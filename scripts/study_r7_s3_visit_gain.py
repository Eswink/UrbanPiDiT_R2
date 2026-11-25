"""G1-B driver: two damped arms per seed, from the migrated p32,132 incumbent (decision 0119).

Layout written under the attempt directory::

    seed41/training/{visit_damped,uniform_damped}/update_0XXXX.pt
    seed41/evaluation/{visit_damped,uniform_damped}/lead_XXXh/{rmse.csv,boundary_rmse.csv,...}
    seed41/probe/{visit_damped,uniform_damped}/progress.jsonl      the rate probe, 400 updates each
    seed41_{arm}_receipt.json            checkpoint digest, s/update, capacity, evaluations
    seed41_{arm}_partial_receipt.json    only when an arm stopped short of its endpoint
    identity_witness.json                exit 3, measured before the first optimizer step
    readings.json                        the four exits
    archive_receipt.json                 code identity

**Why the identity witness is a separate stage artifact.** Exit 3 is a claim about *this*
code under *this* interpreter, so it is measured once per attempt, before training, and
written even though nothing else depends on it. Measuring it after the arms would leave
it possible to report an identity claim the training actually ran past.

**Why the probe runs inside each seed's envelope.** A rate prediction that cannot be
contradicted is how the previous round truncated every treated arm. The probe measures
the real rate at 400 updates and the worker refuses rather than truncating; its compute
comes out of the same envelope it measures against, which is also the honest accounting.

**Why a partial receipt.** The runner raises with ``resume_permitted: false`` and writes
no elapsed time, so an accountant that globs completed receipts silently undercounts a
killed arm. Every non-normal exit writes one, with the update count and seconds recovered
from the arm's own ``progress.jsonl``.
"""
from __future__ import annotations

import argparse
import io
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_visit_gain_parent as parent_module  # noqa: E402
from scripts.r7_s3_visit_gain_identity import (  # noqa: E402
    _identity, _sha256, _write_exclusive, partial_receipt, prepared_protocol,
    validate_protocol)
from scripts import r7_s3_visit_gain_protocol as proto  # noqa: E402

SEEDS = proto.SEEDS
LEADS = proto.LEADS
STEP_HOURS = proto.STEP_HOURS
PER_SEED_SECONDS = proto.PER_SEED_SECONDS
PLANNED_SECONDS = proto.PLANNED_SECONDS
HARD_CAP_SECONDS = proto.HARD_CAP_SECONDS

#: 400 updates is 0.17 epochs: long enough for allocation and autotuning to stop dominating
#: the rate, short enough that being wrong costs minutes rather than a seed's whole envelope.
PROBE_UPDATES = 400
REFUSAL_FRACTION = 0.60

#: The sources whose bytes define this round's execution identity.
EXECUTION_SOURCES = (
    "model/recursive_weather_r7.py", "model/process_forecast_r7.py",
    "model/process_step_r7.py", "model/layers/sdpa.py",
    "model/weather_forecaster_r7.py", "model/coarse_forecast.py",
    "model/coarse_encoder.py", "model/local_solver_state_r7.py",
    "model/process_readout_r7.py", "model/spacetime_conditioning_r7.py",
    "data/r7_long_rollout_dataset.py", "data/r7_zarr_dataset.py", "data/r7_store.py",
    "data/r7_evaluation.py",
    "training/r7_long_rollout_runner.py", "training/r7_recursive_losses.py",
    "training/r7_losses.py", "training/r7_experiment.py", "training/r7_arm_harness.py",
    "training/r7_evaluate.py", "training/r7_boundary_metrics.py",
    "training/r7_climatology_skill.py", "training/r7_rollout_metrics.py",
    "training/r7_simultaneous_stats.py", "training/r7_device_selection.py",
    "scripts/r7_s3_visit_gain_protocol.py", "scripts/r7_s3_visit_gain_parent.py",
    "scripts/r7_s3_visit_gain_verdict.py", "scripts/study_r7_s3_visit_gain.py",
)


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)


def execution_files() -> list[str]:
    missing = [name for name in EXECUTION_SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"declared execution sources are missing: {missing}")
    return list(EXECUTION_SOURCES)


def _sha256(path: Path) -> str:
    from training.r7_arm_harness import sha256_file

    return sha256_file(Path(path))


def _write_exclusive(path: Path, body: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def envelope_check(*, probe_seconds_per_update: float, updates: int,
                   evaluation_seconds: float, remaining_seconds: float,
                   arms_left: int = 1) -> dict:
    """Whether the remaining arms can finish, from the rate the probe just measured."""
    per_arm = int(updates) * float(probe_seconds_per_update) + float(evaluation_seconds)
    need = max(1, int(arms_left)) * per_arm
    fraction = (need / remaining_seconds) if remaining_seconds > 0 else float("inf")
    return {
        "probe_seconds_per_update": round(float(probe_seconds_per_update), 4),
        "updates": int(updates),
        "arms_left": max(1, int(arms_left)),
        "per_arm_seconds": round(per_arm, 1),
        "need_seconds": round(need, 1),
        "remaining_seconds": round(float(remaining_seconds), 1),
        "fraction_of_remaining": (round(fraction, 4) if math.isfinite(fraction) else None),
        "refusal_fraction": REFUSAL_FRACTION,
        "permitted": bool(math.isfinite(fraction) and fraction <= REFUSAL_FRACTION),
    }


def _arm_contract(body: dict, spec: dict, arm: str, parent: Path, parent_sha: str,
                  seed: int, *, total_updates: int) -> dict:
    return {
        "kind": "process", "model": spec, "seed": int(seed),
        "data_identity": body["train_data_identity"], "source_sha256": body["source_sha256"],
        "protocol_sha256": body["protocol_sha256"],
        "training_code_sha256": body["training_code_sha256"],
        "total_updates": int(total_updates), "batch_size": int(body["batch_size"]),
        "steps": int(body["steps"]), "lr": float(body["lr"]),
        "warmup_updates": int(body["warmup_updates"]), "weight_decay": float(body["weight_decay"]),
        "clip": float(body["clip"]), "bf16": False, "process_weight": 0.0,
        "initialization": {
            "parent_run": body["parent"]["run"], "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": parent_sha,
            "parent_endpoint_updates": proto.PARENT_UPDATES, "arm": arm,
            "description": (f"registered p{proto.PARENT_UPDATES} incumbent installed bit for bit; "
                            f"visit_gain={arm} adds no parameter")},
        "autoregression": {
            "physical_steps": 12, "physical_weights": list(body["physical_weights"]),
            "window_sha256": body["train_windows"]["window_sha256"],
            "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
            "supervision": {"kind": "full_grid",
                            "description": "every cell of the target grid is supervised"}},
    }


def _model_and_state(body: dict, seed: int, arm: str):
    """Build one arm's model and initial state from the migrated parent, without training."""
    import torch

    from training.r7_experiment import make_model

    record = next(item for item in body["registry"]
                  if item["arm"] == arm and int(item["seed"]) == int(seed))
    spec = dict(record["model"]["spec"])
    parent = Path(record["parent"]["checkpoint"])
    if _sha256(parent) != record["parent"]["sha256"]:
        raise RuntimeError(f"seed {seed} arm {arm}: the parent moved since freeze")
    saved = torch.load(parent, map_location="cpu", weights_only=True)
    if int(saved["updates"]) != proto.PARENT_UPDATES:
        raise RuntimeError(f"seed {seed} arm {arm}: parent is not at "
                           f"{proto.PARENT_UPDATES} updates")
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", spec)
    state = saved["model"]
    expected = model.state_dict()
    if set(expected) != set(state):
        raise RuntimeError(f"seed {seed} arm {arm}: the parent's tensor keys differ from the "
                           f"arm's model; {arm} must not add a parameter")
    model.load_state_dict(state, strict=True)
    return record, spec, parent, model, state


def _run(output: Path, body: dict, seed: int, arm: str, updates: int, deadline: float,
         phase: str) -> dict:
    """Fine-tune one arm for ``updates`` from the migrated parent. Returns a named record."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout

    record, spec, parent, model, state = _model_and_state(body, seed, arm)
    contract = _arm_contract(body, spec, arm, parent, record["parent"]["sha256"], seed,
                             total_updates=updates)
    folder = output / f"seed{seed}/{phase}/{arm}"
    print(f"[seed {seed}] {phase} {arm} ({spec['visit_gain']}): {updates} updates", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        Path(body["train_manifest"]), folder, model=model, contract=contract,
        parent_weights=state, physical_weights=tuple(body["physical_weights"]),
        steps=int(body["steps"]), updates=int(updates), lr=float(body["lr"]),
        warmup=int(body["warmup_updates"]), seed=int(seed), device_name="cuda:0",
        deadline=deadline, batch_size=int(body["batch_size"]),
        checkpoint_every=int(body["checkpoint_every"]), loader=(4, 4))
    return {"checkpoint": checkpoint, "report": report, "spec": spec, "record": record,
            "parent": parent, "model": model}


def _train_arm(output: Path, body: dict, seed: int, arm: str, deadline: float) -> dict:
    """Fine-tune one arm from the parent, then score its five leads."""
    import torch

    from training.r7_evaluate import EvaluationPreparation, evaluate_local

    ran = _run(output, body, seed, arm, int(body["updates"]), deadline, "training")
    checkpoint, report = ran["checkpoint"], ran["report"]
    if (int(report["updates_this_run"]) != int(body["updates"])
            or int(report["contract"]["batch_size"]) != int(body["batch_size"])
            or report["data_identity"] != body["train_data_identity"]):
        raise RuntimeError(f"seed {seed} arm {arm} did not reach the frozen endpoint")
    losses = [float(entry["loss"]) for entry in report["losses"]]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"seed {seed} arm {arm} produced a nonfinite training loss")
    receipt = {
        "seed": int(seed), "arm": arm, "visit_gain": ran["spec"]["visit_gain"],
        "training_dir": str(Path(checkpoint).parent), "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_report_sha256": _sha256(Path(checkpoint).parent / "training_report.json"),
        "parent_checkpoint_sha256": ran["record"]["parent"]["sha256"],
        "elapsed_seconds": report["elapsed_seconds"],
        "seconds_per_update": round(float(report["elapsed_seconds"]) / int(body["updates"]), 4),
        "updates": int(body["updates"]), "batch_size": int(body["batch_size"]),
        "reasoning_steps": int(body["steps"]),
        "parameters": sum(p.numel() for p in ran["model"].parameters()),
        "loss_curve": {"first": losses[0], "last": losses[-1],
                       "min": min(losses), "max": max(losses), "n": len(losses)},
        "evaluations": {},
    }
    prepared = EvaluationPreparation(Path(body["val_manifest"]), deadline=deadline)
    for lead in LEADS:
        destination = output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
        print(f"[seed {seed}] arm {arm}: evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(body["val_manifest"]), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=int(body["steps"]), boundary_margins=proto.BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        if provenance["n_evaluated"] < 1:
            raise RuntimeError(f"seed {seed} arm {arm} lead {lead}: nothing evaluated")
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "boundary_rmse_csv_sha256": _sha256(destination / "boundary_rmse.csv"),
            "provenance_sha256": _sha256(destination / "provenance.json")}
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    return receipt


def _seed(output: Path, body: dict, seed: int, deadline: float) -> dict:
    probe(output, body, seed, deadline)
    receipts = {}
    for arm, _ in proto.ARMS:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            partial_receipt(output, body, seed, arm, reason="envelope-exhausted-before-start",
                     detail={"remaining_seconds": round(remaining, 1)})
            raise TimeoutError(f"deadline reached before seed {seed} arm {arm}")
        try:
            receipt = _train_arm(output, body, seed, arm, deadline)
        except BaseException as error:      # noqa: BLE001 - recorded, then re-raised
            partial_receipt(output, body, seed, arm, reason=f"{type(error).__name__}: {error}",
                     detail={"remaining_seconds": round(deadline - time.perf_counter(), 1),
                             "refusal_fraction": REFUSAL_FRACTION,
                             "probe_updates": PROBE_UPDATES})
            raise
        _write_exclusive(output / f"seed{seed}_{arm}_receipt.json", receipt)
        receipts[arm] = receipt
    return {
        "seed": int(seed),
        "arms": {arm: {"checkpoint_sha256": receipt["checkpoint_sha256"],
                       "seconds_per_update": receipt["seconds_per_update"],
                       "parameters": receipt["parameters"]}
                 for arm, receipt in receipts.items()},
        "parameter_counts_identical": len({r["parameters"] for r in receipts.values()}) == 1,
    }


def _archive(output: Path, body: dict) -> dict:
    """Verify the code archive's comment and execution sources match the freeze."""
    archive = Path(body["code_archive_input"])
    if any(path.is_symlink() for path in (archive, *archive.parents)):
        raise ValueError("linked archive input forbidden")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("prepared code archive SHA mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        if zipped.comment.decode() != body["code_commit"]:
            raise RuntimeError("archive commit differs from freeze")
        for name, expected in body["execution_files_sha256"].items():
            if name.endswith(".py") and hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                raise RuntimeError(f"archive omits or changes execution source: {name}")
    with (output / "code_commit.txt").open("x", encoding="utf-8") as handle:
        handle.write(body["code_commit"] + "\n")
    with (output / "code.zip").open("xb") as handle:
        handle.write(payload)
    receipt = {"scientific_claim": False, "code_commit": body["code_commit"],
               "code_zip_sha256": _sha256(output / "code.zip")}
    _write_exclusive(output / "archive_receipt.json", receipt)
    return receipt


def worker(output, phase, seed, deadline):
    """One stage of this round, matching the wrapper's ``--worker`` contract."""
    from scripts.r7_m3_offline import deny_network
    from scripts import r7_s3_visit_gain_verdict as verdict_module

    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        body = prepared_protocol(output)
        from scripts.r7_s3_visit_gain_identity import identity_witness

        witness = identity_witness(body)
        _write_exclusive(output / "identity_witness.json", witness)
        print(f"[worker] prepare done: protocol {body['protocol_sha256'][:12]}", flush=True)
        return body
    if phase == "archive":
        return _archive(output, validate_protocol(output))
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside the frozen round")
    from training.r7_s3_v3_screen import pin_declared_gpu
    from training import r7_device_selection as devices
    pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        return verdict_module.build(
            output, protocol_sha256=body["protocol_sha256"],
            migration_path=Path(body["parent_migration"]["run"]))
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _write_exclusive(output / f"seed{seed}_receipt.json", _seed(output, body, seed, deadline))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    arguments = parser.parse_args(argv)
    if not arguments.worker:
        raise SystemExit("this round is driven by `r7_round.py stage-seeds`; invoke the driver "
                         "with --worker <prepare|archive|seed|reading>")
    worker(arguments.out, arguments.worker, arguments.seed, arguments.deadline)


if __name__ == "__main__":
    main()
