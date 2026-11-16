"""The typed-evidence budget round: three capacity-matched arms, one parent, one 4-epoch dose.

The round trains ``incumbent`` (the production configuration, pathway absent), ``generic`` (the four
physical fields fused by one projection) and ``typed`` (the same fields, one projection per type) from
the registered 1600-update parent for 32,132 updates, then scores all five leads. The only capacity
statement the contrast rests on is ``generic == typed``; ``incumbent`` is strictly smaller by the
pathway's own parameters and that number is reported, never explained away.

Layout written under the attempt directory::

    seed41/training/{incumbent,generic,typed}/update_*.pt
    seed41/evaluation/{incumbent,generic,typed}/lead_XXXh/{rmse.csv,boundary_rmse.csv,...}
    seed41_{arm}_receipt.json      per-arm checkpoint digest, s/update, capacity, evaluations
    readings.json                  the per-arm tables and the (a)-(e) clauses
    archive_receipt.json           code identity

**Why one worker per seed.** ``r7_round.py stage-seeds`` spawns one worker per seed with
``CUDA_VISIBLE_DEVICES`` already pinned to the card it cleared, so a worker loads one arm at a time
and never sees another card.

**Why the readings name ``exit_conditions``.** ``tools/check_round_receipts.py:45-47`` requires that
key, and the balance round's mismatch between ``readings.json`` and ``diagnostics.json`` left its
stage permanently "incomplete". This driver writes the content to ``readings.json`` itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_typed_budget_protocol as proto
from training import r7_device_selection as devices

SEEDS = proto.SEEDS
LEADS = proto.LEADS
STEP_HOURS = proto.STEP_HOURS
PER_SEED_SECONDS = proto.PER_SEED_SECONDS
PLANNED_SECONDS = proto.PLANNED_SECONDS
HARD_CAP_SECONDS = proto.HARD_CAP_SECONDS

#: The variables the clauses read. ``t2m`` carries (a) and (b); ``z500`` carries the (e) guard.
CLAUSE_VARIABLES = ("t2m", "z500")


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)


def _region_rmse(path: Path, region: str, variable: str) -> float:
    """One RMSE from a ``boundary_rmse.csv``, refusing a silently absent row."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] == region and row["variable"] == variable:
                return float(row["rmse"])
    raise RuntimeError(f"{region}/{variable} missing from {path}")


def arm_tables(evaluation_root: Path, arm: str) -> dict:
    """``{lead: {variable: rmse}}`` for one arm, over the declared clause variables."""
    table = {}
    for lead in LEADS:
        lead_dir = evaluation_root / arm / f"lead_{lead:03d}h"
        if not lead_dir.is_dir():
            continue
        boundary = lead_dir / "boundary_rmse.csv"
        if not boundary.is_file():
            continue
        table[str(lead)] = {variable: _region_rmse(boundary, proto.REGION, variable)
                            for variable in CLAUSE_VARIABLES}
    return table


def _train_arm(output: Path, body: dict, seed: int, name: str, mode, deadline: float) -> dict:
    """Fine-tune one arm from the pinned parent, then score its five leads."""
    import torch

    from model.typed_evidence_r7 import TypedEvidenceRouter
    from training.r7_arm_harness import sha256_file
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_experiment import make_model

    record = next(item for item in body["registry"]
                  if item["arm"] == name and int(item["seed"]) == seed)
    spec = dict(record["model"]["spec"])
    if mode is not None:
        _, payload = proto.payload()
        spec = dict(spec, typed_evidence=dict(payload))
    parent = Path(record["parent"]["checkpoint"])
    if sha256_file(parent) != record["parent"]["sha256"]:
        raise RuntimeError(f"seed {seed} arm {name}: the parent moved since freeze")

    saved = _load_parent(parent, name, seed)
    model, state, expansion = _expand(saved["model"], spec, mode, seed)
    contract = _arm_contract(body, spec, mode, parent, record["parent"]["sha256"], expansion,
                             name, seed)
    folder = output / f"seed{seed}/training/{name}"
    print(f"[seed {seed}] arm {name} ({mode or 'off'}): {body['updates']} updates", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        Path(body["train_manifest"]), folder, model=model, contract=contract,
        parent_weights=state, physical_weights=tuple(body["physical_weights"]),
        steps=int(body["steps"]), updates=int(body["updates"]), lr=float(body["lr"]),
        warmup=int(body["warmup_updates"]), seed=seed, device_name="cuda:0", deadline=deadline,
        batch_size=int(body["batch_size"]), checkpoint_every=int(body["checkpoint_every"]),
        loader=(4, 4))
    if (int(report["updates_this_run"]) != int(body["updates"])
            or int(report["contract"]["batch_size"]) != int(body["batch_size"])
            or report["data_identity"] != body["train_data_identity"]):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    losses = [float(entry["loss"]) for entry in report["losses"]]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"seed {seed} arm {name} produced a nonfinite training loss")

    receipt = {"seed": seed, "arm": name, "typed_evidence_mode": mode,
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": record["parent"]["sha256"],
               "elapsed_seconds": report["elapsed_seconds"],
               "seconds_per_update": round(report["elapsed_seconds"] / int(body["updates"]), 4),
               "updates": int(body["updates"]), "batch_size": int(body["batch_size"]),
               "reasoning_steps": int(body["steps"]),
               "parameters": sum(p.numel() for p in model.parameters()),
               "expansion": expansion,
               "loss_curve": {"first": losses[0], "last": losses[-1],
                              "min": min(losses), "max": max(losses), "n": len(losses)},
               "evaluations": {}}

    prepared = EvaluationPreparation(Path(body["val_manifest"]), deadline=deadline)
    for lead in LEADS:
        destination = output / f"seed{seed}/evaluation/{name}/lead_{lead:03d}h"
        print(f"[seed {seed}] arm {name}: evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(body["val_manifest"]), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=int(body["steps"]), boundary_margins=proto.BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        if provenance["n_evaluated"] < 1:
            raise RuntimeError(f"seed {seed} arm {name} lead {lead}: no initialisation evaluated")
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination),
            "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    return receipt


def _load_parent(path: Path, name: str, seed: int) -> dict:
    from training.r7_experiment import load_checkpoint, model_code_digest

    saved = load_checkpoint(path)
    if saved["model_code_sha256"] != model_code_digest():
        raise RuntimeError("the parent's code identity is not the current tree's")
    if int(saved["updates"]) != proto.PARENT_UPDATES:
        raise RuntimeError(f"the parent is at {saved['updates']} updates, not "
                           f"{proto.PARENT_UPDATES}")
    return saved


def _expand(parent_state: dict, spec: dict, mode, seed: int) -> tuple:
    """Install the parent bit for bit and add exactly the pathway's tensors, drawn once per seed.

    ``generic`` and ``typed`` must start from the *same* pathway initialization, or their delta
    carries a fresh random draw as well as the routing difference. The seed therefore keys the draw,
    not the arm.
    """
    import torch

    from training.r7_autoregressive_runner import _state_digest
    from training.r7_experiment import make_model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        target = make_model("process", spec)
    target_state = target.state_dict()
    missing = [key for key in target_state if key not in parent_state]
    extra = [key for key in parent_state if key not in target_state]
    if extra:
        raise RuntimeError(f"the parent carries keys the target does not: {sorted(extra)}")
    if mode is None and missing:
        raise RuntimeError("the incumbent arm must not add any key; the parent is its own trunk")
    if mode is not None and not missing:
        raise RuntimeError("a treated arm that adds no key is not carrying the pathway")
    state = dict(parent_state)
    for key in missing:
        state[key] = target_state[key].detach().clone()
    target.load_state_dict(state, strict=True)
    pathway = {key: state[key] for key in sorted(missing)}
    return target, state, {
        "parent_keys": len(parent_state), "new_keys": sorted(missing),
        "new_parameters": int(sum(state[key].numel() for key in missing)),
        "init_seed": int(seed),
        "parent_state_sha256": _state_digest(parent_state),
        "pathway_init_sha256": _state_digest(pathway) if pathway else None,
        # The invariant that is actually achievable across the two treated arms. ``generic`` holds
        # one ``fusion.weight`` of shape (192, 768) where ``typed`` holds four ``projections.N`` of
        # (192, 192): same 147,456 weights, different key names and shapes, because *that block is
        # the axis under test*. Hashing the whole pathway therefore differs by construction, so the
        # achievability test is the **shared** patch encoder, which both arms build from the same
        # seeded draw and which must be bit-identical.
        "shared_encoder_sha256": (_state_digest(shared_encoder(pathway))
                                  if pathway else None),
        "shared_encoder_keys": sorted(shared_encoder(pathway)),
        "projection_keys": sorted(set(pathway) - set(shared_encoder(pathway))),
    }


def shared_encoder(pathway: dict) -> dict:
    """The pathway tensors both treated arms build identically: everything but the projection block.

    ``conv``/``norm`` are the patch encoder; ``fusion.weight`` (generic) and
    ``projections.N.weight`` (typed) are the block the two arms deliberately differ in.
    """
    return {key: value for key, value in pathway.items()
            if "fusion" not in key and "projections" not in key}


def _arm_contract(body: dict, spec: dict, mode, parent: Path, parent_sha: str, expansion: dict,
                  name: str, seed: int) -> dict:
    return {
        "kind": "process", "model": spec, "seed": seed,
        "data_identity": body["train_data_identity"], "source_sha256": body["source_sha256"],
        "protocol_sha256": body["protocol_sha256"],
        "training_code_sha256": body["training_code_sha256"],
        "total_updates": int(body["updates"]), "batch_size": int(body["batch_size"]),
        "steps": int(body["steps"]), "lr": float(body["lr"]),
        "warmup_updates": int(body["warmup_updates"]), "weight_decay": float(body["weight_decay"]),
        "clip": float(body["clip"]), "bf16": False, "process_weight": 0.0,
        "initialization": {
            "parent_run": str(proto.PARENT_RUN),
            "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
            "parent_endpoint_updates": proto.PARENT_UPDATES,
            "arm": name,
            "description": (f"registered 1600-update parent installed bit for bit; "
                            f"typed-evidence mode={mode or 'off'} "
                            f"({expansion['new_parameters']} new parameters)")},
        "autoregression": {"physical_steps": 12,
                           "physical_weights": list(body["physical_weights"]),
                           # ``fine_tune_long_rollout`` refuses a contract whose autoregression
                           # block omits the window digest ("physical steps/weights/window digest
                           # must be predeclared exactly"), and every precedent driver
                           # (spectral-geometry, curve-point) supplies it. Without it the seed
                           # worker would raise before the first update rather than train.
                           "window_sha256": body["train_windows"]["window_sha256"],
                           "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                           # Must be word for word the block ``_supervision_block(None)`` builds:
                           # ``_bind`` compares the declared dict against the live one and raises on
                           # any difference, so declaring only ``{"kind": "full_grid"}`` is refused.
                           "supervision": {"kind": "full_grid",
                                           "description": ("every cell of the target grid is "
                                                           "supervised")}},
    }


def _seed(output: Path, body: dict, seed: int, deadline: float) -> dict:
    receipts = {}
    shared = {}
    projections = {}
    for name, mode in proto.ARMS:
        if time.monotonic() > deadline:
            raise TimeoutError(f"deadline reached before seed {seed} arm {name}")
        receipt = _train_arm(output, body, seed, name, mode, deadline)
        _write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
        shared[name] = receipt["expansion"]["shared_encoder_sha256"]
        projections[name] = receipt["expansion"]["pathway_init_sha256"]
        print(f"[seed {seed}] {name}: {receipt['seconds_per_update']} s/update, "
              f"{receipt['parameters']} parameters", flush=True)
    # B->C is attributable to the routing only if the two treated arms started from one shared
    # encoder draw. The projection block itself must differ -- it is the treatment -- so the check
    # is on the encoder the two arms hold in common, which is the part that has to be identical.
    treated = {shared[proto.GENERIC], shared[proto.TYPED]}
    if len(treated) != 1 or None in treated:
        raise RuntimeError(f"seed {seed}: the two treated arms do not share one encoder "
                           f"initialization: {shared}")
    if projections[proto.GENERIC] == projections[proto.TYPED]:
        raise RuntimeError(f"seed {seed}: the two treated arms hold the same projection block, "
                           f"which would mean the routing switch did not build a different graph")
    return {"seed": seed, "shared_encoder_sha256": shared[proto.TYPED],
            "pathway_init_sha256": {name: projections[name]
                                    for name in (proto.GENERIC, proto.TYPED)},
            "arms": {name: {"checkpoint_sha256": receipts[name]["checkpoint_sha256"],
                            "parameters": receipts[name]["parameters"]}
                     for name in receipts}}


def readings(output: Path, body: dict) -> dict:
    """Assemble the per-arm tables and evaluate (a)-(e)."""
    from scripts import r7_typed_budget_verdict as verdict
    from training.r7_arm_harness import sha256_file

    output = Path(output)
    arms = {name: {} for name, _ in proto.ARMS}
    missing = []
    for seed in SEEDS:
        receipt_path = output / f"seed{seed}_{proto.INCUMBENT}_receipt.json"
        if not receipt_path.is_file():
            missing.append(f"seed{seed}")
            continue
        for name, mode in proto.ARMS:
            receipt = json.loads((output / f"seed{seed}_{name}_receipt.json")
                                 .read_text(encoding="utf-8"))
            table = {}
            for lead in LEADS:
                entry = (receipt.get("evaluations") or {}).get(str(lead))
                if not entry:
                    continue
                table[str(lead)] = {
                    variable: _region_rmse(Path(entry["dir"]) / "boundary_rmse.csv", proto.REGION,
                                           variable)
                    for variable in CLAUSE_VARIABLES}
            if len(table) != len(LEADS):
                missing.append(f"{name}/seed{seed}")
            arms[name][str(seed)] = {"rmse": table, "parameters": receipt["parameters"],
                                     "typed_evidence_mode": receipt["typed_evidence_mode"],
                                     "seconds_per_update": receipt["seconds_per_update"]}
    body_readings = {
        "format": "r7-s3-typed-evidence-budget-readings-v1",
        "protocol_sha256": body["protocol_sha256"],
        "updates": int(body["updates"]), "batch_size": int(body["batch_size"]),
        "epochs": int(body["updates"]) / 32_131,
        "seeds": list(SEEDS), "lead_hours": list(LEADS),
        "region": proto.REGION,
        "variables": list(CLAUSE_VARIABLES),
        "arms": arms,
        "complete": not missing,
        "missing": sorted(set(missing)),
        "closed_dose": body["closed_dose"],
        "independent_arm_argument": body["independent_arm_argument"],
        "preflight": body["preflight"],
        "checks": {
            "capacity": body["preflight"]["clauses"]["capacity_is_matched_where_it_must_be"],
            "forward_arithmetic": body["preflight"]["clauses"]["forward_arithmetic_is_matched"],
            "val_manifest_sha256": sha256_file(Path(body["val_manifest"])),
        },
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "limitations": body["limitations"],
    }
    judgement = verdict.evaluate(body_readings)
    body_readings["exits"] = judgement
    body_readings["exit_conditions"] = judgement["clauses"]
    body_readings["verdict"] = judgement["verdict"]
    return body_readings


def worker(output, phase, seed, deadline):
    """One stage of this round, matching the wrapper's ``--worker`` contract."""
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        return proto.prepared_protocol(output)
    if phase == "archive":
        return _archive(output, proto.validate_protocol(output))
    body = proto.validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside the frozen round")
    from training.r7_s3_v3_screen import pin_declared_gpu
    pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        # The reading path must name its writer here too: the balance round died on a NameError
        # 7.2 s into this branch, after the whole GPU round had already been paid for.
        from scripts import r7_typed_budget_verdict as verdict_module
        verdict_module.write(output / "readings.json", readings(output, body))
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _write_exclusive(output / f"seed{seed}_receipt.json", _seed(output, body, seed, deadline))


def _archive(output: Path, body: dict) -> dict:
    """Verify the code archive's comment and execution sources match the freeze."""
    import hashlib
    from io import BytesIO
    import zipfile
    from training.r7_arm_harness import sha256_file

    archive = Path(body["code_archive_input"])
    if any(path.is_symlink() for path in (archive, *archive.parents)):
        raise ValueError("linked archive input forbidden")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("prepared code archive SHA mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as zipped:
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
               "code_zip_sha256": sha256_file(output / "code.zip")}
    _write_exclusive(output / "archive_receipt.json", receipt)
    return receipt


def _write_exclusive(path: Path, body: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


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
