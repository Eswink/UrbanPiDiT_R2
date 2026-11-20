"""The in-protocol round: three capacity-matched arms, one parent, one 4-epoch dose.

The round trains ``incumbent`` (the production configuration, pathway absent), ``generic`` (the four
physical fields fused by one projection) and ``typed`` (the same fields, one projection per type)
from the registered 1600-update parent for 32,132 updates, then scores all five leads. The only
capacity statement the contrast rests on is ``generic == typed``; ``incumbent`` is strictly smaller
by the pathway's own parameters and that number is reported, never explained away.

Layout written under the attempt directory::

    seed41/training/{incumbent,generic,typed}/update_*.pt
    seed41/evaluation/{incumbent,generic,typed}/lead_XXXh/{rmse.csv,boundary_rmse.csv,...}
    seed41/probe/{incumbent,generic,typed}/progress.jsonl   the rate probe, 400 updates each
    seed41_{arm}_receipt.json          per-arm checkpoint digest, s/update, capacity, evaluations
    seed41_{arm}_partial_receipt.json  only when an arm stopped short of its endpoint
    readings.json                      the per-arm tables and the family verdict
    archive_receipt.json               code identity

**Why a probe stage before the real arms.** The previous round derived its per-seed envelope from the
incumbent rate (1.1536 s/update) while the treated arms ran at 1.39-1.43, so all three ``typed`` arms
were truncated after about 24,500 of 32,132 updates and 29.32 percent of the segment's compute went
to arms that were never scored. A larger envelope is still a prediction about a rate nobody has
measured on this code revision. The probe measures it: three arms, 400 updates each, about twenty
minutes, and the seed worker then sizes its own envelope from what it saw and refuses rather than
truncating.

**Why a partial receipt.** A truncated arm writes nothing, because the runner raises inside its own
exception handler and ``resume_permitted`` is a literal ``False``. The ledger then has to be
reconstructed by hand from ``progress.jsonl``, and an accountant that globs receipts silently
undercounts. Every arm therefore writes a receipt on the way out, complete or not.
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

from scripts import r7_s3_in_protocol_protocol as proto
from training import r7_device_selection as devices

SEEDS = proto.SEEDS
LEADS = proto.LEADS
STEP_HOURS = proto.STEP_HOURS
PER_SEED_SECONDS = proto.PER_SEED_SECONDS
PLANNED_SECONDS = proto.PLANNED_SECONDS
HARD_CAP_SECONDS = proto.HARD_CAP_SECONDS

#: The probe length. 400 updates is 0.012 epochs: enough for allocation and autotuning to stop
#: dominating the rate, small enough that being wrong costs twenty minutes rather than ten hours.
PROBE_UPDATES = 400
#: The arm is refused when its extrapolated completion exceeds this fraction of the remaining
#: envelope. The headroom is deliberately not spent: the later arms in a seed also pay for five-lead
#: scoring, and a worker that consumes 100 percent of its envelope cannot write its receipt.
REFUSAL_FRACTION = 0.60


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)


def envelope_check(*, probe_seconds_per_update: float, updates: int,
                   evaluation_seconds: float, remaining_seconds: float,
                   arms_left: int = 1) -> dict:
    """Decide whether the remaining arms can finish, from the rate the probe measured.

    The requirement is recomputed rather than compared to a frozen constant, so a rate worse than the
    one that sized the envelope shows up here. ``arms_left`` prices the arms still to run at the same
    measured rate, because a seed worker runs all three and the envelope is per seed.
    """
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
        "rule": ("the envelope is a prediction; this re-derives the requirement from the rate the "
                 "probe measured and refuses before the arm is doomed rather than after"),
    }


def measured_rate(progress_path: Path, after_update: int) -> float | None:
    """Seconds per update over the second half of a run's own progress log.

    ``progress.jsonl`` records ``elapsed_seconds`` and the update number every 20 updates, both taken
    from the training loop's own clock, so a difference of two rows is a rate with no evaluation or
    checkpoint cost folded in. The window starts at the midpoint so the allocation-heavy first rows
    are excluded.
    """
    path = Path(progress_path)
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    usable = [row for row in rows
              if isinstance(row.get("update"), int)
              and isinstance(row.get("elapsed_seconds"), (int, float))]
    if len(usable) < 2:
        return None
    latest = usable[-1]
    if int(latest["update"]) < after_update:
        return None
    earlier = next((row for row in reversed(usable)
                    if int(row["update"]) <= max(1, int(latest["update"]) // 2)), None)
    if earlier is None or int(latest["update"]) <= int(earlier["update"]):
        return None
    delta_updates = int(latest["update"]) - int(earlier["update"])
    delta_seconds = float(latest["elapsed_seconds"]) - float(earlier["elapsed_seconds"])
    if delta_updates <= 0 or delta_seconds <= 0.0:
        return None
    return delta_seconds / delta_updates


def _model_and_state(body: dict, seed: int, name: str):
    """Build one arm's model and initial state from the pinned parent, without training it."""
    from training.r7_arm_harness import sha256_file

    record = next(item for item in body["registry"]
                  if item["arm"] == name and int(item["seed"]) == seed)
    mode = record["typed_evidence_mode"]
    spec = dict(record["model"]["spec"])
    if mode is not None:
        _, payload = proto.payload()
        spec = dict(spec, typed_evidence=dict(payload))
    parent = Path(record["parent"]["checkpoint"])
    if sha256_file(parent) != record["parent"]["sha256"]:
        raise RuntimeError(f"seed {seed} arm {name}: the parent moved since freeze")
    saved = _load_parent(parent, name, seed)
    model, state, expansion = _expand(saved["model"], spec, mode, seed)
    return record, mode, spec, parent, model, state, expansion


def _run(output: Path, body: dict, seed: int, name: str, updates: int, deadline: float,
         phase: str) -> dict:
    """Fine-tune one arm for ``updates`` from the pinned parent.

    Returns a **named record**, not a positional tuple. The first attempt at this round shipped a
    ``_run`` that returned six items where both call sites unpacked seven -- the runner's own
    ``(checkpoint, report)`` pair stayed nested instead of being flattened -- so the probe died with
    ``not enough values to unpack (expected 7, got 6)`` after its first 400 updates. Nothing in the
    41 counter-proofs exercised this function, so no test could see it. A dict makes the failure
    mode unavailable: a caller that asks for a key it did not get raises ``KeyError`` naming the key,
    and adding a field cannot silently shift another one's position.
    """
    from training.r7_long_rollout_runner import fine_tune_long_rollout

    record, mode, spec, parent, model, state, expansion = _model_and_state(body, seed, name)
    contract = _arm_contract(body, spec, mode, parent, record["parent"]["sha256"], expansion,
                             name, seed, total_updates=updates)
    folder = output / f"seed{seed}/{phase}/{name}"
    print(f"[seed {seed}] {phase} {name} ({mode or 'off'}): {updates} updates", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        Path(body["train_manifest"]), folder, model=model, contract=contract,
        parent_weights=state, physical_weights=tuple(body["physical_weights"]),
        steps=int(body["steps"]), updates=int(updates), lr=float(body["lr"]),
        warmup=int(body["warmup_updates"]), seed=seed, device_name="cuda:0", deadline=deadline,
        batch_size=int(body["batch_size"]), checkpoint_every=int(body["checkpoint_every"]),
        loader=(4, 4))
    return {"checkpoint": checkpoint, "report": report, "mode": mode, "expansion": expansion,
            "record": record, "parent": parent, "model": model}


def probe(output: Path, body: dict, seed: int, deadline: float) -> dict:
    """Measure all three arms' rates at 400 updates, then refuse or accept from what was measured.

    This is the stage the previous round did not have. Its envelope was a prediction and its guard
    could not contradict it, so three arms ran ten hours into a wall. Here the rate is measured on
    this code revision, on this card, before any arm is committed to 32,132 updates.
    """
    import torch

    from training.r7_arm_harness import sha256_file

    rates = {}
    for name, mode in proto.ARMS:
        ran = _run(output, body, seed, name, PROBE_UPDATES, deadline, "probe")
        checkpoint, report = ran["checkpoint"], ran["report"]
        rate = float(report["elapsed_seconds"]) / PROBE_UPDATES
        rates[name] = {
            "typed_evidence_mode": ran["mode"],
            "elapsed_seconds": float(report["elapsed_seconds"]),
            "updates": PROBE_UPDATES,
            "seconds_per_update": round(rate, 4),
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": sum(parameter.numel() for parameter in ran["model"].parameters()),
            "new_parameters": int(ran["expansion"]["new_parameters"]),
        }
        print(f"[seed {seed}] probe {name}: {rate:.4f} s/update", flush=True)
    treated = max(rates[proto.GENERIC]["seconds_per_update"],
                  rates[proto.TYPED]["seconds_per_update"])
    remaining = deadline - time.perf_counter()
    decision = envelope_check(
        probe_seconds_per_update=treated,
        updates=int(body["updates"]),
        evaluation_seconds=proto.MEASURED_SECONDS_PER_EVALUATION,
        remaining_seconds=remaining,
        arms_left=len(proto.ARMS))
    record = {
        "scientific_claim": False, "test_read": False, "r": 0,
        "seed": seed, "probe_updates": PROBE_UPDATES,
        "arms": rates,
        "treated_seconds_per_update": round(treated, 4),
        "frozen_treated_seconds_per_update": proto.TREATED_SECONDS_PER_UPDATE,
        "frozen_envelope_seconds": PER_SEED_SECONDS,
        "decision": decision,
        "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
        "limitations": ["a rate probe, not a reading: 400 updates of three arms, scored on nothing"],
    }
    _write_exclusive(output / f"seed{seed}_probe.json", record)
    if not decision["permitted"]:
        # Refusing here is the point. The envelope cannot be widened mid-round without editing a
        # frozen protocol, and running anyway would spend the envelope to produce nothing -- which is
        # exactly what the previous round did. The refusal is recorded, the attempt stops, and a
        # larger envelope becomes a new decision rather than a silent override.
        raise RuntimeError(
            f"seed {seed}: the probe measured {treated:.4f} s/update on the treated arms, so the "
            f"remaining arms need {decision['need_seconds']} s of the {decision['remaining_seconds']} "
            f"s left ({decision['fraction_of_remaining']} of it, over the "
            f"{REFUSAL_FRACTION} refusal fraction). Refusing to start a run that cannot finish.")
    return record


def _train_arm(output: Path, body: dict, seed: int, name: str, mode, deadline: float) -> dict:
    """Fine-tune one arm from the pinned parent, then score its five leads."""
    import torch

    from training.r7_arm_harness import sha256_file
    from training.r7_evaluate import EvaluationPreparation, evaluate_local

    ran = _run(output, body, seed, name, int(body["updates"]), deadline, "training")
    checkpoint, report = ran["checkpoint"], ran["report"]
    mode_seen, expansion, record = ran["mode"], ran["expansion"], ran["record"]
    model = ran["model"]
    if (int(report["updates_this_run"]) != int(body["updates"])
            or int(report["contract"]["batch_size"]) != int(body["batch_size"])
            or report["data_identity"] != body["train_data_identity"]):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    losses = [float(entry["loss"]) for entry in report["losses"]]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"seed {seed} arm {name} produced a nonfinite training loss")

    receipt = {"seed": seed, "arm": name, "typed_evidence_mode": mode_seen,
               "training_dir": str(Path(checkpoint).parent), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(Path(checkpoint).parent
                                                     / "training_report.json"),
               "parent_checkpoint_sha256": record["parent"]["sha256"],
               "elapsed_seconds": report["elapsed_seconds"],
               "seconds_per_update": round(report["elapsed_seconds"] / int(body["updates"]), 4),
               "updates": int(body["updates"]), "batch_size": int(body["batch_size"]),
               "reasoning_steps": int(body["steps"]),
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
        # ``generic`` holds one ``fusion.weight`` of shape (192, 768) where ``typed`` holds four
        # ``projections.N`` of (192, 192): the same 147,456 weights under different key names,
        # because *that block is the axis under test*. Hashing the whole pathway therefore differs by
        # construction, so the invariant that must hold is the shared patch encoder.
        "shared_encoder_sha256": (_state_digest(shared_encoder(pathway))
                                  if pathway else None),
        "shared_encoder_keys": sorted(shared_encoder(pathway)),
        "projection_keys": sorted(set(pathway) - set(shared_encoder(pathway))),
    }


def shared_encoder(pathway: dict) -> dict:
    """The pathway tensors both treated arms build identically: everything but the projection block."""
    return {key: value for key, value in pathway.items()
            if "fusion" not in key and "projections" not in key}


def _arm_contract(body: dict, spec: dict, mode, parent: Path, parent_sha: str, expansion: dict,
                  name: str, seed: int, *, total_updates: int) -> dict:
    return {
        "kind": "process", "model": spec, "seed": seed,
        "data_identity": body["train_data_identity"], "source_sha256": body["source_sha256"],
        "protocol_sha256": body["protocol_sha256"],
        "training_code_sha256": body["training_code_sha256"],
        "total_updates": int(total_updates), "batch_size": int(body["batch_size"]),
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
                           # block omits the window digest, and its discovery is the 12-step
                           # long-rollout one. The plain discovery yields a different digest.
                           "window_sha256": body["train_windows"]["window_sha256"],
                           "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                           # Must be word for word the block ``_supervision_block(None)`` builds:
                           # ``_bind`` compares against the live dict and raises on any difference.
                           "supervision": {"kind": "full_grid",
                                           "description": ("every cell of the target grid is "
                                                           "supervised")}},
    }


def _partial(output: Path, body: dict, seed: int, name: str, mode, *, reason: str,
             detail: dict) -> dict:
    """The receipt a stopped arm writes, so the ledger cannot silently undercount it.

    The runner writes ``failed_attempt.json`` with ``resume_permitted: false`` and no elapsed time, so
    a ledger assembled by globbing completed receipts misses the whole arm. This record carries what
    the arm actually consumed, from its own progress log, and is written for every non-normal exit.
    """
    folder = output / f"seed{seed}/training/{name}"
    progress = folder / "progress.jsonl"
    rows = []
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    usable = [row for row in rows if isinstance(row.get("update"), int)
              and isinstance(row.get("elapsed_seconds"), (int, float))]
    reached = int(usable[-1]["update"]) if usable else 0
    elapsed = float(usable[-1]["elapsed_seconds"]) if usable else 0.0
    record = {
        "scientific_claim": False, "test_read": False, "r": 0,
        "seed": seed, "arm": name, "typed_evidence_mode": mode,
        "status": "partial", "reason": reason,
        "updates_reached": reached, "updates_planned": int(body["updates"]),
        "elapsed_seconds": elapsed,
        "seconds_per_update": (round(elapsed / reached, 4) if reached else None),
        "gpu_hours_in_failed_attempt": round(elapsed / 3_600.0, 4),
        "progress_file": str(progress),
        "progress_rows": len(usable),
        "detail": detail,
        "limitations": ["a partial arm scored nothing; it is recorded so the ledger counts the "
                        "compute it consumed, and it contributes no reading to any clause"],
    }
    _write_exclusive(output / f"seed{seed}_{name}_partial_receipt.json", record)
    return record


def _seed(output: Path, body: dict, seed: int, deadline: float) -> dict:
    receipts = {}
    shared = {}
    projections = {}
    # The rate probe runs inside this seed's own envelope, before any arm is committed. It raises
    # when the measured rate cannot finish the remaining arms, which is the whole point: the frozen
    # envelope is a prediction and this is what makes the prediction checkable. A separate stage
    # name is not available -- the wrapper's stage set is fixed and refuses an unknown phase -- and
    # sharing the envelope is also the honest accounting, since the probe's compute comes out of
    # the same budget it is measuring against.
    probe(output, body, seed, deadline)
    for name, mode in proto.ARMS:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            _partial(output, body, seed, name, mode, reason="envelope-exhausted-before-start",
                     detail={"remaining_seconds": round(remaining, 1)})
            raise TimeoutError(f"deadline reached before seed {seed} arm {name}")
        try:
            receipt = _train_arm(output, body, seed, name, mode, deadline)
        except BaseException as error:                     # noqa: BLE001 - recorded, then re-raised
            _partial(output, body, seed, name, mode,
                     reason=f"{type(error).__name__}: {error}",
                     detail={"remaining_seconds": round(deadline - time.perf_counter(), 1),
                             "refusal_fraction": REFUSAL_FRACTION,
                             "probe_updates": PROBE_UPDATES})
            raise
        _write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
        shared[name] = receipt["expansion"]["shared_encoder_sha256"]
        projections[name] = receipt["expansion"]["pathway_init_sha256"]
        print(f"[seed {seed}] {name}: {receipt['seconds_per_update']} s/update, "
              f"{receipt['parameters']} parameters", flush=True)
    # B->C is attributable to the routing only if the two treated arms started from one shared
    # encoder draw. The projection block itself must differ -- it is the treatment -- so the check is
    # on the encoder the two arms hold in common, which is the part that has to be identical.
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
        from scripts import r7_s3_in_protocol_assembly as assembly
        from scripts import r7_s3_in_protocol_verdict as verdict_module
        assembled = assembly.readings(output, body)
        # ``readings.json`` IS this stage's receipt and ``write`` refuses to overwrite, so a
        # premature run would freeze an incomplete reading AND mark the stage done forever: the
        # round would then skip reading on every later attempt. Refuse instead of recording a
        # half-filled table -- a missing arm must stay untested, not become a published null.
        if not assembled.get("complete"):
            raise RuntimeError(
                "refusing to write readings.json: the round is incomplete, missing "
                f"{assembled.get('missing')}; an incomplete reading would become this stage's "
                "immutable receipt and the missing arms would never be scored")
        verdict_module.write(output / "readings.json", assembled)
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
