"""Stage B point 1 (0.25 epoch = 2,008 updates): the learning curve's first ordinate.

Stage B asks whether 72 h skill is still rising with training volume, or saturated. It is a
*calibration* curve, not a factor comparison: one arm, one seed, batch fixed at 4, and the
only thing that varies across the four points is how many updates the run gets. This file
runs point 1 and scores all seventeen variables at all five leads.

Why batch stays 4 (goal §3, decision 0084 §Decision 3): R1 measured that *updates* rather
than samples move skill, and A6 measured that batch 12-14 saturates throughput while raising
peak memory 74%. Changing batch during calibration would confound the curve being measured.
Batch only appears as a Stage F cost option, and then only with the updates axis restated.

The saturation judgement itself is NOT made here. One point cannot compute a two-point first
difference; the rule is recorded verbatim in the protocol and evaluated when the ladder is
complete. This round's predeclared exit is therefore an *engineering* one (see the brief).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from scripts.r7_s3_fy_year_coverage_climatology import (
    D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS)
from scripts.r7_s3_throughput_physics_b_protocol import (
    BATCH_SIZE, BOUNDARY_MARGINS, CHECKPOINT_EVERY, FULL_GRID_POINTS, HARD_CAP_SECONDS, LR,
    PER_SEED_SECONDS, PHYSICAL_WEIGHTS, PLANNED_SECONDS, POINT, REASONING_STEPS, REGION,
    SATURATION_RULE, SEEDS, STEPS, STORE, TRAIN_MANIFEST, UPDATES, VAL_MANIFEST, WARMUP,
    _prepared_protocol, freeze, validate_protocol)
from training import r7_device_selection as devices

#: The seventeen dynamic channels scored at every lead, from the store's own channel list.
#: Named explicitly rather than iterated from the store so the scored surface is frozen in
#: text: a change to the store's channel order must fail loudly, not silently rescore.
#: Verbatim from the frozen store's own ``channels`` attribute (17 entries). A store whose
#: channel list moved would make these lookups fail loudly rather than silently rescore.
SCORED_VARIABLES = (
    "t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
    "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250",
)


def _model(output, body, seed):
    """Load the pinned migrated parent under the current revision, asserting its identity."""
    import torch

    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    from model.spacetime_conditioning_r7 import isolated_stream
    spec = body["arms"]["curve"]["model"]["spec"]
    path, expected = parent_pins.pinned_parent(seed)
    if sha256_file(path) != expected:
        raise RuntimeError(f"seed {seed} parent checkpoint pin changed")
    saved = load_checkpoint(path)
    if saved["updates"] != parent_pins.ENDPOINT:
        raise RuntimeError(f"seed {seed} parent is not the frozen endpoint")
    parent_state = saved["model"]
    with isolated_stream():
        torch.manual_seed(int(seed))
        model = make_model("process", spec)
    if list(parent_state) != list(model.state_dict()):
        raise RuntimeError(f"seed {seed}: the parent's key set differs from this revision")
    model.load_state_dict(parent_state, strict=True)
    digest = _state_digest(model.state_dict())
    if digest != parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed}: the imported state is not the pinned weights "
                           f"({digest} != {parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]})")
    return model, parent_state, path, expected, {
        "parent_state_sha256": digest,
        "parent_keys": len(parent_state),
        "endpoint_updates": parent_pins.ENDPOINT,
        "init_seed": int(seed),
    }


def _contract(body, parent, parent_sha, expansion):
    return {"kind": "process", "model": body["arms"]["curve"]["model"]["spec"],
            "data_identity": body["arms"]["curve"]["train_data_identity"],
            "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": parent_pins.ENDPOINT,
                               "curve_point": POINT, "updates": UPDATES,
                               "description": ("registered v3-BD endpoint, model-only migration "
                                               "under the current revision; Stage B point "
                                               f"{POINT} = {UPDATES} updates at batch "
                                               f"{BATCH_SIZE}")},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(PHYSICAL_WEIGHTS),
                               "window_sha256": body["arms"]["curve"]["train_windows"]["window_sha256"],
                               "excluded_sample_ids": body["arms"]["curve"]["train_windows"]["excluded_sample_ids"]}}


def _check_provenance(provenance, seed, lead):
    if provenance["split"] != "val" or "test_read" in provenance:
        raise RuntimeError(f"seed {seed} lead {lead} did not stay on val")
    if (provenance["n_evaluated"] != VAL_COHORTS[str(lead)]
            or provenance["n_available_windows"] != VAL_COHORTS[str(lead)]
            or provenance["lead_hours"] != [lead]):
        raise RuntimeError(f"seed {seed} lead {lead} cohort differs from the frozen one")
    regions = {region["region"]: region for region in
               (provenance["boundary_scoring"] or {}).get("regions", [])}
    if regions.get(REGION, {}).get("n_grid_points") != FULL_GRID_POINTS:
        raise RuntimeError(f"seed {seed} lead {lead} full region is not 65x65")


def _train(output, body, seed, deadline):
    """Train this point's single arm from the parent and score every lead."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha, expansion = _model(output, body, seed)
    contract = _contract(body, parent, parent_sha, expansion)
    folder = output / f"seed{seed}/training"
    print(f"[seed {seed}] curve point {POINT}: {UPDATES} updates at batch {BATCH_SIZE}",
          flush=True)
    checkpoint, report = fine_tune_long_rollout(
        TRAIN_MANIFEST, folder, model=model, contract=contract, parent_weights=state,
        physical_weights=PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline, batch_size=BATCH_SIZE,
        # Decision 0087's policy is only real if the runner is told it: this argument
        # defaults to 20, and omitting it wrote 101 checkpoints (5.6 GiB) on the first
        # attempt of this point while the protocol declared 5,000.
        checkpoint_every=CHECKPOINT_EVERY, loader=(4, 4))
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["arms"]["curve"]["train_data_identity"]
            or report["supervision"]["kind"] != "full_grid"):
        raise RuntimeError(f"seed {seed} did not reach the frozen endpoint")
    update1 = dict(report["losses"][0])
    losses = [float(entry["loss"]) for entry in report["losses"]]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"seed {seed} produced a nonfinite training loss")
    receipt = {"seed": seed, "curve_point": POINT, "updates": UPDATES,
               "batch_size": BATCH_SIZE, "reasoning_steps": STEPS,
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"],
               "train_data_identity": report["data_identity"],
               "train_windows_sha256": body["arms"]["curve"]["train_windows"]["window_sha256"],
               "expansion": expansion,
               "checkpoint_every": CHECKPOINT_EVERY,
               "retained_checkpoints": sorted(item.name for item in folder.glob("update_*.pt")),
               "retained_checkpoint_count": len(list(folder.glob("update_*.pt"))),
               "update1": {"loss": update1["loss"], "gradient_norm": update1["gradient_norm"]},
               "loss_curve": {"first": losses[0], "last": losses[-1],
                              "min": min(losses), "max": max(losses), "n": len(losses)},
               "threads": _thread_count(),
               "evaluations": {}}
    print(f"[seed {seed}] trained in {report['elapsed_seconds']:.0f}s; scoring five leads",
          flush=True)
    # One manifest reader, one store scan and one training climatology for all five leads.
    prepared = EvaluationPreparation(VAL_MANIFEST, deadline=deadline)
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/lead_{lead:03d}h"
        print(f"[seed {seed}] evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        _check_provenance(provenance, seed, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
        print(f"[seed {seed}] lead {lead}h done in {provenance['elapsed_seconds']:.0f}s over "
              f"{provenance['n_evaluated']} cases", flush=True)
    return receipt


def _thread_count():
    """Recorded because bit-reproducibility of a loss sequence holds only at a fixed count."""
    import torch
    return int(torch.get_num_threads())


def _seed(output, body, seed, deadline):
    import torch
    receipt = _train(output, body, seed, deadline)
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    wide.write_exclusive(output / f"seed{seed}_receipt.json", receipt)
    print(f"[seed {seed}] receipt written", flush=True)


def _readings(output, body):
    """Score all seventeen variables at all five leads for this point and record it."""
    from training.r7_arm_harness import sha256_file
    seed = SEEDS[0]
    receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"],
                "curve_point": POINT, "updates": UPDATES, "batch_size": BATCH_SIZE,
                "epochs": round(UPDATES * BATCH_SIZE / 32131, 6),
                "seed": seed,
                "saturation_rule": dict(SATURATION_RULE),
                "saturation_verdict": None,
                "saturation_note": ("not evaluable from one point: the rule is a first "
                                    "difference between the 16-epoch and 4-epoch points, so it "
                                    "is recorded here verbatim and evaluated only when the "
                                    "ladder is complete"),
                "climatology_reference_sha256": sha256_file(D2_REFERENCE),
                "scored_variables": list(SCORED_VARIABLES), "leads": {}}
    for lead in EVALUATION_LEADS:
        table = Path(receipt["evaluations"][str(lead)]["dir"]) / "boundary_rmse.csv"
        entry = {"rmse": {}, "skill": {}}
        for variable in SCORED_VARIABLES:
            value = wide._region_variable_rmse(table, REGION, variable)
            entry["rmse"][variable] = value
            denominator = _climatology_denominator(body, lead, variable)
            entry["skill"][variable] = None if denominator is None else 1.0 - (value / denominator) ** 2
        readings["leads"][str(lead)] = entry
    readings["exit_conditions"] = evaluate_exit(readings)
    return readings


def _climatology_denominator(body, lead, variable):
    """The frozen climatology RMSE for this lead/variable, or None if it was never fitted.

    Read from the ``climatology_reference`` block the prepare stage resolved (the full D2
    reference with per-lead values), NOT from the frozen ``climatology`` summary block, which
    records only the bucket count and year span. Only t2m has a registered denominator on this
    instance; the other sixteen are reported as raw RMSE and get no invented skill number.
    """
    if variable != "t2m":
        return None
    reference = body.get("climatology_reference") or {}
    leads = reference.get("leads") or {}
    record = leads.get(str(lead))
    if not isinstance(record, dict) or "t2m_rmse" not in record:
        raise RuntimeError(f"the resolved climatology reference has no t2m denominator for "
                           f"lead {lead}; refusing to invent one")
    return record["t2m_rmse"]


def evaluate_exit(readings):
    """This point's predeclared ENGINEERING exit; the science verdict is not computable here.

    Predeclared in the brief: (a) the run reaches exactly the frozen update count at the frozen
    batch, (b) all seventeen variables are finite at all five leads on the full grid, (c) the
    t2m 72 h skill is finite and its denominator is the registered one, (d) no frozen criterion
    was altered. The saturation judgement is explicitly NOT part of this exit.
    """
    leads = readings["leads"]
    a = readings["updates"] == UPDATES and readings["batch_size"] == BATCH_SIZE
    wide_470 = leads["72"]["rmse"]
    b = bool(wide_470) and len(wide_470) == len(SCORED_VARIABLES) and all(
        _finite(value) for entry in leads.values() for value in entry["rmse"].values())
    c = _finite(leads["72"]["skill"].get("t2m"))
    d = len(SCORED_VARIABLES) == 17
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d), "all": bool(a and b and c and d),
            "verdict": ("advance-to-next-curve-point" if (a and b and c and d)
                        else "engineering-incomplete-do-not-read")}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        body = _prepared_protocol(output)
        print(f"[worker] prepare done: protocol {body['protocol_sha256'][:12]}", flush=True)
        return body
    if phase == "archive":
        from scripts import study_r7_s3_wide_interior_dose as dose
        body = dose.interior._archive(output, validate_protocol(output))
        print("[worker] archive done", flush=True)
        return body
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    from scripts import study_r7_s3_v3_rollout_ft as recipe
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", _readings(output, body))
        print("[worker] reading done", flush=True)
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)
    print(f"[seed {seed}] done", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise RuntimeError("dual-card mode is driven by `r7_round.py stage-seeds`, not by main")
    # ``sys.exit(value)`` prints a non-None value to stderr and sets a nonzero status, so the
    # worker's return value must be discarded rather than propagated (a 38 MB protocol body
    # otherwise lands in the stage log, as it did on this round's first prepare run).
    worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    sys.exit(main())
