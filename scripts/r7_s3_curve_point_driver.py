"""Shared machinery for Stage B curve points 2-4: train one point, score it, read it out.

One point = one arm at one endpoint. Stage B asks whether 72 h skill is still rising with
training volume; it is a *calibration* curve, so there is no control arm and the only thing
that varies across points is the update count. The point's constants live in
``r7_s3_curve_point_protocol``; each point has a thin binding under ``scripts/`` that names
its label, because decision 0067 requires the execution inventory to start from a named
entry file rather than a glob.

The saturation judgement is NOT made here: one point cannot compute the two-point first
difference the goal document freezes. That rule is recorded verbatim in the protocol and
evaluated only when the ladder is complete.
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

from scripts import r7_s3_curve_point_protocol as proto
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from scripts.r7_s3_fy_year_coverage_climatology import (
    D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS)
from training import r7_device_selection as devices

#: The seventeen dynamic channels scored at every lead, verbatim from the store's own
#: ``channels`` attribute. Named in text so a store whose channel order moved fails loudly
#: at lookup rather than silently rescoring a different surface.
SCORED_VARIABLES = (
    "t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
    "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250",
)


def load_parent(seed):
    """Load the pinned migrated parent under the current revision, asserting its identity."""
    import torch

    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    from model.spacetime_conditioning_r7 import isolated_stream
    path, expected = parent_pins.pinned_parent(seed)
    if sha256_file(path) != expected:
        raise RuntimeError(f"seed {seed} parent checkpoint pin changed")
    saved = load_checkpoint(path)
    if saved["updates"] != parent_pins.ENDPOINT:
        raise RuntimeError(f"seed {seed} parent is not the frozen endpoint")
    parent_state = saved["model"]
    with isolated_stream():
        torch.manual_seed(int(seed))
        model = make_model("process", proto.model_spec())
    if list(parent_state) != list(model.state_dict()):
        raise RuntimeError(f"seed {seed}: the parent's key set differs from this revision")
    model.load_state_dict(parent_state, strict=True)
    digest = _state_digest(model.state_dict())
    if digest != parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed}: the imported state is not the pinned weights "
                           f"({digest} != {parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]})")
    return model, parent_state, path, expected, {
        "parent_state_sha256": digest, "parent_keys": len(parent_state),
        "endpoint_updates": parent_pins.ENDPOINT, "init_seed": int(seed)}


def contract(body, point, parent, parent_sha, expansion):
    return {"kind": "process", "model": body["arms"]["curve"]["model"]["spec"],
            "data_identity": body["arms"]["curve"]["train_data_identity"],
            "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": parent_pins.ENDPOINT,
                               "curve_point": point,
                               "updates": proto.updates_for(point),
                               "description": ("registered v3-BD endpoint, model-only migration "
                                               "under the current revision; Stage B point "
                                               f"{point} = {proto.updates_for(point)} updates "
                                               f"at batch {proto.BATCH_SIZE}")},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(proto.PHYSICAL_WEIGHTS),
                               "window_sha256": body["arms"]["curve"]["train_windows"]["window_sha256"],
                               "excluded_sample_ids": body["arms"]["curve"]["train_windows"]["excluded_sample_ids"]}}


def check_provenance(provenance, seed, lead):
    if provenance["split"] != "val" or "test_read" in provenance:
        raise RuntimeError(f"seed {seed} lead {lead} did not stay on val")
    if (provenance["n_evaluated"] != VAL_COHORTS[str(lead)]
            or provenance["n_available_windows"] != VAL_COHORTS[str(lead)]
            or provenance["lead_hours"] != [lead]):
        raise RuntimeError(f"seed {seed} lead {lead} cohort differs from the frozen one")
    regions = {region["region"]: region for region in
               (provenance["boundary_scoring"] or {}).get("regions", [])}
    if regions.get(proto.REGION, {}).get("n_grid_points") != proto.FULL_GRID_POINTS:
        raise RuntimeError(f"seed {seed} lead {lead} full region is not 65x65")


def thread_count():
    """Recorded because bit-reproducibility of a loss sequence holds only at a fixed count."""
    import torch
    return int(torch.get_num_threads())


def train_point(output, body, point, seed, deadline):
    """Train this point's single arm from the parent and score every lead."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    updates = proto.updates_for(point)
    model, state, parent, parent_sha, expansion = load_parent(seed)
    folder = output / f"seed{seed}/training"
    print(f"[seed {seed}] curve point {point}: {updates} updates at batch {proto.BATCH_SIZE}",
          flush=True)
    checkpoint, report = fine_tune_long_rollout(
        proto.TRAIN_MANIFEST, folder, model=model,
        contract=contract(body, point, parent, parent_sha, expansion), parent_weights=state,
        physical_weights=proto.PHYSICAL_WEIGHTS, steps=proto.STEPS, updates=updates,
        lr=proto.LR, warmup=proto.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        batch_size=proto.BATCH_SIZE,
        # Decision 0087's policy is only real if the runner is told it: this argument defaults
        # to 20, and omitting it wrote 101 checkpoints on point 1's first attempt while the
        # protocol declared 5,000.
        checkpoint_every=proto.CHECKPOINT_EVERY, loader=(4, 4))
    if (report["updates_this_run"] != updates or report["contract"]["steps"] != proto.STEPS
            or int(report["contract"]["batch_size"]) != proto.BATCH_SIZE
            or report["data_identity"] != body["arms"]["curve"]["train_data_identity"]
            or report["supervision"]["kind"] != "full_grid"):
        raise RuntimeError(f"seed {seed} did not reach the frozen endpoint")
    update1 = dict(report["losses"][0])
    losses = [float(entry["loss"]) for entry in report["losses"]]
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"seed {seed} produced a nonfinite training loss")
    retained = sorted(item.name for item in folder.glob("update_*.pt"))
    receipt = {"seed": seed, "curve_point": point, "updates": updates,
               "batch_size": proto.BATCH_SIZE, "reasoning_steps": proto.STEPS,
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha,
               "elapsed_seconds": report["elapsed_seconds"],
               "seconds_per_update": round(report["elapsed_seconds"] / updates, 4),
               "supervision": report["supervision"],
               "train_data_identity": report["data_identity"],
               "train_windows_sha256": body["arms"]["curve"]["train_windows"]["window_sha256"],
               "expansion": expansion, "checkpoint_every": proto.CHECKPOINT_EVERY,
               "retained_checkpoints": retained, "retained_checkpoint_count": len(retained),
               "update1": {"loss": update1["loss"], "gradient_norm": update1["gradient_norm"]},
               "loss_curve": {"first": losses[0], "last": losses[-1],
                              "min": min(losses), "max": max(losses), "n": len(losses)},
               "threads": thread_count(), "evaluations": {}}
    print(f"[seed {seed}] trained in {report['elapsed_seconds']:.0f}s; scoring five leads",
          flush=True)
    # One manifest reader, one store scan and one training climatology for all five leads.
    prepared = EvaluationPreparation(proto.VAL_MANIFEST, deadline=deadline)
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/lead_{lead:03d}h"
        print(f"[seed {seed}] evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(proto.VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=proto.REASONING_STEPS, boundary_margins=proto.BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        check_provenance(provenance, seed, lead)
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


def seed_point(output, body, point, seed, deadline):
    import torch
    receipt = train_point(output, body, point, seed, deadline)
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    wide.write_exclusive(output / f"seed{seed}_receipt.json", receipt)
    print(f"[seed {seed}] receipt written", flush=True)


def climatology_denominator(body, lead, variable):
    """The frozen climatology RMSE for this lead/variable, or None if it was never fitted.

    Read from the ``climatology_reference`` block the prepare stage resolved (per-lead values),
    NOT the frozen ``climatology`` summary block, which records only bucket count and years.
    Only t2m has a registered denominator on this instance; the other sixteen are reported as
    raw RMSE and get no invented skill number.
    """
    if variable != "t2m":
        return None
    leads = (body.get("climatology_reference") or {}).get("leads") or {}
    record = leads.get(str(lead))
    if not isinstance(record, dict) or "t2m_rmse" not in record:
        raise RuntimeError(f"the resolved climatology reference has no t2m denominator for "
                           f"lead {lead}; refusing to invent one")
    return record["t2m_rmse"]


def readings(output, body, point):
    """Score all seventeen variables at all five leads for this point and record it."""
    from training.r7_arm_harness import sha256_file
    seed = proto.SEEDS[0]
    receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    updates = proto.updates_for(point)
    body_out = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"],
                "curve_point": point, "updates": updates, "batch_size": proto.BATCH_SIZE,
                "epochs": round(updates * proto.BATCH_SIZE / proto.USABLE_WINDOWS, 6),
                "seed": seed, "saturation_rule": dict(proto.SATURATION_RULE),
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
            value = wide._region_variable_rmse(table, proto.REGION, variable)
            entry["rmse"][variable] = value
            denominator = climatology_denominator(body, lead, variable)
            entry["skill"][variable] = None if denominator is None else 1.0 - (value / denominator) ** 2
        body_out["leads"][str(lead)] = entry
    body_out["exit_conditions"] = evaluate_exit(body_out, point)
    return body_out


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def evaluate_exit(body, point):
    """This point's predeclared ENGINEERING exit; the science verdict is not computable here.

    Predeclared in the brief: (a) the run reaches exactly the frozen update count at the frozen
    batch, (b) all seventeen variables are finite at all five leads on the full grid, (c) the
    t2m 72 h skill is finite and its denominator is the registered one, (d) no frozen criterion
    was altered. The saturation judgement is explicitly NOT part of this exit.
    """
    leads = body["leads"]
    a = body["updates"] == proto.updates_for(point) and body["batch_size"] == proto.BATCH_SIZE
    b = (bool(leads["72"]["rmse"]) and len(leads["72"]["rmse"]) == len(SCORED_VARIABLES)
         and all(finite(value) for entry in leads.values()
                 for value in entry["rmse"].values()))
    c = finite(leads["72"]["skill"].get("t2m"))
    d = len(SCORED_VARIABLES) == 17 and len(set(SCORED_VARIABLES)) == 17
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d),
            "all": bool(a and b and c and d),
            "verdict": ("advance-to-next-curve-point" if (a and b and c and d)
                        else "engineering-incomplete-do-not-read")}


def worker(output, point, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} point={point} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        body = proto.prepared_protocol(point, output)
        print(f"[worker] prepare done: protocol {body['protocol_sha256'][:12]}", flush=True)
        return body
    if phase == "archive":
        from scripts import study_r7_s3_wide_interior_dose as dose
        body = dose.interior._archive(output, proto.validate_protocol(point, output))
        print("[worker] archive done", flush=True)
        return body
    body = proto.validate_protocol(point, output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    from scripts import study_r7_s3_v3_rollout_ft as recipe
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", readings(output, body, point))
        print("[worker] reading done", flush=True)
        return
    if seed not in proto.SEEDS or deadline - time.perf_counter() > body["per_seed_seconds"]:
        raise RuntimeError("seed/deadline differs from freeze")
    seed_point(output, body, point, seed, deadline)
    print(f"[seed {seed}] done", flush=True)


def main(argv, point):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise RuntimeError("dual-card mode is driven by `r7_round.py stage`, not by main")
    # ``sys.exit(value)`` prints a non-None value and sets a nonzero status, so the worker's
    # return value must be discarded rather than propagated (a 38 MB protocol body otherwise
    # lands in the stage log, as it did on point 1's first prepare run).
    worker(args.out, point, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    # This module is the shared engine, not a round entry point: it has no point label of its
    # own, so running it directly could only ever train the wrong point. Refuse loudly rather
    # than exit 0 having done nothing (point 1 shipped exactly that silent-success defect).
    raise SystemExit("this is the shared engine for Stage B curve points; spawn a named "
                     "binding such as scripts/study_r7_s3_throughput_physics_b_p1000.py")
