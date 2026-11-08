"""Stage E1b driver: final-only draft supervision vs the registered deep form, three seeds.

One seed runs **both** arms, so the pairing is within-seed by construction and the only thing that
differs is the weight vector. The reading stage then computes the predeclared (a)-(d) from the
receipts, and the exit is whatever those numbers say -- including a close, which decision 0096
commits to recording verbatim.

This driver is deliberately thin over the C2 driver's proven shape: the same pinned parent, the
same full-grid supervision assertion, the same digest-pinned lead tables, the same exclusive
writes. What it does *not* reuse is C2's protocol, which asserts its own axis name and would have
to be gutted to serve here (decision 0099's note on duplication).
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

from scripts import r7_s3_e1b_final_only_supervision_protocol as proto
from scripts import r7_s3_curve_point_large_protocol as curve
from scripts import r7_s3_curve_point_large_driver as curve_driver
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

CANDIDATE, CONTROL = proto.CANDIDATE, proto.CONTROL
SEEDS = proto.SEEDS
UPDATES, STEPS, BATCH_SIZE = proto.UPDATES, proto.STEPS, proto.BATCH_SIZE
#: The five scoring leads live on ``wide``. Writing ``curve.EVALUATION_LEADS`` raised AttributeError
#: only when the reading stage ran (decision 0098), so the name is bound once, here.
LEADS = wide.EVALUATION_LEADS
PER_SEED_SECONDS = proto.budgets()["per_seed_seconds"]
PLANNED_SECONDS = proto.budgets()["planned_seconds"]
HARD_CAP_SECONDS = proto.budgets()["hard_cap_seconds"]


def _parent_model(seed, body):
    """Load the pinned migrated parent under this revision, asserting its identity."""
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
    with isolated_stream():
        torch.manual_seed(int(seed))
        model = make_model("process", proto.arm_spec())
    model.load_state_dict(saved["model"], strict=True)
    digest = _state_digest(model.state_dict())
    if digest != parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed}: the imported state is not the pinned weights")
    return model, saved["model"], path, expected


def _arm_contract(body, arm, parent, parent_sha):
    """The training contract; only ``draft_supervision`` differs between the two arms.

    The weight vector is bound to the registered sparse value for *both* arms, which is what makes
    this round a test of organisation rather than of allocation (contrast E1a).
    """
    return {"kind": "process", "model": proto.arm_spec(),
            "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {
                "parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                "parent_checkpoint": str(parent),
                "parent_checkpoint_sha256": parent_sha,
                "parent_endpoint_updates": parent_pins.ENDPOINT,
                "arm": arm,
                "description": ("registered v3-BD endpoint, model-only migration under the current "
                                f"revision; Stage E1b arm {arm} = {UPDATES} updates at batch "
                                f"{BATCH_SIZE}")},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(proto.SPARSE_WEIGHTS),
                               "draft_supervision": proto.arm_supervision(arm),
                               "window_sha256": body["train_windows"]["window_sha256"],
                               "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}}


def _train_arm(output, body, arm, seed, deadline):
    """One arm's training and scoring; no mask, because this store is full-grid."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha = _parent_model(seed, body)
    folder = output / f"seed{seed}/training/{arm}"
    checkpoint, report = fine_tune_long_rollout(
        curve.TRAIN_MANIFEST, folder, model=model,
        contract=_arm_contract(body, arm, parent, parent_sha), parent_weights=state,
        physical_weights=list(proto.SPARSE_WEIGHTS), steps=STEPS, updates=UPDATES,
        draft_supervision=proto.arm_supervision(arm),
        lr=curve.LR, warmup=curve.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        batch_size=BATCH_SIZE, checkpoint_every=curve.CHECKPOINT_EVERY,
        supervision_mask=None, loader=(4, 4))
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["train_data_identity"]):
        raise RuntimeError(f"seed {seed} arm {arm} did not reach the frozen endpoint")
    # The single factor, read back from the run itself rather than from the spec we sent. The
    # contract also records ``internal_deep_supervision``, which is the mode's own description of
    # itself; asserting both means a mislabelled arm cannot pass as the other mode.
    autoregression = report["contract"]["autoregression"]
    actual = autoregression.get("draft_supervision")
    if actual != proto.arm_supervision(arm):
        raise RuntimeError(f"seed {seed} arm {arm} trained with draft_supervision={actual!r}, "
                           f"not {proto.arm_supervision(arm)!r}")
    expected_deep = arm == proto.CONTROL
    if autoregression.get("internal_deep_supervision") is not expected_deep:
        raise RuntimeError(f"seed {seed} arm {arm} reports internal_deep_supervision="
                           f"{autoregression.get('internal_deep_supervision')!r}, expected "
                           f"{expected_deep}")
    weights = list(autoregression["physical_weights"])
    if weights != list(proto.SPARSE_WEIGHTS):
        raise RuntimeError(f"seed {seed} arm {arm} moved the weight vector, which E1b must not do")
    if report["supervision"]["kind"] != "full_grid":
        raise RuntimeError(f"seed {seed} arm {arm} supervised on {report['supervision']['kind']}")
    receipt = {"seed": seed, "arm": arm, "updates": UPDATES, "batch_size": BATCH_SIZE,
               "reasoning_steps": STEPS, "training_dir": str(folder),
               "draft_supervision": actual, "physical_weights": weights,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha,
               "elapsed_seconds": report["elapsed_seconds"],
               "seconds_per_update": round(report["elapsed_seconds"] / UPDATES, 4),
               "evaluations": {}}
    prepared = EvaluationPreparation(curve.VAL_MANIFEST, deadline=deadline)
    for lead in LEADS:
        destination = output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
        provenance = evaluate_local(
            str(curve.VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=curve.REASONING_STEPS,
            boundary_margins=curve.BOUNDARY_MARGINS, deadline=deadline, prepared=prepared)
        curve_driver.check_provenance(provenance, seed, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv")}
    return receipt


def _seed(output, body, seed, deadline):
    """Train and score both arms for one seed, so the pairing is within-seed by construction."""
    import torch
    receipts = {}
    for arm in (CONTROL, CANDIDATE):
        receipts[arm] = _train_arm(output, body, arm, seed, deadline)
    wide.write_exclusive(output / f"seed{seed}_receipt.json",
                         {"seed": seed, "arms": receipts,
                          "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved())})


def readings(output, body):
    """Score every lead for both arms and compute the predeclared exit.

    The fill sits in the per-lead loop and outside any arm-conditional branch: nesting it inside
    ``if arm == CANDIDATE`` left the control arm empty and made the exit raise ``KeyError`` only
    after ten hours of training (decision 0098).
    """
    from training.r7_arm_harness import sha256_file
    climatology = body.get("climatology_reference") or curve.climatology_reference()
    result = {"scientific_claim": False, "test_read": False,
              "protocol_sha256": body["protocol_sha256"],
              "factor": "final-only vs deep draft supervision at 16 epoch",
              "arms": {CANDIDATE: body["arms"][CANDIDATE], CONTROL: body["arms"][CONTROL]},
              "seeds": {}}
    for seed in SEEDS:
        receipt = json.loads((Path(output) / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {}
        for arm in (CONTROL, CANDIDATE):
            arm_receipt = receipt["arms"][arm]
            entry[arm] = {"rmse": {}, "skill": {},
                          "draft_supervision": arm_receipt["draft_supervision"]}
            for lead in LEADS:
                folder = Path(arm_receipt["evaluations"][str(lead)]["dir"])
                if sha256_file(folder / "boundary_rmse.csv") != \
                        arm_receipt["evaluations"][str(lead)]["boundary_rmse_csv_sha256"]:
                    raise RuntimeError(f"seed {seed} arm {arm} lead {lead} table drifted")
                rmse = wide._region_variable_rmse(folder / "boundary_rmse.csv", curve.REGION, "t2m")
                denominator = climatology["leads"][str(lead)]["t2m_rmse"]
                entry[arm]["rmse"][str(lead)] = rmse
                entry[arm]["skill"][str(lead)] = 1.0 - (rmse / denominator) ** 2
            if sorted(int(key) for key in entry[arm]["rmse"]) != sorted(LEADS):
                raise RuntimeError(f"seed {seed} arm {arm} scored {sorted(entry[arm]['rmse'])}, "
                                   f"not the frozen leads {sorted(LEADS)}")
        result["seeds"][str(seed)] = entry
    result["exit_conditions"] = evaluate_exit(result, body)
    return result


def evaluate_exit(result, body):
    """The predeclared (a)-(d) of decision 0096, each computed from the readings."""
    seeds = result["seeds"]
    delta_72 = {seed: entry[CANDIDATE]["skill"]["72"] - entry[CONTROL]["skill"]["72"]
                for seed, entry in seeds.items()}
    # (a) 72h t2m skill does not regress against the same-code off arm: paired mean delta >= 0.
    a = sum(delta_72.values()) / len(delta_72) >= 0.0
    # (b) the single factor actually bit: the candidate arm trained as final_only, not deep.
    b = all(entry[CANDIDATE]["draft_supervision"] == proto.DRAFT_SUPERVISION_FINAL_ONLY
            for entry in seeds.values())
    # (c) the off arm is the registered deep form verbatim.
    c = all(entry[CONTROL]["draft_supervision"] == proto.DRAFT_SUPERVISION_DEEP
            for entry in seeds.values())
    # (d) nothing frozen was changed.
    d = (result["scientific_claim"] is False and result["test_read"] is False
         and body.get("test_read") is False)
    return {"a_72h_skill_mean_delta": a, "b_factor_effective": b, "c_control_is_registered": c,
            "d_no_criterion_changed": d, "all": bool(a and b and c and d),
            "delta_72h": delta_72,
            "delta_72h_mean": sum(delta_72.values()) / len(delta_72),
            "verdict": ("advance-to-independent-review" if (a and b and c and d)
                        else "permanently-close-final-only-supervision-in-this-regime")}


def worker(output, phase, seed, deadline):
    """One stage of this round, matching the wrapper's ``--worker`` contract."""
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        return proto.prepared_protocol(output)
    if phase == "archive":
        from scripts.study_r7_s3_wide_interior_dose import interior
        return interior._archive(output, proto.validate_protocol(output))
    body = proto.validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    from scripts import study_r7_s3_v3_rollout_ft as recipe
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", readings(output, body))
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise SystemExit("dual-card rounds are driven by `r7_round.py stage-seeds`; invoke this "
                         "driver with --worker <prepare|archive|seed|reading>")
    worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    main()
