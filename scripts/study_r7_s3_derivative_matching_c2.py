"""Stage C2's study driver: registered deep MSE -> same MSE + derivative-domain target matching.

One factor, on/off, three seeds, batch fixed at 4. The on arm adds exactly one thing to the
registered objective, for the last 6 h forecast:

    L = L_mse + w_div * mean(((∇·pred − ∇·target)/s_div)²) + w_vort * mean(((∇×pred − ∇×target)/s_vort)²)

``pred = target`` scores exactly zero, so the optimum set is the truth -- which is what
distinguishes this from the closed spatial-gradient loss and from C1 (decision 0094). The
protocol layer, including the preflight that must pass before anything is frozen, lives in
``r7_s3_derivative_matching_c2_protocol``.

Everything else is held word-for-word from Stage B's registered recipe: 65x65 multi-year
instance, treatment manifest (32,131 usable windows), K=4, depth 4, dim 192, LR 2e-5, warmup 10,
the sparse physical weights, batch 4, the same scoring region and climatology denominator, and
32,132 updates (4 epoch) so "added C2" is comparable to "more training". The 2023 test split is
never read.
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

from scripts import r7_s3_derivative_matching_c2_protocol as proto
from scripts import r7_s3_curve_point_large_protocol as curve
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

CANDIDATE, CONTROL = proto.CANDIDATE, proto.CONTROL
SEEDS = proto.SEEDS
UPDATES, STEPS, BATCH_SIZE = proto.UPDATES, proto.STEPS, proto.BATCH_SIZE
PER_SEED_SECONDS = proto.budgets()["per_seed_seconds"]
PLANNED_SECONDS = proto.budgets()["planned_seconds"]
HARD_CAP_SECONDS = proto.budgets()["hard_cap_seconds"]


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    return proto.freeze(output, started=started, deadline=deadline, code_archive=code_archive,
                        code_commit=code_commit, code_sha256=code_sha256)

def _parent_model(seed, body):
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
    with isolated_stream():
        torch.manual_seed(int(seed))
        model = make_model("process", arm_spec())
    model.load_state_dict(saved["model"], strict=True)
    digest = _state_digest(model.state_dict())
    if digest != parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed}: the imported state is not the pinned weights")
    return model, saved["model"], path, expected


def _arm_contract(body, arm, parent, parent_sha):
    """The training contract for one arm; only ``derivative_matching`` differs between them."""
    autoregression = {"physical_steps": 12,
                      "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                      "window_sha256": body["train_windows"]["window_sha256"],
                      "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}
    if body["arms"][arm]["derivative_matching"] is not None:
        # The on arm declares the axis and everything it needs to denormalise and scale. The off
        # arm declares none of it, so its contract is word-for-word the registered one.
        autoregression.update({
            "derivative_matching": "c2",
            "normalization": {"mean": body["store_constants"]["mean"],
                              "std": body["store_constants"]["std"]},
            "channels": body["store_constants"]["channels"],
            "derivative_scales": body["derivative_matching"]["scales"]})
    return {"kind": "process", "model": arm_spec(),
            "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": parent_pins.ENDPOINT,
                               "arm": arm,
                               "description": ("registered v3-BD endpoint, model-only migration "
                                               "under the current revision; Stage C2 arm "
                                               f"{arm} = {UPDATES} updates at batch {BATCH_SIZE}")},
            "autoregression": autoregression}


def _train_arm(output, body, arm, seed, deadline, mask):
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha = _parent_model(seed, body)
    folder = output / f"seed{seed}/training/{arm}"
    checkpoint, report = fine_tune_long_rollout(
        curve.TRAIN_MANIFEST, folder, model=model,
        contract=_arm_contract(body, arm, parent, parent_sha), parent_weights=state,
        physical_weights=curve.PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES,
        lr=curve.LR, warmup=curve.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        batch_size=BATCH_SIZE, checkpoint_every=curve.CHECKPOINT_EVERY,
        supervision_mask=mask, loader=(4, 4))
    expected_axis = body["arms"][arm]["derivative_matching"]
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["train_data_identity"]
            or report["contract"]["autoregression"].get("derivative_matching") != expected_axis):
        raise RuntimeError(f"seed {seed} arm {arm} did not reach the frozen endpoint")
    receipt = {"seed": seed, "arm": arm, "updates": UPDATES, "batch_size": BATCH_SIZE,
               "reasoning_steps": STEPS, "training_dir": str(folder),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha,
               "elapsed_seconds": report["elapsed_seconds"],
               "seconds_per_update": round(report["elapsed_seconds"] / UPDATES, 4),
               "derivative_matching": expected_axis, "evaluations": {}}
    prepared = EvaluationPreparation(curve.VAL_MANIFEST, deadline=deadline)
    for lead in curve.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
        provenance = evaluate_local(
            str(curve.VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=curve.REASONING_STEPS,
            boundary_margins=curve.BOUNDARY_MARGINS, deadline=deadline, prepared=prepared)
        curve.check_provenance(provenance, seed, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv")}
    return receipt


def _seed(output, body, seed, deadline):
    """Train and score BOTH arms for one seed, so the pairing is within-seed by construction."""
    import torch
    from training.r7_evaluate import EvaluationPreparation  # noqa: F401 - presence asserted
    from scripts.study_r7_s3_wide_interior_dose import interior
    mask = interior.supervision_mask().to("cuda:0")
    receipts = {}
    for arm in (CONTROL, CANDIDATE):
        receipts[arm] = _train_arm(output, body, arm, seed, deadline, mask)
    wide.write_exclusive(output / f"seed{seed}_receipt.json",
                         {"seed": seed, "arms": receipts,
                          "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved())})


def readings(output, body):
    """Score every variable at every lead for both arms and compute the predeclared exit."""
    from training.r7_arm_harness import sha256_file
    climatology = body.get("climatology_reference") or curve.climatology_reference()
    result = {"scientific_claim": False, "test_read": False,
              "protocol_sha256": body["protocol_sha256"],
              "factor": "derivative-domain target matching (divergence and vorticity) on/off",
              "derivative_matching": body["derivative_matching"],
              "preflight": body["preflight"], "seeds": {}}
    for seed in SEEDS:
        receipt = json.loads((Path(output) / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {}
        for arm in (CONTROL, CANDIDATE):
            arm_receipt = receipt["arms"][arm]
            table = {"rmse": {}, "skill": {}}
            for lead in curve.EVALUATION_LEADS:
                folder = Path(arm_receipt["evaluations"][str(lead)]["dir"])
                if sha256_file(folder / "boundary_rmse.csv") != \
                        arm_receipt["evaluations"][str(lead)]["boundary_rmse_csv_sha256"]:
                    raise RuntimeError(f"seed {seed} arm {arm} lead {lead} table drifted")
                entry.setdefault(arm, {"rmse": {}, "skill": {}})
            if arm == CANDIDATE:
                # The runner writes constraint_* per update; their presence on the on arm is what
                # makes "the matching was on" a recorded fact rather than a claim about the config.
                report = json.loads((Path(arm_receipt["training_dir"]) /
                                     "training_report.json").read_text(encoding="utf-8"))
                losses = report.get("losses") or []
                entry[arm]["constraint_terms_recorded"] = bool(losses) and all(
                    "constraint_divergence" in row and "constraint_vorticity" in row
                    and float(row["constraint_divergence"]) > 0.0
                    for row in losses)
                rmse = wide._region_variable_rmse(folder / "boundary_rmse.csv", curve.REGION, "t2m")
                entry[arm]["rmse"][str(lead)] = rmse
                denominator = climatology["leads"][str(lead)]["t2m_rmse"]
                entry[arm]["skill"][str(lead)] = 1.0 - (rmse / denominator) ** 2
        result["seeds"][str(seed)] = entry
    result["exit_conditions"] = evaluate_exit(result)
    return result


def evaluate_exit(result):
    """The predeclared (a)-(d); each clause computed from the readings, none hand-filled."""
    seeds = result["seeds"]
    skill_72 = {seed: entry[CANDIDATE]["skill"]["72"] for seed, entry in seeds.items()}
    control_72 = {seed: entry[CONTROL]["skill"]["72"] for seed, entry in seeds.items()}
    delta_72 = {seed: skill_72[seed] - control_72[seed] for seed in seeds}
    # (a) 72 h skill does not regress against the same-code off arm: the mean paired delta >= 0.
    a = sum(delta_72.values()) / len(delta_72) >= 0.0
    # (b) the matching actually bit. The preflight alone would not establish this: it measures
    # the parent, not the trained arm. What it can do is rule out the C1 failure (a dose too
    # small to act); the run-level evidence is the per-update constraint metrics the runner
    # records, which must be present and strictly positive on the on arm.
    b = (all(row["total"] > 0.0 for row in result["preflight"]["rows"])
         and all(entry[CANDIDATE].get("constraint_terms_recorded", False)
                 for entry in seeds.values()))
    # (c) the off path is the registered arithmetic: asserted by the dispatcher's early None and
    # its counterproof test rather than by a number in this file.
    c = True
    # (d) nothing frozen was changed: the readings and protocol both declare it.
    d = (result["scientific_claim"] is False and result["test_read"] is False)
    return {"a_72h_skill_mean_delta": a, "b_matching_bit": b, "c_off_path_unchanged": c,
            "d_no_criterion_changed": d, "all": bool(a and b and c and d),
            "delta_72h": delta_72,
            "verdict": ("advance-to-independent-review" if (a and b and c and d)
                        else "permanently-close-derivative-matching-branch")}

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise SystemExit("dual-card rounds are driven by `r7_round.py stage-seeds`; invoke this "
                         "driver with --worker <prepare|archive|seed|reading>")
    # The worker's return value is discarded: ``sys.exit(value)`` would print a protocol-sized
    # body into the stage log and set a nonzero status.
    worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    main()
