"""Stage A's study driver: registered deep MSE -> same recipe + beta_j AMSE-analogue weighting.

One factor, on/off, three seeds, batch fixed at 4, 4,000 updates from the p1600 endpoint.

    off: ``spectral_geometry: None``  -- the registered recipe, word for word
    on : ``spectral_geometry: 'amse-analogue'`` + the frozen ``beta_j`` from the protocol

The on arm's loss is FastNet equation (10)'s shape as read from arXiv:2509.17601 §4.1.3
(2026-11-12): ``sum_j beta_j AMSE_j`` over the 17 variables, per forecast step, averaged over
steps -- no ``1/J``, no parallel per-gridpoint MSE inside the term. Everything else is held
from Stage B's registered recipe.

The protocol layer (beta_j freeze, the (e) preflight, freeze/prepare/validate) lives in
``r7_s3_spectral_geometry_a_protocol``. The 2023 test split is never read.
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

from scripts import r7_s3_spectral_geometry_a_protocol as proto
from scripts import r7_s3_curve_point_large_protocol as curve
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

CANDIDATE, CONTROL = proto.CANDIDATE, proto.CONTROL
SEEDS = proto.SEEDS
UPDATES, STEPS, BATCH_SIZE = proto.UPDATES, proto.STEPS, proto.BATCH_SIZE
LEADS = proto.LEADS
PER_SEED_SECONDS = proto.PER_SEED_SECONDS
PLANNED_SECONDS = proto.PLANNED_SECONDS
HARD_CAP_SECONDS = proto.HARD_CAP_SECONDS


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
        model = make_model("process", proto.arm_spec())
    model.load_state_dict(saved["model"], strict=True)
    digest = _state_digest(model.state_dict())
    if digest != parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed}: the imported state is not the pinned weights")
    return model, saved["model"], path, expected


def _arm_contract(body, arm, parent, parent_sha):
    """The training contract for one arm; only ``spectral_geometry`` differs between them."""
    autoregression = {"physical_steps": 12,
                      "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                      "window_sha256": body["train_windows"]["window_sha256"],
                      "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}
    if body["arms"][arm]["spectral_geometry"] is not None:
        # The on arm declares the axis and the frozen per-variable weight. The off arm declares
        # neither, so its contract is word-for-word the registered one.
        autoregression.update({
            "spectral_geometry": "amse-analogue",
            "beta_weights": list(body["beta_weights"]["values"]),
            "beta_channels": list(body["beta_weights"]["channels"])})
    return {"kind": "process", "model": proto.arm_spec(),
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
                                               "under the current revision; Stage A arm "
                                               f"{arm} = {UPDATES} updates at batch {BATCH_SIZE}")},
            "autoregression": autoregression}


def _train_arm(output, body, arm, seed, deadline):
    """One arm's training and five-lead scoring."""
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
        supervision_mask=None, loader=(4, 4))
    expected_axis = body["arms"][arm]["spectral_geometry"]
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["train_data_identity"]
            or report["contract"]["autoregression"].get("spectral_geometry") != expected_axis):
        raise RuntimeError(f"seed {seed} arm {arm} did not reach the frozen endpoint")
    if report["supervision"]["kind"] != "full_grid":
        raise RuntimeError(f"seed {seed} arm {arm} supervised on "
                           f"{report['supervision']['kind']}, not the registered full grid")
    receipt = {"seed": seed, "arm": arm, "updates": UPDATES, "batch_size": BATCH_SIZE,
               "reasoning_steps": STEPS, "training_dir": str(folder),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha,
               "elapsed_seconds": report["elapsed_seconds"],
               "seconds_per_update": round(report["elapsed_seconds"] / UPDATES, 4),
               "spectral_geometry": expected_axis, "evaluations": {}}
    prepared = EvaluationPreparation(curve.VAL_MANIFEST, deadline=deadline)
    for lead in LEADS:
        destination = output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
        provenance = evaluate_local(
            str(curve.VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=curve.REASONING_STEPS,
            boundary_margins=curve.BOUNDARY_MARGINS, deadline=deadline, prepared=prepared)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv")}
    return receipt


def _seed(output, body, seed, deadline):
    """Train and score BOTH arms for one seed, so the pairing is within-seed by construction."""
    import torch
    receipts = {}
    for arm in (CONTROL, CANDIDATE):
        receipts[arm] = _train_arm(output, body, arm, seed, deadline)
    wide.write_exclusive(output / f"seed{seed}_receipt.json",
                         {"seed": seed, "arms": receipts,
                          "owned_cuda_reserved_peak_bytes": int(torch.cuda.max_memory_reserved())})


def _amplitude_rows(output, body, seed):
    """Per-band amplitude ratio and coherence for both arms on the frozen cases.

    This is exit (a)'s direct reading: the AMSE term is supposed to raise fine-band amplitude,
    so ``amp_ratio`` relative to the same seed's off arm is the point where the term either acts
    or does not. The cases are the eight Stage P1 measured, so the two rounds' numbers are
    comparable; they are train-manifest cases because ``ZarrLongRolloutDataset`` refuses any
    other manifest, and that in-sample choice is recorded in the protocol and limitations.
    """
    import numpy as np
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_long_rollout import training_long_rollout
    from training import r7_spectral_geometry as geometry
    from data.r7_long_rollout_dataset import ZarrLongRolloutDataset
    import torch

    cases = list(body["amplitude_cases"]["cases"])
    windows = body["train_windows"]
    dataset = ZarrLongRolloutDataset(
        curve.TRAIN_MANIFEST, physical_steps=len(curve.PHYSICAL_WEIGHTS),
        expected_exclusions=windows["excluded_sample_ids"],
        expected_window_sha256=windows["window_sha256"])
    edges = geometry.band_edges(65, 65)
    receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    arms = {}
    for arm in (CONTROL, CANDIDATE):
        saved = load_checkpoint(receipt["arms"][arm]["checkpoint"])
        with torch.random.fork_rng(devices=[]):
            model = make_model("process", proto.arm_spec())
        model.load_state_dict(saved["model"], strict=True)
        spec = proto.arm_spec()
        if float(spec.get("dropout", -1.0)) != 0.0:
            raise ValueError(f"frozen spec dropout is {spec.get('dropout')!r}, not 0.0")
        model.train()
        per_case = []
        for case in cases:
            batch = {key: value.unsqueeze(0) for key, value in dataset[case].items()
                     if torch.is_tensor(value)}
            result = training_long_rollout(model, batch, curve.REASONING_STEPS,
                                           physical_weights=curve.PHYSICAL_WEIGHTS,
                                           supervision_mask=None)
            forecasts = result.forecasts[0].detach()
            targets = batch["physical_targets"][0]
            leads = []
            for step in range(forecasts.shape[0]):
                leads.append({"lead_hours": 6 * (step + 1),
                              "bands": geometry.band_statistics(forecasts[step], targets[step], edges)})
            per_case.append({"case": case, "leads": leads})
        arms[arm] = _band_aggregate(per_case)
    return {"seed": seed,
            "bands": {arm: arms[arm] for arm in (CONTROL, CANDIDATE)},
            "fine_band_amp_ratio": {
                arm: float(np.mean([row["amp_ratio"] for row in arms[arm]
                                    if row["band"] >= len(arms[arm]) // 2]))
                for arm in (CONTROL, CANDIDATE)}}


def _band_aggregate(per_case) -> list[dict]:
    """Mean amplitude ratio and coherence per band, aggregated in the power domain.

    Summing PSDs before taking the ratio (rather than averaging per-case ratios) keeps a band
    whose target is nearly empty from contributing a wild ratio with full weight -- the same
    convention Stage P1's gate uses, so the two rounds' numbers are comparable.
    """
    import numpy as np
    bands: dict[int, list[dict]] = {}
    for case in per_case:
        for lead in case["leads"]:
            if lead["lead_hours"] not in proto.FROZEN_LEADS:
                continue
            for row in lead["bands"]:
                bands.setdefault(row["band"], []).append(row)
    rows = []
    for band in sorted(bands):
        entries = bands[band]
        sx = float(np.sum([entry["psd_pred"] for entry in entries]))
        sy = float(np.sum([entry["psd_target"] for entry in entries]))
        sn = float(np.sum([entry["cross"] for entry in entries]))
        ratio = float(np.sqrt(sx / sy)) if sy > 0 else float("nan")
        coh = float(sn / np.sqrt(sx * sy)) if sx > 0 and sy > 0 else float("nan")
        rows.append({"band": band, "amp_ratio": ratio, "coherence": coh,
                     "gap": 1.0 - ratio if ratio == ratio else float("nan")})
    return rows


def readings(output, body):
    """Score every variable at every lead for both arms and compute the predeclared exits."""
    from training.r7_arm_harness import sha256_file
    climatology = body.get("climatology_reference") or curve.climatology_reference()
    result = {"scientific_claim": False, "test_read": False,
              "protocol_sha256": body["protocol_sha256"],
              "factor": "beta_j AMSE-analogue normalisation on the spectral_geometry axis",
              "beta_weights": body["beta_weights"], "preflight": body["preflight"],
              "seeds": {}, "amplitude": {}}
    for seed in SEEDS:
        receipt = json.loads((Path(output) / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {}
        for arm in (CONTROL, CANDIDATE):
            arm_receipt = receipt["arms"][arm]
            entry[arm] = {"rmse": {}, "skill": {}}
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
            if arm == CANDIDATE:
                report = json.loads((Path(arm_receipt["training_dir"]) /
                                     "training_report.json").read_text(encoding="utf-8"))
                losses = report.get("losses") or []
                # The on arm must have recorded the weighted term per update, or "the axis was
                # on" is a claim about the config rather than a fact about the run.
                entry[arm]["beta_weighted_terms_recorded"] = bool(losses) and all(
                    "constraint_beta_weighted" in row and float(row["constraint_beta_weighted"]) > 0.0
                    for row in losses)
                entry[arm]["constraint_terms_recorded"] = bool(losses) and all(
                    "constraint_spectral_geometry" in row for row in losses)
        result["seeds"][str(seed)] = entry
        result["amplitude"][str(seed)] = _amplitude_rows(output, body, seed)
    result["exit_conditions"] = evaluate_exit(result)
    return result


def evaluate_exit(result):
    """The predeclared (a)-(e); each clause computed from the readings, none hand-filled."""
    seeds = result["seeds"]
    # (a) fine-band amp_ratio rises against the SAME seed's off arm.
    amp = {}
    for seed in seeds:
        on = result["amplitude"][seed]["fine_band_amp_ratio"][CANDIDATE]
        off = result["amplitude"][seed]["fine_band_amp_ratio"][CONTROL]
        amp[seed] = {"on": on, "off": off, "delta": on - off, "raised": on > off}
    a = all(entry["raised"] for entry in amp.values())
    # (b) 72 h t2m skill paired delta >= 0 for all three seeds.
    delta_72 = {seed: entry[CANDIDATE]["skill"]["72"] - entry[CONTROL]["skill"]["72"]
                for seed, entry in seeds.items()}
    b = all(delta >= 0.0 for delta in delta_72.values())
    # (c) 24/48 h paired delta >= 0 for all three seeds (both leads, every seed).
    delta_mid = {lead: {seed: entry[CANDIDATE]["skill"][lead] - entry[CONTROL]["skill"][lead]
                        for seed, entry in seeds.items()}
                 for lead in ("24", "48")}
    c = all(delta >= 0.0 for per_lead in delta_mid.values() for delta in per_lead.values())
    # (e) preflight: beta_j not all equal, and amse finite on all 8 cases x 12 steps.
    pre = result["preflight"]
    beta = result["beta_weights"]["values"]
    e = (len(set(beta)) == len(beta)
         and len(pre["rows"]) == len(pre["cases"])
         and all(len(row["steps"]) == 12 for row in pre["rows"])
         and all(step["all_finite"] for row in pre["rows"] for step in row["steps"]))
    # (d) is NOT evaluated here on purpose. Decision 0109: a literal `True` is not a criterion.
    # The off path's bit-equivalence is carried by the dispatcher's early `None` return and by
    # ``tests/test_r7_s3_spectral_geometry_a.py::test_off_arm_matches_the_registered_arithmetic``,
    # which fails if the dispatcher ever constructs the axis when it is unset.
    return {"a_fine_band_amp_raised": a, "a_per_seed": {str(k): v for k, v in amp.items()},
            "b_72h_skill_nonnegative": b, "b_delta_72h": {str(k): v for k, v in delta_72.items()},
            "c_mid_skill_nonnegative": c,
            "c_delta_mid": {lead: {str(k): v for k, v in per.items()}
                            for lead, per in delta_mid.items()},
            "d_off_path_not_evaluated_here": ("carried by the dispatcher's None early return plus "
                                              "test_off_arm_matches_the_registered_arithmetic; "
                                              "decision 0109 forbids a literal here"),
            "e_preflight_passed": e,
            "verdict": ("advance-to-independent-review" if (a and b and c and e)
                        else "permanently-close-the-spectral-geometry-branch")}


def worker(output, phase, seed, deadline):
    """One stage of this round, matching the wrapper's ``--worker`` contract."""
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        return proto.prepared_protocol(output)
    if phase == "archive":
        return wide._archive(output, proto.validate_protocol(output))
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
