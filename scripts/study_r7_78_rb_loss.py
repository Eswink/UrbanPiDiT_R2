"""#78 R-B S2 round: the change-scale-weighted training loss.

Single registered contrast: both arms keep the identity decode (R-A's negative
result stays closed) and differ only in the training objective - the incumbent
equal-channel latitude-weighted MSE versus ``w_c * MSE`` with the train-only
per-channel weights ``w_c = (1 / runtime_ratio_c)^2`` normalized to mean one,
which is the stored-space form of the issue's physical formula
``sum_c mean(((forecast - target) / d_c)^2)``.

Mechanism difference from R-A (written before any step): R-A reparameterized
*what the decoder writes* (a representation change) while keeping the objective
equal-channel; R-B keeps the decoder's units and changes *which channels the
optimization emphasizes* (an objective change). R-A's negative result therefore
does not logically falsify R-B, but it lowers the prior: if the equal-channel
objective were the binding constraint, R-B is the lever that shows it.

Everything else is shared: architecture, decode, loss shape (latitude
weighting, FP32 reduction, streamed truncated BPTT), optimizer, schedule,
updates, batch size, clip, validation selection metric and evaluation protocol.
Validation selection deliberately keeps the equal-channel normalized MSE for
both arms: the contrast is the training objective, not the selection rule. The
test split is never opened.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_arm_harness import measure_arms, sha256_file, write_study_tables

SEEDS = (41, 42, 43)
UPDATES = 400
BATCH_SIZE = 2
LR = 2e-4
WEIGHT_DECAY = 1e-4
CLIP = 1.0
WARMUP_UPDATES = 80
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 4
PROCESS_WEIGHT = 0.0
EVALUATION_MAX_SAMPLES = 64
DEADLINE_SECONDS_PER_SEED = 5400.0
PLANNED_SECONDS_ROUND = 7200.0
HARD_CAP_SECONDS_ROUND = 18000.0
COMPARATOR_DEPTH = 0
PROBE_RELATIVE_TOLERANCE = 1e-3

BASE = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4, "window_size": 4,
        "patch_size": 2, "dropout": 0.0, "anchored_processes": 8, "free_processes": 8,
        "use_forecast_feedback": True, "positional_process_readout": True,
        "default_reasoning_steps": REASONING_STEPS}
IDENTITY_ARM = "process_loss_identity"
WEIGHTED_ARM = "process_loss_change_scale"
ARM_NAMES = (IDENTITY_ARM, WEIGHTED_ARM)
PRIMARY_PAIR = (WEIGHTED_ARM, IDENTITY_ARM)
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12)
SIDECAR_DIR = ROOT / "outputs/r7_s1_seasons_2017/change_scale"
PRIMARY_DECISION_TEXT = (
    "Read the seed-paired mean delta of t2m validation RMSE per lead for "
    "'process_loss_change_scale - process_loss_identity', only where the per-seed "
    "deltas agree in sign (disagreement is unresolved and is never averaged into a "
    "verdict). A negative delta means the change-scale-weighted loss has the lower "
    "RMSE. A lead reads 'supported' when every declared seed agrees and the delta is "
    "negative, 'worsened' when every seed agrees and the delta is positive, and "
    "'unresolved' otherwise. Supported requires BOTH 6 h and 12 h supported; anything "
    "else is a negative or mixed result and is reported as such. Falsified if either "
    "lead is worsened or both are unresolved. What was written down before the run: "
    "this pilot screens an objective reweighting on one dev store at 400 updates and "
    "cannot support a scientific claim in either direction; it decides whether the "
    "weighted objective earns a place in the next confirmation instance.")
LIMITATIONS = [
    "one bounded two-arm run at 400 updates per seed on the 2017 dev store; no "
    "convergence, significance or SOTA claim",
    "the dev store's val/test are 6-15 day windows inside one season block of one "
    "year and one region; the final confirmation instance is a separate, larger "
    "acquisition with unseen years",
    "the arms differ only in the training objective's per-channel weights; the "
    "decode stays identity as R-A closed, so this says nothing about the decoded "
    "parameterization",
    "validation checkpoints are still selected on the equal-channel normalized MSE "
    "for both arms, so the reported contrast is the training objective alone",
    "the weights are derived from this store's train-only sidecar; a different "
    "store requires its own sidecar and its own weights",
    "three seeds are consistency evidence, not a significance test; no significance "
    "threshold is introduced",
]


def refused_test_manifest(manifest: Path) -> None:
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this study is validation-only: the test split stays sealed")


MECHANISM_DIFFERENCE_FROM_RA = (
    "R-A reparameterized the decoded increment (Y = X_t + ratio_c * r_c) with "
    "the objective unchanged; it closed worsened. R-B keeps the decode at "
    "identity and reweights the objective per channel, so it tests a different "
    "hypothesis - that the equal-channel objective is the binding constraint - "
    "and is not falsified by R-A's negative result; it is a separate round, "
    "never a combined change.")
OBJECTIVE_TEXT = (
    "measure whether reweighting the training objective per channel by the "
    "train-only change scale changes validation skill relative to the "
    "equal-channel loss under an otherwise identical model, decode and schedule")
MECHANISM_PREDICTION = (
    "if the equal-channel objective underprices channels whose normalized "
    "errors are large relative to their physical 6 h change, the weighted "
    "objective should lower the t2m validation RMSE through a better shared "
    "representation")
MECHANISM_FALSIFIER = (
    "no validation-skill change at all, or a worsened primary cell, reads as "
    "the objective reweighting not being a useful lever at this budget and "
    "instance")
ROUND_SCOPE = (
    "all declared seeds: CPU preflight/probe, training, evaluation, "
    "aggregation; soft overrun is recorded and continues, only the hard cap "
    "truncates")


def load_weight_metadata():
    """Read the published sidecar and derive the frozen R-B weights."""
    from data.preprocess.r7_change_scale import load_change_scale_sidecar
    from data.preprocess.r7_rb_weights import rb_loss_weights

    meta = load_change_scale_sidecar(SIDECAR_DIR)
    derived = rb_loss_weights(meta)
    weights = [float(value) for value in derived["weights"]]
    if len(weights) != 17:
        raise ValueError(f"the sidecar must carry 17 channel weights, got {len(weights)}")
    if any(value <= 0 for value in weights):
        raise ValueError("loss weights must be positive")
    if not all(abs(value - 1.0) > 0 for value in weights):
        raise ValueError("at least one weight must differ from one, or the arms are identical")
    return weights, derived, meta["change_scale_identity"], meta["data_identity"]


def build_arms(weights):
    """Both arms run the identical model; only the trainer objective differs."""
    return ((IDENTITY_ARM, "process", dict(BASE)),
            (WEIGHTED_ARM, "process", dict(BASE)))


def arm_switches(config, *, weights):
    return {"loss_channel_weights": None if config is None else list(weights)}


def loss_weight_probe(model, batch, weights):
    """The wiring metric: the shipped weights must reach the trainer's loss.

    ``backward_streamed_truncated`` at ``reasoning_steps=0`` reduces to exactly
    the weighted per-sample latitude MSE of the backbone draft, so the trainer
    total divided by an independently recomputed uniform-weight analytic value
    must reproduce the weights' effect per arm, and the weighted arm must equal
    the weighted analytic value. Both directions are checked against a fresh
    deep copy so no gradients leak into the probe model.
    """
    import copy as _copy

    from model.r7_halting import forecast_inputs
    from training.r7_halting import per_sample_latitude_mse
    from training.r7_streaming import backward_streamed_truncated

    def analytic(channel_weights):
        with torch.no_grad():
            draft = model.backbone(forecast_inputs(batch)).forecast
        return float(per_sample_latitude_mse(draft, batch["atmos_target"],
                                             batch.get("latitude"),
                                             channel_weights=channel_weights).mean())

    uniform = [1.0] * len(weights)
    analytic_uniform, analytic_weighted = analytic(uniform), analytic(weights)
    if analytic_uniform <= 0:
        raise ValueError("the probe fixture has zero error; it cannot measure a weight")
    report = {"channels": len(weights), "analytic_ratio": analytic_weighted / analytic_uniform}
    ratios = {}
    for name, channel_weights in ((IDENTITY_ARM, None), (WEIGHTED_ARM, weights)):
        arm_model = _copy.deepcopy(model).train()
        result = backward_streamed_truncated(
            arm_model, batch, reasoning_steps=0, process_weight=0.0,
            loss_channel_weights=channel_weights)
        ratios[name] = float(result.total) / analytic_uniform
        del arm_model, result
    report["trainer_over_analytic_uniform"] = ratios
    relative = abs(ratios[WEIGHTED_ARM] - report["analytic_ratio"]) / max(
        abs(report["analytic_ratio"]), 1e-12)
    report["max_relative_error"] = float(relative)
    report["identity_arm_ratio"] = ratios[IDENTITY_ARM]
    report["positive_control_ratio_diversity"] = len({round(float(v), 4)
                                                      for v in weights}) > 1
    report["within_tolerance"] = bool(relative <= PROBE_RELATIVE_TOLERANCE)
    return report


def protocol_payload(manifests_dir, identity, channels, measured, weights, derived,
                     sidecar_identity, schedule=None):
    from training.r7_experiment import canonical_digest

    arms = build_arms(weights)
    effective = {"max_updates": UPDATES, "warmup_updates": WARMUP_UPDATES,
                 "validation_every": VALIDATION_EVERY, "batch_size": BATCH_SIZE,
                 "clip": CLIP}
    if schedule is not None:
        effective.update({key: int(value) if key != "clip" else float(value)
                          for key, value in schedule.items()})
    body = {
        "format": "r7-78-rb-loss-single-factor-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#78 R-B change-scale-weighted loss (S2)",
        "objective": OBJECTIVE_TEXT,
        "mechanism_difference_from_ra": MECHANISM_DIFFERENCE_FROM_RA,
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False, "test_policy": "sealed; only the val split is scored",
            "data_identity": str(identity),
            "split_mode": "time_ranges; four-season 2017 dev store",
            "normalization": "store train-only centered mean/std, applied at read time",
            "change_scale_sidecar": str(SIDECAR_DIR),
            "change_scale_identity": str(sidecar_identity),
            "loss_channel_weights": [float(value) for value in weights],
            "loss_channel_weights_rule": derived["rule"],
            "loss_channel_weights_loss_rule": derived["loss_rule"],
            "loss_channel_weights_degenerate_channels": derived["degenerate_channels"],
        },
        "seeds": list(SEEDS),
        "arms": [{"name": name, "kind": kind, "config": config,
                  "switches": {"loss_channel_weights": (
                      None if name == IDENTITY_ARM else [float(v) for v in weights])},
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in arms],
        "primary_registration": {
            "registered_before_first_optimizer_step": True,
            "variable": PRIMARY_VARIABLE, "leads_hours": list(PRIMARY_LEADS),
            "pair": {"focus": PRIMARY_PAIR[0], "baseline": PRIMARY_PAIR[1],
                     "reading": f"{PRIMARY_PAIR[0]} - {PRIMARY_PAIR[1]}"},
            "decision_text": PRIMARY_DECISION_TEXT,
            "secondary_reporting": ("all 17 variables x 5 leads are reported for the "
                                    "pair; the primary is never re-picked after the run"),
            "aggregate_rule": (f"per-seed sign agreement at depth {COMPARATOR_DEPTH}; a "
                               "cell is improved/worsened only when every declared seed "
                               "agrees in sign, otherwise unresolved"),
        },
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
            "lr_schedule": (f"linear warmup over {effective['warmup_updates']} updates "
                            f"then cosine decay to {MINIMUM_LR_RATIO} of peak at "
                            f"{effective['max_updates']}"),
            "max_updates": effective["max_updates"], "batch_size": effective["batch_size"],
            "clip": effective["clip"],
            "validation_every": effective["validation_every"],
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": (f"{EARLY_STOPPING_PATIENCE} consecutive non-improving "
                               f"checks at {MINIMUM_IMPROVEMENT:.3%} relative"),
            "decode": "identity (R-A closed; the decode is not a variable in this round)",
            "loss_shape": ("latitude-weighted MSE in the stored normalized space, FP32 "
                           "reduction, streamed truncated BPTT; the only difference is "
                           "the per-channel weight vector"),
            "reasoning_steps": REASONING_STEPS, "process_weight": PROCESS_WEIGHT,
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "selection_split": "val only; test sealed",
            "selection_metric": ("equal-channel normalized latitude-weighted MSE for "
                                 "both arms (the contrast is the training objective)"),
        },
        "mechanism_prediction": {
            "text": MECHANISM_PREDICTION,
            "falsifier": MECHANISM_FALSIFIER,
        },
        "budgets": {
            "planned_seconds_round": PLANNED_SECONDS_ROUND,
            "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
            "deadline_seconds_per_seed": DEADLINE_SECONDS_PER_SEED,
            "whole_round_scope": ROUND_SCOPE,
            "frozen_before_any_step": True,
        },
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def verified_registration(protocol_path):
    from training.r7_experiment import canonical_digest

    frozen = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != canonical_digest(
            {key: value for key, value in frozen.items() if key != "protocol_sha256"}):
        raise RuntimeError("the protocol on disk is not the one this run measured")
    registration = frozen["primary_registration"]
    if (registration["variable"] != PRIMARY_VARIABLE
            or tuple(registration["leads_hours"]) != PRIMARY_LEADS
            or (registration["pair"]["focus"], registration["pair"]["baseline"])
            != PRIMARY_PAIR
            or not registration["decision_text"].strip()):
        raise RuntimeError("the primary registration on disk is not the declared one")
    return frozen


def run_seed(manifests_dir, output_dir, *, seed, updates=UPDATES, device_name="cuda",
             deadline_seconds=DEADLINE_SECONDS_PER_SEED,
             warmup_updates=WARMUP_UPDATES, validation_every=VALIDATION_EVERY):
    from data.r7_evaluation import ZarrRolloutDataset
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from torch.utils.data import default_collate
    from training.r7_arm_harness import arm_config, verify_arm_pairing
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (dataset_identity, load_checkpoint, make_model,
                                        model_code_digest, seed_everything, select_device)
    from training.r7_scheduled_runner import run_scheduled_updates

    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    train_manifest, val_manifest = manifests_dir / "train.jsonl", manifests_dir / "val.jsonl"
    for manifest in (train_manifest, val_manifest):
        refused_test_manifest(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")

    weights, derived, sidecar_identity, sidecar_data_identity = load_weight_metadata()
    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    if sidecar_data_identity != identity:
        raise ValueError("the change-scale sidecar was fitted on different data: "
                         f"{sidecar_data_identity} != {identity}")
    channels = int(dataset[0]["coarse_history"].shape[1])
    if channels != len(weights):
        raise ValueError(f"the sidecar weights must match {channels} channels, "
                         f"got {len(weights)}")
    device = select_device(device_name)
    probe_batch = default_collate([dataset[0], dataset[1]])
    validation_dataset = ZarrRolloutDataset(manifests_dir.parent / "cache.zarr", split="val",
                                            lead_hours=VALIDATION_LEADS, history_steps=2,
                                            step_hours=6)

    arms = build_arms(weights)
    resolved_arms = tuple((name, kind, arm_config(kind, channels, config))
                          for name, kind, config in arms)
    measured = measure_arms(arms, channels=channels, probe_batch=probe_batch,
                            reasoning_steps=REASONING_STEPS, seed=seed)
    if (measured[IDENTITY_ARM]["parameters"] != measured[WEIGHTED_ARM]["parameters"]
            or measured[IDENTITY_ARM]["forward_flops"] != measured[WEIGHTED_ARM]["forward_flops"]
            or (measured[IDENTITY_ARM]["forward_backward_flops"]
                != measured[WEIGHTED_ARM]["forward_backward_flops"])):
        raise RuntimeError("the two arms must not differ in parameters or FLOPs: only "
                           "the objective weights differ, so a structural difference "
                           "would mean the contrast is confounded")
    protocol = protocol_payload(manifests_dir, identity, channels, measured, weights,
                                derived, sidecar_identity,
                                schedule={"max_updates": updates,
                                          "warmup_updates": warmup_updates,
                                          "validation_every": validation_every})
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    verified_registration(protocol_path)

    # Wiring check before any step: the shipped weights must reach the trainer's
    # loss in the documented direction, and equal-channel reduction must equal
    # the incumbent path bitwise.
    seed_everything(seed)
    probe_model = make_model("process", resolved_arms[0][2])
    probe = loss_weight_probe(probe_model, probe_batch, weights)
    del probe_model
    if not probe["within_tolerance"]:
        raise RuntimeError(f"the loss-weight wiring probe failed: {probe}")

    results = {
        "format": "r7-78-rb-loss-single-factor-result-v1", "scientific_claim": False,
        "test_read": False, "seed": seed, "seeds": list(SEEDS), "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": model_code_digest(), "protocol_sha256": protocol["protocol_sha256"],
        "protocol": protocol, "loss_weight_probe": probe, "flop_measurements": measured,
        "training": {}, "evaluation": {}, "probes": {}, "budget": {}}

    started = time.perf_counter()
    pairing = verify_arm_pairing(arms, baseline=IDENTITY_ARM, channels=channels, seed=seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("the two arms do not share bitwise identical seeded weights; "
                           "the comparison would be confounded by the initialization")
    shared_pair = pairing["pairwise_shared_tensors"][
        "|".join(sorted((IDENTITY_ARM, WEIGHTED_ARM)))]
    if shared_pair["added_in_first"] or shared_pair["added_in_second"]:
        raise RuntimeError("this contrast must not change the tensor set: the weights "
                           "live in the trainer, so an added/removed state key means "
                           "the switch is confounded with capacity")
    if not shared_pair["same_tensor_set"]:
        raise RuntimeError("the arms must have identical state-dict key sets")
    results["arm_pairing"] = pairing

    anchors = {}
    for name, kind, config in arms:
        if time.perf_counter() - started > deadline_seconds:
            raise RuntimeError(f"{deadline_seconds}s wall limit reached before {name} "
                               "started; stopping rather than overrunning")
        shared = None
        if name == IDENTITY_ARM:
            seed_everything(seed)
            anchors[name] = {key: value.clone()
                             for key, value in make_model(kind, resolved_arms[0][2])
                             .state_dict().items()}
        else:
            shared = anchors[IDENTITY_ARM]
        arm_weights = None if name == IDENTITY_ARM else weights
        checkpoint, report = run_scheduled_updates(
            dataset, kind=kind, model_config=resolved_arms[0][2], data_identity=identity,
            output_dir=output_dir / "training" / name, total_updates=updates,
            batch_size=BATCH_SIZE, steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
            process_weight=PROCESS_WEIGHT, warmup_updates=warmup_updates,
            minimum_lr_ratio=MINIMUM_LR_RATIO, validation_every=validation_every,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            minimum_improvement=MINIMUM_IMPROVEMENT, validation_lead_hours=VALIDATION_LEADS,
            device_name=device_name, validation_dataset=validation_dataset,
            shared_initial_state=shared, loss_channel_weights=arm_weights)
        saved = load_checkpoint(checkpoint)
        if saved["updates"] != report["selected_update"]:
            raise ValueError(f"{name} selected checkpoint/update mismatch")
        applied = report["shared_initial_state"]
        if name != IDENTITY_ARM and applied["ignored_count"]:
            raise RuntimeError(f"{name}: shared-state transfer ignored "
                               f"{applied['ignored_count']} tensors")
        recorded_weights = report["contract"].get("loss_channel_weights")
        if (name == IDENTITY_ARM) != (recorded_weights is None):
            raise RuntimeError(f"{name}: the training contract did not record the "
                               "declared objective switch")
        if recorded_weights is not None and len(recorded_weights) != channels:
            raise RuntimeError(f"{name}: recorded weights do not match {channels} channels")
        results["training"][name] = {
            "protocol_sha256": protocol["protocol_sha256"],
            "switches": {"loss_channel_weights": recorded_weights},
            "updates_run": report["updates_this_run"],
            "selected_update": report["selected_update"],
            "selected_validation_mse": report["selected_validation_mse"],
            "early_stopped": report["early_stopped"],
            "stopped_reason": report["stopped_reason"],
            "elapsed_seconds": report["elapsed_seconds"],
            "seconds_per_update": report["seconds_per_update"],
            "validation_checks": report["validations"],
            "peak_allocated_bytes": report["peak_allocated_bytes"],
            "peak_reserved_bytes": report["peak_reserved_bytes"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": measured[name]["parameters"],
            "shared_initial_state": applied,
            "first_epoch_mean_loss": (report["losses"][0]["loss"] if report["losses"] else None),
            "last_epoch_mean_loss": (report["losses"][-1]["loss"] if report["losses"] else None)}
        print(json.dumps({"trained": name, "seed": seed,
                          "selected_update": report["selected_update"],
                          "seconds": round(report["elapsed_seconds"], 1)}), flush=True)

    results["budget"]["training_seconds_total"] = time.perf_counter() - started

    for name, kind, config in arms:
        for lead in EVALUATION_LEADS:
            run_dir = output_dir / "evaluation" / name / f"lead_{lead:03d}h"
            report = evaluate_local(
                val_manifest, output_dir=run_dir,
                checkpoint=(output_dir / "training" / name
                            / f"update_{results['training'][name]['selected_update']:07d}.pt"),
                lead_hours=(lead,), max_samples=EVALUATION_MAX_SAMPLES,
                device_name=device_name, reasoning_steps=REASONING_STEPS)
            if report["split"] != "val":
                raise ValueError(f"{name} was scored on {report['split']}, not val")
            results["evaluation"][f"{name}@{lead}h"] = {
                "arm": name, "lead_hours": lead, "split": report["split"],
                "n_evaluated": report["n_evaluated"],
                "n_available_windows": report["n_available_windows"],
                "channels": list(report["channels"]), "units": list(report["units"]),
                "elapsed_seconds": report["elapsed_seconds"],
                "trainable_parameters": report["trainable_parameters"],
                "climatology": {key: value for key, value in report["climatology"].items()
                                if key != "bucket_counts"},
                "bucket_counts": report["climatology"]["bucket_counts"],
                "evaluation_dir": str(run_dir), "rmse_csv": str(run_dir / "rmse.csv"),
                "skill_csv": str(run_dir / "climatology_skill.csv")}
    print(json.dumps({"evaluated_seed": seed, "leads": list(EVALUATION_LEADS)}), flush=True)

    for lead in EVALUATION_LEADS:
        counts = {name: results["evaluation"][f"{name}@{lead}h"]["n_evaluated"]
                  for name in ARM_NAMES}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"arms scored on different case counts at {lead}h: {counts}")
        results.setdefault("case_counts_by_lead", {})[str(lead)] = counts[IDENTITY_ARM]

    with (output_dir / "seed_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


def main():
    """Seed mode runs one seed; finalize mode lives in the finalize driver."""
    from scripts.r7_m3_offline import deny_network

    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("seed",))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--manifests", type=Path,
                        default=ROOT / "outputs/r7_s1_seasons_2017/store/manifests")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/r7_78_rb_loss_pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--warmup-updates", type=int, default=WARMUP_UPDATES,
                        help="smoke-only override; the registered run uses the default")
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY,
                        help="smoke-only override; the registered run uses the default")
    parser.add_argument("--deadline-seconds", type=float, default=DEADLINE_SECONDS_PER_SEED)
    args = parser.parse_args()

    if args.seed is None or args.seed not in SEEDS:
        parser.error(f"--mode seed requires --seed in {SEEDS}")
    started = time.perf_counter()
    seed_dir = args.out / f"seed{args.seed}"
    attempt = {"format": "r7-78-rb-loss-seed-attempt-v1", "seed": args.seed,
               "scientific_claim": False, "test_read": False,
               "planned_seconds_round": PLANNED_SECONDS_ROUND,
               "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
               "deadline_seconds_per_seed": args.deadline_seconds,
               "started_perf_counter": started, "status": "running"}
    try:
        result = run_seed(args.manifests, seed_dir, seed=args.seed,
                          updates=args.updates, device_name=args.device,
                          deadline_seconds=args.deadline_seconds,
                          warmup_updates=args.warmup_updates,
                          validation_every=args.validation_every)
    except BaseException as exc:
        attempt.update({"status": "failed", "finalized": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.perf_counter() - started,
                        "no_retry_or_resurrection": True})
        if seed_dir.exists():
            (seed_dir / "attempt.json").write_text(
                json.dumps(attempt, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8")
        raise
    attempt.update({"status": "success", "finalized": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "protocol_sha256": result["protocol_sha256"],
                    "soft_overrun_seconds": max(0.0, time.perf_counter() - started
                                                - PLANNED_SECONDS_ROUND / len(SEEDS))})
    (seed_dir / "attempt.json").write_text(
        json.dumps(attempt, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
