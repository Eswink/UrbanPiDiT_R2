"""#78 R-A S2 single-factor run: the normalized change-scale decode.

The registered contrast is one switch: whether ``CoarseForecastHead`` writes its
decoded increment unchanged (``identity``, bitwise the pre-change behavior) or
multiplies it by the frozen train-only ratio ``d_c / s_c`` supplied by the
published sidecar (``normalized_change_scale``). Everything else is shared:
architecture, inputs, loss (latitude-weighted MSE, unchanged), optimizer,
schedule, updates, batch size, clip, validation, and both arms' initialized
weights (the ratio is a non-persistent buffer, so the two arms share one tensor
set, one parameter count and equal FLOPs).

Mechanism under test (falsifiable): the decoder currently writes its increment
directly in the state-normalized space, where the effective step must already
implicitly match the 6 h change scale - train-only ratios on the dev store span
0.07-0.75, so the incumbent has to learn tendencies roughly an order of
magnitude in the wrong units. Reparameterizing the increment by the physical
change scale should make the same parameter count optimize more directly. The
wiring probe proves the ratio enters the forward path before any step runs; if
validation skill does not move, the mechanism is falsified as a lever.

The registered decision reads only ``t2m`` at 6 h and 12 h on the val split, by
the #60-fixed per-seed sign rule; all 17 variables x 5 leads are reported
alongside and never merged into the verdict. The test split is never opened.
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
DEADLINE_SECONDS_PER_SEED = 4500.0  # per-seed hard stop, frozen before the run
PLANNED_SECONDS_ROUND = 7200.0     # soft budget for all three seeds
HARD_CAP_SECONDS_ROUND = 14400.0   # loose hard cap for all three seeds
COMPARATOR_DEPTH = 0
PROBE_RELATIVE_TOLERANCE = 1e-3

BASE = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4, "window_size": 4,
        "patch_size": 2, "dropout": 0.0, "anchored_processes": 8, "free_processes": 8,
        "use_forecast_feedback": True, "positional_process_readout": True,
        "default_reasoning_steps": REASONING_STEPS}
IDENTITY_ARM = "process_decode_identity"
SCALED_ARM = "process_decode_change_scale"
ARM_NAMES = (IDENTITY_ARM, SCALED_ARM)
PRIMARY_PAIR = (SCALED_ARM, IDENTITY_ARM)
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12)
SIDECAR_DIR = ROOT / "outputs/r7_s1_seasons_2017/change_scale"
PRIMARY_DECISION_TEXT = (
    "Read the seed-paired mean delta of t2m validation RMSE per lead for "
    "'process_decode_change_scale - process_decode_identity', only where the per-seed "
    "deltas agree in sign (disagreement is unresolved and is never averaged into a "
    "verdict). A negative delta means the change-scale decode has the lower RMSE. A "
    "lead reads 'supported' when every declared seed agrees and the delta is negative, "
    "'worsened' when every seed agrees and the delta is positive, and 'unresolved' "
    "otherwise. Supported requires BOTH 6 h and 12 h supported; anything else is a "
    "negative or mixed result and is reported as such. Falsified if either lead is "
    "worsened or both are unresolved. What was written down before the run: this pilot "
    "screens a reparameterization on one dev store at 400 updates and cannot support a "
    "scientific claim in either direction; it decides whether the change-scale decode "
    "earns a place in the next confirmation instance."
)
LIMITATIONS = [
    "one bounded two-arm run at 400 updates per seed on the 2017 dev store; no "
    "convergence, significance or SOTA claim",
    "the dev store's val/test are 6-15 day windows inside one season block of one "
    "year and one region; the final confirmation instance is a separate, larger "
    "acquisition with unseen years",
    "the arms differ only in the decode parameterization, so this says nothing about "
    "the multivariate loss (R-B) or the training recipe (R-C)",
    "validation-split only; the test split was never opened by this study",
    "the sidecar ratio is fitted on this store's train split; a different store "
    "requires its own sidecar",
    "three seeds are consistency evidence, not a significance test; no significance "
    "threshold is introduced",
]


def refused_test_manifest(manifest: Path) -> None:
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this study is validation-only: the test split stays sealed")


def load_ratio_metadata():
    """Read the published sidecar; returns (ratio, sidecar identity, data identity)."""
    from data.preprocess.r7_change_scale import load_change_scale_sidecar

    meta = load_change_scale_sidecar(SIDECAR_DIR)
    ratio = [float(value) for value in meta["runtime_ratio"]]
    if len(ratio) != 17:
        raise ValueError(f"the sidecar must carry 17 channel ratios, got {len(ratio)}")
    if any(value <= 0 for value in ratio):
        raise ValueError("sidecar ratios must be positive")
    return ratio, meta["change_scale_identity"], meta["data_identity"]


def build_arms(ratio):
    """The two arms: identical configs except the decode mode and its fixed ratio."""
    return (
        (IDENTITY_ARM, "process", dict(BASE)),
        (SCALED_ARM, "process", dict(BASE, change_scale_mode="normalized_change_scale",
                                     change_scale_ratio=list(ratio))),
    )


def arm_switches(config):
    return {"change_scale_mode": config.get("change_scale_mode", "identity")}


def decode_scale_probe(identity_model, scaled_model, batch):
    """The wiring metric: the shipped ratio must multiply the initial decode.

    Both models hold bitwise identical weights (same seed), so the only way the
    two backbone forwards can differ is the scale buffer. Per channel the
    measured (scaled - base) / (identity - base) must reproduce the sidecar
    vector: that is what "the scale enters the forward path" means mechanically.
    """
    ratio = scaled_model.backbone.head.change_scale_ratio.detach().clone()
    if scaled_model.backbone.head.change_scale_mode != "normalized_change_scale":
        raise ValueError("the probe requires the scaled head mode")
    if identity_model.backbone.head.change_scale_mode != "identity":
        raise ValueError("the probe requires the identity head mode")
    with torch.no_grad():
        identity_forecast = identity_model.backbone(batch).forecast
        scaled_forecast = scaled_model.backbone(batch).forecast
    base_state = batch.get("atmos_baseline")
    if base_state is None:
        base_state = batch["coarse_history"][:, -1, :identity_forecast.shape[1]]
    delta_identity = identity_forecast - base_state
    delta_scaled = scaled_forecast - base_state
    max_abs_error = []
    for channel in range(delta_identity.shape[1]):
        values = delta_identity[:, channel]
        scale = float(values.abs().max())
        if scale <= 0:
            raise ValueError(f"channel {channel} decodes an all-zero tendency; the "
                             "probe cannot measure the scale")
        error = (delta_scaled[:, channel] - float(ratio[channel]) * values).abs().max()
        max_abs_error.append(float(error / scale))
    worst = max(max_abs_error)
    return {"channels": int(ratio.numel()),
            "sidecar_ratio": [float(value) for value in ratio],
            "max_relative_error": worst,
            "positive_control_ratio_diversity": len({round(float(v), 4)
                                                     for v in ratio}) > 1,
            "within_tolerance": bool(worst <= PROBE_RELATIVE_TOLERANCE)}


def protocol_payload(manifests_dir, identity, channels, measured, ratio,
                     sidecar_identity):
    from training.r7_experiment import canonical_digest

    arms = build_arms(ratio)
    body = {
        "format": "r7-78-change-scale-single-factor-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#78 R-A normalized change-scale decode (S2)",
        "objective": ("measure whether reparameterizing the decoded increment by the "
                      "train-only d_c / s_c ratio changes validation skill relative to "
                      "the bitwise identity decode under an otherwise identical model "
                      "and loss"),
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
            "change_scale_ratio": [float(value) for value in ratio],
            "change_scale_rule": ("train-only contiguous 6 h change RMS over the "
                                  "store's train-only state std; fitted, published "
                                  "and verified before this protocol"),
        },
        "seeds": list(SEEDS),
        "arms": [{"name": name, "kind": kind, "config": config,
                  "switches": arm_switches(config),
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
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates then cosine "
                            f"decay to {MINIMUM_LR_RATIO} of peak at {UPDATES}"),
            "max_updates": UPDATES, "batch_size": BATCH_SIZE, "clip": CLIP,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": (f"{EARLY_STOPPING_PATIENCE} consecutive non-improving "
                               f"checks at {MINIMUM_IMPROVEMENT:.3%} relative"),
            "loss": "latitude-weighted MSE, streamed truncated BPTT (unchanged)",
            "reasoning_steps": REASONING_STEPS, "process_weight": PROCESS_WEIGHT,
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "selection_split": "val only; test sealed",
        },
        "mechanism_prediction": {
            "text": ("the change-scale decode should let the same parameter count reach "
                     "a lower validation loss by writing increments in the physical "
                     "change units; the run records the per-channel wiring probe "
                     "before training so the scale is checked, not assumed"),
            "falsifier": ("no validation-skill change at all, or a worsened primary "
                          "cell, reads as the reparameterization not being a useful "
                          "lever at this budget and instance"),
        },
        "budgets": {
            "planned_seconds_round": PLANNED_SECONDS_ROUND,
            "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
            "deadline_seconds_per_seed": DEADLINE_SECONDS_PER_SEED,
            "whole_round_scope": ("all declared seeds: CPU preflight/probe, training, "
                                  "evaluation, aggregation; soft overrun is recorded and "
                                  "continues, only the hard cap truncates"),
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
             deadline_seconds=DEADLINE_SECONDS_PER_SEED):
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

    ratio, sidecar_identity, sidecar_data_identity = load_ratio_metadata()
    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    if sidecar_data_identity != identity:
        raise ValueError("the change-scale sidecar was fitted on different data: "
                         f"{sidecar_data_identity} != {identity}")
    channels = int(dataset[0]["coarse_history"].shape[1])
    if channels != len(ratio):
        raise ValueError(f"the sidecar ratio must match {channels} channels, "
                         f"got {len(ratio)}")
    device = select_device(device_name)
    probe_batch = default_collate([dataset[0], dataset[1]])
    validation_dataset = ZarrRolloutDataset(manifests_dir.parent / "cache.zarr", split="val",
                                            lead_hours=VALIDATION_LEADS, history_steps=2,
                                            step_hours=6)

    arms = build_arms(ratio)
    resolved_arms = tuple((name, kind, arm_config(kind, channels, config))
                          for name, kind, config in arms)
    measured = measure_arms(arms, channels=channels, probe_batch=probe_batch,
                            reasoning_steps=REASONING_STEPS, seed=seed)
    if (measured[IDENTITY_ARM]["parameters"] != measured[SCALED_ARM]["parameters"]
            or measured[IDENTITY_ARM]["forward_flops"] != measured[SCALED_ARM]["forward_flops"]
            or (measured[IDENTITY_ARM]["forward_backward_flops"]
                != measured[SCALED_ARM]["forward_backward_flops"])):
        raise RuntimeError("the two arms must not differ in parameters or FLOPs: the "
                           "scale is a fixed buffer, so a difference here means the "
                           "contrast is confounded")
    protocol = protocol_payload(manifests_dir, identity, channels, measured, ratio,
                                sidecar_identity)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
    verified_registration(protocol_path)

    # Wiring check before any step: the shipped ratio must multiply the initial
    # decoded tendency, and the identity arm must reproduce the identity mode.
    seed_everything(seed)
    probe_models = {name: make_model("process", resolved)
                    for (name, kind, config), (_, _, resolved)
                    in zip(arms, resolved_arms)}
    probe = decode_scale_probe(probe_models[IDENTITY_ARM], probe_models[SCALED_ARM],
                               probe_batch)
    for model in probe_models.values():
        del model
    if not probe["within_tolerance"]:
        raise RuntimeError(f"the change-scale wiring probe failed: {probe}")

    results = {
        "format": "r7-78-change-scale-single-factor-result-v1", "scientific_claim": False,
        "test_read": False, "seed": seed, "seeds": list(SEEDS), "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": model_code_digest(), "protocol_sha256": protocol["protocol_sha256"],
        "protocol": protocol, "decode_scale_probe": probe, "flop_measurements": measured,
        "training": {}, "evaluation": {}, "probes": {}, "budget": {}}

    started = time.perf_counter()
    pairing = verify_arm_pairing(arms, baseline=IDENTITY_ARM, channels=channels, seed=seed)
    if not pairing["all_shared_pairs_identical"]:
        raise RuntimeError("the two arms do not share bitwise identical seeded weights; "
                           "the comparison would be confounded by the initialization")
    shared_pair = pairing["pairwise_shared_tensors"][
        "|".join(sorted((IDENTITY_ARM, SCALED_ARM)))]
    if shared_pair["added_in_first"] or shared_pair["added_in_second"]:
        raise RuntimeError("this contrast must not change the tensor set: the ratio is a "
                           "non-persistent buffer, so an added/removed state key means "
                           "the switch is confounded with capacity")
    if not shared_pair["same_tensor_set"]:
        raise RuntimeError("the arms must have identical state-dict key sets")
    results["arm_pairing"] = pairing

    anchors = {}
    for (name, kind, config), (_, _, resolved) in zip(arms, resolved_arms):
        if time.perf_counter() - started > deadline_seconds:
            raise RuntimeError(f"{deadline_seconds}s wall limit reached before {name} "
                               "started; stopping rather than overrunning")
        shared = None
        if name == IDENTITY_ARM:
            seed_everything(seed)
            anchors[name] = {key: value.clone()
                             for key, value in make_model(kind, resolved).state_dict().items()}
        else:
            shared = anchors[IDENTITY_ARM]
        checkpoint, report = run_scheduled_updates(
            dataset, kind=kind, model_config=resolved, data_identity=identity,
            output_dir=output_dir / "training" / name, total_updates=updates,
            batch_size=BATCH_SIZE, steps=REASONING_STEPS, seed=seed, lr=LR, clip=CLIP,
            process_weight=PROCESS_WEIGHT, warmup_updates=WARMUP_UPDATES,
            minimum_lr_ratio=MINIMUM_LR_RATIO, validation_every=VALIDATION_EVERY,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            minimum_improvement=MINIMUM_IMPROVEMENT, validation_lead_hours=VALIDATION_LEADS,
            device_name=device_name, validation_dataset=validation_dataset,
            shared_initial_state=shared)
        saved = load_checkpoint(checkpoint)
        if saved["updates"] != report["selected_update"]:
            raise ValueError(f"{name} selected checkpoint/update mismatch")
        applied = report["shared_initial_state"]
        if name != IDENTITY_ARM and applied["ignored_count"]:
            raise RuntimeError(f"{name}: shared-state transfer ignored "
                               f"{applied['ignored_count']} tensors")
        results["training"][name] = {
            "protocol_sha256": protocol["protocol_sha256"],
            "switches": arm_switches(config),
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


def primary_reading(pairs):
    """The preregistered primary cell, read by the registered decision text."""
    block = pairs.get(f"{PRIMARY_PAIR[0]} - {PRIMARY_PAIR[1]}", {}).get("cells", {})
    cells, verdicts = {}, []
    for lead in PRIMARY_LEADS:
        entry = block.get(f"{lead}h|{PRIMARY_VARIABLE}")
        if entry is None or not entry["sign_consistent"]:
            cells[str(lead)] = {"delta_seed_mean": None, "outcome": "unresolved",
                                "seed_deltas": ({} if entry is None else dict(entry["seed_deltas"])),
                                "reading": ("per-seed deltas disagree or the comparator "
                                            "returned no cell; no verdict and no seed "
                                            "mean under the frozen rule")}
            continue
        deltas = list(entry["seed_deltas"].values())
        delta = sum(deltas) / len(deltas)
        outcome = "supported" if delta < 0 else "worsened"
        cells[str(lead)] = {"delta_seed_mean": delta, "outcome": outcome,
                            "sign_consistent": True, "seed_deltas": dict(entry["seed_deltas"]),
                            "reading": f"{outcome}: every seed agrees, delta {delta:+.6f}"}
        verdicts.append(outcome)
    if verdicts and all(value == "supported" for value in verdicts):
        headline = "supported on both registered leads"
    elif any(value == "worsened" for value in verdicts):
        headline = "worsened: a registered lead moved the wrong way"
    elif not verdicts:
        headline = "no sign-consistent primary cell"
    else:
        headline = "mixed/unresolved: the registered leads did not both support"
    return {"primary_pair": list(PRIMARY_PAIR), "variable": PRIMARY_VARIABLE,
            "leads_hours": list(PRIMARY_LEADS), "cells": cells, "headline": headline,
            "reading": ("supported requires 6 h and 12 h both sign-consistent and "
                        "negative; anything else is negative or mixed")}


def main():
    """Seed mode runs one seed; finalize mode merges the declared seeds."""
    from scripts.r7_m3_offline import deny_network

    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("seed", "finalize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--manifests", type=Path,
                        default=ROOT / "outputs/r7_s1_seasons_2017/store/manifests")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/r7_78_change_scale_pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--deadline-seconds", type=float, default=DEADLINE_SECONDS_PER_SEED)
    args = parser.parse_args()

    if args.mode == "seed":
        if args.seed is None or args.seed not in SEEDS:
            parser.error(f"--mode seed requires --seed in {SEEDS}")
        started = time.perf_counter()
        seed_dir = args.out / f"seed{args.seed}"
        attempt = {"format": "r7-78-change-scale-seed-attempt-v1", "seed": args.seed,
                   "scientific_claim": False, "test_read": False,
                   "planned_seconds_round": PLANNED_SECONDS_ROUND,
                   "hard_cap_seconds_round": HARD_CAP_SECONDS_ROUND,
                   "deadline_seconds_per_seed": args.deadline_seconds,
                   "started_perf_counter": started, "status": "running"}
        try:
            result = run_seed(args.manifests, seed_dir, seed=args.seed,
                              updates=args.updates, device_name=args.device,
                              deadline_seconds=args.deadline_seconds)
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

    from training.r7_arm_harness import (comparator_blocks, merge_seed_results,
                                         pair_cells, write_study_tables)

    merged = merge_seed_results(args.out, seeds=SEEDS, arms=ARM_NAMES,
                                fmt="r7-78-change-scale-single-factor-result-v1")
    identity = {"dataset_identity": merged["protocol"]["data"]["data_identity"],
                "change_scale_identity": merged["protocol"]["data"]["change_scale_identity"],
                "model_code_sha256": merged["model_code_sha256"],
                "declared_update_budget": UPDATES, "evaluation_split": "val"}
    table, blocks = comparator_blocks(merged, pairs={PRIMARY_PAIR}, depth=COMPARATOR_DEPTH,
                                      identity=identity)
    pairs = pair_cells(blocks, leads=EVALUATION_LEADS)
    payload = {"format": "r7-78-change-scale-single-factor-comparison-v1",
               "scientific_claim": False, "identity": identity, "pairs": pairs,
               "primary": primary_reading(pairs),
               "rule": ("seed-paired: a cell is improved/worsened only when every "
                        "declared seed's delta agrees in sign; disagreements are "
                        "unresolved and are never averaged into a win"),
               "decode_scale_probe": {str(seed): merged["probes"].get(str(seed), {})
                                      for seed in merged["seeds"]},
               "table": table}
    probe_by_seed = {seed: json.loads((args.out / f"seed{seed}" / "seed_result.json")
                                      .read_text(encoding="utf-8"))["decode_scale_probe"]
                     for seed in merged["seeds"]}
    payload["decode_scale_probe"] = probe_by_seed
    with (args.out / "paired_comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_study_tables(merged, args.out)
    print(json.dumps({
        "complete": True, "seeds": merged["seeds"], "scientific_claim": False,
        "protocol_sha256": merged["protocol_sha256"],
        "model_code_sha256": merged["model_code_sha256"],
        "gpu_hours_training": merged["budget"]["training_seconds_total"] / 3600.0,
        "gpu_hours_evaluation": merged["budget"]["evaluation_seconds_total"] / 3600.0,
        "primary": payload["primary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
