"""B2: controlled multi-seed confirmatory comparison on the real B2 segment (#64).

#64 B2 asks for something B1 could not give: several **pre-declared** seeds, a
**frozen** training protocol, a per-variable/per-lead skill check against
persistence and the train-only climatology decided by criteria written down in
advance, shared initial weights for the arms that share a structure, and free
6/12/24/48/72 h rollouts rather than a single +6 h step.

What this script fixes rather than tunes:

- the seed set is written in ``EXPLORATION_SEEDS`` / ``CONFIRMATORY_SEEDS``
  before any run; every seed is reported, best or not;
- the learning-rate schedule (linear warmup then cosine), the maximum update
  count, the validation frequency, the checkpoint-selection rule and the
  early-stopping rule are all constants in the frozen protocol, and early
  stopping reads **validation only**;
- the skill gate is ``SKILL_CRITERIA``, written before the runs: a fixed
  variable/lead list and required win counts against both parameter-free
  controls, with a stated allowance. No variable is dropped afterwards;
- shared structures get a shared start: the recursive arms are seeded once and
  the process arm's common parameters are copied from the generic arm's
  initialization, with applied/ignored parameter names recorded. The U-Net and
  AFNO arms share no structure with them, so they are matched on data and
  budget only - never claimed as initial-weight-matched.

Two phases, matching #64's order. ``--phase explore`` runs the two exploration
seeds to check the protocol is stable, then the protocol is frozen; ``--phase
confirm`` runs the pre-declared confirmatory seeds. Freezing is enforced by the
protocol digest recorded in ``protocol.json``: the confirm phase refuses to run
if the protocol file it would use differs from the frozen digest, and every
result carries it.

Discipline this script enforces instead of documenting:

- the test split is **never** read; B2's test block is a 2016 engineering
  re-split, not the v2 2021 test candidate, so it stays sealed;
- all arms train on the identical window order for a given seed;
- parameter, FLOPs, wall-time and case-count tables are all emitted, because
  "same updates" is not "same compute".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPLORATION_SEEDS = (41, 42)
CONFIRMATORY_SEEDS = (41, 42, 43)
SEED_POOL = tuple(sorted(set(EXPLORATION_SEEDS) | set(CONFIRMATORY_SEEDS)))

UPDATES = 800
BATCH_SIZE = 2
LR = 2e-4
CLIP = 1.0
WEIGHT_DECAY = 1e-4
WARMUP_UPDATES = 80
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)  # integers: a float horizon is rejected, not silently truncated
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 3
PROCESS_WEIGHT = 0.0  # process arm is generic+process tokens here, not the process loss
NOMINAL_PARAMETERS = 2_800_000
PARAMETER_BAND = 0.05
# The #60 comparator keys rows by (arm, depth, variable, unit, lead). B2 does not
# ablate reasoning depth (#65 owns that), so every row carries one declared depth
# and the comparison is read at it.
COMPARATOR_DEPTH = 0
SCORED_CHANNELS = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850",
                   "v850", "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")

# (name, runner kind, architecture config). native_window/generic keep B1's
# audited configs so the two segments stay comparable; the process arm exists
# because #64 B2 requires the *shared-structure* pair to start from aligned
# common weights, and generic/process are that pair. unet/afno match data and
# budget only.
ARMS = (
    ("unet", "native", {"architecture": "unet", "dim": 66}),
    ("native_window", "native", {"architecture": "window", "dim": 192, "depth": 6,
                                 "heads": 4, "window_size": 4, "patch_size": 2,
                                 "dropout": 0.0}),
    ("afno_small", "native", {"architecture": "afno_small", "dim": 248, "patch_size": 2,
                              "depth": 8, "blocks": 4}),
    ("generic", "generic", {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "latent_tokens": 16,
                            "default_reasoning_steps": REASONING_STEPS}),
    ("process", "process", {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                            "window_size": 4, "patch_size": 2, "dropout": 0.0,
                            "anchored_processes": 8, "free_processes": 8,
                            "use_forecast_feedback": True,
                            "default_reasoning_steps": REASONING_STEPS}),
)
ARM_NAMES = tuple(entry[0] for entry in ARMS)
BASELINES = ("persistence", "climatology")
# #64 B2 sentence 3: the arms that share a structure get aligned common initial
# weights. ``SHARED_INITIALIZATION_ANCHOR`` is the arm whose seeded initialization
# is copied into every other arm in the group before its first step.
SHARED_INITIALIZATION_ARMS = ("generic", "process")
SHARED_INITIALIZATION_ANCHOR = "generic"
NON_SHARED_ARMS = ("unet", "native_window", "afno_small")

# ---- the pre-registered skill gate (#64 B2 sentence 2) ------------------------
# Written before the confirmatory runs. "beat" means a strictly lower RMSE than
# BOTH parameter-free controls in the same cell, averaged over the declared
# seeds by the #60 comparator's seed-paired rule (a cell counts as beaten only
# when every seed's delta agrees in sign, so seed noise cannot buy a pass).
SKILL_CRITERIA = {
    "format": "r7-b2-skill-gate-v1",
    "frozen_before_any_step": True,
    "statement": (
        "The neural arms must beat BOTH parameter-free controls (persistence and "
        "the train-only month-hour climatology) on the required cells below, with "
        "the win decided per seed and required to agree in sign across the "
        "declared seeds. Cells outside the required list are still reported in "
        "full - they are not part of this gate and are never dropped from the "
        "tables."),
    "required_cells": {
        "6h": {"variables": ["t2m", "mslp", "v850", "u10"],
               "required_wins": 4, "of": 4},
        "12h": {"variables": ["t2m", "mslp", "v850"],
                "required_wins": 2, "of": 3},
        "24h": {"variables": ["mslp", "v850"],
                "required_wins": 1, "of": 2},
    },
    "required_arms": ["native_window", "generic"],
    "rationale": (
        "6 h is the trained horizon, so the gate is strictest there (all four "
        "required cells). 12 h and 24 h are free-rollout extrapolation, so the "
        "gate asks for a majority at 12 h and at least one cell at 24 h rather "
        "than pretending free rollout is as reliable as the trained step. "
        "t2m is included at 6/12 h because it is the variable where B1 found the "
        "climatology hardest to beat; a gate that omitted it would be a gate "
        "chosen to pass."),
    "allowance": {
        "unresolved_cells": 1,
        "note": ("one required cell may be unresolved (per-seed deltas disagreeing "
                 "in sign) instead of won; a required cell that is *worse* than "
                 "either control is always a failure and never consumes allowance"),
    },
    "states_are_separate": (
        "'the strong baselines are usable' and 'ours wins' are different states: "
        "this gate only decides whether the neural arms clear the parameter-free "
        "controls, not whether any recursive arm beats a non-recursive one."),
}

FLOP_CONVENTION = (
    "FlopCounterMode over one forward pass under torch.enable_grad(); counts "
    "linear/conv/matmul and aten-dispatched attention matmuls; elementwise and "
    "normalization ops are not counted; no parameter hooks (SDPA is parameterless "
    "and a parameter hook would silently undercount attention). Backward is "
    "measured separately, never assumed to be 2x forward."
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_config(kind, channels, config):
    return {"in_channels": channels, "out_channels": channels,
            "history_steps": 2, **config}


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_flops(model, batch, *, reasoning_steps, recursive):
    from model.r7_halting import forecast_inputs
    from torch.utils.flop_counter import FlopCounterMode

    inputs = forecast_inputs(batch)
    run = ((lambda: model(inputs, reasoning_steps=reasoning_steps))
           if recursive else (lambda: model(inputs)))
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            run()
        forward = int(counter.get_total_flops())
    model.zero_grad()
    with torch.enable_grad():
        with FlopCounterMode(display=False) as counter:
            output = run()
            output.forecast.square().mean().backward()
        forward_backward = int(counter.get_total_flops())
    model.zero_grad()
    return forward, forward_backward


def _csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _trainable_state(model):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()}


def protocol_payload(manifests_dir, identity, channels, measured):
    """The frozen B2 protocol. Its digest is identical in both phases.

    Phase membership is deliberately **outside** this body: if the digest moved
    between the exploration and confirmatory phases, a confirmatory run could
    never reuse an exploration run, and the "frozen protocol" would be two
    protocols. Which phase is executing is recorded in the result file instead.
    """
    body = {
        "format": "r7-b2-multiseed-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#64 B2",
        "objective": ("a controlled multi-seed confirmatory comparison of five "
                      "parameter-matched neural arms against persistence and a "
                      "train-only climatology on the real B2 segment, with free "
                      "6/12/24/48/72 h rollouts; never a SOTA or convergence claim"),
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False,
            "test_policy": ("sealed: B2's test block is an engineering re-split inside "
                            "the 2016 segment, not the v2 2021 test candidate"),
            "data_identity": str(identity),
            "split_mode": "time_ranges (docs/decisions/0005-*.md, 0008-*.md)",
            "normalization": "store train-only centered mean/std, applied at read time",
            "segment": ("36 consecutive January days; train [01-01, 01-25) is byte-identical "
                        "to the frozen D1 train range (same normalization statistics, same "
                        "94 windows), so B1 and B2 share one train distribution"),
        },
        "phases": {
            "explore": {"seeds": list(EXPLORATION_SEEDS),
                        "purpose": "confirm the protocol trains stably before freezing",
                        "seeds_are_exploratory": True},
            "confirm": {"seeds": list(CONFIRMATORY_SEEDS),
                        "purpose": ("pre-declared confirmatory comparison; no seed dropped, "
                                    "every declared seed reported"),
                        "seeds_are_exploratory": False},
            "reuse_rule": ("a confirmatory seed id that was also an exploration seed reuses "
                           "that run only because this protocol digest is unchanged; the "
                           "digest is phase-independent and confirm refuses to run on "
                           "protocol drift"),
        },
        "seed_pool": list(SEED_POOL),
        "seed_policy": ("seeds are written down before running; every declared seed is "
                        "reported whether or not it favours any arm"),
        "arms": [{"name": name, "kind": kind,
                  "model_config": _arm_config(kind, channels, config),
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in ARMS],
        "baselines": [{"name": name, "trainable_parameters": 0,
                       "note": "parameter-free control; outside the parameter band by design"}
                      for name in BASELINES],
        "parameter_gate": {
            "nominal": NOMINAL_PARAMETERS, "band": PARAMETER_BAND,
            "applies_to": "the four neural arms only; parameter-free controls are not padded",
            "measured_spread": ((max(v["parameters"] for v in measured.values())
                                 - min(v["parameters"] for v in measured.values()))
                                / NOMINAL_PARAMETERS),
        },
        "flop_convention": FLOP_CONVENTION,
        "shared_initialization": {
            "applies_to": list(SHARED_INITIALIZATION_ARMS),
            "anchor": SHARED_INITIALIZATION_ANCHOR,
            "rule": ("the anchor arm is built from the declared seed and its initialized "
                     "state_dict is copied into every other arm of the group before that "
                     "arm's first step; applied/ignored parameter names are recorded in "
                     "each run report, so the alignment is auditable rather than assumed"),
            "not_applicable_to": list(NON_SHARED_ARMS),
            "not_applicable_reason": ("U-Net, Swin-like window and AFNO share no structure "
                                      "with the recursive arms, so they are matched on data "
                                      "and budget only, never claimed as weight-matched"),
        },
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY}",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates to the peak, then "
                            f"cosine decay to {MINIMUM_LR_RATIO} of peak at update "
                            f"{UPDATES}"),
            "warmup_updates": WARMUP_UPDATES,
            "minimum_lr_ratio": MINIMUM_LR_RATIO,
            "max_updates": UPDATES,
            "early_stopping_rule": (f"stop when {EARLY_STOPPING_PATIENCE} consecutive "
                                    f"validation checks fail to improve the validation "
                                    f"objective by {MINIMUM_IMPROVEMENT:.1%} relative; "
                                    "reads validation only"),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "minimum_improvement": MINIMUM_IMPROVEMENT,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "checkpoint_selection_rule": ("keep the checkpoint with the lowest mean "
                                          "latitude-weighted normalized validation MSE; "
                                          "ties keep the earlier checkpoint"),
            "clip": CLIP,
            "batch_size": BATCH_SIZE,
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
            "loss": "latitude-weighted MSE (streamed truncated BPTT for the recursive arms)",
            "reasoning_steps": REASONING_STEPS,
            "process_weight": PROCESS_WEIGHT,
            "target_variable": "the store's 17 declared channels, unmodified",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_policy": ("free autoregressive rollout: one evaluation run per "
                                  "(model, lead) holding all other leads fixed, so the "
                                  "case set per lead is the largest one that supports that "
                                  "horizon; cells are compared only within a lead"),
            "selection_split": "val only; test is sealed and never read",
            "scored_channels": list(SCORED_CHANNELS),
            "comparator": ("training/r7_coreasoning_compare.summarize/compare/evaluate_gate "
                           "(the #60-fixed comparator); no third comparator is built"),
            "comparator_depth": COMPARATOR_DEPTH,
        },
        "identity_note": (
            "the comparator identity carries the comparison's declared update budget. "
            "Parameter-free controls are not trained, so for them that field describes "
            "the comparison they belong to rather than a trained-update count; this is "
            "stated so it is never read as 'the baseline trained N updates'."),
        "compute_alignment": {
            "claimed_aligned": ["data", "seed", "sample order", "normalization",
                                "target variable", "optimizer updates", "batch size",
                                "lr schedule", "checkpoint rule", "early-stopping rule"],
            "explicitly_not_aligned": ["forward/backward FLOPs", "wall time",
                                       "peak memory", "parameter count (bounded, not equal)",
                                       "initial weights for arms sharing no structure"],
            "measurement": ("parameter, FLOPs, wall-time and case-count tables are all "
                            "reported; forward-only FLOPs never stand in for training cost"),
        },
        "skill_criteria": SKILL_CRITERIA,
        "scientific_claim": False,
        "limitations": [
            "B2 is one 36-day engineering segment (January 2016) of real ERA5",
            "all of it is January: no seasonal or cross-year conclusion is testable",
            "climatology has only 4 (month,hour) buckets here, so it is NOT a strong "
            "seasonal climatology; a climatology win on t2m is a property of this "
            "data range, not evidence that the baseline is strong",
            "the val/test blocks are engineering re-splits inside 2016, not the v2 2019 "
            "validation year or the 2021 test candidate",
            "not a converged benchmark; bounded GPU checkpoints",
            "the skill gate is a descriptive pre-registered check, not a significance test",
        ],
    }
    from training.r7_experiment import canonical_digest

    return dict(body, protocol_sha256=canonical_digest(body))


def _write_protocol(path, protocol):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _read_protocol(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _trainable_state(model):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()}


def run_phase(manifests_dir, output_dir, *, phase, seeds, updates=UPDATES,
              device_name="cuda", protocol_file=None):
    """Run one phase (explore or confirm) of the B2 comparison."""
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (dataset_identity, load_checkpoint, make_model,
                                        model_code_digest, seed_everything, select_device)
    from training.r7_scheduled_runner import run_scheduled_updates

    if phase not in ("explore", "confirm"):
        raise ValueError("phase must be 'explore' or 'confirm'")
    declared = EXPLORATION_SEEDS if phase == "explore" else CONFIRMATORY_SEEDS
    unexpected = [seed for seed in seeds if seed not in declared]
    if unexpected:
        raise ValueError(f"phase {phase!r} declares seeds {declared}; {unexpected} are not "
                         "among them, and B2 does not run undeclared seeds")

    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    train_manifest = manifests_dir / "train.jsonl"
    val_manifest = manifests_dir / "val.jsonl"
    for manifest in (train_manifest, val_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")

    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    channels = int(dataset[0]["coarse_history"].shape[1])
    device = select_device(device_name)
    from torch.utils.data import default_collate
    probe_batch = default_collate([dataset[0], dataset[1]])
    # The frozen protocol selects and early-stops on the validation split, so the
    # validation windows have to exist before training starts; a missing val set
    # must fail here rather than degrade into "no validation" silently.
    from data.r7_evaluation import ZarrRolloutDataset
    validation_dataset = ZarrRolloutDataset(manifests_dir.parent / "cache.zarr", split="val",
                                            lead_hours=VALIDATION_LEADS, history_steps=2,
                                            step_hours=6)

    measured = {}
    for name, kind, config in ARMS:
        seed_everything(SEED_POOL[0])
        model = make_model(kind, _arm_config(kind, channels, config))
        forward, forward_backward = count_flops(
            model, probe_batch, reasoning_steps=REASONING_STEPS, recursive=(kind != "native"))
        measured[name] = {"parameters": count_parameters(model),
                          "forward_flops": forward,
                          "forward_backward_flops": forward_backward}
        del model

    protocol = protocol_payload(manifests_dir, identity, channels, measured)
    if protocol_file is not None:
        frozen = _read_protocol(protocol_file)
        if frozen.get("protocol_sha256") != protocol.get("protocol_sha256"):
            raise RuntimeError(
                "protocol drift: the frozen protocol file does not match the protocol "
                "this code would run; a pre-registered protocol may not be edited after "
                "the exploration phase")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_protocol(output_dir / "protocol.json", protocol)

    digest = model_code_digest()
    results = {
        "format": "r7-b2-multiseed-result-v1",
        "phase": phase, "seeds": list(seeds),
        "scientific_claim": False, "test_read": False, "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "model_code_sha256": digest, "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "training": {}, "evaluation": {}, "flop_measurements": measured, "budget": {},
        "shared_initialization": {},
    }

    training_started = time.perf_counter()
    anchors = {}
    for seed in seeds:
        seed_results = {}
        for name, kind, config in ARMS:
            run_dir = output_dir / "training" / f"seed{seed}" / name
            resolved = _arm_config(kind, channels, config)
            shared = None
            if name in SHARED_INITIALIZATION_ARMS:
                if name == SHARED_INITIALIZATION_ANCHOR:
                    # Build the anchor's seeded initialization once per seed, and
                    # remember it so every other arm in the group starts from
                    # exactly these common parameters.
                    seed_everything(seed)
                    anchor_model = make_model(kind, resolved)
                    anchors[(seed, name)] = _trainable_state(anchor_model)
                    del anchor_model
                else:
                    shared = anchors.get((seed, SHARED_INITIALIZATION_ANCHOR))
                    if shared is None:
                        raise RuntimeError(
                            f"the shared-initialization anchor "
                            f"{SHARED_INITIALIZATION_ANCHOR!r} must run before {name!r}")
            checkpoint, report = run_scheduled_updates(
                dataset, kind=kind, model_config=resolved,
                data_identity=identity, output_dir=run_dir, total_updates=updates,
                batch_size=BATCH_SIZE, steps=REASONING_STEPS, seed=seed, lr=LR,
                clip=CLIP, process_weight=PROCESS_WEIGHT,
                warmup_updates=WARMUP_UPDATES, minimum_lr_ratio=MINIMUM_LR_RATIO,
                validation_every=VALIDATION_EVERY,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                minimum_improvement=MINIMUM_IMPROVEMENT,
                validation_lead_hours=VALIDATION_LEADS, device_name=device_name,
                validation_dataset=validation_dataset,
                shared_initial_state=shared)
            saved = load_checkpoint(checkpoint)
            if saved["updates"] != report["selected_update"]:
                raise ValueError(f"{name} selected checkpoint/update mismatch")
            samples_seen = int(sum(entry["samples"] for entry in report["losses"]))
            by_epoch = {}
            for entry in report["losses"]:
                by_epoch.setdefault(entry["epoch"], []).append(entry["loss"])
            epoch_means = {str(epoch): sum(values) / len(values)
                           for epoch, values in sorted(by_epoch.items())}
            epoch_sizes = {str(epoch): len(values) for epoch, values in sorted(by_epoch.items())}
            full = max(epoch_sizes.values()) if epoch_sizes else 0
            full_epochs = [epoch for epoch, size in epoch_sizes.items() if size == full]
            first_full, last_full = (min(full_epochs), max(full_epochs)) if full_epochs else (None, None)
            seed_results[name] = {
                "updates_run": report["updates_this_run"],
                "selected_update": report["selected_update"],
                "selected_validation_mse": report["selected_validation_mse"],
                "early_stopped": report["early_stopped"],
                "stopped_reason": report["stopped_reason"],
                "samples_seen": samples_seen,
                "elapsed_seconds": report["elapsed_seconds"],
                "seconds_per_update": report["seconds_per_update"],
                "wall_time_includes_io": True,
                "epoch_mean_loss": epoch_means,
                "epoch_update_counts": epoch_sizes,
                "first_full_epoch": first_full, "last_full_epoch": last_full,
                "loss_ratio_first_to_last_full_epoch": (
                    epoch_means[str(last_full)] / epoch_means[str(first_full)]
                    if full_epochs else None),
                "single_point_note": ("first/last single-update losses are noisy; the "
                                      "full-epoch mean ratio is the reported trend"),
                "validation_checks": report["validations"],
                "peak_allocated_bytes": report["peak_allocated_bytes"],
                "peak_reserved_bytes": report["peak_reserved_bytes"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "parameters": measured[name]["parameters"],
                "shared_initial_state": report["shared_initial_state"],
            }
            print(json.dumps({"trained": name, "seed": seed,
                              "selected_update": report["selected_update"],
                              "epochs": epoch_means,
                              "seconds": round(report["elapsed_seconds"], 1),
                              "early_stopped": report["early_stopped"]}), flush=True)
        results["training"][str(seed)] = seed_results
    results["budget"]["training_seconds_total"] = time.perf_counter() - training_started

    # ---- evaluate every arm and both parameter-free controls --------------------
    for seed in seeds:
        for name in list(ARM_NAMES) + list(BASELINES):
            for lead in EVALUATION_LEADS:
                run_dir = output_dir / "evaluation" / f"seed{seed}" / name / f"lead_{lead:03d}h"
                trained = name not in BASELINES
                report = evaluate_local(
                    val_manifest, output_dir=run_dir,
                    checkpoint=(output_dir / "training" / f"seed{seed}" / name
                                / f"update_{results['training'][str(seed)][name]['selected_update']:07d}.pt")
                    if trained else None,
                    baseline=None if trained else name,
                    lead_hours=(lead,), max_samples=64, device_name=device_name,
                    reasoning_steps=REASONING_STEPS if trained and name != "unet" else None)
                if report["split"] != "val":
                    raise ValueError(f"{name} was evaluated on split {report['split']}, not val")
                results["evaluation"][f"seed{seed}/{name}@{lead}h"] = {
                    "seed": seed, "model": name, "lead_hours": lead, "split": report["split"],
                    "n_evaluated": report["n_evaluated"],
                    "n_available_windows": report["n_available_windows"],
                    "channels": list(report["channels"]), "units": list(report["units"]),
                    "elapsed_seconds": report["elapsed_seconds"],
                    "timing_scope": report["timing_scope"],
                    "parameter_free_baseline": report["parameter_free_baseline"],
                    "trainable_parameters": report["trainable_parameters"],
                    "climatology": {key: value for key, value in report["climatology"].items()
                                    if key != "bucket_counts"},
                    "acc_skill_identity": report["acc_skill_identity"],
                    "evaluation_dir": str(run_dir),
                    "rmse_csv": str(run_dir / "rmse.csv"),
                    "skill_csv": str(run_dir / "climatology_skill.csv"),
                    "acc_csv": str(run_dir / "acc.csv"),
                }
        print(json.dumps({"evaluated_seed": seed, "leads": list(EVALUATION_LEADS)}), flush=True)

    # every model must have been scored on the identical case set at each lead
    models = list(ARM_NAMES) + list(BASELINES)
    for lead in EVALUATION_LEADS:
        for seed in seeds:
            counts = {name: results["evaluation"][f"seed{seed}/{name}@{lead}h"]["n_evaluated"]
                      for name in models}
            if len(set(counts.values())) != 1:
                raise RuntimeError(f"models scored on different case counts at {lead}h "
                                   f"seed {seed}: {counts}")
            results.setdefault("case_counts_by_lead", {}).setdefault(
                f"{lead}h", {})[str(seed)] = counts[models[0]]

    with (output_dir / "b2_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


def _required_lead_hours(label):
    """The integer horizon a required-cell label names (``"6h"`` -> ``6``).

    The frozen criteria are stored exactly as they were written before any run,
    so the label is the declaration and this function is the single place that
    interprets it. Keeping the interpretation here (rather than adding a
    ``lead_hours`` field to the frozen text) is what lets the exploration-phase
    runs stay valid: changing the criteria after the first run would be exactly
    the post-hoc tuning the pre-registration exists to prevent.
    """
    text = str(label).strip().lower()
    if not text.endswith("h") or not text[:-1].isdigit():
        raise ValueError(f"required-cell label {label!r} is not a '<hours>h' horizon")
    hours = int(text[:-1])
    if hours not in EVALUATION_LEADS:
        raise ValueError(f"required cell {label!r} is not among the evaluated leads "
                         f"{EVALUATION_LEADS}")
    return hours


def _unit_for(table, variable):
    for row in table:
        if row["variable"] == variable:
            return row["unit"]
    raise ValueError(f"variable {variable!r} is absent from the comparator table")


def _cell_win_table(comparisons):
    """Win/unresolved/loss per (arm, control, lead, variable) from the comparator."""
    outcomes = {}
    for control, blocks in comparisons.items():
        for block in blocks:
            arm = block["arm"]
            for entry in block["seed_paired"]:
                key = f"{control}|{int(entry['lead_hours'])}|{entry['variable']}"
                outcomes.setdefault(arm, {})[key] = (
                    "won" if entry["direction"] == "improved" else
                    "lost" if entry["direction"] == "worsened" else "unresolved")
    return outcomes


def score_with_comparator(results, output_dir, *, updates=UPDATES):
    """Score every arm through the #60-fixed comparator and read back the gate.

    No third comparator is built: every (arm, seed) evaluation directory feeds
    ``summarize``, then ``compare`` produces the seed-paired blocks against
    persistence and against climatology, and ``evaluate_gate`` reads the
    pre-frozen criteria. The identity carries the comparison's declared update
    budget - a protocol field for the parameter-free controls, stated in the
    protocol rather than silently written as a trained-update count.
    """
    from training.r7_coreasoning_compare import compare, evaluate_gate, summarize

    output_dir = Path(output_dir)
    identity = {
        "dataset_identity": results["protocol"]["data"]["data_identity"],
        "model_code_sha256": results["model_code_sha256"],
        "declared_update_budget": int(updates),
        "evaluation_split": "val",
    }
    records = [{"arm": entry["model"], "seed": entry["seed"], "depth": COMPARATOR_DEPTH,
                "evaluation_dir": entry["evaluation_dir"], "identity": identity}
               for entry in results["evaluation"].values()]
    table = summarize(records)
    comparisons = {control: compare(table, baseline=control, depth=COMPARATOR_DEPTH)
                   for control in BASELINES}

    gates = {}
    for control, blocks in comparisons.items():
        for arm in SKILL_CRITERIA["required_arms"]:
            required = []
            for label, spec in SKILL_CRITERIA["required_cells"].items():
                for variable in spec["variables"]:
                    required.append({"variable": variable,
                                     "unit": _unit_for(table, variable),
                                     "lead_hours": float(_required_lead_hours(label))})
            gates[f"{arm}_vs_{control}"] = evaluate_gate(blocks, {
                "arm": arm, "baseline": control, "required_variables": required,
                "required_direction": "improved",
                # The comparator's allowance is per-variable; the frozen gate's
                # allowance is per-cell and is enforced below on the cell counts,
                # so it stays at zero here rather than being double-counted.
                "allow_unresolved_required": 0,
            })

    cell = _cell_win_table(comparisons)
    allowance = SKILL_CRITERIA["allowance"]["unresolved_cells"]
    verdicts = {}
    for arm in SKILL_CRITERIA["required_arms"]:
        arm_result = {"per_control": {}, "cells": cell.get(arm, {})}
        for control in BASELINES:
            required_wins, won, unresolved, lost = 0, 0, 0, 0
            for label, spec in SKILL_CRITERIA["required_cells"].items():
                required_wins += spec["required_wins"]
                for variable in spec["variables"]:
                    outcome = cell.get(arm, {}).get(
                        f"{control}|{_required_lead_hours(label)}|{variable}")
                    if outcome == "won":
                        won += 1
                    elif outcome == "unresolved":
                        unresolved += 1
                    else:
                        lost += 1
            arm_result["per_control"][control] = {
                "required_wins": required_wins, "won": won,
                "unresolved": unresolved, "lost": lost,
                "comparator_gate_met": gates[f"{arm}_vs_{control}"]["gate_met"],
                "comparator_failures": gates[f"{arm}_vs_{control}"]["failures"],
            }
        arm_result["gate_met"] = all(
            entry["won"] >= entry["required_wins"] and entry["lost"] == 0
            and entry["unresolved"] <= allowance
            for entry in arm_result["per_control"].values())
        arm_result["rationale"] = (
            "passed only when BOTH parameter-free controls are cleared: the required "
            "cells must be won, no required cell may be worse, and at most "
            f"{allowance} required cell may be unresolved")
        verdicts[arm] = arm_result

    payload = {
        "format": "r7-b2-skill-gate-result-v1",
        "scientific_claim": False,
        "criteria": SKILL_CRITERIA,
        "identity": identity,
        "comparator": "training/r7_coreasoning_compare (the #60-fixed comparator)",
        "cell_outcomes": cell,
        "comparator_gates": gates,
        "verdicts": verdicts,
        "gate_met": all(entry["gate_met"] for entry in verdicts.values()),
        "note": ("descriptive pre-registered check; sign consistency across three seeds is "
                 "not a significance test, and this says nothing about whether a recursive "
                 "arm beats a non-recursive one"),
    }
    for name, content in (("b2_skill_gate.json", payload),
                          ("b2_comparator_summary.json", {"table": table,
                                                          "comparisons": comparisons})):
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return payload


def write_tables(results, output_dir):
    """The four required tables: parameters, FLOPs, wall time, cases - plus RMSE."""
    output_dir = Path(output_dir)
    tables = {}
    models = list(ARM_NAMES) + list(BASELINES)

    parameters_path = output_dir / "b2_parameter_table.csv"
    with parameters_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "family", "trainable_parameters",
                         "deviation_from_nominal", "inside_band",
                         "forward_flops", "forward_backward_flops", "fwd_bwd_over_fwd"])
        for entry in results["protocol"]["arms"]:
            deviation = (entry["parameters"] - NOMINAL_PARAMETERS) / NOMINAL_PARAMETERS
            writer.writerow([entry["name"], entry["kind"], entry["parameters"],
                             f"{deviation:+.4%}", str(abs(deviation) <= PARAMETER_BAND),
                             entry["forward_flops"], entry["forward_backward_flops"],
                             f"{entry['forward_backward_flops'] / entry['forward_flops']:.3f}"])
        for name in BASELINES:
            writer.writerow([name, "parameter-free", 0, "n/a", "not_in_band",
                             "0", "0", "n/a"])
    tables["parameters"] = parameters_path.name

    flops_path = output_dir / "b2_flops_table.csv"
    with flops_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "forward_flops", "forward_backward_flops", "batch",
                         "reasoning_steps", "convention"])
        for entry in results["protocol"]["arms"]:
            writer.writerow([entry["name"], entry["forward_flops"],
                             entry["forward_backward_flops"], BATCH_SIZE,
                             REASONING_STEPS if entry["kind"] != "native"
                             or entry["name"] == "generic" else "n/a",
                             FLOP_CONVENTION])
        for name in BASELINES:
            writer.writerow([name, "0", "0", BATCH_SIZE, "n/a",
                             "parameter-free control: no learnable compute"])
    tables["flops"] = flops_path.name

    wall_path = output_dir / "b2_wall_time_table.csv"
    with wall_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "model", "updates_run", "selected_update", "samples_seen",
                         "train_seconds", "seconds_per_update", "early_stopped",
                         "wall_time_includes_io", "eval_seconds_total", "peak_reserved_mib"])
        for seed, per_seed in sorted(results["training"].items(), key=lambda kv: int(kv[0])):
            for name in models:
                train = per_seed.get(name)
                entries = [entry for entry in results["evaluation"].values()
                           if entry["model"] == name and str(entry["seed"]) == seed]
                eval_seconds = sum(entry["elapsed_seconds"] for entry in entries)
                writer.writerow([
                    seed, name,
                    train["updates_run"] if train else 0,
                    train["selected_update"] if train else 0,
                    train["samples_seen"] if train else 0,
                    f"{train['elapsed_seconds']:.3f}" if train else "0.000",
                    f"{train['seconds_per_update']:.6f}" if train else "0.000000",
                    str(bool(train["early_stopped"])) if train else "n/a (no training)",
                    "True" if train else "n/a (no training)",
                    f"{eval_seconds:.3f}",
                    f"{(train['peak_reserved_bytes'] or 0) / 2**20:.1f}" if train else "0.0"])
    tables["wall_time"] = wall_path.name

    cases_path = output_dir / "b2_case_count_table.csv"
    with cases_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "model", "split", "lead_hours", "n_available_windows",
                         "n_evaluated", "test_read"])
        for key in sorted(results["evaluation"],
                          key=lambda k: (results["evaluation"][k]["seed"],
                                         results["evaluation"][k]["model"],
                                         results["evaluation"][k]["lead_hours"])):
            entry = results["evaluation"][key]
            writer.writerow([entry["seed"], entry["model"], entry["split"], entry["lead_hours"],
                             entry["n_available_windows"], entry["n_evaluated"], "False"])
    tables["cases"] = cases_path.name

    rmse_path = output_dir / "b2_rmse_table.csv"
    with rmse_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "model", "variable", "unit", "lead_hours", "rmse",
                         "rmse_climatology", "mse_skill", "n_initializations"])
        for key in sorted(results["evaluation"],
                          key=lambda k: (results["evaluation"][k]["seed"],
                                         results["evaluation"][k]["model"],
                                         results["evaluation"][k]["lead_hours"])):
            entry = results["evaluation"][key]
            rmse = {row["variable"]: row for row in _csv_rows(entry["rmse_csv"])}
            skill = {row["variable"]: row for row in _csv_rows(entry["skill_csv"])}
            for variable in entry["channels"]:
                row = rmse.get(variable)
                if row is None:
                    continue
                skill_row = skill.get(variable, {})
                writer.writerow([entry["seed"], entry["model"], variable, row["unit"],
                                 entry["lead_hours"], row["rmse"],
                                 skill_row.get("rmse_climatology", ""),
                                 skill_row.get("mse_skill", "") or "undefined",
                                 row.get("n_initializations", entry["n_evaluated"])])
    tables["rmse"] = rmse_path.name
    return tables


def merge_phase_runs(run_root, *, phase, seeds):
    """Merge per-seed phase runs into one result, refusing an incomplete phase.

    Seeds run in separate processes (one per GPU) so each writes its own result
    file. Merging refuses to proceed unless every pre-declared seed for the phase
    is present and all of them ran under the same protocol and model code, so an
    incomplete phase can never be reported as a narrowed one.
    """
    run_root = Path(run_root)
    declared = EXPLORATION_SEEDS if phase == "explore" else CONFIRMATORY_SEEDS
    unexpected = [seed for seed in seeds if seed not in declared]
    if unexpected:
        raise ValueError(f"phase {phase!r} declares seeds {declared}; {unexpected} are not "
                         "among them")
    loaded = {}
    for seed in seeds:
        path = run_root / f"seed{seed}" / "b2_result.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"phase {phase!r} declares seed {seed} but {path} is missing; an incomplete "
                "phase is reported as incomplete, never silently narrowed")
        loaded[seed] = json.loads(path.read_text(encoding="utf-8"))
    digests = {result["protocol_sha256"] for result in loaded.values()}
    if len(digests) != 1:
        raise RuntimeError(f"seeds ran under different protocols: {sorted(digests)}")
    codes = {result["model_code_sha256"] for result in loaded.values()}
    if len(codes) != 1:
        raise RuntimeError(f"seeds ran on different model code: {sorted(codes)}")

    first = loaded[seeds[0]]
    merged = {
        "format": "r7-b2-multiseed-result-v1",
        "phase": phase, "seeds": list(seeds),
        "scientific_claim": False, "test_read": False,
        "protocol": first["protocol"], "protocol_sha256": first["protocol_sha256"],
        "model_code_sha256": first["model_code_sha256"],
        "flop_measurements": first["flop_measurements"],
        "segments": {str(seed): {"device": result["device"], "gpu": result["gpu"],
                                 "torch_version": result["torch_version"],
                                 "platform": result["platform"]}
                     for seed, result in loaded.items()},
        "training": {}, "evaluation": {}, "case_counts_by_lead": {},
        "budget": {"training_seconds_total": 0.0},
    }
    for seed, result in loaded.items():
        merged["training"].update(result["training"])
        merged["evaluation"].update(result["evaluation"])
        merged["case_counts_by_lead"].update(result["case_counts_by_lead"])
        merged["budget"]["training_seconds_total"] += result["budget"]["training_seconds_total"]
    # Which phase actually produced each seed's run, and whether it was inherited
    # from an earlier phase. Reuse across phases is declared in the protocol's
    # reuse_rule; recording it per seed keeps that auditable instead of implied.
    merged["seed_provenance"] = {
        str(seed): {"run_phase": result["phase"],
                    "reused_from_earlier_phase": result["phase"] != phase}
        for seed, result in loaded.items()}
    if not merged["training"] or not merged["evaluation"]:
        raise RuntimeError("merged result is empty; refusing to score nothing")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="B2 controlled multiseed comparison on the real B2 segment (#64 B2).")
    parser.add_argument("--mode", required=True, choices=("seed", "finalize"),
                        help=("seed: run one seed in its own process (two GPUs run two "
                               "seeds in parallel); finalize: merge the phase's seeds, "
                               "emit the tables and read the pre-registered gate"))
    parser.add_argument("--seed", type=int, default=None, help="seed mode: which seed")
    parser.add_argument("--manifests", required=True, help="B2 store manifest directory")
    parser.add_argument("--out", required=True,
                        help="run root; seed mode creates <out>/seed<N>/, finalize reads it")
    parser.add_argument("--phase", required=True, choices=("explore", "confirm"))
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=None,
                        help="CUDA device to pin (the two 3090s are not NVLink-coupled)")
    parser.add_argument("--protocol", default=None,
                        help="path to a frozen protocol.json to refuse drift against")
    args = parser.parse_args()
    declared = EXPLORATION_SEEDS if args.phase == "explore" else CONFIRMATORY_SEEDS

    if args.mode == "seed":
        if args.seed is None:
            parser.error("--mode seed requires --seed")
        if args.seed not in declared:
            parser.error(f"--phase {args.phase} declares seeds {declared}; {args.seed} is "
                         "not among them")
        if args.device_index is not None and args.device.startswith("cuda"):
            torch.cuda.set_device(args.device_index)
            args.device = f"cuda:{args.device_index}"
        run_phase(args.manifests, Path(args.out) / f"seed{args.seed}", phase=args.phase,
                  seeds=(args.seed,), updates=args.updates, device_name=args.device,
                  protocol_file=args.protocol)
        return

    results = merge_phase_runs(args.out, phase=args.phase, seeds=declared)
    with (Path(args.out) / "b2_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    tables = write_tables(results, args.out)
    gate = score_with_comparator(results, args.out, updates=args.updates)
    print(json.dumps({
        "phase": args.phase, "seeds": results["seeds"],
        "parameter_spread": results["protocol"]["parameter_gate"]["measured_spread"],
        "case_counts_by_lead": results["case_counts_by_lead"],
        "training_seconds_total": results["budget"]["training_seconds_total"],
        "tables": tables, "protocol_sha256": results["protocol_sha256"],
        "skill_gate_met": gate["gate_met"],
        "verdicts": {arm: entry["gate_met"] for arm, entry in gate["verdicts"].items()},
        "scientific_claim": False,
    }, indent=1))


if __name__ == "__main__":
    main()
