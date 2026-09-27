"""#65 minimal ablation: process supervision (C1), draft feedback (C2), reasoning
depth and compute (C3) on the real B2 segment.

Why this exists. The B2/B3 recursive pair was compared at auxiliary weight 0,
and the #65 pre-diagnostic showed that in that configuration the ``process`` arm
is the ``generic`` arm with a renamed recurrent state plus a readout head that
receives no gradient at all. That is not a test of process supervision. This
harness runs the comparison the issue actually asks for, with the auxiliary
weight **engaged** and the arms otherwise aligned, and it reports the four
tables ("same updates is not same compute") next to the accuracy table.

Frozen before any step
----------------------
The protocol body below - arms, auxiliary weights, seeds, update budget,
schedule, checkpoint rule, early-stopping rule, evaluation leads and the
decision rule - is written to ``protocol.json`` and hashed before the first
optimizer step. The digest is re-derived by every run and compared, so a later
edit to the criteria is visible as drift rather than silently accepted.

Scope discipline
----------------
- ``test`` is never read (``test_read: false``); selection is validation-only.
- One bounded segment, one season: ``REASONING_STEPS`` is *internal* depth and
  never advances physical time; the +6h..+72h leads are a separate axis.
- Failure leaves the output directory in place. A crashed or cancelled run is
  not a result and is never overwritten by a rerun into the same path.

    # C1: does an engaged auxiliary loss change anything?
    python scripts/study_r7_65_ablation.py --phase c1 --device cuda --out outputs/r7_65_c1

    # C2: draft feedback into the reasoner vs into the solver
    python scripts/study_r7_65_ablation.py --phase c2 --device cuda --out outputs/r7_65_c2

    # C3: test-time K sweep from one checkpoint + an independently trained K=1
    python scripts/study_r7_65_ablation.py --phase c3 --device cuda --out outputs/r7_65_c3
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

# ---- frozen budget and schedule (identical to B2 so the two are comparable) --
TRAIN_SEEDS = (41, 42, 43)
UPDATES = 800
BATCH_SIZE = 2
LR = 2e-4
CLIP = 1.0
WARMUP_UPDATES = 80
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
# C1 freezes the training depth at 4 as the issue specifies.
TRAIN_REASONING_STEPS = 4
# C3 sweeps inference depth from one C1 checkpoint and additionally trains an
# independent shallow model, because "K=1 evaluated from a K=4 checkpoint" and
# "a model trained at K=1" are different questions.
INFERENCE_STEPS_C3 = (1, 2, 4)
SHALLOW_TRAINED_STEPS = 1

DEFAULT_MANIFESTS = "outputs/r7_b2_segment/store/manifests"

# ---- C1: the auxiliary supervision axis (#65 C1) -----------------------------
# ``generic16`` is the free-token control; ``process8+8`` is the same structure
# with the anchored readout. The three process rows are the SAME architecture at
# three auxiliary weights, so the only thing that moves inside that group is
# whether - and how hard - the process readout is trained.
C1_AUXILIARY_WEIGHTS = (0.0, 0.01, 0.1)
C1_ARMS = (
    ("generic16_aux0", "generic", {"latent_tokens": 16}, 0.0),
    ("process8_aux0", "process", {}, 0.0),
    ("process8_aux001", "process", {}, 0.01),
    ("process8_aux010", "process", {}, 0.1),
)

# ---- C2: which feedback path (#65 C2) ---------------------------------------
# Both arms keep the process structure and a fixed train depth; the difference is
# whether the recurrent reasoner sees the re-encoded draft, whether only the
# solver does, or neither. Generic gets the same spatial mechanism when it is
# switched on for process, so the process arm is never the only one enhanced.
C2_ARMS = (
    ("process_fb_on_solver_off", "process",
     {"use_forecast_feedback": True, "spatial_solver_feedback": False}),
    ("process_fb_off_solver_off", "process",
     {"use_forecast_feedback": False, "spatial_solver_feedback": False}),
    ("process_fb_on_solver_on", "process",
     {"use_forecast_feedback": True, "spatial_solver_feedback": True}),
    ("generic_fb_on_solver_on", "generic",
     {"use_forecast_feedback": True, "spatial_solver_feedback": True}),
)

# The common window/attention configuration, matching B2 at 2.8M so the
# ablation is described against the same baseline family.
BASE_ARCHITECTURE = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                     "window_size": 4, "patch_size": 2, "dropout": 0.0}

FLOP_CONVENTION = (
    "FlopCounterMode over one forward pass under torch.enable_grad(); counts "
    "linear/conv/matmul and aten-dispatched attention matmuls; elementwise and "
    "normalization ops are not counted; no parameter hooks (SDPA is parameterless "
    "and a parameter hook would silently undercount attention). Backward is "
    "measured separately and never assumed to be exactly 2x forward."
)

LIMITATIONS = [
    "one 36-day engineering segment (January 2016) of real ERA5; single season",
    "climatology here has only 4 (month,hour) buckets, so it is not a strong "
    "seasonal baseline and a climatology win is a property of this data range",
    "val/test are engineering re-splits inside 2016, not the v2 validation year "
    "or the 2021 test candidate; test is sealed and never read",
    "three seeds are not a significance test; agreement in sign is descriptive",
    "bounded checkpoints, not converged training runs",
    "the internal reasoning depth K never advances physical time",
    "no precipitation target exists in this store, so no precipitation skill is claimed",
]
SCIENTIFIC_CLAIM = False


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_config(kind, channels, extra):
    """Build a model config, dropping keys the target kind does not accept.

    ``generic`` always re-encodes the draft into its recurrent context, so it has
    no ``use_forecast_feedback`` switch; passing one would be a TypeError rather
    than a silently ignored argument. The routing question C2 asks is therefore
    only asked of the process arm, and the generic arm that receives the same
    *spatial solver* mechanism is matched on that mechanism alone.
    """
    config = {"in_channels": channels, "out_channels": channels,
              "history_steps": 2, **BASE_ARCHITECTURE, **extra}
    if kind == "process":
        config.setdefault("anchored_processes", 8)
        config.setdefault("free_processes", 8)
        config.setdefault("use_forecast_feedback", True)
    else:
        config.pop("use_forecast_feedback", None)
    return config


def _phase_arms(phase):
    if phase == "c1":
        return tuple((name, kind, dict(extra), weight)
                     for name, kind, extra, weight in C1_ARMS)
    if phase == "c2":
        return tuple((name, kind, dict(extra), 0.1)
                     for name, kind, extra in C2_ARMS)
    # C3 trains the same C1 process arm at the engaged weight plus a shallow
    # control. The training depth is part of the frozen protocol, so it is
    # declared per arm in the protocol body rather than read off the model
    # config: a model whose ``default_reasoning_steps`` is 1 still trains at
    # whatever depth the runner is told, and those must not silently disagree.
    return (
        ("process8_aux010_k4", "process",
         {"use_forecast_feedback": True}, 0.1, 4),
        ("process8_aux010_k1", "process",
         {"use_forecast_feedback": True, "default_reasoning_steps": SHALLOW_TRAINED_STEPS},
         0.1, SHALLOW_TRAINED_STEPS),
    )


def _arm_train_steps(phase, name):
    """The declared training depth of one arm in a phase."""
    for entry in _phase_arms(phase):
        if entry[0] == name:
            return entry[4] if len(entry) > 4 else TRAIN_REASONING_STEPS
    raise ValueError(f"unknown arm {name!r} for phase {phase!r}")


def protocol_payload(phase, manifests_dir, identity, channels, measured):
    """The frozen body. Phase membership lives outside it so the digest cannot
    move between phases."""
    arms = []
    for entry in _phase_arms(phase):
        name, kind, extra, weight = entry[0], entry[1], entry[2], entry[3]
        arms.append({
            "name": name, "kind": kind,
            "model_config": _arm_config(kind, channels, extra),
            "auxiliary_weight": weight,
            "train_reasoning_steps": entry[4] if len(entry) > 4 else TRAIN_REASONING_STEPS,
            "parameters": measured[name]["parameters"],
            "forward_flops": measured[name]["forward_flops"],
            "forward_backward_flops": measured[name]["forward_backward_flops"],
        })
    body = {
        "format": "r7-65-ablation-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#65",
        "phase": phase,
        "objective": ("the minimal ablation #65 asks for: process supervision (C1), "
                      "draft-feedback routing (C2) and reasoning depth / test-time "
                      "compute (C3), each against a control that shares data, budget "
                      "and initialization; never a SOTA or convergence claim"),
        "data": {
            "manifests_dir": str(manifests_dir),
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "data_identity": str(identity),
            "test_read": False,
            "test_policy": "sealed: an engineering re-split inside 2016",
            "normalization": "store train-only centered mean/std, applied at read time",
        },
        "seeds": list(TRAIN_SEEDS),
        "seed_policy": "declared before running; every declared seed is reported",
        "arms": arms,
        "shared_controls": {
            "optimizer": "AdamW",
            "lr": LR, "weight_decay": 1e-4, "clip": CLIP,
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates then cosine "
                            f"to {MINIMUM_LR_RATIO} of peak at update {UPDATES}"),
            "max_updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "train_reasoning_steps": {
                spec[0]: (spec[4] if len(spec) > 4 else TRAIN_REASONING_STEPS)
                for spec in _phase_arms(phase)},
            "early_stopping_rule": (f"stop when {EARLY_STOPPING_PATIENCE} consecutive "
                                    f"validation checks fail to improve by "
                                    f"{MINIMUM_IMPROVEMENT:.1%} relative; validation only"),
            "checkpoint_selection_rule": ("lowest mean latitude-weighted normalized "
                                          "validation MSE; ties keep the earlier one"),
            "selection_split": "val only",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_policy": ("free autoregressive rollout; one evaluation per "
                                  "(model, lead) so each lead uses the largest case set "
                                  "that supports it; cells are compared only within a lead"),
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
        },
        "c3_design": {
            "inference_steps_from_one_checkpoint": list(INFERENCE_STEPS_C3),
            "independently_trained_shallow_steps": SHALLOW_TRAINED_STEPS,
            "note": ("evaluating K=1 from a K=4 checkpoint and training a model at K=1 "
                     "are different questions and are reported separately; K above the "
                     "trained depth is not called free scaling"),
        },
        "decision_rule": {
            "primary_variable": "t2m",
            "secondary_variables": ["mslp", "v850", "u10"],
            "cost_columns_required": ["parameters", "forward_flops",
                                      "forward_backward_flops", "wall_time_seconds",
                                      "case_counts"],
            "statement": ("an arm is only described as better when it improves the "
                          "primary variable at the trained lead with the same sign in "
                          "every declared seed; anything else is reported as unresolved "
                          "or worse. No arm is dropped from the tables."),
        },
        "flop_convention": FLOP_CONVENTION,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "limitations": LIMITATIONS,
    }
    from training.r7_experiment import canonical_digest

    return dict(body, protocol_sha256=canonical_digest(body))


def _write_json_exclusive(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(path, rows, fieldnames):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
            loss = output.forecast.square().mean()
            if getattr(output, "process_predictions", None) is not None \
                    and output.process_predictions.shape[1] > 0:
                loss = loss + output.process_predictions.square().mean()
            loss.backward()
        forward_backward = int(counter.get_total_flops())
    model.zero_grad()
    return forward, forward_backward


def run_phase(manifests_dir, output_dir, *, phase, seeds=TRAIN_SEEDS,
              updates=UPDATES, device_name="cuda", protocol_file=None):
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from data.r7_evaluation import ZarrRolloutDataset
    from torch.utils.data import default_collate
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (dataset_identity, make_model, model_code_digest,
                                        seed_everything, select_device)
    from training.r7_scheduled_runner import run_scheduled_updates

    if phase not in ("c1", "c2", "c3"):
        raise ValueError("phase must be 'c1', 'c2' or 'c3'")
    unexpected = [seed for seed in seeds if seed not in TRAIN_SEEDS]
    if unexpected:
        raise ValueError(f"undeclared seeds {unexpected}; #65 declares {TRAIN_SEEDS}")

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
    probe_batch = default_collate([dataset[0], dataset[1]])
    validation_dataset = ZarrRolloutDataset(
        manifests_dir.parent / "cache.zarr", split="val", lead_hours=VALIDATION_LEADS,
        history_steps=2, step_hours=6)

    arms = _phase_arms(phase)
    # The frozen numbers apply to the declared budget. A short smoke run keeps
    # the same shape but a warmup that fits inside it, so the smoke run is a
    # plumbing check rather than a silently different protocol.
    if updates != UPDATES:
        warmup = max(1, min(WARMUP_UPDATES, updates // 10))
        validation_every = updates if updates < VALIDATION_EVERY else VALIDATION_EVERY
    else:
        warmup = WARMUP_UPDATES
        validation_every = VALIDATION_EVERY
    measured = {}
    for entry in arms:
        name, kind, extra = entry[0], entry[1], entry[2]
        steps_for_arm = entry[4] if len(entry) > 4 else TRAIN_REASONING_STEPS
        seed_everything(TRAIN_SEEDS[0])
        model = make_model(kind, _arm_config(kind, channels, extra))
        forward, forward_backward = count_flops(
            model, probe_batch, reasoning_steps=steps_for_arm,
            recursive=(kind != "native"))
        measured[name] = {"parameters": count_parameters(model),
                          "forward_flops": forward,
                          "forward_backward_flops": forward_backward,
                          "train_reasoning_steps": steps_for_arm}
        del model

    protocol = protocol_payload(phase, manifests_dir, identity, channels, measured)
    if protocol_file is not None:
        frozen = json.loads(Path(protocol_file).read_text(encoding="utf-8"))
        if frozen.get("protocol_sha256") != protocol.get("protocol_sha256"):
            raise RuntimeError(
                "protocol drift: the frozen protocol does not match the protocol this "
                "code would run; a pre-registered protocol may not be edited afterwards")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(output_dir / "protocol.json", protocol)

    results = {
        "format": "r7-65-ablation-result-v1",
        "phase": phase, "seeds": list(seeds),
        "scientific_claim": SCIENTIFIC_CLAIM, "test_read": False,
        "gpu_used": device.type == "cuda",
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "model_code_sha256": model_code_digest(),
        "protocol_sha256": protocol["protocol_sha256"],
        "flop_measurements": measured,
        "training": {}, "evaluation": {},
    }

    training_started = time.perf_counter()
    for seed in seeds:
        per_seed = {}
        for entry in arms:
            name, kind, extra, weight = entry[0], entry[1], entry[2], entry[3]
            steps_for_arm = entry[4] if len(entry) > 4 else TRAIN_REASONING_STEPS
            run_dir = output_dir / "training" / f"seed{seed}" / name
            config = _arm_config(kind, channels, extra)
            checkpoint, report = run_scheduled_updates(
                dataset, kind=kind, model_config=config, data_identity=identity,
                output_dir=run_dir, total_updates=updates, batch_size=BATCH_SIZE,
                steps=steps_for_arm, seed=seed, lr=LR, clip=CLIP,
                process_weight=weight, warmup_updates=warmup,
                minimum_lr_ratio=MINIMUM_LR_RATIO, validation_every=validation_every,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                minimum_improvement=MINIMUM_IMPROVEMENT,
                validation_lead_hours=VALIDATION_LEADS, device_name=device_name,
                validation_dataset=validation_dataset)
            saved_steps = report["contract"]["steps"]
            if saved_steps != steps_for_arm:
                raise RuntimeError(
                    f"{name} trained at steps={saved_steps} but declares "
                    f"{steps_for_arm}; the runner and the declared training depth "
                    "must not disagree (this is how a 'trained K=1' control can "
                    "silently be a second K=4 run)")
            by_epoch = {}
            for entry in report["losses"]:
                by_epoch.setdefault(entry["epoch"], []).append(entry["loss"])
            epoch_means = {str(epoch): sum(values) / len(values)
                           for epoch, values in sorted(by_epoch.items())}
            epoch_sizes = {str(epoch): len(values) for epoch, values in sorted(by_epoch.items())}
            full = max(epoch_sizes.values()) if epoch_sizes else 0
            full_epochs = [epoch for epoch, size in epoch_sizes.items() if size == full]
            first_full, last_full = ((min(full_epochs), max(full_epochs))
                                     if full_epochs else (None, None))
            per_seed[name] = {
                "auxiliary_weight": weight,
                "updates_run": report["updates_this_run"],
                "selected_update": report["selected_update"],
                "selected_validation_mse": report["selected_validation_mse"],
                "early_stopped": report["early_stopped"],
                "stopped_reason": report["stopped_reason"],
                "elapsed_seconds": report["elapsed_seconds"],
                "seconds_per_update": report["seconds_per_update"],
                "samples_seen": int(sum(entry["samples"] for entry in report["losses"])),
                "epoch_mean_loss": epoch_means,
                "epoch_update_counts": epoch_sizes,
                "first_full_epoch": first_full, "last_full_epoch": last_full,
                "loss_ratio_first_to_last_full_epoch": (
                    epoch_means[str(last_full)] / epoch_means[str(first_full)]
                    if full_epochs and epoch_means[str(first_full)] else None),
                "peak_allocated_bytes": report["peak_allocated_bytes"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "parameters": measured[name]["parameters"],
                "forward_flops": measured[name]["forward_flops"],
                "forward_backward_flops": measured[name]["forward_backward_flops"],
            }
            print(json.dumps({"phase": phase, "trained": name, "seed": seed,
                              "selected_update": report["selected_update"],
                              "seconds": round(report["elapsed_seconds"], 1),
                              "early_stopped": report["early_stopped"]}), flush=True)
        results["training"][str(seed)] = per_seed
    results["budget"] = {"training_seconds_total": time.perf_counter() - training_started}

    # ---- evaluation: val only, one run per (arm, seed, lead) -----------------
    # Each arm is evaluated at the depth it was trained at; evaluating a K=1-
    # trained model at K=4 would answer a different question than the one the
    # arm exists to answer, and the two depths are reported separately.
    for seed in seeds:
        for entry in arms:
            name = entry[0]
            steps_for_arm = entry[4] if len(entry) > 4 else TRAIN_REASONING_STEPS
            for lead in EVALUATION_LEADS:
                run_dir = output_dir / "evaluation" / f"seed{seed}" / name / f"lead_{lead:03d}h"
                selected = results["training"][str(seed)][name]["selected_update"]
                report = evaluate_local(
                    val_manifest, output_dir=run_dir,
                    checkpoint=(output_dir / "training" / f"seed{seed}" / name
                                / f"update_{selected:07d}.pt"),
                    lead_hours=(lead,), max_samples=64, device_name=device_name,
                    reasoning_steps=steps_for_arm)
                if report["split"] != "val":
                    raise ValueError(f"{name} was evaluated on split {report['split']}")
                results["evaluation"][f"seed{seed}/{name}@{lead}h"] = {
                    "seed": seed, "arm": name, "lead_hours": lead,
                    "split": report["split"], "n_evaluated": report["n_evaluated"],
                    "channels": list(report["channels"]), "units": list(report["units"]),
                    "elapsed_seconds": report["elapsed_seconds"],
                    "timing_scope": report["timing_scope"],
                    "trainable_parameters": report["trainable_parameters"],
                    "evaluation_reasoning_steps": steps_for_arm,
                    "rmse_csv": str(run_dir / "rmse.csv"),
                    "evaluation_dir": str(run_dir),
                }
        print(json.dumps({"phase": phase, "evaluated_seed": seed}), flush=True)

    # every arm must be scored on the identical case set within a lead
    for lead in EVALUATION_LEADS:
        for seed in seeds:
            counts = {entry[0]: results["evaluation"][f"seed{seed}/{entry[0]}@{lead}h"]["n_evaluated"]
                      for entry in arms}
            if len(set(counts.values())) != 1:
                raise RuntimeError(f"arms scored on different case counts at {lead}h "
                                   f"seed {seed}: {counts}")
            results.setdefault("case_counts_by_lead", {}).setdefault(
                f"{lead}h", {})[str(seed)] = counts[arms[0][0]]

    # ---- C3: test-time depth sweep from the K=4 checkpoint ------------------
    if phase == "c3":
        results["inference_depth_sweep"] = {}
        for seed in seeds:
            source = results["training"][str(seed)]["process8_aux010_k4"]
            for steps in INFERENCE_STEPS_C3:
                for lead in EVALUATION_LEADS:
                    run_dir = (output_dir / "evaluation" / f"seed{seed}"
                               / f"k_infer{steps}" / f"lead_{lead:03d}h")
                    report = evaluate_local(
                        val_manifest, output_dir=run_dir,
                        checkpoint=(output_dir / "training" / f"seed{seed}"
                                    / "process8_aux010_k4"
                                    / f"update_{source['selected_update']:07d}.pt"),
                        lead_hours=(lead,), max_samples=64, device_name=device_name,
                        reasoning_steps=steps)
                    results["inference_depth_sweep"][
                        f"seed{seed}/k{steps}@{lead}h"] = {
                        "seed": seed, "inference_steps": steps, "lead_hours": lead,
                        "n_evaluated": report["n_evaluated"],
                        "elapsed_seconds": report["elapsed_seconds"],
                        "timing_scope": report["timing_scope"],
                        "source_checkpoint": source["checkpoint"],
                        "rmse_csv": str(run_dir / "rmse.csv"),
                    }
            print(json.dumps({"phase": phase, "inference_sweep_seed": seed}), flush=True)

    _write_json_exclusive(output_dir / "ablation_result.json", results)
    _write_tables(results, output_dir, arms)
    return results


def _read_rmse(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_tables(results, output_dir, arms):
    """The four cost tables plus the accuracy table, all machine-readable."""
    parameter_rows, flop_rows, wall_rows, count_rows = [], [], [], []
    for seed in results["seeds"]:
        for entry in arms:
            name = entry[0]
            steps_for_arm = entry[4] if len(entry) > 4 else TRAIN_REASONING_STEPS
            trained = results["training"][str(seed)][name]
            parameter_rows.append({"seed": seed, "arm": name,
                                   "parameters": trained["parameters"],
                                   "nominal_parameters": trained["parameters"]})
            flop_rows.append({"seed": seed, "arm": name,
                              "forward_flops": trained["forward_flops"],
                              "forward_backward_flops": trained["forward_backward_flops"],
                              "train_reasoning_steps": steps_for_arm})
            wall_rows.append({"seed": seed, "arm": name,
                              "wall_time_seconds": trained["elapsed_seconds"],
                              "seconds_per_update": trained["seconds_per_update"],
                              "samples_seen": trained["samples_seen"],
                              "updates_run": trained["updates_run"]})
    for key, entry in results["evaluation"].items():
        count_rows.append({"key": key, "seed": entry["seed"], "arm": entry["arm"],
                           "lead_hours": entry["lead_hours"],
                           "n_evaluated": entry["n_evaluated"],
                           "timing_scope": entry["timing_scope"]})
    _write_csv(output_dir / "parameter_table.csv", parameter_rows,
               ["seed", "arm", "parameters", "nominal_parameters"])
    _write_csv(output_dir / "flops_table.csv", flop_rows,
               ["seed", "arm", "forward_flops", "forward_backward_flops",
                "train_reasoning_steps"])
    _write_csv(output_dir / "wall_time_table.csv", wall_rows,
               ["seed", "arm", "wall_time_seconds", "seconds_per_update",
                "samples_seen", "updates_run"])
    _write_csv(output_dir / "case_count_table.csv", count_rows,
               ["key", "seed", "arm", "lead_hours", "n_evaluated", "timing_scope"])

    rmse_rows = []
    for entry in results["evaluation"].values():
        for row in _read_rmse(entry["rmse_csv"]):
            rmse_rows.append({"seed": entry["seed"], "arm": entry["arm"],
                              "evaluation_lead_hours": entry["lead_hours"],
                              "variable": row.get("variable"), "unit": row.get("unit"),
                              "rmse": row.get("rmse")})
    _write_csv(output_dir / "rmse_table.csv", rmse_rows,
               ["seed", "arm", "evaluation_lead_hours", "variable", "unit", "rmse"])


def main():
    parser = argparse.ArgumentParser(description="#65 minimal ablation (C1/C2/C3).")
    parser.add_argument("--phase", required=True, choices=("c1", "c2", "c3"))
    parser.add_argument("--manifests", default=DEFAULT_MANIFESTS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--protocol-file", default=None,
                        help="a previously written protocol.json to freeze against")
    args = parser.parse_args()
    results = run_phase(args.manifests, args.out, phase=args.phase,
                        seeds=tuple(args.seeds), updates=args.updates,
                        device_name=args.device, protocol_file=args.protocol_file)
    print(json.dumps({"complete": True, "phase": args.phase,
                      "protocol_sha256": results["protocol_sha256"],
                      "training_seconds": results["budget"]["training_seconds_total"]}))


if __name__ == "__main__":
    main()
