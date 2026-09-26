"""B1: parameter-matched neural baselines vs persistence/climatology on real D1.

#64's B1 acceptance is a *controlled* comparison of the audited baseline families:
same data, same seed, same sample order, same normalization, same target variable,
with the parameter count, FLOPs, wall time and case count all recorded rather than
assumed. Four neural arms sit inside a +-5% parameter band at a nominal 2.8M
capacity (measured: 2,789,903-2,831,433, spread 1.49%); persistence and the
train-only climatology enter with **0 trainable parameters** and are never padded
to look comparable.

Discipline this script enforces rather than documents:

- test is **never read** - D1's test block is an engineering re-split inside the
  2016 segment, not the v2 2021 test candidate, so B1 selects on val only.
- the protocol is written **before** the first optimizer step, and its digest is
  echoed into the result file.
- every arm trains on the identical `train.jsonl` window order with one declared
  seed; no early stopping, no per-arm tuning, no seed shopping.
- FLOPs follow `training/r7_budget_audit.py`: `FlopCounterMode` under
  `torch.enable_grad()`, no parameter hooks (attention runs through
  `F.scaled_dot_product_attention`, which has no parameters, so a parameter hook
  would silently undercount every attention matmul).
- "same updates" is not claimed as "same compute": parameter, FLOPs, wall-time and
  case-count tables are all emitted, and the forward-only number is never allowed
  to stand in for the training cost.
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

SEED = 41
UPDATES = 200
BATCH_SIZE = 2
LR = 2e-4
CLIP = 1.0
REASONING_STEPS = 3
PROCESS_WEIGHT = 0.0  # no process head in B1; declared, not defaulted silently
# Per-lead evaluation, one run per (model, lead): the D1 val block is 72 h wide, so a
# joint 6..48 h request would keep only the 3 windows that satisfy every horizon at
# once. Reporting per lead keeps 10/9/7/3 usable cases instead of 3, and the (variable,
# lead) pairing is how the audit requires these numbers to be read.
LEADS = (6, 12, 24, 48)
NOMINAL_PARAMETERS = 2_800_000
PARAMETER_BAND = 0.05
# (name, runner kind, architecture config) - candidates fixed by docs/R7_B1_BASELINE_AUDIT.md
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
)
# parameter-free controls: recorded at 0 parameters, never brought into the band
BASELINES = ("persistence", "climatology")
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
    """Measured, not a paper formula."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_flops(model, batch, *, reasoning_steps, recursive):
    """Forward and forward+backward FLOPs under the repository's FLOP convention."""
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


def run_study(manifests_dir, output_dir, *, updates=UPDATES, device_name="cuda"):
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from training.r7_evaluate import evaluate_local, fit_training_climatology
    from training.r7_experiment import (canonical_digest, dataset_identity, load_checkpoint,
                                        make_model, model_code_digest, seed_everything,
                                        select_device)
    from training.r7_local_runner import run_local_updates

    manifests_dir, output_dir = Path(manifests_dir), Path(output_dir)
    train_manifest = manifests_dir / "train.jsonl"
    val_manifest = manifests_dir / "val.jsonl"
    for manifest in (train_manifest, val_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    if (manifests_dir / "test.jsonl").is_file():
        pass  # present but deliberately never opened by this script
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")

    dataset = ZarrAtmosWindowDataset(train_manifest)
    identity, _ = dataset_identity(train_manifest)
    channels = int(dataset[0]["coarse_history"].shape[1])
    device = select_device(device_name)
    # one fixed batch, reused by every arm so FLOPs and wall time share a shape
    from torch.utils.data import default_collate
    probe_batch = default_collate([dataset[0], dataset[1]])

    measured = {}
    for name, kind, config in ARMS:
        seed_everything(SEED)
        model = make_model(kind, _arm_config(kind, channels, config))
        parameters = count_parameters(model)
        forward, forward_backward = count_flops(
            model, probe_batch, reasoning_steps=REASONING_STEPS, recursive=(kind != "native"))
        measured[name] = {"parameters": parameters, "forward_flops": forward,
                          "forward_backward_flops": forward_backward}
        del model

    protocol = {
        "format": "r7-b1-baseline-comparison-v1",
        "frozen_before_any_step": True,
        "objective": ("four parameter-matched neural families and two parameter-free "
                      "baselines scored on the same real held-out D1 windows; a usable "
                      "baseline comparison, never a SOTA or convergence claim"),
        "issue": "#64 B1",
        "data": {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "test_manifest": str(manifests_dir / "test.jsonl"),
            "test_read": False,
            "test_policy": ("sealed: D1's test block is an engineering re-split inside "
                            "the 2016 segment, not the v2 2021 test candidate"),
            "data_identity": str(identity),
            "store": "outputs/r7_d1_earthmover/store/cache.zarr",
            "split_mode": "time_ranges (docs/decisions/0005-*.md, 0006-*.md)",
            "normalization": "store train-only centered mean/std, applied at read time",
        },
        "arms": [{"name": name, "kind": kind,
                  "model_config": _arm_config(kind, channels, config),
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in ARMS],
        "baselines": [{"name": name, "trainable_parameters": 0,
                       "note": "parameter-free control; deliberately outside the +-5% band"}
                      for name in BASELINES],
        "parameter_gate": {
            "nominal": NOMINAL_PARAMETERS,
            "band": PARAMETER_BAND,
            "applies_to": "the four neural arms only; non-neural baselines are not padded",
            "measured_spread": (max(v["parameters"] for v in measured.values())
                                - min(v["parameters"] for v in measured.values()))
                               / NOMINAL_PARAMETERS,
        },
        "flop_convention": FLOP_CONVENTION,
        "shared_controls": {
            "seed": SEED,
            "optimizer": "AdamW lr 2e-4 weight_decay 1e-4 (runner default, declared)",
            "clip": CLIP,
            "batch_size": BATCH_SIZE,
            "updates": updates,
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
            "loss": "latitude-weighted MSE (streamed truncated BPTT for generic)",
            "reasoning_steps": REASONING_STEPS,
            "process_weight": PROCESS_WEIGHT,
            "target_variable": "the store's 17 declared channels, unmodified",
            "leads_hours": list(LEADS),
            "evaluation_policy": ("one evaluation run per (model, lead): the D1 val block "
                                  "is 72 h wide, so a joint 6..48 h request would keep only "
                                  "the 3 windows satisfying every horizon at once; per-lead "
                                  "runs keep 10/9/7/3. Cells are compared only within a "
                                  "lead - never ranked across leads, because a longer lead "
                                  "uses a different, smaller case set"),
            "selection_split": "val only; test is sealed and never read",
        },
        "compute_alignment": {
            "claimed_aligned": ["data", "seed", "sample order", "normalization",
                                "target variable", "optimizer updates", "batch size"],
            "explicitly_not_aligned": ["forward/backward FLOPs", "wall time",
                                       "peak memory", "parameter count (bounded, not equal)"],
            "measurement": ("parameter, FLOPs, wall-time and case-count tables are all "
                            "reported; forward-only FLOPs never stand in for training cost"),
        },
        "selection_rule": ("fixed 200-update endpoint per arm; no early stopping, no "
                           "per-arm hyperparameter search, no test-driven selection"),
        "scientific_claim": False,
        "limitations": [
            "D1 is one 30-day engineering segment (January 2016) of real ERA5",
            "single pre-declared seed; no multi-seed significance claim",
            "climatology has only 4 (month,hour) buckets, so no seasonal skill is testable",
            "not a converged benchmark; bounded CPU/GPU checkpoints",
            "4 of 17 channels are scored so the table stays readable (see scored_channels)",
        ],
    }
    protocol["protocol_sha256"] = canonical_digest(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"})
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)

    digest = model_code_digest()
    results = {
        "format": "r7-b1-baseline-result-v1",
        "scientific_claim": False,
        "test_read": False,
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "model_code_sha256": digest,
        "protocol": protocol,
        "training": {},
        "evaluation": {},
        "flop_measurements": measured,
        "budget": {},
    }

    # ---- train the four neural arms on the identical window order -----------------
    training_started = time.perf_counter()
    checkpoints = {}
    for name, kind, config in ARMS:
        checkpoint, report = run_local_updates(
            dataset, kind=kind, model_config=_arm_config(kind, channels, config),
            data_identity=identity, output_dir=output_dir / "training" / name,
            total_updates=updates, batch_size=BATCH_SIZE, steps=REASONING_STEPS,
            seed=SEED, lr=LR, clip=CLIP, process_weight=PROCESS_WEIGHT,
            device_name=device_name)
        saved = load_checkpoint(checkpoint)
        if saved["updates"] != updates:
            raise ValueError(f"{name} did not reach the fixed update endpoint")
        checkpoints[name] = checkpoint
        samples_seen = int(sum(entry["samples"] for entry in report["losses"]))
        results["training"][name] = {
            "updates": report["updates_this_run"],
            "samples_seen": samples_seen,
            "elapsed_seconds": report["elapsed_seconds"],
            "seconds_per_update": report["elapsed_seconds"] / report["updates_this_run"],
            "wall_time_includes_io": True,
            "wall_time_note": report["note"],
            "first_loss": report["losses"][0]["loss"],
            "final_loss": report["losses"][-1]["loss"],
            "loss_ratio": (report["losses"][-1]["loss"] / report["losses"][0]["loss"]
                           if report["losses"][0]["loss"] else None),
            "peak_allocated_bytes": report["peak_allocated_bytes"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "parameters": measured[name]["parameters"],
        }
        print(json.dumps({"trained": name, "loss_ratio": results["training"][name]["loss_ratio"],
                          "seconds": report["elapsed_seconds"], "samples": samples_seen}),
              flush=True)
    results["budget"]["training_seconds_total"] = time.perf_counter() - training_started

    # ---- evaluate every arm and both parameter-free controls on the same val cases --
    # One run per (model, lead): a joint 6..48 h request keeps only the 3 windows that
    # satisfy every horizon at once, while per-lead runs keep 10/9/7/3. Cells are only
    # ever compared within a lead, because a longer lead necessarily uses a smaller,
    # different case set (the audit forbids cross-lead ranking for that reason).
    for name in list(checkpoints) + list(BASELINES):
        for lead in LEADS:
            run_dir = output_dir / "evaluation" / name / f"lead_{lead:03d}h"
            report = evaluate_local(
                val_manifest, output_dir=run_dir,
                checkpoint=checkpoints[name] if name in checkpoints else None,
                baseline=None if name in checkpoints else name,
                lead_hours=(lead,), max_samples=64, device_name=device_name)
            if report["split"] != "val":
                raise ValueError(f"{name} was evaluated on split {report['split']}, not val")
            results["evaluation"][f"{name}@{lead}h"] = {
                "model": name, "lead_hours": lead,
                "split": report["split"],
                "n_evaluated": report["n_evaluated"],
                "n_available_windows": report["n_available_windows"],
                "channels": list(report["channels"]),
                "units": list(report["units"]),
                "elapsed_seconds": report["elapsed_seconds"],
                "timing_scope": report["timing_scope"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "parameter_free_baseline": report["parameter_free_baseline"],
                "trainable_parameters": report["trainable_parameters"],
                "climatology": {key: value for key, value in report["climatology"].items()
                                if key != "bucket_counts"},
                "acc_skill_identity": report["acc_skill_identity"],
                "rmse_csv": str(run_dir / "rmse.csv"),
                "acc_csv": str(run_dir / "acc.csv"),
                "skill_csv": str(run_dir / "climatology_skill.csv"),
            }
        print(json.dumps({"evaluated": name, "leads": list(LEADS)}), flush=True)

    # every model must have been scored on the identical case set at each lead
    models = list(checkpoints) + list(BASELINES)
    for lead in LEADS:
        counts = {name: results["evaluation"][f"{name}@{lead}h"]["n_evaluated"]
                  for name in models}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"models scored on different case counts at {lead}h: {counts}")
        results.setdefault("case_counts_by_lead", {})[f"{lead}h"] = counts[str(models[0])]
    results["climatology_selection"] = results["evaluation"]["persistence@6h"]["climatology"]

    with (output_dir / "b1_result.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return results


def write_tables(results, output_dir):
    """The four required tables: parameters, FLOPs, wall time, cases - plus RMSE."""
    output_dir = Path(output_dir)
    tables = {}
    models = list(results["training"]) + [name for name in BASELINES]

    parameters_path = output_dir / "b1_parameter_table.csv"
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

    flops_path = output_dir / "b1_flops_table.csv"
    with flops_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "forward_flops", "forward_backward_flops",
                         "batch", "reasoning_steps", "convention"])
        for entry in results["protocol"]["arms"]:
            writer.writerow([entry["name"], entry["forward_flops"],
                             entry["forward_backward_flops"], "2",
                             REASONING_STEPS if entry["kind"] == "generic" else "n/a",
                             FLOP_CONVENTION])
        for name in BASELINES:
            writer.writerow([name, "0", "0", "2", "n/a",
                             "parameter-free control: no learnable compute"])
    tables["flops"] = flops_path.name

    wall_path = output_dir / "b1_wall_time_table.csv"
    with wall_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "updates", "samples_seen", "train_seconds",
                         "seconds_per_update", "wall_time_includes_io",
                         "eval_seconds_total", "eval_timing_scope"])
        for name in models:
            train = results["training"].get(name)
            eval_seconds = sum(entry["elapsed_seconds"] for key, entry in results["evaluation"].items()
                               if entry["model"] == name)
            scope = next(entry["timing_scope"] for key, entry in results["evaluation"].items()
                         if entry["model"] == name)
            writer.writerow([
                name,
                train["updates"] if train else 0,
                train["samples_seen"] if train else 0,
                f"{train['elapsed_seconds']:.3f}" if train else "0.000",
                f"{train['seconds_per_update']:.6f}" if train else "0.000000",
                "True" if train else "n/a (no training)",
                f"{eval_seconds:.3f}", scope])
    tables["wall_time"] = wall_path.name

    cases_path = output_dir / "b1_case_count_table.csv"
    with cases_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "split", "lead_hours", "n_available_windows",
                         "n_evaluated", "test_read"])
        for key in sorted(results["evaluation"], key=lambda k: (results["evaluation"][k]["model"],
                                                               results["evaluation"][k]["lead_hours"])):
            entry = results["evaluation"][key]
            writer.writerow([entry["model"], entry["split"], entry["lead_hours"],
                             entry["n_available_windows"], entry["n_evaluated"], "False"])
    tables["cases"] = cases_path.name

    # RMSE + climatology skill, joined per (model, variable, lead)
    rmse_path = output_dir / "b1_rmse_table.csv"
    with rmse_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "variable", "unit", "lead_hours", "rmse",
                         "rmse_climatology", "mse_skill", "n_initializations"])
        for key in sorted(results["evaluation"], key=lambda k: (results["evaluation"][k]["model"],
                                                               results["evaluation"][k]["lead_hours"])):
            entry = results["evaluation"][key]
            rmse = {row["variable"]: row for row in _csv_rows(entry["rmse_csv"])}
            skill = {row["variable"]: row for row in _csv_rows(entry["skill_csv"])}
            for variable in entry["channels"]:
                row = rmse.get(variable)
                if row is None:
                    continue
                skill_row = skill.get(variable, {})
                writer.writerow([entry["model"], variable, row["unit"], entry["lead_hours"],
                                 row["rmse"],
                                 skill_row.get("rmse_climatology", ""),
                                 skill_row.get("mse_skill", "") or "undefined",
                                 row["n_initializations"]])
    tables["rmse"] = rmse_path.name
    return tables


def main():
    parser = argparse.ArgumentParser(
        description="B1 controlled baseline comparison on the real D1 store (#64).")
    parser.add_argument("--manifests", required=True, help="D1 store manifest directory")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    args = parser.parse_args()
    results = run_study(args.manifests, args.out, updates=args.updates, device_name=args.device)
    tables = write_tables(results, args.out)
    print(json.dumps({
        "arms": {entry["name"]: entry["parameters"] for entry in results["protocol"]["arms"]},
        "parameter_spread": results["protocol"]["parameter_gate"]["measured_spread"],
        "case_counts_by_lead": results["case_counts_by_lead"],
        "training_seconds_total": results["budget"]["training_seconds_total"],
        "tables": tables,
        "scientific_claim": False,
    }, indent=1))


if __name__ == "__main__":
    main()
