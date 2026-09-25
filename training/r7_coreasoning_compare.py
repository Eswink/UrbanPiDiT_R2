"""Fair-budget co-reasoning comparison for #6, on real regional ERA5.

Issue #6's research gate is that process-aware co-reasoning must beat the generic
recursive baseline **under a comparable budget**, and its fairness contract
requires parameter counts and measured cost alongside accuracy:

1. `generic` recursion — #5's baseline; K=0 removes recursion entirely,
2. `process_no_feedback` — process tokens without forecast feedback,
3. `process_feedback` — process + forecast co-reasoning.

The adaptive arm is evaluated separately (`docs/R7_ADAPTIVE_HALTING.md`), not here.

#5 established that these arms differ by +0.030 % parameters and
+0.0001–0.0003 % forward FLOPs, so a win cannot come from being a larger network.
This module therefore trains every arm at an identical seed with an identical
optimizer budget on identical data, evaluates the fixed depth ablation, and
reports **every variable with its unit** plus the per-variable win/loss count.

An aggregate score is deliberately not produced: averaging RMSE across K, Pa and
m/s would be meaningless. Every seed is retained and a mixed outcome is reported
as mixed. No test split is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (41, 42, 43)
# (arm name, model kind, construction options)
ARMS = (
    ("generic", "generic", {"latent_tokens": 16}),
    ("process_no_feedback", "process",
     {"anchored_processes": 8, "free_processes": 8, "use_forecast_feedback": False}),
    ("process_feedback", "process",
     {"anchored_processes": 8, "free_processes": 8, "use_forecast_feedback": True}),
)
ABLATION_DEPTHS = (0, 1, 3)
CONFIG_BASE = {
    "in_channels": 11, "out_channels": 11, "history_steps": 2,
    "dim": 32, "depth": 2, "heads": 4, "window_size": 4, "patch_size": 2,
    "dropout": 0.0, "default_reasoning_steps": 3,
}
DEFAULT_UPDATES = 200
SPLIT_YEARS = {"train": [2018], "val": [2019], "test": [2020]}


def arm_config(kind, options, channels):
    config = dict(CONFIG_BASE)
    config["in_channels"] = channels
    config["out_channels"] = channels
    config.update(options)
    return config


def protocol_payload(source_sha256, channels, updates, depths, seeds):
    """Frozen before any optimizer step, including every fixed choice."""
    from training.r7_experiment import canonical_digest

    body = {
        "format": "r7-coreasoning-fair-budget-v1",
        "frozen_before_any_step": True,
        "source_sha256": source_sha256,
        "channels": channels,
        "arms": [
            {"name": name, "kind": kind, "model_config": arm_config(kind, options, channels)}
            for name, kind, options in ARMS
        ],
        "seeds": list(seeds),
        "optimizer_updates": updates,
        "batch_size": 2, "lr": 2e-4, "clip": 1.0, "accumulation": 1,
        "training_reasoning_steps": 3, "process_weight": 0.1,
        "ablation_depths": list(depths),
        "evaluation_split": "val", "test_evaluated": False,
        "budget_parity": (
            "arms differ by +0.030% parameters and +0.0001-0.0003% forward FLOPs; "
            "see docs/R7_BUDGET_PARITY.md"
        ),
        "all_seeds_retained": True,
        "limitations": [
            "one small tile over ten January days per year; not full seasons",
            "fixed optimizer endpoint, not extension until a metric improves",
            "validation cases were inspected during earlier iterations",
            "three seeds: no significance test, no convergence or SOTA claim",
            "K=0 is the no-recursion ablation on the same architecture",
        ],
        "scientific_claim": False,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def read_rmse_rows(evaluation_dir):
    """Per-variable RMSE from one evaluation, with its unit and initialization count."""
    path = Path(evaluation_dir) / "rmse.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append({
                "lead_hours": float(record["lead_hours"]),
                "variable": record["variable"],
                "unit": record["unit"],
                "rmse": float(record["rmse"]),
                "n_initializations": int(record["n_initializations"]),
            })
    if not rows:
        raise ValueError(f"empty RMSE table: {path}")
    return rows


def summarize(records):
    """Per-arm, per-depth, per-variable mean and seed spread.

    Every variable and depth is kept: reporting only a favoured variable would
    hide the mixed outcome these arms actually produce.
    """
    grouped = {}
    for record in records:
        for row in read_rmse_rows(record["evaluation_dir"]):
            key = (record["arm"], record["depth"], row["variable"], row["unit"],
                   row["lead_hours"])
            grouped.setdefault(key, []).append(row["rmse"])
    table = []
    for (arm, depth, variable, unit, lead), values in sorted(grouped.items()):
        table.append({
            "arm": arm, "depth": depth, "variable": variable, "unit": unit,
            "lead_hours": lead, "seeds": len(values),
            "rmse_mean": statistics.fmean(values),
            "rmse_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values,
        })
    return table


def compare(table, *, baseline="generic", depth=None):
    """Compare at `depth`, defaulting to the deepest measured ablation."""
    """Per-variable paired comparison of each arm against the baseline.

    A negative `delta` means the arm has the lower RMSE for that variable. The
    win/loss count is reported instead of an aggregate, because a single number
    would average across physical units.
    """
    depths = sorted({row["depth"] for row in table})
    if not depths:
        raise ValueError("no rows to compare")
    if depth is None:
        depth = max(depths)
    if depth not in depths:
        raise ValueError(f"depth {depth} was not measured; available={depths}")
    by_key = {(row["arm"], row["depth"], row["variable"], row["lead_hours"]): row
              for row in table}
    results = []
    for arm in sorted({row["arm"] for row in table} - {baseline}):
        deltas, wins, losses = [], 0, 0
        for (name, row_depth, variable, lead), row in sorted(by_key.items()):
            if name != arm or row_depth != depth:
                continue
            reference = by_key.get((baseline, depth, variable, lead))
            if reference is None:
                continue
            delta = row["rmse_mean"] - reference["rmse_mean"]
            deltas.append({"variable": variable, "unit": row["unit"],
                           "lead_hours": lead, "delta": delta})
            if delta < 0:
                wins += 1
            elif delta > 0:
                losses += 1
        # A win/loss split is only meaningful if the per-seed deltas agree on
        # their sign. Seed-paired deltas that flip sign mean the effect is below
        # seed noise, so those variables must not be counted as established.
        paired = []
        for (name, row_depth, variable, lead), row in sorted(by_key.items()):
            if name != arm or row_depth != depth:
                continue
            reference = by_key.get((baseline, depth, variable, lead))
            if reference is None or row["seeds"] < 2:
                continue
            diffs = [a - b for a, b in zip(row["values"], reference["values"])]
            if not diffs:
                continue
            paired.append({
                "variable": variable, "unit": row["unit"], "lead_hours": lead,
                "seed_deltas": diffs,
                "sign_consistent": all(d < 0 for d in diffs) or all(d > 0 for d in diffs),
                "direction": "improved" if all(d < 0 for d in diffs) else
                             "worsened" if all(d > 0 for d in diffs) else "unresolved",
            })
        established = [row for row in paired if row["sign_consistent"]]
        results.append({
            "arm": arm, "baseline": baseline, "depth": depth,
            "variables_improved": wins, "variables_worsened": losses,
            "deltas": deltas, "seed_paired": paired,
            "variables_sign_consistent": len(established),
            "established_improved": sum(1 for r in established if r["direction"] == "improved"),
            "established_worsened": sum(1 for r in established if r["direction"] == "worsened"),
            "beats_baseline_everywhere": losses == 0 and wins > 0,
            "gate_met": bool(established) and all(
                r["direction"] == "improved" for r in established),
        })
    return results


def run_comparison(source, receipt, output_dir, *, updates=DEFAULT_UPDATES,
                   max_samples=8, seeds=SEEDS, depths=ABLATION_DEPTHS):
    """Publish a fresh cache, train every arm at every seed, then evaluate."""
    import torch

    from data.preprocess.r7_preflight import prepare_local
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import (
        dataset_identity, load_checkpoint, make_model, model_code_digest)
    from training.r7_local_runner import run_local_updates

    source, receipt = Path(source), Path(receipt)
    out = Path(output_dir)
    # Validate the receipt before creating anything, so a rejected run leaves no
    # half-built output directory behind.
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("status") != "downloaded-real-source":
        raise ValueError("a real acquisition receipt is required; no synthetic fallback")
    if payload.get("synthetic_fallback") is not False:
        raise ValueError("receipt must declare synthetic_fallback=False")
    source_sha256 = payload["local_artifact"]["sha256"]
    if not source.is_file():
        raise FileNotFoundError(source)
    # Re-derive the bytes rather than trusting the receipt's own field: a receipt
    # is caller-editable, so this is an integrity check on the input, not proof
    # of provenance (the acquisition receipt carries that separately).
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("source file does not match the receipt's recorded SHA256")
    out.mkdir(parents=True, exist_ok=False)

    prep_config = {
        "split_years": dict(SPLIT_YEARS),
        "history_steps": 2, "history_interval_hours": 6, "lead_time_hours": 6,
        "sample_stride_hours": 6, "time_chunk": 16,
        "compute_process_targets": True,
    }
    dataset_dir = out / "dataset"
    report = prepare_local(source, prep_config, write=True,
                           store_path=dataset_dir / "cache.zarr",
                           manifest_dir=dataset_dir / "manifests", max_raw_gib=0.5)
    train_manifest = dataset_dir / "manifests" / "train.jsonl"
    val_manifest = dataset_dir / "manifests" / "val.jsonl"
    channels = int(report["shape"][1])

    protocol = protocol_payload(source_sha256, channels, updates, depths, seeds)
    with (out / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)

    identity, dataset = dataset_identity(train_manifest)
    digest = model_code_digest()
    torch.set_num_threads(2)
    records, resources = [], []
    for seed in seeds:
        for name, kind, options in ARMS:
            model_config = arm_config(kind, options, channels)
            checkpoint, training = run_local_updates(
                dataset, kind=kind, model_config=model_config, data_identity=identity,
                output_dir=out / "training" / f"{name}_{seed}", total_updates=updates,
                batch_size=2, steps=3, seed=seed, lr=2e-4,
                process_weight=0.1, device_name="cpu")
            saved = load_checkpoint(checkpoint)
            if saved["updates"] != updates:
                raise ValueError("fixed optimizer endpoint not reached")
            resources.append({
                "arm": name, "seed": seed,
                "params": sum(p.numel() for p in make_model(kind, model_config).parameters()),
                "training": training,
            })
            for depth in depths:
                evaluation_dir = out / "evaluation" / f"{name}_{seed}_K{depth}"
                evaluate_local(val_manifest, output_dir=evaluation_dir,
                               checkpoint=checkpoint, lead_hours=(6,),
                               max_samples=max_samples, reasoning_steps=depth,
                               device_name="cpu")
                records.append({"arm": name, "seed": seed, "depth": depth,
                                "evaluation_dir": str(evaluation_dir)})
        print(json.dumps({"seed": seed, "arms_complete": len(ARMS)}), flush=True)

    if model_code_digest() != digest:
        raise RuntimeError("model code changed during the comparison")
    table = summarize(records)
    result = {
        "format": "r7-coreasoning-fair-budget-result-v1",
        "complete": len(records) == len(seeds) * len(ARMS) * len(depths),
        "scientific_claim": False, "gpu_used": False, "test_evaluated": False,
        "source_cloud_requests": 0,
        "protocol": protocol, "protocol_sha256": protocol["protocol_sha256"],
        "preflight": report, "model_code_sha256": digest,
        "resources": resources, "records": records,
        "summary": table, "comparison": compare(table),
        "limitations": protocol["limitations"],
    }
    with (out / "coreasoning_result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fair-budget co-reasoning ablation comparison (#6).")
    parser.add_argument("--source", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    result = run_comparison(args.source, args.receipt, args.out,
                            updates=args.updates, max_samples=args.max_samples)
    print(json.dumps({
        "complete": result["complete"],
        "protocol_sha256": result["protocol_sha256"],
        "comparison": [{k: v for k, v in block.items() if k != "deltas"}
                       for block in result["comparison"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
