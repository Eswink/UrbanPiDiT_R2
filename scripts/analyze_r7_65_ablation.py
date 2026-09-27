"""Score a #65 ablation run through the #60 comparator and read back the results.

The comparator is the only sanctioned comparison path (training/r7_coreasoning_compare),
so this script does not compute deltas itself: it feeds every (arm, seed, lead)
evaluation directory to ``summarize``, asks ``compare`` for seed-paired blocks
against a declared reference arm, and reads the four cost tables the ablation
harness already wrote. It reports every variable and every lead - dropping a
losing variable would be the failure mode the pre-registration exists to stop.

    python scripts/analyze_r7_65_ablation.py --run outputs/r7_65_c1 \
        --reference generic16_aux0 --out outputs/r7_65_c1_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPARATOR_DEPTH = 0


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Score a #65 ablation run (#60 comparator).")
    parser.add_argument("--run", required=True, help="ablation output directory")
    parser.add_argument("--reference", required=True,
                        help="arm the others are compared against")
    parser.add_argument("--out", required=True, help="new analysis directory")
    args = parser.parse_args()

    from training.r7_coreasoning_compare import compare, summarize

    run = Path(args.run)
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result = json.loads((run / "ablation_result.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    # The recorded digest must match the protocol body that is actually on disk,
    # or the analysis would describe a different frozen protocol than the run.
    from training.r7_experiment import canonical_digest

    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if canonical_digest(body) != protocol["protocol_sha256"]:
        raise RuntimeError("protocol.json does not match its own recorded digest")
    if protocol["protocol_sha256"] != result["protocol_sha256"]:
        raise RuntimeError(
            "the run result and the protocol file disagree on the protocol digest; "
            "a protocol may not be edited after the run")

    arms = sorted({entry["arm"] for entry in result["evaluation"].values()})
    if args.reference not in arms:
        raise ValueError(f"reference {args.reference!r} is not among {arms}")
    seeds = [int(seed) for seed in result["seeds"]]

    identity = {
        "dataset_identity": protocol["data"]["data_identity"],
        "model_code_sha256": result["model_code_sha256"],
        "declared_update_budget": int(protocol["shared_controls"]["max_updates"]),
        "evaluation_split": "val",
    }
    records = [{"arm": entry["arm"], "seed": entry["seed"], "depth": COMPARATOR_DEPTH,
                "evaluation_dir": entry["evaluation_dir"], "identity": identity}
               for entry in result["evaluation"].values()]
    table = summarize(records)
    output.mkdir(parents=True, exist_ok=False)

    # ``compare`` returns one block per non-baseline arm, so it is called once and
    # the rows are labelled from ``block["arm"]``. Calling it per arm and
    # relabelling would silently duplicate every block under the wrong name.
    blocks = [block for block in compare(table, baseline=args.reference,
                                         depth=COMPARATOR_DEPTH)
              if block["arm"] != args.reference]
    seen_arms = sorted({block["arm"] for block in blocks})
    if seen_arms != sorted(arm for arm in arms if arm != args.reference):
        raise RuntimeError(f"comparator returned arms {seen_arms}, expected "
                           f"{sorted(arm for arm in arms if arm != args.reference)}")

    # ---- per-cell outcome table: every variable, every lead, no filtering ----
    rows = []
    for block in blocks:
        for entry in block["seed_paired"]:
            rows.append({
                "arm": block["arm"], "reference": args.reference,
                "variable": entry["variable"], "unit": entry["unit"],
                "lead_hours": int(entry["lead_hours"]),
                "direction": entry["direction"],
                "sign_consistent": entry["sign_consistent"],
                "seed_deltas": ";".join(
                    f"{item['seed']}:{item['delta']}" for item in entry["seed_deltas"]),
            })
    _write_csv(output / "paired_cells.csv", rows,
               ["arm", "reference", "variable", "unit", "lead_hours", "direction",
                "sign_consistent", "seed_deltas"])

    counts = {}
    for arm in seen_arms:
        per_arm = {}
        for row in rows:
            if row["arm"] != arm:
                continue
            key = f"{row['lead_hours']}h"
            bucket = per_arm.setdefault(key, {"improved": 0, "worsened": 0, "unresolved": 0})
            bucket[row["direction"]] = bucket.get(row["direction"], 0) + 1
        counts[arm] = per_arm

    # ---- the four cost tables, copied into the analysis for one-place reading -
    cost_tables = {}
    for name in ("parameter_table", "flops_table", "wall_time_table", "case_count_table"):
        path = run / f"{name}.csv"
        if path.is_file():
            cost_tables[name] = _read_csv(path)
    cost_summary = _cost_summary(cost_tables)

    payload = {
        "format": "r7-65-ablation-analysis-v1",
        "scientific_claim": False,
        "run": str(run),
        "reference_arm": args.reference,
        "arms": arms,
        "seeds": seeds,
        "protocol_sha256": result["protocol_sha256"],
        "model_code_sha256": result["model_code_sha256"],
        "test_read": False,
        "paired_counts_by_lead": counts,
        "cost_summary": cost_summary,
        "train_curves": {
            str(seed): {
                arm: {
                    "epoch_mean_loss": entry["epoch_mean_loss"],
                    "epoch_update_counts": entry["epoch_update_counts"],
                    "first_full_epoch": entry["first_full_epoch"],
                    "last_full_epoch": entry["last_full_epoch"],
                    "loss_ratio_first_to_last_full_epoch":
                        entry["loss_ratio_first_to_last_full_epoch"],
                    "selected_update": entry["selected_update"],
                    "early_stopped": entry["early_stopped"],
                    "auxiliary_weight": entry["auxiliary_weight"],
                }
                for arm, entry in per_arm.items()
            }
            for seed, per_arm in result["training"].items()
        },
        "limitations": protocol["limitations"] + [
            "the paired counts are descriptive: agreement in sign across 3 seeds "
            "is not a significance test",
        ],
    }
    with (output / "analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps({"complete": True, "reference": args.reference,
                      "arms_compared": seen_arms,
                      "cells": len(rows)}, ensure_ascii=False))


def _cost_summary(cost_tables):
    """Mean parameter/FLOP/wall/case figures per arm, kept separate by seed."""
    summary = {}
    for row in cost_tables.get("parameter_table", []):
        summary.setdefault(row["arm"], {})["parameters"] = int(row["parameters"])
    for row in cost_tables.get("flops_table", []):
        summary.setdefault(row["arm"], {})["forward_flops"] = int(row["forward_flops"])
        summary.setdefault(row["arm"], {})["forward_backward_flops"] = int(
            row["forward_backward_flops"])
    wall = {}
    for row in cost_tables.get("wall_time_table", []):
        wall.setdefault(row["arm"], []).append(float(row["wall_time_seconds"]))
    for arm, values in wall.items():
        summary.setdefault(arm, {})["wall_time_seconds_mean"] = sum(values) / len(values)
        summary.setdefault(arm, {})["wall_time_seconds_by_seed"] = values
    cases = {}
    for row in cost_tables.get("case_count_table", []):
        cases.setdefault(row["arm"], set()).add(int(row["n_evaluated"]))
    for arm, values in cases.items():
        summary.setdefault(arm, {})["case_counts"] = sorted(values)
    return summary


if __name__ == "__main__":
    main()
