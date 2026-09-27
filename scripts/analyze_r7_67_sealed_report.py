"""#67 sealed test aggregation: per-seed paired block intervals and region strata.

Reads the once-written sealed test report and produces the two analyses the
protocol declares:

1. **day-block paired intervals**, computed per (arm pair, seed) so the
   seed dimension and the weather-sampling dimension stay separate -- the
   protocol forbids treating 3 seeds on one case set as a large independent N;
2. **full / interior / boundary stratification** per arm and seed, from the
   ``boundary_rmse.csv`` each evaluation already wrote.

Everything is derived from the sealed report's own files; no forecast is
regenerated and no model is re-run, so the report cannot drift from its analysis.

    python scripts/analyze_r7_67_sealed_report.py \
        --report outputs/r7_67_sealed_report --out outputs/r7_67_sealed_analysis
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

# Comparisons the protocol's "main table" implies. Zero-parameter controls are
# read from the sealed report's own climatology/persistence columns rather than
# being re-run.
ARM_PAIRS = (
    ("process", "generic"),        # the shared-structure pair
    ("generic", "native_window"),  # recursive vs the strongest non-recursive arm
    ("unet", "native_window"),     # within the non-recursive family
    ("afno_small", "native_window"),
)
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20260927


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description="#67 sealed-report aggregation.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args()

    from scripts.analyze_r7_67_paired_blocks import (
        paired_day_bootstrap, read_per_case, region_summary,
    )

    report_dir = Path(args.report)
    report = json.loads((report_dir / "sealed_test_report.json").read_text(encoding="utf-8"))
    if not report.get("test_read"):
        raise ValueError("this is not a sealed test report")
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)

    evaluation = report["evaluation"]
    seeds = sorted({entry["seed"] for entry in evaluation.values()})
    arms = sorted({entry["arm"] for entry in evaluation.values()})
    leads = sorted({entry["lead_hours"] for entry in evaluation.values()})

    def directory(arm, seed, lead):
        key = f"{arm}_s{seed}@{lead}h"
        if key not in evaluation:
            raise KeyError(key)
        return evaluation[key]["evaluation_dir"]

    # ---- day-block paired intervals, per seed and per lead -------------------
    blocks = {}
    for arm, reference in ARM_PAIRS:
        for lead in leads:
            for seed in seeds:
                arm_cases = read_per_case(directory(arm, seed, lead), "t2m", lead)
                ref_cases = read_per_case(directory(reference, seed, lead), "t2m", lead)
                result = paired_day_bootstrap(ref_cases, arm_cases,
                                              resamples=args.resamples, seed=DEFAULT_SEED)
                blocks[f"{arm}_vs_{reference}@{lead}h_s{seed}"] = {
                    "arm": arm, "reference": reference, "lead_hours": lead, "seed": seed,
                    **result,
                }

    # ---- full / interior / boundary stratification ---------------------------
    regions = {}
    for arm in arms:
        for seed in seeds:
            for lead in leads:
                summary = region_summary({arm: directory(arm, seed, lead)}, "t2m")
                entry = summary.get(arm)
                regions[f"{arm}_s{seed}@{lead}h"] = entry

    payload = {
        "format": "r7-67-sealed-analysis-v1",
        "scientific_claim": False,
        "report": str(report_dir),
        "data_identity": report["data_identity"],
        "test_read": True,
        "seeds": seeds,
        "arms": arms,
        "leads": leads,
        "paired_day_block": blocks,
        "region_stratification": regions,
        "case_counts_by_lead": report.get("case_counts_by_lead"),
        "notes": [
            "day blocks, not cases, are the resampling unit",
            "intervals are for estimation only and never select a model",
            "region stratification reuses the same forecasts; it is an error "
            "decomposition, not a boundary-forcing intervention",
            "seed variation and weather-sampling variation are reported separately",
        ],
    }
    with (output / "sealed_analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    summary = {
        "complete": True,
        "n_pairs": len(blocks),
        "n_region_cells": sum(1 for value in regions.values() if value),
        "excludes_zero_by_pair": {},
    }
    for key, value in blocks.items():
        pair = key.split("@")[0]
        bucket = summary["excludes_zero_by_pair"].setdefault(pair, {"yes": 0, "no": 0})
        bucket["yes" if value.get("interval_excludes_zero") else "no"] += 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
