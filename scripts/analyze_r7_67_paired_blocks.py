"""#67 declared analyses: day-block paired resampling and region stratification.

Two things the #67 protocol declares and that must actually run on results
rather than remain a plan:

1. **Day-block paired resampling.** Forecast initializations on the same UTC day
   are not independent samples, so an interval built from the number of cases
   would be too narrow. Cases are resampled in **day blocks** (all initializations
   of one UTC day move together) and the paired difference between two arms is
   recomputed per resample. This gives an interval for the *arm difference* that
   respects the dependence structure. It is used for interval estimation only -
   never to choose a model.

2. **full / interior / boundary stratification.** Errors are stratified by
   distance to the regional edge using the same forecasts, so this is an error
   decomposition, not a boundary-forcing experiment. The distinction matters and
   is restated in the output.

Both read the per-case MSE that ``evaluate_local`` already writes into
``provenance.json``, so no forecast is regenerated and no model is re-run.

    python scripts/analyze_r7_67_paired_blocks.py \
        --arm <name>=<evaluation_dir> --arm <name>=<evaluation_dir> \
        --reference <name> --out outputs/r7_67_paired_blocks
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20260927
# Margins in grid cells. The tile is 12x12 native cells in the D1/B2 family, so a
# 1-cell ring is 8.3% of the domain per side; 2 cells is reported as the stricter
# variant rather than as the headline.
DEFAULT_MARGINS = (1, 2)


def read_per_case(evaluation_dir, channel="t2m", lead_filter=None):
    """Per-case MSE for one variable, keyed by init_time, from provenance.json.

    Returns ``{init_time: mse}``. The provenance file records ``mse`` as a
    ``[n_leads][n_channels]`` grid per initialization.
    """
    path = Path(evaluation_dir) / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = list(payload["channels"])
    if channel not in channels:
        raise ValueError(f"channel {channel!r} absent from {path}")
    index = channels.index(channel)
    leads = list(payload["lead_hours"])
    if lead_filter is not None:
        if lead_filter not in leads:
            raise ValueError(f"lead {lead_filter} absent from {path}")
        lead_index = leads.index(lead_filter)
    else:
        lead_index = 0
    out = {}
    for entry in payload["initializations"]:
        out[entry["init_time"]] = float(entry["mse"][lead_index][index])
    if not out:
        raise ValueError(f"no initializations in {path}")
    return out


def day_of(init_time):
    return str(init_time)[:10]


def paired_day_bootstrap(reference_cases, arm_cases, *, resamples=DEFAULT_RESAMPLES,
                         seed=DEFAULT_SEED):
    """Paired difference (arm - reference) with a day-block bootstrap interval.

    A negative difference means the arm has the lower MSE. Days, not cases, are
    the resampling unit. Returns None-valued fields when the two arms were not
    scored on identical cases, because a paired statistic is then undefined.
    """
    shared = sorted(set(reference_cases) & set(arm_cases))
    if len(shared) != len(reference_cases) or len(shared) != len(arm_cases):
        return {
            "paired": False,
            "n_reference": len(reference_cases),
            "n_arm": len(arm_cases),
            "n_shared": len(shared),
            "reason": "arms were not scored on identical cases; a paired test is undefined",
        }
    if not shared:
        raise ValueError("no shared cases")
    days = {}
    for init_time in shared:
        days.setdefault(day_of(init_time), []).append(init_time)
    day_keys = sorted(days)
    if len(day_keys) < 2:
        return {
            "paired": True, "n_cases": len(shared), "n_days": len(day_keys),
            "difference": float(np.mean([arm_cases[t] - reference_cases[t] for t in shared])),
            "interval_low": None, "interval_high": None,
            "reason": ("fewer than two day blocks; a block interval cannot be formed and "
                       "a case-level interval would overstate precision"),
        }
    deltas = np.array([arm_cases[t] - reference_cases[t] for t in shared], dtype=np.float64)
    observed = float(deltas.mean())
    generator = np.random.default_rng(seed)
    per_day = {day: np.array([arm_cases[t] - reference_cases[t] for t in times],
                             dtype=np.float64)
               for day, times in days.items()}
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = generator.choice(len(day_keys), size=len(day_keys), replace=True)
        pooled = np.concatenate([per_day[day_keys[position]] for position in picked])
        draws[index] = pooled.mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "paired": True,
        "n_cases": len(shared),
        "n_days": len(day_keys),
        "difference": observed,
        "interval_low": float(low),
        "interval_high": float(high),
        "interval_excludes_zero": bool(low > 0 or high < 0),
        "resamples": int(resamples),
        "resample_seed": int(seed),
        "unit": "day blocks of UTC initializations (cases within a day move together)",
    }


def _read_boundary_csv(evaluation_dir):
    path = Path(evaluation_dir) / "boundary_rmse.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def region_summary(evaluation_dirs, channel="t2m"):
    """Per-arm RMSE by region from the boundary stratification CSVs."""
    per_arm = {}
    for arm, directory in evaluation_dirs.items():
        rows = _read_boundary_csv(directory)
        if rows is None:
            per_arm[arm] = None
            continue
        lookups = {}
        for row in rows:
            if row.get("variable") != channel:
                continue
            lookups[row["region"]] = row
        if not lookups:
            per_arm[arm] = {"status": "no boundary rows for this variable"}
            continue
        regions = {}
        for region, row in lookups.items():
            regions[region] = {
                "rmse": float(row["rmse"]),
                "n_cells": int(float(row.get("n_cells", 0) or 0)),
            }
        per_arm[arm] = {"regions": regions}
    return per_arm


def main():
    parser = argparse.ArgumentParser(description="#67 day-block and region analyses.")
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=EVALUATION_DIR (repeatable)")
    parser.add_argument("--reference", required=True, help="arm name used as the reference")
    parser.add_argument("--channel", default="t2m")
    parser.add_argument("--lead", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    arms = {}
    for item in args.arm:
        if "=" not in item:
            raise ValueError(f"--arm expects NAME=DIR, got {item!r}")
        name, directory = item.split("=", 1)
        arms[name] = directory
    if args.reference not in arms:
        raise ValueError(f"reference {args.reference!r} is not among {sorted(arms)}")

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)

    cases = {name: read_per_case(directory, args.channel, args.lead)
             for name, directory in arms.items()}
    reference = cases[args.reference]

    pairs = {}
    for name in sorted(arms):
        if name == args.reference:
            continue
        pairs[f"{name}_minus_{args.reference}"] = paired_day_bootstrap(
            reference, cases[name], resamples=args.resamples, seed=args.seed)

    payload = {
        "format": "r7-67-paired-block-analysis-v1",
        "scientific_claim": False,
        "channel": args.channel,
        "lead_hours": args.lead,
        "reference_arm": args.reference,
        "arms": sorted(arms),
        "paired_day_block": pairs,
        "region_stratification": region_summary(arms, args.channel),
        "notes": [
            "day blocks, not cases, are the resampling unit because same-day "
            "initializations are not independent samples",
            "the interval is used for estimation only and never to select a model",
            "region stratification uses the same forecasts; it is an error "
            "decomposition, not a boundary-forcing intervention",
        ],
    }
    with (output / "paired_block_analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    summary = {
        "complete": True,
        "reference": args.reference,
        "pairs": {key: {"difference": value.get("difference"),
                        "interval": [value.get("interval_low"), value.get("interval_high")],
                        "n_days": value.get("n_days"),
                        "excludes_zero": value.get("interval_excludes_zero")}
                  for key, value in pairs.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
