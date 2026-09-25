"""Audit whether #7's adaptive-halting gate is even satisfiable on real data.

#7's research gate is: "adaptive model should approach fixed-Kmax accuracy with
meaningfully lower average reasoning depth". That sentence contains an assumption
— that fixed-Kmax is the accuracy to approach. This module tests the assumption
against a measured error-versus-depth curve before any controller is re-tuned,
because a controller cannot be blamed for failing to approach a reference that is
not the ceiling.

Two checks, both computed from an existing multi-depth evaluation:

1. **Is deeper actually better?** Compare the equal-channel normalized objective
   at each measured depth per arm and per seed. If a shallower depth wins
   consistently, the gate's reference point is confounded.
2. **Is the per-variable tolerance satisfiable?** The established policy search
   requires *every* variable and horizon to meet a relative-RMSE tolerance
   separately. This reports, for a grid of tolerances, whether any shallower
   depth qualifies at all — and which variable blocks it.

`scientific_claim: false`. This is an audit of existing bounded CPU results, not
a new training run; it inherits their limits.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TOLERANCES = (0.0, 0.01, 0.05, 0.10)


def load_depth_table(result_path, *, lead_hours=6.0):
    """Per-arm/depth/variable seed values from a comparison result."""
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    table = {}
    for row in payload["summary"]:
        if float(row["lead_hours"]) != float(lead_hours):
            continue
        table[(row["arm"], int(row["depth"]), row["variable"])] = row
    if not table:
        raise ValueError(f"no rows at lead {lead_hours} in {result_path}")
    return payload, table


def normalization_std(store_path):
    """Per-channel training std, used to make channels comparable."""
    import numpy as np
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    names = list(root.attrs["channels"])
    std = np.asarray(root["normalization_std"][:], dtype=np.float64)
    if len(names) != std.size or not (std > 0).all():
        raise ValueError("channel/std mismatch or nonpositive std")
    return names, std


def score(table, names, std, *, arm, depth, seed_index):
    """Equal-channel normalized RMSE for one arm/depth/seed.

    This is the objective the training loss actually minimizes (equal-channel
    MSE on normalized fields), so it — not any single physical variable — is the
    right yardstick for "accuracy" in the gate.
    """
    index = {name: i for i, name in enumerate(names)}
    values = []
    for name in names:
        row = table[(arm, depth, name)]
        values.append(float(row["values"][seed_index]) / float(std[index[name]]))
    if not values:
        raise ValueError("no channels scored")
    return sum(values) / len(values)


def depth_curve(table, names, std, *, arm, seeds):
    """{seed_index: [score at each measured depth]} plus the arithmetic mean."""
    from statistics import fmean

    depths = sorted({depth for (row_arm, depth, _) in table if row_arm == arm})
    if len(depths) < 2:
        raise ValueError(f"arm {arm} has fewer than two measured depths")
    per_seed = {}
    for seed_index in range(len(seeds)):
        per_seed[seed_index] = [
            score(table, names, std, arm=arm, depth=depth, seed_index=seed_index)
            for depth in depths
        ]
    mean = [fmean(per_seed[i][i_depth] for i in per_seed)
            for i_depth in range(len(depths))]
    return {"arm": arm, "depths": depths, "per_seed": per_seed, "mean": mean}


def gauge_depth_benefit(curves, *, shallowest=None):
    """Does the deepest measured depth actually beat the shallowest?

    Counted per seed, so a mean-only win is not mistaken for a consistent one.
    """
    results = []
    for curve in curves:
        depths, per_seed = curve["depths"], curve["per_seed"]
        first = 0 if shallowest is None else depths.index(shallowest)
        last = len(depths) - 1
        if first == last:
            continue
        better = worse = 0
        for seed_index, scores in per_seed.items():
            if scores[last] < scores[first]:
                better += 1
            elif scores[last] > scores[first]:
                worse += 1
        results.append({
            "arm": curve["arm"],
            "shallow_depth": depths[first], "deep_depth": depths[last],
            "shallow_mean": curve["mean"][first], "deep_mean": curve["mean"][last],
            "relative_change": (curve["mean"][last] - curve["mean"][first])
                               / curve["mean"][first],
            "seeds_deeper_better": better, "seeds_deeper_worse": worse,
            "deeper_is_consistently_worse": worse > 0 and better == 0,
        })
    return results


def tolerance_feasibility(table, *, reference_depth, tolerances, baseline_arm,
                          candidate_arm, lead_hours=6.0):
    """Can any shallower depth satisfy the per-variable tolerance?

    Mirrors `select_validation_policy`: every variable must meet the tolerance
    separately, with no aggregation. Reports the blocking variable so the reason
    a search returns "full depth" is visible instead of inferred.
    """
    depths = sorted({depth for (arm, depth, _) in table
                     if arm == candidate_arm and depth < reference_depth})
    variables = sorted({variable for (arm, _, variable) in table if arm == candidate_arm})
    if not depths:
        raise ValueError("no shallower candidates to test")
    rows = []
    for tolerance in tolerances:
        if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerances must be finite and nonnegative")
        qualifying = []
        for depth in depths:
            ratios = {}
            for variable in variables:
                candidate = table[(candidate_arm, depth, variable)]["rmse_mean"]
                reference = table[(baseline_arm, reference_depth, variable)]["rmse_mean"]
                if reference <= 0:
                    raise ValueError(f"nonpositive reference RMSE for {variable}")
                ratios[variable] = candidate / reference
            worst = max(ratios, key=lambda name: ratios[name])
            if all(ratio <= 1.0 + tolerance for ratio in ratios.values()):
                qualifying.append(depth)
            rows.append({
                "tolerance": float(tolerance), "depth": depth,
                "variables_within_tolerance": sum(
                    1 for ratio in ratios.values() if ratio <= 1.0 + tolerance),
                "variables_total": len(variables),
                "feasible": all(ratio <= 1.0 + tolerance for ratio in ratios.values()),
                "blocking_variable": worst, "blocking_ratio": ratios[worst],
            })
        for row in rows:
            if row["tolerance"] == float(tolerance):
                row["any_depth_feasible"] = bool(qualifying)
        for row in rows:
            row.setdefault("any_depth_feasible", False)
    return rows


def audit(result_path, store_path, *, lead_hours=6.0, tolerances=DEFAULT_TOLERANCES,
          baseline_arm="process_feedback", reference_depth=3, seeds=(41, 42, 43)):
    payload, table = load_depth_table(result_path, lead_hours=lead_hours)
    names, std = normalization_std(store_path)
    arms = sorted({arm for (arm, _, _) in table})
    curves = [depth_curve(table, names, std, arm=arm, seeds=seeds) for arm in arms]
    benefit = gauge_depth_benefit(curves)
    feasibility = tolerance_feasibility(
        table, reference_depth=reference_depth, tolerances=tolerances,
        baseline_arm=baseline_arm, candidate_arm=baseline_arm, lead_hours=lead_hours)
    # The gate needs BOTH: a depth reduction that is feasible against the
    # reference, and a reference that is actually the best available accuracy.
    feasible_depths = sorted({row["depth"] for row in feasibility if row["feasible"]})
    reference_is_ceiling = all(not item["deeper_is_consistently_worse"] for item in benefit)
    report = {
        "format": "r7-halting-gate-audit-v1",
        "scientific_claim": False,
        "source_result": str(Path(result_path)),
        "store": str(Path(store_path)),
        "protocol_sha256": payload.get("protocol_sha256"),
        "lead_hours": float(lead_hours),
        "arms": arms, "seeds": list(seeds),
        "depth_curves": curves,
        "depth_benefit": benefit,
        "tolerance_feasibility": feasibility,
        "feasible_shallower_depths": feasible_depths,
        "reference_depth": reference_depth,
        "reference_is_accuracy_ceiling": bool(reference_is_ceiling),
        "gate_satisfiable": bool(feasible_depths) and reference_is_ceiling,
        "verdict": _verdict(feasible_depths, reference_is_ceiling, baseline_arm),
        "limitations": [
            "bounded CPU comparison: 200 updates, dim=32, depth=2, ten January days per year",
            "three seeds: a noise check, not a significance test",
            "validation split only; the test split was not read",
            "one lead time (+6 h)",
            "this audits existing results; it trains nothing and re-tunes nothing",
            "the normalized objective weights channels equally, as the training loss does",
        ],
    }
    return report


def _verdict(feasible_depths, reference_is_ceiling, baseline_arm) -> dict:
    if not reference_is_ceiling and not feasible_depths:
        summary = ("gate not satisfiable as stated: the fixed-Kmax reference is not the "
                   "best available accuracy, and no shallower depth meets the per-variable "
                   "tolerance at any tested tolerance")
    elif not reference_is_ceiling:
        summary = ("gate confounded: fixed-Kmax is not the best available accuracy, so "
                   "'approach fixed-Kmax' would mean approaching a worse objective")
    elif not feasible_depths:
        summary = ("gate blocked by the per-variable tolerance: no shallower depth "
                   "qualifies against the reference")
    else:
        summary = "gate is satisfiable: a shallower depth qualifies and the reference is the ceiling"
    return {"baseline_arm": baseline_arm, "summary": summary,
            "adaptive_benefit_demonstrated": False}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether the #7 adaptive-halting gate is satisfiable.")
    parser.add_argument("--result", required=True,
                        help="a comparison result JSON with per-depth RMSE rows")
    parser.add_argument("--store", required=True, help="the published R7 zarr store")
    parser.add_argument("--out", required=True, help="new JSON report path")
    parser.add_argument("--lead-hours", type=float, default=6.0)
    parser.add_argument("--baseline-arm", default="process_feedback")
    parser.add_argument("--reference-depth", type=int, default=3)
    args = parser.parse_args()
    report = audit(args.result, args.store, lead_hours=args.lead_hours,
                   baseline_arm=args.baseline_arm, reference_depth=args.reference_depth)
    write_report(args.out, report)
    print(json.dumps({
        "gate_satisfiable": report["gate_satisfiable"],
        "reference_is_accuracy_ceiling": report["reference_is_accuracy_ceiling"],
        "feasible_shallower_depths": report["feasible_shallower_depths"],
        "verdict": report["verdict"]["summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
