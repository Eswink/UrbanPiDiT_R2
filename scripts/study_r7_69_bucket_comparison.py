"""M2 two-month evaluation: does climatology still dominate the neural arms?

What this answers. Every earlier R7 segment is a single January, so the
train-only ``(month, hour)`` climatology has four buckets and the scored block
sits inside the same 30 days those buckets were averaged from. On such a segment
a "climatology beats the model" result cannot be separated from the data range.
M2 covers two months (8 buckets) and puts the scored block a week away from the
days the February buckets were fitted on, so the comparison is run again on
strictly less favourable ground for the baseline.

What this script does. It scores every trained arm and both parameter-free
controls on the M2 **test** split at all five declared leads, then reports for
each (lead, variable):

- the arm RMSE in **physical units**, and the same climatology RMSE;
- ``mse_skill`` (unit-free) and the physical ratio, so the two conventions can
  never be divided into each other again;
- the bucket count the climatology was actually fitted with, read from the
  evaluation provenance rather than asserted here.

This script exists because the earlier per-arm tables wrote physical forecast
RMSE next to a **normalized** climatology RMSE under the same unit label, and a
derived "9.09x" ratio was published from that mistake (docs/decisions/0010-*).

    python scripts/study_r7_69_bucket_comparison.py \\
        --train-root outputs/r7_m2_multiseed --out outputs/r7_m2_comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEADS = (6, 12, 24, 48, 72)
ARMS = ("unet", "native_window", "afno_small", "generic", "process")
BASELINES = ("persistence", "climatology")
HEADLINE_VARIABLES = ("t2m", "mslp", "v850", "u10")
MAX_SAMPLES = 64


def _sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_arms(train_root):
    """One selected checkpoint per (arm, seed), read from each training report."""
    train_root = Path(train_root)
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    arms = {}
    for report_path in sorted(train_root.rglob("training_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report.get("selected_checkpoint")
        if not selected or not Path(selected).is_file():
            raise ValueError(f"{report_path} has no existing selected checkpoint")
        seed_dir = report_path.parent.parent.name
        arm = report_path.parent.name
        if not seed_dir.startswith("seed") or arm not in ARMS:
            continue
        arms.setdefault(arm, {})[int(seed_dir[4:])] = {
            "checkpoint": str(selected),
            "checkpoint_sha256": _sha256_file(selected),
            "selected_update": report.get("selected_update"),
            "training_report": str(report_path),
        }
    if not arms:
        raise ValueError(f"no arm training reports under {train_root}")
    missing = [arm for arm in ARMS if arm not in arms]
    if missing:
        raise ValueError(f"training root is missing declared arms {missing}; a partial "
                         "arm set cannot answer the climatology question")
    return arms


def main():
    parser = argparse.ArgumentParser(description="M2 bucket-count comparison (#69).")
    parser.add_argument("--manifests", default="outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    from data.r7_evaluation import ZarrRolloutDataset
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import load_checkpoint, model_code_digest

    manifests = Path(args.manifests)
    store = manifests.parent / "cache.zarr"
    test_manifest = manifests / "test.jsonl"
    if not test_manifest.is_file():
        raise FileNotFoundError(test_manifest)

    arms = discover_arms(args.train_root)
    identities = {load_checkpoint(entry["checkpoint"])["contract"]["data_identity"]
                  for per_seed in arms.values() for entry in per_seed.values()}
    if len(identities) != 1:
        raise RuntimeError(f"checkpoints span {len(identities)} data identities; "
                           "one experiment at a time")
    identity = identities.pop()

    available = {}
    for lead in LEADS:
        dataset = ZarrRolloutDataset(store, split="test", lead_hours=(lead,),
                                     history_steps=2, step_hours=6)
        available[lead] = len(dataset)
    if any(count == 0 for count in available.values()):
        raise ValueError(f"test split lacks declared leads: {available}")

    results = {
        "format": "r7-69-bucket-comparison-v1",
        "scientific_claim": False,
        "test_read": True,
        "read_count": 1,
        "device": args.device,
        "store": str(store),
        "data_identity": identity,
        "model_code_sha256": model_code_digest(),
        "leads": list(LEADS),
        "headline_variables": list(HEADLINE_VARIABLES),
        "case_counts": {str(lead): count for lead, count in available.items()},
        "arms": {arm: {str(seed): entry for seed, entry in sorted(per_seed.items())}
                 for arm, per_seed in sorted(arms.items())},
        "evaluation": {},
    }

    for seed in sorted({s for per_seed in arms.values() for s in per_seed}):
        for name in list(ARMS) + list(BASELINES):
            for lead in LEADS:
                run_dir = output / "evaluation" / f"{name}_s{seed}" / f"lead_{lead:03d}h"
                trained = name not in BASELINES
                checkpoint = (arms[name][seed]["checkpoint"] if trained else None)
                report = evaluate_local(
                    test_manifest, output_dir=run_dir, checkpoint=checkpoint,
                    baseline=None if trained else name, lead_hours=(lead,),
                    max_samples=args.max_samples, device_name=args.device)
                if report["split"] != "test":
                    raise RuntimeError(f"{name} s{seed} scored on {report['split']}, not test")
                results["evaluation"][f"{name}_s{seed}@{lead}h"] = {
                    "arm": name, "seed": seed, "lead_hours": lead,
                    "n_evaluated": report["n_evaluated"],
                    "bucket_counts": report["climatology"]["bucket_counts"],
                    "n_selected_steps": report["climatology"]["n_selected_steps"],
                    "units": list(report["units"]),
                    "rmse_csv": str(run_dir / "rmse.csv"),
                    "skill_csv": str(run_dir / "climatology_skill.csv"),
                    "evaluation_dir": str(run_dir),
                }
        print(json.dumps({"scored_seed": seed}), flush=True)

    for lead in LEADS:
        counts = {key: value["n_evaluated"] for key, value in results["evaluation"].items()
                  if value["lead_hours"] == lead}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"models scored on different case counts at {lead}h: {counts}")
        results.setdefault("case_counts_by_lead", {})[str(lead)] = next(iter(counts.values()))

    buckets = {key: value["bucket_counts"] for key, value in results["evaluation"].items()}
    distinct = {tuple(sorted(counts)) for counts in buckets.values()}
    if len(distinct) != 1:
        raise RuntimeError(f"evaluations disagree on the climatology buckets: {distinct}")
    results["climatology_buckets"] = sorted(next(iter(distinct)))
    results["climatology_bucket_count"] = len(next(iter(distinct)))

    comparison = compare_against_climatology(results)
    results["comparison"] = comparison
    _write_tables(results, comparison, output)
    with (output / "bucket_comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({
        "complete": True,
        "data_identity": identity,
        "climatology_bucket_count": results["climatology_bucket_count"],
        "totals": comparison["totals"],
    }, ensure_ascii=False))


def _physical(entry):
    """Per-variable physical forecast RMSE and climatology RMSE for one evaluation.

    Both columns come from the same evaluation: the forecast RMSE from
    ``rmse.csv`` and the climatology RMSE from ``climatology_skill.csv``. Since
    the units defect was fixed both files are physical, which this asserts rather
    than assumes - a normalized number carrying a physical unit label is exactly
    the failure this script must not reproduce.
    """
    rmse = {row["variable"]: row for row in _read_csv(entry["rmse_csv"])}
    skill = {row["variable"]: row for row in _read_csv(entry["skill_csv"])}
    out = {}
    for variable, row in rmse.items():
        skill_row = skill.get(variable)
        if skill_row is None:
            continue
        units_rmse, units_skill = row["unit"], skill_row["unit"]
        if units_rmse != units_skill:
            raise ValueError(f"{entry['arm']} s{entry['seed']} {entry['lead_hours']}h "
                             f"{variable}: rmse.csv says {units_rmse!r} but "
                             f"climatology_skill.csv says {units_skill!r}; refusing to "
                             "compare across unit labels")
        if units_rmse == "normalized":
            raise ValueError(f"{entry['arm']} {variable}: metrics are normalized; a "
                             "physical-versus-climatology claim cannot be made from them")
        forecast = float(row["rmse"])
        climatology = float(skill_row["rmse_climatology"])
        skill_value = skill_row["mse_skill"]
        out[variable] = {
            "unit": units_rmse,
            "forecast": forecast,
            "climatology": climatology,
            "ratio": forecast / climatology if climatology > 0 else None,
            "mse_skill": float(skill_value) if skill_value not in ("", None) else None,
        }
    return out


def compare_against_climatology(results):
    """Count, per lead, how many (arm, seed, variable) cells beat the climatology.

    The verdict uses ``mse_skill``'s sign (a unit-free ratio of two MSEs on the
    same fields) and the physical RMSE comparison, and asserts they agree. That
    agreement is the check that would have caught the original defect: a
    normalized baseline read as physical disagrees with the skill sign.
    """
    per_lead = {str(lead): {"better": 0, "worse": 0, "unresolved": 0} for lead in LEADS}
    per_variable = {}
    disagreements = []
    rows = []
    for key, entry in sorted(results["evaluation"].items()):
        if entry["arm"] in BASELINES:
            continue
        metrics = _physical(entry)
        for variable, values in sorted(metrics.items()):
            skill_value = values["mse_skill"]
            if skill_value is None:
                outcome = "unresolved"
            else:
                outcome = "better" if skill_value > 0 else "worse"
            physical_outcome = ("better" if values["forecast"] < values["climatology"]
                                else "worse")
            if outcome != physical_outcome:
                disagreements.append({
                    "arm": entry["arm"], "seed": entry["seed"],
                    "lead_hours": entry["lead_hours"], "variable": variable,
                    "mse_skill": skill_value, "physical_outcome": physical_outcome})
            per_lead[str(entry["lead_hours"])][outcome] += 1
            bucket = per_variable.setdefault(variable, {"better": 0, "worse": 0,
                                                        "unresolved": 0})
            bucket[outcome] += 1
            rows.append({
                "arm": entry["arm"], "seed": entry["seed"],
                "lead_hours": entry["lead_hours"], "variable": variable,
                "unit": values["unit"], "rmse": values["forecast"],
                "rmse_climatology": values["climatology"], "ratio": values["ratio"],
                "mse_skill": skill_value, "outcome": outcome,
            })
    totals = {outcome: sum(entry[outcome] for entry in per_lead.values())
              for outcome in ("better", "worse", "unresolved")}
    if disagreements:
        raise RuntimeError(
            "the unit-free skill sign and the physical RMSE comparison disagree in "
            f"{len(disagreements)} cells, which is exactly the symptom of a mixed-unit "
            f"table; first few: {disagreements[:3]}")
    return {"per_lead": per_lead, "per_variable": per_variable, "totals": totals,
            "rows": rows,
            "rule": ("better = mse_skill > 0 (equivalently forecast RMSE < climatology "
                     "RMSE in the same physical units); the two must agree in every cell")}


def _write_tables(results, comparison, output):
    path = output / "m2_cell_table.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("arm", "seed", "lead_hours", "variable",
                                                    "unit", "rmse", "rmse_climatology",
                                                    "ratio", "mse_skill", "outcome"))
        writer.writeheader()
        writer.writerows(comparison["rows"])
    summary = output / "m2_outcome_summary.json"
    summary.write_text(json.dumps({
        "climatology_bucket_count": results["climatology_bucket_count"],
        "climatology_buckets": results["climatology_buckets"],
        "case_counts_by_lead": results.get("case_counts_by_lead"),
        "per_lead": comparison["per_lead"],
        "totals": comparison["totals"],
        "rule": comparison["rule"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Headline variables, physical units, seed mean -- the table the #67 report
    # originally got wrong.
    means = {}
    for row in comparison["rows"]:
        if row["variable"] in HEADLINE_VARIABLES:
            means.setdefault((row["lead_hours"], row["variable"], row["arm"]),
                             []).append(row["rmse"])
    headline = output / "m2_headline_table.csv"
    with headline.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "arm", "rmse_seed_mean", "n_seeds"])
        for (lead, variable, arm), values in sorted(means.items()):
            writer.writerow([lead, variable, arm, float(np.mean(values)), len(values)])
    return {"cells": path.name, "summary": summary.name, "headline": headline.name}


if __name__ == "__main__":
    main()
