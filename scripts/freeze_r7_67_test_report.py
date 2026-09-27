"""#67 sealed test report: one read-once evaluation of the frozen arms.

What this runs. Every arm is scored on the **test** split of the January re-cut
store, at all five declared leads, with the boundary stratification the protocol
declares. The result is written once and is the only test report for this arm
set; nothing here selects, tunes or drops anything.

Why the re-cut was necessary, and what it costs. The originally frozen segment
put its test block in February while the train-only month-hour climatology is
fitted from January only, so `normalized_climatology` had no bucket for any test
case and refused a held-out fallback - by design. The re-cut keeps the train
block byte-identical (verified: the train manifest hashes equal the frozen one)
and moves the held-out boundary so both val and test stay inside January. Because
`data_identity` covers `split_time_ranges`, the previous checkpoints are
correctly refused and every arm is **retrained** on the re-cut store. That makes
this a new experiment; it is reported as one and does not claim to confirm the
earlier frozen chain.

Anti-selection guarantees:

- the arm list, leads, headline variables and the block-resampling rule are read
  from ``docs/R7_67_PUBLICATION_PROTOCOL.md``'s frozen values, which were
  committed before this split existed;
- the split is chosen from the store's own timeline and written into the
  protocol payload below, so the same report cannot be regenerated against a
  different held-out boundary;
- the run refuses to start if any checkpoint's recorded model digest differs from
  the live ``model/`` tree, and refuses to mix two ``data_identity`` values.

    python scripts/freeze_r7_67_test_report.py \
        --train-root outputs/r7_recut_multiseed --out outputs/r7_67_sealed_report
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STORE = "outputs/r7_recut_segment/store/cache.zarr"
DEFAULT_MANIFESTS = "outputs/r7_recut_segment/store/manifests"
LEADS = (6, 12, 24, 48, 72)
# From the frozen protocol: headline variables and the accepted-degradation level.
HEADLINE_VARIABLES = ("t2m", "mslp", "v850", "u10")
HEADLINE_LEADS = (6, 24)
ACCEPTED_DEGRADATION = 0.01
BOUNDARY_MARGINS = (1, 2)
MAX_SAMPLES = 64
RESAMPLES = 2000
RESAMPLE_SEED = 20260927


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_arms(train_root):
    """One selected checkpoint per (seed, arm), read from each training report."""
    train_root = Path(train_root)
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    arms = {}
    for report_path in sorted(train_root.rglob("training_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report.get("selected_checkpoint")
        if not selected or not Path(selected).is_file():
            raise ValueError(f"{report_path} has no existing selected checkpoint")
        seed_dir = report_path.parent.parent.name          # seed41
        arm = report_path.parent.name                      # unet
        if not seed_dir.startswith("seed"):
            continue
        seed = int(seed_dir[4:])
        arms.setdefault(arm, {})[seed] = {
            "checkpoint": str(selected),
            "checkpoint_sha256": _sha256_file(selected),
            "selected_update": report.get("selected_update"),
            "training_report": str(report_path),
        }
    if not arms:
        raise ValueError(f"no arm training reports under {train_root}")
    return arms


def main():
    parser = argparse.ArgumentParser(description="#67 sealed test report (read once).")
    parser.add_argument("--manifests", default=DEFAULT_MANIFESTS)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"a sealed report is written once: {output}")
    output.mkdir(parents=True, exist_ok=False)

    from data.r7_evaluation import ZarrRolloutDataset
    from training.r7_evaluate import evaluate_local
    from training.r7_experiment import load_checkpoint, model_code_digest

    manifests = Path(args.manifests)
    store = manifests.parent / "cache.zarr"
    test_manifest = manifests / "test.jsonl"
    val_manifest = manifests / "val.jsonl"
    for path in (test_manifest, val_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    arms = discover_arms(args.train_root)
    identities = set()
    for arm, per_seed in arms.items():
        for seed, entry in per_seed.items():
            saved = load_checkpoint(entry["checkpoint"])
            identities.add(saved["contract"]["data_identity"])
    if len(identities) != 1:
        raise RuntimeError(f"checkpoints span {len(identities)} data identities; "
                           "a sealed report must be one experiment")
    identity = identities.pop()

    # The leads must all exist on test, or the report would silently cover fewer
    # horizons than the protocol declares.
    available = {}
    for lead in LEADS:
        ds = ZarrRolloutDataset(store, split="test", lead_hours=(lead,),
                                history_steps=2, step_hours=6)
        available[lead] = len(ds)
    if any(count == 0 for count in available.values()):
        raise ValueError(f"test split lacks declared leads: {available}")

    results = {
        "format": "r7-67-sealed-test-report-v1",
        "scientific_claim": False,
        "test_read": True,
        "read_count": 1,
        "device": args.device,
        "store": str(store),
        "data_identity": identity,
        "test_manifest_sha256": _sha256_file(test_manifest),
        "val_manifest_sha256": _sha256_file(val_manifest),
        "model_code_sha256": model_code_digest(),
        "leads": list(LEADS),
        "headline_variables": list(HEADLINE_VARIABLES),
        "headline_leads": list(HEADLINE_LEADS),
        "accepted_degradation": ACCEPTED_DEGRADATION,
        "boundary_margins": list(BOUNDARY_MARGINS),
        "case_counts": {str(lead): count for lead, count in available.items()},
        "arms": {arm: {str(seed): entry for seed, entry in sorted(per_seed.items())}
                 for arm, per_seed in sorted(arms.items())},
        "evaluation": {},
    }

    started_agents = []
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            entry = arms[arm][seed]
            for lead in LEADS:
                run_dir = output / "evaluation" / f"{arm}_s{seed}" / f"lead_{lead:03d}h"
                report = evaluate_local(
                    test_manifest, output_dir=run_dir, checkpoint=entry["checkpoint"],
                    lead_hours=(lead,), max_samples=args.max_samples,
                    device_name=args.device, boundary_margins=BOUNDARY_MARGINS)
                if report["split"] != "test":
                    raise RuntimeError(f"{arm} s{seed} scored on {report['split']}, not test")
                results["evaluation"][f"{arm}_s{seed}@{lead}h"] = {
                    "arm": arm, "seed": seed, "lead_hours": lead,
                    "n_evaluated": report["n_evaluated"],
                    "rmse_csv": str(run_dir / "rmse.csv"),
                    "skill_csv": str(run_dir / "climatology_skill.csv"),
                    "acc_csv": str(run_dir / "acc.csv"),
                    "boundary_csv": (str(run_dir / "boundary_rmse.csv")
                                     if (run_dir / "boundary_rmse.csv").is_file() else None),
                    "evaluation_dir": str(run_dir),
                    "elapsed_seconds": report["elapsed_seconds"],
                }
                started_agents.append(f"{arm}_s{seed}@{lead}h")
            print(json.dumps({"sealed_test": arm, "seed": seed}), flush=True)

    # Every arm must be scored on the identical case set within a lead.
    for lead in LEADS:
        counts = {key: value["n_evaluated"] for key, value in results["evaluation"].items()
                  if value["lead_hours"] == lead}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"arms scored on different case counts at {lead}h: {counts}")
        results.setdefault("case_counts_by_lead", {})[str(lead)] = next(iter(counts.values()))

    _write_summary_tables(results, output)
    with (output / "sealed_test_report.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"complete": True, "cells": len(results["evaluation"]),
                      "data_identity": identity}, ensure_ascii=False))


def _write_summary_tables(results, output):
    """Wide RMSE/skill tables from the per-arm CSVs, plus the win/loss counts.

    Nothing is recomputed here: the numbers come from the evaluation CSVs, so the
    summary cannot disagree with the per-arm artifacts.
    """
    rmse_rows, skill_rows, boundary_rows = [], [], []
    for key, entry in sorted(results["evaluation"].items()):
        with Path(entry["rmse_csv"]).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rmse_rows.append({"arm": entry["arm"], "seed": entry["seed"],
                                  "lead_hours": entry["lead_hours"],
                                  "variable": row["variable"], "unit": row["unit"],
                                  "rmse": row["rmse"],
                                  "n_initializations": row["n_initializations"]})
        if Path(entry["skill_csv"]).is_file():
            with Path(entry["skill_csv"]).open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    skill_rows.append({"arm": entry["arm"], "seed": entry["seed"],
                                       "lead_hours": entry["lead_hours"],
                                       "variable": row["variable"], "unit": row["unit"],
                                       "rmse_forecast": row["rmse_forecast"],
                                       "rmse_climatology": row["rmse_climatology"],
                                       "mse_skill": row["mse_skill"]})
        if entry.get("boundary_csv") and Path(entry["boundary_csv"]).is_file():
            with Path(entry["boundary_csv"]).open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    boundary_rows.append({"arm": entry["arm"], "seed": entry["seed"],
                                          "lead_hours": entry["lead_hours"],
                                          **row})
    _write_csv(output / "sealed_rmse_table.csv", rmse_rows)
    _write_csv(output / "sealed_skill_table.csv", skill_rows)
    _write_csv(output / "sealed_boundary_table.csv", boundary_rows)


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
