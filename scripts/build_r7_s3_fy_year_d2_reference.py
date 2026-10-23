"""Freeze the 48-bucket train-only climatology the year-coverage round divides by.

``evaluate_local`` already computes the climatology baseline from the store's train years
and writes it next to every forecast table, so the denominator exists either way. This
script pre-computes it once, before any training, so the round's skill numbers divide by
a table frozen ahead of the readings rather than by whatever the evaluator happened to
produce on the scoring pass.

The climatology is fit on the store's declared train years (2000-2021) and scored on the
same val cohort the arms are scored on. Nothing here reads the test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_arm_harness import sha256_file
from training.r7_evaluate import evaluate_local

FORMAT = "r7-s3-fy-year-coverage-d2-reference-v1"
VAL_MANIFEST = (ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
                / "store/manifests_treatment/val.jsonl")
LEADS = (6, 12, 24, 48, 72)
BOUNDARY_MARGINS = (1,)
REGION = "full"


def _lead_record(destination, provenance):
    return {
        "n_evaluated": provenance["n_evaluated"],
        "n_available_windows": provenance["n_available_windows"],
        "boundary_rmse_csv": str(destination / "boundary_rmse.csv"),
        "boundary_rmse_sha256": sha256_file(destination / "boundary_rmse.csv"),
        "climatology": provenance["climatology"],
        "bucket_counts": provenance["climatology"]["bucket_counts"],
        "elapsed_seconds": provenance["elapsed_seconds"],
    }


def build(out_dir, *, device_name="cpu"):
    out_dir = Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"fresh exclusive output only: {out_dir}")
    if not VAL_MANIFEST.is_file():
        raise FileNotFoundError(f"the frozen val manifest is missing: {VAL_MANIFEST}")
    leads = {}
    for lead in LEADS:
        destination = out_dir / f"climatology/lead_{lead:03d}h"
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=destination, checkpoint=None, baseline="climatology",
            lead_hours=(lead,), max_samples=10**9, device_name=device_name,
            boundary_margins=BOUNDARY_MARGINS)
        if provenance["split"] != "val" or "test_read" in provenance:
            raise RuntimeError(f"lead {lead}: the climatology did not stay on val")
        record = _lead_record(destination, provenance)
        if len(record["bucket_counts"]) != 48:
            raise RuntimeError(f"lead {lead}: the climatology is not 48 buckets")
        leads[str(lead)] = record
        print(f"lead {lead} done {record['n_evaluated']} / {record['n_available_windows']} "
              f"buckets {len(record['bucket_counts'])}", flush=True)
    body = {
        "format": FORMAT, "scientific_claim": False,
        "val_manifest": str(VAL_MANIFEST.relative_to(ROOT)),
        "region": (f"{REGION} 65x65 (boundary_margins={BOUNDARY_MARGINS!r}, "
                   f"read the {REGION!r} region)"),
        "training_years": sorted({int(year) for year in
                                  leads[str(LEADS[-1])]["climatology"]["training_years"]}),
        "n_selected_steps": leads[str(LEADS[-1])]["climatology"]["n_selected_steps"],
        "leads": leads,
    }
    (out_dir / "reference.json").write_text(
        json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    body = build(args.out, device_name=args.device)
    print(json.dumps({"training_years": body["training_years"],
                      "n_selected_steps": body["n_selected_steps"],
                      "leads": {lead: record["n_evaluated"]
                                for lead, record in body["leads"].items()}}, indent=1))
    print("wrote reference.json")


if __name__ == "__main__":
    main()
