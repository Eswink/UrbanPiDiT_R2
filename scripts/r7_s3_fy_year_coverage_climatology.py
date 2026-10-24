"""The D2 climatology reference the year-coverage round divides by, and its frozen constants.

Split out of ``scripts/study_r7_s3_fy_year_coverage.py`` when that driver reached the R-051
600-line hard cap, the same way the other round drivers grew a ``*_support`` module. This is
pure data plus a byte-verification: the reference is the only frozen input the round reads
from outside its own instance directory, and the leads it carries are exactly the leads the
round evaluates, which is why the lead list lives here too.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_wide_single_factor as wide

EVALUATION_LEADS = wide.EVALUATION_LEADS
VAL_COHORTS = {"6": 1458, "12": 1457, "24": 1455, "48": 1451, "72": 1447}
CLIMATOLOGY_T2M_RMSE = {
    "6": 3.9188406572176904, "12": 3.9196037113045894, "24": 3.921682294513083,
    "48": 3.9253820690090073, "72": 3.926891361747067,
}
CLIMATOLOGY_YEARS, CLIMATOLOGY_BUCKETS, CLIMATOLOGY_STEPS = list(range(2000, 2022)), 48, 32144
D2_REFERENCE = ROOT / "outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json"
D2_FORMAT = "r7-s3-fy-year-coverage-d2-reference-v1"


def climatology_reference():
    """Load and byte-verify the D2 climatology reference this round divides by."""
    from training.r7_arm_harness import sha256_file
    if not D2_REFERENCE.is_file():
        raise FileNotFoundError(f"the year-coverage climatology reference is missing: {D2_REFERENCE}")
    reference = json.loads(D2_REFERENCE.read_text(encoding="utf-8"))
    if reference.get("format") != D2_FORMAT or reference.get("scientific_claim") is not False:
        raise RuntimeError("the climatology reference is not the frozen D2 artifact")
    leads = {}
    for lead in EVALUATION_LEADS:
        record = reference["leads"][str(lead)]
        if record["n_evaluated"] != record["n_available_windows"] \
                or record["n_evaluated"] != VAL_COHORTS[str(lead)]:
            raise RuntimeError(f"D2 lead {lead} does not cover the frozen val cohort")
        if len(record["bucket_counts"]) != CLIMATOLOGY_BUCKETS:
            raise RuntimeError(f"D2 lead {lead} does not carry the 48 buckets")
        csv_path = Path(record["boundary_rmse_csv"])
        if sha256_file(csv_path) != record["boundary_rmse_sha256"]:
            raise RuntimeError(f"D2 lead {lead} boundary table drifted from its digest")
        with (csv_path.parent / "rmse.csv").open(encoding="utf-8") as handle:
            t2m = next(row for row in csv.DictReader(handle) if row["variable"] == "t2m")
        value, unit = float(t2m["rmse"]), t2m["unit"]
        if unit != "K" or abs(value - CLIMATOLOGY_T2M_RMSE[str(lead)]) > 1e-12:
            raise RuntimeError(f"D2 lead {lead} t2m RMSE differs from the frozen constant")
        leads[str(lead)] = {"t2m_rmse": value, "unit": unit,
                            "n_evaluated": record["n_evaluated"],
                            "boundary_rmse_sha256": record["boundary_rmse_sha256"]}
    return {"path": str(D2_REFERENCE), "sha256": sha256_file(D2_REFERENCE),
            "format": D2_FORMAT, "bucket_count": CLIMATOLOGY_BUCKETS,
            "training_years": CLIMATOLOGY_YEARS, "n_selected_steps": CLIMATOLOGY_STEPS,
            "leads": leads}
