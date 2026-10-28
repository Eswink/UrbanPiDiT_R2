"""Read one finished R1 seed's per-variable RMSE and emit the evidence-page rows.

The evidence page's 17-variable x 5-lead table is transcribed from each seed's
``boundary_rmse.csv`` with the official reader, not recomputed from anything else.
Handing that transcription to a script that prints the exact markdown rows is the
difference between "copied carefully" and "provably the same numbers"; the page has
been wrong once already from a hand-copied column.

    .venv/bin/python scripts/read_r1_seed_rows.py --seed 43 --arm treatment
    .venv/bin/python scripts/read_r1_seed_rows.py --seed 43 --arm treatment --json

``--json`` emits the same values as a mapping, for diffing against ``readings.json``
(the t2m column must agree bit-for-bit with what the driver's own ``_readings`` got).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01"
VARIABLES = ("t2m", "u10", "v10", "mslp", "t500", "t850", "u250", "u500", "u850",
             "v250", "v500", "v850", "z250", "z500", "z850", "q500", "q850")
LEADS = (6, 12, 24, 48, 72)
REGION = "full"


def region_variable_rmse(path: Path, region: str, variable: str) -> float:
    """The repo's own reader (``scripts/study_r7_s3_wide_single_factor.py:186``)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] == region and row["variable"] == variable:
                return float(row["rmse"])
    raise RuntimeError(f"{region}/{variable} missing from {path}")


def table_path(seed: int, arm: str, lead: int, output: Path = OUTPUT) -> Path:
    return output / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h/boundary_rmse.csv"


def seed_rows(seed: int, arm: str, output: Path = OUTPUT) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {}
    for variable in VARIABLES:
        values[variable] = {}
        for lead in LEADS:
            path = table_path(seed, arm, lead, output)
            if not path.is_file():
                raise FileNotFoundError(f"lead {lead} h for seed {seed} ({arm}) is not on disk: {path}")
            values[variable][lead] = region_variable_rmse(path, REGION, variable)
    return values


def markdown_rows(values: dict[str, dict[int, float]], seed: int) -> list[str]:
    return [f"| {variable} | {lead} h | {values[variable][lead]:.4f} | `seed {seed}` |"
            for variable in VARIABLES for lead in LEADS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", default="treatment", choices=("treatment", "control"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        values = seed_rows(args.seed, args.arm, Path(args.out))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({variable: {str(lead): value for lead, value in leads.items()}
                          for variable, leads in values.items()}, indent=1))
    else:
        print("\n".join(markdown_rows(values, args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
