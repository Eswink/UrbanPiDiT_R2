"""Assembly and judging for the in-protocol round: per-arm tables and the family verdict.

Split out of the driver to keep each file one job and to stay inside R-051's 600-line cap. The
driver trains and scores; this module turns the receipts back into the tables the verdict reads. The
separation matters for one reason beyond size: the reading stage must be runnable on an attempt
directory without importing a training entry point, so a judging run cannot depend on the training
stack being importable, let alone on a GPU.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_in_protocol_protocol as proto

LEADS = proto.LEADS
SEEDS = proto.SEEDS


def _region_rmse(path: Path, region: str, variable: str) -> float:
    """One RMSE from a ``boundary_rmse.csv``, refusing a silently absent row."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] == region and row["variable"] == variable:
                return float(row["rmse"])
    raise RuntimeError(f"{region}/{variable} missing from {path}")




def climatology_table(directory: Path) -> dict:
    """``{lead: {variable: climatology mse}}`` for one arm's evaluation tree.

    The verdict divides every per-case delta by the cell's climatology MSE, so the table travels with
    the arm rather than being re-read at judging time. ``rmse_climatology`` is squared rather than
    back-derived from ``mse_skill``, so the denominator does not depend on the forecast it normalises.
    """
    table = {}
    for lead in LEADS:
        path = Path(directory) / f"lead_{lead:03d}h" / "climatology_skill.csv"
        if not path.is_file():
            continue
        row = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for entry in csv.DictReader(handle):
                value = float(entry["rmse_climatology"])
                row[str(entry["variable"])] = value * value
        table[str(lead)] = row
    return table




def readings(output: Path, body: dict) -> dict:
    """Assemble the per-arm tables and evaluate the families."""
    from scripts import r7_s3_in_protocol_verdict as verdict
    from training.r7_arm_harness import sha256_file

    output = Path(output)
    arms = {name: {} for name, _ in proto.ARMS}
    missing = []
    for seed in SEEDS:
        for name, _mode in proto.ARMS:
            receipt_path = output / f"seed{seed}_{name}_receipt.json"
            if not receipt_path.is_file():
                missing.append(f"{name}/seed{seed}")
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            evaluations = receipt.get("evaluations") or {}
            if len(evaluations) != len(LEADS):
                missing.append(f"{name}/seed{seed}: {len(evaluations)} of {len(LEADS)} leads")
            arms[name][str(seed)] = {
                "parameters": receipt["parameters"],
                "typed_evidence_mode": receipt["typed_evidence_mode"],
                "seconds_per_update": receipt["seconds_per_update"],
                "evaluation_dirs": {str(lead): entry["dir"]
                                    for lead, entry in evaluations.items()},
                # The climatology denominators travel with the arm: the verdict divides every
                # per-case delta by the cell's climatology MSE, and re-reading the CSVs at judging
                # time would let the denominator drift away from the arm it belongs to.
                "climatology_mse": climatology_table(output / f"seed{seed}/evaluation/{name}"),
            }
    body_readings = {
        "format": "r7-s3-typed-evidence-in-protocol-readings-v1",
        "protocol_sha256": body["protocol_sha256"],
        "updates": int(body["updates"]), "batch_size": int(body["batch_size"]),
        "epochs": int(body["updates"]) / 32_131,
        "seeds": list(SEEDS), "lead_hours": list(LEADS),
        "channel_order": proto.store_channels(),
        "region": proto.REGION,
        "families": body["families"],
        "thresholds": body["thresholds"],
        "arms": arms,
        "complete": not missing,
        "missing": sorted(set(missing)),
        "closed_doses": body["closed_doses"],
        "independent_arm_argument": body["independent_arm_argument"],
        "preflight": body["preflight"],
        "checks": {
            "capacity": body["preflight"]["clauses"]["capacity_is_matched_where_it_must_be"],
            "forward_arithmetic": body["preflight"]["clauses"]["forward_arithmetic_is_matched"],
            "envelope": body["preflight"]["clauses"]["envelope_covers_the_treated_rate"],
            "val_manifest_sha256": sha256_file(Path(body["val_manifest"])),
        },
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "limitations": body["limitations"],
    }
    judgement = verdict.evaluate(body_readings)
    body_readings["exits"] = judgement
    body_readings["exit_conditions"] = judgement["clauses"]
    body_readings["verdict"] = judgement["verdict"]
    return body_readings


