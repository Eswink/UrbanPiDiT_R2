"""Segment B: prove the family machinery assembles, on readings that already exist, for free.

This is the round's zero-GPU gate. It reads the previous round's fifteen ``boundary_rmse.csv`` and
``provenance.json`` files, assembles both families exactly as the verdict will, and reports whether
the machinery can run at all: every cell built, one shared case set per family, enough calendar
blocks for the bootstrap. Nothing here trains, scores, or decides anything.

**What it is not.** The floor and the family partition were chosen after looking at these numbers, so
a reading computed from them cannot confirm the hypothesis -- it is the reason the hypothesis was
worth stating. The output carries ``exploratory: true`` and ``scientific_claim: false`` for that
reason, and the evidence page may cite it only as the origin of the thresholds, never as evidence
that the pathway works. Only the in-protocol arms can test the frozen criterion.

**Why assemble rather than trust.** ``simultaneous_paired_interval`` fails closed on a family whose
cells do not share one case-key set, and each lead scores a slightly different cohort. Running the
assembly here means that failure is a twenty-second CPU error before any GPU second is spent, which
is the same reason the preflight exists.

    python scripts/r7_s3_joint_criterion.py --dry-check
    python scripts/r7_s3_joint_criterion.py --out outputs/r7_s3_joint_criterion_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The previous round's attempt directory: three seeds x three arms, of which six receipts are
#: complete (incumbent and generic) and three ``typed`` arms were truncated. The probe uses the six.
PREVIOUS_RUN = ROOT / "outputs/r7_s3_typed_evidence_budget_20261116_attempt01"
PREVIOUS_ARMS = ("incumbent", "generic")
PREVIOUS_SEEDS = (41, 42, 43)
PREVIOUS_LEADS = (6, 12, 24, 48, 72)


def previous_arms() -> dict:
    """The previous round's arms in the shape the verdict reads, including climatology tables.

    Only directories that exist are recorded; a missing one is left out so the assembly reports it
    as an incomplete cell rather than raising a path error deep inside a bootstrap.
    """
    arms = {name: {} for name in PREVIOUS_ARMS}
    for seed in PREVIOUS_SEEDS:
        for name in PREVIOUS_ARMS:
            root = PREVIOUS_RUN / f"seed{seed}/evaluation/{name}"
            dirs = {}
            climatology = {}
            for lead in PREVIOUS_LEADS:
                directory = root / f"lead_{lead:03d}h"
                if (directory / "provenance.json").is_file():
                    dirs[str(lead)] = str(directory)
                    table = _climatology_table(directory)
                    if table:
                        climatology[str(lead)] = table
            if dirs:
                arms[name][str(seed)] = {"evaluation_dirs": dirs, "climatology_mse": climatology,
                                         "parameters": None}
    return arms


def _climatology_table(directory: Path) -> dict:
    """``{variable: climatology mse}`` from one lead's ``climatology_skill.csv``."""
    import csv

    path = Path(directory) / "climatology_skill.csv"
    if not path.is_file():
        return {}
    table = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row["rmse_climatology"])
            table[str(row["variable"])] = value * value
    return table


def dry_check() -> dict:
    """Assemble both families on the previous round's data and report whether they can run."""
    from scripts import r7_s3_in_protocol_protocol as proto
    from scripts import r7_s3_in_protocol_verdict as verdict

    arms = previous_arms()
    families = proto.families()
    order = proto.store_channels()
    report = {"scientific_claim": False, "test_read": False, "r": 0,
              "exploratory": True, "gpu_hours": 0.0, "source_run": str(PREVIOUS_RUN),
              "families": {}}
    for label, body in families.items():
        channels = list(body["channels"])
        try:
            assembled = verdict.family_cells(arms, list(PREVIOUS_SEEDS), list(PREVIOUS_LEADS),
                                             channels, "generic", "incumbent",
                                             channel_order=order)
        except Exception as error:                          # noqa: BLE001 - reported, not hidden
            report["families"][label] = {
                "assembled": False, "error": f"{type(error).__name__}: {error}",
                "expected_grids": body["grids"]}
            continue
        cells = assembled["cells"]
        complete = {cell: row for cell, row in cells.items()
                    if len(row["by_seed"]) == len(PREVIOUS_SEEDS)}
        outcome = {"status": "insufficient-evidence", "reason": "no complete cell"}
        if complete:
            try:
                outcome = verdict.build_family(
                    complete, block_length=proto.BLOCK_LENGTH_CASES,
                    resamples=proto.RESAMPLES, seed=proto.RESAMPLE_SEED,
                    alpha=proto.ALPHA_FAMILY)
            except Exception as error:                      # noqa: BLE001 - reported, not hidden
                outcome = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        report["families"][label] = {
            "assembled": True,
            "expected_grids": body["grids"],
            "n_cells_built": len(cells),
            "n_cells_complete": len(complete),
            "n_missing_reports": len(assembled["missing"]),
            "missing_sample": assembled["missing"][:5],
            "can_run": outcome.get("status") == "ok",
            "instrument": {key: outcome.get(key) for key in
                           ("status", "reason", "n_cases", "n_blocks", "block_length",
                            "n_union_cases", "n_dropped_cases", "critical_value", "alpha")},
            "reading": _summary(outcome),
        }
    report["all_families_can_run"] = all(entry.get("can_run") for entry in
                                         report["families"].values())
    report["rule"] = ("this proves the machinery assembles on real per-case rows and that each "
                      "family has one shared case set with enough calendar blocks. It is not a "
                      "test of the criterion: the floor and the partition were chosen after these "
                      "numbers were seen, so any reading here is exploratory by construction")
    report["limitations"] = [
        "computed on the previous round's arms, of which only incumbent and generic are complete; "
        "the typed arm is absent, so this is generic - incumbent, not the capacity-matched pair",
        "the floor and family partition were chosen after these numbers were seen; a positive "
        "reading here is the motivation for the threshold, never evidence that it holds",
        "the previous round's arms are a different protocol digest and a different code revision "
        "history; no number here may be carried into a registered reading",
    ]
    return report


def _summary(outcome: dict) -> dict:
    """The cell-level reading, kept separate from the instrument's own diagnostics."""
    cells = outcome.get("cells") or {}
    if not cells:
        return {}
    ordered = sorted(cells.items(), key=lambda item: item[1]["difference"], reverse=True)
    return {"n_cells": len(cells),
            "best": {cell: row["difference"] for cell, row in ordered[:5]},
            "worst": {cell: row["difference"] for cell, row in ordered[-5:]},
            "n_excluding_zero_positive": sum(1 for row in cells.values()
                                             if row["excludes_zero"] and row["difference"] > 0),
            "n_excluding_zero_negative": sum(1 for row in cells.values()
                                             if row["excludes_zero"] and row["difference"] < 0)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-check", action="store_true",
                        help="assemble the families and report; writes nothing")
    parser.add_argument("--out", type=Path, help="write the report here")
    arguments = parser.parse_args(argv)
    from scripts.r7_m3_offline import deny_network
    deny_network()
    report = dry_check()
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    summary = {"all_families_can_run": report["all_families_can_run"],
               "families": {label: {key: entry.get(key) for key in
                                    ("assembled", "n_cells_built", "n_cells_complete", "can_run")}
                            | {"instrument": entry.get("instrument", {}).get("n_cases")}
                            for label, entry in report["families"].items()},
               "out": str(arguments.out) if arguments.out else None}
    print(json.dumps(summary, indent=2))
    return 0 if report["all_families_can_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
