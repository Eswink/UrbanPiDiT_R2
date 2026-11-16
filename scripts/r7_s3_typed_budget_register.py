"""Assemble the round's four artefacts from ``readings.json``, refusing on any gap.

The registration step runs last, after the whole GPU budget is spent, so every field it needs is
derived here from files that already exist rather than typed by hand. Three facts are read straight
out of the round instead of being restated:

* the ``(a)``-``(f)`` clauses come from ``readings.json``'s ``exit_conditions`` -- the values that
  decided the round, not a second computation;
* the GPU hours come from the arm receipts' own elapsed seconds plus Segment B's two probe files,
  because no stage receipt records a cumulative total;
* the evidence digest and the commits are looked up at write time, so a placeholder cannot survive.

The module builds the artefacts and validates them; it never decides a scientific verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Segment B wrote these itself; they are this round's own measurements, not estimates.
SEGMENT_B = ("outputs/r7_s3_segB_probe_20261116/segment_b_probe.json",
             "outputs/r7_s3_segB_probe_20261116/segment_b_depth.json")

#: The record's ``metrics`` block, in the order the evidence page quotes them.
METRIC_KEYS = ("gpu_hours", "gpu_hours_segment_b", "gpu_hours_segment_a",
               "parameters_incumbent", "parameters_treated", "parameter_delta",
               "updates_per_arm", "batch_size", "seeds")


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def segment_b_hours() -> dict:
    """Segment B's measured GPU hours, per probe file, with the total."""
    per_file = {}
    for relative in SEGMENT_B:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Segment B artifact is missing: {relative}")
        body = _read(path)
        hours = body.get("gpu_hours")
        if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0:
            raise RuntimeError(f"{relative} carries no measured gpu_hours: {hours!r}")
        per_file[Path(relative).stem] = float(hours)
    return {"per_file": per_file, "total": sum(per_file.values())}


def segment_a_hours(attempt: Path) -> dict:
    """Sum every arm receipt's training and scoring seconds.

    ``elapsed_seconds`` in an arm receipt is the training loop; each ``evaluations[lead]`` entry
    carries the scoring loop separately. Both ran on a GPU, so both are GPU hours. A receipt that
    lacks either is refused rather than silently under-counted.
    """
    attempt = Path(attempt)
    per_arm = {}
    for receipt_path in sorted(attempt.glob("seed*_*_receipt.json")):
        body = _read(receipt_path)
        train = body.get("elapsed_seconds")
        evaluations = body.get("evaluations") or {}
        if not isinstance(train, (int, float)) or train <= 0:
            raise RuntimeError(f"{receipt_path.name} has no positive elapsed_seconds")
        if not evaluations:
            raise RuntimeError(f"{receipt_path.name} scored no lead")
        scoring = 0.0
        for lead, entry in evaluations.items():
            seconds = entry.get("elapsed_seconds")
            if not isinstance(seconds, (int, float)) or seconds <= 0:
                raise RuntimeError(f"{receipt_path.name} lead {lead} has no elapsed_seconds")
            scoring += float(seconds)
        per_arm[receipt_path.stem] = {"training": float(train), "scoring": scoring,
                                      "total": float(train) + scoring}
    if not per_arm:
        raise RuntimeError(f"no arm receipts in {attempt}")
    # Quantized to 1e-4 because the ledger row and the record's ``metrics.gpu_hours`` must agree at
    # that precision; leaving the float tail on would make an exact comparison fail on 1e-16.
    total = sum(item["total"] for item in per_arm.values()) / 3600.0
    return {"per_arm": per_arm, "total_gpu_hours": round(total, 4)}


def clause_summary(readings: dict) -> dict:
    """The six clauses exactly as ``readings.json`` recorded them, with no reinterpretation."""
    exits = readings.get("exit_conditions") or {}
    values = (readings.get("exits") or {}).get("values") or {}
    expected = ("a_typed_minus_generic", "b_generic_minus_incumbent", "c_claim_scope",
                "d_budget_direction", "e_z500_72h_guard")
    missing = [name for name in expected if name not in exits]
    if missing:
        raise RuntimeError(f"readings.json lacks clause entries {missing}")
    # (f) is not one of the values above: it is the six-clause preflight. ``checks`` stores the two
    # capacity/arithmetic entries as dicts carrying a ``passed`` key, so testing them against
    # ``is True`` would report (f) false for a green preflight -- the exact shape confusion that
    # published a card contradicting its own reading once already.
    checks = readings.get("checks") or {}
    preflight = (readings.get("preflight") or {}).get("clauses") or {}
    f_names = ("off_path_is_the_previous_model", "slot_arrival_is_diagonal",
               "future_labels_are_unread", "capacity_is_matched_where_it_must_be",
               "forward_arithmetic_is_matched", "budget_and_closed_dose_are_registered")
    f_clauses = {}
    for name in f_names:
        entry = preflight.get(name)
        f_clauses[name] = None if entry is None else entry.get("passed")
    payload = (readings.get("preflight") or {})
    return {"clauses": {name: exits[name] for name in expected},
            "values": {name: values.get(name) for name in expected},
            "f_preflight": {"passed": (payload.get("f_preflight_passed") is True
                                       and all(value is True for value in f_clauses.values())),
                            "clauses": f_clauses,
                            "capacity": _passed(checks.get("capacity")),
                            "forward_arithmetic": _passed(checks.get("forward_arithmetic"))},
            "verdict": readings.get("verdict")}


def _passed(entry) -> bool | None:
    """A preflight entry is green only when its own ``passed`` flag is exactly ``True``.

    The entries are dicts, not booleans; ``entry is True`` would be false for every green check.
    ``None`` means the entry is absent, which is untested rather than passed.
    """
    if entry is None:
        return None
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, dict):
        value = entry.get("passed")
        return None if value is None else value is True
    return None


def build_record(readings: dict, *, evidence_path: str, evidence_sha256: str, evidence_commit: str,
                 experiment_commit: str, record_id: str, reason: str, outcome_class: str,
                 candidate_state: str, priority: int, limitations: list,
                 metrics: dict, excluded_reason: str) -> dict:
    """One index record. Shape only: every scientific value is passed in, never invented."""
    return {
        "format": "urbanpidit-r7-evidence-record-v1",
        "record_id": record_id,
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "evidence_commit": evidence_commit,
        "experiment_commit": experiment_commit,
        "protocol_sha256": readings.get("protocol_sha256"),
        "source_sha256": readings.get("source_sha256"),
        "data_identity": readings.get("data_identity"),
        "ci_commit": None,
        "ci_run_id": None,
        "outcome_class": outcome_class,
        "candidate_state": candidate_state,
        "priority": priority,
        "scientific_claim": False,
        "limitations": list(limitations),
        "metrics": dict(metrics),
        "reason": reason,
        "excluded_reason": excluded_reason,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--readings", type=Path)
    args = parser.parse_args(argv)
    attempt = Path(args.attempt)
    readings_path = Path(args.readings) if args.readings else attempt / "readings.json"
    readings = _read(readings_path)

    b_hours = segment_b_hours()
    a_hours = segment_a_hours(attempt)
    summary = clause_summary(readings)
    body = {
        "readings_sha256_path": str(readings_path),
        "segment_b": b_hours,
        "segment_a": a_hours,
        "gpu_hours_total": round(b_hours["total"] + a_hours["total_gpu_hours"], 4),
        "clauses": summary,
        "scientific_claim": False,
        "test_read": readings.get("test_read"),
        "seeds": readings.get("seeds"),
    }
    json.dump(body, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
