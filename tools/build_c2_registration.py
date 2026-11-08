"""Build Stage C2's evidence-index record from artifacts, then validate it before writing.

Why this exists. Two records in ``docs/R7_EVIDENCE_INDEX.jsonl`` were written with a 7-character
commit SHA padded to 40 characters with zeros, which passes the shape check and names no git
object (decision 0103). Hand-assembling a record is where that happens: every field is a digest,
a SHA, or a counted value, and any of them can be mistyped silently. This script derives every
field from files and from ``git rev-parse``, so there is nothing to transcribe.

What it will NOT do: decide anything scientific. ``outcome_class``, ``candidate_state``,
``priority``, ``reason`` and ``limitations`` are arguments, because they are human judgements
about what the numbers mean. The verdict itself comes from ``readings.json``'s
``exit_conditions``, which the driver computed.

    python tools/build_c2_registration.py \
        --readings outputs/.../readings.json \
        --protocol outputs/.../protocol.json \
        --evidence docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md \
        --record-id s3-derivative-matching-c2-result \
        --out /tmp/record.json

Exit codes: 0 = a record that passes ``validate_record``; 1 = it does not (nothing written);
2 = usage error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_r7_evidence_index import (  # noqa: E402
    RECORD_FORMAT,
    _sha256,
    validate_record,
)


def _git(*args: str) -> str:
    """Run a read-only git command and return its stripped output."""
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def _gpu_hours_from_ledger(attempt: Path) -> tuple[float, int]:
    """Sum the stage ledger's elapsed seconds, the same accounting earlier rows used.

    Earlier rounds recorded "sum of the stage ledger's elapsed_seconds / 3600" rather than
    wall-clock times a card count, because the ledger measures actual stage occupancy and excludes
    the gaps between stages. Returns (hours, number of rows summed).
    """
    ledger = attempt / "stage_ledger.jsonl"
    if not ledger.is_file():
        raise SystemExit(f"no stage ledger at {ledger}")
    total, rows = 0.0, 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        total += float(record.get("elapsed_seconds") or 0.0)
        rows += 1
    return round(total / 3600.0, 6), rows


def build(
    *,
    readings_path: Path,
    protocol_path: Path,
    evidence_path: Path,
    record_id: str,
    outcome_class: str,
    candidate_state: str,
    priority: int,
    reason: str,
    limitations: list[str],
    excluded_reason: str | None,
) -> dict:
    """Assemble the record from artifacts. Every digest is computed here, never typed."""
    readings = json.loads(readings_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    attempt = protocol_path.parent
    hours, ledger_rows = _gpu_hours_from_ledger(attempt)
    exits = readings["exit_conditions"]

    record = {
        "format": RECORD_FORMAT,
        "record_id": record_id,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        # HEAD, full 40 characters, straight from git: the field that was once padded.
        "evidence_commit": _git("rev-parse", "HEAD"),
        "experiment_commit": protocol["code_commit"],
        "protocol_sha256": protocol["protocol_sha256"],
        "data_identity": protocol["train_data_identity"],
        "source_sha256": protocol.get("source_sha256"),
        "outcome_class": outcome_class,
        "candidate_state": candidate_state,
        "priority": priority,
        "scientific_claim": False,
        "limitations": list(limitations),
        "metrics": {
            "gpu_hours": hours,
            "stage_ledger_rows": ledger_rows,
            "all_exits_true": bool(exits["all"]),
            "exit_verdict": exits["verdict"],
            "delta_72h": {seed: round(value, 6) for seed, value in exits["delta_72h"].items()},
            "test_read": bool(readings["test_read"]),
        },
        "reason": reason,
        "ci_run_id": None,
    }
    if candidate_state != "candidate":
        if not excluded_reason:
            raise SystemExit("a non-candidate record needs --excluded-reason")
        record["excluded_reason"] = excluded_reason
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readings", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--outcome-class", required=True)
    parser.add_argument("--candidate-state", required=True)
    parser.add_argument("--priority", required=True, type=int)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--limitation", action="append", default=[], dest="limitations")
    parser.add_argument("--excluded-reason")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.evidence.is_file():
        # The record pins the page's digest, so the page has to exist first. This ordering is
        # the trap that bit once already: a record written before its page cannot be validated.
        print(f"ERROR evidence page does not exist: {args.evidence}")
        return 1

    record = build(
        readings_path=args.readings, protocol_path=args.protocol, evidence_path=args.evidence,
        record_id=args.record_id, outcome_class=args.outcome_class,
        candidate_state=args.candidate_state, priority=args.priority, reason=args.reason,
        limitations=args.limitations, excluded_reason=args.excluded_reason,
    )
    errors = validate_record(record, ROOT, 1)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        print("nothing written")
        return 1
    args.out.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes); validate_record passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
