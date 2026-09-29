"""Counterproofs for the read-only R7 evidence index validator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_r7_evidence_index import (
    RECORD_FORMAT,
    check_brief_sync,
    load_index,
    main,
    render_brief,
    validate_record,
)


COMMIT = "a" * 40


def make_record(root: Path, *, record_id: str = "candidate-one") -> dict:
    evidence = root / "evidence.md"
    evidence.write_text("scientific_claim: false\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    return {
        "format": RECORD_FORMAT,
        "record_id": record_id,
        "evidence_path": "evidence.md",
        "evidence_sha256": digest,
        "evidence_commit": COMMIT,
        "experiment_commit": COMMIT,
        "protocol_sha256": "b" * 64,
        "data_identity": "c" * 64,
        "source_sha256": None,
        "outcome_class": "negative",
        "candidate_state": "candidate",
        "priority": 90,
        "scientific_claim": False,
        "reason": "Retain the negative result for human review.",
        "limitations": ["fixture is not weather truth"],
        "metrics": {"test_read": False, "seeds": 3},
        "ci_run_id": "12345",
        "excluded_reason": "Follow-up still requires a new frozen protocol and authorization.",
    }


def write_index(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n", encoding="utf-8")


def test_valid_record_checks_source_digest(tmp_path):
    record = make_record(tmp_path)
    assert validate_record(record, tmp_path, 1) == []


def test_tampered_evidence_is_rejected(tmp_path):
    record = make_record(tmp_path)
    (tmp_path / "evidence.md").write_text("changed\n", encoding="utf-8")
    errors = validate_record(record, tmp_path, 1)
    assert any("evidence_sha256 does not match" in error for error in errors)


@pytest.mark.parametrize("field,value", [
    ("evidence_path", "../outside.md"),
    ("evidence_commit", "short-sha"),
    ("protocol_sha256", "short-digest"),
    ("candidate_state", "run-now"),
])
def test_invalid_identity_or_state_is_rejected(tmp_path, field, value):
    record = make_record(tmp_path)
    record[field] = value
    errors = validate_record(record, tmp_path, 1)
    assert errors


def test_candidate_without_protocol_is_rejected(tmp_path):
    record = make_record(tmp_path)
    record["protocol_sha256"] = None
    errors = validate_record(record, tmp_path, 1)
    assert any("candidate requires protocol_sha256" in error for error in errors)


def test_non_candidate_requires_exclusion_reason(tmp_path):
    record = make_record(tmp_path)
    record["candidate_state"] = "needs-review"
    record.pop("excluded_reason")
    errors = validate_record(record, tmp_path, 1)
    assert any("require excluded_reason" in error for error in errors)


def test_duplicate_record_id_is_rejected(tmp_path):
    first = make_record(tmp_path, record_id="duplicate")
    second = make_record(tmp_path, record_id="duplicate")
    index = tmp_path / "index.jsonl"
    write_index(index, [first, second])
    records, errors = load_index(index, tmp_path)
    assert len(records) == 2
    assert any("duplicate record_id" in error for error in errors)


def test_brief_sync_accepts_canonical_render_and_rejects_stale_text(tmp_path):
    record = make_record(tmp_path)
    index = tmp_path / "index.jsonl"
    brief = tmp_path / "brief.md"
    write_index(index, [record])
    brief.write_text(render_brief([record]) + "\n", encoding="utf-8")
    assert check_brief_sync(index, brief, tmp_path) == []
    brief.write_text(render_brief([record]) + "\n\n", encoding="utf-8")
    assert any("does not match" in error for error in check_brief_sync(index, brief, tmp_path))


def test_brief_sync_rejects_invalid_index_before_render(tmp_path):
    record = make_record(tmp_path)
    record.pop("metrics")
    index = tmp_path / "index.jsonl"
    brief = tmp_path / "brief.md"
    write_index(index, [record])
    brief.write_text("not canonical\n", encoding="utf-8")
    errors = check_brief_sync(index, brief, tmp_path)
    assert any("metrics" in error for error in errors)
    assert not any("KeyError" in error for error in errors)


def test_brief_sync_cli_exit_codes(tmp_path):
    record = make_record(tmp_path)
    index = tmp_path / "index.jsonl"
    brief = tmp_path / "brief.md"
    write_index(index, [record])
    brief.write_text(render_brief([record]) + "\n", encoding="utf-8")
    assert main(["--index", str(index), "--root", str(tmp_path), "--check-brief", str(brief)]) == 0
    brief.write_text("stale\n", encoding="utf-8")
    assert main(["--index", str(index), "--root", str(tmp_path), "--check-brief", str(brief)]) == 1


def test_checked_in_brief_matches_index():
    root = Path(__file__).resolve().parents[1]
    index = root / "docs" / "R7_EVIDENCE_INDEX.jsonl"
    brief = root / "docs" / "R7_CANDIDATE_BRIEF.md"
    assert check_brief_sync(index, brief, root) == []


def test_brief_orders_only_by_explicit_priority_and_keeps_exclusions(tmp_path):
    high = make_record(tmp_path, record_id="high")
    low = make_record(tmp_path, record_id="low")
    high["priority"] = 90
    low["priority"] = 10
    low["candidate_state"] = "not-candidate"
    brief = render_brief([low, high])
    assert brief.index("## high") < brief.index("## low")
    assert "Excluded from runnable candidates" in brief
    assert "not a scientific verdict" in brief
    assert "not a scientific score" in brief
