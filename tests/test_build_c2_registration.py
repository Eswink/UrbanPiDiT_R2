"""The C2 registration record is derived from artifacts, and refuses hand-typed identity.

Decision 0103: two index records carried a 7-character commit padded to 40 with zeros, which
passes a shape check and names no object. Hand-assembling a record is exactly how that happens,
because every field is a digest or a SHA. These tests pin that the builder computes them.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools/build_c2_registration.py"
PROTOCOL = ROOT / "outputs/r7_s3_derivative_matching_c2_20261108_attempt06/protocol.json"
EVIDENCE = ROOT / "docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md"

EXITS_CLOSED = {"a_72h_skill_mean_delta": False, "b_matching_bit": True,
                "c_off_path_unchanged": True, "d_no_criterion_changed": True,
                "all": False, "delta_72h": {"41": -0.003, "42": 0.001, "43": -0.002},
                "verdict": "permanently-close-derivative-matching-branch"}


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (tmp_path / "stage_ledger.jsonl").write_text(
        json.dumps({"phase": "prepare", "elapsed_seconds": 3600.0}) + "\n"
        + json.dumps({"phase": "reading", "elapsed_seconds": 60.0}) + "\n", encoding="utf-8")
    readings = {"scientific_claim": False, "test_read": False, "seeds": {},
                "protocol_sha256": protocol["protocol_sha256"], "exit_conditions": EXITS_CLOSED}
    (tmp_path / "readings.json").write_text(json.dumps(readings), encoding="utf-8")
    return tmp_path / "readings.json", tmp_path / "protocol.json"


def _run(tmp_path: Path, out: Path) -> subprocess.CompletedProcess:
    readings, protocol = _fixtures(tmp_path)
    return subprocess.run(
        [sys.executable, str(BUILDER), "--readings", str(readings), "--protocol", str(protocol),
         "--evidence", "docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md",
         "--record-id", "s3-rehearsal", "--outcome-class", "mixed",
         "--candidate-state", "not-candidate", "--priority", "50", "--reason", "rehearsal",
         "--limitation", "rehearsal only", "--excluded-reason", "rehearsal",
         "--out", str(out)], capture_output=True, text=True, timeout=300)


def test_the_record_carries_a_real_full_sha_and_derived_gpu_hours(tmp_path):
    """The commit comes from git at full length; the hours come from the ledger."""
    out = tmp_path / "record.json"
    done = _run(tmp_path, out)
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads(out.read_text(encoding="utf-8"))

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    assert record["evidence_commit"] == head
    assert len(record["evidence_commit"]) == 40
    assert subprocess.run(["git", "cat-file", "-e", record["evidence_commit"]],
                          cwd=ROOT).returncode == 0
    # 3600 + 60 seconds, not a wall clock and not typed. The builder rounds to 6 decimals
    # because every earlier ledger row is recorded at that precision, so the tolerance is the
    # rounding step, not float noise.
    assert record["metrics"]["gpu_hours"] == pytest.approx(3660 / 3600, abs=1e-6)
    assert record["metrics"]["stage_ledger_rows"] == 2
    assert record["metrics"]["exit_verdict"] == EXITS_CLOSED["verdict"]
    assert record["scientific_claim"] is False


def test_a_record_whose_page_is_missing_is_refused(tmp_path):
    """The page must exist before its digest can be pinned -- the ordering trap from earlier."""
    out = tmp_path / "record.json"
    readings, protocol = _fixtures(tmp_path)
    done = subprocess.run(
        [sys.executable, str(BUILDER), "--readings", str(readings), "--protocol", str(protocol),
         "--evidence", "docs/DOES_NOT_EXIST_PAGE.md", "--record-id", "s3-rehearsal",
         "--outcome-class", "mixed", "--candidate-state", "not-candidate", "--priority", "50",
         "--reason", "x", "--limitation", "y", "--excluded-reason", "z", "--out", str(out)],
        capture_output=True, text=True, timeout=300)
    assert done.returncode == 1
    assert not out.exists()


def test_a_non_candidate_needs_an_exclusion_reason(tmp_path):
    """A not-candidate record without a reason fails validation, so nothing is written."""
    out = tmp_path / "record.json"
    readings, protocol = _fixtures(tmp_path)
    done = subprocess.run(
        [sys.executable, str(BUILDER), "--readings", str(readings), "--protocol", str(protocol),
         "--evidence", "docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md", "--record-id", "s3-x",
         "--outcome-class", "mixed", "--candidate-state", "not-candidate", "--priority", "50",
         "--reason", "x", "--limitation", "y", "--out", str(out)],
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert not out.exists()
