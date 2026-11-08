"""The receipt checker must catch the failure it was built for, not merely report success.

write_exclusive opens the target with "x" and then dumps with allow_nan=False, so a non-finite
value leaves a partially written file -- and for the reading stage that half-file IS the receipt,
which makes the wrapper skip the stage as done forever. These tests build that exact stub by
calling the real write_exclusive, rather than by writing an arbitrary bad string, so the guard is
pinned against the production failure shape.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools/check_round_receipts.py"


def _run(attempt: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECKER), "--attempt", str(attempt), *extra],
                          capture_output=True, text=True, timeout=300)


def test_a_partial_write_is_caught_and_the_stub_really_is_partial(tmp_path):
    """Reproduce the production failure: write_exclusive leaves a truncated file behind."""
    sys.path.insert(0, str(ROOT))
    from scripts.study_r7_s3_wide_single_factor import write_exclusive

    stub = tmp_path / "readings.json"
    with pytest.raises(ValueError, match="not JSON compliant"):
        write_exclusive(stub, {"a": 1, "b": float("nan"), "c": 2})

    # The premise of the whole guard: the file exists and does not parse.
    assert stub.is_file() and stub.stat().st_size < 40
    with pytest.raises(json.JSONDecodeError):
        json.loads(stub.read_text(encoding="utf-8"))

    done = _run(tmp_path, "--require-reading")
    assert done.returncode == 1
    report = json.loads(done.stdout)
    reading = next(item for item in report["results"] if item["stage"] == "reading")
    assert reading["ok"] is False
    assert "truncated or invalid JSON" in reading["reason"]


def test_a_complete_document_passes_and_missing_seeds_fail(tmp_path):
    """Both directions, so a checker that always fails would not pass this file."""
    (tmp_path / "prepared_protocol.json").write_text(json.dumps({"k": 1}), encoding="utf-8")
    (tmp_path / "archive_receipt.json").write_text(json.dumps({"k": 1}), encoding="utf-8")
    (tmp_path / "readings.json").write_text(json.dumps({"scientific_claim": False}),
                                            encoding="utf-8")
    done = _run(tmp_path, "--require-reading")
    assert done.returncode == 0, done.stdout

    done = _run(tmp_path, "--require-seeds", "41", "42", "43")
    assert done.returncode == 1
    missing = [item["stage"] for item in json.loads(done.stdout)["failed"]]
    assert missing == ["seed41", "seed42", "seed43"]


def test_a_one_armed_seed_receipt_is_refused(tmp_path):
    """Half a paired comparison must fail here, not surface as a KeyError after hours of work."""
    for name in ("prepared_protocol.json", "archive_receipt.json"):
        (tmp_path / name).write_text(json.dumps({"k": 1}), encoding="utf-8")
    (tmp_path / "seed41_receipt.json").write_text(
        json.dumps({"seed": 41, "arms": {"off": {}}}), encoding="utf-8")

    done = _run(tmp_path, "--require-seeds", "41")
    assert done.returncode == 1
    entry = next(item for item in json.loads(done.stdout)["results"] if item["stage"] == "seed41")
    assert entry["ok"] is False
    assert "expected 2 arms" in entry["reason"]

    (tmp_path / "seed41_receipt.json").write_text(
        json.dumps({"seed": 41, "arms": {"off": {}, "on": {}}}), encoding="utf-8")
    assert _run(tmp_path, "--require-seeds", "41").returncode == 0


def test_a_missing_attempt_directory_is_a_usage_error(tmp_path):
    done = _run(tmp_path / "nope")
    assert done.returncode == 2
