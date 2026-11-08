"""The receipt guard: a half-written ``readings.json`` must not count as a finished stage.

The failure this covers is silent. ``write_exclusive`` opens the target with ``open("x")`` and
then dumps with ``allow_nan=False``; a non-finite value raises *inside* the write, leaving a
truncated file that is nonetheless the reading stage's receipt, so the orchestrator skips the
stage as done. These tests assert the guard reports that file as failed -- and, in the other
direction, that a complete receipt passes, because a guard that only ever fails proves nothing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _guard():
    """Load tools/check_round_receipts.py by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("check_round_receipts", ROOT / "tools/check_round_receipts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attempt(tmp_path: Path, *, readings: str | None, arms: tuple[str, ...] = ("off", "on")):
    folder = tmp_path / "attempt"
    folder.mkdir()
    (folder / "prepared_protocol.json").write_text('{"stage": "prepare"}', encoding="utf-8")
    (folder / "archive_receipt.json").write_text('{"stage": "archive"}', encoding="utf-8")
    (folder / "seed41_receipt.json").write_text(
        json.dumps({"arms": {name: {} for name in arms}}), encoding="utf-8")
    if readings is not None:
        (folder / "readings.json").write_text(readings, encoding="utf-8")
    return folder


def test_a_truncated_readings_file_is_not_a_finished_stage(tmp_path):
    """Exactly the bytes ``allow_nan=False`` leaves behind: 21 characters, no closing braces."""
    guard = _guard()
    # Measured against the real writer: json.dump raises after these 21 bytes.
    folder = _attempt(tmp_path, readings='{\n  "a": 1.0,\n  "b": ')
    report = guard.check(folder, [41], require_reading=True)
    assert report["passed"] is False
    assert [item["stage"] for item in report["failed"]] == ["reading"]
    assert "truncated" in report["failed"][0]["reason"]


def test_a_complete_readings_file_passes(tmp_path):
    """The other direction: a well-formed receipt must not be rejected."""
    guard = _guard()
    folder = _attempt(tmp_path, readings=json.dumps({"exit_conditions": {"all": True}}))
    report = guard.check(folder, [41], require_reading=True)
    assert report["passed"] is True
    assert report["checked"] == 4


def test_a_one_armed_seed_receipt_is_refused(tmp_path):
    """A missing arm is half a paired comparison; refusing it here beats a KeyError later."""
    guard = _guard()
    folder = _attempt(tmp_path, readings=json.dumps({}), arms=("off",))
    report = guard.check(folder, [41], require_reading=True)
    assert report["passed"] is False
    assert "expected 2 arms" in report["failed"][0]["reason"]


def test_a_missing_seed_receipt_is_reported_by_seed(tmp_path):
    """During a live round the seeds are genuinely absent; that must be named, not crashed on."""
    guard = _guard()
    folder = _attempt(tmp_path, readings=None)
    report = guard.check(folder, [41, 42, 43], require_reading=False)
    assert report["passed"] is False
    assert [item["stage"] for item in report["failed"]] == ["seed42", "seed43"]


def test_the_cli_exit_code_carries_the_verdict(tmp_path):
    """Piping to another program must not be able to hide the failure."""
    guard = _guard()
    folder = _attempt(tmp_path, readings='{"half": ')
    assert guard.main(["--attempt", str(folder), "--require-reading", "--quiet"]) == 1
    (folder / "readings.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert guard.main(["--attempt", str(folder), "--require-reading", "--quiet"]) == 0
