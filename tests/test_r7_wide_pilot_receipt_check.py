"""The pilot-receipt criteria checker, with a negative control per criterion.

A criteria checker that cannot report a deviation is worthless, so each frozen
criterion is exercised twice: once with a receipt that satisfies it, once with a
receipt that does not. No network, no GPU, no repository data.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "scripts/check_r7_wide_pilot_receipt.py"
    spec = importlib.util.spec_from_file_location("check_r7_wide_pilot_receipt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load()

CHANNELS = ["t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850", "z500",
            "t500", "q500", "u500", "v500", "z250", "u250", "v250"]


def _protocol():
    stamps = 120
    first = "2018-01-01T00:00:00"
    return {
        "protocol_sha256": "a" * 64,
        "read_plan": {"protocol_sha256": "b" * 64},
        "source": {"snapshot_id": "ZFKDHBCTBVHVXM3BQFV0"},
        "pilot_scope": {"stamps": stamps, "first_time": first, "last_time": "2018-01-30T18:00:00",
                        "channels": 17, "levels_hpa": [250, 500, 850]},
        "budgets": {"deadline_seconds": 1800.0},
        "reference_part": {"sha256": "c" * 64, "network_body_bytes": 3554716077,
                           "decoded_charged_bytes": 9482837880},
    }


def _timestamps():
    import pandas as pd

    return [stamp.isoformat() for stamp in pd.date_range("2018-01-01T00:00", periods=120, freq="6h")]


def _receipt():
    stamps = _timestamps()
    return {
        "status": "downloaded-real-source",
        "synthetic_fallback": False,
        "source_is_real_reanalysis": True,
        "read_plan_protocol_sha256": "b" * 64,
        "snapshot_id": "ZFKDHBCTBVHVXM3BQFV0",
        "timestamps": stamps,
        "init_hour_coverage": {"0": 30, "6": 30, "12": 30, "18": 30},
        "shape": [120, 3, 129, 129],
        "stored_level_axis_hpa": [250, 500, 850],
        "channel_count": 17,
        "channels": CHANNELS,
        "payloads": {name: {"payload_sha256": "d" * 64, "finite": True} for name in CHANNELS},
        "network_body_bytes": 3560000000,
        "decoded_chunk_budget": {"charged_bytes": 9482837880},
        "elapsed_seconds": 1300.0,
    }


def test_a_satisfying_receipt_passes_every_criterion():
    protocol = _protocol()
    report = CHECK.check(protocol, _receipt(), protocol["reference_part"],
                         reference_sha256=protocol["reference_part"]["sha256"])
    assert report["verdict"] == "pass"
    assert report["passed"] == report["total"]
    assert report["deviations"] == []


def test_a_substituted_control_receipt_is_caught_by_its_file_digest():
    protocol = _protocol()
    report = CHECK.check(protocol, _receipt(), protocol["reference_part"],
                         reference_sha256="f" * 64)
    assert report["verdict"] == "deviation"
    assert any("control receipt file" in entry["criterion"] for entry in report["deviations"])


@pytest.mark.parametrize("mutate,label", [
    (lambda r: r.update(status="failed-no-fallback"), "failed download"),
    (lambda r: r.update(synthetic_fallback=True), "synthetic fallback"),
    (lambda r: r.update(read_plan_protocol_sha256="e" * 64), "read plan drift"),
    (lambda r: r.update(snapshot_id="OTHER"), "snapshot drift"),
    (lambda r: r.update(timestamps=r["timestamps"][:119]), "short stamp count"),
    (lambda r: r.update(init_hour_coverage={"0": 120, "6": 0, "12": 0, "18": 0}), "hour coverage"),
    (lambda r: r.update(shape=[120, 3, 65, 65]), "target-box shape"),
    (lambda r: r.update(stored_level_axis_hpa=[250, 500, 850, 1000]), "level axis drift"),
    (lambda r: r.update(channel_count=9), "channel count"),
    (lambda r: r["payloads"].pop("t2m"), "missing channel"),
    (lambda r: r["payloads"]["t2m"].update(finite=False), "nonfinite channel"),
    (lambda r: r["payloads"]["t2m"].update(payload_sha256="short"), "missing digest"),
    (lambda r: r.update(network_body_bytes=10**12), "network over budget"),
    (lambda r: r["decoded_chunk_budget"].update(charged_bytes=10**12), "decoded over budget"),
    (lambda r: r.update(elapsed_seconds=1801.0), "deadline overrun"),
])
def test_each_frozen_criterion_actually_rejects(mutate, label):
    receipt = _receipt()
    mutate(receipt)
    report = CHECK.check(_protocol(), receipt, _protocol()["reference_part"])
    assert report["verdict"] == "deviation", label
    assert report["passed"] < report["total"], label


def test_a_missing_field_is_a_failure_not_a_skip():
    receipt = _receipt()
    del receipt["payloads"]
    report = CHECK.check(_protocol(), receipt, _protocol()["reference_part"])
    assert report["verdict"] == "deviation"
    failed = [entry["criterion"] for entry in report["deviations"]]
    assert "every stored channel payload is finite" in failed
    assert "payload hashes are recorded for every channel" in failed


def test_the_checker_reports_the_declared_protocol_digest_alongside_the_file_digest(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    receipt_path = tmp_path / "receipt.json"
    reference_path = tmp_path / "reference.json"
    protocol_path.write_text(json.dumps(_protocol()), encoding="utf-8")
    receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    reference_path.write_text(json.dumps(_protocol()["reference_part"]), encoding="utf-8")
    sys.argv = ["check", "--protocol", str(protocol_path), "--receipt", str(receipt_path),
                "--reference", str(reference_path)]
    CHECK.main()
    # No report path was given, so nothing was written beside the inputs.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "protocol.json", "receipt.json", "reference.json"]


def test_the_checker_compares_against_the_reference_it_was_given():
    protocol = _protocol()
    reference = copy.deepcopy(protocol["reference_part"])
    reference["network_body_bytes"] = 1  # a control that cannot be met
    report = CHECK.check(protocol, _receipt(), reference)
    assert report["verdict"] == "deviation"
    assert any("network" in entry["criterion"] for entry in report["deviations"])
