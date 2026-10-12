"""Unit tests for the multi-card device selection (decision 0047).

No GPU, no ``nvidia-smi``: every probe is injected, so the arithmetic and the selection
order are pinned without a device.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20
GIB = 2**30


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def devices():
    return _load("r7_device_selection_under_test", "training/r7_device_selection.py")


def _probe(cards):
    """A fake ``nvidia-smi``: ``(uuid, free_mib, total_mib)`` rows."""

    def probe(argv):
        assert argv[0] == "nvidia-smi"
        body = "".join(f"{uuid}, {free}, {total}\n" for uuid, free, total in cards)
        return {"exit_code": 0, "stdout": body, "stderr": ""}

    return probe


def test_seed_order_rotates_over_the_frozen_list(devices):
    uuids = ["GPU-a", "GPU-b"]
    assert devices.seed_order(uuids, 0) == ["GPU-a", "GPU-b"]
    assert devices.seed_order(uuids, 1) == ["GPU-b", "GPU-a"]
    assert devices.seed_order(uuids, 2) == ["GPU-a", "GPU-b"]
    assert devices.seed_order(["GPU-a"], 5) == ["GPU-a"]
    with pytest.raises(ValueError, match="empty"):
        devices.seed_order([], 0)


def test_the_gate_requires_footprint_plus_margin_plus_the_neighbour_reserve(devices):
    probe = _probe([("GPU-a", 9000, 24576), ("GPU-b", 30000, 24576)])
    gate = devices.gate_candidates(["GPU-a", "GPU-b"], estimated_peak_bytes=8704 * MIB,
                                   margin_bytes=512 * MIB, reserve_bytes=6 * GIB, runner=probe)
    assert gate["required_bytes"] // MIB == 15360  # 8704 + 512 + 6144
    assert gate["uuid"] == "GPU-b" and gate["passed"] is True
    assert [row["uuid"] for row in gate["observations"]] == ["GPU-a", "GPU-b"]
    assert gate["observations"][0]["passed"] is False


def test_a_short_card_is_reported_with_its_numbers_instead_of_only_blocking(devices):
    probe = _probe([("GPU-a", 100, 24576), ("GPU-b", 200, 24576)])
    gate = devices.gate_candidates(["GPU-a", "GPU-b"], estimated_peak_bytes=GIB, margin_bytes=GIB,
                                   reserve_bytes=6 * GIB, runner=probe)
    assert gate["passed"] is False and gate["uuid"] is None
    assert [row["free_bytes"] // MIB for row in gate["observations"]] == [100, 200]


def test_a_measured_peak_larger_than_the_estimate_wins(devices):
    probe = _probe([("GPU-a", 20000, 24576)])
    gate = devices.gate_candidates(["GPU-a"], estimated_peak_bytes=GIB,
                                   owned_reserved_peak_bytes=12000 * MIB,
                                   margin_bytes=GIB, reserve_bytes=0, runner=probe)
    assert gate["required_bytes"] // MIB == 13024  # max(1024, 12000) + 1024
    assert gate["passed"] is True


def test_the_gate_is_unavailable_without_nvidia_smi(devices):
    dead = lambda argv: {"exit_code": 127, "stdout": "", "stderr": "nope"}  # noqa: E731
    gate = devices.gate_candidates(["GPU-a"], estimated_peak_bytes=1, margin_bytes=1,
                                   reserve_bytes=1, runner=dead)
    assert gate["available"] is False and gate["passed"] is False and gate["uuid"] is None


def test_protocol_fields_single_names_only_the_declared_card(devices):
    fields = devices.protocol_fields(devices.MODE_SINGLE, single_uuid="GPU-a")
    assert fields == {"device_mode": "single", "gpu_uuids": ["GPU-a"],
                      "neighbour_reserve_bytes": devices.NEIGHBOUR_RESERVE_BYTES}
    assert devices.NEIGHBOUR_RESERVE_BYTES == 6 * GIB


def test_protocol_fields_dual_lists_both_cards_with_the_declared_one_first(devices):
    probe = _probe([("GPU-b", 1, 24576), ("GPU-a", 1, 24576)])
    fields = devices.protocol_fields(devices.MODE_DUAL, single_uuid="GPU-a", runner=probe)
    assert fields["device_mode"] == "dual" and fields["gpu_uuids"] == ["GPU-a", "GPU-b"]


def test_protocol_fields_rejects_an_unknown_mode(devices):
    with pytest.raises(ValueError, match="device mode"):
        devices.protocol_fields("triple", single_uuid="GPU-a")


def test_protocol_fields_refuses_a_card_nvidia_smi_cannot_see(devices):
    probe = _probe([("GPU-b", 1, 24576)])
    with pytest.raises(RuntimeError, match="not visible"):
        devices.protocol_fields(devices.MODE_DUAL, single_uuid="GPU-a", runner=probe)


def test_assignment_keeps_the_legacy_rule_when_there_is_no_list(devices):
    assert devices.assignment_is_valid({"gpu_uuid": "GPU-a"}, "GPU-a") is True
    assert devices.assignment_is_valid({"gpu_uuid": "GPU-b"}, "GPU-a") is False


def test_assignment_requires_the_declared_card_to_head_the_frozen_list(devices):
    assert devices.assignment_is_valid(
        {"gpu_uuid": "GPU-a", "gpu_uuids": ["GPU-a", "GPU-b"]}, "GPU-a") is True
    assert devices.assignment_is_valid(
        {"gpu_uuid": "GPU-a", "gpu_uuids": ["GPU-b", "GPU-a"]}, "GPU-a") is False
    assert devices.assignment_is_valid(
        {"gpu_uuid": "GPU-a", "gpu_uuids": ["GPU-b"]}, "GPU-a") is False


def test_assigned_uuid_uses_the_parent_pin_and_refuses_a_card_outside_the_freeze(devices, monkeypatch):
    body = {"gpu_uuid": "GPU-a", "gpu_uuids": ["GPU-a", "GPU-b"]}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    assert devices.assigned_uuid(body) == "GPU-b"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c")
    with pytest.raises(RuntimeError, match="not in the frozen device list"):
        devices.assigned_uuid(body)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert devices.assigned_uuid(body) == "GPU-a"


def test_frozen_uuids_falls_back_to_the_legacy_single_card(devices):
    assert devices.frozen_uuids({"gpu_uuid": "GPU-a"}) == ["GPU-a"]
    assert devices.frozen_uuids(
        {"gpu_uuid": "GPU-a", "gpu_uuids": ["GPU-a", "GPU-b"]}) == ["GPU-a", "GPU-b"]
    assert devices.frozen_uuids({}) == []
