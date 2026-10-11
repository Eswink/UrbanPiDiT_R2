"""Cross-attempt seed reuse and the GPU co-residency probe.

Split from ``tests/test_r7_round.py`` to stay under the 400-line report threshold (R-021).
The reuse gate is decision 0045: every whitelisted field must match exactly, so most of
these tests are counterproofs that a single drifted field refuses reuse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

UUID = "GPU-00000000-0000-0000-0000-000000000000"
GPU_CSV = f"{UUID}, 30000, 24576" + chr(10)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def support():
    return _load("r7_round_support_reuse", "tools/r7_round_support.py")


def _ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}


def _fail(code: int = 1, stderr: str = "boom") -> dict:
    return {"exit_code": code, "stdout": "", "stderr": stderr}


def test_gpu_headroom_reports_unavailable_when_the_probe_fails(support):
    record = support.gpu_headroom(UUID, estimated_peak_bytes=2**31, margin_bytes=2**31,
                                  runner=lambda argv, cwd=None: _fail(127, "no nvidia-smi"))
    assert record["available"] is False


def test_gpu_headroom_reports_unavailable_without_the_card(support):
    record = support.gpu_headroom("GPU-missing", estimated_peak_bytes=2**31, margin_bytes=2**31,
                                  runner=lambda argv, cwd=None: _ok(GPU_CSV))
    assert record["available"] is False
    assert record["passed"] is False


def test_gpu_headroom_counterproof_insufficient_memory_fails(support):
    record = support.gpu_headroom(UUID, estimated_peak_bytes=2**31, margin_bytes=2**31,
                                  runner=lambda argv, cwd=None: _ok(f"{UUID}, 1024, 24576\n"))
    assert record["passed"] is False


def test_gpu_headroom_passes_with_enough_free_memory(support):
    record = support.gpu_headroom(UUID, estimated_peak_bytes=2**31, margin_bytes=2**31,
                                  runner=lambda argv, cwd=None: _ok(GPU_CSV))
    assert record["passed"] is True
    assert record["required_bytes"] == 4096 * 2**20


def test_reuse_refuses_when_there_is_no_previous_attempt(support):
    decision = support.reuse_decision(None, _protocol(), {})
    assert decision["reusable"] == []
    assert decision["reason"] == "no previous attempt"


def test_reuse_counterproof_a_receipt_claiming_another_seed_is_refused(support):
    receipts = _receipts(41)
    receipts["41"] = {"seed": 42, "reasoning_steps": 8, "updates": 2400}
    decision = support.reuse_decision(_protocol(), _protocol(), receipts)
    assert decision["reusable"] == []
    assert "seed" in decision["refused"]["41"]


def test_reuse_refuses_a_seed_whose_receipt_disagrees(support):
    receipts = _receipts(41, 42)
    receipts["42"] = {"seed": 42, "reasoning_steps": 4, "updates": 2400}
    decision = support.reuse_decision(_protocol(), _protocol(), receipts)
    assert decision["reusable"] == ["41"]
    assert "42" in decision["refused"]


def test_reuse_counterproof_each_guarded_field_drifts_alone(support):
    for field in support.REUSE_PROTOCOL_FIELDS:
        drifted = _protocol(**{field: "drifted-value"})
        decision = support.reuse_decision(_protocol(), drifted, _receipts(41))
        assert decision["reusable"] == [], f"{field} drift was accepted"
        assert field in decision["reason"]


def test_reuse_refuses_everything_when_one_protocol_field_drifts(support):
    decision = support.reuse_decision(_protocol(), _protocol(updates=800), _receipts(41, 42))
    assert decision["reusable"] == []
    assert "updates" in decision["reason"]


def test_reuse_accepts_an_identical_protocol(support):
    decision = support.reuse_decision(_protocol(), _protocol(), _receipts(41, 42))
    assert decision["reusable"] == ["41", "42"]
    assert decision["refused"] == {}


def _receipts(*seeds) -> dict:
    return {str(seed): {"seed": seed, "reasoning_steps": 8, "updates": 2400} for seed in seeds}


def _protocol(**overrides) -> dict:
    body = {"protocol_sha256": "a" * 64, "code_zip_sha256": "b" * 64, "code_commit": "c" * 40,
            "train_data_identity": "d" * 64, "val_data_identity": "e" * 64,
            "source_inventory_sha256": "f" * 64, "supervision": {"mask": "interior_32"},
            "updates": 2400, "reasoning_steps": 8, "physical_weights": [1, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
            "lr": 2e-5, "warmup": 100, "weight_decay": 0.0, "batch_size": 8, "clip": 1.0,
            "per_seed_seconds": 18000.0, "planned_seconds": 54000.0, "hard_cap_seconds": 72000.0}
    body.update(overrides)
    return body
