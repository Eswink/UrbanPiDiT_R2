"""CPU-only M3 order, allocator initialization and original-error counterproofs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from training import r7_m3_driver as driver
from training import r7_m3_protocol as protocol
from training import r7_m3_results as results
from training import r7_m3_worker as worker
from test_r7_m3_driver import _authorization, _fake_round, _profile_fixture, frozen


def _raise(error):
    raise error


def test_original_child_failure_survives_cleanup_and_receipt_failure(frozen, monkeypatch):
    clock, spawned, _, processes, popen, snapshot = _fake_round(frozen, fail_at=6)
    original_reap, original_write = driver.reap_owned, driver.write_json
    def reap(process, *args, **kwargs):
        if process.code:
            raise RuntimeError("owned reap fixture failure")
        return original_reap(process, *args, **kwargs)
    def write(path, value):
        if path.name.endswith(".timing.json") and value["job"]["phase"] == "evaluate":
            raise OSError("receipt fixture failure")
        return original_write(path, value)
    monkeypatch.setattr(driver, "reap_owned", reap)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(RuntimeError, match="failed with exit 2") as caught:
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    assert len(spawned) == 7
    assert all(not process.signals for process in processes)
    assert any("receipt fixture failure" in note for note in caught.value.__notes__)
    assert any("unreaped" in note for note in caught.value.__notes__)
    receipt = protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")
    assert receipt["status"] == "failed" and len(receipt["jobs_completed"]) == 6
    assert "exit 2" in receipt["failure_reason"]


def test_cleanup_failure_after_success_stops_and_never_reports_success(frozen, monkeypatch):
    clock, spawned, _, _, popen, snapshot = _fake_round(frozen)
    monkeypatch.setattr(driver, "reap_owned", lambda *a, **k: _raise(RuntimeError("unreaped fixture")))
    with pytest.raises(RuntimeError, match="unreaped fixture"):
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    assert len(spawned) == 1
    timing = protocol.read_json(Path(frozen["output"]) / "workers" / (protocol.job_key(spawned[0]) + ".timing.json"))
    assert timing["status"] == "failed-unreaped"
    assert protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")["status"] == "failed"


def test_failed_execution_receipt_preserves_error_and_charged_facts(frozen, monkeypatch):
    clock, spawned, _, _, popen, snapshot = _fake_round(frozen, fail_at=0)
    original = driver.write_json
    def write(path, value):
        if path.name == "execution_attempt.json":
            raise OSError("execution receipt denied fixture")
        return original(path, value)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(RuntimeError, match="exit 2") as caught:
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    assert len(spawned) == 1
    assert caught.value.m3_execution_attempt["gpu_phase_elapsed_seconds"] == 1.25
    assert any("execution receipt denied" in note for note in caught.value.__notes__)


def test_original_worker_failure_survives_result_publication_failure(frozen, monkeypatch):
    monkeypatch.setattr(worker, "deny_network", lambda: None)
    monkeypatch.setattr(worker, "verify_code", lambda _: _raise(ValueError("original worker identity failure")))
    monkeypatch.setattr(worker, "write_json", lambda *a: _raise(OSError("result publication denied")))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixed")
    with pytest.raises(ValueError, match="original worker identity") as caught:
        worker.run_worker(Path(frozen["output"]) / "protocol.json", protocol.planned_jobs()[0], float("inf"))
    assert any("result publication denied" in note for note in caught.value.__notes__)


def test_eval_peak_reset_requires_init_and_rejects_postinit_residual(monkeypatch):
    initialized, calls = [False], []
    def allocated(device):
        calls.append(("read", initialized[0]))
        return 0
    def initialize():
        initialized[0] = True
        calls.append(("init", None))
    def select(device):
        assert initialized[0]
        calls.append(("select", device))
    def reset(device):
        assert initialized[0]
        calls.append(("reset", device))
    monkeypatch.setattr(torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", allocated)
    monkeypatch.setattr(torch.cuda, "init", initialize)
    monkeypatch.setattr(torch.cuda, "set_device", select)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    assert worker.zero_allocator_baseline("cuda:0") == {"allocated_bytes": 0, "reserved_bytes": 0}
    assert calls[:2] == [("read", False), ("read", False)]
    assert calls[2:4] == [("init", None), ("select", "cuda:0")]
    assert calls[-1] == ("reset", "cuda:0")
    initialized[0] = False
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 1 if initialized[0] else 0)
    with pytest.raises(ValueError, match="initialized.*zero"):
        worker.zero_allocator_baseline("cuda:0")


def test_cpu_prepare_costs_before_protocol_and_source_reverified_before_freeze(tmp_path, monkeypatch):
    authorization = tmp_path / "auth.json"
    _authorization(authorization)
    output = tmp_path / "out"
    data = {"channels": [f"channel{i}" for i in range(17)], "units": ["physical"] * 17,
            "data_identity": "a" * 64}
    events = []
    measurement = {"parameters": 1, "trainable_parameters": 1, "forward_flops": 1, "forward_backward_flops": 2}
    monkeypatch.setattr(driver, "dataset_pins", lambda *_: (data, object(), object()))
    monkeypatch.setattr(driver, "source_identity", lambda *_: {"source_sha256": "b" * 64})
    monkeypatch.setattr(driver, "sidecar_pins", lambda *a: ({"path": "sidecar", "identity": "c" * 64}, {}))
    monkeypatch.setattr(worker, "make_context", lambda *a: object())
    monkeypatch.setattr(worker, "deny_network", lambda: None)
    monkeypatch.setattr(worker, "M3Dataset", lambda *a, **k: [{"coarse_history": torch.zeros(2, 17, 2, 2)}] * 2)
    import training.r7_m3_profile as profile
    def costs(*a):
        assert not (output / "protocol.json").exists()
        events.append("costs")
        return _profile_fixture()
    monkeypatch.setattr(profile, "profile_arms", costs)
    monkeypatch.setattr(driver, "archive_code", lambda *_: {"model_code_sha256": "d" * 64})
    monkeypatch.setattr(driver, "verify_input_pins", lambda _: events.append("pins-after-costs"))
    result = driver.prepare(tmp_path / "inputs", tmp_path / "sidecar", output, authorization, "GPU-fixed")
    assert events == ["costs", "pins-after-costs"]
    assert protocol.verify_protocol(output / "protocol.json") == result
    assert protocol.read_json(output / "prepare_attempt.json")["status"] == "prepared-not-run"
    assert not (output / "attempt.json").exists()
    assert not (output / "run_started.json").exists()


def test_postgpu_source_failure_blocks_finalization_and_retains_charge(frozen, monkeypatch, tmp_path):
    output = Path(frozen["output"])
    (output / "workers").rmdir()
    auth = tmp_path / "auth.json"
    authorization = _authorization(auth)
    value = deepcopy(frozen)
    value["authorization"] = authorization
    value["authorization_sha256"] = driver.sha256_file(auth)
    value["protocol_sha256"] = protocol.digest({k: v for k, v in value.items() if k != "protocol_sha256"})
    (output / "protocol.json").write_text(__import__("json").dumps(value), encoding="utf-8")
    checks = []
    def pins(_):
        checks.append("pin")
        if len(checks) == 2:
            raise ValueError("source changed after GPU fixture")
    execution = {"status": "success", "jobs_completed": protocol.planned_jobs(), "gpu_phase_elapsed_seconds": 12.,
                 "gpu_hours_charged": 12 / 3600, "budget_limited": False}
    monkeypatch.setattr(driver, "verify_code", lambda _: None)
    monkeypatch.setattr(driver, "verify_input_pins", pins)
    monkeypatch.setattr(driver, "execute_jobs", lambda _: execution)
    monkeypatch.setattr(results, "finalize", lambda *a: pytest.fail("changed source must block CPU finalize"))
    with pytest.raises(ValueError, match="source changed"):
        driver.run_bounded_round(output, auth)
    receipt = protocol.read_json(output / "attempt.json")
    assert receipt["status"] == "failed" and not receipt["finalized"]
    assert receipt["gpu_hours_charged"] == 12 / 3600
    assert checks == ["pin", "pin"]


@pytest.mark.parametrize("failure", ["arm", "seed", "hash", "component", "norm", "flops"])
def test_frozen_profile_requires_all_arms_seeds_components_and_measured_costs(frozen, failure):
    value = deepcopy(frozen)
    profile = value["cpu_profile"]
    if failure == "arm":
        profile["gradient_ownership"].pop("input_aux")
    elif failure == "seed":
        profile["pairing"].pop("42")
    elif failure == "hash":
        profile["pairing"]["41"]["full_initial_state_sha256"]["input_aux"] = "f" * 64
    elif failure == "component":
        profile["gradient_ownership"]["aux_off"]["components"].pop("draft")
    elif failure == "norm":
        profile["gradient_ownership"]["aux_off"]["components"]["total"]["groups"]["readout"]["norm"] = -1
    else:
        profile["measurements"]["input_aux"]["forward_flops"] = 1
    value["protocol_sha256"] = protocol.digest({k: v for k, v in value.items() if k != "protocol_sha256"})
    with pytest.raises(ValueError):
        protocol.validate_protocol(value)


def test_original_round_failure_survives_attempt_receipt_denial(frozen, tmp_path, monkeypatch):
    output = Path(frozen["output"])
    auth = tmp_path / "auth.json"
    _authorization(auth)
    original = driver.write_json
    monkeypatch.setattr(driver, "verify_code", lambda _: _raise(ValueError("original round code mismatch")))
    def write(path, value):
        if path.name == "attempt.json":
            raise OSError("attempt receipt denied")
        return original(path, value)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(ValueError, match="authorization or output") as caught:
        driver.run_bounded_round(output, auth)
    assert any("attempt receipt denied" in note for note in caught.value.__notes__)
    assert (output / "run_started.json").exists()


def test_secondary_stderr_failure_cannot_mask_original_exception(monkeypatch):
    import sys
    class FailedStream:
        def write(self, value):
            raise OSError("secondary stderr ENOSPC fixture")
        def flush(self):
            raise OSError("secondary stderr flush fixture")
    original = ValueError("original worker failure")
    monkeypatch.setattr(sys, "stderr", FailedStream())
    protocol.preserve_artifact_error(original, OSError("receipt denied"), label="publication")
    assert any("receipt denied" in note for note in original.__notes__)
    assert any("secondary stderr ENOSPC" in note for note in original.__notes__)
    with pytest.raises(ValueError, match="original worker failure"):
        raise original
    with pytest.raises(OSError, match="receipt denied without original"):
        protocol.preserve_artifact_error(None, OSError("receipt denied without original"), label="publication")


def test_overbudget_or_partial_finalize_never_reads_result_artifacts(frozen, monkeypatch):
    monkeypatch.setattr(results, "validate_full_set", lambda *a: pytest.fail("invalid execution must stop first"))
    with pytest.raises(ValueError, match="over-budget"):
        results.finalize(frozen["output"], frozen,
                         {"status": "success", "jobs_completed": protocol.planned_jobs(), "gpu_phase_elapsed_seconds": 1801})
    with pytest.raises(ValueError, match="partial"):
        results.finalize(frozen["output"], frozen,
                         {"status": "failed", "jobs_completed": [], "gpu_phase_elapsed_seconds": 1})
