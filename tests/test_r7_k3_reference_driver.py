"""Fake process/GPU receipts test orchestration; no NVIDIA invocation or CUDA initialization."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import types

from test_r7_k3_reference_support import no_actual_runtime
import pytest
import r7_k3_reference_driver as driver
import r7_k3_reference_worker as worker
from r7_k3_reference_support import GPU_UUID, LIMITATIONS, clock_identity, planned_jobs, read_json, sha256_file, write_json


class FakeClock:
    def __init__(self, start=0): self.value = float(start)
    def __call__(self): return self.value
    def advance(self, amount): self.value += amount


class FakeProcess:
    pid = 987654321
    def __init__(self, clock, *, duration=1, code=0, hanging=False):
        self.clock, self.duration, self.code, self.hanging = clock, duration, code, hanging
        self.exited, self.terminated, self.killed, self.waits = False, 0, 0, 0
    def poll(self): return self.code if self.exited else None
    def wait(self, timeout):
        self.waits += 1
        if self.hanging and not self.terminated:
            self.clock.advance(timeout)
            raise subprocess.TimeoutExpired("synthetic", timeout)
        self.clock.advance(min(self.duration, timeout))
        self.exited = True
        return self.code
    def terminate(self): self.terminated += 1
    def kill(self): self.killed += 1


def protocol(output):
    return {"output": str(output), "protocol_sha256": "synthetic_protocol", "hard_cap_seconds": 3600,
            "cleanup_reserve_seconds": 10, "clock": clock_identity(0.),
            "gpu": {"uuid": GPU_UUID, "estimated_peak_mib": 342, "headroom_margin_mib": 2048},
            "code": {"source_sha256": "synthetic_source"}, "archive": {"files": {}}}


def inventory(free=10000, uuid=GPU_UUID):
    return {"uuid": uuid, "free_mib": free, "neighbors": [{"pid": 12345, "used_mib": "5000"}], "read_only": True}


def fake_verifiers(monkeypatch):
    for name in ("verify_inputs", "verify_extraction", "verify_runner"):
        monkeypatch.setattr(driver, name, lambda *a, **k: None)
    monkeypatch.setattr(worker, "verify_companion", lambda *a, **k: None)


@pytest.mark.parametrize("free,observed,allowed", [(2390, 0, True), (2389, 0, False), (3072, 1024, True), (3071, 1024, False)])
def test_shared_headroom_uses_max_observed_plus2048(free, observed, allowed):
    gpu = protocol(Path("/tmp/synthetic"))["gpu"]
    if allowed:
        result = driver.verify_headroom(inventory(free), gpu, observed)
        assert result["required_free_mib"] == max(342, observed) + 2048
        assert result["neighbors"][0]["pid"] == 12345
    else:
        with pytest.raises(ValueError, match="headroom"): driver.verify_headroom(inventory(free), gpu, observed)


def test_gpu_inventory_is_read_only_fixed_uuid_no_fallback():
    calls = []
    def query(args, **kwargs):
        calls.append(args)
        if "--query-gpu=uuid,index,memory.free,memory.total" in args:
            return f"{GPU_UUID}, 1, 3000, 24576\nGPU-other, 0, 10000, 24576\n"
        return f"{GPU_UUID}, 12345, 5000\nGPU-other, 23456, 100\n"
    snapshot = driver.gpu_snapshot(GPU_UUID, query=query)
    assert snapshot["uuid"] == GPU_UUID and snapshot["neighbors"] == [{"pid": 12345, "used_mib": "5000"}]
    assert snapshot["read_only"] is True and len(calls) == 2
    assert all(args[0] == "nvidia-smi" and "--query-" in args[1] for args in calls)
    with pytest.raises(ValueError): driver.gpu_snapshot("GPU-missing", query=query)
    with pytest.raises(ValueError): driver.verify_headroom(inventory(uuid="GPU-other"), protocol(Path("/tmp"))["gpu"])


@pytest.mark.parametrize("malformed", ["gpu", "apps"])
def test_malformed_inventory_fail_closed(malformed):
    def query(args, **kwargs):
        if malformed == "gpu" or "--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in args: return "malformed\n"
        return f"{GPU_UUID}, 1, 10000, 24576\n"
    with pytest.raises(ValueError): driver.gpu_snapshot(GPU_UUID, query=query)


class FakeCuda:
    def __init__(self, before=(0, 0), after=(0, 0), initialized=False, devices=1):
        self.before, self.after, self.initialized, self.devices = before, after, initialized, devices
        self.calls = []
    def is_initialized(self): return self.initialized
    def memory_allocated(self, d): self.calls.append("allocated"); return (self.after if self.initialized else self.before)[0]
    def memory_reserved(self, d): self.calls.append("reserved"); return (self.after if self.initialized else self.before)[1]
    def init(self): self.calls.append("init"); self.initialized = True
    def device_count(self): return self.devices
    def set_device(self, d): self.calls.append("set")
    def reset_peak_memory_stats(self, d): self.calls.append("reset_peak")


def test_fresh_allocator_0_0_before_and_after_init_no_clear():
    cuda = FakeCuda()
    fake = types.SimpleNamespace(cuda=cuda, device=lambda d: d)
    result = worker.zero_allocator_baseline(fake)
    assert result["pre_init"] == result["initialized"] == {"allocated_bytes": 0, "reserved_bytes": 0}
    assert cuda.calls == ["allocated", "reserved", "init", "set", "allocated", "reserved", "reset_peak"]
    assert result["pre_init_is_initialized"] is False


@pytest.mark.parametrize("kwargs", [{"before": (1, 0)}, {"before": (0, 1)}, {"after": (0, 1)}, {"initialized": True}, {"devices": 2}])
def test_allocator_nonfresh_or_nonzero_or_multiple_devices_refused(kwargs):
    cuda = FakeCuda(**kwargs)
    fake = types.SimpleNamespace(cuda=cuda, device=lambda d: d)
    with pytest.raises(ValueError): worker.zero_allocator_baseline(fake)
    assert "reset_peak" not in cuda.calls


def test_worker_environment_no_active_pythonpath_and_command_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    env = driver.worker_environment()
    assert env["PYTHONPATH"] == "" and env["CUDA_VISIBLE_DEVICES"] == GPU_UUID and env["PYTHONDONTWRITEBYTECODE"] == "1"
    command = driver.worker_command(protocol(tmp_path), {"seed": 42, "lead": 72}, 1234.)
    assert "archive_worker" in command and "--deadline" in command and command[-1] == "1234.0"
    assert str(tmp_path / "companion_code/r7_k3_reference.py") in command
    assert "--repo" not in command and "--output" not in command


def test_prepare_freezes_before_extraction_import_spawn_and_no_runtime_data(monkeypatch, tmp_path):
    clock, events = FakeClock(12), []
    out = tmp_path / "fresh"
    p = protocol(out)
    p.update(repo=str(tmp_path / "repo"), archive={"path": "synthetic.zip", "files": {}}, limitations=LIMITATIONS)
    def build(*args, **kwargs): events.append("build_bytes"); clock.advance(3); return p
    def archive(protocol):
        assert (out / "protocol.json").is_file()
        events.append("archive_companion_after_freeze")
    def extract(*args, **kwargs):
        assert (out / "protocol.json").is_file()
        events.append("extract_after_freeze")
        clock.advance(4)
    monkeypatch.setattr(driver, "build_protocol", build)
    monkeypatch.setattr(driver, "archive_companion", archive)
    monkeypatch.setattr(driver, "extract_archive", extract)
    monkeypatch.setattr(driver, "verify_inputs", lambda *a, **k: events.append("verify_bytes"))
    result = driver.prepare(tmp_path / "repo", out, 12., clock=clock)
    assert result == p and events == ["build_bytes", "archive_companion_after_freeze", "extract_after_freeze", "verify_bytes"]
    receipt = read_json(out / "prepare_attempt.json")
    assert receipt["status"] == "prepared-not-run" and receipt["whole_elapsed_seconds"] == 7
    assert receipt["weather_samples_read"] == receipt["checkpoint_deserializations"] == receipt["cuda_jobs"] == 0
    with pytest.raises(FileExistsError): driver.prepare(tmp_path / "repo", out, 12., clock=clock)


def test_owned_job_before_each_spawn_and_record_reap(monkeypatch, tmp_path):
    (tmp_path / "workers").mkdir()
    clock, events, processes = FakeClock(20), [], []
    p, job = protocol(tmp_path), {"seed": 41, "lead": 6}
    receipt = {"status": "success", "peak_reserved_bytes": 500}
    write_json(tmp_path / "workers/seed41_lead006h.json", receipt)
    monkeypatch.setattr(driver, "verify_worker", lambda *a, **k: None)
    def snapshot(uuid): events.append("headroom"); clock.advance(1); return inventory()
    def spawn(command, **kwargs):
        events.append("spawn")
        assert kwargs["cwd"] == tmp_path / "archived_code" and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU_UUID
        process = FakeProcess(clock)
        processes.append(process)
        return process
    assert driver.run_job(p, job, 100., 0., clock=clock, snapshot_fn=snapshot, popen_factory=spawn) == receipt
    timing = read_json(tmp_path / "workers/seed41_lead006h.timing.json")
    assert events == ["headroom", "spawn"] and timing["status"] == "success"
    assert timing["cleanup"] == "already-exited-owned-worker" and processes[0].terminated == 0


@pytest.mark.parametrize("failure", ["headroom", "exit", "timeout", "receipt"])
def test_any_owned_job_failure_stops_without_retry_and_preserves_timing(monkeypatch, tmp_path, failure):
    (tmp_path / "workers").mkdir()
    clock, processes = FakeClock(20), []
    def spawn(*args, **kwargs):
        process = FakeProcess(clock, code=7 if failure == "exit" else 0, hanging=failure == "timeout")
        processes.append(process)
        return process
    snapshot = lambda uuid: inventory(0 if failure == "headroom" else 10000)
    with pytest.raises((ValueError, TimeoutError, FileNotFoundError)):
        driver.run_job(protocol(tmp_path), {"seed": 41, "lead": 6}, 100., 0., clock=clock, snapshot_fn=snapshot, popen_factory=spawn)
    timing = read_json(tmp_path / "workers/seed41_lead006h.timing.json")
    assert timing["status"] == "failed" and "failure_reason" in timing
    assert len(processes) == (0 if failure == "headroom" else 1)
    if failure == "timeout":
        assert processes[0].terminated == 1 and processes[0].exited
        assert timing["budget_limited"] is True


def test_run_all_continuous_earliest_clock_soft_overrun_allowed_and_observed_peak(monkeypatch, tmp_path):
    clock = FakeClock(1000.)  # Preparation already consumed1000s from original clock0.
    fake_verifiers(monkeypatch)
    seen = []
    def job(p, j, deadline, observed_peak, **kwargs):
        seen.append((j, deadline, observed_peak))
        clock.advance(100)
        return {"peak_reserved_bytes": (500 + len(seen) * 100) * 2**20}
    monkeypatch.setattr(driver, "run_job", job)
    def finalize(p, completed, **kwargs):
        clock.advance(11)
        write_json(tmp_path / "reference_result.json", {"synthetic": True})
        return {"synthetic": True}
    monkeypatch.setattr(driver, "finalize", finalize)
    result = driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: inventory())
    attempt = read_json(tmp_path / "attempt.json")
    assert result == {"synthetic": True} and attempt["status"] == "success"
    assert attempt["whole_elapsed_seconds"] == 2011 and attempt["soft_overrun_seconds"] == 211
    assert attempt["gpu_phase_elapsed_seconds"] == 1000 and attempt["gpu_hours_charged"] == 1000 / 3600
    assert [r[0] for r in seen] == planned_jobs() and all(r[1] == 3600 for r in seen)
    assert [r[2] for r in seen] == [0., *range(600, 1500, 100)]
    assert attempt["jobs_completed"] == planned_jobs() and attempt["finalized"] is True
    with pytest.raises(ValueError): driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: inventory())


def test_whole_hard_deadline_includes_preparation_without_clock_reset(monkeypatch, tmp_path):
    clock = FakeClock(3601)
    fake_verifiers(monkeypatch)
    monkeypatch.setattr(driver, "run_job", lambda *a, **k: pytest.fail("hard-expired round spawned"))
    with pytest.raises(TimeoutError): driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: pytest.fail("hard-expired inventory"))
    attempt = read_json(tmp_path / "attempt.json")
    assert attempt["status"] == "failed" and attempt["budget_limited"] is True and not attempt["jobs_completed"]
    assert attempt["whole_elapsed_seconds"] == 3601 and attempt["gpu_hours_charged"] == 0


def test_first_failure_preserved_all_completed_jobs_and_no_retry(monkeypatch, tmp_path):
    clock, calls = FakeClock(0), []
    fake_verifiers(monkeypatch)
    def run(p, job, *args, **kwargs):
        calls.append(job)
        clock.advance(2)
        if len(calls) == 3: raise RuntimeError("original synthetic failure")
        return {"peak_reserved_bytes": 2**20}
    monkeypatch.setattr(driver, "run_job", run)
    monkeypatch.setattr(driver, "finalize", lambda *a, **k: pytest.fail("failed subset finalized"))
    with pytest.raises(RuntimeError, match="original synthetic failure"):
        driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: inventory())
    attempt = read_json(tmp_path / "attempt.json")
    assert calls == planned_jobs()[:3] and attempt["jobs_completed"] == planned_jobs()[:2]
    assert attempt["status"] == "failed" and attempt["partial"] is True and attempt["finalized"] is False
    assert attempt["whole_elapsed_seconds"] == attempt["gpu_phase_elapsed_seconds"] == 6


def test_publication_secondary_error_never_replaces_original_failure(monkeypatch, tmp_path):
    clock = FakeClock(0)
    fake_verifiers(monkeypatch)
    monkeypatch.setattr(driver, "run_job", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("primary")))
    original = driver.write_json
    def write(path, value):
        if Path(path).name == "attempt.json": raise OSError("publication")
        return original(path, value)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(RuntimeError, match="primary") as exc:
        driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: inventory())
    assert any("publication" in note for note in exc.value.__notes__)


def test_finalize_refuses_missing_jobs_before_reading_any_results(tmp_path):
    with pytest.raises(ValueError, match="all exact10"):
        driver.finalize(protocol(tmp_path), planned_jobs()[:-1])
    assert not (tmp_path / "reference_result.json").exists() and not (tmp_path / "metrics.csv").exists()


def test_worker_uuid_refused_before_project_or_torch_import(tmp_path, monkeypatch):
    p = protocol(tmp_path)
    monkeypatch.setattr(worker, "read_protocol", lambda path: p)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    monkeypatch.setattr(worker, "check_deadline", lambda deadline: None)  # Synthetic boot anchor0, no real clock dependency.
    with pytest.raises(ValueError, match="exact fixed"):
        worker.run_worker(tmp_path / "protocol.json", {"seed": 41, "lead": 6}, 3590.)
    assert not (tmp_path / "workers/seed41_lead006h.json").exists()


def test_frozen_companion_archive_and_source_receipt_tamper_refused(tmp_path):
    from r7_k3_reference_identity import runner_identity
    source = Path(__file__).resolve().parents[1] / "tools"
    p = protocol(tmp_path)
    p["code"] = runner_identity(source)
    driver.archive_companion(p)
    worker.verify_companion(p, tmp_path / "companion_code")
    path = tmp_path / "companion_code.zip"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archive bytes changed"):
        worker.verify_companion(p, tmp_path / "companion_code")


def test_prepare_identity_failure_sealed_before_spawn_and_existing_output_not_resurrected(monkeypatch, tmp_path):
    clock = FakeClock(0)
    output = tmp_path / "failed_prepare"
    monkeypatch.setattr(driver, "build_protocol", lambda *a, **k: (_ for _ in ()).throw(ValueError("tampered original bytes")))
    monkeypatch.setattr(driver, "archive_companion", lambda *a, **k: pytest.fail("failed qualification archived"))
    with pytest.raises(ValueError, match="tampered original bytes"):
        driver.prepare(tmp_path / "repo", output, 0., clock=clock)
    seal = read_json(output / "failed_attempt_seal.json")
    assert seal["status"] == "failed" and seal["finalized"] is False
    assert read_json(output / "prepare_attempt.json")["status"] == "failed"
    with pytest.raises(FileExistsError): driver.prepare(tmp_path / "repo", output, 0., clock=clock)


def test_byte_identity_failure_sealed_before_admission_and_spawn(monkeypatch, tmp_path):
    clock = FakeClock(0)
    fake_verifiers(monkeypatch)
    monkeypatch.setattr(driver, "verify_inputs", lambda *a, **k: (_ for _ in ()).throw(ValueError("tampered frozen input")))
    monkeypatch.setattr(driver, "run_job", lambda *a, **k: pytest.fail("tampered input spawned"))
    with pytest.raises(ValueError, match="tampered frozen input"):
        driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: pytest.fail("tampered input admitted"))
    seal = read_json(tmp_path / "failed_attempt_seal.json")
    attempt = read_json(tmp_path / "attempt.json")
    assert seal["status"] == attempt["status"] == "failed" and attempt["finalized"] is False
    assert attempt["jobs_completed"] == [] and attempt["gpu_hours_charged"] == 0
    with pytest.raises(ValueError): driver.run_all(protocol(tmp_path), 0., clock=clock)


@pytest.mark.parametrize("late_phase", ["final_result_hash", "attempt_publication"])
def test_hard_cap_during_last_hash_or_publication_cannot_be_accepted(monkeypatch, tmp_path, late_phase):
    clock = FakeClock(0)
    fake_verifiers(monkeypatch)
    monkeypatch.setattr(driver, "run_job", lambda *a, **k: {"peak_reserved_bytes": 0})
    def finalize(*args, **kwargs):
        write_json(tmp_path / "reference_result.json", {"status": "complete-not-accepted"})
        clock.advance(3599)
        return {"status": "complete-not-accepted"}
    monkeypatch.setattr(driver, "finalize", finalize)
    original_hash, original_write = driver.sha256_file, driver.write_json
    def late_hash(path, **kwargs):
        value = original_hash(path)
        if Path(path).name == "reference_result.json" and late_phase == "final_result_hash": clock.advance(2)
        return value
    def late_write(path, value):
        original_write(path, value)
        if Path(path).name == "attempt.json" and late_phase == "attempt_publication": clock.advance(2)
    monkeypatch.setattr(driver, "sha256_file", late_hash)
    monkeypatch.setattr(driver, "write_json", late_write)
    with pytest.raises(TimeoutError): driver.run_all(protocol(tmp_path), 0., clock=clock, snapshot_fn=lambda uuid: inventory())
    seal = read_json(tmp_path / "failed_attempt_seal.json")
    assert seal["status"] == "failed" and seal["finalized"] is False
    assert read_json(tmp_path / "reference_result.json")["status"] == "complete-not-accepted"
    if late_phase == "final_result_hash":
        assert read_json(tmp_path / "attempt.json")["finalized"] is False
    with pytest.raises(ValueError): driver.run_all(protocol(tmp_path), 0., clock=clock)


@pytest.mark.parametrize("mutation", ["reset_anchor", "cross_boot"])
def test_frozen_clock_reset_or_crossboot_sealed_before_input_or_spawn(monkeypatch, tmp_path, mutation):
    p = protocol(tmp_path)
    started = 1. if mutation == "reset_anchor" else 0.
    if mutation == "cross_boot": p["clock"]["boot_id"] = "00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(driver, "verify_inputs", lambda *a, **k: pytest.fail("invalid clock touched inputs"))
    with pytest.raises(ValueError):
        driver.run_all(p, started, clock=FakeClock(2), snapshot_fn=lambda uuid: pytest.fail("invalid clock admitted CUDA"))
    assert read_json(tmp_path / "failed_attempt_seal.json")["finalized"] is False
    assert read_json(tmp_path / "attempt.json")["jobs_completed"] == []


def test_worker_deadline_cannot_be_reset_from_frozen_round(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "read_protocol", lambda path: protocol(tmp_path))
    monkeypatch.setattr(worker, "check_deadline", lambda deadline: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    with pytest.raises(ValueError, match="exact frozen"):
        worker.run_worker(tmp_path / "protocol.json", {"seed": 41, "lead": 6}, 99999.)
    assert not (tmp_path / "workers/seed41_lead006h.json").exists()
