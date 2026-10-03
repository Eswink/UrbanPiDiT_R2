"""CPU/fake-CUDA counterproofs only; all artifacts in tmp_path, no real samples."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import socket
import subprocess
import time
from types import SimpleNamespace

import pytest
import torch

from tools import r7_v2_precision_probe as probe

UUID = "GPU-12345678-1234-1234-1234-123456789abc"
INPUT_NAMES = ("checkpoint", "original_protocol", "codezip", "sidecar", "trainmanifest")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # Register originals with monkeypatch before deny_network changes them.
    for obj, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection")):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Clock:
    def __init__(self, value=100.):
        self.value = value

    def __call__(self):
        return self.value


class OwnedProcess:
    def __init__(self, clock, *, timed_out=False, kill_needed=False, exit_code=0):
        self.pid = 321
        self.clock, self.timed_out, self.kill_needed, self.exit_code = clock, timed_out, kill_needed, exit_code
        self.events, self.done = [], False

    def poll(self):
        return self.exit_code if self.done else None

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if self.timed_out and not self.done:
            self.clock.value += timeout
            raise subprocess.TimeoutExpired("owned-fake", timeout)
        self.done = True
        return self.exit_code

    def terminate(self):
        self.events.append(("terminate", self.pid))
        if not self.kill_needed:
            self.done = True

    def kill(self):
        self.events.append(("kill", self.pid))
        self.done = True


def fake_identity(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    inputs = {}
    for name in INPUT_NAMES:
        file = parent / ("train.jsonl" if name == "trainmanifest" else name + ".json")
        file.write_text(name)
        inputs[name] = str(file)
    return {"paths": inputs, "sha256": {n: probe.sha256_file(p) for n, p in inputs.items()},
            "model_spec": {"in_channels": 17, "out_channels": 17, "detach_between_steps": False},
            "windows": {"window_sha256": "a" * 64, "excluded_sample_ids": ["boundary"]},
            "selected_windows": [{"sample_id": "train0"}, {"sample_id": "train1"}],
            "parent_import": {"data": {"data_identity": "d" * 64, "sources": {"source_sha256": "e" * 64}},
                              "report_sha256": "f" * 64, "initialization_contract": {"model_code_sha256": "c" * 64}}}


def frozen(tmp_path, monkeypatch, clock=None):
    clock = clock or Clock()
    identity = fake_identity(tmp_path)
    monkeypatch.setattr(probe, "boot_id", lambda: "test-boot")
    output = tmp_path / "probe_attempt01"
    output.mkdir()
    body = {"format": "r7-v2-precision-probe-v1", "scientific_claim": False, "limitations": probe.LIMITATIONS,
            "test_read": False, "frozen_before_any_step": True, "output": str(output),
            "round_started_perf_counter": clock(), "monotonic_boot_id": "test-boot",
            "planned_seconds": 900., "hard_cap_seconds": 1800., "inputs": identity,
            "code": {"model_code_sha256": "c" * 64, "files": {}, "artifacts": {}, "source_tree_sha256": probe.digest({})},
            "controls": probe.CONTROLS, "precisions": ["fp32", "bf16"],
            "gpu": {"policy": "shared", "uuid": UUID, "estimated_peak_mib": 4096, "headroom_margin_mib": 2048}}
    protocol = {**body, "protocol_sha256": probe.digest(body)}
    probe.write_json(output / "protocol.json", protocol)
    monkeypatch.setattr(probe, "verify_code", lambda _p, check: check())
    monkeypatch.setattr(probe, "verify_inputs", lambda _p, check: (check(), None))
    return protocol, output, clock


def receipt(protocol, precision):
    checks = {"finite": True, "full_gradient": True, "poison_forecasts_equal": True, "poison_loss_changed": True,
              "two_step_forward_calls": 2, "one_step_forward_calls": 1, "internal_k_calls": 8,
              "calendar_trace": [{}, {}], "valid_time_phase_trace": [[], []],
              "l12_gradient_ownership": {n: [1.] for n in ("first_forecast", "first_encoder", "first_reader",
                                                        "first_internal_k", "encoder_parameters", "reader_parameters")}}
    return {"status": "success", "scientific_claim": False, "limitations": probe.LIMITATIONS,
            "test_read": False, "precision": precision, "protocol_sha256": protocol["protocol_sha256"],
            "baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "elapsed_seconds": 3.,
            "resume_exact": {n: True for n in ("weights", "optimizer", "rng", "losses")},
            "checks": checks, "input_tensor_sha256": {"history": "a" * 64},
            "model_code_sha256": protocol["code"]["model_code_sha256"], "actual_optimizer_updates": 4,
            "branches": [{"branch": branch, "updates": 2, "losses": [{}, {}]} for branch in ("uninterrupted", "intentional_resume")]}


def fake_execution(monkeypatch, protocol, output, clock, *, timed_out=False, kill_needed=False, bad=None):
    events, processes = [], []

    def snapshot(uuid):
        events.append(("snapshot", uuid))
        return {"uuid": uuid, "free_mib": 16384, "neighbors": [{"pid": 999, "used_mib": "1024"}], "read_only": True}

    def popen(command, **kwargs):
        precision = command[command.index("--worker") + 1]
        events.append(("spawn", precision, command, kwargs))
        process = OwnedProcess(clock, timed_out=timed_out, kill_needed=kill_needed)
        processes.append(process)
        if not timed_out:
            clock.value += 3.
            payload = receipt(protocol, precision)
            if bad:
                payload.update(bad)
            probe.write_json(output / "workers" / (precision + ".json"), payload)
        return process

    return snapshot, popen, events, processes


def test_socket_deny_is_installed_without_connections():
    probe.deny_network()
    with socket.socket() as connection:
        for method in (connection.connect, connection.connect_ex):
            with pytest.raises(RuntimeError, match="offline"):
                method(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("127.0.0.1", 9))


@pytest.mark.parametrize("name", ["data/raw/x", "data/interim/x", "data/processed/x", "legacy_v6/x", "model/legacy_v531/x"])
def test_output_rejects_protected_paths_before_write(name):
    with pytest.raises(ValueError, match="protected"):
        probe.output_path(probe.ROOT / name, fresh=True)


def test_output_rejects_symlink_ancestor_old_outputs_and_parent_overlap(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        probe.output_path(link / "never-created", fresh=True)
    with pytest.raises(FileExistsError, match="exclusive"):
        probe.output_path(real, fresh=True)
    with pytest.raises(ValueError, match="overlaps"):
        probe.output_path(real / "child", inputs=[real], fresh=True)
    assert not (real / "never-created").exists()
    assert probe.output_path(tmp_path / "fresh", inputs=[real], fresh=True) == tmp_path / "fresh"


def test_prepare_anchor_precedes_identity_and_has_no_sample_reads(tmp_path, monkeypatch):
    identity = fake_identity(tmp_path)
    output, clock, seen = tmp_path / "probe", Clock(), []
    monkeypatch.setattr(probe, "boot_id", lambda: "test-boot")

    def qualify(inputs, check):
        entry = probe.read_json(output / "prepare_entry.json")
        assert entry["round_started_perf_counter"] == 100.
        assert entry["planned_seconds"] == 900. and entry["hard_cap_seconds"] == 1800.
        with pytest.raises(RuntimeError, match="offline"):
            socket.create_connection(("127.0.0.1", 9))
        clock.value += 12.
        seen.append("metadata-only")
        return identity, None, None

    monkeypatch.setattr(probe, "qualify_inputs", qualify)
    monkeypatch.setattr(probe, "archive_code", lambda _output, _check: {"model_code_sha256": "c" * 64})
    p = probe.prepare(identity["paths"], output, gpu_uuid=UUID, clock=clock)
    assert p["round_started_perf_counter"] == 100. and seen == ["metadata-only"]
    assert p["controls"]["updates_per_path"] == 2 and p["controls"]["internal_deep_supervision"] is True
    report = probe.read_json(output / "prepare_attempt.json")
    assert report["status"] == "prepared-not-run" and report["gpu_hours_charged"] == 0
    assert report["whole_elapsed_seconds"] == 12.
    assert not (output / "workers").exists()
    with pytest.raises(FileExistsError):
        probe.prepare(identity["paths"], output, gpu_uuid=UUID, clock=clock)


def test_prepare_identity_failure_is_recorded_not_retried(tmp_path, monkeypatch):
    identity = fake_identity(tmp_path)
    output = tmp_path / "failed"
    monkeypatch.setattr(probe, "qualify_inputs", lambda *_args: (_ for _ in ()).throw(ValueError("identity missing")))
    with pytest.raises(ValueError, match="identity missing"):
        probe.prepare(identity["paths"], output, gpu_uuid=UUID)
    report = probe.read_json(output / "attempt.json")
    assert report["status"] == "failed" and report["scientific_claim"] is False
    assert report["gpu_hours_charged"] == 0 and "identity missing" in report["failure_reason"]


@pytest.mark.parametrize("change", [{"controls": {"updates_per_path": 3}}, {"monotonic_boot_id": "another-boot"},
                                    {"planned_seconds": 1800.}, {"scientific_claim": True}])
def test_rehashed_protocol_still_rejects_changed_frozen_contract(tmp_path, monkeypatch, change):
    p, output, _ = frozen(tmp_path, monkeypatch)
    p.update(change)
    p["protocol_sha256"] = probe.digest({k: v for k, v in p.items() if k != "protocol_sha256"})
    path = output / "altered.json"
    probe.write_json(path, p)
    with pytest.raises(ValueError):
        probe.verify_protocol(path)


def test_missing_input_identity_and_finite_json_are_rejected(tmp_path, monkeypatch):
    p, output, _ = frozen(tmp_path, monkeypatch)
    del p["inputs"]["sha256"]["checkpoint"]
    p["protocol_sha256"] = probe.digest({k: v for k, v in p.items() if k != "protocol_sha256"})
    path = output / "missing.json"
    probe.write_json(path, p)
    with pytest.raises(ValueError, match="five"):
        probe.verify_protocol(path)
    bad = tmp_path / "nonfinite.json"
    bad.write_text('{"loss": NaN}')
    with pytest.raises(ValueError):
        probe.read_json(bad)


def test_fresh_cuda_requires_exact_zero_baseline_and_never_clears_cache(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    events = []
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           set_device=lambda index: events.append(("set", index)), init=lambda: events.append("init"),
                           memory_allocated=lambda _index: 0, memory_reserved=lambda _index: 0,
                           is_bf16_supported=lambda: True, reset_peak_memory_stats=lambda _index: events.append("peak"))
    fake = SimpleNamespace(cuda=cuda)
    assert probe.fresh_cuda(fake, UUID, "bf16") == {"baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0}
    assert events == [("set", 0), "init", "peak"]
    cuda.memory_reserved = lambda _index: 1
    with pytest.raises(ValueError, match="0/0"):
        probe.fresh_cuda(fake, UUID, "fp32")
    cuda.memory_reserved = lambda _index: 0
    cuda.is_bf16_supported = lambda: False
    with pytest.raises(ValueError, match="BF16 unavailable"):
        probe.fresh_cuda(fake, UUID, "bf16")
    cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="unavailable"):
        probe.fresh_cuda(fake, UUID, "fp32")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="UUID"):
        probe.fresh_cuda(fake, UUID, "fp32")


def test_run_has_two_fresh_uuid_bound_direct_workers_and_whole_billing(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    clock.value += 25.  # Frozen prepare/run interval is included.
    snapshot, popen, events, processes = fake_execution(monkeypatch, p, output, clock)
    result = probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    assert result["status"] == "success" and result["finalized"]
    assert [e[0] for e in events] == ["snapshot", "spawn", "snapshot", "spawn"]
    assert len(processes) == 2 and all(process.done for process in processes)
    for event in events:
        if event[0] == "spawn":
            command, env = event[2], event[3]["env"]
            assert command[0] == str(probe.ROOT / ".venv/bin/python") and "-I" in command
            assert env["CUDA_VISIBLE_DEVICES"] == UUID and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert "PYTHONPATH" not in env
    assert result["whole_elapsed_seconds"] == 31. and result["gpu_phase_elapsed_seconds"] == 6.
    assert result["soft_overrun_seconds"] == 0 and result["scientific_claim"] is False
    assert probe.read_json(output / "closeout.json")["attempt_sha256"] == probe.sha256_file(output / "attempt.json")
    with pytest.raises(ValueError, match="resurrection"):
        probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)


def test_soft_budget_overrun_does_not_stop_or_relax_endpoint(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    clock.value += 901.
    snapshot, popen, _, _ = fake_execution(monkeypatch, p, output, clock)
    result = probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    assert result["status"] == "success" and len(result["workers"]) == 2
    assert result["soft_overrun_seconds"] == 7. and result["whole_elapsed_seconds"] == 907.


def test_hard_cap_stops_before_snapshot_spawn_and_bills_prepare_interval(tmp_path, monkeypatch):
    _, output, clock = frozen(tmp_path, monkeypatch)
    clock.value += 1791.
    events = []
    with pytest.raises(probe.BudgetLimited):
        probe.run(output, snapshot_fn=lambda *_a: events.append("snapshot"),
                  popen_factory=lambda *_a, **_k: events.append("spawn"), clock=clock)
    report = probe.read_json(output / "attempt.json")
    assert events == [] and report["status"] == "failed" and report["budget_limited"]
    assert report["gpu_hours_charged"] == 0 and report["whole_elapsed_seconds"] == 1791.


@pytest.mark.parametrize("kill_needed", [False, True])
def test_timeout_reaps_only_direct_owned_handle_without_neighbor_signals(tmp_path, monkeypatch, kill_needed):
    p, output, clock = frozen(tmp_path, monkeypatch)
    snapshot, popen, events, processes = fake_execution(monkeypatch, p, output, clock, timed_out=True, kill_needed=kill_needed)
    with pytest.raises(probe.BudgetLimited, match="timed out"):
        probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    assert len(processes) == 1 and len(events) == 2
    assert ("terminate", 321) in processes[0].events
    assert (("kill", 321) in processes[0].events) is kill_needed
    assert not any(event == ("terminate", 999) or event == ("kill", 999) for event in processes[0].events)
    assert probe.read_json(output / "attempt.json")["status"] == "failed"
    assert probe.read_json(output / "attempt.json")["gpu_hours_charged"] > 0


def test_headroom_or_identity_failure_never_spawns(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    spawn = []
    with pytest.raises(RuntimeError, match="headroom"):
        probe.run(output, snapshot_fn=lambda uuid: {"uuid": uuid, "free_mib": 1},
                  popen_factory=lambda *_a, **_k: spawn.append("spawn"), clock=clock)
    assert spawn == []
    assert probe.read_json(output / "attempt.json")["gpu_hours_charged"] == 0


@pytest.mark.parametrize("bad", [{"peak_reserved_bytes": None}, {"peak_reserved_bytes": float("inf")},
                                 {"baseline_reserved_bytes": 1}, {"resume_exact": {"weights": False}},
                                 {"checks": {"finite": False, "full_gradient": True}}, {"status": "skipped"}])
def test_worker_acceptance_rejects_missing_nonfinite_or_partial_evidence(bad):
    p = {"protocol_sha256": "a" * 64, "code": {"model_code_sha256": "c" * 64}}
    payload = receipt(p, "fp32")
    payload.update(bad)
    with pytest.raises(ValueError):
        probe.validate_worker_result(payload, p, "fp32")


def small_model_and_batch():
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    torch.manual_seed(41)
    model = ProcessForecastCoReasoner(in_channels=17, out_channels=17, dim=16, depth=1, heads=2,
                                     window_size=2, patch_size=2, anchored_processes=2, free_processes=2,
                                     default_reasoning_steps=4, spacetime_inputs=True, positional_process_readout=True,
                                     detach_between_steps=False).train()
    batch = {"coarse_history": torch.randn(2, 2, 17, 4, 4), "atmos_target": torch.randn(2, 17, 4, 4),
             "future_target": torch.randn(2, 17, 4, 4), "latitude": torch.linspace(-30, 30, 4),
             "longitude": torch.linspace(-120, 120, 4), "lead_time_hours": torch.full((2,), 6.),
             "init_calendar_year": torch.tensor([2024., 2023.]), "init_day_of_year": torch.tensor([59., 365.]),
             "init_utc_hour": torch.tensor([18., 18.]), "init_time": ["2024-02-28T18:00:00", "2023-12-31T18:00:00"]}
    return model, batch


def test_actual_rollout_two_updates_full_l12_gradient_poison_calendar_and_one_step_cpu():
    model, batch = small_model_and_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    original = deepcopy(model.state_dict())
    for _ in range(2):
        row = probe.timed_update(torch, model, optimizer, batch, "fp32", time.perf_counter() + 30.)
        assert row["elapsed_seconds"] > 0 and row["gradient_norm"] > 0
    assert any(not torch.equal(original[n], value) for n, value in model.state_dict().items())
    result = probe.diagnostic_checks(torch, model, batch, "fp32")
    assert result["two_step_forward_calls"] == 2 and result["internal_k_calls"] == 8
    assert result["one_step_forward_calls"] == 1 and result["poison_forecasts_equal"] and result["poison_loss_changed"]
    assert all(all(norm > 0 for norm in values) for values in result["l12_gradient_ownership"].values())
    assert result["calendar_trace"][1]["init_calendar_year"] == [2024, 2024]
    assert result["calendar_trace"][1]["init_day_of_year"] == [60, 1]
    assert result["finite"] and result["full_gradient"]


def test_detached_internal_path_and_nonfinite_gradients_have_counterproofs(monkeypatch):
    model, batch = small_model_and_batch()
    original = model.forward
    monkeypatch.setattr(model, "forward", lambda *a, **k: original(*a, **dict(k, detach_between_steps=True)))
    with pytest.raises(ValueError, match="missing gradient|zero gradient"):
        probe.diagnostic_checks(torch, model, batch, "fp32")
    with pytest.raises(ValueError, match="nonfinite"):
        probe.finite_tensors(torch, {"grad": torch.tensor([float("nan")])}, "gradient")
    with pytest.raises(ValueError, match="missing tensor"):
        probe.finite_tensors(torch, {"grad": None}, "gradient")


def test_real_current_format_checkpoint_body_resume_is_exact_cpu(tmp_path, monkeypatch):
    from training.r7_experiment import load_checkpoint, model_code_digest
    model, batch = small_model_and_batch()
    p = {"output": str(tmp_path / "probe"), "protocol_sha256": "a" * 64,
         "code": {"model_code_sha256": model_code_digest()},
         "inputs": {"model_spec": {"in_channels": 17, "out_channels": 17, "detach_between_steps": False},
                    "parent_import": {"report_sha256": "b" * 64,
                                      "data": {"data_identity": "c" * 64, "sources": {"source_sha256": "d" * 64}}}}}
    events = []
    monkeypatch.setattr(probe, "verify_code", lambda _p, check: events.append("code"))
    monkeypatch.setattr(probe, "verify_inputs", lambda _p, check: events.append("inputs"))
    reports, acceptance = probe.resume_paths(torch, model, batch, p, "fp32", time.perf_counter() + 30.)
    assert all(acceptance.values()) and events == ["code", "inputs"]
    assert len(reports) == 2 and all(row["updates"] == 2 for row in reports)
    resumed = Path(reports[1]["checkpoint"])
    saved = load_checkpoint(resumed, expected=reports[1]["signature"])
    assert saved["format"] == "r7-local-v1" and saved["updates"] == 2
    assert saved["contract"]["total_updates"] == 2 and saved["contract"]["steps"] == 4
    assert probe.read_json(resumed.parent / "intentional_stop.json")["status"] == "intentional-stop-not-failed"
    assert not (resumed.parent / "failed_attempt.json").exists()
    changed = deepcopy(saved["model"])
    key = next(k for k, v in changed.items() if v.is_floating_point())
    changed[key].view(-1)[0] += 1e-7
    assert not probe.exact_equal(torch, saved["model"], changed)
    with pytest.raises(ValueError, match="identity differs"):
        load_checkpoint(resumed, expected="e" * 64)
    with pytest.raises(FileExistsError):
        probe.resume_paths(torch, model, batch, p, "fp32", time.perf_counter() + 30.)


def test_wrong_model_digest_and_stale_calendar_are_rejected(tmp_path, monkeypatch):
    from training.r7_experiment import canonical_digest, load_checkpoint
    from model.spacetime_conditioning_r7 import phase_features
    checkpoint = tmp_path / "wrong_model.pt"
    contract = {"total_updates": 2}
    torch.save({"format": "r7-local-v1", "model_code_sha256": "0" * 64,
                "contract": contract, "signature": canonical_digest(contract)}, checkpoint)
    with pytest.raises(ValueError, match="implementation differs"):
        load_checkpoint(checkpoint)
    model, batch = small_model_and_batch()
    initial = phase_features(batch["init_utc_hour"], batch["init_day_of_year"], batch["lead_time_hours"],
                             init_calendar_year=batch["init_calendar_year"])
    def stale(_module, args):
        features = args[0].clone()
        features[:, :, 4:] = initial[:, None]
        return (features,)
    handle = model.backbone.spacetime.net.register_forward_pre_hook(stale)
    try:
        with pytest.raises(AssertionError):
            probe.trace_rollout(torch, model, batch, "fp32")
    finally:
        handle.remove()


def test_changed_protocol_failure_is_terminal_and_publication_keeps_original(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    p["controls"] = dict(probe.CONTROLS, seed=42)
    p["protocol_sha256"] = probe.digest({k: v for k, v in p.items() if k != "protocol_sha256"})
    (output / "protocol.json").write_text(json.dumps(p))
    with pytest.raises(ValueError, match="contract changed"):
        probe.run(output, clock=clock)
    assert probe.read_json(output / "attempt.json")["status"] == "failed"
    with pytest.raises(ValueError, match="resurrection"):
        probe.run(output, clock=clock)
    original = ValueError("initial failure")
    monkeypatch.setattr(probe, "write_json", lambda *_a: (_ for _ in ()).throw(OSError("storage full")))
    probe.publish(tmp_path / "unpublished.json", {}, original)
    assert "storage full" in original.__notes__[0] and str(original) == "initial failure"


def test_actual_code_closure_is_code_only_and_size_limits_are_met():
    import ast
    paths = probe.source_paths()
    assert "tools/r7_v2_precision_probe.py" in paths
    assert "training/r7_autoregressive_rollout.py" in paths and "training/r7_parent_import.py" in paths
    assert "data/r7_autoregressive_dataset.py" in paths and "training/r7_m3_driver.py" in paths
    assert not any("outputs" in Path(p).parts or any("legacy" in part for part in Path(p).parts) for p in paths)
    assert "tools/r7_v2_precision_support.py" in paths
    for file in (Path(probe.__file__), probe.ROOT / "tools/r7_v2_precision_support.py", Path(__file__)):
        text = file.read_text()
        assert len(text.splitlines()) <= 600
        tree = ast.parse(text)
        assert all(n.end_lineno - n.lineno + 1 <= 200 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
