"""Temporary CPU/fake-CUDA engineering only; no outputs, weather, GPU or archive loads."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from tools import r7_v2_package_precision_probe as probe
from tools import r7_v2_package_precision_support as support

UUID = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for obj, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection")):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(*, tiny=False):
    from training.r7_experiment import make_model
    process = dict(probe.BASE_MODEL, anchored_processes=8, free_processes=8, periodic_width=False,
                   spatial_solver_feedback=False, spacetime_field_mode="fields", pooled_readout_query=False,
                   **{flag: True for flag in probe.PACKAGE_FLAGS})
    if tiny:
        process.update(dim=16, depth=1, heads=2, window_size=2)
    generic = {k: v for k, v in process.items() if k not in ("anchored_processes", "free_processes")}
    generic["latent_tokens"] = 16
    old = dict(process, **{flag: False for flag in probe.PACKAGE_FLAGS if flag not in ("solver_gate_proposal", "solver_state_recurrence")})
    specs = {"old_ours": {"kind": "process", "model": old}, "process": {"kind": "process", "model": process},
             "matched_generic": {"kind": "generic", "model": generic}}
    mapping = {}
    for arm, spec in specs.items():
        with torch.random.fork_rng(devices=[]):
            model = make_model(spec["kind"], spec["model"])
        mapping[arm] = {}
        for key in model.state_dict():
            source = "process_queries" if key == "latent" else key.replace("cell.", "reasoning_cell.", 1) if key.startswith("cell.") else key
            source = source.replace("latent_to_context.", "process_to_context.")
            mapping[arm][key] = source
    return {"mode": "two_step", "model_specs": specs, "initialization": {"anchor": specs["process"], "mapping": mapping}}


def batch():
    generator = torch.Generator().manual_seed(9)
    return {"coarse_history": torch.randn(2, 2, 17, 5, 7, generator=generator),
            "atmos_target": torch.randn(2, 17, 5, 7, generator=generator), "future_target": torch.randn(2, 17, 5, 7, generator=generator),
            "latitude": torch.linspace(-30, 30, 5), "longitude": torch.linspace(-179, 179, 7), "lead_time_hours": torch.full((2,), 6.),
            "init_calendar_year": torch.tensor([2024., 2023.]), "init_day_of_year": torch.tensor([59., 365.]), "init_utc_hour": torch.tensor([18., 18.]),
            "history_offsets_hours": torch.tensor([[-6., 0.], [-6., 0.]]),
            "init_time": ["2024-02-28T18:00:00", "2023-12-31T18:00:00"]}


class Clock:
    def __init__(self, value=100.):
        self.value = value

    def __call__(self):
        return self.value


class OwnedProcess:
    def __init__(self, clock, *, timeout=False, kill_needed=False):
        self.clock, self.timeout, self.kill_needed = clock, timeout, kill_needed
        self.events, self.done, self.pid = [], False, 321

    def poll(self):
        return 0 if self.done else None

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if self.timeout and not self.done:
            self.clock.value += timeout
            raise subprocess.TimeoutExpired("owned", timeout)
        self.done = True
        return 0

    def terminate(self):
        self.events.append(("terminate", self.pid))
        if not self.kill_needed:
            self.done = True

    def kill(self):
        self.events.append(("kill", self.pid))
        self.done = True


def frozen(tmp_path, monkeypatch):
    clock, output = Clock(), tmp_path / "probe_attempt01"
    output.mkdir()
    inputs = tmp_path / "input"
    inputs.mkdir()
    configfile = inputs / "configuration.json"
    config = configuration()
    probe.write_json(configfile, config)
    train, sidecar = inputs / "train.jsonl", inputs / "scale_metadata.json"
    train.write_text("metadata only", encoding="utf-8")
    sidecar.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(probe, "boot_id", lambda: "test-boot")
    body = {"format": "r7-v2-package-precision-v1", "scientific_claim": False, "limitations": probe.LIMITATIONS,
            "test_read": False, "frozen_before_any_step": True, "output": str(output), "round_started_perf_counter": clock(),
            "monotonic_boot_id": "test-boot", "planned_seconds": 900., "hard_cap_seconds": 1800.,
            "controls": probe.CONTROLS, "precisions": list(probe.PRECISIONS),
            "gpu": {"policy": "shared", "uuid": UUID, "estimated_peak_mib": 4096, "headroom_margin_mib": 2048},
            "configuration": {"path": str(configfile), "file_sha256": probe.sha256_file(configfile), "value": config,
                              "mapping_reports": {"41": {"process": {"target_state_sha256": "a" * 64}}}},
            "inputs": {"trainmanifest": str(train), "sidecar": str(sidecar), "data_identity": "d" * 64,
                       "sources": {"source_sha256": "e" * 64}, "sidecar_sha256": "f" * 64},
            "code": {"model_code_sha256": "c" * 64}}
    p = {**body, "protocol_sha256": probe.digest(body)}
    probe.write_json(output / "protocol.json", p)
    monkeypatch.setattr(probe, "verify_code", lambda _p, check: check())
    monkeypatch.setattr(probe, "verify_inputs", lambda _p, check: check())
    return p, output, clock


def receipt(p, precision):
    checks = {"finite": True, "full_gradient": True, "poison_forecasts_equal": True, "poison_loss_changed": True,
              "two_step_forward_calls": 2, "one_step_forward_calls": 1, "internal_k_calls": 8, "query_reuse_calls": 8,
              "calendar_trace": [{}, {}], "known_feature_sha256": ["a" * 64, "b" * 64],
              "l12_gradient_ownership": {n: [1.] for n in support.GRADIENT_GROUPS},
              "active_parameter_gradients": {"known.weight": 1.}, "idle_parameters": {"process_readout.weight": "inapplicable"}}
    return {"status": "success", "scientific_claim": False, "limitations": probe.LIMITATIONS, "test_read": False,
            "precision": precision, "protocol_sha256": p["protocol_sha256"], "model_code_sha256": p["code"]["model_code_sha256"],
            "configuration_sha256": p["configuration"]["file_sha256"], "initial_state_sha256": "a" * 64,
            "actual_optimizer_updates": 4, "baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "elapsed_seconds": 3., "checks": checks,
            "resume_exact": {n: True for n in ("weights", "optimizer", "rng", "losses")}, "input_tensor_sha256": {"history": "a" * 64},
            "branches": [{"branch": branch, "updates": 2, "losses": [{}, {}]} for branch in ("uninterrupted", "intentional_resume")]}


def fake_execution(p, output, clock, *, timeout=False, kill_needed=False):
    events, processes = [], []
    def snapshot(uuid):
        events.append(("snapshot", uuid))
        return {"uuid": uuid, "free_mib": 16384, "neighbors": [{"pid": 999, "used_mib": "1024"}], "read_only": True}
    def popen(command, **kwargs):
        precision = command[command.index("--worker") + 1]
        events.append(("spawn", precision, command, kwargs))
        process = OwnedProcess(clock, timeout=timeout, kill_needed=kill_needed)
        processes.append(process)
        if not timeout:
            clock.value += 3.
            probe.write_json(output / "workers" / (precision + ".json"), receipt(p, precision))
        return process
    return snapshot, popen, events, processes


def test_default_cli_and_offline_connections_are_independent():
    assert probe.DEFAULT_OUTPUT.name == "r7_v2_package_precision_probe_20261003_attempt01"
    assert 0 < probe.CLI_ENTRY_PERF_COUNTER <= time.perf_counter()
    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    anchor = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLI_ENTRY_PERF_COUNTER" for t in n.targets))
    assert anchor.lineno < next(n.lineno for n in tree.body if isinstance(n, ast.Import) and any(a.name == "argparse" for a in n.names))
    assert "M3" in probe.LIMITATIONS[2] and probe.CONTROLS["loss_dtype"] == "torch.float32"
    probe.deny_network()
    with socket.socket() as connection:
        for method in (connection.connect, connection.connect_ex):
            with pytest.raises(RuntimeError, match="offline"):
                method(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("127.0.0.1", 9))


def test_output_exclusive_protected_symlink_and_nested_old_rejections(tmp_path):
    for name in ("data/raw/x", "data/interim/x", "data/processed/x", "legacy_v6/x"):
        with pytest.raises(ValueError, match="protected"):
            probe.output_path(probe.ROOT / name, fresh=True)
    old = tmp_path / "old"
    old.mkdir()
    (old / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="nested old"):
        probe.output_path(old / "child", fresh=True)
    with pytest.raises(FileExistsError):
        probe.output_path(old, fresh=True)
    link = tmp_path / "link"
    link.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        probe.output_path(link / "child", fresh=True)
    with pytest.raises(ValueError, match="overlaps"):
        probe.output_path(tmp_path / "child", [tmp_path])
    assert not (old / "child").exists()


@pytest.mark.parametrize("change", ["known_context_inputs", "source_position_markers", "source_role_markers", "draft_query_feedback", "detach_between_steps", "dim"])
def test_configuration_strict_full_package_does_not_silently_fallback(change):
    config = configuration()
    config["model_specs"]["process"]["model"][change] = 1 if change in ("dim", "known_context_inputs") else False if change != "detach_between_steps" else True
    with pytest.raises(ValueError):
        probe.validate_configuration(config)


def test_actual_capacity_and_three_seed_complete_mapping_are_qualified(tmp_path):
    config = configuration()
    path = tmp_path / "C_configuration.json"
    probe.write_json(path, config)
    result = probe.qualify_configuration(path, lambda: None)
    assert set(result["mapping_reports"]) == {"41", "42", "43"}
    for rows in result["mapping_reports"].values():
        assert set(rows) == set(probe.ARMS)
        assert len({r["anchor_state_sha256"] for r in rows.values()}) == 1
        assert all(r["uninitialized_target_keys"] == [] and r["optimizer_reset"] and not r["resume"] for r in rows.values())
    assert result["value"]["model_specs"]["process"]["model"]["dim"] == 192
    assert "latent" in config["initialization"]["mapping"]["matched_generic"]


def test_missing_or_wrong_target_mapping_fails_before_partial_initialization():
    from training.r7_v2_profile import seeded_mapped_model
    config = configuration(tiny=True)
    del config["initialization"]["mapping"]["matched_generic"]["latent"]
    with pytest.raises(ValueError, match="cover every"):
        seeded_mapped_model(config, 41, "matched_generic")
    config = configuration(tiny=True)
    config["initialization"]["mapping"]["matched_generic"]["latent"] = "backbone.head.decoder.bias"
    with pytest.raises(ValueError, match="mapping mismatch"):
        seeded_mapped_model(config, 41, "matched_generic")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_actual_new_models_two_updates_l12_grad_query_reuse_and_poison_cpu(precision):
    from training.r7_v2_profile import seeded_mapped_model
    config, data = configuration(tiny=True), batch()
    process, _, _ = seeded_mapped_model(config, 41, "process")
    generic, _, _ = seeded_mapped_model(config, 41, "matched_generic")
    outputs = []
    for model in (process, generic):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        for _ in range(2):
            row = support.timed_update(torch, model, optimizer, data, precision, time.perf_counter() + 30., probe)
            assert row["loss_dtype"] == "torch.float32" and row["gradient_norm"] > 0 and row["elapsed_seconds"] > 0
        result = support.diagnostic_checks(torch, model, data, precision, probe)
        assert result["full_gradient"] and result["query_reuse_calls"] == 8 and result["poison_forecasts_equal"]
        assert result["calendar_trace"][1]["history_offsets_hours"] == [[-6., 0.], [-6., 0.]]
        assert all(norm > 0 for values in result["l12_gradient_ownership"].values() for norm in values)
        assert all(name.startswith(support.IDLE_PREFIXES) for name in result["idle_parameters"])
        assert model.backbone.known_context.projection.weight.grad is not None
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
            from training.r7_autoregressive_rollout import training_two_step
            outputs.append(training_two_step(model, data).forecasts.detach())
    assert torch.equal(outputs[0], outputs[1]), "mapped Generic did not match actual Process update path"


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_current_format_same_endpoint_resume_is_exact_cpu(tmp_path, monkeypatch, precision):
    from training.r7_experiment import load_checkpoint, model_code_digest
    from training.r7_v2_profile import seeded_mapped_model
    config = configuration(tiny=True)
    model, initial, _ = seeded_mapped_model(config, 41, "process")
    model.train()
    p = {"output": str(tmp_path / "probe"), "protocol_sha256": "a" * 64, "code": {"model_code_sha256": model_code_digest()},
         "configuration": {"file_sha256": "b" * 64, "value": config, "mapping_reports": {"41": {"process": initial}}},
         "inputs": {"data_identity": "d" * 64, "sources": {"source_sha256": "e" * 64}}}
    events = []
    monkeypatch.setattr(probe, "verify_code", lambda _p, check: events.append("code"))
    monkeypatch.setattr(probe, "verify_inputs", lambda _p, check: events.append("inputs"))
    rows, exact = support.resume_paths(torch, model, batch(), p, precision, time.perf_counter() + 30., probe)
    assert all(exact.values()) and events == ["code", "inputs"]
    assert len(rows) == 2 and all(r["updates"] == 2 for r in rows)
    endpoint = Path(rows[1]["checkpoint"])
    saved = load_checkpoint(endpoint, expected=rows[1]["signature"])
    assert saved["format"] == "r7-local-v1" and saved["contract"]["initialization"] == initial
    assert saved["contract"]["configuration_sha256"] == "b" * 64 and saved["updates"] == 2
    assert probe.read_json(endpoint.parent / "intentional_stop.json")["status"] == "intentional-stop-not-failed"
    assert not (endpoint.parent / "failed_attempt.json").exists()
    changed = deepcopy(saved["model"])
    key = next(k for k, v in changed.items() if v.is_floating_point())
    changed[key].view(-1)[0] += 1e-7
    assert not support.exact_equal(torch, saved["model"], changed)
    with pytest.raises(ValueError, match="identity differs"):
        load_checkpoint(endpoint, expected="0" * 64)
    fresh_model, _, _ = seeded_mapped_model(config, 41, "process")
    fresh_model.train()
    with pytest.raises(FileExistsError):
        support.resume_paths(torch, fresh_model, batch(), p, precision, time.perf_counter() + 30., probe)


@pytest.mark.parametrize("fault", ["projection", "roles", "query", "detached", "shadow"])
def test_new_known_role_query_internal_k_faults_cannot_pass(monkeypatch, fault):
    from training.r7_v2_profile import seeded_mapped_model
    model, _, _ = seeded_mapped_model(configuration(tiny=True), 41, "process")
    model.train()
    data = batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        support.timed_update(torch, model, optimizer, data, "fp32", time.perf_counter() + 30., probe)
    if fault == "projection":
        original = model.backbone.known_context.forward
        monkeypatch.setattr(model.backbone.known_context, "forward", lambda *a, **k: original(*a, **k) * 0)
    elif fault == "roles":
        model.role_context.requires_grad_(False)
    elif fault == "query":
        original = model.process_conditioning
        monkeypatch.setattr(model, "process_conditioning", lambda *a, **k: original(*a, **dict(k, draft_tokens=k["draft_tokens"].clone())))
    elif fault == "detached":
        original = model.forward
        monkeypatch.setattr(model, "forward", lambda *a, **k: original(*a, **dict(k, detach_between_steps=True)))
    else:
        model.known_context_inputs = 1
    with pytest.raises(ValueError):
        support.diagnostic_checks(torch, model, data, "fp32", probe)


def test_changed_history_metadata_and_missing_loss_finite_fail_closed():
    from training.r7_v2_profile import seeded_mapped_model
    model, _, _ = seeded_mapped_model(configuration(tiny=True), 41, "process")
    model.train()
    data = batch()
    data["history_offsets_hours"] = torch.tensor([[-12., 0.], [-6., 0.]])
    with pytest.raises(ValueError, match="explicit ordered|physical"):
        support.trace_rollout(torch, model, data, "fp32", probe)
    with pytest.raises(ValueError, match="nonfinite"):
        support.finite_tensors(torch, {"grad": torch.tensor([float("nan")])}, "gradient")
    with pytest.raises(ValueError, match="missing"):
        support.finite_tensors(torch, {"grad": None}, "gradient")


def test_fresh_cuda_zero_baseline_no_clear_or_precision_fallback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    calls = []
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, set_device=lambda i: calls.append(("set", i)),
                           init=lambda: calls.append("init"), memory_allocated=lambda i: 0, memory_reserved=lambda i: 0,
                           is_bf16_supported=lambda: True, reset_peak_memory_stats=lambda i: calls.append("peak"))
    fake = SimpleNamespace(cuda=cuda)
    assert probe.fresh_cuda(fake, UUID, "bf16") == {"baseline_allocated_bytes": 0, "baseline_reserved_bytes": 0}
    assert calls == [("set", 0), "init", "peak"]
    cuda.memory_reserved = lambda i: 1
    with pytest.raises(ValueError, match="0/0"):
        probe.fresh_cuda(fake, UUID, "fp32")
    cuda.memory_reserved = lambda i: 0
    cuda.is_bf16_supported = lambda: False
    with pytest.raises(ValueError, match="BF16 unavailable"):
        probe.fresh_cuda(fake, UUID, "bf16")
    cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="no CPU fallback"):
        probe.fresh_cuda(fake, UUID, "fp32")


def test_prepare_freezes_entry_before_cpu_qualification_without_sample_read(tmp_path, monkeypatch):
    p, unused, clock = frozen(tmp_path, monkeypatch)
    output = tmp_path / "prepare_only"
    events = []
    def qualify_config(path, check):
        entry = probe.read_json(output / "prepare_entry.json")
        assert entry["planned_seconds"] == 900. and entry["hard_cap_seconds"] == 1800. and entry["round_started_perf_counter"] == 100.
        with pytest.raises(RuntimeError, match="offline"):
            socket.create_connection(("127.0.0.1", 9))
        events.append("CPU-mapping")
        return p["configuration"]
    monkeypatch.setattr(probe, "qualify_configuration", qualify_config)
    monkeypatch.setattr(probe, "qualify_inputs", lambda *_a: (p["inputs"], None))
    monkeypatch.setattr(probe, "archive_code", lambda *_a: p["code"])
    got = probe.prepare(p["configuration"]["path"], p["inputs"]["trainmanifest"], p["inputs"]["sidecar"], output, gpu_uuid=UUID, clock=clock)
    assert got["controls"] == probe.CONTROLS and events == ["CPU-mapping"]
    assert probe.read_json(output / "prepare_attempt.json")["gpu_hours_charged"] == 0
    assert not (output / "workers").exists()


def test_two_direct_workers_soft_overrun_and_continuous_prepare_gap_bill(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    clock.value += 901.
    snapshot, popen, events, processes = fake_execution(p, output, clock)
    result = probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    assert result["status"] == "success" and len(result["workers"]) == 2
    assert [e[0] for e in events] == ["snapshot", "spawn", "snapshot", "spawn"]
    assert len(processes) == 2 and all(p.done for p in processes)
    assert result["whole_elapsed_seconds"] == 907. and result["gpu_phase_elapsed_seconds"] == 6. and result["soft_overrun_seconds"] == 7.
    for event in events:
        if event[0] == "spawn":
            assert event[2][0] == str(probe.ROOT / ".venv/bin/python") and "-I" in event[2]
            assert event[3]["env"]["CUDA_VISIBLE_DEVICES"] == UUID and event[3]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert probe.read_closeout(output)["attempt_sha256"] == probe.sha256_file(output / "attempt.json")
    with pytest.raises(ValueError, match="resurrection"):
        probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)


def test_hard_deadline_and_headroom_stop_before_spawn(tmp_path, monkeypatch):
    _, output, clock = frozen(tmp_path, monkeypatch)
    clock.value += 1791.
    events = []
    with pytest.raises(probe.BudgetLimited):
        probe.run(output, snapshot_fn=lambda *_a: events.append("snapshot"), popen_factory=lambda *_a, **_k: events.append("spawn"), clock=clock)
    assert events == [] and probe.read_json(output / "attempt.json")["budget_limited"]
    assert probe.read_json(output / "attempt.json")["gpu_hours_charged"] == 0


@pytest.mark.parametrize("kill_needed", [False, True])
def test_timeout_signals_only_direct_owned_process_not_neighbors(tmp_path, monkeypatch, kill_needed):
    p, output, clock = frozen(tmp_path, monkeypatch)
    snapshot, popen, events, processes = fake_execution(p, output, clock, timeout=True, kill_needed=kill_needed)
    with pytest.raises(probe.BudgetLimited):
        probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    assert len(processes) == 1 and len(events) == 2
    assert ("terminate", 321) in processes[0].events and (("kill", 321) in processes[0].events) is kill_needed
    assert not any(e == ("terminate", 999) or e == ("kill", 999) for e in processes[0].events)
    assert probe.read_json(output / "attempt.json")["status"] == "failed"


@pytest.mark.parametrize("bad", [{"status": "skipped"}, {"peak_reserved_bytes": None}, {"baseline_reserved_bytes": 1},
                                 {"configuration_sha256": "0" * 64}, {"resume_exact": {"weights": True}}, {"actual_optimizer_updates": 2}])
def test_receipt_missing_identity_finite_or_endpoint_cannot_pass(tmp_path, monkeypatch, bad):
    p, _, _ = frozen(tmp_path, monkeypatch)
    result = receipt(p, "fp32")
    result.update(bad)
    with pytest.raises(ValueError):
        support.validate_worker_result(result, p, "fp32", probe)


def test_rehashed_configuration_and_reboot_remain_terminal_failures(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    p["monotonic_boot_id"] = "another-boot"
    p["protocol_sha256"] = probe.digest({k: v for k, v in p.items() if k != "protocol_sha256"})
    (output / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError, match="same-boot"):
        probe.run(output, clock=clock)
    assert probe.read_json(output / "attempt.json")["status"] == "failed"
    with pytest.raises(ValueError, match="resurrection"):
        probe.run(output, clock=clock)


@pytest.mark.parametrize("fault", ["zip", "identity", "malformed-json"])
def test_early_own_failure_is_claimed_sealed_and_not_retryable(tmp_path, monkeypatch, fault):
    p, output, clock = frozen(tmp_path, monkeypatch)
    events = []
    def fail(_p, check):
        raise ValueError(fault + " qualification failed")
    if fault == "zip":
        monkeypatch.setattr(probe, "verify_code", fail)
    elif fault == "identity":
        monkeypatch.setattr(probe, "verify_inputs", fail)
    else:
        (output / "protocol.json").write_text("{malformed", encoding="utf-8")
    with pytest.raises(ValueError):
        probe.run(output, snapshot_fn=lambda *_a: events.append("snapshot"), clock=clock)
    assert events == [] and (output / "run_started.json").is_file()
    assert probe.read_json(output / "attempt.json")["status"] == "failed"
    assert probe.read_json(output / "closeout.json")["finalized"] is False
    with pytest.raises(ValueError, match="resurrection"):
        probe.run(output, clock=clock)


def test_late_hash_consumes_hard_budget_and_authoritative_closeout_rejects_success(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    snapshot, popen, _, processes = fake_execution(p, output, clock)
    original = probe.sha256_file
    def delayed_hash(path, check=lambda: None):
        value = original(path, check)
        if Path(path).name == "attempt.json":
            clock.value = p["round_started_perf_counter"] + 1801.
        return value
    monkeypatch.setattr(probe, "sha256_file", delayed_hash)
    with pytest.raises(probe.BudgetLimited):
        probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    closeout = probe.read_json(output / "closeout.json")
    assert len(processes) == 2 and all(process.done for process in processes)
    assert closeout["status"] == "failed" and closeout["budget_limited"] and not closeout["finalized"]
    assert closeout["whole_elapsed_seconds"] == 1801. and closeout["soft_overrun_seconds"] == 901.


def test_closeout_publication_over_hard_cap_invalidates_immutable_candidates(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    snapshot, popen, _, _ = fake_execution(p, output, clock)
    original, hashes = probe.write_json, {}
    def delayed_closeout(path, payload):
        original(path, payload)
        if Path(path).name in ("attempt.json", "closeout.json"):
            hashes[Path(path).name] = probe.sha256_file(path)
        if Path(path).name == "closeout.json":
            clock.value = p["round_started_perf_counter"] + 1801.
    monkeypatch.setattr(probe, "write_json", delayed_closeout)
    error, returned = None, None
    try:
        returned = probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    except probe.BudgetLimited as exc:
        error = exc
    candidate = probe.read_json(output / "closeout.json")
    assert error is not None, f"returned {returned['status']} with accepted closeout {candidate['status']} after whole=1801"
    assert candidate["status"] == "success" and candidate["finalized"]
    assert all(probe.sha256_file(output / name) == pin for name, pin in hashes.items())
    marker = probe.read_json(output / "publication_failure.json")
    assert marker["status"] == "failed" and not marker["finalized"] and marker["budget_limited"]
    assert marker["whole_elapsed_seconds"] == 1801. and marker["soft_overrun_seconds"] == 901.
    assert marker["invalidates"] == ["attempt.json", "closeout.json"]
    with pytest.raises(ValueError, match="publication failure"):
        probe.read_closeout(output)
    with pytest.raises(ValueError, match="resurrection"):
        probe.run(output, clock=clock)


def test_late_postflight_hash_cannot_finalize_a_round(tmp_path, monkeypatch):
    p, output, clock = frozen(tmp_path, monkeypatch)
    snapshot, popen, _, _ = fake_execution(p, output, clock)
    calls = []
    def late_verification(_p, check):
        calls.append(1)
        if len(calls) == 2:
            clock.value = p["round_started_perf_counter"] + 1801.
        check()
    monkeypatch.setattr(probe, "verify_code", late_verification)
    with pytest.raises(probe.BudgetLimited):
        probe.run(output, snapshot_fn=snapshot, popen_factory=popen, clock=clock)
    assert probe.read_json(output / "attempt.json")["status"] == "failed"
    assert probe.read_json(output / "attempt.json")["finalized"] is False
    assert probe.read_json(output / "closeout.json")["whole_elapsed_seconds"] == 1801.


def test_shadow_source_module_fails_closed_and_head_change_invalidates_code(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "model.shadow_probe", SimpleNamespace(__file__=str(tmp_path / "model.py")))
    with pytest.raises(ValueError, match="shadow active-source"):
        support.require_source_runtime(probe)
    monkeypatch.delitem(sys.modules, "model.shadow_probe")
    # No archive/checkpoint deserialization occurs: runtime HEAD is checked first.
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: "b" * 40 + "\n")
    with pytest.raises(ValueError, match="HEAD/source"):
        probe.verify_code({"output": str(tmp_path), "code": {"base_commit": "a" * 40, "source_root": str(probe.ROOT)}}, lambda: None)


def test_low_headroom_stops_without_neighbor_interference(tmp_path, monkeypatch):
    _, output, clock = frozen(tmp_path, monkeypatch)
    spawns = []
    with pytest.raises(RuntimeError, match="headroom"):
        probe.run(output, snapshot_fn=lambda uuid: {"uuid": uuid, "free_mib": 6143, "neighbors": [{"pid": 999}]},
                  popen_factory=lambda *_a, **_k: spawns.append(1), clock=clock)
    assert spawns == [] and probe.read_json(output / "attempt.json")["gpu_hours_charged"] == 0


def test_prepare_early_archive_failure_seals_and_empty_allocator_requires_uuid(tmp_path, monkeypatch):
    p, _, clock = frozen(tmp_path, monkeypatch)
    output = tmp_path / "prepare_fail"
    monkeypatch.setattr(probe, "qualify_configuration", lambda *_a: p["configuration"])
    monkeypatch.setattr(probe, "qualify_inputs", lambda *_a: (p["inputs"], None))
    monkeypatch.setattr(probe, "archive_code", lambda *_a: (_ for _ in ()).throw(ValueError("early ZIP identity")))
    with pytest.raises(ValueError, match="early ZIP"):
        probe.prepare(p["configuration"]["path"], p["inputs"]["trainmanifest"], p["inputs"]["sidecar"], output, gpu_uuid=UUID, clock=clock)
    assert probe.read_json(output / "attempt.json")["status"] == "failed"
    assert probe.read_json(output / "attempt.json")["gpu_hours_charged"] == 0
    with pytest.raises(FileExistsError):
        probe.prepare(p["configuration"]["path"], p["inputs"]["trainmanifest"], p["inputs"]["sidecar"], output, gpu_uuid=UUID, clock=clock)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="UUID"):
        probe.fresh_cuda(SimpleNamespace(cuda=None), UUID, "fp32")


def test_code_closure_owns_helpers_excludes_weather_old_probe_and_size_limits():
    paths = probe.source_paths()
    assert "tools/r7_v2_package_precision_probe.py" in paths and "tools/r7_v2_package_precision_support.py" in paths
    assert "training/r7_v2_profile.py" in paths and "model/known_context_r7.py" in paths
    assert not any("outputs" in Path(n).parts or any("legacy" in p for p in Path(n).parts) for n in paths)
    assert "tools/r7_v2_precision_probe.py" not in paths and "tools/r7_v2_precision_support.py" not in paths
    assert all(Path(module.__file__).resolve().is_relative_to(probe.ROOT) for module in (probe, support))
    for name in ("tools/r7_v2_package_precision_probe.py", "tools/r7_v2_package_precision_support.py", "tests/test_r7_v2_package_precision_probe.py"):
        text = (probe.ROOT / name).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 600
        assert all(n.end_lineno - n.lineno + 1 <= 200 for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef))
