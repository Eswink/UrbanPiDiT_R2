"""CPU-only fake workers and adversarial guards; no real data, archive execution or GPU."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from training import r7_v2_driver as driver
from training import r7_v2_identity as identity
from training import r7_v2_protocol as protocol
from training import r7_v2_worker as worker


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now


class FakeProcess:
    def __init__(self, clock, *, seconds=1, code=0, timeout=False, cleanup_fail=False):
        self.clock, self.seconds, self.code, self.timeout = clock, seconds, code, timeout
        self.cleanup_fail, self.done, self.signals, self.pid = cleanup_fail, False, [], 12345
    def wait(self, timeout):
        if self.done:
            return self.code
        if self.signals and self.cleanup_fail:
            raise OSError("owned reap failed fixture")
        if self.timeout and not self.signals:
            self.clock.now += timeout
            raise subprocess.TimeoutExpired("owned fixture", timeout)
        self.clock.now += self.seconds
        self.done = True
        return self.code
    def poll(self):
        return self.code if self.done else None
    def terminate(self):
        self.signals.append("terminate-owned")
    def kill(self):
        self.signals.append("kill-owned")


@pytest.fixture(autouse=True)
def restore_process_guards(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _refreeze(value):
    value["protocol_sha256"] = protocol.digest({key: item for key, item in value.items() if key != "protocol_sha256"})
    return value


def _configuration():
    specs = {arm: {"kind": "generic" if arm == "matched_generic" else "process",
                   "model": {"detach_between_steps": False}} for arm in protocol.C_ARMS}
    return {"mode": "l6", "model_specs": specs, "initialization": {
            "anchor": specs["process"], "mapping": {arm: {"weight": "weight"} for arm in protocol.C_ARMS}},
            "primary": {"variable": "t2m"}, "tolerances": {"degradation": 0},
            "case_unit_selection": "exact paired case units", "adaptive_gate": {"start": False},
            "reference": "independent B evidence", "selection_evidence": {"sha256": "e" * 64}}


@pytest.fixture
def frozen(tmp_path):
    output = tmp_path / "r7_74_autoregressive_20261003_attempt01"
    output.mkdir()
    windows = {"input_windows": 3, "usable_windows": 2, "excluded_sample_ids": ["boundary"],
               "exclusions": [{"sample_id": "boundary", "reason": "t12_outside_train_split"}],
               "window_sha256": "f" * 64, "selection": "exact train windows", "test_read": False, "state_fields_read": False}
    data = {"data_identity": "d" * 64, "channels": [f"var{i}" for i in range(17)],
            "units": ["K"] * 17, "test_read": False, "evaluation_cases": {
                str(lead): {"n_available": 1, "cases": [["2016-02-17T00:00:00", ["2016-02-17T06:00:00"]]]}
                for lead in protocol.LEADS}}
    parents = {str(seed): {"original_output": str(tmp_path / "parent"), "model_spec": {"detach_between_steps": False},
                          "checkpoint": str(tmp_path / f"parent{seed}.pt"), "file_sha256": {}}
               for seed in protocol.B_SEEDS}
    profile = {"measurements": {str(seed): {arm: {"forward_flops": 1} for arm in protocol.B_ARMS}
                                for seed in protocol.B_SEEDS},
               "pairing": {str(seed): {arm: {"model_spec": {"detach_between_steps": False}} for arm in protocol.B_ARMS}
                           for seed in protocol.B_SEEDS}}
    value = protocol.build_protocol(stage="B", output=output, manifests=tmp_path / "manifests",
        data=data, sources={"source_sha256": "s" * 64}, sidecar={"identity": "i" * 64, "path": str(tmp_path / "scale_metadata.json")}, windows=windows,
        parents=parents, profile=profile, code={"model_code_sha256": "a" * 64, "source_tree_sha256": "b" * 64,
        "code_zip_sha256": "c" * 64}, gpu_uuid="GPU-fixed", round_started_perf_counter=0.0,
        boot_id=protocol.monotonic_boot_id())
    protocol.write_json(output / "protocol.json", value, output=output)
    protocol.write_json(output / "prepare_attempt.json", {"status": "prepared-not-run",
        "protocol_sha256": value["protocol_sha256"]}, output=output)
    return value


def _fake(frozen, monkeypatch, *, cpu_seconds=10, seconds=1, fail_at=None, timeout_at=None,
          cleanup_fail=False, free=9999, result_change=None, cpu_error=False, peak_mib=2100):
    clock, jobs, processes, queries, aggregates = FakeClock(), [], [], [], []
    def preflight(value, *, check):
        clock.now += cpu_seconds
        check()
        if cpu_error:
            raise ValueError("CPU input pin mismatch fixture")
    monkeypatch.setattr(driver, "_runtime_preflight", preflight)
    monkeypatch.setattr(driver, "_runtime_postflight", lambda *args, **kwargs: None)
    def snapshot(uuid):
        queries.append(uuid)
        clock.now += .25
        return {"uuid": uuid, "free_mib": free, "read_only": True,
                "neighbors": [{"pid": 999999, "used_mib": "15000"}]}
    def popen(command, **kwargs):
        index = len(jobs)
        job = frozen["jobs"][index]
        assert command[:4] == [str(protocol.ROOT / ".venv/bin/python"), "-B", "-m", "training.r7_v2_worker"]
        assert kwargs["cwd"] == protocol.ROOT and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixed"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "4" and "PYTHONPATH" not in kwargs["env"]
        assert float(command[command.index("--deadline") + 1]) == 10790
        jobs.append(job)
        process = FakeProcess(clock, seconds=seconds, code=2 if index == fail_at else 0,
                              timeout=index == timeout_at, cleanup_fail=cleanup_fail)
        processes.append(process)
        if index != fail_at and index != timeout_at:
            entry = {"status": "success", "job": job, "scientific_claim": False, "limitations": ["CPU engineering fixture"],
                     "test_read": False, "protocol_sha256": frozen["protocol_sha256"],
                     "model_code_sha256": frozen["code"]["model_code_sha256"],
                     "source_tree_sha256": frozen["code"]["source_tree_sha256"], "code_zip_sha256": frozen["code"]["code_zip_sha256"],
                     "data_identity": frozen["data"]["data_identity"], "source_sha256": frozen["sources"]["source_sha256"],
                     "sidecar_identity": frozen["sidecar"]["identity"], "windows_sha256": protocol.digest(frozen["windows"]),
                     "baseline": {"allocated_bytes": 0, "reserved_bytes": 0}, "elapsed_seconds": seconds,
                     "peak_allocated_bytes": 100, "peak_reserved_bytes": peak_mib * 2 ** 20,
                     "updates_run": frozen["arm_configs"][job["arm"]]["updates"]}
            if result_change:
                entry.update(result_change)
            protocol.write_json(protocol.worker_result_path(frozen["output"], job), entry, output=frozen["output"])
        return process
    def aggregate(output, value, execution):
        aggregates.append(deepcopy(execution))
        assert execution["status"] == "results-complete" and execution["finalized"] is False
        assert execution["jobs_completed"] == frozen["jobs"]
        assert not (output / "attempt.json").exists() and not (output / "execution_attempt.json").exists()
        clock.now += 5
        return {"scientific_claim": False, "limitations": ["descriptive fixture"], "paused": False}
    return clock, jobs, processes, queries, aggregates, snapshot, popen, aggregate


def _run(frozen, fake, **kwargs):
    clock, _, _, _, _, snapshot, popen, aggregate = fake
    return driver.run_bounded_round(frozen["output"], clock=clock, snapshot_fn=snapshot, popen_factory=popen,
                                    finalize_fn=kwargs.pop("finalize_fn", aggregate), **kwargs)


def _attempt(frozen):
    return protocol.read_json(Path(frozen["output"]) / "attempt.json")


def test_b_independent_inventory_budget_and_contract(frozen):
    assert protocol.verify_protocol(Path(frozen["output"]) / "protocol.json") == frozen
    assert len(frozen["jobs"]) == 36 and all(job["phase"] == "train" for job in frozen["jobs"][:6])
    assert all(job["phase"] == "evaluate" and job["reasoning_steps"] == 4 for job in frozen["jobs"][6:])
    assert frozen["planned_seconds"] == 5400 and frozen["hard_cap_seconds"] == 10800
    assert frozen["arm_configs"]["equal_compute_l6"]["updates"] == 400
    contract = protocol.child_contract(frozen, frozen["jobs"][0], {"detach_between_steps": False})
    assert contract["autoregression"]["excluded_sample_ids"] == ["boundary"]
    assert contract["autoregression"]["windows"] == frozen["windows"] and contract["autoregression"]["lambda12"] == 0
    rollout_contract = protocol.child_contract(frozen, frozen["jobs"][1], {"detach_between_steps": False})
    assert rollout_contract["autoregression"]["lambda12"] == .5
    assert "process_supervision" not in contract and frozen["reporting"] == protocol.B_REPORTING
    assert "deep supervision per physical step" in frozen["shared_controls"]["gradient_recipe"]
    assert frozen["baseline_reporting"]["required"] == ["persistence", "climatology"]


def test_c_independent_configuration_inventory_and_science_fields():
    configuration = _configuration()
    jobs = protocol.planned_jobs("C", configuration)
    assert len(jobs) == 144 and sum(job["phase"] == "train" for job in jobs) == 9
    assert {job["reasoning_steps"] for job in jobs[9:]} == {1, 2, 4}
    assert protocol.BUDGETS["C"] == (10800, 21600)
    for field in ("primary", "tolerances", "case_unit_selection", "adaptive_gate", "reference", "selection_evidence"):
        changed = deepcopy(configuration)
        changed[field] = None
        with pytest.raises(ValueError, match="nonempty"):
            protocol.planned_jobs("C", changed)
    with pytest.raises(ValueError):
        protocol.planned_jobs("C")


@pytest.mark.parametrize("field,value", [("hard_cap_seconds", 1800), ("hard_cap_seconds", 10801),
    ("planned_seconds", 1800), ("cleanup_reserve_seconds", 0), ("test_read", True),
    ("scientific_claim", True), ("no_total_gpu_cap", False), ("no_retry_or_resurrection", False)])
def test_rehashed_changed_budgets_or_flags_are_refused(frozen, field, value):
    changed = deepcopy(frozen)
    changed[field] = value
    with pytest.raises(ValueError):
        protocol.validate_protocol(_refreeze(changed))


@pytest.mark.parametrize("kind", ["jobs", "uuid", "margin", "policy", "old_cap", "reporting", "windows", "parents"])
def test_frozen_recipe_jobs_and_identity_guard_counterproofs(frozen, kind):
    changed = deepcopy(frozen)
    if kind == "jobs": changed["jobs"].reverse()
    elif kind == "uuid": changed["gpu"]["uuid"] = "GPU-a,GPU-b"
    elif kind == "margin": changed["gpu"]["headroom_margin_mib"] = 0
    elif kind == "policy": changed["gpu"]["policy"] = "exclusive"
    elif kind == "old_cap": changed["gpu"]["max_gpu_seconds"] = 1800
    elif kind == "reporting": changed["reporting"]["primary"]["degradation_tolerance"] = .5
    elif kind == "windows": changed["windows"]["excluded_sample_ids"] *= 2
    else: changed["parents"].pop("42")
    with pytest.raises(ValueError): protocol.validate_protocol(_refreeze(changed))


def test_all_training_before_evaluation_continuous_cost_and_observed_gate(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch)
    fake[0].now = 100  # prepare/run gap is part of frozen whole clock.
    _run(frozen, fake)
    attempt = _attempt(frozen)
    assert fake[1] == frozen["jobs"] and len(fake[3]) == 36 and len(fake[4]) == 1
    assert attempt["started_perf_counter"] == 0 and attempt["run_entry_perf_counter"] == 100
    assert attempt["whole_elapsed_seconds"] == 160 and attempt["gpu_phase_elapsed_seconds"] == 44.75
    assert attempt["gpu_hours_charged"] == 44.75 / 3600 and attempt["finalized"]
    assert fake[4][0]["whole_elapsed_seconds"] == 155 and not fake[4][0]["finalized"]
    assert attempt["headroom_checks"][0]["snapshot"]["required_free_mib"] == 4096
    assert attempt["headroom_checks"][1]["snapshot"]["required_free_mib"] == 4148
    assert all(not process.signals for process in fake[2])


def test_soft_overrun_continues_all_and_records_overrun(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch, cpu_seconds=5401)
    _run(frozen, fake)
    attempt = _attempt(frozen)
    assert len(fake[1]) == 36 and attempt["status"] == "success"
    assert attempt["soft_overrun_seconds"] == attempt["whole_elapsed_seconds"] - 5400
    assert not attempt["budget_limited"] and attempt["soft_budget_exceeded"]


@pytest.mark.parametrize("seconds", [10790, 10801])
def test_whole_cap_includes_cpu_prepare_gap_and_stops_before_spawn(frozen, monkeypatch, seconds):
    fake = _fake(frozen, monkeypatch, cpu_seconds=seconds)
    with pytest.raises(driver.BudgetLimited): _run(frozen, fake)
    assert not fake[1] and not fake[4]
    assert _attempt(frozen)["gpu_hours_charged"] == 0 and _attempt(frozen)["budget_limited"]


def test_cpu_identity_failure_stops_without_gpu_and_no_resurrection(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch, cpu_error=True)
    with pytest.raises(ValueError, match="input pin"): _run(frozen, fake)
    assert not fake[1] and _attempt(frozen)["jobs_completed"] == []
    with pytest.raises(FileExistsError, match="no retry"): _run(frozen, fake)
    assert _attempt(frozen)["status"] == "failed"


def test_any_worker_failure_stops_exact_endpoint_without_retry(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="exit 2; stop without retry"): _run(frozen, fake)
    attempt = _attempt(frozen)
    assert len(fake[1]) == 3 and len(attempt["jobs_completed"]) == 2 and not fake[4]
    assert attempt["failed_job_key"] == protocol.job_key(frozen["jobs"][2])
    assert attempt["gpu_phase_elapsed_seconds"] == 3.5 and attempt["partial"] and not attempt["finalized"]
    with pytest.raises(FileExistsError): _run(frozen, fake)


def test_timeout_only_reaps_owned_handle_and_charges_cleanup(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch, timeout_at=0)
    with pytest.raises(driver.BudgetLimited, match="hard timeout"): _run(frozen, fake)
    assert len(fake[1]) == 1 and fake[2][0].signals == ["terminate-owned"] and fake[2][0].done
    assert _attempt(frozen)["whole_elapsed_seconds"] == 10791 and not fake[4]
    assert _attempt(frozen)["gpu_phase_elapsed_seconds"] == 10780.75


def test_cleanup_and_receipt_errors_do_not_mask_original_or_undercharge(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch, timeout_at=0, cleanup_fail=True)
    original = driver.write_json
    def write(path, value, **kwargs):
        if path.name.endswith(".timing.json"): raise OSError("timing publication denied fixture")
        return original(path, value, **kwargs)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(driver.BudgetLimited, match="hard timeout") as caught: _run(frozen, fake)
    assert any("unreaped" in note for note in caught.value.__notes__)
    assert any("publication denied" in note for note in caught.value.__notes__)
    assert caught.value.v2_attempt["owned_unreaped"] and caught.value.v2_attempt["status"] == "failed"


@pytest.mark.parametrize("kind", ["low", "uuid", "actual_peak"])
def test_every_spawn_headroom_uuid_and_peak_update_fail_closed(frozen, monkeypatch, kind):
    fake = _fake(frozen, monkeypatch, free=2000 if kind == "low" else 4096 if kind == "actual_peak" else 9999)
    if kind == "uuid":
        fake = (*fake[:5], lambda uuid: {"uuid": "GPU-other", "free_mib": 9999}, *fake[6:])
    with pytest.raises(RuntimeError, match="headroom"): _run(frozen, fake)
    assert len(fake[1]) == (1 if kind == "actual_peak" else 0) and not fake[4]
    assert not any(process.signals for process in fake[2])


@pytest.mark.parametrize("change", [{"status": "skipped"}, {"protocol_sha256": "bad"},
    {"windows_sha256": "bad"}, {"peak_reserved_bytes": -1}, {"peak_allocated_bytes": float("inf")},
    {"baseline": {"allocated_bytes": 1, "reserved_bytes": 0}}, {"updates_run": 199}])
def test_malformed_partial_or_skipped_child_result_stops(frozen, monkeypatch, change):
    fake = _fake(frozen, monkeypatch, result_change=change)
    # JSON forbids infinity before launch; other cases reach receipt verification.
    with pytest.raises(ValueError): _run(frozen, fake)
    assert len(fake[1]) == 1 and not fake[4] and _attempt(frozen)["status"] == "failed"


@pytest.mark.parametrize("kind", ["error", "late", "claim", "missing"])
def test_aggregation_failure_never_seals_success(frozen, monkeypatch, kind):
    fake = _fake(frozen, monkeypatch)
    def aggregate(*args):
        if kind == "error": raise ValueError("aggregate fixture error")
        if kind == "late": fake[0].now = 10801
        if kind == "missing": return {}
        return {"scientific_claim": kind == "claim", "limitations": ["CPU fixture"]}
    with pytest.raises((ValueError, driver.BudgetLimited)): _run(frozen, fake, finalize_fn=aggregate)
    assert len(fake[1]) == 36 and _attempt(frozen)["status"] == "failed" and not _attempt(frozen)["finalized"]


def test_failed_final_publication_keeps_original_error_and_full_cost(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch, fail_at=0)
    original = driver.write_json
    def write(path, value, **kwargs):
        if path.name == "execution_attempt.json": raise OSError("final receipt denied fixture")
        return original(path, value, **kwargs)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(RuntimeError, match="exit 2") as caught: _run(frozen, fake)
    assert caught.value.v2_attempt["gpu_hours_charged"] == 1 / 3600
    assert any("receipt denied" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("name", ["workers", "run_started.json", "attempt.json", "unexpected.pt"])
def test_partial_attempt_is_not_repaired(frozen, name):
    path = Path(frozen["output"]) / name
    path.mkdir() if name == "workers" else path.write_text("opaque evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="no retry"):
        driver.run_bounded_round(frozen["output"], snapshot_fn=lambda _: pytest.fail("must not query GPU"))
    assert path.exists()


def test_cross_boot_and_future_anchor_are_refused_before_any_spawn(frozen, monkeypatch):
    fake = _fake(frozen, monkeypatch)
    monkeypatch.setattr(driver, "monotonic_boot_id", lambda: "different-boot")
    with pytest.raises(ValueError, match="same-boot"): _run(frozen, fake)
    assert not fake[1] and _attempt(frozen)["gpu_hours_charged"] == 0


@pytest.mark.parametrize("prefix", ["data/raw", "data/processed", "data/interim", "legacy_v6", "outputs/original"])
def test_protected_nested_or_old_outputs_refused_without_finally_writes(tmp_path, prefix):
    target = tmp_path / prefix / protocol.DEFAULT_OUTPUT.name
    with pytest.raises(ValueError):
        driver.prepare(tmp_path / "missing", tmp_path / "missing", target, "GPU-fixed")
    assert not target.exists()
    with pytest.raises(ValueError): driver.run_bounded_round(tmp_path / "outputs/r7_73_process_supervision")


def test_symlink_ancestor_and_worker_output_symlink_refused_without_writes(frozen, tmp_path):
    parent, alias = tmp_path / "actual", tmp_path / "alias"
    parent.mkdir()
    alias.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        driver.prepare(tmp_path / "missing", tmp_path / "missing", alias / protocol.DEFAULT_OUTPUT.name, "GPU-fixed")
    assert not list(parent.iterdir())
    output = Path(frozen["output"])
    (output / "workers").symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        protocol.write_json(output / "workers/forbidden.json", {}, output=output)
    assert not list(parent.iterdir())


def test_socket_denial_both_entrypoints_and_fresh_allocator_counterproofs():
    worker.deny_network()
    with pytest.raises(RuntimeError, match="offline"): socket.create_connection(("example.invalid", 443))
    with pytest.raises(RuntimeError, match="offline"): socket.socket.connect(None, ("example.invalid", 443))
    calls = []
    cuda = SimpleNamespace(memory_allocated=lambda _: 0, memory_reserved=lambda _: 0,
                           init=lambda: calls.append("init"), set_device=lambda _: calls.append("device"),
                           reset_peak_memory_stats=lambda _: calls.append("reset"))
    assert worker.zero_allocator_baseline("cuda:0", torch_module=SimpleNamespace(cuda=cuda)) == {"allocated_bytes": 0, "reserved_bytes": 0}
    assert calls == ["init", "device", "reset"]
    cuda.memory_allocated = lambda _: 1
    with pytest.raises(ValueError, match="pre-init zero"): worker.zero_allocator_baseline("cuda:0", torch_module=SimpleNamespace(cuda=cuda))
    cuda.memory_allocated = lambda _: 0
    cuda.memory_reserved = lambda _: 1 if len(calls) > 3 else 0
    with pytest.raises(ValueError, match="remain zero"): worker.zero_allocator_baseline("cuda:0", torch_module=SimpleNamespace(cuda=cuda))


def test_stdlib_worker_bootstrap_denies_before_project_imports_in_subprocess(tmp_path):
    code = """import importlib.util,sys,socket
from pathlib import Path
spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1]); module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name.startswith(('training.','data.','model.')) for name in sys.modules)
sys.path.insert(0,str(Path(sys.argv[1]).resolve().parents[1]))
module.deny_network()
assert not any(name.startswith(('training.','data.','model.')) for name in sys.modules)
for call in (lambda:socket.create_connection(('example.invalid',443)),lambda:socket.socket.connect(None,('example.invalid',443))):
    try: call()
    except RuntimeError: pass
    else: raise AssertionError('outbound denial absent')
print('cpu-offline-bootstrap')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(protocol.ROOT / "training/r7_v2_worker.py")],
                            cwd=tmp_path, text=True, capture_output=True, timeout=10, env=driver.worker_environment(""))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cpu-offline-bootstrap"


def test_source_closure_archive_excludes_outputs_tests_and_rejects_changed_helpers(tmp_path, monkeypatch):
    root, output = tmp_path / "sources", tmp_path / protocol.DEFAULT_OUTPUT.name
    output.mkdir()
    roots = ("training/r7_v2_identity.py", "training/r7_v2_results.py")
    for name in (*roots, "training/__init__.py", "training/r7_v2_cells.py", "model/__init__.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU source fixture\n", encoding="utf-8")
    (root / roots[1]).write_text("from .r7_v2_cells import combine\n", encoding="utf-8")
    monkeypatch.setattr(identity, "ROOT", root)
    monkeypatch.setattr(identity, "ARCHIVE_ROOTS", roots)
    monkeypatch.setattr(identity.subprocess, "check_output", lambda command, **kwargs:
                        "a" * 40 + "\n" if "rev-parse" in command else " M unrelated_owner.py\n")
    code = identity.archive_code(output)
    assert "training/r7_v2_cells.py" in code["files"] and code["working_tree_modified"]
    assert code["base_commit"] == "a" * 40 and "unrelated_owner.py" not in code["files"]
    assert all(not any(part in ("tests", "outputs", "manifests") for part in Path(name).parts) for name in code["files"])
    value = {"output": str(output), "code": code}
    identity.verify_code(value)
    # Unrelated dirtiness can change without widening or invalidating the closure.
    monkeypatch.setattr(identity.subprocess, "check_output", lambda command, **kwargs:
                        "a" * 40 + "\n" if "rev-parse" in command else " M another_owner.py\n")
    identity.verify_code(value)
    monkeypatch.setattr(identity.subprocess, "check_output", lambda *args, **kwargs: "b" * 40 + "\n")
    with pytest.raises(ValueError, match="HEAD"): identity.verify_code(value)
    monkeypatch.setattr(identity.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    saved_status = (output / "code_status.txt").read_text(encoding="utf-8")
    (output / "code_status.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Git identity"): identity.verify_code(value)
    (output / "code_status.txt").write_text(saved_status, encoding="utf-8")
    (root / "training/r7_v2_cells.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"): identity.verify_code(value)
    with zipfile.ZipFile(output / "code.zip", "a") as archive: archive.writestr("tests/test_unrelated.py", "# no\n")
    with pytest.raises(ValueError): identity.verify_code(value)


def test_prepare_failure_seals_only_new_safe_output_and_no_real_inputs(tmp_path, monkeypatch):
    output = tmp_path / protocol.DEFAULT_OUTPUT.name
    monkeypatch.setattr(driver, "pin_inputs", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("preflight failure fixture")))
    # The subject here is the seal path, not GPU state. On a GPU host an earlier
    # test in the same process may already have initialized CUDA, and prepare()'s
    # "CPU prepare must not initialize CUDA" guard would then fire before the
    # fixture error; that guard is exercised by its own test. Answer the query
    # with the CPU-clean precondition so this test keeps asserting exactly what
    # it always did: the failure receipt, the charge of zero and the refusal.
    import torch
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(ValueError, match="preflight failure"): driver.prepare(tmp_path / "manifests", tmp_path / "scale", output, "GPU-fixed")
    receipt = protocol.read_json(output / "attempt.json")
    assert receipt["status"] == "failed" and receipt["gpu_hours_charged"] == 0 and not receipt["finalized"]
    assert not (output / "protocol.json").exists()
    with pytest.raises(FileExistsError): driver.prepare(tmp_path / "manifests", tmp_path / "scale", output, "GPU-fixed")


def test_unclaimed_execute_and_unsafe_worker_protocol_never_write(frozen, tmp_path):
    with pytest.raises(ValueError, match="claimed run"):
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json")
    unsafe = tmp_path / "outputs/r7_73_process_supervision"
    unsafe.mkdir(parents=True)
    with pytest.raises(ValueError, match="independent"):
        worker.run_worker(unsafe / "protocol.json", frozen["jobs"][0], 1.)
    assert list(unsafe.iterdir()) == []


def _baseline_schema(value, job, initializations):
    rows = [{"baseline_kind": kind, "region": region, "variable": channel, "unit": unit,
             "lead_hours": job["lead"], "n_initializations": 1, "zero_train_updates": 0,
             "parameters": 0, "trainable_parameters": 0}
            for kind in ("persistence", "climatology") for region in protocol.REGIONS
            for channel, unit in zip(value["data"]["channels"], value["data"]["units"])]
    return {"climatology": {"kind": "train-only-month-hour-grid-mean-v1"}, "baseline_definition": {
            "persistence": {"kind": "known-last-history-frame", "training_updates": 0},
            "climatology": {"kind": "train-only-month-hour", "training_updates": 0},
            "case_pairing": "same exact requested-lead initializations and full/interior/edge_2"},
            "baseline_region_metrics": [{**row, "n_initializations": len(initializations)} for row in rows],
            "baseline_initializations": [{**item, "region_metrics": deepcopy(rows)} for item in initializations]}


def test_actual_child_evaluation_sidecar_guard_and_worker_forwarding(frozen, monkeypatch):
    from training import r7_v2_evaluation as evaluation
    from training.r7_process_training_contract import verify_evaluation_sidecar
    assert verify_evaluation_sidecar({}, None) is None
    with pytest.raises(ValueError, match="sidecar override"):
        verify_evaluation_sidecar({}, "/forbidden-scale")
    output, job = Path(frozen["output"]), frozen["jobs"][6]
    protocol.make_directory(output / "workers", output)
    train_job = frozen["jobs"][0]
    directory = protocol.make_directory(protocol.train_output_dir(output, job["seed"], job["arm"]), output)
    checkpoint = directory / "update_0000200.pt"
    checkpoint.write_bytes(b"opaque CPU checkpoint fixture")
    entry = {"status": "success", "job": train_job, "protocol_sha256": frozen["protocol_sha256"],
             "checkpoint": str(checkpoint), "checkpoint_sha256": protocol.sha256_file(checkpoint)}
    protocol.write_json(protocol.worker_result_path(output, train_job), entry, output=output)
    def evaluate(manifest, selected, folder, **kwargs):
        # Call the production contract gate, not a permissive substitute.
        assert verify_evaluation_sidecar({}, kwargs["process_scale_sidecar"]) is None
        assert selected == checkpoint and manifest.name == "val.jsonl" and kwargs["max_samples"] == 32
        protocol.make_directory(folder, output)
        cases = frozen["data"]["evaluation_cases"][str(job["lead"])]["cases"]
        provenance = {"split": "val", "test_read": False, "scientific_claim": False,
            "channels": frozen["data"]["channels"], "units": frozen["data"]["units"],
            "training_identity": frozen["data"]["data_identity"], "n_evaluated": len(cases),
            "n_available_windows": 1, "lead_hours": [job["lead"]], "reasoning_steps": 4,
            "initializations": [{"sample_id": "fixture-case", "init_time": init, "valid_times": times} for init, times in cases]}
        provenance.update(_baseline_schema(frozen, job, provenance["initializations"]))
        for name in ("region_metrics.csv", "per_case_metrics.csv", "baseline_region_metrics.csv", "baseline_per_case_metrics.csv"):
            (folder / name).write_text("CPU schema fixture\n", encoding="utf-8")
        protocol.write_json(folder / "provenance.json", provenance, output=output)
        return provenance
    monkeypatch.setattr(evaluation, "evaluate_v2", evaluate)
    result = worker.evaluate_worker(frozen, job, driver.time.perf_counter() + 30)
    assert result["n_evaluated"] == 1 and result["reasoning_steps"] == 4
    assert set(result["artifact_sha256"]) == {"region_metrics.csv", "per_case_metrics.csv", "provenance.json",
                                               "baseline_region_metrics.csv", "baseline_per_case_metrics.csv"}
    assert Path(result["baseline_region_metrics_csv"]).name == "baseline_region_metrics.csv"
    assert frozen["sidecar"]["identity"] == "i" * 64  # Parent/source identity was not removed.


def test_input_and_parent_pin_counterproofs(frozen, tmp_path, monkeypatch):
    pinned = (frozen["data"], frozen["sources"], frozen["sidecar"], frozen["windows"], None, None, None)
    verifier = identity.verify_parent_pins
    monkeypatch.setattr(identity, "pin_inputs", lambda *args, **kwargs: pinned)
    monkeypatch.setattr(identity, "verify_parent_pins", lambda *args, **kwargs: None)
    assert identity.verify_input_pins(frozen) == pinned
    changed = deepcopy(frozen)
    changed["windows"]["window_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact windows changed"): identity.verify_input_pins(changed)
    path = tmp_path / "opaque_parent.pt"
    path.write_bytes(b"accepted parent CPU identity")
    original = tmp_path / "parent_protocol.json"
    original.write_text(json.dumps({"protocol_sha256": "a" * 64}), encoding="utf-8")
    parent = {"checkpoint": str(path), "file_sha256": {"checkpoint": protocol.sha256_file(path)},
              "import_report": {"resume": False}, "import_report_sha256": protocol.digest({"resume": False}),
              "original_protocol": str(original), "original_protocol_sha256": "a" * 64}
    # Exercise the production parent verifier despite the input-isolation stub above.
    verifier({"parents": {"41": parent}})
    path.write_bytes(b"tampered parent")
    with pytest.raises(ValueError, match="parent input changed"): verifier({"parents": {"41": parent}})


def test_profile_counts_real_one_two_forward_and_backward_cpu(monkeypatch):
    import torch
    from training import r7_v2_profile as profile
    class ToyObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
        def forward(self, value):
            return self.linear(value)
    def objective(model, batch, mode, **kwargs):
        first = model(batch)
        second = model(first) if mode == "two_step" else None
        l6 = first.float().square().mean()
        l12 = None if second is None else second.float().square().mean()
        return SimpleNamespace(loss=l6 if l12 is None else l6 + .5 * l12, l6=l6, l12=l12)
    monkeypatch.setattr(profile, "objective", objective)
    model = ToyObjective()
    batch = torch.ones(1, 2)
    one = profile.measure_cost(model, batch, "l6")
    two = profile.measure_cost(model, batch, "two_step")
    assert one["actual_model_forward_calls"] == 1 and two["actual_model_forward_calls"] == 2
    assert two["forward_flops"] > one["forward_flops"] > 0
    assert two["forward_backward_flops"] > two["forward_flops"] and two["gradient_norm"] > 0
    monkeypatch.setattr(profile, "objective", lambda model, batch, mode: objective(model, batch, "two_step"))
    with pytest.raises(ValueError, match="exactly one/two"): profile.measure_cost(model, batch, "l6")


@pytest.mark.parametrize("stage", ["B", "C"])
def test_baseline_reporting_mandatory_before_freeze_without_added_jobs(frozen, stage):
    value = deepcopy(frozen)
    if stage == "C":
        value.update(stage=stage, configuration=_configuration(), parents={}, planned_seconds=10800, hard_cap_seconds=21600)
        value.update(jobs=protocol.planned_jobs(stage, value["configuration"]), arm_configs=protocol.arm_configs(stage, value["configuration"]))
    assert protocol.validate_protocol(_refreeze(value))["baseline_reporting"] == protocol.BASELINE_REPORTING
    assert len(value["jobs"]) == (36 if stage == "B" else 144)
    value["baseline_reporting"]["required"].remove("persistence")
    with pytest.raises(ValueError, match="mandatory B/C"): protocol.validate_protocol(_refreeze(value))


@pytest.mark.parametrize("change", ["missing", "trained", "cases", "region", "variable", "duplicate"])
def test_missing_or_unpaired_baseline_provenance_refused(frozen, change):
    job = frozen["jobs"][6]
    inits = [{"sample_id": "fixture-case", "init_time": "2016-02-17T00:00:00", "valid_times": ["2016-02-17T06:00:00"]}]
    value = {"initializations": inits, **_baseline_schema(frozen, job, inits)}
    if change == "missing": value.pop("baseline_definition")
    elif change == "trained": value["baseline_definition"]["persistence"]["training_updates"] = 1
    elif change == "cases": value["baseline_initializations"][0]["sample_id"] = "wrong-case"
    elif change == "region": value["baseline_region_metrics"][0]["region"] = "narrowed"
    elif change == "variable": value["baseline_initializations"][0]["region_metrics"].pop()
    else: value["baseline_region_metrics"][0] = value["baseline_region_metrics"][1]
    with pytest.raises(ValueError): worker.verify_baseline_provenance(frozen, job, value)
