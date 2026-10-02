"""CPU/tmp_path counterproofs for isolated M3 complement scheduling, pins and failure receipts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from scripts import r7_m3_complement_worker as worker
from training import r7_m3_complement_driver as driver
from training import r7_m3_complement_protocol as protocol


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now


class FakeProcess:
    def __init__(self, clock, seconds=1.0, code=0, timeout=False, cleanup_fail=False):
        self.clock, self.seconds, self.code, self.timeout = clock, seconds, code, timeout
        self.cleanup_fail, self.done, self.signals, self.pid = cleanup_fail, False, [], 12345
    def wait(self, timeout):
        if self.done:
            return self.code
        if self.cleanup_fail and self.signals:
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


def _refreeze(value):
    value["protocol_sha256"] = protocol.digest({key: item for key, item in value.items() if key != "protocol_sha256"})
    return value


@pytest.fixture
def frozen(tmp_path):
    old, output = tmp_path / "original", tmp_path / "complement"
    old.mkdir()
    output.mkdir()
    original = {"protocol_sha256": "a" * 64, "gpu": {"uuid": "GPU-fixed"},
                "code": {"code_zip_sha256": "b" * 64, "source_tree_sha256": "c" * 64,
                         "model_code_sha256": "d" * 64}}
    (old / "protocol.json").write_text(json.dumps(original), encoding="utf-8")
    pins = {name: "1" * 64 for name in protocol.original_file_names()}
    pins["code.zip"] = "b" * 64
    value = protocol.build_protocol(output=output, original_output=old, original=original,
                                    original_files=pins, code={"source_root": str(tmp_path / "sources")},
                                    round_started_perf_counter=0.0, boot_id=protocol.monotonic_boot_id())
    protocol.write_json(output / "protocol.json", value)
    return value


def _fake_round(frozen, monkeypatch, *, cpu_seconds=10.0, seconds=1.0, fail_at=None,
                timeout_at=None, free=9999, malformed=False, cpu_error=False, cleanup_fail=False):
    clock, spawned, processes, snapshots, aggregate_calls = FakeClock(), [], [], [], []
    def preflight(path, value, deadline, execution, **kwargs):
        clock.now += cpu_seconds
        driver.check_budget(deadline, clock=clock)
        if cpu_error:
            raise ValueError("CPU identity mismatch fixture")
        (Path(value["output"]) / "archived_code").mkdir()
    monkeypatch.setattr(driver, "_runtime_preflight", preflight)
    monkeypatch.setattr(driver, "_runtime_postflight", lambda *args, **kwargs: None)
    def snapshot(uuid):
        snapshots.append(uuid)
        clock.now += .25
        return {"uuid": uuid, "free_mib": free, "neighbors": [{"pid": 999999, "used_mib": "15000"}], "read_only": True}
    def popen(command, **kwargs):
        index = len(spawned)
        job = frozen["jobs"][index]
        assert kwargs["cwd"] == Path(frozen["output"]) / "archived_code"
        assert command[3] == str(protocol.ROOT / "scripts" / "r7_m3_complement_worker.py")
        assert command[:3] == [str(protocol.ROOT / ".venv" / "bin" / "python"), "-I", "-B"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixed"
        assert "PYTHONPATH" not in kwargs["env"]
        assert float(command[command.index("--deadline") + 1]) == 3590.0
        spawned.append(job)
        process = FakeProcess(clock, seconds=seconds, code=2 if index == fail_at else 0,
                              timeout=index == timeout_at, cleanup_fail=cleanup_fail)
        processes.append(process)
        if index != fail_at and index != timeout_at:
            original = protocol.read_json(Path(frozen["original_output"]) / "protocol.json")
            entry = {"status": "success", "scientific_claim": False, "limitations": ["CPU engineering fixture"],
                     "test_read": False, "job": job, "seed": job["seed"], "arm": job["arm"], "lead_hours": job["lead"],
                     "protocol_sha256": frozen["original_protocol_sha256"],
                     "complement_protocol_sha256": frozen["protocol_sha256"], "split": "val",
                     "evaluation_dir": str(protocol.evaluation_dir(frozen["output"], job)),
                     "peak_allocated_bytes": 1000, "peak_reserved_bytes": 350 * 2 ** 20, "elapsed_seconds": seconds,
                     "original_training_receipt_sha256": "1" * 64,
                     "archived_evaluator_code": original["code"]}
            if malformed:
                entry["status"] = "skipped"
            protocol.write_json(Path(frozen["output"]) / "workers" / (protocol.job_key(job) + ".json"), entry)
        return process
    def finalize(output, value, execution):
        from training.r7_m3_complement_results import _execution
        _execution(execution, output, value)  # Actual shared schema, not just a permissive mock.
        aggregate_calls.append(deepcopy(execution))
        assert execution["status"] == "evaluations-complete" and not execution["finalized"]
        assert execution["jobs_completed"] == value["jobs"]
        assert not (output / "execution_attempt.json").exists()
        clock.now += 5
        return {"paused": False, "scientific_claim": False, "limitations": ["frozen descriptive fixture"]}
    return clock, spawned, processes, snapshots, aggregate_calls, snapshot, popen, finalize


def _run(frozen, fake, **kwargs):
    clock, _, _, _, _, snapshot, popen, finalize = fake
    return driver.run_bounded_round(frozen["output"], clock=clock, snapshot_fn=snapshot,
                                    popen_factory=popen, finalize_fn=kwargs.pop("finalize_fn", finalize), **kwargs)


def test_protocol_exact23_no_training_and_static109(frozen):
    assert protocol.verify_protocol(Path(frozen["output"]) / "protocol.json") == frozen
    assert len(frozen["jobs"]) == len(protocol.planned_jobs()) == 23
    assert all(job["phase"] == "evaluate" for job in frozen["jobs"])
    assert frozen["jobs"][0] == {"phase": "evaluate", "seed": 41, "arm": "input_aux", "lead": 24}
    assert len(protocol.original_file_names()) == len(set(protocol.original_file_names())) == 109
    assert "workers/evaluate_seed41_input_aux_lead024h.log" in protocol.original_file_names()
    assert frozen["planned_seconds"] == 1800 and frozen["hard_cap_seconds"] == 3600
    assert frozen["cleanup_reserve_seconds"] == 10 and frozen["training_updates"] == 0


@pytest.mark.parametrize("field,value", [("planned_seconds", 1801), ("hard_cap_seconds", 1800),
    ("cleanup_reserve_seconds", 0), ("training_updates", 1), ("test_read", True), ("scientific_claim", True)])
def test_rehashed_changes_to_frozen_budget_flags_refused(frozen, field, value):
    changed = deepcopy(frozen)
    changed[field] = value
    with pytest.raises(ValueError, match="frozen"):
        protocol.validate_protocol(_refreeze(changed))


@pytest.mark.parametrize("change", ["jobs", "pins", "shared", "estimate", "margin", "uuid", "nested"])
def test_changed_job_inventory_paths_or_shared_contract_refused(frozen, change):
    value = deepcopy(frozen)
    if change == "jobs": value["jobs"].pop()
    elif change == "pins": value["original_file_sha256"].pop("attempt.json")
    elif change == "shared": value["gpu"]["policy"] = "exclusive"
    elif change == "estimate": value["gpu"]["estimated_peak_mib"] = 1
    elif change == "margin": value["gpu"]["headroom_margin_mib"] = 0
    elif change == "uuid": value["gpu"]["uuid"] = "GPU-a,GPU-b"
    else: value["output"] = value["original_output"] + "/complement"
    with pytest.raises(ValueError):
        protocol.validate_protocol(_refreeze(value))


def test_whole_deadline_includes_cpu_and_cli_entry_gpu_continuous(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, cpu_seconds=100)
    fake[0].now = 20  # CLI captured zero before parent imports.
    outcome = _run(frozen, fake, started_perf_counter=0.0)
    execution = protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")
    assert not outcome["paused"] and len(fake[1]) == len(fake[3]) == 23
    assert execution["status"] == "success" and execution["finalized"]
    assert execution["whole_elapsed_seconds"] == 153.75
    assert execution["gpu_phase_elapsed_seconds"] == 28.5  # excludes first headroom query, includes all later gaps.
    assert execution["gpu_hours_charged"] == 28.5 / 3600
    assert fake[4][0]["whole_elapsed_seconds"] == 148.75
    assert all(not process.signals for process in fake[2])
    assert len(execution["headroom_checks"]) == 23
    assert execution["headroom_checks"][0]["snapshot"]["neighbors"][0]["pid"] == 999999


def test_soft_overrun_continues_all23_and_aggregation(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, cpu_seconds=1801)
    _run(frozen, fake)
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert len(fake[1]) == 23 and len(fake[4]) == 1
    assert attempt["status"] == "success" and not attempt["budget_limited"]
    assert attempt["soft_overrun_seconds"] == attempt["whole_elapsed_seconds"] - 1800


def test_scientific_pause_is_full_coverage_not_failed_partial(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch)
    def pause(*args):
        return {"paused": True, "scientific_claim": False, "limitations": ["frozen unresolved pause"], "advance_next_node": False}
    assert _run(frozen, fake, finalize_fn=pause)["paused"]
    for name in ("execution_attempt.json", "attempt.json"):
        entry = protocol.read_json(Path(frozen["output"]) / name)
        assert entry["status"] == "paused" and entry["finalized"] and entry["scientific_halt"]
        assert not entry["partial"] and not entry["budget_limited"] and len(entry["jobs_completed"]) == 23


@pytest.mark.parametrize("cpu_seconds", [3590, 3700])
def test_late_cpu_preflight_cannot_spawn_and_still_publishes_failure(frozen, monkeypatch, cpu_seconds):
    fake = _fake_round(frozen, monkeypatch, cpu_seconds=cpu_seconds)
    with pytest.raises(driver.BudgetLimited):
        _run(frozen, fake)
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert fake[1] == [] and fake[4] == []
    assert attempt["status"] == "failed" and attempt["budget_limited"]
    assert attempt["gpu_hours_charged"] == 0 and attempt["whole_elapsed_seconds"] == cpu_seconds


def test_cpu_identity_error_publishes_unambiguous_zero_gpu_receipt(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, cpu_error=True)
    with pytest.raises(ValueError, match="CPU identity"):
        _run(frozen, fake)
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert attempt["status"] == "failed" and not attempt["finalized"]
    assert attempt["jobs_completed"] == [] and attempt["gpu_hours_charged"] == 0
    assert not attempt["budget_limited"] and fake[1] == []


def test_real_failure_stops_exact_key_without_retry(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="exit 2; stop without retry"):
        _run(frozen, fake)
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert len(fake[1]) == 3 and len(attempt["jobs_completed"]) == 2 and fake[4] == []
    assert attempt["failed_job_key"] == protocol.job_key(frozen["jobs"][2])
    assert attempt["partial"] and attempt["status"] == "failed"
    assert attempt["gpu_phase_elapsed_seconds"] == 3.5
    with pytest.raises(FileExistsError, match="already"):
        _run(frozen, fake)
    assert len(fake[1]) == 3


def test_timeout_reserves_cleanup_reaps_only_owned_and_charges_failure(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, timeout_at=0)
    with pytest.raises(driver.BudgetLimited):
        _run(frozen, fake)
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert len(fake[1]) == 1 and fake[2][0].signals == ["terminate-owned"] and fake[2][0].done
    assert attempt["whole_elapsed_seconds"] == 3591 and attempt["budget_limited"]
    assert attempt["gpu_phase_elapsed_seconds"] == 3580.75 and fake[4] == []
    timing = protocol.read_json(Path(frozen["output"]) / "workers" / (protocol.job_key(fake[1][0]) + ".timing.json"))
    assert timing["cleanup"] == "terminated-owned-worker"
    assert timing["last_owned_reap_perf_counter"] == 3591


def test_cleanup_and_timing_publication_do_not_mask_original_error(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, timeout_at=0, cleanup_fail=True)
    original_write = driver.write_json
    def write(path, value):
        if path.name.endswith(".timing.json"):
            raise OSError("timing publication denied fixture")
        return original_write(path, value)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(driver.BudgetLimited, match="hard timeout") as caught:
        _run(frozen, fake)
    assert len(fake[1]) == 1
    assert any("unreaped" in note for note in caught.value.__notes__)
    assert any("publication denied" in note for note in caught.value.__notes__)
    assert caught.value.m3_complement_execution_attempt["status"] == "failed"


@pytest.mark.parametrize("kind", ["low", "uuid", "skipped"])
def test_uuid_headroom_and_skipped_results_stop_without_aggregation(frozen, monkeypatch, kind):
    fake = _fake_round(frozen, monkeypatch, free=2000 if kind == "low" else 9999, malformed=kind == "skipped")
    if kind == "uuid":
        snapshot = fake[5]
        fake = (*fake[:5], lambda uuid: dict(snapshot(uuid), uuid="GPU-other"), *fake[6:])
    with pytest.raises((RuntimeError, ValueError)):
        _run(frozen, fake)
    assert len(fake[1]) == (1 if kind == "skipped" else 0)
    assert fake[4] == []
    assert protocol.read_json(Path(frozen["output"]) / "attempt.json")["status"] == "failed"


def test_aggregate_failure_and_late_aggregate_never_publish_success(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch)
    def finalize(*args):
        fake[0].now = 3601
        return {"paused": False, "scientific_claim": False, "limitations": ["CPU fixture"]}
    with pytest.raises(driver.BudgetLimited):
        _run(frozen, fake, finalize_fn=finalize)
    assert len(fake[1]) == 23
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert attempt["status"] == "failed" and not attempt["finalized"]
    assert attempt["whole_elapsed_seconds"] == 3601 and attempt["budget_limited"]


def test_attempt_publication_failure_preserves_original_and_charge(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, fail_at=0)
    original = driver.write_json
    def write(path, value):
        if path.name == "execution_attempt.json": raise OSError("execution publication denied fixture")
        return original(path, value)
    monkeypatch.setattr(driver, "write_json", write)
    with pytest.raises(RuntimeError, match="exit 2") as caught:
        _run(frozen, fake)
    assert caught.value.m3_complement_execution_attempt["gpu_hours_charged"] == 1 / 3600
    assert any("publication denied" in note for note in caught.value.__notes__)
    assert protocol.read_json(Path(frozen["output"]) / "attempt.json")["status"] == "failed"


@pytest.mark.parametrize("name", ["workers", "archived_code", "run_started.json", "execution_attempt.json"])
def test_partial_and_repeat_attempts_are_not_repaired(frozen, name):
    path = Path(frozen["output"]) / name
    path.mkdir() if name in ("workers", "archived_code") else path.write_text("partial evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="no rerun"):
        driver.run_bounded_round(frozen["output"], snapshot_fn=lambda _: pytest.fail("must not query GPU"))
    assert path.exists()


def _archive_fixture(tmp_path, *, unsafe=None, changed=False):
    old = tmp_path / "original_zip"
    old.mkdir()
    names = ["model/__init__.py", "training/r7_m3_protocol.py", "training/r7_m3_identity.py"]
    names += [f"training/source_{index:03d}.py" for index in range(77)]
    if unsafe is not None: names[-1] = unsafe
    contents = {name: b"# pinned CPU archive fixture\n" for name in names}
    files = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    with zipfile.ZipFile(old / "code.zip", "x") as archive:
        for name, value in contents.items():
            if unsafe == "training/symlink.py" and name == unsafe:
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, value)
            else:
                archive.writestr(name, b"changed" if changed and name == names[0] else value)
    model = hashlib.sha256(b"__init__.py\0" + contents["model/__init__.py"]).hexdigest()
    original = {"code": {"files": files, "source_tree_sha256": protocol.digest(files),
                         "model_code_sha256": model, "code_zip_sha256": protocol.sha256_file(old / "code.zip")}}
    return old, original


def test_safe_extraction_only_new_directory_and_member_pins(tmp_path):
    old, original = _archive_fixture(tmp_path)
    output = tmp_path / "complement_zip"
    output.mkdir()
    before = protocol.sha256_file(old / "code.zip")
    value = {"output": str(output), "original_output": str(old)}
    root = protocol.extract_original_archive(value, original)
    assert len(list(root.rglob("*.py"))) == 80
    assert protocol.sha256_file(old / "code.zip") == before
    assert all(protocol.sha256_file(root / name) == pin for name, pin in original["code"]["files"].items())
    with pytest.raises(FileExistsError): protocol.extract_original_archive(value, original)


@pytest.mark.parametrize("unsafe", ["../escape.py", "/absolute.py", "training/../escape.py", "training\\escape.py", "tests/test_hidden.py", "training/symlink.py"])
def test_archive_traversal_symlink_or_unpinned_member_refused_before_extraction(tmp_path, unsafe):
    old, original = _archive_fixture(tmp_path, unsafe=unsafe)
    output = tmp_path / "complement_zip"
    output.mkdir()
    with pytest.raises(ValueError):
        protocol.extract_original_archive({"output": str(output), "original_output": str(old)}, original)
    assert not (output / "archived_code").exists() and not (tmp_path / "escape.py").exists()


@pytest.mark.parametrize("change", ["member", "zip", "tree", "model", "extra"])
def test_archive_digest_or_exact_member_set_rejects_tampering(tmp_path, change):
    old, original = _archive_fixture(tmp_path, changed=change == "member")
    if change == "zip": original["code"]["code_zip_sha256"] = "0" * 64
    elif change == "tree": original["code"]["source_tree_sha256"] = "0" * 64
    elif change == "model": original["code"]["model_code_sha256"] = "0" * 64
    elif change == "extra":
        with zipfile.ZipFile(old / "code.zip", "a") as archive: archive.writestr("training/unpinned.py", "# extra")
        original["code"]["code_zip_sha256"] = protocol.sha256_file(old / "code.zip")
    with pytest.raises(ValueError): protocol.verify_original_archive(old, original)


def test_static_opaque109_hashing_rejects_partial_test_or_changed_original(tmp_path, monkeypatch):
    old = tmp_path / "original_files"
    old.mkdir()
    for name in protocol.original_file_names():
        path = old / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque fixture, not weather or valid JSON")
    monkeypatch.setattr(protocol, "read_json", lambda _: pytest.fail("opaque pinning must not parse contents"))
    pins = protocol.pin_original_files(old)
    assert len(pins) == 109
    protocol.verify_original_files({"original_output": str(old), "original_file_sha256": pins})
    (old / "attempt.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pins changed"):
        protocol.verify_original_files({"original_output": str(old), "original_file_sha256": pins})
    (old / "test.jsonl").write_bytes(b"sealed fixture")
    with pytest.raises(ValueError, match="inventory"):
        protocol.pin_original_files(old)


def test_new_code_closure_includes_aggregator_split_sources_and_changed_pin(tmp_path, monkeypatch):
    root, output = tmp_path / "sources", tmp_path / "source_archive"
    output.mkdir()
    for name in protocol.DRIVER_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU code fixture\n", encoding="utf-8")
    (root / "training/r7_m3_complement_results.py").write_text("from .r7_m3_complement_cells import evaluate\n", encoding="utf-8")
    child = root / "training/r7_m3_complement_cells.py"
    child.write_text("def evaluate(): return None\n", encoding="utf-8")
    monkeypatch.setattr(protocol, "ROOT", root)
    monkeypatch.setattr(protocol.subprocess, "check_output", lambda args, **kwargs: "a" * 40 + "\n" if "rev-parse" in args else " M fixture\n")
    code = protocol.archive_driver_code(output)
    assert "training/r7_m3_complement_cells.py" in code["files"]
    assert set(protocol.DRIVER_SOURCES) <= set(code["files"])
    value = {"output": str(output), "code": code}
    protocol.verify_driver_code(value)
    child.write_text("# altered after freeze\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        protocol.verify_driver_code(value)


def test_worker_socket_deny_and_origin_assertions(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    worker.deny_network()
    with pytest.raises(RuntimeError, match="offline"): socket.create_connection(("example.invalid", 443))
    with pytest.raises(RuntimeError, match="offline"): socket.socket.connect(None, ("example.invalid", 443))
    root = tmp_path / "archived_code"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "path", [str(root)])
    # Actual imports in pytest are deliberately outside the archive and must be rejected.
    with pytest.raises(ValueError, match="escaped"):
        worker.assert_archive_origins(root)


def test_archive_bootstrap_is_stdlib_before_project_imports_in_fresh_process(tmp_path):
    root, source = tmp_path / "archived_code", tmp_path / "current_source"
    root.mkdir()
    source.mkdir()
    (root / "training").mkdir()
    (root / "training/__init__.py").write_text("", encoding="utf-8")
    (root / "training/isolated_probe.py").write_text("ORIGIN='archive'\n", encoding="utf-8")
    (source / "training").mkdir()
    (source / "training/__init__.py").write_text("raise RuntimeError('current import forbidden')\n", encoding="utf-8")
    code = """import importlib.util,sys,socket
from pathlib import Path
path=Path(sys.argv[1]); root=Path(sys.argv[2]); source=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('isolated_bootstrap',path)
w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
w.deny_network()
sys.path.insert(0,str(source))
w.isolate_archive(root,source)
import training.isolated_probe as probe
w.assert_archive_origins(root)
assert probe.ORIGIN=='archive' and Path(sys.path[0])==root
try: socket.create_connection(('example.invalid',443))
except RuntimeError: pass
else: raise AssertionError('network denial absent')
print('isolated-cpu-success')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                             str(protocol.ROOT / "scripts/r7_m3_complement_worker.py"), str(root), str(source)],
                            cwd=root, env=driver.worker_environment(""), text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated-cpu-success"


def test_prepare_static_freeze_failure_receipt_and_no_actual_gpu(frozen, monkeypatch, tmp_path):
    output = tmp_path / "prepare_only"
    monkeypatch.setattr(driver, "pin_original_files", lambda *args, **kwargs: frozen["original_file_sha256"])
    monkeypatch.setattr(driver, "qualify_original", lambda _: {"protocol_sha256": frozen["original_protocol_sha256"],
                         "code": {"code_zip_sha256": frozen["original_code_zip_sha256"], "model_code_sha256": "d" * 64},
                         "gpu": {"uuid": "GPU-fixed"}})
    monkeypatch.setattr(driver, "verify_original_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "archive_driver_code", lambda _: {"source_root": str(tmp_path / "sources")})
    monkeypatch.setattr(driver, "verify_original_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "verify_driver_code", lambda *args, **kwargs: None)
    value = driver.prepare(frozen["original_output"], output)
    assert protocol.verify_protocol(output / "protocol.json") == value
    assert not (output / "archived_code").exists() and not (output / "workers").exists()
    assert protocol.read_json(output / "prepare_attempt.json")["status"] == "prepared-not-run"
    with pytest.raises(FileExistsError): driver.prepare(frozen["original_output"], output)
    monkeypatch.setattr(driver, "qualify_original", lambda _: (_ for _ in ()).throw(ValueError("partial selected399 fixture")))
    failed = tmp_path / "prepare_failure"
    with pytest.raises(ValueError, match="selected399"):
        driver.prepare(frozen["original_output"], failed)
    entry = protocol.read_json(failed / "attempt.json")
    assert entry["status"] == "failed" and entry["gpu_hours_charged"] == 0
    assert not (failed / "protocol.json").exists()


@pytest.mark.parametrize("target", ["old", "outside", "accepted"])
def test_real_cli_rejected_receipt_never_writes_original_but_valid_failure_kept(frozen, tmp_path, target):
    import time
    output = Path(frozen["output"])
    root = output / "archived_code"
    root.mkdir()
    if target == "old": receipt = Path(frozen["original_output"]) / "forbidden_receipt.json"
    elif target == "outside": receipt = tmp_path / "outside_receipt.json"
    else: receipt = output / "accepted_failed.json"
    before = protocol.sha256_file(Path(frozen["original_output"]) / "protocol.json")
    result = subprocess.run([sys.executable, "-I", "-B", str(protocol.ROOT / "scripts/r7_m3_complement_worker.py"),
                             "--protocol", str(output / "protocol.json"), "--mode", "verify",
                             "--deadline", repr(time.perf_counter() + 30), "--receipt", str(receipt)],
                            cwd=root, env=driver.worker_environment(""), text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert protocol.sha256_file(Path(frozen["original_output"]) / "protocol.json") == before
    if target == "accepted":
        entry = protocol.read_json(receipt)
        assert entry["status"] == "failed" and "failure_reason" in entry
    else:
        assert not receipt.exists() and "new exclusive output receipt" in result.stderr


def test_frozen_prepare_start_includes_interstage_gap_and_reboot_is_refused(frozen, monkeypatch):
    fake = _fake_round(frozen, monkeypatch, cpu_seconds=10)
    fake[0].now = 100  # prepare anchor was frozen at zero, run begins after 100-second gap.
    _run(frozen, fake)
    attempt = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert attempt["started_perf_counter"] == 0 and attempt["run_entry_perf_counter"] == 100
    assert attempt["whole_elapsed_seconds"] == 143.75
    assert attempt["gpu_phase_elapsed_seconds"] == 28.5


def test_evaluation_rejects_ancestor_symlink_before_archived_evaluator(frozen, monkeypatch):
    old, output = Path(frozen["original_output"]), Path(frozen["output"])
    (old / "seed41").mkdir()
    (output / "seed41").symlink_to(old / "seed41", target_is_directory=True)
    original_files = {path.relative_to(old).as_posix() for path in old.rglob("*")}
    monkeypatch.setattr(worker, "training_receipt", lambda *args, **kwargs: ({"checkpoint": "fixture"}, "1" * 64))
    cuda = SimpleNamespace(device=lambda name: name)
    modules = {"torch": cuda,
               "worker": SimpleNamespace(zero_allocator_baseline=lambda _: {"allocated_bytes": 0, "reserved_bytes": 0}),
               "evaluate_local": lambda *args, **kwargs: pytest.fail("must reject ancestors before archived evaluator")}
    import time
    with pytest.raises(ValueError, match="ancestor symlinks"):
        worker.evaluate_complement(frozen, {}, modules, frozen["jobs"][0], time.perf_counter() + 30)
    assert {path.relative_to(old).as_posix() for path in old.rglob("*")} == original_files
    assert not (old / "seed41/evaluation").exists()


def test_frozen_prepare_boot_id_prevents_cross_reboot_resume(frozen, monkeypatch):
    monkeypatch.setattr(driver, "monotonic_boot_id", lambda: "different-boot")
    with pytest.raises(ValueError, match="same-boot"):
        driver.run_bounded_round(frozen["output"], clock=FakeClock(),
                                 snapshot_fn=lambda _: pytest.fail("must not query GPU"))
    entry = protocol.read_json(Path(frozen["output"]) / "attempt.json")
    assert entry["status"] == "failed" and entry["gpu_hours_charged"] == 0
