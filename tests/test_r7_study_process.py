"""CPU counterproofs for directly owned study processes and isolated rounds."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts import study_r7_s3_v3_rollout_isolated as isolated
from scripts import study_r7_s3_v3_rollout_ft_worker as worker
from training.r7_study_process import bounded_process, worker_environment


def test_owned_process_success_and_failure(tmp_path):
    success = bounded_process([sys.executable, "-c", "print('observed')"], cwd=tmp_path,
                              log_path=tmp_path / "success.log", deadline=time.perf_counter() + 10)
    assert success["status"] == "success" and success["reaped"] is True
    assert success["returncode"] == 0 and success["signals"] == []
    assert (tmp_path / "success.log").read_text().strip() == "observed"
    failure = bounded_process([sys.executable, "-c", "raise ValueError('fault')"], cwd=tmp_path,
                              log_path=tmp_path / "failure.log", deadline=time.perf_counter() + 10)
    assert failure["status"] == "failed" and failure["returncode"] != 0
    assert "ValueError: fault" in (tmp_path / "failure.log").read_text()


def test_expired_deadline_never_spawns(tmp_path):
    report = bounded_process([sys.executable, "-c", "raise RuntimeError('must not run')"],
                             cwd=tmp_path, log_path=tmp_path / "absent.log",
                             deadline=time.perf_counter() - 1)
    assert report["status"] == "deadline-before-spawn" and report["pid"] is None
    assert not (tmp_path / "absent.log").exists()


def test_blocking_worker_is_reaped_without_signalling_neighbor(tmp_path):
    neighbor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    try:
        report = bounded_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                 cwd=tmp_path, log_path=tmp_path / "blocked.log",
                                 deadline=time.perf_counter() + 0.4, grace_seconds=0.3)
        assert report["status"] == "deadline-exceeded" and report["reaped"] is True
        assert report["elapsed_seconds"] < 5
        assert report["pid"] != neighbor.pid and neighbor.poll() is None
        assert all(action["pid"] == report["pid"] for action in report["signals"])
    finally:
        neighbor.terminate()
        neighbor.wait(timeout=5)


def test_sigterm_ignored_escalates_only_the_owned_pid(tmp_path):
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
    report = bounded_process([sys.executable, "-c", code], cwd=tmp_path,
                             log_path=tmp_path / "ignore.log", deadline=time.perf_counter() + 0.8,
                             grace_seconds=0.2)
    assert "ready" in (tmp_path / "ignore.log").read_text()
    assert report["status"] == "deadline-exceeded" and report["reaped"] is True
    assert [action["action"] for action in report["signals"]] == ["terminate", "kill"]
    assert report["elapsed_seconds"] >= 0.8 and report["elapsed_seconds"] < 5
    assert report["hard_overrun_seconds"] > 0


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "10"])
def test_invalid_deadline_is_rejected_before_spawn(tmp_path, deadline):
    with pytest.raises(ValueError, match="finite"):
        bounded_process([sys.executable, "-c", "print('no')"], cwd=tmp_path,
                        log_path=tmp_path / "absent.log", deadline=deadline)
    assert not (tmp_path / "absent.log").exists()


def test_worker_environment_rejects_gpu_conflict(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    with pytest.raises(ValueError, match="conflicts"):
        worker_environment(worker.GPU_UUID)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    environment = worker_environment(worker.GPU_UUID)
    assert environment["CUDA_VISIBLE_DEVICES"] == worker.GPU_UUID
    assert environment["OMP_NUM_THREADS"] == "4"


def test_worker_damaged_frozen_protocol_is_refused(tmp_path):
    from training.r7_experiment import canonical_digest
    body = {"gpu": {"uuid": worker.GPU_UUID}, "execution": {"files_sha256": {}},
            "arms": {"control": {"pins": {"d3_result_sha256": "0" * 64}}}}
    body["protocol_sha256"] = canonical_digest(body)
    body["gpu"]["uuid"] = "wrong"
    (tmp_path / "protocol.json").write_text(json.dumps(body))
    with pytest.raises(RuntimeError, match="digest"):
        worker.frozen_protocol(tmp_path)


def test_worker_code_drift_is_refused(tmp_path, monkeypatch):
    from training.r7_experiment import canonical_digest
    body = {"gpu": {"uuid": worker.GPU_UUID}, "execution": {
        "files_sha256": {"scripts/study_r7_s3_v3_rollout_ft.py": "0" * 64}}}
    body["protocol_sha256"] = canonical_digest(body)
    (tmp_path / "protocol.json").write_text(json.dumps(body))
    monkeypatch.setattr(worker, "execution_files", lambda: ["scripts/study_r7_s3_v3_rollout_ft.py"])
    with pytest.raises(RuntimeError, match="code drift"):
        worker.frozen_protocol(tmp_path)


def test_isolated_recipe_and_new_uuid_keep_original_facts():
    assert worker.GPU_UUID == "GPU-9d1624af-9d77-aa7c-0620-b6cb778f4ced"
    assert recipe.GPU_UUID == "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"
    assert recipe.HARD_CAP_SECONDS_ROUND == 7200 and recipe.FT_UPDATES == 200
    assert worker.HARD_CAP_SECONDS == 10800 and worker.PLANNED_SECONDS == 5400
    assert worker.PER_SEED_SECONDS == 3600
    assert recipe.SEEDS == (41, 42, 43) and recipe.GATE_TOLERANCE == 0
    assert isolated.DEFAULT_OUT.name.endswith("attempt02")


def _fake_worker(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import json,sys,time
from pathlib import Path
phase, directory, mode, seed = sys.argv[1:]
root=Path(directory)
if phase=='prepare':
    body={'train_data_identity':'a'*64,'val_data_identity':'b'*64,
          'arms':{},'flop_probe':{},'limitations':['CPU control fixture, not weather evidence']}
    (root/'prepared_protocol.json').write_text(json.dumps(body))
elif phase=='reading':
    if mode=='read_block':
        time.sleep(30)
    body={'decision':'registered-negative','gate_pre_screen':{'failures':[1]},
          'primary_verdict':{'overall':'supported'}}
    (root/'readings.json').write_text(json.dumps(body))
elif mode=='block' and seed=='42':
    time.sleep(30)
elif mode=='error' and seed=='42':
    raise RuntimeError('injected seed failure')
else:
    (root/f'seed{seed}_receipt.json').write_text(json.dumps({'seed':int(seed),'evaluations':{},
                                                          'owned_cuda_reserved_peak_bytes':123456}))
''', encoding="utf-8")
    return script


def _fixture_round(tmp_path, monkeypatch, mode):
    script = _fake_worker(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(isolated, "HARD_CAP_SECONDS", 15.0)
    monkeypatch.setattr(isolated, "PER_SEED_SECONDS", 0.5)
    monkeypatch.setattr(isolated, "_command", lambda phase, out, deadline, seed=None:
                        [sys.executable, str(script), phase, str(out), mode, str(seed)])
    monkeypatch.setattr(recipe._shared(), "gpu_gate", lambda uuid, **kwargs: {"passed": True, "uuid": uuid})
    return tmp_path / "attempt"


@pytest.mark.parametrize("mode", ["error", "block"])
def test_seed_failure_stops_round_and_no_complete_is_written(tmp_path, monkeypatch, mode):
    out = _fixture_round(tmp_path, monkeypatch, mode)
    with pytest.raises(RuntimeError, match="seed 42"):
        isolated.run_attempt(out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["seeds_completed"] == ["41"] and failure["scientific_claim"] is False
    assert not (out / "seed43_receipt.json").exists()
    assert not (out / "result.json").exists() and not (out / "attempt.json").exists()
    assert all(process["reaped"] for process in failure["processes"])
    assert failure["elapsed_seconds_total"] > 0


def test_complete_cpu_control_does_not_omit_any_seed(tmp_path, monkeypatch):
    out = _fixture_round(tmp_path, monkeypatch, "success")
    result = isolated.run_attempt(out)
    assert sorted(result["seeds"]) == ["41", "42", "43"]
    assert len(result["processes"]) == 5 and len(result["gpu_gates"]) == 3
    assert json.loads((out / "attempt.json").read_text())["status"] == "complete"
    assert result["scientific_claim"] is False and result["test_read"] is False
    assert not (out / "failure.json").exists()


def test_round_deadline_bounds_preparation(tmp_path, monkeypatch):
    out = _fixture_round(tmp_path, monkeypatch, "success")
    monkeypatch.setattr(isolated, "HARD_CAP_SECONDS", 0.4)
    monkeypatch.setattr(isolated, "_command", lambda *args:
                        [sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(RuntimeError, match="preparation"):
        isolated.run_attempt(out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["seeds_completed"] == []
    assert failure["processes"][0]["status"] == "deadline-exceeded"
    assert not (out / "protocol.json").exists()
    assert failure["elapsed_seconds_total"] < 8


def test_existing_attempt_is_never_overwritten(tmp_path):
    out = tmp_path / "attempt"
    out.mkdir()
    original = out / "failure.json"
    original.write_text("immutable")
    with pytest.raises(FileExistsError, match="fresh"):
        isolated.run_attempt(out)
    assert original.read_text() == "immutable"


def test_round_deadline_bounds_paired_reading(tmp_path, monkeypatch):
    out = _fixture_round(tmp_path, monkeypatch, "read_block")
    monkeypatch.setattr(isolated, "HARD_CAP_SECONDS", 2.0)
    with pytest.raises(RuntimeError, match="paired reading"):
        isolated.run_attempt(out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["seeds_completed"] == ["41", "42", "43"]
    assert failure["processes"][-1]["phase"] == "reading"
    assert failure["processes"][-1]["status"] == "deadline-exceeded"
    assert failure["elapsed_seconds_total"] < 10
    assert not (out / "attempt.json").exists()


def test_output_ancestors_are_rejected_before_writes(tmp_path):
    old = tmp_path / "completed"
    old.mkdir()
    (old / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="old evidence"):
        isolated.fresh_output(old / "attempt")
    link = tmp_path / "linked"
    link.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        isolated.fresh_output(link / "attempt")
    with pytest.raises(ValueError, match="protected"):
        isolated.fresh_output(isolated.ROOT / "data/raw/forbidden_attempt")
    assert not (old / "attempt").exists()


def test_frozen_inventory_cannot_be_empty(tmp_path):
    from training.r7_experiment import canonical_digest
    body = {"execution": {"files_sha256": {}}}
    body["protocol_sha256"] = canonical_digest(body)
    (tmp_path / "protocol.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory"):
        worker.frozen_protocol(tmp_path)


def test_worker_deadline_cannot_exceed_frozen_round():
    now = time.perf_counter()
    body = {"timing": {"started_perf_counter": now, "deadline_perf_counter": now + 10},
            "budgets": {"hard_cap_seconds_round": 10}}
    worker.validate_deadline(body, now + 5)
    with pytest.raises(RuntimeError, match="outside"):
        worker.validate_deadline(body, now + 11)
    with pytest.raises(RuntimeError, match="disagree"):
        worker.validate_deadline({**body, "budgets": {"hard_cap_seconds_round": 11}}, now + 5)


def test_known_peak_reaches_the_next_spawn_gate(tmp_path, monkeypatch):
    out = _fixture_round(tmp_path, monkeypatch, "success")
    peaks = []
    def gate(uuid, **kwargs):
        if "owned_reserved_peak_bytes" in kwargs:
            peaks.append(kwargs["owned_reserved_peak_bytes"])
        return {"passed": True, "uuid": uuid}
    monkeypatch.setattr(recipe._shared(), "gpu_gate", gate)
    isolated.run_attempt(out)
    assert peaks == [0, 123456, 123456]
