"""CPU counterproofs for the separate N1 evaluation-memory supplement."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import subprocess
import zipfile

import pytest

from training import r7_n1_cost_replay as support
from scripts import measure_r7_n1_eval_cost as parent
from scripts import measure_r7_n1_eval_worker as worker


def authorization():
    return {"status": "authorized", "channel": "AskUserQuestion", "user_response": "authorized",
            "scope": "n1-evaluation-cost-supplement", "gpu_seconds_cap": 324.0,
            "training_updates": 0, "evaluations": 30, "test_read": False, "automatic_retry": False}


def original_protocol():
    body = {"data": {"manifests_dir": "fixture/manifests", "val_manifest": "fixture/manifests/val.jsonl",
                     "data_identity": "data", "source_identity": {"source_sha256": "source"}}}
    return dict(body, protocol_sha256=support.digest(body))


@pytest.fixture(autouse=True)
def pin_fixture_protocol(monkeypatch):
    monkeypatch.setattr(support, "ORIGINAL_PROTOCOL_SHA", original_protocol()["protocol_sha256"])


def protocol():
    original = original_protocol()
    body = {"format": "r7-n1-evaluation-cost-supplement-v1", "gpu_seconds_cap": 324.0,
            "training_updates": 0, "test_read": False,
            "original_protocol_sha256": support.ORIGINAL_PROTOCOL_SHA,
            "original_protocol": original, "val_manifest": original["data"]["val_manifest"],
            "data_identity": original["data"]["data_identity"], "source_identity": original["data"]["source_identity"],
            "authorization": authorization(),
            "tasks": [{"seed": s, "arm": a, "lead": h} for s in support.SEEDS
                      for a in support.ARMS for h in support.LEADS]}
    return dict(body, protocol_sha256=support.digest(body))


class FakeCuda:
    def __init__(self, allocated=0, reserved=0, peak=128, peak_reserved=256):
        self.events = []
        self.allocated, self.reserved = allocated, reserved
        self.peak, self.peak_reserved = peak, peak_reserved

    def set_device(self, device):
        self.events.append(("set", device))

    def synchronize(self, device):
        self.events.append(("sync", device))

    def empty_cache(self):
        self.events.append(("empty",))

    def memory_allocated(self, device):
        self.events.append(("allocated", device))
        return self.allocated

    def memory_reserved(self, device):
        self.events.append(("reserved", device))
        return self.reserved

    def reset_peak_memory_stats(self, device):
        self.events.append(("reset", device))

    def max_memory_allocated(self, device):
        self.events.append(("peak", device))
        return self.peak

    def max_memory_reserved(self, device):
        self.events.append(("peak_reserved", device))
        return self.peak_reserved

    def get_device_properties(self, device):
        self.events.append(("uuid", device))
        return SimpleNamespace(uuid="GPU-a")


def test_reset_and_read_use_explicit_device_and_zero_baseline(monkeypatch):
    cuda = FakeCuda()
    monkeypatch.setattr(support.gc, "collect", lambda: cuda.events.append(("gc",)))
    assert support.reset_measurement(cuda, "cuda:1") == {"allocated_bytes": 0, "reserved_bytes": 0}
    assert cuda.events == [("set", "cuda:1"), ("gc",), ("sync", "cuda:1"), ("empty",),
                           ("sync", "cuda:1"), ("allocated", "cuda:1"), ("reserved", "cuda:1"), ("reset", "cuda:1")]
    assert support.read_peaks(cuda, "cuda:1") == {"peak_allocated_bytes": 128, "peak_reserved_bytes": 256}
    assert cuda.events[-3:] == [("sync", "cuda:1"), ("peak", "cuda:1"), ("peak_reserved", "cuda:1")]


@pytest.mark.parametrize("allocated,reserved", [(1, 0), (0, 1), (1, 2)])
def test_nonzero_baseline_refuses_before_reset(allocated, reserved):
    cuda = FakeCuda(allocated, reserved)
    with pytest.raises(RuntimeError, match="zero allocated/reserved"):
        support.reset_measurement(cuda, "cuda:1")
    assert not any(event[0] == "reset" for event in cuda.events)


@pytest.mark.parametrize("peak,reserved", [(0, 0), (2, 1), (-1, 3)])
def test_invalid_peak_refused(peak, reserved):
    with pytest.raises(RuntimeError, match="invalid"):
        support.read_peaks(FakeCuda(peak=peak, peak_reserved=reserved), "cuda:1")


@pytest.mark.parametrize("field,value", [("status", "pending"), ("user_response", ""),
    ("scope", "other"), ("gpu_seconds_cap", 325), ("training_updates", 1),
    ("evaluations", 29), ("test_read", True), ("automatic_retry", True), ("channel", "notification")])
def test_named_authorization_fail_closed(field, value):
    data = authorization()
    data[field] = value
    with pytest.raises(ValueError, match="authorization"):
        support.verify_authorization(data)


def test_no_authorization_means_no_archive_or_cuda_access(tmp_path, monkeypatch):
    path = tmp_path / "authorization.json"
    support.write_json(path, {"status": "pending"})
    monkeypatch.setattr(parent, "inspect_archive", lambda *_: pytest.fail("archive accessed"))
    monkeypatch.setattr(parent, "gpu_exclusive", lambda *_: pytest.fail("GPU accessed"))
    with pytest.raises(ValueError, match="authorization"):
        parent.run(tmp_path / "archive", tmp_path / "out", path, "cuda:1")
    assert not (tmp_path / "out").exists()


def test_registration_rehashed_missing_secondary_cell_refuses(tmp_path):
    value = protocol()
    support.write_json(tmp_path / "p.json", value)
    assert support.verify_registration(tmp_path / "p.json") == value
    value["tasks"].pop()
    value["protocol_sha256"] = support.digest({k: v for k, v in value.items() if k != "protocol_sha256"})
    support.write_json(tmp_path / "missing.json", value)
    with pytest.raises(ValueError, match="exactly"):
        support.verify_registration(tmp_path / "missing.json")


@pytest.mark.parametrize("field,value", [("gpu_seconds_cap", 325), ("training_updates", 1),
                                         ("test_read", True), ("original_protocol_sha256", "0" * 64)])
def test_registration_rehashed_fixed_controls_refused(tmp_path, field, value):
    p = protocol()
    p[field] = value
    p["protocol_sha256"] = support.digest({k: v for k, v in p.items() if k != "protocol_sha256"})
    support.write_json(tmp_path / "p.json", p)
    with pytest.raises(ValueError, match="mismatch"):
        support.verify_registration(tmp_path / "p.json")


def test_pinned_file_mismatch_symlink_escape_and_exclusive_output(tmp_path):
    p = tmp_path / "a.json"
    support.write_json(p, {"safe": True})
    with pytest.raises(FileExistsError):
        support.write_json(p, {"safe": False})
    before = p.read_bytes()
    with pytest.raises(ValueError, match="digest"):
        support.pinned_file(p, "0" * 64)
    (tmp_path / "link").symlink_to(p)
    with pytest.raises(ValueError, match="symlink"):
        support.pinned_file(tmp_path / "link", support.sha256(p))
    with pytest.raises(ValueError, match="escapes"):
        support.pinned_file(p, support.sha256(p), tmp_path / "elsewhere")
    assert p.read_bytes() == before


@pytest.mark.parametrize("name", ["../model/a.py", "/model/a.py", "model\\a.py"])
def test_archive_unsafe_paths_refused(tmp_path, name):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "x") as bundle:
        bundle.writestr(name, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        support.extract_code(z, tmp_path / "extract")
    assert not (tmp_path / "model").exists()


def test_archive_extracts_active_python_only_and_exact_bytes(tmp_path):
    z = tmp_path / "a.zip"
    names = ["training/r7_evaluate.py", "training/r7_experiment.py", "model/__init__.py",
             "training/r7_frozen_z_intervention.py", "data/r7_evaluation.py"]
    with zipfile.ZipFile(z, "x") as bundle:
        for name in names:
            bundle.writestr(name, b"# archived\n")
        bundle.writestr("model/legacy_v531/test.py", "no")
        bundle.writestr("data/raw/payload.nc", "no")
    entries = support.extract_code(z, tmp_path / "extract")
    assert {entry["path"] for entry in entries} == set(names)
    assert all((tmp_path / "extract" / name).read_bytes() == b"# archived\n" for name in names)
    assert not (tmp_path / "extract/model/legacy_v531").exists()
    assert not (tmp_path / "extract/data/raw").exists()


def valid_rows():
    return [{"seed": s, "arm": a, "lead": h, "n_cases": support.COUNTS[h],
             "baseline": {"allocated_bytes": 0, "reserved_bytes": 0},
             "peak_allocated_bytes": 128, "peak_reserved_bytes": 256, "elapsed_seconds": 1.0,
             "replay": {"exact_numeric_replay": True}}
            for s in support.SEEDS for a in support.ARMS for h in support.LEADS]


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_case", "baseline", "replay", "nan"])
def test_full_measurement_set_fail_closed(change):
    rows = valid_rows()
    support.validate_measurements(rows)
    if change == "missing": rows.pop()
    if change == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    if change == "wrong_case": rows[0]["n_cases"] = 1
    if change == "baseline": rows[0]["baseline"]["reserved_bytes"] = 1
    if change == "replay": rows[0]["replay"]["exact_numeric_replay"] = False
    if change == "nan": rows[0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        support.validate_measurements(rows)


def evaluation_files(tmp_path):
    original, replay = tmp_path / "original", tmp_path / "replay"
    original.mkdir()
    replay.mkdir()
    report = {"split": "val", "n_evaluated": 22, "initializations": [{"init_time": "time", "mse": [[1.0]]}]}
    support.write_json(original / "provenance.json", report)
    support.write_json(replay / "provenance.json", report)
    (replay / "acc.csv").write_text("acc\n", encoding="utf-8")
    (replay / "climatology_skill.csv").write_text("skill\n", encoding="utf-8")
    fields = ["lead_hours", "variable", "rmse", "unit", "n_initializations"]
    for path in (original / "rmse.csv", replay / "rmse.csv"):
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(lead_hours=6, variable=v, rmse=1.0, unit="K", n_initializations=22) for v in support.VARIABLES)
    task = {"original_dir": str(original), "rmse_sha256": support.sha256(original / "rmse.csv"),
            "provenance_sha256": support.sha256(original / "provenance.json")}
    return task, replay, report


def test_exact_replay_rejects_case_or_rmse_change(tmp_path):
    task, output, report = evaluation_files(tmp_path)
    assert support.compare_evaluation(task, output, report)["exact_numeric_replay"]
    changed = copy.deepcopy(report)
    changed["initializations"][0]["mse"] = [[1.0000000000000002]]
    with pytest.raises(ValueError, match="provenance"):
        support.compare_evaluation(task, output, changed)
    (output / "rmse.csv").write_text((output / "rmse.csv").read_text().replace("1.0", "1.0000000000000002"), encoding="utf-8")
    with pytest.raises(ValueError, match="RMSE"):
        support.compare_evaluation(task, output, report)


def test_worker_measurement_sequence_and_record(tmp_path, monkeypatch):
    task, output, report = evaluation_files(tmp_path)
    task.update(seed=41, arm=support.ARMS[0], lead=6, checkpoint={"path": str(output), "sha256": "pin"})
    cuda = FakeCuda()
    torch = SimpleNamespace(cuda=cuda, device=lambda value: value)
    monkeypatch.setattr(support, "gpu_exclusive", lambda *_a, **_k: None)
    monkeypatch.setattr(support, "verify_task_inputs", lambda *_: cuda.events.append(("inputs",)))
    monkeypatch.setattr(support.gc, "collect", lambda: cuda.events.append(("gc",)))
    def evaluate(*args, **kwargs):
        assert cuda.events[-1] == ("reset", "cuda:0")
        cuda.events.append(("evaluate", "cuda:0"))
        assert kwargs["device_name"] == "cuda:0"
        assert kwargs["deadline"] == 1e12 and kwargs["reasoning_steps"] == 3
        return report
    p = {"device": "cuda:1", "execution_device": "cuda:0", "val_manifest": "val.jsonl",
         "protocol_sha256": "p", "physical_gpu_uuid": "GPU-a"}
    row = worker.measure_task(task, p, output, evaluate, torch, support, 1e12)
    assert row["baseline"] == {"allocated_bytes": 0, "reserved_bytes": 0}
    assert row["peak_allocated_bytes"] == 128 and row["peak_reserved_bytes"] == 256
    assert support.read_json(output / "cost_measurement.json") == row
    assert cuda.events[-3:] == [("sync", "cuda:0"), ("peak", "cuda:0"), ("peak_reserved", "cuda:0")]
    assert cuda.events[0] == ("inputs",) and row["gpu_uuid"] == "GPU-a"


def test_expired_deadline_refuses_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(support.time, "perf_counter", lambda: 10)
    with pytest.raises(RuntimeError, match="deadline"):
        worker.measure_task({}, {}, tmp_path, lambda *_: pytest.fail("evaluation ran"), None, support, 9)


@pytest.mark.parametrize("busy", [False, True])
def test_gpu_mapping_and_exclusivity(monkeypatch, busy):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    responses = iter(["0, GPU-a\n1, GPU-b\n", f"GPU-b, {12345 if busy else support.os.getpid()}\n"])
    def run(arguments, **kwargs):
        assert kwargs["shell"] is False
        return SimpleNamespace(stdout=next(responses))
    monkeypatch.setattr(support.subprocess, "run", run)
    if busy:
        with pytest.raises(RuntimeError, match="occupied"):
            support.gpu_exclusive("cuda:0")
    else:
        assert support.gpu_exclusive("cuda:0")["physical_selector"] == "1"


def test_owned_child_timeout_kills_only_child_and_waits(tmp_path, monkeypatch):
    events = []
    class Child:
        def communicate(self, **kwargs):
            assert '"seed": 41' in kwargs["input"]
            events.append("communicate")
            raise subprocess.TimeoutExpired("owned", 1)
        def poll(self): return None
        def kill(self): events.append("kill")
        def wait(self): events.append("wait")
    def launch(arguments, **kwargs):
        assert arguments == ["./.venv/bin/python", "-B", "measurement/scripts/measure_r7_n1_eval_worker.py"]
        assert kwargs["shell"] is False
        return Child()
    monkeypatch.setattr(parent.subprocess, "Popen", launch)
    monkeypatch.setattr(parent.time, "perf_counter", lambda: 0)
    with pytest.raises(subprocess.TimeoutExpired):
        parent.run_owned_child(tmp_path, tmp_path, 41, support.ARMS[0], deadline=1, environment={})
    assert events == ["communicate", "kill", "wait"]


@pytest.mark.parametrize("field,value", [("device", "cuda:0"), ("protocol_sha256", "wrong"),
    ("original_checkpoint_sha256", "wrong"), ("evaluation_dir", "/outside"), ("gpu_uuid", "GPU-b")])
def test_measurement_identity_is_bound_to_task(tmp_path, field, value, monkeypatch):
    rows = valid_rows()
    p = protocol()
    p.update(device="cuda:1", execution_device="cuda:0", physical_gpu_uuid="GPU-a")
    monkeypatch.setattr(support, "evaluation_artifacts", lambda *_: {})
    monkeypatch.setattr(support, "compare_evaluation", lambda *_: {"exact_numeric_replay": True})
    def stored(path):
        return next((r for r in rows if Path(r["evaluation_dir"]) / "cost_measurement.json" == path), {})
    monkeypatch.setattr(support, "read_json", stored)
    for task, row in zip(p["tasks"], rows):
        task["checkpoint"] = {"sha256": "pin"}
        row.update(device="cuda:1", execution_device="cuda:0", protocol_sha256=p["protocol_sha256"],
                   gpu_uuid="GPU-a", evaluation_artifacts_sha256={},
                   original_checkpoint_sha256="pin", evaluation_dir=str(
                       tmp_path / f"seed{row['seed']}" / row["arm"] / f"lead_{row['lead']:03d}h"))
    support.validate_measurements(rows, p, tmp_path)
    rows[0][field] = value
    with pytest.raises(ValueError, match="bind"):
        support.validate_measurements(rows, p, tmp_path)


def test_expired_parent_never_spawns_child(tmp_path, monkeypatch):
    monkeypatch.setattr(parent.time, "perf_counter", lambda: 10)
    monkeypatch.setattr(parent.subprocess, "Popen", lambda *_a, **_k: pytest.fail("spawned"))
    with pytest.raises(RuntimeError, match="deadline"):
        parent.run_owned_child(tmp_path, tmp_path, 41, support.ARMS[0], deadline=9, environment={})
    assert not list(tmp_path.glob("*.log"))


def test_gpu_queries_are_bounded_by_remaining_deadline(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(support.time, "perf_counter", lambda: 10)
    calls = []
    responses = iter(["0, GPU-a\n1, GPU-b\n", ""])
    def run(arguments, **kwargs):
        calls.append(kwargs["timeout"])
        return SimpleNamespace(stdout=next(responses))
    monkeypatch.setattr(support.subprocess, "run", run)
    support.gpu_exclusive("cuda:1", deadline=12)
    assert calls == [2, 2]
    with pytest.raises(RuntimeError, match="deadline"):
        support.gpu_exclusive("cuda:1", deadline=9)
    assert calls == [2, 2]


def test_frozen_wrapper_bytes_match_archive(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    code = tmp_path / "code"
    for p in (repo, out, code): p.mkdir()
    names = ("training/r7_n1_cost_replay.py", "scripts/measure_r7_n1_eval_worker.py")
    with zipfile.ZipFile(out / "measurement_code.zip", "x") as bundle:
        for name in names: bundle.writestr(name, "# pinned\n")
    import hashlib
    identity = {"measurement_source_sha256": {n: hashlib.sha256(b"# pinned\n").hexdigest() for n in names}}
    monkeypatch.setattr(parent, "ROOT", repo)
    info = parent.freeze_wrapper(out, code, identity)
    assert Path(info["root"]) == code / "measurement"
    assert all((code / "measurement" / n).read_bytes() == b"# pinned\n" for n in names)
    assert (code / ".venv").is_symlink()


def test_failure_after_worker_is_charged_and_retained(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    auth = tmp_path / "auth.json"
    support.write_json(auth, authorization())
    monkeypatch.setattr(parent, "inspect_archive", lambda *_: {"original_protocol": {"data": {"manifests_dir": str(tmp_path)}}})
    monkeypatch.setattr(parent, "reserve_output", lambda *_: out)
    monkeypatch.setattr(parent, "archive_wrapper", lambda *_: {})
    monkeypatch.setattr(parent, "extract_code", lambda *_: [])
    monkeypatch.setattr(parent, "freeze_wrapper", lambda *_: {})
    monkeypatch.setattr(parent, "registration", lambda *_: protocol())
    monkeypatch.setattr(parent, "gpu_exclusive", lambda *_a, **_k: {"gpu_uuid": "GPU-a"})
    monkeypatch.setattr(parent.time, "perf_counter", lambda: 100.0)
    calls = []
    def fail(*args, **kwargs):
        assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-a"
        calls.append((args[2], args[3]))
        monkeypatch.setattr(parent.time, "perf_counter", lambda: 125.0)
        raise RuntimeError("worker failure")
    monkeypatch.setattr(parent, "run_owned_child", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        parent.run(tmp_path, out, auth, "cuda:1")
    attempt = support.read_json(out / "attempt.json")
    assert attempt["status"] == "failed" and attempt["gpu_phase_seconds"] == 25.0
    assert attempt["gpu_hours_charged"] == 25 / 3600
    assert calls == [(41, support.ARMS[0])]
    assert (out / "protocol.json").is_file() and not (out / "result.json").exists()


def test_network_is_denied(monkeypatch):
    import socket
    with monkeypatch.context() as patch:
        patch.setattr(socket.socket, "connect", socket.socket.connect)
        patch.setattr(socket, "create_connection", socket.create_connection)
        parent.deny_network()
        with pytest.raises(RuntimeError, match="offline"):
            socket.create_connection(("example.invalid", 443))


@pytest.mark.parametrize("change", ["test_path", "data_identity", "source_identity", "original_protocol"])
def test_rehashed_validation_scope_refuses_before_use(tmp_path, change):
    value = protocol()
    if change == "test_path": value["val_manifest"] = "fixture/manifests/test.jsonl"
    if change == "data_identity": value["data_identity"] = "other"
    if change == "source_identity": value["source_identity"] = {}
    if change == "original_protocol":
        value["original_protocol"]["data"]["val_manifest"] = "fixture/manifests/test.jsonl"
    value["protocol_sha256"] = support.digest({k: v for k, v in value.items() if k != "protocol_sha256"})
    path = tmp_path / "protocol.json"
    support.write_json(path, value)
    with pytest.raises(ValueError, match="validation"):
        support.verify_registration(path)


def test_changed_source_refuses_before_cuda_or_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "verify_validation_scope", lambda *_: {})
    def changed(*args):
        raise ValueError("source digest mismatch")
    monkeypatch.setattr(support, "verify_source", changed)
    monkeypatch.setattr(support, "gpu_exclusive", lambda *_a, **_k: pytest.fail("CUDA checked before source"))
    with pytest.raises(ValueError, match="source digest"):
        worker.measure_task({}, {}, tmp_path, lambda *_a, **_k: pytest.fail("evaluation ran"), None, support, 1e12)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("split", ["test", "train", None])
def test_manifest_must_be_val_even_after_source_verification(tmp_path, monkeypatch, split):
    manifest = tmp_path / "val.jsonl"
    manifest.write_text("\n".join(support.json.dumps({"split": split}) for _ in range(22)), encoding="utf-8")
    monkeypatch.setattr(support, "verify_validation_scope", lambda *_: {})
    monkeypatch.setattr(support, "verify_source", lambda *_: {})
    with pytest.raises(ValueError, match="validation-only manifest"):
        support.verify_task_inputs({}, {"val_manifest": str(manifest)})


@pytest.mark.parametrize("uuid", ["GPU-a", "GPU-b"])
def test_actual_cuda_uuid_is_bound_to_checked_device(uuid):
    if uuid == "GPU-a":
        assert support.verify_cuda_device(FakeCuda(), "cuda:1", uuid) == uuid
    else:
        with pytest.raises(RuntimeError, match="UUID differs"):
            support.verify_cuda_device(FakeCuda(), "cuda:1", uuid)


def test_uuid_exclusivity_does_not_assume_numeric_ordinal(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    responses = iter(["0, GPU-a\n1, GPU-b\n", "GPU-b, 12345\n"])
    monkeypatch.setattr(support.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=next(responses)))
    selected = support.gpu_exclusive("cuda:1", gpu_uuid="GPU-a")
    assert selected["gpu_uuid"] == "GPU-a"


def test_occupied_after_evaluation_refuses_row_publication(tmp_path, monkeypatch):
    task, directory, report = evaluation_files(tmp_path)
    task.update(seed=41, arm=support.ARMS[0], lead=6, checkpoint={"path": "fixture", "sha256": "pin"})
    monkeypatch.setattr(support, "verify_task_inputs", lambda *_: None)
    occupied = [False]
    def check(*args, **kwargs):
        if occupied[0]: raise RuntimeError("device occupied")
    monkeypatch.setattr(support, "gpu_exclusive", check)
    def evaluate(*args, **kwargs):
        occupied[0] = True
        return report
    p = {"device": "cuda:1", "execution_device": "cuda:0", "physical_gpu_uuid": "GPU-a", "val_manifest": "val.jsonl"}
    torch = SimpleNamespace(cuda=FakeCuda(), device=lambda value: value)
    with pytest.raises(RuntimeError, match="occupied"):
        worker.measure_task(task, p, directory, evaluate, torch, support, 1e12)
    assert not (directory / "cost_measurement.json").exists()


@pytest.mark.parametrize("change", ["absent", "provenance", "rmse", "measurement", "acc"])
def test_final_collection_requires_all_bound_replay_artifacts(tmp_path, change):
    rows, p = valid_rows(), protocol()
    p.update(device="cuda:1", execution_device="cuda:0", physical_gpu_uuid="GPU-a")
    for task, row in zip(p["tasks"], rows):
        directory = tmp_path / f"seed{row['seed']}" / row["arm"] / f"lead_{row['lead']:03d}h"
        fixture = directory.parent / f"fixture_{row['lead']}"
        fixture.mkdir(parents=True)
        source, replay, report = evaluation_files(fixture)
        replay.rename(directory)
        task.update(source, checkpoint={"sha256": "pin"})
        row["replay"] = {"rmse_rows": 17, "case_count": 22, "exact_numeric_replay": True}
        row.update(device="cuda:1", execution_device="cuda:0", gpu_uuid="GPU-a", protocol_sha256=p["protocol_sha256"],
                   original_checkpoint_sha256="pin", evaluation_dir=str(directory),
                   evaluation_artifacts_sha256=support.evaluation_artifacts(directory))
        support.write_json(directory / "cost_measurement.json", row)
    for seed in support.SEEDS:
        for arm in support.ARMS:
            selected = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
            support.write_json(tmp_path / f"worker_seed{seed}_{arm}.json",
                               {"status": "success", "protocol_sha256": p["protocol_sha256"], "rows": selected})
    assert parent.collect_rows(tmp_path, p) == rows
    first = Path(rows[0]["evaluation_dir"])
    if change == "absent": first.rename(first.parent / "absent")
    else:
        names = {"provenance": "provenance.json", "rmse": "rmse.csv",
                 "measurement": "cost_measurement.json", "acc": "acc.csv"}
        name = names[change]
        (first / name).write_text("{}", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        parent.collect_rows(tmp_path, p)
