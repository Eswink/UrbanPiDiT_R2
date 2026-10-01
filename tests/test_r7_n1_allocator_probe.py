"""CPU-only counterproofs for frozen N1 allocator diagnostics, without real data."""
from __future__ import annotations

import builtins
import gc
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import weakref

import pytest

from scripts import probe_r7_n1_allocator as probe


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return {"original_torch_version": "2.11.0+cu128-exact", "physical_gpu_uuid": "GPU-fixture",
            "execution_device": "cuda:0", "protocol_sha256": "protocol-pin", "limitations": ["fixture-only"],
            "val_manifest": "fixture-val-manifest-not-read",
            "tasks": [{"seed": 41, "arm": probe.PROBE_ARM, "lead": lead,
                       "checkpoint": {"path": "fixture-checkpoint-not-read", "sha256": "checkpoint-pin"}}
                      for lead in (6, 12)]}


class FakeCuda:
    def __init__(self):
        self.events = []
        self.bytes = (0, 0)
        self.snapshot_error = False
        self.references = []
        self.clock = 10.0

    def init(self):
        self.events.append(("init",))

    def set_device(self, device):
        self.events.append(("set", device))

    def synchronize(self, device):
        self.events.append(("sync", device))

    def empty_cache(self):
        self.events.append(("empty",))
        assert all(reference() is None for reference in self.references)


class FakeTensor:
    pass


class FakeTorch:
    def __init__(self, cuda, *, before=(64, 128), after=(0, 0), api=True, clear_error=False):
        self.cuda = cuda
        self.__version__ = "2.11.0+cu128-exact"
        self.float32 = "float32"
        self.before, self.after, self.clear_error = before, after, clear_error
        self._C = SimpleNamespace()
        if api:
            self._C._cuda_clearCublasWorkspaces = self.clear

    def device(self, value):
        assert value == "cuda:0"
        return value

    def set_num_threads(self, value):
        assert value == 4
        self.cuda.events.append(("threads", value))

    def ones(self, shape, *, dtype, device):
        assert shape == (1024, 1024) and dtype == self.float32 and device == "cuda:0"
        self.cuda.events.append(("ones", device))
        tensor = FakeTensor()
        self.cuda.references.append(weakref.ref(tensor))
        return tensor

    def matmul(self, left, right):
        assert isinstance(left, FakeTensor) and isinstance(right, FakeTensor)
        self.cuda.events.append(("matmul",))
        self.cuda.bytes = self.before
        tensor = FakeTensor()
        self.cuda.references.append(weakref.ref(tensor))
        return tensor

    def clear(self):
        self.cuda.events.append(("clear",))
        assert all(reference() is None for reference in self.cuda.references)
        if self.clear_error:
            raise RuntimeError("private clear unsupported")
        self.cuda.bytes = self.after


class FakeSupport:
    def __init__(self, cuda):
        self.cuda = cuda

    def check_deadline(self, deadline):
        if self.cuda.clock >= deadline:
            raise RuntimeError("supplement deadline exhausted")

    def gpu_headroom(self, device, deadline, *, gpu_uuid, refusal_path):
        self.check_deadline(deadline)
        assert device == "cuda:0" and gpu_uuid == "GPU-fixture"
        assert refusal_path.name == "guard_refusal.json"
        self.cuda.events.append(("headroom",))

    def verify_cuda_device(self, cuda, device, uuid):
        assert cuda is self.cuda and device == "cuda:0" and uuid == "GPU-fixture"
        self.cuda.events.append(("uuid", uuid))

    def reset_measurement(self, cuda, device, *, refusal_path):
        self.cuda.events.append(("reset", device))
        if cuda.bytes != (0, 0):
            write_json(refusal_path, {"allocator": self.allocator_record(cuda, device)})
            raise RuntimeError("zero allocated/reserved baseline required")

    def allocator_record(self, cuda, device):
        assert cuda is self.cuda and device == "cuda:0"
        allocated, reserved = cuda.bytes
        cuda.events.extend([("allocated", device), ("reserved", device), ("snapshot", device)])
        snapshot = {"sha256": hashlib.sha256(b"fixture-only").hexdigest(), "segments": 0,
                    "total_bytes": reserved, "block_state_bytes": {}, "raw_segments": []}
        if cuda.snapshot_error:
            snapshot = {"error": "fixture snapshot unavailable"}
        return {"allocated_bytes": allocated, "reserved_bytes": reserved, "memory_snapshot": snapshot}

    def verify_task_inputs(self, task, value):
        self.cuda.events.append(("inputs", task["lead"]))

    def compare_evaluation(self, task, directory, report):
        self.cuda.events.append(("compare", task["lead"]))
        assert report["lead_hours"] == [task["lead"]]
        return {"exact_numeric_replay": True, "case_count": report["n_evaluated"], "rmse_rows": 17}

    @staticmethod
    def write_json(path, value):
        write_json(path, value)

    @staticmethod
    def sha256(path):
        return file_sha(path)


@pytest.fixture(autouse=True)
def cpu_boundary(monkeypatch):
    original_import = builtins.__import__
    def offline_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch" and "torch" not in sys.modules:
            pytest.fail("real torch import forbidden by CPU probe tests")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", offline_import)
    monkeypatch.setattr(socket.socket, "connect", lambda *_a, **_k: pytest.fail("network attempted"))
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("network attempted"))


def rig(monkeypatch, **kwargs):
    cuda = FakeCuda()
    torch = FakeTorch(cuda, **kwargs)
    support = FakeSupport(cuda)
    monkeypatch.setattr(probe.time, "perf_counter", lambda: cuda.clock)
    collect = gc.collect
    def collected():
        cuda.events.append(("gc",))
        return collect()
    monkeypatch.setattr(probe.gc, "collect", collected)
    return cuda, torch, support


@pytest.mark.parametrize("before,after,classification", [
    ((4_194_304, 4_194_304), (0, 0), "torch-releasable-workspace-family"),
    ((64, 128), (32, 128), "torch-process-residue"),
    ((0, 128), (0, 64), "torch-process-residue"),
    ((0, 0), (0, 0), "needs-project-probe"),
])
def test_p1_three_classifications_exact_bytes_snapshots_and_sequence(tmp_path, monkeypatch, before, after, classification):
    cuda, torch, support = rig(monkeypatch, before=before, after=after)
    record = probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    assert record == read_json(tmp_path / "probe_residue.json")
    assert record["status"] == "success" and record["classification"] == classification
    assert record["attribution_confirmed"] is (classification != "needs-project-probe")
    assert record["requires_p2"] is (classification == "needs-project-probe")
    assert record["initial"]["allocated_bytes"] == record["initial"]["reserved_bytes"] == 0
    assert (record["before_clear"]["allocated_bytes"], record["before_clear"]["reserved_bytes"]) == before
    assert (record["after_clear"]["allocated_bytes"], record["after_clear"]["reserved_bytes"]) == after
    assert all("sha256" in record[key]["memory_snapshot"] for key in ("initial", "before_clear", "after_clear"))
    assert record["torch_version"] == torch.__version__ and record["gpu_uuid"] == "GPU-fixture"
    assert record["deadline_perf_counter"] == 70.0 and record["cap_seconds"] == 60.0
    assert record["started_perf_counter"] == record["finished_perf_counter"] == 10.0
    assert record["elapsed_seconds"] == record["matmul_call_seconds"] == 0.0
    assert record["scientific_claim"] is False and record["accepted_cost_rows"] is False
    assert record["test_read"] is False and record["training_updates"] == 0 and record["limitations"]
    assert record["private_api"]["available"] is True and record["private_api"]["called"] is True
    index = cuda.events.index(("matmul",))
    assert cuda.events[index + 1:] == [("gc",), ("sync", "cuda:0"), ("empty",), ("sync", "cuda:0"),
        ("allocated", "cuda:0"), ("reserved", "cuda:0"), ("snapshot", "cuda:0"), ("clear",),
        ("gc",), ("sync", "cuda:0"), ("empty",), ("sync", "cuda:0"),
        ("allocated", "cuda:0"), ("reserved", "cuda:0"), ("snapshot", "cuda:0")]
    assert not (tmp_path / "probe_p2").exists()


@pytest.mark.parametrize("api,clear_error", [(False, False), (True, True)])
def test_residue_with_missing_or_failing_private_api_is_not_confirmation(tmp_path, monkeypatch, api, clear_error):
    cuda, torch, support = rig(monkeypatch, api=api, clear_error=clear_error)
    with pytest.raises(RuntimeError, match="private"):
        probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    record = read_json(tmp_path / "probe_residue.json")
    assert record["status"] == "failed" and record["classification"] == "unattributed"
    assert record["attribution_confirmed"] is False and record["requires_p2"] is False
    assert record["before_clear"]["allocated_bytes"] == record["after_clear"]["allocated_bytes"] == 64
    assert record["private_api"]["available"] is api and record["private_api"]["called"] is api


def test_missing_private_api_with_zero_bytes_can_only_request_project_probe(tmp_path, monkeypatch):
    _, torch, support = rig(monkeypatch, before=(0, 0), api=False)
    record = probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    assert record["classification"] == "needs-project-probe" and record["requires_p2"] is True
    assert record["attribution_confirmed"] is False and record["private_api"]["called"] is False


@pytest.mark.parametrize("mode", ["P1", "P2"])
def test_nonzero_fresh_initial_refuses_before_matmul_or_project_import(tmp_path, monkeypatch, mode):
    cuda, torch, support = rig(monkeypatch, before=(0, 0))
    if mode == "P2":
        probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    cuda.events.clear()
    cuda.bytes = (16, 32)
    with pytest.raises(RuntimeError, match="zero allocated/reserved"):
        if mode == "P1":
            probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
        else:
            probe.probe_p2(torch, support, protocol(), tmp_path, 70.0,
                           load_evaluate=lambda: pytest.fail("project imported"))
    path = tmp_path / ("probe_residue.json" if mode == "P1" else "probe_project_path.json")
    record = read_json(path)
    assert record["initial"]["allocated_bytes"] == 16 and record["initial"]["reserved_bytes"] == 32
    assert record["status"] == "failed" and record["attribution_confirmed"] is False
    assert (tmp_path / "guard_refusal.json").is_file()
    assert ("matmul",) not in cuda.events and ("clear",) not in cuda.events


@pytest.mark.parametrize("stage", ["initial", "before_clear", "after_clear"])
def test_snapshot_failure_preserves_raw_error_and_stops_unattributed(tmp_path, monkeypatch, stage):
    cuda, torch, support = rig(monkeypatch)
    original = support.allocator_record
    calls = []
    def snapshot(*args):
        calls.append(len(calls))
        cuda.snapshot_error = len(calls) == {"initial": 1, "before_clear": 2, "after_clear": 3}[stage]
        return original(*args)
    monkeypatch.setattr(support, "allocator_record", snapshot)
    with pytest.raises(ValueError, match="memory_snapshot"):
        probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    record = read_json(tmp_path / "probe_residue.json")
    assert record[stage]["memory_snapshot"] == {"error": "fixture snapshot unavailable"}
    assert record["classification"] == "unattributed" and record["attribution_confirmed"] is False


def p2_rig(tmp_path, monkeypatch):
    cuda, torch, support = rig(monkeypatch, before=(0, 0), after=(0, 0))
    probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    cuda.events.clear()
    return cuda, torch, support


def fake_evaluate(cuda, residues):
    values = iter(residues)
    def evaluate(manifest, **kwargs):
        assert manifest == "fixture-val-manifest-not-read"
        assert kwargs["checkpoint"] == "fixture-checkpoint-not-read" and kwargs["max_samples"] == 32
        assert kwargs["reasoning_steps"] == 3 and kwargs["device_name"] == "cuda:0"
        assert kwargs["deadline"] == 70.0
        directory = kwargs["output_dir"]
        assert not directory.exists()
        assert "probe_p2" in directory.parts
        directory.mkdir(parents=True, exist_ok=False)
        lead = kwargs["lead_hours"][0]
        cuda.events.append(("evaluate", lead))
        cuda.bytes = next(values)
        return {"lead_hours": [lead], "n_evaluated": 22 if lead == 6 else 21}
    return evaluate


@pytest.mark.parametrize("residues", [[(128, 256), (128, 256)], [(0, 0), (0, 256)]])
def test_p2_only_after_zero_p1_replays_first_two_in_one_process_without_release(tmp_path, monkeypatch, residues):
    cuda, torch, support = p2_rig(tmp_path, monkeypatch)
    p1_bytes = (tmp_path / "probe_residue.json").read_bytes()
    origin_checks = []
    record = probe.probe_p2(torch, support, protocol(), tmp_path, 70.0,
                           fake_evaluate(cuda, residues), verify_origins=lambda: origin_checks.append(True))
    assert record == read_json(tmp_path / "probe_project_path.json")
    assert record["classification"] == "project-path-residue" and record["attribution_confirmed"] is True
    assert record["status"] == "success" and record["cap_seconds"] == 120.0
    assert record["p1_probe_sha256"] == hashlib.sha256(p1_bytes).hexdigest()
    assert record["initial"]["allocated_bytes"] == record["initial"]["reserved_bytes"] == 0
    assert [item["lead"] for item in record["evaluations"]] == [6, 12]
    assert [(item["after_evaluation"]["allocated_bytes"], item["after_evaluation"]["reserved_bytes"])
            for item in record["evaluations"]] == residues
    assert all(item["replay"]["exact_numeric_replay"] for item in record["evaluations"])
    assert len(origin_checks) == 3 and cuda.events.count(("reset", "cuda:0")) == 1
    assert ("clear",) not in cuda.events and ("matmul",) not in cuda.events
    assert (tmp_path / "probe_residue.json").read_bytes() == p1_bytes
    assert not list(tmp_path.rglob("cost_measurement.json")) and not (tmp_path / "seed41").exists()


def test_p2_zero_residue_stops_parent_with_unexplained_failure(tmp_path, monkeypatch):
    cuda, torch, support = p2_rig(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="unexplained-original-failure"):
        probe.probe_p2(torch, support, protocol(), tmp_path, 70.0, fake_evaluate(cuda, [(0, 0), (0, 0)]))
    record = read_json(tmp_path / "probe_project_path.json")
    assert record["status"] == "failed" and record["classification"] == "unexplained-original-failure"
    assert record["attribution_confirmed"] is False and len(record["evaluations"]) == 2


@pytest.mark.parametrize("failure", ["evaluate", "compare", "false_replay", "snapshot", "deadline"])
def test_p2_errors_retain_bytes_but_never_confirm_or_run_second_task(tmp_path, monkeypatch, failure):
    cuda, torch, support = p2_rig(tmp_path, monkeypatch)
    original = fake_evaluate(cuda, [(64, 128), (64, 128)])
    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        if failure == "evaluate":
            raise RuntimeError("evaluation failed")
        if failure == "snapshot":
            cuda.snapshot_error = True
        return result
    if failure in ("compare", "false_replay", "deadline"):
        def compare(*args):
            if failure == "compare":
                raise ValueError("exact replay differs")
            if failure == "deadline":
                cuda.clock = 70.0
            return {"exact_numeric_replay": failure != "false_replay"}
        monkeypatch.setattr(support, "compare_evaluation", compare)
    with pytest.raises((RuntimeError, ValueError)):
        probe.probe_p2(torch, support, protocol(), tmp_path, 70.0, evaluate)
    record = read_json(tmp_path / "probe_project_path.json")
    assert record["status"] == "failed" and record["classification"] == "unattributed"
    assert record["attribution_confirmed"] is False and len(record["evaluations"]) == 1
    assert record["evaluations"][0]["after_evaluation"]["allocated_bytes"] == 64
    assert ("evaluate", 12) not in cuda.events


@pytest.mark.parametrize("field,value", [("classification", "torch-process-residue"), ("status", "failed"),
    ("requires_p2", False), ("protocol_sha256", "other"), ("gpu_uuid", "GPU-other"),
    ("torch_version", "other"), ("elapsed_seconds", 60.001), ("training_updates", True)])
def test_p2_requires_bound_successful_zero_p1_proof(tmp_path, monkeypatch, field, value):
    cuda, torch, support = p2_rig(tmp_path, monkeypatch)
    prior = read_json(tmp_path / "probe_residue.json")
    prior[field] = value
    (tmp_path / "probe_residue.json").write_text(json.dumps(prior), encoding="utf-8")
    with pytest.raises(ValueError, match="P2 requires"):
        probe.probe_p2(torch, support, protocol(), tmp_path, 70.0,
                       load_evaluate=lambda: pytest.fail("project imported"))
    assert not cuda.events and not (tmp_path / "probe_project_path.json").exists()


def test_p2_nonzero_p1_bytes_refuse_even_with_forged_zero_classification(tmp_path, monkeypatch):
    cuda, torch, support = p2_rig(tmp_path, monkeypatch)
    prior = read_json(tmp_path / "probe_residue.json")
    prior["before_clear"]["reserved_bytes"] = 1
    (tmp_path / "probe_residue.json").write_text(json.dumps(prior), encoding="utf-8")
    with pytest.raises(ValueError, match="P2 forbidden"):
        probe.probe_p2(torch, support, protocol(), tmp_path, 70.0, lambda *_: pytest.fail("evaluate called"))
    assert not cuda.events


@pytest.mark.parametrize("mode", ["P1", "P2"])
def test_proof_files_exclusive_create_and_no_probe_retry(tmp_path, monkeypatch, mode):
    cuda, torch, support = rig(monkeypatch, before=(0, 0))
    name = "probe_residue.json" if mode == "P1" else "probe_project_path.json"
    path = tmp_path / name
    path.write_bytes(b"existing-proof")
    with pytest.raises(FileExistsError):
        function = probe.probe_p1 if mode == "P1" else probe.probe_p2
        function(torch, support, protocol(), tmp_path, 70.0)
    assert path.read_bytes() == b"existing-proof" and not cuda.events


def request_fixture(tmp_path):
    output = tmp_path / "input_repo" / "outputs" / "probe-run"
    output.mkdir(parents=True)
    metadata = {"wrapper_location": {"input_repo": str(tmp_path / "input_repo")}}
    write_json(output / "protocol.json", metadata)
    return {"protocol": str(output / "protocol.json"), "output": str(output),
            "code_root": str(tmp_path / "code"), "mode": "P1", "deadline": 70.0}


@pytest.mark.parametrize("mode,deadline", [("P1", 70.0), ("P2", 130.0)])
def test_request_exact_fields_frozen_support_namespace_and_deadline_boundary(tmp_path, monkeypatch, mode, deadline):
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 10.0)
    request = request_fixture(tmp_path)
    request.update(mode=mode, deadline=deadline)
    args = probe.parse_request(request)
    assert args.mode == mode and args.deadline == deadline
    assert args.protocol == args.output / "protocol.json" and args.code_root == tmp_path / "code"
    assert args.support == Path(probe.__file__).resolve().parents[1] / "training/r7_n1_cost_replay.py"


@pytest.mark.parametrize("deadline", [True, "70", None, float("nan"), float("inf"), 10.0, 9.0, 70.0001])
def test_p1_invalid_expired_or_overlong_deadline_refuses_before_metadata(tmp_path, monkeypatch, deadline):
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 10.0)
    value = {"protocol": str(tmp_path / "missing"), "output": str(tmp_path), "code_root": str(tmp_path),
             "mode": "P1", "deadline": deadline}
    with pytest.raises((ValueError, RuntimeError), match="deadline"):
        probe.parse_request(value)


@pytest.mark.parametrize("change", ["extra", "missing", "mode", "output", "protocol", "symlink", "p2_cap"])
def test_request_scope_refusals(tmp_path, monkeypatch, change):
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 10.0)
    request = request_fixture(tmp_path)
    if change == "extra": request["seed"] = 41
    if change == "missing": request.pop("mode")
    if change == "mode": request["mode"] = "P3"
    if change == "output":
        output = tmp_path / "elsewhere"
        output.mkdir()
        write_json(output / "protocol.json", {"wrapper_location": {"input_repo": str(tmp_path / "input_repo")}})
        request.update(output=str(output), protocol=str(output / "protocol.json"))
    if change == "protocol": request["protocol"] = str(tmp_path / "different.json")
    if change == "symlink":
        link = tmp_path / "linked-output"
        link.symlink_to(request["output"], target_is_directory=True)
        request.update(output=str(link), protocol=str(link / "protocol.json"))
    if change == "p2_cap": request.update(mode="P2", deadline=130.001)
    with pytest.raises(ValueError):
        probe.parse_request(request)


def frozen_runtime(tmp_path, monkeypatch):
    value = protocol()
    code = tmp_path / "archived_code"
    wrapper = code / "measurement"
    output = tmp_path / "input_repo" / "outputs" / "probe-run"
    output.mkdir(parents=True)
    names = (*probe.REQUIRED_WRAPPER_FILES, "scripts/measure_r7_n1_eval_worker.py")
    archived = ("training/r7_evaluate.py", "training/r7_experiment.py", "model/__init__.py")
    for root, files in ((wrapper, names), (code, archived)):
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# CPU fixture source\n", encoding="utf-8")
    support = FakeSupport(FakeCuda())
    support.__file__ = str(wrapper / probe.REQUIRED_WRAPPER_FILES[1])
    support.plain_path = probe._plain_path
    support.pinned_file = lambda path, sha, root: pinned(path, sha, root)
    support.verify_registration = lambda path: read_json(path)
    support.verify_pins = lambda pins: support.cuda.events.append(("pins",))
    value.update(wrapper_location={"root": str(wrapper), "input_repo": str(tmp_path / "input_repo")},
                 code_identity={"measurement_source_sha256": {n: file_sha(wrapper / n) for n in names}},
                 extracted_code=[{"path": n, "sha256": file_sha(code / n)} for n in archived], input_pins=[])
    write_json(output / "protocol.json", value)
    args = SimpleNamespace(protocol=output / "protocol.json", code_root=code, output=output,
                           mode="P1", deadline=70.0, support=Path(support.__file__))
    monkeypatch.setattr(probe, "__file__", str(wrapper / probe.REQUIRED_WRAPPER_FILES[0]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 10.0)
    return args, value, support


def pinned(path, sha, root):
    resolved = probe._plain_path(path)
    if not resolved.is_relative_to(probe._plain_path(root)) or file_sha(resolved) != sha:
        raise ValueError("frozen file digest/root mismatch")


def test_frozen_runtime_verifies_registration_source_task_wrapper_and_archived_pins(tmp_path, monkeypatch):
    args, value, support = frozen_runtime(tmp_path, monkeypatch)
    assert probe.verify_runtime(args, support) == value
    assert support.cuda.events == [("pins",), ("inputs", 6), ("inputs", 12)]
    imported = probe.load_support(args.support)
    assert Path(imported.__file__) == args.support


@pytest.mark.parametrize("change", ["support_origin", "probe_hash", "wrapper_hash", "archived_hash",
    "missing_probe_pin", "missing_evaluator_pin", "source_pin", "task_pin", "registration", "uuid", "two_uuids", "first_two"])
def test_frozen_runtime_refuses_changed_identity_before_any_cuda(tmp_path, monkeypatch, change):
    args, value, support = frozen_runtime(tmp_path, monkeypatch)
    if change == "support_origin": support.__file__ = str(tmp_path / "live_support.py")
    if change == "probe_hash": Path(probe.__file__).write_text("# changed\n", encoding="utf-8")
    if change == "wrapper_hash": args.support.write_text("# changed\n", encoding="utf-8")
    if change == "archived_hash": (args.code_root / "training/r7_evaluate.py").write_text("# changed\n", encoding="utf-8")
    if change == "missing_probe_pin": value["code_identity"]["measurement_source_sha256"].pop(probe.REQUIRED_WRAPPER_FILES[0])
    if change == "missing_evaluator_pin": value["extracted_code"].pop(0)
    if change in ("source_pin", "task_pin", "registration"):
        def refused(*args):
            raise ValueError("fixture identity refusal")
        function = {"source_pin": "verify_pins", "task_pin": "verify_task_inputs", "registration": "verify_registration"}[change]
        setattr(support, function, refused)
    if change == "uuid": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    if change == "two_uuids": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture,GPU-other")
    if change == "first_two": value["tasks"].reverse()
    args.protocol.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        probe.verify_runtime(args, support)
    assert not any(event[0] in ("init", "headroom", "set", "reset", "uuid") for event in support.cuda.events)


@pytest.mark.parametrize("origin", ["live", "missing", "unpinned"])
def test_project_import_origins_reject_live_or_unpinned_modules(tmp_path, monkeypatch, origin):
    code = tmp_path / "code"
    code.mkdir()
    support = SimpleNamespace(plain_path=probe._plain_path, pinned_file=pinned)
    value = {"extracted_code": []}
    path = tmp_path / "live.py" if origin == "live" else code / "unpinned.py"
    path.write_text("# fixture\n", encoding="utf-8")
    module = SimpleNamespace(__file__=str(path)) if origin != "missing" else SimpleNamespace()
    monkeypatch.setattr(probe.sys, "modules", {"training.r7_evaluate": module})
    with pytest.raises(ValueError, match="non-archived"):
        probe.verify_import_origins(value, code, support)


def test_project_import_origins_require_actual_archived_bytes(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    source = code / "evaluator.py"
    source.write_text("# fixture\n", encoding="utf-8")
    value = {"extracted_code": [{"path": "evaluator.py", "sha256": file_sha(source)}]}
    support = SimpleNamespace(plain_path=probe._plain_path, pinned_file=pinned)
    monkeypatch.setattr(probe.sys, "modules", {"training.r7_evaluate": SimpleNamespace(__file__=str(source))})
    probe.verify_import_origins(value, code, support)
    source.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        probe.verify_import_origins(value, code, support)


def test_p1_run_never_loads_or_imports_project_evaluator_model(tmp_path, monkeypatch):
    cuda, torch, support = rig(monkeypatch, before=(0, 0))
    args = SimpleNamespace(support=tmp_path / "support.py", mode="P1", output=tmp_path, deadline=70.0)
    claims = []
    monkeypatch.setattr(support, "claim_parent_launch", lambda *values: claims.append(values), raising=False)
    monkeypatch.setattr(probe, "load_support", lambda _: support)
    monkeypatch.setattr(probe, "verify_runtime", lambda *_: protocol())
    monkeypatch.setattr(probe, "load_archived_evaluator", lambda *_: pytest.fail("P1 project evaluator import"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    original_import = builtins.__import__
    def no_project(name, *args, **kwargs):
        if name.split(".")[0] in ("training", "model", "data"):
            pytest.fail("project import in P1")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_project)
    result = probe.run(args)
    assert result["classification"] == "needs-project-probe" and ("matmul",) in cuda.events
    assert claims == [(protocol(), tmp_path, "probe_P1", 70.0)]


@pytest.mark.parametrize("mode", ["P1", "P2"])
def test_parent_claim_refusal_precedes_torch_import_and_any_cuda(tmp_path, monkeypatch, mode):
    cuda, _, support = rig(monkeypatch)
    args = SimpleNamespace(support=tmp_path / "support.py", mode=mode, output=tmp_path, deadline=70.0)
    events = []
    monkeypatch.setattr(probe, "load_support", lambda _: support)
    monkeypatch.setattr(probe, "verify_runtime", lambda *_: events.append("runtime") or protocol())
    monkeypatch.setattr(probe, "verify_p1_gate", lambda *_: events.append("p1_gate"))
    def refused(value, output, kind, deadline):
        assert (value, output, kind, deadline) == (protocol(), tmp_path, "probe_" + mode, 70.0)
        events.append("claim")
        raise ValueError("parent launch capability refused")
    monkeypatch.setattr(support, "claim_parent_launch", refused, raising=False)
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            pytest.fail("torch imported before successful parent claim")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(ValueError, match="parent launch capability"):
        probe.run(args)
    assert events == (["runtime", "p1_gate", "claim"] if mode == "P2" else ["runtime", "claim"])
    assert not cuda.events and not list(tmp_path.iterdir())


def test_exact_torch_version_mismatch_refuses_before_cuda_or_proof(tmp_path, monkeypatch):
    cuda, torch, support = rig(monkeypatch)
    torch.__version__ = "2.11.0+cu128"
    with pytest.raises(ValueError, match="exact version"):
        probe.probe_p1(torch, support, protocol(), tmp_path, 70.0)
    assert not cuda.events and not list(tmp_path.iterdir())


def test_probe_network_connections_fail_closed(monkeypatch):
    probe.deny_network()
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("example.invalid", 443))
