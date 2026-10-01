"""CPU counterproofs for N1 fresh-cell scheduling and read-only headroom guards."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import measure_r7_n1_eval_cost as parent
from scripts import measure_r7_n1_eval_worker as worker
from training import r7_n1_cost_replay as support
from test_r7_n1_eval_cost import FakeCuda, authorization, original_protocol, protocol, valid_rows


@pytest.mark.parametrize("free", [0, 2047])
def test_headroom_refusal_persists_query_bytes_before_raise(tmp_path, monkeypatch, free):
    responses = iter([f"0, GPU-a, {free}, 24576\n1, GPU-b, 9000, 24576\n", "GPU-a, 12345\n"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(support.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=next(responses)))
    path = tmp_path / "guard_refusal.json"
    with pytest.raises(RuntimeError, match="headroom insufficient"):
        support.gpu_headroom("cuda:0", refusal_path=path)
    record = support.read_json(path)
    assert record["device_query"]["free_bytes"] == free * 1024 * 1024
    assert record["device_query"]["external_pids"] == [12345]
    assert record["scientific_claim"] is False and record["status"] == "refused"


def test_headroom_boundary_allows_neighbors_without_signals(monkeypatch):
    responses = iter(["0, GPU-a, 2048, 24576\n1, GPU-b, 9000, 24576\n", "GPU-a, 12345\n"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=next(responses))
    monkeypatch.setattr(support.subprocess, "run", query)
    record = support.gpu_headroom("cuda:0")
    assert record["free_mib"] == 2048 and record["external_pids"] == [12345]
    assert len(calls) == 2 and all(command[0] == "nvidia-smi" for command in calls)
    assert all(any(item.startswith("--query-") for item in command) for command in calls)


@pytest.mark.parametrize("preferred_free", [0, 2048])
def test_startup_chooses_available_gpu_not_exclusive_gpu(tmp_path, monkeypatch, preferred_free):
    responses = iter([f"0, GPU-a, {preferred_free}, 24576\n1, GPU-b, 2048, 24576\n",
                      "GPU-a, 12345\nGPU-b, 54321\n"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(support.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=next(responses)))
    record = support.select_shared_gpu("cuda:0", 1e12, tmp_path / "guard_refusal.json")
    assert record["gpu_uuid"] == ("GPU-a" if preferred_free else "GPU-b")
    assert len(record["startup_observations"]) == 1
    assert not (tmp_path / "guard_refusal.json").exists()


def test_startup_wait_is_bounded_read_only_and_retains_both_cards(tmp_path, monkeypatch):
    clock, sleeps, calls = [0.0], [], []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(support.time, "perf_counter", lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="0, GPU-a, 2047, 24576\n1, GPU-b, 0, 24576\n"
                               if "--query-gpu=index,uuid,memory.free,memory.total" in command else "")
    monkeypatch.setattr(support.time, "sleep", sleep)
    monkeypatch.setattr(support.subprocess, "run", query)
    path = tmp_path / "guard_refusal.json"
    with pytest.raises(RuntimeError, match="both GPUs"):
        support.select_shared_gpu("cuda:0", 1000, path, wait_seconds=120)
    assert sleeps == [60, 60] and clock[0] == 120
    assert len(support.read_json(path)["observations"]) == 2
    assert all(command[0] == "nvidia-smi" for command in calls)


def test_allocator_refusal_contains_snapshot_summary_before_reset(tmp_path):
    class SnapshotCuda(FakeCuda):
        def memory_snapshot(self):
            return [{"device": 1, "total_size": 256,
                     "blocks": [{"state": "active_allocated", "size": 128}]}]
    cuda = SnapshotCuda(128, 256)
    path = tmp_path / "guard_refusal.json"
    with pytest.raises(RuntimeError, match="zero allocated/reserved"):
        support.reset_measurement(cuda, "cuda:1", refusal_path=path)
    record = support.read_json(path)["allocator"]
    assert record["allocated_bytes"] == 128 and record["reserved_bytes"] == 256
    assert record["memory_snapshot"]["block_state_bytes"] == {"active_allocated": 128}
    assert record["memory_snapshot"]["segments"] == 1
    assert not any(event[0] == "reset" for event in cuda.events)


@pytest.mark.parametrize("field,value", [("device_policy", "exclusive"), ("min_free_mib", 2047),
    ("evaluations_per_child", 5), ("whole_wall_seconds_cap", 1201)])
def test_v2_registration_rejects_rehashed_topology_or_resource_drift(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(support, "ORIGINAL_PROTOCOL_SHA", original_protocol()["protocol_sha256"])
    p = protocol()
    p[field] = value
    p["protocol_sha256"] = support.digest({key: item for key, item in p.items() if key != "protocol_sha256"})
    support.write_json(tmp_path / "protocol.json", p)
    with pytest.raises(ValueError, match="mismatch"):
        support.verify_registration(tmp_path / "protocol.json")


@pytest.mark.parametrize("field,value", [("scope", "n1-evaluation-cost-supplement"),
    ("device_policy", "exclusive"), ("min_free_mib", 2047), ("probe_seconds_cap", 61),
    ("conditional_p2_seconds_cap", 121), ("whole_wall_seconds_cap", 1201),
    ("failure_policy", "retry")])
def test_v2_authorization_does_not_inherit_v1_or_expand_scope(field, value):
    value_map = authorization()
    value_map[field] = value
    with pytest.raises(ValueError, match="authorization"):
        support.verify_authorization(value_map)


def fixture_parent(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    auth = tmp_path / "authorization.json"
    support.write_json(auth, authorization())
    monkeypatch.setattr(support, "ORIGINAL_PROTOCOL_SHA", original_protocol()["protocol_sha256"])
    p = protocol()
    monkeypatch.setattr(parent, "inspect_archive", lambda *_: {"original_protocol": {"data": {"manifests_dir": str(tmp_path)}}})
    monkeypatch.setattr(parent, "reserve_output", lambda *_: out)
    monkeypatch.setattr(parent, "archive_wrapper", lambda *_: {"measurement_source_sha256": {}})
    monkeypatch.setattr(parent, "extract_code", lambda *_: [])
    monkeypatch.setattr(parent, "freeze_wrapper", lambda *_: {})
    p["input_pins"] = []
    monkeypatch.setattr(parent, "registration", lambda *_: copy.deepcopy(p))
    monkeypatch.setattr(parent, "select_shared_gpu", lambda *_a, **_k: {"gpu_uuid": "GPU-a", "physical_device": "cuda:0"})
    monkeypatch.setattr(parent, "gpu_headroom", lambda *_a, **_k: {"gpu_uuid": "GPU-a", "free_mib": 2048})
    monkeypatch.setattr(parent, "verify_pins", lambda *_: None)
    def verify_probe(report, *_args):
        if report.get("status") != "success":
            raise RuntimeError("residue probe cannot attribute a failed observation")
    monkeypatch.setattr(parent, "verify_probe_report", verify_probe)
    monkeypatch.setattr(parent, "collect_rows", lambda *_: valid_rows())
    monkeypatch.setattr(parent, "write_cost_tables", lambda output, *_: support.write_json(output / "cost_views.json", {}))
    clock = [100.0]
    monkeypatch.setattr(parent.time, "perf_counter", lambda: clock[0])
    return out, auth, clock


def test_parent_launches_exactly_thirty_single_cell_children_after_probe(tmp_path, monkeypatch):
    out, auth, clock = fixture_parent(tmp_path, monkeypatch)
    sequence = []
    def probe(*args, **kwargs):
        sequence.append("P1")
        clock[0] += 1
        return {"status": "success", "attribution_confirmed": True, "requires_p2": False}
    def child(output, code, seed, arm, lead, **kwargs):
        sequence.append((seed, arm, lead))
        assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-a"
        assert kwargs["deadline"] == 996.0
        clock[0] += 1
    monkeypatch.setattr(parent, "run_probe", probe)
    monkeypatch.setattr(parent, "run_owned_child", child)
    parent.run(tmp_path, out, auth, "cuda:1")
    assert sequence == ["P1", *itertools.product(support.SEEDS, support.ARMS, support.LEADS)]
    assert len(list(out.glob("spawn_seed*.json"))) == 30
    assert support.read_json(out / "attempt.json")["gpu_phase_seconds"] == 31
    assert support.read_json(out / "result.json")["status"] == "success"


@pytest.mark.parametrize("probe_result", [
    {"status": "failed", "attribution_confirmed": False, "requires_p2": False},
    {"status": "success", "attribution_confirmed": False, "requires_p2": False},
])
def test_unattributed_probe_stops_before_any_evaluation_and_charges(tmp_path, monkeypatch, probe_result):
    out, auth, clock = fixture_parent(tmp_path, monkeypatch)
    def probe(*args, **kwargs):
        clock[0] += 3
        return probe_result
    monkeypatch.setattr(parent, "run_probe", probe)
    monkeypatch.setattr(parent, "run_owned_child", lambda *_a, **_k: pytest.fail("evaluation launched"))
    with pytest.raises(RuntimeError, match="cannot attribute"):
        parent.run(tmp_path, out, auth, "cuda:1")
    assert support.read_json(out / "attempt.json")["gpu_phase_seconds"] == 3
    assert not (out / "result.json").exists()


def test_zero_torch_probe_runs_conditional_p2_before_evaluations(tmp_path, monkeypatch):
    out, auth, clock = fixture_parent(tmp_path, monkeypatch)
    calls = []
    def probe(output, code, mode, **kwargs):
        calls.append(mode)
        clock[0] += 1
        return {"status": "success", "requires_p2": mode == "P1", "attribution_confirmed": mode == "P2"}
    monkeypatch.setattr(parent, "run_probe", probe)
    def stop(*args, **kwargs):
        calls.append("first-cell")
        raise RuntimeError("test stop")
    monkeypatch.setattr(parent, "run_owned_child", stop)
    with pytest.raises(RuntimeError, match="test stop"):
        parent.run(tmp_path, out, auth, "cuda:1")
    assert calls == ["P1", "P2", "first-cell"]
    assert support.read_json(out / "attempt.json")["gpu_phase_seconds"] == 2


def test_owned_worker_request_binds_a_single_lead(tmp_path, monkeypatch):
    requests = []
    support.write_json(tmp_path / "protocol.json", {"protocol_sha256": "protocol"})
    class Child:
        pid = 123
        returncode = 0
        def communicate(self, input, timeout):
            requests.append(json.loads(input))
        def poll(self): return 0
    monkeypatch.setattr(parent.subprocess, "Popen", lambda *_a, **_k: Child())
    parent.run_owned_child(tmp_path, tmp_path, 41, support.ARMS[0], 12, deadline=1e12, environment={})
    assert requests[0]["lead"] == 12
    assert (tmp_path / f"worker_seed41_{support.ARMS[0]}_lead012.log").is_file()


@pytest.mark.parametrize("change", ["missing", "wrong", "boolean"])
def test_worker_request_requires_one_original_lead(tmp_path, change):
    support.write_json(tmp_path / "protocol.json", {"wrapper_location": {"input_repo": str(tmp_path)}})
    output = tmp_path / "outputs/out"
    output.mkdir(parents=True)
    support.write_json(output / "protocol.json", {"wrapper_location": {"input_repo": str(tmp_path)}})
    request = {"protocol": str(output / "protocol.json"), "code_root": str(tmp_path), "output": str(output),
               "seed": 41, "arm": support.ARMS[0], "lead": 6, "deadline": 1e12}
    if change == "missing": request.pop("lead")
    if change == "wrong": request["lead"] = 18
    if change == "boolean": request["lead"] = True
    with pytest.raises(ValueError, match="request"):
        worker.parse_request(request)


def test_collector_refuses_multi_cell_worker_summary(tmp_path):
    p = protocol()
    support.write_json(tmp_path / f"worker_seed41_{support.ARMS[0]}_lead006.json",
                       {"status": "success", "protocol_sha256": p["protocol_sha256"], "rows": valid_rows()[:2]})
    with pytest.raises(ValueError, match="single-task"):
        parent.collect_rows(tmp_path, p)


def probe_proof(mode="P1", *, needs_p2=False):
    zero = {"allocated_bytes": 0, "reserved_bytes": 0, "memory_snapshot": {}}
    residue = {"allocated_bytes": 128, "reserved_bytes": 256, "memory_snapshot": {}}
    return {"format": "r7-n1-allocator-probe-v1", "mode": mode, "status": "success", "failure_reason": None,
            "protocol_sha256": "protocol", "gpu_uuid": "GPU-a", "execution_device": "cuda:0",
            "torch_version": "torch", "scientific_claim": False, "test_read": False, "training_updates": 0,
            "accepted_cost_rows": False, "cap_seconds": 60 if mode == "P1" else 120, "elapsed_seconds": 1.0,
            "initial": copy.deepcopy(zero), "before_clear": copy.deepcopy(zero if needs_p2 else residue),
            "after_clear": copy.deepcopy(zero), "requires_p2": needs_p2, "attribution_confirmed": not needs_p2,
            "classification": "needs-project-probe" if needs_p2 else "torch-releasable-workspace-family"}


def proof_protocol():
    return {"protocol_sha256": "protocol", "physical_gpu_uuid": "GPU-a", "execution_device": "cuda:0",
            "original_torch_version": "torch"}


@pytest.mark.parametrize("field,value", [("status", "failed"), ("protocol_sha256", "wrong"),
    ("gpu_uuid", "wrong"), ("torch_version", "wrong"), ("mode", "P2"), ("elapsed_seconds", 61),
    ("requires_p2", True), ("attribution_confirmed", False)])
def test_parent_probe_proof_rejects_failure_identity_or_branch_drift(tmp_path, field, value):
    report = probe_proof()
    support.write_json(tmp_path / "probe_residue.json", report)
    assert support.verify_probe_report(report, proof_protocol(), "P1", tmp_path) == report
    report[field] = value
    (tmp_path / "probe_residue.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="probe"):
        support.verify_probe_report(report, proof_protocol(), "P1", tmp_path)


def test_failed_p1_cannot_trigger_p2_or_hide_behind_success(tmp_path, monkeypatch):
    out, auth, clock = fixture_parent(tmp_path, monkeypatch)
    calls = []
    def probe(output, code, mode, **kwargs):
        calls.append(mode)
        clock[0] += 1
        return {"status": "failed", "requires_p2": True, "attribution_confirmed": False}
    monkeypatch.setattr(parent, "run_probe", probe)
    monkeypatch.setattr(parent, "run_owned_child", lambda *_a, **_k: pytest.fail("evaluation launched"))
    with pytest.raises(RuntimeError, match="cannot attribute"):
        parent.run(tmp_path, out, auth, "cuda:1")
    assert calls == ["P1"] and support.read_json(out / "attempt.json")["gpu_phase_seconds"] == 1


@pytest.mark.parametrize("target", ["same", "descendant", "ancestor"])
def test_output_never_writes_in_or_above_frozen_v1_failure_directory(tmp_path, monkeypatch, target):
    root = tmp_path / "repo"
    frozen = root / "outputs/r7_n1_eval_cost_supplement"
    frozen.mkdir(parents=True)
    archive, manifests = root / "outputs/original", root / "data/source/store/manifests"
    monkeypatch.setattr(parent, "ROOT", root)
    output = frozen if target == "same" else frozen / "v2" if target == "descendant" else root / "outputs"
    with pytest.raises(ValueError, match="overlaps"):
        parent.reserve_output(output, archive, manifests)
    assert list(frozen.iterdir()) == []


def test_owned_reap_timeout_is_bounded_and_preserves_no_neighbor_control(tmp_path):
    events = []
    class Child:
        def communicate(self, **kwargs): raise parent.subprocess.TimeoutExpired("owned", 1)
        def poll(self): return None
        def kill(self): events.append("owned-kill")
        def wait(self, timeout):
            events.append(timeout)
            raise parent.subprocess.TimeoutExpired("owned-reap", timeout)
    with pytest.raises(RuntimeError, match="exit unconfirmed"):
        parent.wait_owned_child(Child(), {}, 1e12, tmp_path / "owned.log")
    assert events == ["owned-kill", 2.0]


@pytest.mark.parametrize("change", ["valid", "expired", "oversized", "wrong_parent", "wrong_token", "replay"])
def test_parent_launch_claim_is_deadline_bound_single_use(tmp_path, monkeypatch, change):
    p = {"output_dir": str(tmp_path), "protocol_sha256": "protocol",
         "gpu_phase_deadline_perf_counter": 200.0, "whole_deadline_perf_counter": 300.0}
    deadline = 200.0
    monkeypatch.setattr(support.time, "perf_counter", lambda: 100.0)
    monkeypatch.setenv("N1_COST_LAUNCH_ID", "a" * 64)
    record = {"kind": "probe_P1", "launch_id": "a" * 64, "parent_pid": support.os.getppid(),
              "deadline": deadline, "protocol_sha256": "protocol"}
    if change == "expired": deadline = 99.0
    if change == "oversized": deadline = 201.0
    if change == "wrong_parent": record["parent_pid"] += 1
    if change == "wrong_token": record["launch_id"] = "b" * 64
    support.write_json(tmp_path / "launch_probe_P1.json", record)
    if change == "replay": support.claim_parent_launch(p, tmp_path, "probe_P1", deadline)
    if change == "valid":
        assert support.claim_parent_launch(p, tmp_path, "probe_P1", deadline) == record
        assert support.read_json(tmp_path / "claimed_probe_P1.json")["worker_pid"] == support.os.getpid()
    else:
        with pytest.raises((ValueError, FileExistsError)):
            support.claim_parent_launch(p, tmp_path, "probe_P1", deadline)


@pytest.mark.parametrize("field", ["seed", "lead"])
def test_fractional_worker_identifier_refuses_before_cuda(tmp_path, field):
    output = tmp_path / "outputs/run"
    output.mkdir(parents=True)
    support.write_json(output / "protocol.json", {"wrapper_location": {"input_repo": str(tmp_path)}})
    request = {"protocol": str(output / "protocol.json"), "code_root": str(tmp_path), "output": str(output),
               "seed": 41, "arm": support.ARMS[0], "lead": 6, "deadline": 1e12}
    request[field] = float(request[field])
    with pytest.raises(ValueError, match="outside its named scope"):
        worker.parse_request(request)


@pytest.mark.parametrize("diagnostic", ["allocator", "write"])
def test_failed_evaluation_keeps_original_exception_if_diagnostic_fails(tmp_path, monkeypatch, diagnostic):
    from test_r7_n1_eval_cost import evaluation_files
    task, directory, report = evaluation_files(tmp_path)
    task.update(seed=41, arm=support.ARMS[0], lead=6, checkpoint={"path": "fixture", "sha256": "pin"})
    monkeypatch.setattr(support, "verify_task_inputs", lambda *_: None)
    monkeypatch.setattr(support, "gpu_headroom", lambda *_a, **_k: {"gpu_uuid": "GPU-a", "free_mib": 2048})
    original_error = RuntimeError("original evaluator failure")
    def evaluate(*args, **kwargs): raise original_error
    if diagnostic == "allocator":
        monkeypatch.setattr(support, "allocator_record", lambda *_: (_ for _ in ()).throw(ValueError("diagnostic allocator")))
    else:
        original_write = support.write_json
        def write(path, value):
            if path.name.endswith("_failed_call.json"): raise ValueError("diagnostic write")
            original_write(path, value)
        monkeypatch.setattr(support, "write_json", write)
    p = {"device": "cuda:1", "execution_device": "cuda:0", "physical_gpu_uuid": "GPU-a", "val_manifest": "val"}
    torch = SimpleNamespace(cuda=FakeCuda(), device=lambda value: value)
    with pytest.raises(RuntimeError, match="original evaluator failure") as failure:
        worker.measure_task(task, p, directory, evaluate, torch, support, 1e12)
    assert failure.value is original_error
    assert (directory.parent / f"{directory.name}_device_before.json").is_file()
    assert not (directory / "cost_measurement.json").exists()
