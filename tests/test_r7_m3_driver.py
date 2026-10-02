"""M3 engineering counterproofs: tmp artifacts, CPU/fake gates, no real GPU/data."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from training import r7_m3_driver as driver
from training import r7_m3_identity as identity
from training import r7_m3_profile as profile
from training import r7_m3_protocol as protocol
from training import r7_m3_results as results
from training import r7_m3_worker as worker
from test_r7_process_supervision_moments import CHANNEL_NAMES, make_batch, make_context
from test_r7_process_supervision_paths import make_model


def _profile_fixture():
    measurement = {"parameters": 100, "trainable_parameters": 100,
                   "forward_flops": 200, "forward_backward_flops": 400}
    groups = {name: {"norm": 0.0} for name in ("shared", "query", "readout", "forecast_history", "forecast_drafts")}
    return {"scientific_claim": False, "limitations": ["tmp fake profile"], "device": "cpu",
            "reasoning_steps": 4, "probe_batch_size": 2, "gradient_probe_seed": 41,
            "flop_convention": "elementwise arithmetic uncounted",
            "measurements": {arm: dict(measurement) for arm in protocol.ARM_NAMES},
            "pairing": {str(seed): {"all_shared_pairs_identical": True, "same_tensor_set": True,
                         "full_initial_state_sha256": {arm: "d" * 64 for arm in protocol.ARM_NAMES}}
                        for seed in protocol.SEEDS},
            "gradient_ownership": {arm: {"scientific_claim": False, "limitations": ["fake"], "sample_id": ["fake0"],
                                   "components": {name: {"loss": 0.0, "groups": deepcopy(groups)}
                                                  for name in ("forecast", "input", "future", "draft", "total")}}
                                   for arm in protocol.ARM_NAMES}}


@pytest.fixture
def frozen(tmp_path):
    data = {"channels": list(CHANNEL_NAMES), "units": ["physical-unit"] * 17,
            "data_identity": "a" * 64, "val_data_identity": "b" * 64,
            "normalization_mean": [0.0] * 17, "normalization_std": [1.0] * 17,
            "evaluation_cases": {str(lead): {"n_available": 1,
                                  "cases": [["2016-02-18T00:00:00", [f"case+{lead}h"]]]}
                                  for lead in protocol.LEADS}}
    measurement = {"parameters": 100, "trainable_parameters": 100,
                   "forward_flops": 200, "forward_backward_flops": 400}
    pairing = {str(seed): {"all_shared_pairs_identical": True,
                          "full_initial_state_sha256": {arm: "d" * 64 for arm in protocol.ARM_NAMES}}
               for seed in protocol.SEEDS}
    payload = protocol.build_protocol(
        manifests=tmp_path / "inputs", output=tmp_path, dataset=data,
        sources={"val_manifest_sha256": "e" * 64},
        sidecar={"path": str(tmp_path / "sidecar" / "scale_metadata.json"), "identity": "f" * 64},
        code={"model_code_sha256": "c" * 64},
        profile=_profile_fixture(),
        authorization={}, authorization_sha256="1" * 64, gpu_uuid="GPU-fixed", estimated_peak_mib=342)
    protocol.write_json(tmp_path / "protocol.json", payload)
    (tmp_path / "workers").mkdir()
    return payload


def _refreeze(value):
    value["protocol_sha256"] = protocol.digest({key: item for key, item in value.items() if key != "protocol_sha256"})
    return value


def _authorization(path):
    payload = {"status": "authorized", "scope": protocol.AUTHORIZATION_SCOPE,
               "user_response": "named bounded M3 scope, retain failures/partial, stop without retry",
               "bounds": {"seeds": list(protocol.SEEDS), "arms": list(protocol.ARM_NAMES),
                          "updates": 400, "reasoning_steps": 4, "max_gpu_seconds": 3600.0,
                          "max_round_seconds": 1800.0, "max_worker_seconds": 1800.0, "gpu_policy": "shared"}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_protocol_exact_three_arms_and_frozen_budget(frozen):
    assert protocol.verify_protocol(Path(frozen["output"]) / "protocol.json") == frozen
    assert len(protocol.planned_jobs()) == 36
    assert sum(job["phase"] == "train" for job in protocol.planned_jobs()) == 6
    assert len({protocol.job_key(job) for job in protocol.planned_jobs()}) == 36
    assert all(arm[2] == dict(protocol.RW_A_CONFIG, default_reasoning_steps=4) for arm in protocol.ARMS)
    assert all(not arm[2].get("local_solver_state", False) for arm in protocol.ARMS)
    assert frozen["gpu"]["max_round_seconds"] == 1800
    assert frozen["gpu"]["max_gpu_seconds"] == 3600
    assert all(sum(weights.values()) in (0, .1) for weights in protocol.WEIGHTS.values())


@pytest.mark.parametrize("change", ["seed", "steps", "weight", "headroom", "round", "jobs", "units", "count"])
def test_protocol_refuses_rehashed_control_changes(frozen, change):
    invalid = deepcopy(frozen)
    if change == "seed":
        invalid["seeds"].append(43)
    elif change == "steps":
        invalid["arms"][0]["model_config"]["default_reasoning_steps"] = 3
    elif change == "weight":
        invalid["arms"][1]["supervision_weights"]["input_diagnostic_weight"] = .01
    elif change == "headroom":
        invalid["gpu"]["headroom_margin_mib"] = 0
    elif change == "round":
        invalid["gpu"]["max_round_seconds"] = 3600
    elif change == "jobs":
        invalid["jobs"].pop()
    elif change == "units":
        invalid["data"]["units"][0] = "normalized"
    else:
        invalid["arms"][0]["forward_backward_flops"] = 0
    with pytest.raises(ValueError):
        protocol.validate_protocol(_refreeze(invalid))


def test_authorization_is_named_exact_and_before_readers(tmp_path, monkeypatch):
    path = tmp_path / "authorization.json"
    actual = _authorization(path)
    assert protocol.verify_authorization(path) == actual
    actual["bounds"]["updates"] = 399
    path.write_text(json.dumps(actual), encoding="utf-8")
    monkeypatch.setattr(driver, "dataset_pins", lambda *_: pytest.fail("unauthorized data read"))
    with pytest.raises(ValueError, match="named M3"):
        driver.prepare(tmp_path / "inputs", tmp_path / "sidecar", tmp_path / "out", path, "GPU-fixed")
    assert not (tmp_path / "out").exists()


def test_shared_headroom_records_neighbor_and_includes_peak_margin(frozen):
    commands = []
    def query(args, **kwargs):
        commands.append(args)
        return ("GPU-other, 0, 200, 24576\nGPU-fixed, 1, 2400, 24576\n"
                if "--query-gpu=uuid,index,memory.free,memory.total" in args
                else "GPU-fixed, 999999, 15000\n")
    snapshot = driver.gpu_snapshot("GPU-fixed", query=query)
    gate = driver.verify_headroom(snapshot, frozen["gpu"])
    assert gate["required_free_mib"] == 2390
    assert gate["neighbors"][0]["pid"] == 999999
    assert all(command[0] == "nvidia-smi" for command in commands)
    with pytest.raises(RuntimeError, match="headroom"):
        driver.verify_headroom(snapshot, frozen["gpu"], observed_peak_mib=1000)
    with pytest.raises(ValueError, match="UUID"):
        driver.gpu_snapshot("GPU-missing", query=query)


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now


class FakeProcess:
    def __init__(self, clock, seconds=1, code=0, timeout=False):
        self.clock, self.seconds, self.code, self.timeout = clock, seconds, code, timeout
        self.pid, self.signals, self.done = 12345, [], False
    def wait(self, timeout):
        if self.done:
            return self.code
        if self.timeout and not self.signals:
            self.clock.now += timeout
            raise subprocess.TimeoutExpired("fake-owned", timeout)
        self.clock.now += self.seconds
        self.done = True
        return self.code
    def poll(self):
        return self.code if self.done else None
    def terminate(self):
        self.signals.append("terminate-owned")
    def kill(self):
        self.signals.append("kill-owned")


def _fake_round(frozen, *, seconds=1, fail_at=None, timeout_at=None, free=9999):
    clock, spawned, snapshots, processes = FakeClock(), [], [], []
    jobs = protocol.planned_jobs()
    def snapshot(uuid):
        snapshots.append(uuid)
        clock.now += .25
        return {"uuid": uuid, "free_mib": free, "neighbors": [{"pid": 88888}]}
    def popen(args, **kwargs):
        job = jobs[len(spawned)]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixed"
        assert args[args.index("--device") + 1] == "cuda:0"
        index = len(spawned)
        spawned.append(job)
        process = FakeProcess(clock, seconds=seconds, code=2 if index == fail_at else 0,
                              timeout=index == timeout_at)
        processes.append(process)
        if index != fail_at and index != timeout_at:
            protocol.write_json(worker.worker_result_path(frozen["output"], job),
                                {"status": "success", "job": job, "protocol_sha256": frozen["protocol_sha256"],
                                 "peak_reserved_bytes": 350 * 2 ** 20})
        return process
    return clock, spawned, snapshots, processes, popen, snapshot


def test_sequential_exact36_fresh_uuid_bound_processes_contiguous_charge(frozen):
    clock, spawned, snapshots, processes, popen, snapshot = _fake_round(frozen)
    attempt = driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                                  popen_factory=popen, snapshot_fn=snapshot)
    assert spawned == protocol.planned_jobs()
    assert len(snapshots) == len(processes) == 36
    assert attempt["gpu_phase_elapsed_seconds"] == 45.0
    assert attempt["gpu_hours_charged"] == 45 / 3600
    assert all(not process.signals for process in processes)
    assert attempt["status"] == "success"


@pytest.mark.parametrize("phase", ["train", "evaluate"])
def test_any_child_failure_stops_full_round_without_retry(frozen, phase):
    failure = 1 if phase == "train" else 6
    clock, spawned, _, processes, popen, snapshot = _fake_round(frozen, fail_at=failure)
    with pytest.raises(RuntimeError, match="no retry"):
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    attempt = protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")
    assert len(spawned) == failure + 1
    assert len(attempt["jobs_completed"]) == failure
    assert attempt["status"] == "failed" and attempt["partial"]
    assert all(not process.signals for process in processes)


def test_timeout_reaps_only_owned_handle_and_charges_cleanup(frozen):
    clock, spawned, _, processes, popen, snapshot = _fake_round(frozen, timeout_at=0)
    with pytest.raises(driver.BudgetLimited):
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    attempt = protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")
    assert len(spawned) == 1
    assert processes[0].signals == ["terminate-owned"]
    assert processes[0].done
    assert attempt["gpu_phase_elapsed_seconds"] == 1791
    assert attempt["budget_limited"] and attempt["status"] == "failed"


def test_global_budget_counts_gaps_and_stops_before_more_spawns(frozen):
    clock, spawned, _, _, popen, snapshot = _fake_round(frozen, seconds=100)
    with pytest.raises(driver.BudgetLimited):
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    assert len(spawned) < 36
    attempt = protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")
    assert attempt["gpu_phase_elapsed_seconds"] > len(spawned) * 100
    assert attempt["budget_limited"]


def test_low_headroom_fails_before_popen_and_preserves_receipt(frozen):
    clock, spawned, _, _, popen, snapshot = _fake_round(frozen, free=2000)
    with pytest.raises(RuntimeError, match="headroom"):
        driver.execute_jobs(Path(frozen["output"]) / "protocol.json", clock=clock,
                            popen_factory=popen, snapshot_fn=snapshot)
    assert spawned == []
    assert protocol.read_json(Path(frozen["output"]) / "execution_attempt.json")["jobs_completed"] == []


def test_no_repeat_and_x_artifacts_preserve_originals(frozen):
    output = Path(frozen["output"])
    protocol.write_json(output / "attempt.json", {"status": "failed"})
    with pytest.raises(FileExistsError, match="already"):
        driver.run_bounded_round(output, output / "missing-authorization.json")
    with pytest.raises(FileExistsError):
        protocol.write_json(output / "attempt.json", {"status": "success"})
    assert protocol.read_json(output / "attempt.json")["status"] == "failed"


def test_worker_offline_and_zero_baseline_counterproof(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    worker.deny_network()
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("example.invalid", 443))
    with pytest.raises(RuntimeError, match="offline"):
        socket.socket.connect(None, ("example.invalid", 443))
    calls = []
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append(device))
    assert worker.zero_allocator_baseline("cuda:0") == {"allocated_bytes": 0, "reserved_bytes": 0}
    assert calls == ["cuda:0"]
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 1)
    with pytest.raises(ValueError, match="zero"):
        worker.zero_allocator_baseline("cuda:0")


def test_m3_adapter_pins_real_int64_timestamps_without_legacy_changes(tmp_path, monkeypatch):
    import data.r7_store as store
    stamps = np.array([10, 20, 30], dtype=np.int64)
    root = {"time_ns": stamps}
    record = {"split": "train"}
    original = {"coarse_history": torch.zeros(2, 17, 2, 2)}
    class Reader:
        manifest = tmp_path / "train.jsonl"
        records = [record]
        def __len__(self):
            return 1
        def _store(self, _):
            return root
        def __getitem__(self, _):
            return original
    monkeypatch.setattr(store, "validate_record", lambda r, rec: [0, 1, 2])
    dataset = worker.M3Dataset(Reader.manifest, reader=Reader())
    sample = dataset[0]
    assert sample["history_time_ns"].dtype == torch.int64
    assert sample["history_time_ns"].tolist() == [10, 20]
    assert sample["init_time_ns"].item() == 20
    assert sample["target_time_ns"].item() == sample["valid_time_ns"].item() == 30
    assert sample["split"] == "train" and "split" not in original
    with pytest.raises(ValueError, match="train.jsonl"):
        worker.M3Dataset(tmp_path / "val.jsonl", reader=Reader())


def test_source_pins_never_open_test_or_decode_source(tmp_path, monkeypatch):
    manifests = tmp_path / "segment/store/manifests"
    manifests.mkdir(parents=True)
    source = manifests.parent.parent / "source.bin"
    source.write_bytes(b"engineering source identity fixture, not weather")
    source_hash = identity.sha256_file(source)
    for name, value in (("source_preflight.json", {"schema_version": 1, "source_path": str(source),
                        "fingerprint": {"scope": "full-local-file", "sha256": source_hash, "bytes": source.stat().st_size}}),
                        ("BUILD_COMPLETE.json", {"schema_version": 1, "build_complete": True})):
        protocol.write_json(manifests / name, value)
    protocol.write_json(manifests.parent.parent / "source_receipt.json", {"synthetic_fallback": False,
                        "local_artifact": {"sha256": source_hash}})
    for split in ("train", "val", "test"):
        (manifests / f"{split}.jsonl").write_text("opaque fixture", encoding="utf-8")
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != "test.jsonl"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    pins = identity.source_identity(manifests)
    assert pins["source_sha256"] == source_hash
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="disagree"):
        identity.source_identity(manifests)


def test_archive_only_active_import_closure_and_config(tmp_path):
    code = identity.archive_code(tmp_path)
    with zipfile.ZipFile(tmp_path / "code.zip") as archive:
        names = set(archive.namelist())
    assert {"training/r7_m3_worker.py", "training/r7_process_training_contract.py",
            "data/preprocess/r7_process_scale_sidecar.py", "model/process_forecast_r7.py"} <= names
    assert not any(name.startswith(("tests/", "outputs/", "docs/", "legacy")) or "test.jsonl" in name for name in names)
    assert code["source_tree_sha256"] == protocol.digest(code["files"])
    assert all(name.endswith((".py", ".txt", ".toml")) for name in names)


def test_actual_fullforward_newloss_profile_records_branch_ownership():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        batch = make_batch(1)
        batch["sample_id"] = ["fake-first-train-case"]
        context = make_context()
        measurements, ownership = {}, {}
        for arm in protocol.ARM_NAMES:
            model = make_model().train()
            initial = profile.state_hash(model.state_dict())
            measurements[arm] = profile.measure_cost(model, batch, context, protocol.WEIGHTS[arm])
            ownership[arm] = profile.gradient_profile(model, batch, context, protocol.WEIGHTS[arm])
            assert profile.state_hash(model.state_dict()) == initial
            assert all(parameter.grad is None for parameter in model.parameters())
        assert len({item["parameters"] for item in measurements.values()}) == 1
        assert all(item["forward_backward_flops"] > item["forward_flops"] > 0 for item in measurements.values())
        assert ownership["aux_off"]["components"]["input"]["loss"] == 0
        assert ownership["input_aux"]["components"]["input"]["groups"]["readout"]["norm"] > 0
        assert ownership["future_draft_aux"]["components"]["future"]["groups"]["readout"]["norm"] > 0
        assert ownership["future_draft_aux"]["components"]["draft"]["groups"]["readout"]["norm"] == 0
        assert ownership["future_draft_aux"]["components"]["draft"]["groups"]["forecast_history"]["norm"] > 0
        assert ownership["future_draft_aux"]["components"]["draft"]["groups"]["forecast_drafts"]["norm"] > 0
    finally:
        torch.set_num_threads(previous)


def test_profile_flops_include_new_loss_not_just_forecast():
    model = torch.nn.Linear(3, 3)
    # Controlled full-forward loss adds a real matrix multiply observable by FLOP counter.
    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = model
        def forward(self, batch, reasoning_steps):
            return SimpleNamespace(forecast=self.linear(batch["coarse_history"]))
    called = []
    def loss_fn(batch, out, **kwargs):
        called.append(kwargs)
        value = out.forecast
        if kwargs["draft_diagnostic_weight"]:
            value = value @ torch.eye(3)
        return SimpleNamespace(total=value.square().mean())
    batch = {"coarse_history": torch.ones(2, 3)}
    measured = profile.measure_cost(ProbeModel(), batch, None, protocol.WEIGHTS["future_draft_aux"], loss_fn=loss_fn)
    plain = profile.measure_cost(ProbeModel(), batch, None, protocol.WEIGHTS["aux_off"], loss_fn=loss_fn)
    assert measured["forward_flops"] > plain["forward_flops"]
    assert all(call["process_weight"] == 0 for call in called)
    assert all(torch.is_grad_enabled() for _ in called)


def _csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _complete_fixture(frozen):
    output = Path(frozen["output"])
    channels = frozen["data"]["channels"]
    for job in protocol.planned_jobs():
        arm, seed = job["arm"], job["seed"]
        contract = protocol.expected_training_contract(frozen, arm)
        entry = {"status": "success", "job": job, "scientific_claim": False, "limitations": ["tmp engineering fixture"],
                 "test_read": False, "protocol_sha256": frozen["protocol_sha256"],
                 "model_code_sha256": frozen["code"]["model_code_sha256"], "data_identity": frozen["data"]["data_identity"],
                 "process_supervision": contract, "peak_allocated_bytes": 1000, "peak_reserved_bytes": 2000,
                 "elapsed_seconds": 1.0}
        folder = worker.train_output_dir(output, seed, arm)
        checkpoint = folder / "update_0000400.pt"
        if job["phase"] == "train":
            folder.mkdir(parents=True)
            checkpoint.write_bytes(b"tmp fake checkpoint, never loaded as real model")
            report = {"updates_this_run": 400, "selected_update": 400, "early_stopped": False,
                      "selection_split": "val", "contract": {"process_supervision": contract, "data_identity": frozen["data"]["data_identity"]},
                      "validations": [{"update": value} for value in (100, 200, 300, 400)],
                      "losses": [{"update": value} for value in range(1, 401)]}
            protocol.write_json(folder / "training_report.json", report)
            entry.update(checkpoint=str(checkpoint), checkpoint_sha256=identity.sha256_file(checkpoint),
                         training_report=str(folder / "training_report.json"),
                         training_report_sha256=identity.sha256_file(folder / "training_report.json"),
                         selected_update=400, updates_run=400, early_stopped=False, seconds_per_update=.0025,
                         initial_state_sha256="d" * 64, torch_version=str(torch.__version__),
                         shared_initial_state={"provided": True, "applied_count": 1, "ignored_count": 0})
        else:
            lead = job["lead"]
            directory = output / f"seed{seed}" / "evaluation" / arm / f"lead_{lead:03d}h"
            directory.mkdir(parents=True)
            error = {"aux_off": 2., "input_aux": 1., "future_draft_aux": 3.}[arm]
            rows = [{"lead_hours": lead, "variable": name, "unit": "physical-unit", "rmse": error,
                     "n_initializations": 1} for name in channels]
            _csv(directory / "rmse.csv", list(rows[0]), rows)
            rows = [{"lead_hours": lead, "variable": name, "pooled_acc": .5,
                     "status": "defined", "n_initializations": 1} for name in channels]
            _csv(directory / "acc.csv", list(rows[0]), rows)
            rows = [{"lead_hours": lead, "variable": name, "unit": "physical-unit", "rmse_forecast": error,
                     "rmse_climatology": 4., "mse_skill": 1 - error ** 2 / 16, "n_initializations": 1} for name in channels]
            _csv(directory / "climatology_skill.csv", list(rows[0]), rows)
            cases = frozen["data"]["evaluation_cases"][str(lead)]["cases"]
            provenance = {"initializations": [{"init_time": cases[0][0], "valid_times": cases[0][1], "mse": [[error ** 2] * 17]}],
                          "channels": channels, "units": frozen["data"]["units"], "lead_hours": [lead], "n_evaluated": 1,
                          "evaluation_manifest_sha256": frozen["sources"]["val_manifest_sha256"], "split": "val",
                          "checkpoint_sha256": identity.sha256_file(checkpoint), "training_identity": frozen["data"]["data_identity"],
                          "process_scale_sidecar_identity": frozen["sidecar"]["identity"], "training_protocol_sha256": frozen["protocol_sha256"]}
            protocol.write_json(directory / "provenance.json", provenance)
            entry.update(arm=arm, lead_hours=lead, split="val", channels=channels, units=frozen["data"]["units"],
                         evaluation_dir=str(directory), rmse_csv=str(directory / "rmse.csv"),
                         skill_csv=str(directory / "climatology_skill.csv"), acc_csv=str(directory / "acc.csv"),
                         n_evaluated=1, n_available_windows=1, baseline={"allocated_bytes": 0, "reserved_bytes": 0},
                         checkpoint=str(checkpoint), checkpoint_sha256=identity.sha256_file(checkpoint),
                         artifact_sha256={name: identity.sha256_file(directory / name) for name in
                                          ("provenance.json", "rmse.csv", "acc.csv", "climatology_skill.csv")})
        protocol.write_json(worker.worker_result_path(output, job), entry)
        (output / "workers" / (protocol.job_key(job) + ".log")).write_text("fixture log", encoding="utf-8")
        protocol.write_json(output / "workers" / (protocol.job_key(job) + ".timing.json"), {"job": job})
    for name in ("code.zip", "code_commit.txt", "code_status.txt", "cpu_profile.json"):
        (output / name).write_text("tmp engineering fixture", encoding="utf-8")
    execution = {"status": "success", "jobs_completed": protocol.planned_jobs(), "gpu_phase_elapsed_seconds": 45.0}
    protocol.write_json(output / "execution_attempt.json", execution)
    return execution


def test_complete510_cells_all85pairs_and_negative_results_tables(frozen):
    execution = _complete_fixture(frozen)
    outcome = results.finalize(frozen["output"], frozen, execution)
    assert not outcome["paused"]
    assert not outcome["advance_next_node"]
    comparison = protocol.read_json(Path(frozen["output"]) / "paired_comparison.json")
    assert len(comparison["table"]) == 255
    assert all(len(pair["cells"]) == 85 for pair in comparison["pairs"].values())
    assert comparison["pairs"]["input_aux - aux_off"]["totals"]["improved"] == 85
    assert comparison["pairs"]["future_draft_aux - aux_off"]["totals"]["worsened"] == 85
    assert len(results._read_csv(Path(frozen["output"]) / "rmse_table.csv")) == 510
    assert len(results._read_csv(Path(frozen["output"]) / "acc_table.csv")) == 510
    assert len(results._read_csv(Path(frozen["output"]) / "allocator_table.csv")) == 36


@pytest.mark.parametrize("failure", ["missing", "skip", "updates", "seed", "baseline", "cases", "units", "rmse", "acc", "protocol"])
def test_incomplete_or_unpaired_full_set_never_partial_merges(frozen, failure):
    _complete_fixture(frozen)
    job = protocol.planned_jobs()[0 if failure == "updates" else -1]
    path = worker.worker_result_path(frozen["output"], job)
    entry = protocol.read_json(path)
    if failure == "missing":
        path.unlink()
    else:
        if failure == "skip":
            entry["status"] = "skipped"
        elif failure == "updates":
            entry["updates_run"] = 399
        elif failure == "seed":
            entry["job"]["seed"] = 43
        elif failure == "baseline":
            entry["baseline"]["allocated_bytes"] = 1
        elif failure == "units":
            entry["units"][0] = "normalized"
        elif failure == "protocol":
            entry["protocol_sha256"] = "0" * 64
        else:
            name = {"cases": "provenance.json", "rmse": "rmse.csv", "acc": "acc.csv"}[failure]
            artifact = Path(entry["evaluation_dir"]) / name
            if failure == "cases":
                value = protocol.read_json(artifact)
                value["initializations"][0]["init_time"] = "different-case-same-count"
                artifact.write_text(json.dumps(value), encoding="utf-8")
            else:
                rows = results._read_csv(artifact)[:-1]
                artifact.unlink()
                _csv(artifact, list(rows[0]), rows)
            entry["artifact_sha256"][name] = identity.sha256_file(artifact)
        path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        results.validate_full_set(frozen["output"], frozen)
    assert not (Path(frozen["output"]) / "merged_result.json").exists()


def test_unresolved_pauses_without_new_significance_threshold():
    pairs = {"input_aux - aux_off": {"cells": {"6h|t2m": {"outcome": "unresolved"}},
                                  "totals": {"improved": 0, "worsened": 0, "unresolved": 1}}}
    outcome = results.descriptive_outcome(pairs)
    assert outcome["paused"] and not outcome["advance_next_node"]
    assert outcome["scientific_claim"] is False and outcome["scientific_gate_evaluated"] is False
