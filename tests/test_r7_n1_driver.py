"""N1 driver guards use CPU engineering fixtures only, never archived outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    path = ROOT / "scripts" / "study_r7_72_frozen_z.py"
    spec = importlib.util.spec_from_file_location("r7_n1_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_costs_apply_intervention_before_both_flop_measurements(monkeypatch):
    driver = _driver()
    seen = []

    def measure(model, batch, *, reasoning_steps):
        frozen = hasattr(model, "_frozen_z_value")
        seen.append(frozen)
        if frozen:
            assert not model.solver_init.requires_grad
            assert all(not p.requires_grad for p in model.solver_cell.parameters())
            assert model.solver_gate.score.weight.requires_grad
        return 100, 300

    monkeypatch.setattr(driver, "count_flops", measure)
    specs = {str(seed): driver.make_spec(seed, (2, 3), 192, 2) for seed in driver.SEEDS}
    measured = driver.measured_arms({}, 1, specs)
    assert seen == [False, False, True]
    plain, frozen = measured[driver.ARM_NAMES[1]], measured[driver.FROZEN_Z_ARM]
    assert plain["parameters"] == frozen["parameters"]
    assert plain["trainable_parameters"] > frozen["trainable_parameters"]


def test_pairing_covers_original_weights_and_non_global_randomness():
    driver = _driver()
    spec = driver.make_spec(41, (2, 3), 192, 2)
    pairing = driver.pairing_for_seed(1, 41, spec)
    assert pairing["all_shared_pairs_identical"]
    frozen = pairing["frozen_z_installation"]
    assert frozen["all_original_parameters_bitwise_unchanged"]
    assert frozen["global_cpu_rng_unchanged"]
    assert frozen["same_shape"] == [1, 6, 192]


def test_deadline_has_a_failure_counterproof(monkeypatch):
    driver = _driver()
    monkeypatch.setattr(driver.time, "perf_counter", lambda: 10.0)
    driver.check_deadline(11.0)
    with pytest.raises(RuntimeError, match="deadline"):
        driver.check_deadline(10.0)


def test_gpu_exclusivity_refuses_other_process_without_stopping_it(monkeypatch):
    driver = _driver()
    monkeypatch.setattr(driver.os, "getpid", lambda: 123)
    monkeypatch.setattr(driver.subprocess, "check_output", lambda args, **kw:
                        "GPU-target\n" if "--query-gpu=uuid" in args
                        else "GPU-other, 900\nGPU-target, 123\n")
    driver.verify_gpu_exclusive("cuda:1")
    monkeypatch.setattr(driver.subprocess, "check_output", lambda args, **kw:
                        "GPU-target\n" if "--query-gpu=uuid" in args
                        else "GPU-target, 456\n")
    with pytest.raises(RuntimeError, match="occupied"):
        driver.verify_gpu_exclusive("cuda:1")
    with pytest.raises(ValueError, match="explicit"):
        driver.verify_gpu_exclusive("cuda")


def test_offline_connections_are_denied(monkeypatch):
    driver = _driver()
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    driver.deny_network()
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("example.invalid", 443))
    with pytest.raises(RuntimeError, match="offline"):
        socket.socket.connect(None, ("example.invalid", 443))


def _source_fixture(tmp_path):
    driver = _driver()
    manifests = tmp_path / "segment" / "store" / "manifests"
    manifests.mkdir(parents=True)
    source = manifests.parent.parent / "source.bin"
    source.write_bytes(b"engineering identity bytes, not weather data")
    digest = driver.sha256_file(source)
    (manifests / "source_preflight.json").write_text(json.dumps({
        "source_path": str(source), "fingerprint": {"sha256": digest, "bytes": source.stat().st_size}
    }), encoding="utf-8")
    (manifests.parent.parent / "source_receipt.json").write_text(json.dumps({
        "local_artifact": {"sha256": digest}, "synthetic_fallback": False
    }), encoding="utf-8")
    (manifests / "BUILD_COMPLETE.json").write_text(
        json.dumps({"schema_version": 1, "build_complete": True}), encoding="utf-8")
    for name in ("train.jsonl", "val.jsonl"):
        (manifests / name).write_text("{}\n", encoding="utf-8")
    return driver, manifests, source


def test_source_identity_is_separate_and_never_opens_test(tmp_path, monkeypatch):
    driver, manifests, source = _source_fixture(tmp_path)
    opened = []
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name != "test.jsonl"
        opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    identity = driver.source_identity(manifests)
    assert identity["source_sha256"] == driver.sha256_file(source)
    assert "test.jsonl" not in opened
    assert {"train.jsonl", "val.jsonl", "BUILD_COMPLETE.json"} <= set(opened)


@pytest.mark.parametrize("failure", ["source", "marker", "receipt"])
def test_source_identity_refuses_unpublished_or_mismatched_bytes(tmp_path, failure):
    driver, manifests, source = _source_fixture(tmp_path)
    if failure == "source":
        source.write_bytes(b"changed engineering fixture")
    elif failure == "marker":
        (manifests / "BUILD_COMPLETE.json").write_text(
            json.dumps({"schema_version": 1, "build_complete": False}), encoding="utf-8")
    else:
        (manifests.parent.parent / "source_receipt.json").write_text(json.dumps({
            "local_artifact": {"sha256": driver.sha256_file(source)}, "synthetic_fallback": True
        }), encoding="utf-8")
    with pytest.raises(ValueError):
        driver.source_identity(manifests)


def test_missing_user_authorization_refuses_before_any_reader_or_cuda(tmp_path, monkeypatch):
    driver = _driver()
    import data.r7_zarr_dataset as dataset_module
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({"status": "pending"}), encoding="utf-8")

    def refused(*args, **kwargs):
        raise AssertionError("must stop before dataset or CUDA access")

    monkeypatch.setattr(dataset_module, "ZarrAtmosWindowDataset", refused)
    monkeypatch.setattr(torch.cuda, "set_device", refused)
    with pytest.raises(ValueError, match="actual recorded user authorization"):
        driver.run(tmp_path / "manifests", tmp_path / "output", authorization, "cuda:1")
    assert not (tmp_path / "output").exists()


def test_complete_merge_cannot_hide_cap_overrun(tmp_path, monkeypatch):
    driver = _driver()
    merged = {"protocol": {"limitations": ["fixture only"]}, "budget": {}}
    monkeypatch.setattr(driver, "merge_seed_results", lambda *args, **kw: merged)
    with pytest.raises(RuntimeError, match="cap"):
        driver.finalize(tmp_path, driver.GPU_CAP_HOURS * 3600 + 1)
    assert not (tmp_path / "merged_result.json").exists()


def test_training_checks_deadline_before_protocol_reader_or_optimizer(monkeypatch, tmp_path):
    driver = _driver()
    monkeypatch.setattr(driver.time, "perf_counter", lambda: 20.0)
    with pytest.raises(RuntimeError, match="deadline"):
        driver.train_arm(None, None, tmp_path, tmp_path / "no-protocol.json",
                         name=driver.FROZEN_Z_ARM, kind="process", config={}, channels=1,
                         identity="fixture", seed=41, spec={}, shared={}, device="cuda:1",
                         deadline=19.0)


def test_table_writer_is_exclusive(tmp_path):
    driver = _driver()
    driver.write_json(tmp_path / "result.json", {"scientific_claim": False})
    with pytest.raises(FileExistsError):
        driver.write_json(tmp_path / "result.json", {"scientific_claim": True})
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["scientific_claim"] is False


def test_cost_table_adds_trainable_counts_only_for_declared_protocol(tmp_path):
    import csv
    driver = _driver()
    merged = {"protocol": {"arms": [{"name": "fixture", "parameters": 10,
                                     "trainable_parameters": 6, "forward_flops": 100,
                                     "forward_backward_flops": 300, "switches": {}}]},
              "training": {}, "evaluation": {}}
    driver.write_study_tables(merged, tmp_path)
    with (tmp_path / "arm_table.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["parameters"] == "10"
    assert rows[0]["trainable_parameters"] == "6"
