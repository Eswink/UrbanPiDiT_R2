"""Synthetic-only complement contract/clock/inventory/correction acceptance tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import zipfile

import pytest

import training.r7_v2_stats_complement as complement
from training.r7_v2_stats_sources import (
    CORRECTED, OLD_OBJECTIVE, corrected_source, digest, extract_corrected, read_json,
    sha256_file, source_metadata, verify_archive, verify_inventory, write_json,
)


def tiny_source(tmp_path):
    source = tmp_path / "sealed_failed_source"
    source.mkdir()
    jobs = [{"phase": "train" if i < 6 else "evaluate", "seed": 41 if i % 2 == 0 else 42,
             "arm": f"arm_{i}", "lead": None if i < 6 else 6, "reasoning_steps": 4} for i in range(36)]
    original = {"stage": "B", "jobs": jobs, "planned_seconds": 5400., "hard_cap_seconds": 10800.,
                "gpu": {"uuid": "GPU-fixture", "estimated_peak_mib": 2048, "headroom_margin_mib": 2048}}
    model = b"# synthetic archive model, never instantiated\n"
    validator = ("def verify():\n    for loss in report['losses']:\n" + OLD_OBJECTIVE).encode()
    contents = {"model/synthetic_model.py": model, CORRECTED: validator}
    files = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    md = hashlib.sha256(b"synthetic_model.py\0" + model).hexdigest()
    with zipfile.ZipFile(source / "code.zip", "x") as archive:
        for name, value in contents.items(): archive.writestr(name, value)
    original["code"] = {"files": files, "source_tree_sha256": digest(files),
                        "code_zip_sha256": sha256_file(source / "code.zip"), "model_code_sha256": md}
    original["protocol_sha256"] = digest(original)
    write_json(source / "protocol.json", original)
    execution = {"status": "failed", "finalized": False, "protocol_sha256": original["protocol_sha256"], "stage": "B",
                 "jobs_planned": jobs, "jobs_completed": jobs, "jobs_results": [], "headroom_checks": [],
                 "partial": False, "budget_limited": False, "owned_unreaped": False, "test_read": False, "scientific_claim": False,
                 "first_gpu_spawn_started_perf_counter": 10., "last_owned_gpu_reap_perf_counter": 110., "started_perf_counter": 0.,
                 "gpu_phase_elapsed_seconds": 100., "gpu_hours_charged": 100 / 3600, "billing_scope": "continuous",
                 "ended_perf_counter": 120., "whole_elapsed_seconds": 120., "soft_overrun_seconds": 0.,
                 "failure_reason": "ValueError: same-case metric mismatch: training objective"}
    (source / "workers").mkdir()
    for index, job in enumerate(jobs):
        lead = "" if job["lead"] is None else "_lead006h"
        name = f"{job['phase']}_seed{job['seed']}_{job['arm']}{lead}_k4.json"
        receipt_path = source / "workers" / name
        write_json(receipt_path, {"job": job, "status": "success", "protocol_sha256": original["protocol_sha256"],
                                 "scientific_claim": False, "test_read": False, "elapsed_seconds": 1., "peak_reserved_bytes": 0})
        snapshot = {"uuid": "GPU-fixture", "read_only": True, "required_free_mib": 4096,
                    "free_mib": 10000, "observed_owned_peak_mib": 0.}
        spawn = 10. + index * 2.
        reap = spawn + 1.
        if index == 35: reap = 110.
        write_json(source / "workers" / name.replace(".json", ".timing.json"), {
            "job": job, "status": "success", "returncode": 0, "protocol_sha256": original["protocol_sha256"],
            "headroom": snapshot, "cleanup": "already-exited", "scientific_claim": False, "test_read": False,
            "limitations": ["fixture"], "spawned_perf_counter": spawn, "last_owned_reap_perf_counter": reap,
            "ended_perf_counter": reap})
        execution["jobs_results"].append({"job": job, "result": str(receipt_path), "elapsed_seconds": 1., "peak_reserved_bytes": 0})
        execution["headroom_checks"].append({"job": job, "snapshot": snapshot})
    write_json(source / "execution_attempt.json", execution)
    write_json(source / "attempt.json", execution)
    pins = {item.relative_to(source).as_posix(): sha256_file(item) for item in source.rglob("*") if item.is_file()}
    pins_path = tmp_path / "source_pins.json"
    write_json(pins_path, {"source_attempt": str(source), "files_sha256": pins})
    return source, original, pins_path, pins


def test_prepare_freezes_whole_anchor_gap_budget_and_exact_correction_without_metrics(tmp_path):
    source, original, pins, files = tiny_source(tmp_path)
    anchor = 10.
    clock = lambda: anchor + 20.
    output = tmp_path / "independent_metadata"
    protocol = complement.prepare(pins, output, round_started_perf_counter=anchor, clock=clock)
    assert protocol["round_started_perf_counter"] == anchor
    assert protocol["monotonic_boot_id"] == complement.boot_id()
    assert protocol["planned_seconds"] == 600 and protocol["hard_cap_seconds"] == 1200
    assert protocol["source_status"] == "failed" and len(protocol["source_inventory"]) == 36
    assert protocol["source_files_digest"] == digest(files)
    assert verify_inventory(source, files) == len(files)
    assert complement.clock_fields(anchor, protocol, clock=clock)["whole_elapsed_seconds"] >= 20.
    assert not (source / "provenance.json").exists()
    with pytest.raises(FileExistsError): complement.prepare(pins, output)


@pytest.mark.parametrize("change", ["tamper", "missing", "extra"])
def test_exact_failed_source_inventory_refuses_any_changed_member(tmp_path, change):
    source, _, _, files = tiny_source(tmp_path)
    if change == "tamper": (source / "protocol.json").write_text("{}", encoding="utf-8")
    elif change == "missing": (source / "protocol.json").rename(source / "protocol.missing")
    else: (source / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError): verify_inventory(source, files)


@pytest.mark.parametrize("change", ["success", "partial", "cost", "worker", "cause"])
def test_source_qualification_does_not_upgrade_failed_or_silent_intersect(tmp_path, change):
    source, original, _, _ = tiny_source(tmp_path)
    attempt = read_json(source / "attempt.json")
    if change == "success": attempt["status"] = "success"
    elif change == "partial": attempt["jobs_completed"].pop()
    elif change == "cost": attempt["gpu_phase_elapsed_seconds"] = 99.
    elif change == "cause": attempt["failure_reason"] = "unrelated fault"
    else:
        worker = next((source / "workers").glob("*.json"))
        entry = read_json(worker); entry["status"] = "skipped"; worker.write_text(json.dumps(entry), encoding="utf-8")
    (source / "attempt.json").write_text(json.dumps(attempt), encoding="utf-8")
    with pytest.raises(ValueError): source_metadata(source, original["protocol_sha256"])


def test_only_exact_objective_validator_delta_is_allowed_model_archive_stays_identical(tmp_path):
    source, original, pins, _ = tiny_source(tmp_path)
    protocol = complement.prepare(pins, tmp_path / "complement")
    original_validator = verify_archive(source, original)
    correction = protocol["correction"]
    runtime = extract_corrected(source, original, tmp_path / "runtime", correction)
    assert sha256_file(runtime / CORRECTED) == correction["corrected_validator_sha256"]
    assert (runtime / "model/synthetic_model.py").read_bytes() == b"# synthetic archive model, never instantiated\n"
    assert corrected_source(original_validator) == (runtime / CORRECTED).read_bytes()
    with pytest.raises(ValueError): corrected_source(corrected_source(original_validator))
    bad = deepcopy(correction); bad["corrected_validator_sha256"] = "0" * 64
    with pytest.raises(ValueError): extract_corrected(source, original, tmp_path / "bad_runtime", bad)


@pytest.mark.parametrize("anchor", [0., 10., 1_000_000_000.])
def test_prepare_hard_timeout_seals_new_failure_without_touching_old(tmp_path, anchor):
    source, _, pins, files = tiny_source(tmp_path)
    output = tmp_path / "timed_prepare"
    clock = lambda: anchor + 1210.
    with pytest.raises(TimeoutError): complement.prepare(pins, output, round_started_perf_counter=anchor, clock=clock)
    attempt = read_json(output / "attempt.json")
    assert attempt["status"] == "failed" and attempt["finalized"] is False
    assert attempt["whole_elapsed_seconds"] >= 1210.
    assert attempt["gpu_hours_charged"] == 0.
    assert verify_inventory(source, files) == len(files)


def test_run_same_boot_and_no_retry_or_resurrection_are_required(tmp_path):
    _, _, pins, _ = tiny_source(tmp_path)
    output = tmp_path / "run_refusal"
    protocol = complement.prepare(pins, output)
    protocol["monotonic_boot_id"] = "other-boot"
    (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="same-boot"): complement.run(output / "protocol.json")


def test_complement_output_cannot_overlap_source_or_protected_data(tmp_path):
    source, _, _, _ = tiny_source(tmp_path)
    with pytest.raises(ValueError): complement.output_path(source, source)
    with pytest.raises(ValueError): complement.output_path(source / "child", source)
    with pytest.raises(ValueError): complement.output_path(tmp_path / "data/raw/new", source)


@pytest.mark.parametrize("live,timeout", [(False, False), (True, False), (True, True)])
def test_only_owned_cpu_handle_is_reaped_with_bounded_wait(live, timeout):
    import subprocess
    class Child:
        def __init__(self): self.killed = False; self.waits = []
        def poll(self): return None if live else 0
        def kill(self): self.killed = True
        def wait(self, *, timeout):
            self.waits.append(timeout)
            if live and timeout == 0: raise subprocess.TimeoutExpired("own", timeout)
            return 0
    child = Child()
    result = complement.reap_cpu_owned(child, 10., clock=lambda: 10. if timeout else 9.)
    assert child.killed is live
    assert child.waits == [0 if not live or timeout else 1.]
    assert result["owned_unreaped"] is timeout


def test_fresh_namespace_child_can_import_archive_training_without_current_models(tmp_path):
    import subprocess
    runtime = tmp_path / "runtime"
    (runtime / "training").mkdir(parents=True)
    (runtime / "training/__init__.py").write_text("", encoding="utf-8")
    (runtime / "training/archive_only.py").write_text("ORIGIN = 'original-archive'\n", encoding="utf-8")
    support = tmp_path / "support"
    (support / "training").mkdir(parents=True)
    helper = support / "training/helper.py"
    helper.write_text("from training.archive_only import ORIGIN\nassert ORIGIN == 'original-archive'\n", encoding="utf-8")
    script = ("import sys,types,importlib.util;sys.path.insert(0,sys.argv[1]);p=types.ModuleType('stats_support');"
              "p.__path__=[sys.argv[2]];sys.modules['stats_support']=p;"
              "s=importlib.util.spec_from_file_location('stats_support.helper',sys.argv[3]);"
              "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)")
    result = subprocess.run([__import__("sys").executable, "-B", "-c", script, str(runtime), str(support / "training"), str(helper)],
                            cwd=runtime, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key,bad", [("protocol_sha256", "0" * 64), ("stage", "C")])
def test_both_source_seals_must_bind_actual_canonical_b_protocol(tmp_path, key, bad):
    source, original, _, _ = tiny_source(tmp_path)
    for name in ("attempt.json", "execution_attempt.json"):
        seal = read_json(source / name)
        seal[key] = bad
        (source / name).write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError):
        source_metadata(source, original["protocol_sha256"])


@pytest.mark.parametrize("key", ["partial", "budget_limited", "owned_unreaped", "test_read", "scientific_claim"])
def test_execution_flags_cannot_contradict_failed_source_qualification(tmp_path, key):
    source, original, _, _ = tiny_source(tmp_path)
    seal = read_json(source / "execution_attempt.json")
    seal[key] = True
    (source / "execution_attempt.json").write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError):
        source_metadata(source, original["protocol_sha256"])


@pytest.mark.parametrize("bad", ["archive", "protocol-json", "boot"])
def test_run_claim_seals_prevalidation_failure_and_prevents_restore_resurrection(tmp_path, bad):
    source, _, pins, files = tiny_source(tmp_path)
    output = tmp_path / "claim_refusal"
    complement.prepare(pins, output)
    name = "complement_code.zip" if bad == "archive" else "protocol.json"
    saved = (output / name).read_bytes()
    if bad == "archive": (output / name).write_bytes(b"tampered archive")
    elif bad == "protocol-json": (output / name).write_bytes(b"{invalid json")
    else:
        protocol = read_json(output / name)
        protocol["monotonic_boot_id"] = "other-boot"
        (output / name).write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)): complement.run(output / "protocol.json")
    seal = read_json(output / "attempt.json")
    assert seal["status"] == "failed" and seal["finalized"] is False
    assert seal["owned_unreaped"] is False and seal["gpu_hours_charged"] == 0.
    failed_bytes = (output / "attempt.json").read_bytes()
    (output / name).write_bytes(saved)
    with pytest.raises(FileExistsError): complement.run(output / "protocol.json")
    assert (output / "attempt.json").read_bytes() == failed_bytes
    assert verify_inventory(source, files) == len(files)


@pytest.mark.parametrize("bad", ["pins-json", "boot"])
def test_prepare_exclusively_claims_and_seals_all_preflight_failures(tmp_path, bad):
    source, _, pins, files = tiny_source(tmp_path)
    output = tmp_path / "preflight_refusal"
    if bad == "pins-json": pins.write_text("{invalid json", encoding="utf-8")
    arguments = {"monotonic_boot_id": "other-boot"} if bad == "boot" else {}
    with pytest.raises(ValueError): complement.prepare(pins, output, **arguments)
    seal = read_json(output / "attempt.json")
    assert seal["status"] == "failed" and seal["finalized"] is False
    failed_bytes = (output / "attempt.json").read_bytes()
    with pytest.raises(FileExistsError): complement.prepare(pins, output)
    assert (output / "attempt.json").read_bytes() == failed_bytes
    assert verify_inventory(source, files) == len(files)


def fake_success_child(monkeypatch, output):
    """No aggregation/evaluation: parent lifecycle only, with an already-exited owned fake."""
    class Child:
        def poll(self): return 0
        def kill(self): raise AssertionError("exited fake must not be killed")
        def wait(self, *, timeout):
            assert timeout >= 0
            return 0
    def launch(*args, **kwargs):
        write_json(output / "provenance.json", {"candidate_selection": {"status": "fixture-only"}})
        write_json(output / "artifact_manifest.json", {"scientific_claim": False,
                   "files_sha256": {"worker.log": sha256_file(output / "worker.log")}})
        return Child()
    monkeypatch.setattr(complement.subprocess, "Popen", launch)


@pytest.mark.parametrize("late_work", ["provenance-hash", "manifest-hash", "cleanup"])
def test_final_hash_and_owned_cleanup_crossing_hard_limit_never_seals_success(tmp_path, monkeypatch, late_work):
    source, _, pins, files = tiny_source(tmp_path)
    output = tmp_path / "late_final_work"
    protocol = complement.prepare(pins, output)
    anchor = protocol["round_started_perf_counter"]
    elapsed = [20.]
    monkeypatch.setattr(complement.time, "perf_counter", lambda: anchor + elapsed[0])
    fake_success_child(monkeypatch, output)
    if late_work.endswith("-hash"):
        real_hash = complement.sha256_file
        def delayed_hash(value, **kwargs):
            result = real_hash(value, **kwargs)
            delayed_name = "provenance.json" if late_work == "provenance-hash" else "artifact_manifest.json"
            if Path(value) == output / delayed_name:
                elapsed[0] = 1201.
            return result
        monkeypatch.setattr(complement, "sha256_file", delayed_hash)
    else:
        real_reap = complement.reap_cpu_owned
        def delayed_reap(*args, **kwargs):
            result = real_reap(*args, **kwargs)
            elapsed[0] = 1201.
            return result
        monkeypatch.setattr(complement, "reap_cpu_owned", delayed_reap)
    with pytest.raises(TimeoutError): complement.run(output / "protocol.json")
    seal = read_json(output / "attempt.json")
    assert seal["status"] == "failed" and seal["finalized"] is False
    assert seal["coverage_complete"] is False and seal["budget_limited"] is True
    assert seal["whole_elapsed_seconds"] >= 1201.
    assert seal["owned_unreaped"] is False
    failed_bytes = (output / "attempt.json").read_bytes()
    with pytest.raises(FileExistsError): complement.run(output / "protocol.json")
    assert (output / "attempt.json").read_bytes() == failed_bytes
    assert verify_inventory(source, files) == len(files)


def test_successful_parent_seal_has_explicit_complete_false_safety_flags(tmp_path, monkeypatch):
    source, _, pins, files = tiny_source(tmp_path)
    output = tmp_path / "parent_seal_only"
    complement.prepare(pins, output)
    fake_success_child(monkeypatch, output)
    result = complement.run(output / "protocol.json")
    assert result == read_json(output / "attempt.json")
    assert result["finalized"] is True and result["coverage_complete"] is True
    assert result["status"] == "aggregation-complete" and result["failure_reason"] is None
    assert all(result[key] is False for key in ("budget_limited", "partial", "owned_unreaped", "test_read", "scientific_claim"))
    assert result["whole_elapsed_seconds"] < result["hard_cap_seconds"] - result["cleanup_reserve_seconds"]
    assert verify_inventory(source, files) == len(files)
    assert result["worker_log_sha256"] == sha256_file(output / "worker.log")
    assert read_json(output / "artifact_manifest.json")["files_sha256"]["worker.log"] == result["worker_log_sha256"]


def test_worker_log_changed_after_owned_reap_cannot_finalize(tmp_path, monkeypatch):
    _, _, pins, _ = tiny_source(tmp_path)
    output = tmp_path / "worker_log_after_reap"
    complement.prepare(pins, output)
    fake_success_child(monkeypatch, output)
    original = complement.reap_cpu_owned
    def poison_after_reap(*args, **kwargs):
        result = original(*args, **kwargs)
        with (output / "worker.log").open("a", encoding="utf-8") as stream:
            stream.write("unregistered final worker bytes")
        return result
    monkeypatch.setattr(complement, "reap_cpu_owned", poison_after_reap)
    with pytest.raises(ValueError, match="worker log"):
        complement.run(output / "protocol.json")
    seal = read_json(output / "attempt.json")
    assert seal["status"] == "failed" and seal["finalized"] is False
    assert seal["owned_unreaped"] is False
    before = (output / "attempt.json").read_bytes()
    with pytest.raises(FileExistsError): complement.run(output / "protocol.json")
    assert (output / "attempt.json").read_bytes() == before


def test_final_owned_serialization_crossing_limit_is_resealed_failed_not_success(tmp_path, monkeypatch):
    _, _, pins, _ = tiny_source(tmp_path)
    output = tmp_path / "late_serialization"
    protocol = complement.prepare(pins, output)
    anchor, elapsed = protocol["round_started_perf_counter"], [10.]
    monkeypatch.setattr(complement.time, "perf_counter", lambda: anchor + elapsed[0])
    fake_success_child(monkeypatch, output)
    real_seal = complement.seal_owned
    def delayed_seal(stream, payload, **kwargs):
        real_seal(stream, payload, **kwargs)
        if payload["finalized"]: elapsed[0] = 1201.
    monkeypatch.setattr(complement, "seal_owned", delayed_seal)
    with pytest.raises(TimeoutError): complement.run(output / "protocol.json")
    result = read_json(output / "attempt.json")
    assert result["status"] == "failed" and result["finalized"] is False
    assert result["budget_limited"] is True and result["whole_elapsed_seconds"] >= 1201.
    with pytest.raises(FileExistsError): complement.run(output / "protocol.json")


def test_bounded_json_archive_and_inventory_checks_refuse_unbounded_transport(tmp_path, monkeypatch):
    import training.r7_v2_stats_sources as sources
    source, original, _, files = tiny_source(tmp_path)
    monkeypatch.setattr(sources, "MAX_JSON_BYTES", 8)
    with pytest.raises(ValueError, match="bounded"): read_json(source / "protocol.json")
    monkeypatch.setattr(sources, "MAX_SOURCE_BYTES", 8)
    with pytest.raises(ValueError, match="byte cap"): verify_archive(source, original)
    calls = []
    def exhausted():
        calls.append(True)
        if len(calls) >= 3: raise TimeoutError("fixture deadline")
    with pytest.raises(TimeoutError): verify_inventory(source, files, check=exhausted)
    assert len(calls) == 3


def test_negative_final_publication_marker_cannot_qualify_source(tmp_path):
    source, original, _, _ = tiny_source(tmp_path)
    (source / "publication_failure").write_text("fixture failed publication", encoding="ascii")
    with pytest.raises(ValueError, match="publication"): source_metadata(source, original["protocol_sha256"])


def test_prepare_claim_must_not_create_even_empty_directory_within_old_sealed_source(tmp_path):
    source, _, pins, files = tiny_source(tmp_path)
    output = source / "new_metadata"
    with pytest.raises(ValueError, match="sealed source"): complement.prepare(pins, output)
    assert not output.exists()
    assert verify_inventory(source, files) == len(files)
