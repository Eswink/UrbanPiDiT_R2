"""Stdlib bounded orchestration: fresh10 CUDA jobs; soft1800/hard3600 whole earliest CPU clock."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from r7_k3_reference_cases import validate_evaluation
from r7_k3_reference_identity import (
    build_protocol, companion_zip_bytes, extract_archive, read_protocol, verify_extraction, verify_inputs, verify_runner,
)
from r7_k3_reference_support import (
    GPU_UUID, LIMITATIONS, check_deadline, digest, job_key, local_path, output_path, planned_jobs,
    preserve_error, read_json, require, sha256_file, validate_clock, write_json,
)


def gpu_snapshot(uuid, *, query=subprocess.check_output):
    """Original M3-driver NVIDIA read-only contract; never signal/query-clear neighbors."""
    cards = query(["nvidia-smi", "--query-gpu=uuid,index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                  text=True, timeout=5)
    rows = []
    for line in cards.splitlines():
        fields = [x.strip() for x in line.split(",")]
        require(len(fields) == 4, "malformed GPU UUID/headroom inventory")
        rows.append(dict(uuid=fields[0], index=int(fields[1]), free_mib=int(fields[2]), total_mib=int(fields[3])))
    matches = [r for r in rows if r["uuid"] == uuid]
    require(len(matches) == 1, "bound physical GPU UUID missing/duplicated; no fallback")
    apps = query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                 text=True, timeout=5)
    neighbors = []
    for line in apps.splitlines():
        fields = [x.strip() for x in line.split(",")]
        require(len(fields) == 3, "malformed read-only GPU process inventory")
        if fields[0] == uuid:
            neighbors.append({"pid": int(fields[1]), "used_mib": fields[2]})
    return {**matches[0], "neighbors": neighbors, "all_cards": rows, "read_only": True, "queried_unix_seconds": time.time()}


def verify_headroom(snapshot, gpu, observed_peak_mib=0):
    require(math.isfinite(observed_peak_mib) and observed_peak_mib >= 0, "finite nonnegative owned peak required")
    required = max(gpu["estimated_peak_mib"], math.ceil(observed_peak_mib)) + gpu["headroom_margin_mib"]
    require(snapshot["uuid"] == gpu["uuid"] == GPU_UUID and snapshot["free_mib"] >= required,
            f"shared GPU UUID/headroom refused: need {required}MiB; no neighbor interference/fallback")
    return {**snapshot, "required_free_mib": required, "observed_owned_peak_mib": observed_peak_mib}


def worker_environment():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUID, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}


def reap_owned(process, deadline, *, clock=time.perf_counter):
    """Only direct Popen handle; no os.kill/raw PID/process group/neighbor management."""
    if process.poll() is not None:
        process.wait(timeout=0)
        return "already-exited-owned-worker"
    process.terminate()
    try:
        process.wait(timeout=max(0.0, min(3.0, deadline - clock())))
        return "terminated-and-reaped-owned-worker"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.0, deadline - clock()))
        return "killed-and-reaped-owned-worker"


def archive_companion(protocol):
    output = Path(protocol["output"])
    root = output / "companion_code"
    root.mkdir(exist_ok=False)
    content = companion_zip_bytes(protocol["code"]["root"], protocol["code"]["files"])
    with (output / "companion_code.zip").open("xb") as stream:
        stream.write(content)
    require(sha256_file(output / "companion_code.zip") == protocol["code"]["companion_zip_sha256"], "companion archive digest differs")
    for name, expected in protocol["code"]["files"].items():
        source = Path(protocol["code"]["root"]) / name
        require(sha256_file(source) == expected, "companion freeze drift before archival")
        with (root / name).open("xb") as stream:
            stream.write(source.read_bytes())
    write_json(output / "source_identity.json", protocol["code"])


def prepare(repo, output, started, *, clock=time.perf_counter):
    """Byte/manifest/metadata preparation only: never import torch, weather dataset or load checkpoints."""
    output = output_path(output, repo, fresh=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    write_json(output / "prepare_started.json", {"started_perf_counter": started, "started_unix_seconds": time.time(),
               "hard_cap_seconds": 3600, "planned_seconds": 1800, "scientific_claim": False, "limitations": LIMITATIONS,
               "clock_scope": "CLI entry before companion imports/preparation/archive/checkpins; whole round includes gaps/aggregation/owned cleanup; anchor and boot are frozen in protocol digest"})
    deadline = started + 3600
    check = lambda: check_deadline(deadline, clock=clock)
    error = None
    try:
        check()
        protocol = build_protocol(repo, output, started=started, check=check)
        write_json(output / "protocol.json", protocol)  # Freeze BEFORE extraction, any project import or CUDA spawn.
        archive_companion(protocol)
        check()
        extract_archive(protocol["archive"]["path"], protocol["archive"]["files"], output / "archived_code", check=check)
        verify_inputs(protocol, check=check)
        (output / "workers").mkdir(exist_ok=False)
        (output / "cases").mkdir(exist_ok=False)
        check()
        return protocol
    except BaseException as exc:
        error = exc
        raise
    finally:
        elapsed = clock() - started
        receipt = {"status": "failed" if error else "prepared-not-run", "scientific_claim": False, "limitations": LIMITATIONS,
                   "test_read": False, "weather_samples_read": 0, "checkpoint_deserializations": 0, "cuda_jobs": 0,
                   "whole_elapsed_seconds": elapsed, "soft_overrun_seconds": max(0.0, elapsed - 1800),
                   "failure_reason": None if error is None else f"{type(error).__name__}: {error}"}
        try:
            if error is not None:
                write_json(output / "failed_attempt_seal.json", {"status": "failed", "finalized": False,
                           "scientific_claim": False, "limitations": LIMITATIONS, "phase": "prepare",
                           "failure_reason": receipt["failure_reason"], "ended_perf_counter": clock()})
            write_json(output / "prepare_attempt.json", receipt)
            if error is None:
                check()  # Include preparation receipt publication, not only input/zip work.
        except BaseException as exc:
            if error is None:
                try:
                    write_json(output / "failed_attempt_seal.json", {"status": "failed", "finalized": False,
                               "scientific_claim": False, "limitations": LIMITATIONS, "phase": "prepare-publication",
                               "failure_reason": f"{type(exc).__name__}: {exc}", "ended_perf_counter": clock()})
                except BaseException as additional:
                    exc.add_note(f"prepare failure seal publication failed: {additional}")
            preserve_error(error, exc, "CPU preparation receipt publication failed")


def worker_command(protocol, job, deadline):
    return [sys.executable, "-B", str(Path(protocol["output"]) / "companion_code/r7_k3_reference.py"),
            "--mode", "archive_worker", "--protocol", str(Path(protocol["output"]) / "protocol.json"),
            "--seed", str(job["seed"]), "--lead", str(job["lead"]), "--deadline", repr(deadline)]


def verify_worker(receipt, protocol, job):
    output = Path(protocol["output"])
    result = output / "cases" / (job_key(job) + ".json")
    require(receipt.get("status") == "success" and receipt.get("job") == job
            and receipt.get("protocol_sha256") == protocol["protocol_sha256"] and receipt.get("result") == str(result),
            "exact successful worker job/protocol/result required; failure/skip/partial never accepted")
    require(receipt.get("result_sha256") == sha256_file(result), "worker result digest changed")
    require(receipt.get("source_sha256") == protocol["code"]["source_sha256"]
            and receipt.get("archive_sha256") == protocol["archive"]["sha256"]
            and receipt.get("companion_zip_sha256") == protocol["code"]["companion_zip_sha256"]
            and receipt.get("model_code_sha256") == protocol["archive"]["model_code_sha256"]
            and receipt.get("data_identity") == protocol["data"]["data_identity"]
            and receipt.get("checkpoint_sha256") == protocol["parents"][str(job["seed"])]["checkpoint_sha256"],
            "worker exact source/data/parent identity required")
    allocator = receipt["allocator"]
    require(allocator["pre_init"] == allocator["initialized"] == {"allocated_bytes": 0, "reserved_bytes": 0}
            and allocator["pre_init_is_initialized"] is False and allocator["device"] == "cuda:0", "genuine fresh measured0/0 allocator required")
    require(isinstance(receipt["peak_reserved_bytes"], int) and receipt["peak_reserved_bytes"] >= receipt["peak_allocated_bytes"] >= 0,
            "valid measured own memory peaks required")
    report = read_json(result)
    parent = protocol["parents"][str(job["seed"])]
    require(report["checkpoint_sha256"] == parent["checkpoint_sha256"] and report["signature"] == parent["signature"]
            and report["calendar"] == protocol["calendar"] and report["extra_model_forward_for_regions_or_baselines"] == 0,
            "original K3 strict loader/calendar/one-trajectory qualification differs")
    validate_evaluation(report, protocol, job)
    return report


def run_job(protocol, job, hard_deadline, observed_peak_mib, *, clock=time.perf_counter,
            snapshot_fn=gpu_snapshot, popen_factory=subprocess.Popen):
    output = Path(protocol["output"])
    key = job_key(job)
    timing = {"status": "failed", "job": job, "scientific_claim": False, "limitations": LIMITATIONS,
              "started_perf_counter": clock(), "ownership": "direct Popen handle only"}
    process, error = None, None
    try:
        check_deadline(hard_deadline - 10, clock=clock)
        timing["headroom"] = verify_headroom(snapshot_fn(GPU_UUID), protocol["gpu"], observed_peak_mib)
        check_deadline(hard_deadline - 10, clock=clock)
        deadline = hard_deadline - 10
        command = worker_command(protocol, job, deadline)
        timing.update(command=command, spawned_perf_counter=clock(), execution_deadline=deadline)
        with (output / "workers" / (key + ".log")).open("x", encoding="utf-8") as log:
            process = popen_factory(command, cwd=output / "archived_code", env=worker_environment(), stdout=log, stderr=subprocess.STDOUT)
            timing["owned_pid"] = process.pid
            try:
                code = process.wait(timeout=max(0.0, deadline - clock()))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("whole-round hard cap reached during owned worker; no retry") from exc
            timing.update(returncode=code, reaped_perf_counter=clock())
            require(code == 0, f"K3 {key} exited {code}; whole round stops without retry")
            check_deadline(deadline, clock=clock)
        receipt = read_json(output / "workers" / (key + ".json"))
        verify_worker(receipt, protocol, job)
        timing["status"] = "success"
        return receipt
    except BaseException as exc:
        error = exc
        timing.update(failure_reason=f"{type(exc).__name__}: {exc}", budget_limited=isinstance(exc, TimeoutError))
        raise
    finally:
        cleanup_error = None
        if process is not None:
            try:
                timing["cleanup"] = reap_owned(process, hard_deadline, clock=clock)
                timing["last_owned_reap_perf_counter"] = clock()
            except BaseException as exc:
                cleanup_error = exc
                timing.update(status="failed-unreaped", cleanup_failure=f"{type(exc).__name__}: {exc}")
        timing["ended_perf_counter"] = clock()
        try:
            write_json(output / "workers" / (key + ".timing.json"), timing)
        except BaseException as exc:
            preserve_error(error or cleanup_error, exc, "owned-job timing receipt publication failed")
        if cleanup_error is not None:
            preserve_error(error, cleanup_error, "owned worker cleanup failed/unreaped")


def finalize(protocol, completed, *, check=lambda: None):
    require(completed == planned_jobs(), "all exact10 jobs required; no filtered successful subset")
    rows, files = [], {}
    for job in completed:
        check()
        path = Path(protocol["output"]) / "workers" / (job_key(job) + ".json")
        receipt = read_json(path)
        report = verify_worker(receipt, protocol, job)
        rows.extend(report["rows"])
        files[str(path)] = sha256_file(path)
        files[receipt["result"]] = receipt["result_sha256"]
    require(len(rows) == 2 * 5 * 17 * 3 * 3, "complete1530 region/forecast rows required")
    output = Path(protocol["output"])
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "complete-not-accepted", "acceptance_authority": "successful attempt.json plus absence of failed_attempt_seal.json",
              "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
              "protocol_sha256": protocol["protocol_sha256"], "source_sha256": protocol["code"]["source_sha256"],
              "archive_sha256": protocol["archive"]["sha256"], "model_code_sha256": protocol["archive"]["model_code_sha256"],
              "jobs_completed": completed, "training_updates": 0, "reference_rmse_cells": 170, "all_region_forecast_rows": len(rows),
              "artifact_sha256": {**files, str(output / "metrics.csv"): sha256_file(output / "metrics.csv")},
              "calendar": protocol["calendar"], "reproducibility": protocol["reproducibility"]}
    check()
    write_json(output / "reference_result.json", result)
    check()
    return result


def run_all(protocol, started, *, clock=time.perf_counter, snapshot_fn=gpu_snapshot, popen_factory=subprocess.Popen):
    output = Path(protocol["output"])
    require(not any((output / name).exists() for name in ("run_started.json", "attempt.json", "failed_attempt_seal.json")),
            "already attempted output; no retry/resume")
    deadline = started + protocol["hard_cap_seconds"]
    check = lambda: check_deadline(deadline, clock=clock)
    attempt = {"status": "failed", "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
               "protocol_sha256": protocol["protocol_sha256"], "started_perf_counter": started,
               "planned_seconds": 1800, "hard_cap_seconds": 3600, "jobs_planned": planned_jobs(), "jobs_completed": [],
               "budget_limited": False, "finalized": False, "gpu_phase_elapsed_seconds": 0.0}
    write_json(output / "run_started.json", {"started_perf_counter": started, "entry_perf_counter": clock(), "scientific_claim": False})
    error, gpu_started, gpu_ended = None, None, None
    try:
        validate_clock(protocol["clock"], started)
        check()
        verify_inputs(protocol, check=check)
        verify_extraction(output / "archived_code", protocol["archive"]["files"], check=check)
        verify_runner(protocol["code"], check=check)
        from r7_k3_reference_worker import verify_companion
        verify_companion(protocol, output / "companion_code", check=check)
        gpu_started = clock()  # Includes read-only admission, all startup/gaps/eval/owned cleanup.
        attempt["startup_headroom"] = verify_headroom(snapshot_fn(GPU_UUID), protocol["gpu"])
        observed_peak = 0.0
        for job in planned_jobs():
            check()
            receipt = run_job(protocol, job, deadline, observed_peak, clock=clock, snapshot_fn=snapshot_fn, popen_factory=popen_factory)
            observed_peak = max(observed_peak, receipt["peak_reserved_bytes"] / 2**20)
            attempt["jobs_completed"].append(job)
        gpu_ended = clock()
        verify_inputs(protocol, check=check)
        verify_extraction(output / "archived_code", protocol["archive"]["files"], check=check)
        verify_companion(protocol, output / "companion_code", check=check)
        result = finalize(protocol, attempt["jobs_completed"], check=check)
        check()
        result_hash = sha256_file(output / "reference_result.json", check=check)
        verify_inputs(protocol, check=check)
        verify_extraction(output / "archived_code", protocol["archive"]["files"], check=check)
        verify_companion(protocol, output / "companion_code", check=check)
        check()
        attempt.update(status="success", finalized=True, result_sha256=result_hash)
        return result
    except BaseException as exc:
        error = exc
        attempt.update(status="failed", finalized=False, failure_reason=f"{type(exc).__name__}: {exc}", budget_limited=isinstance(exc, TimeoutError))
        try:
            verify_inputs(protocol, check=check)
            verify_extraction(output / "archived_code", protocol["archive"]["files"], check=check)
            attempt["failure_exit_identity_verification"] = "verified"
        except BaseException as additional:
            attempt["failure_exit_identity_verification"] = f"not-verified: {type(additional).__name__}: {additional}"
            error.add_note(attempt["failure_exit_identity_verification"])
        raise
    finally:
        ended = clock()
        elapsed = ended - started
        gpu_elapsed = 0.0 if gpu_started is None else (gpu_ended if gpu_ended is not None else ended) - gpu_started
        attempt.update(ended_perf_counter=ended, whole_elapsed_seconds=elapsed, whole_hours_charged=elapsed / 3600,
                       soft_overrun_seconds=max(0.0, elapsed - 1800), gpu_phase_elapsed_seconds=gpu_elapsed,
                       gpu_hours_charged=gpu_elapsed / 3600, partial=0 < len(attempt["jobs_completed"]) < 10,
                       cleanup_or_publication_diagnostics=list(getattr(error, "__notes__", [])),
                       billing_scope="whole earliest CPU clock includes prepare/imports/archive/checkpins/startup/gaps/evaluation/aggregation/cleanup; separate continuous GPU phase")
        try:
            if error is not None:
                write_json(output / "failed_attempt_seal.json", {"status": "failed", "finalized": False,
                           "scientific_claim": False, "limitations": LIMITATIONS, "protocol_sha256": protocol["protocol_sha256"],
                           "failure_reason": attempt["failure_reason"], "ended_perf_counter": clock(),
                           "authority": "terminal failure seal; never accepted/retried even if candidate results exist"})
            write_json(output / "attempt.json", attempt)
            if error is None:
                check()  # Receipt serialization/publication itself is inside the whole-round hard clock.
        except BaseException as exc:
            if error is None:
                try:
                    write_json(output / "failed_attempt_seal.json", {"status": "failed", "finalized": False,
                               "scientific_claim": False, "limitations": LIMITATIONS, "protocol_sha256": protocol["protocol_sha256"],
                               "failure_reason": f"{type(exc).__name__}: {exc}", "ended_perf_counter": clock(),
                               "authority": "post-publication failure invalidates premature candidate/attempt success"})
                except BaseException as additional:
                    exc.add_note(f"failure seal publication failed: {type(additional).__name__}: {additional}")
            preserve_error(error, exc, "whole-round failure/success receipt publication failed")
