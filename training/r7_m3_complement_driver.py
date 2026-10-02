"""Single-attempt zero-training M3 complement with archive isolation and owned cleanup."""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .r7_m3_complement_protocol import (
    CLEANUP_RESERVE_SECONDS, HARD_CAP_SECONDS, LIMITATIONS, PLANNED_SECONDS, ROOT,
    archive_driver_code, build_protocol, evaluation_dir, extract_original_archive, job_key,
    local_path, monotonic_boot_id, pin_original_files, planned_jobs, preserve_artifact_error, qualify_original,
    read_json, sha256_file, verify_driver_code, verify_original_archive, verify_original_files,
    verify_protocol, write_json,
)


class BudgetLimited(TimeoutError):
    pass


def gpu_snapshot(uuid):
    from .r7_m3_driver import gpu_snapshot as original_snapshot
    return original_snapshot(uuid)


def verify_headroom(snapshot, gpu, observed_peak_mib=0):
    from .r7_m3_driver import verify_headroom as original_headroom
    return original_headroom(snapshot, gpu, observed_peak_mib)


def reap_owned(process, hard_deadline, *, clock=time.perf_counter):
    from .r7_m3_driver import reap_owned as original_reap
    return original_reap(process, hard_deadline, clock=clock)


def check_budget(hard_deadline, *, clock=time.perf_counter, reserve=CLEANUP_RESERVE_SECONDS):
    if clock() >= hard_deadline - reserve:
        raise BudgetLimited("complement whole-round hard cap exhausted; cleanup reserve retained")


def worker_environment(uuid):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "OMP_NUM_THREADS": "4",
                   "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    environment.pop("PYTHONPATH", None)
    return environment


def worker_command(protocol_path, deadline, *, job=None, receipt=None):
    command = [str(ROOT / ".venv" / "bin" / "python"), "-I", "-B",
               str(ROOT / "scripts" / "r7_m3_complement_worker.py"),
               "--protocol", str(protocol_path), "--deadline", repr(deadline),
               "--mode", "evaluate" if job is not None else "verify"]
    if job is not None:
        command.extend(["--seed", str(job["seed"]), "--arm", job["arm"], "--lead", str(job["lead"])])
    else:
        command.extend(["--receipt", str(receipt)])
    return command


def _owned_call(protocol, command, key, hard_deadline, execution, *, job, clock, popen_factory):
    """Only directly held Popen handles; no raw PID/group/neighbor signals or retries."""
    output = Path(protocol["output"])
    timing = {"job": job, "status": "failed", "scientific_claim": False, "limitations": LIMITATIONS,
              "test_read": False, "started_perf_counter": clock(), "command": command,
              "ownership": "direct Popen handle only", "gpu_charged": job is not None}
    if job is not None:
        timing["headroom"] = execution["headroom_checks"][-1]["snapshot"]
    process, original_error, cleanup_error = None, None, None
    try:
        check_budget(hard_deadline, clock=clock)
        deadline = hard_deadline - protocol["cleanup_reserve_seconds"]
        with (output / "workers" / (key + ".log")).open("x", encoding="utf-8") as log:
            spawned = clock()
            check_budget(hard_deadline, clock=clock)
            timing.update(spawned_perf_counter=spawned, deadline_perf_counter=deadline)
            if job is not None and execution.get("first_gpu_spawn_started_perf_counter") is None:
                execution["first_gpu_spawn_started_perf_counter"] = spawned
            process = popen_factory(command, cwd=output / "archived_code",
                                    env=worker_environment(protocol["gpu"]["uuid"] if job is not None else ""),
                                    stdout=log, stderr=subprocess.STDOUT)
            timing["owned_pid"] = process.pid
            try:
                code = process.wait(timeout=max(0.0, deadline - clock()))
            except subprocess.TimeoutExpired as exc:
                raise BudgetLimited(f"complement {key} owned worker hard timeout; stop without retry") from exc
            timing.update(returncode=code, reaped_perf_counter=clock())
            if code != 0:
                raise RuntimeError(f"complement {key} failed with exit {code}; stop without retry")
            check_budget(hard_deadline, clock=clock)
        timing["status"] = "success"
    except BaseException as exc:
        original_error = exc
        timing.update(failure_reason=f"{type(exc).__name__}: {exc}", budget_limited=isinstance(exc, BudgetLimited))
        raise
    finally:
        if process is not None:
            try:
                timing["cleanup"] = reap_owned(process, hard_deadline, clock=clock)
                timing["last_owned_reap_perf_counter"] = clock()
            except BaseException as exc:
                cleanup_error = exc
                timing.update(status="failed-unreaped", cleanup_failure=f"{type(exc).__name__}: {exc}")
                if original_error is not None:
                    original_error.add_note("owned worker cleanup failed/unreaped: " + str(exc))
        if job is not None and execution.get("first_gpu_spawn_started_perf_counter") is not None:
            execution["last_owned_gpu_reap_perf_counter"] = clock()
        timing["ended_perf_counter"] = clock()
        try:
            write_json(output / "workers" / (key + ".timing.json"), timing)
        except BaseException as exc:
            preserve_artifact_error(original_error or cleanup_error, exc, label="complement timing publication failed")
        if cleanup_error is not None:
            preserve_artifact_error(original_error, cleanup_error, label="complement owned worker failed/unreaped")
    return timing


def _verify_cpu_child(protocol_path, protocol, hard_deadline, execution, *, key, clock, popen_factory):
    receipt = Path(protocol["output"]) / "workers" / (key + ".json")
    command = worker_command(protocol_path, hard_deadline - protocol["cleanup_reserve_seconds"], receipt=receipt)
    _owned_call(protocol, command, key, hard_deadline, execution, job=None, clock=clock, popen_factory=popen_factory)
    result = read_json(receipt)
    if (result.get("status") != "success" or result.get("test_read") is not False
            or result.get("protocol_sha256") != protocol["original_protocol_sha256"]
            or result.get("complement_protocol_sha256") != protocol["protocol_sha256"]):
        raise ValueError("archive-isolated CPU qualification receipt missing/mismatched")
    check_budget(hard_deadline, clock=clock)
    return result


def _run_job(protocol_path, protocol, job, hard_deadline, observed_peak_mib, execution, *, clock,
             snapshot_fn, popen_factory):
    key = job_key(job)
    check_budget(hard_deadline, clock=clock)
    snapshot = verify_headroom(snapshot_fn(protocol["gpu"]["uuid"]), protocol["gpu"], observed_peak_mib)
    check_budget(hard_deadline, clock=clock)
    execution["headroom_checks"].append({"job": job, "snapshot": snapshot})
    command = worker_command(protocol_path, hard_deadline - protocol["cleanup_reserve_seconds"], job=job)
    _owned_call(protocol, command, key, hard_deadline, execution, job=job, clock=clock, popen_factory=popen_factory)
    result = read_json(Path(protocol["output"]) / "workers" / (key + ".json"))
    if (result.get("status") != "success" or result.get("job") != job
            or result.get("protocol_sha256") != protocol["original_protocol_sha256"]
            or result.get("complement_protocol_sha256") != protocol["protocol_sha256"]
            or result.get("scientific_claim") is not False or not result.get("limitations")
            or result.get("test_read") is not False or result.get("seed") != job["seed"]
            or result.get("arm") != job["arm"] or result.get("lead_hours") != job["lead"]
            or result.get("split") != "val" or result.get("evaluation_dir") != str(evaluation_dir(protocol["output"], job))):
        raise ValueError("owned exact-job receipt/protocol/path missing or mismatched; no skip/retry")
    for name in ("peak_reserved_bytes", "peak_allocated_bytes", "elapsed_seconds"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("finite actual owned evaluation peaks/timing required")
    training_name = f"workers/train_seed{job['seed']}_{job['arm']}.json"
    original = read_json(Path(protocol["original_output"]) / "protocol.json")
    if (result.get("original_training_receipt_sha256") != protocol["original_file_sha256"][training_name]
            or result.get("archived_evaluator_code") != {name: original["code"][name] for name in
                                                       ("code_zip_sha256", "source_tree_sha256", "model_code_sha256")}):
        raise ValueError("original selected training receipt/archived evaluator pins differ")
    check_budget(hard_deadline, clock=clock)
    return result


def prepare(original_output, output, *, started_perf_counter=None, clock=time.perf_counter):
    """CPU only; freeze the earliest same-boot whole-round clock before any pin work."""
    started = clock() if started_perf_counter is None else started_perf_counter
    if not math.isfinite(started) or started < 0 or started > clock():
        raise ValueError("valid monotonic prepare-entry start required")
    boot_id = monotonic_boot_id()
    hard_deadline = started + HARD_CAP_SECONDS
    check = lambda: check_budget(hard_deadline, clock=clock)
    original_output, output = local_path(original_output), local_path(output)
    if (output.is_relative_to(original_output) or original_output.is_relative_to(output)
            or output.exists() or output.is_symlink()):
        raise FileExistsError("complement requires a new exclusive output disjoint from original")
    output.mkdir(parents=True, exist_ok=False)
    original_error = None
    try:
        check()
        pins = pin_original_files(original_output, check=check)
        original = qualify_original(original_output)
        verify_original_archive(original_output, original, check=check)
        code = archive_driver_code(output)
        check()
        code["model_code_sha256"] = original["code"]["model_code_sha256"]
        protocol = build_protocol(output=output, original_output=original_output, original=original,
                                  original_files=pins, code=code, round_started_perf_counter=started, boot_id=boot_id)
        verify_original_files(protocol, check=check)
        check()
        write_json(output / "protocol.json", protocol)
        verify_protocol(output / "protocol.json")
        verify_driver_code(protocol, check=check)
        check()
        write_json(output / "prepare_attempt.json", {"status": "prepared-not-run", "scientific_claim": False,
                   "limitations": LIMITATIONS, "test_read": False, "elapsed_seconds": clock() - started,
                   "started_perf_counter": started, "ended_perf_counter": clock(),
                   "monotonic_boot_id": boot_id, "prepared_unix_seconds": time.time(),
                   "gpu_hours_charged": 0.0, "archived_runtime_qualification_pending": True})
        return protocol
    except BaseException as exc:
        original_error = exc
        try:
            write_json(output / "attempt.json", {"status": "failed", "phase": "cpu-prepare",
                       "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
                       "gpu_phase_elapsed_seconds": 0.0, "gpu_hours_charged": 0.0,
                       "whole_elapsed_seconds": clock() - started,
                       "started_perf_counter": started, "ended_perf_counter": clock(), "monotonic_boot_id": boot_id,
                       "soft_overrun_seconds": max(0.0, clock() - started - PLANNED_SECONDS),
                       "budget_limited": isinstance(exc, BudgetLimited),
                       "failure_reason": f"{type(exc).__name__}: {exc}", "finalized": False})
        except BaseException as additional:
            preserve_artifact_error(original_error, additional, label="complement prepare failure receipt publication failed")
        raise


def _runtime_preflight(protocol_path, protocol, hard_deadline, execution, *, clock, popen_factory):
    check = lambda: check_budget(hard_deadline, clock=clock)
    verify_driver_code(protocol, check=check)
    verify_original_files(protocol, check=check)
    original = qualify_original(protocol["original_output"])
    if (original["protocol_sha256"] != protocol["original_protocol_sha256"]
            or original["gpu"]["uuid"] != protocol["gpu"]["uuid"]):
        raise ValueError("bound original training protocol/UUID changed")
    extract_original_archive(protocol, original, check=check)
    _verify_cpu_child(protocol_path, protocol, hard_deadline, execution, key="archived_preflight",
                      clock=clock, popen_factory=popen_factory)
    check()


def _runtime_postflight(protocol_path, protocol, hard_deadline, execution, *, clock, popen_factory):
    check = lambda: check_budget(hard_deadline, clock=clock)
    verify_original_files(protocol, check=check)
    verify_driver_code(protocol, check=check)
    _verify_cpu_child(protocol_path, protocol, hard_deadline, execution, key="archived_postflight",
                      clock=clock, popen_factory=popen_factory)
    check()


def _update_costs(execution, started, *, clock):
    now = clock()
    first = execution.get("first_gpu_spawn_started_perf_counter")
    last = execution.get("last_owned_gpu_reap_perf_counter")
    gpu = 0.0 if first is None else max(0.0, (last if last is not None else now) - first)
    whole = max(0.0, now - started)
    execution.update(gpu_phase_elapsed_seconds=gpu, gpu_hours_charged=gpu / 3600.0,
                     whole_elapsed_seconds=whole, soft_overrun_seconds=max(0.0, whole - PLANNED_SECONDS),
                     ended_perf_counter=now, partial=bool(execution["jobs_completed"]) and
                     execution["jobs_completed"] != execution["jobs_planned"])


def finalize_complement(output, protocol, execution):
    from .r7_m3_complement_results import finalize_complement as aggregate
    return aggregate(output, protocol, execution)


def _publish_attempt(output, execution, outcome, original_error):
    attempt = {**execution, "outcome": outcome, "phase": "validation-complement",
               "original_failed_preserved": True,
               "cleanup_or_publication_diagnostics": list(getattr(original_error, "__notes__", []))}
    if original_error is not None:
        original_error.m3_complement_execution_attempt = dict(execution)
        original_error.m3_complement_attempt = dict(attempt)
    for name, payload in (("execution_attempt.json", execution), ("attempt.json", attempt)):
        try:
            write_json(output / name, payload)
        except BaseException as exc:
            exc.m3_complement_execution_attempt = dict(execution)
            if original_error is None:
                original_error = exc
                execution.update(status="failed", finalized=False, failure_reason=f"{type(exc).__name__}: {exc}")
                attempt.update(execution, outcome=outcome)
            else:
                preserve_artifact_error(original_error, exc, label=f"complement {name} publication failed")
    return original_error


def run_bounded_round(output, *, snapshot_fn=None, popen_factory=None, clock=time.perf_counter,
                      started_perf_counter=None, finalize_fn=None):
    """Whole clock includes frozen prepare entry and startup interval, same boot. Never retry."""
    started = clock() if started_perf_counter is None else started_perf_counter
    if not math.isfinite(started) or started > clock():
        raise ValueError("valid monotonic run-entry start required")
    output = local_path(output)
    if any((output / name).exists() or (output / name).is_symlink()
           for name in ("attempt.json", "execution_attempt.json", "run_started.json", "workers", "archived_code")):
        raise FileExistsError("complement already attempted/partially extracted; no rerun or regeneration")
    snapshot_fn = gpu_snapshot if snapshot_fn is None else snapshot_fn
    popen_factory = subprocess.Popen if popen_factory is None else popen_factory
    finalize_fn = finalize_complement if finalize_fn is None else finalize_fn
    hard_deadline = started + HARD_CAP_SECONDS
    execution = {"status": "failed", "scientific_claim": False, "limitations": LIMITATIONS, "test_read": False,
                 "started_perf_counter": started, "hard_deadline_perf_counter": hard_deadline,
                 "jobs_completed": [], "jobs_planned": planned_jobs(), "failed_job_key": None,
                 "failure_reason": None, "budget_limited": False, "partial": False, "finalized": False,
                 "evaluation_coverage_complete": False, "continuous_clock": True,
                 "headroom_checks": [], "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                 "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
                 "cap_gpu_seconds": HARD_CAP_SECONDS, "cap_round_seconds": HARD_CAP_SECONDS,
                 "first_gpu_spawn_started_perf_counter": None, "last_owned_gpu_reap_perf_counter": None,
                 "billing_scope": "continuous before first CUDA spawn/startup through last owned GPU reap, gaps/failures/cleanup included; CPU-only children excluded",
                 "execution_whole_scope": "run entry through pre-aggregation snapshot; aggregation/cleanup included only by final attempt.json",
                 "whole_round_cost_reference": str(output / "attempt.json")}
    original_error, outcome = None, None
    try:
        write_json(output / "run_started.json", {"scientific_claim": False, "limitations": LIMITATIONS,
                   "test_read": False, "started_perf_counter": started, "started_unix_seconds": time.time()})
        check_budget(hard_deadline, clock=clock)
        protocol_path = output / "protocol.json"
        protocol = verify_protocol(protocol_path)
        anchor = protocol["round_started_perf_counter"]
        if protocol["monotonic_boot_id"] != monotonic_boot_id() or anchor > started:
            raise ValueError("same-boot frozen prepare clock required; no resume across reboot/future anchor")
        started = min(started, anchor)
        hard_deadline = started + protocol["hard_cap_seconds"]
        execution.update(started_perf_counter=started, hard_deadline_perf_counter=hard_deadline,
                         run_entry_perf_counter=execution["started_perf_counter"], monotonic_boot_id=protocol["monotonic_boot_id"])
        check_budget(hard_deadline, clock=clock)
        execution.update(protocol_sha256=protocol["protocol_sha256"],
                         original_protocol_sha256=protocol["original_protocol_sha256"],
                         billing_scope=protocol["gpu"]["billing"])
        (output / "workers").mkdir(exist_ok=False)
        _runtime_preflight(protocol_path, protocol, hard_deadline, execution, clock=clock, popen_factory=popen_factory)
        observed_peak_mib = 0.0
        for job in protocol["jobs"]:
            execution["failed_job_key"] = job_key(job)
            result = _run_job(protocol_path, protocol, job, hard_deadline, observed_peak_mib, execution,
                              clock=clock, snapshot_fn=snapshot_fn, popen_factory=popen_factory)
            observed_peak_mib = max(observed_peak_mib, result["peak_reserved_bytes"] / 2 ** 20)
            execution["jobs_completed"].append(job)
        execution["failed_job_key"] = None
        _runtime_postflight(protocol_path, protocol, hard_deadline, execution, clock=clock, popen_factory=popen_factory)
        _update_costs(execution, started, clock=clock)
        check_budget(hard_deadline, clock=clock)
        if execution["jobs_completed"] != protocol["jobs"]:
            raise ValueError("all 23 exact successful evaluations required before aggregation")
        execution.update(status="evaluations-complete", evaluation_coverage_complete=True)
        execution["aggregation_started_perf_counter"] = clock()
        # Stage facts are not an early success receipt. Pass a separate dict to the aggregator.
        outcome = finalize_fn(output, protocol, dict(execution))
        execution["aggregation_ended_perf_counter"] = clock()
        check_budget(hard_deadline, clock=clock)
        if (not isinstance(outcome, dict) or type(outcome.get("paused")) is not bool
                or outcome.get("scientific_claim") is not False or not outcome.get("limitations")):
            raise ValueError("frozen descriptive_outcome including scientific pause required")
        execution.update(status="paused" if outcome["paused"] else "success", finalized=True,
                         scientific_halt=outcome["paused"], outcome=outcome)
    except BaseException as exc:
        original_error = exc
        execution.update(status="failed", finalized=False, failure_reason=f"{type(exc).__name__}: {exc}",
                         budget_limited=isinstance(exc, BudgetLimited))
    finally:
        _update_costs(execution, started, clock=clock)
        if execution["whole_elapsed_seconds"] >= HARD_CAP_SECONDS:
            if original_error is None:
                original_error = BudgetLimited("whole round including aggregation/owned cleanup exceeded hard cap")
            execution.update(status="failed", finalized=False, budget_limited=True,
                             failure_reason=f"{type(original_error).__name__}: {original_error}")
        execution["execution_whole_scope"] = (
            "frozen prepare CLI entry through freeze, startup interval, CPU identity/extraction, GPU work, "
            "aggregation and owned cleanup; receipt serialization follows this final snapshot")
        execution["cleanup_or_publication_diagnostics"] = list(getattr(original_error, "__notes__", []))
        original_error = _publish_attempt(output, execution, outcome, original_error)
    if original_error is not None:
        raise original_error
    return outcome
