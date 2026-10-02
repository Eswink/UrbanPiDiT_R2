"""Prepare before updates, then 36 sequential owned workers under a continuous cap."""
from __future__ import annotations

import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .r7_arm_harness import sha256_file
from .r7_m3_identity import (
    ROOT, archive_code, dataset_pins, local_path, sidecar_pins, source_identity,
    verify_code, verify_input_pins,
)
from .r7_m3_protocol import (
    CLEANUP_SECONDS, GLOBAL_SECONDS, LIMITATIONS, ROUND_SECONDS, WORKER_SECONDS,
    build_protocol, job_key, planned_jobs, preserve_artifact_error, read_json, verify_authorization,
    verify_protocol, write_json,
)

DEFAULT_ESTIMATED_PEAK_MIB = 342
ESTIMATE_SOURCE = "docs/R7_N1_PIVOT_AUDIT.md section5.3: historical RW_A reserved max342MiB, not a current M3 measurement"


class BudgetLimited(TimeoutError):
    pass


def gpu_snapshot(uuid, *, query=subprocess.check_output):
    """Read-only inventory; a neighbor PID is recorded, never signaled or refused."""
    cards = query(["nvidia-smi", "--query-gpu=uuid,index,memory.free,memory.total",
                   "--format=csv,noheader,nounits"], text=True, timeout=5)
    rows = []
    for line in cards.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise ValueError("malformed GPU UUID/headroom inventory")
        rows.append({"uuid": fields[0], "index": int(fields[1]),
                     "free_mib": int(fields[2]), "total_mib": int(fields[3])})
    matching = [row for row in rows if row["uuid"] == uuid]
    if len(matching) != 1:
        raise ValueError("bound physical GPU UUID missing or duplicated; no fallback")
    selected = matching[0]
    apps = query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                  "--format=csv,noheader,nounits"], text=True, timeout=5)
    neighbors = []
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            raise ValueError("malformed read-only GPU process inventory")
        if fields[0] == uuid:
            neighbors.append({"pid": int(fields[1]), "used_mib": fields[2]})
    return {**selected, "neighbors": neighbors, "queried_unix_seconds": time.time(),
            "all_cards": rows, "read_only": True}


def verify_headroom(snapshot, gpu, observed_peak_mib=0):
    required = max(gpu["estimated_peak_mib"], math.ceil(observed_peak_mib)) + gpu["headroom_margin_mib"]
    if snapshot["uuid"] != gpu["uuid"] or snapshot["free_mib"] < required:
        raise RuntimeError(f"shared GPU UUID/headroom refused: need {required}MiB; no neighbor interference")
    return {**snapshot, "required_free_mib": required, "observed_owned_peak_mib": observed_peak_mib}


def worker_command(protocol_path, job, deadline):
    args = [str(ROOT / ".venv" / "bin" / "python"), "-B", "-m", "training.r7_m3_worker",
            "--protocol", str(protocol_path), "--phase", job["phase"], "--seed", str(job["seed"]),
            "--arm", job["arm"], "--device", "cuda:0", "--deadline", repr(deadline)]
    if job["lead"] is not None:
        args.extend(["--lead", str(job["lead"])])
    return args


def worker_environment(uuid):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}


def reap_owned(process, hard_deadline, *, clock=time.perf_counter):
    """Only this directly-created Popen handle; no raw PIDs/process groups/signals."""
    if process.poll() is not None:
        process.wait(timeout=0)
        return "already-exited"
    process.terminate()
    try:
        process.wait(timeout=max(0.0, min(3.0, hard_deadline - clock())))
        return "terminated-owned-worker"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.0, hard_deadline - clock()))
        return "killed-and-reaped-owned-worker"


def _run_job(protocol_path, protocol, job, hard_deadline, observed_peak_mib, *, clock,
             snapshot_fn, popen_factory):
    output = Path(protocol["output"])
    key = job_key(job)
    timing = {"job": job, "scientific_claim": False, "limitations": LIMITATIONS,
              "status": "failed", "started_perf_counter": clock(), "ownership": "direct Popen handle only"}
    process, original_error = None, None
    cleanup_deadline = hard_deadline
    try:
        if clock() >= hard_deadline - CLEANUP_SECONDS:
            raise BudgetLimited("M3 continuous round deadline exhausted before spawn")
        timing["headroom"] = verify_headroom(snapshot_fn(protocol["gpu"]["uuid"]),
                                            protocol["gpu"], observed_peak_mib)
        spawned = clock()
        cleanup_deadline = min(hard_deadline, spawned + WORKER_SECONDS)
        deadline = cleanup_deadline - CLEANUP_SECONDS
        if spawned >= deadline:
            raise BudgetLimited("M3 round deadline exhausted during headroom query")
        command = worker_command(protocol_path, job, deadline)
        timing.update(command=command, spawned_perf_counter=spawned, deadline_perf_counter=deadline)
        with (output / "workers" / (key + ".log")).open("x", encoding="utf-8") as log:
            process = popen_factory(command, cwd=ROOT, env=worker_environment(protocol["gpu"]["uuid"]),
                                    stdout=log, stderr=subprocess.STDOUT)
            timing["owned_pid"] = process.pid
            try:
                code = process.wait(timeout=max(0.0, deadline - clock()))
            except subprocess.TimeoutExpired as exc:
                raise BudgetLimited("M3 owned worker hard timeout; full round stops without retry") from exc
            timing.update(returncode=code, reaped_perf_counter=clock())
            if code != 0:
                raise RuntimeError(f"M3 {key} failed with exit {code}; full round stops, no retry")
            if clock() > deadline:
                raise BudgetLimited("M3 worker returned after execution deadline")
        from .r7_m3_worker import worker_result_path
        result = read_json(worker_result_path(output, job))
        if (result.get("status") != "success" or result.get("job") != job
                or result.get("protocol_sha256") != protocol["protocol_sha256"]):
            raise ValueError("owned worker missing successful exact-job/protocol result; no skip")
        timing["status"] = "success"
        return result
    except BaseException as exc:
        original_error = exc
        timing["failure_reason"] = f"{type(exc).__name__}: {exc}"
        timing["budget_limited"] = isinstance(exc, BudgetLimited)
        raise
    finally:
        cleanup_error = None
        if process is not None:
            try:
                timing["cleanup"] = reap_owned(process, cleanup_deadline, clock=clock)
                timing["last_owned_reap_perf_counter"] = clock()
            except BaseException as exc:
                cleanup_error = exc
                timing.update(status="failed-unreaped", cleanup_failure=f"{type(exc).__name__}: {exc}")
                if original_error is not None:
                    original_error.add_note("owned worker cleanup failed/unreaped: " + str(exc))
        timing["ended_perf_counter"] = clock()
        try:
            write_json(output / "workers" / (key + ".timing.json"), timing)
        except BaseException as exc:
            preserve_artifact_error(original_error or cleanup_error, exc, label="M3 timing receipt publication failed")
        if cleanup_error is not None:
            preserve_artifact_error(original_error, cleanup_error, label="M3 owned worker failed/unreaped")


def execute_jobs(protocol_path, *, snapshot_fn=gpu_snapshot, popen_factory=subprocess.Popen,
                 clock=time.perf_counter):
    """Fakeable CPU gate. Billing includes startup, all gaps, scoring, owned cleanup."""
    protocol = verify_protocol(protocol_path)
    output = Path(protocol["output"])
    started = clock()
    hard_deadline = started + min(protocol["gpu"]["max_gpu_seconds"],
                                  protocol["gpu"]["max_round_seconds"])
    attempt = {"status": "failed", "scientific_claim": False, "limitations": LIMITATIONS,
               "test_read": False, "started_perf_counter": started,
               "hard_deadline_perf_counter": hard_deadline, "jobs_completed": [],
               "jobs_planned": planned_jobs(), "failure_reason": None, "budget_limited": False,
               "cap_gpu_seconds": GLOBAL_SECONDS, "cap_round_seconds": ROUND_SECONDS,
               "billing_scope": protocol["gpu"]["billing"], "finalized": False}
    original_error = None
    try:
        observed_peak_mib = 0
        for job in planned_jobs():
            result = _run_job(protocol_path, protocol, job, hard_deadline, observed_peak_mib,
                              clock=clock, snapshot_fn=snapshot_fn, popen_factory=popen_factory)
            observed_peak_mib = max(observed_peak_mib, result["peak_reserved_bytes"] / 2 ** 20)
            attempt["jobs_completed"].append(job)
        if clock() > hard_deadline:
            raise BudgetLimited("M3 whole round including last reap exceeded frozen cap")
        attempt["status"] = "success"
    except BaseException as exc:
        original_error = exc
        attempt.update(failure_reason=f"{type(exc).__name__}: {exc}",
                       budget_limited=isinstance(exc, BudgetLimited),
                       cleanup_or_publication_diagnostics=list(getattr(exc, "__notes__", [])))
        raise
    finally:
        elapsed = clock() - started
        attempt.update(gpu_phase_elapsed_seconds=elapsed, gpu_hours_charged=elapsed / 3600.0,
                       ended_perf_counter=clock(), partial=bool(attempt["jobs_completed"]) and
                       len(attempt["jobs_completed"]) != len(planned_jobs()))
        # Keep charged execution facts even when the filesystem rejects a receipt.
        if original_error is not None:
            original_error.m3_execution_attempt = dict(attempt)
        try:
            write_json(output / "execution_attempt.json", attempt)
        except BaseException as exc:
            exc.m3_execution_attempt = dict(attempt)
            preserve_artifact_error(original_error, exc, label="M3 execution receipt publication failed")
    return attempt


def prepare(manifests, sidecar, output, authorization_path, gpu_uuid,
            estimated_peak_mib=DEFAULT_ESTIMATED_PEAK_MIB):
    from torch.utils.data import default_collate
    import torch
    from .r7_m3_profile import profile_arms
    from .r7_m3_worker import M3Dataset, deny_network, make_context
    authorization = verify_authorization(authorization_path)
    requested = Path(output)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("M3 requires a new exclusive output directory")
    output, manifests = requested.resolve(), local_path(manifests)
    output.mkdir(parents=True, exist_ok=False)
    started, failure, original_error = time.perf_counter(), None, None
    try:
        deny_network()
        torch.set_num_threads(4)
        data, reader, root = dataset_pins(manifests)
        sources = source_identity(manifests, root)
        pins, metadata = sidecar_pins(sidecar, data, sources, reader, root)
        context = make_context(metadata, root)
        dataset = M3Dataset(manifests / "train.jsonl", reader=reader)
        if len(dataset) < 2:
            raise ValueError("fixed batch2 requires at least two actual train cases")
        probe = default_collate([dataset[0], dataset[1]])
        first_case = default_collate([dataset[0]])
        profile = profile_arms(probe, first_case, context)
        code = archive_code(output)
        protocol = build_protocol(manifests=manifests, output=output, dataset=data,
                                  sources=sources, sidecar=pins, code=code, profile=profile,
                                  authorization=authorization,
                                  authorization_sha256=sha256_file(authorization_path),
                                  gpu_uuid=gpu_uuid, estimated_peak_mib=estimated_peak_mib)
        protocol["gpu"]["estimate_source"] = ESTIMATE_SOURCE if estimated_peak_mib == 342 else "explicit engineering peak estimate, not measured in this round"
        from .r7_m3_protocol import digest
        protocol["protocol_sha256"] = digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
        verify_input_pins(protocol)
        write_json(output / "cpu_profile.json", profile)
        write_json(output / "protocol.json", protocol)
        verify_protocol(output / "protocol.json")
        write_json(output / "environment.json", {"scientific_claim": False, "limitations": LIMITATIONS,
                                                 "python": sys.version, "torch": str(torch.__version__),
                                                 "platform": platform.platform(), "phase": "cpu-prepare"})
        return protocol
    except BaseException as exc:
        original_error = exc
        failure = f"{type(exc).__name__}: {exc}"
        try:
            write_json(output / "attempt.json", {"status": "failed", "phase": "cpu-prepare",
                       "scientific_claim": False, "limitations": LIMITATIONS,
                       "test_read": False, "gpu_hours_charged": 0.0, "failure_reason": failure})
        except BaseException as additional:
            preserve_artifact_error(exc, additional, label="M3 prepare failure receipt publication failed")
        raise
    finally:
        try:
            write_json(output / "prepare_attempt.json", {"status": "failed" if failure else "prepared-not-run",
                       "scientific_claim": False, "limitations": LIMITATIONS,
                       "elapsed_seconds": time.perf_counter() - started,
                       "gpu_hours_charged": 0.0, "failure_reason": failure,
                       "historical_cost_note": LIMITATIONS[-1]})
        except BaseException as exc:
            preserve_artifact_error(original_error, exc, label="M3 prepare receipt publication failed")


def run_bounded_round(output, authorization_path):
    """No retries/resume: CPU identities are checked before and after owned GPU work."""
    output = local_path(output)
    if (output / "attempt.json").exists() or (output / "run_started.json").exists():
        raise FileExistsError("M3 round has already been attempted; no repeated jobs")
    authorization = verify_authorization(authorization_path)
    write_json(output / "run_started.json", {"scientific_claim": False, "limitations": LIMITATIONS,
                                            "authorized_scope": authorization["scope"], "started_unix_seconds": time.time()})
    started, execution, failure, outcome = time.perf_counter(), None, None, None
    original_error = None
    try:
        protocol = verify_protocol(output / "protocol.json")
        if (protocol["authorization"] != authorization
                or sha256_file(authorization_path) != protocol["authorization_sha256"]
                or protocol["output"] != str(output)):
            raise ValueError("frozen named authorization or output path differs")
        verify_code(protocol)
        verify_input_pins(protocol)
        (output / "workers").mkdir(exist_ok=False)
        execution = execute_jobs(output / "protocol.json")
        # Last worker is reaped; CPU identity verification/finalization is not charged twice.
        verify_code(protocol)
        verify_input_pins(protocol)
        from .r7_m3_results import finalize
        outcome = finalize(output, protocol, execution)
        return outcome
    except BaseException as exc:
        original_error = exc
        failure = f"{type(exc).__name__}: {exc}"
        if execution is None:
            execution = getattr(exc, "m3_execution_attempt", None)
        if execution is None and (output / "execution_attempt.json").is_file():
            try:
                execution = read_json(output / "execution_attempt.json")
            except BaseException as additional:
                preserve_artifact_error(exc, additional, label="M3 execution receipt recovery failed")
        raise
    finally:
        execution = execution or {"jobs_completed": [], "gpu_phase_elapsed_seconds": 0.0,
                                  "gpu_hours_charged": 0.0, "budget_limited": False}
        completed = execution["jobs_completed"]
        try:
            write_json(output / "attempt.json", {"scientific_claim": False, "limitations": LIMITATIONS,
                     "status": "failed" if failure else ("paused" if outcome["paused"] else "success"),
                     "test_read": False, "failure_reason": failure, "budget_limited": execution["budget_limited"],
                     "partial": bool(completed) and len(completed) != 36, "jobs_completed": completed,
                     "gpu_phase_elapsed_seconds": execution["gpu_phase_elapsed_seconds"],
                     "gpu_hours_charged": execution["gpu_hours_charged"], "cap_gpu_seconds": GLOBAL_SECONDS,
                     "cap_round_seconds": ROUND_SECONDS, "whole_elapsed_seconds": time.perf_counter() - started,
                     "cpu_finalize_excluded_from_gpu_charge": True, "finalized": failure is None,
                     "cleanup_or_publication_diagnostics": list(getattr(original_error, "__notes__", [])),
                     "outcome": outcome})
        except BaseException as exc:
            preserve_artifact_error(original_error, exc, label="M3 whole-round receipt publication failed")
