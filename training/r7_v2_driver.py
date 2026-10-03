"""Single-attempt v2 CPU prepare, sequential owned workers and final whole-round seal."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .r7_v2_identity import archive_code, pin_inputs, pin_parents, verify_code, verify_input_pins
from .r7_v2_protocol import (
    BUDGETS, CLEANUP_SECONDS, LIMITATIONS, ROOT, build_protocol, digest, job_key, local_path,
    make_directory, monotonic_boot_id, preserve_error, read_json, safe_output, sha256_file,
    verify_protocol, worker_result_path, write_json, write_path,
)


class BudgetLimited(TimeoutError):
    pass


def gpu_snapshot(uuid):
    from .r7_m3_driver import gpu_snapshot as snapshot
    return snapshot(uuid)


def verify_headroom(snapshot, gpu, observed_peak_mib=0):
    from .r7_m3_driver import verify_headroom as gate
    return gate(snapshot, gpu, observed_peak_mib)


def reap_owned(process, hard_deadline, *, clock=time.perf_counter):
    from .r7_m3_driver import reap_owned as reap
    return reap(process, hard_deadline, clock=clock)


def check_budget(hard_deadline, *, clock=time.perf_counter, reserve=CLEANUP_SECONDS):
    if clock() >= hard_deadline - reserve:
        raise BudgetLimited("v2 frozen whole-round hard deadline exhausted; no retry or resurrection")


def worker_environment(uuid):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "OMP_NUM_THREADS": "4",
                   "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    environment.pop("PYTHONPATH", None)
    return environment


def worker_command(protocol_path, job, deadline):
    args = [str(ROOT / ".venv/bin/python"), "-B", "-m", "training.r7_v2_worker",
            "--protocol", str(protocol_path), "--phase", job["phase"], "--seed", str(job["seed"]),
            "--arm", job["arm"], "--reasoning-steps", str(job["reasoning_steps"]),
            "--device", "cuda:0", "--deadline", repr(deadline)]
    if job["lead"] is not None:
        args.extend(["--lead", str(job["lead"])])
    return args


def _costs(execution, protocol, *, clock):
    now = clock()
    started = execution["started_perf_counter"]
    first, last = execution["first_gpu_spawn_started_perf_counter"], execution["last_owned_gpu_reap_perf_counter"]
    gpu = 0.0 if first is None else max(0.0, (now if execution.get("owned_unreaped") or last is None else last) - first)
    whole = max(0.0, now - started)
    execution.update(ended_perf_counter=now, whole_elapsed_seconds=whole,
                     gpu_phase_elapsed_seconds=gpu, gpu_hours_charged=gpu / 3600.0,
                     soft_overrun_seconds=max(0.0, whole - protocol["planned_seconds"]),
                     soft_budget_exceeded=whole > protocol["planned_seconds"],
                     partial=bool(execution["jobs_completed"]) and execution["jobs_completed"] != execution["jobs_planned"])


def _execution(protocol):
    return {"status": "failed", "finalized": False, "scientific_claim": False,
            "limitations": list(protocol["limitations"]), "test_read": False,
            "protocol_sha256": protocol["protocol_sha256"], "stage": protocol["stage"],
            "started_perf_counter": protocol["round_started_perf_counter"],
            "hard_deadline_perf_counter": protocol["round_started_perf_counter"] + protocol["hard_cap_seconds"],
            "monotonic_boot_id": protocol["monotonic_boot_id"], "planned_seconds": protocol["planned_seconds"],
            "hard_cap_seconds": protocol["hard_cap_seconds"], "cleanup_reserve_seconds": CLEANUP_SECONDS,
            "jobs_planned": protocol["jobs"], "jobs_completed": [], "jobs_results": [],
            "headroom_checks": [], "failed_job_key": None, "failure_reason": None, "budget_limited": False,
            "first_gpu_spawn_started_perf_counter": None, "last_owned_gpu_reap_perf_counter": None,
            "owned_unreaped": False, "continuous_clock": True, "no_total_gpu_cap": True,
            "billing_scope": protocol["gpu"]["billing"], "partial": False,
            "whole_round_cost_reference": protocol["whole_round_cost_reference"]}


def _validate_result(result, protocol, job):
    expected = {"protocol_sha256": protocol["protocol_sha256"], "model_code_sha256": protocol["code"]["model_code_sha256"],
                "source_tree_sha256": protocol["code"]["source_tree_sha256"], "code_zip_sha256": protocol["code"]["code_zip_sha256"],
                "data_identity": protocol["data"]["data_identity"], "source_sha256": protocol["sources"]["source_sha256"],
                "sidecar_identity": protocol["sidecar"]["identity"], "windows_sha256": digest(protocol["windows"])}
    if (result.get("job") != job or result.get("status") != "success" or result.get("scientific_claim") is not False
            or result.get("test_read") is not False or not result.get("limitations")
            or any(result.get(key) != value for key, value in expected.items())):
        raise ValueError("successful exact worker job/source/protocol receipt required; no skipped or partial jobs")
    for name in ("peak_allocated_bytes", "peak_reserved_bytes", "elapsed_seconds"):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("finite measured owned peaks and elapsed time required")
    if result.get("baseline") != {"allocated_bytes": 0, "reserved_bytes": 0}:
        raise ValueError("fresh measured zero CUDA allocator baseline required for every worker")
    if job["phase"] == "train" and result.get("updates_run") != protocol["arm_configs"][job["arm"]]["updates"]:
        raise ValueError("frozen complete training update endpoint required")
    return result


def _worker_exit_error(protocol, job, code, deadline, *, clock):
    """Distinguish owned child deadline exits from early failures, preserving exact receipts."""
    receipt, receipt_error = None, None
    path = worker_result_path(protocol["output"], job)
    if path.exists():
        try:
            value = read_json(path)
            if (value.get("job") == job and value.get("protocol_sha256") == protocol["protocol_sha256"]
                    and value.get("status") == "failed"):
                receipt = value
        except BaseException as exc:
            receipt_error = exc
    exact_budget = (receipt is not None and receipt.get("budget_limited") is True
                    and receipt.get("deadline_perf_counter") == deadline
                    and receipt.get("monotonic_boot_id") == protocol["monotonic_boot_id"])
    budget = clock() >= deadline or exact_budget
    detail = "" if receipt is None else f"; worker reported {receipt.get('failure_reason', 'unspecified failure')}"
    error = (BudgetLimited if budget else RuntimeError)(
        f"v2 {job_key(job)} failed with exit {code}; stop without retry{detail}")
    error.v2_worker_failure = receipt
    if receipt_error is not None:
        error.add_note(f"worker failure receipt unreadable: {type(receipt_error).__name__}: {receipt_error}")
    return error


def _run_job(protocol_path, protocol, job, execution, observed_peak_mib, *, clock, snapshot_fn, popen_factory):
    output, hard_deadline = safe_output(protocol["output"]), execution["hard_deadline_perf_counter"]
    key = job_key(job)
    timing = {"job": job, "status": "failed", "scientific_claim": False, "limitations": list(LIMITATIONS),
              "test_read": False, "started_perf_counter": clock(), "protocol_sha256": protocol["protocol_sha256"],
              "ownership": "only the directly returned Popen handle; never raw PIDs or groups"}
    process, error, cleanup_error = None, None, None
    try:
        check_budget(hard_deadline, clock=clock)
        snapshot = verify_headroom(snapshot_fn(protocol["gpu"]["uuid"]), protocol["gpu"], observed_peak_mib)
        timing["headroom"] = snapshot
        execution["headroom_checks"].append({"job": job, "snapshot": snapshot})
        check_budget(hard_deadline, clock=clock)
        deadline = hard_deadline - protocol["cleanup_reserve_seconds"]
        command = worker_command(protocol_path, job, deadline)
        timing.update(command=command, deadline_perf_counter=deadline)
        with write_path(output / "workers" / (key + ".log"), output).open("x", encoding="utf-8") as log:
            spawned = clock()
            check_budget(hard_deadline, clock=clock)
            if execution["first_gpu_spawn_started_perf_counter"] is None:
                execution["first_gpu_spawn_started_perf_counter"] = spawned
            timing["spawned_perf_counter"] = spawned
            process = popen_factory(command, cwd=ROOT, env=worker_environment(protocol["gpu"]["uuid"]),
                                    stdout=log, stderr=subprocess.STDOUT)
            timing["owned_pid"] = process.pid
            try:
                code = process.wait(timeout=max(0.0, deadline - clock()))
            except subprocess.TimeoutExpired as exc:
                raise BudgetLimited("v2 owned worker hard timeout; entire attempt stops, no retry") from exc
            timing.update(returncode=code, reaped_perf_counter=clock())
            if code != 0:
                raise _worker_exit_error(protocol, job, code, deadline, clock=clock)
            check_budget(hard_deadline, clock=clock)
        result = _validate_result(read_json(worker_result_path(output, job)), protocol, job)
        timing["status"] = "success"
        return result
    except BaseException as exc:
        error = exc
        timing.update(failure_reason=f"{type(exc).__name__}: {exc}", budget_limited=isinstance(exc, BudgetLimited))
        if getattr(exc, "v2_worker_failure", None) is not None:
            timing["worker_failure_receipt"] = exc.v2_worker_failure
            execution["worker_failure_receipt"] = exc.v2_worker_failure
        raise
    finally:
        if process is not None:
            try:
                timing["cleanup"] = reap_owned(process, hard_deadline, clock=clock)
                timing["last_owned_reap_perf_counter"] = clock()
                execution["last_owned_gpu_reap_perf_counter"] = clock()
            except BaseException as exc:
                cleanup_error = exc
                execution["owned_unreaped"] = True
                timing.update(status="failed-unreaped", cleanup_failure=f"{type(exc).__name__}: {exc}")
                if error is not None:
                    preserve_error(error, exc, "owned cleanup failed/unreaped")
        elif "spawned_perf_counter" in timing:
            execution["last_owned_gpu_reap_perf_counter"] = clock()
            timing["cleanup"] = "launch failed before an owned handle was returned"
        timing["ended_perf_counter"] = clock()
        try:
            write_json(output / "workers" / (key + ".timing.json"), timing, output=output)
        except BaseException as exc:
            preserve_error(error or cleanup_error, exc, "v2 timing receipt publication failed")
        if cleanup_error is not None:
            preserve_error(error, cleanup_error, "owned worker unreaped")


def _runtime_preflight(protocol, *, check):
    verify_code(protocol, check=check)
    verify_input_pins(protocol, check=check)


def _runtime_postflight(protocol, *, check):
    verify_input_pins(protocol, check=check)
    verify_code(protocol, check=check)


def execute_jobs(protocol_path, *, execution=None, snapshot_fn=None, popen_factory=None, clock=time.perf_counter):
    """All trains precede all single-lead/K fresh evals; CPU fake-worker injection supported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    protocol = verify_protocol(protocol_path)
    if execution is None:
        raise ValueError("execute_jobs requires a claimed run execution; use run_bounded_round")
    if execution["protocol_sha256"] != protocol["protocol_sha256"] or execution["jobs_completed"]:
        raise ValueError("new exact protocol execution required; no resume/retry")
    snapshot_fn = gpu_snapshot if snapshot_fn is None else snapshot_fn
    popen_factory = subprocess.Popen if popen_factory is None else popen_factory
    observed = 0.0
    try:
        for job in protocol["jobs"]:
            execution["failed_job_key"] = job_key(job)
            result = _run_job(protocol_path, protocol, job, execution, observed, clock=clock,
                              snapshot_fn=snapshot_fn, popen_factory=popen_factory)
            observed = max(observed, result["peak_reserved_bytes"] / 2 ** 20)
            execution["jobs_completed"].append(job)
            execution["jobs_results"].append({"job": job, "result": str(worker_result_path(protocol["output"], job)),
                                              "peak_reserved_bytes": result["peak_reserved_bytes"],
                                              "elapsed_seconds": result["elapsed_seconds"]})
        execution.update(status="results-complete", failed_job_key=None)
        return execution
    finally:
        _costs(execution, protocol, clock=clock)


def prepare(manifests, sidecar, output, gpu_uuid, *, stage="B", configuration=None,
            round_started_perf_counter=None, boot_id=None, clock=time.perf_counter):
    """CPU-only identity/parent/profile preparation, before independent protocol freeze."""
    from scripts.r7_m3_offline import deny_network
    deny_network()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = clock() if round_started_perf_counter is None else round_started_perf_counter
    boot = monotonic_boot_id() if boot_id is None else boot_id
    if (stage not in BUDGETS or isinstance(started, bool) or not math.isfinite(started)
            or started < 0 or started > clock() or boot != monotonic_boot_id()):
        raise ValueError("valid earliest same-boot prepare CLI clock required")
    output = safe_output(output)  # Refusal happens outside any finally publication.
    if output.exists():
        raise FileExistsError("new exclusive attempt output required; existing outputs are read-only")
    manifests, sidecar = local_path(manifests), local_path(sidecar)
    output.mkdir(parents=True, exist_ok=False)
    planned, hard = BUDGETS[stage]
    deadline = started + hard
    check = lambda: check_budget(deadline, clock=clock)
    try:
        check()
        write_json(output / "prepare_started.json", {"scientific_claim": False, "limitations": list(LIMITATIONS),
                   "round_started_perf_counter": started, "monotonic_boot_id": boot, "stage": stage}, output=output)
        from .r7_v2_profile import profile_arms
        import torch
        torch.set_num_threads(4)
        if torch.cuda.is_initialized():
            raise ValueError("CPU prepare must not initialize CUDA")
        data, sources, pins, windows, _, _, _ = pin_inputs(manifests, sidecar, check=check)
        parents = pin_parents(manifests, pins["path"], data, sources, check=check) if stage == "B" else {}
        profile = profile_arms(manifests, windows, parents, stage=stage, configuration=configuration, check=check)
        check()
        code = archive_code(output, check=check)
        protocol = build_protocol(stage=stage, output=output, manifests=manifests, data=data,
                                  sources=sources, sidecar=pins, windows=windows, parents=parents,
                                  profile=profile, code=code, gpu_uuid=gpu_uuid,
                                  round_started_perf_counter=started, boot_id=boot, configuration=configuration)
        _runtime_preflight(protocol, check=check)
        write_json(output / "cpu_profile.json", profile, output=output)
        write_json(output / "environment.json", {"scientific_claim": False, "limitations": list(LIMITATIONS),
                   "python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
                   "phase": "cpu-prepare", "cuda_initialized": False}, output=output)
        check()
        write_json(output / "protocol.json", protocol, output=output)
        verify_protocol(output / "protocol.json")
        check()
        write_json(output / "prepare_attempt.json", {"status": "prepared-not-run", "scientific_claim": False,
                   "limitations": list(LIMITATIONS), "protocol_sha256": protocol["protocol_sha256"],
                   "test_read": False, "started_perf_counter": started, "ended_perf_counter": clock(),
                   "monotonic_boot_id": boot, "elapsed_seconds": clock() - started,
                   "soft_overrun_seconds": max(0.0, clock() - started - planned), "gpu_hours_charged": 0.0}, output=output)
        return protocol
    except BaseException as exc:
        attempt = {"status": "failed", "phase": "cpu-prepare", "finalized": False,
                   "scientific_claim": False, "limitations": list(LIMITATIONS), "test_read": False,
                   "started_perf_counter": started, "ended_perf_counter": clock(), "monotonic_boot_id": boot,
                   "whole_elapsed_seconds": clock() - started, "soft_overrun_seconds": max(0.0, clock() - started - planned),
                   "planned_seconds": planned, "hard_cap_seconds": hard, "gpu_phase_elapsed_seconds": 0.0,
                   "gpu_hours_charged": 0.0, "failure_reason": f"{type(exc).__name__}: {exc}",
                   "budget_limited": isinstance(exc, BudgetLimited), "no_retry_or_resurrection": True}
        exc.v2_attempt = attempt
        try:
            write_json(output / "attempt.json", attempt, output=output)
        except BaseException as additional:
            preserve_error(exc, additional, "v2 prepare failure receipt publication failed")
        raise


def finalize_results(output, protocol, execution):
    from .r7_v2_results import finalize
    return finalize(output, protocol, execution)


def _seal_unqualified(output, error, run_entry, *, clock):
    """Claimed startup failed before a trustworthy execution could be constructed."""
    ended = clock()
    attempt = {"status": "failed", "phase": "v2-unqualified-startup", "finalized": False,
               "scientific_claim": False, "limitations": [*LIMITATIONS,
                   "protocol/execution identity unqualified; no whole-round clock or frozen digest asserted"],
               "test_read": False, "protocol_identity_qualified": False,
               "run_entry_perf_counter": run_entry, "ended_perf_counter": ended,
               "run_entry_elapsed_seconds": None if run_entry is None else max(0.0, ended - run_entry),
               "failure_reason": f"{type(error).__name__}: {error}",
               "budget_limited": isinstance(error, BudgetLimited), "no_retry_or_resurrection": True}
    error.v2_attempt = attempt
    try:
        write_json(output / "attempt.json", attempt, output=output)
    except BaseException as additional:
        preserve_error(error, additional, "v2 unqualified startup seal publication failed")
    return error


def _publication_check(output, execution, attempt, error, *, clock):
    """Never rewrite a cost snapshot; a failure marker overrides any apparent success."""
    pins, publication_error = {}, None
    try:
        for name in ("execution_attempt.json", "attempt.json"):
            pins[name] = sha256_file(output / name)
    except BaseException as exc:
        publication_error = exc
    checked = clock()  # Includes final receipt serialization and the last hash.
    late = checked >= execution["hard_deadline_perf_counter"]
    if late:
        budget = BudgetLimited("whole attempt final publication exceeded frozen hard deadline; acceptance refused")
        if publication_error is not None:
            preserve_error(budget, publication_error, "v2 final receipt hash failed")
        publication_error = budget
    if publication_error is None and error is None:
        return None
    if publication_error is not None:
        if error is None:
            error = publication_error
        elif late and not isinstance(error, BudgetLimited):
            preserve_error(publication_error, error, "original v2 attempt failure")
            error = publication_error
        else:
            preserve_error(error, publication_error, "v2 final publication failure")
    failure = {"status": "failed", "phase": "v2-final-publication", "finalized": False,
               "authoritative": True, "acceptance_refused": True, "no_retry_or_resurrection": True,
               "scientific_claim": False, "test_read": False, "limitations": list(execution["limitations"]),
               "protocol_sha256": execution["protocol_sha256"], "stage": execution["stage"],
               "hard_deadline_perf_counter": execution["hard_deadline_perf_counter"],
               "publication_checked_perf_counter": checked, "receipt_sha256": pins,
               "supersedes": ["attempt.json", "execution_attempt.json"],
               "failure_reason": f"{type(error).__name__}: {error}",
               "budget_limited": late or isinstance(error, BudgetLimited),
               "cleanup_or_publication_diagnostics": list(getattr(error, "__notes__", []))}
    execution.update(status="failed", finalized=False, budget_limited=failure["budget_limited"],
                     failure_reason=failure["failure_reason"])
    attempt.update(status="failed", finalized=False, budget_limited=failure["budget_limited"],
                   failure_reason=failure["failure_reason"], acceptance_refused=True,
                   authoritative_failure_reference=str(output / "publication_failure.json"))
    error.v2_execution_attempt, error.v2_attempt = dict(execution), dict(attempt)
    error.v2_publication_failure = failure
    try:
        write_json(output / "publication_failure.json", failure, output=output)
    except BaseException as additional:
        preserve_error(error, additional, "v2 authoritative publication failure marker failed")
    return error


def _publish_final(output, protocol, execution, outcome, error, *, clock):
    _costs(execution, protocol, clock=clock)
    if clock() >= execution["hard_deadline_perf_counter"]:
        error = error or BudgetLimited("whole attempt including aggregation and owned cleanup exceeded frozen hard deadline")
    if error is not None:
        execution.update(status="failed", finalized=False, failure_reason=f"{type(error).__name__}: {error}",
                         budget_limited=isinstance(error, BudgetLimited))
    execution["whole_clock_scope"] = protocol["whole_clock_scope"] + "; final receipt serialization follows snapshot"
    execution["cleanup_or_publication_diagnostics"] = list(getattr(error, "__notes__", []))
    try:
        write_json(output / "execution_attempt.json", execution, output=output)
    except BaseException as exc:
        if error is None:
            error = exc
            execution.update(status="failed", finalized=False, failure_reason=f"{type(exc).__name__}: {exc}")
        else:
            preserve_error(error, exc, "v2 final execution receipt publication failed")
    _costs(execution, protocol, clock=clock)
    if clock() >= execution["hard_deadline_perf_counter"]:
        error = error or BudgetLimited("whole attempt exceeded hard deadline before final seal")
        execution.update(status="failed", finalized=False, budget_limited=True, failure_reason=f"{type(error).__name__}: {error}")
    attempt = {**execution, "phase": "v2-whole-attempt", "outcome": outcome,
               "seal": "post-aggregation-and-owned-cleanup independent final cost receipt",
               "cleanup_or_publication_diagnostics": list(getattr(error, "__notes__", []))}
    try:
        write_json(output / "attempt.json", attempt, output=output)
    except BaseException as exc:
        if error is None:
            error = exc
            attempt.update(status="failed", finalized=False, failure_reason=f"{type(exc).__name__}: {exc}")
        else:
            preserve_error(error, exc, "v2 final attempt seal publication failed")
    return _publication_check(output, execution, attempt, error, clock=clock)


def run_bounded_round(output, *, snapshot_fn=None, popen_factory=None, clock=time.perf_counter, finalize_fn=None):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = safe_output(output)  # Unsafe/old/symlink paths never receive finally receipts.
    allowed = {"prepare_started.json", "prepare_attempt.json", "protocol.json", "cpu_profile.json", "code.zip", "environment.json",
               "code_commit.txt", "code_status.txt"}
    if not output.is_dir() or any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError("attempt already started/failed/partial; no retry or resurrection")
    # Claim with O_EXCL before trusting any prepared input. A losing claimant never seals.
    protocol, execution, error, outcome, run_entry = None, None, None, None, None
    claim = write_path(output / "run_started.json", output).open("x", encoding="utf-8")
    try:
        with claim:
            run_entry = clock()
            json.dump({"scientific_claim": False, "limitations": list(LIMITATIONS),
                       "protocol_identity_qualified": False, "started_perf_counter": run_entry}, claim,
                      ensure_ascii=False, indent=2, allow_nan=False)
            claim.write("\n")
        protocol = verify_protocol(output / "protocol.json")
        execution = _execution(protocol)
        execution["run_entry_perf_counter"] = run_entry
        check = lambda: check_budget(execution["hard_deadline_perf_counter"], clock=clock)
        prepared = read_json(output / "prepare_attempt.json")
        if prepared.get("status") != "prepared-not-run" or prepared.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("successful exact prepare receipt required before a first run")
        if protocol["monotonic_boot_id"] != monotonic_boot_id() or protocol["round_started_perf_counter"] > clock():
            raise ValueError("same-boot frozen prepare anchor required; no cross-boot or future-anchor resurrection")
        check()
        _runtime_preflight(protocol, check=check)
        make_directory(output / "workers", output)
        execute_jobs(output / "protocol.json", execution=execution, snapshot_fn=snapshot_fn,
                     popen_factory=popen_factory, clock=clock)
        _runtime_postflight(protocol, check=check)
        _costs(execution, protocol, clock=clock)
        if execution["jobs_completed"] != protocol["jobs"]:
            raise ValueError("full exact frozen train/evaluation inventory required for aggregation")
        execution.update(status="results-complete", finalized=False, evaluation_coverage_complete=True)
        execution["aggregation_started_perf_counter"] = clock()
        outcome = (finalize_results if finalize_fn is None else finalize_fn)(output, protocol, dict(execution))
        execution["aggregation_ended_perf_counter"] = clock()
        if (not isinstance(outcome, dict) or outcome.get("scientific_claim") is not False
                or not outcome.get("limitations") or type(outcome.get("paused", False)) is not bool):
            raise ValueError("explicit nonscientific descriptive aggregate outcome required")
        _runtime_postflight(protocol, check=check)
        check_budget(execution["hard_deadline_perf_counter"], clock=clock, reserve=0)
        execution.update(status="paused" if outcome.get("paused", False) else "success", finalized=True,
                         scientific_halt=outcome.get("paused", False))
    except BaseException as exc:
        error = exc
    finally:
        if execution is None:
            error = _seal_unqualified(output, error, run_entry, clock=clock)
        else:
            error = _publish_final(output, protocol, execution, outcome, error, clock=clock)
    if error is not None:
        raise error
    return outcome
