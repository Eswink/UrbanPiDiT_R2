"""Wall-clock limits for directly owned, non-spawning study workers.

Only the Popen created here may receive a signal. Workers must not start their
own background work: terminating the direct child is not a process-tree guard.
"""
from __future__ import annotations

import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def bounded_process(arguments, *, cwd, log_path, deadline, env=None, grace_seconds=5.0):
    if (not isinstance(arguments, (list, tuple)) or not arguments
            or any(not isinstance(value, str) or not value for value in arguments)):
        raise ValueError("arguments must be a nonempty list of nonempty strings")
    for value, name in ((deadline, "deadline"), (grace_seconds, "grace_seconds")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if grace_seconds <= 0:
        raise ValueError("grace_seconds must be positive")
    started = time.perf_counter()
    receipt = {"status": "not-started", "pid": None, "returncode": None,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "started_perf_counter": started, "deadline_perf_counter": deadline,
               "signals": [], "log_path": str(log_path)}
    if started >= deadline:
        receipt.update(status="deadline-before-spawn", elapsed_seconds=0.0,
                       ended_perf_counter=started, reaped=True, hard_overrun_seconds=0.0)
        return receipt
    process = None
    with Path(log_path).open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(list(arguments), cwd=cwd, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, close_fds=True)
            receipt["pid"] = process.pid
            try:
                process.wait(timeout=max(0.0, deadline - time.perf_counter()))
                receipt["status"] = "success" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                receipt["status"] = "deadline-exceeded"
                _reap_owned(process, receipt, grace_seconds)
        except BaseException:
            if process is not None and process.poll() is None:
                _reap_owned(process, receipt, grace_seconds)
            raise
        finally:
            ended = time.perf_counter()
            receipt.update(returncode=None if process is None else process.poll(),
                           elapsed_seconds=ended - started, ended_perf_counter=ended,
                           hard_overrun_seconds=max(0.0, ended - deadline),
                           reaped=process is None or process.poll() is not None,
                           ended_utc=datetime.now(timezone.utc).isoformat())
    return receipt


def _reap_owned(process, receipt, grace_seconds):
    for action in ("terminate", "kill"):
        if process.poll() is not None:
            break
        try:
            getattr(process, action)()
            receipt["signals"].append({"action": action, "pid": process.pid,
                                       "at_perf_counter": time.perf_counter()})
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        raise RuntimeError(f"owned worker {process.pid} could not be reaped; no further workers permitted")


def worker_environment(gpu_uuid=None):
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    environment["OPENBLAS_NUM_THREADS"] = "4"
    if gpu_uuid is not None:
        if environment.get("CUDA_VISIBLE_DEVICES") not in (None, "", gpu_uuid):
            raise ValueError("CUDA_VISIBLE_DEVICES conflicts with the declared GPU UUID")
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return environment
