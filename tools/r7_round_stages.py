"""Per-stage driver wrapper for the campaign round (decisions 0044 / 0045).

This is the half of the round support that actually moves work: finding the attempt
directory, deciding whether a stage is already done, freezing a protocol through the
driver's own ``freeze``, spawning one driver worker per call, and the commit/push
primitives. The parsing and accounting half lives in ``tools/r7_round_support.py``,
which this module loads by path so ``RoundError`` is one shared class.

The stage wrapper exists because the driver has **no resume**: every artifact is created
exclusively and a finished stage re-run fails loudly. Skipping is therefore the
orchestrator's job, and it is what lets an interrupted round be continued.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

_SUPPORT_PATH = Path(__file__).resolve().parent / "r7_round_support.py"
_spec = importlib.util.spec_from_file_location("r7_round_support", _SUPPORT_PATH)
support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(support)

ROOT = support.ROOT
PYTHON = support.PYTHON
RoundError = support.RoundError
default_runner = support.default_runner
read_json = support.read_json
sha256_file = support.sha256_file
gpu_headroom = support.gpu_headroom
next_attempt_dir = support.next_attempt_dir
PROTECTED_BRANCHES = ("main", "master")

#: stage name -> the file whose presence proves the stage finished
STAGE_RECEIPTS = {
    "prepare": "prepared_protocol.json",
    "archive": "archive_receipt.json",
    "reading": "readings.json",
}
STAGE_ORDER = ("prepare", "archive", "seed", "reading")
DRIVER_REQUIRED_NAMES = ("freeze", "SEEDS", "PER_SEED_SECONDS", "PLANNED_SECONDS", "HARD_CAP_SECONDS")
WORKER_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV", "TMPDIR")


def current_attempt_dir(stem: str, outputs: Path | None = None) -> Path:
    """The newest attempt directory for this round; refuse if the round never froze."""
    root = Path(outputs or (ROOT / "outputs"))
    candidates = sorted(
        (path for path in root.glob(f"{stem}_attempt*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not candidates:
        raise RoundError(f"no attempt directory exists yet for stem {stem!r}")
    return candidates[-1]


def stage_receipt(output: Path, phase: str, seed: int | None = None) -> Path:
    """The file that proves this stage finished (decision 0044 layer 3)."""
    if phase == "seed":
        if seed is None:
            raise RoundError("the seed stage needs a seed value")
        return Path(output) / f"seed{seed}_receipt.json"
    if phase not in STAGE_RECEIPTS:
        raise RoundError(f"unknown stage {phase!r}; expected one of {STAGE_ORDER}")
    return Path(output) / STAGE_RECEIPTS[phase]


def stage_plan(output: Path, phase: str, seed: int | None = None) -> dict[str, Any]:
    """Decide whether a stage still has work to do.

    The driver itself never skips: every artifact is created exclusively, so re-running
    a finished stage fails loudly. This is the skip-if-done layer the orchestrator owes
    (decision 0044), and it is what lets a *fresh* run continue a half-done round.
    """
    receipt = stage_receipt(output, phase, seed)
    done = receipt.is_file()
    return {
        "phase": phase,
        "seed": seed,
        "receipt": str(receipt),
        "already_done": done,
        "receipt_sha256": sha256_file(receipt) if done else None,
        "skip": done,
    }


def load_driver(driver_rel: str, root: Path | None = None) -> Any:
    """Import a study driver by path and assert the interface this wrapper needs."""
    base = Path(root or ROOT)
    path = base / driver_rel
    if not path.is_file():
        raise RoundError(f"driver not found: {path}")
    spec = importlib.util.spec_from_file_location("r7_round_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in DRIVER_REQUIRED_NAMES if not hasattr(module, name)]
    if missing:
        raise RoundError(f"{driver_rel} does not expose {missing}; refusing to guess")
    return module


def freeze_round(
    spec: dict[str, Any],
    *,
    code_archive: str,
    code_commit: str,
    code_sha256: str,
    root: Path | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Create a fresh attempt directory and write ``preparation_protocol.json``.

    Delegates to the driver's own ``freeze`` so the protocol constants stay in one place.
    """
    base = Path(root or ROOT)
    now = clock or time.perf_counter
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit or ""):
        raise RoundError("an exact code commit (40 lowercase hex) is required")
    if not re.fullmatch(r"[0-9a-f]{64}", code_sha256 or ""):
        raise RoundError("an exact code SHA256 (64 lowercase hex) is required")
    driver = load_driver(spec["driver"], base)
    output = next_attempt_dir(spec["output_stem"], base / "outputs")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RoundError(f"attempt directory already exists: {output}") from exc
    started = now()
    deadline = started + float(driver.HARD_CAP_SECONDS)
    body = driver.freeze(
        output,
        started=started,
        deadline=deadline,
        code_archive=Path(code_archive).absolute(),
        code_commit=code_commit,
        code_sha256=code_sha256,
    )
    return {
        "attempt_dir": str(output),
        "protocol_sha256": body["protocol_sha256"],
        "deadline_perf_counter": deadline,
        "planned_seconds": float(driver.PLANNED_SECONDS),
        "hard_cap_seconds": float(driver.HARD_CAP_SECONDS),
        "per_seed_seconds": float(driver.PER_SEED_SECONDS),
        "seeds": list(driver.SEEDS),
        "passed": True,
    }


def worker_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """A small, explicit environment: the worker pins its own GPU by declared UUID."""
    return {key: value for key, value in (source or os.environ).items() if key in WORKER_ENV_KEYS}


def run_stage(
    output: Path,
    driver_rel: str,
    phase: str,
    *,
    seed: int | None = None,
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
    protocol: dict[str, Any] | None = None,
    clock: Callable[[], float] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run exactly one driver stage, skipping it when its receipt already exists.

    The deadline is taken from the frozen ``protocol.json``, never invented here: for a
    seed it is ``min(round_deadline, now + per_seed_seconds)``, which is what the worker
    validates. GPU stages are gated on a read-only headroom probe first.
    """
    base = Path(root or ROOT)
    now = clock or time.perf_counter
    output = Path(output)
    plan = stage_plan(output, phase, seed)
    if plan["skip"]:
        return {**plan, "status": "skipped", "passed": True}
    body = protocol if protocol is not None else read_json(output / "protocol.json")
    deadline = float(body["deadline_perf_counter"])
    per_seed = float(body["per_seed_seconds"])
    limit = min(deadline, now() + per_seed) if phase == "seed" else deadline
    if limit <= now():
        raise RoundError(f"the round's hard deadline has already passed; refusing to spawn {phase}")
    gate = None
    if phase in {"seed", "reading"}:
        gate = gpu_headroom(
            body["gpu_uuid"],
            estimated_peak_bytes=int(body["estimated_peak_bytes"]),
            owned_reserved_peak_bytes=int(body.get("owned_reserved_peak_bytes", 0)),
            margin_bytes=int(body["headroom_margin_bytes"]),
            runner=runner,
        )
        if not gate["passed"]:
            return {**plan, "status": "headroom-blocked", "passed": False, "gate": gate}
    argv = [PYTHON, str(base / driver_rel), "--worker", phase, "--out", str(output), "--deadline", str(limit)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    run = runner or default_runner
    started = now()
    result = run(argv, base, worker_environment(env), max(1.0, limit - started) + 60.0)
    elapsed = now() - started
    tag = phase if seed is None else f"seed{seed}"
    record = {
        "phase": phase,
        "seed": seed,
        "exit_code": result["exit_code"],
        "status": "success" if result["exit_code"] == 0 else "failed",
        "passed": result["exit_code"] == 0,
        "elapsed_seconds": elapsed,
        "soft_overrun_seconds": max(0.0, elapsed - float(body["planned_seconds"])),
        "hard_overrun_seconds": max(0.0, elapsed - float(body["hard_cap_seconds"])),
        "gate": gate,
        "receipt": str(stage_receipt(output, phase, seed)),
        "stdout_tail": result["stdout"].strip().splitlines()[-5:],
        "stderr_tail": result["stderr"].strip().splitlines()[-5:],
    }
    log_path = output / f"{tag}.log"
    if not log_path.exists():
        log_path.write_text(result["stdout"] + result["stderr"], encoding="utf-8")
    process_path = output / f"{tag}_process.json"
    if not process_path.exists():
        process_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def git_branch(root: Path | None = None, runner: Callable[..., dict[str, Any]] | None = None) -> str:
    base = Path(root or ROOT)
    run = runner or default_runner
    result = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], base)
    if result["exit_code"] != 0:
        raise RoundError(f"cannot read the branch: {result['stderr'].strip()}")
    return result["stdout"].strip()


def commit_round(
    spec: dict[str, Any],
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Commit exactly the round-spec's declared paths, never on a protected branch.

    The workflow cannot pass paths on the command line, so the paths and the message come
    from the round-spec. A protected branch is refused outright: pushing to ``main`` and
    merging are reserved (decision 0030).
    """
    base = Path(root or ROOT)
    run = runner or default_runner
    branch = git_branch(base, run)
    if branch in PROTECTED_BRANCHES:
        raise RoundError(f"refusing to commit on protected branch {branch!r}")
    paths = list(spec.get("commit_paths") or [])
    message = spec.get("commit_message")
    if not paths:
        raise RoundError("round-spec needs commit_paths before committing")
    if not isinstance(message, str) or not message.strip():
        raise RoundError("round-spec needs commit_message before committing")
    staged = run(["git", "add", "--", *paths], base)
    if staged["exit_code"] != 0:
        return {"passed": False, "stage": "add", "tail": staged["stderr"].strip().splitlines()[-3:]}
    committed = run(["git", "commit", "-m", message, "--", *paths], base)
    if committed["exit_code"] != 0:
        return {"passed": False, "stage": "commit", "tail": committed["stderr"].strip().splitlines()[-3:]}
    sha = run(["git", "rev-parse", "HEAD"], base)["stdout"].strip()
    return {"passed": True, "branch": branch, "sha": sha, "paths": paths, "message": message}


def push_round(
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Push the current branch. Plain ``git push``: no refspec to spell, no force."""
    base = Path(root or ROOT)
    run = runner or default_runner
    branch = git_branch(base, run)
    if branch in PROTECTED_BRANCHES:
        raise RoundError(f"refusing to push protected branch {branch!r}")
    result = run(["git", "push"], base, None, 600.0)
    return {"passed": result["exit_code"] == 0, "branch": branch,
            "tail": (result["stdout"] + result["stderr"]).strip().splitlines()[-5:]}
