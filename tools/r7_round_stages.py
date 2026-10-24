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

import concurrent.futures
from datetime import datetime, timezone
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

_LIVENESS_PATH = Path(__file__).resolve().parent / "r7_round_liveness.py"
_liveness_spec = importlib.util.spec_from_file_location("r7_round_liveness", _LIVENESS_PATH)
liveness = importlib.util.module_from_spec(_liveness_spec)
_liveness_spec.loader.exec_module(liveness)

_DEVICES_PATH = Path(__file__).resolve().parents[1] / "training" / "r7_device_selection.py"
_devices_spec = importlib.util.spec_from_file_location("r7_device_selection", _DEVICES_PATH)
devices = importlib.util.module_from_spec(_devices_spec)
_devices_spec.loader.exec_module(devices)

ROOT = support.ROOT
PYTHON = support.PYTHON
RoundError = support.RoundError
default_runner = support.default_runner
read_json = support.read_json
sha256_file = support.sha256_file
gpu_headroom = support.gpu_headroom
next_attempt_dir = support.next_attempt_dir
running_worker_pids = liveness.running_worker_pids
measured_peak_bytes = liveness.measured_peak_bytes
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

#: The driver's own orchestrator (``run_attempt``) promotes the prepare worker's
#: ``prepared_protocol.json`` to ``protocol.json`` before archive/seed/reading. Run one
#: stage at a time, this wrapper owes that promotion, or every later stage reads a file
#: that was never written. The prepare stage's own frozen body is the one ``freeze``
#: wrote (``preparation_protocol.json``), since ``protocol.json`` does not exist yet.
FROZEN_PROTOCOL = "protocol.json"
PREPARATION_PROTOCOL = "preparation_protocol.json"
PREPARED_PROTOCOL = "prepared_protocol.json"
STAGE_LEDGER = "stage_ledger.jsonl"


def _append_stage_ledger(output: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append one line of stage history to ``<attempt>/stage_ledger.jsonl``.

    The driver's own ``run_attempt`` printed a ``phase_done`` line to its log, but the
    per-stage path never runs that loop, so an attempt driven by ``r7_round.py stage``
    had no stage ledger at all. This restores it as an append-only sidecar: a resumed
    attempt keeps every earlier line, and the file lives under ``tools/`` so it costs no
    execution identity. It records what the orchestrator did, including skips and
    refusals, not just the stages that ran.
    """
    body = {"ledger": "r7-stage-v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with (Path(output) / STAGE_LEDGER).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


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


def stage_plan(
    output: Path,
    phase: str,
    seed: int | None = None,
    *,
    driver_rel: str | None = None,
) -> dict[str, Any]:
    """Decide whether a stage still has work to do.

    The driver itself never skips: every artifact is created exclusively, so re-running
    a finished stage fails loudly. This is the skip-if-done layer the orchestrator owes
    (decision 0044), and it is what lets a *fresh* run continue a half-done round.

    A missing receipt is not enough to spawn: an interrupted orchestrator can leave its
    worker running, so the plan also reports whether one is still alive. The liveness
    scan needs the driver path; without it (``driver_rel is None``) the plan reports
    only the receipt half and never claims the stage is idle.
    """
    receipt = stage_receipt(output, phase, seed)
    done = receipt.is_file()
    live = []
    if not done and driver_rel:
        live = running_worker_pids(driver_rel, phase, output, seed)
    return {
        "phase": phase,
        "seed": seed,
        "receipt": str(receipt),
        "already_done": done,
        "receipt_sha256": sha256_file(receipt) if done else None,
        "already_running": bool(live),
        "running_pids": [item["pid"] for item in live],
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


def worker_environment(source: dict[str, str] | None = None, device: str | None = None) -> dict[str, str]:
    """A small, explicit environment: the worker pins its own GPU by declared UUID.

    ``device`` (decision 0047) pins the worker to the card this stage was assigned. It is
    set explicitly rather than allowed through from the ambient environment, and the
    driver re-validates it against the frozen device list instead of trusting it.
    """
    environment = {key: value for key, value in (source or os.environ).items() if key in WORKER_ENV_KEYS}
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def promote_prepared_protocol(output: Path) -> bool:
    """Materialise ``protocol.json`` from the prepare worker's ``prepared_protocol.json``.

    The driver's own ``run_attempt`` does this between prepare and archive; a per-stage
    orchestrator must do it too. Idempotent: a no-op once ``protocol.json`` exists, and a
    no-op before prepare has written ``prepared_protocol.json``.
    """
    output = Path(output)
    prepared = output / PREPARED_PROTOCOL
    target = output / FROZEN_PROTOCOL
    if not prepared.is_file() or target.is_file():
        return False
    with target.open("x", encoding="utf-8") as handle:
        handle.write(prepared.read_text(encoding="utf-8"))
    return True


def stage_protocol(output: Path, phase: str) -> dict[str, Any]:
    """The frozen body that governs this stage's deadline and GPU gate.

    Archive/seed/reading read ``protocol.json`` (promoted after prepare). The prepare
    stage itself reads the body ``freeze`` wrote, since ``protocol.json`` is not written
    until prepare returns.
    """
    output = Path(output)
    if (output / FROZEN_PROTOCOL).is_file():
        return read_json(output / FROZEN_PROTOCOL)
    if phase == "prepare" and (output / PREPARATION_PROTOCOL).is_file():
        return read_json(output / PREPARATION_PROTOCOL)
    return read_json(output / FROZEN_PROTOCOL)


def run_stage(
    output: Path,
    driver_rel: str,
    phase: str,
    *,
    seed: int | None = None,
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
    clock: Callable[[], float] | None = None,
    device: str | None = None,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run exactly one driver stage, skipping it when its receipt already exists.

    The deadline is taken from the frozen ``protocol.json``, never invented here: for a
    seed it is ``min(round_deadline, now + per_seed_seconds)``, which is what the worker
    validates. GPU stages are gated on a read-only headroom probe first.

    A stage with no receipt but a **live worker** is refused, not re-run: after an
    interruption the orphan keeps training, and a second worker on the same attempt
    directory would fight it for the GPU and the output tree. Nothing here signals the
    orphan — the caller waits for it, or the operator decides.
    """
    base = Path(root or ROOT)
    now = clock or time.perf_counter
    output = Path(output)
    plan = stage_plan(output, phase, seed, driver_rel=driver_rel)
    if plan["skip"]:
        return _append_stage_ledger(output, {**plan, "status": "skipped", "passed": True})
    if plan["already_running"]:
        return _append_stage_ledger(output, {**plan, "status": "already-running", "passed": False,
                "reason": "a worker for this stage is still running; wait for its receipt"})
    promote_prepared_protocol(output)
    body = stage_protocol(output, phase)
    deadline = float(body["deadline_perf_counter"])
    per_seed = float(body["per_seed_seconds"])
    limit = min(deadline, now() + per_seed) if phase == "seed" else deadline
    if limit <= now():
        raise RoundError(f"the round's hard deadline has already passed; refusing to spawn {phase}")
    observed_gate = gate
    if phase in {"seed", "reading"}:
        if observed_gate is None:
            # The protocol's ``estimated_peak_bytes`` is a freeze-time guess; a finished
            # seed's receipt carries what the run actually reserved. Gate on the larger of
            # the two, so an already-measured footprint cannot be under-declared.
            owned_peak = max(int(body.get("owned_reserved_peak_bytes", 0)), measured_peak_bytes(output))
            if device is None:
                observed_gate = gpu_headroom(
                    body["gpu_uuid"],
                    estimated_peak_bytes=int(body["estimated_peak_bytes"]),
                    owned_reserved_peak_bytes=owned_peak,
                    margin_bytes=int(body["headroom_margin_bytes"]),
                    runner=runner,
                )
            else:
                # A protocol that names its cards takes the multi-card path (decision 0047):
                # the parent already picked this stage's card, so only that one is probed, and
                # the declared neighbour reserve is added on top of footprint and margin.
                observed_gate = devices.gate_candidates(
                    [device],
                    estimated_peak_bytes=int(body["estimated_peak_bytes"]),
                    owned_reserved_peak_bytes=owned_peak,
                    margin_bytes=int(body["headroom_margin_bytes"]),
                    reserve_bytes=int(body.get("neighbour_reserve_bytes", 0)),
                    runner=runner,
                )
        if not observed_gate["passed"]:
            return _append_stage_ledger(output, {**plan, "status": "headroom-blocked", "passed": False,
                                                "gate": observed_gate})
        # A scheduler that probes several cards hands back the card that cleared the gate.
        if device is not None and observed_gate.get("uuid"):
            device = observed_gate["uuid"]
    argv = [PYTHON, str(base / driver_rel), "--worker", phase, "--out", str(output), "--deadline", str(limit)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    run = runner or default_runner
    started = now()
    result = run(argv, base, worker_environment(device=device), max(1.0, limit - started) + 60.0)
    elapsed = now() - started
    if phase == "prepare" and result["exit_code"] == 0:
        promote_prepared_protocol(output)
    tag = phase if seed is None else f"seed{seed}"
    record = {
        "phase": phase,
        "seed": seed,
        "device": device,
        "exit_code": result["exit_code"],
        "status": "success" if result["exit_code"] == 0 else "failed",
        "passed": result["exit_code"] == 0,
        "elapsed_seconds": elapsed,
        "soft_overrun_seconds": max(0.0, elapsed - float(body["planned_seconds"])),
        "hard_overrun_seconds": max(0.0, elapsed - float(body["hard_cap_seconds"])),
        "gate": observed_gate,
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
    return _append_stage_ledger(output, record)


def run_seeds(
    output: Path,
    spec: dict[str, Any],
    *,
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
    clock: Callable[[], float] | None = None,
    wait_seconds: float = 3600.0,
    poll_seconds: float = 60.0,
) -> dict[str, Any]:
    """Run the round's remaining seeds with **at most one worker per frozen card**.

    A protocol that names a single card degenerates to the sequential order the round
    used before: one worker at a time, seeds in the spec's order. With two cards the
    second seed starts immediately and a third starts as soon as a card frees, so no card
    is ever shared by two workers of this round (decision 0047). The card a seed prefers
    is a rotation over the frozen list, which keeps three seeds from piling onto card 0.

    ``headroom-blocked`` is waited out read-only, exactly as the workflow's stage loop
    does, and never by signalling a neighbour (R-054). A genuine stage failure stops the
    round from starting further seeds; seeds already running are left to finish and are
    recorded either way.
    """
    output = Path(output)
    now = clock or time.perf_counter
    body = stage_protocol(output, "seed")
    uuids = devices.frozen_uuids(body)
    if not uuids:
        raise RoundError("the frozen protocol declares no device")
    driver = spec["driver"]
    order = {seed: index for index, seed in enumerate(spec["seeds"])}
    pending = [seed for seed in spec["seeds"] if not stage_receipt(output, "seed", seed).is_file()]
    deadlines = {seed: now() + wait_seconds for seed in pending}
    free, records, stop = list(uuids), [], False
    owned_peak = max(int(body.get("owned_reserved_peak_bytes", 0)), measured_peak_bytes(output))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(uuids)) as pool:
        running: dict[Any, tuple[int, str]] = {}
        while (pending and not stop) or running:
            while pending and free and not stop:
                seed = pending.pop(0)
                candidates = [card for card in devices.seed_order(uuids, order[seed]) if card in free]
                # Probe every free card in the seed's preference order before spawning: a
                # blocked first choice must not pin the seed to that card when another free
                # card clears the same frozen gate. The gate is still the protocol's.
                gate = devices.gate_candidates(
                    candidates,
                    estimated_peak_bytes=int(body["estimated_peak_bytes"]),
                    owned_reserved_peak_bytes=owned_peak,
                    margin_bytes=int(body["headroom_margin_bytes"]),
                    reserve_bytes=int(body.get("neighbour_reserve_bytes", 0)),
                    runner=runner,
                )
                device = gate["uuid"] if gate["passed"] else candidates[0]
                free.remove(device)
                future = pool.submit(run_stage, output, driver, "seed", seed=seed, root=root,
                                     runner=runner, clock=clock, device=device, gate=gate)
                running[future] = (seed, device)
            if not running:
                break
            done, _ = concurrent.futures.wait(list(running),
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            retry = False
            for future in done:
                seed, device = running.pop(future)
                free.append(device)
                record = future.result()
                records.append(record)
                if record["status"] == "headroom-blocked" and now() < deadlines[seed]:
                    pending.insert(0, seed)
                    retry = True
                elif not record["passed"]:
                    stop = True
            if retry and not stop and pending:
                time.sleep(max(0.0, min(poll_seconds, deadlines[pending[0]] - now())))
    # A seed that was blocked and then waited out leaves two records: score each seed by its
    # last one, so a retried seed is not reported as blocked forever. The earlier observation
    # stays in ``seeds`` as the audit trail.
    final = {record["seed"]: record for record in records}
    blocked = [seed for seed, record in final.items() if record["status"] == "headroom-blocked"]
    return {
        "mode": "multi-card" if len(uuids) > 1 else "single-card",
        "devices": uuids,
        "seeds": records,
        "headroom_blocked": blocked,
        "stopped": stop,
        "passed": bool(records) and not stop and not blocked,
    }


def git_branch(root: Path | None = None, runner: Callable[..., dict[str, Any]] | None = None) -> str:
    base = Path(root or ROOT)
    run = runner or default_runner
    result = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], base)
    if result["exit_code"] != 0:
        raise RoundError(f"cannot read the branch: {result['stderr'].strip()}")
    return result["stdout"].strip()


def _is_nothing_to_commit(result: dict[str, Any]) -> bool:
    """True when ``git commit`` failed only because the declared paths had no changes.

    A resumed round re-runs its implementer against files that are already committed, so
    ``git commit -- <paths>`` exits nonzero with "nothing to commit". That is not a failure
    of the work, and stopping the run on it would make an interrupted round unresumable.
    Any other failure — a hook rejection, a lock, a bad identity — still fails.
    """
    text = ((result.get("stdout") or "") + "\n" + (result.get("stderr") or "")).lower()
    return any(phrase in text for phrase in (
        "nothing to commit", "no changes added to commit", "nothing added to commit"))


def _commit_declared(
    spec: dict[str, Any],
    *,
    phase: str,
    paths_key: str,
    message_key: str,
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Commit exactly the round-spec's declared paths, never on a protected branch.

    The workflow cannot pass paths on the command line, so the paths and the message come
    from the round-spec. A protected branch is refused outright: pushing to ``main`` and
    merging are reserved (decision 0030). ``phase`` names which of the round's two commits
    this is, so a caller's log says which one failed.
    """
    base = Path(root or ROOT)
    run = runner or default_runner
    branch = git_branch(base, run)
    if branch in PROTECTED_BRANCHES:
        raise RoundError(f"refusing to commit on protected branch {branch!r}")
    paths = list(spec.get(paths_key) or [])
    message = spec.get(message_key)
    if not paths:
        raise RoundError(f"round-spec needs {paths_key} before committing")
    if not isinstance(message, str) or not message.strip():
        raise RoundError(f"round-spec needs {message_key} before committing")
    staged = run(["git", "add", "--", *paths], base)
    if staged["exit_code"] != 0:
        return {"passed": False, "phase": phase, "stage": "add",
                "tail": staged["stderr"].strip().splitlines()[-3:]}
    committed = run(["git", "commit", "-m", message, "--", *paths], base)
    if committed["exit_code"] != 0:
        if _is_nothing_to_commit(committed):
            return {"passed": True, "phase": phase, "branch": branch, "sha": "",
                    "paths": paths, "message": message, "noop": True,
                    "tail": (committed.get("stdout") or "").strip().splitlines()[-3:]}
        return {"passed": False, "phase": phase, "stage": "commit",
                "tail": committed["stderr"].strip().splitlines()[-3:]}
    sha = run(["git", "rev-parse", "HEAD"], base)["stdout"].strip()
    return {"passed": True, "phase": phase, "branch": branch, "sha": sha,
            "paths": paths, "message": message}


def commit_round(
    spec: dict[str, Any],
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Commit the round's implementation paths (``commit_paths``) before the GPU round."""
    return _commit_declared(spec, phase="implementation", paths_key="commit_paths",
                            message_key="commit_message", root=root, runner=runner)


def commit_register(
    spec: dict[str, Any],
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Commit the round's registration paths after the reading stage.

    Reads ``register_paths`` / ``register_message`` when the round-spec declares them and
    falls back to ``commit_paths`` / ``commit_message`` otherwise, so a brief written before
    this split keeps working. The two lists must be separate because the registration
    documents — the evidence page above all — do not exist yet when the implementation
    commit runs, and staging a path git cannot find fails the whole commit.
    """
    paths_key = "register_paths" if spec.get("register_paths") else "commit_paths"
    message_key = "register_message" if spec.get("register_message") else "commit_message"
    return _commit_declared(spec, phase="registration", paths_key=paths_key,
                            message_key=message_key, root=root, runner=runner)


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
