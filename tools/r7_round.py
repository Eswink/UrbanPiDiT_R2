"""Thin CLI over ``tools/r7_round_support.py`` for the campaign round driver.

Every subcommand prints one JSON document on stdout. **No subcommand takes a runtime
value on the command line**: a workflow can only pass compile-time literals, so the
current round is resolved from ``campaign-state.current_round_goal`` and every other
value comes from the round-spec block or the frozen ``protocol.json``.

    tools/r7_round.py state
    tools/r7_round.py verify
    tools/r7_round.py round-guard
    tools/r7_round.py attempt
    tools/r7_round.py stage-plan <phase> [<seed>]
    tools/r7_round.py orphans
    tools/r7_round.py freeze
    tools/r7_round.py stage <phase> [<seed>]
    tools/r7_round.py reuse-plan <previous_attempt_dir>
    tools/r7_round.py ledger-plan <new_gpu_hours>
    tools/r7_round.py commit-guard <path> [<path> ...]
    tools/r7_round.py commit-round
    tools/r7_round.py push-round

Exit codes: 0 on success, 1 when a gate reports not-passed, 2 on a usage or precondition
error.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS = Path(__file__).resolve().parent
USAGE = ("usage: r7_round.py {state|verify|round-guard|attempt|stage-plan|orphans|freeze|stage|"
         "reuse-plan|ledger-plan|commit-guard|commit-round|push-round}")


def load_support() -> Any:
    """Load the sibling support module by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("r7_round_support", TOOLS / "r7_round_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stages() -> Any:
    """Load the per-stage wrapper, which itself loads the support module by path."""
    spec = importlib.util.spec_from_file_location("r7_round_stages", TOOLS / "r7_round_stages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_liveness() -> Any:
    """Load the read-only stage-liveness module by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("r7_round_liveness", TOOLS / "r7_round_liveness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = load_support()
stages = load_stages()
liveness = load_liveness()


def current_goal() -> Path:
    """The round the campaign is on, per ``campaign-state`` — never from argv."""
    return support.ROOT / support.campaign_state()["current_round_goal"]


def current_spec() -> dict[str, Any]:
    return support.round_spec(current_goal())


def current_attempt(spec: dict[str, Any] | None = None) -> Path:
    body = spec if spec is not None else current_spec()
    return stages.current_attempt_dir(body["output_stem"])


def _seed(rest: Sequence[str], offset: int = 1) -> int | None:
    return int(rest[offset]) if len(rest) > offset else None


def round_guard(rest: Sequence[str]) -> dict[str, Any]:
    """``round-guard`` — the round's own deterministic gate: new tests plus commit content.

    This is what stands in for the hooks a workflow's subagents probably do not run.
    """
    if rest:
        raise support.RoundError("round-guard takes no arguments")
    spec = current_spec()
    tests = support.run_pytest(spec.get("test_path"), root=support.ROOT)
    guard = support.commit_guard(spec.get("commit_paths") or [], root=support.ROOT)
    return {"passed": bool(tests["passed"] and guard["passed"]), "tests": tests, "commit_guard": guard}


def attempt(rest: Sequence[str]) -> dict[str, Any]:
    if rest:
        raise support.RoundError("attempt takes no arguments")
    return {"attempt_dir": str(current_attempt()), "goal": str(current_goal()), "passed": True}


def stage_plan(rest: Sequence[str]) -> dict[str, Any]:
    if len(rest) not in {1, 2}:
        raise support.RoundError("stage-plan needs <phase> [<seed>]")
    spec = current_spec()
    return stages.stage_plan(current_attempt(spec), rest[0], _seed(rest), driver_rel=spec["driver"])


def orphans(rest: Sequence[str]) -> dict[str, Any]:
    """``orphans`` — read-only: which stages still have a live worker.

    The answer to "may I re-run this stage, or is an orphan still training?" after an
    interruption. Never signals anything.
    """
    if rest:
        raise support.RoundError("orphans takes no arguments")
    spec = current_spec()
    try:
        attempt = current_attempt(spec)
    except support.RoundError as exc:
        return {"attempt_dir": None, "stages": [], "running": [], "reason": str(exc)}
    return liveness.stage_liveness(spec, attempt)


def freeze(rest: Sequence[str]) -> dict[str, Any]:
    """``freeze`` — fresh attempt directory plus the preparation protocol.

    The code archive path comes from the round-spec; the commit and its SHA256 are read
    here, so argv stays literal.
    """
    if rest:
        raise support.RoundError("freeze takes no arguments")
    spec = current_spec()
    if "code_archive" not in spec:
        raise support.RoundError("round-spec needs code_archive before freezing")
    archive = support.ROOT / spec["code_archive"]
    if not archive.is_file():
        raise support.RoundError(f"code archive is missing: {archive}")
    commit = support.default_runner(["git", "rev-parse", "HEAD"], support.ROOT)["stdout"].strip()
    return stages.freeze_round(spec, code_archive=str(archive), code_commit=commit,
                               code_sha256=support.sha256_file(archive), root=support.ROOT)


def stage(rest: Sequence[str]) -> dict[str, Any]:
    """``stage <phase> [<seed>]`` — one driver stage, skipped when its receipt exists."""
    if len(rest) not in {1, 2}:
        raise support.RoundError("stage needs <phase> [<seed>]")
    spec = current_spec()
    return stages.run_stage(current_attempt(spec), spec["driver"], rest[0], seed=_seed(rest),
                            root=support.ROOT)


def reuse_plan(rest: Sequence[str]) -> dict[str, Any]:
    """``reuse-plan <previous_attempt_dir>`` — which seeds may be reused (decision 0045)."""
    if len(rest) != 1:
        raise support.RoundError("reuse-plan needs <previous_attempt_dir>")
    previous = Path(rest[0])
    new_protocol_path = current_attempt() / "protocol.json"
    new_protocol = support.read_json(new_protocol_path)
    previous_protocol_path = previous / "protocol.json"
    has_previous = previous_protocol_path.is_file()
    previous_protocol = support.read_json(previous_protocol_path) if has_previous else None
    receipts = {}
    for seed in new_protocol.get("seeds", ()):
        receipt_path = previous / f"seed{seed}_receipt.json"
        if receipt_path.is_file():
            receipts[str(seed)] = support.read_json(receipt_path)
    decision = support.reuse_decision(previous_protocol, new_protocol, receipts)
    decision["previous_attempt"] = str(previous)
    decision["previous_protocol_file_sha256"] = support.sha256_file(previous_protocol_path) if has_previous else None
    decision["passed"] = True
    return decision


def ledger_plan(rest: Sequence[str]) -> dict[str, Any]:
    """``ledger-plan <new_gpu_hours>`` — the arithmetic the new row must satisfy."""
    if len(rest) != 1:
        raise support.RoundError("ledger-plan needs <new_gpu_hours>")
    state = support.campaign_state()
    return support.ledger_plan(support.ledger_rows(), float(state["cap_gpu_h"]), float(rest[0]))


def dispatch(command: str, rest: Sequence[str]) -> dict[str, Any]:
    handlers = {
        "state": lambda: support.state_report(),
        "verify": lambda: support.run_validators(),
        "round-guard": lambda: round_guard(rest),
        "attempt": lambda: attempt(rest),
        "stage-plan": lambda: stage_plan(rest),
        "orphans": lambda: orphans(rest),
        "freeze": lambda: freeze(rest),
        "stage": lambda: stage(rest),
        "reuse-plan": lambda: reuse_plan(rest),
        "ledger-plan": lambda: ledger_plan(rest),
        "commit-guard": lambda: support.commit_guard(rest),
        "commit-round": lambda: stages.commit_round(current_spec()),
        "push-round": lambda: stages.push_round(),
    }
    if command not in handlers:
        raise support.RoundError(f"unknown subcommand {command!r}")
    return handlers[command]()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(json.dumps({"error": USAGE}))
        return 2
    try:
        payload = dispatch(arguments[0], arguments[1:])
    except support.RoundError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
