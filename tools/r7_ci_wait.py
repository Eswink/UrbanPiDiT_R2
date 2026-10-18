"""Wait for the GitHub Actions run of a commit, without blocking the round forever.

The round driver needs to know whether the exact commit it just pushed is green before
it freezes and trains. This tool polls the public Actions API anonymously through
``curl`` (no Python network client is imported, so R-016 and R-025 stay untouched) and
exits 0 only for a run that is ``completed`` with conclusion ``success``.

Everything a caller needs is on the command line, so a workflow can invoke it with a
fixed argv: ``tools/r7_ci_wait.py --ref HEAD``.

Exit codes: 0 green, 1 not green within the budget, 2 usage or transport error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

DEFAULT_REPO = "Eswink/UrbanPiDiT_R2"
DEFAULT_BRANCH = "r7/weather-reasoning"
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_INTERVAL_SECONDS = 60.0
CURL_TIMEOUT_SECONDS = 30
RUNS_PER_PAGE = 20
GREEN = "success"
# A push creates one run per workflow. The 17 label-gated experiment workflows are
# skipped by design and their run ids can be newer than the main CPU run, so "newest
# run for this SHA" is not the round's CI. Pick the CPU workflow by name when it is
# present; fall back to the newest matching run when it is not (e.g. in unit tests).
DEFAULT_WORKFLOW_NAME = "R7 CPU CI"


class CiWaitError(RuntimeError):
    """The status could not be determined; the caller must not read this as green."""


def default_runner(argv: Sequence[str], cwd: str | None = None) -> dict[str, Any]:
    completed = subprocess.run(  # fixed argv, no shell
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=CURL_TIMEOUT_SECONDS,
    )
    return {"exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def resolve_sha(ref: str, runner: Callable[..., dict[str, Any]] | None = None) -> str:
    run = runner or default_runner
    result = run(["git", "rev-parse", ref])
    if result["exit_code"] != 0:
        raise CiWaitError(f"cannot resolve {ref!r}: {result['stderr'].strip()}")
    return result["stdout"].strip()


def runs_url(repo: str, branch: str) -> str:
    return f"https://api.github.com/repos/{repo}/actions/runs?branch={branch}&per_page={RUNS_PER_PAGE}"


def fetch_runs(repo: str, branch: str, runner: Callable[..., dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    run = runner or default_runner
    result = run(["curl", "-s", runs_url(repo, branch)])
    if result["exit_code"] != 0:
        raise CiWaitError(f"curl failed: {result['stderr'].strip()}")
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        raise CiWaitError(f"the Actions API did not return JSON: {exc}") from exc
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise CiWaitError(f"unexpected Actions payload keys: {sorted(payload)[:6]}")
    return runs


def pick_run(runs: Sequence[dict[str, Any]], sha: str,
             workflow_name: str = DEFAULT_WORKFLOW_NAME) -> dict[str, Any] | None:
    """The round's run for this exact commit, or ``None`` if it has not appeared.

    Prefer the named CPU workflow: a push also creates the 17 label-gated experiment
    workflows, which are ``skipped`` by design and can carry the higher run id. Without
    the preference, a skipped experiment run would be classified as the commit's CI and
    every round would stop "not-green" after the real CPU run had passed.
    """
    matching = [item for item in runs if item.get("head_sha") == sha]
    if not matching:
        return None
    preferred = [item for item in matching if item.get("name") == workflow_name]
    return max(preferred or matching, key=lambda item: int(item.get("id", 0)))


def verdict(run: dict[str, Any] | None, sha: str) -> dict[str, Any]:
    """Classify a run. Only ``completed`` + ``success`` counts as passing."""
    if run is None:
        return {"state": "absent", "green": False, "sha": sha}
    status, conclusion = run.get("status"), run.get("conclusion")
    green = status == "completed" and conclusion == GREEN
    return {
        "state": "green" if green else ("pending" if status != "completed" else "not-green"),
        "green": green,
        "sha": sha,
        "run_id": run.get("id"),
        "name": run.get("name"),
        "status": status,
        "conclusion": conclusion,
        "html_url": run.get("html_url"),
    }


def wait_for_green(
    sha: str,
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    runner: Callable[..., dict[str, Any]] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Poll until the commit's run is green, or until the budget runs out.

    Never raises on a red or slow CI: an exhausted budget is reported, not thrown, so
    the caller can stop the round cleanly and write a resume command.
    """
    now = clock or time.monotonic
    pause = sleep or time.sleep
    started = now()
    history: list[dict[str, Any]] = []
    while True:
        outcome = verdict(pick_run(fetch_runs(repo, branch, runner), sha), sha)
        history.append({"at_seconds": round(now() - started, 1), "state": outcome["state"]})
        if outcome["green"]:
            return {"passed": True, "budget_exhausted": False, "last": outcome, "history": history}
        elapsed = now() - started
        if elapsed >= timeout_seconds:
            return {"passed": False, "budget_exhausted": True, "last": outcome, "history": history}
        pause(min(interval_seconds, max(0.0, timeout_seconds - elapsed)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    args = parser.parse_args(argv)
    try:
        sha = resolve_sha(args.ref)
        payload = wait_for_green(
            sha,
            repo=args.repo,
            branch=args.branch,
            timeout_seconds=args.timeout_seconds,
            interval_seconds=args.interval_seconds,
        )
    except CiWaitError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 2
    payload["sha"] = sha
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
