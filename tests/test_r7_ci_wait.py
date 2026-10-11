"""Counterproofs for ``tools/r7_ci_wait.py``.

No network client is imported and no request is made: every test injects a runner. The
classification is the part that must not drift, because a mis-read status would let a
red commit be frozen as if it were green.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHA = "0df159d81e04fada4efaefe01855c6dfdab8d197"


@pytest.fixture()
def ciw():
    spec = importlib.util.spec_from_file_location("r7_ci_wait", ROOT / "tools/r7_ci_wait.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}


def _run(sha: str = SHA, status: str = "completed", conclusion: str = "success", run_id: int = 7) -> dict:
    return {"id": run_id, "name": "ci", "head_sha": sha, "status": status, "conclusion": conclusion,
            "html_url": "https://example.invalid/run"}


def _payload(runs) -> str:
    return json.dumps({"workflow_runs": list(runs)})


# --- classification ----------------------------------------------------------------


def test_verdict_green_only_for_completed_success(ciw):
    assert ciw.verdict(_run(), SHA)["green"] is True


@pytest.mark.parametrize("status,conclusion", [
    ("in_progress", None),
    ("queued", None),
    ("completed", "failure"),
    ("completed", "cancelled"),
    ("completed", "skipped"),
    ("completed", "timed_out"),
    ("completed", "action_required"),
    ("completed", None),
])
def test_verdict_counterproof_nothing_else_counts_as_passing(ciw, status, conclusion):
    assert ciw.verdict(_run(status=status, conclusion=conclusion), SHA)["green"] is False


def test_verdict_reports_absence_explicitly(ciw):
    outcome = ciw.verdict(None, SHA)
    assert outcome["state"] == "absent"
    assert outcome["green"] is False


# --- run selection -----------------------------------------------------------------


def test_pick_run_requires_the_exact_sha(ciw):
    assert ciw.pick_run([_run(sha="a" * 40)], SHA) is None


def test_pick_run_takes_the_newest_matching_run(ciw):
    older = _run(run_id=1, conclusion="failure")
    newer = _run(run_id=9, conclusion="success")
    assert ciw.pick_run([newer, older], SHA)["id"] == 9


# --- transport ---------------------------------------------------------------------


def test_resolve_sha_uses_git(ciw):
    assert ciw.resolve_sha("HEAD", runner=lambda argv, cwd=None: _ok(SHA + "\n")) == SHA


def test_resolve_sha_refuses_a_broken_ref(ciw):
    with pytest.raises(ciw.CiWaitError):
        ciw.resolve_sha("nope", runner=lambda argv, cwd=None: {"exit_code": 128, "stdout": "", "stderr": "bad"})


def test_fetch_runs_parses_the_payload(ciw):
    runs = ciw.fetch_runs("o/r", "b", runner=lambda argv, cwd=None: _ok(_payload([_run()])))
    assert runs[0]["head_sha"] == SHA


def test_fetch_runs_counterproof_non_json_is_an_error(ciw):
    with pytest.raises(ciw.CiWaitError):
        ciw.fetch_runs("o/r", "b", runner=lambda argv, cwd=None: _ok("rate limited"))


def test_fetch_runs_counterproof_an_unexpected_payload_shape_is_an_error(ciw):
    with pytest.raises(ciw.CiWaitError):
        ciw.fetch_runs("o/r", "b", runner=lambda argv, cwd=None: _ok(json.dumps({"message": "Not Found"})))


def test_fetch_runs_counterproof_a_failing_curl_is_an_error(ciw):
    with pytest.raises(ciw.CiWaitError):
        ciw.fetch_runs("o/r", "b", runner=lambda argv, cwd=None: {"exit_code": 6, "stdout": "", "stderr": "dns"})


def test_runs_url_is_anonymous_and_branch_scoped(ciw):
    url = ciw.runs_url("Eswink/UrbanPiDiT_R2", "r7/weather-reasoning")
    assert url.startswith("https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs?branch=")
    assert "per_page=" in url


# --- waiting -----------------------------------------------------------------------


def test_wait_returns_immediately_when_green(ciw):
    payload = ciw.wait_for_green(SHA, runner=lambda argv, cwd=None: _ok(_payload([_run()])))
    assert payload["passed"] is True
    assert payload["history"] == [{"at_seconds": 0.0, "state": "green"}]


def test_wait_polls_until_the_run_appears_then_turns_green(ciw):
    state = {"polls": 0}

    def runner(argv, cwd=None):
        state["polls"] += 1
        if state["polls"] == 1:
            return _ok(_payload([_run(sha="a" * 40)]))
        return _ok(_payload([_run()]))

    payload = ciw.wait_for_green(SHA, runner=runner, sleep=lambda _: None)
    assert payload["passed"] is True
    assert [item["state"] for item in payload["history"]] == ["absent", "green"]


def test_wait_reports_budget_exhaustion_instead_of_raising(ciw):
    clock = iter([0.0, 10.0, 2000.0])
    payload = ciw.wait_for_green(SHA, timeout_seconds=100.0, interval_seconds=5.0,
                                 runner=lambda argv, cwd=None: _ok(_payload([_run(status="in_progress", conclusion=None)])),
                                 sleep=lambda _: None, clock=lambda: next(clock))
    assert payload["passed"] is False
    assert payload["budget_exhausted"] is True
    assert payload["last"]["state"] == "pending"


def test_wait_counterproof_a_cancelled_run_never_becomes_green(ciw):
    clock = iter([0.0, 0.0, 500.0])
    payload = ciw.wait_for_green(SHA, timeout_seconds=100.0,
                                 runner=lambda argv, cwd=None: _ok(_payload([_run(conclusion="cancelled")])),
                                 sleep=lambda _: None, clock=lambda: next(clock))
    assert payload["passed"] is False
    assert payload["last"]["conclusion"] == "cancelled"


# --- CLI ---------------------------------------------------------------------------


def test_main_exit_code_is_one_when_not_green(ciw, monkeypatch, capsys):
    monkeypatch.setattr(ciw, "default_runner", lambda argv, cwd=None: _ok(_payload([_run(status="queued", conclusion=None)])))
    assert ciw.main(["--ref", "HEAD", "--timeout-seconds", "0"]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_main_exit_code_is_two_on_a_transport_error(ciw, monkeypatch, capsys):
    monkeypatch.setattr(ciw, "default_runner", lambda argv, cwd=None: _ok("not json"))
    assert ciw.main(["--ref", "HEAD"]) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_main_is_green_for_a_successful_run(ciw, monkeypatch, capsys):
    def runner(argv, cwd=None):
        return _ok(SHA + "\n") if argv[0] == "git" else _ok(_payload([_run()]))

    monkeypatch.setattr(ciw, "default_runner", runner)
    assert ciw.main(["--ref", "HEAD"]) == 0
    assert json.loads(capsys.readouterr().out)["sha"] == SHA
