"""Stage, freeze and GPU-round tests for ``tools/r7_round_support.py``.

Split from ``tests/test_r7_round.py`` to stay under the 600-line hard cap (R-051). The
helpers here are deliberately self-contained: this file tests the per-stage driver
wrapper, which needs a synthetic driver and a synthetic protocol and nothing else.

No GPU, no network, no ``outputs/``: every outside call is injected.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAKE_DRIVER = """from __future__ import annotations
import json
from pathlib import Path

SEEDS = (41, 42, 43)
PER_SEED_SECONDS = 18000.0
PLANNED_SECONDS = 54000.0
HARD_CAP_SECONDS = 72000.0

def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    body = {"protocol_sha256": "a" * 64, "started_perf_counter": started,
            "deadline_perf_counter": deadline, "planned_seconds": PLANNED_SECONDS,
            "hard_cap_seconds": HARD_CAP_SECONDS, "per_seed_seconds": PER_SEED_SECONDS,
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256, "seeds": list(SEEDS),
            "gpu_uuid": "GPU-test", "estimated_peak_bytes": 2 ** 31,
            "headroom_margin_bytes": 2 ** 31}
    (Path(output) / "preparation_protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return body
"""

def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture()
def support():
    """The per-stage wrapper, which re-exports the shared names it needs."""
    return _load("r7_round_stages_under_test", "tools/r7_round_stages.py")

def _ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}

def _fail(code: int = 1, stderr: str = "boom") -> dict:
    return {"exit_code": code, "stdout": "", "stderr": stderr}

def _fake_driver(root: Path) -> str:
    target = root / "scripts/study_fake.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(FAKE_DRIVER, encoding="utf-8")
    return "scripts/study_fake.py"

def _protocol_file(output: Path, **overrides) -> Path:
    body = {"protocol_sha256": "a" * 64, "deadline_perf_counter": 10_000.0, "per_seed_seconds": 1800.0,
            "planned_seconds": 5400.0, "hard_cap_seconds": 7200.0, "gpu_uuid": "GPU-test",
            "estimated_peak_bytes": 2 ** 31, "headroom_margin_bytes": 2 ** 31}
    body.update(overrides)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "protocol.json"
    target.write_text(json.dumps(body), encoding="utf-8")
    return target

def test_current_attempt_dir_takes_the_newest_and_refuses_when_absent(support, tmp_path):
    outputs = tmp_path / "outputs"
    with pytest.raises(support.RoundError):
        support.current_attempt_dir("stem", outputs)
    (outputs / "stem_attempt01").mkdir(parents=True)
    (outputs / "stem_attempt02").mkdir()
    assert support.current_attempt_dir("stem", outputs).name.endswith("attempt02")

def test_stage_plan_skips_only_when_the_receipt_exists(support, tmp_path):
    assert support.stage_plan(tmp_path, "prepare")["skip"] is False
    (tmp_path / "prepared_protocol.json").write_text("{}", encoding="utf-8")
    plan = support.stage_plan(tmp_path, "prepare")
    assert plan["skip"] is True
    assert plan["receipt_sha256"] is not None

def test_stage_plan_maps_every_stage_to_its_receipt(support, tmp_path):
    assert support.stage_plan(tmp_path, "archive")["receipt"].endswith("archive_receipt.json")
    assert support.stage_plan(tmp_path, "reading")["receipt"].endswith("readings.json")
    assert support.stage_plan(tmp_path, "seed", 42)["receipt"].endswith("seed42_receipt.json")

def test_stage_plan_refuses_an_unknown_phase_and_a_seedless_seed(support, tmp_path):
    with pytest.raises(support.RoundError):
        support.stage_plan(tmp_path, "nonsense")
    with pytest.raises(support.RoundError):
        support.stage_plan(tmp_path, "seed")

def test_run_stage_counterproof_a_done_stage_never_spawns(support, tmp_path):
    (tmp_path / "prepared_protocol.json").write_text("{}", encoding="utf-8")
    calls = []

    def runner(*args):
        calls.append(args)
        return _ok()

    record = support.run_stage(tmp_path, "scripts/study_fake.py", "prepare", runner=runner)
    assert record["status"] == "skipped"
    assert calls == []

def test_run_stage_spawns_the_driver_worker_with_the_frozen_deadline(support, tmp_path):
    _protocol_file(tmp_path)
    seen = {}

    def runner(argv, cwd=None, env=None, timeout=None):
        seen["argv"] = list(argv)
        seen["env"] = env
        seen["timeout"] = timeout
        return _ok("done\n")

    record = support.run_stage(tmp_path, "scripts/study_fake.py", "prepare", runner=runner,
                               clock=lambda: 1000.0)
    assert record["passed"] is True
    assert seen["argv"][1].endswith("scripts/study_fake.py")
    assert seen["argv"][2:5] == ["--worker", "prepare", "--out"]
    assert seen["argv"][-2:] == ["--deadline", "10000.0"]
    assert "--seed" not in seen["argv"]
    assert set(seen["env"]) <= set(support.WORKER_ENV_KEYS)
    assert (tmp_path / "prepare_process.json").is_file()

def test_run_stage_clamps_a_seed_deadline_to_the_per_seed_budget(support, tmp_path):
    _protocol_file(tmp_path, deadline_perf_counter=999_999.0, per_seed_seconds=1800.0)
    seen = {}

    def probe(argv, cwd=None, env=None, timeout=None):
        if argv[0] == "nvidia-smi":
            return _ok("GPU-test, 30000, 24576" + chr(10))
        seen["argv"] = list(argv)
        return _ok()

    record = support.run_stage(tmp_path, "scripts/study_fake.py", "seed", seed=41, runner=probe,
                               clock=lambda: 1000.0)
    assert record["passed"] is True
    assert seen["argv"][-4:] == ["--deadline", "2800.0", "--seed", "41"]
    assert record["gate"]["passed"] is True

def test_run_stage_blocks_a_seed_when_headroom_is_insufficient(support, tmp_path):
    _protocol_file(tmp_path)
    calls = []

    def runner(argv, cwd=None, env=None, timeout=None):
        if argv[0] == "nvidia-smi":
            return _ok("GPU-test, 512, 24576\n")
        calls.append(list(argv))
        return _ok()

    record = support.run_stage(tmp_path, "scripts/study_fake.py", "seed", seed=41, runner=runner,
                               clock=lambda: 1000.0)
    assert record["status"] == "headroom-blocked"
    assert record["passed"] is False
    assert calls == []

def test_run_stage_refuses_to_spawn_after_the_hard_deadline(support, tmp_path):
    _protocol_file(tmp_path, deadline_perf_counter=500.0)
    with pytest.raises(support.RoundError):
        support.run_stage(tmp_path, "scripts/study_fake.py", "prepare", runner=lambda *a: _ok(),
                          clock=lambda: 1000.0)

def test_run_stage_counterproof_a_failing_worker_is_not_passing(support, tmp_path):
    _protocol_file(tmp_path)
    record = support.run_stage(tmp_path, "scripts/study_fake.py", "prepare",
                               runner=lambda *a: {"exit_code": 1, "stdout": "", "stderr": "boom"},
                               clock=lambda: 1000.0)
    assert record["passed"] is False
    assert record["status"] == "failed"

def test_worker_environment_keeps_only_the_allowlist(support):
    env = support.worker_environment({"PATH": "/bin", "SECRET": "x", "HOME": "/root"})
    assert env == {"PATH": "/bin", "HOME": "/root"}

def test_freeze_round_creates_a_fresh_attempt_and_delegates(support, tmp_path):
    driver = _fake_driver(tmp_path)
    archive = tmp_path / "outputs" / "code_input.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"zip")
    payload = support.freeze_round({"driver": driver, "output_stem": "r7_s3_fake_20261011"},
                                   code_archive=str(archive), code_commit="c" * 40,
                                   code_sha256="d" * 64, root=tmp_path, clock=lambda: 1000.0)
    assert Path(payload["attempt_dir"]).name.endswith("attempt01")
    assert payload["hard_cap_seconds"] == 72000.0
    assert (Path(payload["attempt_dir"]) / "preparation_protocol.json").is_file()

def test_freeze_round_counterproof_a_short_commit_is_refused(support, tmp_path):
    with pytest.raises(support.RoundError):
        support.freeze_round({"driver": _fake_driver(tmp_path), "output_stem": "s"},
                             code_archive="x", code_commit="short", code_sha256="d" * 64, root=tmp_path)

def test_freeze_round_counterproof_a_driver_missing_names_is_refused(support, tmp_path):
    stub = tmp_path / "scripts/study_stub.py"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(support.RoundError) as excinfo:
        support.freeze_round({"driver": "scripts/study_stub.py", "output_stem": "s"},
                             code_archive="x", code_commit="c" * 40, code_sha256="d" * 64, root=tmp_path)
    assert "does not expose" in str(excinfo.value)

# --- commit and push primitives ----------------------------------------------------

COMMIT_SPEC = {"commit_paths": ["tools/r7_round.py", "docs/plans/0020.md"],
               "commit_message": "feat(r7): round driver support"}

def _git_runner(branch: str = "r7/weather-reasoning", sha: str = "e" * 40, fail_on: str = ""):
    calls = []

    def runner(argv, cwd=None, env=None, timeout=None):
        calls.append(list(argv))
        if argv[0] == "git" and argv[1] == "rev-parse":
            return _ok(branch + "\n") if "--abbrev-ref" in argv else _ok(sha + "\n")
        if argv[0] == "git" and argv[1] in {"add", "commit"} and argv[1] == fail_on:
            return _fail(1, "git refused")
        if argv[0] == "git" and argv[1] == "push":
            return _fail(1, "rejected") if fail_on == "push" else _ok("up to date")
        return _ok()

    runner.calls = calls
    return runner

def test_commit_round_counterproof_a_protected_branch_is_refused(support):
    with pytest.raises(support.RoundError):
        support.commit_round(COMMIT_SPEC, runner=_git_runner(branch="main"))

def test_commit_round_refuses_without_paths_or_message(support):
    with pytest.raises(support.RoundError):
        support.commit_round({"commit_message": "x"}, runner=_git_runner())
    with pytest.raises(support.RoundError):
        support.commit_round({"commit_paths": ["a"]}, runner=_git_runner())

def test_commit_round_commits_only_the_declared_paths(support):
    runner = _git_runner()
    payload = support.commit_round(COMMIT_SPEC, runner=runner)
    assert payload["passed"] is True
    assert payload["sha"] == "e" * 40
    add = next(call for call in runner.calls if call[1] == "add")
    commit = next(call for call in runner.calls if call[1] == "commit")
    assert add == ["git", "add", "--", *COMMIT_SPEC["commit_paths"]]
    assert commit[3] == COMMIT_SPEC["commit_message"]
    assert commit[-len(COMMIT_SPEC["commit_paths"]):] == COMMIT_SPEC["commit_paths"]

def test_commit_round_counterproof_a_failing_add_is_not_passing(support):
    payload = support.commit_round(COMMIT_SPEC, runner=_git_runner(fail_on="add"))
    assert payload["passed"] is False
    assert payload["stage"] == "add"

def test_push_round_counterproof_a_protected_branch_is_refused(support):
    with pytest.raises(support.RoundError):
        support.push_round(runner=_git_runner(branch="main"))

def test_push_round_pushes_without_a_refspec(support):
    runner = _git_runner()
    assert support.push_round(runner=runner)["passed"] is True
    assert [call for call in runner.calls if call[1] == "push"] == [["git", "push"]]

def test_push_round_counterproof_a_rejected_push_is_not_passing(support):
    assert support.push_round(runner=_git_runner(fail_on="push"))["passed"] is False
