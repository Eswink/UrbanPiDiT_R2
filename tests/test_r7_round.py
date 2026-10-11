"""Counterproofs for ``tools/r7_round_support.py`` and ``tools/r7_round.py``.

Everything here runs without a GPU, without a network and without ``outputs/``: the
round support module takes an injectable runner for exactly this reason. The earlier
dose round turned CI red by reading a real ``outputs/`` tree from a test, so no test in
this file may depend on one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

STATE_BLOCK = "<!-- campaign-state: " + json.dumps({
    "current_node": "S3", "previous_node": "S3", "current_round_goal": "docs/goals/s3-x.md",
    "previous_round_goal": "docs/goals/s3-b.md", "previous_round_evidence": "docs/R7_B.md",
    "cap_gpu_h": 20.0, "used_gpu_h": 3.5, "remaining_gpu_h": 16.5}) + " -->"

MASTER_TEMPLATE = """# synthetic master

""" + STATE_BLOCK + """

## §7 账本

| 轮次 | 实测 GPU-h | 累计 | 证据 |
| --- | --- | --- | --- |
| 第一轮 | 1.5 | 1.5 | `docs/R7_A.md`（索引 `record:round-a`） |
| 第二轮 | 2.0 | 3.5 | `docs/R7_B.md`（索引 `record:round-b`） |
| **合计已用** | **3.5** | — | 会计字段 |

## §8 进度块

- **下一动作（S3）**：跑第三轮。
"""

def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture()
def support():
    return _load("r7_round_support", "tools/r7_round_support.py")

@pytest.fixture()
def cli():
    return _load("r7_round_cli", "tools/r7_round.py")

def _write_master(root: Path, text: str = MASTER_TEMPLATE) -> Path:
    target = root / "docs/goals/main-model-climatology-campaign.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target

def _ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": ""}

def _fail(code: int = 1, stderr: str = "boom") -> dict:
    return {"exit_code": code, "stdout": "", "stderr": stderr}

# --- campaign state ---------------------------------------------------------------

def test_campaign_state_parses_the_required_keys(support, tmp_path):
    body = support.campaign_state(_write_master(tmp_path))
    assert body["used_gpu_h"] == 3.5
    assert body["remaining_gpu_h"] == 16.5

def test_campaign_state_refuses_a_missing_block(support, tmp_path):
    master = _write_master(tmp_path, "# no state here\n")
    with pytest.raises(support.RoundError):
        support.campaign_state(master)

def test_campaign_state_refuses_a_missing_key(support, tmp_path):
    master = _write_master(tmp_path, '<!-- campaign-state: {"current_node": "S3"} -->\n')
    with pytest.raises(support.RoundError):
        support.campaign_state(master)

def test_campaign_state_refuses_two_blocks(support, tmp_path):
    text = MASTER_TEMPLATE + '\n<!-- campaign-state: {"current_node": "S3"} -->\n'
    with pytest.raises(support.RoundError):
        support.campaign_state(_write_master(tmp_path, text))

# --- round spec -------------------------------------------------------------------

SPEC = {"arm": "s3-reasoning-depth", "driver": "scripts/study_x.py", "output_stem": "r7_s3_x_20261011",
        "control_record": "s3-wide-interior-dose", "seeds": [41, 42, 43]}

def _write_spec(root: Path, body: dict) -> Path:
    target = root / "docs/goals/s3-x.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# round\n\n<!-- round-spec: {json.dumps(body)} -->\n", encoding="utf-8")
    return target

def test_round_spec_round_trips(support, tmp_path):
    assert support.round_spec(_write_spec(tmp_path, SPEC)) == SPEC

def test_round_spec_refuses_a_missing_key(support, tmp_path):
    incomplete = {key: value for key, value in SPEC.items() if key != "control_record"}
    with pytest.raises(support.RoundError):
        support.round_spec(_write_spec(tmp_path, incomplete))

def test_round_spec_refuses_an_absent_block(support, tmp_path):
    target = tmp_path / "docs/goals/s3-x.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# no spec\n", encoding="utf-8")
    with pytest.raises(support.RoundError):
        support.round_spec(target)

# --- ledger -----------------------------------------------------------------------

def test_ledger_rows_skip_the_total_row(support, tmp_path):
    rows = support.ledger_rows(_write_master(tmp_path))
    assert [row["gpu_hours"] for row in rows] == [1.5, 2.0]
    assert rows[0]["record_ids"] == ["round-a"]
    assert rows[1]["cumulative"] == 3.5

def test_ledger_plan_arithmetic(support, tmp_path):
    plan = support.ledger_plan(support.ledger_rows(_write_master(tmp_path)), 20.0, 4.2867)
    assert plan["previous_total_gpu_h"] == 3.5
    assert plan["new_total_gpu_h"] == 7.7867
    assert plan["remaining_gpu_h"] == 12.2133

def test_ledger_plan_counterproof_a_wrong_new_total_is_visible(support, tmp_path):
    rows = support.ledger_rows(_write_master(tmp_path))
    plan = support.ledger_plan(rows, 20.0, 4.2867)
    assert plan["new_total_gpu_h"] != plan["previous_total_gpu_h"] + 4.2

# --- attempt dirs -----------------------------------------------------------------

def test_next_attempt_dir_starts_at_one(support, tmp_path):
    (tmp_path / "outputs").mkdir()
    assert support.next_attempt_dir("r7_s3_x_20261011", tmp_path / "outputs").name.endswith("attempt01")

def test_next_attempt_dir_skips_taken_numbers(support, tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "r7_s3_x_20261011_attempt01").mkdir(parents=True)
    (outputs / "r7_s3_x_20261011_attempt02").mkdir()
    assert support.next_attempt_dir("r7_s3_x_20261011", outputs).name.endswith("attempt03")

def test_next_attempt_dir_refuses_an_unsafe_stem(support, tmp_path):
    with pytest.raises(support.RoundError):
        support.next_attempt_dir("../escape", tmp_path)

# --- cross-attempt seed reuse (decision 0045) --------------------------------------

# --- validators and the commit gate ------------------------------------------------

def test_run_validators_passes_only_when_every_command_is_zero(support):
    calls = []

    def runner(argv, cwd=None):
        calls.append(tuple(argv))
        return _ok("ok")

    assert support.run_validators(runner=runner)["passed"] is True
    assert len(calls) == len(support.VALIDATOR_COMMANDS)

def test_run_validators_counterproof_one_red_is_not_passing(support):
    def runner(argv, cwd=None):
        return _fail(1, "R-051 red") if any("check_conventions.py" in item for item in argv) else _ok()

    verdict = support.run_validators(runner=runner)
    assert verdict["passed"] is False
    assert any(item["exit_code"] == 1 for item in verdict["results"])

def test_commit_guard_passes_on_zero(support):
    seen = {}

    def runner(argv, cwd=None):
        seen["argv"] = list(argv)
        return _ok()

    assert support.commit_guard(["tools/r7_round.py"], runner=runner)["passed"] is True
    assert seen["argv"][-1] == "tools/r7_round.py"
    assert "--paths" in seen["argv"]

def test_commit_guard_counterproof_a_red_checker_blocks(support):
    guard = support.commit_guard(["docs/x.md"], runner=lambda argv, cwd=None: _fail(1, "blocking rule"))
    assert guard["passed"] is False
    assert guard["exit_code"] == 1

def test_commit_guard_refuses_an_empty_path_list(support):
    with pytest.raises(support.RoundError):
        support.commit_guard([])

# --- gpu co-residency probe --------------------------------------------------------


# --- state report ------------------------------------------------------------------

def test_state_report_assembles_the_round_context(support, tmp_path):
    _write_master(tmp_path)
    (tmp_path / "docs/goals/s3-x.md").write_text("# another brief\n", encoding="utf-8")
    (tmp_path / "docs/R7_EVIDENCE_INDEX.jsonl").write_text(
        json.dumps({"record_id": "round-b"}) + "\n", encoding="utf-8")

    def runner(argv, cwd=None):
        is_head = argv[:2] == ["git", "rev-parse"] and argv[-1] == "HEAD"
        return _ok("0df159d8\n" if is_head else "r7/weather-reasoning\n")

    report = support.state_report(tmp_path, runner=runner)
    assert report["campaign_state"]["current_node"] == "S3"
    assert report["ledger_rows"] == 2
    assert report["ledger_total_gpu_h"] == 3.5
    assert report["last_record_ids"] == ["round-b"]
    assert report["next_action"].startswith("下一动作")
    assert "s3-x.md" in report["goal_briefs"]

def test_state_report_reads_the_real_repository(support):
    """The real master must stay parseable; this is the no-drift guard."""
    report = support.state_report()
    assert report["campaign_state"]["current_node"] in {"S0", "S1", "S2", "S3", "S4"}
    assert report["ledger_rows"] > 0
    assert report["next_action"].startswith("下一动作")

# --- CLI ---------------------------------------------------------------------------

def test_cli_exposes_the_support_module(cli):
    assert cli.support.MASTER_REL.name == "main-model-climatology-campaign.md"

def test_cli_rejects_an_unknown_subcommand(cli):
    with pytest.raises(cli.support.RoundError):
        cli.dispatch("nonsense", [])

def test_cli_refuses_malformed_arguments(cli):
    with pytest.raises(cli.support.RoundError):
        cli.reuse_plan([])
    with pytest.raises(cli.support.RoundError):
        cli.ledger_plan([])

def test_cli_usage_exit_code_is_two(cli, capsys):
    assert cli.main([]) == 2
    assert "usage" in capsys.readouterr().out

def test_cli_ledger_plan_prints_arithmetic(cli, capsys):
    assert cli.main(["ledger-plan", "1.0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["new_row_gpu_h"] == 1.0
    assert payload["new_total_gpu_h"] == payload["previous_total_gpu_h"] + 1.0

def test_cli_reuse_plan_refuses_when_the_previous_protocol_is_absent(cli, tmp_path, monkeypatch, capsys):
    _round_root(cli, tmp_path, monkeypatch)
    attempt = tmp_path / "outputs" / "r7_s3_x_20261011_attempt01"
    attempt.mkdir(parents=True)
    (attempt / "protocol.json").write_text(json.dumps({"seeds": [41]}), encoding="utf-8")
    assert cli.main(["reuse-plan", str(tmp_path / "missing")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reusable"] == []
    assert payload["previous_protocol_file_sha256"] is None
    assert payload["reason"] == "no previous attempt"

def _round_root(cli, tmp_path, monkeypatch) -> Path:
    """A synthetic repo whose campaign-state points at a synthetic round brief."""
    _write_master(tmp_path)
    _spec(tmp_path, _fake_driver(tmp_path))
    monkeypatch.setattr(cli.support, "ROOT", tmp_path)
    monkeypatch.setattr(cli.stages, "ROOT", tmp_path)
    return tmp_path

def test_cli_zero_argument_subcommands_refuse_arguments(cli):
    for handler in (cli.attempt, cli.freeze, cli.round_guard):
        with pytest.raises(cli.support.RoundError):
            handler(["surplus"])

def test_cli_stage_subcommands_validate_arity(cli):
    for handler in (cli.stage, cli.stage_plan):
        with pytest.raises(cli.support.RoundError):
            handler([])
        with pytest.raises(cli.support.RoundError):
            handler(["prepare", "41", "extra"])

def test_cli_attempt_resolves_the_round_from_campaign_state(cli, tmp_path, monkeypatch):
    _round_root(cli, tmp_path, monkeypatch)
    (tmp_path / "outputs" / "r7_s3_x_20261011_attempt01").mkdir(parents=True)
    payload = cli.attempt([])
    assert payload["attempt_dir"].endswith("attempt01")
    assert payload["goal"].endswith("docs/goals/s3-x.md")

def test_cli_freeze_refuses_a_missing_code_archive(cli, tmp_path, monkeypatch):
    _round_root(cli, tmp_path, monkeypatch)
    with pytest.raises(cli.support.RoundError) as excinfo:
        cli.freeze([])
    assert "code archive is missing" in str(excinfo.value)

def test_cli_stage_plan_reports_skip_without_a_receipt(cli, tmp_path, monkeypatch):
    _round_root(cli, tmp_path, monkeypatch)
    (tmp_path / "outputs" / "r7_s3_x_20261011_attempt01").mkdir(parents=True)
    assert cli.stage_plan(["prepare"])["skip"] is False

FAKE_DRIVER = """from __future__ import annotations
import json
from pathlib import Path

SEEDS = (41, 42, 43)
PER_SEED_SECONDS = 18000.0
PLANNED_SECONDS = 54000.0
HARD_CAP_SECONDS = 72000.0

def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    body = {"protocol_sha256": "a" * 64, "deadline_perf_counter": deadline,
            "per_seed_seconds": PER_SEED_SECONDS, "seeds": list(SEEDS)}
    (Path(output) / "preparation_protocol.json").write_text(json.dumps(body), encoding="utf-8")
    return body
"""

def _fake_driver(root: Path) -> str:
    target = root / "scripts/study_fake.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(FAKE_DRIVER, encoding="utf-8")
    return "scripts/study_fake.py"

def _spec(root: Path, driver: str) -> Path:
    return _write_spec(root, {**SPEC, "driver": driver, "code_archive": "outputs/code_input.zip"})

def test_run_pytest_is_a_no_op_without_a_declared_test_path(support):
    assert support.run_pytest(None)["skipped"] is True

def test_run_pytest_counterproof_a_red_module_is_not_passing(support):
    seen = {}

    def runner(argv, cwd=None, env=None, timeout=None):
        seen["argv"] = list(argv)
        return _fail(1, "1 failed")

    verdict = support.run_pytest(["tests/test_x.py"], runner=runner)
    assert verdict["passed"] is False
    assert seen["argv"][-1] == "tests/test_check_conventions.py"
    assert "no:cacheprovider" in seen["argv"]
