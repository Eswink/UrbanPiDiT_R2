"""Tests for the campaign-state checker (decision 0025).

The checker guards the cross-checks that used to be manual: the master plan's
current node against the previous round's next action, the GPU ledger's
arithmetic, and whether the previous round's evidence is registered. Every
mechanical rule gets a counterproof on a fixture repository, and one test
asserts the real repository currently satisfies all of them - a rule that only
passes on fixtures would be worthless.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

BRIEF_BODY = (
    "# {title}\n\n"
    "## §0 Objective（可粘贴；实测 20 字符）\n\n"
    "> Do the round and name docs/goals/{name} so the verifier can find the criteria.\n\n"
    "## §2 交付物清单\n\n"
    "| # | 交付物 | 证据形态 |\n| --- | --- | --- |\n| D1 | one thing | test name |\n\n"
    "## §3 判据与证据来源\n\n"
    "> Criteria come from docs/R7_B2_MULTISEED.md only.\n\n"
    "## §5 预算与停止\n\n"
    "| 项 | 值 |\n| --- | --- |\n| 本轮上限 | 0.5 GPU-h |\n"
    "| 停止条件 | over budget |\n\n"
    "## §7 进度块\n\n"
    "- **状态**：`active`\n- **下一动作**：{next_action}\n"
)

LEDGER = (
    "| 轮次 | 实测 GPU-h | 累计 | 证据 |\n"
    "| --- | --- | --- | --- |\n"
    "| round one | 1.0 | 1.0 | `docs/R7_ONE.md` + 索引记录 `record:one` |\n"
    "| round two | 0.5 | 1.5 | `docs/R7_TWO.md` |\n"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load("check_campaign_state", TOOLS / "check_campaign_state.py")


def _state(**overrides) -> str:
    state = {
        "current_node": "N1",
        "previous_node": "N0",
        "current_round_goal": "docs/goals/round-b.md",
        "previous_round_goal": "docs/goals/round-a.md",
        "previous_round_evidence": "docs/R7_TWO.md",
        "cap_gpu_h": 24.0,
        "used_gpu_h": 1.5,
        "remaining_gpu_h": 22.5,
    }
    state.update(overrides)
    return "<!-- campaign-state: " + json.dumps(state, ensure_ascii=False) + " -->\n"


def build_repo(
    tmp_path: Path,
    *,
    ledger: str = LEDGER,
    state: str | None = None,
    current_node_marker: str = "N1",
    previous_next_action: str = "start node N1",
    records: list[dict] | None = None,
) -> Path:
    goals = tmp_path / "docs" / "goals"
    goals.mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "R7_ONE.md").write_text("# one\n", encoding="utf-8")
    (tmp_path / "docs" / "R7_TWO.md").write_text("# two\n", encoding="utf-8")
    (goals / "main-model-v2-campaign.md").write_text(
        "# campaign\n\n" + (state or _state()) + "\n## §7 账本\n\n" + ledger, encoding="utf-8")
    (goals / "round-a.md").write_text(
        BRIEF_BODY.format(title="round a", name="round-a.md",
                          next_action=previous_next_action)
        + "\n<!-- round-node: N0 -->\n", encoding="utf-8")
    (goals / "round-b.md").write_text(
        BRIEF_BODY.format(title="round b", name="round-b.md", next_action="finish D1")
        + f"\n<!-- round-node: {current_node_marker} -->\n", encoding="utf-8")
    if records is None:
        records = [{"record_id": "one", "evidence_path": "docs/R7_ONE.md",
                    "metrics": {"gpu_hours": 1.0}},
                   {"record_id": "two", "evidence_path": "docs/R7_TWO.md",
                    "evidence_commit": "a" * 40,
                    "metrics": {"gpu_hours": 0.5}}]
    index = tmp_path / "docs" / "R7_EVIDENCE_INDEX.jsonl"
    index.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                     encoding="utf-8")
    return tmp_path


def failures(report: dict, prefix: str) -> list[str]:
    return [finding for finding in report["failures"] if finding.startswith(prefix)]


def test_conforming_fixture_passes(checker, tmp_path):
    report = checker.check(build_repo(tmp_path))
    assert report["failures"] == []
    assert report["node"] == "N1"
    assert any("not machine-checked" in note for note in report["notes"])


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"ledger": LEDGER.replace("| 0.5 | 1.5 |", "| 0.6 | 1.5 |")}, "C-02"),
        ({"ledger": LEDGER.replace("| 1.0 | 1.0 |", "| 1.0 | 0.9 |")}, "C-02"),
        ({"ledger": LEDGER.replace("`docs/R7_ONE.md`", "`docs/R7_MISSING.md`")}, "C-03"),
        ({"previous_next_action": "keep going"}, "C-04"),
        ({"current_node_marker": "N2"}, "C-05"),
        ({"records": [{"record_id": "one", "evidence_path": "docs/R7_ONE.md",
                       "metrics": {"gpu_hours": 1.25}}]}, "C-03"),
        ({"records": []}, "C-06"),
    ],
)
def test_each_rule_rejects_its_violation(checker, tmp_path, mutation, expected):
    report = checker.check(build_repo(tmp_path, **mutation))
    assert failures(report, expected), (mutation, report)


def test_unparseable_state_fails(checker, tmp_path):
    report = checker.check(build_repo(tmp_path, state="no machine block here\n"))
    assert failures(report, "C-01")


def test_hard_drift_exits_nonzero(checker, tmp_path, capsys):
    repo = build_repo(tmp_path, current_node_marker="N9")
    assert checker.main(["--root", str(repo), "--quiet"]) == 1
    capsys.readouterr()
    assert checker.main(["--root", str(repo), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["failures"]


def test_evidence_commit_must_exist_when_git_can_answer(checker, tmp_path):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not available")
    repo = build_repo(tmp_path, records=[
        {"record_id": "one", "evidence_path": "docs/R7_ONE.md",
         "metrics": {"gpu_hours": 1.0}},
        {"record_id": "two", "evidence_path": "docs/R7_TWO.md",
         "evidence_commit": "0" * 40},
    ])
    assert failures(checker.check(repo), "C-06") == []
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    report = checker.check(repo)
    assert failures(report, "C-06")


def test_checker_does_not_modify_what_it_reads(checker):
    tracked = [ROOT / "docs/goals/main-model-v2-campaign.md",
               ROOT / "docs/goals/main-model-v2-pivot-audit.md",
               ROOT / "docs/R7_EVIDENCE_INDEX.jsonl"]
    before = {path: path.read_text(encoding="utf-8") for path in tracked}
    checker.check(ROOT)
    assert {path: path.read_text(encoding="utf-8") for path in tracked} == before


def test_real_repository_satisfies_every_rule(checker):
    report = checker.check(ROOT)
    assert report["failures"] == [], report
