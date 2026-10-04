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


NEW_CAMPAIGN = "docs/goals/main-model-climatology-campaign.md"
START_PLAN = "docs/plans/0016-main-model-climatology-campaign.md"
HANDOFF_GOAL = "docs/goals/v2-issue-closeout.md"
HANDOFF_EVIDENCE = "docs/R7_C_ACTUAL_CONFIRMATION.md"
EVIDENCE_INDEX = "docs/R7_EVIDENCE_INDEX.jsonl"
START_LEDGER = (
    "| 轮次 | 实测 GPU-h | 累计 | 证据 |\n"
    "| --- | --- | --- | --- |\n"
    f"| S0 documentation startup | 0.0 | 0.0 | `{START_PLAN}` |\n"
)


def _startup_state(**overrides) -> str:
    values = {
        "current_node": "S0", "previous_node": "N5",
        "current_round_goal": NEW_CAMPAIGN,
        "previous_round_goal": HANDOFF_GOAL,
        "previous_round_evidence": HANDOFF_EVIDENCE,
        "cap_gpu_h": 0.0, "used_gpu_h": 0.0, "remaining_gpu_h": 0.0,
    }
    values.update(overrides)
    return _state(**values)


def build_new_campaign(
    tmp_path: Path, *, ledger: str = START_LEDGER, state: str | None = None,
    current_node_marker: str | None = "S0", previous_node_marker: str = "N5",
    previous_next_action: str = f"handoff to S0 at {NEW_CAMPAIGN}",
    records: list[dict] | None = None,
) -> Path:
    """Two independent masters; the new master is also its own current brief."""
    repo = build_repo(tmp_path)
    plan = repo / START_PLAN
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Documentation startup; no experiment, 0 GPU-h\n", encoding="utf-8")
    (repo / HANDOFF_EVIDENCE).write_text("# Handoff evidence fixture\n", encoding="utf-8")
    (repo / HANDOFF_GOAL).write_text(
        BRIEF_BODY.format(title="closeout", name=Path(HANDOFF_GOAL).name,
                          next_action=previous_next_action)
        + f"\n<!-- round-node: {previous_node_marker} -->\n", encoding="utf-8")
    marker = (f"\n<!-- round-node: {current_node_marker} -->\n"
              if current_node_marker is not None else "")
    (repo / NEW_CAMPAIGN).write_text(
        BRIEF_BODY.format(title="new campaign", name=Path(NEW_CAMPAIGN).name,
                          next_action="freeze S0 prerequisites; not complete")
        + marker + (state if state is not None else _startup_state())
        + "\n## GPU ledger (accounting-only)\n\n" + ledger, encoding="utf-8")
    if records is None:
        index = repo / EVIDENCE_INDEX
        records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
        records.append({"record_id": "handoff", "evidence_path": HANDOFF_EVIDENCE,
                        "evidence_commit": "a" * 40, "metrics": {"gpu_hours": 0.0}})
    (repo / EVIDENCE_INDEX).write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return repo


def test_selected_master_passes_without_changing_defaults_or_files(checker, tmp_path, capsys):
    repo = build_new_campaign(tmp_path)
    before = {path: path.read_bytes() for path in repo.rglob("*") if path.is_file()}
    default_report = checker.check(root=repo)
    selected_report = checker.check(root=repo, campaign_doc=NEW_CAMPAIGN)
    assert default_report["failures"] == [] and default_report["node"] == "N1"
    assert selected_report["failures"] == [] and selected_report["node"] == "S0"
    assert any("0.0 GPU-h" in note and "not machine-checked" in note
               for note in selected_report["notes"])
    assert checker.main(["--root", str(repo), "--campaign", NEW_CAMPAIGN, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == selected_report
    assert checker.main(["--root", str(repo), "--campaign", NEW_CAMPAIGN, "--quiet"]) == 0
    assert "failures=0" in capsys.readouterr().out
    assert checker.main(["--root", str(repo), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == default_report
    assert checker.check(repo) == default_report
    assert checker.CAMPAIGN_DOC == "docs/goals/main-model-v2-campaign.md"
    assert (checker.TOLERANCE, checker.RECORD_TOLERANCE, checker.GIT_TIMEOUT_SECONDS) \
        == (1e-3, 1e-4, 20)
    assert {path: path.read_bytes() for path in repo.rglob("*") if path.is_file()} == before


def test_missing_selected_master_names_it_without_default_fallback(checker, tmp_path, capsys):
    repo = build_repo(tmp_path)
    report = checker.check(repo, campaign_doc=NEW_CAMPAIGN)
    assert report["failures"] == [f"C-01 master plan missing: {NEW_CAMPAIGN}"]
    assert checker.main(["--root", str(repo), "--campaign", NEW_CAMPAIGN]) == 1
    output = capsys.readouterr().out
    assert NEW_CAMPAIGN in output and checker.CAMPAIGN_DOC not in output
    assert checker.check(repo)["failures"] == []


@pytest.mark.parametrize("state", [
    "no machine state here\n", "<!-- campaign-state: {broken JSON} -->\n",
])
def test_selected_master_missing_or_bad_state_fails(checker, tmp_path, state):
    repo = build_new_campaign(tmp_path, state=state)
    report = checker.check(repo, campaign_doc=NEW_CAMPAIGN)
    assert failures(report, "C-01")
    assert all(NEW_CAMPAIGN in finding and checker.CAMPAIGN_DOC not in finding
               for finding in report["failures"])


@pytest.mark.parametrize("key", [
    "current_node", "previous_node", "current_round_goal", "previous_round_goal",
    "previous_round_evidence", "cap_gpu_h", "used_gpu_h", "remaining_gpu_h",
])
def test_selected_master_missing_state_key_still_fails(checker, tmp_path, key):
    state = checker.parse_state(_startup_state())
    del state[key]
    repo = build_new_campaign(tmp_path, state=f"<!-- campaign-state: {json.dumps(state)} -->\n")
    report = checker.check(repo, campaign_doc=NEW_CAMPAIGN)
    assert any(NEW_CAMPAIGN in finding and key in finding for finding in failures(report, "C-01"))


@pytest.mark.parametrize("mutation, expected", [
    ({"ledger": START_LEDGER.replace("| 0.0 | 0.0 |", "| 0.1 | 0.0 |")}, "C-02"),
    ({"ledger": START_LEDGER.replace("| 0.0 | 0.0 |", "| 0.0 | 0.1 |")}, "C-02"),
    ({"state": _startup_state(remaining_gpu_h=1.0)}, "C-02"),
    ({"ledger": ""}, "C-02"),
    ({"ledger": START_LEDGER.replace(START_PLAN, "docs/plans/missing.md")}, "C-03"),
    ({"ledger": START_LEDGER.replace(f"`{START_PLAN}`", "no evidence pointer")}, "C-03"),
    ({"ledger": START_LEDGER.replace(f"`{START_PLAN}`", f"`{START_PLAN}` record:unknown")}, "C-03"),
    ({"previous_next_action": "stay on N5 without handoff"}, "C-04"),
    ({"previous_node_marker": "N4"}, "C-04"),
    ({"state": _startup_state(previous_round_goal="docs/goals/missing.md")}, "C-04"),
    ({"current_node_marker": "S1"}, "C-05"),
    ({"current_node_marker": None}, "C-05"),
    ({"state": _startup_state(current_round_goal="docs/goals/missing.md")}, "C-05"),
    ({"records": []}, "C-06"),
    ({"records": [{"record_id": "handoff", "evidence_path": HANDOFF_EVIDENCE}]}, "C-06"),
])
def test_selected_master_keeps_each_rule_strict(checker, tmp_path, mutation, expected):
    report = checker.check(build_new_campaign(tmp_path, **mutation), campaign_doc=NEW_CAMPAIGN)
    assert failures(report, expected), (mutation, report)


@pytest.mark.parametrize("missing", [HANDOFF_EVIDENCE, "docs/R7_EVIDENCE_INDEX.jsonl"])
def test_selected_master_missing_evidence_or_index_fails(checker, tmp_path, missing):
    repo = build_new_campaign(tmp_path)
    (repo / missing).unlink()
    assert failures(checker.check(repo, campaign_doc=NEW_CAMPAIGN), "C-06")


def test_selected_master_record_gpu_hours_mismatch_fails(checker, tmp_path):
    repo = build_new_campaign(tmp_path, ledger=START_LEDGER.replace(
        f"`{START_PLAN}`", f"`{START_PLAN}` record:handoff"), records=[
            {"record_id": "handoff", "evidence_path": HANDOFF_EVIDENCE,
             "evidence_commit": "a" * 40, "metrics": {"gpu_hours": 0.1}},
        ])
    assert failures(checker.check(repo, campaign_doc=NEW_CAMPAIGN), "C-03")


@pytest.mark.parametrize("mutation", [
    ("<!-- round-node: S0 -->", "<!-- round-node: S1 -->"),
    (f"name {NEW_CAMPAIGN}", "name docs/goals/wrong-identity.md"),
    ("## §2 交付物清单", "## absent-section"),
])
def test_selected_master_current_round_marker_identity_and_structure_fail(
    checker, tmp_path, mutation,
):
    repo = build_new_campaign(tmp_path)
    path = repo / NEW_CAMPAIGN
    original, replacement = mutation
    path.write_text(path.read_text(encoding="utf-8").replace(original, replacement), encoding="utf-8")
    assert failures(checker.check(repo, campaign_doc=NEW_CAMPAIGN), "C-05")


def test_selected_master_inaccessible_evidence_commit_fails(checker, tmp_path):
    repo = build_new_campaign(tmp_path, records=[
        {"record_id": "handoff", "evidence_path": HANDOFF_EVIDENCE,
         "evidence_commit": "0" * 40},
    ])
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    report = checker.check(repo, campaign_doc=NEW_CAMPAIGN)
    assert any("does not exist" in finding for finding in failures(report, "C-06"))


@pytest.mark.parametrize("campaign, detail", [
    ("../docs/goals/outside.md", "traversal"),
    ("docs/goals/../goals/master.md", "traversal"),
    ("docs/goals/../../docs/goals/master.md", "traversal"),
    ("docs/master.md", "docs/goals/*.md"),
    ("other/goals/master.md", "docs/goals/*.md"),
    ("docs/goals/master.txt", "docs/goals/*.md"),
    ("docs/goals/nested/master.md", "docs/goals/*.md"),
    ("docs/goals", "docs/goals/*.md"),
    ("", "docs/goals/*.md"),
])
def test_unsafe_campaign_path_fails_in_api_and_cli(checker, tmp_path, capsys, campaign, detail):
    repo = build_repo(tmp_path)
    report = checker.check(repo, campaign_doc=campaign)
    assert any(detail in finding for finding in failures(report, "C-01"))
    assert checker.main(["--root", str(repo), "--campaign", campaign]) == 1
    assert detail in capsys.readouterr().out


def test_absolute_campaign_path_is_rejected_even_inside_repo(checker, tmp_path, capsys):
    repo = build_new_campaign(tmp_path)
    campaign = str(repo / NEW_CAMPAIGN)
    assert any("not absolute" in finding for finding in failures(
        checker.check(repo, campaign_doc=campaign), "C-01"))
    assert checker.main(["--root", str(repo), "--campaign", campaign]) == 1
    assert "not absolute" in capsys.readouterr().out


@pytest.mark.parametrize("link", [NEW_CAMPAIGN, "docs/goals", "docs"])
def test_campaign_symlink_escape_is_rejected_before_reading(
    checker, tmp_path, capsys, monkeypatch, link,
):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "master.md" if link == NEW_CAMPAIGN else outside
    if link == NEW_CAMPAIGN:
        target.write_text("not a campaign; must never be read\n", encoding="utf-8")
    elif link == "docs/goals":
        (outside / Path(NEW_CAMPAIGN).name).write_text("outside master\n", encoding="utf-8")
    else:
        (outside / "goals").mkdir()
        (outside / "goals" / Path(NEW_CAMPAIGN).name).write_text("outside master\n", encoding="utf-8")
    destination = repo / link
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(target, target_is_directory=link != NEW_CAMPAIGN)

    def forbid_read(*args, **kwargs):
        pytest.fail("unsafe campaign must be rejected before any document is read")

    monkeypatch.setattr(Path, "read_text", forbid_read)
    report = checker.check(repo, campaign_doc=NEW_CAMPAIGN)
    assert any("resolves outside the repository" in finding for finding in failures(report, "C-01"))
    assert checker.main(["--root", str(repo), "--campaign", NEW_CAMPAIGN]) == 1
    assert "symlink escape" in capsys.readouterr().out
