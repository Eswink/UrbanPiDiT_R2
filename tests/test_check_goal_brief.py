"""Tests for the goal-brief checker (the mechanical part of R-034).

Two properties are guarded beyond the ordinary accept/reject cases:

* **counterproofs** - every mechanical rule must have a brief that violates it
  and is rejected, so the checker cannot pass everything;
* **no drift** - the canonical example embedded in the ``goal-loop`` skill must
  be accepted by this checker, and every rule id the checker enforces must be
  documented in the skill, so the documented contract and the enforced contract
  cannot diverge.

All briefs are written under ``tmp_path``; the checker itself is read-only and
one test asserts that by re-reading the real brief it was pointed at.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SKILL = ROOT / ".agents" / "skills" / "goal-loop" / "SKILL.md"
GOALS = ROOT / "docs" / "goals"
DEFAULT_NAME = "example-goal-brief.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    assert (TOOLS / "check_goal_brief.py").is_file()
    return _load("check_goal_brief", TOOLS / "check_goal_brief.py")


def brief_text(
    *,
    slug: str = "example-goal-brief",
    objective_lines: int = 1,
    objective_repeat: int = 0,
    cite_brief: bool = True,
    deliverables: bool = True,
    criteria: bool = True,
    criteria_pointer: bool = True,
    budget: bool = True,
    progress: bool = True,
) -> str:
    """A minimal conforming brief; each test mutates exactly one aspect."""
    body = f"This objective is short enough and names docs/goals/{slug}.md so the " \
           f"verifier can find the full criteria."
    if not cite_brief:
        body = "This objective is short enough but names no file at all."
    body += "X" * objective_repeat
    objective = "\n".join([f"> {body}"] * objective_lines)
    parts = [f"# 目标：{slug}", "", "## 0. objective（实测 1 字符，上限 4000）", "", objective, ""]
    if deliverables:
        parts += ["## 2. 交付物清单", "", "| # | 交付物 | 证据形态 |", "| --- | --- | --- |",
                  "| D1 | 一件可核对的事 | 测试名与结果 |", ""]
    if criteria:
        parts += ["## 3. 判据与证据来源", ""]
        parts += ["> 判据引用 docs/R7_B2_MULTISEED.md（文档指针）。" if criteria_pointer
                  else "> 判据就是我觉得比之前好。", ""]
    if budget:
        parts += ["## 5. 预算与停止", "", "| 项 | 值 |", "| --- | --- |",
                  "| 本轮上限 | 0.5 GPU-h |", "| 停止条件 | 超预算即停 |", ""]
    if progress:
        parts += ["## 8. 进度", "", "- 状态：active", "- 已完成：无", "- 未做：全部",
                  "- 下一动作：写 D1 的测试", ""]
    return "\n".join(parts)


def write_brief(tmp_path: Path, *, name: str = DEFAULT_NAME, text: str | None = None,
                **kwargs) -> Path:
    slug = name.removesuffix(".md")
    if text is None:
        text = brief_text(slug=slug, **kwargs)
    path = tmp_path / "docs" / "goals" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def rules(result: dict, key: str) -> set[str]:
    return {finding["rule"] for finding in result[key]}


def skill_example() -> str:
    """The fenced ``markdown`` example embedded in the goal-loop skill."""
    text = SKILL.read_text(encoding="utf-8")
    match = re.search(r"```markdown\n(?P<body>.*?)\n```", text, re.DOTALL)
    assert match, "the goal-loop skill must embed a ```markdown example"
    return textwrap.dedent(match.group("body"))


def test_valid_brief_passes(checker, tmp_path):
    result = checker.check_brief(write_brief(tmp_path), root=tmp_path)
    assert result["failures"] == []


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"criteria": False}, "G-07"),
        ({"criteria_pointer": False}, "G-07"),
        ({"objective_lines": 2}, "G-03"),
        ({"objective_repeat": 4000}, "G-04"),
        ({"cite_brief": False}, "G-05"),
        ({"deliverables": False}, "G-06"),
        ({"budget": False}, "G-08"),
    ],
)
def test_each_rule_rejects_its_violation(checker, tmp_path, kwargs, expected):
    path = write_brief(tmp_path, **kwargs)
    result = checker.check_brief(path, root=tmp_path)
    assert expected in rules(result, "failures"), (kwargs, result)


def test_non_kebab_name_is_rejected(checker, tmp_path):
    path = write_brief(tmp_path, name="Example_Brief.md")
    assert "G-01" in rules(checker.check_brief(path, root=tmp_path), "failures")


def test_banned_token_name_is_rejected(checker, tmp_path):
    path = write_brief(tmp_path, name="example-draft.md")
    assert "G-01" in rules(checker.check_brief(path, root=tmp_path), "failures")


def test_brief_outside_goals_dir_is_rejected(checker, tmp_path):
    path = tmp_path / "docs" / "example-brief.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief_text(), encoding="utf-8")
    assert "G-01" in rules(checker.check_brief(path, root=tmp_path), "failures")


def test_missing_progress_block_is_advisory_only(checker, tmp_path):
    path = write_brief(tmp_path, progress=False)
    result = checker.check_brief(path, root=tmp_path)
    assert result["failures"] == []
    assert "A-01" in rules(result, "advisories")


def test_readme_is_exempt_from_the_name_rule(checker, tmp_path):
    path = tmp_path / "docs" / "goals" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# index\n", encoding="utf-8")
    assert "G-01" not in rules(checker.check_brief(path, root=tmp_path), "failures")


def test_skill_example_is_accepted(checker, tmp_path):
    """Self-consistency: the documented example must pass the enforced rules."""
    path = write_brief(tmp_path, text=skill_example())
    result = checker.check_brief(path, root=tmp_path)
    assert result["failures"] == [], result


def test_skill_documents_every_mechanical_rule(checker):
    """Anti-drift: a rule the checker enforces must be documented in the skill."""
    source = (TOOLS / "check_goal_brief.py").read_text(encoding="utf-8")
    enforced = set(re.findall(r"[GA]-\d{2}", source))
    assert enforced >= {"G-01", "G-02", "G-03", "G-04", "G-05", "G-06", "G-07",
                        "G-08", "A-01", "A-02"}
    skill = SKILL.read_text(encoding="utf-8")
    missing = sorted(rule for rule in enforced if rule not in skill)
    assert missing == [], f"rules not documented in the skill: {missing}"


def test_current_briefs_written_to_the_convention_pass(checker):
    for name in ("m1-and-rw-a-iteration.md", "v2-round-two-attribution.md"):
        result = checker.check_brief(GOALS / name)
        assert result["failures"] == [], (name, result)


def test_exit_codes_and_json_report(checker, tmp_path, capsys):
    good = write_brief(tmp_path, name="good-brief.md")
    base = ["--root", str(tmp_path)]
    assert checker.main(["--brief", str(good), *base, "--quiet"]) == 0
    bad = write_brief(tmp_path, name="bad-brief.md", criteria=False)
    assert checker.main(["--brief", str(bad), *base, "--quiet"]) == 1
    capsys.readouterr()
    assert checker.main(["--brief", str(bad), *base, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["failures"] >= 1


def test_checker_does_not_modify_the_brief_it_reads(checker):
    path = GOALS / "v2-round-two-attribution.md"
    before = path.read_text(encoding="utf-8")
    checker.check_brief(path)
    assert path.read_text(encoding="utf-8") == before
