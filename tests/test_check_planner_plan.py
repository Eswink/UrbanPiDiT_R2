"""Tests for the planner-plan validator (R-026: every gate needs a falsification).

Two properties are guarded beyond the ordinary accept/reject cases:

* **no drift** - the validator's protected-path list must equal the convention
  checker's `ARCHIVAL_PREFIXES` plus the three read-only data directories, so a
  change in one place cannot silently weaken the other;
* **self-consistency** - the canonical example embedded in the delegation skill
  must be accepted by this validator with exactly the required key set, so the
  documented contract and the enforced contract cannot diverge.

The validator is read-only, and one test asserts that too.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SKILL = ROOT / ".agents" / "skills" / "planner-delegation" / "SKILL.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    assert (TOOLS / "check_planner_plan.py").is_file()
    return _load("check_planner_plan", TOOLS / "check_planner_plan.py")


@pytest.fixture(scope="module")
def checker():
    return _load("check_conventions_for_planner_test", TOOLS / "check_conventions.py")


def valid_plan() -> dict:
    """A minimal fully-conforming plan; each test mutates one aspect of it."""
    return {
        "slug": "demo-iteration-plan",
        "objective": "One short single-line objective for the next iteration.",
        "steps": [
            {"id": "probe", "action": "read the current state",
             "verify": "commands and their observed output", "depends_on": [],
             "files": []},
            {"id": "build", "action": "make the change",
             "verify": "targeted test passes", "depends_on": ["probe"],
             "files": ["tools/check_planner_plan.py"]},
        ],
        "verification": ["python -m pytest -q", "python tools/check_conventions.py"],
        "risks": [{"risk": "the contract drifts from actual use",
                   "mitigation": "self-consistency test on the embedded example"}],
        "not_doing": ["no scientific criteria changes"],
        "stop_conditions": ["budget or scope exceeded"],
        "science_criteria_refs": ["docs/R7_B1_BASELINE_AUDIT.md"],
        "files_to_touch": ["tools/check_planner_plan.py"],
    }


def test_valid_plan_passes(validator):
    assert validator.validate_plan(valid_plan()) == []


def test_optional_keys_are_accepted(validator):
    plan = valid_plan()
    plan["open_questions"] = ["which data range should the next iteration use?"]
    plan["budget_notes"] = "informational only; the budget ceiling is set by the repo"
    assert validator.validate_plan(plan) == []


# ------------------------------------------------------------------ rejections

@pytest.mark.parametrize("key", [
    "slug", "objective", "steps", "verification", "risks",
    "not_doing", "stop_conditions", "science_criteria_refs", "files_to_touch",
])
def test_missing_required_key_is_rejected(validator, key):
    plan = valid_plan()
    del plan[key]
    assert validator.validate_plan(plan), f"a plan without {key!r} must be rejected"


def test_unknown_top_level_key_is_rejected(validator):
    plan = valid_plan()
    plan["extra_idea"] = "not in the contract"
    failures = validator.validate_plan(plan)
    assert failures and "unknown keys" in failures[0]


def test_slug_must_be_kebab_case(validator):
    """A slug is kebab-case only; separators and capitals are refused.

    A leading number is *not* refused here - numbering is assigned when the work
    is archived (R-032), and a slug that happens to start with digits is still
    syntactically kebab-case. What must fail is any non-kebab spelling.
    """
    for bad in ("Demo_Plan", "demo plan", "Demo-Plan", "demo--plan", "-demo", "demo-"):
        plan = valid_plan()
        plan["slug"] = bad
        assert validator.validate_plan(plan), f"slug {bad!r} must be rejected"


def test_objective_must_be_single_line_and_bounded(validator):
    plan = valid_plan()
    plan["objective"] = "first line\nsecond line"
    assert validator.validate_plan(plan), "a multi-line objective must be rejected"
    plan = valid_plan()
    plan["objective"] = "x" * (validator.OBJECTIVE_MAX_CHARS + 1)
    assert validator.validate_plan(plan), "an over-long objective must be rejected"


def test_empty_steps_rejected(validator):
    plan = valid_plan()
    plan["steps"] = []
    assert validator.validate_plan(plan)


def test_step_requires_all_fields(validator):
    for missing in validator.STEP_KEYS:
        plan = valid_plan()
        del plan["steps"][0][missing]
        assert validator.validate_plan(plan), f"a step without {missing!r} must be rejected"


def test_step_unknown_key_rejected(validator):
    plan = valid_plan()
    plan["steps"][0]["estimate"] = "5 minutes"
    failures = validator.validate_plan(plan)
    assert failures and "unknown keys" in failures[0]


def test_duplicate_step_id_rejected(validator):
    plan = valid_plan()
    plan["steps"][1]["id"] = "probe"
    assert validator.validate_plan(plan)


def test_dependency_on_unknown_id_rejected(validator):
    plan = valid_plan()
    plan["steps"][1]["depends_on"] = ["nonexistent"]
    assert validator.validate_plan(plan)


def test_forward_dependency_rejected(validator):
    """A step may only depend on earlier steps, so cycles are unrepresentable."""
    plan = valid_plan()
    plan["steps"][0]["depends_on"] = ["build"]
    failures = validator.validate_plan(plan)
    assert failures and any("earlier step id" in item for item in failures)


def test_risks_need_a_mitigation(validator):
    plan = valid_plan()
    plan["risks"] = [{"risk": "something could go wrong"}]
    assert validator.validate_plan(plan)
    plan = valid_plan()
    plan["risks"] = [{"risk": "r", "mitigation": "   "}]
    assert validator.validate_plan(plan), "a blank mitigation must be rejected"


def test_empty_not_doing_and_stop_conditions_rejected(validator):
    for key in ("not_doing", "stop_conditions", "verification"):
        plan = valid_plan()
        plan[key] = []
        assert validator.validate_plan(plan), f"an empty {key!r} must be rejected"


def test_bare_threshold_in_science_refs_rejected(validator):
    """The planner cites where a frozen criterion lives; it never authors one."""
    for bad in ("threshold 1%", "RMSE <= 2.5", "docs/R7_B2_MULTISEED", "R7_TASK_QUEUE.md"):
        plan = valid_plan()
        plan["science_criteria_refs"] = [bad]
        failures = validator.validate_plan(plan)
        assert failures, f"{bad!r} is not a document pointer and must be rejected"


@pytest.mark.parametrize("path", [
    "data/raw/era5.nc",
    "data/interim/tmp.zarr",
    "data/processed/out.npz",
    "legacy_v531_full/model.py",
    "legacy_v6/anything.py",
])
def test_protected_paths_are_rejected(validator, path):
    plan = valid_plan()
    plan["files_to_touch"] = [path]
    failures = validator.validate_plan(plan)
    assert failures and "read-only" in failures[0]


def test_absolute_and_traversing_paths_rejected(validator):
    plan = valid_plan()
    plan["files_to_touch"] = ["/etc/passwd"]
    assert validator.validate_plan(plan), "an absolute path must be rejected"
    plan = valid_plan()
    plan["files_to_touch"] = ["../outside.md"]
    assert validator.validate_plan(plan), "an upward traversal must be rejected"


def test_non_object_plan_rejected(validator):
    assert validator.validate_plan(["not", "an", "object"])
    assert validator.validate_plan("just a string")


# ------------------------------------------------------- drift and consistency

def test_protected_prefixes_match_the_convention_checker(validator, checker):
    """The validator must not keep a second, drifting copy of the prefix list."""
    expected = ("data/raw/", "data/interim/", "data/processed/") + tuple(checker.ARCHIVAL_PREFIXES)
    assert validator.protected_prefixes() == expected


def test_skill_example_is_accepted_and_key_complete(validator):
    """The documented contract must equal the enforced contract."""
    assert SKILL.is_file(), f"the delegation skill is missing: {SKILL}"
    text = SKILL.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "the skill must embed a canonical example JSON block"
    example = json.loads(blocks[0])
    assert validator.validate_plan(example) == [], \
        "the example embedded in the skill must pass the validator"
    assert set(example) >= set(validator.REQUIRED_KEYS), \
        "the example must show every required key"
    assert not set(example) - set(validator.REQUIRED_KEYS + validator.OPTIONAL_KEYS), \
        "the example must not show keys outside the contract"


# --------------------------------------------------------------------- the CLI

def _write(tmp_path: Path, plan) -> Path:
    target = tmp_path / "plan.json"
    target.write_text(json.dumps(plan), encoding="utf-8")
    return target


def test_cli_exit_codes(validator, tmp_path, capsys):
    good = _write(tmp_path, valid_plan())
    assert validator.main(["--plan", str(good), "--quiet"]) == 0
    capsys.readouterr()
    bad = _write(tmp_path, {})
    assert validator.main(["--plan", str(bad), "--quiet"]) == 1
    capsys.readouterr()
    assert validator.main(["--plan", str(tmp_path / "absent.json")]) == 2
    capsys.readouterr()


def test_cli_rejects_malformed_json(validator, tmp_path, capsys):
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")
    assert validator.main(["--plan", str(target)]) == 2
    capsys.readouterr()


# ------------------------------------------------- the reply-transport contract

def test_load_accepts_bare_json(validator):
    plan, fenced = validator.load_plan_text(json.dumps(valid_plan()))
    assert plan["slug"] == "demo-iteration-plan" and fenced is False


@pytest.mark.parametrize("lang", ["json", "JSON", ""])
def test_load_accepts_one_enclosing_fence(validator, lang):
    """A real delegation returned one fence despite the planner's instruction.

    Tolerating exactly one enclosing fence keeps the reply-to-object contract
    usable without accepting prose around the plan.
    """
    body = json.dumps(valid_plan(), ensure_ascii=False)
    text = f"```{lang}\n{body}\n```"
    plan, fenced = validator.load_plan_text(text)
    assert plan["slug"] == "demo-iteration-plan" and fenced is True


@pytest.mark.parametrize("text", [
    "Here is the plan:\n```json\n{}\n```",
    "```json\n{}\n```\nHope this helps!",
    "```json\n{}\n```\n```json\n{}\n```",
    "```json\n{}\n",
    "plain prose with no JSON at all",
])
def test_load_rejects_prose_and_extra_fences(validator, text):
    with pytest.raises(ValueError):
        validator.load_plan_text(text)


def test_cli_accepts_fenced_plan_and_reports_it(validator, tmp_path, capsys):
    body = json.dumps(valid_plan(), ensure_ascii=False)
    target = tmp_path / "fenced.json"
    target.write_text(f"```json\n{body}\n```", encoding="utf-8")
    assert validator.main(["--plan", str(target), "--quiet"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == {"verified": True, "fence_stripped": True, "failures": 0}


def test_cli_rejects_fenced_plan_with_prose(validator, tmp_path, capsys):
    target = tmp_path / "prose.json"
    target.write_text("Sure! Here you go:\n```json\n{}\n```", encoding="utf-8")
    assert validator.main(["--plan", str(target)]) == 2
    capsys.readouterr()


def test_cli_quiet_output_is_json_with_the_verdict(validator, tmp_path, capsys):
    good = _write(tmp_path, valid_plan())
    validator.main(["--plan", str(good), "--quiet"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == {"verified": True, "fence_stripped": False, "failures": 0}


def test_validator_is_read_only(validator, tmp_path):
    """The validator must not create, move or delete anything."""
    _write(tmp_path, valid_plan())
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    validator.main(["--plan", str(tmp_path / "plan.json"), "--quiet"])
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert before == after
