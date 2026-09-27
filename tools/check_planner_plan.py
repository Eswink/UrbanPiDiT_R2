"""Validate a planner-produced plan before it is converted into repo artifacts.

Why this exists. A read-only planner sub-agent returns a plan as JSON and never
touches the repository (see `.agents/skills/planner-delegation/SKILL.md` and
decision 0012). The planner's own prompt is user-level configuration outside
version control, so it can drift; the JSON contract checked here is the one
thing the repository can actually hold fixed. A malformed or over-reaching plan
is therefore refused *before* it becomes a goal long-form, a `docs/plans/`
archive entry, or a task-queue row.

Two boundaries are mechanical rather than advisory:

* **Scientific criteria are not delegated.** `science_criteria_refs` must be
  *document pointers* (each containing a slash and ending in `.md`). A bare
  number or threshold is refused, which encodes "the planner cites where a
  frozen criterion lives; it does not author one".
* **Protected paths are not planned against.** `files_to_touch` may not name
  `data/raw|interim|processed` or an archival snapshot. The prefix list is
  derived from `tools/check_conventions.py` so the two can never drift apart.

Read-only: this module never writes. Exit codes: 0 = valid, 1 = violations,
2 = usage error.

    python tools/check_planner_plan.py --plan /tmp/plan.json
    python tools/check_planner_plan.py --plan /tmp/plan.json --quiet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent

# The plan is a work artifact, so the slug carries no number: numbering is
# assigned when the work is archived into docs/plans/NNNN-<slug>.md (R-032).
SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
OBJECTIVE_MAX_CHARS = 400

REQUIRED_KEYS = (
    "slug",
    "objective",
    "steps",
    "verification",
    "risks",
    "not_doing",
    "stop_conditions",
    "science_criteria_refs",
    "files_to_touch",
)
OPTIONAL_KEYS = ("open_questions", "budget_notes")
STEP_KEYS = ("id", "action", "verify", "depends_on", "files")

# Paths that must never appear in a plan's intended file list. The archival
# half is imported from the convention checker; the read-only data directories
# are stated in AGENTS.md hard constraints and R-004.
READONLY_DATA_PREFIXES = ("data/raw/", "data/interim/", "data/processed/")


def _load_archival_prefixes() -> tuple:
    """Derive archival prefixes from the checker so the two cannot drift."""
    spec = importlib.util.spec_from_file_location(
        "check_conventions", TOOLS / "check_conventions.py")
    module = importlib.util.module_from_spec(spec)
    # Loaded under a non-__main__ name, so the checker's CLI does not run.
    spec.loader.exec_module(module)
    return tuple(module.ARCHIVAL_PREFIXES)


def protected_prefixes() -> tuple:
    """All path prefixes a plan may not target."""
    return READONLY_DATA_PREFIXES + _load_archival_prefixes()


def _require_nonempty_str(value, where: str, out: list) -> bool:
    if not isinstance(value, str) or not value.strip():
        out.append(f"{where}: must be a nonempty string")
        return False
    return True


def _require_str_list(value, where: str, out: list, *, allow_empty: bool = True) -> bool:
    if not isinstance(value, list):
        out.append(f"{where}: must be a list")
        return False
    if not value and not allow_empty:
        out.append(f"{where}: must not be empty")
        return False
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            out.append(f"{where}[{index}]: must be a nonempty string")
            return False
    return True


def _check_slug(value, out: list) -> None:
    if not _require_nonempty_str(value, "slug", out):
        return
    if not SLUG_RE.fullmatch(value):
        out.append(f"slug: {value!r} must be kebab-case [a-z0-9-] with no number prefix")


def _check_objective(value, out: list) -> None:
    if not _require_nonempty_str(value, "objective", out):
        return
    if len(value) > OBJECTIVE_MAX_CHARS:
        out.append(f"objective: {len(value)} chars exceeds {OBJECTIVE_MAX_CHARS}")
    if "\n" in value:
        out.append("objective: must be a single line")


def _check_steps(value, out: list) -> None:
    if not isinstance(value, list) or not value:
        out.append("steps: must be a nonempty list")
        return
    seen: list = []
    for index, step in enumerate(value):
        where = f"steps[{index}]"
        if not isinstance(step, dict):
            out.append(f"{where}: must be an object")
            continue
        extra = sorted(set(step) - set(STEP_KEYS))
        if extra:
            out.append(f"{where}: unknown keys {extra}; allowed {list(STEP_KEYS)}")
        missing = [key for key in STEP_KEYS if key not in step]
        if missing:
            out.append(f"{where}: missing keys {missing}")
            continue
        step_id = step["id"]
        if not _require_nonempty_str(step_id, f"{where}.id", out):
            continue
        if step_id in seen:
            out.append(f"{where}.id: duplicate step id {step_id!r}")
            continue
        _require_nonempty_str(step["action"], f"{where}.action", out)
        _require_nonempty_str(step["verify"], f"{where}.verify", out)
        _require_str_list(step["files"], f"{where}.files", out)
        if _require_str_list(step["depends_on"], f"{where}.depends_on", out):
            for dep in step["depends_on"]:
                if dep not in seen:
                    # Requires the dependency to appear strictly earlier, so a
                    # cycle or a forward reference is unrepresentable.
                    out.append(f"{where}.depends_on: {dep!r} is not an earlier step id; "
                               "steps must be listed in dependency order")
        seen.append(step_id)


def _check_risks(value, out: list) -> None:
    if not isinstance(value, list) or not value:
        out.append("risks: must be a nonempty list (this project requires the costs)")
        return
    for index, item in enumerate(value):
        where = f"risks[{index}]"
        if not isinstance(item, dict):
            out.append(f"{where}: must be an object")
            continue
        extra = sorted(set(item) - {"risk", "mitigation"})
        if extra:
            out.append(f"{where}: unknown keys {extra}; allowed ['risk', 'mitigation']")
        for key in ("risk", "mitigation"):
            if key not in item:
                out.append(f"{where}: missing key {key!r}")
            else:
                _require_nonempty_str(item[key], f"{where}.{key}", out)


def _check_science_refs(value, out: list) -> None:
    """Criteria must be pointers to frozen documents, never authored values."""
    if not _require_str_list(value, "science_criteria_refs", out, allow_empty=False):
        return
    for index, ref in enumerate(value):
        if "/" not in ref or not ref.endswith(".md"):
            out.append(f"science_criteria_refs[{index}]: {ref!r} is not a document pointer; "
                       "scientific criteria live in the repo and are cited, not set here")


def _check_files(value, out: list) -> None:
    if not _require_str_list(value, "files_to_touch", out, allow_empty=False):
        return
    protected = protected_prefixes()
    for index, rel in enumerate(value):
        if Path(rel).is_absolute():
            out.append(f"files_to_touch[{index}]: {rel!r} must be repo-relative")
            continue
        if ".." in Path(rel).parts:
            out.append(f"files_to_touch[{index}]: {rel!r} must not traverse upward")
            continue
        normalised = rel if rel.endswith("/") else rel + "/"
        for prefix in protected:
            if normalised.startswith(prefix) or rel == prefix.rstrip("/"):
                out.append(f"files_to_touch[{index}]: {rel!r} is under the read-only "
                           f"prefix {prefix!r}")
                break


def validate_plan(plan) -> list:
    """Return a list of violation strings; empty means the plan is acceptable."""
    out: list = []
    if not isinstance(plan, dict):
        return ["plan: top level must be a JSON object"]
    unknown = sorted(set(plan) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        out.append(f"plan: unknown keys {unknown}; "
                   f"allowed {list(REQUIRED_KEYS + OPTIONAL_KEYS)}")
    missing = [key for key in REQUIRED_KEYS if key not in plan]
    if missing:
        out.append(f"plan: missing required keys {missing}")

    if "slug" in plan:
        _check_slug(plan["slug"], out)
    if "objective" in plan:
        _check_objective(plan["objective"], out)
    if "steps" in plan:
        _check_steps(plan["steps"], out)
    if "verification" in plan:
        _require_str_list(plan["verification"], "verification", out, allow_empty=False)
    if "risks" in plan:
        _check_risks(plan["risks"], out)
    if "not_doing" in plan:
        _require_str_list(plan["not_doing"], "not_doing", out, allow_empty=False)
    if "stop_conditions" in plan:
        _require_str_list(plan["stop_conditions"], "stop_conditions", out,
                          allow_empty=False)
    if "science_criteria_refs" in plan:
        _check_science_refs(plan["science_criteria_refs"], out)
    if "files_to_touch" in plan:
        _check_files(plan["files_to_touch"], out)
    if "open_questions" in plan:
        _require_str_list(plan["open_questions"], "open_questions", out)
    if "budget_notes" in plan:
        _require_nonempty_str(plan["budget_notes"], "budget_notes", out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a planner-produced plan JSON (read-only).")
    parser.add_argument("--plan", required=True, help="path to the plan JSON file")
    parser.add_argument("--quiet", action="store_true", help="print the verdict only")
    args = parser.parse_args(argv)

    path = Path(args.plan)
    if not path.is_file():
        print(f"error: plan file not found: {path}", file=sys.stderr)
        return 2
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"error: unreadable plan JSON: {exc}", file=sys.stderr)
        return 2

    failures = validate_plan(plan)
    report = {
        "plan": str(path),
        "verified": not failures,
        "required_keys": list(REQUIRED_KEYS),
        "protected_prefixes": list(protected_prefixes()),
        "failures": failures,
    }
    if args.quiet:
        print(json.dumps({"verified": report["verified"],
                          "failures": len(failures)}, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
