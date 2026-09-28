"""Check a goal long-form brief against the mechanically decidable part of R-034.

Why this exists. R-034 (docs/rules/artifact-storage.md) says a goal prompt is a
long-form file under ``docs/goals/`` plus a short objective that only repeats the
essentials and points at that file. The rule is declared "manual" because the
*content* of an objective cannot be judged mechanically - but its *structure*
can: the objective must sit in a quoting block, must be a single paragraph, must
fit the client's 4000-character objective limit, and must name its own brief,
because the completion verifier reads only the conversation and a criterion that
lives in a file it cannot open is a criterion that does not exist.

This module therefore checks structure only, and it never writes. It is NOT a
convention rule: it adds no blocking rule, is not wired into the Stop hook or CI,
and its rule ids (``G-01``...) deliberately differ from the ``R-0xx`` namespace
owned by docs/rules/. Failures and advisories are separated so that briefs
written before the convention can be reported without being retro-fitted - the
existing briefs are evidence and are not rewritten.

Exit codes: 0 = no failures, 1 = failures, 2 = usage error.

    python tools/check_goal_brief.py --brief docs/goals/v2-round-two-attribution.md
    python tools/check_goal_brief.py --brief docs/goals --quiet
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

# The client's limit is counted in code points, not bytes; a 1373-character
# objective is a normal brief and a 4001-character one is rejected at the door.
OBJECTIVE_MAX_CHARS = 4000

GOALS_DIR = "docs/goals/"
KEBAB = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.md")
HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
QUOTE_LINE = re.compile(r"^>\s?")
DOC_POINTER = re.compile(r"docs/[\w./-]+\.md")

# Headings that must exist for the brief to be usable by an executor and by a
# verifier that cannot open files. Matching is by substring so the wording can
# change without breaking the contract.
OBJECTIVE_HEADING = "objective"
DELIVERABLES_HEADING = "交付物"
CRITERIA_HEADING = "判据"
BUDGET_HEADING = "预算"
STOP_HEADING = "停止"
PROGRESS_HEADING = "进度"
CHAR_COUNT_MARKER = "字符"


def _load_conventions():
    """Load tools/check_conventions.py without running its CLI.

    The banned-token list is imported rather than re-typed so this checker and
    the blocking convention rule cannot drift apart on what a bad file name is.
    """
    spec = importlib.util.spec_from_file_location(
        "check_conventions", TOOLS / "check_conventions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def banned_token_pattern() -> re.Pattern:
    return _load_conventions()._BANNED_TOKEN_RE


def _sections(text: str) -> list[tuple[str, str]]:
    """Split into (heading, body) pairs, keeping any preamble as ''."""
    lines = text.splitlines()
    out: list[tuple[str, list[str]]] = [("", [])]
    for line in lines:
        match = HEADING.match(line)
        if match:
            out.append((match.group(2), []))
        else:
            out[-1][1].append(line)
    return [(heading, "\n".join(body)) for heading, body in out]


def _quote_blocks(body: str) -> list[list[str]]:
    """Contiguous runs of quoted lines, each as its list of stripped lines."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if QUOTE_LINE.match(line):
            if current is None:
                current = []
                blocks.append(current)
            current.append(QUOTE_LINE.sub("", line))
        else:
            current = None
    return blocks


def _first_paragraph(block: list[str]) -> str:
    """The objective paragraph: quoted text up to the first blank quoted line."""
    paragraph: list[str] = []
    for line in block:
        if not line.strip():
            break
        paragraph.append(line.rstrip())
    return "\n".join(paragraph)


def check_brief(path: Path, *, root: Path = ROOT) -> dict:
    """Return ``{"failures": [...], "advisories": [...]}`` for one brief."""
    failures: list[dict] = []
    advisories: list[dict] = []

    def fail(rule: str, detail: str) -> None:
        failures.append({"rule": rule, "detail": detail})

    def advise(rule: str, detail: str) -> None:
        advisories.append({"rule": rule, "detail": detail})

    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        fail("G-01", "brief must live inside the repository")
        rel = path.as_posix()

    name = path.name
    if name != "README.md":
        if not rel.startswith(GOALS_DIR):
            fail("G-01", f"brief must live under {GOALS_DIR}")
        if not KEBAB.fullmatch(name):
            fail("G-01", "brief name must be kebab-case.md")
        elif banned_token_pattern().search(name):
            fail("G-01", "brief name must not use a banned token (R-043 word list)")

    if not path.is_file():
        fail("G-02", "brief file is missing")
        return {"failures": failures, "advisories": advisories}

    text = path.read_text(encoding="utf-8")
    sections = _sections(text)

    objective_sections = [(heading, body) for heading, body in sections
                          if OBJECTIVE_HEADING in heading.lower()]
    if not objective_sections:
        fail("G-02", "no objective section (heading must contain 'objective')")
        objective_heading, objective_body = "", ""
    else:
        objective_heading, objective_body = objective_sections[0]
        blocks = _quote_blocks(objective_body)
        if not blocks:
            fail("G-02", "objective section has no quoted objective block")
        else:
            paragraph = _first_paragraph(blocks[0])
            if not paragraph.strip():
                fail("G-02", "objective block starts with an empty line")
            else:
                if "\n" in paragraph:
                    fail("G-03", "objective must be a single paragraph (one line)")
                if len(paragraph) > OBJECTIVE_MAX_CHARS:
                    fail("G-04", f"objective is {len(paragraph)} code points, "
                                 f"over the {OBJECTIVE_MAX_CHARS} limit")
                if rel not in paragraph:
                    fail("G-05", f"objective must name its own brief ({rel}) so the "
                                 f"verifier can locate the full criteria")

    headings = [heading for heading, _ in sections]
    if not any(DELIVERABLES_HEADING in heading for heading in headings):
        fail("G-06", "no deliverables section (heading must contain 交付物)")
    criteria_bodies = [body for heading, body in sections
                       if CRITERIA_HEADING in heading]
    if not criteria_bodies:
        fail("G-07", "no criteria section (heading must contain 判据)")
    elif not DOC_POINTER.search(criteria_bodies[0]):
        fail("G-07", "criteria section must cite at least one docs/...md source")
    if not any(BUDGET_HEADING in heading or STOP_HEADING in heading
               for heading in headings):
        fail("G-08", "no budget/stop section (heading must contain 预算 or 停止)")

    if not any(PROGRESS_HEADING in heading for heading in headings):
        advise("A-01", "no progress block (状态机 + 已完成/未做/下一动作) yet")
    if objective_body and CHAR_COUNT_MARKER not in objective_body \
            and CHAR_COUNT_MARKER not in objective_heading:
        advise("A-02", "objective section does not state its measured length")

    return {"failures": failures, "advisories": advisories}


def check_paths(paths, *, root: Path = ROOT) -> dict:
    """Check every ``--brief`` target; directories expand to ``*.md``."""
    briefs: dict[str, dict] = {}
    root = Path(root)
    for target in paths:
        target = Path(target)
        candidates = (sorted(path for path in target.glob("*.md")
                             if path.name != "README.md")
                      if target.is_dir() else [target])
        for path in candidates:
            try:
                rel = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = path.as_posix()
            briefs[rel] = check_brief(path, root=root)
    failures = sum(len(entry["failures"]) for entry in briefs.values())
    advisories = sum(len(entry["advisories"]) for entry in briefs.values())
    return {"briefs": briefs, "failures": failures, "advisories": advisories}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--brief", action="append", default=[],
                        help="goal brief file or directory (repeatable); defaults "
                             f"to {GOALS_DIR}")
    parser.add_argument("--root", default=str(ROOT),
                        help="repository root the briefs are relative to "
                             "(default: this checkout)")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the summary line")
    parser.add_argument("--json", action="store_true",
                        help="emit the full report as JSON")
    args = parser.parse_args(argv)

    root = Path(args.root)
    targets = args.brief or [str(root / GOALS_DIR.rstrip("/"))]
    report = check_paths(targets, root=root)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif not args.quiet:
        for rel, entry in report["briefs"].items():
            for finding in entry["failures"]:
                print(f"{rel}: {finding['rule']} FAIL {finding['detail']}")
            for finding in entry["advisories"]:
                print(f"{rel}: {finding['rule']} note {finding['detail']}")
        print(f"briefs={len(report['briefs'])} failures={report['failures']} "
              f"advisories={report['advisories']}")
    else:
        print(f"briefs={len(report['briefs'])} failures={report['failures']}")

    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
