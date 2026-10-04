"""Check the campaign master plan against the round documents it drives (decision 0025).

Why this exists. The R7 main-model line is driven by a campaign master plan
(``docs/goals/main-model-v2-campaign.md``) plus one goal brief per round, and
before this tool every cross-check between them was manual: the GPU ledger was
hand-written prose nobody re-added, and nothing compared "the master plan's
current node" with "the previous round's next action". A session that skipped a
round, mis-added the budget or started from a stale next-action could not be
caught by anything in CI.

This module checks the mechanically decidable part only:

- ``C-01`` the master plan carries a parseable ``campaign-state`` block with the
  required keys;
- ``C-02`` the ledger table adds up: rows sum to ``used_gpu_h``,
  ``cap_gpu_h - used_gpu_h`` equals ``remaining_gpu_h``, and the cumulative
  column is the running sum (tolerance 1e-3);
- ``C-03`` every ledger row cites an existing ``docs/...md`` file, and when the
  row names ``record:<id>`` that evidence-index record exists and its
  ``gpu_hours`` agrees (tolerance 1e-4); rows with no index record are listed as
  ``unbacked`` notes rather than silently accepted;
- ``C-04`` the previous round brief exists and its progress block's next action
  names the master plan's current node;
- ``C-05`` the current round brief exists, declares ``<!-- round-node: X -->``
  equal to the current node, and passes the structural brief check;
- ``C-06`` the previous round's evidence page is registered in
  ``docs/R7_EVIDENCE_INDEX.jsonl`` and the recorded ``evidence_commit`` exists in git.

It never writes, runs no shell, and does **not** judge scientific direction:
passing means "nothing here contradicts the master plan", not "the next node is
right". Rule ids use the ``C-xx`` namespace so they cannot be confused with
``docs/rules``' ``R-0xx`` or ``check_goal_brief.py``'s ``G-xx``.

Exit codes: 0 = no hard drift, 1 = hard drift, 2 = usage error.

    .venv/bin/python tools/check_campaign_state.py
    .venv/bin/python tools/check_campaign_state.py --json
    .venv/bin/python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent

CAMPAIGN_DOC = "docs/goals/main-model-v2-campaign.md"
INDEX = "docs/R7_EVIDENCE_INDEX.jsonl"

STATE_RE = re.compile(r"<!--\s*campaign-state:\s*(\{.*?\})\s*-->", re.S)
ROUND_NODE_RE = re.compile(r"<!--\s*round-node:\s*([A-Za-z0-9_.-]+)\s*-->")
DOC_POINTER_RE = re.compile(r"docs/[\w./-]+\.md")
RECORD_REF_RE = re.compile(r"record:([A-Za-z0-9_.-]+)")
LEDGER_ROW_RE = re.compile(
    r"^\|\s*(?P<name>[^|]+?)\s*\|\s*\*{0,2}(?P<gpu>\d+(?:\.\d+)?)\*{0,2}\s*\|"
    r"\s*\*{0,2}(?P<cum>[\d.]+|—|-)\*{0,2}\s*\|\s*(?P<evidence>[^|]+?)\s*\|\s*$",
    re.M)

REQUIRED_STATE_KEYS = ("current_node", "previous_node", "current_round_goal",
                       "previous_round_goal", "previous_round_evidence",
                       "cap_gpu_h", "used_gpu_h", "remaining_gpu_h")
TOLERANCE = 1e-3
RECORD_TOLERANCE = 1e-4
GIT_TIMEOUT_SECONDS = 20


def _load_brief_checker():
    """Load tools/check_goal_brief.py in-process instead of forking a shell.

    Imported the same way that checker imports check_conventions.py: the
    structural contract lives in one place and this tool cannot drift from it.
    """
    spec = importlib.util.spec_from_file_location(
        "check_goal_brief", TOOLS / "check_goal_brief.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_state(text: str) -> dict | None:
    match = STATE_RE.search(text)
    if not match:
        return None
    try:
        state = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return state if isinstance(state, dict) else None


def ledger_rows(text: str) -> list[dict]:
    """Ledger rows carrying a GPU number; the total row is skipped."""
    rows = []
    for match in LEDGER_ROW_RE.finditer(text):
        if match.group("cum") in ("—", "-"):
            continue
        rows.append({
            "name": match.group("name").strip(),
            "gpu_h": float(match.group("gpu")),
            "cumulative": float(match.group("cum")),
            "evidence": match.group("evidence").strip(),
        })
    return rows


def load_index(root: Path) -> dict[str, dict]:
    path = root / INDEX
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record.get("record_id"), str):
            records[record["record_id"]] = record
    return records


def record_gpu_hours(record: dict) -> float | None:
    metrics = record.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("gpu_hours"), (int, float)):
        return float(metrics["gpu_hours"])
    if isinstance(record.get("gpu_hours"), (int, float)):
        return float(record["gpu_hours"])
    return None


def commit_exists(root: Path, sha: str) -> bool | None:
    """True/False when git can answer, None when it cannot be asked at all."""
    if not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=root, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode == 0


def check_state_block(campaign_text: str, failures: list[str],
                      notes: list[str], campaign_doc: str = CAMPAIGN_DOC) -> dict | None:
    """C-01: the machine-readable state block exists and carries its keys."""
    state = parse_state(campaign_text)
    if state is None:
        failures.append(f"C-01 no parseable campaign-state block in {campaign_doc}")
        return None
    missing = [key for key in REQUIRED_STATE_KEYS if key not in state]
    if missing:
        failures.append(f"C-01 campaign-state block in {campaign_doc} lacks keys: "
                        f"{', '.join(missing)}")
    return state


def check_ledger(campaign_text: str, state: dict, records: dict[str, dict],
                 root: Path, failures: list[str], notes: list[str]) -> None:
    """C-02 arithmetic, C-03 evidence pointers and index-backed values."""
    rows = ledger_rows(campaign_text)
    if not rows:
        failures.append("C-02 no ledger rows found in the master plan")
        return
    total = round(sum(row["gpu_h"] for row in rows), 6)
    used = float(state.get("used_gpu_h", float("nan")))
    cap = float(state.get("cap_gpu_h", float("nan")))
    remaining = float(state.get("remaining_gpu_h", float("nan")))
    if abs(total - used) > TOLERANCE:
        failures.append(f"C-02 ledger rows sum to {total}, state says used_gpu_h={used}")
    if abs((cap - used) - remaining) > TOLERANCE:
        failures.append(
            f"C-02 cap - used = {round(cap - used, 6)}, state says remaining_gpu_h={remaining}")
    running = 0.0
    for row in rows:
        running = round(running + row["gpu_h"], 6)
        if abs(running - row["cumulative"]) > TOLERANCE:
            failures.append(f"C-02 cumulative for '{row['name']}' is {row['cumulative']}, "
                            f"running sum is {running}")
    for row in rows:
        pointers = DOC_POINTER_RE.findall(row["evidence"])
        if not pointers:
            failures.append(f"C-03 ledger row '{row['name']}' cites no docs/...md evidence")
        for pointer in pointers:
            if not (root / pointer).is_file():
                failures.append(f"C-03 ledger row '{row['name']}' cites missing file {pointer}")
        _check_row_record(row, records, notes, failures)


def _check_row_record(row: dict, records: dict[str, dict], notes: list[str],
                      failures: list[str]) -> None:
    refs = RECORD_REF_RE.findall(row["evidence"])
    if not refs:
        notes.append(f"C-03 ledger row '{row['name']}' ({row['gpu_h']} GPU-h) has no "
                     f"evidence-index record: value not machine-checked")
        return
    for record_id in refs:
        record = records.get(record_id)
        if record is None:
            failures.append(f"C-03 ledger row '{row['name']}' cites unknown record "
                            f"'{record_id}'")
            continue
        gpu_hours = record_gpu_hours(record)
        if gpu_hours is None:
            notes.append(f"C-03 record '{record_id}' carries no gpu_hours metric")
        elif abs(gpu_hours - row["gpu_h"]) > RECORD_TOLERANCE:
            failures.append(f"C-03 ledger row '{row['name']}' says {row['gpu_h']} GPU-h, "
                            f"record '{record_id}' says {gpu_hours}")


def check_previous_round(state: dict, records: dict[str, dict], root: Path,
                         failures: list[str], notes: list[str]) -> None:
    """C-04 the previous round points here; C-06 its evidence is registered."""
    previous_goal = state.get("previous_round_goal")
    if isinstance(previous_goal, str) and previous_goal:
        previous_path = root / previous_goal
        if not previous_path.is_file():
            failures.append(f"C-04 previous round brief missing: {previous_goal}")
        else:
            _check_previous_next_action(previous_path, state, failures, notes)
    _check_previous_evidence(state, records, root, failures, notes)


def _check_previous_next_action(previous_path: Path, state: dict,
                                failures: list[str], notes: list[str]) -> None:
    text = previous_path.read_text(encoding="utf-8")
    graded = True
    marker = ROUND_NODE_RE.search(text)
    if marker is None:
        notes.append(f"C-04 {previous_path.name} predates the round-node marker; node "
                     f"identity checked through its next action only")
    elif isinstance(state.get("previous_node"), str) \
            and marker.group(1) != state["previous_node"]:
        failures.append(f"C-04 {previous_path.name} declares round-node {marker.group(1)}, "
                        f"master plan says previous_node={state['previous_node']}")
    if state.get("previous_round_predates_mechanism") is True:
        graded = False
        notes.append(f"C-04 {previous_path.name} predates the node ids in this mechanism; "
                     f"the next action is not graded against current_node "
                     f"(declared by previous_round_predates_mechanism)")
    action_tail = text[text.rfind("下一动作"):] if "下一动作" in text else ""
    if not action_tail:
        failures.append(f"C-04 {previous_path.name} has no 下一动作 entry")
    elif graded and isinstance(state.get("current_node"), str) \
            and state["current_node"] not in action_tail:
        failures.append(
            f"C-04 {previous_path.name}'s next action does not name the master plan's "
            f"current node {state['current_node']}")


def _check_previous_evidence(state: dict, records: dict[str, dict], root: Path,
                             failures: list[str], notes: list[str]) -> None:
    evidence = state.get("previous_round_evidence")
    if not isinstance(evidence, str) or not evidence:
        return
    if not (root / evidence).is_file():
        failures.append(f"C-06 previous round evidence missing: {evidence}")
    registered = [record for record in records.values()
                  if record.get("evidence_path") == evidence]
    if not registered:
        failures.append(f"C-06 previous round evidence {evidence} is not registered in "
                        f"{INDEX}")
    for record in registered:
        commit = record.get("evidence_commit")
        if not isinstance(commit, str) or not commit:
            failures.append(f"C-06 record '{record.get('record_id')}' has no evidence_commit")
            continue
        exists = commit_exists(root, commit)
        if exists is None:
            notes.append(f"C-06 commit existence for '{commit[:12]}' not checkable here")
        elif not exists:
            failures.append(f"C-06 record '{record.get('record_id')}' binds commit "
                            f"{commit[:12]} which does not exist in this repository")


def check_current_round(state: dict, root: Path, failures: list[str]) -> None:
    """C-05 the current round brief exists, declares its node and is well formed."""
    current_goal = state.get("current_round_goal")
    if not isinstance(current_goal, str) or not current_goal:
        return
    current_path = root / current_goal
    if not current_path.is_file():
        failures.append(f"C-05 current round brief missing: {current_goal}")
        return
    text = current_path.read_text(encoding="utf-8")
    marker = ROUND_NODE_RE.search(text)
    current_node = state.get("current_node")
    if marker is None:
        failures.append(f"C-05 {current_goal} carries no <!-- round-node: X --> marker")
    elif marker.group(1) != current_node:
        failures.append(f"C-05 {current_goal} declares round-node {marker.group(1)}, "
                        f"master plan says current_node={current_node}")
    brief_report = _load_brief_checker().check_brief(current_path, root=root)
    if brief_report["failures"]:
        detail = "; ".join(f"{finding['rule']} {finding['detail']}"
                           for finding in brief_report["failures"])
        failures.append(f"C-05 {current_goal} fails the brief structure check: {detail}")


def _campaign_path(root: Path, campaign_doc: str) -> Path:
    """Accept only repository-relative docs/goals/*.md without symlink escape."""
    relative = Path(campaign_doc)
    if relative.is_absolute():
        raise ValueError("campaign path must be repository-relative, not absolute")
    if ".." in relative.parts:
        raise ValueError("campaign path must not contain '..' traversal")
    if len(relative.parts) != 3 or relative.parts[:2] != ("docs", "goals") \
            or relative.suffix != ".md":
        raise ValueError("campaign path must name docs/goals/*.md")
    path = root / relative
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"campaign path cannot be resolved safely: {exc}") from exc
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("campaign path resolves outside the repository (symlink escape)")
    return resolved


def check(root: Path = ROOT, campaign_doc: str = CAMPAIGN_DOC) -> dict:
    failures: list[str] = []
    notes: list[str] = []
    try:
        campaign_path = _campaign_path(root, campaign_doc)
    except ValueError as exc:
        return {"failures": [f"C-01 invalid master plan path {campaign_doc!r}: {exc}"],
                "notes": [], "node": None}
    if not campaign_path.is_file():
        return {"failures": [f"C-01 master plan missing: {campaign_doc}"],
                "notes": [], "node": None}
    campaign_text = campaign_path.read_text(encoding="utf-8")
    state = check_state_block(campaign_text, failures, notes, campaign_doc)
    if state is None:
        return {"failures": failures, "notes": notes, "node": None}
    records = load_index(root)
    check_ledger(campaign_text, state, records, root, failures, notes)
    check_previous_round(state, records, root, failures, notes)
    check_current_round(state, root, failures)
    return {"failures": failures, "notes": notes, "node": state.get("current_node")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT),
                        help="repository root (default: this checkout)")
    parser.add_argument("--campaign", default=CAMPAIGN_DOC,
                        help=f"repository-relative docs/goals/*.md (default: {CAMPAIGN_DOC})")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the summary line")
    parser.add_argument("--json", action="store_true",
                        help="emit the full report as JSON")
    args = parser.parse_args(argv)

    report = check(Path(args.root), campaign_doc=args.campaign)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.quiet:
        print(f"campaign_state failures={len(report['failures'])} notes={len(report['notes'])}")
    else:
        for finding in report["failures"]:
            print(f"FAIL {finding}")
        for note in report["notes"]:
            print(f"note {note}")
        print(f"campaign_state node={report['node']} failures={len(report['failures'])} "
              f"notes={len(report['notes'])}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
