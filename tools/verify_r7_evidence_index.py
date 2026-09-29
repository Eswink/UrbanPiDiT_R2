"""Validate a read-only R7 evidence index and render a human brief.

The index is an explicit handoff between frozen evidence and future planning.
This module never recomputes metrics or infers a scientific verdict. It checks
source identity and preserves exclusion reasons, then orders records by the
human-provided triage priority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


RECORD_FORMAT = "urbanpidit-r7-evidence-record-v1"
OUTCOME_CLASSES = frozenset({
    "engineering-positive", "audit", "negative", "mixed", "unresolved",
})
CANDIDATE_STATES = frozenset({"candidate", "blocked", "needs-review", "not-candidate"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    relative = Path(value)
    if ".." in relative.parts or relative == Path("."):
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, str) and SHA256_RE.fullmatch(value) is not None)


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_nonempty(item) for item in value)


def _json_scalars(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, list):
        return all(_json_scalars(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_scalars(item) for key, item in value.items())
    return False


def validate_record(record: Any, root: Path, line_number: int) -> list[str]:
    """Return errors for one record; no errors means it is indexable."""
    where = f"line {line_number}"
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{where}: record must be an object"]
    if record.get("format") != RECORD_FORMAT:
        errors.append(f"{where}: format must be {RECORD_FORMAT!r}")
    for key in ("record_id", "evidence_path", "evidence_commit", "reason"):
        if not _nonempty(record.get(key)):
            errors.append(f"{where}: {key} must be a nonempty string")
    record_id = record.get("record_id")
    if isinstance(record_id, str) and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", record_id):
        errors.append(f"{where}: record_id must be kebab-case")

    evidence_path = _safe_relative(root, record.get("evidence_path"))
    if evidence_path is None:
        errors.append(f"{where}: evidence_path must be a safe repository-relative path")
    elif not evidence_path.is_file():
        errors.append(f"{where}: evidence_path does not exist: {record['evidence_path']}")
    elif not isinstance(record.get("evidence_sha256"), str) or not SHA256_RE.fullmatch(record["evidence_sha256"]):
        errors.append(f"{where}: evidence_sha256 must be a full lowercase SHA256")
    elif _sha256(evidence_path) != record["evidence_sha256"]:
        errors.append(f"{where}: evidence_sha256 does not match {record['evidence_path']}")

    for key in ("evidence_commit", "experiment_commit"):
        value = record.get(key)
        if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
            errors.append(f"{where}: {key} must be a full lowercase 40-character SHA")
    for key in ("protocol_sha256", "data_identity", "source_sha256"):
        if not _digest_or_none(record.get(key)):
            errors.append(f"{where}: {key} must be a full SHA256 or null")

    if record.get("outcome_class") not in OUTCOME_CLASSES:
        errors.append(f"{where}: outcome_class must be one of {sorted(OUTCOME_CLASSES)}")
    if record.get("candidate_state") not in CANDIDATE_STATES:
        errors.append(f"{where}: candidate_state must be one of {sorted(CANDIDATE_STATES)}")
    priority = record.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
        errors.append(f"{where}: priority must be an integer in [0, 100]")
    if record.get("scientific_claim") is not False:
        errors.append(f"{where}: scientific_claim must be false")
    if not _string_list(record.get("limitations")):
        errors.append(f"{where}: limitations must be a nonempty list of strings")
    if not isinstance(record.get("metrics"), dict) or not _json_scalars(record["metrics"]):
        errors.append(f"{where}: metrics must be a JSON object of scalar/list values")
    if record.get("candidate_state") == "candidate":
        for key in ("protocol_sha256", "data_identity"):
            if record.get(key) is None:
                errors.append(f"{where}: candidate requires {key}")
    else:
        if not _nonempty(record.get("excluded_reason")):
            errors.append(f"{where}: non-candidate records require excluded_reason")
    ci_run_id = record.get("ci_run_id")
    if ci_run_id is not None and not _nonempty(str(ci_run_id)):
        errors.append(f"{where}: ci_run_id must be a nonempty string or null")
    return errors


def load_index(path: Path, root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and validate an evidence JSONL file."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"cannot read index {path}: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc}")
            continue
        errors.extend(validate_record(record, root, line_number))
        if isinstance(record, dict):
            record_id = record.get("record_id")
            if record_id in seen:
                errors.append(f"line {line_number}: duplicate record_id {record_id!r}")
            seen.add(record_id)
            records.append(record)
    if not records:
        errors.append("index must contain at least one record")
    return records, errors


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def render_brief(records: Iterable[dict[str, Any]], errors: Iterable[str] = ()) -> str:
    """Render a source-linked human brief without changing any record."""
    records = sorted(records, key=lambda item: (-item["priority"], item["record_id"]))
    errors = list(errors)
    lines = [
        "# R7 Evidence Candidate Brief",
        "",
        "> This is an evidence index, not a scientific verdict. Priority is a human triage field.",
        "> No entry authorizes training, data access, GPU use, or a change to a frozen criterion.",
        "",
        f"Records: {len(records)}; human-review candidates: "
        f"{sum(item['candidate_state'] == 'candidate' for item in records)}",
        "",
    ]
    if errors:
        lines.extend(["## Index Errors", "", *[f"- {error}" for error in errors], ""])
    for record in records:
        lines.extend([
            f"## {record['record_id']}",
            "",
            f"- Outcome class: `{record['outcome_class']}`; candidate state: `{record['candidate_state']}`",
            f"- Human triage priority: `{record['priority']}` (not a scientific score)",
            f"- Evidence: `{record['evidence_path']}` (SHA256 `{record['evidence_sha256']}`)",
            f"- Evidence commit: `{record['evidence_commit']}`; experiment commit: `{record['experiment_commit']}`",
            f"- Protocol SHA256: `{record.get('protocol_sha256') or 'not recorded'}`; "
            f"data identity: `{record.get('data_identity') or 'not recorded'}`",
            f"- Reason: {record['reason']}",
            "- Limitations:",
            *[f"  - {item}" for item in record["limitations"]],
        ])
        if record.get("ci_run_id") is not None:
            lines.append(f"- CI run: `{record['ci_run_id']}`")
        if record.get("excluded_reason"):
            lines.append(f"- Excluded from runnable candidates: {record['excluded_reason']}")
        metrics = ", ".join(
            f"{key}={_format_value(value)}" for key, value in sorted(record["metrics"].items())
        )
        lines.extend([f"- Recorded metrics (not recomputed): {metrics or 'none'}", ""])
    return "\n".join(lines)


def check_brief_sync(index_path: Path, brief_path: Path, root: Path) -> list[str]:
    """Return read-only errors when a checked-in brief is stale or altered."""
    records, errors = load_index(index_path, root.resolve())
    if errors:
        return errors
    try:
        actual = brief_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"cannot read brief {brief_path}: {exc}"]
    expected = render_brief(records) + "\n"
    if actual != expected:
        return [f"brief does not match canonical render: {brief_path}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and render an R7 evidence JSONL index.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--render", action="store_true", help="render a source-linked Markdown brief")
    parser.add_argument("--check-brief", type=Path, help="check a checked-in brief against the canonical render")
    args = parser.parse_args(argv)
    if args.render and args.check_brief is not None:
        parser.error("--render and --check-brief are mutually exclusive")
    root = args.root.resolve()
    if args.check_brief is not None:
        errors = check_brief_sync(args.index, args.check_brief, root)
        for error in errors:
            print(f"ERROR {error}")
        if not errors:
            records, _ = load_index(args.index, root)
            print(f"PASS evidence index and brief: {len(records)} records")
        return 1 if errors else 0
    records, errors = load_index(args.index, root)
    if args.render:
        print(render_brief(records, errors))
    else:
        for error in errors:
            print(f"ERROR {error}")
        if not errors:
            print(f"PASS evidence index: {len(records)} records")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
