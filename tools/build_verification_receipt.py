"""Build a verification receipt from an existing bounded-run directory.

This tool does not run a study and does not infer a scientific result. It only
summarises files already present, so a failed run can leave a useful terminal
record without being presented as a successful experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verification_profiles import DEFAULT_PROFILE_ID, ReceiptProfile, get_profile, pointer_values
from tools.verify_experiment_receipt import STRICT_FORMAT, STATUSES, validate_receipt



def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_path(root: Path, profile: ReceiptProfile, role: str) -> Path:
    relative = profile.path_for(role)
    if relative is None:
        raise ValueError(f"profile {profile.profile_id} has no {role} role")
    return root / relative


def _identity_values(result: dict[str, Any], profile: ReceiptProfile) -> list[str]:
    values: list[str] = []
    for pointer in profile.result_identity_pointers:
        for value in pointer_values(result, pointer):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"result identity pointer {pointer} contains a nonempty string requirement")
            values.append(value)
    unique = sorted(set(values))
    if len(unique) > 1:
        raise ValueError("result identity fields do not agree")
    return unique


def _limitations(result: dict[str, Any] | None, profile: ReceiptProfile) -> list[str]:
    if result is not None:
        for pointer in profile.result_limitations_pointers:
            values = pointer_values(result, pointer)
            for value in values:
                if isinstance(value, list):
                    items = [item for item in value if isinstance(item, str) and item.strip()]
                    if items:
                        return items
    return ["Run did not produce a complete result; inspect failure_reason and artifacts."]


def _commit_sha(root: Path, supplied: str | None) -> str:
    path = root / "code_commit.txt"
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    if supplied:
        return supplied
    return os.environ.get("GITHUB_SHA", "")


def _artifact_entries(root: Path, profile: ReceiptProfile) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for role, relative in profile.role_paths:
        path = root / relative
        if path.is_file():
            entries.append({
                "type": profile.type_for(relative),
                "path": relative,
                "sha256": _sha256(path),
                "role": role,
            })
    return entries


def build_receipt(
    root: Path,
    *,
    run_id: str,
    workflow: str,
    status: str,
    commit_sha: str | None = None,
    failure_reason: str | None = None,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> dict[str, Any]:
    """Build and exclusively write a strict receipt under ``root``."""
    root = Path(root)
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    if not run_id or not workflow:
        raise ValueError("run_id and workflow are required")
    if status != "success" and not isinstance(failure_reason, str):
        raise ValueError(f"{status} receipts require failure_reason")
    profile = get_profile(profile_id)
    root.mkdir(parents=True, exist_ok=True)

    protocol_path = _profile_path(root, profile, "protocol")
    result_path = _profile_path(root, profile, "primary_result")
    source_path = _profile_path(root, profile, "source_data")
    source_receipt_path = _profile_path(root, profile, "source_receipt")
    protocol = _read_json(protocol_path) if protocol_path.is_file() else None
    result = _read_json(result_path) if result_path.is_file() else None
    commit = _commit_sha(root, commit_sha)
    if not commit:
        raise ValueError("a full commit SHA is required from code_commit.txt or --commit-sha")

    artifacts = _artifact_entries(root, profile)
    artifact_paths = {entry["path"] for entry in artifacts}
    required_paths = [profile.path_for(role) for role in profile.required_roles]
    required_paths = [path for path in required_paths if path is not None]

    protocol_digest = None
    source_digest = None
    if protocol is not None:
        values = pointer_values(protocol, profile.protocol_digest_pointer)
        if len(values) == 1 and isinstance(values[0], str):
            protocol_digest = values[0]
        values = pointer_values(protocol, profile.protocol_source_pointer)
        if len(values) == 1 and isinstance(values[0], str):
            source_digest = values[0]
    if source_path.is_file():
        actual_source_digest = _sha256(source_path)
        if source_digest is not None and source_digest != actual_source_digest:
            raise ValueError("protocol source digest does not match source artifact bytes")
        source_digest = actual_source_digest
    elif status == "success":
        raise ValueError(f"successful run is missing source artifact: {source_path.relative_to(root)}")

    data_identity = None
    if result is not None:
        identities = _identity_values(result, profile)
        data_identity = identities[0] if identities else None

    if status == "success":
        missing = sorted(set(required_paths) - artifact_paths)
        if missing:
            raise ValueError(f"successful run is missing required artifacts: {', '.join(missing)}")
        if protocol_digest is None:
            raise ValueError("successful run protocol lacks the profile digest field")
        if source_digest is None:
            raise ValueError("successful run lacks source identity")
        if data_identity is None:
            raise ValueError("successful run result lacks data/training identity")
        if result is None or result.get("scientific_claim") is not False:
            raise ValueError("successful run result must declare scientific_claim=false")

    receipt: dict[str, Any] = {
        "format": STRICT_FORMAT,
        "profile_id": profile.profile_id,
        "verification_scope": profile.verification_scope,
        "run_id": run_id,
        "workflow": workflow,
        "status": status,
        "commit_sha": commit,
        "protocol_sha256": protocol_digest,
        "source_sha256": source_digest,
        "data_identity": data_identity,
        "scientific_claim": False,
        "limitations": _limitations(result, profile),
        "failure_reason": failure_reason if status != "success" else None,
        "protocol_path": profile.path_for("protocol"),
        "commit_path": profile.path_for("code_commit"),
        "result_path": profile.path_for("primary_result") if result_path.is_file() else None,
        "source_path": profile.path_for("source_data"),
        "source_receipt_path": profile.path_for("source_receipt"),
        "role_paths": dict(profile.role_paths),
        "required_artifacts": required_paths,
        "output_artifacts": artifacts,
    }
    report = validate_receipt(receipt, root)
    if status == "success" and not report.accepted:
        raise ValueError("constructed successful receipt failed validation: " + "; ".join(report.errors))
    if status != "success" and not report.valid_record:
        raise ValueError("constructed terminal receipt failed validation: " + "; ".join(report.errors))
    destination = root / "verification_receipt.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a bounded-run verification receipt.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--status", required=True, choices=sorted(STATUSES))
    parser.add_argument("--commit-sha", default=None)
    parser.add_argument("--failure-reason", default=None)
    parser.add_argument("--profile", dest="profile_id", default=DEFAULT_PROFILE_ID)
    args = parser.parse_args(argv)
    try:
        build_receipt(
            args.root,
            run_id=args.run_id,
            workflow=args.workflow,
            status=args.status,
            commit_sha=args.commit_sha,
            failure_reason=args.failure_reason,
            profile_id=args.profile_id,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR {exc}")
        return 1
    print(f"WROTE {args.root / 'verification_receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
