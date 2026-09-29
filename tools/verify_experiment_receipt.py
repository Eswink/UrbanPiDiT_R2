"""Validate machine-readable verification receipts without running experiments.

A receipt records engineering provenance and artifact integrity. It does not
judge forecast quality or turn a failed, queued, skipped, or cancelled run into
a pass. The validator is deliberately read-only: callers create the receipt,
then this module checks it against the files under ``--root``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verification_profiles import ReceiptProfile, get_profile, pointer_values


FORMAT = "urbanpidit-verification-receipt-v1"
STRICT_FORMAT = "urbanpidit-verification-receipt-v1.1"
STATUSES = frozenset({"success", "partial", "failed", "cancelled", "queued", "skipped"})
NON_SUCCESS_STATUSES = STATUSES - {"success"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ReceiptReport:
    """The result of one read-only receipt validation."""

    status: str | None
    errors: tuple[str, ...]

    @property
    def valid_record(self) -> bool:
        """Whether the receipt is structurally valid, regardless of pass status."""
        return not self.errors

    @property
    def accepted(self) -> bool:
        """Whether the receipt describes an accepted successful run."""
        return self.valid_record and self.status == "success"


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _safe_relative_path(root: Path, value: Any) -> Path | None:
    """Return a path contained by root, or None for an unsafe/non-relative path."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    relative = Path(value)
    if ".." in relative.parts or relative == Path("."):
        return None
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read JSON {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON artifact {path} must contain an object"
    return value, None


def _canonical_digest(value: dict[str, Any], digest_pointer: str) -> str | None:
    """Hash a protocol body after removing its root digest field."""
    parts = digest_pointer.split("/") if isinstance(digest_pointer, str) else []
    if len(parts) != 2 or parts[0] != "" or not parts[1] or "/" in parts[1]:
        return None
    digest_key = parts[1].replace("~1", "/").replace("~0", "~")
    body = {key: item for key, item in value.items() if key != digest_key}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _nonempty_strings(values: list[Any]) -> list[str]:
    return [value for value in values if isinstance(value, str) and value.strip()]


def _validate_artifact_list(
    receipt: dict[str, Any], root: Path, errors: list[str]
) -> dict[str, Path]:
    entries = receipt.get("output_artifacts")
    if not isinstance(entries, list):
        errors.append("output_artifacts must be a list")
        return {}

    seen: set[str] = set()
    resolved: dict[str, Path] = {}
    for index, entry in enumerate(entries):
        where = f"output_artifacts[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        path_value = entry.get("path")
        artifact_path = _safe_relative_path(root, path_value)
        if artifact_path is None:
            errors.append(f"{where}.path must be a safe path relative to the receipt root")
            continue
        path_key = Path(path_value).as_posix()
        if path_key in seen:
            errors.append(f"{where}.path is duplicated: {path_key}")
        seen.add(path_key)
        if not _is_nonempty_string(entry.get("type")):
            errors.append(f"{where}.type must be a nonempty string")
        if not _is_sha256(entry.get("sha256")):
            errors.append(f"{where}.sha256 must be a lowercase SHA256 digest")
        if not artifact_path.is_file():
            errors.append(f"{where}.path does not exist as a file: {path_key}")
            continue
        if _is_sha256(entry.get("sha256")) and _sha256(artifact_path) != entry["sha256"]:
            errors.append(f"{where}.sha256 does not match {path_key}")
        resolved[path_key] = artifact_path
    return resolved


def _validate_required_artifacts(
    receipt: dict[str, Any], artifacts: dict[str, Path], status: str | None, errors: list[str]
) -> list[str]:
    required = receipt.get("required_artifacts")
    if not isinstance(required, list):
        errors.append("required_artifacts must be a list")
        return []
    normalised: list[str] = []
    for index, value in enumerate(required):
        if not isinstance(value, str) or _safe_relative_path(Path("."), value) is None:
            errors.append(f"required_artifacts[{index}] must be a safe relative path")
            continue
        path_key = Path(value).as_posix()
        if path_key in normalised:
            errors.append(f"required_artifacts[{index}] is duplicated: {path_key}")
        normalised.append(path_key)
    if status == "success":
        for path_key in normalised:
            if path_key not in artifacts:
                errors.append(f"successful receipt is missing required artifact: {path_key}")
    return normalised


def _validate_result_artifact(
    receipt: dict[str, Any],
    artifacts: dict[str, Path],
    errors: list[str],
    profile: ReceiptProfile | None = None,
) -> dict[str, Any] | None:
    result_path = receipt.get("result_path")
    if result_path is None:
        if receipt.get("status") == "success":
            errors.append("successful receipt must declare result_path")
        return None
    if not isinstance(result_path, str) or Path(result_path).is_absolute() or ".." in Path(result_path).parts:
        errors.append("result_path must be a safe relative path")
        return None
    result_key = Path(result_path).as_posix()
    if result_key not in artifacts:
        errors.append(f"result_path is not listed in output_artifacts: {result_key}")
        return None
    result, error = _read_json(artifacts[result_key])
    if error:
        errors.append(error)
        return None
    assert result is not None
    if result.get("scientific_claim") is not False:
        errors.append(f"result artifact {result_key} must declare scientific_claim=false")

    pointers = profile.result_limitations_pointers if profile else ("/limitations",)
    limitation_lists: list[list[Any]] = []
    for pointer in pointers:
        values = pointer_values(result, pointer)
        limitation_lists.extend(value for value in values if isinstance(value, list))
    if not limitation_lists or not any(
        items and all(isinstance(item, str) and item.strip() for item in items)
        for items in limitation_lists
    ):
        errors.append(f"result artifact {result_key} must contain nonempty limitations")
    return result


def _profile_role_paths(
    receipt: dict[str, Any], profile: ReceiptProfile, root: Path, errors: list[str]
) -> dict[str, str]:
    """Resolve fixed profile roles and reject receipt-side path substitution."""
    expected = dict(profile.role_paths)
    declared = receipt.get("role_paths")
    if declared is not None:
        if not isinstance(declared, dict):
            errors.append("role_paths must be an object")
        else:
            for role, path in expected.items():
                if declared.get(role) != path:
                    errors.append(f"role_paths[{role!r}] does not match profile {path!r}")
            extras = set(declared) - set(expected)
            if extras:
                errors.append(f"role_paths contains unknown role(s): {sorted(extras)}")
    for role, path in expected.items():
        if _safe_relative_path(root, path) is None:
            errors.append(f"profile role {role!r} has an unsafe path")
    return expected


def _profile_identity(
    receipt: dict[str, Any],
    root: Path,
    artifacts: dict[str, Path],
    required: list[str],
    profile: ReceiptProfile,
    result: dict[str, Any] | None,
    commit_sha: str | None,
    errors: list[str],
) -> None:
    paths = _profile_role_paths(receipt, profile, root, errors)
    status = receipt.get("status")

    for role in profile.required_roles:
        path = paths.get(role)
        if path is None:
            errors.append(f"profile is missing required role: {role}")
        elif status == "success" and path not in artifacts:
            errors.append(f"successful receipt is missing required role artifact: {role} ({path})")
        if status == "success" and path is not None and path not in required:
            errors.append(f"successful receipt required_artifacts omits profile role: {role} ({path})")

    for field, role in (("protocol_path", "protocol"), ("source_path", "source_data"),
                        ("source_receipt_path", "source_receipt"), ("commit_path", "code_commit")):
        declared = receipt.get(field)
        expected = paths.get(role)
        if declared is not None and declared != expected:
            errors.append(f"{field} does not match profile role {role}: {expected}")
        if status == "success" and declared is None:
            errors.append(f"successful receipt must declare {field}")

    result_path = receipt.get("result_path")
    expected_result = paths.get("primary_result")
    if result_path is not None and result_path != expected_result:
        errors.append(f"result_path does not match profile primary_result: {expected_result}")
    elif status == "success" and result_path != expected_result:
        errors.append(f"successful receipt result_path must match profile primary_result: {expected_result}")
    elif status == "success" and result_path not in artifacts:
        errors.append(f"successful receipt is missing primary result artifact: {expected_result}")

    commit_key = paths.get("code_commit")
    if commit_key in artifacts and isinstance(commit_sha, str):
        try:
            recorded = artifacts[commit_key].read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read code commit artifact: {exc}")
        else:
            if recorded != commit_sha:
                errors.append("commit_sha does not match the code_commit artifact")
    elif status == "success":
        errors.append(f"successful receipt is missing code commit artifact: {commit_key}")

    protocol_key = paths.get("protocol")
    protocol = None
    if protocol_key in artifacts:
        protocol, error = _read_json(artifacts[protocol_key])
        if error:
            errors.append(error)
        else:
            digest_values = pointer_values(protocol, profile.protocol_digest_pointer)
            if len(digest_values) != 1 or not _is_sha256(digest_values[0]):
                errors.append("protocol digest field must contain one full lowercase SHA256")
            else:
                protocol_digest = digest_values[0]
                if receipt.get("protocol_sha256") != protocol_digest:
                    errors.append("protocol_sha256 does not match the protocol artifact")
                if profile.canonical_protocol_digest:
                    canonical = _canonical_digest(protocol, profile.protocol_digest_pointer)
                    if canonical is None or canonical != protocol_digest:
                        errors.append("protocol digest does not match the canonical protocol body")
            source_values = pointer_values(protocol, profile.protocol_source_pointer)
            if len(source_values) != 1 or not _is_sha256(source_values[0]):
                if status == "success":
                    errors.append("protocol source digest must contain one full lowercase SHA256")
            elif source_values[0] != receipt.get("source_sha256"):
                errors.append("protocol source digest does not match receipt source_sha256")
    elif status == "success":
        errors.append(f"successful receipt is missing protocol artifact: {protocol_key}")

    source_key = paths.get("source_data")
    source_digest = None
    if source_key in artifacts:
        source_digest = _sha256(artifacts[source_key])
        if receipt.get("source_sha256") != source_digest:
            errors.append("source_sha256 does not match the source artifact bytes")
    elif status == "success":
        errors.append(f"successful receipt is missing source artifact: {source_key}")

    source_receipt_key = paths.get("source_receipt")
    if source_receipt_key in artifacts:
        source_receipt, error = _read_json(artifacts[source_receipt_key])
        if error:
            errors.append(error)
        else:
            assert source_receipt is not None
            digest_values = pointer_values(source_receipt, profile.source_receipt_digest_pointer)
            if digest_values:
                if len(digest_values) != 1 or not _is_sha256(digest_values[0]):
                    errors.append("source receipt digest must contain one full lowercase SHA256")
                elif source_digest is not None and digest_values[0] != source_digest:
                    errors.append("source receipt digest does not match source artifact bytes")
            elif status == "success":
                errors.append("successful receipt source receipt lacks source digest")
            byte_values = pointer_values(source_receipt, profile.source_receipt_bytes_pointer)
            if byte_values:
                if len(byte_values) != 1 or not isinstance(byte_values[0], int) or isinstance(byte_values[0], bool):
                    errors.append("source receipt byte count must contain one integer")
                elif source_key in artifacts and byte_values[0] != artifacts[source_key].stat().st_size:
                    errors.append("source receipt byte count does not match source artifact")
            elif status == "success":
                errors.append("successful receipt source receipt lacks source byte count")
    elif status == "success":
        errors.append(f"successful receipt is missing source receipt artifact: {source_receipt_key}")

    if result is not None:
        identity_values: list[str] = []
        for pointer in profile.result_identity_pointers:
            values = pointer_values(result, pointer)
            if values and any(not _is_nonempty_string(value) for value in values):
                errors.append(f"result identity pointer {pointer} contains a non-string value")
            identity_values.extend(_nonempty_strings(values))
        unique = set(identity_values)
        if len(unique) > 1:
            errors.append("result identity fields do not agree")
        elif status == "success" and not unique:
            errors.append("successful receipt result lacks data/training identity")
        elif unique and receipt.get("data_identity") not in unique:
            errors.append("data_identity does not match the result identity")


def validate_receipt(receipt: Any, root: Path) -> ReceiptReport:
    """Validate one receipt and its referenced files under ``root``."""
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ReceiptReport(None, ("receipt must contain a JSON object",))

    format_value = receipt.get("format")
    strict = format_value == STRICT_FORMAT
    if format_value not in (FORMAT, STRICT_FORMAT):
        errors.append(f"format must be {FORMAT!r} or {STRICT_FORMAT!r}")
    for field in ("run_id", "workflow"):
        if not _is_nonempty_string(receipt.get(field)):
            errors.append(f"{field} must be a nonempty string")

    status = receipt.get("status")
    if status not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    if not isinstance(receipt.get("scientific_claim"), bool) or receipt.get("scientific_claim") is not False:
        errors.append("scientific_claim must be false")
    limitations = receipt.get("limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        errors.append("limitations must be a nonempty list of strings")

    commit_sha = receipt.get("commit_sha")
    if not isinstance(commit_sha, str) or COMMIT_RE.fullmatch(commit_sha) is None:
        errors.append("commit_sha must be a full lowercase 40-character commit SHA")

    for field in ("protocol_sha256", "source_sha256"):
        value = receipt.get(field)
        if value is not None and not _is_sha256(value):
            errors.append(f"{field} must be a lowercase SHA256 digest or null")
    data_identity = receipt.get("data_identity")
    if data_identity is not None and not _is_nonempty_string(data_identity):
        errors.append("data_identity must be a nonempty string or null")

    if status == "success":
        for field in ("protocol_sha256", "source_sha256", "data_identity"):
            if receipt.get(field) in (None, ""):
                errors.append(f"successful receipt must declare {field}")
    elif status in NON_SUCCESS_STATUSES and not _is_nonempty_string(receipt.get("failure_reason")):
        errors.append(f"{status} receipt must declare failure_reason")

    root = root.resolve()
    artifacts = _validate_artifact_list(receipt, root, errors)
    required = _validate_required_artifacts(receipt, artifacts, status, errors)

    profile = None
    if strict:
        profile_id = receipt.get("profile_id")
        if not _is_nonempty_string(profile_id):
            errors.append("strict receipt must declare profile_id")
        else:
            try:
                profile = get_profile(profile_id)
            except ValueError as exc:
                errors.append(str(exc))
        if profile is not None and receipt.get("verification_scope") != profile.verification_scope:
            errors.append("verification_scope does not match the selected profile")

    result = _validate_result_artifact(receipt, artifacts, errors, profile if strict else None)
    if strict and profile is not None:
        _profile_identity(receipt, root, artifacts, required, profile, result, commit_sha, errors)
    else:
        protocol_path = receipt.get("protocol_path", "protocol.json")
        if protocol_path is not None:
            protocol_key = Path(protocol_path).as_posix() if isinstance(protocol_path, str) else ""
            if protocol_key in artifacts and receipt.get("protocol_sha256") is not None:
                protocol, error = _read_json(artifacts[protocol_key])
                if error:
                    errors.append(error)
                elif protocol.get("protocol_sha256") != receipt["protocol_sha256"]:
                    errors.append("protocol_sha256 does not match the protocol artifact")

        commit_path = receipt.get("commit_path", "code_commit.txt")
        if commit_path is not None:
            commit_key = Path(commit_path).as_posix() if isinstance(commit_path, str) else ""
            if commit_key in artifacts and isinstance(commit_sha, str):
                recorded = artifacts[commit_key].read_text(encoding="utf-8").strip()
                if recorded != commit_sha:
                    errors.append("commit_sha does not match the code_commit artifact")

    if status == "success" and not required:
        errors.append("successful receipt must declare at least one required artifact")
    return ReceiptReport(status if isinstance(status, str) else None, tuple(errors))


def load_receipt(path: Path) -> dict[str, Any]:
    """Read a receipt JSON object, raising a concise error on malformed input."""
    value, error = _read_json(path)
    if error:
        raise ValueError(error)
    assert value is not None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only verification receipt validator.")
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        help="root for relative artifact paths; defaults to the receipt's parent",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="return nonzero unless the valid receipt status is success",
    )
    args = parser.parse_args(argv)
    try:
        receipt = load_receipt(args.receipt)
    except ValueError as exc:
        print(f"ERROR {exc}")
        return 1
    report = validate_receipt(receipt, args.root or args.receipt.parent)
    for error in report.errors:
        print(f"ERROR {error}")
    if report.errors:
        return 1
    if report.accepted:
        print("PASS valid verification receipt: status=success")
        return 0
    print(f"RECORDED status={report.status}; not an accepted success")
    return 1 if args.require_success else 0


if __name__ == "__main__":
    raise SystemExit(main())
