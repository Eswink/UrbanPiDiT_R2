"""Explicit artifact-layout profiles for machine-readable verification receipts.

Profiles describe engineering provenance only. They do not select experiments,
interpret metrics, or authenticate a source beyond the bytes and declarations
that the receipt validator can inspect locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReceiptProfile:
    """A fixed artifact layout and its identity-field locations."""

    profile_id: str
    role_paths: tuple[tuple[str, str], ...]
    required_roles: tuple[str, ...]
    artifact_types: tuple[tuple[str, str], ...]
    protocol_digest_pointer: str
    protocol_source_pointer: str
    source_receipt_digest_pointer: str
    source_receipt_bytes_pointer: str
    result_limitations_pointers: tuple[str, ...]
    result_identity_pointers: tuple[str, ...]
    verification_scope: str = "full-local-file"
    canonical_protocol_digest: bool = True

    def path_for(self, role: str) -> str | None:
        """Return the exact root-relative path for a logical role."""
        return dict(self.role_paths).get(role)

    def type_for(self, path: str) -> str:
        """Return the stable artifact type for a profile path."""
        return dict(self.artifact_types).get(path, "artifact")


CPU_STUDY_PROFILE = ReceiptProfile(
    profile_id="cpu-study-flat-v1.1",
    role_paths=(
        ("protocol", "protocol.json"),
        ("primary_result", "study_result.json"),
        ("source_data", "source/era5_pressure_pilot.nc"),
        ("source_receipt", "source/receipt.json"),
        ("code_commit", "code_commit.txt"),
        ("code_archive", "code.zip"),
        ("metrics_rmse", "seed_rmse.csv"),
        ("metrics_summary", "seed_summary.csv"),
    ),
    required_roles=(
        "protocol", "primary_result", "source_data", "source_receipt",
        "code_commit", "code_archive", "metrics_rmse", "metrics_summary",
    ),
    artifact_types=(
        ("protocol.json", "protocol"),
        ("study_result.json", "result"),
        ("source/era5_pressure_pilot.nc", "source"),
        ("source/receipt.json", "source-receipt"),
        ("code_commit.txt", "code-identity"),
        ("code.zip", "code-archive"),
        ("seed_rmse.csv", "metrics"),
        ("seed_summary.csv", "metrics"),
    ),
    protocol_digest_pointer="/protocol_sha256",
    protocol_source_pointer="/source_sha256",
    source_receipt_digest_pointer="/source_netcdf_sha256",
    source_receipt_bytes_pointer="/source_netcdf_bytes",
    result_limitations_pointers=("/limitations",),
    result_identity_pointers=(
        "/data_identity", "/training_identity",
        "/records/*/report/data_identity", "/records/*/report/training_identity",
    ),
)

CONTINUOUS_CONTROL_PROFILE = ReceiptProfile(
    profile_id="continuous-nested-v1",
    role_paths=(
        ("protocol", "protocol.json"),
        ("primary_result", "continuous_control_result.json"),
        ("source_data", "dataset/source/era5_continuous250d.nc"),
        ("source_receipt", "dataset/source/receipt.json"),
        ("code_commit", "code_commit.txt"),
        ("code_archive", "code.zip"),
    ),
    required_roles=(
        "protocol", "primary_result", "source_data", "source_receipt",
        "code_commit", "code_archive",
    ),
    artifact_types=(
        ("protocol.json", "protocol"),
        ("continuous_control_result.json", "result"),
        ("dataset/source/era5_continuous250d.nc", "source"),
        ("dataset/source/receipt.json", "source-receipt"),
        ("code_commit.txt", "code-identity"),
        ("code.zip", "code-archive"),
    ),
    protocol_digest_pointer="/protocol_sha256",
    protocol_source_pointer="/source_sha256",
    source_receipt_digest_pointer="/source_netcdf_sha256",
    source_receipt_bytes_pointer="/source_netcdf_bytes",
    result_limitations_pointers=("/limitations",),
    result_identity_pointers=(
        "/data_identity", "/training_identity",
        "/records/*/report/data_identity", "/records/*/report/training_identity",
    ),
)

BASELINE_PROFILE = ReceiptProfile(
    profile_id="baseline-flat-v1",
    role_paths=(
        ("protocol", "protocol.json"),
        ("primary_result", "baseline_study_result.json"),
        ("source_data", "source/era5_four_season.nc"),
        ("source_receipt", "source/receipt.json"),
        ("code_commit", "code_commit.txt"),
        ("code_archive", "code.zip"),
        ("metrics", "baseline_rmse_cpu.csv"),
    ),
    required_roles=(
        "protocol", "primary_result", "source_data", "source_receipt",
        "code_commit", "code_archive", "metrics",
    ),
    artifact_types=(
        ("protocol.json", "protocol"),
        ("baseline_study_result.json", "result"),
        ("source/era5_four_season.nc", "source"),
        ("source/receipt.json", "source-receipt"),
        ("code_commit.txt", "code-identity"),
        ("code.zip", "code-archive"),
        ("baseline_rmse_cpu.csv", "metrics"),
    ),
    protocol_digest_pointer="/signature",
    protocol_source_pointer="/source_sha256",
    source_receipt_digest_pointer="/source_netcdf_sha256",
    source_receipt_bytes_pointer="/source_netcdf_bytes",
    result_limitations_pointers=("/limitations", "/limits"),
    result_identity_pointers=(
        "/data_identity", "/training_identity",
        "/records/*/report/data_identity", "/records/*/report/training_identity",
    ),
)

PROFILES = {
    profile.profile_id: profile
    for profile in (CPU_STUDY_PROFILE, CONTINUOUS_CONTROL_PROFILE, BASELINE_PROFILE)
}
DEFAULT_PROFILE_ID = CPU_STUDY_PROFILE.profile_id


def get_profile(profile_id: str) -> ReceiptProfile:
    """Return a known profile or raise a concise configuration error."""
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown verification profile: {profile_id}") from exc


def pointer_values(document: Any, pointer: str) -> list[Any]:
    """Read values from a JSON pointer with ``*`` list/dict expansion."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return []
    current = [document]
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        next_values: list[Any] = []
        for value in current:
            if part == "*":
                if isinstance(value, dict):
                    next_values.extend(value.values())
                elif isinstance(value, list):
                    next_values.extend(value)
            elif isinstance(value, dict) and part in value:
                next_values.append(value[part])
            elif isinstance(value, list) and part.isdigit():
                index = int(part)
                if index < len(value):
                    next_values.append(value[index])
        current = next_values
    return current
