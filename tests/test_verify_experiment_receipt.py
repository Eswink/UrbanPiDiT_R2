"""Offline counterproofs for the verification receipt validator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_verification_receipt import build_receipt
from tools.verify_experiment_receipt import FORMAT, STRICT_FORMAT, main as verify_main, validate_receipt


COMMIT = "a" * 40
PROTOCOL = "b" * 64


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(root: Path, *, status: str = "success", result: dict | None = None) -> dict:
    result_path = root / "study_result.json"
    write_json(result_path, result or {
        "scientific_claim": False,
        "limitations": ["fixture only"],
        "data_identity": "data-fixture-001",
    })
    source_path = root / "source" / "era5_pressure_pilot.nc"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"verified source")
    source_digest = sha256(source_path)
    protocol_path = root / "protocol.json"
    protocol_body = {"source_sha256": source_digest}
    protocol_digest = hashlib.sha256(
        json.dumps(protocol_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(protocol_path, {**protocol_body, "protocol_sha256": protocol_digest})
    commit_path = root / "code_commit.txt"
    commit_path.write_text(COMMIT + "\n", encoding="utf-8")
    source_receipt_path = root / "source" / "receipt.json"
    write_json(source_receipt_path, {
        "status": "downloaded-real-source",
        "source_netcdf_sha256": source_digest,
        "source_netcdf_bytes": source_path.stat().st_size,
    })
    for relative, content in (
        ("seed_rmse.csv", "variable,rmse\nt2m,1.0\n"),
        ("seed_summary.csv", "variable,mean\nt2m,1.0\n"),
        ("code.zip", b"archived code"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    artifacts = [
        {"type": "protocol", "path": "protocol.json", "sha256": sha256(protocol_path)},
        {"type": "result", "path": "study_result.json", "sha256": sha256(result_path)},
        {"type": "code", "path": "code_commit.txt", "sha256": sha256(commit_path)},
        {"type": "source", "path": "source/era5_pressure_pilot.nc", "sha256": sha256(source_path)},
        {"type": "source-receipt", "path": "source/receipt.json", "sha256": sha256(source_receipt_path)},
        {"type": "metrics", "path": "seed_rmse.csv", "sha256": sha256(root / "seed_rmse.csv")},
        {"type": "metrics", "path": "seed_summary.csv", "sha256": sha256(root / "seed_summary.csv")},
        {"type": "code-archive", "path": "code.zip", "sha256": sha256(root / "code.zip")},
    ]
    return {
        "format": FORMAT,
        "run_id": "run-fixture-001",
        "workflow": "r7-cpu-study",
        "status": status,
        "commit_sha": COMMIT,
        "protocol_sha256": protocol_digest,
        "source_sha256": source_digest,
        "data_identity": "data-fixture-001",
        "scientific_claim": False,
        "limitations": ["fixture is not weather truth"],
        "failure_reason": "fixture terminal state" if status != "success" else None,
        "protocol_path": "protocol.json",
        "commit_path": "code_commit.txt",
        "result_path": "study_result.json",
        "required_artifacts": [
            "protocol.json", "study_result.json", "seed_rmse.csv", "seed_summary.csv",
            "source/era5_pressure_pilot.nc", "source/receipt.json", "code_commit.txt", "code.zip",
        ],
        "output_artifacts": artifacts,
    }


def make_nested_run(root: Path) -> None:
    """Create a continuous-control-shaped layout without running a study."""
    source = root / "dataset" / "source" / "era5_continuous250d.nc"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"nested verified source")
    source_digest = sha256(source)
    protocol_body = {"source_sha256": source_digest, "updates": 800}
    protocol_digest = hashlib.sha256(
        json.dumps(protocol_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(root / "protocol.json", {**protocol_body, "protocol_sha256": protocol_digest})
    write_json(root / "dataset" / "source" / "receipt.json", {
        "source_netcdf_sha256": source_digest,
        "source_netcdf_bytes": source.stat().st_size,
    })
    write_json(root / "continuous_control_result.json", {
        "scientific_claim": False,
        "limitations": ["nested fixture only"],
        "training_identity": "training-fixture-001",
    })
    (root / "code_commit.txt").write_text(COMMIT + "\n", encoding="utf-8")
    (root / "code.zip").write_bytes(b"nested archived code")


def test_builder_creates_a_receipt_that_validator_accepts(tmp_path):
    receipt = make_run(tmp_path)
    # The fixture builder writes the source/protocol/result/commit files; replace
    # the hand-built object with the generator's output for the same directory.
    (tmp_path / "verification_receipt.json").unlink(missing_ok=True)
    generated = build_receipt(
        tmp_path,
        run_id="run-fixture-001",
        workflow="r7-cpu-study",
        status="success",
        commit_sha=COMMIT,
    )
    report = validate_receipt(generated, tmp_path)
    assert report.accepted


def test_strict_builder_and_validator_bind_source_identity(tmp_path):
    make_run(tmp_path)
    generated = build_receipt(
        tmp_path,
        run_id="run-strict-001",
        workflow="r7-cpu-study",
        status="success",
        commit_sha=COMMIT,
    )
    assert generated["format"] == STRICT_FORMAT
    assert validate_receipt(generated, tmp_path).accepted


def test_strict_source_receipt_digest_tampering_is_rejected(tmp_path):
    make_run(tmp_path)
    generated = build_receipt(
        tmp_path,
        run_id="run-strict-002",
        workflow="r7-cpu-study",
        status="success",
        commit_sha=COMMIT,
    )
    receipt_path = tmp_path / "source" / "receipt.json"
    source_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_receipt["source_netcdf_sha256"] = "d" * 64
    write_json(receipt_path, source_receipt)
    for entry in generated["output_artifacts"]:
        if entry["path"] == "source/receipt.json":
            entry["sha256"] = sha256(receipt_path)
    report = validate_receipt(generated, tmp_path)
    assert not report.valid_record
    assert "source receipt digest does not match source artifact bytes" in report.errors


def test_strict_protocol_body_tampering_is_rejected(tmp_path):
    make_run(tmp_path)
    generated = build_receipt(
        tmp_path,
        run_id="run-strict-003",
        workflow="r7-cpu-study",
        status="success",
        commit_sha=COMMIT,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["unapproved_change"] = True
    write_json(protocol_path, protocol)
    for entry in generated["output_artifacts"]:
        if entry["path"] == "protocol.json":
            entry["sha256"] = sha256(protocol_path)
    report = validate_receipt(generated, tmp_path)
    assert not report.valid_record
    assert "protocol digest does not match the canonical protocol body" in report.errors


def test_strict_nested_profile_is_reusable(tmp_path):
    make_nested_run(tmp_path)
    generated = build_receipt(
        tmp_path,
        run_id="run-nested-001",
        workflow="r7-continuous-control",
        status="success",
        commit_sha=COMMIT,
        profile_id="continuous-nested-v1",
    )
    report = validate_receipt(generated, tmp_path)
    assert report.accepted
    assert generated["result_path"] == "continuous_control_result.json"
    assert generated["source_path"] == "dataset/source/era5_continuous250d.nc"


def test_strict_result_and_commit_paths_are_bound(tmp_path):
    make_run(tmp_path)
    generated = build_receipt(
        tmp_path,
        run_id="run-strict-004",
        workflow="r7-cpu-study",
        status="success",
        commit_sha=COMMIT,
    )
    generated["result_path"] = "other.json"
    generated["commit_sha"] = "e" * 40
    report = validate_receipt(generated, tmp_path)
    assert not report.valid_record
    assert any("result_path" in error for error in report.errors)
    assert "commit_sha does not match the code_commit artifact" in report.errors


def test_require_success_exit_codes(tmp_path):
    make_run(tmp_path)
    (tmp_path / "verification_receipt.json").unlink(missing_ok=True)
    cancelled = build_receipt(
        tmp_path,
        run_id="run-cancelled-001",
        workflow="r7-cpu-study",
        status="cancelled",
        commit_sha=COMMIT,
        failure_reason="cancelled before study",
    )
    receipt_path = tmp_path / "verification_receipt.json"
    write_json(receipt_path, cancelled)
    assert verify_main(["--receipt", str(receipt_path), "--root", str(tmp_path)]) == 0
    assert verify_main(["--receipt", str(receipt_path), "--root", str(tmp_path), "--require-success"]) == 1


def test_builder_preserves_failure_state_without_faking_success(tmp_path):
    generated = build_receipt(
        tmp_path,
        run_id="run-fixture-002",
        workflow="r7-cpu-study",
        status="cancelled",
        commit_sha=COMMIT,
        failure_reason="workflow was cancelled before study output existed",
    )
    assert generated["status"] == "cancelled"
    assert generated["scientific_claim"] is False
    report = validate_receipt(generated, tmp_path)
    assert report.valid_record
    assert not report.accepted
def test_success_receipt_checks_files_and_identity(tmp_path):
    receipt = make_run(tmp_path)
    report = validate_receipt(receipt, tmp_path)
    assert report.valid_record
    assert report.accepted
    assert report.errors == ()


@pytest.mark.parametrize("status", ["partial", "failed", "cancelled", "queued", "skipped"])
def test_non_success_terminal_states_are_recorded_but_not_accepted(tmp_path, status):
    receipt = make_run(tmp_path, status=status)
    report = validate_receipt(receipt, tmp_path)
    assert report.valid_record
    assert not report.accepted
    assert report.status == status


def test_non_success_state_without_reason_is_invalid(tmp_path):
    receipt = make_run(tmp_path, status="cancelled")
    receipt.pop("failure_reason")
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert any("failure_reason" in error for error in report.errors)


def test_success_requires_all_identity_fields(tmp_path):
    receipt = make_run(tmp_path)
    receipt["protocol_sha256"] = None
    receipt["source_sha256"] = None
    receipt["data_identity"] = None
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert sum("successful receipt must declare" in error for error in report.errors) == 3


def test_digest_tampering_is_rejected(tmp_path):
    receipt = make_run(tmp_path)
    (tmp_path / "source" / "era5_pressure_pilot.nc").write_bytes(b"tampered")
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert any("does not match source/era5_pressure_pilot.nc" in error for error in report.errors)


def test_missing_required_artifact_is_rejected_for_success(tmp_path):
    receipt = make_run(tmp_path)
    receipt["required_artifacts"].append("missing.json")
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert "successful receipt is missing required artifact: missing.json" in report.errors


def test_result_must_be_explicitly_non_scientific_and_limited(tmp_path):
    receipt = make_run(tmp_path, result={"scientific_claim": True, "limitations": []})
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert any("scientific_claim=false" in error for error in report.errors)
    assert any("nonempty limitations" in error for error in report.errors)


def test_unsafe_artifact_path_is_rejected(tmp_path):
    receipt = make_run(tmp_path)
    receipt["output_artifacts"][0]["path"] = "../outside.json"
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert any("safe path" in error for error in report.errors)


def test_unknown_status_is_rejected(tmp_path):
    receipt = make_run(tmp_path, status="passed")
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert any("status must be one of" in error for error in report.errors)


def test_protocol_and_commit_identity_are_cross_checked(tmp_path):
    receipt = make_run(tmp_path)
    receipt["protocol_sha256"] = "d" * 64
    receipt["commit_sha"] = "e" * 40
    report = validate_receipt(receipt, tmp_path)
    assert not report.valid_record
    assert "protocol_sha256 does not match the protocol artifact" in report.errors
    assert "commit_sha does not match the code_commit artifact" in report.errors
