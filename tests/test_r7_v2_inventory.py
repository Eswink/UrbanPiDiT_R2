"""Pure metadata/opaque-byte inventory guards; no real B, tensors, weather or GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training import r7_v2_results as results
from training import r7_v2_protocol as protocol


@pytest.fixture
def owned(tmp_path):
    output = tmp_path / "r7_v2_comparison_20261003_attempt01"
    folder = output / "seed41/training/process"
    folder.mkdir(parents=True)
    files = {
        "protocol.json": b"opaque fixture protocol metadata",
        "seed41/training/process/fine_tune_started.json": b"opaque startup metadata",
        "seed41/training/process/update_0000020.pt": b"opaque intermediate bytes, not tensor serialization",
        "seed41/training/process/update_0000400.pt": b"opaque endpoint bytes, not tensor serialization",
        "seed41/training/process/training_report.json": b"opaque report metadata",
        "stats01worker.log": b"owned extra fixture log",
    }
    for name, content in files.items():
        (output / name).write_bytes(content)
    required = [output / name for name in ("protocol.json", "seed41/training/process/update_0000400.pt",
                                           "seed41/training/process/training_report.json")]
    return output, files, required


def test_all_regular_startup_intermediate_and_extra_logs_pinned_before_after(owned):
    from training import r7_v2_inventory as inventory
    output, files, required = owned
    pins = inventory.pin_inventory(output, required)
    assert set(pins) == set(files)
    assert all(pins[name] == protocol.sha256_file(output / name) for name in files)
    protocol.write_json(output / "artifact_manifest.json", {"files_sha256": pins}, output=output)
    assert inventory.verify_inventory(output, pins) == pins
    assert inventory.pin_inventory(output, required) == pins
    assert (output / "seed41/training/process/update_0000020.pt").read_bytes() == files["seed41/training/process/update_0000020.pt"]


def test_only_exact_three_root_files_excluded_nested_names_included(owned):
    from training import r7_v2_inventory as inventory
    output, files, required = owned
    excluded = {"artifact_manifest.json", "attempt.json", "execution_attempt.json"}
    assert set(inventory.EXCLUDED_RECURSIVE_OR_FUTURE) == excluded
    nested = output / "nested"
    nested.mkdir()
    for name in excluded:
        (output / name).write_bytes(b"opaque future receipt")
        (nested / name).write_bytes(b"ordinary nested file with same basename")
    pins = inventory.pin_inventory(output, required)
    assert set(pins) == set(files) | {"nested/" + name for name in excluded}
    assert inventory.verify_inventory(output, pins) == pins


@pytest.mark.parametrize("kind", ["file", "directory", "broken", "excluded", "root"])
def test_any_file_directory_or_broken_symlink_refused(owned, tmp_path, kind):
    from training import r7_v2_inventory as inventory
    output, _, required = owned
    external = tmp_path / "external"
    external.mkdir()
    (external / "opaque.txt").write_bytes(b"external must not be consumed")
    if kind == "root":
        alias = tmp_path / "aliases" / output.name
        alias.parent.mkdir()
        alias.symlink_to(output, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            inventory.pin_inventory(alias, required)
        return
    link = output / ("attempt.json" if kind == "excluded" else "linked")
    target = external if kind == "directory" else external / ("missing" if kind == "broken" else "opaque.txt")
    link.symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="symlink"):
        inventory.pin_inventory(output, required)
    assert (external / "opaque.txt").read_bytes() == b"external must not be consumed"


def test_fifo_special_file_refused_without_open_or_hash(owned, monkeypatch):
    from training import r7_v2_inventory as inventory
    output, _, required = owned
    os.mkfifo(output / "fifo")
    monkeypatch.setattr(inventory, "sha256_file", lambda *args: pytest.fail("special file must fail before any hash"))
    with pytest.raises(ValueError, match="special file"):
        inventory.pin_inventory(output, required)


@pytest.mark.parametrize("kind", ["file", "broken", "directory"])
def test_publication_failure_marker_always_refuses_even_with_opaque_contents(owned, kind):
    from training import r7_v2_inventory as inventory
    output, _, required = owned
    pins = inventory.pin_inventory(output, required)
    marker = output / "publication_failure.json"
    if kind == "file":
        marker.write_bytes(b"invalid JSON must still prohibit acceptance")
    elif kind == "broken":
        marker.symlink_to(output / "absent")
    else:
        marker.mkdir()
    with pytest.raises(ValueError, match="authoritative publication failure"):
        inventory.pin_inventory(output, required)
    with pytest.raises(ValueError, match="authoritative publication failure"):
        inventory.verify_inventory(output, pins)


@pytest.mark.parametrize("kind", ["missing", "directory", "excluded"])
def test_required_validated_subset_cannot_be_satisfied_by_scan_or_exclusion(owned, kind):
    from training import r7_v2_inventory as inventory
    output, _, required = owned
    missing = output / "missing-required.json"
    if kind == "directory":
        missing.mkdir()
    elif kind == "excluded":
        missing = output / "attempt.json"
        missing.write_bytes(b"excluded cannot satisfy a mandatory selected artifact")
    with pytest.raises(ValueError, match="required validated artifacts missing"):
        inventory.pin_inventory(output, [*required, missing])


@pytest.mark.parametrize("change", ["addition", "removal", "bytes", "symlink"])
def test_post_manifest_exact_inventory_and_opaque_hash_changes_refused(owned, change):
    from training import r7_v2_inventory as inventory
    output, _, required = owned
    pins = inventory.pin_inventory(output, required)
    protocol.write_json(output / "artifact_manifest.json", {"files_sha256": pins}, output=output)
    intermediate = output / "seed41/training/process/update_0000020.pt"
    if change == "addition":
        (output / "late.log").write_bytes(b"late owned log")
    elif change == "removal":
        intermediate.unlink()
    elif change == "bytes":
        intermediate.write_bytes(b"changed opaque checkpoint bytes")
    else:
        (output / "late-link").symlink_to(output / "missing")
    with pytest.raises(ValueError, match="inventory"):
        inventory.verify_inventory(output, pins)


def test_addition_during_last_hash_refused_on_second_enumeration(owned, monkeypatch):
    from training import r7_v2_inventory as inventory
    output, _, required = owned
    pins = inventory.pin_inventory(output, required)
    original = inventory.sha256_file
    last = sorted(pins)[-1]

    def hash_then_add(path):
        result = original(path)
        if path.relative_to(output).as_posix() == last:
            (output / "hash-late.log").write_bytes(b"owned mutation during final hashing")
        return result

    monkeypatch.setattr(inventory, "sha256_file", hash_then_add)
    with pytest.raises(ValueError, match="inventory changed"):
        inventory.verify_inventory(output, pins)


@pytest.fixture
def metadata_finalize(owned, monkeypatch):
    """Exercise actual finalize publication tail, stubbing science gates, not inventory."""
    output, _, required = owned
    profile = {"model_code_sha256": "a" * 64, "source_tree_sha256": "b" * 64,
        "code_zip_sha256": "c" * 64, "base_commit": "d" * 40, "working_tree_modified": True,
        "code_commit_sha256": "e" * 64, "code_status_sha256": "f" * 64}
    frozen = {"stage": "C", "arm_configs": {arm: {} for arm in protocol.C_ARMS}, "code": profile,
              "data": {"data_identity": "1" * 64}, "sources": {"source_sha256": "2" * 64},
              "sidecar": {"identity": "3" * 64}, "windows": {}, "protocol_sha256": "4" * 64,
              "whole_round_cost_reference": str(output / "attempt.json")}
    selected = [{"checkpoint": str(required[1]), "training_report": str(required[2])}]
    for name in ("cpu_profile.json", "code.zip", "code_commit.txt", "code_status.txt", "environment.json",
                 "prepare_started.json", "prepare_attempt.json", "run_started.json"):
        (output / name).write_bytes(b"opaque prepare metadata only")
    monkeypatch.setattr(results, "validate_protocol", lambda value: value)
    monkeypatch.setattr(results, "verify_execution", lambda *args: [])
    monkeypatch.setattr(results, "validate_full_set", lambda *args: ({(41, "process"): selected[0]}, {}, []))
    monkeypatch.setattr(results, "aggregate_metrics", lambda *args, **kwargs: [{} for _ in range(3 * 765)])
    monkeypatch.setattr(results, "paired_comparisons", lambda *args, **kwargs:
                        {"fixture": {"cells": {str(index): {} for index in range(765)}}})
    monkeypatch.setattr(results, "descriptive_outcome", lambda *args: {
        "scientific_claim": False, "limitations": ["metadata publication test only; science gates stubbed"], "paused": False})
    monkeypatch.setattr(results, "publish_tables", lambda *args: [])
    from training import r7_v2_frontier
    monkeypatch.setattr(r7_v2_frontier, "evaluate_adaptive_gate", lambda *args, **kwargs: {"status": "fixture-only"})
    return output, frozen, required


def test_finalize_pins_actual_startup_intermediate_and_owned_extra_log(metadata_finalize):
    output, frozen, _ = metadata_finalize
    results.finalize(output, frozen, {"scientific_claim": False})
    manifest = protocol.read_json(output / "artifact_manifest.json")
    for name in ("seed41/training/process/fine_tune_started.json",
                 "seed41/training/process/update_0000020.pt", "stats01worker.log"):
        assert manifest["files_sha256"][name] == protocol.sha256_file(output / name)
    excluded = {"artifact_manifest.json", "attempt.json", "execution_attempt.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} - excluded
    assert set(manifest["files_sha256"]) == actual
    assert set(manifest["excluded_recursive_or_future"]) == excluded
    assert manifest["files_digest"] == protocol.digest(manifest["files_sha256"])


def test_finalize_missing_required_selected_artifact_refuses(metadata_finalize):
    output, frozen, required = metadata_finalize
    required[1].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        results.finalize(output, frozen, {"scientific_claim": False})
    assert not (output / "artifact_manifest.json").exists()


@pytest.mark.parametrize("change", ["addition", "bytes", "marker"])
def test_finalize_rechecks_inventory_and_hashes_after_manifest_publication(metadata_finalize, monkeypatch, change):
    output, frozen, _ = metadata_finalize
    original = results.write_json

    def publish_then_change(path, value, **kwargs):
        result = original(path, value, **kwargs)
        if path.name == "artifact_manifest.json":
            target = (output / "late.log" if change == "addition" else output / "publication_failure.json"
                      if change == "marker" else output / "stats01worker.log")
            target.write_bytes(b"opaque metadata changed after manifest publication")
        return result

    monkeypatch.setattr(results, "write_json", publish_then_change)
    with pytest.raises(ValueError, match="inventory|authoritative publication failure"):
        results.finalize(output, frozen, {"scientific_claim": False})
    assert (output / "artifact_manifest.json").is_file()
