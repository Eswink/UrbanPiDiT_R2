"""CPU tests for the #62 audit pack builder and the pack-only table rebuild."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "scripts" / "build_r7_gpu_audit_pack.py"
REBUILD = ROOT / "scripts" / "rebuild_r7_gpu_tables.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sweep_json(status="ok", dtype="bf16-autocast/fp32-master"):
    return {
        "format": "r7-gpu-single-device-sweep-v1",
        "protocol": {"gpu_name": "NVIDIA GeForce RTX 3090",
                     "torch_version": "2.11.0+cu128", "cuda_version": "12.8",
                     "dtype": dtype, "warmup_steps": 1, "measure_steps": 2,
                     "timing_rule": "sync on both edges"},
        "rows": [{"kind": "generic", "k": 4, "training_mode": "full_bptt",
                  "activation_checkpointing": 0, "batch_size": 4, "grid": [12, 12],
                  "in_channels": 11, "dim": 128, "depth": 4, "heads": 4, "window": 4,
                  "patch_size": 2, "dtype": dtype, "status": status,
                  "peak_allocated_bytes": 100 * 2**20,
                  "peak_reserved_bytes": 120 * 2**20, "step_seconds": 0.05}]}


def _seed_outputs(tmp_path):
    source = tmp_path / "outputs" / "gpu_sweep"
    cell = source / "cells" / "generic_K4_full_bptt_ck0"
    cell.mkdir(parents=True)
    (cell / "sweep.json").write_text(json.dumps(_sweep_json()), encoding="utf-8")
    bad = source / "cells" / "generic_K8_full_bptt_ck0"
    bad.mkdir(parents=True)
    (bad / "sweep.json").write_text(json.dumps(_sweep_json(status="nan")),
                                    encoding="utf-8")
    # A binary, an oversized file and a credential lookalike must be excluded.
    (source / "ckpt.pt").write_bytes(b"\x00" * 8)
    (source / "huge.log").write_text("x" * (3 * 1024 * 1024), encoding="utf-8")
    (source / "leak.log").write_text("BEGIN OPENSSH PRIVATE KEY\nnope", encoding="utf-8")
    return source


def test_pack_builder_copies_text_excludes_binaries_and_secrets(tmp_path):
    module = _load("pack_builder", PACK)
    source = _seed_outputs(tmp_path)
    out = tmp_path / "pack"
    manifest = module.build_pack([source], out, model_code_digest="a" * 64)
    assert manifest["format"] == "r7-gpu-audit-pack-v1"
    assert manifest["model_code_sha256"] == "a" * 64
    assert manifest["git_commit"] is None or len(manifest["git_commit"]) == 40
    copied = {entry["path"] for entry in manifest["files"]}
    assert "gpu_sweep/cells/generic_K4_full_bptt_ck0/sweep.json" in copied
    assert manifest["file_count"] == len(manifest["files"]) + 1
    excluded_reasons = {entry["path"]: entry["reason"] for entry in manifest["excluded"]}
    assert any("ckpt.pt" in path for path in excluded_reasons)
    assert any("huge.log" in path for path in excluded_reasons)
    assert any("leak.log" in path for path in manifest["credentials_skipped"])
    assert not (out / "gpu_sweep" / "ckpt.pt").exists()
    assert not (out / "gpu_sweep" / "leak.log").exists()
    statuses = [entry["status"] for entry in manifest["failed_or_skipped_cells"]]
    assert statuses == ["nan"]
    # Write-once: a second build into the same directory must fail.
    with pytest.raises(FileExistsError):
        module.build_pack([source], out, model_code_digest=None)
    # Every copied file's recorded hash matches its bytes; the manifest's own
    # hash lives in the sidecar (it cannot contain itself).
    import hashlib
    for entry in manifest["files"]:
        digest = hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"], entry["path"]
    sidecar = (out / "MANIFEST.sha256").read_text(encoding="utf-8").split()[0]
    assert sidecar == manifest["manifest_sha256"]
    assert hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest() == sidecar


def test_rebuild_derives_tables_from_the_pack_alone_and_flags_problems(tmp_path):
    builder = _load("pack_builder", PACK)
    rebuilder = _load("rebuild_tables", REBUILD)
    source = _seed_outputs(tmp_path)
    pack = tmp_path / "pack"
    builder.build_pack([source], pack, model_code_digest="b" * 64)
    tables = tmp_path / "tables"
    report = rebuilder.rebuild(pack, tables)
    assert report["format"] == "r7-gpu-tables-rebuilt-v1"
    assert report["source_pack"]["model_code_sha256"] == "b" * 64
    # Only the healthy cell becomes a table row; the nan cell is flagged.
    assert len(report["memory_timing_tables"]) == 1
    row = report["memory_timing_tables"][0]
    assert row["kind"] == "generic" and row["k"] == 4
    assert row["training_mode"] == "full_bptt"
    assert row["activation_checkpointing"] is False
    assert row["peak_allocated_mib_mean"] == 100.0
    assert row["step_ms_mean"] == 50.0
    flagged = {problem["cell"] for problem in report["metadata_problems"]}
    assert any("generic_K8" in cell for cell in flagged)
    assert report["metadata_consistent"] is False
    # No multiseed index in this pack: the gap is stated, not invented.
    assert report["multiseed_cells"] is None
    assert any("cells.jsonl" in note for note in report["notes"])
    markdown = (tables / "tables.md").read_text(encoding="utf-8")
    assert "| generic | K=4 | full_bptt |" in markdown
    with pytest.raises(FileExistsError):
        rebuilder.rebuild(pack, tables)


def test_rebuild_rejects_a_directory_without_a_manifest(tmp_path):
    rebuilder = _load("rebuild_tables", REBUILD)
    with pytest.raises(SystemExit, match="not an audit pack"):
        rebuilder.rebuild(tmp_path / "empty", tmp_path / "tables")


def test_rebuild_joins_only_cells_with_one_contract(tmp_path):
    builder = _load("pack_builder", PACK)
    rebuilder = _load("rebuild_tables", REBUILD)
    source = tmp_path / "outputs" / "gpu_sweep"
    cell = source / "cells" / "generic_K4_full_bptt_ck0"
    cell.mkdir(parents=True)
    consistent = _sweep_json()
    (cell / "sweep.json").write_text(json.dumps(consistent), encoding="utf-8")
    other = source / "cells_b" / "generic_K4_full_bptt_ck0"
    other.mkdir(parents=True)
    conflicting = _sweep_json()
    conflicting["rows"][0]["dim"] = 256  # same row key, different contract
    conflicting["rows"][0]["step_seconds"] = 0.07
    (other / "sweep.json").write_text(json.dumps(conflicting), encoding="utf-8")
    pack = tmp_path / "pack"
    builder.build_pack([source], pack, model_code_digest=None)
    report = rebuilder.rebuild(pack, tmp_path / "tables")
    # Cells with a different metadata contract form their own row instead of
    # being averaged into one row with a different config.
    assert report["metadata_consistent"] is True
    rows = report["memory_timing_tables"]
    assert len(rows) == 2
    by_dim = {row["contract"]["dim"]: row for row in rows}
    assert by_dim[128]["cells"] == 1 and by_dim[128]["step_ms_mean"] == 50.0
    assert by_dim[256]["cells"] == 1 and by_dim[256]["step_ms_mean"] == 70.0
