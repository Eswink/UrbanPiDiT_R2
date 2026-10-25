"""The FY multi-year anchor-table builder: pure helpers and its frozen target.

The builder's ``derive`` reads the store under ``outputs/``, which CI does not have, so
this file covers the parts that decide what gets published: the bucket mapping, the
digest the anchor module will recompute at install time, the module's spec schema, and
the instance/manifest paths the tool is pinned to. Nothing here touches ``outputs/``.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "fy_anchor_table_build", ROOT / "scripts/r7_s3_fy_anchor_table_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bucket_index_reads_utc_month_and_synoptic_hour():
    import numpy as np
    builder = _builder()
    stamps = np.array([
        "2000-01-01T00:00", "2000-01-01T06:00", "2000-12-31T18:00", "2021-07-15T12:00",
    ], dtype="datetime64[m]").astype("datetime64[ns]").astype("int64")
    months, hours = builder.bucket_index(stamps)
    assert list(months) == [1, 1, 12, 7]
    assert list(hours) == [0, 6, 18, 12]


def test_table_digest_is_the_digest_the_anchor_module_recomputes():
    import torch

    from model.climatology_anchor_r7 import _table_digest
    builder = _builder()
    table = torch.arange(48, dtype=torch.float32).reshape(1, 1, 4, 12) / 7.0
    assert builder.table_digest(table) == _table_digest(table)
    assert builder.table_digest(table) == hashlib.sha256(
        table.contiguous().numpy().tobytes(order="C")).hexdigest()


def test_a_minimal_spec_validates_against_the_anchor_module_schema():
    import torch

    from model.climatology_anchor_r7 import TrainClimatologyAnchor
    builder = _builder()
    table = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    spec = {
        "format": builder.FORMAT, "kind": builder.KIND, "channels": ["t2m"], "units": ["K"],
        "source_sha256": "a" * 64, "train_data_identity": "b" * 64,
        "climatology_mean_identity_sha256": "c" * 64,
        "normalization_mean": [0.0], "normalization_std": [1.0],
        "latitude": [0.0, 1.0], "longitude": [0.0, 1.0],
        "bucket_keys": [[1, 0]], "bucket_counts": [3], "training_years": [2000],
        "n_selected_steps": 3, "selection": "declared_train_years",
        "table_sha256": builder.table_digest(table), "table_shape": [1, 1, 2, 2],
    }
    anchor = TrainClimatologyAnchor(spec)
    anchor.install(table)
    assert bool(anchor.ready)


def test_a_spec_with_the_wrong_key_set_is_rejected():
    import torch

    from model.climatology_anchor_r7 import TrainClimatologyAnchor
    builder = _builder()
    table = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    spec = {
        "format": builder.FORMAT, "kind": builder.KIND, "channels": ["t2m"], "units": ["K"],
        "source_sha256": "a" * 64, "train_data_identity": "b" * 64,
        "climatology_mean_identity_sha256": "c" * 64,
        "normalization_mean": [0.0], "normalization_std": [1.0],
        "latitude": [0.0, 1.0], "longitude": [0.0, 1.0],
        "bucket_keys": [[1, 0]], "bucket_counts": [3], "training_years": [2000],
        "n_selected_steps": 3, "selection": "declared_train_years",
        "table_sha256": builder.table_digest(table), "table_shape": [1, 1, 2, 2],
        "extra": 1,
    }
    with pytest.raises(ValueError):
        TrainClimatologyAnchor(spec)


def test_the_builder_is_pinned_to_the_multi_year_instance_not_the_wide_one():
    """A silent repoint to the wide store would produce a 129x129 table for a 65x65 round."""
    builder = _builder()
    assert builder.INSTANCE.name == "r7_s3_fy_multiyear_20261023_attempt01"
    assert builder.CACHE == builder.STORE / "cache.zarr"
    assert builder.TRAIN_MANIFEST == builder.STORE / "manifests_treatment/train.jsonl"
    assert builder.SOURCE_PREFLIGHT == builder.INSTANCE / "preflight_store_report.json"
    assert builder.GRID_POINTS == 65
    assert "wide" not in str(builder.INSTANCE)
    assert builder.DEFAULT_OUT.name.startswith("r7_s3_fy_anchor_table_")
