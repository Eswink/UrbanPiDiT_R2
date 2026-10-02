"""Exercise actual inverse identity gates with in-memory readers, no manifests."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from training.r7_process_training_contract import _actual_train_inverse, _root_inverse

DATA_IDENTITY = "d" * 64


class Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
    def __getitem__(self, index):
        return self.values[index]
    @property
    def shape(self):
        return self.values.shape


class Root(dict):
    def __init__(self):
        super().__init__(normalization_mean=Array(np.arange(17)),
                         normalization_std=Array(np.arange(17) + 1))
        self.attrs = {"channels": [f"channel{i}" for i in range(17)]}


def fake_reader(tmp_path, monkeypatch, identity=DATA_IDENTITY):
    import training.r7_experiment as experiment
    root = Root()
    reader = SimpleNamespace(manifest=tmp_path / "manifests" / "train.jsonl",
                             records=[{"store_path": "../cache.zarr"}], _store=lambda record: root)
    monkeypatch.setattr(experiment, "dataset_identity", lambda path: (identity, reader))
    metadata = {"data_identity": DATA_IDENTITY, "train_manifest": str(reader.manifest),
                "store": str(tmp_path / "cache.zarr")}
    return metadata, root


def test_actual_inverse_reads_fixed_normalization_and_order(tmp_path, monkeypatch):
    metadata, root = fake_reader(tmp_path, monkeypatch)
    fixed = _actual_train_inverse(metadata, DATA_IDENTITY)
    assert fixed["channel_names"] == root.attrs["channels"]
    assert fixed["normalization_mean"] == np.arange(17).astype(float).tolist()
    assert fixed["normalization_std"] == (np.arange(17) + 1).astype(float).tolist()
    assert fixed["data_identity"] == DATA_IDENTITY
    assert not Path(metadata["train_manifest"]).exists()


@pytest.mark.parametrize("kind", ["sidecar_identity", "actual_identity", "store"])
def test_actual_inverse_rejects_distinct_identity_or_store(tmp_path, monkeypatch, kind):
    metadata, root = fake_reader(tmp_path, monkeypatch, identity="e" * 64 if kind == "actual_identity" else DATA_IDENTITY)
    if kind == "sidecar_identity":
        metadata["data_identity"] = "e" * 64
    if kind == "store":
        metadata["store"] = str(tmp_path / "different.zarr")
    with pytest.raises(ValueError, match="identity|store"):
        _actual_train_inverse(metadata, DATA_IDENTITY)


@pytest.mark.parametrize("kind", ["mean_width", "std_zero", "std_nan", "duplicate_name"])
def test_actual_root_inverse_rejects_invalid_normalization_or_order(kind):
    root = Root()
    if kind == "mean_width":
        root["normalization_mean"] = Array(np.zeros(16))
    if kind == "std_zero":
        root["normalization_std"] = Array(np.zeros(17))
    if kind == "std_nan":
        root["normalization_std"] = Array(np.full(17, np.nan))
    if kind == "duplicate_name":
        root.attrs["channels"][1] = root.attrs["channels"][0]
    with pytest.raises(ValueError):
        _root_inverse(root, DATA_IDENTITY)
