from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.loader import BeijingWeatherDataset, compute_train_climatology_mean_field, parse_hour_from_filename
from utils.config_builder import build_datamodule_kwargs


DYNAMIC_VARS = ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"]
STATIC_VARS = ["landcover", "building_surface", "buildings", "building_volume", "population"]


def _write_norm_files(root: str) -> None:
    np.savez(Path(root) / "normalize_mean_train.npz", **{name: np.array([0.0], dtype=np.float32) for name in DYNAMIC_VARS})
    np.savez(Path(root) / "normalize_std_train.npz", **{name: np.array([1.0], dtype=np.float32) for name in DYNAMIC_VARS})
    np.savez(Path(root) / "normalize_static_mean_train.npz", **{name: np.array([0.0], dtype=np.float32) for name in STATIC_VARS[1:]})
    np.savez(Path(root) / "normalize_static_std_train.npz", **{name: np.array([1.0], dtype=np.float32) for name in STATIC_VARS[1:]})


def test_climatology_npy_mean_field_counts_each_file_once() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = Path(tmp) / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        for i, hour in enumerate([0, 6, 12]):
            arr = np.zeros((12, 2, 2), dtype=np.float32)
            for ch in range(7):
                arr[ch] = float((ch + 1) * 10 + i)
            np.save(train_dir / f"2020_01_01_{hour:02d}.npy", arr)

        clim = compute_train_climatology_mean_field(tmp, ["d2m", "t2m", "v10"])

        np.testing.assert_allclose(clim["d2m"], np.full((2, 2), 11.0, dtype=np.float32))
        np.testing.assert_allclose(clim["t2m"], np.full((2, 2), 31.0, dtype=np.float32))
        np.testing.assert_allclose(clim["v10"], np.full((2, 2), 71.0, dtype=np.float32))


def test_climatology_npz_mean_field_uses_time_axis() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = Path(tmp) / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        d2m = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 3.0), np.full((2, 2), 5.0)], axis=0)
        t2m = np.stack([np.full((2, 2), 10.0), np.full((2, 2), 14.0)], axis=0)
        np.savez(train_dir / "seq.npz", d2m=d2m.astype(np.float32), t2m=t2m.astype(np.float32))

        clim = compute_train_climatology_mean_field(tmp, ["d2m", "t2m"])

        np.testing.assert_allclose(clim["d2m"], np.full((2, 2), 3.0, dtype=np.float32))
        np.testing.assert_allclose(clim["t2m"], np.full((2, 2), 12.0, dtype=np.float32))


def test_expose_hour_is_config_gated_for_npy_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = Path(tmp) / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        _write_norm_files(tmp)
        for i, hour in enumerate([0, 6, 12, 18]):
            arr = np.zeros((12, 2, 2), dtype=np.float32)
            arr[:7] = float(i)
            arr[7] = 1.0
            arr[8:] = 2.0
            np.save(train_dir / f"2020_01_01_{hour:02d}.npy", arr)

        common = dict(
            split_dir=train_dir,
            k=2,
            delta_t=1,
            lead_times=None,
            dynamic_vars=DYNAMIC_VARS,
            static_vars=STATIC_VARS,
            include_static=True,
            broadcast_static=True,
            normalize_root=tmp,
        )
        ds_off = BeijingWeatherDataset(**common, expose_hour=False)
        ds_on = BeijingWeatherDataset(**common, expose_hour=True)

        assert "hour_of_day" not in ds_off[0]
        sample = ds_on[0]
        assert int(sample["hour_of_day"].item()) == 6
        assert bool(sample["diurnal_available"].item()) is True


def test_parse_hour_and_builder_gate_defaults() -> None:
    assert parse_hour_from_filename("/x/2020_01_02_00.npy") == 0
    assert parse_hour_from_filename("/x/2020_01_02_23.npy") == 23
    assert parse_hour_from_filename("/x/not_timestamp.npy") is None

    cfg = {
        "data_root": "/tmp/data",
        "k": 2,
        "delta_t": 1,
        "dynamic_vars": DYNAMIC_VARS,
        "static_vars": STATIC_VARS,
    }
    assert build_datamodule_kwargs(cfg)["expose_hour"] is False
    cfg["data"] = {"expose_hour": True}
    assert build_datamodule_kwargs(cfg)["expose_hour"] is True