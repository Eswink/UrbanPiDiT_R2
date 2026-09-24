"""Determinism and leakage tests for the R7 ERA5 converter.

Acceptance 1 for #13 is that the converter is deterministic *and* leakage-safe.
The existing pipeline tests already cover the leakage split and train-only
statistics; these tests add the byte-level rebuild guarantee, which nothing
asserted before: two independent builds of the same input must produce the same
stored windows, the same manifest records and the same statistics, and held-out
years must not be able to move those statistics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.preprocess.r7_era5 import (
    ERA5ChannelSpec,
    _training_stats,
    build_r7_era5_npz_from_dataset,
    stack_era5_channels,
)

SPECS = (
    ERA5ChannelSpec("t2m", name="t2m"),
    ERA5ChannelSpec("u", 850, "u850"),
    ERA5ChannelSpec("u", 500, "u500"),
)
SPLITS = {"train": [2018], "val": [2019], "test": [2020]}


def _fixture():
    times = []
    for year in (2018, 2019, 2020):
        times.extend(pd.date_range(f"{year}-01-01T00:00:00", periods=8, freq="6h"))
    time = pd.DatetimeIndex(times)
    lat = np.array([40.0, 39.75, 39.5], dtype=np.float32)
    lon = np.array([115.0, 115.25, 115.5, 115.75], dtype=np.float32)
    level = np.array([850, 500], dtype=np.int32)
    spatial = np.add.outer(
        np.arange(len(lat), dtype=np.float32),
        np.arange(len(lon), dtype=np.float32),
    )
    base = np.empty((len(time), len(lat), len(lon)), dtype=np.float32)
    u = np.empty((len(time), len(level), len(lat), len(lon)), dtype=np.float32)
    for i, ts in enumerate(time):
        offset = {2018: 0.0, 2019: 100.0, 2020: 200.0}[ts.year]
        base[i] = 280.0 + offset + 0.2 * i + spatial
        u[i, 0] = 5.0 + 0.1 * offset + 0.1 * i + 0.5 * spatial
        # A genuinely different level pattern, so channel-wise normalization
        # cannot make 500 hPa collapse onto 850 hPa.
        u[i, 1] = 20.0 + 0.1 * offset + 0.3 * i + 2.0 * spatial
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), base),
            "u": (("time", "level", "latitude", "longitude"), u),
        },
        coords={"time": time, "level": level, "latitude": lat, "longitude": lon},
    )


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _records_without_path(records: list[dict]) -> list[dict]:
    """Manifest rows minus the field that legitimately encodes the destination.

    The stored `path` is a relative pointer to the caller's own output directory,
    so two builds at different destinations must differ there and agree
    everywhere else.
    """
    return [{k: v for k, v in row.items() if k != "path"} for row in records]


def _build(tmp_path: Path, label: str, dataset=None):
    manifest_dir = tmp_path / f"manifests_{label}"
    paths = build_r7_era5_npz_from_dataset(
        _fixture() if dataset is None else dataset,
        out_dir=tmp_path / f"processed_{label}",
        manifest_dir=manifest_dir,
        specs=SPECS,
        split_years=SPLITS,
        history_steps=2,
        history_interval_hours=6,
        lead_time_hours=6,
        sample_stride_hours=6,
        expected_grid_spacing_deg=0.25,
        source_label="synthetic-xarray-era5-fixture",
    )
    return manifest_dir, paths


def _payload_digests(manifest_dir: Path, paths: dict[str, Path]) -> dict[str, str]:
    """Hash the stored window bytes and coordinate arrays per split."""
    digests = {}
    for split, path in paths.items():
        digest = hashlib.sha256()
        for record in _records(path):
            data = np.load((manifest_dir / record["path"]).resolve())
            digest.update(record["sample_id"].encode("utf-8"))
            for key in ("coarse_history", "atmos_target"):
                digest.update(np.ascontiguousarray(data[key], dtype="<f4").tobytes())
            digest.update(np.ascontiguousarray(data["latitude"], dtype="<f4").tobytes())
            digest.update(np.ascontiguousarray(data["longitude"], dtype="<f4").tobytes())
        digests[split] = digest.hexdigest()
    return digests


def test_rebuild_is_bit_identical_and_manifests_are_stable(tmp_path: Path):
    manifest_a, paths_a = _build(tmp_path, "a")
    manifest_b, paths_b = _build(tmp_path, "b")

    assert _payload_digests(manifest_a, paths_a) == _payload_digests(manifest_b, paths_b)
    for split in ("train", "val", "test"):
        assert _records_without_path(_records(paths_a[split])) == _records_without_path(
            _records(paths_b[split])
        )
        # The relative pointer must resolve to this build's own window, and each
        # row must live under the split directory it claims.
        for record in _records(paths_a[split]):
            resolved = (manifest_a / record["path"]).resolve()
            assert resolved.is_file()
            assert resolved.parent.name == split
            assert resolved.name == f"{record['sample_id']}.npz"

    normalization_a = json.loads((manifest_a / "normalization.json").read_text(encoding="utf-8"))
    normalization_b = json.loads((manifest_b / "normalization.json").read_text(encoding="utf-8"))
    assert normalization_a == normalization_b
    assert normalization_a["channels"] == ["t2m", "u850", "u500"]
    assert normalization_a["computed_from_years"] == [2018]

    provenance_a = json.loads((manifest_a / "provenance.json").read_text(encoding="utf-8"))
    provenance_b = json.loads((manifest_b / "provenance.json").read_text(encoding="utf-8"))
    # The only legitimate difference is the caller-declared output location.
    assert set(provenance_a) == set(provenance_b)
    for key in set(provenance_a) - {"source"}:
        assert provenance_a[key] == provenance_b[key], key


def test_held_out_years_cannot_move_the_statistics():
    """Leakage guard: shifting only val/test must leave train statistics intact."""
    baseline = _fixture()
    state, _, _, _, _ = stack_era5_channels(baseline, SPECS)
    years = np.asarray(pd.DatetimeIndex(baseline["time"].values).year, dtype=np.int32)
    mean, std = _training_stats(state, years, {2018})

    perturbed = _fixture()
    shifted = perturbed["u"].values.copy()
    index = pd.DatetimeIndex(perturbed["time"].values)
    held_out = np.asarray([year in (2019, 2020) for year in index.year])
    shifted[held_out] += 300.0
    perturbed = perturbed.assign(u=(("time", "level", "latitude", "longitude"), shifted))
    t2m_shifted = perturbed["t2m"].values.copy()
    t2m_shifted[held_out] += 40.0
    perturbed = perturbed.assign(t2m=(("time", "latitude", "longitude"), t2m_shifted))

    other_state, _, _, _, _ = stack_era5_channels(perturbed, SPECS)
    other_mean, other_std = _training_stats(other_state, years, {2018})
    assert np.array_equal(mean, other_mean)
    assert np.array_equal(std, other_std)

    # And the perturbed years really were perturbed, so the test cannot pass
    # merely because the assignment was a no-op.
    assert not np.array_equal(
        np.asarray(baseline["u"].values), np.asarray(perturbed["u"].values)
    )
