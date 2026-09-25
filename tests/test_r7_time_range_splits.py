"""Tests for the explicit time-range split mode of the audited zarr builder.

D1 is one continuous 30-day slice of a single year, so the year-based split
contract cannot express three non-empty splits. ``split_time_ranges`` adds
half-open ``[start, stop)`` engineering sub-splits with the same fail-closed
guarantees: disjoint, chronological, never crossing a boundary, and confined
to the declared train years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from data.preprocess.r7_era5 import ERA5ChannelSpec
from data.preprocess.r7_era5_zarr import (
    _window_records_in_ranges,
    build_r7_era5_zarr_from_dataset,
    parse_split_time_ranges,
)
from data.r7_store import validate_store

RANGES = {
    "train": [["2016-01-01T00:00:00", "2016-01-10T00:00:00"]],
    "val": [["2016-01-10T00:00:00", "2016-01-15T00:00:00"]],
    "test": [["2016-01-15T00:00:00", "2016-01-20T00:00:00"]],
}


def regional_dataset(stamps):
    """A tiny physical-unit regional dataset in the audited source schema."""
    times = pd.date_range("2016-01-01", periods=stamps, freq="6h")
    lat = np.arange(43., 26.99, -0.25)
    lon = np.arange(107., 123.01, 0.25)
    t, c, h, w = len(times), 2, len(lat), len(lon)
    rng = np.random.default_rng(7)
    data_vars = {
        "2m_temperature": (("time", "latitude", "longitude"),
                           rng.normal(280., 3., (t, h, w)).astype("f4")),
        "temperature": (("time", "level", "latitude", "longitude"),
                        rng.normal(270., 5., (t, 3, h, w)).astype("f4")),
    }
    ds = xr.Dataset(data_vars, coords={
        "time": times, "level": np.asarray([850., 500., 250.], dtype="int32"),
        "latitude": lat, "longitude": lon})
    ds["2m_temperature"].attrs["units"] = "K"
    ds["temperature"].attrs["units"] = "K"
    ds["level"].attrs["units"] = "hPa"
    return ds


SPECS = (ERA5ChannelSpec("2m_temperature", name="t2m"),
         ERA5ChannelSpec("temperature", 850, "t850"),
         ERA5ChannelSpec("temperature", 500, "t500"))


def build(tmp_path, stamps=60, ranges=RANGES, split_years=None):
    ds = regional_dataset(stamps)
    paths = build_r7_era5_zarr_from_dataset(
        ds, store_path=tmp_path / "store.zarr", manifest_dir=tmp_path / "manifests",
        specs=SPECS, split_years=split_years or {"train": [2016], "val": [2017], "test": [2018]},
        split_time_ranges=ranges, compute_process_targets=False,
        time_chunk=16, spatial_chunk=(64, 64))
    return paths


def test_parse_rejects_bad_ranges():
    with pytest.raises(ValueError, match="exactly"):
        parse_split_time_ranges({"train": [["2016-01-01", "2016-02-01"]]})
    with pytest.raises(ValueError, match="start < stop"):
        parse_split_time_ranges({"train": [["2016-02-01", "2016-01-01"]],
                                 "val": [["2016-03-01", "2016-04-01"]],
                                 "test": [["2016-05-01", "2016-06-01"]]})
    overlapping = {"train": [["2016-01-01", "2016-05-01"], ["2016-04-01", "2016-06-01"]],
                   "val": [["2016-07-01", "2016-08-01"]],
                   "test": [["2016-09-01", "2016-10-01"]]}
    with pytest.raises(ValueError, match="overlap"):
        parse_split_time_ranges(overlapping)
    disordered = {"train": [["2016-06-01", "2016-07-01"]],
                  "val": [["2016-02-01", "2016-03-01"]],
                  "test": [["2016-09-01", "2016-10-01"]]}
    with pytest.raises(ValueError, match="chronological"):
        parse_split_time_ranges(disordered)
    with pytest.raises(ValueError, match="naive"):
        parse_split_time_ranges({"train": [["2016-01-01T00:00:00+00:00", "2016-02-01T00:00:00+00:00"]],
                                 "val": [["2016-03-01", "2016-04-01"]],
                                 "test": [["2016-05-01", "2016-06-01"]]})


def test_windows_never_cross_a_range_boundary():
    times = pd.date_range("2016-01-01", periods=80, freq="6h")  # 20 days
    ranges = parse_split_time_ranges(RANGES)
    records = _window_records_in_ranges(
        times, split_ranges=ranges, store_path=__import__("pathlib").Path("s.zarr"),
        manifest_dir=__import__("pathlib").Path("m"), history_steps=2,
        history_interval_hours=6, lead_time_hours=6, sample_stride_hours=6)
    # 2016-01-09T18:00 init would need the 2016-01-10T00:00 target inside the
    # train range [01, 10); it must be excluded, not reassigned.
    train_inits = [pd.Timestamp(r["init_time"]) for r in records["train"]]
    assert max(train_inits) == pd.Timestamp("2016-01-09T12:00")
    val_inits = [pd.Timestamp(r["init_time"]) for r in records["val"]]
    assert min(val_inits) == pd.Timestamp("2016-01-10T06:00")
    for split in ("train", "val", "test"):
        for record in records[split]:
            assert record["lead_time_hours"] == 6
            times_iso = record["history_times"] + [record["target_time"]]
            stamps = [pd.Timestamp(value) for value in times_iso]
            span = (stamps[-1] - stamps[0]).total_seconds() / 3600
            assert span == 12


def test_builder_publishes_range_split_store(tmp_path):
    paths = build(tmp_path, stamps=80)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    validate_store(root)
    assert root.attrs["split_mode"] == "time_ranges"
    assert root.attrs["split_time_ranges"]["train"] == [
        ["2016-01-01T00:00:00", "2016-01-10T00:00:00"]]
    assert root.attrs["normalization_years"] == [2016]
    counts = {split: len(paths[split].read_text(encoding="utf-8").splitlines())
              for split in ("train", "val", "test")}
    assert counts == {"train": 34, "val": 18, "test": 18}
    # normalization statistics must come from the train range only
    mean = np.asarray(root["normalization_mean"])
    ds = regional_dataset(80)
    train = ds["2m_temperature"].sel(time=slice("2016-01-01", "2016-01-09T18:00")).values
    np.testing.assert_allclose(mean[0], train.mean(), rtol=1e-4)


def test_builder_refuses_ranges_outside_declared_train_years(tmp_path):
    with pytest.raises(ValueError, match="declared train years"):
        build(tmp_path, split_years={"train": [2017], "val": [2018], "test": [2019]})


def test_builder_refuses_year_normalization_mismatch(tmp_path):
    ds = regional_dataset(20)
    ranges = {"train": [["2016-01-01T00:00:00", "2016-01-06T00:00:00"]],
              "val": [["2016-01-06T00:00:00", "2016-01-10T00:00:00"]],
              "test": [["2016-01-10T00:00:00", "2016-01-20T00:00:00"]]}
    # the range covers only 2016 while two train years are declared
    with pytest.raises(ValueError, match="declared train years are"):
        build_r7_era5_zarr_from_dataset(
            ds, store_path=tmp_path / "store.zarr", manifest_dir=tmp_path / "manifests",
            specs=SPECS, split_years={"train": [2016, 2017], "val": [2018], "test": [2019]},
            split_time_ranges=ranges, compute_process_targets=False)


def test_builder_refuses_empty_split(tmp_path):
    ds = regional_dataset(12)
    ranges = {"train": [["2016-01-01T00:00:00", "2016-01-03T00:00:00"]],
              "val": [["2016-01-03T00:00:00", "2016-01-05T00:00:00"]],
              "test": [["2016-01-20T00:00:00", "2016-01-25T00:00:00"]]}
    with pytest.raises(ValueError, match="every split needs"):
        build_r7_era5_zarr_from_dataset(
            ds, store_path=tmp_path / "store.zarr", manifest_dir=tmp_path / "manifests",
            specs=SPECS, split_years={"train": [2016], "val": [2017], "test": [2018]},
            split_time_ranges=ranges, compute_process_targets=False)


def test_year_mode_unchanged(tmp_path):
    """The default year-based path must behave exactly as before the extension."""
    from data.preprocess.r7_era5_zarr import _window_records
    times = pd.date_range("2018-01-01", periods=40, freq="6h")
    records = _window_records(times, split_sets={"train": {2018}, "val": {2019}, "test": {2020}},
                              store_path=__import__("pathlib").Path("s.zarr"),
                              manifest_dir=__import__("pathlib").Path("m"),
                              history_steps=2, history_interval_hours=6,
                              lead_time_hours=6, sample_stride_hours=6)
    assert len(records["train"]) == 38 and records["val"] == [] and records["test"] == []
    blocks = []
    for year in (2018, 2019, 2020):
        block = regional_dataset(40)
        block = block.assign_coords(time=pd.date_range(f"{year}-01-01", periods=40, freq="6h"))
        blocks.append(block)
    ds = xr.concat(blocks, dim="time")
    paths = build_r7_era5_zarr_from_dataset(
        ds, store_path=tmp_path / "store.zarr", manifest_dir=tmp_path / "manifests",
        specs=SPECS, split_years={"train": [2018], "val": [2019], "test": [2020]},
        time_chunk=16, spatial_chunk=(64, 64), compute_process_targets=False)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    validate_store(root)
    assert "split_mode" not in root.attrs
    assert len(paths["train"].read_text(encoding="utf-8").splitlines()) == 38
