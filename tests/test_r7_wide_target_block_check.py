"""The wide target-block check, with its own positive and negative controls.

A comparison tool that only ever reports "pass" proves nothing, so this file
builds a synthetic wide/reference pair whose central block is identical, then
perturbs one cell, one unit and one stamp to prove the check actually rejects
each of them. No network, no GPU, no repository data.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "scripts/verify_r7_wide_target_block.py"
    spec = importlib.util.spec_from_file_location("verify_r7_wide_target_block", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load()
LEVELS = [250, 500, 850]
PRESSURE = ("geopotential", "temperature", "specific_humidity",
            "u_component_of_wind", "v_component_of_wind")
SURFACE = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
           "mean_sea_level_pressure")
UNITS = {"geopotential": "m**2 s**-2", "temperature": "K", "specific_humidity": "kg kg**-1",
         "u_component_of_wind": "m s**-1", "v_component_of_wind": "m s**-1",
         "2m_temperature": "K", "10m_u_component_of_wind": "m s**-1",
         "10m_v_component_of_wind": "m s**-1", "mean_sea_level_pressure": "Pa"}


def _dataset(path, height, width, *, times, row_offset=0.0, column_offset=0.0, units=None, seed=1):
    import xarray as xr

    rng = np.random.default_rng(seed)
    latitude = 27.0 + 0.25 * (np.arange(height) + row_offset)
    longitude = 107.0 + 0.25 * (np.arange(width) + column_offset)
    data = {}
    for name in PRESSURE:
        values = rng.standard_normal((len(times), len(LEVELS), height, width)).astype(np.float32)
        data[name] = (("time", "level", "latitude", "longitude"), values + 300.0)
    for name in SURFACE:
        values = rng.standard_normal((len(times), height, width)).astype(np.float32)
        data[name] = (("time", "latitude", "longitude"), values + 280.0)
    dataset = xr.Dataset(data, coords={"time": times, "level": np.asarray(LEVELS, dtype=np.int32),
                                       "latitude": latitude.astype(np.float32),
                                       "longitude": longitude.astype(np.float32)})
    for name in PRESSURE + SURFACE:
        dataset[name].attrs["units"] = (units or UNITS)[name]
    dataset.to_netcdf(path, engine="h5netcdf")
    return dataset


def _pair(tmp_path, *, geometry=(9, 9, 5, 2), stamps=4, perturb=None,
          wide_units=None, reference_units=None):
    """A wide/reference pair whose central block is *literally* the reference data."""
    import pandas as pd
    import xarray as xr

    height, width, points, margin = geometry
    times = pd.date_range("2018-01-01T00:00", periods=stamps, freq="6h")
    wide = tmp_path / "wide.nc"
    reference = tmp_path / "reference.nc"
    _dataset(reference, points, points, times=times, units=reference_units,
             row_offset=float(margin), column_offset=float(margin), seed=2)
    _dataset(wide, height, width, times=times, units=wide_units, seed=1)
    rows, columns = slice(margin, margin + points), slice(margin, margin + points)
    with xr.open_dataset(wide) as source, xr.open_dataset(reference) as target:
        dataset = source.load()
        for name in PRESSURE + SURFACE:
            values = dataset[name].values
            values[..., rows, columns] = target[name].values
            dataset[name].values[...] = values
        if perturb is not None:
            values = dataset[perturb["variable"]].values
            values[(slice(None),) * (values.ndim - 2)
                   + (perturb["row"] + margin, perturb["column"] + margin)] += perturb["delta"]
            dataset[perturb["variable"]].values[...] = values
    dataset.to_netcdf(wide, engine="h5netcdf")
    return wide, reference, rows, columns


def test_identical_central_block_passes(tmp_path):
    wide, reference, rows, columns = _pair(tmp_path)
    report = CHECK.compare_central_block(wide, reference, rows=rows, columns=columns)
    assert report["verdict"] == "pass"
    assert report["stamps"] == 4
    assert report["stacked_channels"]["bit_identical"] is True
    assert report["stacked_channels"]["count"] == 17
    assert report["units_match"] is True
    assert all(value["bit_identical"] for value in report["stored_variables"].values())


def test_a_single_perturbed_cell_fails(tmp_path):
    wide, reference, rows, columns = _pair(
        tmp_path, perturb={"variable": "2m_temperature", "row": 3, "column": 4, "delta": 0.5})
    report = CHECK.compare_central_block(wide, reference, rows=rows, columns=columns)
    assert report["verdict"] == "fail"
    assert report["stored_variables"]["2m_temperature"]["bit_identical"] is False
    assert report["stored_variables"]["2m_temperature"]["max_abs_diff"] == pytest.approx(0.5)
    assert report["stored_variables"]["geopotential"]["bit_identical"] is True
    assert report["stacked_channels"]["bit_identical"] is False


def test_a_perturbation_outside_the_central_block_is_ignored(tmp_path):
    # The block selector must actually select: a difference outside the central
    # block is not this check's business and must not be reported as one.
    import xarray as xr

    wide, reference, rows, columns = _pair(tmp_path, stamps=2)
    with xr.open_dataset(wide) as source:
        dataset = source.load()
    values = dataset["2m_temperature"].values
    values[0, 0, 0] += 9.0  # corner cell, outside the central 5x5 block
    dataset["2m_temperature"].values[...] = values
    dataset.to_netcdf(wide, engine="h5netcdf")
    report = CHECK.compare_central_block(wide, reference, rows=rows, columns=columns)
    assert report["verdict"] == "pass"
    assert report["stored_variables"]["2m_temperature"]["bit_identical"] is True


def test_a_unit_change_fails(tmp_path):
    wide, reference, rows, columns = _pair(
        tmp_path, wide_units=dict(UNITS, **{"2m_temperature": "degC"}))
    report = CHECK.compare_central_block(wide, reference, rows=rows, columns=columns)
    assert report["verdict"] == "fail"
    assert report["units_match"] is False
    assert report["stored_variables"]["2m_temperature"]["unit_wide"] == "degC"


def test_a_missing_stamp_fails_closed(tmp_path):
    import pandas as pd

    wide, reference, rows, columns = _pair(tmp_path, stamps=4)
    # Replace the reference with a file that starts later, so a wide stamp has no
    # counterpart; the check must refuse rather than silently compare fewer stamps.
    import xarray as xr

    with xr.open_dataset(reference) as source:
        dataset = source.load()
    later = dataset.isel(time=slice(1, None))
    later.to_netcdf(reference, engine="h5netcdf")
    report = CHECK.compare_central_block(wide, reference, rows=rows, columns=columns)
    assert report["verdict"] == "fail"
    assert "absent from the reference" in report["error"]


def test_the_default_block_comes_from_the_wide_read_plan(tmp_path):
    from data.download.read_plan_wide import central_block

    assert central_block() == (slice(32, 97), slice(32, 97))


def test_a_reference_longer_than_the_wide_file_is_handled_by_time_not_by_index(tmp_path):
    # The registered reference covers every split year while a pilot part covers
    # one season, so the shared stamps sit at non-zero reference positions. A
    # positional index into the wide file would silently read the wrong stamps.
    import pandas as pd
    import xarray as xr

    times = pd.date_range("2018-01-01T00:00", periods=4, freq="6h")
    reference = tmp_path / "reference.nc"
    wide = tmp_path / "wide.nc"
    _dataset(reference, 5, 5, times=times, row_offset=2.0, column_offset=2.0, seed=2)
    _dataset(wide, 9, 9, times=times[1:], seed=1)
    with xr.open_dataset(wide) as source, xr.open_dataset(reference) as target:
        dataset = source.load()
        for name in PRESSURE + SURFACE:
            values = dataset[name].values
            values[..., 2:7, 2:7] = target[name].values[1:]
            dataset[name].values[...] = values
    dataset.to_netcdf(wide, engine="h5netcdf")
    report = CHECK.compare_central_block(wide, reference, rows=slice(2, 7), columns=slice(2, 7))
    assert report["verdict"] == "pass"
    assert report["stamps"] == 3
    assert report["stacked_channels"]["bit_identical"] is True


def test_a_wrong_stamp_alignment_is_caught(tmp_path):
    import pandas as pd
    import xarray as xr

    times = pd.date_range("2018-01-01T00:00", periods=4, freq="6h")
    reference = tmp_path / "reference.nc"
    wide = tmp_path / "wide.nc"
    _dataset(reference, 5, 5, times=times, row_offset=2.0, column_offset=2.0, seed=2)
    _dataset(wide, 9, 9, times=times[1:], seed=1)
    with xr.open_dataset(wide) as source, xr.open_dataset(reference) as target:
        dataset = source.load()
        for name in PRESSURE + SURFACE:
            values = dataset[name].values
            values[..., 2:7, 2:7] = target[name].values[:-1]  # off by one stamp
            dataset[name].values[...] = values
    dataset.to_netcdf(wide, engine="h5netcdf")
    report = CHECK.compare_central_block(wide, reference, rows=slice(2, 7), columns=slice(2, 7))
    assert report["verdict"] == "fail"
    assert report["stacked_channels"]["bit_identical"] is False
