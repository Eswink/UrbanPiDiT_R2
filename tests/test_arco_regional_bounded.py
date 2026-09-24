"""Offline tests for the bounded regional ARCO extractor.

No test here opens the network: the source is reached only through the explicit
optional real-data smoke, which is gated on an already-present local file. These
tests cover the deterministic planning, attestation, budget and failure-record
logic with synthetic xarray fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.download.arco_regional_bounded import (
    CHANNELS,
    LEVEL_VARIABLES,
    PRESSURE_LEVELS_HPA,
    SOURCE_VARIABLES,
    BudgetExceeded,
    RegionalPlan,
    attest_levels,
    build_regional_dataset,
    check_channel_units,
    extract_regional_region,
    plan_timestamps,
    region_budget,
)

UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m s**-1",
    "10m_v_component_of_wind": "m s**-1",
    "mean_sea_level_pressure": "Pa",
    "geopotential": "m**2 s**-2",
    "temperature": "K",
    "specific_humidity": "kg kg**-1",
    "u_component_of_wind": "m s**-1",
    "v_component_of_wind": "m s**-1",
}


def _source_fixture(*, times, lat, lon, level=PRESSURE_LEVELS_HPA, units=None, fill=280.0):
    """Synthetic dataset shaped exactly like the audited ARCO source."""
    units = dict(UNITS) if units is None else dict(units)
    shape3 = (len(times), len(lat), len(lon))
    shape4 = (len(times), len(level), len(lat), len(lon))
    spatial = np.add.outer(
        np.arange(len(lat), dtype=np.float32),
        np.arange(len(lon), dtype=np.float32),
    )
    data_vars = {}
    for variable in SOURCE_VARIABLES:
        if variable in LEVEL_VARIABLES:
            base = np.empty(shape4, dtype=np.float32)
            for k in range(len(level)):
                base[:, k] = fill + k + spatial
        else:
            base = np.empty(shape3, dtype=np.float32)
            for t in range(len(times)):
                base[t] = fill + t + spatial
        data_vars[variable] = (
            ("time", "level", "latitude", "longitude") if variable in LEVEL_VARIABLES
            else ("time", "latitude", "longitude"),
            base,
        )
    dataset = xr.Dataset(
        data_vars,
        coords={
            "time": pd.DatetimeIndex(times),
            "level": np.asarray(level, dtype=np.int32),
            "latitude": np.asarray(lat, dtype=np.float32),
            "longitude": np.asarray(lon, dtype=np.float32),
        },
    )
    for variable, values in dataset.data_vars.items():
        values.attrs["units"] = units[variable]
    return dataset


def _block(dataset, times):
    """Build one decoded block straight from the fixture, bypassing the network."""
    lat = np.asarray(dataset["latitude"].values, dtype=np.float32)
    lon = np.asarray(dataset["longitude"].values, dtype=np.float32)
    index = {t.isoformat(): i for i, t in enumerate(pd.DatetimeIndex(dataset["time"].values))}
    frames = {
        variable: np.stack([np.asarray(dataset[variable].values[index[pd.Timestamp(t).isoformat()]], dtype=np.float32) for t in times])
        for variable in SOURCE_VARIABLES
    }
    return {"frames": frames, "latitude": lat, "longitude": lon, "times": list(times)}


def test_plan_is_deterministic_and_spans_declared_seasons():
    plan = RegionalPlan()
    plan.validate()
    first, blocks_first = plan_timestamps(plan)
    second, blocks_second = plan_timestamps(plan)
    assert first == second and blocks_first == blocks_second
    assert len(blocks_first) == len(plan.years) * len(plan.season_starts)
    assert len(first) == len(blocks_first) * plan.block_steps
    for key, times in blocks_first.items():
        assert len(times) == plan.block_steps
        assert len(set(times)) == len(times)
        year, month, _ = (int(v) for v in key.split("-"))
        assert year in plan.years
        assert month in {m for m, _ in plan.season_starts}
    # Every block is one continuous 6-hourly sequence with no gap.
    for times in blocks_first.values():
        parsed = pd.DatetimeIndex(times)
        assert np.array_equal(np.diff(parsed.values).astype("timedelta64[h]").astype(int), [6] * (len(times) - 1))


def test_plan_rejects_invalid_declarations():
    with pytest.raises(ValueError):
        RegionalPlan(block_steps=2).validate()
    with pytest.raises(ValueError):
        RegionalPlan(years=()).validate()
    with pytest.raises(ValueError):
        RegionalPlan(season_starts=((13, 1),)).validate()
    with pytest.raises(ValueError):
        RegionalPlan(roi=(43.0, 27.0, 107.0, 123.0)).validate()
    with pytest.raises(ValueError):
        RegionalPlan(max_chunk_bytes=0).validate()


def test_level_attestation_accepts_audited_set_and_rejects_others():
    report = attest_levels(PRESSURE_LEVELS_HPA)
    assert report["units"] == "hPa"
    assert report["conversion_applied"] is False
    assert report["levels"] == list(PRESSURE_LEVELS_HPA)
    # A plausible-looking but different axis must not be silently accepted.
    with pytest.raises(ValueError):
        attest_levels((1000, 850, 700, 500, 300, 250, 200, 150, 100, 50, 10, 5, 1))
    with pytest.raises(ValueError):
        attest_levels((850, 500))
    # Reordered is not the audited axis either: this is an identity check.
    with pytest.raises(ValueError):
        attest_levels(tuple(sorted(PRESSURE_LEVELS_HPA, reverse=True)))


def test_channel_units_are_checked_verbatim():
    ds = _source_fixture(
        times=pd.date_range("2018-01-01", periods=2, freq="6h"),
        lat=(41.0, 40.0),
        lon=(115.0, 115.25),
    )
    observed = check_channel_units(ds, CHANNELS)
    assert observed["2m_temperature"] == "K"
    assert observed["geopotential"] == "m**2 s**-2"

    wrong = dict(UNITS, **{"2m_temperature": "degC"})
    celsius = _source_fixture(
        times=pd.date_range("2018-01-01", periods=2, freq="6h"),
        lat=(41.0, 40.0),
        lon=(115.0, 115.25),
        units=wrong,
    )
    with pytest.raises(ValueError, match="no guessed conversion"):
        check_channel_units(celsius, CHANNELS)

    missing = _source_fixture(
        times=pd.date_range("2018-01-01", periods=2, freq="6h"),
        lat=(41.0, 40.0),
        lon=(115.0, 115.25),
    ).drop_vars("geopotential")
    with pytest.raises(KeyError):
        check_channel_units(missing, CHANNELS)


def test_budget_estimate_counts_whole_touched_chunks():
    times = pd.date_range("2018-01-01", periods=4, freq="6h")
    lat = np.arange(41.0, 38.9, -0.25)
    lon = np.arange(115.0, 117.1, 0.25)
    ds = _source_fixture(times=times, lat=lat, lon=lon)
    # ARCO stores one global slab per pressure variable and timestamp, so the
    # estimate must count whole touched chunks rather than the cropped ROI.
    for variable in SOURCE_VARIABLES:
        if variable in LEVEL_VARIABLES:
            ds[variable].encoding["chunks"] = (1, len(PRESSURE_LEVELS_HPA), 721, 1440)
        else:
            ds[variable].encoding["chunks"] = (1, 721, 1440)
    plan = RegionalPlan(years=(2018,), season_starts=((1, 1),), block_steps=4)
    report = region_budget(ds, plan, [t.isoformat() for t in times])
    assert set(report["variables"]) == set(SOURCE_VARIABLES)
    for name, row in report["variables"].items():
        assert row["touched_chunks"] == len(times), name
        assert row["estimated_uncompressed_bytes"] == (
            row["touched_chunks"] * row["uncompressed_bytes_per_full_chunk"]
        ), name
    assert report["estimated_uncompressed_bytes"] == sum(
        row["estimated_uncompressed_bytes"] for row in report["variables"].values()
    )
    # Levels widen the touched slab, not the number of touched chunks.
    slab = report["variables"]["temperature"]["uncompressed_bytes_per_full_chunk"]
    surface = report["variables"]["2m_temperature"]["uncompressed_bytes_per_full_chunk"]
    assert slab > surface


def _write_local_source(dataset, path: Path) -> Path:
    """Persist a fixture as a local zarr so the full flow can run offline."""
    dataset.to_zarr(path, mode="w", zarr_format=2)
    return path


def _full_flow_plan(**overrides):
    base = dict(years=(2018,), season_starts=((1, 1),), block_steps=3, deadline_seconds=120.0)
    base.update(overrides)
    return RegionalPlan(**base)


def _plan_fixture(plan):
    """Fixture holding exactly the timestamps the plan asks for."""
    times = pd.DatetimeIndex([t for block in plan_timestamps(plan)[1].values() for t in block])
    lat = np.arange(41.0, 38.9, -0.25, dtype=np.float32)
    lon = np.arange(115.0, 117.1, 0.25, dtype=np.float32)
    return _source_fixture(times=times, lat=lat, lon=lon)


def test_full_flow_writes_contract_dataset_and_honest_receipt(tmp_path: Path):
    plan = _full_flow_plan()
    source = _write_local_source(_plan_fixture(plan), tmp_path / "source.zarr")
    out = tmp_path / "subset.nc"
    receipt = tmp_path / "receipt.json"
    report = extract_regional_region(out, receipt, plan=plan, source=str(source))

    assert out.exists() and report["status"] == "downloaded-real-source"
    assert report["synthetic_fallback"] is False
    assert report["scientific_claim"] is False
    assert report["interpolation"] is False
    assert report["protocol"]["selection"].startswith("exact 6-hourly")
    assert report["level_attestation"]["units"] == "hPa"
    assert report["level_attestation"]["conversion_applied"] is False
    assert len(report["timestamps"]) == plan.block_steps * len(plan.years) * len(plan.season_starts)
    assert report["local_artifact"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    on_disk = json.loads(receipt.read_text(encoding="utf-8"))
    assert on_disk == report

    built = xr.open_dataset(out, engine="h5netcdf")
    try:
        assert set(built.data_vars) == set(SOURCE_VARIABLES)
        assert np.array_equal(
            np.asarray(built["level"].values), np.asarray(PRESSURE_LEVELS_HPA, dtype=np.int32)
        )
        assert built["level"].attrs["units"] == "hPa"
        assert built.attrs["native_grid_spacing_deg"] == 0.25
        assert built.attrs["spatial_resampling"] == "none"
        # Payload hashes must describe the values actually stored.
        for name, payload in report["payloads"].items():
            assert payload["units"] == UNITS[payload["source_variable"]]
            assert np.isfinite([payload["min"], payload["max"]]).all()
            assert payload["min"] <= payload["max"]
    finally:
        built.close()


def test_full_flow_refuses_to_exceed_the_declared_chunk_cap(tmp_path: Path):
    plan = _full_flow_plan(max_chunk_bytes=1)
    source = _write_local_source(_plan_fixture(plan), tmp_path / "source.zarr")
    out = tmp_path / "subset.nc"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(BudgetExceeded):
        extract_regional_region(out, receipt, plan=plan, source=str(source))
    assert not out.exists()
    record = json.loads(receipt.read_text(encoding="utf-8"))
    assert record["status"] == "failed-no-fallback"
    assert record["synthetic_fallback"] is False


def test_assembled_dataset_keeps_source_schema_and_full_pressure_axis():
    times = pd.date_range("2018-01-01", periods=4, freq="6h")
    lat = np.arange(41.0, 38.9, -0.25, dtype=np.float32)
    lon = np.arange(115.0, 117.1, 0.25, dtype=np.float32)
    ds = _source_fixture(times=times, lat=lat, lon=lon)
    plan = RegionalPlan(years=(2018,), season_starts=((1, 1),), block_steps=4)
    block_times = [t.isoformat() for t in times]
    built = build_regional_dataset(plan, [_block(ds, block_times)])

    assert set(built.data_vars) == set(SOURCE_VARIABLES)
    assert built.sizes["time"] == len(times)
    assert built.sizes["latitude"] == lat.size and built.sizes["longitude"] == lon.size
    assert np.array_equal(np.asarray(built["level"].values), np.asarray(PRESSURE_LEVELS_HPA, dtype=np.int32))
    assert built["level"].attrs["units"] == "hPa"
    for variable in SOURCE_VARIABLES:
        assert built[variable].attrs["units"] == UNITS[variable]
    # Values must survive the assembly unmixed between levels.
    plane = np.asarray(built["temperature"].values)
    assert plane.ndim == 4
    delta = plane[:, 1] - plane[:, 0]
    assert np.allclose(delta, 1.0)


def test_assembly_concatenates_blocks_in_order_without_duplicates():
    times = pd.date_range("2018-01-01", periods=6, freq="6h")
    lat = np.arange(41.0, 40.9, -0.25, dtype=np.float32)
    lon = np.arange(115.0, 115.9, 0.25, dtype=np.float32)
    ds = _source_fixture(times=times, lat=lat, lon=lon)
    plan = RegionalPlan(years=(2018,), season_starts=((1, 1),), block_steps=3)
    first = [t.isoformat() for t in times[:3]]
    second = [t.isoformat() for t in times[3:]]
    built = build_regional_dataset(plan, [_block(ds, first), _block(ds, second)])
    assert built.sizes["time"] == 6
    assert np.array_equal(
        pd.DatetimeIndex(built["time"].values).values,
        pd.DatetimeIndex(times).values,
    )
    with pytest.raises(ValueError, match="duplicate timestamps"):
        build_regional_dataset(plan, [_block(ds, first), _block(ds, first)])


def test_assembly_rejects_changed_coordinates_between_blocks():
    times = pd.date_range("2018-01-01", periods=6, freq="6h")
    lat = np.arange(41.0, 40.9, -0.25, dtype=np.float32)
    lon = np.arange(115.0, 115.9, 0.25, dtype=np.float32)
    ds = _source_fixture(times=times, lat=lat, lon=lon)
    other = _source_fixture(times=times, lat=lat + 1.0, lon=lon)
    plan = RegionalPlan(years=(2018,), season_starts=((1, 1),), block_steps=3)
    first = [t.isoformat() for t in times[:3]]
    second = [t.isoformat() for t in times[3:]]
    with pytest.raises(ValueError, match="ROI coordinates changed"):
        build_regional_dataset(plan, [_block(ds, first), _block(other, second)])


def test_failure_records_no_fallback_and_never_writes_a_field(tmp_path: Path):
    """An unreachable source must leave an audit record and no synthetic data."""
    out = tmp_path / "subset.nc"
    receipt = tmp_path / "receipt.json"
    plan = RegionalPlan(years=(2018,), season_starts=((1, 1),), block_steps=3,
                        deadline_seconds=30.0, max_chunk_bytes=2**30)
    with pytest.raises(Exception):
        extract_regional_region(out, receipt, plan=plan, source="/nonexistent/source.zarr")
    assert not out.exists()
    record = json.loads(receipt.read_text(encoding="utf-8"))
    assert record["status"] == "failed-no-fallback"
    assert record["synthetic_fallback"] is False
    assert record["scientific_claim"] is False
    assert record["source"] == "/nonexistent/source.zarr"


def test_existing_output_is_refused_without_touching_it(tmp_path: Path):
    out = tmp_path / "subset.nc"
    out.write_bytes(b"keep me")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(FileExistsError):
        extract_regional_region(out, receipt, plan=RegionalPlan(), source="/nonexistent/source.zarr")
    assert out.read_bytes() == b"keep me"
    assert not receipt.exists()


# --- Optional real-data smoke (Acceptance item 3) -------------------------
# Runs only when a real subset fetched by this module is already on disk. The
# checkout does not track it and nothing here downloads: a clean clone and CI
# simply skip. No synthetic stand-in is ever substituted for the real file.
REAL_SUBSET = Path("outputs/r7_regional_real/source.nc")
REAL_RECEIPT = Path("outputs/r7_regional_real/source_receipt.json")

requires_real_subset = pytest.mark.skipif(
    not (REAL_SUBSET.exists() and REAL_RECEIPT.exists()),
    reason=(
        "optional real ARCO subset is not tracked in a clean checkout; "
        "no synthetic fallback is permitted"
    ),
)


@requires_real_subset
def test_real_subset_matches_its_receipt_and_stays_physical():
    """The pinned real subset must be self-consistent and physically plausible."""
    receipt = json.loads(REAL_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "downloaded-real-source"
    assert receipt["synthetic_fallback"] is False
    assert receipt["scientific_claim"] is False
    assert receipt["interpolation"] is False
    assert receipt["level_attestation"]["units"] == "hPa"
    assert receipt["level_attestation"]["conversion_applied"] is False
    assert receipt["local_artifact"]["sha256"] == hashlib.sha256(REAL_SUBSET.read_bytes()).hexdigest()
    assert receipt["local_artifact"]["bytes"] == REAL_SUBSET.stat().st_size

    with xr.open_dataset(REAL_SUBSET, engine="h5netcdf") as ds:
        assert ds.attrs["native_grid_spacing_deg"] == 0.25
        assert ds.attrs["spatial_resampling"] == "none"
        assert np.array_equal(
            np.asarray(ds["level"].values), np.asarray(PRESSURE_LEVELS_HPA, dtype=np.int32)
        )
        assert ds["level"].attrs["units"] == "hPa"
        times = pd.DatetimeIndex(ds["time"].values)
        assert len(times) == len(receipt["timestamps"])
        assert times.is_monotonic_increasing and times.is_unique
        assert sorted(set(times.year.tolist())) == sorted(
            int(y) for y in receipt["protocol"]["years"]
        )
        # Every planned block must survive as a continuous 6-hourly sequence.
        for block in receipt["blocks"].values():
            parsed = pd.DatetimeIndex(block)
            assert parsed.isin(times).all()
            assert np.array_equal(
                np.diff(parsed.values).astype("timedelta64[h]").astype(int),
                [6] * (len(block) - 1),
            )
        # Physical sanity: the June-April four-season range for East Asia. These
        # bounds are deliberately loose checks for gross unit/misread errors,
        # not a calibration claim.
        t2m = np.asarray(ds["2m_temperature"].values)
        t850 = np.asarray(ds["temperature"].sel(level=850).values)
        t500 = np.asarray(ds["temperature"].sel(level=500).values)
        for name, values, low, high in (
            ("t2m", t2m, 200.0, 335.0),
            ("t850", t850, 200.0, 320.0),
            ("t500", t500, 200.0, 300.0),
        ):
            assert np.isfinite(values).all(), name
            assert low <= float(values.min()) and float(values.max()) <= high, name
        # Lower troposphere must be warmer than the mid-troposphere everywhere.
        assert np.all(t850 > t500)
        q = np.asarray(ds["specific_humidity"].sel(level=850).values)
        assert float(q.min()) >= 0.0 and float(q.max()) < 0.05
        mslp = np.asarray(ds["mean_sea_level_pressure"].values)
        assert 85_000.0 <= float(mslp.min()) and float(mslp.max()) <= 110_000.0
        # The exact channels the R7 process diagnostics require are present.
        for variable in SOURCE_VARIABLES:
            assert variable in ds.data_vars
