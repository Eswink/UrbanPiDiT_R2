"""Offline tests for the Earthmover spatial-namespace D1 read path (#63).

The fixtures shrink the global axes but keep the exact guard semantics; the
real frozen chunk geometry is pinned separately against the measured layout in
docs/R7_D1_READ_SPEED.md. No network access happens in this module.
"""
from __future__ import annotations

import contextlib
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import data.download.earthmover_spatial_d1 as module
from data.download.earthmover_pilot import DecodedBudget
from data.download.earthmover_spatial_d1 import (
    PRESSURE_DIMS,
    SPATIAL_PRESSURE_CHUNKS,
    SPATIAL_SURFACE_CHUNKS,
    SURFACE_DIMS,
    d1_fields,
    extract_d1,
    preflight,
    read_stamp,
    stamp_plan,
    validate_spatial_namespace,
    _requested_times,
)

SNAPSHOT = "ZFKDHBCTBVHVXM3BQFV0"


def test_frozen_spatial_chunk_geometry_is_pinned():
    # Measured chunk shapes from docs/R7_D1_READ_SPEED.md; a silent upstream
    # rechunk would invalidate the cost model, so the real constants are pinned
    # here independently of the shrunk test fixtures.
    assert SPATIAL_SURFACE_CHUNKS == (1, 721, 1440)
    assert SPATIAL_PRESSURE_CHUNKS == (1, 1, 721, 1440)


def test_requested_times_are_the_frozen_d1_window():
    stamps, request = _requested_times()
    assert len(stamps) == 120 and request["steps"] == 120
    assert stamps[0] == pd.Timestamp("2016-01-01T00:00")
    assert stamps[-1] == pd.Timestamp("2016-01-30T18:00")
    assert sorted({stamp.hour for stamp in stamps}) == [0, 6, 12, 18]


def test_d1_fields_follow_the_frozen_plan():
    fields, shared = d1_fields()
    by_variable = {field["variable"]: field for field in fields}
    assert shared == [250, 500, 850]
    assert by_variable["2m_temperature"]["group"] == "single"
    assert by_variable["2m_temperature"]["short_name"] == "t2m"
    assert by_variable["mean_sea_level_pressure"]["short_name"] == "msl"
    for name, short in (("geopotential", "z"), ("temperature", "t"), ("specific_humidity", "q"),
                        ("u_component_of_wind", "u"), ("v_component_of_wind", "v")):
        assert by_variable[name]["group"] == "pressure"
        assert by_variable[name]["short_name"] == short
        assert by_variable[name]["levels"] == [250, 500, 850]
    channels = [name for field in fields for name in field["channel_names"]]
    assert len(channels) == 17 and len(set(channels)) == 17
    reads = sum(1 if field["group"] == "single" else len(field["levels"]) for field in fields)
    assert reads == 19  # 4 surface + 5 pressure x 3 shared levels


class SpatialArray:
    """Minimal zarr-like array with basic slicing and guard-relevant metadata."""

    def __init__(self, x, chunks, dims=(), attrs=None):
        self.x, self.shape, self.dtype, self.chunks = x, x.shape, x.dtype, chunks
        self.shards = None
        self.calls = []
        self.metadata = type("Meta", (), {"dimension_names": dims})()
        self.attrs = attrs or {}

    def __getitem__(self, item):
        self.calls.append(item)
        return self.x[item]


def spatial_root(stamps=6):
    """A shrunk spatial-namespace tree with the same guard-relevant semantics."""
    lat = np.arange(43., 26.99, -0.25)
    lon = np.arange(107., 123.01, 0.25)
    levels = np.array([50., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
    times = pd.date_range("2016-01-01", periods=stamps, freq="6h")
    hours = np.asarray((times - pd.Timestamp("1940-01-01")) / pd.Timedelta(hours=1), dtype="i8")
    time_attrs = {"units": "hours since 1940-01-01 00:00:00", "calendar": "proleptic_gregorian"}
    root = {}
    for group in ("single", "pressure"):
        root[f"{group}/spatial"] = {
            "latitude": SpatialArray(lat.copy(), (len(lat),), ("latitude",)),
            "longitude": SpatialArray(lon.copy(), (len(lon),), ("longitude",)),
            "valid_time": SpatialArray(hours.copy(), (stamps,), ("valid_time",), dict(time_attrs)),
        }
    root["pressure/spatial"]["pressure_level"] = SpatialArray(
        levels.copy(), (len(levels),), ("pressure_level",), {"units": "hPa"})
    dims = {"single": SURFACE_DIMS, "pressure": PRESSURE_DIMS}
    chunks = {"single": module.SPATIAL_SURFACE_CHUNKS, "pressure": module.SPATIAL_PRESSURE_CHUNKS}
    units = {"t2m": "K", "u10": "m s**-1", "v10": "m s**-1", "msl": "Pa",
             "z": "m**2 s**-2", "t": "K", "q": "kg kg**-1", "u": "m s**-1", "v": "m s**-1"}
    for field in d1_fields()[0]:
        shape = (stamps, len(levels), len(lat), len(lon)) if field["group"] == "pressure" \
            else (stamps, len(lat), len(lon))
        values = np.arange(np.prod(shape), dtype="f4").reshape(shape)
        root[f"{field['group']}/spatial"][field["short_name"]] = SpatialArray(
            values, chunks[field["group"]], dims[field["group"]], {"units": units[field["short_name"]]})
    return root, times


@pytest.fixture
def shrunk_geometry(monkeypatch):
    """Shrink the pinned chunk geometry to the fixture's 65-point axes."""
    monkeypatch.setattr(module, "SPATIAL_SURFACE_CHUNKS", (1, 65, 65))
    monkeypatch.setattr(module, "SPATIAL_PRESSURE_CHUNKS", (1, 1, 65, 65))


@pytest.fixture
def frozen_request(monkeypatch):
    """Replace the real 120-stamp D1 request with the fixture's stamps."""
    times = pd.date_range("2016-01-01", periods=6, freq="6h")
    request = {"days": 1, "start": "2016-01-01T00:00", "steps": 6,
               "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
               "init_times": 6, "season": "Jan"}
    monkeypatch.setattr(module, "_requested_times", lambda: (list(times), request))
    return times


class FakeSession:
    def __init__(self, root):
        self.store = type("Store", (), {"_root": root})()
        self.snapshot_id = SNAPSHOT


class FakeIc:
    __version__ = "0.0-test"


class _FakeConfig:
    @staticmethod
    def set(settings):
        return contextlib.nullcontext()


class FakeZarr:
    __version__ = "0.0-test"
    config = _FakeConfig()

    @staticmethod
    def open_group(store, mode):
        return store._root


def test_validate_plan_and_read_stamp_crop_exactly(shrunk_geometry):
    root, times = spatial_root(stamps=4)
    budget = DecodedBudget(limit=2 * 2**30)
    plan = validate_spatial_namespace(root, list(times[:4]), budget)
    assert len(plan["stamps"]) == 4
    assert plan["per_stamp_field_reads"] == 19
    assert plan["level_attestation"]["conversion_applied"] is False
    frames = read_stamp(plan, 1, budget)
    levels = np.asarray(root["pressure/spatial"]["pressure_level"].x)
    z = root["pressure/spatial"]["z"]
    # pressure frames carry the stored level axis [250, 500, 850] on axis 0
    for stored_row, level in enumerate(plan["shared_levels"]):
        source_row = int(np.flatnonzero(levels == float(level))[0])
        np.testing.assert_array_equal(frames["geopotential"][stored_row], z.x[1, source_row])
    np.testing.assert_array_equal(frames["2m_temperature"],
                                  root["single/spatial"]["t2m"].x[1])
    surface_calls = len(root["single/spatial"]["t2m"].calls)
    pressure_calls = len(root["pressure/spatial"]["z"].calls)
    assert surface_calls == 1 and pressure_calls == 3  # one chunk per read; 3 shared levels
    assert budget.reads == 2 * 3 + 1 + 19  # coords per group + level axis + one stamp of fields


def test_stamp_plan_refuses_missing_timestamps():
    _, times = spatial_root(stamps=4)
    with pytest.raises(ValueError, match="absent"):
        stamp_plan(times, [pd.Timestamp("2016-01-03T03:00")])


@pytest.mark.parametrize("failure", ["units", "chunks", "coords", "dims", "packed"])
def test_metadata_guards_reject_before_weather_reads(shrunk_geometry, failure):
    root, times = spatial_root(stamps=2)
    if failure == "units":
        root["single/spatial"]["t2m"].attrs["units"] = "degC"
    elif failure == "chunks":
        root["single/spatial"]["t2m"].chunks = (2, 65, 65)
    elif failure == "coords":
        root["single/spatial"]["latitude"].x[0] = 43.5
    elif failure == "dims":
        root["single/spatial"]["t2m"].metadata.dimension_names = ("valid_time", "lat", "longitude")
    else:
        root["pressure/spatial"]["t"].attrs["scale_factor"] = 0.5
    budget = DecodedBudget(limit=2 * 2**30)
    with pytest.raises(ValueError):
        validate_spatial_namespace(root, list(times[:2]), budget)
    assert not root["single/spatial"]["t2m"].calls
    assert not root["pressure/spatial"]["z"].calls


def test_budget_rejects_before_any_field_read(shrunk_geometry, frozen_request, tmp_path, monkeypatch):
    root, _ = spatial_root(stamps=6)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    monkeypatch.setattr(module, "_net_recv_bytes", lambda: 10**9)
    # Coordinates (~1 KiB) fit; the ~1.9 MiB field estimate for six stamps does
    # not, so the budget must fire before any weather chunk is touched.
    with pytest.raises(RuntimeError, match="BEFORE"):
        extract_d1(tmp_path / "source.nc", tmp_path / "receipt.json",
                   decoded_budget_bytes=2**20)
    assert not root["single/spatial"]["t2m"].calls
    stored = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed-no-fallback"
    assert stored["synthetic_fallback"] is False


def test_extract_d1_publishes_artifact_and_honest_receipt(shrunk_geometry, frozen_request,
                                                          tmp_path, monkeypatch):
    root, times = spatial_root(stamps=6)
    readings = iter([10**9, 10**9 + 500 * 2**20])
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    monkeypatch.setattr(module, "_net_recv_bytes", lambda: next(readings))
    out_nc = tmp_path / "source.nc"
    receipt_path = tmp_path / "receipt.json"
    receipt = extract_d1(out_nc, receipt_path)
    assert receipt["status"] == "downloaded-real-source"
    assert receipt["source_is_real_reanalysis"] is True
    assert receipt["synthetic_fallback"] is False and receipt["scientific_claim"] is False
    assert len(receipt["timestamps"]) == 6
    assert receipt["init_hour_coverage"] == {"0": 2, "6": 2, "12": 1, "18": 1}
    assert receipt["network_body_bytes"] == 500 * 2**20
    assert receipt["local_artifact"]["sha256"] == hashlib.sha256(out_nc.read_bytes()).hexdigest()
    assert receipt["local_artifact"]["bytes"] == out_nc.stat().st_size
    assert len(receipt["payloads"]) == 17
    assert all(payload["finite"] for payload in receipt["payloads"].values())
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stored["status"] == receipt["status"]
    # the audited converter must stack exactly the 17 planned channels from it
    import xarray as xr
    from data.preprocess.r7_era5 import DEFAULT_R7_ERA5_CHANNELS, stack_era5_channels

    dataset = xr.open_dataset(out_nc, engine="h5netcdf")
    state, names, nc_times, lat, lon = stack_era5_channels(dataset, DEFAULT_R7_ERA5_CHANNELS)
    assert state.shape == (6, 17, 65, 65) and len(names) == 17
    np.testing.assert_array_equal(np.asarray(nc_times), times.values)
    dataset.close()


def test_extract_d1_failure_leaves_failed_no_fallback(shrunk_geometry, frozen_request,
                                                      tmp_path, monkeypatch):
    root, _ = spatial_root(stamps=6)
    readings = iter([10**9, 10**9 + 1024])
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    monkeypatch.setattr(module, "_net_recv_bytes", lambda: next(readings))

    def explode(plan, budget, deadline):
        raise ValueError("source turned nonfinite mid-stream")
    monkeypatch.setattr(module, "_collect_frames", explode)
    out_nc = tmp_path / "source.nc"
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="nonfinite"):
        extract_d1(out_nc, receipt_path)
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stored["status"] == "failed-no-fallback"
    assert stored["synthetic_fallback"] is False
    assert not out_nc.exists()


def test_extract_d1_refuses_existing_outputs(shrunk_geometry, frozen_request, tmp_path, monkeypatch):
    root, _ = spatial_root(stamps=2)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    out_nc = tmp_path / "source.nc"
    out_nc.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        extract_d1(out_nc, tmp_path / "receipt.json")


def test_preflight_writes_nothing_by_default(shrunk_geometry, frozen_request, tmp_path, monkeypatch, capsys):
    root, _ = spatial_root(stamps=6)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    report = preflight()
    assert report["mode"] == "read-only-preflight"
    assert report["scientific_training_certified"] is False
    assert report["stamps_found"] == 6
    assert capsys.readouterr().out
    assert list(tmp_path.glob("*")) == []


def test_preflight_report_path_is_exclusive(shrunk_geometry, frozen_request, tmp_path, monkeypatch):
    root, _ = spatial_root(stamps=6)
    monkeypatch.setattr(module, "_open_pinned_session",
                        lambda: (FakeSession(root), FakeIc(), FakeZarr()))
    target = tmp_path / "report.json"
    preflight(target)
    assert target.is_file()
    with pytest.raises(FileExistsError):
        preflight(target)
