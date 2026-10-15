"""Offline tests for the full-year (FY) extraction bookkeeping.

No network access: the plan arithmetic, the merge re-validation and the failure
receipt are exercised on synthetic NetCDF parts with the D1 read helpers stood
in. The guard semantics being pinned are the ones that decide whether a real
batch may publish:

- a month window is exactly ``days_in_month x 4`` contiguous 6-hourly stamps on
  the four UTC init hours, and a full year is one unbroken 6-hourly run;
- leap years are handled by the calendar, not by a frozen 30-day constant;
- a merge refuses any part whose bytes do not match its own receipt hash, any
  part that is not a completed real download, an overlapping union, a union that
  differs from the recorded stamps, and (when required) a non-contiguous axis;
- a failed part leaves a ``failed-no-fallback`` receipt with
  ``synthetic_fallback: false`` and no output artifact;
- existing outputs are refused rather than truncated.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from data.download import earthmover_spatial_fy as fy
from data.download.season_plan_fy import full_year_plan, planned_stamps

SNAPSHOT = "ZFKDHBCTBVHVXM3BQFV0"


def _plan(years=(2017,), months=(1,)):
    return full_year_plan(list(years), months=months)


def test_a_month_window_is_contiguous_six_hourly_on_the_four_init_hours():
    plan = _plan(months=(1,))
    assert plan["total_stamps"] == 31 * 4 == 124
    stamps = [pd.Timestamp(value) for value in plan["blocks"][0]["stamps"]]
    assert stamps[0] == pd.Timestamp("2017-01-01T00:00")
    assert stamps[-1] == pd.Timestamp("2017-01-31T18:00")
    assert {(right - left) for left, right in zip(stamps, stamps[1:])} == {pd.Timedelta(hours=6)}
    assert sorted({stamp.hour for stamp in stamps}) == [0, 6, 12, 18]


def test_leap_years_are_taken_from_the_calendar_not_a_frozen_length():
    feb_2017 = _plan(years=(2017,), months=(2,))
    feb_2020 = _plan(years=(2020,), months=(2,))
    assert feb_2017["total_stamps"] == 28 * 4 == 112
    assert feb_2020["total_stamps"] == 29 * 4 == 116


def test_a_full_year_is_one_unbroken_six_hourly_run():
    for year, expected in ((2017, 365 * 4), (2020, 366 * 4)):
        stamps = planned_stamps(full_year_plan([year]))
        assert len(stamps) == expected
        assert stamps.equals(pd.date_range(f"{year}-01-01T00:00", f"{year}-12-31T18:00", freq="6h"))
    assert full_year_plan([2017])["parts"] == 12


def test_plan_refuses_out_of_range_year_oversized_batch_and_bad_months():
    with pytest.raises(ValueError):
        full_year_plan([1939])
    with pytest.raises(ValueError):
        full_year_plan(list(range(2000, 2009)))
    with pytest.raises(ValueError):
        full_year_plan([2017, 2017])
    with pytest.raises(ValueError):
        full_year_plan([2017], months=())
    with pytest.raises(ValueError):
        full_year_plan([2017], months=(1, 13))
    with pytest.raises(ValueError):
        full_year_plan([2017], months=(1, 1))


def test_plan_report_estimates_are_positive_and_scale_with_stamps():
    one = fy.plan_report([2017], months=(1,))
    year = fy.plan_report([2017])
    assert one["plan"]["total_stamps"] == 124 and year["plan"]["total_stamps"] == 1460
    assert 0 < one["estimated_network_bytes"] < year["estimated_network_bytes"]
    assert 0 < one["estimated_decoded_bytes"] < year["estimated_decoded_bytes"]
    assert year["year_stamp_counts"] == {2017: 1460}


_SURFACE = {"2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure"}
_PRESSURE = ("geopotential", "temperature", "specific_humidity",
             "u_component_of_wind", "v_component_of_wind")
_LEVELS = (250, 500, 850)


def _part_dataset(stamps):
    """All 17 frozen channels, so the merge exercises the real channel contract."""
    import xarray as xr

    times = list(stamps)
    latitude = np.linspace(43.0, 42.25, 4, dtype=np.float32)
    longitude = np.linspace(107.0, 107.75, 4, dtype=np.float32)
    data_vars = {}
    for index, name in enumerate(sorted(_SURFACE)):
        values = np.full((len(times), 4, 4), 250.0 + index, dtype=np.float32)
        data_vars[name] = (("time", "latitude", "longitude"), values)
    for index, name in enumerate(_PRESSURE):
        values = np.full((len(times), len(_LEVELS), 4, 4), 250.0 + index, dtype=np.float32)
        data_vars[name] = (("time", "level", "latitude", "longitude"), values)
    dataset = xr.Dataset(data_vars, coords={
        "time": times, "level": np.asarray(_LEVELS, dtype=np.int32),
        "latitude": latitude, "longitude": longitude})
    dataset["level"].attrs = {"units": "hPa"}
    dataset.attrs.update({"source_snapshot_id": SNAPSHOT})
    return dataset


def _write_part(tmp_path, name, stamps, *, status="downloaded-real-source", shift_hash=0):
    nc = tmp_path / f"part_{name}.nc"
    dataset = _part_dataset(stamps)
    dataset.to_netcdf(nc, engine="h5netcdf")
    digest = hashlib.sha256(nc.read_bytes()).hexdigest()
    if shift_hash:
        digest = f"{int(digest, 16) ^ shift_hash:064x}"
    receipt = {
        "status": status,
        "part": {"years": [2017], "months": [name],
                 "first_time": str(stamps[0]), "last_time": str(stamps[-1]),
                 "stamps": len(stamps),
                 "blocks": [{"month": name, "year": 2017,
                             "first_time": str(stamps[0]), "last_time": str(stamps[-1]),
                             "stamps": [str(stamp) for stamp in stamps]}]},
        "local_artifact": {"path": nc.name, "bytes": nc.stat().st_size, "sha256": digest},
        "timestamps": [str(stamp) for stamp in stamps],
        "network_body_bytes": 123,
        "elapsed_seconds": 4.5,
    }
    rc = tmp_path / f"part_{name}_receipt.json"
    rc.write_text(json.dumps(receipt), encoding="utf-8")
    return nc, rc


def _month(year, month):
    days = pd.Period(f"{year}-{month:02d}").days_in_month
    return list(pd.date_range(f"{year}-{month:02d}-01", periods=days * 4, freq="6h"))


def test_merge_accepts_verified_parts_and_records_the_union(tmp_path):
    january, february = _month(2017, 1), _month(2017, 2)
    parts = [_write_part(tmp_path, 1, january), _write_part(tmp_path, 2, february)]
    out_nc, rc = tmp_path / "source.nc", tmp_path / "merge.json"
    result = fy.merge_fy_parts(parts, out_nc, rc, expected_stamps=len(january) + len(february))
    assert result["status"] == "merged-real-source" and result["n_parts"] == 2
    assert result["stamps"] == 124 + 112
    assert result["synthetic_fallback"] is False and result["scientific_claim"] is False
    assert result["require_contiguous"] is False
    assert out_nc.is_file() and rc.is_file()


def test_merge_asserts_contiguity_only_when_asked(tmp_path):
    january, february = _month(2017, 1), _month(2017, 2)
    parts = [_write_part(tmp_path, 1, january), _write_part(tmp_path, 2, february)]
    total = len(january) + len(february)
    result = fy.merge_fy_parts(parts, tmp_path / "ok.nc", tmp_path / "ok.json",
                               expected_stamps=total, require_contiguous=True)
    assert result["require_contiguous"] is True
    gap = [_write_part(tmp_path, 1, january), _write_part(tmp_path, 3, _month(2017, 3))]
    with pytest.raises(ValueError, match="not one contiguous"):
        fy.merge_fy_parts(gap, tmp_path / "gap.nc", tmp_path / "gap.json",
                          expected_stamps=len(january) + len(_month(2017, 3)),
                          require_contiguous=True)
    assert not (tmp_path / "gap.nc").exists()


def test_merge_refuses_a_part_whose_bytes_do_not_match_its_receipt(tmp_path):
    parts = [_write_part(tmp_path, 1, _month(2017, 1), shift_hash=1)]
    with pytest.raises(ValueError, match="does not match its receipt hash"):
        fy.merge_fy_parts(parts, tmp_path / "out.nc", tmp_path / "out.json", expected_stamps=124)
    assert not (tmp_path / "out.nc").exists()


def test_merge_refuses_an_incomplete_part_status(tmp_path):
    parts = [_write_part(tmp_path, 1, _month(2017, 1), status="failed-no-fallback")]
    with pytest.raises(ValueError, match="not a completed download"):
        fy.merge_fy_parts(parts, tmp_path / "out.nc", tmp_path / "out.json", expected_stamps=124)
    assert not (tmp_path / "out.nc").exists()


def test_merge_refuses_overlapping_parts_and_a_wrong_expected_total(tmp_path):
    january = _month(2017, 1)
    parts = [_write_part(tmp_path, 1, january), _write_part(tmp_path, 2, january)]
    with pytest.raises(ValueError, match="overlap"):
        fy.merge_fy_parts(parts, tmp_path / "out.nc", tmp_path / "out.json",
                          expected_stamps=2 * len(january))
    with pytest.raises(ValueError, match="plan"):
        fy.merge_fy_parts([_write_part(tmp_path, 3, _month(2017, 3))], tmp_path / "out2.nc",
                          tmp_path / "out2.json", expected_stamps=99)


def test_merge_refuses_existing_outputs_without_touching_them(tmp_path):
    part = _write_part(tmp_path, 1, _month(2017, 1))
    marker = tmp_path / "merge.json"
    marker.write_text("keep-me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fy.merge_fy_parts([part], tmp_path / "out.nc", marker, expected_stamps=124)
    assert marker.read_text(encoding="utf-8") == "keep-me"


def test_a_missing_part_artifact_is_refused(tmp_path):
    _, rc = _write_part(tmp_path, 1, _month(2017, 1))
    with pytest.raises(FileNotFoundError):
        fy.merge_fy_parts([(tmp_path / "absent.nc", rc)], tmp_path / "out.nc",
                          tmp_path / "out.json", expected_stamps=124)


def test_extract_refuses_existing_output_and_does_not_launch_a_read(tmp_path):
    out_nc = tmp_path / "source.nc"
    out_nc.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        fy.extract_fy_part(out_nc, tmp_path / "receipt.json", years=[2017], months=(1,))
    assert out_nc.read_bytes() == b"occupied"


def test_a_failed_read_leaves_failed_no_fallback_and_no_artifact(tmp_path, monkeypatch):
    def explode():
        raise RuntimeError("pinned session unavailable")

    monkeypatch.setattr(fy, "_open_pinned_session", explode)
    out_nc, rc = tmp_path / "source.nc", tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="pinned session unavailable"):
        fy.extract_fy_part(out_nc, rc, years=[2017], months=(1,))
    assert not out_nc.exists()
    receipt = json.loads(rc.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed-no-fallback"
    assert receipt["synthetic_fallback"] is False and receipt["scientific_claim"] is False
    assert receipt["segment"] == "fy-full-year-multi-year"


# --- batch driver guards (all fire before any byte is fetched) -----------------

_DRIVER = "scripts/r7_fy_batch_download.py"


def _protocol(**overrides):
    """A valid 1-month protocol, mutated per test."""
    payload = {
        "format": "r7-s3-fy-batch-protocol-v1",
        "plan": {"years": [2017], "months": [1], "total_stamps": 124, "parts": 1},
        "concurrency": {"pool": 1, "max_host_mb_s": 60.0},
        "budgets": {"per_part_deadline_seconds": 3600.0, "per_part_watchdog_seconds": 3660.0,
                    "per_part_decoded_gib_cap": 16.0, "free_space_gate_gib": 0.001},
        "scientific_claim": False,
    }
    payload.update(overrides)
    return payload


def _write_protocol(tmp_path, payload):
    body = {k: v for k, v in payload.items() if k != "protocol_sha256"}
    payload = dict(payload)
    payload["protocol_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_driver(protocol, out_dir):
    return subprocess.run([sys.executable, _DRIVER, "--protocol", str(protocol),
                           "--out-dir", str(out_dir)],
                          cwd=".", capture_output=True, text=True, timeout=120)


def test_batch_driver_refuses_a_missing_protocol(tmp_path):
    done = _run_driver(tmp_path / "absent.json", tmp_path / "batch")
    assert done.returncode != 0 and "missing protocol" in done.stderr


def test_batch_driver_refuses_a_tampered_protocol_digest(tmp_path):
    path = _write_protocol(tmp_path, _protocol())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plan"]["total_stamps"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    done = _run_driver(path, tmp_path / "batch")
    assert done.returncode != 0 and "digest mismatch" in done.stderr


def test_batch_driver_refuses_a_plan_that_disagrees_with_the_code(tmp_path):
    path = _write_protocol(tmp_path, _protocol(
        plan={"years": [2017], "months": [1], "total_stamps": 999, "parts": 1}))
    done = _run_driver(path, tmp_path / "batch")
    assert done.returncode != 0 and "stamps" in done.stderr


def test_batch_driver_refuses_a_nonempty_output_directory(tmp_path):
    path = _write_protocol(tmp_path, _protocol())
    out_dir = tmp_path / "batch"
    out_dir.mkdir()
    (out_dir / "keep-me").write_text("occupied", encoding="utf-8")
    done = _run_driver(path, out_dir)
    assert done.returncode != 0 and "refusing non-empty" in done.stderr
    assert (out_dir / "keep-me").read_text(encoding="utf-8") == "occupied"


def _batch_dir(tmp_path, name, stamps, *, tamper=False):
    """A minimal completed batch: a source.nc plus its BUILD_COMPLETE.json."""
    import xarray as xr

    directory = tmp_path / name
    directory.mkdir()
    values = np.arange(len(stamps) * 2 * 2, dtype=np.float32).reshape(len(stamps), 2, 2)
    dataset = xr.Dataset({"2m_temperature": (("time", "latitude", "longitude"), values)},
                         coords={"time": list(stamps),
                                 "latitude": np.array([43.0, 42.75], dtype=np.float32),
                                 "longitude": np.array([107.0, 107.25], dtype=np.float32)})
    dataset.attrs["source_snapshot_id"] = SNAPSHOT
    source = directory / "source.nc"
    dataset.to_netcdf(source, engine="h5netcdf")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if tamper:
        digest = f"{int(digest, 16) ^ 1:064x}"
    (directory / "BUILD_COMPLETE.json").write_text(json.dumps({
        "source_nc_sha256": digest, "source_nc_bytes": source.stat().st_size,
        "plan": {"years": sorted({s.year for s in stamps})}}), encoding="utf-8")
    return directory


def test_combine_concatenates_verified_batches_into_one_contiguous_axis(tmp_path):
    first = list(pd.date_range("2017-01-01", periods=4, freq="6h"))
    second = list(pd.date_range("2017-01-02", periods=4, freq="6h"))
    done = subprocess.run([sys.executable, _DRIVER, "--combine",
                           "--batches", str(_batch_dir(tmp_path, "a", first)),
                           str(_batch_dir(tmp_path, "b", second)),
                           "--combined-nc", str(tmp_path / "all.nc"),
                           "--combined-receipt", str(tmp_path / "all.json")],
                          cwd=".", capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    receipt = json.loads((tmp_path / "all.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "combined-real-source"
    assert receipt["n_batches"] == 2 and receipt["stamps"] == 8
    assert receipt["require_contiguous"] is True
    assert (tmp_path / "all.nc").is_file()


def test_combine_refuses_a_batch_that_is_not_complete(tmp_path):
    directory = tmp_path / "a"
    directory.mkdir()
    done = subprocess.run([sys.executable, _DRIVER, "--combine", "--batches", str(directory),
                           "--combined-nc", str(tmp_path / "all.nc"),
                           "--combined-receipt", str(tmp_path / "all.json")],
                          cwd=".", capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "not complete" in done.stderr
    assert not (tmp_path / "all.nc").exists()


def test_combine_refuses_a_batch_whose_source_no_longer_matches_its_hash(tmp_path):
    stamps = list(pd.date_range("2017-01-01", periods=4, freq="6h"))
    done = subprocess.run([sys.executable, _DRIVER, "--combine",
                           "--batches", str(_batch_dir(tmp_path, "a", stamps, tamper=True)),
                           "--combined-nc", str(tmp_path / "all.nc"),
                           "--combined-receipt", str(tmp_path / "all.json")],
                          cwd=".", capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "unverified batch" in done.stderr
    assert not (tmp_path / "all.nc").exists()
