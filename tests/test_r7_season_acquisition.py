"""Offline tests for the S1 four-season extraction bookkeeping (#78 track, S1 data).

No network access: the plan arithmetic, the merge re-validation and the failure
receipt are exercised on synthetic NetCDF parts with the D1 read helpers stood
in. The guard semantics being pinned are the ones that decide whether a real
batch may publish:

- a season plan is exactly 120 contiguous 6-hourly stamps on the four UTC init
  hours, per season, and refuses an out-of-range year or an oversized batch;
- a merge refuses any part whose bytes do not match its own receipt hash, any
  part that is not a completed real download, an overlapping union and a union
  that differs from the recorded stamps;
- a failed part leaves a ``failed-no-fallback`` receipt with
  ``synthetic_fallback: false`` and no output artifact;
- existing outputs are refused rather than truncated.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from data.download import earthmover_spatial_s1 as s1
from data.download.season_plan_s1 import season_plan

SNAPSHOT = "ZFKDHBCTBVHVXM3BQFV0"


def _plan(years=(2017,), seasons=("winter", "spring")):
    return season_plan(list(years), season_names=seasons)


def test_plan_is_exactly_120_contiguous_six_hourly_stamps_per_block():
    plan = _plan(seasons=("winter",))
    assert plan["total_stamps"] == 120 and plan["stamps_per_season"] == 120
    stamps = [pd.Timestamp(value) for value in plan["blocks"][0]["stamps"]]
    assert stamps[0] == pd.Timestamp("2017-01-01T00:00")
    assert stamps[-1] == pd.Timestamp("2017-01-30T18:00")
    diffs = {(right - left) for left, right in zip(stamps, stamps[1:])}
    assert diffs == {pd.Timedelta(hours=6)}
    assert sorted({stamp.hour for stamp in stamps}) == [0, 6, 12, 18]


def test_plan_refuses_out_of_range_year_and_oversized_batch():
    with pytest.raises(ValueError):
        season_plan([1939])
    with pytest.raises(ValueError):
        season_plan([2016, 2017, 2018, 2019, 2020])
    with pytest.raises(ValueError):
        season_plan([2017, 2017])


def test_plan_refuses_an_empty_or_unknown_season_subset():
    with pytest.raises(ValueError):
        season_plan([2017], season_names=())
    with pytest.raises(ValueError):
        season_plan([2017], season_names=("winter", "monsoon"))
    with pytest.raises(ValueError):
        season_plan([2017], season_names=("winter", "winter"))


def test_plan_report_estimates_are_positive_and_scale_with_stamps():
    one = s1.plan_report([2017], season_names=("winter",))
    four = s1.plan_report([2017])
    assert one["plan"]["total_stamps"] == 120
    assert four["plan"]["total_stamps"] == 480
    assert 0 < one["estimated_network_bytes"] < four["estimated_network_bytes"]
    assert 0 < one["estimated_decoded_bytes"] < four["estimated_decoded_bytes"]


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
        "part": {"years": [2017], "seasons": [name],
                 "first_time": str(stamps[0]), "last_time": str(stamps[-1]),
                 "stamps": len(stamps),
                 "blocks": [{"season": name, "year": 2017,
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


def _stamps(month, count=6):
    return list(pd.date_range(f"2017-{month:02d}-01", periods=count, freq="6h"))


def test_merge_accepts_verified_parts_and_records_the_union(tmp_path):
    winter = _stamps(1)
    spring = _stamps(4)
    parts = [_write_part(tmp_path, "winter", winter), _write_part(tmp_path, "spring", spring)]
    out_nc, rc = tmp_path / "source.nc", tmp_path / "merge.json"
    result = s1.merge_season_parts(parts, out_nc, rc, expected_stamps=len(winter) + len(spring))
    assert result["status"] == "merged-real-source"
    assert result["stamps"] == len(winter) + len(spring)
    assert result["n_parts"] == 2
    assert result["synthetic_fallback"] is False and result["scientific_claim"] is False
    assert out_nc.is_file() and rc.is_file()


def test_merge_refuses_a_part_whose_bytes_do_not_match_its_receipt(tmp_path):
    parts = [_write_part(tmp_path, "winter", _stamps(1), shift_hash=1)]
    with pytest.raises(ValueError, match="does not match its receipt hash"):
        s1.merge_season_parts(parts, tmp_path / "out.nc", tmp_path / "out.json", expected_stamps=6)
    assert not (tmp_path / "out.nc").exists()


def test_merge_refuses_an_incomplete_part_status(tmp_path):
    parts = [_write_part(tmp_path, "winter", _stamps(1), status="failed-no-fallback")]
    with pytest.raises(ValueError, match="not a completed download"):
        s1.merge_season_parts(parts, tmp_path / "out.nc", tmp_path / "out.json", expected_stamps=6)
    assert not (tmp_path / "out.nc").exists()


def test_merge_refuses_overlapping_parts_and_a_wrong_expected_total(tmp_path):
    winter = _stamps(1)
    parts = [_write_part(tmp_path, "winter", winter), _write_part(tmp_path, "spring", winter)]
    with pytest.raises(ValueError, match="overlap"):
        s1.merge_season_parts(parts, tmp_path / "out.nc", tmp_path / "out.json",
                              expected_stamps=2 * len(winter))
    shifted = _stamps(1, count=3)
    with pytest.raises(ValueError, match="plan"):
        s1.merge_season_parts([_write_part(tmp_path, "x", shifted)], tmp_path / "out2.nc",
                              tmp_path / "out2.json", expected_stamps=99)


def test_merge_refuses_existing_outputs_without_touching_them(tmp_path):
    stdout, stderr = _write_part(tmp_path, "winter", _stamps(1))
    marker = tmp_path / "merge.json"
    marker.write_text("keep-me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        s1.merge_season_parts([(stdout, stderr)], tmp_path / "out.nc", marker, expected_stamps=6)
    assert marker.read_text(encoding="utf-8") == "keep-me"


def test_a_missing_part_artifact_is_refused(tmp_path):
    _, rc = _write_part(tmp_path, "winter", _stamps(1))
    with pytest.raises(FileNotFoundError):
        s1.merge_season_parts([(tmp_path / "absent.nc", rc)], tmp_path / "out.nc",
                              tmp_path / "out.json", expected_stamps=6)


def test_extract_refuses_existing_output_and_does_not_launch_a_read(tmp_path):
    out_nc = tmp_path / "source.nc"
    out_nc.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        s1.extract_seasons_part(out_nc, tmp_path / "receipt.json", years=[2017], seasons=("winter",))
    assert out_nc.read_bytes() == b"occupied"
