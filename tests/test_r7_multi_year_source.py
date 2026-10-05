"""Offline tests for the S3 multi-batch source combiner (no network, no GPU)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from data.download import multi_year_source as combine

_SURFACE = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure")
_PRESSURE = ("geopotential", "temperature", "specific_humidity",
             "u_component_of_wind", "v_component_of_wind")
_LEVELS = (250, 500, 850)


def _dataset(stamps, *, offset=0.0, latitude=None):
    import xarray as xr

    latitude = np.linspace(43.0, 42.25, 4, dtype=np.float32) if latitude is None else latitude
    longitude = np.linspace(107.0, 107.75, 4, dtype=np.float32)
    data_vars = {}
    for index, name in enumerate(_SURFACE):
        values = np.full((len(stamps), 4, 4), 250.0 + index + offset, dtype=np.float32)
        data_vars[name] = (("time", "latitude", "longitude"), values)
    for index, name in enumerate(_PRESSURE):
        values = np.full((len(stamps), len(_LEVELS), 4, 4), 250.0 + index + offset,
                         dtype=np.float32)
        data_vars[name] = (("time", "level", "latitude", "longitude"), values)
    dataset = xr.Dataset(data_vars, coords={
        "time": list(stamps), "level": np.asarray(_LEVELS, dtype=np.int32),
        "latitude": latitude, "longitude": longitude})
    dataset["level"].attrs = {"units": "hPa"}
    dataset.attrs.update({"source_snapshot_id": "ZFKDHBCTBVHVXM3BQFV0",
                          "license": "CC-BY-4.0"})
    return dataset


def _write_source(tmp_path, name, stamps, *, offset=0.0, status="merged-real-source",
                  shift_hash=0, latitude=None):
    nc = tmp_path / f"{name}.nc"
    _dataset(stamps, offset=offset, latitude=latitude).to_netcdf(nc, engine="h5netcdf")
    digest = hashlib.sha256(nc.read_bytes()).hexdigest()
    if shift_hash:
        digest = f"{int(digest, 16) ^ shift_hash:064x}"
    receipt = {"status": status,
               "local_artifact": {"path": nc.name, "bytes": nc.stat().st_size,
                                  "sha256": digest}}
    rc = tmp_path / f"{name}_receipt.json"
    rc.write_text(json.dumps(receipt), encoding="utf-8")
    return nc, rc


def _stamps(year, month, count=6):
    return list(pd.date_range(f"{year}-{month:02d}-01", periods=count, freq="6h"))


def test_combine_accepts_verified_sources_in_chronological_order(tmp_path):
    first = _stamps(2017, 1)
    second = _stamps(2022, 1, count=4)
    pairs = [_write_source(tmp_path, "s1", first),
             _write_source(tmp_path, "b2", second, offset=3.0)]
    out_nc, rc = tmp_path / "combined.nc", tmp_path / "combined.json"
    result = combine.combine_sources(pairs, out_nc, rc,
                                     expected_stamps=len(first) + len(second))
    assert result["status"] == "combined-real-source"
    assert result["stamps"] == len(first) + len(second)
    assert result["n_sources"] == 2
    assert result["synthetic_fallback"] is False and result["scientific_claim"] is False
    assert out_nc.is_file() and rc.is_file()
    import xarray as xr
    with xr.open_dataset(out_nc) as merged:
        times = pd.DatetimeIndex(merged["time"].values)
        assert times[0] == pd.Timestamp(first[0]) and times[-1] == pd.Timestamp(second[-1])
        assert bool(np.isfinite(merged["2m_temperature"].values).all())
    assert result["channel_count"] == 17


def test_combine_refuses_a_source_whose_bytes_do_not_match_its_receipt(tmp_path):
    pairs = [_write_source(tmp_path, "s1", _stamps(2017, 1), shift_hash=1)]
    with pytest.raises(ValueError, match="does not match its receipt hash"):
        combine.combine_sources(pairs, tmp_path / "out.nc", tmp_path / "out.json",
                                expected_stamps=6)
    assert not (tmp_path / "out.nc").exists()


def test_combine_refuses_an_unmerged_source_status(tmp_path):
    pairs = [_write_source(tmp_path, "s1", _stamps(2017, 1), status="downloaded-real-source")]
    with pytest.raises(ValueError, match="not a completed merge"):
        combine.combine_sources(pairs, tmp_path / "out.nc", tmp_path / "out.json",
                                expected_stamps=6)


def test_combine_refuses_overlaps_and_a_wrong_total(tmp_path):
    shared = _stamps(2017, 1)
    overlapping = [_write_source(tmp_path, "a", shared), _write_source(tmp_path, "b", shared)]
    with pytest.raises(ValueError, match="overlap"):
        combine.combine_sources(overlapping, tmp_path / "out.nc", tmp_path / "out.json",
                                expected_stamps=2 * len(shared))
    single = [_write_source(tmp_path, "c", shared)]
    with pytest.raises(ValueError, match="expected 99 stamps"):
        combine.combine_sources(single, tmp_path / "out2.nc", tmp_path / "out2.json",
                                expected_stamps=99)


def test_combine_refuses_a_grid_or_level_mismatch(tmp_path):
    shifted_lat = np.linspace(43.0, 42.0, 4, dtype=np.float32)
    pairs = [_write_source(tmp_path, "a", _stamps(2017, 1)),
             _write_source(tmp_path, "b", _stamps(2022, 1), latitude=shifted_lat)]
    with pytest.raises(ValueError, match="coordinate 'latitude' differs"):
        combine.combine_sources(pairs, tmp_path / "out.nc", tmp_path / "out.json",
                                expected_stamps=12)


def test_combine_refuses_out_of_order_inputs_instead_of_sorting(tmp_path):
    late = _write_source(tmp_path, "late", _stamps(2022, 1), offset=1.0)
    early = _write_source(tmp_path, "early", _stamps(2017, 1))
    with pytest.raises(ValueError, match="sorted union"):
        combine.combine_sources([late, early], tmp_path / "out.nc", tmp_path / "out.json",
                                expected_stamps=12)


def test_combine_refuses_existing_outputs_without_touching_them(tmp_path):
    pairs = [_write_source(tmp_path, "a", _stamps(2017, 1))]
    marker = tmp_path / "out.json"
    marker.write_text("keep-me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        combine.combine_sources(pairs, tmp_path / "out.nc", marker, expected_stamps=6)
    assert marker.read_text(encoding="utf-8") == "keep-me"


def test_combine_refuses_a_missing_artifact(tmp_path):
    _, rc = _write_source(tmp_path, "a", _stamps(2017, 1))
    with pytest.raises(FileNotFoundError):
        combine.combine_sources([(tmp_path / "absent.nc", rc)], tmp_path / "out.nc",
                                tmp_path / "out.json", expected_stamps=6)


def test_combine_records_every_source_identity_in_the_receipt(tmp_path):
    pairs = [_write_source(tmp_path, "s1", _stamps(2017, 1)),
             _write_source(tmp_path, "b2", _stamps(2022, 1), offset=1.0)]
    result = combine.combine_sources(pairs, tmp_path / "out.nc", tmp_path / "out.json",
                                     expected_stamps=12)
    assert [entry["path"] for entry in result["sources"]] == ["s1.nc", "b2.nc"]
    for entry, (nc, _) in zip(result["sources"], pairs):
        assert entry["sha256"] == hashlib.sha256(nc.read_bytes()).hexdigest()
