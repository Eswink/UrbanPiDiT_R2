from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.r7_dataset import ManifestAtmosNPZDataset
from data.preprocess.r7_era5 import (
    ERA5ChannelSpec,
    build_r7_era5_npz_from_dataset,
)


def _fixture():
    blocks=[]
    for year,offset in [(2018,0.0),(2019,100.0),(2020,200.0)]:
        t=pd.date_range(
            f"{year}-01-01T00:00:00",
            periods=8,
            freq="6h",
        )
        blocks.extend(t)
    time=pd.DatetimeIndex(blocks)
    lat=np.array([40.0,39.75,39.5],dtype=np.float32)
    lon=np.array([115.0,115.25,115.5,115.75],dtype=np.float32)
    level=np.array([850,500],dtype=np.int32)

    nt=len(time)
    base=np.zeros((nt,len(lat),len(lon)),dtype=np.float32)
    u=np.zeros((nt,len(level),len(lat),len(lon)),dtype=np.float32)
    for i,ts in enumerate(time):
        year_offset={2018:0.0,2019:100.0,2020:200.0}[ts.year]
        spatial=np.add.outer(
            np.arange(len(lat),dtype=np.float32),
            np.arange(len(lon),dtype=np.float32),
        )
        base[i]=year_offset+i*0.1+spatial
        u[i,0]=year_offset+10.0+i*0.2+spatial
        # Keep a genuinely different level-specific pattern so channel-wise
        # normalization cannot make 500 hPa collapse onto 850 hPa.
        u[i,1]=year_offset+20.0+i*0.3+2.0*spatial

    return xr.Dataset(
        {
            "t2m":(("time","latitude","longitude"),base),
            "u":(("time","level","latitude","longitude"),u),
        },
        coords={
            "time":time,
            "level":level,
            "latitude":lat,
            "longitude":lon,
        },
    )


def _records(path:Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_r7_era5_converter_is_leakage_safe_and_native(tmp_path:Path):
    specs=(
        ERA5ChannelSpec("t2m",name="t2m"),
        ERA5ChannelSpec("u",850,"u850"),
        ERA5ChannelSpec("u",500,"u500"),
    )
    manifests=tmp_path/"manifests"
    paths=build_r7_era5_npz_from_dataset(
        _fixture(),
        out_dir=tmp_path/"processed",
        manifest_dir=manifests,
        specs=specs,
        split_years={
            "train":[2018],
            "val":[2019],
            "test":[2020],
        },
        history_steps=2,
        history_interval_hours=6,
        lead_time_hours=6,
        sample_stride_hours=6,
        expected_grid_spacing_deg=0.25,
        source_label="synthetic-xarray-era5-fixture",
    )

    for split,year in [("train",2018),("val",2019),("test",2020)]:
        records=_records(paths[split])
        assert records
        for rec in records:
            assert pd.Timestamp(rec["init_time"]).year==year
            assert pd.Timestamp(rec["target_time"]).year==year
            assert all(
                pd.Timestamp(t).year==year
                for t in rec["history_times"]
            )

    train_ds=ManifestAtmosNPZDataset(paths["train"])
    sample=train_ds[0]
    assert sample["coarse_history"].shape==(2,3,3,4)
    assert sample["atmos_target"].shape==(3,3,4)
    assert float(sample["grid_spacing_deg"])==0.25

    norm=json.loads(
        (manifests/"normalization.json").read_text(encoding="utf-8")
    )
    assert norm["channels"]==["t2m","u850","u500"]
    assert norm["computed_from_years"]==[2018]

    provenance=json.loads(
        (manifests/"provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["native_grid_spacing_deg"]==0.25
    assert provenance["target_semantics"]=="native ERA5 grid; no spatial upsampling"

    # 2019 was offset by +100 but normalization uses 2018 only. If validation
    # statistics had leaked into normalization, this magnitude would collapse.
    val_sample=ManifestAtmosNPZDataset(paths["val"])[0]
    assert float(val_sample["atmos_target"].mean())>10.0


def test_pressure_level_selection_is_exact(tmp_path:Path):
    specs=(
        ERA5ChannelSpec("u",850,"u850"),
        ERA5ChannelSpec("u",500,"u500"),
    )
    paths=build_r7_era5_npz_from_dataset(
        _fixture(),
        out_dir=tmp_path/"processed",
        manifest_dir=tmp_path/"manifests",
        specs=specs,
        split_years={
            "train":[2018],
            "val":[2019],
            "test":[2020],
        },
        expected_grid_spacing_deg=0.25,
    )
    sample=ManifestAtmosNPZDataset(paths["train"])[0]
    assert sample["coarse_history"].shape[1]==2
    # The two pressure levels carry different spatiotemporal structures and
    # must remain distinguishable after channel-wise standardization.
    assert not np.allclose(
        sample["coarse_history"][0,0].numpy(),
        sample["coarse_history"][0,1].numpy(),
    )



def test_missing_timestamp_is_never_bridged(tmp_path:Path):
    ds=_fixture()
    # Remove one required 6-hour timestamp from the training year. The
    # converter may keep other valid windows, but no sample may bridge the gap.
    missing=pd.Timestamp("2018-01-01T12:00:00")
    ds=ds.sel(time=ds.time!=np.datetime64(missing))

    paths=build_r7_era5_npz_from_dataset(
        ds,
        out_dir=tmp_path/"processed",
        manifest_dir=tmp_path/"manifests",
        specs=(ERA5ChannelSpec("t2m",name="t2m"),),
        split_years={
            "train":[2018],
            "val":[2019],
            "test":[2020],
        },
        history_steps=2,
        history_interval_hours=6,
        lead_time_hours=6,
        sample_stride_hours=6,
        expected_grid_spacing_deg=0.25,
    )

    for record in _records(paths["train"]):
        required=[
            pd.Timestamp(t) for t in record["history_times"]
        ]+[pd.Timestamp(record["target_time"])]
        assert missing not in required


def test_irregular_grid_cannot_masquerade_as_native_025(tmp_path:Path):
    ds=_fixture().assign_coords(
        longitude=np.array(
            [115.0,115.25,115.50,115.90],
            dtype=np.float32,
        )
    )
    with pytest.raises(ValueError,match="longitude 非规则网格"):
        build_r7_era5_npz_from_dataset(
            ds,
            out_dir=tmp_path/"processed",
            manifest_dir=tmp_path/"manifests",
            specs=(ERA5ChannelSpec("t2m",name="t2m"),),
            split_years={
                "train":[2018],
                "val":[2019],
                "test":[2020],
            },
            expected_grid_spacing_deg=0.25,
        )



def test_stack_preserves_single_timestamp_and_single_pressure_level():
    ds=_fixture().sel(level=[850]).isel(time=slice(0,1))
    from data.preprocess.r7_era5 import stack_era5_channels

    state,names,times,lat,lon=stack_era5_channels(
        ds,
        (
            ERA5ChannelSpec("t2m",name="t2m"),
            ERA5ChannelSpec("u",850,"u850"),
        ),
    )
    assert state.shape==(1,2,3,4)
    assert names==["t2m","u850"]
    assert len(times)==1


def test_stack_drops_only_singleton_auxiliary_dims():
    ds=_fixture()
    ds["t2m_aux"]=ds["t2m"].expand_dims(expver=[1])
    from data.preprocess.r7_era5 import stack_era5_channels

    state,_,_,_,_=stack_era5_channels(
        ds,(ERA5ChannelSpec("t2m_aux",name="t2m_aux"),)
    )
    assert state.shape==(24,1,3,4)

    # Use a fresh Dataset here. Reusing `ds` would already define expver=[1],
    # and xarray assignment would align the [1,5] variable onto that existing
    # coordinate, silently reducing the test variable back to a singleton.
    bad=_fixture()
    bad["t2m_bad"]=bad["t2m"].expand_dims(expver=[1,5])
    assert int(bad["t2m_bad"].sizes["expver"])==2
    with pytest.raises(ValueError,match="未处理非单例维度"):
        stack_era5_channels(
            bad,(ERA5ChannelSpec("t2m_bad",name="t2m_bad"),)
        )
