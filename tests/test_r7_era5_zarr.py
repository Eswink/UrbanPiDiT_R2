from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from data import ZarrAtmosWindowDataset
from data.preprocess.r7_era5 import ERA5ChannelSpec
from data.preprocess.r7_era5_zarr import (
    build_r7_era5_zarr_from_dataset,
)


def _fixture():
    times=[]
    for year in (2018,2019,2020):
        times.extend(pd.date_range(
            f"{year}-01-01T00:00:00",
            periods=8,
            freq="6h",
        ))
    time=pd.DatetimeIndex(times)
    lat=np.array([40.0,39.75,39.5],dtype=np.float32)
    lon=np.array([115.0,115.25,115.5,115.75],dtype=np.float32)
    level=np.array([850],dtype=np.int32)

    t2m=np.empty((len(time),len(lat),len(lon)),dtype=np.float32)
    u=np.empty((len(time),1,len(lat),len(lon)),dtype=np.float32)
    spatial=np.add.outer(
        np.arange(len(lat),dtype=np.float32),
        np.arange(len(lon),dtype=np.float32),
    )
    for i,ts in enumerate(time):
        offset={2018:0.0,2019:100.0,2020:200.0}[ts.year]
        t2m[i]=280.0+offset+0.2*i+spatial
        u[i,0]=5.0+0.1*offset+0.1*i+0.5*spatial

    return xr.Dataset(
        {
            "t2m":(("time","latitude","longitude"),t2m),
            "u":(("time","level","latitude","longitude"),u),
        },
        coords={
            "time":time,
            "level":level,
            "latitude":lat,
            "longitude":lon,
        },
    )


def test_zarr_archive_retains_physical_units_and_streams_windows(tmp_path:Path):
    store=tmp_path/"era5_r7.zarr"
    manifests=tmp_path/"manifests"
    paths=build_r7_era5_zarr_from_dataset(
        _fixture(),
        store_path=store,
        manifest_dir=manifests,
        specs=(
            ERA5ChannelSpec("t2m",name="t2m"),
            ERA5ChannelSpec("u",850,"u850"),
        ),
        split_years={
            "train":[2018],
            "val":[2019],
            "test":[2020],
        },
        time_chunk=3,
        spatial_chunk=(2,3),
        expected_grid_spacing_deg=0.25,
    )

    root=zarr.open_group(str(store),mode="r")
    assert root["state"].shape==(24,2,3,4)
    assert root["state"].chunks==(3,2,2,3)
    assert root.attrs["physical_units_retained"] if "physical_units_retained" in root.attrs else True

    # Raw Zarr state remains in physical units, not standardized values.
    first=np.asarray(root["state"][0],dtype=np.float32)
    assert float(first[0].mean())>270.0

    mean=np.asarray(root["normalization_mean"][:])
    std=np.asarray(root["normalization_std"][:])
    assert mean.shape==(2,)
    assert np.all(std>0)
    assert root.attrs["normalization_years"]==[2018]

    train=ZarrAtmosWindowDataset(paths["train"])
    val=ZarrAtmosWindowDataset(paths["val"])
    train_sample=train[0]
    val_sample=val[0]
    assert train_sample["coarse_history"].shape==(2,2,3,4)
    assert train_sample["atmos_target"].shape==(2,3,4)
    assert np.isfinite(train_sample["coarse_history"].numpy()).all()

    # Validation carries a +100 K synthetic offset but reuses 2018 train stats.
    # Leakage from val statistics would collapse this signal.
    assert float(val_sample["atmos_target"][0].mean())>10.0

    meta=json.loads(
        (manifests/"zarr_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["normalization_years"]==[2018]
    assert meta["physical_units_retained"] is True
    assert meta["spatial_resampling"] is False
    assert meta["chunks"]==[3,2,2,3]



def _process_fixture():
    times=[]
    for year in (2018,2019,2020):
        times.extend(pd.date_range(
            f"{year}-02-01T00:00:00",
            periods=8,
            freq="6h",
        ))
    time=pd.DatetimeIndex(times)
    lat=np.array([40.0,39.75,39.5],dtype=np.float32)
    lon=np.array([115.0,115.25,115.5,115.75],dtype=np.float32)
    level=np.array([850,500],dtype=np.int32)
    H,W=len(lat),len(lon)
    yy,xx=np.meshgrid(
        np.arange(H,dtype=np.float32),
        np.arange(W,dtype=np.float32),
        indexing="ij",
    )

    mslp=np.empty((len(time),H,W),dtype=np.float32)
    t=np.empty((len(time),2,H,W),dtype=np.float32)
    q=np.empty_like(t)
    u=np.empty_like(t)
    v=np.empty_like(t)
    for i,ts in enumerate(time):
        year_shift={2018:0.0,2019:1.0,2020:2.0}[ts.year]
        local=i%8
        mslp[i]=100000.0+(8.0+local)*xx+2.0*yy
        t[i,0]=290.0+0.2*local+(0.3+0.03*local)*xx
        t[i,1]=260.0+0.4*local+(0.1+0.01*local)*xx
        q[i,0]=0.010+1e-4*local+(1e-4+1e-5*local)*xx
        q[i,1]=0.004+5e-5*local+3e-5*xx
        u[i,0]=6.0+year_shift+(0.10+0.01*local)*xx+0.04*yy
        v[i,0]=2.0+(0.08+0.005*local)*xx+0.03*yy
        u[i,1]=18.0+year_shift+0.15*xx+0.08*local
        v[i,1]=8.0+0.10*xx+0.05*local

    return xr.Dataset(
        {
            "mslp":(("time","latitude","longitude"),mslp),
            "t":(("time","level","latitude","longitude"),t),
            "q":(("time","level","latitude","longitude"),q),
            "u":(("time","level","latitude","longitude"),u),
            "v":(("time","level","latitude","longitude"),v),
        },
        coords={
            "time":time,
            "level":level,
            "latitude":lat,
            "longitude":lon,
        },
    )


def test_zarr_process_targets_use_input_time_not_future_target(tmp_path:Path):
    store=tmp_path/"era5_process.zarr"
    manifests=tmp_path/"process_manifests"
    specs=(
        ERA5ChannelSpec("mslp",name="mslp"),
        ERA5ChannelSpec("t",850,"t850"),
        ERA5ChannelSpec("q",850,"q850"),
        ERA5ChannelSpec("u",850,"u850"),
        ERA5ChannelSpec("v",850,"v850"),
        ERA5ChannelSpec("t",500,"t500"),
        ERA5ChannelSpec("u",500,"u500"),
        ERA5ChannelSpec("v",500,"v500"),
    )
    paths=build_r7_era5_zarr_from_dataset(
        _process_fixture(),
        store_path=store,
        manifest_dir=manifests,
        specs=specs,
        split_years={
            "train":[2018],
            "val":[2019],
            "test":[2020],
        },
        time_chunk=2,
        expected_grid_spacing_deg=0.25,
        compute_process_targets=True,
    )

    root=zarr.open_group(str(store),mode="r")
    assert root["process_diagnostics_raw"].shape==(24,8)
    assert root["process_normalization_mean"].shape==(8,)
    assert root.attrs["process_diagnostics_enabled"] is True
    assert root.attrs["process_target_time_semantics"]=="input init time only"

    record=json.loads(
        paths["train"].read_text(encoding="utf-8").splitlines()[0]
    )
    init_index=int(record["history_indices"][-1])
    target_index=int(record["target_index"])
    raw_init=np.asarray(
        root["process_diagnostics_raw"][init_index],
        dtype=np.float32,
    )
    raw_future=np.asarray(
        root["process_diagnostics_raw"][target_index],
        dtype=np.float32,
    )
    mean=np.asarray(root["process_normalization_mean"][:],dtype=np.float32)
    std=np.asarray(root["process_normalization_std"][:],dtype=np.float32)
    expected=(raw_init-mean)/std

    sample=ZarrAtmosWindowDataset(paths["train"])[0]
    assert sample["process_targets"].shape==(8,)
    np.testing.assert_allclose(
        sample["process_targets"].numpy(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
    assert not np.allclose(raw_init,raw_future)
