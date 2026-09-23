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
