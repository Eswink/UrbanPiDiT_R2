from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
