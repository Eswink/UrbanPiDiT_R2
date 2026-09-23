"""Bounded real WeatherBench2 surface pilot for R7 CPU integration tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from .chunk_budget import estimate_bundle_chunk_bytes
from .wb2_public_probe import WB2_ERA5_025, describe_dataset
from data.preprocess.grid import regular_latlon_spacing

SURFACE_VARIABLES=(
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
PILOT_TIMES=(
    "2018-01-01T00:00:00",
    "2018-01-01T06:00:00",
    "2018-01-01T12:00:00",
    "2019-01-01T00:00:00",
    "2019-01-01T06:00:00",
    "2019-01-01T12:00:00",
    "2020-01-01T00:00:00",
    "2020-01-01T06:00:00",
    "2020-01-01T12:00:00",
)
# 16 x 24 native 0.25-degree points.
PILOT_ROI=(38.25,42.0,114.0,119.75)
DEFAULT_CHUNK_CAP=192*1024*1024

EXPECTED_UNITS={
    "2m_temperature":"K",
    "10m_u_component_of_wind":"m s**-1",
    "10m_v_component_of_wind":"m s**-1",
    "mean_sea_level_pressure":"Pa",
}


def _exact_time_indices(values, desired:Sequence[str]) -> list[int]:
    source=np.asarray(values).astype("datetime64[ns]")
    if source.ndim!=1 or source.size<1:
        raise ValueError("time coordinate must be nonempty 1-D")
    targets=np.asarray(desired,dtype="datetime64[ns]")
    indices=np.searchsorted(source,targets)
    if (indices>=source.size).any():
        raise ValueError("requested pilot time outside source")
    matched=source[indices]
    if not np.array_equal(matched,targets):
        missing=[str(t) for t,m in zip(targets,matched) if t!=m]
        raise ValueError(f"source is missing exact pilot times: {missing}")
    if len(set(int(v) for v in indices))!=len(indices):
        raise ValueError("pilot times are not unique")
    return [int(v) for v in indices]


def _roi_slice(values,lower:float,upper:float,name:str) -> slice:
    arr=np.asarray(values,dtype=np.float64)
    if arr.ndim!=1 or arr.size<2 or not np.isfinite(arr).all():
        raise ValueError(f"{name} coordinate must be finite 1-D")
    mask=(arr>=lower-1e-8)&(arr<=upper+1e-8)
    indices=np.flatnonzero(mask)
    if indices.size<2 or not np.array_equal(indices,np.arange(indices[0],indices[-1]+1)):
        raise ValueError(f"{name} ROI is not one contiguous source slice")
    return slice(int(indices[0]),int(indices[-1])+1)


def select_surface_pilot(
    ds:xr.Dataset,
    *,
    max_chunk_bytes:int=DEFAULT_CHUNK_CAP,
)->tuple[xr.Dataset,dict]:
    missing=[v for v in SURFACE_VARIABLES if v not in ds]
    if missing:
        raise KeyError(f"WeatherBench2 missing surface variables: {missing}")
    for name,unit in EXPECTED_UNITS.items():
        if str(ds[name].attrs.get("units",""))!=unit:
            raise ValueError(f"{name} units changed: {ds[name].attrs.get('units')!r} != {unit!r}")

    times=_exact_time_indices(ds["time"].values,PILOT_TIMES)
    south,north,west,east=PILOT_ROI
    lat_slice=_roi_slice(ds["latitude"].values,south,north,"latitude")
    lon_slice=_roi_slice(ds["longitude"].values,west,east,"longitude")

    metadata=describe_dataset(ds,SURFACE_VARIABLES)
    selections={
        name:{
            "time":times,
            "latitude":lat_slice,
            "longitude":lon_slice,
        }
        for name in SURFACE_VARIABLES
    }
    budget=estimate_bundle_chunk_bytes(
        metadata,selections,max_bytes=int(max_chunk_bytes)
    )

    # The first field-value read happens only after the source-chunk budget gate.
    subset=ds[list(SURFACE_VARIABLES)].isel(
        time=times,latitude=lat_slice,longitude=lon_slice
    ).load()
    if subset.sizes["time"]!=9 or subset.sizes["latitude"]!=16 or subset.sizes["longitude"]!=24:
        raise ValueError(f"unexpected pilot subset shape: {dict(subset.sizes)}")
    spacing=regular_latlon_spacing(
        np.asarray(subset["latitude"].values),
        np.asarray(subset["longitude"].values),
    )
    if not np.isclose(spacing,0.25,rtol=0,atol=1e-6):
        raise ValueError(f"pilot source spacing changed: {spacing}")
    for name in SURFACE_VARIABLES:
        values=np.asarray(subset[name].values)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")

    receipt={
        "source":WB2_ERA5_025,
        "access":"anonymous-public-gcs",
        "variables":list(SURFACE_VARIABLES),
        "units":{name:EXPECTED_UNITS[name] for name in SURFACE_VARIABLES},
        "selected_times":[str(np.datetime64(t,"ns")) for t in PILOT_TIMES],
        "roi":{"south":south,"north":north,"west":west,"east":east},
        "native_grid_spacing_deg":float(spacing),
        "shape":[9,4,16,24],
        "source_chunk_budget":budget,
        "interpolation":False,
        "scientific_training_ready":False,
        "limitations":[
            "one exact +6h transition per train/val/test year",
            "surface-only bundle; no pressure-level process supervision",
            "chunk estimate is an uncompressed source-chunk bound, not measured network bytes",
        ],
    }
    return subset,receipt


def download_surface_pilot(
    output_nc:str|Path,
    receipt_json:str|Path,
    *,
    source:str=WB2_ERA5_025,
    max_chunk_bytes:int=DEFAULT_CHUNK_CAP,
)->dict:
    output_nc,receipt_json=Path(output_nc),Path(receipt_json)
    for path in (output_nc,receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    output_nc.parent.mkdir(parents=True,exist_ok=True)
    receipt_json.parent.mkdir(parents=True,exist_ok=True)
    ds=xr.open_zarr(source,chunks=None,storage_options={"token":"anon"})
    try:
        subset,receipt=select_surface_pilot(ds,max_chunk_bytes=max_chunk_bytes)
        receipt["source"]=source
        encoding={
            name:{"compression":"gzip","compression_opts":1,"chunksizes":(1,16,24)}
            for name in SURFACE_VARIABLES
        }
        subset.to_netcdf(output_nc,engine="h5netcdf",encoding=encoding)
    finally:
        ds.close()
    digest=hashlib.sha256(output_nc.read_bytes()).hexdigest()
    receipt["local_artifact"]={
        "path":output_nc.name,
        "bytes":output_nc.stat().st_size,
        "sha256":digest,
    }
    with receipt_json.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False)
    return receipt
