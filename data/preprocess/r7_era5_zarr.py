from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .r7_era5 import (
    DEFAULT_R7_ERA5_CHANNELS,
    ERA5ChannelSpec,
    _coord_name,
    _grid_spacing,
    _validate_year_splits,
    stack_era5_channels,
)


def _write_jsonl(path:Path,records:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")


def _window_records(
    times,
    *,
    split_sets:Mapping[str,set[int]],
    store_path:Path,
    manifest_dir:Path,
    history_steps:int,
    history_interval_hours:int,
    lead_time_hours:int,
    sample_stride_hours:int,
)->dict[str,list[dict]]:
    import pandas as pd

    index={int(ts.value):i for i,ts in enumerate(times)}
    records_by_split={key:[] for key in ("train","val","test")}
    hour_ns=3_600_000_000_000

    for split,allowed_years in split_sets.items():
        for init_time in times:
            if int(init_time.year) not in allowed_years:
                continue
            if (
                init_time.minute!=0
                or init_time.second!=0
                or (int(init_time.value)//hour_ns)%sample_stride_hours!=0
            ):
                continue

            history_times=[
                init_time-pd.Timedelta(
                    hours=history_interval_hours*(history_steps-1-j)
                )
                for j in range(history_steps)
            ]
            target_time=init_time+pd.Timedelta(hours=lead_time_hours)
            required=history_times+[target_time]

            if any(int(t.year) not in allowed_years for t in required):
                continue
            keys=[int(t.value) for t in required]
            if any(k not in index for k in keys):
                continue

            sid=(
                f"era5z_{split}_{init_time:%Y%m%d%H}_"
                f"p{lead_time_hours:03d}h"
            )
            records_by_split[split].append({
                "sample_id":sid,
                "store_path":os.path.relpath(
                    store_path.resolve(),manifest_dir.resolve()
                ),
                "split":split,
                "history_indices":[index[k] for k in keys[:-1]],
                "target_index":index[keys[-1]],
                "history_times":[t.isoformat() for t in history_times],
                "init_time":init_time.isoformat(),
                "target_time":target_time.isoformat(),
                "lead_time_hours":int(lead_time_hours),
            })

    return records_by_split


def build_r7_era5_zarr_from_dataset(
    ds,
    *,
    store_path:str|Path,
    manifest_dir:str|Path,
    specs:Iterable[ERA5ChannelSpec]=DEFAULT_R7_ERA5_CHANNELS,
    split_years:Mapping[str,Sequence[int]],
    history_steps:int=2,
    history_interval_hours:int=6,
    lead_time_hours:int=6,
    sample_stride_hours:int=6,
    expected_grid_spacing_deg:float|None=0.25,
    source_label:str="ERA5 regional subset",
    time_chunk:int=64,
    spatial_chunk:tuple[int,int]=(64,64),
)->dict[str,Path]:
    """Stream physical-unit regional ERA5 into one chunked Zarr store.

    The Zarr state array is intentionally *not normalized*. Channel-wise
    training-period mean/std are stored separately and applied by the Dataset at
    read time. This preserves physical units for later meteorological process
    diagnostics and avoids duplicating normalized copies of the archive.
    """
    try:
        import pandas as pd
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "R7 Zarr preprocessing requires pandas and zarr; "
            "install requirements-r7-data.txt"
        ) from exc

    if history_steps<1:
        raise ValueError("history_steps 必须 >= 1")
    if min(history_interval_hours,lead_time_hours,sample_stride_hours)<=0:
        raise ValueError("history/lead/stride hours 必须为正数")
    if time_chunk<1 or min(spatial_chunk)<1:
        raise ValueError("chunk sizes 必须为正数")

    split_sets=_validate_year_splits(split_years)
    specs=tuple(specs)
    if not specs:
        raise ValueError("至少需要一个 ERA5 channel spec")

    time_name=_coord_name(ds,("time","valid_time"))
    lat_name=_coord_name(ds,("latitude","lat"))
    lon_name=_coord_name(ds,("longitude","lon"))
    ds=ds.sortby(time_name)

    times=pd.DatetimeIndex(np.asarray(ds[time_name].values))
    if not times.is_monotonic_increasing:
        raise ValueError("ERA5 time 必须单调递增")
    if times.has_duplicates:
        raise ValueError("ERA5 time 存在重复")
    if len(times)<1:
        raise ValueError("ERA5 time 为空")

    latitude=np.asarray(ds[lat_name].values,dtype=np.float32)
    longitude=np.asarray(ds[lon_name].values,dtype=np.float32)
    spacing=_grid_spacing(latitude,longitude)
    if (
        expected_grid_spacing_deg is not None
        and not np.isclose(
            spacing,float(expected_grid_spacing_deg),rtol=0,atol=1e-4
        )
    ):
        raise ValueError(
            f"网格分辨率 {spacing}° != expected "
            f"{expected_grid_spacing_deg}°"
        )

    years=np.asarray(times.year,dtype=np.int32)
    train_mask=np.isin(years,list(split_sets["train"]))
    if not bool(train_mask.any()):
        raise ValueError("输入数据中没有 train years")

    store_path=Path(store_path)
    manifest_dir=Path(manifest_dir)
    store_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_dir.mkdir(parents=True,exist_ok=True)

    # Validate channel semantics on the first timestamp before creating the
    # archive. Each later iteration materializes only one time chunk.
    first,names,_,lat_check,lon_check=stack_era5_channels(
        ds.isel({time_name:slice(0,1)}),specs
    )
    if not np.array_equal(lat_check,latitude):
        raise ValueError("latitude 坐标在 channel stack 后发生变化")
    if not np.array_equal(lon_check,longitude):
        raise ValueError("longitude 坐标在 channel stack 后发生变化")

    T=len(times)
    C=len(names)
    H=len(latitude)
    W=len(longitude)

    root=zarr.open_group(str(store_path),mode="w")
    state=root.create_array(
        "state",
        shape=(T,C,H,W),
        chunks=(
            min(int(time_chunk),T),
            C,
            min(int(spatial_chunk[0]),H),
            min(int(spatial_chunk[1]),W),
        ),
        dtype="f4",
    )
    root.create_array(
        "latitude",
        data=latitude,
        chunks=(H,),
    )
    root.create_array(
        "longitude",
        data=longitude,
        chunks=(W,),
    )
    root.create_array(
        "time_ns",
        data=times.asi8.astype(np.int64),
        chunks=(min(int(time_chunk),T),),
    )

    sum_c=np.zeros(C,dtype=np.float64)
    sumsq_c=np.zeros(C,dtype=np.float64)
    train_count=0

    for start in range(0,T,int(time_chunk)):
        stop=min(T,start+int(time_chunk))
        block,block_names,block_times,block_lat,block_lon=stack_era5_channels(
            ds.isel({time_name:slice(start,stop)}),specs
        )
        if block_names!=names:
            raise ValueError("ERA5 channel order changed between chunks")
        if not np.array_equal(block_lat,latitude):
            raise ValueError("latitude changed between chunks")
        if not np.array_equal(block_lon,longitude):
            raise ValueError("longitude changed between chunks")
        if not np.array_equal(
            np.asarray(block_times),np.asarray(times[start:stop])
        ):
            raise ValueError("time coordinate changed between chunks")

        state[start:stop]=block
        local_train=train_mask[start:stop]
        if bool(local_train.any()):
            selected=block[local_train].astype(np.float64)
            sum_c+=selected.sum(axis=(0,2,3))
            sumsq_c+=(selected*selected).sum(axis=(0,2,3))
            train_count+=int(selected.shape[0])*H*W

    if train_count<=0:
        raise RuntimeError("无法从 train years 计算 normalization")
    mean=sum_c/train_count
    variance=np.maximum(sumsq_c/train_count-mean*mean,1e-12)
    std=np.sqrt(variance)

    root.create_array(
        "normalization_mean",
        data=mean.astype(np.float32),
        chunks=(C,),
    )
    root.create_array(
        "normalization_std",
        data=std.astype(np.float32),
        chunks=(C,),
    )
    root.attrs.update({
        "source":source_label,
        "channels":names,
        "native_grid_spacing_deg":float(spacing),
        "target_semantics":"native ERA5 grid; physical units retained; no spatial upsampling",
        "normalization":"training-years-only mean/std; applied at read time",
        "normalization_years":sorted(split_sets["train"]),
        "split_years":{
            key:sorted(value) for key,value in split_sets.items()
        },
    })

    records_by_split=_window_records(
        times,
        split_sets=split_sets,
        store_path=store_path,
        manifest_dir=manifest_dir,
        history_steps=history_steps,
        history_interval_hours=history_interval_hours,
        lead_time_hours=lead_time_hours,
        sample_stride_hours=sample_stride_hours,
    )
    for split,records in records_by_split.items():
        if not records:
            raise RuntimeError(
                f"{split} 没有可用窗口；检查年份、时间范围和 cadence"
            )
        _write_jsonl(manifest_dir/f"{split}.jsonl",records)

    metadata={
        "source":source_label,
        "store_path":os.path.relpath(
            store_path.resolve(),manifest_dir.resolve()
        ),
        "channels":names,
        "shape":[T,C,H,W],
        "chunks":list(state.chunks),
        "native_grid_spacing_deg":float(spacing),
        "history_steps":int(history_steps),
        "history_interval_hours":int(history_interval_hours),
        "lead_time_hours":int(lead_time_hours),
        "sample_stride_hours":int(sample_stride_hours),
        "split_years":{
            key:sorted(value) for key,value in split_sets.items()
        },
        "samples_by_split":{
            key:len(value) for key,value in records_by_split.items()
        },
        "normalization_years":sorted(split_sets["train"]),
        "physical_units_retained":True,
        "spatial_resampling":False,
    }
    (manifest_dir/"zarr_metadata.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )
    return {
        split:manifest_dir/f"{split}.jsonl"
        for split in ("train","val","test")
    }


def build_r7_era5_zarr_from_path(
    source:str|Path,
    **kwargs,
)->dict[str,Path]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "R7 ERA5 path adapter requires optional dependency xarray; "
            "install requirements-r7-data.txt"
        ) from exc

    source=Path(source)
    if source.is_dir() or source.suffix.lower()==".zarr":
        ds=xr.open_zarr(source,chunks=None)
    else:
        ds=xr.open_dataset(source)
    try:
        return build_r7_era5_zarr_from_dataset(
            ds,
            source_label=str(source),
            **kwargs,
        )
    finally:
        ds.close()
