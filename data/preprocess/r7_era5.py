from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
from .grid import regular_latlon_spacing
from .contracts import chronological_splits, fresh_outputs, utc_time_index


@dataclass(frozen=True)
class ERA5ChannelSpec:
    variable: str
    level_hpa: int | None = None
    name: str | None = None

    @property
    def channel_name(self) -> str:
        if self.name:
            return self.name
        if self.level_hpa is None:
            return self.variable
        return f"{self.variable}_{self.level_hpa}hPa"


DEFAULT_R7_ERA5_CHANNELS = (
    ERA5ChannelSpec("2m_temperature", name="t2m"),
    ERA5ChannelSpec("10m_u_component_of_wind", name="u10"),
    ERA5ChannelSpec("10m_v_component_of_wind", name="v10"),
    ERA5ChannelSpec("mean_sea_level_pressure", name="mslp"),
    ERA5ChannelSpec("geopotential", 850, "z850"),
    ERA5ChannelSpec("temperature", 850, "t850"),
    ERA5ChannelSpec("specific_humidity", 850, "q850"),
    ERA5ChannelSpec("u_component_of_wind", 850, "u850"),
    ERA5ChannelSpec("v_component_of_wind", 850, "v850"),
    ERA5ChannelSpec("geopotential", 500, "z500"),
    ERA5ChannelSpec("temperature", 500, "t500"),
    ERA5ChannelSpec("specific_humidity", 500, "q500"),
    ERA5ChannelSpec("u_component_of_wind", 500, "u500"),
    ERA5ChannelSpec("v_component_of_wind", 500, "v500"),
    ERA5ChannelSpec("geopotential", 250, "z250"),
    ERA5ChannelSpec("u_component_of_wind", 250, "u250"),
    ERA5ChannelSpec("v_component_of_wind", 250, "v250"),
)


def _coord_name(obj, candidates:Sequence[str]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(f"找不到坐标，候选={list(candidates)}")


def _exact_level(da, level_name:str, level_hpa:int):
    values=np.asarray(da[level_name].values)
    idx=np.flatnonzero(np.isclose(values.astype(float),float(level_hpa)))
    if idx.size!=1:
        raise ValueError(f"{da.name} 不包含唯一 pressure level={level_hpa} hPa; available={values.tolist()}")
    return values[int(idx[0])].item()


def stack_era5_channels(ds,specs:Iterable[ERA5ChannelSpec]):
    """Materialize an already-subset regional dataset, not a global archive."""
    time_name=_coord_name(ds,("time","valid_time"))
    lat_name=_coord_name(ds,("latitude","lat"))
    lon_name=_coord_name(ds,("longitude","lon"))
    specs=tuple(specs)
    names=[spec.channel_name for spec in specs]
    if not names or len(names)!=len(set(names)):
        raise ValueError('channel specs must have unique, nonempty names')
    ds=ds.sortby(time_name)
    arrays=[]
    for spec in specs:
        if spec.variable not in ds:
            raise KeyError(f"ERA5 变量不存在: {spec.variable}")
        da=ds[spec.variable]
        if spec.level_hpa is not None:
            level_name=_coord_name(da,("level","pressure_level","isobaricInhPa"))
            selected=_exact_level(da,level_name,spec.level_hpa)
            da=da.sel({level_name:selected},drop=True)
        required={time_name,lat_name,lon_name}
        if not required.issubset(set(da.dims)):
            raise ValueError(f"{spec.channel_name} dims={da.dims} 缺少 {required}")
        for dim in [dim for dim in da.dims if dim not in required]:
            if int(da.sizes[dim])!=1:
                raise ValueError(f"{spec.channel_name} 存在未处理非单例维度 {dim}={int(da.sizes[dim])}")
            da=da.isel({dim:0},drop=True)
        arr=np.asarray(da.transpose(time_name,lat_name,lon_name).values,dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{spec.channel_name} 包含非有限值")
        arrays.append(arr)
    return (np.stack(arrays,axis=1), names, np.asarray(ds[time_name].values),
        np.asarray(ds[lat_name].values,dtype=np.float32), np.asarray(ds[lon_name].values,dtype=np.float32))


def _validate_year_splits(split_years:Mapping[str,Sequence[int]])->dict[str,set[int]]:
    return chronological_splits(split_years)


def _grid_spacing(latitude:np.ndarray,longitude:np.ndarray)->float:
    return regular_latlon_spacing(latitude,longitude)


def _training_stats(state,years,train_years,eps=1e-6):
    mask=np.isin(years,list(train_years))
    if not bool(mask.any()):
        raise ValueError("输入数据中没有 train years")
    train=state[mask].astype(np.float64)
    return train.mean(axis=(0,2,3)).astype(np.float32),np.maximum(train.std(axis=(0,2,3)),eps).astype(np.float32)


def _write_jsonl(path:Path,records:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")


@fresh_outputs('out_dir','manifest_dir')
def build_r7_era5_npz_from_dataset(
    ds, *, out_dir:str|Path, manifest_dir:str|Path,
    specs:Iterable[ERA5ChannelSpec]=DEFAULT_R7_ERA5_CHANNELS,
    split_years:Mapping[str,Sequence[int]], history_steps:int=2,
    history_interval_hours:int=6, lead_time_hours:int=6,
    sample_stride_hours:int=6, expected_grid_spacing_deg:float|None=0.25,
    source_label:str="ERA5 regional subset",
)->dict[str,Path]:
    """Small regional windows only. New outputs, strict splits, train-only stats."""
    import pandas as pd
    values=(history_steps,history_interval_hours,lead_time_hours,sample_stride_hours)
    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<1 for x in values):
        raise ValueError('history, lead and stride must be positive integers')
    split_sets=_validate_year_splits(split_years)
    specs=tuple(specs)
    state,names,times_raw,latitude,longitude=stack_era5_channels(ds,specs)
    times=utc_time_index(times_raw)
    spacing=_grid_spacing(latitude,longitude)
    if expected_grid_spacing_deg is not None and not np.isclose(spacing,float(expected_grid_spacing_deg),rtol=0,atol=1e-4):
        raise ValueError(f"网格分辨率 {spacing}° != expected {expected_grid_spacing_deg}°")
    years=np.asarray(times.year,dtype=np.int32)
    mean,std=_training_stats(state,years,split_sets["train"])
    normalized=((state-mean[None,:,None,None])/std[None,:,None,None]).astype(np.float32)
    index={int(ts.value):i for i,ts in enumerate(times)}
    out_dir,manifest_dir=Path(out_dir),Path(manifest_dir)
    out_dir.mkdir(parents=True,exist_ok=False)
    manifest_dir.mkdir(parents=True,exist_ok=False)
    records_by_split={key:[] for key in ("train","val","test")}
    for split,allowed_years in split_sets.items():
        split_out=out_dir/split
        split_out.mkdir()
        for init_time in times:
            if int(init_time.year) not in allowed_years or int(init_time.value) % (sample_stride_hours*3_600_000_000_000):
                continue
            history_times=[init_time-pd.Timedelta(hours=history_interval_hours*(history_steps-1-j)) for j in range(history_steps)]
            target_time=init_time+pd.Timedelta(hours=lead_time_hours)
            required_times=history_times+[target_time]
            if any(int(t.year) not in allowed_years for t in required_times):
                continue
            keys=[int(t.value) for t in required_times]
            if any(k not in index for k in keys):
                continue
            sid=f"era5_{split}_{init_time:%Y%m%d%H}_p{lead_time_hours:03d}h"
            npz_path=split_out/f"{sid}.npz"
            with npz_path.open('xb') as f:
                np.savez_compressed(f, coarse_history=normalized[[index[k] for k in keys[:-1]]],
                    atmos_target=normalized[index[keys[-1]]], lead_time_hours=np.asarray(lead_time_hours,dtype=np.float32),
                    latitude=latitude,longitude=longitude,grid_spacing_deg=np.asarray(spacing,dtype=np.float32))
            records_by_split[split].append({
                "sample_id":sid,"path":os.path.relpath(npz_path.resolve(),manifest_dir.resolve()),
                "split":split,"history_times":[t.isoformat() for t in history_times],
                "init_time":init_time.isoformat(),"target_time":target_time.isoformat(),"lead_time_hours":int(lead_time_hours)})
    for split,records in records_by_split.items():
        if not records:
            raise RuntimeError(f"{split} 没有可用窗口；检查年份、时间范围和 cadence")
        _write_jsonl(manifest_dir/f"{split}.jsonl",records)
    normalization={"channels":names,"mean":mean.tolist(),"std":std.tolist(),"computed_from_years":sorted(split_sets["train"])}
    with (manifest_dir/"normalization.json").open('x',encoding='utf-8') as f:
        json.dump(normalization,f,ensure_ascii=False,indent=2)
    provenance={
        'schema_version':1,"source":source_label,"native_grid_spacing_deg":spacing,
        "expected_grid_spacing_deg":expected_grid_spacing_deg,
        "channels":[asdict(spec)|{"channel_name":spec.channel_name} for spec in specs],
        "history_steps":int(history_steps),"history_interval_hours":int(history_interval_hours),
        "lead_time_hours":int(lead_time_hours),"sample_stride_hours":int(sample_stride_hours),
        "split_years":{key:sorted(value) for key,value in split_sets.items()},
        "samples_by_split":{key:len(value) for key,value in records_by_split.items()},
        "normalization":"training-years-only mean/std","target_semantics":"native ERA5 grid; no spatial upsampling"}
    with (manifest_dir/"provenance.json").open('x',encoding='utf-8') as f:
        json.dump(provenance,f,ensure_ascii=False,indent=2)
    return {split:manifest_dir/f"{split}.jsonl" for split in records_by_split}


def build_r7_era5_npz_from_path(source:str|Path,**kwargs)->dict[str,Path]:
    import xarray as xr
    source=Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    ds=xr.open_zarr(source,chunks=None) if source.is_dir() or source.suffix.lower()==".zarr" else xr.open_dataset(source)
    try:
        return build_r7_era5_npz_from_dataset(ds,source_label=str(source),**kwargs)
    finally:
        ds.close()
