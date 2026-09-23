from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


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
        raise ValueError(
            f"{da.name} 不包含唯一 pressure level={level_hpa} hPa; "
            f"available={values.tolist()}"
        )
    return values[int(idx[0])].item()


def stack_era5_channels(
    ds,
    specs:Iterable[ERA5ChannelSpec],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a regional ERA5 subset into [time, channel, lat, lon].

    This helper is intentionally for already-subset regional data. Multi-decade
    cloud-scale extraction should first cache/subset the source Zarr.
    """
    time_name=_coord_name(ds,("time","valid_time"))
    lat_name=_coord_name(ds,("latitude","lat"))
    lon_name=_coord_name(ds,("longitude","lon"))

    ds=ds.sortby(time_name)
    arrays=[]
    names=[]
    for spec in specs:
        if spec.variable not in ds:
            raise KeyError(f"ERA5 变量不存在: {spec.variable}")
        da=ds[spec.variable]

        if spec.level_hpa is not None:
            level_name=_coord_name(
                da,
                ("level","pressure_level","isobaricInhPa"),
            )
            selected=_exact_level(da,level_name,spec.level_hpa)
            da=da.sel({level_name:selected},drop=True)

        required={time_name,lat_name,lon_name}
        if not required.issubset(set(da.dims)):
            raise ValueError(
                f"{spec.channel_name} dims={da.dims} 缺少 {required}"
            )

        # Preserve required singleton dimensions (especially one-timestamp
        # streaming chunks). Only auxiliary singleton dimensions may be dropped.
        extra=[dim for dim in da.dims if dim not in required]
        for dim in extra:
            if int(da.sizes[dim])!=1:
                raise ValueError(
                    f"{spec.channel_name} 存在未处理非单例维度 "
                    f"{dim}={int(da.sizes[dim])}"
                )
            da=da.isel({dim:0},drop=True)

        da=da.transpose(time_name,lat_name,lon_name)
        arr=np.asarray(da.values,dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{spec.channel_name} 包含非有限值")
        arrays.append(arr)
        names.append(spec.channel_name)

    if not arrays:
        raise ValueError("至少需要一个 ERA5 channel spec")
    state=np.stack(arrays,axis=1)
    times=np.asarray(ds[time_name].values)
    latitude=np.asarray(ds[lat_name].values,dtype=np.float32)
    longitude=np.asarray(ds[lon_name].values,dtype=np.float32)
    return state,names,times,latitude,longitude


def _validate_year_splits(
    split_years:Mapping[str,Sequence[int]],
)->dict[str,set[int]]:
    required={"train","val","test"}
    if set(split_years)!=required:
        raise ValueError(
            f"split_years 必须且只能包含 {sorted(required)}"
        )
    normalized={
        key:{int(y) for y in years}
        for key,years in split_years.items()
    }
    if any(not years for years in normalized.values()):
        raise ValueError("train/val/test years 均不能为空")
    if normalized["train"] & normalized["val"]:
        raise ValueError("train/val years 重叠")
    if normalized["train"] & normalized["test"]:
        raise ValueError("train/test years 重叠")
    if normalized["val"] & normalized["test"]:
        raise ValueError("val/test years 重叠")
    return normalized


def _grid_spacing(latitude:np.ndarray,longitude:np.ndarray)->float:
    if latitude.size<2 or longitude.size<2:
        raise ValueError("latitude/longitude 至少各需要 2 个点")

    lat_diff=np.abs(np.diff(latitude.astype(float)))
    lon_diff=np.abs(np.diff(longitude.astype(float)))
    if np.any(lat_diff<=0) or np.any(lon_diff<=0):
        raise ValueError("latitude/longitude 存在重复坐标")
    if not np.allclose(lat_diff,lat_diff[0],rtol=0,atol=1e-5):
        raise ValueError(
            f"latitude 非规则网格: diffs={lat_diff.tolist()}"
        )
    if not np.allclose(lon_diff,lon_diff[0],rtol=0,atol=1e-5):
        raise ValueError(
            f"longitude 非规则网格: diffs={lon_diff.tolist()}"
        )

    dlat=float(lat_diff[0])
    dlon=float(lon_diff[0])
    if not np.isclose(dlat,dlon,rtol=0,atol=1e-5):
        raise ValueError(f"非等经纬网格: dlat={dlat}, dlon={dlon}")
    return (dlat+dlon)/2.0


def _training_stats(
    state:np.ndarray,
    years:np.ndarray,
    train_years:set[int],
    eps:float=1e-6,
)->tuple[np.ndarray,np.ndarray]:
    mask=np.isin(years,list(train_years))
    if not bool(mask.any()):
        raise ValueError("输入数据中没有 train years")
    train=state[mask].astype(np.float64)
    mean=train.mean(axis=(0,2,3))
    std=train.std(axis=(0,2,3))
    std=np.maximum(std,eps)
    return mean.astype(np.float32),std.astype(np.float32)


def _write_jsonl(path:Path,records:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")


def build_r7_era5_npz_from_dataset(
    ds,
    *,
    out_dir:str|Path,
    manifest_dir:str|Path,
    specs:Iterable[ERA5ChannelSpec]=DEFAULT_R7_ERA5_CHANNELS,
    split_years:Mapping[str,Sequence[int]],
    history_steps:int=2,
    history_interval_hours:int=6,
    lead_time_hours:int=6,
    sample_stride_hours:int=6,
    expected_grid_spacing_deg:float|None=0.25,
    source_label:str="ERA5 regional subset",
)->dict[str,Path]:
    """Build a leakage-safe R7 NPZ fixture/dataset from a regional xarray Dataset.

    Splits are assigned by raw timestamp year before any forecast windows are
    constructed. Normalization statistics are computed from train years only.

    NPZ is intended for small/medium cached regional experiments and smoke tests.
    A chunked Zarr backend remains the preferred full-production storage path.
    """
    import pandas as pd

    if history_steps<1:
        raise ValueError("history_steps 必须 >= 1")
    if history_interval_hours<=0 or lead_time_hours<=0:
        raise ValueError("history/lead hours 必须为正数")
    if sample_stride_hours<=0:
        raise ValueError("sample_stride_hours 必须为正数")

    split_sets=_validate_year_splits(split_years)
    specs=tuple(specs)
    state,names,times_raw,latitude,longitude=stack_era5_channels(
        ds,specs
    )
    times=pd.DatetimeIndex(times_raw)
    if not times.is_monotonic_increasing:
        raise ValueError("ERA5 time 必须单调递增")
    if times.has_duplicates:
        raise ValueError("ERA5 time 存在重复")

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
    mean,std=_training_stats(state,years,split_sets["train"])
    normalized=(state-mean[None,:,None,None])/std[None,:,None,None]
    normalized=normalized.astype(np.float32)

    index={int(ts.value):i for i,ts in enumerate(times)}
    out_dir=Path(out_dir)
    manifest_dir=Path(manifest_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    manifest_dir.mkdir(parents=True,exist_ok=True)

    records_by_split={key:[] for key in ("train","val","test")}
    for split,allowed_years in split_sets.items():
        split_out=out_dir/split
        split_out.mkdir(parents=True,exist_ok=True)
        for init_idx,init_time in enumerate(times):
            if int(init_time.year) not in allowed_years:
                continue
            if (
                init_time.minute!=0
                or init_time.second!=0
                or init_time.hour % sample_stride_hours != 0
            ):
                continue

            history_times=[
                init_time-pd.Timedelta(
                    hours=history_interval_hours*(history_steps-1-j)
                )
                for j in range(history_steps)
            ]
            target_time=init_time+pd.Timedelta(hours=lead_time_hours)
            required_times=history_times+[target_time]

            if any(int(t.year) not in allowed_years for t in required_times):
                continue
            keys=[int(t.value) for t in required_times]
            if any(k not in index for k in keys):
                continue

            history_indices=[index[k] for k in keys[:-1]]
            target_index=index[keys[-1]]
            sid=(
                f"era5_{split}_{init_time:%Y%m%d%H}_"
                f"p{lead_time_hours:03d}h"
            )
            npz_path=split_out/f"{sid}.npz"
            np.savez_compressed(
                npz_path,
                coarse_history=normalized[history_indices],
                atmos_target=normalized[target_index],
                lead_time_hours=np.asarray(
                    lead_time_hours,dtype=np.float32
                ),
                latitude=latitude,
                longitude=longitude,
                grid_spacing_deg=np.asarray(spacing,dtype=np.float32),
            )
            records_by_split[split].append({
                "sample_id":sid,
                "path":os.path.relpath(
                    npz_path.resolve(),manifest_dir.resolve()
                ),
                "split":split,
                "history_times":[t.isoformat() for t in history_times],
                "init_time":init_time.isoformat(),
                "target_time":target_time.isoformat(),
                "lead_time_hours":int(lead_time_hours),
            })

    for split,records in records_by_split.items():
        if not records:
            raise RuntimeError(
                f"{split} 没有可用窗口；检查年份、时间范围和 cadence"
            )
        _write_jsonl(manifest_dir/f"{split}.jsonl",records)

    normalization={
        "channels":names,
        "mean":mean.tolist(),
        "std":std.tolist(),
        "computed_from_years":sorted(split_sets["train"]),
    }
    (manifest_dir/"normalization.json").write_text(
        json.dumps(normalization,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )
    provenance={
        "source":source_label,
        "native_grid_spacing_deg":spacing,
        "expected_grid_spacing_deg":expected_grid_spacing_deg,
        "channels":[asdict(spec) | {"channel_name":spec.channel_name} for spec in specs],
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
        "normalization":"training-years-only mean/std",
        "target_semantics":"native ERA5 grid; no spatial upsampling",
    }
    (manifest_dir/"provenance.json").write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )
    return {
        split:manifest_dir/f"{split}.jsonl"
        for split in ("train","val","test")
    }


def build_r7_era5_npz_from_path(
    source:str|Path,
    **kwargs,
)->dict[str,Path]:
    """Open a local NetCDF/Zarr regional ERA5 subset and build R7 samples."""
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
        return build_r7_era5_npz_from_dataset(
            ds,
            source_label=str(source),
            **kwargs,
        )
    finally:
        ds.close()
