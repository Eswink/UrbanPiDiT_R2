from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
from .contracts import fresh_outputs, RunningMoments, utc_time_index
from .process_diagnostics import PROCESS_DIAGNOSTIC_NAMES, compute_process_diagnostic_vector
from .r7_era5 import DEFAULT_R7_ERA5_CHANNELS, ERA5ChannelSpec, _coord_name, _grid_spacing, _validate_year_splits, stack_era5_channels


def _write_jsonl(path, records):
    with Path(path).open('x', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')


def _window_records(times, *, split_sets, store_path, manifest_dir,
                    history_steps, history_interval_hours, lead_time_hours, sample_stride_hours):
    import pandas as pd
    index = {int(ts.value): i for i, ts in enumerate(times)}
    records = {key: [] for key in ('train', 'val', 'test')}
    for split, years in split_sets.items():
        for init in times:
            if init.year not in years or int(init.value) % (sample_stride_hours*3_600_000_000_000):
                continue
            history = [init-pd.Timedelta(hours=history_interval_hours*(history_steps-1-j)) for j in range(history_steps)]
            target = init+pd.Timedelta(hours=lead_time_hours)
            required = history+[target]
            if any(t.year not in years or int(t.value) not in index for t in required):
                continue
            records[split].append({
                'sample_id': f'era5z_{split}_{init:%Y%m%d%H}_p{lead_time_hours:03d}h',
                'store_path': os.path.relpath(store_path.resolve(), manifest_dir.resolve()),
                'split': split, 'history_indices': [index[int(t.value)] for t in history],
                'target_index': index[int(target.value)],
                'history_times': [t.isoformat() for t in history],
                'init_time': init.isoformat(), 'target_time': target.isoformat(),
                'lead_time_hours': lead_time_hours,
            })
    return records


@fresh_outputs('store_path', 'manifest_dir')
def build_r7_era5_zarr_from_dataset(
    ds, *, store_path: str|Path, manifest_dir: str|Path,
    specs: Iterable[ERA5ChannelSpec] = DEFAULT_R7_ERA5_CHANNELS,
    split_years: Mapping[str, Sequence[int]], history_steps: int = 2,
    history_interval_hours: int = 6, lead_time_hours: int = 6,
    sample_stride_hours: int = 6, expected_grid_spacing_deg: float|None = .25,
    source_label: str = 'ERA5 regional subset', time_chunk: int = 64,
    spatial_chunk: tuple[int,int] = (64,64), compute_process_targets: bool = False,
) -> dict[str, Path]:
    """Publish a new local, physical-unit store with train-only centered moments.

    Failure leaves an inspectable incomplete output. Reuse/overwrite is refused;
    choose a new destination to retry. This does not authenticate source_label.
    """
    import zarr
    ints = (history_steps, history_interval_hours, lead_time_hours, sample_stride_hours, time_chunk, *spatial_chunk)
    if len(spatial_chunk) != 2 or any(isinstance(x, bool) or not isinstance(x, (int,np.integer)) or x < 1 for x in ints):
        raise ValueError('steps, cadence and chunk sizes must be positive integers')
    splits = _validate_year_splits(split_years)
    specs = tuple(specs)
    time_name = _coord_name(ds, ('time','valid_time'))
    lat_name, lon_name = _coord_name(ds, ('latitude','lat')), _coord_name(ds, ('longitude','lon'))
    ds = ds.sortby(time_name)
    times = utc_time_index(ds[time_name].values)
    lat, lon = np.asarray(ds[lat_name].values,dtype=np.float32), np.asarray(ds[lon_name].values,dtype=np.float32)
    spacing = _grid_spacing(lat, lon)
    if expected_grid_spacing_deg is not None and not np.isclose(spacing, expected_grid_spacing_deg, rtol=0, atol=1e-4):
        raise ValueError('native grid spacing differs from expected resolution')
    first, names, _, _, _ = stack_era5_channels(ds.isel({time_name:slice(0,1)}), specs)
    if len(set(names)) != len(names):
        raise ValueError('duplicate channel names')
    if compute_process_targets:
        compute_process_diagnostic_vector(first[0], names, lat, lon)
    del first
    years = np.asarray(times.year)
    train_mask = np.isin(years, list(splits['train']))
    if not train_mask.any():
        raise ValueError('no training observations')
    store_path, manifest_dir = Path(store_path), Path(manifest_dir)
    records = _window_records(times, split_sets=splits, store_path=store_path,
        manifest_dir=manifest_dir, history_steps=history_steps,
        history_interval_hours=history_interval_hours, lead_time_hours=lead_time_hours,
        sample_stride_hours=sample_stride_hours)
    if any(not rows for rows in records.values()):
        raise ValueError('every split needs at least one complete exact-time window')
    T,C,H,W = len(times),len(names),len(lat),len(lon)
    manifest_dir.mkdir(parents=True, exist_ok=False)
    root = zarr.open_group(str(store_path), mode='w-')
    root.attrs.update({'schema_version':1, 'build_complete':False, 'time_unit':'ns'})
    chunks = (min(time_chunk,T), C, min(spatial_chunk[0],H), min(spatial_chunk[1],W))
    state = root.create_array('state', shape=(T,C,H,W), chunks=chunks, dtype='f4')
    root.create_array('latitude', data=lat, chunks=(H,))
    root.create_array('longitude', data=lon, chunks=(W,))
    root.create_array('time_ns', data=times.asi8, chunks=(min(time_chunk,T),))
    moments = RunningMoments(C)
    process_array = None
    process_moments = RunningMoments(len(PROCESS_DIAGNOSTIC_NAMES))
    if compute_process_targets:
        process_array = root.create_array('process_diagnostics_raw', shape=(T,len(PROCESS_DIAGNOSTIC_NAMES)),
            chunks=(min(time_chunk,T),len(PROCESS_DIAGNOSTIC_NAMES)), dtype='f4')
    for start in range(0,T,time_chunk):
        stop = min(T,start+time_chunk)
        block, block_names, block_times, block_lat, block_lon = stack_era5_channels(ds.isel({time_name:slice(start,stop)}), specs)
        if block_names != names or not np.array_equal(block_lat,lat) or not np.array_equal(block_lon,lon):
            raise ValueError('channel/coordinate schema changed between chunks')
        if not np.array_equal(utc_time_index(block_times).asi8, times[start:stop].asi8):
            raise ValueError('time coordinate changed between chunks')
        state[start:stop] = block
        selected = block[train_mask[start:stop]]
        moments.update(selected.transpose(0,2,3,1).reshape(-1,C))
        if process_array is not None:
            processes = np.stack([compute_process_diagnostic_vector(frame,names,lat,lon) for frame in block]).astype(np.float32)
            process_array[start:stop] = processes
            process_moments.update(processes[train_mask[start:stop]])
    mean, std = moments.finish()
    root.create_array('normalization_mean', data=mean, chunks=(C,))
    root.create_array('normalization_std', data=std, chunks=(C,))
    if process_array is not None:
        pmean, pstd = process_moments.finish()
        root.create_array('process_normalization_mean', data=pmean, chunks=(len(pmean),))
        root.create_array('process_normalization_std', data=pstd, chunks=(len(pstd),))
    common = {
        'schema_version':1, 'time_unit':'ns', 'source':str(source_label),
        'channels':names, 'units':[str(ds[s.variable].attrs.get('units','unknown')) for s in specs],
        'native_grid_spacing_deg':float(spacing), 'physical_units_retained':True,
        'spatial_resampling':False, 'normalization_years':sorted(splits['train']),
        'split_years':{k:sorted(v) for k,v in splits.items()},
        'normalization':'training-years-only centered population mean/std; applied at read time',
        'process_diagnostics_enabled':bool(compute_process_targets),
        'process_diagnostic_names':list(PROCESS_DIAGNOSTIC_NAMES) if compute_process_targets else [],
        'process_target_time_semantics':'input init time only',
    }
    root.attrs.update(common)
    root.attrs['target_semantics'] = 'native ERA5 grid; physical units retained; no spatial upsampling'
    for split, rows in records.items():
        _write_jsonl(manifest_dir/f'{split}.jsonl', rows)
    metadata = dict(common, store_path=os.path.relpath(store_path.resolve(),manifest_dir.resolve()),
        shape=[T,C,H,W], chunks=list(chunks), history_steps=history_steps,
        history_interval_hours=history_interval_hours, lead_time_hours=lead_time_hours,
        sample_stride_hours=sample_stride_hours, samples_by_split={k:len(v) for k,v in records.items()})
    with (manifest_dir/'zarr_metadata.json').open('x',encoding='utf-8') as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)
    root.attrs['build_complete'] = True
    return {split:manifest_dir/f'{split}.jsonl' for split in records}


def build_r7_era5_zarr_from_path(source: str|Path, **kwargs):
    import xarray as xr
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    ds = xr.open_zarr(source,chunks=None) if source.is_dir() or source.suffix.lower()=='.zarr' else xr.open_dataset(source)
    try:
        return build_r7_era5_zarr_from_dataset(ds,source_label=str(source),**kwargs)
    finally:
        ds.close()
