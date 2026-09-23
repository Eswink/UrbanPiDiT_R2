"""Production-facing metadata/unit audit; defaults to read-only local preflight."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .contracts import chronological_splits,utc_time_index
from .grid import regular_latlon_spacing
from .r7_era5 import ERA5ChannelSpec,DEFAULT_R7_ERA5_CHANNELS,_coord_name,stack_era5_channels
from .r7_era5_zarr import _window_records,build_r7_era5_zarr_from_dataset


def _unit(value):
    return str(value).strip().lower().replace('**','^').replace(' ','').replace('^','')


SI_UNITS={
    'K':{'k','kelvin'}, 'Pa':{'pa','pascal','pascals'},
    'm s-1':{'ms-1','m/s'}, 'kg kg-1':{'kgkg-1','kg/kg','1','dimensionless'},
    'm2 s-2':{'m2s-2','m2/s2'},
}


def canonical_unit(channel):
    if channel in ('t2m','d2m') or channel.startswith('t') and channel[1:].isdigit():
        return 'K'
    if channel in ('mslp','sp'):
        return 'Pa'
    if channel[:1] in ('u','v') and channel[1:].isdigit():
        return 'm s-1'
    if channel[:1]=='q' and channel[1:].isdigit():
        return 'kg kg-1'
    if channel[:1]=='z' and channel[1:].isdigit():
        return 'm2 s-2'
    raise ValueError(f'no audited unit definition for {channel}; use an explicit supported channel name')


@contextmanager
def local_source(path):
    import xarray as xr
    if '://' in str(path):
        raise ValueError('local source only; remote URLs are not opened')
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ds=xr.open_zarr(path,chunks=None) if path.is_dir() else xr.open_dataset(path,engine='h5netcdf')
    try:
        yield ds
    finally:
        ds.close()


def inspect_era5(ds,config):
    """Read coordinate metadata plus one selected frame; no output publication."""
    specs=tuple(ERA5ChannelSpec(**s) for s in config['channels']) if 'channels' in config else DEFAULT_R7_ERA5_CHANNELS
    names=[s.channel_name for s in specs]
    if not names or len(set(names))!=len(names):
        raise ValueError('unique nonempty channel definitions required')
    splits=chronological_splits(config['split_years'])
    tn=_coord_name(ds,('time','valid_time'))
    ln=_coord_name(ds,('latitude','lat'))
    xn=_coord_name(ds,('longitude','lon'))
    times=utc_time_index(ds[tn].values)
    latitude,longitude=np.asarray(ds[ln].values),np.asarray(ds[xn].values)
    spacing=regular_latlon_spacing(latitude,longitude)
    if not np.isclose(spacing,.25,rtol=0,atol=1e-5):
        raise ValueError('production R7 preflight currently requires 0.25-degree input spacing')
    channels=[]
    for spec in specs:
        da=ds[spec.variable]
        expected=canonical_unit(spec.channel_name)
        observed=da.attrs.get('units','')
        if _unit(observed) not in SI_UNITS[expected]:
            raise ValueError(f'{spec.channel_name} units {observed!r} != required {expected}; no guessed conversion')
        if spec.level_hpa is not None:
            pn=_coord_name(da,('level','pressure_level','isobaricInhPa'))
            if _unit(da[pn].attrs.get('units','')) not in {'hpa','millibar','millibars','mbar'}:
                raise ValueError('pressure levels require explicit hPa/millibar units')
        channels.append({'variable':spec.variable,'name':spec.channel_name,'level_hpa':spec.level_hpa,
            'observed_units':str(observed),'canonical_units':expected})
    # Uses the same exact level/dimension/finite guard as the actual builder.
    stack_era5_channels(ds.isel({tn:slice(0,1)}),specs)
    options={name:config.get(name,default) for name,default in (
        ('history_steps',2),('history_interval_hours',6),('lead_time_hours',6),('sample_stride_hours',6))}
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in options.values()):
        raise ValueError('positive integer cadence parameters required')
    records=_window_records(times,split_sets=splits,store_path=Path('unwritten.zarr'),
        manifest_dir=Path('unwritten-manifest'),**options)
    counts={k:len(v) for k,v in records.items()}
    if not all(counts.values()):
        raise ValueError('every split needs complete exact-time windows')
    raw_bytes=len(times)*len(specs)*len(latitude)*len(longitude)*4
    report={'schema_version':1,'mode':'read-only-preflight','scientific_training_certified':False,
        'shape':[len(times),len(specs),len(latitude),len(longitude)],'grid_spacing_deg':float(spacing),
        'channels':channels,'split_years':{k:sorted(v) for k,v in splits.items()},'windows':counts,
        'raw_state_bytes':int(raw_bytes),'raw_state_GiB':raw_bytes/2**30,'time_first':times[0].isoformat(),
        'time_last':times[-1].isoformat(),'cadence':options,
        'limitations':['one frame checked for values; full finite checks happen while writing',
            'raw state estimate excludes metadata/compression overhead and is not a disk-space guarantee',
            'unit metadata checks do not authenticate source observations']}
    return report,specs


def source_fingerprint(path,max_hash_bytes=64*2**20):
    """Bounded fingerprint. Never call a partial fingerprint a whole-source hash."""
    path=Path(path)
    if path.is_file():
        size=path.stat().st_size
        if size<=max_hash_bytes:
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1<<20),b''):
                    digest.update(chunk)
            return {'scope':'full-local-file','sha256':digest.hexdigest(),'bytes':size}
        return {'scope':'file-stat-only','bytes':size,'mtime_ns':path.stat().st_mtime_ns,'sha256':None}
    metadata=[p for p in [path/'zarr.json',path/'.zgroup',path/'.zattrs',path/'.zmetadata'] if p.is_file()]
    digest=hashlib.sha256()
    for p in sorted(metadata):
        if p.stat().st_size>max_hash_bytes:
            raise ValueError('root metadata exceeds bounded hash budget')
        digest.update(p.name.encode()+b'\0'+p.read_bytes())
    return {'scope':'zarr-root-metadata-only','sha256':digest.hexdigest(),'files':[p.name for p in metadata]}


def prepare_local(source,config,*,write=False,store_path=None,manifest_dir=None,max_raw_gib=None):
    with local_source(source) as ds:
        report,specs=inspect_era5(ds,config)
        report['source_path']=str(Path(source).resolve())
        report['fingerprint']=source_fingerprint(source)
        if not write:
            return report
        if store_path is None or manifest_dir is None or max_raw_gib is None:
            raise ValueError('write requires explicit fresh store/manifest paths and raw-GiB cap')
        if not math.isfinite(max_raw_gib) or max_raw_gib<=0 or report['raw_state_GiB']>max_raw_gib:
            raise ValueError('raw output estimate exceeds explicit cap')
        if config.get('compute_process_targets',False):
            from .process_diagnostics import compute_process_diagnostic_vector
            tn=_coord_name(ds,('time','valid_time'))
            state,names,_,lat,lon=stack_era5_channels(ds.isel({tn:slice(0,1)}),specs)
            compute_process_diagnostic_vector(state[0],names,lat,lon)
        paths=build_r7_era5_zarr_from_dataset(ds,store_path=store_path,manifest_dir=manifest_dir,
            specs=specs,split_years=config['split_years'],**report['cadence'],
            compute_process_targets=bool(config.get('compute_process_targets',False)),
            time_chunk=config.get('time_chunk',16),source_label=str(Path(source).resolve()))
        report['mode']='written-local-cache'
        report['manifests']={k:str(v) for k,v in paths.items()}
        with (Path(manifest_dir)/'source_preflight.json').open('x',encoding='utf-8') as f:
            json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
        return report
