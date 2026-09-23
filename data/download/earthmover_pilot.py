"""Pinned anonymous ERA5 tile extraction with decoded-chunk (not HTTP) limits."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from numbers import Integral, Real
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import numpy as np

SNAPSHOT = 'ZFKDHBCTBVHVXM3BQFV0'
SOURCE = 's3://earthmover-icechunk-era5/icechunkV2'
FIELDS = (('t2m', None, 't2m'), ('u10', None, 'u10'), ('v10', None, 'v10'),
          ('msl', None, 'mslp'), ('t', 850, 't850'), ('t', 500, 't500'),
          ('q', 850, 'q850'), ('u', 850, 'u850'), ('u', 500, 'u500'),
          ('v', 850, 'v850'), ('v', 500, 'v500'))


@dataclass
class DecodedBudget:
    limit: int = 192 * 2**20
    used: int = 0
    reads: int = 0
    seconds: float = 600.
    started: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        for key, minimum in (('limit', 1), ('used', 0), ('reads', 0)):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, Integral) or v < minimum:
                raise ValueError(f'invalid integer budget {key}')
        if self.used > self.limit:
            raise ValueError('budget already exhausted')
        for key in ('seconds', 'started'):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v):
                raise ValueError(f'invalid finite budget {key}')
        if self.seconds <= 0:
            raise ValueError('deadline duration must be positive')

    def check(self, n):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError('byte charge must be a nonnegative integer')
        if self.used + n > self.limit:
            raise RuntimeError('decoded-chunk budget exceeded BEFORE field read')
        if time.monotonic() - self.started > self.seconds:
            raise RuntimeError('extraction deadline exceeded')

    def charge(self, n):
        self.check(n)
        self.used += n
        self.reads += 1


def declared_missing_values(array):
    """CF declarations only, never generic Zarr allocation fill values.

    Xarray's Zarr3 floating _FillValue encoding is base64 of one little-endian
    double (FillValueCoder). The pinned source uses this encoding for NaN.
    Numeric missing_value lists remain numeric; arbitrary strings are refused.
    """
    values = []
    attrs = dict(getattr(array, 'attrs', {}))
    for key in ('_FillValue', 'missing_value'):
        if key not in attrs:
            continue
        value = attrs[key]
        if key == '_FillValue' and isinstance(value, str) and np.dtype(array.dtype).kind == 'f':
            try:
                encoded = base64.b64decode(value, validate=True)
                if len(encoded) != 8:
                    raise ValueError('expected eight-byte float fill encoding')
                value = struct.unpack('<d', encoded)[0]
            except (ValueError, struct.error) as exc:
                raise ValueError('invalid encoded floating CF _FillValue') from exc
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or raw.ndim > 1 or not raw.size:
            raise ValueError(f'invalid numeric CF {key}')
        values.extend(raw.ravel().tolist())
    return values


def selection_plan(array, indices):
    """Only unsharded numeric arrays and explicit unique integer axis indices."""
    if getattr(array, 'shards', None) is not None:
        raise ValueError('sharded source requires a different audited read plan')
    shape, chunks = tuple(array.shape), tuple(array.chunks)
    if len(indices) != len(shape) or len(chunks) != len(shape) or any(
        isinstance(c, bool) or not isinstance(c, Integral) or c < 1 for c in shape + chunks
    ):
        raise ValueError('selection/chunk dimensionality mismatch')
    dtype = np.dtype(array.dtype)
    if dtype.kind not in 'iuf':
        raise ValueError('numeric source dtype required')
    declared_missing_values(array)
    idx = tuple(np.asarray(i) for i in indices)
    for i, size in zip(idx, shape):
        if i.ndim != 1 or not len(i) or i.dtype.kind not in 'iu' or len(np.unique(i)) != len(i):
            raise ValueError('indices must be unique nonempty integer vectors')
        if np.any(i < 0) or np.any(i >= size):
            raise ValueError('selection index out of bounds')
    out_bytes = math.prod(map(len, idx)) * dtype.itemsize
    if out_bytes > 8 * 2**20:
        raise ValueError('individual selected output exceeds 8 MiB')
    chunk_bytes = math.prod(chunks) * dtype.itemsize
    if chunk_bytes > 8 * 2**20:
        raise ValueError('single decoded source chunk exceeds 8 MiB')
    coords = tuple(product(*(np.unique(i // c).tolist() for i, c in zip(idx, chunks))))
    return idx, coords, chunk_bytes


def bounded_selection(array, indices, budget):
    """Read each touched chunk once, retain only its explicitly requested cells."""
    idx, coords, chunk_bytes = selection_plan(array, indices)
    missing = declared_missing_values(array)
    budget.check(len(coords) * chunk_bytes)
    output = np.empty(tuple(map(len, idx)), dtype=array.dtype)
    for coord in coords:
        starts = [k * c for k, c in zip(coord, array.chunks)]
        slices = tuple(slice(s, min(s + c, n)) for s, c, n in zip(starts, array.chunks, array.shape))
        budget.charge(chunk_bytes)
        chunk = np.asarray(array[slices])
        if chunk.shape != tuple(sl.stop - sl.start for sl in slices):
            raise ValueError('source chunk read returned an unexpected shape')
        positions = [np.flatnonzero((i >= sl.start) & (i < sl.stop)) for i, sl in zip(idx, slices)]
        local = [i[p] - s for i, p, s in zip(idx, positions, starts)]
        chosen = chunk[np.ix_(*local)]
        if not np.isfinite(chosen).all() or np.any(np.abs(chosen.astype(np.float64)) > 1e30):
            raise ValueError('source missing/nonfinite values; no synthetic fallback')
        if any(np.any(chosen == sentinel) for sentinel in missing):
            raise ValueError('source contains a declared CF missing value; no replacement')
        output[np.ix_(*positions)] = chosen
    return output


def extract_dataset(root, *, snapshot_id=SNAPSHOT, sampling='january'):
    """One exact provider tile and a fixed bounded calendar sampling profile."""
    import pandas as pd
    import xarray as xr
    from data.preprocess.grid import regular_latlon_spacing
    from .seasonal_sampling import requested_times, PROFILES
    wanted = requested_times(sampling)
    groups = {k: root[k + '/temporal'] for k in ('single', 'pressure')}
    budget = DecodedBudget()
    coordinates = {}
    for kind, group in groups.items():
        coordinates[kind] = {n: bounded_selection(group[n], (np.arange(group[n].shape[0]),), budget)
                             for n in ('latitude', 'longitude', 'valid_time')}
    for name in coordinates['single']:
        if not np.array_equal(coordinates['single'][name], coordinates['pressure'][name]):
            raise ValueError('surface/pressure coordinate mismatch')
    sc = coordinates['single']
    spacing = regular_latlon_spacing(sc['latitude'], sc['longitude'])
    if not np.isclose(spacing, .25, rtol=0, atol=1e-5):
        raise ValueError('source grid must have native 0.25-degree spacing')
    time_attrs = dict(groups['single']['valid_time'].attrs)
    pressure_time_attrs = dict(groups['pressure']['valid_time'].attrs)
    for key in ('units', 'calendar'):
        if time_attrs.get(key) != pressure_time_attrs.get(key):
            raise ValueError('surface/pressure time encoding mismatch')
    times = pd.DatetimeIndex(xr.coding.times.decode_cf_datetime(sc['valid_time'],
        time_attrs['units'], time_attrs.get('calendar', 'standard'), use_cftime=False))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('source times must be unique, increasing and valid')
    ti = times.get_indexer(wanted)
    if (ti < 0).any():
        raise ValueError('exact requested timestamps absent')
    yi = np.flatnonzero((sc['latitude'] >= 39.25) & (sc['latitude'] <= 42.))
    xi = np.flatnonzero((sc['longitude'] >= 114.) & (sc['longitude'] <= 116.75))
    if (len(yi), len(xi)) != (12, 12):
        raise ValueError('expected exact provider 0.25-degree 12x12 tile')
    if not (np.allclose(np.sort(sc['latitude'][yi]), np.arange(39.25,42.01,.25), rtol=0, atol=1e-5)
            and np.allclose(np.sort(sc['longitude'][xi]), np.arange(114.,116.76,.25), rtol=0, atol=1e-5)):
        raise ValueError('selected coordinates do not match the declared tile')
    level_array = groups['pressure']['pressure_level']
    if str(level_array.attrs.get('units')) != 'hPa':
        raise ValueError('explicit hPa pressure coordinate required')
    levels = bounded_selection(level_array, (np.arange(level_array.shape[0]),), budget)
    plans = []
    for variable, level, name in FIELDS:
        group = groups['single' if level is None else 'pressure']
        array = group[variable]
        expected = ('valid_time', 'latitude', 'longitude') if level is None else ('valid_time', 'pressure_level', 'latitude', 'longitude')
        if tuple(array.metadata.dimension_names) != expected:
            raise ValueError(f'unexpected dimensions for {name}')
        lengths = {n: len(v) for n, v in sc.items()}
        lengths['pressure_level'] = len(levels)
        if tuple(array.shape) != tuple(lengths[n] for n in expected):
            raise ValueError(f'field/coordinate shape mismatch for {name}')
        if np.dtype(array.dtype) != np.dtype('float32'):
            raise ValueError('pilot preserves float32 source payloads without guessed conversion')
        attrs = dict(array.attrs)
        if 'scale_factor' in attrs or 'add_offset' in attrs:
            raise ValueError('packed data require explicit decoding audit')
        from data.preprocess.r7_preflight import canonical_unit, SI_UNITS, _unit
        if _unit(attrs.get('units', '')) not in SI_UNITS[canonical_unit(name)]:
            raise ValueError(f'unit mismatch for {name}')
        if level is None:
            indices = (ti, yi, xi)
        else:
            li = np.flatnonzero(levels == level)
            if len(li) != 1:
                raise ValueError(f'exact unique {level} hPa absent')
            indices = (ti, li, yi, xi)
        if sampling == 'continuous-250d':
            reference = times.get_indexer(requested_times('four-season'))
            if (reference < 0).any() or set(ti // array.chunks[0]) != set(reference // array.chunks[0]):
                raise ValueError('continuous profile would change the approved seasonal time-chunk set')
        _, touched, each = selection_plan(array, indices)
        plans.append((array, indices, variable, level, name, attrs, len(touched) * each))
    budget.check(sum(p[-1] for p in plans))
    variables, receipts = {}, []
    for array, indices, variable, level, name, attrs, decoded in plans:
        values = bounded_selection(array, indices, budget)
        if level is not None:
            values = values[:, 0]
        variables[name] = (('time', 'latitude', 'longitude'), values,
                           {'units': attrs['units'], 'source_variable': variable,
                            'source_pressure_hpa': -1 if level is None else level})
        receipts.append({'name': name, 'source_variable': variable, 'pressure_hpa': level,
                         'shape': list(values.shape), 'source_chunks': list(array.chunks),
                         'decoded_chunk_bytes': decoded, 'units': attrs['units'],
                         'payload_sha256': hashlib.sha256(values.astype('<f4').tobytes()).hexdigest(),
                         'min': float(values.min()), 'max': float(values.max())})
        print(json.dumps({'extracted': name, 'shape': list(values.shape), 'decoded_budget_used': budget.used}), flush=True)
    ds = xr.Dataset(variables, coords={'time': wanted.values, 'latitude': sc['latitude'][yi],
                                     'longitude': sc['longitude'][xi]},
        attrs={'source': SOURCE, 'snapshot_id': snapshot_id, 'license': 'CC-BY-4.0',
               'attribution': 'Copernicus C3S/ECMWF ERA5; NSF NCAR historical archive; Earthmover Icechunk edition',
               'processing': 'exact time, pressure and grid subset only; no interpolation',
               'scientific_training_ready': 'false: bounded CPU integration pilot'})
    report = {'source': SOURCE, 'snapshot_id': snapshot_id, 'field_values_loaded': True,
              'source_is_real_reanalysis': True, 'scientific_claim': False,
              'grid_spacing_deg': .25, 'years': [2018, 2019, 2020], 'times_per_year': len(wanted) // 3,
              'sampling_profile': sampling, 'months': list(PROFILES[sampling]),
              'selected_times_utc': [t.isoformat() for t in wanted],
              'coordinates': {'latitude': ds.latitude.values.tolist(), 'longitude': ds.longitude.values.tolist()},
              'decoded_budget_bytes': budget.limit, 'decoded_charged_bytes': budget.used,
              'decoded_chunk_reads': budget.reads, 'variables': receipts,
              'budget_scope': 'sum of uncompressed touched chunk sizes; NOT measured HTTP transfer or peak RAM',
              'network_body_bytes': None, 'elapsed_seconds': time.monotonic() - budget.started,
              'license': 'CC-BY-4.0', 'catalog': 'https://registry.opendata.aws/earthmover-era5/',
              'source_dois': ['10.24381/cds.adbb2d47', '10.24381/cds.bd0915c6', '10.5065/BH6N-5N20']}
    if sampling == 'continuous-250d':
        report['time_coverage_by_year'] = {
            str(y): {'first': wanted[wanted.year == y][0].isoformat(),
                     'last': wanted[wanted.year == y][-1].isoformat(), 'count': 1000}
            for y in (2018, 2019, 2020)
        }
        report['same_time_chunks_as_four_season_verified'] = True
        report['cropped_state_float32_bytes'] = len(wanted) * len(FIELDS) * 12 * 12 * 4
    return ds, report


def download_pilot(path, receipt_path, *, sampling='january'):
    from .seasonal_sampling import requested_times
    requested_times(sampling)  # Reject unknown profiles before opening the network.
    path, receipt_path = Path(path), Path(receipt_path)
    if any(p.exists() or p.is_symlink() for p in (path, receipt_path)):
        raise FileExistsError('output must be new')
    import icechunk as ic
    import zarr
    import numcodecs.zarr3
    try:
        from icechunk.storage import s3_storage
    except ImportError:
        s3_storage = ic.s3_storage
    storage = s3_storage(bucket='earthmover-icechunk-era5', prefix='icechunkV2', region='us-east-1', anonymous=True)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session(snapshot_id=SNAPSHOT)
    if session.snapshot_id != SNAPSHOT:
        raise RuntimeError('snapshot pin not honored')
    with zarr.config.set({'async.concurrency': 1}):
        root = zarr.open_group(store=session.store, mode='r')
        ds, report = extract_dataset(root, snapshot_id=session.snapshot_id, sampling=sampling)
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.era5-', suffix='.nc', dir=path.parent)
    os.close(fd)
    try:
        ds.to_netcdf(temp, engine='h5netcdf', mode='w')
        if sampling == 'continuous-250d' and Path(temp).stat().st_size > 32 * 2**20:
            raise ValueError('continuous profile NetCDF exceeds32MiB publication cap')
        os.link(temp, path)
    finally:
        os.unlink(temp)
    report.update({'source_netcdf_bytes': path.stat().st_size,
        'source_netcdf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'icechunk_version': ic.__version__, 'zarr_version': zarr.__version__})
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return path, report
