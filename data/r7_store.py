"""Read-only validation shared by training and evaluation of R7 stores."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .preprocess.contracts import chronological_splits, parse_split_time_ranges, utc_time_index
from .preprocess.grid import regular_latlon_spacing

HOUR_NS = 3_600_000_000_000

YEAR_SPLIT_MODE = 'years'
TIME_RANGE_SPLIT_MODE = 'time_ranges'


def split_time_labels(root):
    """Per-timestep split ownership under the store's own declared semantics.

    Returns ``None`` for the default **year** mode, where callers must keep
    using ``root.attrs['split_years']`` exactly as before. When the store
    declares ``split_mode == 'time_ranges'`` (decision 0005) it returns an
    array with one entry per store timestep holding ``'train'``/``'val'``/
    ``'test'``, or ``''`` for a step outside every declared interval.

    **Precedence.** In time-range mode the declared ``split_time_ranges`` is
    authoritative and ``split_years`` is *not* consulted for membership: a
    one-year engineering segment necessarily carries placeholder year splits
    (D1 declares train/val/test years 2016/2017/2018 while every observation is
    from 2016), so the year declaration cannot describe the real partition.
    ``split_years`` is retained in the metadata because older artifacts and
    ``validate_store`` still reference it, and it keeps its meaning as the
    declared normalization/leakage guard - the builder already requires every
    range to fall inside the declared train years.

    Fails closed on a store that declares the range mode without a parseable
    declaration, so a malformed store cannot silently fall back to year
    semantics.
    """
    if root.attrs.get('split_mode', YEAR_SPLIT_MODE) != TIME_RANGE_SPLIT_MODE:
        return None
    ranges = parse_split_time_ranges(root.attrs['split_time_ranges'])
    stamps = utc_time_index(np.asarray(root['time_ns'][:]).astype('datetime64[ns]')).asi8
    labels = np.full(stamps.shape, '', dtype=object)
    for split, rows in ranges.items():
        for start_ns, stop_ns in rows:
            labels[(stamps >= start_ns) & (stamps < stop_ns)] = split
    return labels


def require_complete_manifest(manifest):
    marker = Path(manifest).parent/'BUILD_COMPLETE.json'
    if not marker.is_file():
        raise ValueError('incomplete/unversioned R7 manifest; rebuild in a new directory')
    metadata = json.loads(marker.read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1 or metadata.get('build_complete') is not True:
        raise ValueError('manifest build is not complete')


def normalization(root, prefix='normalization', count=None):
    mean = np.asarray(root[prefix+'_mean'][:], dtype=np.float32)
    std = np.asarray(root[prefix+'_std'][:], dtype=np.float32)
    if mean.ndim != 1 or std.shape != mean.shape or (count is not None and len(mean) != count):
        raise ValueError('normalization/channel shape mismatch')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError('normalization must be finite with positive std')
    return mean, std


def validate_store(root):
    if root.attrs.get('schema_version') != 1 or root.attrs.get('build_complete') is not True:
        raise ValueError('incomplete/unversioned R7 store; rebuild in a new directory')
    if root.attrs.get('time_unit') != 'ns':
        raise ValueError('R7 store timestamps must explicitly use ns')
    shape = root['state'].shape
    if len(shape) != 4 or min(shape) < 1:
        raise ValueError('state must be nonempty [T,C,H,W]')
    names = list(root.attrs['channels'])
    if len(names) != shape[1] or len(set(names)) != len(names):
        raise ValueError('duplicate/missing channel metadata')
    lat, lon = np.asarray(root['latitude'][:]), np.asarray(root['longitude'][:])
    if lat.shape != (shape[2],) or lon.shape != (shape[3],):
        raise ValueError('coordinate shape mismatch')
    spacing = regular_latlon_spacing(lat, lon)
    if not np.isclose(spacing,root.attrs['native_grid_spacing_deg'],rtol=0,atol=1e-5):
        raise ValueError('grid metadata disagrees with coordinates')
    raw = np.asarray(root['time_ns'][:])
    if raw.dtype.kind != 'i' or raw.shape != (shape[0],):
        raise ValueError('time_ns must be integer [T]')
    utc_time_index(raw.astype('datetime64[ns]'))
    splits = chronological_splits(root.attrs['split_years'])
    if set(root.attrs['normalization_years']) != splits['train']:
        raise ValueError('normalization years must exactly match training years')
    normalization(root,count=shape[1])
    if 'process_diagnostics_raw' in root:
        pshape = root['process_diagnostics_raw'].shape
        if pshape != (shape[0],len(root.attrs['process_diagnostic_names'])):
            raise ValueError('process metadata/shape mismatch')
        normalization(root,'process_normalization',pshape[1])
    return raw


def validate_record(root, record):
    import pandas as pd
    h = record['history_indices']
    indices = list(h)+[record['target_index']]
    if not h or any(isinstance(i,bool) or not isinstance(i,int) or i < 0 or i >= root['state'].shape[0] for i in indices):
        raise ValueError('invalid record indices')
    stamps = list(record['history_times'])+[record['target_time']]
    if len(stamps) != len(indices):
        raise ValueError('timestamp/index count mismatch')
    times = utc_time_index(stamps)
    actual = np.array([int(root['time_ns'][i]) for i in indices], dtype=np.int64)
    if not np.array_equal(actual,times.asi8):
        raise ValueError('record timestamps do not match store indices')
    init = pd.Timestamp(record['init_time'])
    if init.tz is not None or init.value != actual[-2]:
        raise ValueError('init_time must equal the final history timestamp')
    lead = record['lead_time_hours']
    if isinstance(lead,bool) or not isinstance(lead,(int,float)) or not np.isfinite(lead) or lead <= 0:
        raise ValueError('positive lead hours required')
    if actual[-1]-actual[-2] != lead*HOUR_NS:
        raise ValueError('lead time does not match timestamps')
    if len(h)>1 and not np.all(np.diff(actual[:-1]) == np.diff(actual[:-1])[0]):
        raise ValueError('history timestamps have irregular cadence')
    labels = split_time_labels(root)
    if labels is None:
        years = set(root.attrs['split_years'][record['split']])
        if any(t.year not in years for t in times):
            raise ValueError('forecast window crosses split years')
    else:
        owner = record['split']
        for index in indices:
            if labels[index] != owner:
                raise ValueError(
                    f"forecast window crosses split boundaries: index {index} belongs "
                    f"to {labels[index] or 'no declared range'}, not {owner}")
    return indices
