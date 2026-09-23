"""Fixed calendar sampling and target-free balanced manifest selection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROFILES = {'january': (1,), 'four-season': (1, 4, 7, 9),
            'continuous-250d': tuple(range(1, 10))}


def requested_times(profile='january'):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('unknown bounded sampling profile')
    if profile == 'continuous-250d':
        # Exactly250days in each year; leap-year end date is deliberately different.
        return pd.DatetimeIndex(np.concatenate([
            pd.date_range(f'{y}-01-01', periods=1000, freq='6h').values
            for y in (2018, 2019, 2020)
        ]))
    return pd.DatetimeIndex(np.concatenate([
        pd.date_range(f'{y}-{m:02d}-01', periods=32, freq='6h').values
        for y in (2018, 2019, 2020) for m in PROFILES[profile]
    ]))


def select_balanced_records(records, available_times, *, count_per_month=6):
    """First N complete 72h trajectories in EACH fixed sampled month.

    Inspect timestamps only, not target values, predictions or errors. Preserve
    original records and chronological ordering. No across-gap trajectories.
    """
    if isinstance(count_per_month, bool) or not isinstance(count_per_month, int) or not 1 <= count_per_month <= 6:
        raise ValueError('one to six cases per month required')
    times = pd.DatetimeIndex(available_times)
    if times.hasnans or times.tz is not None or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('valid unique increasing UTC-naive available times required')
    if not records or len({r['init_time'] for r in records}) != len(records):
        raise ValueError('nonempty unique initialization records required')
    splits = {r['split'] for r in records}
    if len(splits) != 1 or next(iter(splits)) not in ('val', 'test'):
        raise ValueError('balanced selection is held-out only')
    available = set(times.as_unit('ns').asi8.tolist())
    selected = []
    for month in PROFILES['four-season']:
        eligible = []
        for record in sorted(records, key=lambda r: r['init_time']):
            init = pd.Timestamp(record['init_time'])
            if pd.isna(init) or init.tz is not None:
                raise ValueError('invalid initialization time')
            if init.month != month:
                continue
            stamps = [pd.Timestamp(t) for t in record['history_times']]
            if not stamps or stamps[-1] != init:
                raise ValueError('history must end at initialization')
            if any(pd.isna(t) or t.tz is not None for t in stamps):
                raise ValueError('invalid history time')
            # Include all six-hour transition times, not just reported horizons.
            required = stamps + [init + pd.Timedelta(hours=h) for h in range(0, 73, 6)]
            if all(t.value in available and t.year == init.year for t in required):
                eligible.append(record)
        if len(eligible) < count_per_month:
            raise ValueError(f'month {month} lacks complete predeclared trajectories')
        selected.extend(dict(r) for r in eligible[:count_per_month])
    return sorted(selected, key=lambda r: r['init_time'])


def write_balanced_manifest(manifest, available_times):
    manifest = Path(manifest)
    from data.r7_store import require_complete_manifest
    require_complete_manifest(manifest)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = select_balanced_records(records, available_times)
    output = manifest.with_name(manifest.stem + '_balanced.jsonl')
    with output.open('x', encoding='utf-8') as f:
        for record in selected:
            f.write(json.dumps(record, sort_keys=True) + '\n')
    return output
