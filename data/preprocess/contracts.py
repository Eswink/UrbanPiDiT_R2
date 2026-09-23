"""Strict publication/time/statistics contracts for local R7 datasets."""
from __future__ import annotations
from functools import wraps
from inspect import signature
from pathlib import Path
import json
import numpy as np


def fresh_outputs(*names):
    """Reserve new disjoint paths; never remove dataset contents.

    A failed build remains inspectable and lacks BUILD_COMPLETE.json. Only
    locks owned by this call are removed. This is not a distributed lock.
    """
    def decorate(function):
        sig = signature(function)
        @wraps(function)
        def run(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            paths = [Path(bound.arguments[n]).absolute() for n in names]
            resolved = [p.resolve() for p in paths]
            for i, p in enumerate(resolved):
                if any(p == q or p in q.parents or q in p.parents for q in resolved[:i]):
                    raise ValueError('output paths must be distinct and non-nested')
            def check_new():
                for p in paths:
                    if p.exists() or p.is_symlink():
                        raise FileExistsError(f'refusing existing output: {p}')
            owned = []
            try:
                check_new()
                for p in sorted(resolved):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    lock = p.with_name(p.name + '.r7-build.lock')
                    with lock.open('x', encoding='utf-8') as f:
                        f.write('R7 local output reservation\n')
                    owned.append(lock)
                check_new()
                result = function(*args, **kwargs)
                manifest = Path(bound.arguments['manifest_dir'])
                with (manifest/'BUILD_COMPLETE.json').open('x', encoding='utf-8') as f:
                    json.dump({'schema_version': 1, 'build_complete': True}, f)
                return result
            finally:
                for lock in owned:
                    lock.unlink(missing_ok=True)
        return run
    return decorate


def chronological_splits(split_years):
    if set(split_years) != {'train', 'val', 'test'}:
        raise ValueError('split_years must contain train, val and test exactly')
    result = {}
    for key, values in split_years.items():
        values = list(values)
        if not values or any(isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) for y in values):
            raise ValueError('split years must be nonempty integer sequences')
        result[key] = set(map(int, values))
    if not (max(result['train']) < min(result['val']) and max(result['val']) < min(result['test'])):
        raise ValueError('year splits must be disjoint and chronological: train < val < test')
    return result


def utc_time_index(values):
    import pandas as pd
    times = pd.DatetimeIndex(values)
    if times.tz is not None:
        raise ValueError('use explicitly UTC-naive timestamps')
    if len(times) == 0 or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('time must be nonempty, finite, unique and increasing')
    return times.as_unit('ns')


class RunningMoments:
    """Mergeable centered population moments over finite [N, features] blocks."""
    def __init__(self, features: int):
        self.count = 0
        self.mean = np.zeros(features, dtype=np.float64)
        self.m2 = np.zeros(features, dtype=np.float64)

    def update(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1:] != self.mean.shape or not np.isfinite(x).all():
            raise ValueError('moments need finite [N, features] blocks')
        n = len(x)
        if n == 0:
            return
        mean = x.mean(axis=0)
        m2 = ((x-mean)**2).sum(axis=0)
        delta = mean-self.mean
        total = self.count+n
        self.m2 += m2+delta*delta*(self.count*n/total)
        self.mean += delta*(n/total)
        self.count = total

    def finish(self, eps=1e-6):
        if self.count == 0:
            raise ValueError('no training observations for moments')
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be positive')
        std = np.maximum(np.sqrt(np.maximum(self.m2/self.count, 0)), eps)
        return self.mean.astype(np.float32), std.astype(np.float32)
