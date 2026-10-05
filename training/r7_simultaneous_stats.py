"""Simultaneous paired time-block bootstrap for the S4 confirmation family.

The main-model climatology contract requires one *simultaneous* interval for
every registered confirmation cell (total set, each declared year, each season
group x leads x variables): the resampling unit is a time block of paired
initializations, not a pixel, case or seed, and the family-wise coverage is
controlled by the max-statistic (bootstrap-t style) construction rather than by
17 x 5 unrelated intervals.

Design frozen in the protocol that calls it; this module only implements it:

- inputs are per-case paired series keyed by init time, so the two arms must be
  scored on identical cases or the statistic is undefined;
- blocks are moving blocks of a declared length in cases (the protocol fixes
  the length from train/val dependence before test); a block never spans a
  split, year or season boundary because the caller passes one group's cases;
- each resample draws whole blocks with replacement, recomputes every cell's
  paired mean difference, and standardizes it by that resample's own block
  standard error, so the family's maximum over cells gives the simultaneous
  critical value;
- at least one resample must produce a finite critical value, and too few
  blocks for the declared block length is reported as ``insufficient-evidence``
  rather than as a pass.

No network, no model, no data file: this is pure statistics over case series.
"""
from __future__ import annotations

import math

import numpy as np

MINIMUM_BLOCKS = 4


def block_axis(n_cases, block_length):
    """Moving-block supports for one series: start offsets of whole blocks."""
    for value, name in ((n_cases, "n_cases"), (block_length, "block_length")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if block_length > n_cases:
        raise ValueError("block_length cannot exceed the number of cases")
    starts = np.arange(n_cases - block_length + 1)
    return starts


def block_index_draw(rng, starts, block_length, n_cases):
    """One moving-block resample's case indices (with repetition)."""
    picks = rng.choice(starts.size, size=math.ceil(n_cases / block_length), replace=True)
    blocks = [np.arange(starts[pick], starts[pick] + block_length) for pick in picks]
    return np.concatenate(blocks)[:n_cases]


def simultaneous_paired_interval(series, *, block_length, resamples=4000, seed,
                                 alpha):
    """Max-statistic simultaneous interval for every cell of one family.

    ``series`` maps a cell id to ``(case_keys, deltas)``: the sorted case keys
    shared by both arms and the paired difference ``focus - baseline`` per case.
    All cells must share the identical case key set, otherwise the family is not
    comparable and the call fails closed.

    Returns the observed per-cell mean, its block standard error, the
    simultaneous interval for every cell and the shared critical value. A
    family with fewer than ``MINIMUM_BLOCKS`` blocks returns
    ``insufficient-evidence`` with no interval.
    """
    if not series:
        raise ValueError("at least one cell is required")
    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)):
        raise ValueError("block_length must be an integer")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) \
            or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    key_sets = {tuple(keys) for keys, _ in series.values()}
    if len(key_sets) != 1:
        raise ValueError("every cell in the family must share the identical case keys; "
                         "a paired interval across different case sets is undefined")
    keys = list(next(iter(key_sets)))
    n_cases = len(keys)
    observed = {cell: float(np.mean(deltas)) for cell, (_, deltas) in series.items()}

    starts = block_axis(n_cases, block_length)
    n_blocks = int(starts.size)
    if n_blocks < MINIMUM_BLOCKS:
        return {"status": "insufficient-evidence", "cells": observed,
                "n_cases": n_cases, "block_length": int(block_length),
                "n_blocks": n_blocks, "reason": f"fewer than {MINIMUM_BLOCKS} blocks",
                "alpha": float(alpha)}

    matrix = np.column_stack([np.asarray(series[cell][1], dtype=np.float64)
                              for cell in sorted(series)])
    cell_names = sorted(series)
    rng = np.random.default_rng(seed)
    # Bootstrap standard error per cell, from the same resampling scheme.
    draws = np.empty((resamples, len(cell_names)), dtype=np.float64)
    for index in range(resamples):
        picks = block_index_draw(rng, starts, int(block_length), n_cases)
        draws[index] = matrix[picks].mean(axis=0)
    observed_vector = matrix.mean(axis=0)
    centered = draws - observed_vector
    se = centered.std(axis=0, ddof=1)
    # Guard against degenerate cells: a zero-variance cell cannot be standardized
    # and would otherwise inflate the max statistic silently.
    scale = np.where(se > 0, se, np.nan)
    standardized = centered / scale
    max_abs = np.nanmax(np.abs(standardized), axis=1)
    finite = max_abs[np.isfinite(max_abs)]
    if finite.size == 0:
        return {"status": "insufficient-evidence", "cells": observed,
                "n_cases": n_cases, "block_length": int(block_length),
                "n_blocks": n_blocks,
                "reason": "every cell has zero block standard error",
                "alpha": float(alpha)}
    critical = float(np.quantile(finite, 1.0 - alpha))
    cells = {}
    for position, cell in enumerate(cell_names):
        half = critical * float(scale[position]) if np.isfinite(scale[position]) else None
        low = observed_vector[position] - half if half is not None else None
        high = observed_vector[position] + half if half is not None else None
        cells[cell] = {
            "difference": float(observed_vector[position]),
            "block_standard_error": (float(scale[position])
                                     if np.isfinite(scale[position]) else None),
            "simultaneous_low": low, "simultaneous_high": high,
            "excludes_zero": (bool(low > 0 or high < 0) if half is not None else None),
            "interval_undefined": half is None,
        }
    return {"status": "ok", "alpha": float(alpha), "critical_value": critical,
            "n_cases": n_cases, "block_length": int(block_length), "n_blocks": n_blocks,
            "resamples": int(resamples), "resample_seed": int(seed),
            "cells": cells,
            "rule": ("one max-statistic critical value covers the whole declared "
                     "family; a cell passes only when its simultaneous interval "
                     "excludes zero in the registered direction")}


def skill_from_paired_mse(model_mse, climatology_mse):
    """MSE skill per case: 1 - mse_model / mse_climatology; zero denominator is undefined."""
    model = np.asarray(model_mse, dtype=np.float64)
    reference = np.asarray(climatology_mse, dtype=np.float64)
    if model.shape != reference.shape or model.ndim != 1:
        raise ValueError("paired MSE arrays must be equal-length 1-D vectors")
    if not np.isfinite(model).all() or not np.isfinite(reference).all():
        raise ValueError("paired MSE values must be finite")
    if np.any(reference <= 0):
        raise ValueError("climatology MSE must be strictly positive per case")
    return 1.0 - model / reference
