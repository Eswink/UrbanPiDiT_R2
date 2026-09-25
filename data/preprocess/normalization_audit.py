"""Audit the shared eps=1e-6 std floor against small-magnitude labels (#63).

The store normalizations (`r7_era5._training_stats`, `contracts.Stats.finish`)
floor the per-channel standard deviation at eps=1e-6. For channels whose raw
values (or process-diagnostic labels) live around 1e-8 — moisture advection and
flux convergence among them — that floor divides by 1e-6 where the real spread
is 1e-8, so normalized targets collapse to ~1e-2 of the intended unit scale and
can look "almost zero" to the loss.

This audit is read-only bookkeeping: it reports the raw value/variance, the
floored std, whether the floor is active, and the near-zero fraction before and
after normalization. It changes nothing; any new normalization is decided by
train statistics and versioned separately.
"""
from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-6


def audit_channel(values, eps=DEFAULT_EPS, near_zero=0.01):
    """Per-channel floor audit. `values` is any 1-D+ float array; channel axes
    are reduced by the caller. Returns the raw stats, the floored std and the
    near-zero fractions; the input is never modified."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("finite nonempty values required")
    flat = values.reshape(-1)
    std = float(flat.std())
    floored = max(std, float(eps))
    scale = flat / floored
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": std,
        "eps": float(eps),
        "floored_std": floored,
        "floor_active": std < float(eps),
        "std_to_eps_ratio": std / float(eps),
        "max_abs": float(np.abs(flat).max()),
        "near_zero_fraction_raw": float((np.abs(flat) < near_zero * eps).mean()),
        "near_zero_fraction_normalized": float((np.abs(scale) < near_zero).mean()),
        "values_preserved": True,
    }


def audit_label_set(label_arrays, eps=DEFAULT_EPS, near_zero=0.01):
    """Audit every process-diagnostic channel; returns one record per name."""
    report = {}
    for name, values in label_arrays.items():
        report[name] = audit_channel(values, eps=eps, near_zero=near_zero)
    crushed = sorted(name for name, record in report.items()
                     if record["floor_active"])
    return {
        "format": "r7-normalization-floor-audit-v1",
        "scientific_claim": False,
        "channels": report,
        "floor_active_channels": crushed,
        "note": "audit only; any new normalization comes from train statistics "
                "and is versioned before use",
    }
