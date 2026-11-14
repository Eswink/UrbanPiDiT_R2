"""Exit (e): the five balance-round preflight clauses, run before any GPU second is spent.

This is the cheapest gate in the round and the only one that can catch a unit mistake before
hours are spent. Each clause is here for a reason recorded in the paper trail, not for symmetry:

1. **The store's ``z*`` channels are geopotential, not height.** ``scripts/r7_s3_operator_j1_probe.py:70-90``
   records a sign slip that inflated a 36.6 m residual to 8193 m -- a 224x error that would have
   read as "the analysis violates hydrostatic balance" and manufactured a direction. The clause
   back-solves the implied upper pressure from a measured thickness; the right units land near
   500 hPa, ``z/g`` lands above 750.
2. **``f`` has a known magnitude and sign.** ``2*Omega*sin(35 deg) = 8.365e-5 s^-1``. The repo had
   no Coriolis parameter in active code before this round, so there is nothing to inherit.
3. **The geostrophic statistic's coverage is measured, not assumed.** ``Ro < 0.1`` selects roughly
   a tenth of the grid; a balance number over 7% of cells is not a domain-wide result.
4. **Every field is finite on 65x65.** A NaN would silently poison every median downstream.
5. **The ``delta_HB`` offset is a baseline, not a threshold** (plan 0033 §2.1).

The six arms live in `scripts/r7_s3_balance_protocol.py`, which owns the registry and the frozen
protocol; this module owns only the measurement.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STORE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr"
VAL_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/val.jsonl"

#: Three frames spread over the train split, so the offset is not read off one weather regime.
PROBE_FRAMES = (1500, 5000, 25000)
#: Latitude band of the ROI, for the Coriolis magnitude check.
REFERENCE_LATITUDES = (27.0, 35.0, 43.0)
#: ``2*Omega*sin(35 deg)``. A factor-of-ten departure means the parameter is not ``f``.
EXPECTED_CORIOLIS_35 = 8.3652e-05


def _load_frame(store, index: int) -> dict:
    """One frame in **physical units**, keyed by channel name."""
    import numpy as np

    attributes = dict(store.attrs)
    channels = list(attributes["channels"])
    frame = np.asarray(store["state"][index], dtype=np.float64)
    return {name: frame[position] for position, name in enumerate(channels)}


def _grid(store) -> tuple:
    import numpy as np

    return (np.asarray(store["latitude"][:], dtype=np.float64),
            np.asarray(store["longitude"][:], dtype=np.float64))


def clause_units(store) -> dict:
    """Back out the implied upper pressure from a measured 850-500 thickness."""
    import torch

    from training import r7_balance_diagnostics as balance

    frame = _load_frame(store, PROBE_FRAMES[0])
    thickness = torch.from_numpy(frame["z500"] - frame["z850"])
    mean_temperature = torch.from_numpy(0.5 * (frame["t850"] + frame["t500"]))
    measured_m = float(thickness.mean()) / balance.GRAVITY

    implied_ln = measured_m / ((balance.R_DRY / balance.GRAVITY) * float(mean_temperature.mean()))
    implied_upper_hpa = 850.0 / math.exp(implied_ln)

    # The counter-factual: geopotential divided by g a second time.
    wrong_m = measured_m / balance.GRAVITY
    wrong_ln = wrong_m / ((balance.R_DRY / balance.GRAVITY) * float(mean_temperature.mean()))
    wrong_upper_hpa = 850.0 / math.exp(wrong_ln)

    passed = 470.0 < implied_upper_hpa < 530.0
    return {
        "passed": bool(passed),
        "measured_thickness_m": measured_m,
        "implied_upper_hpa": implied_upper_hpa,
        "implied_upper_hpa_if_z_were_divided_by_g": wrong_upper_hpa,
        "rule": "the correct units imply an upper level within 470-530 hPa; dividing by g "
                "implies one above 750 hPa",
    }


def clause_coriolis() -> dict:
    """``f`` at three ROI latitudes against the closed-form value."""
    import torch

    from training import r7_balance_diagnostics as balance

    rows = {}
    for latitude in REFERENCE_LATITUDES:
        f = float(balance.coriolis_parameter(torch.tensor(latitude)))
        rows[str(latitude)] = {
            "f": f,
            "analytic": 2.0 * balance.OMEGA_EARTH * math.sin(math.radians(latitude)),
        }
    reference = rows["35.0"]["f"]
    ratio = reference / EXPECTED_CORIOLIS_35
    return {
        "passed": bool(0.99 < ratio < 1.01 and all(
            row["f"] > 0.0 for row in rows.values())),
        "latitudes": rows,
        "f_at_35_over_expected": ratio,
    }


def clause_geostrophic_coverage(store) -> dict:
    """How much of the grid the ``Ro < 0.1`` mask actually keeps, per level."""
    import torch

    from training import r7_balance_diagnostics as balance

    latitude, longitude = _grid(store)
    frame = _load_frame(store, PROBE_FRAMES[0])
    levels = {}
    for level in ("850", "500", "250"):
        departure, mask = balance.geostrophic_departure(
            torch.from_numpy(frame[f"u{level}"]), torch.from_numpy(frame[f"v{level}"]),
            torch.from_numpy(frame[f"z{level}"]), latitude, longitude)
        selected = departure[mask]
        levels[level] = {
            "grid_fraction": float(mask.double().mean()),
            "ws_ag_median": float(selected.median()) if selected.numel() else None,
            "n_cells": int(mask.sum()),
        }
    fractions = [row["grid_fraction"] for row in levels.values()]
    return {
        "passed": bool(all(0.01 < value < 0.9 for value in fractions)),
        "levels": levels,
        "rule": "coverage must be non-degenerate: neither zero (mask empty) nor most of the "
                "grid (Ro<0.1 would then not be selecting anything)",
    }


def clause_hydrostatic_offset(store) -> dict:
    """The ``delta_HB`` offset across frames, recorded as the difference baseline."""
    import torch

    from training import r7_balance_diagnostics as balance

    rows = {}
    for index in PROBE_FRAMES:
        frame = _load_frame(store, index)
        deviation = balance.hydrostatic_deviation(
            torch.from_numpy(frame["z850"]), torch.from_numpy(frame["z500"]),
            torch.from_numpy(frame["t850"]), torch.from_numpy(frame["t500"]),
            850.0, 500.0,
            torch.from_numpy(frame["q850"]), torch.from_numpy(frame["q500"]))
        rows[str(index)] = {
            "median": float(deviation.median()),
            "mean": float(deviation.mean()),
            "std": float(deviation.std()),
            "finite": bool(torch.isfinite(deviation).all()),
        }
    medians = [row["median"] for row in rows.values()]
    finite = all(row["finite"] for row in rows.values())
    return {
        "passed": bool(finite and max(abs(value) for value in medians) < 0.2),
        "frames": rows,
        "baseline_median": sum(medians) / len(medians),
        "rule": "every frame finite and |median| < 0.2, i.e. a small systematic offset rather "
                "than a unit error (a unit error gives order 1 or more)",
        "arm_comparison": "differences only: this offset is shared by every arm, so only "
                          "arm-to-arm gaps are meaningful",
    }


def clause_finiteness(store) -> dict:
    """Every channel finite on 65x65, and the 250 hPa level really has no temperature."""
    import numpy as np

    frame = _load_frame(store, PROBE_FRAMES[0])
    bad = [name for name, values in frame.items() if not np.isfinite(values).all()]
    shape_ok = all(values.shape == (65, 65) for values in frame.values())
    return {
        "passed": bool(not bad and shape_ok),
        "non_finite_channels": bad,
        "shape_ok": shape_ok,
        "channels_missing_at_250": [name for name in ("t250", "q250")
                                    if name not in frame],
        "note": "the store carries z/u/v at 250 hPa but no temperature or humidity, so the "
                "500-250 thickness substitutes t500 and must be reported separately",
    }


def preflight() -> dict:
    """All five clauses on real store frames; no model, no forecast, no GPU."""
    import zarr

    store = zarr.open_group(str(STORE), mode="r")
    clauses = {
        "units_are_geopotential_not_height": clause_units(store),
        "coriolis_magnitude": clause_coriolis(),
        "geostrophic_coverage_is_measured": clause_geostrophic_coverage(store),
        "hydrostatic_offset_is_a_baseline": clause_hydrostatic_offset(store),
        "fields_are_finite": clause_finiteness(store),
    }
    return {
        "format": "r7-s3-balance-preflight-v1",
        "store": str(STORE),
        "val_manifest": str(VAL_MANIFEST),
        "probe_frames": list(PROBE_FRAMES),
        "clauses": clauses,
        "e_preflight_passed": bool(all(clause["passed"] for clause in clauses.values())),
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "gpu_hours": 0.0,
        "limitations": [
            "this is a preflight, not a reading: it touches three train frames and no forecast",
            "the delta_HB offset is real (traced to the analysis) and is carried as a "
            "difference baseline, never as an absolute threshold",
        ],
    }
