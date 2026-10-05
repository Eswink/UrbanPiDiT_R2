"""Tests for the S4 simultaneous paired block bootstrap (pure statistics, no data)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_simultaneous_stats import (  # noqa: E402
    block_axis, block_index_draw, simultaneous_paired_interval, skill_from_paired_mse,
)


def _series(values, keys=None):
    keys = keys if keys is not None else [f"2017-01-{i+1:02d}T00:00:00"
                                          for i in range(len(values))]
    return keys, np.asarray(values, dtype=np.float64)


def test_block_axis_and_draw_cover_every_case_once_per_block():
    starts = block_axis(10, 3)
    assert starts.tolist() == list(range(8))
    rng = np.random.default_rng(1)
    draw = block_index_draw(rng, starts, 3, 10)
    assert len(draw) == 10
    assert draw.min() >= 0 and draw.max() <= 9
    # every drawn triple is one contiguous supported block
    for value in set(draw.tolist()):
        assert np.any((starts <= value) & (value < starts + 3))


def test_block_axis_refuses_impossible_parameters():
    with pytest.raises(ValueError):
        block_axis(5, 6)
    with pytest.raises(ValueError):
        block_axis(0, 1)
    with pytest.raises(ValueError):
        block_axis(5, True)


def test_interval_covers_strong_signal_and_rejects_null():
    keys = [f"2017-01-{i+1:02d}T00:00:00" for i in range(40)]
    strong = {"6h|t2m": _series([-0.5 + 0.01 * (i % 3) for i in range(40)], keys)}
    out = simultaneous_paired_interval(strong, block_length=4, resamples=800, seed=7,
                                       alpha=0.05)
    assert out["status"] == "ok"
    cell = out["cells"]["6h|t2m"]
    assert cell["excludes_zero"] is True and cell["difference"] < 0
    assert cell["simultaneous_high"] < 0


def test_interval_does_not_claim_significance_for_a_null_series():
    keys = [f"2017-01-{i+1:02d}T00:00:00" for i in range(40)]
    rng = np.random.default_rng(3)
    null = {"6h|t2m": _series(rng.normal(0.0, 0.4, size=40), keys)}
    out = simultaneous_paired_interval(null, block_length=4, resamples=800, seed=7,
                                       alpha=0.05)
    assert out["status"] == "ok"
    assert out["cells"]["6h|t2m"]["excludes_zero"] is False


def test_family_requires_identical_case_keys():
    a = _series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    b = ("2017-02-01T00:00:00", np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                            0.7, 0.8, 0.9, 1.0]))
    with pytest.raises(ValueError, match="identical case keys"):
        simultaneous_paired_interval({"x": a, "y": b}, block_length=2, resamples=10,
                                     seed=1, alpha=0.05)


def test_insufficient_blocks_is_reported_not_passed():
    keys = [f"2017-01-{i+1:02d}T00:00:00" for i in range(6)]
    out = simultaneous_paired_interval({"6h|t2m": _series([0.1] * 6, keys)},
                                       block_length=4, resamples=100, seed=1, alpha=0.05)
    # 6 cases with 4-length blocks -> 3 supports < MINIMUM_BLOCKS
    assert out["status"] == "insufficient-evidence"
    assert out["n_blocks"] == 3
    assert "critical_value" not in out
    # the observed mean is still reported, but no interval is manufactured
    assert out["cells"]["6h|t2m"] == pytest.approx(0.1)


def test_max_statistic_is_at_least_as_wide_as_a_single_cell_interval():
    keys = [f"2017-01-{i+1:02d}T00:00:00" for i in range(60)]
    rng = np.random.default_rng(11)
    cells = {f"{lead}h|t2m": _series(rng.normal(-0.1, 0.5, 60), keys)
             for lead in (6, 12)}
    family = simultaneous_paired_interval(cells, block_length=5, resamples=1000,
                                          seed=5, alpha=0.05)
    single = simultaneous_paired_interval({"6h|t2m": cells["6h|t2m"]},
                                          block_length=5, resamples=1000, seed=5,
                                          alpha=0.05)
    width_family = (family["cells"]["6h|t2m"]["simultaneous_high"]
                    - family["cells"]["6h|t2m"]["simultaneous_low"])
    width_single = (single["cells"]["6h|t2m"]["simultaneous_high"]
                    - single["cells"]["6h|t2m"]["simultaneous_low"])
    assert family["critical_value"] >= single["critical_value"]
    assert width_family >= width_single


def test_zero_variance_cell_is_undefined_rather_than_silently_inflating():
    keys = [f"2017-01-{i+1:02d}T00:00:00" for i in range(30)]
    cells = {"constant": _series([0.5] * 30, keys),
             "noisy": _series(np.linspace(-0.2, 0.2, 30), keys)}
    out = simultaneous_paired_interval(cells, block_length=3, resamples=500, seed=9,
                                       alpha=0.05)
    assert out["status"] == "ok"
    assert out["cells"]["constant"]["interval_undefined"] is True
    assert out["cells"]["constant"]["simultaneous_low"] is None


def test_skill_formula_and_zero_denominator_refusal():
    skill = skill_from_paired_mse([1.0, 2.0], [2.0, 4.0])
    assert skill.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        skill_from_paired_mse([1.0], [0.0])
    with pytest.raises(ValueError):
        skill_from_paired_mse([1.0, 2.0], [1.0, float("nan")])
