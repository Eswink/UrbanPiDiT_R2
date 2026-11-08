"""The C1 reproduction tool must compute what it claims, and its constants must stay pinned.

Decision 0094 closed Stage C1 permanently on a measurement the repository could not recompute;
tools/reproduce_c1_preflight.py exists to close that gap for the analysis-field half. These tests
pin the two things that would silently invalidate it: that the hinge is the module's own arithmetic
(not a re-implementation), and that the case indices and recorded values match the published page.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "reproduce_c1_preflight", ROOT / "tools/reproduce_c1_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_case_indices_are_the_published_eight():
    """The eight window indices come from C1's per-case table; a typo would change every average."""
    tool = _tool()
    assert tool.C1_CASES == (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)
    assert len(set(tool.C1_CASES)) == 8


def test_the_recorded_values_are_the_ones_the_page_and_module_state():
    """Both published sources are carried, so the comparison cannot quietly drop one."""
    tool = _tool()
    assert tool.RECORDED["result_page_mean_abs_div"] == 2.51e-05
    assert tool.RECORDED["result_page_p95_abs_div"] == 7.30e-05
    assert tool.RECORDED["module_comment_mean"] == 9.4e-06
    assert tool.RECORDED["module_comment_p95"] == 5.6e-05


def test_the_hinge_is_the_production_function_not_a_reimplementation():
    """A hand-written hinge could drift from the frozen module; assert identity of the arithmetic."""
    tool = _tool()
    source = (ROOT / "tools/reproduce_c1_preflight.py").read_text(encoding="utf-8")
    assert "from training.r7_physical_constraints import" in source
    assert "divergence_penalty" in source
    # The tool must not re-derive the hinge itself.
    assert "clamp_min" not in source, "the hinge must come from the module, not be re-implemented"
    assert tool is not None


@pytest.mark.skipif(
    not (ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr").exists(),
    reason="store not present in this checkout",
)
def test_measure_returns_finite_values_above_the_scale():
    """On one real frame the analysis field's mean |div| exceeds DIVERGENCE_SCALE.

    This is the qualitative claim the C1 veto rests on: the real atmosphere is not layer-wise
    non-divergent, so penalising divergence is not neutral. Pinned so a future change that
    flattens the field (or mis-denormalises it) fails here.
    """
    tool = _tool()
    rows = tool.measure(ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr",
                        (1013,))
    row = rows[0]
    # Pin the magnitude, not just "bigger than the scale": with denormalisation removed the raw
    # normalised winds also exceed 1e-5, so a one-sided inequality passes on the defect (measured:
    # the first version of this test did exactly that). The denormalised analysis field at this
    # frame measures ~4.7e-05 s^-1; the raw field is a different quantity in different units.
    assert row["mean_abs_div"] == pytest.approx(4.7084e-05, rel=0.05), row
    assert 0.0 <= row["fraction_above_deadband"] <= 1.0
    assert row["scale"] == 1e-05 and row["tolerance"] == 1.0
