"""CPU counterproofs for the Stage D white-box probe; 0 GPU-h, no weather, no claim.

The probe reports two measurements the campaign's Stage D asks for -- which diagnostic each
anchored P slot tracks, and whether Z's per-position progress relates to the static fields --
so the tests that matter are the ones proving the *arithmetic* cannot quietly report a
diagrammatic answer: correlations must be computed from the inputs, a constant vector must
yield None rather than a fabricated 0.0, and shape mismatches must raise instead of
broadcasting into a meaningless number.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from tools import r7_s3_whitebox_probe as probe


def test_the_two_counted_surfaces_are_eight_and_eight():
    """The campaign names 8 anchored slots and 8 diagnostic targets; both are asserted, not assumed."""
    assert probe.ANCHORED_SLOTS == 8
    from model.r7_process_tensor_diagnostics import PROCESS_DIAGNOSTIC_NAMES
    assert len(PROCESS_DIAGNOSTIC_NAMES) == 8
    assert len(set(PROCESS_DIAGNOSTIC_NAMES)) == 8


def test_the_typed_local_fields_are_a_declared_subset():
    """Four of the eight become spatial maps; naming them explicitly stops a silent reorder."""
    from model.r7_process_tensor_diagnostics import PROCESS_DIAGNOSTIC_NAMES
    assert len(probe.TYPED_LOCAL_FIELD_NAMES) == 4
    assert set(probe.TYPED_LOCAL_FIELD_NAMES) <= set(PROCESS_DIAGNOSTIC_NAMES)


def test_a_diagonal_alignment_is_reported_as_aligned():
    """When slot i genuinely tracks diagnostic i, every slot must report its own index."""
    rng = np.random.default_rng(0)
    diagnostics = rng.normal(size=(240, 8))
    # Slot i is diagnostic i plus a little noise: the documented alignment, made literal.
    slots = diagnostics + 0.05 * rng.normal(size=(240, 8))
    body = probe.alignment_matrix(slots, diagnostics)
    assert body["aligned_slots"] == 8
    assert all(entry["matches_documented_alignment"] for entry in body["per_slot_best"])
    # The diagonal must dominate: |rho_ii| is the max of every row.
    for i, row in enumerate(body["spearman"]):
        assert abs(row[i]) == pytest.approx(max(abs(v) for v in row), abs=1e-12)


def test_a_permuted_alignment_is_reported_as_not_aligned():
    """A counterexample must come back as a counterexample, not be rounded into agreement."""
    rng = np.random.default_rng(1)
    diagnostics = rng.normal(size=(240, 8))
    slots = diagnostics[:, ::-1] + 0.05 * rng.normal(size=(240, 8))
    body = probe.alignment_matrix(slots, diagnostics)
    assert body["aligned_slots"] == 0
    assert body["per_slot_best"][0]["best_diagnostic"] == 7


def test_a_constant_series_yields_none_not_a_fabricated_zero():
    """A slot that never moves has no correlation; reporting 0.0 would invent a measurement."""
    constant = np.ones(50)
    varying = np.linspace(-1, 1, 50)
    assert probe._correlation(constant, varying) is None
    assert probe._correlation(varying, varying) == pytest.approx(1.0)
    assert probe._correlation(varying, -varying) == pytest.approx(-1.0)


def test_correlation_rejects_mismatched_lengths():
    with pytest.raises(RuntimeError, match="equal, non-degenerate"):
        probe._correlation([1.0, 2.0, 3.0], [1.0, 2.0])


def test_spearman_catches_a_monotone_nonlinear_relation():
    """A saturating slot tracks its diagnostic in rank but not linearly; the probe must see it."""
    x = np.linspace(0.0, 10.0, 200)
    y = np.log1p(x)
    body = probe.alignment_matrix(y.reshape(-1, 1), x.reshape(-1, 1))
    assert body["spearman"][0][0] == pytest.approx(1.0)
    assert abs(body["pearson"][0][0]) < 0.99


def test_progress_correlation_requires_matching_shapes():
    """A broadcast pair would produce a number with no meaning, so a mismatch must raise."""
    progress = np.arange(12.0).reshape(3, 4)
    ok = probe.progress_correlation(progress, {"terrain": np.arange(12.0).reshape(3, 4)})
    assert ok["n_positions"] == 12
    assert ok["correlations"]["terrain"]["pearson"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="has shape"):
        probe.progress_correlation(progress, {"terrain": np.zeros((4, 3))})


def test_progress_correlation_reports_the_field_summary():
    progress = np.array([[0.0, 1.0], [2.0, 0.0]])
    body = probe.progress_correlation(progress, {"land": np.array([[0.0, 1.0], [1.0, 0.0]])})
    assert body["progress_summary"]["nonzero"] == 2
    assert body["progress_summary"]["max"] == 2.0


def test_the_probe_declares_no_scientific_claim_and_no_threshold():
    """Stage D fixes no numeric bar, so the tool must not imply one."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    body = ast.parse(source)
    assert "scientific_claim\": False" in source or 'scientific_claim": False' in source
    # No pass/fail threshold is defined anywhere in the module.
    assert "threshold" not in source.lower() or "no threshold" in source.lower()
    assert not any(isinstance(n, ast.Assign) and any(
        getattr(t, "id", "") in {"PASS_THRESHOLD", "FAIL_THRESHOLD", "MIN_CORRELATION"}
        for t in n.targets) for n in ast.walk(body))


def test_the_probe_is_runnable_and_writes_exclusively(tmp_path):
    """R-001: an existing output is never truncated."""
    out = tmp_path / "probe.json"
    assert probe.main(["--out", str(out)]) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["all_checks"] is True
    assert body["scientific_claim"] is False and body["gpu_hours"] == 0.0
    with pytest.raises(FileExistsError):
        probe.main(["--out", str(out)])


def test_the_probe_has_a_main_guard():
    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    guards = [n for n in tree.body
              if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
              and getattr(n.test.left, "id", None) == "__name__"]
    assert guards and "main" in ast.dump(guards[0])
