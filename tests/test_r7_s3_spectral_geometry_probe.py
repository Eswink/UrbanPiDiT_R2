"""Counterproofs for the Stage P spectral-geometry probe; CPU-only, no data, no claim.

Two of these tests exist because the first draft of the probe was **wrong** in exactly those
two ways, and neither error would have raised anything -- both produced plausible-looking
numbers:

1. **The half-spectrum bug.** ``rfft2`` returns only the non-negative-frequency half, and
   ``norm='ortho'`` does not compensate for the omitted conjugate partners, so a naive
   ``sum(|spectrum|^2)`` reports **half** the energy. The probe's Parseval check read 0.499
   relative error -- a 50% discrepancy that looks like a units bug in the *data* rather than
   in the *measurement*. ``test_parseval_needs_hermitian_weights`` pins the weight vector and
   ``test_naive_energy_sum_is_half`` shows what the naive form would return, so the fix cannot
   be quietly reverted.
2. **The degenerate band edges.** The first frequency helper mapped the last rfft index of an
   odd-length axis back to ``-1``, so the computed corner fell to ``sqrt(2)`` instead of
   ``sqrt(32^2+32^2)``, collapsing every band above wavenumber 1.4 into one. The probe
   reported a *passing* decomposition over two bands. ``test_band_edges_reach_the_corner``
   pins the real corner.

The guard tests are the decision-0105 requirement: a check that cannot fail is not a check.
``test_guard_rejects_an_inert_comparison`` feeds the guard a comparison that must be positive
and asserts it is, and ``test_zero_at_truth_is_exact`` asserts exact zero rather than a
tolerance.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/r7_s3_spectral_geometry_probe.py")


def _module():
    spec = importlib.util.spec_from_file_location("spectral_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


def _smooth(height=65, width=65):
    rows = torch.arange(height, dtype=torch.float64)[:, None]
    cols = torch.arange(width, dtype=torch.float64)[None, :]
    return (torch.sin(rows / 3.0) * torch.cos(cols / 5.0)
            + 0.3 * torch.sin(rows / 1.7) * torch.cos(cols / 1.3))


# --- the two bugs the first draft had ------------------------------------------------

def test_naive_energy_sum_is_half(probe):
    """Without the Hermitian weights the spectrum reports about half the energy.

    Precisely: every interior column stands for a conjugate pair and counts twice, so the
    naive sum is ``(spatial + E_dc) / 2`` when the width is odd (the last rfft column is a
    genuine frequency, not Nyquist) and ``(spatial + E_dc + E_nyquist) / 2`` when it is even.
    On this 65-wide grid only DC is single-counted. That is exactly the form which made the
    first draft's Parseval check read 0.499: close to half but never exactly, so it reads like
    a problem in the data rather than in the measurement.

    Asserted rather than tolerated so that "just drop the weights" fails loudly.
    """
    field = _smooth()
    spectrum = torch.fft.rfft2(field, norm="ortho")
    naive = float((spectrum.abs() ** 2).sum())
    spatial = float((field ** 2).sum())
    single = float((spectrum[:, 0].abs() ** 2).sum())
    if field.shape[-1] % 2 == 0:
        single += float((spectrum[:, -1].abs() ** 2).sum())
    assert naive == pytest.approx(0.5 * (spatial + single), rel=1e-12)
    assert naive < 0.52 * spatial, "the naive sum must be near half, not near one"


def test_parseval_needs_hermitian_weights(probe):
    """With the weights, Parseval holds to float noise on this grid."""
    field = _smooth()
    assert probe.parseval_error(field) < 1e-12


def test_hermitian_weights_double_only_the_interior_columns(probe):
    """DC never doubles; for even width Nyquist does not either; everything between does."""
    weights = probe._hermitian_weights(64)[0].tolist()
    assert weights[0] == 1.0
    assert weights[-1] == 1.0
    assert all(value == 2.0 for value in weights[1:-1])
    odd = probe._hermitian_weights(65)[0].tolist()
    assert odd[0] == 1.0
    assert odd[-1] == 2.0


def test_band_edges_reach_the_corner(probe):
    """The last edge must be the true corner Nyquist, not a collapsed sqrt(2).

    The first draft's frequency helper mapped the final rfft index of an odd-length axis to
    ``-1``, which put the corner at ``sqrt(2)`` and merged every band above wavenumber 1.4 --
    while still reporting a passing decomposition.
    """
    edges = probe.band_edges(65, 65)
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(math.sqrt(32.0 ** 2 + 32.0 ** 2), rel=1e-12)
    assert edges == sorted(edges)
    assert len({round(edge, 12) for edge in edges}) == len(edges)


def test_band_edges_are_strictly_increasing_and_cover_the_spectrum(probe):
    """Every rfft2 bin must land in exactly one band."""
    edges = probe.band_edges(65, 65)
    index, count = probe.radial_band_index(65, 65, edges)
    assert count == len(edges) - 1
    assert index.shape == (65, 33)
    assert int(index.min()) >= 0 and int(index.max()) <= count - 1
    assert all(int((index == band).sum()) > 0 for band in range(count))


# --- decision-0105: the guard must be able to fail ------------------------------------

def test_zero_at_truth_is_exact(probe):
    """pred == target must score *exactly* zero, not approximately."""
    field = _smooth()
    result = probe.band_spectra(field, field, probe.band_edges(65, 65))
    for row in result["bands"]:
        assert row["psd_pred"] == row["psd_target"]
        assert row["amp_ratio"] == 1.0
        assert row["gap"] == 0.0
        assert row["coherence"] == pytest.approx(1.0)


def test_guard_sees_a_coarse_shift(probe):
    """Moving fine-band energy into the coarse band must be visible to the guard."""
    field = _smooth()
    edges = probe.band_edges(65, 65)
    moved = probe._coarse_only(field, edges)
    moved_power = float((moved ** 2).sum())
    truth_power = float((field ** 2).sum())
    assert moved_power < truth_power, "dropping bands must remove energy"
    assert probe._guard_sees_it(moved, field, [2, 3, 4, 5]) is True


def test_guard_rejects_an_inert_comparison(probe):
    """With no fine bands to compare the guard must refuse rather than pass vacuously."""
    field = _smooth()
    assert probe._guard_sees_it(field, field, []) is False


def test_p0_hard_gate_passes_on_a_real_grid(probe):
    """All four P0 conditions together, on the 65x65 grid the model actually uses."""
    result = probe.p0_checks(_smooth())
    assert result["parseval_ok"] is True
    assert result["dct_agrees_with_parseval"] is True
    assert result["zero_at_truth"] is True
    assert result["guard_sees_coarse_shift"] is True
    assert result["passes_hard_gate"] is True
    assert result["band_edges"][-1] == pytest.approx(math.sqrt(32.0 ** 2 + 32.0 ** 2))


# --- coherence and amplitude semantics ------------------------------------------------

def test_amplitude_ratio_is_nan_when_the_target_band_is_empty(probe):
    """An all-zero target band must not be scored as a perfect match."""
    prediction = _smooth()
    target = torch.zeros_like(prediction)
    result = probe.band_spectra(prediction, target, probe.band_edges(65, 65))
    for row in result["bands"]:
        assert row["amp_ratio"] != row["amp_ratio"] or row["amp_ratio"] == float("inf")
        assert row["gap"] != row["gap"] or row["gap"] == float("-inf")


def test_a_scaled_prediction_has_the_scaling_as_its_amplitude_ratio(probe):
    """``pred = c * target`` must give ``amp_ratio == c`` in every band, coherence 1."""
    target = _smooth()
    result = probe.band_spectra(3.0 * target, target, probe.band_edges(65, 65))
    for row in result["bands"]:
        assert row["amp_ratio"] == pytest.approx(3.0, rel=1e-12)
        assert row["coherence"] == pytest.approx(1.0)
        assert row["overshoot"] is True
        assert row["gap"] == pytest.approx(-2.0)


def test_shrunk_prediction_reports_a_positive_gap(probe):
    """A damped prediction is the case MSE is said to prefer, so its gap must be positive."""
    target = _smooth()
    result = probe.band_spectra(0.5 * target, target, probe.band_edges(65, 65))
    for row in result["bands"]:
        assert row["amp_ratio"] == pytest.approx(0.5, rel=1e-12)
        assert row["gap"] == pytest.approx(0.5)
        assert row["overshoot"] is False


# --- the refusal surface: Stage P has no training stages ------------------------------

def test_refused_workers_are_declared(probe):
    """A preflight that writes no protocol must refuse the training subcommands."""
    assert "prepare" in probe.REFUSED_WORKERS
    assert "archive" in probe.REFUSED_WORKERS
    assert "seed" in probe.REFUSED_WORKERS


@pytest.mark.parametrize("worker", ["prepare", "archive", "seed"])
def test_training_workers_exit_nonzero(probe, worker):
    """Reaching for the training machinery must fail before anything is written."""
    with pytest.raises(SystemExit) as caught:
        probe.main(["--worker", worker, "--stage", "p0"])
    assert caught.value.code not in (0, None)


def test_an_unknown_worker_also_exits(probe):
    with pytest.raises(SystemExit):
        probe.main(["--worker", "nonsense", "--stage", "p0"])


def test_the_probe_declares_no_scientific_claim(probe):
    """The record must carry the three standing flags the campaign requires."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source
    assert '"r": 0' in source


def test_the_thresholded_seam_ratio_is_not_thresholded(probe):
    """Decision 0110: the seam ratio is *reported*, never compared against a constant.

    A constant chosen after seeing the number is a fitted parameter, so the source must not
    contain a seam tolerance comparison.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "seam_energy_ratio" in source
    assert "seam_max" not in source and "natural_seam" not in source
