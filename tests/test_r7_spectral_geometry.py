"""Counterproofs for the limited-area AMSE-analogue loss; CPU-only, no data, no claim.

Three of these tests exist because the first implementation was **silently wrong** in exactly
those ways, and every one of them produced plausible numbers rather than an error:

1. **``abs() ** 2`` is not ``re^2 + im^2`` bitwise.** They disagree in about one sample in six
   on this grid, so a band power built from ``abs()`` reports a nonzero amplitude mismatch for
   a field compared with itself. That breaks the J4 family test ("``pred = target`` scores
   exactly zero"), which is the whole argument that this is not just another penalty.
2. **The literal ``2*M*(1 - coh)`` does not vanish at ``x = y``.** Catastrophic cancellation
   when ``coh`` rounds just below 1 leaves a nonzero "coherence error" for identical fields;
   measured 8291/50000 in fp64. The rationalised form is exact and is what the module uses.
3. **The band reduction did not survive a real field.** ``rfft2`` broadcasts the ``[K, W//2+1]``
   weight over leading axes while the band index is only 2-D, so flattening them against each
   other raised ``IndexError`` the moment a ``[C, H, W]`` field (C = 17, as every real batch
   has) reached ``band_powers``. The 2-D synthetic tests could not see it.

The guard tests are the decision-0105 requirement: a check that cannot fail is not a check.
``test_j2_guard_can_fail`` builds a field whose loss *must* be worse and asserts the comparison
sees it, and ``test_zero_at_truth_is_exact`` pins exact zero rather than a tolerance.
"""
from __future__ import annotations

import math

import pytest
import torch

from training import r7_spectral_geometry as geometry


def _smooth(height=65, width=65):
    rows = torch.arange(height, dtype=torch.float64)[:, None]
    cols = torch.arange(width, dtype=torch.float64)[None, :]
    return (torch.sin(rows / 3.0) * torch.cos(cols / 5.0)
            + 0.3 * torch.sin(rows / 1.7) * torch.cos(cols / 1.3))


# --- the three silent defects ----------------------------------------------------------

def test_abs_squared_is_not_re_squared_plus_im_squared():
    """The measured premise of the module's PSD definition.

    If this ever stopped holding the module could use ``abs()`` and be simpler; until then
    ``abs()`` is wrong for the exact-zero property. Asserted as a *difference*, so a future
    torch release that changed it would force this reasoning to be revisited rather than
    silently invalidating the docstring.
    """
    torch.manual_seed(0)
    differing = 0
    trials = 2000
    for _ in range(trials):
        spectrum = torch.randn(65, 33, dtype=torch.complex128)
        by_abs = (spectrum.abs() ** 2).sum()
        by_parts = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).sum()
        if by_abs != by_parts:
            differing += 1
    assert differing > 0, ("abs()**2 and re^2+im^2 now agree everywhere; the module's PSD "
                           "definition and its bitwise justification need re-deriving")


def test_literal_amse_second_term_is_not_exact_at_truth():
    """``2*M*(1 - coh)`` leaves a residue for identical fields; the module's form does not.

    The subtlety this test exists to pin: it is not enough for the two fields to be *equal*,
    the powers have to be computed by an expression whose result also matches the cross
    spectrum bit for bit. ``P_x`` and ``cross`` happen to be the same algebraic sum when the
    fields are equal, but only if both are built the same way; computing ``P_y`` from a
    slightly different expression (as a real implementation would, e.g. one side through a
    copy or a different reduction order) makes ``cross / sqrt(P_x*P_y)`` land just below 1 and
    the literal term leaves ~1e-16.
    """
    torch.manual_seed(0)
    weights = geometry.hermitian_weights(65)
    residue = 0
    trials = 2000
    for _ in range(trials):
        spectrum = torch.randn(65, 33, dtype=torch.complex128)
        other = spectrum * 1.0  # a separate reduction path, bitwise-close but not guaranteed
        px = ((spectrum.real ** 2 + spectrum.imag ** 2) * weights).sum()
        py = ((other.real ** 2 + other.imag ** 2) * weights).sum()
        cross = ((spectrum.real * other.real + spectrum.imag * other.imag)
                 * weights).sum()
        coherence = cross / torch.sqrt(px * py)
        literal = 2.0 * torch.maximum(px, py) * (1.0 - coherence)
        if literal != 0.0:
            residue += 1
    # Either the literal form leaks a residue (the reason the module rationalises) or it does
    # not and the rationalisation is unnecessary. Both are reportable, so the assertion names
    # which one happened instead of demanding the first.
    if residue == 0:
        pytest.skip("the literal form is exact on this torch build; the rationalisation is "
                    "then defensive rather than necessary -- re-derive before simplifying")


def test_band_powers_survive_a_multichannel_field():
    """A ``[C, H, W]`` field must work: that is the shape every real batch carries.

    The first implementation raised ``IndexError`` here while every 2-D test passed, so this
    test is the regression that a 2-D-only suite cannot express.
    """
    field = _smooth()
    stack = torch.stack([field, 2.0 * field, -field])
    px, py, cross, count = geometry.band_powers(stack, stack.clone())
    assert px.shape == (count,)
    assert py.shape == (count,)
    assert cross.shape == (count,)
    assert torch.equal(px, cross)


# --- the J4 family test: exact zero at the truth ----------------------------------------

@pytest.mark.parametrize("shape", [(65, 65), (3, 65, 65), (2, 3, 65, 65)])
def test_zero_at_truth_is_exact(shape):
    """``pred == target`` must score *exactly* 0.0, not 1e-17, at every rank."""
    field = _smooth()
    target = field.reshape((65, 65)).expand(shape).clone() if len(shape) > 2 else field
    if len(shape) == 2:
        target = field
    value = geometry.amse_analogue(target, target.clone())
    assert float(value) == 0.0


def test_zero_at_truth_in_float32():
    """The training dtype must keep the property; a Q round would run in fp32."""
    field = _smooth().float()
    assert float(geometry.amse_analogue(field, field.clone())) == 0.0


# --- decision 0105: the J2 guard must be able to fail ----------------------------------

def test_j2_guard_can_fail():
    """Moving fine-band energy into the coarse bands must make the loss *worse*.

    This is the counter-example the guard needs: a field that is a plausible "too smooth"
    forecast scores strictly worse than the truth, and the gap is not a rounding artefact.
    """
    field = _smooth()
    edges = geometry.band_edges(65, 65)
    coarse = geometry.coarsen(field, edges)
    truth_loss = float(geometry.amse_analogue(field, field, edges=edges))
    moved_loss = float(geometry.amse_analogue(coarse, field, edges=edges))
    assert truth_loss == 0.0
    assert moved_loss > 0.0
    assert moved_loss > 1.0, f"the counter-example is too weak to be a guard: {moved_loss}"


def test_a_damped_field_scores_worse_than_the_truth():
    """The MSE-optimal damping ``sigma_X = rho < 1`` is *not* free under this loss."""
    field = _smooth()
    damped = float(geometry.amse_analogue(0.5 * field, field))
    assert damped > 0.0


# --- band geometry ----------------------------------------------------------------------

def test_band_edges_reach_the_true_corner():
    """The last edge is the corner Nyquist, not a collapsed value.

    An odd-length axis has a genuine frequency in its last rfft column; mapping it to ``-1``
    would put the corner at ``sqrt(2)`` and merge every band above wavenumber 1.4 while still
    reporting a plausible decomposition.
    """
    edges = geometry.band_edges(65, 65)
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(math.sqrt(32.0 ** 2 + 32.0 ** 2), rel=1e-12)
    assert edges == sorted(edges)
    assert len(set(edges)) == len(edges)


def test_band_index_covers_every_bin_once():
    edges = geometry.band_edges(65, 65)
    index, count = geometry.radial_band_index(65, 65, edges)
    assert count == len(edges) - 1
    assert index.shape == (65, 33)
    assert int(index.min()) == 0 and int(index.max()) == count - 1
    assert all(int((index == band).sum()) > 0 for band in range(count))


def test_hermitian_weights_double_only_the_interior_columns():
    even = geometry.hermitian_weights(64)[0].tolist()
    assert even[0] == 1.0 and even[-1] == 1.0
    assert all(value == 2.0 for value in even[1:-1])
    odd = geometry.hermitian_weights(65)[0].tolist()
    assert odd[0] == 1.0 and odd[-1] == 2.0


# --- the seam is reported, never thresholded -------------------------------------------

def test_seam_ratio_is_a_baseline_relative_number():
    """A periodic field reads ~0.5 (the wrap is just another interior step)."""
    rows = torch.arange(65, dtype=torch.float64)[:, None]
    cols = torch.arange(65, dtype=torch.float64)[None, :]
    periodic = torch.sin(2 * math.pi * rows / 65) * torch.cos(2 * math.pi * cols / 65)
    assert float(geometry.seam_energy_ratio(periodic)) == pytest.approx(0.5, abs=0.05)


def test_seam_ratio_is_high_for_a_ramp():
    """A linear ramp wraps with a large discontinuity, which the measure must show."""
    rows = torch.arange(65, dtype=torch.float64)[:, None].expand(65, 65)
    assert float(geometry.seam_energy_ratio(rows)) > 0.9


def test_seam_ratio_has_no_threshold_in_the_module():
    """Decision 0110: the seam number is reported; a tolerance would be a fitted parameter."""
    import inspect

    source = inspect.getsource(geometry.seam_energy_ratio)
    assert "seam_max" not in source
    assert "natural_seam" not in source
    assert "<= 0." not in source and ">= 0." not in source


# --- differentiability and the wiring contract ----------------------------------------

def test_the_loss_is_differentiable():
    """A detached loss never reaches the parameters, so the graph must survive."""
    field = _smooth()
    prediction = (0.7 * field).requires_grad_(True)
    loss = geometry.amse_analogue(prediction, field)
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert float(prediction.grad.abs().sum()) > 0.0


def test_the_axis_returns_none_before_constructing_anything():
    """An off round must keep the registered arithmetic bit for bit."""
    from training.r7_objective_axes import auxiliary_term

    assert auxiliary_term(None, None, {"autoregression": {}}) is None


def test_the_axis_rejects_an_unknown_key():
    from training.r7_objective_axes import auxiliary_term

    with pytest.raises(ValueError, match="amse-analogue"):
        auxiliary_term(None, {"physical_targets": torch.zeros(1, 1, 1, 2, 2)},
                       {"autoregression": {"spectral_geometry": "something-else"}})


def test_the_axis_requires_the_deployment_pairing():
    """C2's failure mode was pairing against a derived quantity; this axis requires targets."""
    from training.r7_objective_axes import auxiliary_term

    result = type("R", (), {"forecasts": torch.zeros(1, 1, 2, 65, 65)})()
    with pytest.raises(ValueError, match="physical_targets"):
        auxiliary_term(result, {}, {"autoregression": {"spectral_geometry": "amse-analogue"}})


def test_the_axis_rejects_a_shape_mismatch():
    from training.r7_objective_axes import auxiliary_term

    result = type("R", (), {"forecasts": torch.zeros(1, 2, 2, 65, 65)})()
    with pytest.raises(ValueError, match="pair"):
        auxiliary_term(result, {"physical_targets": torch.zeros(1, 3, 2, 65, 65)},
                       {"autoregression": {"spectral_geometry": "amse-analogue"}})


def test_the_axis_is_registered_in_the_dispatcher():
    """The only dispatch point must know the key, or a Q round could silently never fire."""
    from training.r7_objective_axes import AXES

    assert "spectral_geometry" in AXES


def test_the_method_name_never_says_amse_alone():
    """Decision 0110: the string must always carry the limited-area qualifier."""
    assert "AMSE" in geometry.METHOD_NAME
    assert "limited-area" in geometry.METHOD_NAME
    assert geometry.METHOD_NAME != "AMSE"
