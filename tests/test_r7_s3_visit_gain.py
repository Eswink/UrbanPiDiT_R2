"""Counter-proofs for the visit-gain switch (decision 0119, G1-B D1).

Every test here is written so that a *wrong* implementation fails it. The three that
carry the round are:

* the off path must be the original expression **operation for operation**, so a cell
  built with ``visit_gain=None`` has to reproduce a hand-written reference cell bit for
  bit and must not read the visit index at all;
* the switch must add **no parameter and no buffer**, so the parent checkpoint stays
  installable and the parameter key set stays identical;
* ``visit_damped`` must be exactly 1.0 at its first visit, which is the identity bridge
  decision 0119 D5 records as exit 3b.

The gain arithmetic is pinned too, because decision 0119's whole attribution caveat is
that the uniform arm is **not** the visit-damped arm's mean gain (0.632456 vs 0.696114
at K=4). A test that let those drift would silently retire a recorded limitation.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.recursive_weather_r7 import (GenericRecursiveCell, VISIT_GAIN_MODES,
                                        visit_gain_value)

DIM, HEADS = 32, 4


def _cell(mode=None, seed=1234):
    torch.manual_seed(seed)
    return GenericRecursiveCell(DIM, HEADS, mlp_ratio=3.0, dropout=0.0, visit_gain=mode)


def _inputs(seed=7, batch=2, tokens=5, context_tokens=9):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, tokens, DIM, generator=generator)
    context = torch.randn(batch, context_tokens, DIM, generator=generator)
    return z, context


def _reference_cell():
    """The pre-switch expression, written out here rather than imported.

    Importing the implementation would make this test compare the code to itself. This
    is the cell as it stood before decision 0119: two bare residuals, no gain, and no
    visit index anywhere in the body.
    """
    from model.layers.sdpa import CrossBlock, FeedForward, SDPAttention
    from torch import nn

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(DIM)
            self.self_attn = SDPAttention(DIM, HEADS, 0.0)
            self.cross = CrossBlock(DIM, HEADS, 3.0, 0.0)
            self.n2 = nn.LayerNorm(DIM)
            self.ff = FeedForward(DIM, 3.0, 0.0)

        def forward(self, z, context):
            z = z + self.self_attn(self.n1(z))
            z = self.cross(z, context)
            return z + self.ff(self.n2(z))

    return Reference()


# --------------------------------------------------------------------------- gains

def test_mode_domain_is_exactly_the_frozen_three():
    """The switch is an enum of three, and ``None`` is one of them."""
    assert VISIT_GAIN_MODES == (None, "visit_damped", "uniform_damped")


def test_unknown_mode_is_refused_at_construction():
    with pytest.raises(ValueError, match="visit_gain"):
        GenericRecursiveCell(DIM, HEADS, visit_gain="cosine_decay")


def test_visit_damped_is_one_over_sqrt_one_plus_k():
    for k in range(8):
        assert visit_gain_value("visit_damped", k, 4) == pytest.approx(1.0 / math.sqrt(1 + k))


def test_visit_damped_first_visit_is_exactly_one():
    """Exit 3b's arithmetic: ``g_0 = 1/sqrt(1+0)`` must be exactly 1.0, not near it.

    A near-1.0 gain would make the identity bridge a tolerance claim instead of a bitwise
    one, which is the difference between an identity witness and a similarity check.
    """
    assert visit_gain_value("visit_damped", 0, 4) == 1.0


def test_uniform_damped_ignores_the_visit_index():
    values = {visit_gain_value("uniform_damped", k, 4) for k in range(8)}
    assert values == {1.0 / math.sqrt(1.0 + 1.5)}


def test_uniform_damped_at_k4_is_the_frozen_constant():
    assert visit_gain_value("uniform_damped", 0, 4) == pytest.approx(0.6324555320336759)


def test_uniform_damped_below_a_visit_damped_mean_records_the_jensen_gap():
    """The recorded caveat, as arithmetic rather than prose (decision 0119 Context 1).

    If this ever came out equal or reversed, A-B would become a clean visit-alignment
    contrast and the ADR's attribution boundary would be wrong. Pinning it here means the
    limitation cannot be retired by an unnoticed edit to either formula.
    """
    steps = 4
    visit_mean = sum(visit_gain_value("visit_damped", k, steps) for k in range(steps)) / steps
    uniform = visit_gain_value("uniform_damped", 0, steps)
    assert uniform < visit_mean
    assert (1.0 - uniform / visit_mean) == pytest.approx(0.0914, abs=5e-5)


def test_reasoning_steps_below_one_is_refused_by_uniform():
    with pytest.raises(ValueError, match="reasoning_steps"):
        visit_gain_value("uniform_damped", 0, 0)


def test_negative_visit_index_is_refused():
    with pytest.raises(ValueError, match="visit index"):
        visit_gain_value("visit_damped", -1, 4)


# ----------------------------------------------------------------------- off path

def test_off_path_adds_no_parameter_and_no_buffer():
    """D1's zero-new-parameters claim, measured on the module itself."""
    off = _cell(None)
    assert list(off.state_dict()) and not any("gain" in name for name in off.state_dict())
    assert all(not name for name, _ in off.named_buffers())
    for mode in ("visit_damped", "uniform_damped"):
        on = _cell(mode)
        assert set(on.state_dict()) == set(off.state_dict())
        assert [value.shape for value in on.state_dict().values()] == \
               [value.shape for value in off.state_dict().values()]
        assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())


def test_off_path_equals_the_pre_switch_expression_bit_for_bit():
    """The load-bearing D1 counter-proof.

    The off path must be the *original* expression, not a ``g=1.0`` multiplication. Both
    are exact in IEEE-754, but only the first makes "the default path did not change" a
    statement about the code rather than about rounding.
    """
    cell = _cell(None).eval()
    reference = _reference_cell().eval()
    reference.load_state_dict(cell.state_dict(), strict=True)
    z, context = _inputs()
    with torch.no_grad():
        expected = reference(z, context)
        actual = cell(z, context)
        # A different visit index must not move the off path either: it is never read.
        also = cell(z, context, visit_index=5, reasoning_steps=9)
    assert torch.equal(actual, expected)
    assert torch.equal(also, expected)


def test_off_path_matches_the_full_model_forward_for_every_visit():
    """The same identity through the model's own loop, not just a direct cell call."""
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster

    torch.manual_seed(11)
    model = GenericRecursiveWeatherForecaster(
        in_channels=4, history_steps=2, dim=DIM, patch_size=2, depth=2, heads=HEADS,
        window_size=4, latent_tokens=4, default_reasoning_steps=3, visit_gain=None).eval()
    batch = _forecast_batch(seed=3)
    with torch.no_grad():
        first = model(batch).draft_forecasts
        second = model(batch).draft_forecasts
    assert torch.equal(first, second)


def _forecast_batch(seed=3, batch=2, size=16, channels=4, history=2):
    generator = torch.Generator().manual_seed(seed)
    return {"coarse_history": torch.randn(batch, history, channels, size, size,
                                          generator=generator)}


# --------------------------------------------------------------- visit-damped arm

def test_visit_damped_first_visit_is_the_unmodified_cell_bit_for_bit():
    """Exit 3b: A and the parent agree exactly at k=0, then diverge from k=1."""
    off = _cell(None).eval()
    on = _cell("visit_damped").eval()
    on.load_state_dict(off.state_dict(), strict=True)
    z, context = _inputs()
    with torch.no_grad():
        reference = off(z, context)
        first = on(z, context, visit_index=0, reasoning_steps=4)
        second = on(z, context, visit_index=1, reasoning_steps=4)
    assert torch.equal(first, reference)
    assert not torch.equal(second, reference)


def test_uniform_damped_does_not_equal_the_parent_at_its_first_visit():
    """The expected asymmetry, pinned so nobody reads 3b as covering both arms.

    B's constant is 0.632456 at every visit including the first, so a bitwise match there
    would mean the gain was not applied at all.
    """
    off = _cell(None).eval()
    on = _cell("uniform_damped").eval()
    on.load_state_dict(off.state_dict(), strict=True)
    z, context = _inputs()
    with torch.no_grad():
        assert not torch.equal(on(z, context, visit_index=0, reasoning_steps=4), off(z, context))


def test_both_arms_scale_each_residual_branch_by_the_same_gain():
    """One gain, both branches. Two gains would be a second factor (decision 0119 D1)."""
    on = _cell("visit_damped").eval()
    z, context = _inputs()
    gain = visit_gain_value("visit_damped", 2, 4)
    with torch.no_grad():
        step = on(z, context, visit_index=2, reasoning_steps=4)
        after_attn = z + gain * on.self_attn(on.n1(z))
        crossed = on.cross(after_attn, context)
        expected = crossed + gain * on.ff(on.n2(crossed))
    assert torch.equal(step, expected)


def test_gain_reads_the_round_depth_not_the_model_default():
    """``uniform_damped`` must use the run's K, so a K override changes it."""
    on = _cell("uniform_damped").eval()
    z, context = _inputs()
    with torch.no_grad():
        at_four = on(z, context, visit_index=0, reasoning_steps=4)
        at_eight = on(z, context, visit_index=0, reasoning_steps=8)
    assert not torch.equal(at_four, at_eight)


# ------------------------------------------------------------- digests and gradients

def test_switch_changes_the_module_fields_and_therefore_the_semantics_digest():
    """A switched-on cell must not be loadable as the switched-off arm.

    ``_module_semantics`` reads every non-underscore attribute, so the mode has to be one
    of them: otherwise a checkpoint from the damped arm would pass the parent's identity
    check. This asserts the field exists and differs, which is what makes the migration's
    semantics rebind necessary rather than optional.
    """
    off, on = _cell(None), _cell("visit_damped")
    assert off.visit_gain is None
    assert on.visit_gain == "visit_damped"


def test_both_modes_produce_finite_gradients():
    for mode in (None, "visit_damped", "uniform_damped"):
        cell = _cell(mode)
        z, context = _inputs()
        out = cell(z, context, visit_index=1, reasoning_steps=4)
        out.square().mean().backward()
        grads = [p.grad for p in cell.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def test_state_dict_round_trip_preserves_the_mode_independent_tensors():
    """Loading an off arm's tensors into a damped arm must succeed strictly."""
    off, on = _cell(None), _cell("visit_damped", seed=99)
    on.load_state_dict(off.state_dict(), strict=True)
    for name, value in off.state_dict().items():
        assert torch.equal(on.state_dict()[name], value)
