"""#72 M2-B: the local gated solver state, its boundaries and its capacity.

``docs/R7_MAIN_MODEL_V2_DESIGN.md`` section 3 freezes the RW-B equations and
section 3.1 names the three boundaries that have to hold for the result to *be*
RW-B rather than a renamed version of what was already measured negative. Each
boundary has a test here, and the tests that assert a property also carry the
counterproof that the assertion is not vacuous:

- the gate may not cut the proposal's gradient - asserted by a non-zero gradient,
  and shown to be a real measurement by saturating the gate and watching that
  gradient collapse;
- the proposal is anchored to ``X_t``, not accumulated on the draft - asserted by
  moving the anchor and watching the proposal move with it;
- ``spatial_solver_feedback`` is not re-opened as a new method - it is a separate
  0-parameter switch and stays off in every RW-B configuration built here.

The capacity question is a declared engineering target, not a scientific one: the
design contract asks for an increment no larger than a quarter of the main model,
and the last test measures it at the audited ``dim=192`` configuration rather than
at the small test size, because a ratio measured on a toy model says nothing about
the model that would be trained.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.local_solver_state_r7 import (GATE_INITIAL_PROBABILITY, PositionGate,
    blend_forecast, expand_token_gate, step_encoding)
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import forecast_inputs

SMALL = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": 16,
         "depth": 1, "heads": 2, "window_size": 4, "patch_size": 2, "dropout": 0.0,
         "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 2,
         "spacetime_inputs": True, "positional_process_readout": True}
RW_B = dict(SMALL, local_solver_state=True)
ROLES = dict(SMALL, source_role_markers=True)
BOTH = dict(SMALL, source_role_markers=True, local_solver_state=True)
# The same model with the new switches *named* and set to False: "off" has to mean
# the previous implementation whether the caller left the argument out or wrote the
# default down explicitly.
OFF = dict(SMALL, source_role_markers=False, local_solver_state=False)


def model_with(**switches) -> ProcessForecastCoReasoner:
    torch.manual_seed(11)
    return ProcessForecastCoReasoner(**switches)


def batch(hw=(6, 6), size=2) -> dict:
    generator = torch.Generator().manual_seed(5)
    rows, columns = hw
    return {
        "coarse_history": torch.randn(size, 2, 3, rows, columns, generator=generator),
        "atmos_target": torch.randn(size, 3, rows, columns, generator=generator),
        "lead_time_hours": torch.full((size,), 6.0),
        "latitude": torch.linspace(-30.0, 30.0, rows),
        "longitude": torch.linspace(0.0, 40.0, columns),
        "init_utc_hour": torch.full((size,), 3.0),
        "init_day_of_year": torch.full((size,), 40.0),
    }


def state_keys(model: ProcessForecastCoReasoner) -> set[str]:
    return set(model.state_dict())


def rw_b_parameter_names(model: ProcessForecastCoReasoner) -> set[str]:
    return {name for name in state_keys(model)
            if name.startswith(("solver_init", "solver_cell", "solver_gate", "proposal_head"))}


def test_the_new_switches_are_boolean_only():
    for name in ("source_role_markers", "local_solver_state"):
        with pytest.raises(ValueError, match=name):
            model_with(**dict(SMALL, **{name: 1}))


def test_local_solver_state_demands_the_positional_read_and_draft_feedback():
    """A substitution nobody asked for is worse than a refusal."""
    with pytest.raises(ValueError, match="positional_process_readout"):
        model_with(**dict(SMALL, positional_process_readout=False, local_solver_state=True))
    with pytest.raises(ValueError, match="use_forecast_feedback"):
        model_with(**dict(SMALL, use_forecast_feedback=False, local_solver_state=True))


def test_switches_off_add_no_parameter_and_no_state_key():
    """Off means the previous model, down to the tensors it publishes."""
    plain = model_with(**SMALL)
    other = model_with(**OFF)
    assert state_keys(other) == state_keys(plain)
    assert sum(p.numel() for p in other.parameters()) == sum(
        p.numel() for p in plain.parameters())
    for name, tensor in plain.state_dict().items():
        assert torch.equal(tensor, other.state_dict()[name]), name


def test_switches_off_reproduce_the_forward_bit_for_bit():
    plain = model_with(**SMALL).eval()
    with torch.no_grad():
        expected = plain(forecast_inputs(batch())).forecast
        assert torch.equal(model_with(**OFF).eval()(
            forecast_inputs(batch())).forecast, expected)


def test_turning_a_switch_on_marks_exactly_its_own_parameters():
    """The added tensors must be the named ones, not a reshuffle of the old ones."""
    plain = set(model_with(**SMALL).state_dict())
    assert set(model_with(**ROLES).state_dict()) - plain == {"role_context", "role_draft"}
    added = set(model_with(**RW_B).state_dict()) - plain
    assert all(name.startswith(("solver_init", "solver_cell", "solver_gate",
                                "proposal_head")) for name in added), added


def test_the_solver_state_is_one_vector_per_patch_and_is_returned():
    model = model_with(**RW_B).eval()
    with torch.no_grad():
        out = model(forecast_inputs(batch()))
    assert out.solver_state is not None
    assert out.solver_state.shape == out.context_tokens.shape


def test_without_the_switch_no_solver_state_is_reported():
    model = model_with(**SMALL).eval()
    with torch.no_grad():
        assert model(forecast_inputs(batch())).solver_state is None


def test_k_zero_returns_the_initial_forecast_and_no_state():
    model = model_with(**RW_B).eval()
    with torch.no_grad():
        out = model(forecast_inputs(batch()), reasoning_steps=0)
    assert torch.equal(out.forecast, out.initial_forecast)
    assert out.solver_state is None
    assert out.final_correction.abs().max() == 0


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_forward_runs_at_each_depth_and_the_parameter_count_never_moves(steps):
    model = model_with(**RW_B).eval()
    before = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        out = model(forecast_inputs(batch()), reasoning_steps=steps)
    assert out.draft_forecasts.shape[1] == steps + 1
    assert sum(p.numel() for p in model.parameters()) == before


def test_every_rw_b_parameter_receives_a_nonzero_gradient():
    """A parameter with no gradient is a parameter the run cannot shape."""
    model = model_with(**RW_B).train()
    model.zero_grad()
    model(forecast_inputs(batch())).forecast.square().mean().backward()
    names = rw_b_parameter_names(model)
    assert names, "the configuration built no RW-B parameters"
    missing = [name for name, p in model.named_parameters()
               if name in names and (p.grad is None or not torch.isfinite(p.grad).all()
                                     or p.grad.abs().max() == 0)]
    assert not missing, f"parameters without a usable gradient: {missing}"


def test_the_proposal_carries_gradient_through_the_gate_and_the_test_can_fail():
    """The boundary, plus the counterproof that it is measured and not assumed."""
    def proposal_gradient(saturation: float) -> float:
        model = model_with(**RW_B).train()
        with torch.no_grad():
            model.solver_gate.score.bias.fill_(saturation)
        model.zero_grad()
        model(forecast_inputs(batch())).forecast.square().mean().backward()
        grads = [p.grad.abs().max().item() for name, p in model.named_parameters()
                 if name.startswith("proposal_head")]
        assert grads, "no proposal parameters to measure"
        return max(grads)

    gentle = proposal_gradient(math.log(GATE_INITIAL_PROBABILITY / (1 - GATE_INITIAL_PROBABILITY)))
    assert gentle > 0, "the gentle gate cut the proposal's gradient"
    assert proposal_gradient(-50.0) < gentle * 1e-6, (
        "saturating the gate did not collapse the proposal gradient, so the "
        "non-zero result above does not show anything about the gate")


def test_the_gate_starts_gentle_and_never_saturates():
    gate = PositionGate(dim=64)
    torch.manual_seed(3)
    values = gate(torch.randn(4, 25, 64))
    assert values.shape == (4, 25, 1)
    mean = values.mean().item()
    assert 0.5 * GATE_INITIAL_PROBABILITY < mean < 1.5 * GATE_INITIAL_PROBABILITY, mean
    assert 0.0 < values.min().item() and values.max().item() < 1.0
    assert torch.isfinite(values).all()


def test_the_gate_can_differ_between_positions():
    """A gate that is one scalar for the whole grid is not a per-position gate."""
    model = model_with(**RW_B).eval()
    with torch.no_grad():
        out = model(forecast_inputs(batch()))
        z = out.solver_state
        gate = model.solver_gate(z)
    assert gate.shape[:2] == z.shape[:2]
    assert gate.std().item() > 0


def test_the_expanded_gate_replicates_each_token_over_its_own_patch():
    """Nearest replication plus the decoder's crop, asserted against a manual build."""
    token_hw, patch_size, output_hw = (3, 4), 2, (5, 7)
    gate = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    expanded = expand_token_gate(gate, token_hw, output_hw, patch_size)
    assert expanded.shape == (1, 1, 5, 7)
    grid = gate.reshape(1, 3, 4)
    for row in range(output_hw[0]):
        for column in range(output_hw[1]):
            assert expanded[0, 0, row, column] == grid[0, row // patch_size, column // patch_size]


def test_the_blend_is_exactly_the_documented_equation():
    generator = torch.Generator().manual_seed(7)
    draft = torch.randn(2, 3, 5, 5, generator=generator)
    proposal = torch.randn(2, 3, 5, 5, generator=generator)
    gate = torch.rand(2, 1, 5, 5, generator=generator)
    expected = draft + gate * (proposal - draft)
    assert torch.equal(blend_forecast(draft, proposal, gate), expected)
    # The endpoints are limits, not identities: ``d + 1*(p - d)`` rounds to ``p``
    # only to within an ulp of the intermediate subtraction.
    ones = torch.ones_like(gate)
    assert torch.allclose(blend_forecast(draft, proposal, ones), proposal, atol=1e-6)
    assert torch.equal(blend_forecast(draft, proposal, torch.zeros_like(gate)), draft)


def test_the_blend_refuses_a_gate_of_the_wrong_shape():
    draft = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError, match="expanded gate"):
        blend_forecast(draft, draft, torch.zeros(1, 3, 4, 4))


def test_the_proposal_moves_with_the_anchor_it_is_decoded_against():
    """Anchoring, asserted directly: shift X_t and the proposal shifts with it."""
    model = model_with(**RW_B).eval()
    shifted = batch()
    with torch.no_grad():
        out = model(forecast_inputs(shifted))
        z = out.solver_state
        anchor = out.initial_forecast - out.final_correction
        other = model.proposal_head(z, out.token_hw, out.forecast.shape[-2:], anchor + 1.0)
        base = model.proposal_head(z, out.token_hw, out.forecast.shape[-2:], anchor)
    assert torch.allclose(other[0] - base[0], torch.ones_like(other[0]), atol=1e-5)
    assert torch.equal(other[1], base[1]), "the tendency must not depend on the anchor"


def test_the_anchored_proposal_is_required_when_the_state_is_on():
    from model.process_step_r7 import process_reasoning_step
    model = model_with(**RW_B).eval()
    with torch.no_grad():
        base = model.backbone(forecast_inputs(batch()))
        process = model.process_queries.expand(2, -1, -1)
        with pytest.raises(ValueError, match="anchor"):
            process_reasoning_step(model, process, base.context_tokens,
                                   base.forecast, base.token_hw)


@pytest.mark.parametrize("hw", [(5, 7), (7, 5), (1, 3), (3, 1)])
def test_odd_and_nondivisible_grids_round_trip(hw):
    model = model_with(**RW_B).train()
    data = batch(hw=hw, size=1)
    out = model(forecast_inputs(data))
    assert out.forecast.shape[-2:] == hw
    assert out.solver_state.shape[1] == math.ceil(hw[0] / 2) * math.ceil(hw[1] / 2)
    out.forecast.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_step_encoding_is_fixed_and_distinguishes_depth():
    first = step_encoding(0, 16, device=torch.device("cpu"), dtype=torch.float32)
    again = step_encoding(0, 16, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(first, again)
    assert not torch.equal(step_encoding(1, 16, device=torch.device("cpu"),
                                         dtype=torch.float32), first)
    with pytest.raises(ValueError):
        step_encoding(-1, 16, device=torch.device("cpu"), dtype=torch.float32)


def test_capacity_increment_at_the_audited_configuration():
    """The design contract's 25% target, measured at dim=192 - the audited size."""
    torch.manual_seed(13)
    audited = {"in_channels": 17, "out_channels": 17, "history_steps": 2, "dim": 192,
               "depth": 4, "heads": 4, "window_size": 4, "patch_size": 2, "dropout": 0.0,
               "anchored_processes": 8, "free_processes": 8, "default_reasoning_steps": 3,
               "use_forecast_feedback": True, "spacetime_inputs": True,
               "positional_process_readout": True}
    rw_a = ProcessForecastCoReasoner(**audited)
    rw_b = ProcessForecastCoReasoner(**dict(audited, local_solver_state=True))
    before = sum(p.numel() for p in rw_a.parameters())
    after = sum(p.numel() for p in rw_b.parameters())
    increment = after - before
    assert before == 2_968_259, f"the audited RW-A arm changed size: {before}"
    assert increment > 0
    assert increment <= 0.25 * before, (
        f"RW-B adds {increment} parameters ({increment / before:.1%} of the main model), "
        "above the frozen 25% target; shrink the module hidden before freezing")


def test_the_retired_spatial_feedback_stays_off_in_every_rw_b_configuration():
    """C2 measured this switch negative; RW-B must not be it under a new name."""
    for switches in (SMALL, ROLES, RW_B, BOTH):
        assert model_with(**switches).spatial_solver_feedback is False
