"""Round-two D1/D4: the pooled-query switch is a capacity control, and it is pinned.

The round-two question is whether the *position dependence* of the RW-A read, rather
than the extra capacity the RW-A module brings, is what changes a trained run. The
control arm therefore has to be the same module with the same parameters reading the
same keys - with only the query's position dependence removed. The tests here state
that in the two directions that matter:

- `test_the_pooled_query_reads_the_same_vector_at_every_position` (D4): with the
  switch on, every output position is **bitwise** identical, and perturbing one
  process token moves every position by exactly the same amount. With the switch
  off, the positions differ bitwise and the perturbation response is uneven - so the
  test cannot pass for a readout that ignores the switch in either direction.
- `test_the_pooled_query_ignores_the_token_grid_shape`: the mean is taken *before*
  the position encoding is added, so reshaping the same 12 tokens from 3x4 to 2x6
  cannot move the pooled read while it does move the positional one. Pooling after
  the encoding would leave the encoding's mean inside the control arm and fail here.

The remaining tests cover what D1 requires of the switch itself: `bool`-only
construction in the style of the existing switches, no parameter - and so no
checkpoint key - added by turning it on, and a refusal to accept a pooled query
without the positional readout that owns it (a silently ignored switch is a
configuration error, not a control).
"""
from __future__ import annotations

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_readout_r7 import PositionalProcessReadout
from training.r7_experiment import make_model, seed_everything

DIM = 16
PROCESSES = 4
TOKEN_HW = (3, 4)
TOKENS = TOKEN_HW[0] * TOKEN_HW[1]
BASE = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": DIM, "depth": 1,
        "heads": 2, "window_size": 2, "patch_size": 2, "dropout": 0.0,
        "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 1,
        "positional_process_readout": True}


def _readout(*, pooled: bool, seed: int = 3) -> PositionalProcessReadout:
    torch.manual_seed(seed)
    return PositionalProcessReadout(DIM, heads=2, pooled_readout_query=pooled)


def _tensors(seed: int = 4):
    generator = torch.Generator().manual_seed(seed)
    process = torch.randn(1, PROCESSES, DIM, generator=generator)
    context = torch.randn(1, TOKENS, DIM, generator=generator)
    return process, context


def test_the_pooled_query_reads_the_same_vector_at_every_position():
    """D4 (pooled direction): all positions bitwise equal, as is the response."""
    pooled = _readout(pooled=True)
    process, context = _tensors()
    read = pooled(process, context, TOKEN_HW)
    assert read.shape == (1, TOKENS, DIM)
    for position in range(1, TOKENS):
        assert torch.equal(read[:, 0], read[:, position]), position
    delta = torch.zeros_like(process)
    delta[:, 1] = 1.0
    response = (pooled(process + delta, context, TOKEN_HW)
                - pooled(process, context, TOKEN_HW)).detach()
    assert float(response.abs().max()) > 0.0, "the pooled read stopped seeing the tokens"
    for position in range(1, TOKENS):
        assert torch.equal(response[:, 0], response[:, position]), position


def test_the_positional_query_differs_across_positions():
    """D4 (other direction): the unpooled read must not be flat, or the arms collapse."""
    positional = _readout(pooled=False)
    process, context = _tensors()
    read = positional(process, context, TOKEN_HW)
    differences = [position for position in range(1, TOKENS)
                   if not torch.equal(read[:, 0], read[:, position])]
    assert differences, "every output position is identical without pooling"
    delta = torch.zeros_like(process)
    delta[:, 1] = 1.0
    response = (positional(process + delta, context, TOKEN_HW)
                - positional(process, context, TOKEN_HW)).detach().norm(dim=-1)[0]
    assert float(response.max() - response.min()) > 0.0


def test_the_pooled_query_ignores_the_token_grid_shape():
    """Pooled before the encoding: the same tokens on another grid cannot move it."""
    process, context = _tensors()
    for pooled, expect_equal in ((True, True), (False, False)):
        readout = _readout(pooled=pooled)
        square = readout(process, context, (3, 4))
        wide = readout(process, context, (2, 6))
        assert torch.equal(square, wide) is expect_equal, pooled


def test_the_switch_is_bool_only():
    """Same construction style as the model's other switches."""
    for bad in (1, 0, "pooled", None):
        with pytest.raises(ValueError):
            PositionalProcessReadout(DIM, heads=2, pooled_readout_query=bad)
    with pytest.raises(ValueError):
        make_model("process", dict(BASE, pooled_readout_query=1))


def test_a_pooled_query_without_the_positional_readout_is_refused():
    """The switch cannot be silently ignored: it owns no readout on its own."""
    without_readout = {name: value for name, value in BASE.items()
                       if name != "positional_process_readout"}
    with pytest.raises(ValueError):
        make_model("process", dict(without_readout, pooled_readout_query=True))


def test_the_pooled_switch_adds_no_parameter():
    """D1: the control arm is the same module with the same parameter set.

    Turned on, it must not add, drop or reorder a single tensor - otherwise the
    round-two positional and pooled arms would differ in capacity as well as in
    position dependence, exactly the confound the round-one run could not remove.
    """
    seed_everything(11)
    positional = make_model("process", dict(BASE))
    seed_everything(11)
    pooled = make_model("process", dict(BASE, pooled_readout_query=True))
    assert isinstance(pooled, ProcessForecastCoReasoner)
    # Default off: the switch exists but neither the model nor its reader turns it on.
    assert positional.pooled_readout_query is False
    assert positional.process_reader.pooled_readout_query is False
    assert pooled.pooled_readout_query is True
    assert pooled.process_reader.pooled_readout_query is True
    before, after = positional.state_dict(), pooled.state_dict()
    assert set(before) == set(after)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert sum(p.numel() for p in positional.parameters()) == sum(
        p.numel() for p in pooled.parameters())


def test_the_pooled_read_is_position_independent_through_the_model_read():
    """The same statement at the model's own read, which is what the arms train."""
    seed_everything(13)
    model = make_model("process", dict(BASE, pooled_readout_query=True)).eval()
    process, context = _tensors()
    with torch.no_grad():
        read = model.process_conditioning(process, context, TOKEN_HW)
    assert read.ndim == 3 and read.shape[-2] == TOKENS
    for position in range(1, TOKENS):
        assert torch.equal(read[:, 0], read[:, position]), position
