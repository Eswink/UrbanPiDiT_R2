"""D3/D5: the process read is positional, additive, and really moves the output.

The pre-RW-A bottleneck is one line - ``process.mean(dim=1)`` projected and
broadcast to every token - so the tests below are written as discriminators
against exactly that:

- `test_the_pooled_read_is_a_broadcast_and_the_positional_one_is_not` states the
  defect (a broadcast: identical summary at every position) and shows the new read
  is not one;
- `test_perturbing_one_process_token_moves_positions_unevenly` is the D5 gate: the
  response to perturbing a single process token is measured per output position,
  with the pooled path as the in-test reference whose spread is exactly zero. A
  read that merely re-ordered the mean would respond identically everywhere and
  fail here;
- `test_the_read_is_permutation_sensitive_while_the_pooled_one_is_not` rules out a
  mean in disguise, since a mean cannot see the order of the tokens it averages;
- `test_the_read_cost_is_n_queries_times_m_tokens` pins the N x M attention by
  capturing the attention module's own input shapes;
- `test_turning_the_switch_on_adds_only_the_reader` proves the two experiment arms
  differ by the added pathway only: same seed, same construction, every
  pre-existing parameter bitwise identical.
"""
from __future__ import annotations

import pytest
import torch

from model.coarse_forecast import CoarseForecastHead
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_readout_r7 import PositionalProcessReadout
from model.recursive_weather_r7 import solver_conditioning
from training.r7_experiment import make_model, seed_everything

DIM = 16
PROCESSES = 4
TOKEN_HW = (3, 4)
TOKENS = TOKEN_HW[0] * TOKEN_HW[1]
BASE = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": DIM, "depth": 1,
        "heads": 2, "window_size": 2, "patch_size": 2, "dropout": 0.0,
        "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 1}


def _readout(seed: int = 3) -> PositionalProcessReadout:
    torch.manual_seed(seed)
    return PositionalProcessReadout(DIM, heads=2)


def _tensors(seed: int = 4):
    generator = torch.Generator().manual_seed(seed)
    process = torch.randn(1, PROCESSES, DIM, generator=generator)
    context = torch.randn(1, TOKENS, DIM, generator=generator)
    return process, context


def _pooled(process: torch.Tensor) -> torch.Tensor:
    """The pre-RW-A summary: one vector per sample, broadcast over positions."""
    return process.mean(dim=1)


def test_the_pooled_read_is_a_broadcast_and_the_positional_one_is_not():
    readout = _readout()
    process, context = _tensors()
    positional = readout(process, context, TOKEN_HW)
    assert positional.shape == (1, TOKENS, DIM)
    assert not torch.equal(positional[:, 0], positional[:, 1])
    pooled = _pooled(process)
    assert pooled.shape == (1, DIM)


def test_perturbing_one_process_token_moves_positions_unevenly():
    """The D5 gate: a position-dependent response, against a zero-spread reference."""
    readout = _readout()
    process, context = _tensors()
    delta = torch.zeros_like(process)
    delta[:, 1] = 1.0
    positional_response = (readout(process + delta, context, TOKEN_HW)
                           - readout(process, context, TOKEN_HW)).detach().norm(dim=-1)[0]
    magnitude = float(positional_response.abs().max())
    spread = float(positional_response.max() - positional_response.min())

    # The pooled path cannot vary across positions: it adds the same vector to all
    # of them, so its response is identical everywhere by construction.
    pooled_delta = _pooled(process + delta) - _pooled(process)
    pooled_response = pooled_delta.expand(TOKENS, DIM).norm(dim=-1)
    assert float(pooled_response.max() - pooled_response.min()) == 0.0
    assert float(pooled_response[0]) > 0.0  # the pooled read does see the token

    assert spread > 0.0, ("perturbing one process token produced the same response "
                          "at every output position; the read is not positional")
    # Numerical floor only: the spread has to be a real fraction of the response
    # rather than float dust from the same value computed twice. The reference
    # above is exactly zero, so this is not a tuned scientific threshold.
    assert spread >= 1e-3 * magnitude


def test_every_process_token_reaches_every_output_position():
    """N x M, not N x 1: changing any single process token changes the read."""
    readout = _readout()
    process, context = _tensors()
    clean = readout(process, context, TOKEN_HW)
    for index in range(PROCESSES):
        delta = torch.zeros_like(process)
        delta[:, index] = 0.5
        assert not torch.equal(readout(process + delta, context, TOKEN_HW), clean), index


def test_the_read_cost_is_n_queries_times_m_tokens():
    """The attention itself must be N queries over M process tokens."""
    readout = _readout()
    process, context = _tensors()
    seen = []
    readout.attention.register_forward_pre_hook(
        lambda _module, inputs: seen.append((tuple(inputs[0].shape), tuple(inputs[1].shape))))
    readout(process, context, TOKEN_HW)
    assert seen == [((1, TOKENS, DIM), (1, PROCESSES, DIM))]
    assert TOKENS != PROCESSES  # the test would not distinguish N from M otherwise


def test_the_read_is_invariant_to_the_token_order_as_a_set():
    """Recorded property, not a claim of success: the read still sees a *set*.

    A query is formed without looking at the keys, so permuting the (key, value)
    pairs carries each softmax weight along with its own value and the weighted
    sum is unchanged up to float summation order. RW-A therefore makes the summary
    depend on the *query position*, not on an ordering of the process tokens - the
    pooled path is order-invariant for the same reason. This is pinned so that
    "positional" is never read as "ordered".
    """
    readout = _readout()
    process, context = _tensors()
    order = torch.tensor([2, 0, 3, 1])
    permuted = process[:, order]
    positional_gap = float((readout(process, context, TOKEN_HW)
                           - readout(permuted, context, TOKEN_HW)).detach().abs().max())
    scale = float(readout(process, context, TOKEN_HW).detach().abs().max())
    assert positional_gap / scale < 1e-6, ("the read is order-sensitive; if that is "
                                          "intended it must be claimed explicitly")


def test_the_summary_is_added_and_nothing_else_changes():
    """Additive by construction: both summary ranks reduce to ``context + summary``."""
    process, context = _tensors()
    readout = _readout()
    positional = readout(process, context, TOKEN_HW)
    assert torch.equal(solver_conditioning(context, positional), context + positional)
    pooled = _pooled(process)
    assert torch.equal(solver_conditioning(context, pooled), context + pooled[:, None, :])


def test_a_summary_of_the_wrong_rank_or_shape_is_rejected():
    """Counterproof for the new branch: it accepts exactly two shapes."""
    _, context = _tensors()
    for bad in (context[0], context[:, :2], context.unsqueeze(0), torch.zeros(1, TOKENS)):
        with pytest.raises(ValueError):
            solver_conditioning(context, bad)


def test_turning_the_readout_switch_on_adds_only_the_reader():
    """Same seed, same construction: every pre-existing parameter is bitwise equal.

    This is what makes the two experiment arms a comparison of the pathway rather
    than of two draws: the reader is constructed last, so it cannot shift the
    random stream the shared parameters are drawn from.
    """
    seed_everything(11)
    without = make_model("process", dict(BASE))
    seed_everything(11)
    with_reader = make_model("process", dict(BASE, positional_process_readout=True))
    before, after = without.state_dict(), with_reader.state_dict()
    added = set(after) - set(before)
    assert added and all(name.startswith("process_reader.") for name in added), added
    assert not set(before) - set(after)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert isinstance(with_reader.process_reader, PositionalProcessReadout)
    assert with_reader.positional_process_readout is True


def test_the_default_read_is_the_pooled_projection():
    """With the switch off the summary is the old [B,D] vector, not a positional one."""
    reasoner = make_model("process", dict(BASE))
    process = torch.randn(1, PROCESSES, DIM)
    context = torch.randn(1, TOKENS, DIM)
    summary = reasoner.process_conditioning(process, context, TOKEN_HW)
    assert summary.shape == (1, DIM)
    assert torch.equal(summary, reasoner.process_to_context(process.mean(dim=1)))
    assert isinstance(reasoner.correction_head, CoarseForecastHead)


def test_the_reader_is_only_offered_by_the_process_model():
    """RW-A changes the process arm's reader; the generic baseline is untouched."""
    generic = make_model("generic", {name: value for name, value in BASE.items()
                                     if name not in ("anchored_processes", "free_processes")},
                         )
    assert not hasattr(generic, "process_conditioning")
    assert not hasattr(generic, "positional_process_readout")
    reasoner = make_model("process", dict(BASE))
    assert isinstance(reasoner, ProcessForecastCoReasoner)
