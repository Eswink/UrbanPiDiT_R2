"""Bridge (e)'s identities and the frozen constant, asserted on a built router.

Split from ``tests/test_r7_s3_position_retention_65x65.py`` to stay under R-051's 600-line
cap. Everything here runs on a constructed ``TypedEvidenceRouter`` under a fixed seed: no
store, sidecar, manifest or trained checkpoint. The float32 statements live here because they
are properties of ``model/typed_evidence_r7.py``, not of the round's protocol.
"""
from __future__ import annotations

import math

import pytest
import torch

from scripts import r7_s3_position_retention_support as support
from model.typed_evidence_r7 import TYPED_SPATIAL, TypedEvidenceRouter

STORE_CHANNELS = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                  "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
#: 65x65 at patch size 2 -- the token count the round's grid actually produces.
PATCH_POSITIONS = 1089


def _synthetic_payload():
    """A valid typed-evidence payload: real channel names, invented statistics."""
    return {"channels": list(STORE_CHANNELS),
            "denorm_mean": [0.0] * len(STORE_CHANNELS),
            "denorm_std": [1.0] * len(STORE_CHANNELS),
            "field_scale": [1.0] * 4, "field_mean": [0.0] * 4, "field_std": [1.0] * 4,
            "field_active": [True] * 4}


def _router(mode, *, seed=41, position_scale=support.POSITION_SCALE):
    """A constructed router under a fixed seed, so a repeated construction is comparable."""
    payload = _synthetic_payload()
    torch.manual_seed(seed)
    return TypedEvidenceRouter(192, 5, mode=mode, **payload, position_scale=position_scale)


# --- the position-scale semantics, asserted on a built router ----------------------------------

def test_the_router_carries_the_frozen_scale_as_a_plain_float():
    router = _router(TYPED_SPATIAL)
    assert router.position_scale == 8000.0
    assert not isinstance(router.position_scale, torch.nn.Parameter)
    parameters = {name for name, _ in router.named_parameters()}
    assert "position_scale" not in parameters
    assert "position_query" in parameters


def test_the_scale_adds_no_parameters():
    small = _router(TYPED_SPATIAL, position_scale=1.0)
    large = _router(TYPED_SPATIAL, position_scale=8000.0)
    assert (sum(p.numel() for p in small.parameters())
            == sum(p.numel() for p in large.parameters()))


def test_the_router_refuses_a_non_positive_or_non_finite_scale():
    payload = _synthetic_payload()
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            TypedEvidenceRouter(192, 5, mode=TYPED_SPATIAL, **payload, position_scale=value)


def test_a_zero_query_gives_exactly_uniform_weights():
    """The exact half of bridge (e) 4: zero scores make softmax exactly ``1/P``.

    This is bitwise and testable: ``softmax(zeros)`` is the float32 neighbour of ``1/P``, the
    same value for every position. It is the aggregation *to* the pooled mean that is only
    accurate to float32 accumulation, so the two halves are asserted separately.
    """
    router = _router(TYPED_SPATIAL)
    router.position_query.data.zero_()
    scores = (torch.randn(2, PATCH_POSITIONS, router.dim) @ router.position_query) \
        * router.position_scale / (router.dim ** 0.5)
    assert torch.equal(scores, torch.zeros_like(scores))
    weights = torch.softmax(scores, dim=1)
    p = PATCH_POSITIONS
    assert torch.equal(weights, torch.full_like(weights, 1.0 / p))


def test_a_zero_query_aggregates_to_the_pooled_mean_up_to_float32_accumulation():
    """Bridge (e) 4 as it can honestly be stated, and why it cannot be a bitwise equality.

    ``_position_attention`` multiplies each position by ``1/P`` and sums; ``mean(dim=1)`` sums
    and then divides by ``P``. Those are the same real number and different float32 programs,
    so the reachable claim is a bound (measured worst case: 2.98e-08 over 64 seeds), not zero.
    The driver's frozen criterion is this bound, and the reading records the measured value.
    """
    eps = torch.finfo(torch.float32).eps
    router = _router(TYPED_SPATIAL)
    router.position_query.data.zero_()
    worst_ratio = 0.0
    for seed in range(8):
        value = torch.randn(2, PATCH_POSITIONS, router.dim,
                            generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            zeroed = router._position_attention(value)
            pooled = value.mean(dim=1)
        difference = float((zeroed - pooled).abs().max())
        bound = 2 * PATCH_POSITIONS * eps * float(value.abs().max())
        assert difference <= bound, (seed, difference, bound)
        worst_ratio = max(worst_ratio, difference / bound)
    assert worst_ratio < 1.0
    # The counterexample that makes a bound necessary rather than timid: ``_position_attention``
    # multiplies each position by 1/P and sums, while ``mean(dim=1)`` sums and then divides by P.
    # Any nonzero input separates the two float32 programs, so equality is unreachable.
    value = torch.randn(2, PATCH_POSITIONS, router.dim, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert not torch.equal(router._position_attention(value), value.mean(dim=1))


def test_the_aggregation_identity_does_not_depend_on_the_frozen_constant():
    """The bridge must hold at every scale, or it would be testing the constant instead."""
    eps = torch.finfo(torch.float32).eps
    for scale in (1.0, 2000.0, 8000.0, 1e6):
        router = _router(TYPED_SPATIAL, position_scale=scale)
        router.position_query.data.zero_()
        value = torch.randn(1, PATCH_POSITIONS, router.dim,
                            generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            zeroed = router._position_attention(value)
            pooled = value.mean(dim=1)
        difference = float((zeroed - pooled).abs().max())
        assert difference <= 2 * PATCH_POSITIONS * eps * float(value.abs().max()), scale


def test_the_query_starts_at_exactly_zero():
    for seed in (41, 42, 43):
        assert float(_router(TYPED_SPATIAL, seed=seed).position_query.detach().norm()) == 0.0


def test_a_non_zero_query_does_change_the_aggregation():
    """The counterexample to the identity: without it the round would be vacuous."""
    router = _router(TYPED_SPATIAL)
    with torch.no_grad():
        router.position_query.copy_(torch.full_like(router.position_query, 0.013))
    value = torch.randn(1, PATCH_POSITIONS, router.dim, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        weighted = router._position_attention(value)
    assert not torch.equal(weighted, value.mean(dim=1))
    assert float((weighted - value.mean(dim=1)).abs().max()) > 0.0


def test_the_scale_moves_the_softmax_entropy_the_way_the_freeze_assumes():
    """Higher scale => lower normalised entropy, monotonically, at a fixed query."""
    value = torch.randn(PATCH_POSITIONS, 192, generator=torch.Generator().manual_seed(3))
    query = torch.randn(192, generator=torch.Generator().manual_seed(4))
    query = query / query.norm() * 0.0136

    def entropy(scale):
        weights = torch.softmax((value @ query) * scale / math.sqrt(192), dim=0)
        return float(-(weights * torch.log(weights + 1e-30)).sum() / math.log(PATCH_POSITIONS))

    values = [entropy(scale) for scale in (1.0, 2000.0, 8000.0, 20000.0)]
    assert values[0] > values[1] > values[2] > values[3]
    assert values[0] > 0.999, "scale 1 must leave the softmax numerically uniform"
    assert values[2] < 0.999, "the frozen constant must actually move the softmax"


def test_the_entropy_band_and_the_one_hot_floor_are_the_frozen_values():
    assert support.ENTROPY_BAND == (0.90, 0.99)
    assert support.ENTROPY_ONE_HOT_FLOOR == 0.5
    assert support.ENTROPY_ONE_HOT_FLOOR < support.ENTROPY_BAND[0]
    low, high = support.QUERY_NORM_RANGE
    assert low < high and (low, high) == (0.0074, 0.0136)
