"""CPU counterproofs for the #79 typed-evidence pathway (synthetic fixtures only)."""
from __future__ import annotations

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.typed_evidence_r7 import (
    EVIDENCE_TYPES, GENERIC_FUSION, TYPED_ROUTING, TypedEvidenceRouter,
)

CHANNELS = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
            "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
DIM = 16
PATCH = 2
MODEL = dict(dim=DIM, depth=1, heads=2, window_size=2, patch_size=PATCH, dropout=0.0,
             anchored_processes=8, free_processes=8, use_forecast_feedback=True,
             positional_process_readout=True, default_reasoning_steps=2,
             history_steps=2)
SMALL = dict(MODEL, dim=32, depth=1, heads=2)


def payload(*, active=None, mode_scale=None):
    count = EVIDENCE_TYPES
    return dict(channels=CHANNELS,
                denorm_mean=[0.0] * len(CHANNELS), denorm_std=[1.0] * len(CHANNELS),
                field_scale=[1.0] * count, field_mean=[0.0] * count,
                field_std=[1.0] * count,
                field_active=[True] * count if active is None else active)


def router(mode, **kwargs):
    torch.manual_seed(0)
    return TypedEvidenceRouter(DIM, PATCH, mode=mode, **payload(**kwargs))


def batch(b=2, h=8, w=8, channels=len(CHANNELS)):
    return {"coarse_history": torch.randn(b, 2, channels, h, w),
            "lead_time_hours": torch.tensor(6.0),
            "latitude": torch.linspace(39.0, 41.0, h),
            "longitude": torch.linspace(115.0, 117.0, w)}


# ---------------------------------------------------------------- constructor guards

def test_mode_and_capacity_guards():
    with pytest.raises(ValueError):
        router("not_a_mode")
    with pytest.raises(ValueError):
        TypedEvidenceRouter(DIM, PATCH, mode=TYPED_ROUTING, **payload(active=[False] * 4))
    rotten = payload()
    rotten["field_std"] = [1.0, -1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        TypedEvidenceRouter(DIM, PATCH, mode=TYPED_ROUTING, **rotten)
    missing = payload()
    missing["channels"] = ("t2m", "u10")
    with pytest.raises(ValueError):
        TypedEvidenceRouter(DIM, PATCH, mode=TYPED_ROUTING, **missing)


def test_model_refuses_typed_evidence_on_too_few_anchored_slots():
    with pytest.raises(ValueError):
        ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                  typed_evidence_mode=TYPED_ROUTING,
                                  typed_evidence=payload(),
                                  **dict(MODEL, anchored_processes=3, free_processes=5))
    with pytest.raises(ValueError):
        ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                  typed_evidence_mode=TYPED_ROUTING,
                                  typed_evidence=dict(payload(), extra=1), **MODEL)


# ---------------------------------------------------------------- per-type arrival

def test_typed_mode_routes_each_field_only_to_its_own_slot():
    model = router(TYPED_ROUTING)
    fields = torch.randn(2, EVIDENCE_TYPES, 6, 6)
    base = model.token_evidence(fields)
    assert base.shape == (2, EVIDENCE_TYPES, DIM)
    for index in range(EVIDENCE_TYPES):
        poisoned = fields.clone()
        poisoned[:, index] += 1.5
        moved = (model.token_evidence(poisoned) - base).abs().amax(dim=(0, 2))
        assert float(moved[index].detach()) > 0
        for other in range(EVIDENCE_TYPES):
            if other != index:
                assert float(moved[other].detach()) == 0.0


def test_generic_mode_moves_the_fused_vector_from_every_field():
    model = router(GENERIC_FUSION)
    fields = torch.randn(2, EVIDENCE_TYPES, 6, 6)
    base = model.token_evidence(fields)
    assert base.shape == (2, DIM)
    for index in range(EVIDENCE_TYPES):
        poisoned = fields.clone()
        poisoned[:, index] += 1.5
        moved = (model.token_evidence(poisoned) - base).abs().amax()
        assert float(moved.detach()) > 0


def test_inject_touches_only_the_evidence_slots():
    torch.manual_seed(0)
    process = torch.randn(2, 16, DIM)
    typed = router(TYPED_ROUTING)
    evidence = typed.token_evidence(torch.randn(2, EVIDENCE_TYPES, 6, 6))
    updated = typed.inject(process, evidence)
    assert torch.equal(updated[:, EVIDENCE_TYPES:], process[:, EVIDENCE_TYPES:])
    assert not torch.equal(updated[:, :EVIDENCE_TYPES], process[:, :EVIDENCE_TYPES])
    generic = router(GENERIC_FUSION)
    fused = generic.token_evidence(torch.randn(2, EVIDENCE_TYPES, 6, 6))
    updated = generic.inject(process, fused)
    assert torch.equal(updated[:, EVIDENCE_TYPES:], process[:, EVIDENCE_TYPES:])
    # one fused vector is added to every evidence slot, so each slot is exactly
    # its old value plus that same vector ((a+b)-a is not b in FP, so the sum
    # is compared directly rather than through a subtraction)
    assert all(torch.equal(updated[:, index], process[:, index] + fused)
               for index in range(EVIDENCE_TYPES))


def test_masked_field_contributes_exactly_zero():
    model = router(TYPED_ROUTING, active=[True, False, True, True])
    fields = torch.randn(2, EVIDENCE_TYPES, 6, 6)
    fields[:, 1] = 1e6  # a masked type must not leak into anything
    evidence = model.token_evidence(fields)
    assert torch.equal(evidence[:, 1], torch.zeros_like(evidence[:, 1]))
    baseline = router(TYPED_ROUTING, active=[True, False, True, True])
    quiet = fields.clone()
    quiet[:, 1] = 0.0
    assert torch.equal(model.token_evidence(quiet), baseline.token_evidence(quiet))


# ---------------------------------------------------------------- model integration

def test_off_path_is_bitwise_the_previous_model():
    inputs = batch()
    torch.manual_seed(3)
    without = ProcessForecastCoReasoner(in_channels=17, out_channels=17, **SMALL)
    torch.manual_seed(3)
    explicit_off = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                             typed_evidence_mode=None, **SMALL)
    state = without.state_dict()
    assert set(state) == set(explicit_off.state_dict())
    assert all(torch.equal(state[key], explicit_off.state_dict()[key]) for key in state)
    without.eval()
    explicit_off.eval()
    with torch.no_grad():
        first = without(inputs).forecast
        second = explicit_off(inputs).forecast
    assert torch.equal(first, second)


def test_evidence_changes_the_fixed_forward_and_only_through_the_pathway():
    inputs = batch()
    torch.manual_seed(4)
    plain = ProcessForecastCoReasoner(in_channels=17, out_channels=17, **SMALL)
    torch.manual_seed(4)
    typed = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                      typed_evidence_mode=TYPED_ROUTING,
                                      typed_evidence=payload(), **SMALL)
    plain.eval()
    typed.eval()
    with torch.no_grad():
        reference = plain(inputs).forecast
        modified = typed(inputs).forecast
    # same weights on the shared trunk; the only difference is the evidence
    shared = plain.state_dict()
    assert all(torch.equal(shared[key], typed.state_dict()[key]) for key in shared)
    assert not torch.equal(reference, modified)
    # the state before the first step is the queries expansion plus evidence
    queries = plain.process_queries.expand(reference.shape[0], -1, -1)
    with torch.no_grad():
        seeded = typed.initial_process_state(
            queries, inputs, anchor=inputs["coarse_history"][:, -1],
            draft=reference)
    assert seeded.shape == queries.shape
    assert torch.equal(seeded[:, EVIDENCE_TYPES:], queries[:, EVIDENCE_TYPES:])
    assert not torch.equal(seeded[:, :EVIDENCE_TYPES], queries[:, :EVIDENCE_TYPES])
    # off-path model returns the very tensor it was given
    assert plain.initial_process_state(queries, inputs,
                                       anchor=inputs["coarse_history"][:, -1],
                                       draft=reference) is queries


def test_future_labels_are_never_read_and_poisoning_them_is_a_no_op():
    inputs = batch()
    poisoned = dict(inputs)
    poisoned["atmos_target"] = torch.full_like(inputs["coarse_history"][:, -1], 1e6)
    poisoned["process_targets"] = torch.full((2, 8), 1e6)
    torch.manual_seed(5)
    model = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                      typed_evidence_mode=TYPED_ROUTING,
                                      typed_evidence=payload(), **SMALL)
    model.eval()
    with torch.no_grad():
        clean = model(inputs).forecast
        dirty = model(poisoned).forecast
    assert torch.equal(clean, dirty)


def test_same_seed_arms_share_the_trunk_and_params_are_capacity_matched():
    inputs = batch()
    torch.manual_seed(6)
    control = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                        typed_evidence_mode=GENERIC_FUSION,
                                        typed_evidence=payload(), **SMALL)
    torch.manual_seed(6)
    candidate = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                          typed_evidence_mode=TYPED_ROUTING,
                                          typed_evidence=payload(), **SMALL)
    left = control.state_dict()
    right = candidate.state_dict()
    shared = [key for key in left if key in right]
    assert all(torch.equal(left[key], right[key]) for key in shared)
    control_params = sum(p.numel() for p in control.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    assert control_params == candidate_params
    # and the seed is not consumed: a plain model from the same seed has the
    # same shared trunk values as both arms
    torch.manual_seed(6)
    plain = ProcessForecastCoReasoner(in_channels=17, out_channels=17, **SMALL)
    assert all(torch.equal(plain.state_dict()[key], left[key]) for key in plain.state_dict())
    control.eval()
    candidate.eval()
    with torch.no_grad():
        assert not torch.equal(control(inputs).forecast, candidate(inputs).forecast)


def test_router_buffers_do_not_enter_a_state_dict():
    model = router(TYPED_ROUTING)
    assert all(not key.startswith("typed_evidence.") or "weight" in key or "bias" in key
               for key in model.state_dict())
    assert "typed_evidence.field_std" not in model.state_dict()
    torch.manual_seed(0)
    clone = TypedEvidenceRouter(DIM, PATCH, mode=TYPED_ROUTING, **payload())
    clone.load_state_dict(model.state_dict(), strict=True)


def test_gradients_reach_the_pathway_and_survive_a_training_step():
    torch.manual_seed(7)
    model = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                      typed_evidence_mode=TYPED_ROUTING,
                                      typed_evidence=payload(), **SMALL)
    model.train()
    inputs = batch()
    inputs["atmos_target"] = torch.randn_like(inputs["coarse_history"][:, -1])
    for key in ("coarse_history", "latitude", "longitude"):
        inputs[key] = inputs[key].detach()
    prediction = model(inputs, reasoning_steps=2).forecast
    loss = (prediction - inputs["atmos_target"]).square().mean()
    loss.backward()
    for name, parameter in model.typed_evidence.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    source_grads = [parameter.grad is not None for _, parameter in
                    model.backbone.named_parameters()]
    assert any(source_grads)


@pytest.mark.parametrize("mode", [TYPED_ROUTING, GENERIC_FUSION])
def test_streamed_and_fixed_paths_start_from_the_same_evidence_state(mode):
    from model.r7_halting import forecast_inputs
    from training.r7_streaming import _recursive_step

    inputs = batch()
    torch.manual_seed(8)
    model = ProcessForecastCoReasoner(in_channels=17, out_channels=17,
                                      typed_evidence_mode=mode,
                                      typed_evidence=payload(), **SMALL)
    model.train()
    base = model.backbone(forecast_inputs(inputs))
    queries = model.process_queries.expand(base.forecast.shape[0], -1, -1)
    seeded = model.initial_process_state(queries, inputs, anchor=base.base_state,
                                         draft=base.forecast)
    step = _recursive_step(model, seeded, base.context_tokens, base.forecast,
                           base.token_hw, step_index=0, anchor=base.base_state)
    direct = model(inputs, reasoning_steps=1)
    assert torch.equal(step.draft, direct.forecast)
