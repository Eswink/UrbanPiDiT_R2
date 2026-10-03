"""ADR 0035 source-key positions: complete token equations, not pooled moments.

Synthetic engineering fixtures only. Positions/roles stay fixed across probes;
source keys are ephemeral and may not contaminate the query or solver carry.
"""
from __future__ import annotations

import copy

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import AdaptiveProcessForecaster, forecast_inputs
from model.recursive_weather_r7 import (
    GenericRecursiveWeatherForecaster, process_to_generic_state_key,
    reasoning_source_key, recurrent_key,
)
from training.r7_halting import per_sample_latitude_mse
from training.r7_streaming import backward_streamed_truncated

SMALL = dict(in_channels=3, out_channels=3, history_steps=2, dim=16, depth=1,
             heads=2, window_size=2, patch_size=2, dropout=0.0,
             default_reasoning_steps=4, spacetime_inputs=True, positional_process_readout=True)
TOKEN_HW = (3, 4)
ORDER = torch.tensor([7, 0, 10, 3, 9, 2, 11, 5, 1, 8, 4, 6])


def _model(kind='process', local=False, **extra):
    torch.manual_seed(31)
    config = dict(SMALL, local_solver_state=local, **extra)
    if kind == 'generic':
        return GenericRecursiveWeatherForecaster(**config, latent_tokens=4,
            solver_state_recurrence=local, solver_gate_proposal=local)
    return ProcessForecastCoReasoner(**config, anchored_processes=2, free_processes=2)


def _pair(local=False, **extra):
    generic, process = _model('generic', local, **extra), _model('process', local, **extra)
    mapped = {process_to_generic_state_key(name): value.clone()
              for name, value in process.state_dict().items()
              if process_to_generic_state_key(name) is not None}
    assert set(mapped) == set(generic.state_dict())
    generic.load_state_dict(mapped, strict=True)
    return generic, process


def _batch(hw=(5, 7), size=2):
    generator = torch.Generator().manual_seed(7)
    return {'coarse_history': torch.randn(size, 2, 3, *hw, generator=generator),
            'atmos_target': torch.randn(size, 3, *hw, generator=generator),
            'latitude': torch.linspace(-40., 40., hw[0]),
            'longitude': torch.linspace(-120., 100., hw[1]),
            'init_utc_hour': torch.full((size,), 18.), 'init_day_of_year': torch.full((size,), 59.),
            'init_calendar_year': torch.full((size,), 2024),
            'lead_time_hours': torch.full((size,), 6.),
            'history_offsets_hours': torch.tensor([-6., 0.]).expand(size, -1).clone()}


def _assert_outputs(generic, process):
    for name in ('forecast', 'initial_forecast', 'draft_forecasts', 'final_correction', 'context_tokens'):
        assert torch.equal(getattr(generic, name), getattr(process, name)), name
    assert torch.equal(generic.latent_state, process.process_state)
    assert (generic.solver_state is None) == (process.solver_state is None)
    if generic.solver_state is not None:
        assert torch.equal(generic.solver_state, process.solver_state)


def _assert_gradients(generic, process):
    parameters = dict(generic.named_parameters())
    for name, parameter in process.named_parameters():
        mapped = process_to_generic_state_key(name)
        if mapped is None:
            assert parameter.grad is None, name
        else:
            other = parameters[mapped].grad
            assert (other is None) == (parameter.grad is None), name
            if other is not None:
                assert torch.equal(other, parameter.grad), name
                assert torch.isfinite(other).all(), name


@pytest.mark.parametrize('kind', ['generic', 'process'])
def test_constructor_bool_dependencies_and_independent_no_feedback_pooled_modes(kind):
    for field in ('source_position_markers', 'known_context_inputs'):
        with pytest.raises(ValueError, match=field):
            _model(kind, **{field: 1})
    with pytest.raises(ValueError, match='positional_process_readout'):
        _model(kind, source_position_markers=True, positional_process_readout=False)
    with pytest.raises(ValueError, match='spacetime_inputs'):
        _model(kind, known_context_inputs=True, spacetime_inputs=False)
    model = _model(kind, source_position_markers=True, source_role_markers=False,
                   use_forecast_feedback=False, draft_query_feedback=False,
                   known_context_inputs=False, spacetime_inputs=False, pooled_readout_query=True)
    assert model.source_position_markers is True and model.known_context_inputs is False
    assert model.backbone.known_context_inputs is False
    assert torch.isfinite(model(forecast_inputs(_batch()), reasoning_steps=2).forecast).all()
    known = _model(kind, known_context_inputs=True)
    assert known.known_context_inputs is True and known.backbone.known_context_inputs is True
    assert hasattr(known.backbone, 'known_context')


@pytest.mark.parametrize('kind', ['generic', 'process'])
def test_source_markers_add_no_state_parameters_rng_and_off_is_the_original_call(kind):
    plain = _model(kind, source_role_markers=True)
    state, rng = plain.state_dict(), torch.random.get_rng_state().clone()
    explicit = _model(kind, source_role_markers=True, source_position_markers=False,
                      known_context_inputs=False)
    assert torch.equal(rng, torch.random.get_rng_state())
    enabled = _model(kind, source_role_markers=True, source_position_markers=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(state) == set(explicit.state_dict()) == set(enabled.state_dict())
    for name, value in state.items():
        assert torch.equal(value, enabled.state_dict()[name]), name
        assert torch.equal(value, explicit.state_dict()[name]), name
    generator = torch.Generator().manual_seed(19)
    context, draft = [torch.randn(2, 12, 16, generator=generator) for _ in range(2)]
    expected = recurrent_key(context, draft, role_context=plain.role_context, role_draft=plain.role_draft)
    original = plain.process_reader.position_encoding
    plain.process_reader.position_encoding = lambda *a, **k: (_ for _ in ()).throw(AssertionError('basis called off'))
    try:
        assert torch.equal(reasoning_source_key(plain, context, draft, TOKEN_HW), expected)
        assert torch.equal(reasoning_source_key(plain, context, None, TOKEN_HW), context + plain.role_context)
    finally:
        plain.process_reader.position_encoding = original
    for model in (plain, explicit):
        model(forecast_inputs(_batch())).forecast.square().mean().backward()
    for (name, a), (other, b) in zip(plain.named_parameters(), explicit.named_parameters()):
        assert name == other and (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name


def test_full_source_key_is_position_once_then_role_and_never_mutates_inputs():
    model = _model(source_position_markers=True, source_role_markers=True)
    generator = torch.Generator().manual_seed(19)
    context, draft = [torch.randn(2, 12, 16, generator=generator) for _ in range(2)]
    saved_context, saved_draft = context.clone(), draft.clone()
    position = model.process_reader.position_encoding(TOKEN_HW, device=context.device, dtype=context.dtype)
    expected = torch.cat([(context + position) + model.role_context,
                          (draft + position) + model.role_draft], dim=1)
    key = reasoning_source_key(model, context, draft, TOKEN_HW)
    assert torch.equal(key, expected)
    assert torch.equal(reasoning_source_key(model, context, draft, TOKEN_HW), expected)
    assert torch.equal(context, saved_context) and torch.equal(draft, saved_draft)
    assert torch.equal(reasoning_source_key(model, context, None, TOKEN_HW),
                       (context + position) + model.role_context)
    for bad in (draft[:, :1], draft.double(), draft.to('meta')):
        with pytest.raises(ValueError, match='source_position_markers'):
            reasoning_source_key(model, context, bad, TOKEN_HW)
    with pytest.raises(ValueError, match='token grid'):
        reasoning_source_key(model, context, draft, (2, 4))


def test_fixed_roles_full_joint_key_equation_latent_set_invariance_and_mismatch_counterproof():
    model = _model(source_position_markers=True, source_role_markers=True,
                   draft_query_feedback=True).eval()
    generator = torch.Generator().manual_seed(19)
    context, draft = [torch.randn(2, 12, 16, generator=generator) for _ in range(2)]
    latent = model.process_queries.expand(2, -1, -1)
    basis = model.process_reader.position_encoding(TOKEN_HW, device=context.device, dtype=context.dtype)
    key = reasoning_source_key(model, context, draft, TOKEN_HW)
    expanded_order = torch.cat([ORDER, ORDER + 12])
    with torch.no_grad():
        reference = model._reason(latent, key)
    original = model.process_reader.position_encoding
    try:
        model.process_reader.position_encoding = lambda *a, **k: basis[ORDER]
        joint = reasoning_source_key(model, context[:, ORDER], draft[:, ORDER], TOKEN_HW)
        assert torch.equal(joint, key[:, expanded_order])
        with torch.no_grad():
            transformed = model._reason(latent, joint)
        torch.testing.assert_close(transformed, reference, rtol=1e-5, atol=1e-6)
        # Same parameters and roles; the fault is ONLY that positions do not move.
        model.process_reader.position_encoding = lambda *a, **k: basis
        wrong = reasoning_source_key(model, context[:, ORDER], draft[:, ORDER], TOKEN_HW)
        with pytest.raises(AssertionError):
            assert torch.equal(wrong, key[:, expanded_order])
        with torch.no_grad():
            broken = model._reason(latent, wrong)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(broken, reference, rtol=1e-5, atol=1e-6)
    finally:
        model.process_reader.position_encoding = original


def test_joint_full_query_read_equivariance_and_same_predicate_rejects_content_position_mismatch():
    model = _model(source_position_markers=True, source_role_markers=True,
                   draft_query_feedback=True).eval()
    reader = model.process_reader
    generator = torch.Generator().manual_seed(19)
    context, draft = [torch.randn(2, 12, 16, generator=generator) for _ in range(2)]
    process = model.process_queries.expand(2, -1, -1)
    basis = reader.position_encoding(TOKEN_HW, device=context.device, dtype=context.dtype)
    queries = []
    hook = reader.attention.register_forward_pre_hook(lambda module, inputs: queries.append(inputs[0].clone()))
    original = reader.position_encoding
    try:
        reference = reader(process, context, TOKEN_HW, draft_tokens=draft)
        reader.position_encoding = lambda *a, **k: basis[ORDER]
        joint = reader(process, context[:, ORDER], TOKEN_HW, draft_tokens=draft[:, ORDER])
        assert torch.equal(queries[1], queries[0][:, ORDER])
        torch.testing.assert_close(joint, reference[:, ORDER], rtol=1e-5, atol=1e-6)
        reader.position_encoding = lambda *a, **k: basis
        wrong = reader(process, context[:, ORDER], TOKEN_HW, draft_tokens=draft[:, ORDER])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(queries[2], queries[0][:, ORDER], rtol=1e-5, atol=1e-6)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(wrong, reference[:, ORDER], rtol=1e-5, atol=1e-6)
    finally:
        hook.remove()
        reader.position_encoding = original


@pytest.mark.parametrize('kind', ['generic', 'process'])
def test_shared_step_key_copies_do_not_enter_query_solver_or_carry(kind):
    model = _model(kind, True, source_position_markers=True, source_role_markers=True,
                   draft_query_feedback=True)
    keys, encodings, reads, solver_sources = [], [], [], []
    cell = model.cell if kind == 'generic' else model.reasoning_cell
    hooks = [cell.register_forward_pre_hook(lambda module, inputs: keys.append(inputs[1].clone())),
             model.draft_encoder.register_forward_hook(lambda module, inputs, output: encodings.append(output[0])),
             model.process_reader.register_forward_pre_hook(
                 lambda module, inputs, kwargs: reads.append((inputs[1], kwargs['draft_tokens'])), with_kwargs=True),
             model.solver_cell.register_forward_pre_hook(
                 lambda module, inputs, kwargs: solver_sources.append((kwargs['context'], kwargs['draft_tokens'])),
                 with_kwargs=True)]
    try:
        output = model(forecast_inputs(_batch()), reasoning_steps=2)
    finally:
        for hook in hooks:
            hook.remove()
    position = model.process_reader.position_encoding(output.token_hw, device=output.context_tokens.device,
                                                       dtype=output.context_tokens.dtype)
    assert len(keys) == len(encodings) == len(reads) == len(solver_sources) == 2
    for key, encoding, read, sources in zip(keys, encodings, reads, solver_sources):
        expected = torch.cat([(output.context_tokens + position) + model.role_context,
                              (encoding + position) + model.role_draft], dim=1)
        assert torch.equal(key, expected)
        assert torch.equal(read[0], output.context_tokens) and torch.equal(sources[0], output.context_tokens)
        assert read[1] is encoding and sources[1] is encoding


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('amp_dtype', [None, torch.bfloat16])
def test_matched_full_forward_and_all_gradients_fp32_bf16(local, amp_dtype):
    generic, process = _pair(local, source_position_markers=True, source_role_markers=True,
                             draft_query_feedback=True)
    outputs = []
    for model in (generic, process):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp_dtype is not None):
            output = model(forecast_inputs(_batch()), reasoning_steps=4)
        output.draft_forecasts.float().square().mean().backward()
        outputs.append(output)
        assert torch.isfinite(output.forecast).all()
        assert model.draft_encoder.patch.weight.grad.abs().max() > 0
        assert model.role_context.grad.abs().max() > 0 and model.role_draft.grad.abs().max() > 0
    _assert_outputs(*outputs)
    _assert_gradients(generic, process)


@pytest.mark.parametrize('steps', [0, 1, 2, 4])
def test_full_matched_k_depths_on_odd_native_grid(steps):
    generic, process = _pair(source_position_markers=True)
    with torch.no_grad():
        outputs = [model.eval()(forecast_inputs(_batch((65, 65), 1)), reasoning_steps=steps)
                   for model in (generic, process)]
    _assert_outputs(*outputs)
    assert outputs[0].forecast.shape == (1, 3, 65, 65)
    assert outputs[0].draft_forecasts.shape[1] == steps + 1
    if steps == 0:
        assert torch.equal(outputs[0].forecast, outputs[0].initial_forecast)


@pytest.mark.parametrize('local', [False, True])
def test_streamed_fixed_truncated_match_and_adaptive_force_full_poison_boundary(local):
    generic, process = _pair(local, source_position_markers=True, source_role_markers=True,
                             draft_query_feedback=True)
    batch = _batch()
    logs = []
    for model in (generic, process):
        reference = copy.deepcopy(model)
        output = reference(forecast_inputs(batch), reasoning_steps=3, detach_between_steps=True)
        errors = torch.stack([per_sample_latitude_mse(draft, batch['atmos_target'], batch['latitude']).mean()
                              for draft in output.draft_forecasts.unbind(1)])
        weights = torch.linspace(1., 2., 4)
        loss = (errors * weights / weights.sum()).sum()
        loss.backward()
        result = backward_streamed_truncated(model, batch, reasoning_steps=3, process_weight=0)
        assert torch.equal(result.final_forecast, output.forecast)
        assert torch.equal(result.draft_errors, errors.detach())
        torch.testing.assert_close(result.total, loss.detach(), rtol=1e-5, atol=1e-6)
        for (name, a), (other, b) in zip(reference.named_parameters(), model.named_parameters()):
            assert name == other and (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6, msg=name)
        logs.append(result)
    _assert_gradients(generic, process)
    assert torch.equal(logs[0].final_forecast, logs[1].final_forecast)
    poison = dict(batch, atmos_target=torch.tensor(float('nan')), process_targets=torch.tensor(float('nan')),
                  atmos_baseline=torch.tensor(float('nan')), future_state=torch.tensor(float('nan')))
    process.eval()
    with torch.no_grad():
        fixed = process(forecast_inputs(batch), reasoning_steps=4)
        adaptive = AdaptiveProcessForecaster(process).eval()(poison, max_steps=4, min_steps=4,
                                                            force_full_depth=True)
        assert torch.equal(fixed.forecast, adaptive.forecast)
        assert torch.equal(fixed.forecast, process(forecast_inputs(poison)).forecast)
        assert torch.equal(generic.eval()(forecast_inputs(batch)).forecast,
                           generic(forecast_inputs(poison)).forecast)


def test_no_feedback_markers_keep_context_role_and_do_not_require_draft_or_known_fields():
    generic, process = _pair(source_position_markers=True, source_role_markers=True,
                             use_forecast_feedback=False, spacetime_inputs=False)
    outputs = [model(forecast_inputs(_batch()), reasoning_steps=2) for model in (generic, process)]
    _assert_outputs(*outputs)
    for model, output in zip((generic, process), outputs):
        output.forecast.square().mean().backward()
        assert model.role_context.grad.abs().max() > 0 and model.role_draft.grad is None
        assert all(parameter.grad is None for parameter in model.draft_encoder.parameters())
    _assert_gradients(generic, process)
