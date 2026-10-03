"""ADR 0034: direct draft -> query, not indirect process/solver feedback.

All weather tensors are synthetic CPU engineering fixtures. Fixed P/context/pos
isolate the new edge; cutting that edge must fail the same causal/gradient probes.
"""
from __future__ import annotations

import copy

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_readout_r7 import PositionalProcessReadout
from model.process_step_r7 import ProcessStepInput, process_reasoning_step
from model.r7_halting import AdaptiveProcessForecaster, forecast_inputs
from model.recursive_weather_r7 import (
    GenericRecursiveWeatherForecaster, GenericStepInput, generic_reasoning_step,
    process_to_generic_state_key,
)
from training.r7_halting import per_sample_latitude_mse
from training.r7_streaming import backward_streamed_truncated

SMALL = dict(in_channels=3, out_channels=3, history_steps=2, dim=16, depth=1,
             heads=2, window_size=2, patch_size=2, dropout=0.0,
             default_reasoning_steps=4, spacetime_inputs=True,
             positional_process_readout=True)


def _model(kind='process', local=False, **switches):
    torch.manual_seed(31)
    options = dict(SMALL, local_solver_state=local, **switches)
    if kind == 'generic':
        return GenericRecursiveWeatherForecaster(**options, latent_tokens=4,
            solver_state_recurrence=local, solver_gate_proposal=local)
    return ProcessForecastCoReasoner(**options, anchored_processes=2, free_processes=2)


def _pair(local=False, **switches):
    generic, process = _model('generic', local, **switches), _model('process', local, **switches)
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
            'init_utc_hour': torch.tensor([18., 12.])[:size],
            'init_day_of_year': torch.tensor([59., 365.])[:size],
            'init_calendar_year': torch.tensor([2024, 2023])[:size],
            'lead_time_hours': torch.full((size,), 6.)}


def _assert_mapped_gradients(generic, process):
    parameters = dict(generic.named_parameters())
    for name, value in process.named_parameters():
        key = process_to_generic_state_key(name)
        if key is None:
            assert value.grad is None, name
            continue
        other = parameters[key].grad
        assert (other is None) == (value.grad is None), name
        if other is not None:
            assert torch.equal(other, value.grad), name
            assert torch.isfinite(other).all(), name


def _assert_same_output(generic, process):
    for name in ('forecast', 'initial_forecast', 'draft_forecasts', 'final_correction', 'context_tokens'):
        assert torch.equal(getattr(generic, name), getattr(process, name)), name
    assert torch.equal(generic.latent_state, process.process_state)
    assert (generic.solver_state is None) == (process.solver_state is None)
    if generic.solver_state is not None:
        assert torch.equal(generic.solver_state, process.solver_state)


def _reader_tensors():
    generator = torch.Generator().manual_seed(19)
    return (torch.randn(2, 4, 16, generator=generator),
            torch.randn(2, 12, 16, generator=generator),
            torch.randn(2, 12, 16, generator=generator))


def _assert_local_draft_response(reader, process, context, draft):
    changed = draft.clone()
    changed[:, 5, 0] += 2.0
    queries, keys, norm_inputs = [], [], []
    query_hook = reader.attention.register_forward_pre_hook(
        lambda module, inputs: (queries.append(inputs[0].detach().clone()),
                                keys.append(inputs[1].detach().clone())) and None)
    norm_hook = reader.query_norm.register_forward_pre_hook(
        lambda module, inputs: norm_inputs.append(inputs[0].detach().clone()))
    try:
        clean = reader(process, context, (3, 4), draft_tokens=draft)
        other = reader(process, context, (3, 4), draft_tokens=changed)
    finally:
        query_hook.remove()
        norm_hook.remove()
    assert torch.equal(norm_inputs[0], context + draft)
    assert torch.equal(norm_inputs[1], context + changed)
    assert torch.equal(keys[0], keys[1]), 'P/key-value must stay fixed'
    assert not torch.equal(queries[0][:, 5], queries[1][:, 5]), 'local draft never reached query'
    assert not torch.equal(clean[:, 5], other[:, 5]), 'local query never reached read'
    untouched = [index for index in range(12) if index != 5]
    assert torch.equal(queries[0][:, untouched], queries[1][:, untouched])
    assert torch.equal(clean[:, untouched], other[:, untouched])


def test_fixed_process_context_position_only_local_draft_changes_query_and_cut_branch_fails_same_probe():
    torch.manual_seed(23)
    reader = PositionalProcessReadout(16, 2, draft_query_feedback=True)
    process, context, draft = _reader_tensors()
    _assert_local_draft_response(reader, process, context, draft)
    reader.draft_query_feedback = False
    with pytest.raises(AssertionError):
        _assert_local_draft_response(reader, process, context, draft)


@pytest.mark.parametrize('pooled', [False, True])
def test_query_is_normalization_of_sum_and_pooled_control_has_no_position_encoding(pooled):
    reader = PositionalProcessReadout(16, 2, pooled_readout_query=pooled, draft_query_feedback=True)
    process, context, draft = _reader_tensors()
    queries = []
    hook = reader.attention.register_forward_pre_hook(lambda module, inputs: queries.append(inputs[0]))
    try:
        result = reader(process, context, (3, 4), draft_tokens=draft)
    finally:
        hook.remove()
    normalized = reader.query_norm(context + draft)
    expected = normalized.mean(1, keepdim=True).expand(-1, 12, -1) if pooled else normalized + (
        reader.position_encoding((3, 4), device=context.device, dtype=context.dtype))
    assert torch.equal(queries[0], expected)
    assert not torch.equal(normalized, reader.query_norm(context) + reader.query_norm(draft))
    if pooled:
        assert torch.equal(result, result[:, :1].expand_as(result))
        hook = reader.position_encoding
        reader.position_encoding = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('position called'))
        try:
            assert torch.equal(reader(process, context, (3, 4), draft_tokens=draft), result)
        finally:
            reader.position_encoding = hook


def test_fixed_process_reader_and_draft_gradients_are_nonzero_and_cut_query_fails_gradient_probe():
    reader = PositionalProcessReadout(16, 2, draft_query_feedback=True)
    process, context, draft = _reader_tensors()
    draft.requires_grad_()
    reader(process, context, (3, 4), draft_tokens=draft).square().mean().backward()
    assert draft.grad is not None and torch.isfinite(draft.grad).all() and draft.grad.abs().max() > 0
    for name, parameter in reader.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().max() > 0, name
    reader.zero_grad(set_to_none=True)
    cut = draft.detach().requires_grad_()
    reader.draft_query_feedback = False
    read = reader(process, context, (3, 4), draft_tokens=cut)
    gradient = torch.autograd.grad(read.square().mean(), cut, allow_unused=True)[0]
    with pytest.raises(AssertionError):
        assert gradient is not None and gradient.abs().max() > 0


def test_reader_requires_aligned_shape_device_dtype_only_when_enabled():
    process, context, draft = _reader_tensors()
    reader = PositionalProcessReadout(16, 2, draft_query_feedback=True)
    for bad, message in ((None, 'aligned'), (draft[:, :1], 'aligned'),
                         (draft.double(), 'dtypes'), (draft.to('meta'), 'devices')):
        with pytest.raises(ValueError, match=message):
            reader(process, context, (3, 4), draft_tokens=bad)
    reader.draft_query_feedback = False
    assert torch.equal(reader(process, context, (3, 4)),
                       reader(process, context, (3, 4), draft_tokens=draft.double()))


@pytest.mark.parametrize('kind', ['generic', 'process'])
def test_flag_is_actual_boolean_with_dependencies_and_feedback_override_rejection(kind):
    with pytest.raises(ValueError, match='draft_query_feedback'):
        _model(kind, draft_query_feedback=1)
    for extra, message in (({'positional_process_readout': False}, 'positional_process_readout'),
                           ({'use_forecast_feedback': False}, 'use_forecast_feedback')):
        with pytest.raises(ValueError, match=message):
            _model(kind, draft_query_feedback=True, **extra)
    model = _model(kind, draft_query_feedback=True)
    assert model.draft_query_feedback is True and model.process_reader.draft_query_feedback is True
    for steps in (0, 2):
        with pytest.raises(ValueError, match='use_forecast_feedback'):
            model(forecast_inputs(_batch()), reasoning_steps=steps, use_forecast_feedback=False)
    base = model.backbone(forecast_inputs(_batch()))
    conditioning = model.latent_conditioning if kind == 'generic' else model.process_conditioning
    state = model.latent if kind == 'generic' else model.process_queries
    with pytest.raises(ValueError, match='aligned'):
        conditioning(state.expand(2, -1, -1), base.context_tokens, base.token_hw)
    step, tensors = ((generic_reasoning_step, GenericStepInput) if kind == 'generic'
                     else (process_reasoning_step, ProcessStepInput))
    with pytest.raises(ValueError, match='use_forecast_feedback'):
        step(model, tensors(state.expand(2, -1, -1), base.context_tokens, base.forecast),
             base.token_hw, use_forecast_feedback=False)
    with pytest.raises(ValueError, match='draft_query_feedback'):
        PositionalProcessReadout(16, 2, draft_query_feedback=1)


@pytest.mark.parametrize('kind', ['generic', 'process'])
def test_flag_adds_no_parameter_state_or_random_draw_and_default_is_explicit_off(kind):
    plain = _model(kind)
    state, rng = plain.state_dict(), torch.random.get_rng_state().clone()
    explicit = _model(kind, draft_query_feedback=False)
    assert plain.draft_query_feedback is False and plain.process_reader.draft_query_feedback is False
    assert torch.equal(rng, torch.random.get_rng_state())
    enabled = _model(kind, draft_query_feedback=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(state) == set(enabled.state_dict()) == set(explicit.state_dict())
    for name, value in state.items():
        assert torch.equal(value, enabled.state_dict()[name]), name
        assert torch.equal(value, explicit.state_dict()[name]), name
    with torch.no_grad():
        default = plain.eval()(forecast_inputs(_batch()))
        off = explicit.eval()(forecast_inputs(_batch()))
    assert torch.equal(default.draft_forecasts, off.draft_forecasts)
    # Existing conditioning callers with no new argument remain valid when off.
    conditioning = plain.latent_conditioning if kind == 'generic' else plain.process_conditioning
    recurrent = default.latent_state if kind == 'generic' else default.process_state
    assert conditioning(recurrent, default.context_tokens, default.token_hw).shape == default.context_tokens.shape


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('amp_dtype', [None, torch.bfloat16])
def test_mapped_full_forecast_and_every_gradient_match_fp32_bf16(local, amp_dtype):
    generic, process = _pair(local, draft_query_feedback=True, source_role_markers=True)
    outputs = []
    for model in (generic, process):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp_dtype is not None):
            output = model.train()(forecast_inputs(_batch()), reasoning_steps=4)
        output.draft_forecasts.float().square().mean().backward()
        outputs.append(output)
        assert torch.isfinite(output.forecast).all()
        assert model.draft_encoder.patch.weight.grad.abs().max() > 0
        for name, parameter in model.process_reader.named_parameters():
            assert parameter.grad is not None and parameter.grad.abs().max() > 0, name
    _assert_same_output(*outputs)
    _assert_mapped_gradients(generic, process)


@pytest.mark.parametrize('steps', [0, 1, 2, 4])
def test_mapped_k_depths_preserve_the_odd_native_65_grid(steps):
    generic, process = _pair(draft_query_feedback=True)
    with torch.no_grad():
        outputs = [model.eval()(forecast_inputs(_batch((65, 65), size=1)), reasoning_steps=steps)
                   for model in (generic, process)]
    _assert_same_output(*outputs)
    assert outputs[0].forecast.shape == (1, 3, 65, 65)
    assert outputs[0].context_tokens.shape == (1, 33 * 33, 16)
    assert outputs[0].draft_forecasts.shape[1] == steps + 1


@pytest.mark.parametrize('local', [False, True])
def test_existing_draft_encoding_is_passed_to_reader_once_per_step_in_both_models(local):
    for model in _pair(local, draft_query_feedback=True):
        encoded, observed = [], []
        encoder_hook = model.draft_encoder.register_forward_hook(
            lambda module, inputs, output: encoded.append(output[0]))
        reader_hook = model.process_reader.register_forward_pre_hook(
            lambda module, inputs, kwargs: observed.append(kwargs['draft_tokens']), with_kwargs=True)
        try:
            model(forecast_inputs(_batch()), reasoning_steps=2)
        finally:
            encoder_hook.remove()
            reader_hook.remove()
        assert len(encoded) == len(observed) == 2
        assert all(a is b for a, b in zip(encoded, observed))


@pytest.mark.parametrize('local', [False, True])
def test_streamed_and_retained_truncated_share_drafts_losses_and_gradients(local):
    generic, process = _pair(local, draft_query_feedback=True)
    logs = []
    for model in (generic, process):
        reference = copy.deepcopy(model).train()
        batch = _batch()
        output = reference(forecast_inputs(batch), reasoning_steps=3, detach_between_steps=True)
        errors = torch.stack([per_sample_latitude_mse(draft, batch['atmos_target'],
            batch['latitude']).mean() for draft in output.draft_forecasts.unbind(1)])
        weights = torch.linspace(1., 2., 4)
        loss = (errors * (weights / weights.sum())).sum()
        loss.backward()
        result = backward_streamed_truncated(model.train(), batch, reasoning_steps=3, process_weight=0)
        assert torch.equal(output.forecast, result.final_forecast)
        assert torch.equal(errors.detach(), result.draft_errors)
        torch.testing.assert_close(result.total, loss.detach(), rtol=1e-5, atol=1e-6)
        for (name, a), (other, b) in zip(reference.named_parameters(), model.named_parameters()):
            assert name == other and (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6, msg=name)
        logs.append(result)
    for name, value in vars(logs[0]).items():
        assert torch.equal(value, getattr(logs[1], name)), name
    _assert_mapped_gradients(generic, process)


@pytest.mark.parametrize('local', [False, True])
def test_adaptive_force_full_uses_the_same_enabled_step_and_poison_is_filtered(local):
    model = _model(local=local, draft_query_feedback=True).eval()
    clean = _batch()
    poisoned = dict(clean, atmos_target=torch.tensor(float('nan')),
                    atmos_baseline=torch.tensor(float('nan')), process_targets=torch.tensor(float('nan')),
                    future_state=torch.tensor(float('nan')), process_future_targets=torch.tensor(float('nan')))
    with torch.no_grad():
        reference = model(forecast_inputs(clean), reasoning_steps=4)
        changed = model(forecast_inputs(poisoned), reasoning_steps=4)
        adaptive = AdaptiveProcessForecaster(model).eval()(poisoned, max_steps=4, min_steps=4,
                                                          force_full_depth=True)
    assert torch.equal(reference.forecast, changed.forecast)
    assert torch.equal(reference.forecast, adaptive.forecast)
    assert torch.equal(reference.process_predictions[:, -1], adaptive.process_predictions)
    assert adaptive.reasoning_steps_per_sample.tolist() == [4, 4]
    generic, _ = _pair(local, draft_query_feedback=True)
    with torch.no_grad():
        assert torch.equal(generic.eval()(forecast_inputs(clean)).forecast,
                           generic(forecast_inputs(poisoned)).forecast)


def test_no_feedback_source_roles_preserve_context_role_and_disable_only_draft_role():
    generic, process = _pair(source_role_markers=True, use_forecast_feedback=False)
    contexts = []
    for model in (generic, process):
        captured = []
        cell = model.cell if isinstance(model, GenericRecursiveWeatherForecaster) else model.reasoning_cell
        hook = cell.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[1].clone()))
        try:
            output = model(forecast_inputs(_batch()), reasoning_steps=2)
        finally:
            hook.remove()
        assert len(captured) == 2
        assert all(torch.equal(value, output.context_tokens + model.role_context) for value in captured)
        assert model.role_draft.grad is None
        output.forecast.square().mean().backward()
        assert model.role_context.grad is not None and model.role_context.grad.abs().max() > 0
        assert model.role_draft.grad is None
        assert all(parameter.grad is None for parameter in model.draft_encoder.parameters())
        contexts.append(output)
    _assert_same_output(*contexts)
    _assert_mapped_gradients(generic, process)
    for model in (generic, process):
        model.zero_grad(set_to_none=True)
        streamed = backward_streamed_truncated(model.train(), _batch(), reasoning_steps=2, process_weight=0)
        assert torch.equal(streamed.final_forecast, contexts[0].forecast)
    _assert_mapped_gradients(generic, process)
    # The same role configuration works when feedback is disabled per call too.
    for model in _pair(source_role_markers=True):
        assert torch.equal(model(forecast_inputs(_batch()), use_forecast_feedback=False,
                                 reasoning_steps=2).forecast, contexts[0].forecast)
