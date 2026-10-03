"""Matched Generic V2: complete shared structure, known inputs, no diagnostics.

Weather tensors are synthetic engineering fixtures, not forecasting evidence.
Process weights are explicitly mapped, strictly loaded and compared bitwise;
missing readers, perturbed weights and disconnected gradients must be detected.
"""
from __future__ import annotations

import copy

import pytest
import torch

from model.local_solver_state_r7 import LocalSolverState, PositionGate
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_readout_r7 import PositionalProcessReadout
from model.process_step_r7 import ProcessStepInput, process_reasoning_step
from model.r7_halting import DECLARED_MODEL_INPUTS, forecast_inputs
from model.recursive_weather_r7 import (
    GenericRecursiveCell, GenericRecursiveWeatherForecaster, GenericStepInput,
    generic_reasoning_step, process_to_generic_state_key, recurrent_key,
    solver_conditioning,
)
from training.r7_halting import per_sample_latitude_mse
from training.r7_streaming import backward_streamed_truncated

SMALL = dict(in_channels=3, out_channels=3, history_steps=2, dim=16, depth=1,
             heads=2, window_size=2, patch_size=2, dropout=0.0,
             default_reasoning_steps=4)
RW_A = dict(spacetime_inputs=True, positional_process_readout=True)
RW_B = dict(RW_A, source_role_markers=True, local_solver_state=True,
            solver_state_recurrence=True, solver_gate_proposal=True)
CASES = [pytest.param({}, id="historical-pooled"),
         pytest.param(RW_A, id="rw-a"),
         pytest.param(dict(RW_A, pooled_readout_query=True), id="rw-a-pooled-control"),
         pytest.param(RW_B, id="rw-b"),
         pytest.param(dict(RW_B, solver_state_recurrence=False), id="rw-b-no-recurrence"),
         pytest.param(dict(RW_B, solver_gate_proposal=False), id="rw-b-no-gate-proposal"),
         pytest.param(dict(RW_B, solver_state_recurrence=False, solver_gate_proposal=False),
                      id="rw-b-both-subtractions")]
NEW_SWITCHES = ("positional_process_readout", "pooled_readout_query", "local_solver_state",
                "solver_state_recurrence", "solver_gate_proposal")
POISON_FIELDS = ("atmos_target", "atmos_baseline", "process_targets", "future_state",
                 "process_input_targets", "process_future_targets", "process_draft_targets",
                 "atmos_target_t12", "init_year", "init_time_ns")


def _batch(hw=(5, 7), size=2):
    generator = torch.Generator().manual_seed(7)
    return {"coarse_history": torch.randn(size, 2, 3, *hw, generator=generator),
            "atmos_target": torch.randn(size, 3, *hw, generator=generator),
            "lead_time_hours": torch.tensor([6.0, 12.0])[:size],
            "latitude": torch.linspace(-40, 40, hw[0]),
            "longitude": torch.linspace(-120, 100, hw[1]),
            "init_utc_hour": torch.tensor([18.0, 12.0])[:size],
            "init_day_of_year": torch.tensor([59.0, 365.0])[:size],
            "init_calendar_year": torch.tensor([2024, 2023])[:size],
            "history_offsets_hours": torch.tensor([[-6., 0.], [-6., 0.]])[:size]}


def _generic(switches=None, seed=31, **extra):
    torch.manual_seed(seed)
    return GenericRecursiveWeatherForecaster(**dict(SMALL, latent_tokens=4,
                                                   **(switches or {}), **extra))


def _mapped_state(process):
    mapped = {}
    for name, tensor in process.state_dict().items():
        key = process_to_generic_state_key(name)
        if key is not None:
            assert key not in mapped, f"non-injective map at {name}"
            mapped[key] = tensor.clone()
    return mapped


def _pair(switches=None, seed=31, **extra):
    options = dict(switches or {}, **extra)
    generic = _generic(options, seed)
    torch.manual_seed(seed)
    process = ProcessForecastCoReasoner(**dict(SMALL, anchored_processes=2,
                                              free_processes=2, **options))
    mapped, target = _mapped_state(process), generic.state_dict()
    assert set(mapped) == set(target), "a forecast-facing weight was omitted"
    for name, value in mapped.items():
        assert value.shape == target[name].shape, name
        assert value.dtype == target[name].dtype, name
    generic.load_state_dict(mapped, strict=True)
    return generic, process


def _assert_tensor_equal(actual, expected, name):
    if expected is None:
        assert actual is None, name
    else:
        assert actual is not None and torch.equal(actual, expected), name


def _assert_outputs_equal(generic, process):
    for name in ("forecast", "initial_forecast", "draft_forecasts", "final_correction",
                 "context_tokens", "solver_state"):
        _assert_tensor_equal(getattr(generic, name), getattr(process, name), name)
    _assert_tensor_equal(generic.latent_state, process.process_state, "recurrent state")
    assert generic.token_hw == process.token_hw
    assert generic.reasoning_steps == process.reasoning_steps


def _assert_mapped_gradients_equal(generic, process):
    target = dict(generic.named_parameters())
    for name, parameter in process.named_parameters():
        key = process_to_generic_state_key(name)
        if key is None:
            assert parameter.grad is None, f"diagnostic head trained by forecast loss: {name}"
        else:
            _assert_tensor_equal(target[key].grad, parameter.grad, key)
            if target[key].grad is not None:
                assert torch.isfinite(target[key].grad).all(), key


def _assert_group_gradient(model, prefix):
    gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
    assert gradients and any(g is not None and g.abs().max() > 0 for g in gradients), prefix
    assert all(torch.isfinite(g).all() for g in gradients if g is not None), prefix


def _retained_loss(model, batch, steps):
    output = model(forecast_inputs(batch), reasoning_steps=steps, detach_between_steps=True)
    errors = torch.stack([per_sample_latitude_mse(draft, batch["atmos_target"],
                                                 batch["latitude"]).mean()
                          for draft in output.draft_forecasts.unbind(1)])
    weights = torch.linspace(1.0, 2.0, steps + 1)
    weights /= weights.sum()
    return output, errors, (weights * errors).sum()


@pytest.mark.parametrize("switches", CASES)
def test_parameter_map_and_module_structure_include_the_complete_reader_and_solver(switches):
    generic, process = _pair(switches)
    assert not isinstance(generic, ProcessForecastCoReasoner)
    assert not any(hasattr(generic, name) for name in
                   ("anchored_processes", "free_processes", "process_readout", "_process_prediction"))
    assert isinstance(generic.cell, GenericRecursiveCell)
    assert type(generic.cell) is type(process.reasoning_cell)
    modules = dict(generic.named_modules())
    mapped_modules = {}
    for name, module in process.named_modules():
        if name == "process_readout" or name.startswith("process_readout."):
            continue
        key = process_to_generic_state_key(name + ".").removesuffix(".")
        mapped_modules[key] = module
        if name:
            assert type(modules[key]) is type(module), name
    assert set(modules) == set(mapped_modules)
    diagnostic = {n for n in process.state_dict() if process_to_generic_state_key(n) is None}
    assert diagnostic == {"process_readout.0.weight", "process_readout.0.bias",
                          "process_readout.1.weight", "process_readout.1.bias"}
    assert sum(p.numel() for p in process.parameters()) - sum(
        p.numel() for p in generic.parameters()) == 3 * SMALL["dim"] + 1
    if generic.positional_process_readout:
        assert isinstance(generic.process_reader, PositionalProcessReadout)
        assert generic.process_reader.dim == process.process_reader.dim == SMALL["dim"]
        assert generic.process_reader.heads == process.process_reader.heads == SMALL["heads"]
    if generic.local_solver_state:
        assert isinstance(generic.solver_cell, LocalSolverState)
        assert isinstance(generic.solver_gate, PositionGate)
        assert type(generic.proposal_head) is type(process.proposal_head)


@pytest.mark.parametrize("local", [False, True])
def test_audited_dim192_parameter_gap_is_only_the_577_parameter_diagnostic_head(local):
    config = dict(in_channels=17, out_channels=17, history_steps=2, dim=192, depth=4,
                  heads=4, window_size=4, patch_size=2, dropout=0.0)
    switches = dict(RW_A, local_solver_state=local)
    generic = GenericRecursiveWeatherForecaster(**config, latent_tokens=16, **switches,
                                                solver_state_recurrence=local,
                                                solver_gate_proposal=local)
    process = ProcessForecastCoReasoner(**config, anchored_processes=8, free_processes=8,
                                       **switches)
    assert set(_mapped_state(process)) == set(generic.state_dict())
    difference = sum(p.numel() for p in process.parameters()) - sum(
        p.numel() for p in generic.parameters())
    assert difference == 577
    if not local:
        assert sum(p.numel() for p in generic.parameters()) == 2_967_682


@pytest.mark.parametrize("switches", [RW_A, RW_B])
def test_cell_state_mapping_and_shapes_are_verified_before_the_forecast(switches):
    generic, process = _pair(switches)
    for name, value in generic.cell.state_dict().items():
        assert torch.equal(value, process.reasoning_cell.state_dict()[name]), name
    with torch.no_grad():
        base = generic.backbone(forecast_inputs(_batch()))
        draft, _ = generic.draft_encoder(base.forecast)
        key = recurrent_key(base.context_tokens, draft)
        latent = generic.latent.expand(2, -1, -1)
        actual, expected = generic.cell(latent, key), process.reasoning_cell(latent, key)
    assert actual.shape == expected.shape == (2, 4, 16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("switches", CASES)
@pytest.mark.parametrize("steps", [0, 1, 2, 4])
def test_mapped_forward_is_bitwise_equivalent_at_each_depth(switches, steps):
    generic, process = _pair(switches)
    with torch.no_grad():
        actual = generic.eval()(forecast_inputs(_batch()), reasoning_steps=steps)
        expected = process.eval()(forecast_inputs(_batch()), reasoning_steps=steps)
    _assert_outputs_equal(actual, expected)
    assert actual.forecast.shape == (2, 3, 5, 7)
    assert actual.draft_forecasts.shape == (2, steps + 1, 3, 5, 7)
    assert actual.latent_state.shape == (2, 4, 16)
    assert not hasattr(actual, "process_predictions")
    if generic.local_solver_state and generic.solver_state_recurrence and steps:
        assert actual.solver_state.shape == actual.context_tokens.shape == (2, 12, 16)
    else:
        assert actual.solver_state is None
    if steps == 0:
        assert torch.equal(actual.forecast, actual.initial_forecast)
        assert actual.final_correction.abs().max() == 0


@pytest.mark.parametrize("switches", CASES)
@pytest.mark.parametrize("checkpointed,detach", [(False, False), (True, False), (False, True)])
def test_full_forecast_loss_and_every_mapped_gradient_are_bitwise_equivalent(
        switches, checkpointed, detach):
    generic, process = _pair(switches, activation_checkpointing=checkpointed)
    batch = forecast_inputs(_batch())
    generic_history = batch["coarse_history"].clone().requires_grad_()
    process_history = batch["coarse_history"].clone().requires_grad_()
    actual = generic.train()(dict(batch, coarse_history=generic_history),
                             reasoning_steps=3, detach_between_steps=detach)
    expected = process.train()(dict(batch, coarse_history=process_history),
                               reasoning_steps=3, detach_between_steps=detach)
    _assert_outputs_equal(actual, expected)
    for output in (actual, expected):
        output.draft_forecasts.square().mean().backward()
    _assert_mapped_gradients_equal(generic, process)
    assert torch.equal(generic_history.grad, process_history.grad)
    constant_proposal = (generic.local_solver_state and generic.solver_gate_proposal
                         and not generic.solver_state_recurrence)
    for prefix in ("cell.", "draft_encoder.", "process_reader."):
        if prefix == "process_reader." and not generic.positional_process_readout:
            continue
        if constant_proposal:
            # Process's declared subtraction computes but never consumes this read.
            assert all(p.grad is None for name, p in generic.named_parameters()
                       if name.startswith(prefix)), prefix
        else:
            _assert_group_gradient(generic, prefix)
    if generic.local_solver_state:
        if generic.solver_state_recurrence and generic.solver_gate_proposal:
            _assert_group_gradient(generic, "solver_cell.")
        else:
            assert all(p.grad is None for p in generic.solver_cell.parameters())
        if generic.solver_gate_proposal:
            _assert_group_gradient(generic, "proposal_head.")
            _assert_group_gradient(generic, "solver_gate.")


def test_strict_mapping_and_forward_comparison_detect_missing_or_changed_reader_weights():
    generic, process = _pair(RW_A)
    mapped = _mapped_state(process)
    name = "process_reader.attention.v.weight"
    missing = dict(mapped)
    del missing[name]
    with pytest.raises(RuntimeError, match="Missing key"):
        generic.load_state_dict(missing, strict=True)
    wrong_shape = dict(mapped, **{name: mapped[name][:1]})
    with pytest.raises(RuntimeError, match="size mismatch"):
        generic.load_state_dict(wrong_shape, strict=True)
    changed = dict(mapped, **{name: torch.zeros_like(mapped[name])})
    generic.load_state_dict(changed, strict=True)
    with torch.no_grad():
        actual = generic.eval()(forecast_inputs(_batch()))
        expected = process.eval()(forecast_inputs(_batch()))
    with pytest.raises(AssertionError, match="forecast"):
        _assert_outputs_equal(actual, expected)


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_reader_runs_the_same_n_by_m_compute_in_generic_process_and_pooled_control(pooled, streamed):
    generic, process = _pair(dict(RW_A, pooled_readout_query=pooled))
    traces = []
    for model in (generic, process):
        shapes = []
        handle = model.process_reader.attention.register_forward_pre_hook(
            lambda module, inputs: shapes.append(tuple(tuple(t.shape) for t in inputs)))
        try:
            if streamed:
                backward_streamed_truncated(model.train(), _batch(), reasoning_steps=3,
                                            process_weight=0)
            else:
                model.eval()(forecast_inputs(_batch()), reasoning_steps=3)
        finally:
            handle.remove()
        traces.append(shapes)
    assert traces[0] == traces[1] == [((2, 12, 16), (2, 4, 16))] * 3
    with torch.no_grad():
        base = generic.backbone(forecast_inputs(_batch()))
        read = generic.latent_conditioning(generic.latent.expand(2, -1, -1),
                                           base.context_tokens, base.token_hw)
    assert read.shape == (2, 12, 16)
    flat = torch.equal(read, read[:, :1].expand_as(read))
    assert flat is pooled


@pytest.mark.parametrize("switches", [RW_A, RW_B])
def test_known_input_whitelist_and_target_diagnostic_baseline_poison_do_not_change_forecasts(switches):
    generic, process = _pair(switches)
    clean = forecast_inputs(_batch())
    poisoned = dict(clean, **{name: torch.tensor(float("nan")) for name in POISON_FIELDS})
    assert set(forecast_inputs(poisoned)) == set(clean) == set(DECLARED_MODEL_INPUTS)
    assert not set(POISON_FIELDS).intersection(clean)
    for model in (generic, process):
        seen = []
        hook = model.backbone.register_forward_pre_hook(
            lambda module, inputs: seen.append(set(inputs[0])))
        try:
            with torch.no_grad():
                expected = model.eval()(clean)
                actual = model(forecast_inputs(poisoned))
        finally:
            hook.remove()
        assert seen == [set(DECLARED_MODEL_INPUTS)] * 2
        assert torch.equal(actual.forecast, expected.forecast)
        assert torch.equal(actual.draft_forecasts, expected.draft_forecasts)
        assert torch.isfinite(actual.forecast).all()
        # A deliberately bypassed whitelist exposes the forbidden baseline.
        direct = dict(clean, atmos_baseline=torch.full_like(expected.forecast, float("nan")))
        with torch.no_grad():
            assert not torch.isfinite(model(direct).forecast).all()


@pytest.mark.parametrize("field", DECLARED_MODEL_INPUTS)
def test_every_declared_known_input_is_shared_and_affects_the_context(field):
    generic, process = _pair(RW_A, known_context_inputs=True)
    inputs = forecast_inputs(_batch())
    changed = dict(inputs)
    shift = -1 if field == 'init_day_of_year' else (1 if field == 'init_calendar_year' else 0.5)
    changed[field] = (torch.tensor([[-12., 0.], [-6., 0.]])
                      if field == 'history_offsets_hours' else inputs[field] + shift)
    with torch.no_grad():
        reference = generic.eval()(inputs)
        actual = generic(changed)
        expected = process.eval()(changed)
    _assert_outputs_equal(actual, expected)
    assert not torch.equal(reference.context_tokens, actual.context_tokens), field
    assert not torch.equal(reference.forecast, actual.forecast), field


@pytest.mark.parametrize("switches", [RW_A, RW_B])
def test_reader_gradient_probe_detects_a_disconnected_reader(switches):
    live = _generic(switches).train()
    live(forecast_inputs(_batch())).forecast.square().mean().backward()
    _assert_group_gradient(live, "process_reader.")
    broken = _generic(switches).train()
    hook = broken.process_reader.register_forward_hook(lambda module, inputs, output: output.detach())
    try:
        broken(forecast_inputs(_batch())).forecast.square().mean().backward()
    finally:
        hook.remove()
    with pytest.raises(AssertionError, match="process_reader"):
        _assert_group_gradient(broken, "process_reader.")


@pytest.mark.parametrize("switches", [RW_A, RW_B])
def test_encoded_draft_has_a_nonzero_gradient_to_latent_and_the_probe_detects_cut_feedback(switches):
    def gradient(cut):
        model, captured = _generic(switches).train(), []
        def capture(module, inputs, output):
            tokens, hw = output
            captured.append(tokens)
            return (tokens.detach(), hw) if cut else output
        hook = model.draft_encoder.register_forward_hook(capture)
        try:
            output = model(forecast_inputs(_batch()), reasoning_steps=2)
            return torch.autograd.grad(output.latent_state.square().mean(), captured[0],
                                       allow_unused=True)[0]
        finally:
            hook.remove()
    live = gradient(False)
    assert live is not None and torch.isfinite(live).all() and live.abs().max() > 0
    assert gradient(True) is None


@pytest.mark.parametrize("switches", CASES)
def test_fixed_truncated_and_streamed_share_all_drafts_losses_and_parameter_gradients(switches):
    reference = _generic(switches).train()
    streamed = copy.deepcopy(reference)
    batch = _batch()
    output, errors, loss = _retained_loss(reference, batch, 4)
    loss.backward()
    result = backward_streamed_truncated(streamed, batch, reasoning_steps=4, process_weight=0)
    assert torch.equal(result.final_forecast, output.forecast)
    assert torch.equal(result.draft_errors, errors.detach())
    torch.testing.assert_close(result.total, loss.detach(), rtol=1e-5, atol=1e-6)
    for (name, a), (other, b) in zip(reference.named_parameters(), streamed.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6, msg=name)
            assert torch.isfinite(b.grad).all(), name
    assert result.process == 0
    assert all(not value.requires_grad for value in vars(result).values())


@pytest.mark.parametrize("switches", [RW_A, RW_B])
@pytest.mark.parametrize("amp_dtype", [None, torch.bfloat16])
def test_streamed_generic_and_process_mapped_loss_and_gradients_are_bitwise_equal(switches, amp_dtype):
    generic, process = _pair(switches)
    results = [backward_streamed_truncated(model.train(), _batch(), reasoning_steps=3,
                                           process_weight=0, amp_dtype=amp_dtype)
               for model in (generic, process)]
    for name, value in vars(results[0]).items():
        assert torch.equal(value, getattr(results[1], name)), name
        assert torch.isfinite(value).all(), name
    assert results[0].total.dtype == torch.float32
    _assert_mapped_gradients_equal(generic, process)
    _assert_group_gradient(generic, "process_reader.")
    if generic.local_solver_state:
        _assert_group_gradient(generic, "solver_cell.")
        _assert_group_gradient(generic, "proposal_head.")


@pytest.mark.parametrize("switches", [RW_A, RW_B])
def test_full_bf16_generic_and_process_forecasts_and_gradients_are_finite_and_equal(switches):
    generic, process = _pair(switches)
    outputs = []
    for model in (generic, process):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model.train()(forecast_inputs(_batch()), reasoning_steps=3)
        output.forecast.float().square().mean().backward()
        outputs.append(output)
    _assert_outputs_equal(*outputs)
    _assert_mapped_gradients_equal(generic, process)


@pytest.mark.parametrize("switches", [RW_A, RW_B])
def test_standalone_generic_step_and_process_step_agree_and_require_the_rw_b_anchor(switches):
    generic, process = _pair(switches)
    base = generic.backbone(forecast_inputs(_batch()))
    latent, process_state = generic.latent.expand(2, -1, -1), process.process_queries.expand(2, -1, -1)
    draft, process_draft = base.forecast, base.forecast
    solver, process_solver = None, None
    for step in range(3):
        result = generic_reasoning_step(generic, GenericStepInput(latent, base.context_tokens, draft),
            base.token_hw, solver_state=solver, step_index=step, anchor=base.base_state)
        other = process_reasoning_step(process,
            ProcessStepInput(process_state, base.context_tokens, process_draft), base.token_hw,
            solver_state=process_solver, step_index=step, anchor=base.base_state)
        for name in ("draft", "correction", "solver_state"):
            _assert_tensor_equal(getattr(result, name), getattr(other, name), name)
        assert torch.equal(result.latent, other.process)
        assert not hasattr(result, "prediction")
        latent, draft, solver = result.latent, result.draft, result.solver_state
        process_state, process_draft, process_solver = other.process, other.draft, other.solver_state
    if generic.local_solver_state:
        with pytest.raises(ValueError, match="anchor"):
            generic_reasoning_step(generic, GenericStepInput(latent, base.context_tokens, draft), base.token_hw)
        with pytest.raises(ValueError, match="use_forecast_feedback"):
            generic(forecast_inputs(_batch()), use_forecast_feedback=False)


def test_all_new_switches_default_off_add_no_weights_and_explicit_off_is_bitwise():
    plain = _generic()
    plain_rng = torch.random.get_rng_state().clone()
    explicit = _generic({name: False for name in NEW_SWITCHES})
    assert torch.equal(plain_rng, torch.random.get_rng_state())
    for name in NEW_SWITCHES:
        assert getattr(plain, name) is False
    assert set(plain.state_dict()) == set(explicit.state_dict())
    for name, value in plain.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    for model in (plain, explicit):
        model.train()(forecast_inputs(_batch())).forecast.square().mean().backward()
    for (name, a), (other, b) in zip(plain.named_parameters(), explicit.named_parameters()):
        assert name == other
        _assert_tensor_equal(a.grad, b.grad, name)
    with torch.no_grad():
        assert torch.equal(plain.eval()(forecast_inputs(_batch())).forecast,
                           explicit.eval()(forecast_inputs(_batch())).forecast)


@pytest.mark.parametrize("switches", [RW_A, RW_B, dict(RW_A, pooled_readout_query=True)])
def test_added_components_are_initialized_last_with_an_isolated_random_stream(switches):
    plain = _generic()
    expected_rng = torch.random.get_rng_state().clone()
    added = _generic(switches)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for name, value in plain.state_dict().items():
        assert torch.equal(value, added.state_dict()[name]), name
    assert any(name.startswith("process_reader.") for name in added.state_dict())


def test_default_step_is_the_original_pooled_update_operation_for_operation():
    model = _generic().eval()
    with torch.no_grad():
        base = model.backbone(forecast_inputs(_batch()))
        latent, draft = model.latent.expand(2, -1, -1), base.forecast
        tokens, _ = model.draft_encoder(draft)
        expected_latent = model._cell(latent, torch.cat([base.context_tokens, tokens], dim=1))
        summary = model.latent_to_context(expected_latent.mean(dim=1))
        conditioned = solver_conditioning(base.context_tokens, summary, tokens)
        expected_draft, expected_correction = model.correction_head(
            conditioned, base.token_hw, draft.shape[-2:], draft)
        actual = generic_reasoning_step(model, GenericStepInput(latent, base.context_tokens, draft),
                                       base.token_hw)
    assert torch.equal(actual.latent, expected_latent)
    assert torch.equal(actual.draft, expected_draft)
    assert torch.equal(actual.correction, expected_correction)
    assert actual.solver_state is None


@pytest.mark.parametrize("name", NEW_SWITCHES + ("use_forecast_feedback",))
def test_switches_are_boolean_only(name):
    with pytest.raises(ValueError, match=name):
        _generic({name: 1})


@pytest.mark.parametrize("options,field", [
    ({"pooled_readout_query": True}, "positional_process_readout"),
    ({"local_solver_state": True}, "positional_process_readout"),
    (dict(RW_B, use_forecast_feedback=False), "use_forecast_feedback"),
    ({"solver_state_recurrence": True}, "local_solver_state"),
    ({"solver_gate_proposal": True}, "local_solver_state"),
])
def test_invalid_switch_combinations_are_rejected_instead_of_silently_ignored(options, field):
    with pytest.raises(ValueError, match=field):
        _generic(options)


def test_feedback_off_is_an_explicit_matched_capacity_ablation_not_an_ignored_input():
    generic, process = _pair(dict(RW_A, use_forecast_feedback=False))
    outputs = [model.train()(forecast_inputs(_batch()), reasoning_steps=2)
               for model in (generic, process)]
    _assert_outputs_equal(*outputs)
    for output in outputs:
        output.forecast.square().mean().backward()
    _assert_mapped_gradients_equal(generic, process)
    with pytest.raises(AssertionError, match="draft_encoder"):
        _assert_group_gradient(generic, "draft_encoder.")
    _assert_group_gradient(generic, "cell.")
