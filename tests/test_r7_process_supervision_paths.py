"""Forecast/halting isolation and loss gradient ownership on tiny CPU fixtures."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import AdaptiveProcessForecaster, forecast_inputs
from training.r7_process_forecast_losses import (
    auxiliary_process_loss_one_step, process_forecast_coreasoning_loss,
)
from training.r7_process_supervision import process_gradient_ownership
from test_r7_process_supervision_moments import make_batch, make_context


def make_model():
    torch.manual_seed(7)
    return ProcessForecastCoReasoner(in_channels=17, history_steps=2, out_channels=17,
                                    dim=16, depth=1, heads=2, patch_size=1, window_size=4,
                                    dropout=0., anchored_processes=8, free_processes=2,
                                    positional_process_readout=True, pooled_readout_query=True)


class NoFutureRead(dict):
    FORBIDDEN = ("atmos_target", "future_diagnostic_targets", "future_diagnostics", "true_error", "target_time_ns")

    def __getitem__(self, key):
        if key in self.FORBIDDEN:
            raise AssertionError(f"inference read a forbidden future field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.FORBIDDEN:
            raise AssertionError(f"inference queried a forbidden future field {key}")
        return super().get(key, default)


def test_poison_future_changes_only_training_loss_not_forward_or_diagnostics():
    context, batch, model = make_context(), make_batch(), make_model().eval()
    poison = dict(batch, atmos_target=batch["atmos_target"] + 10.,
                  future_diagnostic_targets=torch.full((2, 8), 1e10),
                  process_targets=torch.full((2, 8), -1e10), true_error=torch.full((2,), 1e8))
    with torch.no_grad():
        first = model(forecast_inputs(batch), reasoning_steps=3)
        second = model(forecast_inputs(poison), reasoning_steps=3)
        inference_a = context.fields_from_output(batch, first, training=False)
        inference_b = context.fields_from_output(poison, second, training=False)
        loss_a = process_forecast_coreasoning_loss(batch, first, process_weight=0.,
                                                  future_diagnostic_weight=.05, draft_diagnostic_weight=.05,
                                                  process_supervision_context=context)
        loss_b = process_forecast_coreasoning_loss(poison, second, process_weight=0.,
                                                  future_diagnostic_weight=.05, draft_diagnostic_weight=.05,
                                                  process_supervision_context=context)
    assert torch.equal(first.forecast, second.forecast)
    assert torch.equal(first.process_predictions, second.process_predictions)
    assert torch.equal(first.draft_forecasts, second.draft_forecasts)
    assert torch.equal(inference_a.input_diagnostics, inference_b.input_diagnostics)
    assert torch.equal(inference_a.draft_diagnostics, inference_b.draft_diagnostics)
    assert inference_a.future_diagnostic_targets is inference_b.future_diagnostic_targets is None
    assert not torch.equal(loss_a.future_diagnostic_targets, loss_b.future_diagnostic_targets)
    assert not torch.equal(loss_a.draft_diagnostics, loss_b.draft_diagnostics)
    assert not torch.equal(loss_a.total, loss_b.total)


def test_poison_future_does_not_change_halting_or_read_true_targets():
    batch, model = make_batch(), make_model().eval()
    adapter = AdaptiveProcessForecaster(model).eval()
    poison = dict(batch, atmos_target=torch.full_like(batch["atmos_target"], float("nan")),
                  future_diagnostic_targets=torch.full((2, 8), -1e20), true_error=object())
    with torch.no_grad():
        first = adapter(batch, max_steps=3, min_steps=1, allow_untrained=True)
        second = adapter(NoFutureRead(poison), max_steps=3, min_steps=1, allow_untrained=True)
    for name in ("forecast", "process_predictions", "reasoning_steps_per_sample", "active_masks",
                 "decision_masks", "predicted_gains", "continue_probabilities"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name


def test_target_free_inference_has_complete_input_draft_diagnostics_and_halting():
    context, batch, model = make_context(), make_batch(), make_model().eval()
    batch.pop("atmos_target")
    batch.pop("target_time_ns")
    guarded = NoFutureRead(batch)
    with torch.no_grad():
        out = model(forecast_inputs(guarded), reasoning_steps=3)
        fields = context.fields_from_output(guarded, out, training=False)
        adaptive = AdaptiveProcessForecaster(model).eval()(guarded, max_steps=3, allow_untrained=True)
        final = context.fields_from_output(guarded, SimpleNamespace(draft_forecasts=adaptive.forecast[:, None]),
                                           training=False)
    assert fields.future_diagnostic_targets is None
    assert fields.future_diagnostic_targets_time_ns is None
    assert fields.draft_diagnostics.shape == (2, 4, 8)
    assert final.draft_diagnostics.shape == (2, 1, 8)
    assert torch.isfinite(final.draft_diagnostics).all()
    assert torch.equal(final.draft_diagnostics_time_ns[:, 0], batch["valid_time_ns"])


@pytest.mark.parametrize("task", ["input", "future", "draft"])
def test_real_model_auxiliary_gradient_ownership_and_read_only_norm_report(task):
    context, batch, model = make_context(), make_batch(), make_model().train()
    out = model(forecast_inputs(batch), reasoning_steps=2)
    out.draft_forecasts.retain_grad()
    kwargs = {f"{task}_diagnostic_weight": .1}
    breakdown = process_forecast_coreasoning_loss(batch, out, process_weight=0., forecast_weight=0.,
                                                 process_supervision_context=context, **kwargs)
    component = {"input": breakdown.input_diagnostics, "future": breakdown.future_diagnostic_targets,
                 "draft": breakdown.draft_diagnostics}[task]
    component.backward()
    ownership = process_gradient_ownership(model, out.draft_forecasts)
    assert ownership["process_queries"]["norm"] > 0
    assert ownership["shared"]["norm"] > 0
    assert ownership["readout"]["norm"] > 0
    if task == "draft":
        assert ownership["draft_forecast"]["norm"] > 0
        assert ownership["draft_tensor"]["has_grad"] is True
        assert ownership["draft_tensor"]["norm"] > 0
        assert all(p.grad is None for p in model.process_readout.parameters())
    else:
        assert any(p.grad is not None for p in model.process_readout.parameters())
        assert ownership["draft_tensor"]["has_grad"] is False
    original = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    process_gradient_ownership(model)
    for previous, parameter in zip(original, model.parameters()):
        assert (previous is None) == (parameter.grad is None)
        if previous is not None:
            assert torch.equal(previous, parameter.grad)


def test_one_step_bf16_forward_has_fp32_aux_losses_and_finite_gradients():
    context, batch, model = make_context(), make_batch(), make_model().train()
    fields = context.fields_from_batch(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(forecast_inputs(batch), reasoning_steps=1)
        result = auxiliary_process_loss_one_step(batch, out.draft_forecasts[:, 1], out.process_predictions[:, 0],
                                                process_supervision_context=context, prepared_fields=fields,
                                                future_diagnostic_weight=.05, draft_diagnostic_weight=.05)
    assert result.total.dtype == result.future_diagnostic_targets.dtype == result.draft_diagnostics.dtype == torch.float32
    result.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(gradient.dtype == torch.float32 and torch.isfinite(gradient).all() for gradient in gradients)
    assert process_gradient_ownership(model)["draft_forecast"]["norm"] > 0
