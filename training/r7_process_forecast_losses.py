"""Forecast supervision and separately named input/future/draft process tasks."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import torch
from torch.nn import functional as F

from .r7_recursive_losses import deep_supervised_forecast_mse
from .r7_process_supervision import (
    ProcessDiagnosticContext, ProcessDiagnosticFields, validate_diagnostic_fields,
)


@dataclass
class ProcessDiagnosticLossBreakdown:
    total: torch.Tensor
    input_diagnostics: torch.Tensor
    future_diagnostic_targets: torch.Tensor
    draft_diagnostics: torch.Tensor


@dataclass
class ProcessForecastLossBreakdown:
    total: torch.Tensor
    forecast: torch.Tensor
    process: torch.Tensor  # legacy input-time process loss only
    input_diagnostics: torch.Tensor | None = None
    future_diagnostic_targets: torch.Tensor | None = None
    draft_diagnostics: torch.Tensor | None = None


def _nonnegative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _aux_weights(input_diagnostic_weight, future_diagnostic_weight, draft_diagnostic_weight):
    return tuple(_nonnegative(value, name) for value, name in (
        (input_diagnostic_weight, "input_diagnostic_weight"),
        (future_diagnostic_weight, "future_diagnostic_weight"),
        (draft_diagnostic_weight, "draft_diagnostic_weight"),
    ))


def _predictions(value, fields):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 3 or min(value.shape) < 1
            or value.shape[0] != fields.input_diagnostics.shape[0] or value.shape[-1] != 8
            or not bool(torch.isfinite(value).all())):
        raise ValueError("enabled process predictions must exactly match finite [B,K,8]; no width truncation")
    return value


def _target_mse(prediction, target, active):
    if prediction.shape[0] != target.shape[0] or prediction.shape[-1] != target.shape[-1]:
        raise ValueError("enabled process loss target/prediction width or batch mismatch")
    target = target.detach().to(device=prediction.device, dtype=torch.float32)
    mask = active.to(prediction.device)
    expanded = target[:, None, :].expand_as(prediction)
    return F.mse_loss(prediction.float()[..., mask], expanded[..., mask])


def process_diagnostic_loss(
    process_predictions: torch.Tensor | None, fields: ProcessDiagnosticFields,
    *, input_diagnostic_weight: float = 0.0, future_diagnostic_weight: float = 0.0,
    draft_diagnostic_weight: float = 0.0,
) -> ProcessDiagnosticLossBreakdown:
    """Unweighted component MSEs and their weighted sum; no per-K budget policy.

    The two process-head tasks compare process_predictions to differently sourced
    targets. Draft supervision compares D(Y_k) directly to train D(Y*), retaining
    gradients through the fixed physical-unit inverse and spherical differences.
    Active channels alone enter the mean; masked channels cannot amplify noise.
    """
    iw, fw, dw = _aux_weights(input_diagnostic_weight, future_diagnostic_weight, draft_diagnostic_weight)
    active = validate_diagnostic_fields(fields, need_input=iw > 0,
                                       need_future=fw > 0 or dw > 0, need_draft=dw > 0)
    zero = fields.input_diagnostics.new_zeros(())
    if not (iw or fw or dw):
        return ProcessDiagnosticLossBreakdown(zero, zero, zero, zero)
    if iw or fw:
        prediction = _predictions(process_predictions, fields)
    if dw and (iw or fw) and fields.draft_diagnostics.shape[:2] != prediction.shape[:2]:
        raise ValueError("draft/process reasoning-step shape mismatch")
    with torch.autocast(fields.input_diagnostics.device.type, enabled=False):
        inputs = _target_mse(prediction, fields.input_diagnostics, active) if iw else zero
        future = _target_mse(prediction, fields.future_diagnostic_targets, active) if fw else zero
        draft = _target_mse(fields.draft_diagnostics, fields.future_diagnostic_targets, active) if dw else zero
        total = iw * inputs + fw * future + dw * draft
    if not bool(torch.isfinite(total)):
        raise ValueError("nonfinite auxiliary process loss")
    return ProcessDiagnosticLossBreakdown(total, inputs, future, draft)


def auxiliary_process_loss_one_step(
    batch: Mapping, draft: torch.Tensor, prediction: torch.Tensor | None, *,
    process_supervision_context: ProcessDiagnosticContext,
    prepared_fields: ProcessDiagnosticFields | None = None,
    input_diagnostic_weight: float = 0.0, future_diagnostic_weight: float = 0.0,
    draft_diagnostic_weight: float = 0.0,
) -> ProcessDiagnosticLossBreakdown:
    """Streamed loss for ONE Y_k/p_k, k>=1; caller owns division by K.

    Cache fields_from_batch(..., need_future=fw>0 or dw>0) once per batch. This
    function never calls a model and never mutates batch/output or halting state.
    Future/observed fields stay detached; the draft diagnostic stays differentiable.
    """
    iw, fw, dw = _aux_weights(input_diagnostic_weight, future_diagnostic_weight, draft_diagnostic_weight)
    if not isinstance(process_supervision_context, ProcessDiagnosticContext):
        raise ValueError("enabled auxiliary tasks require process_supervision_context metadata")
    if not isinstance(draft, torch.Tensor) or draft.ndim != 4:
        raise ValueError("one-step draft must be [B,17,H,W]")
    fields = prepared_fields
    if fields is None:
        fields = process_supervision_context.fields_from_batch(batch, training=True, need_future=bool(fw or dw))
    process_supervision_context.validate_prepared_fields(fields, batch, need_future=bool(fw or dw))
    if dw:
        fields = process_supervision_context.with_drafts(fields, draft, batch["latitude"], batch["longitude"])
    predictions = None if prediction is None else prediction.unsqueeze(1)
    return process_diagnostic_loss(predictions, fields, input_diagnostic_weight=iw,
                                   future_diagnostic_weight=fw, draft_diagnostic_weight=dw)


def _legacy_process_loss(batch, predictions):
    target = batch["process_targets"].to(device=predictions.device, dtype=predictions.dtype)
    if (target.ndim != 2 or predictions.ndim != 3
            or target.shape != (predictions.shape[0], predictions.shape[-1])
            or predictions.shape[-1] < 1):
        raise ValueError("enabled legacy process targets must exactly match prediction width; no truncation")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(predictions).all()):
        raise ValueError("nonfinite legacy process predictions/targets")
    expanded = target.unsqueeze(1).expand(-1, predictions.shape[1], -1)
    return F.mse_loss(predictions, expanded)


def process_forecast_coreasoning_loss(
    batch: Mapping[str, torch.Tensor], out, *, forecast_weight: float = 1.0,
    process_weight: float = 0.1, final_weight: float = 2.0,
    input_diagnostic_weight: float = 0.0, future_diagnostic_weight: float = 0.0,
    draft_diagnostic_weight: float = 0.0,
    process_supervision_context: ProcessDiagnosticContext | None = None,
    diagnostic_fields: ProcessDiagnosticFields | None = None,
) -> ProcessForecastLossBreakdown:
    """Old default/disabled numerical path is preserved; new tasks default off.

    New switches are not aliases for process_weight: enabling both old and new
    supervision is an error. No task/threshold/0.1 budget is silently selected.
    Full loss averages Y1..YK diagnostic MSE (not Y0), matching streamed one-step
    loss accumulated with /K. fields_from_output can still expose all Y0..YK.
    """
    fw = _nonnegative(forecast_weight, "forecast_weight")
    pw = _nonnegative(process_weight, "process_weight")
    final = _nonnegative(final_weight, "final_weight")
    iw, future_w, dw = _aux_weights(input_diagnostic_weight, future_diagnostic_weight, draft_diagnostic_weight)
    enabled = bool(iw or future_w or dw)
    if pw > 0 and enabled:
        raise ValueError("legacy process_weight and new diagnostic weights cannot both be enabled")
    if enabled and not isinstance(process_supervision_context, ProcessDiagnosticContext):
        raise ValueError("enabled diagnostic tasks require process_supervision_context metadata")
    forecast = deep_supervised_forecast_mse(out.draft_forecasts, batch["atmos_target"],
                                          batch.get("latitude"), final_weight=final)
    process = forecast.new_zeros(())
    if pw > 0 and "process_targets" in batch and out.process_predictions.shape[1] > 0:
        process = _legacy_process_loss(batch, out.process_predictions)
    # Keep the exact legacy expression/order when all new switches are off.
    total = fw * forecast + pw * process
    inputs = future = draft = forecast.new_zeros(())
    if enabled:
        if out.draft_forecasts.ndim != 5 or out.draft_forecasts.shape[1] < 2:
            raise ValueError("enabled auxiliary tasks require at least one reasoning step")
        fields = diagnostic_fields
        if fields is None:
            fields = process_supervision_context.fields_from_batch(
                batch, training=True, need_future=bool(future_w or dw))
        process_supervision_context.validate_prepared_fields(fields, batch, need_future=bool(future_w or dw))
        if dw:
            times = getattr(out, "draft_time_ns", None)
            if times is not None:
                if tuple(times.shape) != tuple(out.draft_forecasts.shape[:2]):
                    raise ValueError("draft_time_ns must match all draft reasoning steps")
                times = times[:, 1:]
            fields = process_supervision_context.with_drafts(
                fields, out.draft_forecasts[:, 1:], batch["latitude"], batch["longitude"], draft_time_ns=times)
        elif fields.draft_diagnostics is not None:
            fields = replace(fields, draft_diagnostics=None, draft_diagnostics_time_ns=None)
        if (iw or future_w) and out.process_predictions.shape[1] != out.draft_forecasts.shape[1] - 1:
            raise ValueError("process predictions/draft reasoning-step count mismatch")
        aux = process_diagnostic_loss(out.process_predictions, fields, input_diagnostic_weight=iw,
                                     future_diagnostic_weight=future_w, draft_diagnostic_weight=dw)
        inputs, future, draft = aux.input_diagnostics, aux.future_diagnostic_targets, aux.draft_diagnostics
        total = total + aux.total
    return ProcessForecastLossBreakdown(total, forecast, process, inputs, future, draft)
