"""Differentiable +6/+12 h training, separate from no-grad evaluation rollout.

The two axes are explicit: each physical transition runs K reasoning steps with
``detach_between_steps=False``; the second transition consumes the first final
draft without detach. On each physical step the initial and all K drafts use the
existing normalized K-axis deep supervision (weights linearly 1..2, sum to 1).
The independent physical axis is L6 + lambda12 * L12 in normalized space.
Targets never enter a model input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from model.r7_halting import forecast_inputs, positive_int
from model.spacetime_conditioning_r7 import advance_calendar_time, require_spacetime_fields
from model.known_context_r7 import require_history_offsets, uses_known_context
from .r7_halting import per_sample_latitude_mse
from .r7_recursive_losses import deep_supervised_forecast_mse

STEP_HOURS = 6


@dataclass
class TrainingRolloutOutput:
    forecasts: torch.Tensor  # [B,physical_steps,C,H,W]; graph intentionally retained
    loss: torch.Tensor
    l6: torch.Tensor
    l12: torch.Tensor | None


def _sample_field(value, name, history):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.device != history.device or value.ndim > 1:
        raise ValueError(f"{name} must be scalar or [B] on the history device")
    field = value.reshape(-1)
    if value.ndim == 0:
        field = field.expand(history.shape[0])
    if field.shape != (history.shape[0],) or not torch.isfinite(field).all():
        raise ValueError(f"{name} must be finite with one value per sample")
    return field


def _advance_inputs(inputs):
    """Rebase known initialization metadata by +6, keeping transition lead at 6.

    Calendar year is optional, and old init_year is never read. Without an
    explicit year an unambiguous interior-day advance is valid; an unknown leap
    day/year-boundary carry raises rather than guessing a year or rolling to an
    illegal ordinal. The Zarr producer always supplies the explicit year.
    """
    result = dict(inputs)
    history = inputs["coarse_history"]
    hour = inputs.get("init_utc_hour")
    day = inputs.get("init_day_of_year")
    year = inputs.get("init_calendar_year")
    if hour is None and day is None and year is None:
        return result
    if hour is None or day is None:
        raise ValueError("calendar advance requires both init_utc_hour and init_day_of_year")
    hour = _sample_field(hour, "init_utc_hour", history).float()
    day = _sample_field(day, "init_day_of_year", history).float()
    if ((hour < 0) | (hour >= 24)).any() or ((day < 1) | (day > 366) | (day != day.floor())).any():
        raise ValueError("invalid UTC hour or integer day of year")
    lead = history.new_full((history.shape[0],), float(STEP_HOURS), dtype=torch.float32)
    if year is not None:
        year = _sample_field(year, "init_calendar_year", history)
        valid_year, valid_day, valid_hour = advance_calendar_time(year, day, hour, lead)
        result["init_calendar_year"] = valid_year
        result["init_day_of_year"] = valid_day
        result["init_utc_hour"] = valid_hour
    else:
        elapsed = hour + STEP_HOURS
        valid_day = day + torch.floor(elapsed / 24)
        if ((day >= 365) & (valid_day > day)).any():
            raise ValueError("init_calendar_year required for a leap/year-boundary day carry; init_year is not read")
        result["init_day_of_year"] = valid_day
        result["init_utc_hour"] = elapsed.remainder(24)
    return result


def _training_inputs(model, batch, reasoning_steps):
    positive_int(reasoning_steps, "reasoning_steps")
    if not torch.is_grad_enabled() or not model.training:
        raise RuntimeError("differentiable training requires grad-enabled model.train(), not evaluation rollout")
    inputs = forecast_inputs(batch)
    history = inputs["coarse_history"]
    if not torch.is_tensor(history) or history.ndim != 5 or min(history.shape) < 1:
        raise ValueError("coarse_history must be nonempty [B,T,C,H,W]")
    if not history.is_floating_point() or not torch.isfinite(history).all():
        raise ValueError("coarse_history must be finite floating fields")
    if "lead_time_hours" in inputs:
        lead = _sample_field(inputs["lead_time_hours"], "lead_time_hours", history)
        if (lead != STEP_HOURS).any():
            raise ValueError("both transition lead embeddings must be fixed at 6 hours")
    inputs["lead_time_hours"] = history.new_full((history.shape[0],), float(STEP_HOURS), dtype=torch.float32)
    if uses_known_context(model):
        if "history_offsets_hours" not in inputs:
            raise KeyError("known_context_inputs requires history_offsets_hours for training rollout")
        require_history_offsets(inputs["history_offsets_hours"], batch_size=history.shape[0],
            history_steps=history.shape[1], device=history.device, cadence_hours=STEP_HOURS)
    if getattr(model, "spacetime_inputs", False):
        require_spacetime_fields(inputs, history_shape=history.shape[-2:], batch_size=history.shape[0])
    return inputs


def _forward_final(model, inputs, reasoning_steps):
    # This is the model's ordinary differentiable forward, NOT the streamed
    # trainer or the @no_grad adaptive/evaluation rollout. Explicit False also
    # overrides a transferred parent's truncated internal-K configuration.
    output = model(inputs, reasoning_steps=reasoning_steps, detach_between_steps=False)
    forecast = output.forecast
    history = inputs["coarse_history"]
    expected = (history.shape[0], *history.shape[2:])
    if (not torch.is_tensor(forecast) or forecast.shape != expected
            or forecast.device != history.device or not forecast.is_floating_point()):
        raise ValueError("forecast must include every dynamic channel on the same [B,C,H,W] grid/device")
    if not torch.isfinite(forecast).all():
        raise ValueError("nonfinite training forecast")
    if not forecast.requires_grad:
        raise RuntimeError("training forecast has no gradient graph; no-grad/detached inference is forbidden")
    if getattr(output, "reasoning_steps", reasoning_steps) != reasoning_steps:
        raise ValueError("training forward did not run the declared K reasoning steps")
    drafts = getattr(output, "draft_forecasts", None)
    if (not torch.is_tensor(drafts) or drafts.shape != (expected[0], reasoning_steps + 1, *expected[1:])
            or drafts.device != forecast.device or not drafts.is_floating_point()):
        raise ValueError("initial and all K draft_forecasts are required; no final-only fallback")
    if not torch.isfinite(drafts).all():
        raise ValueError("nonfinite training drafts")
    if not drafts.requires_grad or not torch.equal(drafts[:, -1], forecast):
        raise ValueError("final draft must equal the pushed forecast and retain its gradient graph")
    return forecast, drafts


def _draft_loss(forecast, drafts, target, latitude):
    if not torch.is_tensor(target) or not target.is_floating_point() or target.device != forecast.device:
        raise ValueError("training target must be floating on the forecast device")
    # Strict shape/finite/latitude validation before the existing objective; this
    # guard does not substitute a different loss or change its normalized weights.
    per_sample_latitude_mse(forecast, target, latitude)
    with torch.autocast(forecast.device.type, enabled=False):
        loss = deep_supervised_forecast_mse(drafts.float(), target.float(), latitude, final_weight=2.)
    if loss.dtype != torch.float32 or not torch.isfinite(loss):
        raise ValueError("deep-supervised training loss must be finite FP32")
    return loss


def training_one_step(model: nn.Module, batch: Mapping[str, torch.Tensor],
                      reasoning_steps: int = 4) -> TrainingRolloutOutput:
    """L6 control with the same full-K path and normalized deep supervision."""
    inputs = _training_inputs(model, batch, reasoning_steps)
    forecast, drafts = _forward_final(model, inputs, reasoning_steps)
    loss = _draft_loss(forecast, drafts, batch["atmos_target"], batch.get("latitude"))
    return TrainingRolloutOutput(forecast[:, None], loss, loss, None)


def training_two_step(model: nn.Module, batch: Mapping[str, torch.Tensor],
                      reasoning_steps: int = 4, lambda12: float = .5) -> TrainingRolloutOutput:
    """Two model calls; second history's final frame equals pred1, with full BPTT.

    ``atmos_target`` supervises +6 only; ``future_target`` supervises +12 only.
    Coordinates remain known static inputs. Initialization UTC/day/year advance
    by one physical step, while *both* lead embeddings remain the single 6 h
    transition lead. No target, future observation, process proxy or old
    ``init_year`` can pass the shared forecast-input whitelist.
    """
    if isinstance(lambda12, bool) or not isinstance(lambda12, (int, float)) or not math.isfinite(lambda12) or lambda12 < 0:
        raise ValueError("lambda12 must be finite and nonnegative")
    inputs = _training_inputs(model, batch, reasoning_steps)
    next_inputs = _advance_inputs(inputs)
    first, first_drafts = _forward_final(model, inputs, reasoning_steps)
    history = inputs["coarse_history"]
    # Promote instead of rounding pred1 to a lower precision history: second's
    # last history frame must be numerically equal to pred1, also under BF16.
    dtype = torch.promote_types(history.dtype, first.dtype)
    next_inputs["coarse_history"] = torch.cat(
        [history[:, 1:].to(dtype), first[:, None].to(dtype)], dim=1)
    second, second_drafts = _forward_final(model, next_inputs, reasoning_steps)
    l6 = _draft_loss(first, first_drafts, batch["atmos_target"], batch.get("latitude"))
    l12 = _draft_loss(second, second_drafts, batch["future_target"], batch.get("latitude"))
    loss = l6 + lambda12 * l12
    if not torch.isfinite(loss):
        raise ValueError("nonfinite two-step training loss")
    return TrainingRolloutOutput(torch.stack((first, second), dim=1), loss, l6, l12)
