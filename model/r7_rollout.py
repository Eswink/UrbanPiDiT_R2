"""Target-free, fixed-cadence autoregressive rollout for R7 models (issue #22)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .r7_halting import positive_int


def validate_horizons(values: Sequence[int], step_hours: int | None = None) -> tuple[int, ...]:
    horizons = tuple(values)
    if not horizons:
        raise ValueError("at least one lead horizon is required")
    for value in horizons:
        positive_int(value, "lead horizon")
        if step_hours is not None and value % step_hours:
            raise ValueError("lead horizons must be multiples of step_hours")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("lead horizons must be strictly increasing and unique")
    return horizons


@dataclass
class RolloutOutput:
    forecasts: torch.Tensor  # [B,L,C,H,W], snapshots at requested horizons
    lead_hours: tuple[int, ...]
    cumulative_reasoning_steps: torch.Tensor  # [B,L], NOT latency/FLOPs
    model_calls: int


@torch.no_grad()
def autoregressive_rollout(model: nn.Module, batch: Mapping[str, torch.Tensor], *,
                           lead_hours: Sequence[int] = (6, 12, 24, 48, 72),
                           step_hours: int = 6, history_interval_hours: int = 6,
                           inference_kwargs: Mapping | None = None) -> RolloutOutput:
    """Advance all dynamic channels; never use future labels/boundary reanalysis.

    Equal history/forecast cadence is required. Models with fewer outputs than
    input dynamic channels are rejected: missing future channels cannot be
    silently filled from observations. Model mode is not changed implicitly.
    All calls condition on the SINGLE transition lead, not cumulative horizon.
    """
    positive_int(step_hours, "step_hours")
    positive_int(history_interval_hours, "history_interval_hours")
    if history_interval_hours != step_hours:
        raise ValueError("history cadence must equal forecast step cadence")
    horizons = validate_horizons(lead_hours, step_hours)
    if any(module.training for module in model.modules()):
        raise RuntimeError("rollout requires model.eval()")
    kwargs = dict(inference_kwargs or {})
    allowed = {"reasoning_steps", "max_steps", "min_steps", "force_full_depth",
               "allow_untrained", "use_forecast_feedback", "detach_between_steps"}
    if set(kwargs) - allowed:
        raise ValueError("inference_kwargs contains unsupported conditioning fields")
    history = batch["coarse_history"]
    if history.ndim != 5 or min(history.shape) < 1 or not history.is_floating_point():
        raise ValueError("coarse_history must be floating, nonempty [B,T,C,H,W]")
    if not torch.isfinite(history).all():
        raise ValueError("history contains nonfinite values")
    # Clone once to protect callers even if an inference implementation mutates input.
    history = history.detach().clone()
    b, _, c, h, w = history.shape
    single_lead = history.new_full((b,), float(step_hours))
    cumulative = torch.zeros(b, dtype=torch.long, device=history.device)
    fields, work = [], []
    for transition in range(1, horizons[-1] // step_hours + 1):
        # Deliberate input whitelist. Original targets/metadata never reach model.
        out = model({"coarse_history": history,
                     "lead_time_hours": single_lead.clone()}, **kwargs)
        forecast = out.forecast
        if forecast.shape != (b, c, h, w) or forecast.device != history.device:
            raise ValueError("rollout requires every input dynamic channel on the same grid/device")
        if not torch.isfinite(forecast).all():
            raise ValueError("nonfinite forecast; rollout aborted")
        counts = getattr(out, "reasoning_steps_per_sample", None)
        if counts is None:
            depth = getattr(out, "reasoning_steps", 0)
            if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
                raise ValueError("reasoning_steps must be a nonnegative integer")
            counts = torch.full_like(cumulative, depth)
        if counts.shape != (b,) or counts.dtype != torch.long or (counts < 0).any():
            raise ValueError("per-sample reasoning work must be nonnegative int64 [B]")
        cumulative += counts.to(cumulative.device)
        valid_lead = transition * step_hours
        if valid_lead in horizons:
            fields.append(forecast.detach().clone())
            work.append(cumulative.clone())
        history = torch.cat([history[:, 1:], forecast[:, None].to(history.dtype)], dim=1)
    return RolloutOutput(torch.stack(fields, 1), horizons, torch.stack(work, 1), transition)
