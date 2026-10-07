"""Target-free, fixed-cadence autoregressive rollout for R7 models (issue #22)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .r7_halting import positive_int
from .known_context_r7 import HISTORY_CONTEXT_FIELDS, require_history_offsets, uses_known_context
from .spacetime_conditioning_r7 import (
    CALENDAR_INPUT_FIELDS, SPACETIME_INPUT_FIELDS, advance_calendar_time, require_spacetime_fields,
)


def conditions_on_accumulated_lead(model: nn.Module) -> bool:
    """Whether this model's declared inputs include the *accumulated* lead.

    A model that conditions on the initialization-time phase fields needs the
    lead it is actually at: the phase is ``init + lead``, and every transition
    moves the valid time by one cadence step. The pooled baseline models take the
    single transition lead they were trained with, so the rollout asks the model
    which convention it was configured for instead of assuming one. The answer
    comes from the same declared switch the model's own forward branches on.
    Explicit calendar metadata supersedes accumulated lead: the rollout advances
    its known initialization date and keeps every physical transition's lead fixed.
    """
    forecaster = getattr(model, "forecaster", model)
    return bool(getattr(forecaster, "spacetime_inputs", False))


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


def rollout_model_input(sample: Mapping[str, torch.Tensor], *, lead_hours: float = 6.0,
                        device: torch.device | None = None) -> dict[str, torch.Tensor]:
    """Model inputs for one *unbatched* held-out rollout window.

    The batch dimension is added to the history only. Grid coordinates stay
    ``[H]``/``[W]`` (they describe the token grid, not the samples), and the
    declared initialization-time fields are carried through the same whitelist
    the training paths use, so the training batch, the validation rollout and the
    published evaluation cannot disagree about what the model is given.
    """
    from .r7_halting import forecast_inputs

    batch = forecast_inputs(sample)
    history = batch["coarse_history"]
    if history.ndim != 4:
        raise ValueError("an unbatched [T,C,H,W] history is required for one window")
    batch["coarse_history"] = history.unsqueeze(0)
    if "history_offsets_hours" in batch and torch.is_tensor(batch["history_offsets_hours"]):
        batch["history_offsets_hours"] = batch["history_offsets_hours"].unsqueeze(0)
    batch["lead_time_hours"] = history.new_full((1,), float(lead_hours))
    if device is not None:
        batch = {name: value.to(device) for name, value in batch.items()}
    return batch


@torch.no_grad()
def autoregressive_rollout(model: nn.Module, batch: Mapping[str, torch.Tensor], *,
                           lead_hours: Sequence[int] = (6, 12, 24, 48, 72),
                           step_hours: int = 6, history_interval_hours: int = 6,
                           inference_kwargs: Mapping | None = None) -> RolloutOutput:
    """Advance all dynamic channels; never use future labels/boundary reanalysis.

    Equal history/forecast cadence is required. Models with fewer outputs than
    input dynamic channels are rejected: missing future channels cannot be
    silently filled from observations. Model mode is not changed implicitly.
    Pooled models condition on the SINGLE transition lead, not cumulative
    horizon. The archived space-time route without calendar metadata retains its
    accumulated-lead convention. With explicit init_calendar_year, each transition
    advances known year/day/hour and keeps lead fixed at step_hours; phase therefore
    matches the Gregorian valid date without feeding a 72h lead to a 6h transition.

    Grid coordinates are unchanged. Caller calendar tensors are never mutated;
    targets, baselines and undeclared metadata never reach the model.
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
    declared = {name: batch[name] for name in SPACETIME_INPUT_FIELDS + CALENDAR_INPUT_FIELDS
                + HISTORY_CONTEXT_FIELDS if name in batch}
    if uses_known_context(model):
        if "history_offsets_hours" not in declared:
            raise KeyError("known_context_inputs requires history_offsets_hours for physical rollout")
        require_history_offsets(declared["history_offsets_hours"], batch_size=b,
            history_steps=history.shape[1], device=history.device, cadence_hours=step_hours)
    from .climatology_anchor_r7 import uses_climatology_anchor
    climate_mode = uses_climatology_anchor(model)
    if climate_mode and step_hours != 6:
        raise ValueError('climatology_anchor rollout requires a fixed 6-hour transition')
    if climate_mode and 'init_calendar_year' not in declared:
        raise KeyError('climatology_anchor rollout requires explicit init_calendar_year')
    accumulated_lead = conditions_on_accumulated_lead(model)
    calendar = None
    if climate_mode or (accumulated_lead and "init_calendar_year" in declared):
        _, _, hour, day = require_spacetime_fields(batch, history_shape=(h, w), batch_size=b)
        calendar = advance_calendar_time(declared["init_calendar_year"], day, hour,
                                         history.new_zeros(b, dtype=torch.float64))
    cumulative = torch.zeros(b, dtype=torch.long, device=history.device)
    fields, work = [], []
    for transition in range(1, horizons[-1] // step_hours + 1):
        # Deliberate input whitelist. Original targets/metadata never reach model.
        step = {"coarse_history": history}
        step.update(declared)
        if calendar is not None:
            year, day, hour = calendar
            step.update(init_calendar_year=year, init_day_of_year=day, init_utc_hour=hour)
        step["lead_time_hours"] = history.new_full(
            (b,), float(transition * step_hours
                        if accumulated_lead and calendar is None else step_hours))
        out = model(step, **kwargs)
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
        if calendar is not None:
            calendar = advance_calendar_time(calendar[0], calendar[1], calendar[2],
                                             history.new_full((b,), float(step_hours),
                                                              dtype=torch.float64))
    return RolloutOutput(torch.stack(fields, 1), horizons, torch.stack(work, 1), transition)
