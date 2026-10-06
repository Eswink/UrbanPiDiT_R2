"""Direct multi-physical-step supervision with full physical and internal-K BPTT.

Targets only supervise losses. Every 6 h transition, including zero-weighted
ones, pushes its final forecast into the next history without detaching. This
module selects neither a scientific horizon/threshold nor a training stage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import torch
from torch import nn

from .r7_autoregressive_rollout import _advance_inputs, _draft_loss, _forward_final, _training_inputs


@dataclass
class LongTrainingRolloutOutput:
    forecasts: torch.Tensor  # [B,N,C,H,W], graph retained
    loss: torch.Tensor
    per_step_losses: torch.Tensor  # [N], normalized all-draft supervision
    l6: torch.Tensor
    l12: torch.Tensor | None


def validate_physical_weights(physical_weights) -> tuple[float, ...]:
    """Finite nonnegative numeric sequence, nonempty with at least one positive."""
    if isinstance(physical_weights, (str, bytes)):
        raise ValueError("physical_weights must be a sequence of finite nonnegative numeric weights")
    try:
        supplied = tuple(physical_weights)
    except TypeError as error:
        raise ValueError("physical_weights must be a sequence of finite nonnegative numeric weights") from error
    weights = []
    for value in supplied:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("physical_weights must be finite nonnegative numeric weights; bool/string forbidden")
        try:
            weight = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("physical_weights must be finite nonnegative numeric weights") from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("physical_weights must be finite nonnegative numeric weights")
        weights.append(weight)
    if not weights or not any(weight > 0 for weight in weights):
        raise ValueError("physical_weights must be nonempty with at least one positive weight")
    return tuple(weights)


def _physical_targets(batch, history, physical_steps):
    targets = batch["physical_targets"]
    expected = (history.shape[0], physical_steps, *history.shape[2:])
    if (not torch.is_tensor(targets) or not targets.is_floating_point()
            or targets.shape != expected or targets.device != history.device or targets.dtype != history.dtype):
        raise ValueError("physical_targets must be floating [B,N,C,H,W] on the exact history grid/device/dtype")
    if not torch.isfinite(targets).all():
        raise ValueError("physical_targets must be finite")
    return targets


def training_long_rollout(model: nn.Module, batch: Mapping[str, torch.Tensor], reasoning_steps: int = 4,
                          *, physical_weights) -> LongTrainingRolloutOutput:
    """Weighted sum of unchanged K-axis draft losses over N physical 6 h steps.

    N is the number of physical weights, with no maximum horizon here. Known
    initialization calendar fields advance before every subsequent call; lead
    remains 6 h and regular relative history offsets remain unchanged. History
    and forecast dtypes are promoted, never rounded down when pushing a frame.
    ``physical_targets``, compatibility targets and old ``init_year`` cannot
    pass the existing model-input whitelist.
    """
    weights = validate_physical_weights(physical_weights)
    inputs = _training_inputs(model, batch, reasoning_steps)
    targets = _physical_targets(batch, inputs["coarse_history"], len(weights))
    forecasts, losses = [], []
    for step in range(len(weights)):
        forecast, drafts = _forward_final(model, inputs, reasoning_steps)
        forecasts.append(forecast)
        losses.append(_draft_loss(forecast, drafts, targets[:, step], batch.get("latitude")))
        if step + 1 < len(weights):
            next_inputs = _advance_inputs(inputs)
            history = inputs["coarse_history"]
            dtype = torch.promote_types(history.dtype, forecast.dtype)
            next_inputs["coarse_history"] = torch.cat(
                [history[:, 1:].to(dtype), forecast[:, None].to(dtype)], dim=1)
            inputs = next_inputs
    # Sequential addition preserves the existing L6 + .5*L12 arithmetic for N=2.
    loss = weights[0] * losses[0]
    for weight, step_loss in zip(weights[1:], losses[1:]):
        loss = loss + weight * step_loss
    if not torch.isfinite(loss):
        raise ValueError("nonfinite long-rollout training loss")
    per_step_losses = torch.stack(losses)
    return LongTrainingRolloutOutput(torch.stack(forecasts, dim=1), loss, per_step_losses,
                                     losses[0], losses[1] if len(weights) >= 2 else None)
