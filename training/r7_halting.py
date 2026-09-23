"""Frozen-forecaster, train-only gain calibration for R7.4 (issue #19)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch.nn import functional as F

from model.r7_halting import AdaptiveProcessForecaster, positive_int


def per_sample_latitude_mse(prediction: torch.Tensor, target: torch.Tensor,
                            latitude: torch.Tensor | None = None) -> torch.Tensor:
    """Equal-channel MSE on normalized fields, reduced in FP32 to [B]."""
    if prediction.ndim != 4 or prediction.shape != target.shape or min(prediction.shape) < 1:
        raise ValueError("prediction/target must have equal nonempty [B,C,H,W] shapes")
    error = (prediction.float() - target.float()).square()
    if not torch.isfinite(error).all():
        raise ValueError("nonfinite forecast error")
    if latitude is None:
        return error.mean((1, 2, 3))
    lat = torch.as_tensor(latitude, dtype=torch.float32, device=error.device)
    b, _, h, _ = prediction.shape
    if lat.shape == (h,):
        lat = lat.unsqueeze(0).expand(b, h)
    if lat.shape != (b, h) or not torch.isfinite(lat).all() or (lat.abs() > 90).any():
        raise ValueError("latitude must be finite [H] or [B,H] within [-90,90]")
    weight = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    total = weight.sum(1)
    if (total <= 0).any():
        raise ValueError("latitude weights have zero area")
    # Average over channels/longitude first, then area-normalize latitude.
    return (error.mean((1, 3)) * weight).sum(1) / total


@torch.no_grad()
def next_step_gain_targets(errors: torch.Tensor, threshold: float = 0.0):
    """errors are E1..EK. Supervise only the K-1 observed transitions."""
    if errors.ndim != 2 or errors.shape[0] < 1 or errors.shape[1] < 2:
        raise ValueError("errors must be [B,K] with K >= 2")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    errors = errors.detach().float()
    if not torch.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("errors must be finite and nonnegative")
    gains = errors[:, :-1] - errors[:, 1:]
    return gains, (gains > threshold).float()


@dataclass
class GainCalibrationLoss:
    total: torch.Tensor
    gain: torch.Tensor
    continuation: torch.Tensor
    target_gains: torch.Tensor
    continue_targets: torch.Tensor


def controller_calibration_loss(adapter: AdaptiveProcessForecaster,
                                batch: Mapping[str, torch.Tensor], *,
                                max_steps: int = 4) -> GainCalibrationLoss:
    """Stream a frozen teacher's drafts; retain only small features and errors.

    This is a training-set operation. Future targets are used ONLY to construct
    training labels, never as inputs to the forecaster/controller. It neither
    changes backbone weights nor guarantees held-out forecast improvements.
    """
    positive_int(max_steps, "max_steps")
    if max_steps < 2:
        raise ValueError("calibration requires at least two reasoning steps")
    target = batch["atmos_target"]
    errors, features = [], []
    # Restore even mixed train/eval module modes after frozen teacher execution.
    modes = [(module, module.training) for module in adapter.forecaster.modules()]
    adapter.forecaster.eval()
    try:
        with torch.no_grad():
            base, process = adapter.initial_state(batch)
            draft = base.forecast
            for step in range(1, max_steps + 1):
                process, draft, delta, prediction = adapter.reasoning_step(
                    process, base.context_tokens, draft, base.token_hw)
                errors.append(per_sample_latitude_mse(draft, target, batch.get("latitude")))
                if step < max_steps:
                    features.append(adapter.controller.features(prediction, draft, delta, step))
    finally:
        for module, training in modes:
            module.training = training
    gains, continues = next_step_gain_targets(
        torch.stack(errors, 1), float(adapter.gain_threshold.item()))
    # Teacher graph has already been released; only controller activations live.
    feat = torch.stack(features, 1)
    predicted_gain, logit = adapter.controller(feat.flatten(0, 1))
    predicted_gain, logit = predicted_gain.view_as(gains), logit.view_as(continues)
    gain_loss = F.smooth_l1_loss(predicted_gain.float(), gains)
    continue_loss = F.binary_cross_entropy_with_logits(logit.float(), continues)
    return GainCalibrationLoss(gain_loss + continue_loss, gain_loss, continue_loss, gains, continues)


def calibrate_controller_step(adapter: AdaptiveProcessForecaster, optimizer: torch.optim.Optimizer,
                              batch: Mapping[str, torch.Tensor], *, max_steps: int = 4,
                              gradient_clip: float = 1.0) -> GainCalibrationLoss:
    """One explicit optimizer update; optimizer ownership is checked first."""
    expected = {id(p) for p in adapter.controller.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if expected != actual:
        raise ValueError("optimizer must contain exactly the trainable controller parameters")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive and finite")
    optimizer.zero_grad(set_to_none=True)
    loss = controller_calibration_loss(adapter, batch, max_steps=max_steps)
    if not torch.isfinite(loss.total):
        raise ValueError("nonfinite calibration loss; optimizer not stepped")
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(adapter.controller.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    adapter.controller.optimizer_updates.add_(1)
    # Do not retain computation graphs in callers' training logs.
    return GainCalibrationLoss(*(value.detach() for value in (
        loss.total, loss.gain, loss.continuation, loss.target_gains, loss.continue_targets)))
