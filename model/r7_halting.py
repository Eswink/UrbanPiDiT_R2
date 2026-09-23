"""Opt-in R7.4 gain controller and active-subset inference (issue #19)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, TYPE_CHECKING

import torch
from torch import nn
from .recursive_weather_r7 import solver_conditioning

if TYPE_CHECKING:
    from .process_forecast_r7 import ProcessForecastCoReasoner


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def forecast_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Whitelist initialization-time fields; never forward targets/baselines."""
    result = {"coarse_history": batch["coarse_history"]}
    if "lead_time_hours" in batch:
        result["lead_time_hours"] = batch["lead_time_hours"]
    return result


class ForecastGainController(nn.Module):
    """Predict signed next-step MSE gain and a CONTINUE logit.

    The summaries are detached deliberately: stage-one controller fitting must
    not change the forecasting backbone to make stopping artificially easy.
    No current-error, target, or observed process label is an input feature.
    """

    def __init__(self, processes: int, channels: int, hidden: int = 64):
        super().__init__()
        self.processes = positive_int(processes, "processes")
        self.channels = positive_int(channels, "channels")
        self.feature_dim = processes + 3 * channels + 1
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, positive_int(hidden, "hidden")),
            nn.SiLU(), nn.Linear(hidden, 2),
        )
        self.register_buffer("optimizer_updates", torch.zeros((), dtype=torch.long))

    def features(self, process: torch.Tensor, draft: torch.Tensor,
                 correction: torch.Tensor, step: int) -> torch.Tensor:
        positive_int(step, "step")
        if draft.ndim != 4 or draft.shape != correction.shape:
            raise ValueError("draft/correction must have equal [B,C,H,W] shapes")
        if draft.shape[1] != self.channels or process.shape != (draft.shape[0], self.processes):
            raise ValueError("controller channel/process dimensions do not match")
        # FP32 summaries prevent half-precision squaring overflow.
        p, y, d = process.detach().float(), draft.detach().float(), correction.detach().float()
        return torch.cat([
            p, y.mean((-2, -1)), y.square().mean((-2, -1)).sqrt(),
            d.square().mean((-2, -1)).sqrt(),
            y.new_full((y.shape[0], 1), math.log1p(step)),
        ], dim=-1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("controller expects [B, feature_dim]")
        result = self.net(features.detach())
        return result[:, 0].float(), result[:, 1].float()


@dataclass
class AdaptiveForecastOutput:
    forecast: torch.Tensor
    process_predictions: torch.Tensor
    reasoning_steps_per_sample: torch.Tensor
    active_masks: torch.Tensor       # [B,S], true means that sample ran step s
    decision_masks: torch.Tensor     # [B,S], true means the controller was queried
    predicted_gains: torch.Tensor    # zero where decision_masks is false
    continue_probabilities: torch.Tensor


class AdaptiveProcessForecaster(nn.Module):
    """Wrap a pretrained R7.3 forecaster without changing its fixed-depth path.

    Start with at least one reasoning step. Inference caches one context and one
    current forecast; it does NOT compute all K drafts and select one afterwards.
    A controller with zero optimizer updates is rejected unless explicitly
    overridden for engineering tests. Updates alone do not certify calibration.
    """

    def __init__(self, forecaster: ProcessForecastCoReasoner, hidden: int = 64,
                 gain_threshold: float = 0.0, probability_threshold: float = 0.5):
        super().__init__()
        if not math.isfinite(gain_threshold) or gain_threshold < 0:
            raise ValueError("gain_threshold must be finite and nonnegative")
        if not math.isfinite(probability_threshold) or not 0 < probability_threshold < 1:
            raise ValueError("probability_threshold must be between zero and one")
        self.forecaster = forecaster
        self.controller = ForecastGainController(
            forecaster.anchored_processes, forecaster.out_channels, hidden)
        # Persist policy thresholds with weights for reproducible restoration.
        self.register_buffer("gain_threshold", torch.tensor(float(gain_threshold)))
        self.register_buffer("probability_threshold", torch.tensor(float(probability_threshold)))

    def initial_state(self, batch):
        base = self.forecaster.backbone(forecast_inputs(batch))
        if base.forecast.shape[0] < 1:
            raise ValueError("empty batches are unsupported")
        process = self.forecaster.process_queries.expand(base.forecast.shape[0], -1, -1)
        return base, process

    def reasoning_step(self, process, context, draft, token_hw):
        """Exactly the fixed R7.3 recurrence; equivalence is regression-tested."""
        model = self.forecaster
        recurrent_context = context
        draft_tokens = None
        if model.use_forecast_feedback:
            draft_tokens, draft_hw = model.draft_encoder(draft)
            if tuple(draft_hw) != tuple(token_hw):
                raise ValueError("draft/context token grids differ")
            recurrent_context = torch.cat([context, draft_tokens], dim=1)
        process = model._reason(process, recurrent_context)
        prediction = model._process_prediction(process)
        summary = model.process_to_context(process.mean(dim=1))
        conditioned = solver_conditioning(context, summary, draft_tokens,
            spatial_feedback=model.spatial_solver_feedback and model.use_forecast_feedback)
        draft, correction = model.correction_head(
            conditioned, token_hw, draft.shape[-2:], draft)
        return process, draft, correction, prediction

    @torch.no_grad()
    def forward(self, batch: Mapping[str, torch.Tensor], *, max_steps: int = 4,
                min_steps: int = 1, force_full_depth: bool = False,
                allow_untrained: bool = False) -> AdaptiveForecastOutput:
        positive_int(max_steps, "max_steps")
        positive_int(min_steps, "min_steps")
        if min_steps > max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if self.training or self.forecaster.training or self.controller.training:
            raise RuntimeError("adaptive inference requires eval()")
        needs_controller = not force_full_depth and min_steps < max_steps
        if needs_controller and not allow_untrained and self.controller.optimizer_updates.item() == 0:
            raise RuntimeError("controller has no training updates; calibrate before inference")

        base, process = self.initial_state(batch)
        context, draft = base.context_tokens, base.forecast
        process = process.clone()
        batch_size = draft.shape[0]
        active = torch.ones(batch_size, dtype=torch.bool, device=draft.device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=draft.device)
        pp = draft.new_zeros(batch_size, self.forecaster.anchored_processes)
        active_trace, decision_trace, gain_trace, probability_trace = [], [], [], []

        for step in range(1, max_steps + 1):
            selected = active.nonzero(as_tuple=False).flatten()
            active_trace.append(active.clone())
            p, y, delta, prediction = self.reasoning_step(
                process[selected], context[selected], draft[selected], base.token_hw)
            process[selected] = p.to(process.dtype)
            draft[selected] = y.to(draft.dtype)
            pp[selected] = prediction.to(pp.dtype)
            steps[selected] += 1
            decisions = torch.zeros_like(active)
            gains = torch.zeros(batch_size, dtype=torch.float32, device=draft.device)
            probabilities = torch.zeros_like(gains)
            if needs_controller and min_steps <= step < max_steps:
                features = self.controller.features(prediction, y, delta, step)
                gain, logit = self.controller(features)
                probability = torch.sigmoid(logit)
                decisions[selected] = True
                gains[selected], probabilities[selected] = gain, probability
                finite = torch.isfinite(gain) & torch.isfinite(logit)
                keep = (gain > self.gain_threshold) & (probability >= self.probability_threshold)
                # Invalid controller output cannot justify an early exit.
                active[selected] = keep | ~finite
            decision_trace.append(decisions)
            gain_trace.append(gains)
            probability_trace.append(probabilities)
            if not bool(active.any()):
                break

        return AdaptiveForecastOutput(
            draft, pp, steps, torch.stack(active_trace, 1), torch.stack(decision_trace, 1),
            torch.stack(gain_trace, 1), torch.stack(probability_trace, 1))
