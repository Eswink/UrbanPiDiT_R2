"""Single-device streamed truncated backward; full-BPTT paths stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster, solver_conditioning
from model.r7_halting import forecast_inputs
from .r7_halting import per_sample_latitude_mse


@dataclass
class StreamedBackwardResult:
    total: torch.Tensor
    forecast: torch.Tensor
    process: torch.Tensor
    draft_errors: torch.Tensor  # detached [K+1] scalars, not full draft graphs
    final_forecast: torch.Tensor  # only the final detached field


def _nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _validate(model, batch, steps, amp_dtype):
    if not isinstance(model, (GenericRecursiveWeatherForecaster, ProcessForecastCoReasoner)):
        raise TypeError("only the R7 generic/process single-device models are supported")
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError("streamed backward requires train() and enabled gradients")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("reasoning_steps must be a nonnegative integer")
    if amp_dtype not in (None, torch.bfloat16):
        raise ValueError("only FP32 or explicit BF16 autocast is supported; no FP16 scaler")
    history, target = batch["coarse_history"], batch["atmos_target"]
    if history.ndim != 5 or min(history.shape) < 1:
        raise ValueError("coarse_history must be nonempty [B,T,C,H,W]")
    if target.shape != (history.shape[0], model.out_channels, *history.shape[-2:]):
        raise ValueError("atmos_target shape does not match model outputs")
    params = list(model.parameters())
    if not any(p.requires_grad for p in params):
        raise ValueError("model has no trainable parameters")
    if any(p.dtype != torch.float32 or p.device != history.device for p in params):
        raise ValueError("parameters must be FP32 and on the history device")
    if target.device != history.device or history.device.type not in ("cpu", "cuda"):
        raise ValueError("target/history must share a supported CPU/CUDA device")
    if torch.is_autocast_enabled(history.device.type):
        raise ValueError("pass amp_dtype explicitly, outside any autocast context")
    return history.device.type


def _recursive_step(model, state, context, draft, token_hw):
    process_model = isinstance(model, ProcessForecastCoReasoner)
    feedback = not process_model or model.use_forecast_feedback
    recurrent_context = context
    tokens = None
    if feedback:
        tokens, hw = model.draft_encoder(draft)
        if tuple(hw) != tuple(token_hw):
            raise ValueError("draft/context token grids differ")
        recurrent_context = torch.cat([context, tokens], 1)
    if process_model:
        state = model._reason(state, recurrent_context)
        prediction = model._process_prediction(state)
        summary = model.process_to_context(state.mean(1))
    else:
        state = model._cell(state, recurrent_context)
        prediction = None
        summary = model.latent_to_context(state.mean(1))
    conditioned = solver_conditioning(context, summary, tokens,
        spatial_feedback=model.spatial_solver_feedback and feedback)
    draft, _ = model.correction_head(
        conditioned, token_hw, draft.shape[-2:], draft)
    return state, draft, prediction


def backward_streamed_truncated(model, batch: Mapping[str, torch.Tensor], *,
                                reasoning_steps: int = 4, final_weight: float = 2.0,
                                forecast_weight: float = 1.0, process_weight: float = 0.1,
                                loss_scale: float = 1.0,
                                amp_dtype: torch.dtype | None = None) -> StreamedBackwardResult:
    """Accumulate gradients without zeroing or stepping an optimizer.

    Matches detach_between_steps=True (Y0 is still differentiable through round
    one). One backbone graph is retained; each recurrent graph is freed before
    the next forward. Detached interface leaves collect adjoints which are sent
    through BOTH original backbone roots once, at the end. No retain_graph.

    Losses/reductions are FP32. This is truncated BPTT, not full BPTT. Targets
    are used only in training losses. On failure, callers must clear gradients;
    train_streamed_update handles that cleanup. Distributed/scaled-FP16 training
    is deliberately outside this utility's contract.
    """
    device_type = _validate(model, batch, reasoning_steps, amp_dtype)
    fw = _nonnegative(forecast_weight, "forecast_weight")
    pw = _nonnegative(process_weight, "process_weight")
    scale = _nonnegative(loss_scale, "loss_scale")
    if not math.isfinite(final_weight) or final_weight <= 0 or fw <= 0 or scale <= 0:
        raise ValueError("final_weight, forecast_weight and loss_scale must be positive")
    target = batch["atmos_target"].detach().float()
    latitude = batch.get("latitude")
    process_target = None
    if isinstance(model, ProcessForecastCoReasoner) and pw > 0 and "process_targets" in batch:
        process_target = batch["process_targets"].detach().float()
        if process_target.shape != (target.shape[0], model.anchored_processes):
            raise ValueError("process_targets must exactly match [B, anchored_processes]")
        if process_target.device != target.device or not torch.isfinite(process_target).all():
            raise ValueError("process targets must be finite and on the target device")

    def autocast():
        return torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp_dtype is not None)

    with autocast():
        base = model.backbone(forecast_inputs(batch))
    initial = base.forecast.detach().requires_grad_(base.forecast.requires_grad)
    context = base.context_tokens.detach().requires_grad_(base.context_tokens.requires_grad)
    weights = torch.linspace(1.0, final_weight, reasoning_steps + 1,
                             device=target.device, dtype=torch.float32)
    weights /= weights.sum()
    draft = initial
    query = model.process_queries if isinstance(model, ProcessForecastCoReasoner) else model.latent
    state = query.expand(target.shape[0], -1, -1)
    errors = []
    forecast_log, process_log = target.new_zeros(()), target.new_zeros(())
    for step in range(reasoning_steps + 1):
        prediction = None
        if step:
            with autocast():
                state, draft, prediction = _recursive_step(model, state, context, draft, base.token_hw)
        mse = per_sample_latitude_mse(draft, target, latitude).mean()
        ploss = target.new_zeros(())
        if step and process_target is not None:
            ploss = F.mse_loss(prediction.float(), process_target) / reasoning_steps
        term = (fw * weights[step] * mse + pw * ploss) * scale
        if not torch.isfinite(term):
            raise ValueError("nonfinite loss; optimizer must not step")
        if term.requires_grad:
            term.backward()  # independent recurrent graph ends here
        errors.append(mse.detach())
        forecast_log += weights[step] * mse.detach()
        process_log += ploss.detach()
        if step:  # preserve Y0 -> round 1 exactly as the existing truncated path
            state, draft = state.detach(), draft.detach()
        del term, mse, ploss, prediction

    roots, adjoints = [], []
    for root, leaf in ((base.forecast, initial), (base.context_tokens, context)):
        if root.requires_grad and leaf.grad is not None:
            roots.append(root)
            adjoints.append(leaf.grad)
    if roots:
        torch.autograd.backward(roots, adjoints)  # summed VJP; encoder graph consumed once
    return StreamedBackwardResult(
        fw * forecast_log + pw * process_log, forecast_log, process_log,
        torch.stack(errors), draft.detach())


def train_streamed_update(model, optimizer: torch.optim.Optimizer,
                          microbatches: Sequence[Mapping[str, torch.Tensor]], *,
                          gradient_clip: float = 1.0, **kwargs) -> list[StreamedBackwardResult]:
    """One update for an explicit microbatch group, weighted by sample counts.

    Performs no weight mutation until every backward succeeds. Uneven last
    microbatches do not get overweighted. Return values are graph-free logs.
    The caller owns epochs, data loading, scheduling and checkpointing.
    """
    if not microbatches or not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("nonempty microbatches and positive finite gradient_clip required")
    if "loss_scale" in kwargs:
        raise ValueError("loss_scale is managed by sample-weighted accumulation")
    params = [p for p in model.parameters() if p.requires_grad]
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(params) or {id(p) for p in actual} != {id(p) for p in params}:
        raise ValueError("optimizer must contain exactly the trainable model parameters")
    sizes = [int(b["coarse_history"].shape[0]) for b in microbatches]
    if min(sizes) < 1:
        raise ValueError("empty microbatches are unsupported")
    optimizer.zero_grad(set_to_none=True)
    logs = []
    try:
        for batch, size in zip(microbatches, sizes):
            logs.append(backward_streamed_truncated(
                model, batch, loss_scale=size / sum(sizes), **kwargs))
        torch.nn.utils.clip_grad_norm_(params, gradient_clip, error_if_nonfinite=True)
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return logs
