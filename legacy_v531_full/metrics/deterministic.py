from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch


Reduction = str


def finite_mask(*tensors: torch.Tensor) -> torch.Tensor:
    if not tensors:
        raise ValueError("at least one tensor is required")
    mask = torch.ones_like(tensors[0], dtype=torch.bool)
    for tensor in tensors:
        if tensor.shape != tensors[0].shape:
            raise ValueError(f"shape mismatch: {tuple(tensor.shape)} vs {tuple(tensors[0].shape)}")
        mask = mask & torch.isfinite(tensor)
    return mask


def _dims_for_reduction(x: torch.Tensor, reduction: Reduction) -> Optional[Tuple[int, ...]]:
    reduction = str(reduction)
    if reduction == "none":
        return None
    if reduction == "mean":
        return tuple(range(x.ndim))
    if reduction == "channel_mean":
        if x.ndim < 2:
            raise ValueError("channel_mean requires a channel dimension at dim=1")
        return tuple(i for i in range(x.ndim) if i != 1)
    if reduction == "batch_mean":
        if x.ndim < 1:
            raise ValueError("batch_mean requires a batch dimension at dim=0")
        return tuple(i for i in range(x.ndim) if i != 0)
    raise ValueError(f"unknown reduction={reduction!r}")


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None, *, reduction: Reduction = "mean") -> torch.Tensor:
    if mask is None:
        mask = torch.isfinite(x)
    else:
        mask = mask.to(device=x.device, dtype=torch.bool) & torch.isfinite(x)
    dims = _dims_for_reduction(x, reduction)
    safe = torch.where(mask, x, torch.zeros_like(x))
    if dims is None:
        return torch.where(mask, x, torch.full_like(x, float("nan")))
    count = mask.sum(dim=dims)
    total = safe.sum(dim=dims)
    value = total / count.clamp_min(1).to(dtype=x.dtype)
    return torch.where(count > 0, value, torch.full_like(value, float("nan")))


def mse(pred: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean") -> torch.Tensor:
    mask = finite_mask(pred, target)
    return masked_mean((pred - target).square(), mask, reduction=reduction)


def rmse(pred: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean") -> torch.Tensor:
    return torch.sqrt(mse(pred, target, reduction=reduction))


def mae(pred: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean") -> torch.Tensor:
    mask = finite_mask(pred, target)
    return masked_mean(torch.abs(pred - target), mask, reduction=reduction)


def bias(pred: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean") -> torch.Tensor:
    mask = finite_mask(pred, target)
    return masked_mean(pred - target, mask, reduction=reduction)


def pearson_corr(pred: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean", eps: float = 1e-8) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    mask = finite_mask(pred, target)
    dims = _dims_for_reduction(pred, reduction)
    if dims is None:
        raise ValueError("pearson_corr does not support reduction='none'")

    p_mean = masked_mean(pred, mask, reduction=reduction)
    t_mean = masked_mean(target, mask, reduction=reduction)
    for dim in sorted(dims):
        p_mean = p_mean.unsqueeze(dim)
        t_mean = t_mean.unsqueeze(dim)

    p = torch.where(mask, pred - p_mean, torch.zeros_like(pred))
    t = torch.where(mask, target - t_mean, torch.zeros_like(target))
    num = (p * t).sum(dim=dims)
    den = torch.sqrt((p.square().sum(dim=dims)) * (t.square().sum(dim=dims)))
    value = num / den.clamp_min(eps)
    return torch.where(den > eps, value, torch.full_like(value, float("nan")))


def deterministic_metrics(pred: torch.Tensor, target: torch.Tensor, *, reduction: Reduction = "mean") -> Dict[str, torch.Tensor]:
    return {
        "mse": mse(pred, target, reduction=reduction),
        "rmse": rmse(pred, target, reduction=reduction),
        "mae": mae(pred, target, reduction=reduction),
        "bias": bias(pred, target, reduction=reduction),
        "corr": pearson_corr(pred, target, reduction=reduction),
    }


__all__ = [
    "finite_mask",
    "masked_mean",
    "mse",
    "rmse",
    "mae",
    "bias",
    "pearson_corr",
    "deterministic_metrics",
]