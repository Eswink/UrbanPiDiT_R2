from __future__ import annotations

from typing import Dict

import torch

from .deterministic import finite_mask, masked_mean, rmse


def _validate_ensemble(ensemble: torch.Tensor, target: torch.Tensor | None = None) -> tuple[int, int, int, int, int]:
    if ensemble.ndim != 5:
        raise ValueError(f"ensemble must be [B,M,C,H,W], got {tuple(ensemble.shape)}")
    b, m, c, h, w = ensemble.shape
    if target is not None and target.shape != (b, c, h, w):
        raise ValueError(f"target shape {tuple(target.shape)} does not match ensemble {(b, m, c, h, w)}")
    return b, m, c, h, w


def ensemble_mean(ensemble: torch.Tensor) -> torch.Tensor:
    _validate_ensemble(ensemble)
    finite = torch.isfinite(ensemble)
    count = finite.sum(dim=1)
    total = torch.where(finite, ensemble, torch.zeros_like(ensemble)).sum(dim=1)
    value = total / count.clamp_min(1).to(dtype=ensemble.dtype)
    return torch.where(count > 0, value, torch.full_like(value, float("nan")))


def ensemble_spread(ensemble: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    mean = ensemble_mean(ensemble).unsqueeze(1)
    finite = torch.isfinite(ensemble) & torch.isfinite(mean)
    count = finite.sum(dim=1)
    centered = torch.where(finite, ensemble - mean, torch.zeros_like(ensemble))
    var = centered.square().sum(dim=1) / count.clamp_min(1).to(dtype=ensemble.dtype)
    spread = torch.sqrt(torch.where(count > 0, var, torch.full_like(var, float("nan"))))
    return masked_mean(spread, torch.isfinite(spread), reduction=reduction)


def ensemble_crps(ensemble: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    _validate_ensemble(ensemble, target)
    valid = torch.isfinite(target) & torch.isfinite(ensemble).all(dim=1)
    term1 = torch.mean(torch.abs(ensemble - target.unsqueeze(1)), dim=1)
    pairwise = torch.abs(ensemble.unsqueeze(2) - ensemble.unsqueeze(1))
    term2 = 0.5 * torch.mean(pairwise, dim=(1, 2))
    crps = term1 - term2
    return masked_mean(crps, valid, reduction=reduction)


def probability_of_event(ensemble: torch.Tensor, threshold: float | torch.Tensor, *, op: str = ">=") -> torch.Tensor:
    _validate_ensemble(ensemble)
    thr = torch.as_tensor(threshold, device=ensemble.device, dtype=ensemble.dtype)
    if op == ">=":
        event = ensemble >= thr
    elif op == ">":
        event = ensemble > thr
    elif op == "<=":
        event = ensemble <= thr
    elif op == "<":
        event = ensemble < thr
    else:
        raise ValueError(f"unknown event op={op!r}")
    finite = torch.isfinite(ensemble)
    count = finite.sum(dim=1)
    prob = (event & finite).sum(dim=1).to(dtype=ensemble.dtype) / count.clamp_min(1).to(dtype=ensemble.dtype)
    return torch.where(count > 0, prob, torch.full_like(prob, float("nan")))


def brier_score(prob: torch.Tensor, event: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    if prob.shape != event.shape:
        raise ValueError(f"shape mismatch: prob={tuple(prob.shape)} event={tuple(event.shape)}")
    event_f = event.to(device=prob.device, dtype=prob.dtype)
    mask = finite_mask(prob, event_f)
    return masked_mean((prob - event_f).square(), mask, reduction=reduction)


def spread_skill_ratio(ensemble: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    mean = ensemble_mean(ensemble)
    skill = rmse(mean, target, reduction="mean")
    spread = ensemble_spread(ensemble, reduction="mean")
    return spread / skill.clamp_min(eps)


def probabilistic_metrics(ensemble: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    mean = ensemble_mean(ensemble)
    return {
        "crps": ensemble_crps(ensemble, target, reduction="mean"),
        "ensemble_mean_rmse": rmse(mean, target, reduction="mean"),
        "ensemble_spread": ensemble_spread(ensemble, reduction="mean"),
        "spread_skill_ratio": spread_skill_ratio(ensemble, target),
    }


__all__ = [
    "ensemble_mean",
    "ensemble_spread",
    "ensemble_crps",
    "probability_of_event",
    "brier_score",
    "spread_skill_ratio",
    "probabilistic_metrics",
]