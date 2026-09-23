from __future__ import annotations

from typing import Dict

import torch


def _paired_finite(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={tuple(a.shape)} b={tuple(b.shape)}")
    a = a.flatten().float()
    b = b.flatten().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.numel() == 0:
        raise ValueError("no finite paired samples")
    return a, b


def paired_bootstrap_delta(
    losses_a: torch.Tensor,
    losses_b: torch.Tensor,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    a, b = _paired_finite(losses_a, losses_b)
    diff = a - b
    n = int(diff.numel())
    generator = torch.Generator(device=diff.device).manual_seed(int(seed))
    idx = torch.randint(0, n, (int(n_boot), n), generator=generator, device=diff.device)
    boot = diff[idx].mean(dim=1)
    alpha = (1.0 - float(ci)) / 2.0
    lo = torch.quantile(boot, alpha)
    hi = torch.quantile(boot, 1.0 - alpha)
    std = diff.std(unbiased=True) if n > 1 else torch.zeros((), device=diff.device, dtype=diff.dtype)
    effect = diff.mean() / std.clamp_min(1e-8)
    return {
        "n": float(n),
        "delta_mean": float(diff.mean().detach().cpu()),
        "ci_low": float(lo.detach().cpu()),
        "ci_high": float(hi.detach().cpu()),
        "cohen_dz": float(effect.detach().cpu()),
    }


def paired_permutation_test(
    losses_a: torch.Tensor,
    losses_b: torch.Tensor,
    *,
    n_permutations: int = 999,
    seed: int = 0,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    a, b = _paired_finite(losses_a, losses_b)
    diff = a - b
    observed = diff.mean()
    n = int(diff.numel())
    generator = torch.Generator(device=diff.device).manual_seed(int(seed))
    signs = torch.randint(0, 2, (int(n_permutations), n), generator=generator, device=diff.device, dtype=torch.int64)
    signs = signs.to(dtype=diff.dtype).mul_(2.0).sub_(1.0)
    samples = (diff.unsqueeze(0) * signs).mean(dim=1)

    if alternative == "two-sided":
        extreme = samples.abs() >= observed.abs()
    elif alternative == "greater":
        extreme = samples >= observed
    elif alternative == "less":
        extreme = samples <= observed
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    p_value = (extreme.sum().to(dtype=diff.dtype) + 1.0) / (float(n_permutations) + 1.0)
    return {
        "n": float(n),
        "delta_mean": float(observed.detach().cpu()),
        "p_value": float(p_value.detach().cpu()),
    }


__all__ = ["paired_bootstrap_delta", "paired_permutation_test"]