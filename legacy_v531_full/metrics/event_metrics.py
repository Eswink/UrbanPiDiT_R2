from __future__ import annotations

from typing import Dict

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    value = num / den.clamp_min(1).to(dtype=num.dtype)
    return torch.where(den > 0, value, torch.full_like(value, float("nan")))


def event_mask(x: torch.Tensor, threshold: float | torch.Tensor, *, op: str = ">=") -> torch.Tensor:
    thr = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
    if op == ">=":
        return x >= thr
    if op == ">":
        return x > thr
    if op == "<=":
        return x <= thr
    if op == "<":
        return x < thr
    raise ValueError(f"unknown event op={op!r}")


def contingency_table(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    op: str = ">=",
) -> Dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    valid = torch.isfinite(pred) & torch.isfinite(target)
    pred_event = event_mask(pred, threshold, op=op) & valid
    target_event = event_mask(target, threshold, op=op) & valid
    hits = (pred_event & target_event).sum()
    false_alarms = (pred_event & ~target_event & valid).sum()
    misses = (~pred_event & target_event & valid).sum()
    correct_negatives = (~pred_event & ~target_event & valid).sum()
    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_negatives": correct_negatives,
        "valid": valid.sum(),
    }


def event_scores(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    op: str = ">=",
) -> Dict[str, torch.Tensor]:
    table = contingency_table(pred, target, threshold=threshold, op=op)
    h = table["hits"].to(dtype=pred.dtype)
    fa = table["false_alarms"].to(dtype=pred.dtype)
    m = table["misses"].to(dtype=pred.dtype)
    cn = table["correct_negatives"].to(dtype=pred.dtype)
    total = table["valid"].to(dtype=pred.dtype)

    out = dict(table)
    out.update(
        {
            "pod": _safe_div(h, h + m),
            "far": _safe_div(fa, h + fa),
            "csi": _safe_div(h, h + fa + m),
            "frequency_bias": _safe_div(h + fa, h + m),
            "accuracy": _safe_div(h + cn, total),
            "precision": _safe_div(h, h + fa),
            "recall": _safe_div(h, h + m),
            "f1": _safe_div(2.0 * h, 2.0 * h + fa + m),
        }
    )
    return out


__all__ = ["event_mask", "contingency_table", "event_scores"]