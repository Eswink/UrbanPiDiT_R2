from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def model_size_mb(model: nn.Module, *, trainable_only: bool = False, include_buffers: bool = True) -> float:
    bytes_total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        bytes_total += int(p.numel() * p.element_size())
    if include_buffers:
        for b in model.buffers():
            bytes_total += int(b.numel() * b.element_size())
    return float(bytes_total) / (1024.0**2)


def tensor_memory_mb(*tensors: torch.Tensor) -> float:
    return float(sum(int(t.numel() * t.element_size()) for t in tensors)) / (1024.0**2)


def _call_model(model: nn.Module, sample: Any):
    if isinstance(sample, dict):
        return model(**sample)
    if isinstance(sample, (tuple, list)):
        return model(*sample)
    return model(sample)


def _sync(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_latency_ms(
    model: nn.Module,
    sample: Any,
    *,
    warmup: int = 5,
    iterations: int = 20,
    device: Optional[torch.device | str] = None,
) -> float:
    dev = torch.device(device) if device is not None else None
    model.eval()
    with torch.no_grad():
        for _ in range(int(warmup)):
            _ = _call_model(model, sample)
        _sync(dev)
        start = time.perf_counter()
        for _ in range(int(iterations)):
            _ = _call_model(model, sample)
        _sync(dev)
    elapsed = time.perf_counter() - start
    return float(elapsed) * 1000.0 / float(max(int(iterations), 1))


def efficiency_summary(model: nn.Module) -> Dict[str, float]:
    return {
        "params": float(count_parameters(model, trainable_only=False)),
        "trainable_params": float(count_parameters(model, trainable_only=True)),
        "model_size_mb": model_size_mb(model),
    }


__all__ = [
    "count_parameters",
    "model_size_mb",
    "tensor_memory_mb",
    "measure_inference_latency_ms",
    "efficiency_summary",
]