"""模型诊断日志辅助工具。

该模块把 forward diagnostics 中可稳定聚合的数值项筛出，供训练、验证和测试日志使用。
文本、形状列表和高维摘要不会进入 Lightning 标量日志。
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict, Mapping, Optional

import torch


DEFAULT_LOG_PREFIXES = (
    "proxy/",
    "process_adaln/",
    "urban_control/",
    "process_graph/",
    "micromet/",
    "residual_prediction/",
    "leakage/",
    "model/use_",
)


def _is_scalar_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.detach().numel() == 1


def _to_scalar_tensor(value: Any, *, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if _is_scalar_tensor(value):
        tensor = value.detach().float().reshape(())
        return tensor.to(device=device) if device is not None else tensor
    if isinstance(value, bool):
        target_device = device if device is not None else torch.device("cpu")
        return torch.tensor(float(value), dtype=torch.float32, device=target_device)
    if isinstance(value, Number):
        target_device = device if device is not None else torch.device("cpu")
        return torch.tensor(float(value), dtype=torch.float32, device=target_device)
    return None


def _should_log_key(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def collect_model_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    log_prefix: str = "train/diagnostics",
    include_prefixes: tuple[str, ...] = DEFAULT_LOG_PREFIXES,
    device: Optional[torch.device] = None,
    max_items: int = 128,
) -> Dict[str, torch.Tensor]:
    """提取可进入训练日志的标量诊断项。

    Args:
        diagnostics: 模型 forward 返回的 diagnostics。
        log_prefix: 输出日志键前缀。
        include_prefixes: 允许进入日志的 diagnostics 前缀。
        device: 返回张量所在设备。
        max_items: 最多输出项数，避免日志过宽。

    Returns:
        形如 `{log_prefix}/{diagnostic_key}: scalar_tensor}` 的字典。
    """

    logs: Dict[str, torch.Tensor] = {}
    clean_prefix = log_prefix.rstrip("/")
    for key in sorted(str(k) for k in diagnostics.keys()):
        if len(logs) >= int(max_items):
            break
        if not _should_log_key(key, include_prefixes):
            continue
        value = _to_scalar_tensor(diagnostics[key], device=device)
        if value is None or not torch.isfinite(value):
            continue
        logs[f"{clean_prefix}/{key}"] = value
    return logs


__all__ = ["DEFAULT_LOG_PREFIXES", "collect_model_diagnostics"]