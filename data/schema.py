from __future__ import annotations
from typing import Mapping, Any
import torch

REQUIRED_KEYS = (
    "coarse_history", "urban_history", "urban_static", "urban_baseline", "urban_target"
)


def validate_sample(sample: Mapping[str, Any], *, batched: bool = False) -> None:
    """验证 V6 多尺度数据契约，尽早阻止尺度/通道静默错配。"""
    missing = [k for k in REQUIRED_KEYS if k not in sample]
    if missing:
        raise KeyError(f"缺少必要字段: {missing}")
    offset = 1 if batched else 0
    for name in REQUIRED_KEYS:
        x = sample[name]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor，实际为 {type(x)!r}")
    if sample["coarse_history"].ndim != 5 - (0 if batched else 1):
        raise ValueError("coarse_history 形状应为 [B,T,C,H,W] 或单样本 [T,C,H,W]")
    if sample["urban_history"].ndim != 5 - (0 if batched else 1):
        raise ValueError("urban_history 形状应为 [B,T,C,H,W] 或单样本 [T,C,H,W]")
    if sample["urban_static"].ndim != 4 - (0 if batched else 1):
        raise ValueError("urban_static 形状应为 [B,C,H,W] 或单样本 [C,H,W]")
    if sample["urban_baseline"].shape[-2:] != sample["urban_target"].shape[-2:]:
        raise ValueError("urban_baseline 与 urban_target 的空间尺寸必须一致")
    if sample["urban_history"].shape[-2:] != sample["urban_target"].shape[-2:]:
        raise ValueError("urban_history 与 urban_target 的空间尺寸必须一致")
    if sample["urban_static"].shape[-2:] != sample["urban_target"].shape[-2:]:
        raise ValueError("urban_static 与 urban_target 的空间尺寸必须一致")
