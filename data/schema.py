from __future__ import annotations
from typing import Mapping, Any
import torch

REQUIRED_KEYS = (
    "coarse_history", "urban_history", "urban_static", "urban_baseline", "urban_target"
)

R7_FORECAST_REQUIRED_KEYS = (
    "coarse_history", "atmos_target"
)


def validate_sample(sample: Mapping[str, Any], *, batched: bool = False) -> None:
    """验证 V6 多尺度数据契约，尽早阻止尺度/通道静默错配。"""
    missing = [k for k in REQUIRED_KEYS if k not in sample]
    if missing:
        raise KeyError(f"缺少必要字段: {missing}")
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


def validate_forecast_sample(
    sample:Mapping[str,Any],
    *,
    batched:bool=False,
)->None:
    """Validate the R7 forecast-native atmospheric data contract."""
    missing=[k for k in R7_FORECAST_REQUIRED_KEYS if k not in sample]
    if missing:
        raise KeyError(f"缺少 R7 forecast-native 字段: {missing}")

    history=sample['coarse_history']
    target=sample['atmos_target']
    if not isinstance(history,torch.Tensor) or not isinstance(target,torch.Tensor):
        raise TypeError('coarse_history 与 atmos_target 必须是 torch.Tensor')

    expected_history_ndim=5 if batched else 4
    expected_target_ndim=4 if batched else 3
    if history.ndim!=expected_history_ndim:
        raise ValueError(
            'coarse_history 形状应为 [B,T,C,H,W] 或单样本 [T,C,H,W]'
        )
    if target.ndim!=expected_target_ndim:
        raise ValueError(
            'atmos_target 形状应为 [B,C,H,W] 或单样本 [C,H,W]'
        )
    if history.shape[-2:]!=target.shape[-2:]:
        raise ValueError('coarse_history 与 atmos_target 空间尺寸必须一致')
    if target.shape[-3]>history.shape[-3]:
        raise ValueError('R7.1 要求 atmos_target channels <= input channels')

    if 'lead_time_hours' in sample:
        lead=sample['lead_time_hours']
        if not isinstance(lead,torch.Tensor):
            raise TypeError('lead_time_hours 必须是 torch.Tensor')

    if 'latitude' in sample:
        lat=sample['latitude']
        if not isinstance(lat,torch.Tensor):
            raise TypeError('latitude 必须是 torch.Tensor')
        if lat.shape[-1]!=history.shape[-2]:
            raise ValueError('latitude 长度必须匹配 H')

    if 'longitude' in sample:
        lon=sample['longitude']
        if not isinstance(lon,torch.Tensor):
            raise TypeError('longitude 必须是 torch.Tensor')
        if lon.shape[-1]!=history.shape[-1]:
            raise ValueError('longitude 长度必须匹配 W')
