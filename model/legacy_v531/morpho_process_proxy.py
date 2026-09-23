"""形态过程代理编码器。

该模块把城市静态形态字段转换为一组无量纲 process proxy fields，
并同时提供多尺度空间特征给后续调制路径使用。所有代理量都只是
morphology-derived proxy，不表示真实物理参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PROCESS_PROXY_NAMES = (
    "roughness_proxy",
    "drag_proxy",
    "heat_storage_proxy",
    "impervious_proxy",
    "evap_proxy",
    "ventilation_block_proxy",
    "anthropogenic_heat_proxy",
)


@dataclass(frozen=True)
class MorphologyProcessProxyOutput:
    """形态过程代理编码器输出。"""

    proxy_fields: Dict[str, torch.Tensor]
    level_features: List[torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]


def _as_channels(values: Iterable[int], fallback: tuple[int, ...]) -> tuple[int, ...]:
    channels = tuple(int(v) for v in values)
    return channels if channels else fallback


def _resize_like(value: torch.Tensor, ref: torch.Tensor, *, mode: str = "nearest") -> torch.Tensor:
    if value.shape[-2:] == ref.shape[-2:]:
        return value
    if mode == "nearest":
        return F.interpolate(value, size=ref.shape[-2:], mode="nearest")
    return F.interpolate(value, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _normalize_maps(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value
    mean = value.mean(dim=(-2, -1), keepdim=True)
    std = value.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (value - mean) / std


def _hour_modulation(hour_of_day: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if hour_of_day is None:
        return ref.new_zeros(ref.size(0), 1, 1, 1)
    hour = hour_of_day.to(device=ref.device, dtype=ref.dtype).reshape(-1)
    if hour.numel() == 1 and ref.size(0) > 1:
        hour = hour.expand(ref.size(0))
    if hour.numel() != ref.size(0):
        raise ValueError(f"hour_of_day 需要 shape=[B]，但得到 {tuple(hour_of_day.shape)}")
    radians = (hour.clamp(0.0, 23.0) - 6.0) / 24.0 * (2.0 * torch.pi)
    return torch.sin(radians).clamp_min(0.0).view(-1, 1, 1, 1)


class ConvNormAct(nn.Module):
    """轻量卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MorphologyProcessProxyEncoder(nn.Module):
    """静态形态到 process proxy fields 的统一编码器。"""

    def __init__(
        self,
        static_channels: int = 5,
        hidden_channels: Iterable[int] = (64, 128, 256),
        categorical_scale: float = 20.0,
        dropout: float = 0.0,
        enabled: bool = True,
        use_diurnal_modulation: bool = True,
        proxy_init: str = "zero",
        init_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.static_channels = int(static_channels)
        self.hidden_channels = _as_channels(hidden_channels, (64, 128, 256))
        self.categorical_scale = float(max(categorical_scale, 1.0))
        self.enabled = bool(enabled)
        self.use_diurnal_modulation = bool(use_diurnal_modulation)
        self.init_gate = float(init_gate)

        channels = self.hidden_channels
        self.stem = ConvNormAct(self.static_channels, channels[0], dropout=dropout)
        self.level1 = ConvNormAct(channels[0], channels[1], stride=2, dropout=dropout)
        self.level2 = ConvNormAct(channels[1], channels[2], stride=2, dropout=dropout)
        self.proxy_head = nn.Conv2d(channels[0], len(PROCESS_PROXY_NAMES), kernel_size=1)
        self.fallback_proj = nn.Conv2d(self.static_channels, channels[0], kernel_size=1)
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

        if str(proxy_init).lower() == "zero":
            nn.init.zeros_(self.proxy_head.weight)
            nn.init.zeros_(self.proxy_head.bias)

    def _coerce_static(
        self,
        ref: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts = []
        diagnostics = self._empty_input_diagnostics(ref)
        if static_cont is not None and static_cont.numel() > 0:
            cont = static_cont.to(device=ref.device, dtype=ref.dtype)
            parts.append(cont)
            diagnostics["proxy/static_cont_channels"] = ref.new_tensor(float(cont.size(1)))
        if static_cat is not None and static_cat.numel() > 0:
            cat = static_cat.to(device=ref.device, dtype=ref.dtype) / self.categorical_scale
            parts.insert(0, cat)
            diagnostics["proxy/static_cat_channels"] = ref.new_tensor(float(cat.size(1)))
        if not parts and static_raw is not None and static_raw.numel() > 0:
            raw = static_raw.to(device=ref.device, dtype=ref.dtype)
            parts.append(raw)
            diagnostics["proxy/static_raw_channels"] = ref.new_tensor(float(raw.size(1)))
            diagnostics["proxy/static_schema_fallback"] = ref.new_tensor(1.0)

        if parts:
            feat = torch.cat(parts, dim=1)
            feat = _resize_like(feat, ref, mode="nearest")
        else:
            feat = ref.new_zeros(ref.size(0), self.static_channels, ref.size(2), ref.size(3))
        feat = self._match_static_channels(feat, ref)
        diagnostics["proxy/static_mapper_input_channels"] = ref.new_tensor(float(feat.size(1)))
        return _normalize_maps(feat), diagnostics

    def _empty_input_diagnostics(self, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = ref.new_tensor(0.0)
        return {
            "proxy/static_schema_fallback": zero,
            "proxy/static_cont_channels": zero,
            "proxy/static_cat_channels": zero,
            "proxy/static_raw_channels": zero,
            "proxy/static_mapper_input_channels": zero,
        }

    def _match_static_channels(self, feat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if feat.size(1) == self.static_channels:
            return feat
        if feat.size(1) > self.static_channels:
            return feat[:, : self.static_channels]
        pad = ref.new_zeros(feat.size(0), self.static_channels - feat.size(1), feat.size(2), feat.size(3))
        return torch.cat([feat, pad], dim=1)

    def _encode_levels(self, feat: torch.Tensor) -> List[torch.Tensor]:
        level0 = self.stem(feat) if self.enabled else self.fallback_proj(feat)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        return [level0, level1, level2]

    def _proxy_fields(
        self,
        level0: torch.Tensor,
        hour_of_day: Optional[torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        raw = self.proxy_head(level0)
        if self.use_diurnal_modulation:
            day = _hour_modulation(hour_of_day, level0)
            heat_idx = PROCESS_PROXY_NAMES.index("heat_storage_proxy")
            evap_idx = PROCESS_PROXY_NAMES.index("evap_proxy")
            raw[:, heat_idx : heat_idx + 1] = raw[:, heat_idx : heat_idx + 1] + 0.1 * day
            raw[:, evap_idx : evap_idx + 1] = raw[:, evap_idx : evap_idx + 1] + 0.05 * day
        proxies = {name: torch.sigmoid(raw[:, idx : idx + 1]) for idx, name in enumerate(PROCESS_PROXY_NAMES)}
        return proxies, raw

    def _source_flags(
        self,
        input_diag: Mapping[str, torch.Tensor],
        ref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        static_raw_channels = input_diag.get("proxy/static_raw_channels", ref.new_tensor(0.0))
        static_cont_channels = input_diag.get("proxy/static_cont_channels", ref.new_tensor(0.0))
        static_cat_channels = input_diag.get("proxy/static_cat_channels", ref.new_tensor(0.0))
        has_raw = (static_raw_channels > 0).to(dtype=ref.dtype)
        has_cont = (static_cont_channels > 0).to(dtype=ref.dtype)
        has_cat = (static_cat_channels > 0).to(dtype=ref.dtype)
        return {
            "proxy/source_has_static_raw": has_raw,
            "proxy/source_has_static_cont": has_cont,
            "proxy/source_has_static_cat": has_cat,
            "proxy/source_has_schema": ((has_cont + has_cat) > 0).to(dtype=ref.dtype),
            "proxy/source_has_landcover": has_cat,
            "proxy/source_has_building_surface": (static_cont_channels >= 1).to(dtype=ref.dtype),
            "proxy/source_has_buildings": (static_cont_channels >= 2).to(dtype=ref.dtype),
            "proxy/source_has_building_volume": (static_cont_channels >= 3).to(dtype=ref.dtype),
            "proxy/source_has_population": (static_cont_channels >= 4).to(dtype=ref.dtype),
            "proxy/source_has_vegetation": ref.new_tensor(0.0),
            "proxy/source_has_svf": ref.new_tensor(0.0),
        }

    def _diagnostics(
        self,
        proxy_fields: Mapping[str, torch.Tensor],
        input_diag: Mapping[str, torch.Tensor],
        source_flags: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        diagnostics = dict(input_diag)
        diagnostics.update(source_flags)
        ref = next(iter(proxy_fields.values()))
        diagnostics["proxy/init_gate"] = ref.new_tensor(self.init_gate)
        diagnostics["proxy/estimated_proxy_fields"] = ref.new_tensor(1.0)
        for name, value in proxy_fields.items():
            diagnostics[f"proxy/{name}_mean"] = value.mean()
            diagnostics[f"proxy/{name}_spatial_std"] = value.std(unbiased=False)
        rough = proxy_fields["roughness_proxy"].flatten(1)
        imperv = proxy_fields["impervious_proxy"].flatten(1)
        evap = proxy_fields["evap_proxy"].flatten(1)
        diagnostics["proxy/roughness_impervious_corr"] = self._safe_corr(rough, imperv)
        diagnostics["proxy/evap_impervious_anticorr"] = self._safe_corr(evap, imperv)
        return diagnostics

    @staticmethod
    def _safe_corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_centered = left - left.mean(dim=1, keepdim=True)
        right_centered = right - right.mean(dim=1, keepdim=True)
        denom = left_centered.norm(dim=1) * right_centered.norm(dim=1)
        corr = (left_centered * right_centered).sum(dim=1) / denom.clamp_min(1e-6)
        return corr.mean()

    def forward(
        self,
        ref: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> MorphologyProcessProxyOutput:
        if ref.ndim != 4:
            raise ValueError(f"ref 必须是 [B,C,H,W]，但得到 {tuple(ref.shape)}")
        static_feat, input_diag = self._coerce_static(
            ref,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
        )
        level_features = self._encode_levels(static_feat)
        proxy_fields, _ = self._proxy_fields(level_features[0], hour_of_day)
        diagnostics = self._diagnostics(proxy_fields, input_diag, self._source_flags(input_diag, ref))
        self.last_diagnostics = diagnostics
        return MorphologyProcessProxyOutput(proxy_fields, level_features, diagnostics)


__all__ = [
    "MorphologyProcessProxyEncoder",
    "MorphologyProcessProxyOutput",
    "PROCESS_PROXY_NAMES",
]