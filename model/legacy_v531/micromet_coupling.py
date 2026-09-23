"""城市微气象过程耦合算子。

该模块把城市冠层相关过程拆成可命名、可消融、可诊断的弱残差分支：
- morphology parameter mapping：静态形态字段到粗糙度、阻力、热储、水汽等代理量；
- momentum drag：形态阻力对近地风场的弱残差阻尼；
- thermal storage：热储/不透水面/人为热代理对温度场的日周期响应；
- moisture evaporation：蒸散/水面/不透水面代理对露点与温度的弱耦合；
- ventilation mixing：历史风场约束下的空间通风混合。

所有分支只读取当前动态状态、历史上下文派生风场、静态字段与可选小时信息。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .morphology_graph import WindAwareMorphologyGraph
except ImportError:  # pragma: no cover
    from models.morphology_graph import WindAwareMorphologyGraph


DEFAULT_DYNAMIC_VARS = ("d2m", "sp", "t2m", "tcc", "tp", "u10", "v10")
MICROMET_PARAM_NAMES = (
    "roughness_proxy",
    "drag_proxy",
    "heat_storage_proxy",
    "impervious_proxy",
    "evap_proxy",
    "ventilation_block_proxy",
    "anthropogenic_heat_proxy",
)


def _ordered_vars(values: Optional[Iterable[str]], fallback: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(str(v) for v in (values or ()))
    return items if items else fallback


def _var_index(var_names: tuple[str, ...], name: str, fallback: Optional[int], channels: int) -> Optional[int]:
    if name in var_names:
        idx = int(var_names.index(name))
        return idx if 0 <= idx < channels else None
    if fallback is None:
        return None
    return int(fallback) if 0 <= int(fallback) < channels else None


def _channel_gate(channels: int, indices: Iterable[Optional[int]]) -> torch.Tensor:
    gate = torch.zeros(1, int(channels), 1, 1, dtype=torch.float32)
    for idx in indices:
        if idx is not None and 0 <= int(idx) < channels:
            gate[:, int(idx), :, :] = 1.0
    return gate


def _normalize_static(feat: torch.Tensor) -> torch.Tensor:
    mean = feat.mean(dim=(-2, -1), keepdim=True)
    std = feat.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (feat - mean) / std


def _resize_like(value: torch.Tensor, ref: torch.Tensor, *, mode: str = "nearest") -> torch.Tensor:
    if value.shape[-2:] == ref.shape[-2:]:
        return value
    if mode == "nearest":
        return F.interpolate(value, size=ref.shape[-2:], mode="nearest")
    return F.interpolate(value, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _hour_factor(hour_of_day: Optional[torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hour_of_day is None:
        return x.new_zeros((x.size(0), 1, 1, 1)), x.new_tensor(0.0)
    hour = hour_of_day.to(device=x.device, dtype=x.dtype).reshape(-1)
    if hour.numel() == 1 and x.size(0) > 1:
        hour = hour.expand(x.size(0))
    if hour.numel() != x.size(0):
        raise ValueError(f"hour_of_day 需要 shape=[B]，但得到 {tuple(hour_of_day.shape)}")
    available = ((hour >= 0.0) & (hour <= 23.0)).to(dtype=x.dtype)
    radians = (hour.clamp(0.0, 23.0) - 6.0) / 24.0 * (2.0 * math.pi)
    daylight = torch.sin(radians).clamp_min(0.0)
    factor = torch.where(available > 0.0, daylight, torch.zeros_like(daylight))
    return factor.view(-1, 1, 1, 1), available.mean()


class MorphologyParameterMapper(nn.Module):
    """把静态形态字段映射为微气象代理参数。"""

    def __init__(
        self,
        static_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.0,
        categorical_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.static_channels = max(int(static_channels), 0)
        self.hidden_channels = max(int(hidden_channels), 4)
        self.categorical_scale = float(max(categorical_scale, 1.0))
        if self.static_channels > 0:
            self.net = nn.Sequential(
                nn.Conv2d(self.static_channels, self.hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(float(dropout)),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_channels, len(MICROMET_PARAM_NAMES), kernel_size=1),
            )
        else:
            self.net = None
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def _process_categorical(self, value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        cat = value.to(device=ref.device, dtype=ref.dtype)
        return cat / self.categorical_scale

    def _coerce_static(
        self,
        ref: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, _, height, width = ref.shape
        zero = ref.new_tensor(0.0)
        diagnostics: Dict[str, torch.Tensor] = {
            "static_schema_fallback": zero,
            "static_cont_channels": zero,
            "static_cat_channels": zero,
            "static_raw_channels": zero,
            "static_mapper_input_channels": zero,
        }
        if self.static_channels <= 0:
            self.last_diagnostics = diagnostics
            return ref.new_zeros(batch, 0, height, width)

        parts = []
        if static_cont is not None and static_cont.numel() > 0:
            cont = static_cont.to(device=ref.device, dtype=ref.dtype)
            parts.append(cont)
            diagnostics["static_cont_channels"] = ref.new_tensor(float(cont.size(1)))
        if static_cat is not None and static_cat.numel() > 0:
            cat = self._process_categorical(static_cat, ref)
            parts.append(cat)
            diagnostics["static_cat_channels"] = ref.new_tensor(float(cat.size(1)))
        if not parts and static_raw is not None and static_raw.numel() > 0:
            raw = static_raw.to(device=ref.device, dtype=ref.dtype)
            parts.append(raw)
            diagnostics["static_raw_channels"] = ref.new_tensor(float(raw.size(1)))
            diagnostics["static_schema_fallback"] = ref.new_tensor(1.0)

        if not parts:
            feat = ref.new_zeros(batch, self.static_channels, height, width)
        else:
            feat = torch.cat(parts, dim=1)
            if feat.size(0) != batch:
                if feat.size(0) == 1:
                    feat = feat.expand(batch, -1, -1, -1)
                else:
                    raise ValueError(f"static batch={feat.size(0)} 与动态 batch={batch} 不一致")
            feat = _resize_like(feat, ref, mode="nearest")
            if feat.size(1) < self.static_channels:
                pad = ref.new_zeros(batch, self.static_channels - feat.size(1), height, width)
                feat = torch.cat([feat, pad], dim=1)
            elif feat.size(1) > self.static_channels:
                feat = feat[:, : self.static_channels]
        diagnostics["static_mapper_input_channels"] = ref.new_tensor(float(feat.size(1)))
        self.last_diagnostics = diagnostics
        return _normalize_static(feat)

    def forward(
        self,
        ref: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.net is None:
            raw = ref.new_zeros(ref.size(0), len(MICROMET_PARAM_NAMES), ref.size(2), ref.size(3))
        else:
            raw = self.net(self._coerce_static(ref, static_raw=static_raw, static_cont=static_cont, static_cat=static_cat))
        out: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(MICROMET_PARAM_NAMES):
            value = raw[:, i : i + 1]
            if name == "drag_proxy":
                out[name] = F.softplus(value).clamp(max=5.0) / 5.0
            else:
                out[name] = torch.sigmoid(value)
        return out


class MomentumDragBranch(nn.Module):
    """近地动量阻力分支。"""

    def __init__(
        self,
        channels: int,
        dynamic_vars: Optional[Iterable[str]] = None,
        max_drag: float = 0.2,
        init_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dynamic_vars = _ordered_vars(dynamic_vars, DEFAULT_DYNAMIC_VARS)
        self.max_drag = float(max(max_drag, 0.0))
        self.idx_u10 = _var_index(self.dynamic_vars, "u10", 5, self.channels)
        self.idx_v10 = _var_index(self.dynamic_vars, "v10", 6, self.channels)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def residual(self, x: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        delta = torch.zeros_like(x)
        if self.idx_u10 is None or self.idx_v10 is None:
            return delta
        drag = (params["roughness_proxy"] + params["drag_proxy"] + params["ventilation_block_proxy"]) / 3.0
        drag = drag.clamp(0.0, 1.0) * self.max_drag
        delta[:, self.idx_u10 : self.idx_u10 + 1] = -drag * x[:, self.idx_u10 : self.idx_u10 + 1]
        delta[:, self.idx_v10 : self.idx_v10 + 1] = -drag * x[:, self.idx_v10 : self.idx_v10 + 1]
        return delta

    def forward(self, x: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.gate) * self.residual(x, params)


class ThermalStorageBranch(nn.Module):
    """热储与日周期响应分支。"""

    def __init__(
        self,
        channels: int,
        dynamic_vars: Optional[Iterable[str]] = None,
        max_scale: float = 0.1,
        init_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dynamic_vars = _ordered_vars(dynamic_vars, DEFAULT_DYNAMIC_VARS)
        self.max_scale = float(max(max_scale, 0.0))
        self.idx_t2m = _var_index(self.dynamic_vars, "t2m", 2, self.channels)
        self.idx_d2m = _var_index(self.dynamic_vars, "d2m", 0, self.channels)
        self.register_buffer("mask", _channel_gate(self.channels, [self.idx_t2m, self.idx_d2m]), persistent=False)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def residual(
        self,
        x: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.zeros_like(x)
        day_factor, available = _hour_factor(hour_of_day, x)
        response = (
            params["heat_storage_proxy"] + params["impervious_proxy"] + params["anthropogenic_heat_proxy"]
        ) / 3.0
        response = response * day_factor * self.max_scale
        if self.idx_t2m is not None:
            delta[:, self.idx_t2m : self.idx_t2m + 1] = response
        if self.idx_d2m is not None:
            delta[:, self.idx_d2m : self.idx_d2m + 1] = 0.25 * response
        return delta * self.mask.to(device=x.device, dtype=x.dtype), available

    def forward(
        self,
        x: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        hour_of_day: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, available = self.residual(x, params, hour_of_day=hour_of_day)
        return torch.tanh(self.gate) * delta, available


class MoistureEvaporationBranch(nn.Module):
    """蒸散与水汽弱耦合分支。"""

    def __init__(
        self,
        channels: int,
        dynamic_vars: Optional[Iterable[str]] = None,
        max_scale: float = 0.1,
        init_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dynamic_vars = _ordered_vars(dynamic_vars, DEFAULT_DYNAMIC_VARS)
        self.max_scale = float(max(max_scale, 0.0))
        self.idx_d2m = _var_index(self.dynamic_vars, "d2m", 0, self.channels)
        self.idx_t2m = _var_index(self.dynamic_vars, "t2m", 2, self.channels)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def residual(self, x: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        delta = torch.zeros_like(x)
        evap = params["evap_proxy"] * (1.0 - params["impervious_proxy"]).clamp(0.0, 1.0) * self.max_scale
        if self.idx_d2m is not None:
            delta[:, self.idx_d2m : self.idx_d2m + 1] = evap
        if self.idx_t2m is not None:
            delta[:, self.idx_t2m : self.idx_t2m + 1] = -0.25 * evap
        return delta

    def forward(self, x: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.gate) * self.residual(x, params)


class VentilationMixingBranch(nn.Module):
    """风向感知通风混合分支。"""

    def __init__(
        self,
        channels: int,
        dynamic_vars: Optional[Iterable[str]] = None,
        graph_cfg: Optional[Mapping[str, Any]] = None,
        max_scale: float = 0.1,
        init_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dynamic_vars = _ordered_vars(dynamic_vars, DEFAULT_DYNAMIC_VARS)
        self.max_scale = float(max(max_scale, 0.0))
        self.idx_u10 = _var_index(self.dynamic_vars, "u10", 5, self.channels)
        self.idx_v10 = _var_index(self.dynamic_vars, "v10", 6, self.channels)
        cfg = dict(graph_cfg or {})
        self.graph = WindAwareMorphologyGraph(
            dim=self.channels,
            k=int(cfg.get("k", 8)),
            morphology_sigma=float(cfg.get("morphology_sigma", cfg.get("sigma", 1.0))),
            spatial_sigma=float(cfg.get("spatial_sigma", cfg.get("sigma", 1.0))),
            wind_temperature=float(cfg.get("wind_temperature", 0.5)),
            wind_strength=float(cfg.get("wind_strength", 0.5)),
            roughness_blocking_strength=float(cfg.get("roughness_blocking_strength", 0.5)),
            enable_cache=bool(cfg.get("enable_cache", False)),
        )
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def _wind_uv(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.idx_u10 is None or self.idx_v10 is None:
            return None
        return x[:, [self.idx_u10, self.idx_v10]]

    def residual(
        self,
        x: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        wind = wind_uv if wind_uv is not None else self._wind_uv(x)
        rough = params.get("roughness_proxy", None)
        adj, diag = self.graph.build_adjacency(
            static_feat=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            spatial_hw=(height, width),
            wind_uv=wind,
            roughness_proxy=rough,
        )
        if adj.size(0) != batch:
            adj = adj.expand(batch, -1, -1)
        mixed = torch.einsum("bnm,bmd->bnd", adj.to(device=x.device, dtype=x.dtype), tokens)
        delta = (mixed - tokens).transpose(1, 2).reshape(batch, channels, height, width)
        block = (1.0 - params["ventilation_block_proxy"].clamp(0.0, 1.0)) * self.max_scale
        return delta * block, diag

    def forward(
        self,
        x: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        delta, diag = self.residual(
            x,
            params,
            static_raw=static_raw,
            static_cont=static_cont,
            static_cat=static_cat,
            wind_uv=wind_uv,
        )
        return torch.tanh(self.gate) * delta, diag


class MicroMetTokenProjector(nn.Module):
    """把物理状态残差投影到 token 空间的 zero-init 适配器。"""

    def __init__(self, in_channels: int, token_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(in_channels))
        self.proj = nn.Linear(int(in_channels), int(token_dim))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, delta_state: torch.Tensor) -> torch.Tensor:
        if delta_state.ndim != 4:
            raise ValueError(f"delta_state 必须是 [B,C,H,W]，但得到 {tuple(delta_state.shape)}")
        tokens = delta_state.flatten(2).transpose(1, 2)
        return self.proj(self.norm(tokens))


class MicroMetCouplingOperator(nn.Module):
    """微气象过程耦合总控算子。"""

    def __init__(
        self,
        in_channels: int,
        static_channels: int = 5,
        dynamic_vars: Optional[Iterable[str]] = None,
        hidden_channels: int = 32,
        dropout: float = 0.0,
        enabled: bool = True,
        mode: str = "pre",
        enable_momentum_drag: bool = True,
        enable_thermal_storage: bool = True,
        enable_moisture_evaporation: bool = True,
        enable_ventilation_mixing: bool = True,
        max_drag: float = 0.2,
        max_branch_scale: float = 0.1,
        graph_cfg: Optional[Mapping[str, Any]] = None,
        init_gate: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.static_channels = max(int(static_channels), 0)
        self.dynamic_vars = _ordered_vars(dynamic_vars, DEFAULT_DYNAMIC_VARS)
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.enable_momentum_drag = bool(enable_momentum_drag)
        self.enable_thermal_storage = bool(enable_thermal_storage)
        self.enable_moisture_evaporation = bool(enable_moisture_evaporation)
        self.enable_ventilation_mixing = bool(enable_ventilation_mixing)
        self.init_gate = float(init_gate)

        self.mapper = MorphologyParameterMapper(
            static_channels=self.static_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.momentum_drag = MomentumDragBranch(
            self.in_channels,
            self.dynamic_vars,
            max_drag=max_drag,
            init_gate=self.init_gate,
        )
        self.thermal_storage = ThermalStorageBranch(
            self.in_channels,
            self.dynamic_vars,
            max_scale=max_branch_scale,
            init_gate=self.init_gate,
        )
        self.moisture_evaporation = MoistureEvaporationBranch(
            self.in_channels,
            self.dynamic_vars,
            max_scale=max_branch_scale,
            init_gate=self.init_gate,
        )
        self.ventilation_mixing = VentilationMixingBranch(
            self.in_channels,
            self.dynamic_vars,
            graph_cfg=graph_cfg,
            max_scale=max_branch_scale,
            init_gate=self.init_gate,
        )
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def _disabled_diag(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = x.new_tensor(0.0)
        return {
            "micromet_enabled": zero,
            "diurnal_available": zero,
            "momentum_drag_enabled": zero,
            "thermal_storage_enabled": zero,
            "moisture_evaporation_enabled": zero,
            "ventilation_mixing_enabled": zero,
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"x 必须是 [B,C,H,W]，但得到 {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"x 通道数={x.size(1)} 与 in_channels={self.in_channels} 不一致")
        if not self.enabled:
            diagnostics = self._disabled_diag(x)
            self.last_diagnostics = diagnostics
            return x, diagnostics

        params = self.mapper(x, static_raw=static_raw, static_cont=static_cont, static_cat=static_cat)
        out = x
        diagnostics: Dict[str, torch.Tensor] = {
            "micromet_enabled": x.new_tensor(1.0),
            "momentum_drag_enabled": x.new_tensor(float(self.enable_momentum_drag)),
            "thermal_storage_enabled": x.new_tensor(float(self.enable_thermal_storage)),
            "moisture_evaporation_enabled": x.new_tensor(float(self.enable_moisture_evaporation)),
            "ventilation_mixing_enabled": x.new_tensor(float(self.enable_ventilation_mixing)),
            "gate/raw_init": x.new_tensor(self.init_gate),
        }
        for name, value in params.items():
            diagnostics[f"param/{name}_mean"] = value.mean()
            diagnostics[f"{name}_mean"] = value.mean()
        diagnostics.update(self.mapper.last_diagnostics)

        if self.enable_momentum_drag:
            out = out + self.momentum_drag(out, params)
            diagnostics["gate/momentum_drag"] = torch.tanh(self.momentum_drag.gate).abs()
        if self.enable_thermal_storage:
            delta, diurnal_available = self.thermal_storage(out, params, hour_of_day=hour_of_day)
            out = out + delta
            diagnostics["diurnal_available"] = diurnal_available
            diagnostics["gate/thermal_storage"] = torch.tanh(self.thermal_storage.gate).abs()
        else:
            diagnostics["diurnal_available"] = x.new_tensor(0.0)
        if self.enable_moisture_evaporation:
            out = out + self.moisture_evaporation(out, params)
            diagnostics["gate/moisture_evaporation"] = torch.tanh(self.moisture_evaporation.gate).abs()
        if self.enable_ventilation_mixing:
            delta, graph_diag = self.ventilation_mixing(
                out,
                params,
                static_raw=static_raw,
                static_cont=static_cont,
                static_cat=static_cat,
                wind_uv=wind_uv,
            )
            out = out + delta
            diagnostics["gate/ventilation_mixing"] = torch.tanh(self.ventilation_mixing.gate).abs()
            diagnostics.update({f"graph/{key}": value for key, value in graph_diag.items()})

        self.last_diagnostics = diagnostics
        return out, diagnostics


__all__ = [
    "MICROMET_PARAM_NAMES",
    "MicroMetCouplingOperator",
    "MicroMetTokenProjector",
    "MomentumDragBranch",
    "MoistureEvaporationBranch",
    "MorphologyParameterMapper",
    "ThermalStorageBranch",
    "VentilationMixingBranch",
]