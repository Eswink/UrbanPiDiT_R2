from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


TensorDict = Dict[str, object]


@dataclass(frozen=True)
class ForecastBatchView:
    x_ctx: torch.Tensor
    dynamic: torch.Tensor
    static: torch.Tensor
    lead_steps: torch.Tensor
    mean: Optional[torch.Tensor]
    std: Optional[torch.Tensor]
    clim: Optional[torch.Tensor]
    hour_of_day: Optional[torch.Tensor] = None

    @property
    def last(self) -> torch.Tensor:
        return self.dynamic[:, :, -1]

    @property
    def previous(self) -> torch.Tensor:
        if self.dynamic.shape[2] < 2:
            return self.dynamic[:, :, -1]
        return self.dynamic[:, :, -2]


@dataclass(frozen=True)
class ForecastModelSpec:
    name: str
    description: str
    uses_static: bool
    trainable: bool
    family: str


def normalize_static_policy(policy: str) -> str:
    value = str(policy or "dynamic_only").strip().lower()
    aliases = {
        "none": "dynamic_only",
        "off": "dynamic_only",
        "no_static": "dynamic_only",
        "dynamic": "dynamic_only",
        "static": "same_static",
        "full": "same_static",
        "same": "same_static",
        "zero": "static_zero",
        "zeros": "static_zero",
        "shuffle": "static_shuffle",
        "spatial_shuffle": "static_shuffle",
    }
    value = aliases.get(value, value)
    allowed = {"dynamic_only", "same_static", "static_zero", "static_shuffle"}
    if value not in allowed:
        raise ValueError(f"Unknown static_policy={policy!r}; expected one of {sorted(allowed)}")
    return value


def _to_list(value: Optional[Iterable[str]]) -> List[str]:
    return [str(v) for v in list(value or [])]


def _lead_tensor(batch: TensorDict, lead_times: Optional[Sequence[int]], device: torch.device) -> torch.Tensor:
    if lead_times is not None:
        values = torch.as_tensor(list(lead_times), dtype=torch.long, device=device)
    else:
        raw = batch.get("lead_times", None)
        if raw is None:
            values = torch.ones(1, dtype=torch.long, device=device)
        elif isinstance(raw, torch.Tensor):
            values = raw.to(device=device, dtype=torch.long)
        else:
            values = torch.as_tensor(raw, dtype=torch.long, device=device)

    if values.ndim == 0:
        values = values.view(1)
    elif values.ndim >= 2:
        values = values.reshape(values.shape[0], -1)[0]
    else:
        values = values.flatten()

    if values.numel() == 0:
        values = torch.ones(1, dtype=torch.long, device=device)
    return values


def _norm_tensor(batch: TensorDict, key: str, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    norm = batch.get("norm", None)
    if not isinstance(norm, dict) or key not in norm:
        return None
    x = norm[key]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(device=device, dtype=dtype)


def _with_batch_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1, -1)
    return x


class ForecastModelBase(nn.Module):
    """统一 baseline 接口：输入历史 batch，输出 [B, L, C, H, W]。"""

    spec = ForecastModelSpec(
        name="base",
        description="abstract forecast baseline",
        uses_static=False,
        trainable=False,
        family="base",
    )

    def __init__(
        self,
        *,
        dynamic_vars: Sequence[str],
        static_vars: Optional[Sequence[str]] = None,
        k: int = 1,
        lead_times: Optional[Sequence[int]] = None,
        static_policy: str = "dynamic_only",
        forecast_protocol: str = "direct",
        one_step_lead: int = 1,
        time_step_hours: float = 6.0,
    ) -> None:
        super().__init__()
        self.dynamic_vars = _to_list(dynamic_vars)
        self.static_vars = _to_list(static_vars)
        self.k = int(k)
        self.lead_times = [int(x) for x in list(lead_times or [])]
        self.static_policy = normalize_static_policy(static_policy)
        self.forecast_protocol = self._normalize_forecast_protocol(forecast_protocol)
        self.one_step_lead = max(1, int(one_step_lead))
        self.time_step_hours = float(time_step_hours)

        if len(self.dynamic_vars) <= 0:
            raise ValueError("dynamic_vars must not be empty")
        if self.k <= 0:
            raise ValueError("k must be positive")

    @staticmethod
    def _normalize_forecast_protocol(protocol: str) -> str:
        value = str(protocol or "direct").strip().lower()
        aliases = {
            "multi_horizon": "direct",
            "multi_horizon_direct": "direct",
            "official": "official_rollout",
            "rollout": "official_rollout",
            "autoregressive": "official_rollout",
            "ar": "official_rollout",
        }
        value = aliases.get(value, value)
        allowed = {"direct", "official_rollout"}
        if value not in allowed:
            raise ValueError(f"Unknown forecast_protocol={protocol!r}; expected one of {sorted(allowed)}")
        return value

    @property
    def out_channels(self) -> int:
        return len(self.dynamic_vars)

    @property
    def dynamic_feature_channels(self) -> int:
        return len(self.dynamic_vars) * self.k

    @property
    def configured_static_channels(self) -> int:
        if self.static_policy == "dynamic_only":
            return 0
        return len(self.static_vars)

    @property
    def feature_channels(self) -> int:
        return self.dynamic_feature_channels + self.configured_static_channels

    def prepare_batch(self, batch: TensorDict, lead_times: Optional[Sequence[int]] = None) -> ForecastBatchView:
        x_ctx = batch["x_ctx"]
        if not isinstance(x_ctx, torch.Tensor):
            x_ctx = torch.as_tensor(x_ctx)
        x_ctx = x_ctx.float()
        if x_ctx.ndim != 4:
            raise ValueError(f"x_ctx must be [B,C,H,W], got shape={tuple(x_ctx.shape)}")

        b, channels, h, w = x_ctx.shape
        dyn_channels = self.dynamic_feature_channels
        if channels < dyn_channels:
            raise ValueError(
                f"x_ctx has {channels} channels, but {dyn_channels} dynamic history channels are required"
            )
        dynamic = x_ctx[:, :dyn_channels].reshape(b, len(self.dynamic_vars), self.k, h, w)
        static = self._extract_static(batch, x_ctx, dyn_channels)
        leads = _lead_tensor(batch, lead_times if lead_times is not None else self.lead_times or None, x_ctx.device)

        mean = _norm_tensor(batch, "mean", x_ctx.device, x_ctx.dtype)
        std = _norm_tensor(batch, "std", x_ctx.device, x_ctx.dtype)
        hour_of_day = self._hour_tensor(batch, x_ctx.device)
        clim = batch.get("clim", None)
        if clim is not None:
            if not isinstance(clim, torch.Tensor):
                clim = torch.as_tensor(clim)
            clim = _with_batch_dim(clim.to(device=x_ctx.device, dtype=x_ctx.dtype), b)

        return ForecastBatchView(
            x_ctx=x_ctx,
            dynamic=dynamic,
            static=static,
            lead_steps=leads,
            mean=mean,
            std=std,
            clim=clim,
            hour_of_day=hour_of_day,
        )

    def _hour_tensor(self, batch: TensorDict, device: torch.device) -> Optional[torch.Tensor]:
        raw = batch.get("hour_of_day", None)
        if raw is None:
            return None
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)
        if raw.ndim == 0:
            raw = raw.view(1)
        return raw.to(device=device, dtype=torch.long).flatten()

    def _extract_static(self, batch: TensorDict, x_ctx: torch.Tensor, dyn_channels: int) -> torch.Tensor:
        b, _, h, w = x_ctx.shape
        if self.static_policy == "dynamic_only":
            return x_ctx.new_zeros((b, 0, h, w))

        static = None
        for key in ("static_raw", "static_cont"):
            value = batch.get(key, None)
            if isinstance(value, torch.Tensor) and value.ndim in (3, 4) and value.numel() > 0:
                static = _with_batch_dim(value.to(device=x_ctx.device, dtype=x_ctx.dtype), b)
                break

        if static is None:
            cont = batch.get("static_cont", None)
            cat = batch.get("static_cat", None)
            parts = []
            if isinstance(cont, torch.Tensor) and cont.numel() > 0:
                parts.append(_with_batch_dim(cont.to(device=x_ctx.device, dtype=x_ctx.dtype), b))
            if isinstance(cat, torch.Tensor) and cat.numel() > 0:
                parts.append(_with_batch_dim(cat.to(device=x_ctx.device, dtype=x_ctx.dtype), b))
            if parts:
                static = torch.cat(parts, dim=1)

        if static is None and x_ctx.shape[1] > dyn_channels:
            static = x_ctx[:, dyn_channels:]

        if static is None:
            static = x_ctx.new_zeros((b, self.configured_static_channels, h, w))

        if self.configured_static_channels > 0:
            if static.shape[1] > self.configured_static_channels:
                static = static[:, : self.configured_static_channels]
            elif static.shape[1] < self.configured_static_channels:
                pad = x_ctx.new_zeros((b, self.configured_static_channels - static.shape[1], h, w))
                static = torch.cat([static, pad], dim=1)

        if self.static_policy == "static_zero":
            static = torch.zeros_like(static)
        elif self.static_policy == "static_shuffle" and static.numel() > 0:
            static = torch.flip(static, dims=(-1,))
        return static

    def make_features(self, view: ForecastBatchView) -> torch.Tensor:
        b, c, k, h, w = view.dynamic.shape
        dynamic_features = view.dynamic.reshape(b, c * k, h, w)
        if view.static.shape[1] == 0:
            return dynamic_features
        return torch.cat([dynamic_features, view.static.to(dtype=dynamic_features.dtype)], dim=1)

    def repeat_over_leads(self, field: torch.Tensor, lead_steps: torch.Tensor) -> torch.Tensor:
        return field.unsqueeze(1).expand(-1, int(lead_steps.numel()), -1, -1, -1).contiguous()

    def climatology_norm(self, view: ForecastBatchView) -> torch.Tensor:
        if view.clim is None:
            return torch.zeros_like(view.last)
        if view.mean is None or view.std is None:
            return view.clim
        mean = _with_batch_dim(view.mean, view.last.shape[0]).to(device=view.last.device, dtype=view.last.dtype)
        std = _with_batch_dim(view.std, view.last.shape[0]).to(device=view.last.device, dtype=view.last.dtype).clamp_min(1e-6)
        return (view.clim - mean) / std

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _roll_history(self, view: ForecastBatchView, pred: torch.Tensor) -> ForecastBatchView:
        pred = pred.to(device=view.dynamic.device, dtype=view.dynamic.dtype)
        if view.dynamic.shape[2] == 1:
            dynamic = pred.unsqueeze(2)
        else:
            dynamic = torch.cat([view.dynamic[:, :, 1:], pred.unsqueeze(2)], dim=2)
        hour_of_day = self._advance_hour(view)
        return replace(view, dynamic=dynamic, hour_of_day=hour_of_day)

    def _advance_hour(self, view: ForecastBatchView) -> Optional[torch.Tensor]:
        if view.hour_of_day is None:
            return None
        hours = int(round(self.time_step_hours * self.one_step_lead))
        return (view.hour_of_day + hours) % 24

    def _one_step_tensor(self, view: ForecastBatchView) -> torch.Tensor:
        return torch.as_tensor(self.one_step_lead, dtype=torch.long, device=view.dynamic.device)

    def _forward_direct(self, view: ForecastBatchView) -> torch.Tensor:
        outputs = [self.forecast_lead(view, lead) for lead in view.lead_steps]
        return torch.stack(outputs, dim=1)

    def _forward_rollout(self, view: ForecastBatchView) -> torch.Tensor:
        requested = [int(x) for x in view.lead_steps.detach().cpu().tolist()]
        if not requested:
            raise ValueError("lead_steps must not be empty for official_rollout")
        max_lead = max(requested)
        outputs: Dict[int, torch.Tensor] = {}
        current = view
        one_step = self._one_step_tensor(view)
        for step in range(1, max_lead + 1):
            pred = self.forecast_lead(current, one_step)
            current = self._roll_history(current, pred)
            if step in requested:
                outputs[step] = pred
        return torch.stack([outputs[step] for step in requested], dim=1)

    def forward(self, batch: TensorDict, lead_times: Optional[Sequence[int]] = None) -> torch.Tensor:
        view = self.prepare_batch(batch, lead_times=lead_times)
        if self.forecast_protocol == "official_rollout":
            return self._forward_rollout(view)
        return self._forward_direct(view)

    @torch.no_grad()
    def predict(self, batch: TensorDict, lead_times: Optional[Sequence[int]] = None) -> torch.Tensor:
        self.eval()
        return self.forward(batch, lead_times=lead_times)