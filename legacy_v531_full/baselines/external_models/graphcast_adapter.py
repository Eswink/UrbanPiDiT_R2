from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec
from .urban_loader_adapter import UrbanBatchAdapter


class GridMessagePassingBlock(nn.Module):
    """GraphCast-style local grid message passing on the UrbanPiDiT domain."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.message = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.update = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        msg = self.message(self.norm(x))
        return x + self.update(torch.cat([x, msg], dim=1))


class UrbanGraphCastAdapter(ForecastModelBase):
    spec = ForecastModelSpec(
        name="graphcast",
        description="GraphCast-style grid graph neural baseline adapted to UrbanPiDiT loader batches",
        uses_static=True,
        trainable=True,
        family="external_weather_baseline",
    )

    def __init__(
        self,
        *,
        dynamic_vars: Sequence[str],
        static_vars: Sequence[str] | None = None,
        k: int = 1,
        lead_times: Sequence[int] | None = None,
        static_policy: str = "same_static",
        hidden_channels: int = 128,
        message_passing_steps: int = 8,
        dropout: float = 0.0,
        residual: bool = True,
        include_coords: bool = True,
        include_hour: bool = False,
        include_time_forcing: bool = True,
        forecast_protocol: str = "direct",
        one_step_lead: int = 1,
        time_step_hours: float = 6.0,
    ) -> None:
        super().__init__(
            dynamic_vars=dynamic_vars,
            static_vars=static_vars,
            k=k,
            lead_times=lead_times,
            static_policy=static_policy,
            forecast_protocol=forecast_protocol,
            one_step_lead=one_step_lead,
            time_step_hours=time_step_hours,
        )
        self.residual = bool(residual)
        self.include_coords = bool(include_coords)
        self.include_time_forcing = bool(include_time_forcing)
        self.adapter = UrbanBatchAdapter(self, include_coords=include_coords, include_hour=include_hour)
        extra = (2 if include_coords else 0) + (1 if include_hour else 0) + (2 if include_time_forcing else 0)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.feature_channels + extra, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [GridMessagePassingBlock(hidden_channels, dropout=dropout) for _ in range(max(1, int(message_passing_steps)))]
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _predict_one(self, view: ForecastBatchView, features: torch.Tensor) -> torch.Tensor:
        z = self.encoder(features)
        for block in self.blocks:
            z = block(z)
        delta = self.decoder(z)
        return view.last + delta if self.residual else delta

    def _hour_maps(self, view: ForecastBatchView) -> list[torch.Tensor]:
        if not (self.adapter.include_hour or self.include_time_forcing):
            return []
        b, _, h, w = view.last.shape
        if view.hour_of_day is None:
            hour = torch.zeros(b, device=view.last.device, dtype=view.last.dtype)
        else:
            hour = view.hour_of_day.to(device=view.last.device, dtype=view.last.dtype)
        maps = []
        if self.adapter.include_hour:
            maps.append((hour.view(-1, 1, 1, 1) / 23.0).expand(b, 1, h, w))
        if self.include_time_forcing:
            phase = hour * (2.0 * torch.pi / 24.0)
            maps.append(torch.sin(phase).view(-1, 1, 1, 1).expand(b, 1, h, w))
            maps.append(torch.cos(phase).view(-1, 1, 1, 1).expand(b, 1, h, w))
        return maps

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        del lead_step
        parts = [self.make_features(view)]
        if self.include_coords:
            parts.append(UrbanBatchAdapter._coords(view.last))
        parts.extend(self._hour_maps(view))
        features = torch.cat(parts, dim=1)
        return self._predict_one(view, features)
