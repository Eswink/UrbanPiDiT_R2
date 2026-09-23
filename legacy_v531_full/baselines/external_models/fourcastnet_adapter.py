from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec
from .urban_loader_adapter import UrbanBatchAdapter


class SpectralMixingBlock(nn.Module):
    """Compact AFNO/FourCastNet-style spectral mixing block for small urban grids.

    The official FourCastNet source is vendored in ``baselines/external_sources``.
    This block provides a loader-compatible PyTorch baseline that can be trained on
    UrbanPiDiT's [B, C, H, W] tensors without requiring ERA5/HDF5 preprocessing.
    """

    def __init__(self, channels: int, modes: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = max(1, int(modes))
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.weight_real = nn.Parameter(torch.zeros(channels, self.modes, self.modes))
        self.weight_imag = nn.Parameter(torch.zeros(channels, self.modes, self.modes))
        hidden = max(channels, int(channels * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        nn.init.normal_(self.weight_real, std=0.02)
        nn.init.normal_(self.weight_imag, std=0.02)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm1(x)
        fft = torch.fft.rfft2(z, norm="ortho")
        h_modes = min(self.modes, fft.shape[-2])
        w_modes = min(self.modes, fft.shape[-1])
        weight = torch.complex(
            self.weight_real[:, :h_modes, :w_modes],
            self.weight_imag[:, :h_modes, :w_modes],
        ).unsqueeze(0)
        out_fft = torch.zeros_like(fft)
        out_fft[:, :, :h_modes, :w_modes] = fft[:, :, :h_modes, :w_modes] * weight
        z = torch.fft.irfft2(out_fft, s=x.shape[-2:], norm="ortho")
        x = residual + z
        x = x + self.mlp(self.norm2(x))
        return x


class UrbanFourCastNetAdapter(ForecastModelBase):
    spec = ForecastModelSpec(
        name="fourcastnet",
        description="FourCastNet/AFNO-style spectral neural baseline adapted to UrbanPiDiT loader batches",
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
        depth: int = 6,
        modes: int = 8,
        dropout: float = 0.0,
        residual: bool = True,
        include_coords: bool = True,
        include_hour: bool = False,
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
        self.hidden_channels = int(hidden_channels)
        self.residual = bool(residual)
        self.include_coords = bool(include_coords)
        self.adapter = UrbanBatchAdapter(self, include_coords=include_coords, include_hour=include_hour)
        extra = (2 if include_coords else 0) + (1 if include_hour else 0)
        in_channels = self.feature_channels + extra
        self.stem = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [SpectralMixingBlock(self.hidden_channels, modes=modes, dropout=dropout) for _ in range(max(1, int(depth)))]
        )
        self.head = nn.Sequential(
            nn.BatchNorm2d(self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _predict_one(self, view: ForecastBatchView, features: torch.Tensor) -> torch.Tensor:
        z = self.stem(features)
        for block in self.blocks:
            z = block(z)
        delta = self.head(z)
        return view.last + delta if self.residual else delta

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        del lead_step
        features = self.make_features(view)
        if self.include_coords:
            coords = UrbanBatchAdapter._coords(view.last)
            features = torch.cat([features, coords], dim=1)
        if self.adapter.include_hour:
            b, _, h, w = view.last.shape
            if view.hour_of_day is None:
                hour_map = torch.zeros((b, 1, h, w), device=view.last.device, dtype=view.last.dtype)
            else:
                hour = view.hour_of_day.to(device=view.last.device, dtype=view.last.dtype)
                hour_map = (hour.view(-1, 1, 1, 1) / 23.0).expand(b, 1, h, w)
            features = torch.cat([features, hour_map], dim=1)
        return self._predict_one(view, features)
