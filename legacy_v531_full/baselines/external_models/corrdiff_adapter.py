from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec
from .urban_loader_adapter import UrbanBatchAdapter


class _ConvResidualBlock(nn.Module):
    """Lightweight residual block used by the Urban CorrDiff-style baseline."""

    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, min(8, channels // 8 or 1)), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _ResidualCorrector(nn.Module):
    """Conditional residual corrector used after the regression mean network.

    This is intentionally a loader-adapted CorrDiff-style module rather than an
    official CorrDiff reproduction.  It follows the regression-then-generative
    residual-correction protocol: a deterministic mean forecast is produced
    first, then a conditional residual corrector refines it under a noise-level
    embedding.  At evaluation time the same corrector can produce an ensemble,
    allowing CRPS to be computed from actual samples.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, steps: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + out_channels + 2, hidden_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([_ConvResidualBlock(hidden_channels, dropout=dropout) for _ in range(max(1, int(steps)))])
        self.decoder = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, min(8, hidden_channels // 8 or 1)), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, features: torch.Tensor, mean_forecast: torch.Tensor, lead_map: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        z = self.encoder(torch.cat([features, mean_forecast, lead_map, noise_level], dim=1))
        for block in self.blocks:
            z = block(z)
        return self.decoder(z)


class UrbanCorrDiffAdapter(ForecastModelBase):
    """NVIDIA PhysicsNeMo CorrDiff-style residual corrective diffusion baseline for UrbanPiDiT batches.

    Fairness contract:
    - Uses the same UrbanPiDiT loader batch keys as the other baselines.
    - Can run with ``same_static``, ``dynamic_only``, ``static_zero`` and
      ``static_shuffle`` policies via ``ForecastModelBase``.
    - Trains from scratch on the local urban data; it is not an official NVIDIA
      CorrDiff checkpoint or official km-scale downscaling reproduction.  The upstream
      PhysicsNeMo CorrDiff example is archived under
      baselines/external_sources/physicsnemo_weather_corrdiff for provenance.
    - Uses the same metric suite and can return an ensemble during evaluation so
      CRPS is a sample-based probabilistic score rather than a placeholder.
    """

    spec = ForecastModelSpec(
        name="corrdiff",
        description=(
            "CorrDiff-style baseline: deterministic regression mean followed by "
            "conditional residual corrective diffusion/refinement, adapted to UrbanPiDiT loader batches"
        ),
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
        hidden_channels: int = 96,
        mean_depth: int = 3,
        correction_steps: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
        include_coords: bool = True,
        include_hour: bool = False,
        noise_scale: float = 0.05,
        train_noise_scale: float | None = None,
        eval_ensemble_size: int = 1,
        return_ensemble_eval: bool = True,
        zero_init_correction: bool = True,
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
        self.noise_scale = float(noise_scale)
        self.train_noise_scale = float(train_noise_scale if train_noise_scale is not None else noise_scale)
        self.eval_ensemble_size = max(1, int(eval_ensemble_size))
        self.return_ensemble_eval = bool(return_ensemble_eval)
        self.include_coords = bool(include_coords)
        self.adapter = UrbanBatchAdapter(self, include_coords=include_coords, include_hour=include_hour)

        extra = (2 if include_coords else 0) + (1 if include_hour else 0) + 1  # lead map
        mean_in_channels = self.feature_channels + extra
        mean_layers: List[nn.Module] = [nn.Conv2d(mean_in_channels, hidden_channels, kernel_size=1), nn.SiLU()]
        for _ in range(max(1, int(mean_depth))):
            mean_layers.append(_ConvResidualBlock(hidden_channels, dropout=dropout))
        mean_layers.append(nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1))
        self.mean_net = nn.Sequential(*mean_layers)
        # Keep the mean head initially close to persistence when residual=True.
        nn.init.zeros_(self.mean_net[-1].weight)
        nn.init.zeros_(self.mean_net[-1].bias)

        corrector_in_channels = self.feature_channels + (2 if include_coords else 0) + (1 if include_hour else 0)
        self.corrector = _ResidualCorrector(
            in_channels=corrector_in_channels,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            steps=correction_steps,
            dropout=dropout,
        )
        self.correction_gate = nn.Parameter(torch.tensor(0.0))
        if not zero_init_correction:
            nn.init.normal_(self.corrector.decoder[-1].weight, mean=0.0, std=1e-3)

    def _lead_map(self, features: torch.Tensor, lead_step: torch.Tensor) -> torch.Tensor:
        b, _, h, w = features.shape
        denom = float(max(self.lead_times) if self.lead_times else max(int(lead_step.item()), 1))
        return (lead_step.to(device=features.device, dtype=features.dtype).view(1, 1, 1, 1) / max(denom, 1.0)).expand(b, 1, h, w)

    def _mean_forecast(self, view: ForecastBatchView, features: torch.Tensor, lead_step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lead_map = self._lead_map(features, lead_step)
        mean_delta = self.mean_net(torch.cat([features, lead_map], dim=1))
        mean_forecast = view.last + mean_delta if self.residual else mean_delta
        return mean_forecast, lead_map

    def _corrected_sample(
        self,
        *,
        features: torch.Tensor,
        mean_forecast: torch.Tensor,
        lead_map: torch.Tensor,
        noise_scale: float,
        stochastic: bool,
    ) -> torch.Tensor:
        b, _, h, w = mean_forecast.shape
        if stochastic and noise_scale > 0:
            noise = torch.randn_like(mean_forecast) * float(noise_scale)
            noisy_mean = mean_forecast + noise
            level = torch.full((b, 1, h, w), float(noise_scale), device=features.device, dtype=features.dtype)
        else:
            noisy_mean = mean_forecast
            level = torch.zeros((b, 1, h, w), device=features.device, dtype=features.dtype)
        corr = self.corrector(features, noisy_mean, lead_map, level)
        return mean_forecast + torch.tanh(self.correction_gate) * corr

    def _predict_one(self, view: ForecastBatchView, features: torch.Tensor, lead_step: torch.Tensor, *, stochastic: bool) -> torch.Tensor:
        mean_forecast, lead_map = self._mean_forecast(view, features, lead_step)
        noise_scale = self.train_noise_scale if self.training else self.noise_scale
        return self._corrected_sample(
            features=features,
            mean_forecast=mean_forecast,
            lead_map=lead_map,
            noise_scale=noise_scale,
            stochastic=stochastic,
        )

    def forward(self, batch, lead_times=None) -> torch.Tensor:
        ub = self.adapter(batch, lead_times=lead_times)
        stochastic = bool(self.training and self.train_noise_scale > 0)
        outputs = [self._predict_one(ub.view, ub.features, lead, stochastic=stochastic) for lead in ub.lead_steps]
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def predict(self, batch, lead_times=None) -> torch.Tensor:
        self.eval()
        ub = self.adapter(batch, lead_times=lead_times)
        if self.return_ensemble_eval and self.eval_ensemble_size > 1:
            ensemble = []
            for _ in range(self.eval_ensemble_size):
                outputs = [self._predict_one(ub.view, ub.features, lead, stochastic=True) for lead in ub.lead_steps]
                ensemble.append(torch.stack(outputs, dim=1))  # [B,L,C,H,W]
            return torch.stack(ensemble, dim=1)  # [B,E,L,C,H,W]
        outputs = [self._predict_one(ub.view, ub.features, lead, stochastic=False) for lead in ub.lead_steps]
        return torch.stack(outputs, dim=1)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        features = self.make_features(view)
        if self.include_coords:
            features = torch.cat([features, UrbanBatchAdapter._coords(view.last)], dim=1)
        if self.adapter.include_hour:
            b, _, h, w = view.last.shape
            if view.hour_of_day is None:
                hour_map = torch.zeros((b, 1, h, w), device=view.last.device, dtype=view.last.dtype)
            else:
                hour = view.hour_of_day.to(device=view.last.device, dtype=view.last.dtype)
                hour_map = (hour.view(-1, 1, 1, 1) / 23.0).expand(b, 1, h, w)
            features = torch.cat([features, hour_map], dim=1)
        return self._predict_one(view, features, lead_step, stochastic=bool(self.training))