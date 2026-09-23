from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec, TensorDict
from .urban_loader_adapter import UrbanBatchAdapter


class DenoisingResidualBlock(nn.Module):
    """Small GenCast-style denoising residual refinement block."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UrbanGenCastAdapter(ForecastModelBase):
    spec = ForecastModelSpec(
        name="gencast",
        description="GenCast-style probabilistic residual-denoising baseline adapted to UrbanPiDiT loader batches",
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
        denoising_steps: int = 7,
        dropout: float = 0.0,
        residual: bool = True,
        include_coords: bool = True,
        include_hour: bool = False,
        stochastic_eval: bool = False,
        noise_scale: float | None = None,
        sigma_min: float = 0.002,
        sigma_max: float = 0.8,
        sigma_data: float = 0.5,
        sigma_sample_mean: float = -1.2,
        sigma_sample_std: float = 1.2,
        rho: float = 7.0,
        num_noise_levels: int = 8,
        num_samples: int = 1,
        stochastic_churn: float = 0.0,
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
        self.stochastic_eval = bool(stochastic_eval)
        self.sigma_min = max(float(sigma_min), 1e-6)
        self.sigma_max = max(float(sigma_max), self.sigma_min)
        self.sigma_data = max(float(sigma_data), 1e-6)
        self.sigma_sample_mean = float(sigma_sample_mean)
        self.sigma_sample_std = max(float(sigma_sample_std), 1e-6)
        self.rho = max(float(rho), 1e-6)
        self.num_noise_levels = max(1, int(num_noise_levels))
        self.num_samples = max(1, int(num_samples))
        self.stochastic_churn = max(0.0, float(stochastic_churn))
        self.noise_scale = float(noise_scale) if noise_scale is not None else self.sigma_max
        self.adapter = UrbanBatchAdapter(self, include_coords=include_coords, include_hour=include_hour)

        cond_extra = (2 if include_coords else 0) + (1 if include_hour else 0)
        in_channels = self.feature_channels + cond_extra + self.out_channels + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DenoisingResidualBlock(hidden_channels, dropout=dropout) for _ in range(max(1, int(denoising_steps)))]
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _condition_features(self, view: ForecastBatchView) -> torch.Tensor:
        parts = [self.make_features(view)]
        if self.include_coords:
            parts.append(UrbanBatchAdapter._coords(view.last))
        if self.adapter.include_hour:
            b, _, h, w = view.last.shape
            if view.hour_of_day is None:
                hour_map = torch.zeros((b, 1, h, w), device=view.last.device, dtype=view.last.dtype)
            else:
                hour = view.hour_of_day.to(device=view.last.device, dtype=view.last.dtype)
                hour_map = (hour.view(-1, 1, 1, 1) / 23.0).expand(b, 1, h, w)
            parts.append(hour_map)
        return torch.cat(parts, dim=1)

    def _as_batch_sigma(self, sigma: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
        sigma = sigma.to(device=reference.device, dtype=reference.dtype).flatten()
        if sigma.numel() == 1:
            sigma = sigma.expand(reference.shape[0])
        if sigma.numel() != reference.shape[0]:
            raise ValueError(f"sigma batch size mismatch: got {sigma.numel()}, expected {reference.shape[0]}")
        return sigma.clamp_min(self.sigma_min)

    def _sigma_shape(self, sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return sigma.view(reference.shape[0], 1, 1, 1)

    def _noise_schedule(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ramp = torch.linspace(0.0, 1.0, self.num_noise_levels, device=device, dtype=dtype)
        min_inv = self.sigma_min ** (1.0 / self.rho)
        max_inv = self.sigma_max ** (1.0 / self.rho)
        return (max_inv + ramp * (min_inv - max_inv)) ** self.rho

    def _sample_training_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = torch.randn(batch_size, device=device, dtype=dtype) * self.sigma_sample_std + self.sigma_sample_mean
        return sigma.exp().clamp(min=self.sigma_min, max=self.sigma_max)

    def _to_model_space(self, view: ForecastBatchView, field: torch.Tensor) -> torch.Tensor:
        return field - view.last if self.residual else field

    def _from_model_space(self, view: ForecastBatchView, value: torch.Tensor) -> torch.Tensor:
        return view.last + value if self.residual else value

    def _select_target(self, batch: TensorDict, view: ForecastBatchView, lead_step: int | None = None) -> torch.Tensor:
        if lead_step is None:
            lead_step = self.one_step_lead
        if "y" in batch:
            y = batch["y"]
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            y = y.to(device=view.last.device, dtype=view.last.dtype)
            if y.ndim == 4:
                return y
            if y.ndim != 5:
                raise ValueError(f"y must be [B,L,C,H,W] or [B,C,H,W], got shape={tuple(y.shape)}")
            lead_values = view.lead_steps.detach().cpu().tolist()
            try:
                idx = [int(x) for x in lead_values].index(int(lead_step))
            except ValueError:
                idx = 0
            return y[:, min(idx, y.shape[1] - 1)]
        if "x0" in batch:
            x0 = batch["x0"]
            if not isinstance(x0, torch.Tensor):
                x0 = torch.as_tensor(x0)
            return x0.to(device=view.last.device, dtype=view.last.dtype)
        raise KeyError("GenCast training_loss requires batch['y'] or batch['x0']")

    def _denoise_model_space(
        self,
        view: ForecastBatchView,
        noisy: torch.Tensor,
        sigma: torch.Tensor | float,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma = self._as_batch_sigma(sigma, noisy)
        sigma_b = self._sigma_shape(sigma, noisy)
        sigma_data = torch.as_tensor(self.sigma_data, device=noisy.device, dtype=noisy.dtype)
        c_skip = sigma_data.square() / (sigma_b.square() + sigma_data.square())
        c_out = sigma_b * sigma_data / torch.sqrt(sigma_b.square() + sigma_data.square())
        c_in = 1.0 / torch.sqrt(sigma_b.square() + sigma_data.square())
        c_noise = (torch.log(sigma_b.clamp_min(1e-12)) / 4.0).expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
        if cond is None:
            cond = self._condition_features(view)
        z = self.encoder(torch.cat([cond, c_in * noisy, c_noise], dim=1))
        for block in self.blocks:
            z = block(z)
        raw = self.decoder(z)
        return c_skip * noisy + c_out * raw

    def training_loss(self, batch: TensorDict, lead_times: Sequence[int] | None = None) -> torch.Tensor:
        view = self.prepare_batch(batch, lead_times=lead_times)
        target = self._select_target(batch, view, lead_step=self.one_step_lead)
        clean = self._to_model_space(view, target)
        sigma = self._sample_training_sigma(clean.shape[0], clean.device, clean.dtype)
        sigma_b = self._sigma_shape(sigma, clean)
        noisy = clean + torch.randn_like(clean) * sigma_b
        denoised = self._denoise_model_space(view, noisy, sigma, cond=self._condition_features(view))
        sigma_data = torch.as_tensor(self.sigma_data, device=clean.device, dtype=clean.dtype)
        weight = (sigma_b.square() + sigma_data.square()) / (sigma_b * sigma_data).square().clamp_min(1e-12)
        return (weight * (denoised - clean).square()).mean()

    def _sample_model_space_once(self, view: ForecastBatchView) -> torch.Tensor:
        cond = self._condition_features(view)
        schedule = self._noise_schedule(view.last.device, view.last.dtype)
        base = torch.zeros_like(view.last) if self.residual else view.last.clone()
        if self.stochastic_eval or self.training:
            current = base + torch.randn_like(base) * schedule[0]
        else:
            current = base
        for idx, sigma_value in enumerate(schedule):
            sigma = torch.full((current.shape[0],), float(sigma_value.detach().cpu()), device=current.device, dtype=current.dtype)
            denoised = self._denoise_model_space(view, current, sigma, cond=cond)
            if idx + 1 < schedule.numel():
                next_sigma = schedule[idx + 1]
                direction = (current - denoised) / sigma.view(current.shape[0], 1, 1, 1).clamp_min(1e-12)
                current = denoised + direction * next_sigma
                if self.stochastic_eval and self.stochastic_churn > 0:
                    current = current + torch.randn_like(current) * float(next_sigma) * self.stochastic_churn
            else:
                current = denoised
        return current

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        del lead_step
        sample_count = self.num_samples if self.stochastic_eval else 1
        samples = [self._from_model_space(view, self._sample_model_space_once(view)) for _ in range(sample_count)]
        return torch.stack(samples, dim=0).mean(dim=0)
