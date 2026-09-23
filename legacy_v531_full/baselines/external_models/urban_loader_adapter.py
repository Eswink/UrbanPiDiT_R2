from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, TensorDict


@dataclass(frozen=True)
class UrbanForecastBatch:
    """Canonical tensor view shared by FourCastNet/GraphCast/GenCast adapters.

    dynamic: [B, C, K, H, W], normalized dynamic history in variable-major order.
    static:  [B, S, H, W], normalized static morphology channels after the selected
             static policy.  Empty when ``static_policy=dynamic_only``.
    features: [B, C*K + S + E, H, W], CNN/GNN-ready feature stack.
    target: optional [B, L, C, H, W], normalized future targets when available.
    lead_steps: [L], integer lead steps.
    coords: [B, 2, H, W], normalized y/x grid coordinates in [-1, 1].
    """

    view: ForecastBatchView
    features: torch.Tensor
    target: Optional[torch.Tensor]
    coords: torch.Tensor
    hour_of_day: Optional[torch.Tensor]

    @property
    def dynamic(self) -> torch.Tensor:
        return self.view.dynamic

    @property
    def static(self) -> torch.Tensor:
        return self.view.static

    @property
    def lead_steps(self) -> torch.Tensor:
        return self.view.lead_steps

    @property
    def last(self) -> torch.Tensor:
        return self.view.last


class UrbanBatchAdapter:
    """Convert UrbanPiDiT loader dictionaries into baseline-friendly tensors.

    This class deliberately uses the same ``ForecastModelBase.prepare_batch`` path
    as existing baselines, so FourCastNet/GraphCast/GenCast receive the same
    history length, normalization statistics, static channels, and lead-time
    tensors as UrbanPiDiT and the lightweight baselines.
    """

    def __init__(self, model: ForecastModelBase, *, include_coords: bool = True, include_hour: bool = False) -> None:
        self.model = model
        self.include_coords = bool(include_coords)
        self.include_hour = bool(include_hour)

    def __call__(self, batch: TensorDict, lead_times: Optional[Sequence[int]] = None) -> UrbanForecastBatch:
        view = self.model.prepare_batch(batch, lead_times=lead_times)
        parts = [self.model.make_features(view)]
        coords = self._coords(view.last)
        if self.include_coords:
            parts.append(coords)
        hour = self._hour_tensor(batch, view.last.device) if self.include_hour else None
        if self.include_hour:
            b, _, h, w = view.last.shape
            if hour is None:
                hour_map = torch.zeros((b, 1, h, w), device=view.last.device, dtype=view.last.dtype)
            else:
                hour_map = (hour.float().view(-1, 1, 1, 1) / 23.0).expand(b, 1, h, w).to(dtype=view.last.dtype)
            parts.append(hour_map)
        target = self._target(batch, view.last.device, view.last.dtype)
        return UrbanForecastBatch(
            view=view,
            features=torch.cat(parts, dim=1),
            target=target,
            coords=coords,
            hour_of_day=hour,
        )

    @staticmethod
    def _coords(reference: torch.Tensor) -> torch.Tensor:
        b, _, h, w = reference.shape
        y = torch.linspace(-1.0, 1.0, h, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, w, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(b, -1, -1, -1).contiguous()

    @staticmethod
    def _hour_tensor(batch: TensorDict, device: torch.device) -> Optional[torch.Tensor]:
        raw = batch.get("hour_of_day", None)
        if raw is None:
            return None
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)
        if raw.ndim == 0:
            raw = raw.view(1)
        return raw.to(device=device, dtype=torch.long).flatten()

    @staticmethod
    def _target(batch: TensorDict, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if "y" in batch:
            y = batch["y"]
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            return y.to(device=device, dtype=dtype)
        if "x0" in batch:
            y = batch["x0"]
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            return y.to(device=device, dtype=dtype).unsqueeze(1)
        return None


def move_batch_to_device(batch: Dict, device: torch.device | str) -> Dict:
    """Recursively move tensors inside the UrbanPiDiT batch to a device."""

    device = torch.device(device)

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    return {k: move(v) for k, v in batch.items()}
