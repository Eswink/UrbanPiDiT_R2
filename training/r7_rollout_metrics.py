"""Per-variable RMSE with correct accumulation across initialization batches."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import torch
from model.r7_rollout import validate_horizons


class RolloutRMSEAccumulator:
    """Equal weight per initialization; latitude-area mean within each field.

    Inputs are normalized [B,L,C,H,W]. A supplied training-only std restores
    physical error units (the shared normalization mean cancels). Without std,
    results are explicitly labeled normalized. No cross-variable aggregate.
    """
    def __init__(self, lead_hours: Sequence[int], variables: Sequence[str], *,
                 training_std: torch.Tensor | Sequence[float] | None = None,
                 units: Sequence[str] | None = None):
        self.lead_hours = validate_horizons(lead_hours)
        self.variables = tuple(variables)
        if not self.variables or any(not isinstance(v, str) or not v for v in self.variables):
            raise ValueError("variables must be nonempty names")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be unique")
        c = len(self.variables)
        self.std = torch.ones(c, dtype=torch.float64)
        if training_std is not None:
            self.std = torch.as_tensor(training_std, dtype=torch.float64).detach().cpu().clone()
            if self.std.shape != (c,) or not torch.isfinite(self.std).all() or (self.std <= 0).any():
                raise ValueError("training_std must be positive finite [C]")
            if units is None:
                raise ValueError("physical metrics require explicit variable units")
            self.units = tuple(units)
        else:
            if units is not None:
                raise ValueError("physical units cannot be claimed without training_std")
            self.units = ("normalized",) * c
        if len(self.units) != c or any(not isinstance(u, str) or not u for u in self.units):
            raise ValueError("units must match variables")
        self.sum_squared_error = torch.zeros(len(self.lead_hours), c, dtype=torch.float64)
        self.initializations = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               latitude: torch.Tensor | None = None) -> None:
        if prediction.ndim != 5 or prediction.shape != target.shape or min(prediction.shape) < 1:
            raise ValueError("prediction/target must have equal nonempty [B,L,C,H,W] shapes")
        b, l, c, h, _ = prediction.shape
        if (l, c) != self.sum_squared_error.shape or prediction.device != target.device:
            raise ValueError("metric horizon/channel/device mismatch")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("metric inputs must be floating point")
        if latitude is None:
            weight = torch.ones(b, h, dtype=torch.float64, device=prediction.device)
        else:
            lat = torch.as_tensor(latitude, dtype=torch.float64, device=prediction.device)
            if lat.shape == (h,):
                lat = lat.unsqueeze(0).expand(b, h)
            if lat.shape != (b, h) or not torch.isfinite(lat).all() or (lat.abs() > 90).any():
                raise ValueError("latitude must be finite [H] or [B,H] within [-90,90]")
            weight = torch.cos(torch.deg2rad(lat)).clamp_min(0)
        total = weight.sum(-1)
        if (total < 1e-12).any():
            raise ValueError("latitude weights have no usable area")
        delta = (prediction.detach().double() - target.detach().double())
        delta = delta * self.std.to(delta.device)[None, None, :, None, None]
        squared = delta.square()
        if not torch.isfinite(squared).all():
            raise ValueError("nonfinite metric input; no samples silently dropped")
        per_initialization = (squared.mean(-1) * weight[:, None, None, :]).sum(-1)
        per_initialization /= total[:, None, None]
        # Only compact [L,C] sufficient statistics survive update().
        self.sum_squared_error += per_initialization.sum(0).cpu()
        self.initializations += b

    def compute(self) -> torch.Tensor:
        if self.initializations == 0:
            raise RuntimeError("no initialization samples were accumulated")
        return (self.sum_squared_error / self.initializations).sqrt()

    def write_csv(self, path: str | Path, *, overwrite: bool = False) -> Path:
        rmse = self.compute()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["lead_hours", "variable", "rmse", "unit", "n_initializations"])
            for i, lead in enumerate(self.lead_hours):
                for j, variable in enumerate(self.variables):
                    writer.writerow([lead, variable, format(float(rmse[i, j]), ".17g"),
                                     self.units[j], self.initializations])
        return path
