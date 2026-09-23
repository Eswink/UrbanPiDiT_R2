from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from metrics import acc as _acc
from metrics import bias as _bias
from metrics import crps_ensemble as _crps_ensemble
from metrics import mae as _mae
from metrics import rmse as _rmse


@dataclass
class BatchTensors:
    """Tensors required to evaluate deterministic/probabilistic forecast metrics.

    The external baseline adapters normally emit deterministic predictions with
    shape [B,C,H,W].  If a future probabilistic adapter emits an ensemble with
    shape [B,M,C,H,W], ``pred_ensemble_norm`` can be supplied and CRPS will use
    the standard ensemble approximation.  For deterministic baselines, CRPS is
    reported as the deterministic single-member CRPS, which is exactly MAE.
    """

    pred_norm: torch.Tensor  # [B,C,H,W]
    target_norm: torch.Tensor  # [B,C,H,W]
    mean: torch.Tensor  # [B,C,1,1] or [C,1,1]
    std: torch.Tensor  # [B,C,1,1] or [C,1,1]
    clim_denorm: Optional[torch.Tensor] = None  # [B,C,H,W] or [C,H,W]
    lat: Optional[torch.Tensor] = None  # [H] or [B,H]
    pred_ensemble_norm: Optional[torch.Tensor] = None  # optional [B,M,C,H,W]


def _ensure_broadcast(mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure mean/std broadcastable to x=[B,C,H,W]."""

    if mean.ndim == 3:
        mean = mean.unsqueeze(0)
    if std.ndim == 3:
        std = std.unsqueeze(0)
    if mean.shape[0] == 1 and x.shape[0] != 1:
        mean = mean.expand(x.shape[0], -1, -1, -1)
    if std.shape[0] == 1 and x.shape[0] != 1:
        std = std.expand(x.shape[0], -1, -1, -1)
    return mean, std


def denorm(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean, std = _ensure_broadcast(mean, std, x_norm)
    return x_norm * std + mean


def _denorm_ensemble(ensemble_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize ensemble predictions shaped [B,M,C,H,W]."""

    if ensemble_norm.ndim != 5:
        raise ValueError(f"pred_ensemble_norm must be [B,M,C,H,W], got {tuple(ensemble_norm.shape)}")
    b, m, c, h, w = ensemble_norm.shape
    flat = ensemble_norm.reshape(b * m, c, h, w)
    mean_b, std_b = _ensure_broadcast(mean, std, ensemble_norm[:, 0])
    mean_flat = mean_b.unsqueeze(1).expand(b, m, c, h, w).reshape(b * m, c, h, w)
    std_flat = std_b.unsqueeze(1).expand(b, m, c, h, w).reshape(b * m, c, h, w)
    return (flat * std_flat + mean_flat).reshape(b, m, c, h, w)


@torch.no_grad()
def compute_batch_metrics(bt: BatchTensors, *, center: bool = True) -> Dict[str, torch.Tensor]:
    """Compute global RMSE/MAE/Bias/CRPS and ACC when climatology is available."""

    pred_denorm = denorm(bt.pred_norm, bt.mean, bt.std)
    target_denorm = denorm(bt.target_norm, bt.mean, bt.std)

    out: Dict[str, torch.Tensor] = {
        "rmse": _rmse(pred_denorm, target_denorm),
        "mae": _mae(pred_denorm, target_denorm),
        "bias": _bias(pred_denorm, target_denorm),
    }

    if bt.pred_ensemble_norm is not None:
        ensemble_denorm = _denorm_ensemble(bt.pred_ensemble_norm, bt.mean, bt.std)
        out["crps"] = _crps_ensemble(ensemble_denorm, target_denorm, reduction="mean")
    else:
        # Deterministic single-member CRPS equals MAE.  Keeping a separate key
        # makes the output schema identical for deterministic and probabilistic
        # baselines, which is useful for tables.
        out["crps"] = _mae(pred_denorm, target_denorm)

    if bt.clim_denorm is not None:
        out["acc"] = _acc(
            pred_denorm,
            target_denorm,
            bt.clim_denorm,
            lat=bt.lat,
            center=bool(center),
            reduction="mean",
        )
    return out


@torch.no_grad()
def compute_batch_metrics_per_channel(bt: BatchTensors, *, center: bool = True) -> Dict[str, torch.Tensor]:
    """Compute per-channel RMSE/MAE/Bias/CRPS and ACC when available."""

    pred_denorm = denorm(bt.pred_norm, bt.mean, bt.std)
    target_denorm = denorm(bt.target_norm, bt.mean, bt.std)

    # [B,C,H,W] -> [C]
    rmse_ch = torch.sqrt(torch.mean((pred_denorm - target_denorm) ** 2, dim=(0, 2, 3)))
    mae_ch = torch.mean(torch.abs(pred_denorm - target_denorm), dim=(0, 2, 3))
    bias_ch = torch.mean(pred_denorm - target_denorm, dim=(0, 2, 3))
    if bt.pred_ensemble_norm is not None:
        ensemble_denorm = _denorm_ensemble(bt.pred_ensemble_norm, bt.mean, bt.std)
        crps_ch = _crps_ensemble(ensemble_denorm, target_denorm, reduction="channel_mean")
    else:
        # Deterministic CRPS = MAE for a single-member predictive distribution.
        crps_ch = mae_ch

    out: Dict[str, torch.Tensor] = {
        "rmse_ch": rmse_ch,
        "mae_ch": mae_ch,
        "bias_ch": bias_ch,
        "crps_ch": crps_ch,
    }

    if bt.clim_denorm is not None:
        acc_ch = _acc(
            pred_denorm,
            target_denorm,
            bt.clim_denorm,
            lat=bt.lat,
            center=bool(center),
            reduction="channel_mean",
        )
        out["acc_ch"] = acc_ch
    return out


def extract_lat(batch: Dict) -> Optional[torch.Tensor]:
    meta = batch.get("meta", None)
    if meta is None:
        return None
    lat = meta.get("lat", None)
    if lat is None:
        return None
    # lat can be [H], [H,W], [B,H], [B,H,W]
    if isinstance(lat, torch.Tensor):
        if lat.ndim == 1:
            return lat
        if lat.ndim == 2:
            # either [B,H] (collated from vector) or [H,W]
            B = int(batch.get("x_ctx", torch.empty(0)).shape[0]) if isinstance(batch.get("x_ctx", None), torch.Tensor) else None
            if B is not None and lat.shape[0] == B:
                # [B,H]
                return lat
            # treat as [H,W]
            return lat[:, 0]
        if lat.ndim == 3:
            # [B,H,W]
            return lat[..., 0]
    return None


def default_clim(batch: Dict) -> Optional[torch.Tensor]:
    clim = batch.get("clim", None)
    if clim is None:
        return None
    if not isinstance(clim, torch.Tensor):
        return None
    return clim
