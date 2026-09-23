import math
from typing import Optional

import torch


def acc(
    pred: torch.Tensor,
    target: torch.Tensor,
    climatology: torch.Tensor,
    *,
    lat: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    center: bool = True,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"pred/target 必须是 4D [B,C,H,W]，但得到 pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"pred 与 target shape 不一致：pred={tuple(pred.shape)} target={tuple(target.shape)}")

    B, C, H, W = pred.shape

    clim = climatology.to(device=pred.device, dtype=pred.dtype)
    if clim.ndim == 3:
        if clim.shape != (C, H, W):
            raise ValueError(f"climatology 形状不匹配：得到 {tuple(clim.shape)}，期望 {(C, H, W)}")
        clim = clim.unsqueeze(0)
    elif clim.ndim == 4:
        if clim.shape == (1, C, H, W):
            pass
        elif clim.shape == (B, C, H, W):
            pass
        else:
            raise ValueError(
                f"climatology 形状不匹配：得到 {tuple(clim.shape)}，期望 {(C, H, W)} 或 {(1, C, H, W)} 或 {(B, C, H, W)}"
            )
    else:
        raise ValueError(f"climatology 必须是 [C,H,W] 或 [1,C,H,W] 或 [B,C,H,W]，但得到 {tuple(clim.shape)}")

    if clim.shape[0] == 1 and B != 1:
        clim = clim.expand(B, -1, -1, -1)

    pred = pred - clim
    target = target - clim

    if weights is None:
        if lat is None:
            w = torch.ones((1, H, 1), dtype=pred.dtype, device=pred.device)
        else:
            lat_t = lat.to(device=pred.device, dtype=pred.dtype)
            if lat_t.ndim == 2:
                lat_t = lat_t[0]
            if lat_t.ndim != 1 or lat_t.numel() != H:
                raise ValueError(f"lat 必须是 [H] 或 [B,H]，但得到 {tuple(lat.shape)}，期望 H={H}")
            w_lat = torch.cos(lat_t * (math.pi / 180.0))
            w = (w_lat / w_lat.mean().clamp_min(eps)).view(1, H, 1)
        w = w.expand(B, H, W)
    else:
        w0 = weights.to(device=pred.device, dtype=pred.dtype)
        if w0.ndim == 2:
            if w0.shape != (H, W):
                raise ValueError(f"weights 形状不匹配：得到 {tuple(w0.shape)}，期望 {(H, W)}")
            w = w0.view(1, H, W).expand(B, H, W)
        elif w0.ndim == 4:
            if w0.shape[-2:] != (H, W):
                raise ValueError(f"weights 末两维不匹配：得到 {tuple(w0.shape)}，期望 (*,*,{H},{W})")
            w = w0[0, 0].view(1, H, W).expand(B, H, W)
        else:
            raise ValueError(f"weights 必须是 [H,W] 或 [1,1,H,W] 形式，但得到 {tuple(w0.shape)}")

    w = w.to(dtype=pred.dtype, device=pred.device)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))

    acc_c = []
    for i in range(C):
        p = pred[:, i]
        t = target[:, i]

        m = torch.isfinite(p) & torch.isfinite(t)
        ww = w * m.to(dtype=pred.dtype)

        p = torch.where(m, p, torch.zeros_like(p))
        t = torch.where(m, t, torch.zeros_like(t))

        if bool(center):
            p = p - (p.sum() / m.sum().clamp_min(1))
            t = t - (t.sum() / m.sum().clamp_min(1))

        num = torch.sum(ww * p * t)
        den = torch.sqrt(torch.sum(ww * p.square()) * torch.sum(ww * t.square()))
        # If either field has (near) zero variance under the mask/weights,
        # the correlation is undefined. Return NaN instead of an arbitrary value.
        v = num / den.clamp_min(eps)
        v = torch.where(den < eps, torch.full_like(v, float("nan")), v)
        acc_c.append(v)

    acc_c = torch.stack(acc_c, dim=0)

    if reduction in ("none", "channel_mean"):
        return acc_c
    if reduction in ("mean", "batch_mean"):
        return torch.nanmean(acc_c)

    raise ValueError(f"未知 reduction: {reduction}")
