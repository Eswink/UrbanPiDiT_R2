from __future__ import annotations
from typing import Optional
import torch


def latitude_weighted_mse(
    prediction:torch.Tensor,
    target:torch.Tensor,
    latitude:Optional[torch.Tensor]=None,
)->torch.Tensor:
    """Cos(latitude)-weighted MSE for regular lat/lon regional grids."""
    error=(prediction-target).square()
    if latitude is None:
        return error.mean()

    lat=torch.as_tensor(
        latitude,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    B=prediction.shape[0]
    H=prediction.shape[-2]
    if lat.ndim==1:
        if lat.numel()!=H:
            raise ValueError('latitude length must match prediction H')
        lat=lat.view(1,H).expand(B,H)
    elif lat.ndim==2:
        if lat.shape!=(B,H):
            raise ValueError('batched latitude must have shape [B,H]')
    else:
        raise ValueError('latitude must be [H] or [B,H]')

    weight=torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    weight=weight/weight.mean(dim=1,keepdim=True).clamp_min(1e-6)
    weight=weight[:,None,:,None]
    return (error*weight).mean()
