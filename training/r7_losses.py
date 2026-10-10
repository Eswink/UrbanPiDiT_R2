from __future__ import annotations
from typing import Optional
import torch


def latitude_weighted_mse(
    prediction:torch.Tensor,
    target:torch.Tensor,
    latitude:Optional[torch.Tensor]=None,
    *,
    mask:Optional[torch.Tensor]=None,
)->torch.Tensor:
    """Cos(latitude)-weighted MSE for regular lat/lon regional grids.

    ``mask`` optionally restricts supervision to a spatial region: it must be
    broadcastable to the trailing ``[H,W]`` grid and the result is then the
    mask-and-latitude weighted mean over the selected cells instead of the mean
    over every cell. ``mask=None`` (the default) is the original arithmetic,
    unchanged bit for bit.
    """
    error=(prediction-target).square()
    if latitude is None and mask is None:
        return error.mean()

    if latitude is None:
        weight=torch.ones_like(error)
    else:
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

    if mask is None:
        return (error*weight).mean()

    selected=torch.as_tensor(mask,device=prediction.device,dtype=prediction.dtype)
    if selected.shape!=error.shape[-2:] and selected.shape!=(1,)*2+error.shape[-2:]:
        raise ValueError('mask must be [H,W] matching the prediction grid')
    selected=selected.reshape((1,1)+tuple(error.shape[-2:]))
    if not torch.isfinite(selected).all() or (selected<0).any():
        raise ValueError('mask must be finite and nonnegative')
    total=(selected>0).sum()
    if int(total)==0:
        raise ValueError('mask selects no cell; a masked objective needs a nonempty region')
    combined=weight*selected
    denominator=combined.expand_as(error).sum()
    if not torch.isfinite(denominator) or float(denominator)<=0:
        raise ValueError('masked objective has a nonpositive weight denominator')
    return (error*combined).sum()/denominator
