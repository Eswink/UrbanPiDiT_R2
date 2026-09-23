from __future__ import annotations
from typing import Optional
import torch
from .r7_losses import latitude_weighted_mse


def deep_supervised_forecast_mse(
    drafts:torch.Tensor,
    target:torch.Tensor,
    latitude:Optional[torch.Tensor]=None,
    *,
    final_weight:float=2.0,
)->torch.Tensor:
    """Deep supervision over [B,S,C,H,W] forecast drafts."""
    if drafts.ndim!=5:
        raise ValueError('drafts must be [B,S,C,H,W]')
    steps=drafts.shape[1]
    if steps<1:
        raise ValueError('drafts must contain at least one forecast')
    weights=torch.linspace(
        1.0,float(final_weight),steps,
        device=drafts.device,dtype=drafts.dtype,
    )
    weights=weights/weights.sum()
    losses=[
        latitude_weighted_mse(
            drafts[:,i],target,latitude
        )
        for i in range(steps)
    ]
    return sum(w*l for w,l in zip(weights,losses))
