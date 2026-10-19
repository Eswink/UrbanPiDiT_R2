from __future__ import annotations
import math
from numbers import Real
from typing import Optional
import torch
from .r7_losses import latitude_weighted_mse

#: The registered supervision organization: initial plus every K draft, weights
#: ``linspace(1, final_weight, K+1)`` normalized to sum one. This is the default and
#: its arithmetic is unchanged bit for bit.
DRAFT_SUPERVISION_DEEP = "deep"
#: Final-draft-only organization: only ``drafts[:, -1]`` carries a direct target term,
#: with the same raw latitude-weighted MSE (mask included) and weight 1.0.
DRAFT_SUPERVISION_FINAL_ONLY = "final_only"
DRAFT_SUPERVISION_MODES = (DRAFT_SUPERVISION_DEEP, DRAFT_SUPERVISION_FINAL_ONLY)
#: ``gradient_lambda=0.0`` is a separate Python branch: the registered arithmetic runs
#: untouched and the spatial-gradient penalty is never evaluated.
ZERO_GRADIENT_LAMBDA = 0.0


def require_gradient_lambda(value) -> float:
    """A frozen design coefficient: finite, nonnegative, no bool/string coercion."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("gradient_lambda must be a finite nonnegative real")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("gradient_lambda must be a finite nonnegative real") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError("gradient_lambda must be a finite nonnegative real")
    return number


def spatial_gradient_mse(
    prediction:torch.Tensor,
    target:torch.Tensor,
    latitude:Optional[torch.Tensor]=None,
    *,
    mask:Optional[torch.Tensor]=None,
)->torch.Tensor:
    """Latitude-weighted MSE of forward differences over selected adjacent pairs.

    ``prediction`` and ``target`` are ``[B,C,H,W]``. With ``mask`` given, only pairs
    whose two endpoints both lie in the mask contribute: horizontal (south-north)
    pairs ``(h, h+1)`` and longitudinal pairs ``(w, w+1)``. The pair weight is the
    mean ``cos(lat)`` of the two rows for a horizontal pair and ``cos(lat)`` of the
    row for a longitudinal pair; the combined error is normalized by the sum of the
    selected pair weights over batch, pairs *and channels*, exactly like
    :func:`latitude_weighted_mse` (whose combined weight is expanded over channels).
    Thus 17 identical channels give the same L_grad as one channel. ``mask=None``
    selects every adjacent pair.
    """
    if prediction.ndim!=4 or target.shape!=prediction.shape:
        raise ValueError('prediction and target must be matching [B,C,H,W] tensors')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('prediction and target must be floating point')
    if prediction.device!=target.device:
        raise ValueError('prediction and target must share a device')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('prediction and target must be finite')
    batch,_,height,width=prediction.shape
    dtype=prediction.dtype
    if mask is None:
        selected=torch.ones((height,width),dtype=torch.bool,device=prediction.device)
    else:
        supplied=torch.as_tensor(mask,device=prediction.device)
        if supplied.shape!=(height,width):
            raise ValueError('mask must be [H,W] matching the prediction grid')
        if not torch.isfinite(supplied).all() or (supplied<0).any():
            raise ValueError('mask must be finite and nonnegative')
        selected=supplied>0
    selected_h=selected[:-1,:]&selected[1:,:]
    selected_w=selected[:,:-1]&selected[:,1:]
    if int(selected_h.sum())+int(selected_w.sum())==0:
        raise ValueError('mask selects no adjacent pair')
    if latitude is None:
        cos_lat=torch.ones((batch,height),device=prediction.device,dtype=dtype)
    else:
        lat=torch.as_tensor(latitude,device=prediction.device,dtype=dtype)
        if lat.ndim==1:
            if lat.numel()!=height:
                raise ValueError('latitude length must match prediction H')
            lat=lat.view(1,height).expand(batch,height)
        elif lat.ndim==2:
            if lat.shape!=(batch,height):
                raise ValueError('batched latitude must have shape [B,H]')
        else:
            raise ValueError('latitude must be [H] or [B,H]')
        cos_lat=torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    # Mean cos of the two endpoints for a horizontal pair; row cos for a zonal pair.
    weight_h=0.5*(cos_lat[:,:-1]+cos_lat[:,1:])
    scale_h=weight_h[:,None,:,None]*selected_h.to(dtype)[None,None,:,:]
    scale_w=cos_lat[:,None,:,None]*selected_w.to(dtype)[None,None,:,:]
    delta_h=(prediction[:,:,1:,:]-prediction[:,:,:-1,:])-(target[:,:,1:,:]-target[:,:,:-1,:])
    delta_w=(prediction[:,:,:,1:]-prediction[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1])
    numerator=(delta_h.square()*scale_h).sum()+(delta_w.square()*scale_w).sum()
    # The channel count enters the denominator just as combined.expand_as(error)
    # does in latitude_weighted_mse, so L_grad is a mean over channels and pairs.
    denominator=(scale_h.sum()+scale_w.sum())*prediction.shape[1]
    if not torch.isfinite(denominator) or float(denominator)<=0:
        raise ValueError('gradient objective has a nonpositive weight denominator')
    return numerator/denominator


def _draft_gradient_penalty(prediction, target, latitude, mask, coefficient):
    return coefficient*spatial_gradient_mse(prediction,target,latitude,mask=mask)


def deep_supervised_forecast_mse(
    drafts:torch.Tensor,
    target:torch.Tensor,
    latitude:Optional[torch.Tensor]=None,
    *,
    final_weight:float=2.0,
    mask:Optional[torch.Tensor]=None,
    draft_supervision:str=DRAFT_SUPERVISION_DEEP,
    gradient_lambda:float=ZERO_GRADIENT_LAMBDA,
)->torch.Tensor:
    """Deep supervision over [B,S,C,H,W] forecast drafts.

    ``mask`` restricts every draft's supervision to the same spatial region
    (see :func:`latitude_weighted_mse`); ``None`` keeps the original objective.

    ``draft_supervision`` selects how the K-axis drafts are organized. The default
    ``deep`` is the registered objective and keeps the original arithmetic exactly.
    ``final_only`` drops every non-final draft's direct target term and scores only
    ``drafts[:, -1]`` with the same masked latitude-weighted MSE and weight 1.0; the
    model forward, including all K+1 drafts and their gradient graph, is untouched.

    ``gradient_lambda`` optionally adds ``lambda * spatial_gradient_mse`` to each
    draft's own MSE (the K-axis weights are applied afterwards). The default ``0.0``
    takes a separate branch that reproduces the registered loss bit for bit and never
    evaluates the spatial gradient.
    """
    if drafts.ndim!=5:
        raise ValueError('drafts must be [B,S,C,H,W]')
    steps=drafts.shape[1]
    if steps<1:
        raise ValueError('drafts must contain at least one forecast')
    if draft_supervision not in DRAFT_SUPERVISION_MODES:
        raise ValueError(f'draft_supervision must be one of {DRAFT_SUPERVISION_MODES}, '
                         f'got {draft_supervision!r}')
    gradient_lambda=require_gradient_lambda(gradient_lambda)
    if draft_supervision==DRAFT_SUPERVISION_FINAL_ONLY:
        loss=latitude_weighted_mse(
            drafts[:,-1],target,latitude,mask=mask
        )
        if gradient_lambda==ZERO_GRADIENT_LAMBDA:
            return loss
        return loss+_draft_gradient_penalty(drafts[:,-1],target,latitude,mask,gradient_lambda)
    weights=torch.linspace(
        1.0,float(final_weight),steps,
        device=drafts.device,dtype=drafts.dtype,
    )
    weights=weights/weights.sum()
    losses=[
        latitude_weighted_mse(
            drafts[:,i],target,latitude,mask=mask
        )
        for i in range(steps)
    ]
    if gradient_lambda==ZERO_GRADIENT_LAMBDA:
        return sum(w*l for w,l in zip(weights,losses))
    penalties=[
        spatial_gradient_mse(drafts[:,i],target,latitude,mask=mask)
        for i in range(steps)
    ]
    return sum(w*(l+gradient_lambda*p) for w,l,p in zip(weights,losses,penalties))
