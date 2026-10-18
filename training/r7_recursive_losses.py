from __future__ import annotations
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


def deep_supervised_forecast_mse(
    drafts:torch.Tensor,
    target:torch.Tensor,
    latitude:Optional[torch.Tensor]=None,
    *,
    final_weight:float=2.0,
    mask:Optional[torch.Tensor]=None,
    draft_supervision:str=DRAFT_SUPERVISION_DEEP,
)->torch.Tensor:
    """Deep supervision over [B,S,C,H,W] forecast drafts.

    ``mask`` restricts every draft's supervision to the same spatial region
    (see :func:`latitude_weighted_mse`); ``None`` keeps the original objective.

    ``draft_supervision`` selects how the K-axis drafts are organized. The default
    ``deep`` is the registered objective and keeps the original arithmetic exactly.
    ``final_only`` drops every non-final draft's direct target term and scores only
    ``drafts[:, -1]`` with the same masked latitude-weighted MSE and weight 1.0; the
    model forward, including all K+1 drafts and their gradient graph, is untouched.
    """
    if drafts.ndim!=5:
        raise ValueError('drafts must be [B,S,C,H,W]')
    steps=drafts.shape[1]
    if steps<1:
        raise ValueError('drafts must contain at least one forecast')
    if draft_supervision not in DRAFT_SUPERVISION_MODES:
        raise ValueError(f'draft_supervision must be one of {DRAFT_SUPERVISION_MODES}, '
                         f'got {draft_supervision!r}')
    if draft_supervision==DRAFT_SUPERVISION_FINAL_ONLY:
        return latitude_weighted_mse(
            drafts[:,-1],target,latitude,mask=mask
        )
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
    return sum(w*l for w,l in zip(weights,losses))
