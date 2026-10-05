from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from .layers.patch_grid import crop_native_grid


def resolve_lead_hours(lead_time_hours, *, batch:int, device:torch.device,
                       dtype:torch.dtype, default_hours:float=6.)->torch.Tensor:
    """Coerce a declared lead to a finite [B] FP32-style tensor of hours.

    The single place that decides what a missing or oddly-shaped lead means, so
    the lead embedding and the space-time phase cannot disagree about it. A
    missing lead becomes ``default_hours`` (the declared transition cadence);
    nothing here reads a clock.
    """
    if lead_time_hours is None:
        hours=torch.full((batch,1),default_hours,device=device,dtype=dtype)
    else:
        hours=torch.as_tensor(lead_time_hours,device=device,dtype=dtype)
        if hours.ndim==0:
            hours=hours.expand(batch).reshape(batch,1)
        else:
            hours=hours.reshape(batch,-1)[:,:1]
    if not torch.isfinite(hours).all():
        raise ValueError('lead_time_hours must be finite')
    return hours


class LeadTimeEmbedding(nn.Module):
    def __init__(self,dim:int,hidden:int|None=None):
        super().__init__()
        hidden=hidden or max(32,dim//2)
        self.net=nn.Sequential(nn.Linear(1,hidden),nn.SiLU(),nn.Linear(hidden,dim))

    def forward(self,lead_time_hours:Optional[torch.Tensor],*,batch:int,device:torch.device,
                dtype:torch.dtype,default_hours:float=6.)->torch.Tensor:
        hours=resolve_lead_hours(lead_time_hours,batch=batch,device=device,dtype=dtype,
            default_hours=default_hours)
        return self.net(hours/24.)


class CoarseForecastHead(nn.Module):
    """Decode full patch cells and crop; never resize a native-grid tendency.

    #78 R-A decode reparameterization. The decoder writes a tendency ``r_c`` in
    the existing normalized-state space; with ``normalized_change_scale`` the
    module instead writes ``Y = X_t + (d_c / s_c) * r_c``, where the ratio is a
    fixed, train-only quantity supplied at construction (the store's train-only
    change scale ``d_c`` over its train-only state scale ``s_c``). The loss is
    untouched: this changes the parameterization of the decoded increment, not
    its units or its channel weighting.

    The ratio is a non-persistent buffer: it is not a trained tensor and adds no
    state-dict key, so the identity and scaled arms share one tensor set and one
    parameter count - the contrast is the parameterization alone. With the mode
    left at ``identity`` nothing scaled is built or multiplied, so the default
    path is bitwise the pre-change implementation.
    """
    MODES = ("identity", "normalized_change_scale")

    def __init__(self,dim:int,out_channels:int,patch_size:int=2,hidden:int|None=None,
                 change_scale_mode:str="identity",change_scale_ratio=None):
        super().__init__()
        if change_scale_mode not in self.MODES:
            raise ValueError(f"change_scale_mode must be one of {self.MODES}")
        self.change_scale_mode=str(change_scale_mode)
        if self.change_scale_mode == "identity":
            if change_scale_ratio is not None:
                raise ValueError("change_scale_ratio with identity mode would be "
                                 "silently ignored; leave it unset or switch modes")
        elif change_scale_ratio is None:
            raise ValueError("normalized_change_scale requires the train-only "
                             "d_c / s_c vector")
        hidden=hidden or max(64,dim//2)
        self.out_channels=int(out_channels)
        self.patch_size=int(patch_size)
        self.norm=nn.LayerNorm(dim)
        self.decode=nn.Sequential(nn.ConvTranspose2d(dim,hidden,kernel_size=patch_size,stride=patch_size),
            nn.GELU(),nn.Conv2d(hidden,self.out_channels,3,padding=1))
        nn.init.normal_(self.decode[-1].weight,mean=0.,std=1e-3)
        nn.init.zeros_(self.decode[-1].bias)
        if self.change_scale_mode != "identity":
            value=torch.as_tensor(change_scale_ratio,dtype=torch.float32)
            if tuple(value.shape)!=(self.out_channels,) or not torch.isfinite(value).all() \
                    or bool((value<=0).any()):
                raise ValueError("change scale ratio must be a finite positive "
                                 f"[{self.out_channels}] vector")
            self.register_buffer("change_scale_ratio",value.clone(),persistent=False)

    def _scaled_tendency(self, tendency: torch.Tensor) -> torch.Tensor:
        if self.change_scale_mode == "identity":
            return tendency
        if tuple(self.change_scale_ratio.shape) != (tendency.shape[1],):
            raise ValueError("change scale ratio must match the decoded channels")
        return tendency * self.change_scale_ratio.to(tendency.dtype).view(1, -1, 1, 1)

    def forward(self,tokens:torch.Tensor,token_hw:tuple[int,int],output_hw:tuple[int,int],base_state:torch.Tensor):
        B,N,D=tokens.shape
        ht,wt=token_hw
        if N!=ht*wt:
            raise ValueError('CoarseForecastHead token 数与 token_hw 不一致')
        if base_state.shape!=(B,self.out_channels,*output_hw):
            raise ValueError('base_state must match output batch/channel/native spatial shape')
        z=self.norm(tokens).transpose(1,2).reshape(B,D,ht,wt)
        tendency=crop_native_grid(self.decode(z),output_hw,self.patch_size)
        tendency=self._scaled_tendency(tendency)
        return base_state+tendency,tendency
