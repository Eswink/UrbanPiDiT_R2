from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping,Optional
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .weather_forecaster_r7 import NativeAtmosForecaster
from .coarse_forecast import CoarseForecastHead
from .layers.sdpa import SDPAttention,CrossBlock,FeedForward
from .layers.patch_grid import pad_patch_grid


def solver_conditioning(context, summary, draft_tokens=None, *, spatial_feedback=False):
    """Optional aligned draft evidence for S(C, P, E(Y)); no added parameters.

    False preserves the original pooled-summary equation exactly. True adds
    the already-encoded draft at the same patch positions, without global
    attention or access to target fields.
    """
    if type(spatial_feedback) is not bool:
        raise ValueError("spatial_feedback must be boolean")
    if context.ndim != 3 or summary.shape != (context.shape[0], context.shape[2]):
        raise ValueError("solver context/summary shapes must be [B,N,D]/[B,D]")
    if summary.device != context.device:
        raise ValueError("solver summary/context devices differ")
    conditioned = context + summary[:, None, :]
    if spatial_feedback:
        if draft_tokens is None or draft_tokens.shape != context.shape or draft_tokens.device != context.device:
            raise ValueError("aligned [B,N,D] draft tokens required for spatial solver feedback")
        conditioned = conditioned + draft_tokens
    return conditioned


class DraftTokenEncoder(nn.Module):
    def __init__(self,in_channels:int,dim:int,patch_size:int):
        super().__init__()
        self.patch_size=patch_size
        self.patch=nn.Conv2d(in_channels,dim,kernel_size=patch_size,stride=patch_size)
        self.norm=nn.LayerNorm(dim)

    def forward(self,x:torch.Tensor):
        z=self.patch(pad_patch_grid(x,self.patch_size))
        hw=z.shape[-2:]
        return self.norm(z.flatten(2).transpose(1,2)),hw


class GenericRecursiveCell(nn.Module):
    def __init__(self,dim:int,heads:int,mlp_ratio:float=3.,dropout:float=0.):
        super().__init__()
        self.n1=nn.LayerNorm(dim)
        self.self_attn=SDPAttention(dim,heads,dropout)
        self.cross=CrossBlock(dim,heads,mlp_ratio,dropout)
        self.n2=nn.LayerNorm(dim)
        self.ff=FeedForward(dim,mlp_ratio,dropout)

    def forward(self,z,context):
        z=z+self.self_attn(self.n1(z))
        z=self.cross(z,context)
        return z+self.ff(self.n2(z))


@dataclass
class RecursiveForecastOutput:
    forecast:torch.Tensor
    initial_forecast:torch.Tensor
    draft_forecasts:torch.Tensor
    final_correction:torch.Tensor
    latent_state:torch.Tensor
    context_tokens:torch.Tensor
    token_hw:tuple[int,int]
    reasoning_steps:int


class GenericRecursiveWeatherForecaster(nn.Module):
    """Generic parameter-shared recursive baseline without process semantics."""
    def __init__(self,in_channels:int,history_steps:int=2,out_channels:Optional[int]=None,
                 dim:int=128,patch_size:int=2,depth:int=4,heads:int=4,window_size:int=8,
                 dropout:float=0.,activation_checkpointing:bool=False,periodic_width:bool=False,
                 default_lead_hours:float=6.,latent_tokens:int=16,default_reasoning_steps:int=4,
                 detach_between_steps:bool=False,spatial_solver_feedback:bool=False):
        super().__init__()
        if type(spatial_solver_feedback) is not bool:
            raise ValueError("spatial_solver_feedback must be boolean")
        self.spatial_solver_feedback=spatial_solver_feedback
        self.out_channels=int(out_channels or in_channels)
        self.dim=int(dim)
        self.patch_size=int(patch_size)
        self.default_reasoning_steps=int(default_reasoning_steps)
        self.detach_between_steps=bool(detach_between_steps)
        self.activation_checkpointing=bool(activation_checkpointing)
        self.backbone=NativeAtmosForecaster(in_channels,history_steps,self.out_channels,dim,patch_size,
            depth,heads,window_size,dropout,activation_checkpointing,periodic_width,default_lead_hours)
        self.latent=nn.Parameter(torch.randn(1,int(latent_tokens),dim)*.02)
        self.draft_encoder=DraftTokenEncoder(self.out_channels,dim,patch_size)
        self.cell=GenericRecursiveCell(dim,heads,mlp_ratio=3.,dropout=dropout)
        self.latent_to_context=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.correction_head=CoarseForecastHead(dim,self.out_channels,patch_size)

    def _cell(self,z,context):
        if self.activation_checkpointing and self.training and z.requires_grad:
            return checkpoint(self.cell,z,context,use_reentrant=False)
        return self.cell(z,context)

    def forward(self,batch:Mapping[str,torch.Tensor],*,reasoning_steps:Optional[int]=None,
                detach_between_steps:Optional[bool]=None)->RecursiveForecastOutput:
        base=self.backbone(batch)
        initial=draft=base.forecast
        context,token_hw=base.context_tokens,base.token_hw
        B=initial.shape[0]
        steps=self.default_reasoning_steps if reasoning_steps is None else int(reasoning_steps)
        if steps<0:
            raise ValueError('reasoning_steps 必须 >= 0')
        detach_flag=self.detach_between_steps if detach_between_steps is None else bool(detach_between_steps)
        z=self.latent.expand(B,-1,-1)
        drafts=[draft]
        final_correction=torch.zeros_like(draft)
        for step in range(steps):
            draft_tokens,draft_hw=self.draft_encoder(draft)
            if tuple(draft_hw)!=tuple(token_hw):
                raise ValueError('draft/context token grid mismatch')
            z=self._cell(z,torch.cat([context,draft_tokens],dim=1))
            summary=self.latent_to_context(z.mean(dim=1))
            conditioned=solver_conditioning(context,summary,draft_tokens,
                spatial_feedback=self.spatial_solver_feedback)
            draft,final_correction=self.correction_head(conditioned,token_hw,initial.shape[-2:],draft)
            drafts.append(draft)
            if detach_flag and step<steps-1:
                z,draft=z.detach(),draft.detach()
        return RecursiveForecastOutput(draft,initial,torch.stack(drafts,dim=1),final_correction,z,context,token_hw,steps)
