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
from .local_solver_state_r7 import ROLE_INITIAL_SCALE
from .spacetime_conditioning_r7 import isolated_stream, require_field_mode


def recurrent_key(context, draft_tokens=None, *, role_context=None, role_draft=None):
    """The one place the recurrent cell's key/value is assembled.

    Before this helper the concatenation ``torch.cat([context, draft_tokens], dim=1)``
    was written out at every call site, and none of them marked which half was the
    context and which was the draft - the gap ``docs/R7_MAIN_MODEL_V2_DESIGN.md``
    section 3.2 records. With ``source_role_markers`` on, each half is shifted by
    its own learned vector first, so the cell *can* tell them apart. With it off
    the original expression is returned, and an unmarked key is bit for bit what
    it always was rather than a numerically similar recomputation.
    """
    if draft_tokens is None:
        if role_draft is not None:
            raise ValueError("a draft role needs draft tokens to mark")
        return context if role_context is None else context + role_context
    if (role_context is None) != (role_draft is None):
        raise ValueError("source roles are declared for both halves or for neither")
    if role_context is None:
        return torch.cat([context, draft_tokens], dim=1)
    if role_context.shape[1] != 1 or role_draft.shape[1] != 1:
        raise ValueError("a source role is one vector per half, shaped [1,1,D]")
    if role_context.shape[-1] != context.shape[-1] or role_draft.shape[-1] != context.shape[-1]:
        raise ValueError("source roles must have the context's feature dimension")
    return torch.cat([context + role_context, draft_tokens + role_draft], dim=1)


def declared_source_roles(model):
    """``(role_context, role_draft)`` for a model, or ``(None, None)``."""
    if not model.source_role_markers:
        return None, None
    return model.role_context, model.role_draft


def solver_conditioning(context, summary, draft_tokens=None, *, spatial_feedback=False):
    """Optional aligned draft evidence for S(C, P, E(Y)); no added parameters.

    ``summary`` is either the pooled ``[B,D]`` vector the pre-RW-A model
    broadcast to every position, or the per-position ``[B,N,D]`` read produced by
    the positional process readout. Both are **added**; the branch is on the rank
    of the summary and nothing else about the update changes.

    False preserves the original pooled-summary equation exactly. True adds
    the already-encoded draft at the same patch positions, without global
    attention or access to target fields.
    """
    if type(spatial_feedback) is not bool:
        raise ValueError("spatial_feedback must be boolean")
    if context.ndim != 3:
        raise ValueError("solver context must be [B,N,D]")
    if summary.ndim == 2:
        if summary.shape != (context.shape[0], context.shape[2]):
            raise ValueError("solver context/summary shapes must be [B,N,D]/[B,D]")
        conditioned = context + summary[:, None, :]
    elif summary.ndim == 3:
        if summary.shape != context.shape:
            raise ValueError("per-position solver summary must match the [B,N,D] context exactly")
        conditioned = context + summary
    else:
        raise ValueError("solver summary must be pooled [B,D] or positional [B,N,D]")
    if summary.device != context.device:
        raise ValueError("solver summary/context devices differ")
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
                 detach_between_steps:bool=False,spatial_solver_feedback:bool=False,
                 spacetime_inputs:bool=False,spacetime_field_mode:str='fields',
                 source_role_markers:bool=False):
        super().__init__()
        for value,name in ((spatial_solver_feedback,'spatial_solver_feedback'),
                           (spacetime_inputs,'spacetime_inputs'),
                           (source_role_markers,'source_role_markers')):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        self.spatial_solver_feedback=spatial_solver_feedback
        self.source_role_markers=source_role_markers
        # Exposed, not just forwarded: the rollout asks the model which lead
        # convention it was configured for.
        self.spacetime_inputs=spacetime_inputs
        self.spacetime_field_mode=require_field_mode(spacetime_field_mode)
        if not self.spacetime_inputs and self.spacetime_field_mode!='fields':
            raise ValueError(
                f"spacetime_field_mode={self.spacetime_field_mode!r} needs "
                "spacetime_inputs=True; with the pathway off it would be silently ignored")
        self.out_channels=int(out_channels or in_channels)
        self.dim=int(dim)
        self.patch_size=int(patch_size)
        self.default_reasoning_steps=int(default_reasoning_steps)
        self.detach_between_steps=bool(detach_between_steps)
        self.activation_checkpointing=bool(activation_checkpointing)
        self.backbone=NativeAtmosForecaster(in_channels,history_steps,self.out_channels,dim,patch_size,
            depth,heads,window_size,dropout,activation_checkpointing,periodic_width,default_lead_hours,
            spacetime_inputs=spacetime_inputs,spacetime_field_mode=self.spacetime_field_mode)
        self.latent=nn.Parameter(torch.randn(1,int(latent_tokens),dim)*.02)
        self.draft_encoder=DraftTokenEncoder(self.out_channels,dim,patch_size)
        self.cell=GenericRecursiveCell(dim,heads,mlp_ratio=3.,dropout=dropout)
        self.latent_to_context=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.correction_head=CoarseForecastHead(dim,self.out_channels,patch_size)
        # Built last and under a rewound stream, for the same reason the space-time
        # pathway is: declaring source roles must not move a single weight the
        # model would have had without them, so the two arms differ by the roles
        # and not by a shifted initialization.
        if self.source_role_markers:
            with isolated_stream():
                self.role_context=nn.Parameter(torch.randn(1,1,dim)*ROLE_INITIAL_SCALE)
                self.role_draft=nn.Parameter(torch.randn(1,1,dim)*ROLE_INITIAL_SCALE)

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
        role_context,role_draft=declared_source_roles(self)
        drafts=[draft]
        final_correction=torch.zeros_like(draft)
        for step in range(steps):
            draft_tokens,draft_hw=self.draft_encoder(draft)
            if tuple(draft_hw)!=tuple(token_hw):
                raise ValueError('draft/context token grid mismatch')
            z=self._cell(z,recurrent_key(context,draft_tokens,
                role_context=role_context,role_draft=role_draft))
            summary=self.latent_to_context(z.mean(dim=1))
            conditioned=solver_conditioning(context,summary,draft_tokens,
                spatial_feedback=self.spatial_solver_feedback)
            draft,final_correction=self.correction_head(conditioned,token_hw,initial.shape[-2:],draft)
            drafts.append(draft)
            if detach_flag and step<steps-1:
                z,draft=z.detach(),draft.detach()
        return RecursiveForecastOutput(draft,initial,torch.stack(drafts,dim=1),final_correction,z,context,token_hw,steps)
