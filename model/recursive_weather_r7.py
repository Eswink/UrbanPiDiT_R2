from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .weather_forecaster_r7 import NativeAtmosForecaster
from .coarse_forecast import CoarseForecastHead
from .layers.sdpa import SDPAttention, CrossBlock, FeedForward


class DraftTokenEncoder(nn.Module):
    def __init__(self,in_channels:int,dim:int,patch_size:int):
        super().__init__()
        self.patch=nn.Conv2d(
            in_channels,dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm=nn.LayerNorm(dim)

    def forward(self,x:torch.Tensor):
        z=self.patch(x)
        hw=z.shape[-2:]
        tokens=z.flatten(2).transpose(1,2)
        return self.norm(tokens),hw


class GenericRecursiveCell(nn.Module):
    """Parameter-shared latent update with no meteorological process semantics."""

    def __init__(
        self,
        dim:int,
        heads:int,
        mlp_ratio:float=3.0,
        dropout:float=0.0,
    ):
        super().__init__()
        self.n1=nn.LayerNorm(dim)
        self.self_attn=SDPAttention(dim,heads,dropout)
        self.cross=CrossBlock(dim,heads,mlp_ratio,dropout)
        self.n2=nn.LayerNorm(dim)
        self.ff=FeedForward(dim,mlp_ratio,dropout)

    def forward(self,z:torch.Tensor,context:torch.Tensor):
        z=z+self.self_attn(self.n1(z))
        z=self.cross(z,context)
        return z+self.ff(self.n2(z))


@dataclass
class RecursiveForecastOutput:
    forecast: torch.Tensor
    initial_forecast: torch.Tensor
    draft_forecasts: torch.Tensor
    final_correction: torch.Tensor
    latent_state: torch.Tensor
    context_tokens: torch.Tensor
    token_hw: tuple[int,int]
    reasoning_steps: int


class GenericRecursiveWeatherForecaster(nn.Module):
    """R7.2 TRM-like weather baseline.

    It repeatedly refines a forecast draft using a generic latent state z.
    There are deliberately no anchored meteorological Process State tokens.
    R7.3 must outperform this model at a comparable parameter/compute budget.
    """

    def __init__(
        self,
        in_channels:int,
        history_steps:int=2,
        out_channels:Optional[int]=None,
        dim:int=128,
        patch_size:int=2,
        depth:int=4,
        heads:int=4,
        window_size:int=8,
        dropout:float=0.0,
        activation_checkpointing:bool=False,
        periodic_width:bool=False,
        default_lead_hours:float=6.0,
        latent_tokens:int=16,
        default_reasoning_steps:int=4,
        detach_between_steps:bool=False,
    ):
        super().__init__()
        self.out_channels=int(out_channels or in_channels)
        self.dim=int(dim)
        self.patch_size=int(patch_size)
        self.default_reasoning_steps=int(default_reasoning_steps)
        self.detach_between_steps=bool(detach_between_steps)
        self.activation_checkpointing=bool(activation_checkpointing)

        self.backbone=NativeAtmosForecaster(
            in_channels=in_channels,
            history_steps=history_steps,
            out_channels=self.out_channels,
            dim=dim,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            window_size=window_size,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
            periodic_width=periodic_width,
            default_lead_hours=default_lead_hours,
        )
        self.latent=nn.Parameter(
            torch.randn(1,int(latent_tokens),dim)*0.02
        )
        self.draft_encoder=DraftTokenEncoder(
            self.out_channels,dim,patch_size
        )
        self.cell=GenericRecursiveCell(
            dim,heads,mlp_ratio=3.0,dropout=dropout
        )
        self.latent_to_context=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,dim),
        )
        self.correction_head=CoarseForecastHead(
            dim=dim,
            out_channels=self.out_channels,
            patch_size=patch_size,
        )

    def _cell(self,z,context):
        if (
            self.activation_checkpointing
            and self.training
            and z.requires_grad
        ):
            return checkpoint(
                self.cell,z,context,use_reentrant=False
            )
        return self.cell(z,context)

    def forward(
        self,
        batch:Mapping[str,torch.Tensor],
        *,
        reasoning_steps:Optional[int]=None,
        detach_between_steps:Optional[bool]=None,
    )->RecursiveForecastOutput:
        base=self.backbone(batch)
        initial=base.forecast
        draft=initial
        context=base.context_tokens
        token_hw=base.token_hw
        B=initial.shape[0]
        output_hw=initial.shape[-2:]

        steps=(
            self.default_reasoning_steps
            if reasoning_steps is None
            else int(reasoning_steps)
        )
        if steps<0:
            raise ValueError('reasoning_steps 必须 >= 0')
        detach_flag=(
            self.detach_between_steps
            if detach_between_steps is None
            else bool(detach_between_steps)
        )

        z=self.latent.expand(B,-1,-1)
        drafts=[draft]
        final_correction=torch.zeros_like(draft)

        for step in range(steps):
            draft_tokens,draft_hw=self.draft_encoder(draft)
            if tuple(draft_hw)!=tuple(token_hw):
                raise ValueError(
                    f'draft token grid {draft_hw} != context grid {token_hw}'
                )
            recurrent_context=torch.cat(
                [context,draft_tokens],dim=1
            )
            z=self._cell(z,recurrent_context)
            z_summary=self.latent_to_context(z.mean(dim=1))
            conditioned=context+z_summary[:,None,:]
            draft,final_correction=self.correction_head(
                conditioned,
                token_hw,
                output_hw,
                draft,
            )
            drafts.append(draft)

            # TRM-like truncated supervision: each draft keeps its own graph,
            # while the next recurrent step may consume detached state/draft.
            if detach_flag and step < steps-1:
                z=z.detach()
                draft=draft.detach()

        return RecursiveForecastOutput(
            forecast=draft,
            initial_forecast=initial,
            draft_forecasts=torch.stack(drafts,dim=1),
            final_correction=final_correction,
            latent_state=z,
            context_tokens=context,
            token_hw=token_hw,
            reasoning_steps=steps,
        )
