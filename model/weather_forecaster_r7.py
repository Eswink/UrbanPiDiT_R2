from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional
import torch
from torch import nn
from .coarse_encoder import CoarseEncoder
from .coarse_forecast import CoarseForecastHead, LeadTimeEmbedding


@dataclass
class R7ForecastOutput:
    forecast: torch.Tensor
    tendency: torch.Tensor
    context_tokens: torch.Tensor
    token_hw: tuple[int,int]


class NativeAtmosForecaster(nn.Module):
    """R7.1 forecast-native regional atmospheric transition model.

    This class intentionally stays separate from the V6 UrbanPiDiTR2 path.
    Later R7 recursive/process reasoning modules build on its context + draft
    forecast without requiring an externally supplied urban_baseline.
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
    ):
        super().__init__()
        self.in_channels=int(in_channels)
        self.history_steps=int(history_steps)
        self.out_channels=int(out_channels or in_channels)
        if self.out_channels>self.in_channels:
            raise ValueError(
                'R7.1 默认 persistence base 要求 out_channels <= in_channels'
            )
        self.default_lead_hours=float(default_lead_hours)

        self.encoder=CoarseEncoder(
            in_channels=self.in_channels,
            history_steps=self.history_steps,
            dim=dim,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            window_size=window_size,
            dropout=dropout,
            use_checkpointing=activation_checkpointing,
            periodic_width=periodic_width,
        )
        self.lead_time=LeadTimeEmbedding(dim)
        self.head=CoarseForecastHead(
            dim=dim,
            out_channels=self.out_channels,
            patch_size=patch_size,
        )

    def forward(
        self,
        batch:Mapping[str,torch.Tensor],
    )->R7ForecastOutput:
        history=batch['coarse_history']
        if history.ndim!=5:
            raise ValueError('coarse_history 必须为 [B,T,C,H,W]')
        B,T,C,H,W=history.shape
        tokens,token_hw=self.encoder(history)

        lead=self.lead_time(
            batch.get('lead_time_hours'),
            batch=B,
            device=tokens.device,
            dtype=tokens.dtype,
            default_hours=self.default_lead_hours,
        )
        context=tokens+lead[:,None,:]

        base=batch.get('atmos_baseline')
        if base is None:
            base=history[:,-1,:self.out_channels]
        forecast,tendency=self.head(
            context,
            token_hw,
            (H,W),
            base,
        )
        return R7ForecastOutput(
            forecast=forecast,
            tendency=tendency,
            context_tokens=context,
            token_hw=token_hw,
        )
