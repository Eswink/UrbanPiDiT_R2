from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping,Optional
import torch
from torch import nn
from .coarse_encoder import CoarseEncoder
from .coarse_forecast import CoarseForecastHead,LeadTimeEmbedding


@dataclass
class R7ForecastOutput:
    forecast:torch.Tensor
    tendency:torch.Tensor
    context_tokens:torch.Tensor
    token_hw:tuple[int,int]


class NativeAtmosForecaster(nn.Module):
    """Native-grid atmospheric transitions; ceil patching preserves finite edges."""
    def __init__(self,in_channels:int,history_steps:int=2,out_channels:Optional[int]=None,
                 dim:int=128,patch_size:int=2,depth:int=4,heads:int=4,window_size:int=8,
                 dropout:float=0.,activation_checkpointing:bool=False,periodic_width:bool=False,
                 default_lead_hours:float=6.):
        super().__init__()
        self.in_channels=int(in_channels)
        self.history_steps=int(history_steps)
        self.out_channels=int(out_channels or in_channels)
        if self.out_channels>self.in_channels:
            raise ValueError('R7.1 默认 persistence base 要求 out_channels <= in_channels')
        self.default_lead_hours=float(default_lead_hours)
        self.encoder=CoarseEncoder(self.in_channels,self.history_steps,dim,patch_size,depth,
            heads,window_size,dropout,activation_checkpointing,periodic_width,pad_to_patch=True)
        self.lead_time=LeadTimeEmbedding(dim)
        self.head=CoarseForecastHead(dim,self.out_channels,patch_size)

    def forward(self,batch:Mapping[str,torch.Tensor])->R7ForecastOutput:
        history=batch['coarse_history']
        if history.ndim!=5:
            raise ValueError('coarse_history 必须为 [B,T,C,H,W]')
        B,T,C,H,W=history.shape
        tokens,token_hw=self.encoder(history)
        lead=self.lead_time(batch.get('lead_time_hours'),batch=B,device=tokens.device,
            dtype=tokens.dtype,default_hours=self.default_lead_hours)
        context=tokens+lead[:,None,:]
        base=batch.get('atmos_baseline')
        if base is None:
            base=history[:,-1,:self.out_channels]
        forecast,tendency=self.head(context,token_hw,(H,W),base)
        return R7ForecastOutput(forecast,tendency,context,token_hw)
