from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping,Optional
import torch
from torch import nn
from .coarse_encoder import CoarseEncoder
from .coarse_forecast import CoarseForecastHead,LeadTimeEmbedding
from .known_context_r7 import KnownContextConditioning
from .spacetime_conditioning_r7 import (SpacetimeConditioning, isolated_stream,
    require_field_mode)


@dataclass
class R7ForecastOutput:
    forecast:torch.Tensor
    tendency:torch.Tensor
    context_tokens:torch.Tensor
    token_hw:tuple[int,int]
    # The known state at the current physical time that ``forecast`` was decoded
    # relative to. RW-B's proposal is anchored to this same tensor; re-deriving it
    # at a call site ("the last history step, unless a baseline was supplied")
    # would be a second definition of X_t, and a second definition of one quantity
    # is how the step implementations drifted apart before.
    base_state:torch.Tensor|None=None


class NativeAtmosForecaster(nn.Module):
    """Native-grid atmospheric transitions; ceil patching preserves finite edges."""
    def __init__(self,in_channels:int,history_steps:int=2,out_channels:Optional[int]=None,
                 dim:int=128,patch_size:int=2,depth:int=4,heads:int=4,window_size:int=8,
                 dropout:float=0.,activation_checkpointing:bool=False,periodic_width:bool=False,
                 default_lead_hours:float=6.,spacetime_inputs:bool=False,
                 spacetime_field_mode:str='fields',known_context_inputs:bool=False):
        super().__init__()
        if type(known_context_inputs) is not bool:
            raise ValueError('known_context_inputs must be boolean')
        if known_context_inputs and not spacetime_inputs:
            raise ValueError('known_context_inputs requires spacetime_inputs=True')
        self.known_context_inputs=known_context_inputs
        if type(spacetime_inputs) is not bool:
            raise ValueError('spacetime_inputs 必须是布尔开关')
        self.spacetime_inputs=spacetime_inputs
        # What the conditioning module is shown, not whether it exists: the
        # control arms of the capacity study keep the module and change only this.
        # A control mode on a switch that is off would be silently ignored, so it
        # is rejected here instead.
        self.spacetime_field_mode=require_field_mode(spacetime_field_mode)
        if not self.spacetime_inputs and self.spacetime_field_mode!='fields':
            raise ValueError(
                f"spacetime_field_mode={self.spacetime_field_mode!r} needs "
                "spacetime_inputs=True; with the pathway off it would be silently ignored")
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
        # Built last and under a rewound stream: with the switch off nothing is
        # constructed, and with it on every pre-existing parameter keeps the value
        # the same seed gives it without the pathway. That is what makes the two
        # arms of the study comparable - they differ by the pathway, not by a
        # shifted initialization.
        if self.spacetime_inputs:
            with isolated_stream():
                self.spacetime=SpacetimeConditioning(
                    dim,patch_size,periodic_width=periodic_width,
                    default_lead_hours=self.default_lead_hours,
                    field_mode=self.spacetime_field_mode)
        if self.known_context_inputs:
            with isolated_stream():
                self.known_context=KnownContextConditioning(
                    dim,self.history_steps,patch_size,periodic_width=periodic_width,
                    field_mode=self.spacetime_field_mode)

    def forward(self,batch:Mapping[str,torch.Tensor])->R7ForecastOutput:
        history=batch['coarse_history']
        if history.ndim!=5:
            raise ValueError('coarse_history 必须为 [B,T,C,H,W]')
        B,T,C,H,W=history.shape
        tokens,token_hw=self.encoder(history)
        lead=self.lead_time(batch.get('lead_time_hours'),batch=B,device=tokens.device,
            dtype=tokens.dtype,default_hours=self.default_lead_hours)
        context=tokens+lead[:,None,:]
        if self.spacetime_inputs:
            context=context+self.spacetime(
                batch,history=history,token_hw=token_hw).to(tokens.dtype)
        if self.known_context_inputs:
            context=context+self.known_context(
                batch,history=history,token_hw=token_hw).to(tokens.dtype)
        base=batch.get('atmos_baseline')
        if base is None:
            base=history[:,-1,:self.out_channels]
        forecast,tendency=self.head(context,token_hw,(H,W),base)
        return R7ForecastOutput(forecast,tendency,context,token_hw,base)
