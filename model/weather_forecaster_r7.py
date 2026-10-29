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
    # Physical X_t (or the historical explicit baseline), never the climatology
    # decode prior. Typed evidence and physical history retain this meaning.
    base_state:torch.Tensor|None=None
    climatology_anchor:torch.Tensor|None=None


class NativeAtmosForecaster(nn.Module):
    """Native-grid atmospheric transitions; ceil patching preserves finite edges."""
    #: The decoder's base semantics: the registered persistence anchor or a zero base.
    #: A zero base makes the decoded tendency the forecast (and every anchored proposal
    #: its own tendency); it is one switch, not a reparameterization of the head.
    DECODER_BASE_MODES = ("persistence", "zero")

    def __init__(self,in_channels:int,history_steps:int=2,out_channels:Optional[int]=None,
                 dim:int=128,patch_size:int=2,depth:int=4,heads:int=4,window_size:int=8,
                 dropout:float=0.,activation_checkpointing:bool=False,periodic_width:bool=False,
                 default_lead_hours:float=6.,spacetime_inputs:bool=False,
                 spacetime_field_mode:str='fields',known_context_inputs:bool=False,
                 change_scale_mode:str='identity',change_scale_ratio=None,
                 climatology_anchor_spec=None,anchor_output_prior:bool=True,
                 decoder_base_mode:str='persistence',tendency_input:bool=False,
                 static_conditioning:bool=False):
        super().__init__()
        if type(static_conditioning) is not bool:
            raise ValueError('static_conditioning must be boolean')
        self.static_conditioning=static_conditioning
        if type(known_context_inputs) is not bool:
            raise ValueError('known_context_inputs must be boolean')
        if known_context_inputs and not spacetime_inputs:
            raise ValueError('known_context_inputs requires spacetime_inputs=True')
        # The anchor can be present and still not be an output prior: the H-D round needs
        # ``C_valid`` for the reasoning recurrence's feedback while the decoder keeps the
        # registered persistence anchor, which is what makes "feedback representation" a
        # single factor instead of a package.
        if type(anchor_output_prior) is not bool:
            raise ValueError('anchor_output_prior must be boolean')
        if decoder_base_mode not in self.DECODER_BASE_MODES:
            raise ValueError(f"decoder_base_mode must be one of {self.DECODER_BASE_MODES}")
        if type(tendency_input) is not bool:
            raise ValueError('tendency_input must be boolean')
        if (decoder_base_mode == 'zero' and anchor_output_prior
                and climatology_anchor_spec is not None):
            # With the climatology output prior on, the head anchor is C_valid, not the
            # base, so zeroing the base would desynchronize the head anchor from the
            # base_state and the local solver. Reject the combination instead of
            # silently applying a different factor.
            raise ValueError("decoder_base_mode='zero' with the climatology output prior "
                             "would leave the decoder's head anchor on C_valid; set "
                             "anchor_output_prior=False")
        self.decoder_base_mode=str(decoder_base_mode)
        self.anchor_output_prior=anchor_output_prior
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
        self.tendency_input=tendency_input
        self.out_channels=int(out_channels or in_channels)
        if self.out_channels>self.in_channels:
            raise ValueError('R7.1 默认 persistence base 要求 out_channels <= in_channels')
        self.default_lead_hours=float(default_lead_hours)
        self.encoder=CoarseEncoder(self.in_channels,self.history_steps,dim,patch_size,depth,
            heads,window_size,dropout,activation_checkpointing,periodic_width,pad_to_patch=True,
            tendency_input=tendency_input)
        self.lead_time=LeadTimeEmbedding(dim)
        self.head=CoarseForecastHead(dim,self.out_channels,patch_size,
            change_scale_mode=change_scale_mode,change_scale_ratio=change_scale_ratio)
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
        # Decision 0069: time-invariant urban static term. Built last, after every existing
        # weight, and it registers no trained parameter at all (the field and the fixed
        # projection are buffers), so turning it on cannot shift any pre-existing weight.
        self.static_conditioning_pathway=None
        if self.static_conditioning:
            from .static_conditioning_r7 import StaticConditioning, load_static_fields
            field,variables=load_static_fields()
            self.static_conditioning_pathway=StaticConditioning(
                field,dim=dim,patch_size=patch_size,periodic_width=periodic_width,
                variables=variables)
        # Fixed zero buffers, constructed after every existing weight; no RNG draw.
        self.climatology_anchor=None
        if climatology_anchor_spec is not None:
            from .climatology_anchor_r7 import TrainClimatologyAnchor
            self.climatology_anchor=TrainClimatologyAnchor(climatology_anchor_spec)
            if self.climatology_anchor.table.shape[1]!=self.out_channels:
                raise ValueError('climatology_anchor channels must match out_channels')
            if self.in_channels!=self.out_channels:
                raise ValueError('climatology_anchor requires all input dynamic channels as outputs')

    def forward(self,batch:Mapping[str,torch.Tensor])->R7ForecastOutput:
        if self.climatology_anchor is not None and 'atmos_baseline' in batch:
            raise ValueError('climatology_anchor is incompatible with atmos_baseline')
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
        if self.static_conditioning_pathway is not None:
            # Time-invariant by construction: the same term at every lead and every reasoning
            # step. The off path above is unchanged operation for operation.
            context=context+self.static_conditioning_pathway(
                token_hw=token_hw).to(tokens.dtype)
        base=batch.get('atmos_baseline')
        if base is None:
            base=history[:,-1,:self.out_channels]
        # One switch with two consumers by construction: the decoder's direct anchor
        # and ``R7ForecastOutput.base_state`` are the same tensor, so a zero base also
        # makes every downstream anchored proposal (``Y = X_t + Decoder(Z)``) its own
        # tendency. The persistence path below is the pre-change body, operation for
        # operation; nothing else reads ``decoder_base_mode``.
        if self.decoder_base_mode=='zero':
            base=torch.zeros_like(base)
        if self.climatology_anchor is not None:
            climate=self.climatology_anchor(batch,history,
                default_lead_hours=self.default_lead_hours)
            # With ``anchor_output_prior`` off the decoder keeps the registered persistence
            # anchor and only the step's feedback reads ``C_valid``; the off path is the
            # pre-existing implementation operation for operation.
            head_anchor=climate if self.anchor_output_prior else base
            forecast,tendency=self.head(context,token_hw,(H,W),head_anchor)
            return R7ForecastOutput(forecast,tendency,context,token_hw,base,climate)
        forecast,tendency=self.head(context,token_hw,(H,W),base)
        return R7ForecastOutput(forecast,tendency,context,token_hw,base)
