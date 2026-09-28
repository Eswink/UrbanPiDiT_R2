from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .coarse_forecast import CoarseForecastHead
from .process_readout_r7 import PositionalProcessReadout
from .recursive_weather_r7 import DraftTokenEncoder, GenericRecursiveCell, solver_conditioning
from .spacetime_conditioning_r7 import isolated_stream, require_field_mode
from .weather_forecaster_r7 import NativeAtmosForecaster


@dataclass
class ProcessForecastReasoningOutput:
    forecast: torch.Tensor
    initial_forecast: torch.Tensor
    draft_forecasts: torch.Tensor
    process_predictions: torch.Tensor
    final_correction: torch.Tensor
    process_state: torch.Tensor
    context_tokens: torch.Tensor
    token_hw: tuple[int,int]
    reasoning_steps: int


class ProcessForecastCoReasoner(nn.Module):
    """R7.3 Process-Forecast Co-Reasoning model.

    The model deliberately shares its backbone, recurrent-cell family, draft
    encoder and forecast-correction family with the generic R7.2 baseline.
    Its methodological difference is a semi-structured Process State:

    - the first anchored_processes slots align to meteorological diagnostic
      proxy targets;
    - the remaining slots are unconstrained latent reasoning tokens;
    - the current forecast draft is re-encoded and fed back into the shared
      recurrent update before every forecast correction.

    Recurrence:
        P_(k+1) = U(P_k, C, E(Y_k))
        Y_(k+1) = Y_k + S(C, P_(k+1), Y_k)

    This is latent recurrent reasoning, not natural-language chain-of-thought.
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
        anchored_processes:int=8,
        free_processes:int=8,
        default_reasoning_steps:int=4,
        detach_between_steps:bool=False,
        use_forecast_feedback:bool=True,
        spatial_solver_feedback:bool=False,
        spacetime_inputs:bool=False,
        spacetime_field_mode:str='fields',
        positional_process_readout:bool=False,
        pooled_readout_query:bool=False,
    ):
        super().__init__()
        for value,name in ((spatial_solver_feedback,'spatial_solver_feedback'),
                           (spacetime_inputs,'spacetime_inputs'),
                           (positional_process_readout,'positional_process_readout'),
                           (pooled_readout_query,'pooled_readout_query')):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        if pooled_readout_query and not positional_process_readout:
            raise ValueError("pooled_readout_query only exists inside the positional "
                             "process readout; turning it on without "
                             "positional_process_readout would be a silently ignored switch")
        self.spatial_solver_feedback=spatial_solver_feedback
        self.spacetime_inputs=spacetime_inputs
        # The capacity-control arms of the round-three study keep the space-time
        # module and change only what it is shown; a control mode with the pathway
        # off would add nothing and say nothing, so it is rejected.
        self.spacetime_field_mode=require_field_mode(spacetime_field_mode)
        if not self.spacetime_inputs and self.spacetime_field_mode!='fields':
            raise ValueError(
                f"spacetime_field_mode={self.spacetime_field_mode!r} needs "
                "spacetime_inputs=True; with the pathway off it would be silently ignored")
        self.positional_process_readout=positional_process_readout
        self.pooled_readout_query=pooled_readout_query
        self.out_channels=int(out_channels or in_channels)
        self.dim=int(dim)
        self.patch_size=int(patch_size)
        self.anchored_processes=int(anchored_processes)
        self.free_processes=int(free_processes)
        self.num_process_tokens=(
            self.anchored_processes+self.free_processes
        )
        if self.anchored_processes<1:
            raise ValueError("anchored_processes 必须 >= 1")
        if self.free_processes<0:
            raise ValueError("free_processes 必须 >= 0")
        if self.num_process_tokens<1:
            raise ValueError("至少需要一个 process token")

        self.default_reasoning_steps=int(default_reasoning_steps)
        self.detach_between_steps=bool(detach_between_steps)
        self.use_forecast_feedback=bool(use_forecast_feedback)
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
            spacetime_inputs=spacetime_inputs,
            spacetime_field_mode=self.spacetime_field_mode,
        )

        self.process_queries=nn.Parameter(
            torch.randn(1,self.num_process_tokens,dim)*0.02
        )
        self.draft_encoder=DraftTokenEncoder(
            self.out_channels,dim,patch_size
        )
        self.reasoning_cell=GenericRecursiveCell(
            dim,heads,mlp_ratio=3.0,dropout=dropout
        )

        self.process_readout=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,1),
        )
        self.process_to_context=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,dim),
        )
        self.correction_head=CoarseForecastHead(
            dim=dim,
            out_channels=self.out_channels,
            patch_size=patch_size,
        )
        # Same rule as the space-time term: the pathway is constructed last and
        # under a rewound stream, so switching it on cannot move any other weight.
        # ``pooled_readout_query`` picks the query construction inside that
        # pathway and changes no parameter, which is what makes the round-two
        # positional and pooled arms a capacity-matched pair.
        if self.positional_process_readout:
            with isolated_stream():
                self.process_reader=PositionalProcessReadout(
                    dim,heads,dropout,pooled_readout_query=self.pooled_readout_query)

    def process_conditioning(
        self,
        process:torch.Tensor,
        context:torch.Tensor,
        token_hw:tuple[int,int],
    )->torch.Tensor:
        """The solver-facing read of the process state, at every output position.

        With the positional switch off this is the pooled ``[B,D]`` summary that
        the pre-RW-A model broadcast; with it on it is a ``[B,N,D]`` read in
        which each position queries the process tokens itself. Either way the
        result is *added* to the solver context by ``solver_conditioning``.
        """
        if self.positional_process_readout:
            return self.process_reader(process,context,token_hw)
        return self.process_to_context(process.mean(dim=1))

    def _reason(
        self,
        process:torch.Tensor,
        recurrent_context:torch.Tensor,
    )->torch.Tensor:
        if (
            self.activation_checkpointing
            and self.training
            and process.requires_grad
        ):
            return checkpoint(
                self.reasoning_cell,
                process,
                recurrent_context,
                use_reentrant=False,
            )
        return self.reasoning_cell(process,recurrent_context)

    def _process_prediction(
        self,
        process:torch.Tensor,
    )->torch.Tensor:
        return self.process_readout(
            process[:,:self.anchored_processes]
        ).squeeze(-1)

    def forward(
        self,
        batch:Mapping[str,torch.Tensor],
        *,
        reasoning_steps:Optional[int]=None,
        detach_between_steps:Optional[bool]=None,
        use_forecast_feedback:Optional[bool]=None,
    )->ProcessForecastReasoningOutput:
        base=self.backbone(batch)
        initial=base.forecast
        draft=initial
        context=base.context_tokens
        token_hw=base.token_hw
        output_hw=initial.shape[-2:]
        B=initial.shape[0]

        steps=(
            self.default_reasoning_steps
            if reasoning_steps is None
            else int(reasoning_steps)
        )
        if steps<0:
            raise ValueError("reasoning_steps 必须 >= 0")
        detach_flag=(
            self.detach_between_steps
            if detach_between_steps is None
            else bool(detach_between_steps)
        )
        feedback_flag=(
            self.use_forecast_feedback
            if use_forecast_feedback is None
            else bool(use_forecast_feedback)
        )

        process=self.process_queries.expand(B,-1,-1)
        drafts=[draft]
        process_predictions=[]
        final_correction=torch.zeros_like(draft)

        for step in range(steps):
            draft_tokens=None
            if feedback_flag:
                draft_tokens,draft_hw=self.draft_encoder(draft)
                if tuple(draft_hw)!=tuple(token_hw):
                    raise ValueError(
                        f"draft token grid {draft_hw} != "
                        f"context grid {token_hw}"
                    )
                recurrent_context=torch.cat(
                    [context,draft_tokens],dim=1
                )
            else:
                recurrent_context=context

            process=self._reason(process,recurrent_context)
            process_predictions.append(
                self._process_prediction(process)
            )

            summary=self.process_conditioning(process,context,token_hw)
            solver_context=solver_conditioning(context,summary,draft_tokens,
                spatial_feedback=self.spatial_solver_feedback and feedback_flag)
            draft,final_correction=self.correction_head(
                solver_context,
                token_hw,
                output_hw,
                draft,
            )
            drafts.append(draft)

            if detach_flag and step < steps-1:
                process=process.detach()
                draft=draft.detach()

        if process_predictions:
            process_trace=torch.stack(process_predictions,dim=1)
        else:
            process_trace=initial.new_empty(
                B,0,self.anchored_processes
            )

        return ProcessForecastReasoningOutput(
            forecast=draft,
            initial_forecast=initial,
            draft_forecasts=torch.stack(drafts,dim=1),
            process_predictions=process_trace,
            final_correction=final_correction,
            process_state=process,
            context_tokens=context,
            token_hw=token_hw,
            reasoning_steps=steps,
        )
