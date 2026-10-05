from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .coarse_forecast import CoarseForecastHead
from .local_solver_state_r7 import (SOLVER_INITIAL_SCALE, LocalSolverState, PositionGate)
from .process_readout_r7 import PositionalProcessReadout, require_position_encoding_mode
from .process_step_r7 import ProcessStepInput, ProcessStepOutput, process_reasoning_step
from .recursive_weather_r7 import (ROLE_INITIAL_SCALE, DraftTokenEncoder,
    GenericRecursiveCell, solver_conditioning)
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
    # The RW-B per-position solver state after the last step, or None when the
    # switch is off. It is returned rather than kept on the module so that a
    # caller can thread it across physical transitions deliberately.
    solver_state: Optional[torch.Tensor]=None


def _validate_switches(*, spatial_solver_feedback, spacetime_inputs,
                       positional_process_readout, pooled_readout_query,
                       source_role_markers, local_solver_state,
                       solver_state_recurrence, solver_gate_proposal,
                       draft_query_feedback, source_position_markers,
                       known_context_inputs, use_forecast_feedback,
                       position_encoding_mode):
    """Reject switch combinations nothing reads; one definition for every arm.

    Extracted from the constructor so the constructor stays inside R-052 while
    the combination rules remain written down exactly once. Every rule here is
    the same statement: a switch turned on where nothing reads it, or turned
    off where the thing it modifies is off, is an error rather than a no-op.
    """
    for value,name in ((spatial_solver_feedback,'spatial_solver_feedback'),
                       (spacetime_inputs,'spacetime_inputs'),
                       (positional_process_readout,'positional_process_readout'),
                       (pooled_readout_query,'pooled_readout_query'),
                       (source_role_markers,'source_role_markers'),
                       (local_solver_state,'local_solver_state'),
                       (solver_state_recurrence,'solver_state_recurrence'),
                       (solver_gate_proposal,'solver_gate_proposal'),
                       (draft_query_feedback,'draft_query_feedback'),
                       (source_position_markers,'source_position_markers'),
                       (known_context_inputs,'known_context_inputs')):
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
    if source_position_markers and not positional_process_readout:
        raise ValueError('source_position_markers requires positional_process_readout=True')
    require_position_encoding_mode(position_encoding_mode,
                                   positional_process_readout=positional_process_readout)
    if known_context_inputs and not spacetime_inputs:
        raise ValueError('known_context_inputs requires spacetime_inputs=True')
    if pooled_readout_query and not positional_process_readout:
        raise ValueError("pooled_readout_query only exists inside the positional "
                         "process readout; turning it on without "
                         "positional_process_readout would be a silently ignored switch")
    if draft_query_feedback and not positional_process_readout:
        raise ValueError('draft_query_feedback requires positional_process_readout=True')
    if draft_query_feedback and not use_forecast_feedback:
        raise ValueError('draft_query_feedback requires use_forecast_feedback=True')
    # RW-B's working state is updated from the per-position process read R_k and
    # from the encoded draft; without either of those two inputs the recurrence
    # would not be the one the design contract froze, so the combination is
    # rejected instead of quietly substituting a pooled read or dropping a term.
    if local_solver_state and not positional_process_readout:
        raise ValueError("local_solver_state updates Z from the per-position read "
                         "R_k; turning it on without positional_process_readout "
                         "would silently substitute the pooled summary")
    if local_solver_state and not use_forecast_feedback:
        raise ValueError("local_solver_state updates Z from the encoded draft "
                         "E(Y_k); turning it on with use_forecast_feedback=False "
                         "would silently drop that term")
    # The subtraction switches default to the full RW-B and only mean anything
    # inside it. A caller that turns a piece off with the whole mechanism off is
    # asking for something nothing reads, so that is an error and not a no-op.
    for value,name in ((solver_state_recurrence,'solver_state_recurrence'),
                       (solver_gate_proposal,'solver_gate_proposal')):
        if value is not True and not local_solver_state:
            raise ValueError(
                f"{name}=False only exists inside local_solver_state; turning it off "
                "with local_solver_state=False would be a silently ignored switch")


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
        position_encoding_mode:str='legacy',
        pooled_readout_query:bool=False,
        source_role_markers:bool=False,
        local_solver_state:bool=False,
        solver_state_recurrence:bool=True,
        solver_gate_proposal:bool=True,
        draft_query_feedback:bool=False,
        source_position_markers:bool=False,
        known_context_inputs:bool=False,
        change_scale_mode:str='identity',
        change_scale_ratio=None,
        typed_evidence_mode:Optional[str]=None,
        typed_evidence=None,
    ):
        super().__init__()
        _validate_switches(
            spatial_solver_feedback=spatial_solver_feedback,
            spacetime_inputs=spacetime_inputs,
            positional_process_readout=positional_process_readout,
            pooled_readout_query=pooled_readout_query,
            source_role_markers=source_role_markers,
            local_solver_state=local_solver_state,
            solver_state_recurrence=solver_state_recurrence,
            solver_gate_proposal=solver_gate_proposal,
            draft_query_feedback=draft_query_feedback,
            source_position_markers=source_position_markers,
            known_context_inputs=known_context_inputs,
            use_forecast_feedback=use_forecast_feedback,
            position_encoding_mode=position_encoding_mode)
        self.spatial_solver_feedback=spatial_solver_feedback
        self.source_position_markers=source_position_markers
        self.known_context_inputs=known_context_inputs
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
        self.position_encoding_mode=position_encoding_mode
        self.pooled_readout_query=pooled_readout_query
        self.draft_query_feedback=draft_query_feedback
        self.source_role_markers=source_role_markers
        self.local_solver_state=local_solver_state
        self.solver_state_recurrence=solver_state_recurrence
        self.solver_gate_proposal=solver_gate_proposal
        self.change_scale_mode=change_scale_mode
        self.typed_evidence_mode=typed_evidence_mode
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
            known_context_inputs=self.known_context_inputs,
            change_scale_mode=change_scale_mode,
            change_scale_ratio=change_scale_ratio,
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
            change_scale_mode=change_scale_mode,
            change_scale_ratio=change_scale_ratio,
        )
        # Same rule as the space-time term: the pathway is constructed last and
        # under a rewound stream, so switching it on cannot move any other weight.
        # ``pooled_readout_query`` picks the query construction inside that
        # pathway and changes no parameter, which is what makes the round-two
        # positional and pooled arms a capacity-matched pair.
        if self.positional_process_readout:
            with isolated_stream():
                self.process_reader=PositionalProcessReadout(
                    dim,heads,dropout,pooled_readout_query=self.pooled_readout_query,
                    draft_query_feedback=self.draft_query_feedback,
                    position_encoding_mode=self.position_encoding_mode)
        # Both RW-B pathways are built last and under a rewound stream as well, so
        # "off" is the previous implementation and not merely something close to it.
        if self.source_role_markers:
            with isolated_stream():
                self.role_context=nn.Parameter(torch.randn(1,1,dim)*ROLE_INITIAL_SCALE)
                self.role_draft=nn.Parameter(torch.randn(1,1,dim)*ROLE_INITIAL_SCALE)
        if self.local_solver_state:
            with isolated_stream():
                self.solver_init=nn.Parameter(
                    torch.randn(1,1,dim)*SOLVER_INITIAL_SCALE)
                self.solver_cell=LocalSolverState(dim)
                self.solver_gate=PositionGate(dim)
                self.proposal_head=CoarseForecastHead(
                    dim=dim,out_channels=self.out_channels,patch_size=patch_size,
                    change_scale_mode=change_scale_mode,
                    change_scale_ratio=change_scale_ratio)
        # #79 typed local evidence: built last under a rewound stream so "off"
        # is the previous implementation bit for bit. The pathway adds exactly
        # one shared patch projection, one projection set and one inject site;
        # the reasoning recurrence downstream of the slot state is untouched.
        if typed_evidence_mode is not None:
            from .typed_evidence_r7 import EVIDENCE_MODES, TypedEvidenceRouter
            if typed_evidence_mode not in EVIDENCE_MODES:
                raise ValueError(f"typed_evidence_mode must be one of {EVIDENCE_MODES}")
            fields = dict(typed_evidence or {})
            expected = {'channels', 'denorm_mean', 'denorm_std', 'field_scale',
                        'field_mean', 'field_std', 'field_active'}
            if set(fields) != expected:
                raise ValueError(f"typed_evidence must carry exactly {sorted(expected)}")
            if self.anchored_processes < 4:
                raise ValueError("typed evidence writes the first four anchored slots; "
                                 "anchored_processes must be >= 4")
            with isolated_stream():
                self.typed_evidence=TypedEvidenceRouter(
                    dim,patch_size,mode=typed_evidence_mode,**fields)

    def initial_process_state(self, process:torch.Tensor, batch:Mapping, *,
                              anchor:torch.Tensor, draft:torch.Tensor)->torch.Tensor:
        """The process state before the first internal step, with #79 evidence.

        Every path that starts the recurrence calls this: the fixed forward, the
        streamed trainer and the adaptive wrapper. With the pathway off this is
        the expanded ``process_queries``, operation for operation.

        ``anchor`` and ``draft`` are the tensors the first step itself consumes:
        ``X_t`` is the known state the decode is anchored to and ``draft`` is the
        initial forecast that becomes ``Y_0``. The caller passes the *same*
        tensors it will step with - in the streamed trainer that is the detached
        interface leaf, so the evidence gradient accumulates on the leaf and is
        sent through the backbone exactly once by the trainer's own final
        adjoint pass, instead of opening a second retained graph. No target,
        future field or validation value is read here.
        """
        if not hasattr(self,"typed_evidence"):
            return process
        evidence = self.typed_evidence.evidence(batch, anchor=anchor, draft=draft)
        return self.typed_evidence.inject(process, evidence)

    def process_conditioning(
        self,
        process:torch.Tensor,
        context:torch.Tensor,
        token_hw:tuple[int,int],
        *,
        draft_tokens:Optional[torch.Tensor]=None,
    )->torch.Tensor:
        """The solver-facing read of the process state, at every output position.

        With the positional switch off this is the pooled ``[B,D]`` summary that
        the pre-RW-A model broadcast; with it on it is a ``[B,N,D]`` read in
        which each position queries the process tokens itself. Either way the
        result is *added* to the solver context by ``solver_conditioning``.
        """
        if self.positional_process_readout:
            if self.draft_query_feedback:
                return self.process_reader(process,context,token_hw,draft_tokens=draft_tokens)
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

        if self.draft_query_feedback and not feedback_flag:
            raise ValueError('draft_query_feedback requires use_forecast_feedback=True')

        process=self.process_queries.expand(B,-1,-1)
        process=self.initial_process_state(process,batch,anchor=base.base_state,draft=draft)
        solver_state=None
        drafts=[draft]
        process_predictions=[]
        final_correction=torch.zeros_like(draft)

        for step in range(steps):
            result=process_reasoning_step(
                self,
                ProcessStepInput(process,context,draft),
                token_hw,
                solver_state=solver_state,
                step_index=step,
                anchor=base.base_state,
                use_forecast_feedback=feedback_flag,
            )
            process=result.process
            solver_state=result.solver_state
            draft=result.draft
            final_correction=result.correction
            process_predictions.append(result.prediction)
            drafts.append(draft)

            if detach_flag and step < steps-1:
                process=process.detach()
                draft=draft.detach()
                if solver_state is not None:
                    solver_state=solver_state.detach()

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
            solver_state=solver_state,
        )
