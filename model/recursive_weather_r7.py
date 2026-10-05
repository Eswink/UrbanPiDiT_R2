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
from .local_solver_state_r7 import (ROLE_INITIAL_SCALE, SOLVER_INITIAL_SCALE,
    LocalSolverState, PositionGate, anchored_proposal, blend_forecast, expand_token_gate)
from .process_readout_r7 import PositionalProcessReadout, require_position_encoding_mode
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


def reasoning_source_key(model, context, draft_tokens, token_hw):
    """P/latent source key: position once, then role.

    Off delegates to the original expression without computing a position basis.
    Marked copies live only in this key: context, draft encoding and recurrent
    carry remain untouched for the reader, solver and next internal step.
    """
    role_context, role_draft = declared_source_roles(model)
    if not model.source_position_markers:
        return recurrent_key(context, draft_tokens, role_context=role_context,
            role_draft=role_draft if draft_tokens is not None else None)
    if context.ndim != 3 or context.shape[-1] != model.dim:
        raise ValueError('source_position_markers needs [B,N,D] context')
    rows, columns = int(token_hw[0]), int(token_hw[1])
    if rows < 1 or columns < 1 or rows * columns != context.shape[1]:
        raise ValueError('source_position_markers token grid must match context')
    if draft_tokens is not None:
        if draft_tokens.shape != context.shape:
            raise ValueError('source_position_markers requires aligned draft/context shapes')
        if draft_tokens.device != context.device or draft_tokens.dtype != context.dtype:
            raise ValueError('source_position_markers draft/context device and dtype must match')
    position = model.process_reader.position_encoding(
        token_hw, device=context.device, dtype=context.dtype).unsqueeze(0)
    marked_context = context + position
    marked_draft = None if draft_tokens is None else draft_tokens + position
    return recurrent_key(marked_context, marked_draft, role_context=role_context,
        role_draft=role_draft if draft_tokens is not None else None)


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


def process_to_generic_state_key(name: str) -> Optional[str]:
    """Explicit name map, not a checkpoint loader or an identity-check bypass.

    Only Process's diagnostic readout is absent. Callers must check complete key
    coverage and tensor shapes/dtypes before loading the mapped state strictly.
    The positional reader and all local solver/proposal weights remain present.
    """
    if name.startswith('process_readout.'):
        return None
    if name == 'process_queries':
        return 'latent'
    for source, target in (('reasoning_cell.', 'cell.'),
                           ('process_to_context.', 'latent_to_context.')):
        if name.startswith(source):
            return target + name[len(source):]
    return name


@dataclass
class GenericStepInput:
    latent: torch.Tensor
    context: torch.Tensor
    draft: torch.Tensor


@dataclass
class GenericStepOutput:
    latent: torch.Tensor
    draft: torch.Tensor
    correction: torch.Tensor
    solver_state: Optional[torch.Tensor] = None


def generic_reasoning_step(model, tensors: GenericStepInput,
                           token_hw: tuple[int, int], *,
                           solver_state: Optional[torch.Tensor] = None,
                           step_index: int = 0, anchor: Optional[torch.Tensor] = None,
                           use_forecast_feedback: Optional[bool] = None) -> GenericStepOutput:
    """Generic fixed/streamed step, equivalent to Process without diagnostics.

    Tokens have no anchored subset or diagnostic prediction. The forecast-facing
    operation order and RW-A/RW-B components match process_reasoning_step; tests
    pin every mapped state tensor, intermediate output and forecast gradient.
    """
    latent, context, draft = tensors.latent, tensors.context, tensors.draft
    feedback = model.use_forecast_feedback if use_forecast_feedback is None else bool(
        use_forecast_feedback)
    if model.local_solver_state and not feedback:
        raise ValueError('local_solver_state requires use_forecast_feedback=True')
    if model.draft_query_feedback and not feedback:
        raise ValueError('draft_query_feedback requires use_forecast_feedback=True')
    draft_tokens = None
    if feedback:
        draft_tokens, draft_hw = model.draft_encoder(draft)
        if tuple(draft_hw) != tuple(token_hw):
            raise ValueError(f'draft token grid {draft_hw} != context grid {token_hw}')
    latent = model._cell(latent, reasoning_source_key(model, context, draft_tokens, token_hw))
    if model.draft_query_feedback:
        summary = model.latent_conditioning(latent, context, token_hw, draft_tokens=draft_tokens)
    else:
        summary = model.latent_conditioning(latent, context, token_hw)
    conditioned = solver_conditioning(context, summary, draft_tokens,
        spatial_feedback=model.spatial_solver_feedback and feedback)
    if not model.local_solver_state:
        draft, correction = model.correction_head(
            conditioned, token_hw, draft.shape[-2:], draft)
        return GenericStepOutput(latent, draft, correction)
    if anchor is None:
        raise ValueError('local_solver_state decodes an absolute proposal and needs X_t anchor')
    if summary.ndim != 3:
        raise ValueError('local_solver_state needs the per-position latent read')
    if solver_state is None or not model.solver_state_recurrence:
        solver_state = model.solver_init.expand(
            context.shape[0], context.shape[1], -1).to(dtype=context.dtype)
        if not model.solver_state_recurrence:
            if not model.solver_gate_proposal:
                draft, correction = model.correction_head(
                    conditioned, token_hw, draft.shape[-2:], draft)
                return GenericStepOutput(latent, draft, correction)
            proposal, _ = anchored_proposal(model.proposal_head, solver_state, token_hw,
                draft.shape[-2:], anchor)
            gate = expand_token_gate(model.solver_gate(solver_state), token_hw,
                draft.shape[-2:], model.patch_size)
            updated = blend_forecast(draft, proposal, gate)
            return GenericStepOutput(latent, updated, updated - draft)
    solver_state = model.solver_cell(solver_state, context=context,
        draft_tokens=draft_tokens, read=summary, step_index=step_index, token_hw=token_hw)
    if not model.solver_gate_proposal:
        draft, correction = model.correction_head(
            conditioned, token_hw, draft.shape[-2:], draft)
        return GenericStepOutput(latent, draft, correction, solver_state)
    proposal, _ = anchored_proposal(model.proposal_head, solver_state, token_hw,
        draft.shape[-2:], anchor)
    gate = expand_token_gate(model.solver_gate(solver_state), token_hw,
        draft.shape[-2:], model.patch_size)
    updated = blend_forecast(draft, proposal, gate)
    return GenericStepOutput(latent, updated, updated - draft, solver_state)


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
    solver_state:Optional[torch.Tensor]=None


class GenericRecursiveWeatherForecaster(nn.Module):
    """Shared recursive baseline with no diagnostic slots or process prediction.

    The historical pooled path and its state keys stay unchanged by default.
    Opt-in RW-A/RW-B paths reuse Process's complete positional reader, local
    update, gate and anchored decoder. The solver sub-switches default off too;
    matching full RW-B requires explicitly enabling both inside local_solver_state.
    """
    def __init__(self,in_channels:int,history_steps:int=2,out_channels:Optional[int]=None,
                 dim:int=128,patch_size:int=2,depth:int=4,heads:int=4,window_size:int=8,
                 dropout:float=0.,activation_checkpointing:bool=False,periodic_width:bool=False,
                 default_lead_hours:float=6.,latent_tokens:int=16,default_reasoning_steps:int=4,
                 detach_between_steps:bool=False,spatial_solver_feedback:bool=False,
                 spacetime_inputs:bool=False,spacetime_field_mode:str='fields',
                 source_role_markers:bool=False,positional_process_readout:bool=False,
                 position_encoding_mode:str='legacy',
                 pooled_readout_query:bool=False,local_solver_state:bool=False,
                 solver_state_recurrence:bool=False,solver_gate_proposal:bool=False,
                 use_forecast_feedback:bool=True,draft_query_feedback:bool=False,
                 source_position_markers:bool=False,known_context_inputs:bool=False,
                 change_scale_mode:str='identity',change_scale_ratio=None):
        super().__init__()
        for value,name in ((spatial_solver_feedback,'spatial_solver_feedback'),
                           (spacetime_inputs,'spacetime_inputs'),
                           (source_role_markers,'source_role_markers'),
                           (positional_process_readout,'positional_process_readout'),
                           (pooled_readout_query,'pooled_readout_query'),
                           (local_solver_state,'local_solver_state'),
                           (solver_state_recurrence,'solver_state_recurrence'),
                           (solver_gate_proposal,'solver_gate_proposal'),
                           (use_forecast_feedback,'use_forecast_feedback'),
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
            raise ValueError('pooled_readout_query requires positional_process_readout=True')
        if draft_query_feedback and not positional_process_readout:
            raise ValueError('draft_query_feedback requires positional_process_readout=True')
        if draft_query_feedback and not use_forecast_feedback:
            raise ValueError('draft_query_feedback requires use_forecast_feedback=True')
        if local_solver_state and not positional_process_readout:
            raise ValueError('local_solver_state requires positional_process_readout=True')
        if local_solver_state and not use_forecast_feedback:
            raise ValueError('local_solver_state requires use_forecast_feedback=True')
        for value,name in ((solver_state_recurrence,'solver_state_recurrence'),
                           (solver_gate_proposal,'solver_gate_proposal')):
            if value and not local_solver_state:
                raise ValueError(f'{name}=True requires local_solver_state=True')
        if int(latent_tokens)<1:
            raise ValueError('latent_tokens must be >= 1')
        self.spatial_solver_feedback=spatial_solver_feedback
        self.source_role_markers=source_role_markers
        self.source_position_markers=source_position_markers
        self.known_context_inputs=known_context_inputs
        self.positional_process_readout=positional_process_readout
        self.position_encoding_mode=position_encoding_mode
        self.pooled_readout_query=pooled_readout_query
        self.draft_query_feedback=draft_query_feedback
        self.local_solver_state=local_solver_state
        self.solver_state_recurrence=solver_state_recurrence
        self.solver_gate_proposal=solver_gate_proposal
        self.change_scale_mode=change_scale_mode
        self.use_forecast_feedback=use_forecast_feedback
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
            spacetime_inputs=spacetime_inputs,spacetime_field_mode=self.spacetime_field_mode,
            known_context_inputs=self.known_context_inputs,
            change_scale_mode=change_scale_mode,change_scale_ratio=change_scale_ratio)
        self.latent=nn.Parameter(torch.randn(1,int(latent_tokens),dim)*.02)
        self.draft_encoder=DraftTokenEncoder(self.out_channels,dim,patch_size)
        self.cell=GenericRecursiveCell(dim,heads,mlp_ratio=3.,dropout=dropout)
        self.latent_to_context=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.correction_head=CoarseForecastHead(dim,self.out_channels,patch_size,
            change_scale_mode=change_scale_mode,change_scale_ratio=change_scale_ratio)
        # Built last and under a rewound stream, for the same reason the space-time
        # pathway is: declaring source roles must not move a single weight the
        # model would have had without them, so the two arms differ by the roles
        # and not by a shifted initialization.
        if self.positional_process_readout:
            with isolated_stream():
                self.process_reader=PositionalProcessReadout(
                    dim,heads,dropout,pooled_readout_query=self.pooled_readout_query,
                    draft_query_feedback=self.draft_query_feedback,
                    position_encoding_mode=self.position_encoding_mode)
        if self.source_role_markers:
            with isolated_stream():
                self.role_context=nn.Parameter(torch.randn(1,1,dim)*ROLE_INITIAL_SCALE)
                self.role_draft=nn.Parameter(torch.randn(1,1,dim)*ROLE_INITIAL_SCALE)
        if self.local_solver_state:
            with isolated_stream():
                self.solver_init=nn.Parameter(torch.randn(1,1,dim)*SOLVER_INITIAL_SCALE)
                self.solver_cell=LocalSolverState(dim)
                self.solver_gate=PositionGate(dim)
                self.proposal_head=CoarseForecastHead(dim,self.out_channels,patch_size,
                    change_scale_mode=change_scale_mode,
                    change_scale_ratio=change_scale_ratio)

    def latent_conditioning(self,latent,context,token_hw,*,draft_tokens=None):
        """Same solver-facing read and dimensions, unconstrained Generic tokens."""
        if self.positional_process_readout:
            if self.draft_query_feedback:
                return self.process_reader(latent,context,token_hw,draft_tokens=draft_tokens)
            return self.process_reader(latent,context,token_hw)
        return self.latent_to_context(latent.mean(dim=1))

    def _cell(self,z,context):
        if self.activation_checkpointing and self.training and z.requires_grad:
            return checkpoint(self.cell,z,context,use_reentrant=False)
        return self.cell(z,context)

    def forward(self,batch:Mapping[str,torch.Tensor],*,reasoning_steps:Optional[int]=None,
                detach_between_steps:Optional[bool]=None,
                use_forecast_feedback:Optional[bool]=None)->RecursiveForecastOutput:
        base=self.backbone(batch)
        initial=draft=base.forecast
        context,token_hw=base.context_tokens,base.token_hw
        B=initial.shape[0]
        steps=self.default_reasoning_steps if reasoning_steps is None else int(reasoning_steps)
        if steps<0:
            raise ValueError('reasoning_steps 必须 >= 0')
        detach_flag=self.detach_between_steps if detach_between_steps is None else bool(detach_between_steps)
        feedback_flag=self.use_forecast_feedback if use_forecast_feedback is None else bool(use_forecast_feedback)
        if self.local_solver_state and not feedback_flag:
            raise ValueError('local_solver_state requires use_forecast_feedback=True')
        if self.draft_query_feedback and not feedback_flag:
            raise ValueError('draft_query_feedback requires use_forecast_feedback=True')
        z=self.latent.expand(B,-1,-1)
        solver_state=None
        drafts=[draft]
        final_correction=torch.zeros_like(draft)
        for step in range(steps):
            result=generic_reasoning_step(self,GenericStepInput(z,context,draft),token_hw,
                solver_state=solver_state,step_index=step,anchor=base.base_state,
                use_forecast_feedback=feedback_flag)
            z,draft,final_correction=result.latent,result.draft,result.correction
            solver_state=result.solver_state
            drafts.append(draft)
            if detach_flag and step<steps-1:
                z,draft=z.detach(),draft.detach()
                if solver_state is not None:
                    solver_state=solver_state.detach()
        return RecursiveForecastOutput(draft,initial,torch.stack(drafts,dim=1),
            final_correction,z,context,token_hw,steps,solver_state)
