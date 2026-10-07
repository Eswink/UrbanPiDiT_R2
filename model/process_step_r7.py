"""The process co-reasoner's internal step, written down exactly once (#72 M2-B).

Before RW-B the same recurrence existed three times - in
``ProcessForecastCoReasoner.forward``, in ``training/r7_streaming.py`` and in
``model/r7_halting.py`` - and #72 names the formula drift that creates as a defect
to close rather than a risk to note. Fixed forward, streamed backward and the
adaptive wrapper now all call :func:`process_reasoning_step`; this module exists so
that "the step" is a named artifact with one body, not a convention about three
files agreeing.

The step is the process co-reasoner's, so it takes the model rather than
reimplementing anything: ``_reason``, ``_process_prediction`` and
``process_conditioning`` stay the model's, and what lives here is the *order* of
the operations and the branch between the pre-RW-B update and RW-B's.

Two states are threaded through a step and neither is implicit:

- ``solver_state`` is RW-B's per-patch working state ``Z``. It is passed in and
  returned rather than stored on the module, so a caller decides how long it lives
  (across the internal steps of one physical transition, and no longer).
- ``anchor`` is physical ``X_t`` and remains required with RW-B on. The optional
  ``tensors.climatology_anchor`` is a distinct fixed ``C_valid`` for this physical
  transition; enabled models require it and decode both RW-B proposals from it.
  Neither anchor nor the absolute draft is cached or advanced during internal K.

With both switches off the body below is the pre-RW-B body, operation for
operation, which is what makes "off" the previous implementation rather than
something close to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from .local_solver_state_r7 import anchored_proposal, blend_forecast, expand_token_gate
from .recursive_weather_r7 import reasoning_source_key, solver_conditioning

if TYPE_CHECKING:
    from .process_forecast_r7 import ProcessForecastCoReasoner


@dataclass
class ProcessStepInput:
    """The tensors one internal step reads.

    Bundled rather than passed one by one: three same-shaped states in a row read
    as an argument list, and the step's signature should be about the step. It also
    keeps the parameter count inside the project's target instead of bending that
    target to fit a long signature.
    """
    process: torch.Tensor
    context: torch.Tensor
    draft: torch.Tensor
    climatology_anchor: Optional[torch.Tensor] = None


@dataclass
class ProcessStepOutput:
    """One internal reasoning step of the process co-reasoner."""
    process: torch.Tensor
    draft: torch.Tensor
    correction: torch.Tensor
    prediction: Optional[torch.Tensor]
    solver_state: Optional[torch.Tensor] = None


def process_reasoning_step(
    model: "ProcessForecastCoReasoner",
    tensors: ProcessStepInput,
    token_hw: tuple[int, int],
    *,
    solver_state: Optional[torch.Tensor] = None,
    step_index: int = 0,
    anchor: Optional[torch.Tensor] = None,
    use_forecast_feedback: Optional[bool] = None,
) -> ProcessStepOutput:
    """One internal step: read the process state, decode, and update the draft.

    ``use_forecast_feedback`` overrides the model's declared setting for this call
    only; the fixed forward passes its per-call argument through here instead of
    branching around the step, which is how the three copies diverged before.
    """
    process, context, draft = tensors.process, tensors.context, tensors.draft
    feedback = model.use_forecast_feedback if use_forecast_feedback is None else bool(
        use_forecast_feedback)
    if model.draft_query_feedback and not feedback:
        raise ValueError('draft_query_feedback requires use_forecast_feedback=True')
    climate = tensors.climatology_anchor
    if model.backbone.climatology_anchor is not None:
        if climate is None:
            raise ValueError('climatology_anchor mode requires C_valid at every process step')
        if (not torch.is_tensor(climate) or climate.shape != draft.shape
                or climate.device != draft.device or not climate.is_floating_point()
                or climate.requires_grad or not torch.isfinite(climate).all()):
            raise ValueError('climatology_anchor must be fixed finite floating on the draft grid/device')
    if model.anomaly_feedback and not feedback:
        raise ValueError('anomaly_feedback requires use_forecast_feedback=True')
    if model.local_solver_state and not feedback:
        raise ValueError('local_solver_state requires use_forecast_feedback=True')
    draft_tokens = None
    if feedback:
        feedback_draft = draft - climate if model.anomaly_feedback else draft
        draft_tokens, draft_hw = model.draft_encoder(feedback_draft)
        if tuple(draft_hw) != tuple(token_hw):
            raise ValueError(f"draft token grid {draft_hw} != context grid {token_hw}")
    process = model._reason(process, reasoning_source_key(model, context, draft_tokens, token_hw))
    prediction = model._process_prediction(process)
    if model.draft_query_feedback:
        summary = model.process_conditioning(process, context, token_hw, draft_tokens=draft_tokens)
    else:
        summary = model.process_conditioning(process, context, token_hw)
    conditioned = solver_conditioning(context, summary, draft_tokens,
        spatial_feedback=model.spatial_solver_feedback and feedback)
    if not model.local_solver_state:
        draft, correction = model.correction_head(
            conditioned, token_hw, draft.shape[-2:], draft)
        return ProcessStepOutput(process, draft, correction, prediction, None)
    if anchor is None:
        raise ValueError(
            "local_solver_state decodes an absolute proposal and needs X_t as its "
            "anchor; without it the proposal is the accumulating tendency RW-B replaces")
    if summary.ndim != 3:
        raise ValueError("local_solver_state needs the per-position process read")
    # Two separate pieces of RW-B, each switchable on its own.
    #
    # ``solver_state_recurrence`` off: the cell is never applied, so ``Z`` is the
    # expanded ``solver_init`` on every step. The proposal and the gate are still
    # applied to it, and nothing is carried to the next step - a state that was
    # threaded onward would quietly be the recurrence again.
    #
    # ``solver_gate_proposal`` off: the step takes the pre-RW-B correction path. The
    # state still advances when the recurrence is on, because that is what "the
    # recurrence is still there" means - but it is idle computation for this step's
    # output, and it is returned as such rather than being silently swapped for
    # something else the switch does not name.
    if solver_state is None or not model.solver_state_recurrence:
        solver_state = model.solver_init.expand(
            context.shape[0], context.shape[1], -1).to(dtype=context.dtype)
        if not model.solver_state_recurrence:
            if not model.solver_gate_proposal:
                draft, correction = model.correction_head(
                    conditioned, token_hw, draft.shape[-2:], draft)
                return ProcessStepOutput(process, draft, correction, prediction, None)
            proposal, _ = anchored_proposal(model.proposal_head, solver_state, token_hw,
                draft.shape[-2:], climate if climate is not None else anchor)
            gate = expand_token_gate(model.solver_gate(solver_state), token_hw,
                draft.shape[-2:], model.patch_size)
            updated = blend_forecast(draft, proposal, gate)
            return ProcessStepOutput(process, updated, updated - draft, prediction, None)
    solver_state = model.solver_cell(solver_state, context=context,
        draft_tokens=draft_tokens, read=summary, step_index=step_index, token_hw=token_hw)
    if not model.solver_gate_proposal:
        draft, correction = model.correction_head(
            conditioned, token_hw, draft.shape[-2:], draft)
        return ProcessStepOutput(process, draft, correction, prediction, solver_state)
    proposal, _ = anchored_proposal(model.proposal_head, solver_state, token_hw,
        draft.shape[-2:], climate if climate is not None else anchor)
    gate = expand_token_gate(model.solver_gate(solver_state), token_hw,
        draft.shape[-2:], model.patch_size)
    updated = blend_forecast(draft, proposal, gate)
    return ProcessStepOutput(process, updated, updated - draft, prediction, solver_state)
