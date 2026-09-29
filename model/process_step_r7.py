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
- ``anchor`` is ``X_t``, the known state at the current physical time. With
  ``local_solver_state`` on it is *required*: a proposal without its anchor is the
  accumulating tendency that RW-B exists to replace, and silently producing that
  would be a different method wearing the switch's name.

With both switches off the body below is the pre-RW-B body, operation for
operation, which is what makes "off" the previous implementation rather than
something close to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from .local_solver_state_r7 import anchored_proposal, blend_forecast, expand_token_gate
from .recursive_weather_r7 import declared_source_roles, recurrent_key, solver_conditioning

if TYPE_CHECKING:
    from .process_forecast_r7 import ProcessForecastCoReasoner


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
    process: torch.Tensor,
    context: torch.Tensor,
    draft: torch.Tensor,
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
    feedback = model.use_forecast_feedback if use_forecast_feedback is None else bool(
        use_forecast_feedback)
    draft_tokens = None
    if feedback:
        draft_tokens, draft_hw = model.draft_encoder(draft)
        if tuple(draft_hw) != tuple(token_hw):
            raise ValueError(f"draft token grid {draft_hw} != context grid {token_hw}")
    role_context, role_draft = declared_source_roles(model)
    process = model._reason(process, recurrent_key(
        context, draft_tokens, role_context=role_context, role_draft=role_draft))
    prediction = model._process_prediction(process)
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
    if solver_state is None:
        solver_state = model.solver_init.expand(
            context.shape[0], context.shape[1], -1).to(dtype=context.dtype)
    solver_state = model.solver_cell(solver_state, context=context,
        draft_tokens=draft_tokens, read=summary, step_index=step_index, token_hw=token_hw)
    proposal, _ = anchored_proposal(model.proposal_head, solver_state, token_hw,
        draft.shape[-2:], anchor)
    gate = expand_token_gate(model.solver_gate(solver_state), token_hw,
        draft.shape[-2:], model.patch_size)
    updated = blend_forecast(draft, proposal, gate)
    return ProcessStepOutput(process, updated, updated - draft, prediction, solver_state)
