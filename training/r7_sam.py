"""Sharpness-aware minimization (SAM) as a single training-update factor.

The registered update is one backward of the registered scalar long-rollout loss, then
the registered ``clip_grad_norm_`` and ``optimizer.step()``. This module adds one
predeclared alternative: the two-step SAM perturbation of Foret et al.
(arXiv:2010.01412v3, section 3.1) using that paper's p=2 dual solution, in which the
ascent step is the registered loss's **global** gradient rescaled to a fixed radius
``rho``:

    g     = grad of the registered scalar loss at w      (one global norm over every
                                                          trainable parameter)
    e     = rho * g / (||g|| + eps)                      (one scalar rescaling)
    w     -> w + e
    g_sam = grad of the *same* registered loss at w + e
    w     -> the saved original parameters, copied back bit for bit
    then the registered ``clip_grad_norm_(clip)`` and ``optimizer.step()`` act on g_sam

Exactly one scalar (the global gradient norm) is reduced; the ascent direction is the
registered gradient itself; the restore is a ``copy_`` of saved tensors rather than
``w - e`` (which is not bitwise invertible in floating point). There is no
per-parameter normalization, no second-order term, no sign flip and no ``rho`` sweep.

``sam_rho=None`` is the registered path: the runner takes its own untouched
``loss.backward()`` branch and never consults this module's rules, so the off path
stays the registered one word for word.

Descriptive diagnostics only: nothing here is a scientific threshold.
"""
from __future__ import annotations

import math

import torch

from .r7_ema import require_ema_decay
from .r7_gradient_surgery import REGISTERED_PHYSICAL_WEIGHTS, require_gradient_surgery
from .r7_recursive_losses import DRAFT_SUPERVISION_DEEP, require_gradient_lambda

#: The fixed denominator of the paper's rescaling; it only keeps a zero gradient finite.
SAM_EPS = 1e-12
#: The paper's own default radius (arXiv:2010.01412v3 section 3.1), predeclared here.
DEFAULT_SAM_RHO = 0.05
#: The report fields every SAM update row must carry.
UPDATE_KEYS = (
    "sam_rho", "sam_grad_norm", "sam_perturbation_norm", "sam_perturbation_ratio",
    "sam_loss", "sam_loss_increase", "sam_update_grad_norm", "sam_restore_residual",
)


def require_sam_rho(rho):
    """``None`` is the registered path; otherwise a finite non-negative radius."""
    if rho is None:
        return None
    if isinstance(rho, bool) or not isinstance(rho, (int, float)):
        raise ValueError(f"sam_rho must be None or a finite non-negative float, got {rho!r}")
    value = float(rho)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"sam_rho must be None or a finite non-negative float, got {rho!r}")
    return value


def sam_semantics(sam_rho):
    """The extra contract/report fields for the SAM update rule.

    ``None`` adds nothing, so the registered contract and report stay word for word
    unchanged. A value records the frozen rule. The helper lives next to the rule it
    describes (and keeps the shared runner inside the R-021 file-size target).
    """
    value = require_sam_rho(sam_rho)
    if value is None:
        return {}
    return {"sam_rho": value,
            "sam_eps": SAM_EPS,
            "sam_perturbation": ("e = rho * g / (||g|| + eps); one global rescaling of the "
                                 "registered gradient, no per-parameter normalization"),
            "sam_forwards_per_update": 2,
            "sam_update_gradient": ("gradient of the same registered scalar loss at w + e, "
                                    "after the parameters are restored"),
            "sam_restore": "saved parameters copied back bit for bit before the registered clip"}


def require_sam_axes(*, physical_weights, draft_supervision, gradient_lambda, ema_decay=None,
                     gradient_surgery="off"):
    """The SAM arm cannot be combined with any closed axis (predeclared guard)."""
    weights = tuple(float(weight) for weight in physical_weights)
    if weights != REGISTERED_PHYSICAL_WEIGHTS:
        raise ValueError("SAM requires the registered 12-step physical-weight contour")
    if draft_supervision != DRAFT_SUPERVISION_DEEP:
        raise ValueError("SAM requires draft_supervision='deep'")
    if require_gradient_lambda(gradient_lambda) != 0.0:
        raise ValueError("SAM cannot be combined with the spatial-gradient axis")
    if require_ema_decay(ema_decay) is not None:
        raise ValueError("SAM cannot be combined with the EMA axis")
    if require_gradient_surgery(gradient_surgery) != "off":
        raise ValueError("SAM cannot be combined with the gradient-surgery axis")


def trainable_parameters(model):
    """Every parameter the registered optimizer would see, in model order."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("the SAM update needs at least one trainable parameter")
    return parameters


def global_norm(tensors):
    """One float64 global norm over a sequence of tensors (``None`` counts as zero)."""
    total = torch.zeros((), dtype=torch.float64)
    for tensor in tensors:
        if tensor is None:
            continue
        total = total + tensor.detach().reshape(-1).to(torch.float64).pow(2).sum()
    return float(total.sqrt())


def global_grad_norm(parameters):
    """The float64 global norm of the current gradients; a missing grad counts as zero."""
    return global_norm([parameter.grad for parameter in parameters])


def ascent_direction(parameters, rho):
    """``e = rho * g / (||g|| + eps)`` per parameter, cast back to that parameter's dtype.

    The rescaling factor is a single scalar for the whole model: the norm is global, so
    no per-parameter normalization is involved. A parameter without a gradient gets an
    exactly zero direction.
    """
    scale = float(rho) / (global_grad_norm(parameters) + SAM_EPS)
    return [torch.zeros_like(parameter) if parameter.grad is None
            else parameter.grad.detach() * scale for parameter in parameters]


def perturb(parameters, directions):
    """``w -> w + e`` under ``no_grad``; the caller owns the saved originals."""
    with torch.no_grad():
        for parameter, direction in zip(parameters, directions):
            parameter.add_(direction)


def restore(parameters, saved):
    """Copy the saved originals back bit for bit (``w - e`` would not be exact)."""
    with torch.no_grad():
        for parameter, original in zip(parameters, saved):
            parameter.copy_(original)


def sam_backward(model, *, rho, first_loss, recompute):
    """One SAM update step; assigns the perturbed-point gradient and returns diagnostics.

    ``first_loss`` is the registered scalar loss already evaluated at the current
    parameters and already freed by its own ``backward()``. ``recompute`` must return a
    fresh long-rollout result at the model's *current* parameters; because the first
    graph is freed before this call, the two graphs never coexist. The first pass's
    gradients are cleared so the second ``backward()`` stores the perturbed-point
    gradient instead of accumulating onto the registered one, and the parameters are
    restored even if the second forward or backward raises.
    """
    parameters = trainable_parameters(model)
    grad_norm = global_grad_norm(parameters)
    directions = ascent_direction(parameters, rho)
    perturbation_norm = global_norm(directions)
    saved = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter in parameters:
            parameter.grad = None
    perturb(parameters, directions)
    try:
        result = recompute()
        result.loss.backward()
    finally:
        restore(parameters, saved)
    update_grad_norm = global_grad_norm(parameters)
    residual = max((float((parameter.detach() - original).abs().max())
                    for parameter, original in zip(parameters, saved)), default=0.0)
    return {
        "sam_rho": float(rho),
        "sam_grad_norm": grad_norm,
        "sam_perturbation_norm": perturbation_norm,
        "sam_perturbation_ratio": (perturbation_norm / float(rho)) if rho else None,
        "sam_loss": float(result.loss.detach()),
        "sam_loss_increase": float(result.loss.detach()) - float(first_loss.detach()),
        "sam_update_grad_norm": update_grad_norm,
        "sam_restore_residual": residual,
    }


def summarize_updates(rows):
    """Aggregate the required per-update SAM diagnostics; descriptive, never a gate.

    ``zero_perturbation_updates`` counts updates whose perturbation norm was exactly
    ``0.0`` — the exact condition under which the SAM update degenerates to the
    registered one. That is a description of the run, not a pass threshold.
    """
    if not rows:
        raise ValueError("the SAM summary needs at least one update row")
    for row in rows:
        missing = [key for key in UPDATE_KEYS if key not in row]
        if missing:
            raise RuntimeError(f"update row is missing SAM diagnostics: {missing}")
    perturbations = [float(row["sam_perturbation_norm"]) for row in rows]
    increases = [float(row["sam_loss_increase"]) for row in rows]
    residuals = [float(row["sam_restore_residual"]) for row in rows]
    return {
        "updates": len(rows),
        "zero_perturbation_updates": sum(1 for value in perturbations if value == 0.0),
        "perturbation_norm_min": min(perturbations),
        "perturbation_norm_max": max(perturbations),
        "perturbation_norm_mean": sum(perturbations) / len(perturbations),
        "grad_norm_mean": sum(float(row["sam_grad_norm"]) for row in rows) / len(rows),
        "loss_increase_min": min(increases),
        "loss_increase_max": max(increases),
        "loss_increase_mean": sum(increases) / len(increases),
        "restore_residual_max": max(residuals),
        "all_perturbations_exactly_zero": all(value == 0.0 for value in perturbations),
    }
