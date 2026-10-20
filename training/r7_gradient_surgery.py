"""Short/long gradient surgery for the registered 12-step long-rollout objective.

The registered objective is a single scalar: the weighted sum of twelve physical-step
losses.  This module keeps that scalar definition and splits the *weighted* nonzero
steps into two predeclared groups, then applies the PCGrad rule (arXiv:2001.06782v4,
Algorithm 1) to the two group gradients:

    group A = steps 0/1/3 (6/12/24 h), weights 1.0/0.5/0.5
    group B = steps 7/11 (48/72 h),  weights 0.5/0.5

If ``dot(g_A, g_B) < 0`` then ``g_A <- g_A - (dot / ||g_B||^2) g_B``; otherwise both
groups are unchanged. The update is ``g_A + g_B``. The order is fixed A then B, the
projection is global (one dot over all trainable parameters), and the reductions run
in float64.  Zero-weight steps belong to neither group; they still advance the
physical history exactly as registered.

The off path is deliberately a separate branch in the runner (``loss.backward()``);
this module is only imported when ``gradient_surgery='pcgrad'`` is declared.  The
two-group sum is algebraically the registered total but not bitwise equal to the
registered scalar's backward (different FP association), which the report records as
a descriptive scalar-loss residual.  Diagnostic only: nothing here is a scientific
threshold.

Report definitions: ``group_norm_ratio = ||g_A|| / ||g_B||``;
``projection_relative_norm = ||(dot/||g_B||^2) g_B|| / ||g_A||`` (0 without a
projection); ``pre_clip_total_norm`` is the float64 norm of the update vector and
``clip_factor = min(1, clip/pre_clip_total_norm)`` is its descriptive projection onto
the registered clip (the actual clipping re-derives the norm from the cast grads).
"""
from __future__ import annotations

import torch

from .r7_ema import require_ema_decay
from .r7_recursive_losses import DRAFT_SUPERVISION_DEEP, require_gradient_lambda

GRADIENT_SURGERY_MODES = ("off", "pcgrad")
#: The registered sparse contour, verbatim from study_r7_s3_wide_single_factor.py:38.
REGISTERED_PHYSICAL_WEIGHTS = (1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
GROUP_A_STEPS = (0, 1, 3)      # 6/12/24 h
GROUP_B_STEPS = (7, 11)        # 48/72 h
GROUP_A_WEIGHTS = (1.0, 0.5, 0.5)
GROUP_B_WEIGHTS = (0.5, 0.5)
PROJECTED_GROUP = "A"
#: The report fields every pcgrad update row must carry.
UPDATE_KEYS = (
    "group_a_loss", "group_b_loss", "group_sum_residual",
    "group_a_grad_norm", "group_b_grad_norm", "group_norm_ratio", "group_global_cosine",
    "conflict", "projection_count", "projected_group", "negative_parameter_dots",
    "projection_norm", "projection_relative_norm", "pre_clip_total_norm", "clip_factor",
)


def require_gradient_surgery(mode):
    """Validate the closed mode vocabulary; there is no third mode and no sweep."""
    if mode not in GRADIENT_SURGERY_MODES:
        raise ValueError(f"gradient_surgery must be one of {GRADIENT_SURGERY_MODES}, got {mode!r}")
    return mode


def require_pcgrad_axes(*, physical_weights, draft_supervision, gradient_lambda, ema_decay=None):
    """The candidate cannot be combined with any closed axis (predeclared guard)."""
    weights = tuple(float(weight) for weight in physical_weights)
    if len(weights) != len(REGISTERED_PHYSICAL_WEIGHTS) or any(
            weight != expected for weight, expected in zip(weights, REGISTERED_PHYSICAL_WEIGHTS)):
        raise ValueError("pcgrad requires the registered 12-step physical-weight contour")
    if draft_supervision != DRAFT_SUPERVISION_DEEP:
        raise ValueError("pcgrad requires draft_supervision='deep'")
    if require_gradient_lambda(gradient_lambda) != 0.0:
        raise ValueError("pcgrad cannot be combined with the spatial-gradient axis")
    if require_ema_decay(ema_decay) is not None:
        raise ValueError("pcgrad cannot be combined with the EMA axis")


def group_losses(per_step_losses, physical_weights=REGISTERED_PHYSICAL_WEIGHTS):
    """The two weighted group sums, built from the same ``per_step_losses`` tensor.

    Steps are accumulated in registered index order with the registered weights, so
    the group sums preserve the registered left-to-right association within a group.
    Zero-weight steps are skipped exactly (``0.0 * loss`` contributes no gradient).
    """
    losses = per_step_losses if torch.is_tensor(per_step_losses) else torch.as_tensor(per_step_losses)
    weights = tuple(float(weight) for weight in physical_weights)
    if losses.dim() != 1 or int(losses.numel()) != len(weights):
        raise ValueError("per_step_losses must be a 1-D tensor with one entry per physical weight")
    if weights != REGISTERED_PHYSICAL_WEIGHTS:
        raise ValueError("the group definition is frozen to the registered physical-weight contour")
    group_a = group_b = None
    for index, (weight, step_loss) in enumerate(zip(weights, losses)):
        if weight == 0.0:
            continue
        if index in GROUP_A_STEPS:
            group_a = weight * step_loss if group_a is None else group_a + weight * step_loss
        elif index in GROUP_B_STEPS:
            group_b = weight * step_loss if group_b is None else group_b + weight * step_loss
        else:  # pragma: no cover - guarded by the frozen weight contour
            raise RuntimeError(f"step {index} carries weight {weight} but belongs to neither group")
    if group_a is None or group_b is None:  # pragma: no cover - guarded above
        raise RuntimeError("both gradient-surgery groups must carry at least one weighted step")
    return group_a, group_b


def _flat_gradient(gradient, parameter):
    if gradient is None:
        return torch.zeros(parameter.numel(), dtype=torch.float64, device=parameter.device)
    return gradient.detach().reshape(-1).to(dtype=torch.float64)


def pcgrad_backward(model, per_step_losses, *, physical_weights=REGISTERED_PHYSICAL_WEIGHTS,
                    registered_loss=None, clip=1.0):
    """Apply the two-group PCGrad rule and assign ``.grad``; return diagnostics.

    Two ``torch.autograd.grad`` calls per update: the first retains the graph, the
    second frees it. Dot products and norms are accumulated in float64. The update
    vector is cast back to each parameter's dtype before ``clip_grad_norm_``.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("gradient surgery needs at least one trainable parameter")
    group_a, group_b = group_losses(per_step_losses, physical_weights)
    grads_a = torch.autograd.grad(group_a, parameters, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(group_b, parameters, allow_unused=True)
    flat_a = [_flat_gradient(gradient, parameter) for gradient, parameter in zip(grads_a, parameters)]
    flat_b = [_flat_gradient(gradient, parameter) for gradient, parameter in zip(grads_b, parameters)]
    device = flat_a[0].device
    local_dots = [(a * b).sum() for a, b in zip(flat_a, flat_b)]
    dot = torch.zeros((), dtype=torch.float64, device=device)
    norm_a_sq = torch.zeros((), dtype=torch.float64, device=device)
    norm_b_sq = torch.zeros((), dtype=torch.float64, device=device)
    for value, a, b in zip(local_dots, flat_a, flat_b):
        dot = dot + value
        norm_a_sq = norm_a_sq + (a * a).sum()
        norm_b_sq = norm_b_sq + (b * b).sum()
    negative_parameter_dots = int((torch.stack(local_dots) < 0).sum())
    dot_value = float(dot)
    norm_a = float(torch.sqrt(norm_a_sq))
    norm_b = float(torch.sqrt(norm_b_sq))
    conflict = dot_value < 0.0
    if conflict:
        coefficient = dot / norm_b_sq
        projected = [a - coefficient * b for a, b in zip(flat_a, flat_b)]
        projection_sq = torch.zeros((), dtype=torch.float64, device=device)
        for b in flat_b:
            projection_sq = projection_sq + ((coefficient * b) ** 2).sum()
        projection_norm = float(torch.sqrt(projection_sq))
        projection_relative = projection_norm / norm_a if norm_a > 0.0 else 0.0
        projection_count, projected_group = 1, PROJECTED_GROUP
    else:
        projected = flat_a
        projection_norm = projection_relative = 0.0
        projection_count, projected_group = 0, None
    update = [a + b for a, b in zip(projected, flat_b)]
    total_sq = torch.zeros((), dtype=torch.float64, device=device)
    for value in update:
        total_sq = total_sq + (value * value).sum()
    pre_clip_total_norm = float(torch.sqrt(total_sq))
    clip_factor = min(1.0, float(clip) / pre_clip_total_norm) if pre_clip_total_norm > 0.0 else 1.0
    for parameter, value, gradient_a, gradient_b in zip(parameters, update, grads_a, grads_b):
        if gradient_a is None and gradient_b is None:
            parameter.grad = None
        else:
            parameter.grad = value.reshape(parameter.shape).to(dtype=parameter.dtype)
    group_sum = float((group_a + group_b).detach())
    registered = float(registered_loss.detach()) if registered_loss is not None else group_sum
    cosine = dot_value / (norm_a * norm_b) if norm_a > 0.0 and norm_b > 0.0 else 0.0
    return {
        "group_a_loss": float(group_a.detach()), "group_b_loss": float(group_b.detach()),
        "group_sum_residual": abs(group_sum - registered),
        "group_a_grad_norm": norm_a, "group_b_grad_norm": norm_b,
        "group_norm_ratio": norm_a / norm_b if norm_b > 0.0 else None,
        "group_global_cosine": cosine,
        "conflict": bool(conflict), "projection_count": projection_count,
        "projected_group": projected_group,
        "negative_parameter_dots": negative_parameter_dots,
        "projection_norm": projection_norm,
        "projection_relative_norm": projection_relative,
        "pre_clip_total_norm": pre_clip_total_norm, "clip_factor": clip_factor,
    }


def summarize_updates(rows):
    """Aggregate the required per-update diagnostics; descriptive, never a gate."""
    if not rows:
        raise ValueError("the surgery summary needs at least one update row")
    for row in rows:
        missing = [key for key in UPDATE_KEYS if key not in row]
        if missing:
            raise RuntimeError(f"update row is missing surgery diagnostics: {missing}")
    cosines = [float(row["group_global_cosine"]) for row in rows]
    ratios = [float(row["group_norm_ratio"]) for row in rows
              if row["group_norm_ratio"] is not None]
    projected = [float(row["projection_relative_norm"]) for row in rows if row["conflict"]]
    conflicts = sum(1 for row in rows if row["conflict"])
    return {
        "updates": len(rows), "conflict_updates": conflicts,
        "conflict_ratio": conflicts / len(rows),
        "projection_count": sum(int(row["projection_count"]) for row in rows),
        "cosine_min": min(cosines), "cosine_mean": sum(cosines) / len(cosines),
        "cosine_max": max(cosines),
        "norm_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "projection_relative_norm_mean": (sum(projected) / len(projected)) if projected else 0.0,
        "negative_parameter_dot_updates": sum(
            1 for row in rows if int(row["negative_parameter_dots"]) > 0),
        "group_a_grad_norm_mean": sum(float(row["group_a_grad_norm"]) for row in rows) / len(rows),
        "group_b_grad_norm_mean": sum(float(row["group_b_grad_norm"]) for row in rows) / len(rows),
        "group_sum_residual_max": max(float(row["group_sum_residual"]) for row in rows),
    }
