"""Stage C2's derivative-domain target matching: prediction's divergence/vorticity vs the target's.

Decision 0095 fixes what this is and how it differs from both closed neighbours:

* **Not the closed spatial-gradient loss.** That penalty reads no target and pushes ``‖∇pred‖²``
  toward zero, so it charges the truth (a real field has gradients) and its optimum is a constant
  field. This module reads the target and is *exactly zero when the prediction equals it* -- the
  optimum set is the truth itself, which is the family difference decision 0084 section 6 asks for.
* **Not Stage C1.** C1 constrained the *magnitude* of divergence without reading a target, and
  decision 0094 measured that the analysis field is charged 10.2x more than the model, so its
  descent direction pointed away from truth. C2 compares prediction against target instead, so
  the inequality that killed C1 does not apply here at all: at ``pred = target`` both terms are 0.

**Dimensionless, because decision 0094 measured what happens otherwise.** C1's first form squared
absolute violations and added ``s^-2`` to a normalized-space MSE; measured at the round's real
starting point it was ``1e-11 .. 2.2e-06`` against ``L_mse ~ 1`` with parameter-gradient ratios
down to ``2.7e-10`` -- the axis could not act, and a null would have been a dose artefact. Each
term here divides by a declared physical scale before squaring. Measured pre-freeze on four train
batches at the p1600 endpoint, the pair then comes to 7.1x-12.9x ``L_mse``: live, not inert.

The scales are *derived from the analysis field's own distribution* (mean ``|div|`` and
``|vort|`` over the train batches), recorded in the protocol, and never tuned against a reading.
"""
from __future__ import annotations

import torch

#: This round's choice, declared rather than inherited: legacy has no derivative-domain *target
#: matching* term to quote. Its ``w_wind_div`` is an untargeted divergence-magnitude penalty,
#: which decision 0094 refuted on this store. Equal weights, no prior reason to prefer one.
W_DIVERGENCE = 0.1
W_VORTICITY = 0.1


def divergence_matching(prediction, target, latitude, longitude, *, scale, weight=W_DIVERGENCE):
    """``weight * mean(((∇·pred − ∇·target)/scale)²)`` -- exactly zero when ``pred == target``.

    Reuses the repository's single spherical-divergence operator so this term and the diagnostics
    cannot drift apart. ``scale`` is in s^-1 and is derived from the analysis distribution.
    """
    from model.r7_process_tensor_diagnostics import spherical_divergence
    if float(scale) <= 0.0:
        raise ValueError("the divergence scale must be positive")
    predicted = spherical_divergence(prediction[0], prediction[1], latitude, longitude)
    observed = spherical_divergence(target[0], target[1], latitude, longitude)
    return float(weight) * (((predicted - observed) / float(scale)) ** 2).mean()


def vorticity_matching(prediction, target, latitude, longitude, *, scale, weight=W_VORTICITY):
    """``weight * mean(((∇×pred − ∇×target)/scale)²)`` -- exactly zero when ``pred == target``."""
    from model.r7_process_tensor_diagnostics import spherical_vorticity
    if float(scale) <= 0.0:
        raise ValueError("the vorticity scale must be positive")
    predicted = spherical_vorticity(prediction[0], prediction[1], latitude, longitude)
    observed = spherical_vorticity(target[0], target[1], latitude, longitude)
    return float(weight) * (((predicted - observed) / float(scale)) ** 2).mean()


def derivative_matching_terms(prediction, target, channel_names, latitude, longitude, *,
                              scales, weights=None):
    """Both terms plus their sum, from denormalised ``[..., C, H, W]`` fields.

    ``channel_names`` is the store's own channel order and is matched **by name**, never by
    position: the store's order is ``t2m,u10,v10,mslp,z850,t850,q850,u850,v850,...`` while the
    derivative operators want ``u850``/``v850``. Decision 0094 found that a positional zip handed
    the wrong fields to every term while still returning plausible numbers, so a missing name is a
    ``KeyError`` here rather than a silent mis-read.

    Both tensors must carry the same channel count and names; the target is the analysis field
    for the same window, in the same units.
    """
    resolved = {"divergence": W_DIVERGENCE, "vorticity": W_VORTICITY}
    if weights is not None:
        unknown = set(weights) - set(resolved)
        if unknown:
            raise KeyError(f"unknown derivative-matching weights: {sorted(unknown)}")
        resolved.update(weights)
    missing_scales = set(resolved) - set(scales)
    if missing_scales:
        raise KeyError(f"missing scales for {sorted(missing_scales)}")
    for name, value in (("prediction", prediction), ("target", target)):
        if value.ndim < 3:
            raise ValueError(f"{name} must be [..., C, H, W]")
        if value.shape[-3] != len(channel_names):
            raise ValueError(f"{name} carries {value.shape[-3]} channels but "
                             f"{len(channel_names)} names were declared")
    index = {name: position for position, name in enumerate(channel_names)}
    absent = [name for name in ("u850", "v850") if name not in index]
    if absent:
        raise KeyError(f"the fields are missing {absent}; they declare {list(channel_names)}")
    u, v = index["u850"], index["v850"]
    divergence = divergence_matching(
        (prediction[..., u, :, :], prediction[..., v, :, :]),
        (target[..., u, :, :], target[..., v, :, :]),
        latitude, longitude, scale=scales["divergence"], weight=resolved["divergence"])
    vorticity = vorticity_matching(
        (prediction[..., u, :, :], prediction[..., v, :, :]),
        (target[..., u, :, :], target[..., v, :, :]),
        latitude, longitude, scale=scales["vorticity"], weight=resolved["vorticity"])
    return {"divergence": divergence, "vorticity": vorticity,
            "weights": dict(resolved), "scales": dict(scales),
            "total": divergence + vorticity}


def matching_term(result, moved, contract):
    """Stage C2's derivative match on the *last* forecast, or ``None`` when the axis is off.

    ``None`` is returned before constructing anything, so a run with the axis unset keeps the
    registered arithmetic bit-for-bit -- a "weight 0" path would still build a graph and still
    change the numbers.

    The target is the batch's own ``atmos_target`` for the same window, i.e. the last physical
    step's analysis field. Both sides are denormalised with the store's declared constants: the
    training batch carries no normalization (the dataset normalises inside ``__getitem__``), so
    the caller must pass them in the contract and their digest is frozen in the protocol. Only
    the last forecast is matched, one application per update, so the single-factor claim is
    about one application rather than a dose (decision 0095).
    """
    axis = contract["autoregression"].get("derivative_matching")
    if axis is None:
        return None
    if axis != "c2":
        raise ValueError(f"derivative_matching must be absent or 'c2', got {axis!r}")
    import numpy as np

    from .r7_physical_constraints import denormalize
    normalization = contract["autoregression"].get("normalization")
    channels = contract["autoregression"].get("channels")
    scales = contract["autoregression"].get("derivative_scales")
    if not normalization or not channels or not scales:
        raise ValueError("the c2 axis requires the store's normalization, channel order and "
                         "the derivative scales")
    mean = torch.as_tensor(np.asarray(normalization["mean"], dtype=np.float32))
    std = torch.as_tensor(np.asarray(normalization["std"], dtype=np.float32))
    forecast = result.forecasts[:, -1]
    device = forecast.device
    physical = denormalize(forecast, (mean.to(device), std.to(device)), tuple(channels))
    target = moved.get("atmos_target")
    if target is None:
        raise ValueError("the c2 axis requires atmos_target in the batch")
    observed = denormalize(target, (mean.to(device), std.to(device)), tuple(channels))
    latitude, longitude = moved.get("latitude"), moved.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError("the c2 axis requires latitude and longitude in the batch")
    return derivative_matching_terms(physical, observed, tuple(channels), latitude, longitude,
                                     scales=dict(scales))
