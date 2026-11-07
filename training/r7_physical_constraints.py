"""Stage C1's untargeted physical constraints: divergence, humidity sign, dew point vs T.

Decision 0092 fixes what this module is and, more importantly, what it is *not*:

* **Not the closed spatial-gradient loss.** That penalty pushes ``‖∇pred‖²`` toward zero, so
  its global optimum is a constant field -- it must trade forecast skill for smoothness, which
  is what the measured three-seed regression showed. The three terms here are satisfied by the
  real atmosphere, so their optimum set *contains* the truth, and on any already-satisfying
  field each returns **exactly 0.0**. That difference -- degenerate optimum versus
  truth-containing optimum -- is the family argument, and it is checkable rather than
  rhetorical.
* **Untargeted.** No term reads a target, a future observation or a clock. They constrain the
  prediction against physical relations only.
* **Not imported from legacy.** ``legacy_v531_full/losses/physical_consistency.py`` bundles
  FFT, gradient, diurnal and drag terms that are closed or unauthorised; this is a new module.
  Its weights are *quoted constants* from that file (``w_wind_div = 0.1``,
  ``w_dew_leq_t = 0.5``), not imported code.

Each term uses ``relu`` rather than ``softplus`` on purpose: a smooth hinge keeps charging a
field that already satisfies the constraint, which would quietly make these "constraints" into
smoothing priors -- the exact failure mode this round must avoid.
"""
from __future__ import annotations

from typing import Mapping

import torch

#: Quoted from legacy (``physical_consistency.py:68-70``), not imported.
W_DIVERGENCE = 0.1
W_DEW_POINT = 0.5
#: This round's choice: legacy has no separate specific-humidity term. Declared, not inherited
#: (decision 0092 Context ④); the evidence page must carry the same disclosure.
W_HUMIDITY = 0.5
#: Numerical tolerance for the divergence term, in the operator's own units (s^-1).
#: Real atmospheres are not exactly non-divergent layer by layer, so penalising all of it would
#: be penalising the truth. Frozen before any run; never tuned against a reading.
DIVERGENCE_TOLERANCE = 1e-5


def divergence_penalty(u, v, latitude, longitude, *, tolerance=DIVERGENCE_TOLERANCE):
    """``mean(relu(|∇·(u,v)| − tolerance)²)`` -- zero for a non-divergent field.

    Reuses the repository's single spherical-divergence operator so this constraint and the
    diagnostics cannot drift apart.
    """
    from model.r7_process_tensor_diagnostics import spherical_divergence
    field = spherical_divergence(u, v, latitude, longitude)
    excess = (field.abs() - float(tolerance)).clamp_min(0.0)
    return (excess ** 2).mean()


def humidity_nonnegative(q):
    """``mean(relu(−q)²)`` -- zero wherever specific humidity is non-negative."""
    return (q.clamp_max(0.0) ** 2).mean()


def dew_point_le_temperature(dew_point, temperature):
    """``mean(relu(Td − T)²)`` -- zero wherever the dew point does not exceed temperature."""
    return ((dew_point - temperature).clamp_min(0.0) ** 2).mean()


def physical_constraint_terms(prediction, latitude, longitude, *,
                              weights: Mapping[str, float] | None = None):
    """The three terms plus their weighted sum, from a denormalised prediction.

    ``prediction`` is ``[..., C, H, W]`` in physical units with named channels. Every term is
    computed from that one tensor; nothing here reads a target.

    Returns a dict with each raw term, the weights used, and ``total``. A caller that wants the
    constraint to be *off* must not call this at all -- there is no "weight 0" path, so an
    off arm cannot accidentally construct a tensor.
    """
    resolved = {"divergence": W_DIVERGENCE, "humidity": W_HUMIDITY, "dew_point": W_DEW_POINT}
    if weights is not None:
        unknown = set(weights) - set(resolved)
        if unknown:
            raise KeyError(f"unknown constraint weights: {sorted(unknown)}")
        resolved.update(weights)
    channels = prediction
    if channels.ndim < 3:
        raise ValueError("prediction must be [..., C, H, W]")
    # Channel order is declared by the caller's tensor layout, never inferred from a width.
    from data.preprocess.process_diagnostics import REQUIRED_PROCESS_CHANNELS
    if channels.shape[-3] < len(REQUIRED_PROCESS_CHANNELS):
        raise ValueError(f"prediction carries {channels.shape[-3]} channels, the divergence "
                         f"operator needs the named set {REQUIRED_PROCESS_CHANNELS}")
    fields = dict(zip(REQUIRED_PROCESS_CHANNELS, channels.unbind(dim=-3)))
    divergence = divergence_penalty(fields["u850"], fields["v850"], latitude, longitude)
    humidity = humidity_nonnegative(fields["q850"])
    # The store has no dew-point channel; Td is derived from q850 and t850 by the standard
    # Magnus relation, which is the same convention the legacy feasibility term used.
    dew_point = _dew_point_celsius(fields["q850"], fields["t850"])
    dew = dew_point_le_temperature(dew_point, fields["t850"])
    total = (resolved["divergence"] * divergence + resolved["humidity"] * humidity
             + resolved["dew_point"] * dew)
    return {"divergence": divergence, "humidity": humidity, "dew_point": dew,
            "weights": dict(resolved), "total": total}


def _dew_point_celsius(q, temperature):
    """Dew point in the temperature channel's own units, via the Magnus relation.

    ``q`` is specific humidity (kg/kg). Vapour pressure ``e = q·p/(0.622 + 0.378q)`` needs a
    pressure; the store's ``mslp`` is not the 850 hPa level pressure, so the standard
    approximation ``Td ≈ T − (1 − RH)/0.05`` would silently import a wrong pressure. Instead the
    dew point is obtained from the saturation vapour pressure curve inverted at the vapour
    pressure implied by ``q`` at 850 hPa, with that pressure declared as a constant.
    """
    pressure = 85000.0          # Pa at the 850 hPa level this term is defined on
    vapour = q.clamp_min(0.0) * pressure / (0.622 + 0.378 * q.clamp_min(0.0))
    # Magnus: e = 6.112 hPa * exp(17.67 Td / (Td + 243.5)), inverted for Td in Celsius.
    e_hpa = (vapour / 100.0).clamp_min(1e-12)
    log_term = torch.log(e_hpa / 6.112)
    dew_c = 243.5 * log_term / (17.67 - log_term)
    # ``temperature`` is in Kelvin on this store (t850), so return Kelvin to compare like with like.
    return dew_c + 273.15


def denormalize(fields, normalization, channels):
    """Map a normalized ``[..., C, H, W]`` tensor back to physical units.

    The training path never sees the store's normalization constants -- the dataset normalises
    inside ``__getitem__`` -- so the constants have to be handed in explicitly and their digest
    recorded in the protocol. That makes "which mean/std were used" an auditable field rather
    than an implicit one, which matters because it changes every constraint's magnitude.

    ``channels`` is the store's own channel order, declared by the caller, never inferred from
    the tensor's width: a silently reordered store must fail here rather than rescale the wrong
    variable.
    """
    mean, std = normalization
    if mean.shape != std.shape or mean.shape[0] != len(channels):
        raise ValueError(f"normalization has {mean.shape[0]} entries, channels has {len(channels)}")
    if fields.shape[-3] != len(channels):
        raise ValueError(f"fields carry {fields.shape[-3]} channels, channels names {len(channels)}")
    shape = (1,) * (fields.ndim - 3) + (len(channels), 1, 1)
    return fields * std.reshape(shape) + mean.reshape(shape)


def constraint_term(result, moved, contract):
    """Stage C1's untargeted constraint on the *last* forecast, or ``None`` when off.

    ``None`` is returned without constructing anything when the axis is off, so the
    registered path never touches this code and cannot acquire a numerical difference from
    it. When on, the term is computed on the denormalised last forecast: the training batch
    carries no normalization constants (the dataset normalises inside ``__getitem__``), so
    the caller must pass them in the contract and their digest is frozen in the protocol.

    Only the last forecast is constrained. Constraining every physical step would multiply
    the cost by the step count and mix "does the constraint help" with "how often is it
    applied"; decision 0092 fixes one application per update, which is what the brief's
    single-factor claim needs.
    """
    axis = contract["autoregression"].get("physical_constraints")
    if axis is None:
        return None
    if axis != "c1":
        raise ValueError(f"physical_constraints must be absent or 'c1', got {axis!r}")
    import numpy as np

    from .r7_physical_constraints import denormalize, physical_constraint_terms
    normalization = contract["autoregression"].get("normalization")
    channels = contract["autoregression"].get("channels")
    if not normalization or not channels:
        raise ValueError("the c1 axis requires the store's normalization and channel order")
    mean = torch.as_tensor(np.asarray(normalization["mean"], dtype=np.float32))
    std = torch.as_tensor(np.asarray(normalization["std"], dtype=np.float32))
    forecast = result.forecasts[:, -1]
    physical = denormalize(forecast, (mean.to(forecast.device), std.to(forecast.device)),
                           tuple(channels))
    latitude, longitude = moved.get("latitude"), moved.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError("the c1 axis requires latitude and longitude in the batch")
    return physical_constraint_terms(physical, latitude, longitude)
