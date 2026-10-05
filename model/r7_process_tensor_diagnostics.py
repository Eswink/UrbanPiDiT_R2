"""Differentiable spherical proxies matching data.preprocess.process_diagnostics.

These are regional diagnostics, not causal labels or closed physical budgets.
Sensitive differences/reductions run outside autocast in FP32 (FP64 is retained
only when the caller supplies FP64, for analytic engineering oracles).

This module lives under ``model/`` because the #79 typed-evidence pathway calls
the operators from inside the model's forward path: ``model_code_sha256`` covers
``model/**.py`` and only those bytes, so a forward-path numerical dependency
under ``training/`` would sit outside every checkpoint's identity digest. The
supervision code that consumes the 8-d vector imports it from here.
"""
from __future__ import annotations

from typing import Sequence

import torch

from data.preprocess.process_diagnostics import (
    EARTH_RADIUS_M, KAPPA_DRY_AIR, PROCESS_DIAGNOSTIC_NAMES,
    REQUIRED_PROCESS_CHANNELS,
)

PROCESS_DIAGNOSTIC_UNITS = (
    "Pa m-1", "s-1", "s-1", "K s-1", "kg kg-1 s-1", "kg kg-1 s-1",
    "K", "m s-1",
)


def _field(value: torch.Tensor) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim < 2 or min(value.shape) < 1):
        raise ValueError("finite floating fields with nonempty spatial dimensions required")
    result = value if value.dtype == torch.float64 else value.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite diagnostic field")
    return result


def _axis(values, name: str, field: torch.Tensor, length: int) -> torch.Tensor:
    raw = torch.as_tensor(values, device=field.device)
    if (raw.dtype == torch.bool or raw.is_complex() or raw.ndim not in (1, 2)
            or raw.shape[-1] != length or length < 2):
        raise ValueError(f"{name} must be a numeric [L] or [B,L] axis with >= 2 points")
    if raw.ndim == 2 and (field.ndim < 3 or raw.shape[0] != field.shape[0]):
        raise ValueError(f"{name} batch/field shape mismatch")
    axis = raw.to(field.dtype)
    if not bool(torch.isfinite(axis).all()):
        raise ValueError(f"{name} contains nonfinite values")
    delta = axis.diff(dim=-1)
    monotone = (delta > 0).all(-1) | (delta < 0).all(-1)
    if not bool(monotone.all()):
        raise ValueError(f"{name} must be strictly monotone")
    if name == "latitude":
        if bool((axis.abs() > 90).any()):
            raise ValueError("latitude outside [-90,90]")
    else:
        convention = ((axis >= -180) & (axis <= 180)).all(-1)
        convention |= ((axis >= 0) & (axis < 360)).all(-1)
        if not bool(convention.all()) or bool((axis.amax(-1) - axis.amin(-1) >= 360).any()):
            raise ValueError("longitude must use one non-wrapped convention without duplicate seam")
    return torch.deg2rad(axis)


def _axis_grid(axis: torch.Tensor, field: torch.Tensor, dim: int) -> torch.Tensor:
    shape = [1] * field.ndim
    shape[dim] = axis.shape[-1]
    if axis.ndim == 2:
        shape[0] = axis.shape[0]
    return axis.reshape(shape)


def _coordinates(field: torch.Tensor, latitude, longitude):
    phi = _axis(latitude, "latitude", field, field.shape[-2])
    lam = _axis(longitude, "longitude", field, field.shape[-1])
    cosine = _axis_grid(phi.cos(), field, -2)
    if bool((cosine.abs() < 1e-4).any()):
        raise ValueError("process diagnostics do not support near-pole grids")
    return _axis_grid(phi, field, -2), _axis_grid(lam, field, -1), cosine


def _gradient(field: torch.Tensor, coordinate: torch.Tensor, dim: int) -> torch.Tensor:
    """Nonuniform differences, second-order edges for L>=3, first-order for L=2."""
    length = field.shape[dim]

    def take(value, start, count=1):
        return value.narrow(dim, start, count)

    spacing = coordinate.diff(dim=dim)
    if length == 2:
        slope = (take(field, 1) - take(field, 0)) / spacing
        return torch.cat((slope, slope), dim=dim)
    # Algebraically the numpy coefficients, expressed as adjacent slopes so a
    # large constant pressure/temperature offset cancels BEFORE FP32 products.
    slopes = field.diff(dim=dim) / spacing
    left, right = take(spacing, 0, length - 2), take(spacing, 1, length - 2)
    interior = (right * take(slopes, 0, length - 2)
                + left * take(slopes, 1, length - 2)) / (left + right)
    a, b = take(spacing, 0), take(spacing, 1)
    first = ((2 * a + b) * take(slopes, 0) - a * take(slopes, 1)) / (a + b)
    a, b = take(spacing, length - 3), take(spacing, length - 2)
    last = ((a + 2 * b) * take(slopes, length - 2) - b * take(slopes, length - 3)) / (a + b)
    return torch.cat((first, interior, last), dim=dim)


def _scalar_gradient(field, phi, lam, cosine):
    return (_gradient(field, lam, -1) / (EARTH_RADIUS_M * cosine),
            _gradient(field, phi, -2) / EARTH_RADIUS_M)


def _divergence(u, v, phi, lam, cosine):
    return (_gradient(u, lam, -1) + _gradient(v * cosine, phi, -2)) / (EARTH_RADIUS_M * cosine)


def _vorticity(u, v, phi, lam, cosine):
    return (_gradient(v, lam, -1) - _gradient(u * cosine, phi, -2)) / (EARTH_RADIUS_M * cosine)


def _matching_fields(*fields):
    values = tuple(_field(value) for value in fields)
    if any(value.shape != values[0].shape or value.device != values[0].device
           or value.dtype != values[0].dtype for value in values[1:]):
        raise ValueError("diagnostic fields must have identical shapes, dtype and device")
    return values


def spherical_scalar_gradient(field, latitude, longitude):
    field = _field(field)
    with torch.autocast(field.device.type, enabled=False):
        return _scalar_gradient(field, *_coordinates(field, latitude, longitude))


def spherical_divergence(u, v, latitude, longitude):
    u, v = _matching_fields(u, v)
    with torch.autocast(u.device.type, enabled=False):
        return _divergence(u, v, *_coordinates(u, latitude, longitude))


def spherical_vorticity(u, v, latitude, longitude):
    u, v = _matching_fields(u, v)
    with torch.autocast(u.device.type, enabled=False):
        return _vorticity(u, v, *_coordinates(u, latitude, longitude))


def horizontal_advection(scalar, u, v, latitude, longitude):
    scalar, u, v = _matching_fields(scalar, u, v)
    with torch.autocast(scalar.device.type, enabled=False):
        dx, dy = _scalar_gradient(scalar, *_coordinates(scalar, latitude, longitude))
        return -(u * dx + v * dy)


def _mean(field, weights):
    return (field * weights).sum((-2, -1)) / weights.sum((-2, -1))


#: Field order of the #79 local evidence: four of the eight frozen proxies kept
#: as maps instead of being reduced to regional scalars.
LOCAL_FIELD_NAMES = (
    "divergence_850", "vorticity_850",
    "temperature_advection_850", "static_stability_850_500",
)

#: The channels :func:`typed_local_fields` reads. A store must carry all four;
#: nothing else from the diagnostic set is touched.
TYPED_EVIDENCE_CHANNELS = ("t500", "t850", "u850", "v850")


def typed_local_fields(
    state: torch.Tensor, channel_names: Sequence[str], latitude, longitude,
) -> torch.Tensor:
    """``[...,4,H,W]`` physical-unit local fields for the #79 typed evidence path.

    Four of the eight proxies of :func:`compute_process_tensor_diagnostics`, kept
    as spatial maps: 850 hPa divergence and vorticity of ``(u850, v850)``,
    horizontal advection of ``t850`` by ``(u850, v850)``, and the 850-500 hPa
    static-stability difference. Same operators, FP32 policy, nonperiodic
    boundaries and unit conventions as the scalar diagnostics; this is the
    single definition both the model's evidence module and the train-only
    sidecar fitter call.
    """
    state = _field(state)
    names = tuple(channel_names)
    if (state.ndim < 3 or len(names) != state.shape[-3]
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("unique channel names must match [...,C,H,W] state")
    missing = [name for name in TYPED_EVIDENCE_CHANNELS if name not in names]
    if missing:
        raise KeyError(f"missing typed-evidence channels: {missing}")
    with torch.autocast(state.device.type, enabled=False):
        fields = dict(zip(names, state.unbind(dim=-3)))
        phi, lam, cosine = _coordinates(fields["t850"], latitude, longitude)
        u, v = fields["u850"], fields["v850"]
        tx, ty = _scalar_gradient(fields["t850"], phi, lam, cosine)
        stability = (fields["t500"] * (1000. / 500.) ** KAPPA_DRY_AIR
                     - fields["t850"] * (1000. / 850.) ** KAPPA_DRY_AIR)
        result = torch.stack((
            _divergence(u, v, phi, lam, cosine), _vorticity(u, v, phi, lam, cosine),
            -(u * tx + v * ty), stability,
        ), dim=-3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite typed local field")
        return result


def _rms(field, weights):
    # vector_norm has a finite zero subgradient; sqrt(sum(x*x)) does not.
    return (torch.linalg.vector_norm(field * weights.sqrt(), dim=(-2, -1))
            / weights.sum((-2, -1)).sqrt())


def compute_process_tensor_diagnostics(
    state: torch.Tensor, channel_names: Sequence[str], latitude, longitude,
) -> torch.Tensor:
    """Return [...,8] physical-unit proxies from state [...,C,H,W].

    Axes are degrees, either shared [H]/[W] or per first batch axis [B,H]/[B,W].
    Nonperiodic boundaries match the numpy oracle; no wrapped seam or poles.
    Channel order is declared explicitly, never inferred from a tensor width.
    """
    state = _field(state)
    names = tuple(channel_names)
    if (state.ndim < 3 or len(names) != state.shape[-3]
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("unique channel names must match [...,C,H,W] state")
    missing = [name for name in REQUIRED_PROCESS_CHANNELS if name not in names]
    if missing:
        raise KeyError(f"missing process diagnostic channels: {missing}")
    with torch.autocast(state.device.type, enabled=False):
        fields = dict(zip(names, state.unbind(dim=-3)))
        pressure = fields["mslp"]
        phi, lam, cosine = _coordinates(pressure, latitude, longitude)
        weights = cosine.expand_as(pressure)
        dx, dy = _scalar_gradient(pressure, phi, lam, cosine)
        u, v = fields["u850"], fields["v850"]
        divergence = _divergence(u, v, phi, lam, cosine)
        vorticity = _vorticity(u, v, phi, lam, cosine)
        tx, ty = _scalar_gradient(fields["t850"], phi, lam, cosine)
        qx, qy = _scalar_gradient(fields["q850"], phi, lam, cosine)
        moisture = -_divergence(fields["q850"] * u, fields["q850"] * v, phi, lam, cosine)
        stability = (fields["t500"] * (1000. / 500.) ** KAPPA_DRY_AIR
                     - fields["t850"] * (1000. / 850.) ** KAPPA_DRY_AIR)
        strength = torch.linalg.vector_norm(torch.stack((dx, dy)), dim=0)
        shear = torch.linalg.vector_norm(torch.stack((fields["u500"] - u, fields["v500"] - v)), dim=0)
        result = torch.stack((
            _mean(strength, weights), _rms(divergence, weights), _rms(vorticity, weights),
            _mean(-(u * tx + v * ty), weights), _mean(-(u * qx + v * qy), weights),
            _mean(moisture, weights), _mean(stability, weights), _mean(shear, weights),
        ), dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite diagnostic output")
        return result
