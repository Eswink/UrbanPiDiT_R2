"""Explicit Gregorian history slots and local mean-solar phase (ADR 0035).

Pure tensor arithmetic, no clocks/targets. The residual is opt-in; the archived
8-feature space-time network and its parameters are not changed. History slot
order is preserved in [annual sin/cos, solar sin/cos, signed hours/24] blocks,
followed by valid-time solar sin/cos. Solar harmonics are computed on native
longitudes before the encoder's padding and patch averaging.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from .layers.patch_grid import pad_patch_grid
from .spacetime_conditioning_r7 import (
    advance_calendar_time, phase_features, require_field_mode, require_spacetime_fields,
)

HISTORY_CONTEXT_FIELDS = ("history_offsets_hours",)


def uses_known_context(model):
    """Explicit optional capability query; unsupported external stubs stay legacy.

    Actual enhanced models expose a strict boolean. True never falls back when
    their metadata/module is absent; required-field validation then fails closed.
    """
    forecaster = getattr(model, "forecaster", model)
    if not hasattr(forecaster, "known_context_inputs"):
        return False
    enabled = forecaster.known_context_inputs
    if type(enabled) is not bool:
        raise ValueError("known_context_inputs must be boolean")
    return enabled


def _numeric_tensor(value, name, device):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError(f"{name} must have a real numeric dtype")
    if value.device != device:
        raise ValueError(f"{name} must be on the history device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach()


def _half_ulp(value):
    """Half the larger finite adjacent spacing in value's ORIGINAL float dtype.

    Convert neighbours before subtraction: low-precision subtraction can overflow.
    At a finite dtype endpoint use the finite neighbour, not an infinite bound.
    """
    widened = value.to(torch.float64)
    above = torch.nextafter(value, torch.full_like(value, float("inf"))).to(torch.float64)
    below = torch.nextafter(value, torch.full_like(value, -float("inf"))).to(torch.float64)
    upper = torch.where(torch.isfinite(above), above - widened, torch.zeros_like(widened))
    lower = torch.where(torch.isfinite(below), widened - below, torch.zeros_like(widened))
    return torch.maximum(upper, lower) / 2


def require_history_offsets(value, *, batch_size, history_steps, device,
                            batched=True, cadence_hours=None):
    """Strict [T]/[B,T] signed offsets; never infer times from sequence indices.

    A regular positive history cadence is independent of forecast lead. Physical
    rollouts additionally supply cadence_hours, because dropping one frame and
    appending a prediction is only valid when the two cadences match. Regularity
    allows only propagated original-dtype half-ULP rounding, not an rtol/atol.
    """
    native = _numeric_tensor(value, "history_offsets_hours", device)
    expected = (batch_size, history_steps) if batched else (history_steps,)
    if tuple(native.shape) != expected:
        raise ValueError(f"history_offsets_hours must be {expected}, got {tuple(native.shape)}")
    if not bool((native[..., -1] == 0).all()):
        raise ValueError("history_offsets_hours must end at zero")
    offsets = native.to(torch.float64)
    if history_steps > 1:
        differences = offsets[..., 1:] - offsets[..., :-1]
        if not bool((differences > 0).all()):
            raise ValueError("history_offsets_hours must be strictly increasing")
        error = torch.zeros_like(differences)
        if native.is_floating_point():
            rounding = _half_ulp(native)
            rounding[..., -1] = 0  # The required terminal zero is exact, not uncertain.
            # d_i = x_{i+1} - x_i: two input rounding cells plus FP64 subtraction.
            error = rounding[..., 1:] + rounding[..., :-1] + _half_ulp(differences)
        residual = differences - differences[..., :1]
        bound = error + error[..., :1]
        if native.is_floating_point():
            bound = bound + _half_ulp(residual)
        if not bool((residual.abs() <= bound).all()):
            raise ValueError("history_offsets_hours must have regular positive cadence")
        if cadence_hours is not None:
            residual = differences - cadence_hours
            bound = error
            if native.is_floating_point():
                cadence = torch.as_tensor(cadence_hours, dtype=torch.float64, device=device)
                bound = bound + _half_ulp(cadence) + _half_ulp(residual)
            if not bool((residual.abs() <= bound).all()):
                raise ValueError("history_offsets_hours cadence must equal physical step cadence")
    return offsets


def _sample(value, name, history):
    tensor = _numeric_tensor(value, name, history.device)
    size = history.shape[0]
    if tensor.ndim == 0:
        return tensor.expand(size).to(torch.float64)
    if tensor.shape != (size,):
        raise ValueError(f"{name} must be scalar or [{size}]")
    return tensor.to(torch.float64)


def require_known_context_fields(batch: Mapping, *, history):
    """Return latitude, longitude, hour, day, year, offsets, lead after validation."""
    if not torch.is_tensor(history) or history.ndim != 5 or min(history.shape) < 1:
        raise ValueError("known context needs nonempty [B,T,C,H,W] history")
    if not history.is_floating_point() or not torch.isfinite(history).all():
        raise ValueError("known context needs finite floating history")
    names = ("latitude", "longitude", "init_utc_hour", "init_day_of_year",
             "init_calendar_year", "history_offsets_hours", "lead_time_hours")
    missing = [name for name in names if name not in batch]
    if missing:
        raise KeyError(f"known_context_inputs requires explicit {missing}; no clock/legacy fallback")
    for name in ("latitude", "longitude"):
        coordinate = _numeric_tensor(batch[name], name, history.device)
        if coordinate.ndim == 2 and coordinate.shape[0] != history.shape[0]:
            raise ValueError(f"{name} batch dimension must match history")
    latitude, longitude, _, _ = require_spacetime_fields(
        batch, history_shape=history.shape[-2:], batch_size=history.shape[0])
    hour, day, year, lead = [_sample(batch[name], name, history) for name in (
        "init_utc_hour", "init_day_of_year", "init_calendar_year", "lead_time_hours")]
    if bool((latitude.abs() > 90).any()):
        raise ValueError("latitude must be in [-90,90]")
    if bool((lead < 0).any()):
        raise ValueError("lead_time_hours must be nonnegative")
    offsets = require_history_offsets(batch["history_offsets_hours"],
        batch_size=history.shape[0], history_steps=history.shape[1], device=history.device)
    advance_calendar_time(year, day, hour, torch.zeros_like(lead))
    return latitude, longitude, hour, day, year, offsets, lead


def local_solar_features(utc_hour, longitude, *, native_hw, token_hw, patch_size,
                         periodic_width=False):
    """[B,N,2]: mean of native sin/cos, not sin/cos of mean longitude."""
    angle = math.tau * utc_hour[:, None, None] / 24.0 + torch.deg2rad(longitude)[None, None, :]
    harmonics = torch.stack([angle.sin(), angle.cos()], dim=1).expand(-1, -1, native_hw[0], -1)
    padded = pad_patch_grid(harmonics, patch_size, periodic_width=periodic_width)
    rows, columns = token_hw
    if tuple(padded.shape[-2:]) != (rows * patch_size, columns * patch_size):
        raise ValueError("known-context padded grid must match encoder token_hw")
    pooled = padded.reshape(utc_hour.numel(), 2, rows, patch_size,
                            columns, patch_size).mean(dim=(3, 5))
    return pooled.flatten(2).transpose(1, 2)


def known_context_features(batch, *, history, token_hw, patch_size, periodic_width=False,
                           field_mode="fields"):
    """[B,N,5*T+2] slot-ordered features, computed and validated before controls."""
    mode = require_field_mode(field_mode)
    _, longitude, hour, day, year, offsets, lead = require_known_context_fields(batch, history=history)
    # Match archived controls: metadata pairings move, declared lead stays put.
    # Validate both original and shuffled date ranges, never turn invalid fields
    # into a capacity-control fallback.
    for index in range(history.shape[1]):
        advance_calendar_time(year, day, hour, offsets[:, index])
    advance_calendar_time(year, day, hour, lead)
    if mode == "shuffled":
        hour, day, year, offsets = [value.roll(1, dims=0) for value in (hour, day, year, offsets)]
    slots = []
    for index in range(history.shape[1]):
        elapsed = offsets[:, index]
        annual = phase_features(hour, day, elapsed, init_calendar_year=year)[:, :2]
        _, _, historical_hour = advance_calendar_time(year, day, hour, elapsed)
        solar = local_solar_features(historical_hour, longitude.to(torch.float64),
            native_hw=history.shape[-2:], token_hw=token_hw, patch_size=patch_size,
            periodic_width=periodic_width)
        slots.append(torch.cat([annual[:, None].expand(-1, solar.shape[1], -1), solar,
                                (elapsed / 24.0)[:, None, None].expand(-1, solar.shape[1], 1)], -1))
    _, _, valid_hour = advance_calendar_time(year, day, hour, lead)
    valid_solar = local_solar_features(valid_hour, longitude.to(torch.float64),
        native_hw=history.shape[-2:], token_hw=token_hw, patch_size=patch_size,
        periodic_width=periodic_width)
    features = torch.cat([*slots, valid_solar], dim=-1).to(torch.float32)
    if mode == "constant":
        return torch.zeros_like(features)
    return features


class KnownContextConditioning(nn.Module):
    """One small context residual; no carry storage and no changes to old8 weights."""
    def __init__(self, dim, history_steps, patch_size, periodic_width=False, field_mode="fields"):
        super().__init__()
        self.history_steps = int(history_steps)
        self.patch_size = int(patch_size)
        self.periodic_width = bool(periodic_width)
        self.field_mode = require_field_mode(field_mode)
        self.projection = nn.Linear(5 * self.history_steps + 2, int(dim))

    def forward(self, batch, *, history, token_hw):
        if history.shape[1] != self.history_steps:
            raise ValueError("known-context history_steps must match configured slot count")
        features = known_context_features(batch, history=history, token_hw=token_hw,
            patch_size=self.patch_size, periodic_width=self.periodic_width, field_mode=self.field_mode)
        return self.projection(features.to(history.dtype))
