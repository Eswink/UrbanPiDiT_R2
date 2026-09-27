"""Known initialization-time conditioning for the main forecaster (#71 M1).

Three things were true before this module existed: the main model saw only
``coarse_history`` and a lead time; its encoder has no positional encoding at
all; and the dataset already carried ``latitude``/``longitude`` while nothing
consumed them. #71 asks for the *known* conditions of a forecast - where, when
and how far ahead - to reach the model as inputs, computed from information
available at the initialization time and never from a future observation, a
target, or the machine clock.

Two deliberate design choices:

- **Per-token geography, not a domain-mean scalar.** ``latitude``/``longitude``
  are averaged inside each patch cell and enter as sin/cos harmonics *per token*.
  Collapsing them to one vector per sample would be another lead-like scalar and
  would throw away exactly the spatial structure this input exists to provide.
- **One joint embedding.** Position and phase are concatenated and passed
  through a single small MLP, so a latitude-season interaction is representable.
  Insolation is such an interaction, so separating the two terms would forbid the
  physically obvious one.

The phase is a pure function of ``init_day_of_year``, ``init_utc_hour`` and the
requested lead (``init_utc_hour + lead``, wrapped by the harmonics themselves).
This module imports no clock source, and a missing field raises instead of
falling back to the wall clock or to a placeholder.

``SPACETIME_INPUT_FIELDS`` is the one declaration of which fields that is; the
whitelist in ``model/r7_halting.py`` and the rollout in ``model/r7_rollout.py``
both import it, so a path cannot quietly carry a different set.
"""
from __future__ import annotations

import contextlib
import math
from typing import Mapping

import torch
from torch import nn

from .coarse_forecast import resolve_lead_hours
from .layers.patch_grid import pad_patch_grid

SPACETIME_INPUT_FIELDS = ("latitude", "longitude", "init_utc_hour", "init_day_of_year")
PHASE_FEATURES = 4
POSITION_FEATURES = 4
DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24.0


@contextlib.contextmanager
def isolated_stream():
    """Construct submodules without consuming the caller's random stream.

    A switch that adds a pathway must not shift the stream every pre-existing
    parameter is drawn from: otherwise two arms built from one seed - one with the
    pathway, one without - differ in *all* their weights and the comparison stops
    being a comparison of the pathway. The added module still receives a normal,
    seed-dependent draw; the stream is simply rewound afterwards, so the modules
    constructed later see exactly the numbers they would have seen.
    """
    state = torch.random.get_rng_state()
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


def require_spacetime_fields(
    batch: Mapping[str, torch.Tensor],
    *,
    history_shape: tuple[int, int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and return (latitude, longitude, hour_of_day, day_of_year).

    Failure is explicit by design: a sample without these fields cannot be
    conditioned on, and the alternatives are exactly the ones #71 forbids -
    reading the server clock, or substituting a constant. There is no default
    branch here to fall into.
    """
    missing = [name for name in SPACETIME_INPUT_FIELDS if name not in batch]
    if missing:
        raise KeyError(
            f"space-time conditioning requires {missing}; a missing field is not "
            "replaced by the server clock or by a placeholder value")
    rows, columns = history_shape
    latitude = _grid_axis(batch["latitude"], rows, "latitude")
    longitude = _grid_axis(batch["longitude"], columns, "longitude")
    hour = _per_sample(batch["init_utc_hour"], "init_utc_hour", batch_size)
    day = _per_sample(batch["init_day_of_year"], "init_day_of_year", batch_size)
    if not torch.isfinite(hour).all() or not torch.isfinite(day).all():
        raise ValueError("init_utc_hour/init_day_of_year must be finite")
    if bool((hour < 0).any()) or bool((hour >= HOURS_PER_DAY).any()):
        raise ValueError("init_utc_hour must be in [0, 24)")
    if bool((day < 1).any()) or bool((day > 366).any()):
        raise ValueError("init_day_of_year must be in [1, 366]")
    return latitude, longitude, hour, day


def _per_sample(value, name: str, batch_size: int) -> torch.Tensor:
    """A declared per-sample field as [B]; never broadcast a mismatched length."""
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor, not {type(value)!r}")
    tensor = value.detach()
    if tensor.ndim == 0:
        return tensor.expand(batch_size).reshape(batch_size)
    flat = tensor.reshape(-1)
    if flat.numel() != batch_size:
        raise ValueError(f"{name} has {flat.numel()} values for a batch of {batch_size}; "
                         "a mismatched length is not broadcast")
    return flat


def _grid_axis(value, size: int, name: str) -> torch.Tensor:
    """One coordinate axis of the grid, as [size].

    A batch collates the coordinates into ``[B, size]`` even though every sample
    describes the same grid, and a single window carries them as ``[size]``. The
    axis is what the patch-averaging needs, so both are accepted; a batch whose
    samples disagree about the grid is rejected, because one forward pass has one
    token grid and silently using the first sample's coordinates would condition
    the other samples on a grid that is not theirs.
    """
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor, not {type(value)!r}")
    tensor = value.detach()
    if tensor.ndim == 1:
        axis = tensor
    elif tensor.ndim == 2 and tensor.shape[-1] == size and tensor.shape[0] >= 1:
        axis = tensor[0]
        if not torch.equal(tensor, axis.unsqueeze(0).expand_as(tensor)):
            raise ValueError(f"{name} differs between the samples of one batch; one "
                             "forward pass has one token grid")
    else:
        raise ValueError(f"{name} must be [{size}] or [B, {size}], got {tuple(tensor.shape)}")
    if axis.shape[0] != size:
        raise ValueError(f"{name} must have {size} values, got {axis.shape[0]}")
    if not torch.isfinite(axis).all():
        raise ValueError(f"{name} must be finite")
    return axis


def position_features(
    latitude: torch.Tensor,
    longitude: torch.Tensor,
    *,
    token_hw: tuple[int, int],
    patch_size: int,
    periodic_width: bool = False,
) -> torch.Tensor:
    """``[N, 4]`` sin/cos harmonics of each token's own patch-averaged coordinates.

    Patch cells average their native coordinates with the same padding the
    encoder applies to the fields (``pad_patch_grid``), so token *i* describes
    the same patch cell the encoder produced it from.
    """
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("position features need one-dimensional latitude/longitude")
    grid = torch.stack([
        latitude.reshape(-1, 1).expand(-1, longitude.shape[0]),
        longitude.reshape(1, -1).expand(latitude.shape[0], -1),
    ], dim=0).unsqueeze(0)
    padded = pad_patch_grid(grid, patch_size, periodic_width=periodic_width)
    rows, columns = token_hw
    if tuple(padded.shape[-2:]) != (rows * patch_size, columns * patch_size):
        raise ValueError("padded coordinate grid does not match the encoder token grid")
    cells = padded.reshape(1, 2, rows, patch_size, columns, patch_size).mean(dim=(3, 5))
    radians = torch.deg2rad(cells)
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=1).reshape(4, rows * columns).transpose(0, 1)


def phase_features(
    init_utc_hour: torch.Tensor,
    init_day_of_year: torch.Tensor,
    lead_hours: torch.Tensor,
) -> torch.Tensor:
    """``[B, 4]`` sin/cos annual and diurnal harmonics of the *valid* time.

    The valid time is ``init + lead`` for both terms: a 24 h lead lands on the
    next day at the same hour and the harmonics repeat, and a 6 h lead moves the
    annual phase by a quarter of a day, so nothing here is computed from the
    initialization time alone and nothing is read from a clock. The wrap across
    midnight and across the year end is the harmonics' own periodicity rather
    than a modular-arithmetic special case.
    """
    hour = init_utc_hour.reshape(-1).to(lead_hours.dtype)
    day = init_day_of_year.reshape(-1).to(lead_hours.dtype)
    if hour.shape != lead_hours.shape or day.shape != lead_hours.shape:
        raise ValueError("phase inputs must be one value per sample")
    valid_hour = hour + lead_hours
    annual = math.tau * (day - 1.0 + valid_hour / HOURS_PER_DAY) / DAYS_PER_YEAR
    diurnal = math.tau * valid_hour / HOURS_PER_DAY
    return torch.stack([torch.sin(annual), torch.cos(annual),
                        torch.sin(diurnal), torch.cos(diurnal)], dim=-1)


class SpacetimeConditioning(nn.Module):
    """Additive per-token space-time term for the forecast token grid (#71 M1)."""

    def __init__(self, dim: int, patch_size: int, periodic_width: bool = False,
                 default_lead_hours: float = 6.0, hidden: int | None = None):
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.periodic_width = bool(periodic_width)
        self.default_lead_hours = float(default_lead_hours)
        width = int(hidden) if hidden is not None else max(8, self.dim // 2)
        if width < 1:
            raise ValueError("hidden width must be positive")
        self.net = nn.Sequential(
            nn.Linear(POSITION_FEATURES + PHASE_FEATURES, width), nn.SiLU(),
            nn.Linear(width, self.dim))

    def forward(self, batch: Mapping[str, torch.Tensor], *,
                history: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
        rows, columns = history.shape[-2:]
        latitude, longitude, hour, day = require_spacetime_fields(
            batch, history_shape=(rows, columns), batch_size=history.shape[0])
        hours = resolve_lead_hours(batch.get("lead_time_hours"), batch=history.shape[0],
                                   device=history.device, dtype=torch.float32,
                                   default_hours=self.default_lead_hours)
        position = position_features(
            latitude.detach().to(torch.float32), longitude.detach().to(torch.float32),
            token_hw=token_hw, patch_size=self.patch_size,
            periodic_width=self.periodic_width)
        phase = phase_features(hour.to(torch.float32), day.to(torch.float32),
                               hours.reshape(-1))
        features = torch.cat([
            position.unsqueeze(0).expand(history.shape[0], -1, -1),
            phase.unsqueeze(1).expand(-1, position.shape[0], -1),
        ], dim=-1)
        return self.net(features.to(history.dtype))
