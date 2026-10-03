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
requested lead. An explicit optional ``init_calendar_year`` selects Gregorian
valid-date/year-length arithmetic; without it the archived 365.25-day harmonics
are unchanged. This module imports no clock source, guesses no calendar year,
and missing required fields raise rather than becoming placeholders.

``SPACETIME_INPUT_FIELDS`` declares the required fields; ``CALENDAR_INPUT_FIELDS``
declares optional known calendar metadata. The whitelist and rollout import both,
so no path can quietly choose another input set. Legacy ``init_year`` stays unread.

``field_mode`` adds the two control arms a capacity attribution needs, *inside*
this module and after the fields have been validated - the dataset, the rollout
and every whitelisted path keep carrying the real fields, and only what the
conditioning MLP is shown changes:

- ``fields`` (default) the real fields, i.e. the implementation that existed
  before the switch;
- ``constant`` the same-shape zeros: the module is present, trained and costs the
  same compute, but its input carries no information, so its contribution is one
  constant additive vector;
- ``shuffled`` the per-sample fields rolled by one along the sample axis: real
  values, wrong pairing.

Neither control arm reads a clock, a target or a future observation, and neither
is a fallback: a missing field still raises before any of this runs.
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
# Optional, explicitly known calendar metadata. init_year remains undeclared;
# absence preserves the archived 365.25-day harmonic, never a guessed year.
CALENDAR_INPUT_FIELDS = ("init_calendar_year",)
FIELD_MODES = ("fields", "constant", "shuffled")
PHASE_FEATURES = 4
POSITION_FEATURES = 4
DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24.0


def require_field_mode(mode: str) -> str:
    """The declared field modes, and nothing else.

    A typo in a study arm is a silent experiment change unless it raises, so this
    is the single place the accepted set is written down and every construction
    path goes through it.
    """
    if not isinstance(mode, str) or mode not in FIELD_MODES:
        raise ValueError(f"field mode must be one of {FIELD_MODES}, got {mode!r}")
    return mode


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


def apply_field_mode(
    mode: str,
    latitude: torch.Tensor,
    longitude: torch.Tensor,
    hour: torch.Tensor,
    day: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace *validated* fields for a control arm; never invent a value.

    ``constant`` returns same-shape zeros, so the conditioning term collapses to
    one vector for every token and every sample - module capacity and compute,
    with no information in the input.

    ``shuffled`` rolls the per-sample fields by one along the sample axis,
    deterministically, so each sample is conditioned on another sample's
    initialization time while the marginal distribution of the values is
    untouched. A batch of one has no other sample to borrow from, so the roll is
    the identity there. That is a property of any within-batch permutation, and it
    is relied on rather than hidden: the validation and published-evaluation paths
    score one window per forward pass, and an arm that raised on them could not be
    scored at all; the arms that use this mode declare the property in their
    protocol. The grid coordinates are batch-invariant by construction -
    ``require_spacetime_fields`` rejects a batch whose samples disagree about the
    grid - so a within-batch roll has nothing to move there, and rotating a
    coordinate axis instead would be a different manipulation than the one this
    mode declares.
    """
    mode = require_field_mode(mode)
    if mode == "fields":
        return latitude, longitude, hour, day
    if mode == "constant":
        return (torch.zeros_like(latitude), torch.zeros_like(longitude),
                torch.zeros_like(hour), torch.zeros_like(day))
    if batch_size < 2:
        return latitude, longitude, hour, day
    return (latitude, longitude,
            torch.roll(hour, 1, dims=0), torch.roll(day, 1, dims=0))


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


def _calendar_year_days(year: torch.Tensor) -> torch.Tensor:
    leap = (year.remainder(4) == 0) & ((year.remainder(100) != 0) | (year.remainder(400) == 0))
    return 365 + leap.to(torch.int64)


def advance_calendar_time(
    init_calendar_year: torch.Tensor,
    init_day_of_year: torch.Tensor,
    init_utc_hour: torch.Tensor,
    lead_hours: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Gregorian (year, ordinal day, UTC hour) from explicitly known fields.

    Integer day arithmetic and the 400/100/4/1-year decomposition avoid any clock
    or host calendar library. All samples advance independently, including year
    boundaries, century exceptions and fractional-hour leads. Invalid metadata
    raises; init_year and a missing calendar year are never interpreted here.
    """
    if not torch.is_tensor(lead_hours) or not lead_hours.is_floating_point():
        raise TypeError("calendar lead_hours must be a floating tensor")
    lead = lead_hours.reshape(-1).to(torch.float64)
    values = [_per_sample(value, name, lead.numel()).to(device=lead.device, dtype=torch.float64)
              for value, name in ((init_calendar_year, "init_calendar_year"),
                                  (init_day_of_year, "init_day_of_year"),
                                  (init_utc_hour, "init_utc_hour"))]
    year, day, hour = values
    for name, value in zip(("init_calendar_year", "init_day_of_year", "init_utc_hour",
                            "lead_time_hours"), (*values, lead)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if bool(((year != year.floor()) | (year < 1) | (year > 9999)).any()):
        raise ValueError("init_calendar_year must be an integer in [1, 9999]")
    year = year.to(torch.int64)
    if bool(((day != day.floor()) | (day < 1) | (day > _calendar_year_days(year))).any()):
        raise ValueError("init_day_of_year must be an integer valid for init_calendar_year")
    if bool(((hour < 0) | (hour >= HOURS_PER_DAY)).any()):
        raise ValueError("init_utc_hour must be in [0, 24)")
    previous = year - 1
    start = (365 * previous + previous // 4 - previous // 100 + previous // 400)
    elapsed_hours = hour + lead
    carry = torch.floor(elapsed_hours / HOURS_PER_DAY)
    ordinal = start.to(torch.float64) + day - 1 + carry
    if bool(((ordinal < 0) | (ordinal >= 3_652_059)).any()):
        raise ValueError("calendar valid time must remain in years [1, 9999]")
    ordinal = ordinal.to(torch.int64)
    centuries400 = ordinal // 146097
    remainder = ordinal.remainder(146097)
    centuries = (remainder // 36524).clamp(max=3)
    remainder = remainder - centuries * 36524
    quadrennia = remainder // 1461
    remainder = remainder - quadrennia * 1461
    years = (remainder // 365).clamp(max=3)
    valid_year = 400 * centuries400 + 100 * centuries + 4 * quadrennia + years + 1
    valid_day = remainder - 365 * years + 1
    valid_hour = (elapsed_hours - carry * HOURS_PER_DAY).to(lead_hours.dtype)
    return valid_year, valid_day, valid_hour


def phase_features(
    init_utc_hour: torch.Tensor,
    init_day_of_year: torch.Tensor,
    lead_hours: torch.Tensor,
    *,
    init_calendar_year: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[B, 4]`` sin/cos annual and diurnal harmonics of the *valid* time.

    The valid time is ``init + lead`` for both terms: a 24 h lead lands on the
    next day at the same hour and the harmonics repeat, and a 6 h lead moves the
    annual phase by a quarter of a day, so nothing here is computed from the
    initialization time alone and nothing is read from a clock. Without explicit
    calendar metadata the archived 365.25-day approximation is retained byte for
    byte. With init_calendar_year, UTC is advanced to an actual Gregorian valid
    date and the annual denominator is that valid year's length.
    """
    hour = init_utc_hour.reshape(-1).to(lead_hours.dtype)
    day = init_day_of_year.reshape(-1).to(lead_hours.dtype)
    if hour.shape != lead_hours.shape or day.shape != lead_hours.shape:
        raise ValueError("phase inputs must be one value per sample")
    if init_calendar_year is not None:
        valid_year, valid_day, valid_hour = advance_calendar_time(
            init_calendar_year, init_day_of_year, init_utc_hour, lead_hours)
        annual = math.tau * (valid_day.to(lead_hours.dtype) - 1.0
                            + valid_hour / HOURS_PER_DAY) / _calendar_year_days(valid_year)
        diurnal = math.tau * valid_hour / HOURS_PER_DAY
    else:
        # Operation-for-operation archived behavior; do not retrofit old results.
        valid_hour = hour + lead_hours
        annual = math.tau * (day - 1.0 + valid_hour / HOURS_PER_DAY) / DAYS_PER_YEAR
        diurnal = math.tau * valid_hour / HOURS_PER_DAY
    return torch.stack([torch.sin(annual), torch.cos(annual),
                        torch.sin(diurnal), torch.cos(diurnal)], dim=-1)


class SpacetimeConditioning(nn.Module):
    """Additive per-token space-time term for the forecast token grid (#71 M1)."""

    def __init__(self, dim: int, patch_size: int, periodic_width: bool = False,
                 default_lead_hours: float = 6.0, hidden: int | None = None,
                 field_mode: str = "fields"):
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.periodic_width = bool(periodic_width)
        self.default_lead_hours = float(default_lead_hours)
        self.field_mode = require_field_mode(field_mode)
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
        calendar_year = None
        if "init_calendar_year" in batch:
            calendar_year = _per_sample(batch["init_calendar_year"], "init_calendar_year",
                                        history.shape[0])
            # Validate original metadata even when a control arm will discard it.
            advance_calendar_time(calendar_year, day, hour, torch.zeros_like(hours).reshape(-1))
            if self.field_mode == "shuffled":
                calendar_year = torch.roll(calendar_year, 1, dims=0)
            elif self.field_mode == "constant":
                # Explicit capacity control, not a missing-metadata fallback:
                # discard the calendar too and retain the archived constant input.
                calendar_year = None
        latitude, longitude, hour, day = apply_field_mode(
            self.field_mode, latitude, longitude, hour, day,
            batch_size=history.shape[0])
        position = position_features(
            latitude.detach().to(torch.float32), longitude.detach().to(torch.float32),
            token_hw=token_hw, patch_size=self.patch_size,
            periodic_width=self.periodic_width)
        phase = phase_features(hour.to(torch.float32), day.to(torch.float32),
                               hours.reshape(-1), init_calendar_year=calendar_year)
        features = torch.cat([
            position.unsqueeze(0).expand(history.shape[0], -1, -1),
            phase.unsqueeze(1).expand(-1, position.shape[0], -1),
        ], dim=-1)
        return self.net(features.to(history.dtype))
