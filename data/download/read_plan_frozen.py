"""Frozen ERA5 data-v2 read plan: scope, splits, and pre-read cost model (#63).

Metadata-first: this module freezes the v2 scope and computes the byte cost of
a planned read **before** any network access, from source chunk geometry alone.
Nothing here downloads, writes data, or reports a scientific claim.

Scope freeze (from the published contract, not re-derived from prose):
- region 0.25deg, 65x65 points, 27-43N / 107-123E
  (`arco_regional_bounded.ROI_*`);
- the 17-channel order comes from `r7_era5.DEFAULT_R7_ERA5_CHANNELS` and is
  asserted, never rewritten;
- candidate splits: train 2016-2018, validation 2019; 2020 already had
  development access and must never be presented as unseen test; the test
  candidate 2021 requires an access/overlap audit before it can be sealed;
- D1 acquires 30 consecutive days from the candidate train years for
  window/unit/IO verification only - no journal technique is claimable from D1;
- halo may be built only from t and its history; future ERA5 fields are never
  used as boundary forcing;
- mean/std/climatology/process normalizations fit on train years only;
- new sources, times or variables require an independent identity; the
  original source pins stay in place.

Actual HTTP bytes are reported as None until measured; decoded-byte estimates
must never be presented as network measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from training.r7_experiment import canonical_digest

GRID_HEIGHT = GRID_WIDTH = 65
GRID_SPACING_DEG = 0.25
ROI_SOUTH, ROI_NORTH, ROI_WEST, ROI_EAST = 27.0, 43.0, 107.0, 123.0
TRAIN_YEARS = (2016, 2017, 2018)
VAL_YEARS = (2019,)
TEST_CANDIDATE_YEARS = (2021,)
DEV_ACCESSED_YEARS = (2020,)
D1_DAYS = 30
D1_START = "2016-01-01T00:00"
INIT_HOURS_UTC = (0, 6, 12, 18)
CADENCE_HOURS = 6
HISTORY_STEPS = 2
LEAD_HOURS = (6, 12, 24, 48, 72)
MAX_LEAD_HOURS = 72
FIRST_STAGE_NEW_ARTIFACT_BYTES_CAP = 16 * 2**30
FIRST_STAGE_DECODED_BYTES_CAP = 64 * 2**30


def channel_plan():
    """The 17-channel v2 plan straight from the published channel spec.

    Returns one row per channel with its source variable, pressure level and
    stored name, and asserts that every level used is inside the audited
    13-level set and that geopotential stays geopotential (m**2 s**-2, never
    silently swapped for geopotential height).
    """
    from data.download.arco_regional_bounded import PRESSURE_LEVELS_HPA
    from data.preprocess.r7_era5 import DEFAULT_R7_ERA5_CHANNELS

    rows = []
    for spec in DEFAULT_R7_ERA5_CHANNELS:
        rows.append({
            "variable": spec.variable,
            "level_hpa": spec.level_hpa,
            "channel": spec.name,
        })
    if len(rows) != 17:
        raise ValueError(f"the published channel spec must have 17 channels, got {len(rows)}")
    levels = {row["level_hpa"] for row in rows if row["level_hpa"] is not None}
    if not levels <= set(PRESSURE_LEVELS_HPA):
        raise ValueError(f"levels outside the audited 13-level set: {sorted(levels)}")
    if any(row["channel"] == "z850" for row in rows):
        # geopotential identity guard: stored under m**2 s**-2 in the source
        # contract; a conversion to geopotential height would be a different
        # channel and must be declared, never silently swapped.
        from data.download.arco_regional_bounded import SOURCE_UNITS
        if SOURCE_UNITS.get("geopotential") != "m**2 s**-2":
            raise ValueError("geopotential unit drift; z must stay geopotential")
    return rows


def complete_windows(total_steps, history_steps=HISTORY_STEPS,
                     max_lead_steps=None, cadence_hours=CADENCE_HOURS,
                     stride_steps=1, gaps=()):
    """Count and place complete windows in a continuous time axis.

    A window needs `history_steps` past frames plus enough future frames for
    the longest lead, and must not cross a gap (a missing frame index) or a
    split boundary (boundaries are expressed as gaps). This is the offline
    check that D1 windows exist and that +72h targets are available.
    """
    if max_lead_steps is None:
        max_lead_steps = MAX_LEAD_HOURS // cadence_hours
    if history_steps < 1 or max_lead_steps < 1 or stride_steps < 1:
        raise ValueError("history, lead and stride must be positive")
    span = history_steps - 1 + max_lead_steps
    if total_steps <= span:
        return 0, []
    gaps = set(gaps)
    starts = []
    for start in range(0, total_steps - span, stride_steps):
        used = range(start, start + span + 1)
        if any(index in gaps for index in used):
            continue
        starts.append(start)
    return len(starts), starts


def init_hour_coverage(starts, cadence_hours=CADENCE_HOURS,
                       init_hours_UTC=INIT_HOURS_UTC, base_hour=0):
    """Windows grouped by their UTC initialization hour."""
    coverage = {hour: 0 for hour in init_hours_UTC}
    for start in starts:
        hour = ((base_hour + start * cadence_hours) % 24)
        if hour in coverage:
            coverage[hour] += 1
    return coverage


@dataclass(frozen=True)
class SourceLayout:
    """Chunk geometry of one source variable, as recorded by a prior probe."""

    name: str
    dims: tuple
    shape: tuple
    chunks: tuple
    dtype_bytes: int

    def __post_init__(self):
        if not (len(self.dims) == len(self.shape) == len(self.chunks)):
            raise ValueError(f"{self.name}: dims/shape/chunks rank mismatch")
        if any(c < 1 for c in self.chunks) or any(s < 1 for s in self.shape):
            raise ValueError(f"{self.name}: invalid shape/chunks")
        if self.dtype_bytes < 1:
            raise ValueError(f"{self.name}: invalid dtype size")


def arco_global_per_time_cost(layouts, times):
    """Decoded bytes for ARCO global-per-time layout (chunk-1: one whole-globe
    field per variable per time), plus the cropped bytes actually retained.

    The chunk union touched is every requested time per requested variable;
    spatial crops ride along free because the chunk already spans the globe.
    http_bytes stays None: the plan budgets decoded bytes, it does not measure
    the network.
    """
    if times < 1:
        raise ValueError("at least one time is required")
    decoded = 0
    cropped = 0
    fields = 0
    for layout in layouts:
        time_index = layout.dims.index("time") if "time" in layout.dims else None
        if time_index is None:
            raise ValueError(f"{layout.name}: no time dimension; not a per-time layout")
        field_cells = math.prod(layout.shape) // layout.shape[time_index]
        fields += 1
        decoded += layout.dtype_bytes * field_cells * times
        cropped += layout.dtype_bytes * GRID_HEIGHT * GRID_WIDTH * times
    return {
        "layout": "arco-global-per-time",
        "times": times,
        "fields": fields,
        "chunk_union_touched": fields * times,
        "decoded_bytes": decoded,
        "cropped_bytes": cropped,
        "disk_peak_bytes": decoded + cropped,
        "http_bytes": None,
    }


def earthmover_temporal_tile_cost(layouts, times):
    """Decoded bytes for the Earthmover temporal-tile layout.

    Each variable is tiled over time (chunk covers `chunks[time_index]`
    consecutive times), so the touched chunk union is per variable the ceil of
    requested times over the tile length, and each touched chunk decodes
    fully.
    """
    if times < 1:
        raise ValueError("at least one time is required")
    decoded = 0
    cropped = 0
    touched = 0
    for layout in layouts:
        if "time" not in layout.dims:
            raise ValueError(f"{layout.name}: no time dimension")
        time_index = layout.dims.index("time")
        tile_length = layout.chunks[time_index]
        cells_per_time = math.prod(layout.chunks) // tile_length
        chunks_touched = math.ceil(times / tile_length)
        touched += chunks_touched
        decoded += chunks_touched * cells_per_time * tile_length * layout.dtype_bytes
        cropped += layout.dtype_bytes * GRID_HEIGHT * GRID_WIDTH * times
    return {
        "layout": "earthmover-temporal-tiles",
        "times": times,
        "chunk_union_touched": touched,
        "decoded_bytes": decoded,
        "cropped_bytes": cropped,
        "disk_peak_bytes": decoded + cropped,
        "http_bytes": None,
    }


def d1_request(days=D1_DAYS, start=D1_START, cadence_hours=CADENCE_HOURS):
    """The D1 continuous 30-day segment: explicit init/valid-time coverage."""
    start_time = datetime.fromisoformat(start)
    steps = days * 24 // cadence_hours
    times = [start_time + timedelta(hours=cadence_hours * i) for i in range(steps)]
    init_times = [t for t in times if t.hour in INIT_HOURS_UTC]
    return {"days": days, "start": start, "steps": steps,
            "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
            "init_times": len(init_times),
            "season": "Jan" if start_time.month == 1 else f"{start_time.month:02d}"}


def frozen_protocol(source_pins=(), cost_rows=()):
    """Machine-readable frozen plan; hashed with the canonical digest."""
    body = {
        "format": "r7-era5-v2-read-plan-v1",
        "frozen_before_any_download": True,
        "scientific_claim": False,
        "region": {"south": ROI_SOUTH, "north": ROI_NORTH,
                   "west": ROI_WEST, "east": ROI_EAST,
                   "points": [GRID_HEIGHT, GRID_WIDTH],
                   "spacing_deg": GRID_SPACING_DEG},
        "channels": [row["channel"] for row in channel_plan()],
        "channel_count": 17,
        "splits": {"train_years": list(TRAIN_YEARS), "val_years": list(VAL_YEARS),
                   "test_candidate_years": list(TEST_CANDIDATE_YEARS),
                   "dev_accessed_years_excluded_from_test": list(DEV_ACCESSED_YEARS)},
        "d1": d1_request(),
        "cadence_hours": CADENCE_HOURS,
        "history_steps": HISTORY_STEPS,
        "lead_hours": list(LEAD_HOURS),
        "init_hours_utc": list(INIT_HOURS_UTC),
        "halo_rule": "halo only from t and its history; future ERA5 is never "
                     "used as boundary forcing",
        "normalization_rule": "mean/std, climatology and process-diagnostic "
                              "normalization fit on train years only; any new "
                              "normalization is versioned",
        "identity_rule": "read-only snapshots; original source pins kept; new "
                         "sources/times/variables require an independent identity",
        "case_identity": "explicit init/valid-time lists per #60; equal counts "
                         "never imply the same cases",
        "budget_caps": {"new_artifacts_bytes": FIRST_STAGE_NEW_ARTIFACT_BYTES_CAP,
                        "decoded_source_bytes": FIRST_STAGE_DECODED_BYTES_CAP},
        "source_pins": list(source_pins),
        "cost_rows": list(cost_rows),
        "limitations": [
            "byte costs are decoded-source estimates from recorded chunk "
            "geometry; http_bytes stays null until actually measured",
            "D1 is an engineering segment; no journal technique is claimable",
            "the 2021 test candidate stays unsealed until its access/overlap "
            "audit passes",
        ],
    }
    return dict(body, protocol_sha256=canonical_digest(body))
