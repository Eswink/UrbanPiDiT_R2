"""Frozen wide-ROI ERA5 read plan for the S3 lateral-context instance.

An independent identity, not a rewrite of ``read_plan_frozen``: that module pins
the 65x65 27-43N/107-123E box and its digest is archived in the D1/B2/S1/S2/S3
acquisition receipts. This module declares a *containing* wider region whose
central 65x65 block is exactly that frozen box, so a wide instance can be
trained on the wider field while the frozen target box stays scoreable at its
already-registered geometry (``training/r7_boundary_metrics.boundary_masks``
with a 32-cell margin gives exactly that interior).

Nothing here downloads, writes data or reports a scientific claim. Every byte
figure is a pre-read estimate from recorded chunk geometry; measured network
bytes are reported only by the downloader's own receipts.

The physical hypothesis this instance exists to test is recorded in
``docs/goals/main-model-climatology-campaign.md``: at 48-72 h the target box is
narrower than the distance a synoptic system travels, so the frozen 16-degree
domain may not contain the information the long leads need. Enlarging the region
is the single factor changed here; the model spec, the 17-channel plan, the
cadence, the splits and the frozen target box are unchanged.
"""
from __future__ import annotations

from training.r7_experiment import canonical_digest

from .read_plan_frozen import (
    CADENCE_HOURS,
    HISTORY_STEPS,
    INIT_HOURS_UTC,
    LEAD_HOURS,
    MAX_LEAD_HOURS,
)

# The frozen target box (read_plan_frozen.ROI_*): never re-derived, asserted.
TARGET_SOUTH, TARGET_NORTH, TARGET_WEST, TARGET_EAST = 27.0, 43.0, 107.0, 123.0
TARGET_POINTS = 65
# Wide region: the target box grown by exactly one half-width (32 cells = 8 deg)
# in every direction. The central block of the wide grid is therefore the target
# box itself, cell for cell, with no interpolation and no re-gridding.
TARGET_MARGIN_CELLS = 32
SPACING_DEG = 0.25
WIDE_POINTS = TARGET_POINTS + 2 * TARGET_MARGIN_CELLS
WIDE_SOUTH = TARGET_SOUTH - TARGET_MARGIN_CELLS * SPACING_DEG
WIDE_NORTH = TARGET_NORTH + TARGET_MARGIN_CELLS * SPACING_DEG
WIDE_WEST = TARGET_WEST - TARGET_MARGIN_CELLS * SPACING_DEG
WIDE_EAST = TARGET_EAST + TARGET_MARGIN_CELLS * SPACING_DEG

# v3-aligned splits (docs/R7_S3_CONFIRMATION_INSTANCE_V3.md). read_plan_frozen
# carries the older v2 split; that plan is untouched and this one is explicit.
TRAIN_YEARS = (2017, 2018, 2019, 2020, 2021)
VAL_YEARS = (2022,)
SEALED_TEST_YEAR = 2023

# Same second-stage caps the S1/B3 acquisitions ran under; a wide part touches
# the same global chunks, so the decoded cost is expected to be unchanged.
WIDE_NEW_ARTIFACT_BYTES_CAP = 100 * 2**30
WIDE_DECODED_BYTES_CAP = 256 * 2**30

# Recorded Earthmover spatial-namespace chunk geometry (earthmover_spatial_d1):
# one whole-globe field per time. The read count is per (source variable, level),
# not per output channel: 4 surface variables plus 5 pressure variables at each of
# the 3 stored levels = 19 field reads per stamp. The registered target-box parts
# measure the same 19 (2287 chunk reads over 120 stamps, coordinates included).
SOURCE_LATITUDE_POINTS = 721
SOURCE_LONGITUDE_POINTS = 1440
SOURCE_DTYPE_BYTES = 4
FIELD_READS_PER_STAMP = 19


def grid_points(span_deg: float, spacing_deg: float = SPACING_DEG) -> int:
    """Inclusive grid-point count of a closed coordinate interval."""
    if span_deg < 0 or spacing_deg <= 0:
        raise ValueError("span must be nonnegative and spacing positive")
    exact = span_deg / spacing_deg
    if abs(exact - round(exact)) > 1e-9:
        raise ValueError(f"span {span_deg} is not an integer number of {spacing_deg}-degree cells")
    return int(round(exact)) + 1


def containment_evidence() -> dict:
    """Assert that the wide grid's central block *is* the frozen target box.

    This is the property the whole comparison rests on: if it fails, the wide
    instance cannot be scored on the registered target box and no shared
    climatology applies, so it is asserted rather than described.
    """
    for span, points in ((TARGET_NORTH - TARGET_SOUTH, TARGET_POINTS),
                         (TARGET_EAST - TARGET_WEST, TARGET_POINTS),
                         (WIDE_NORTH - WIDE_SOUTH, WIDE_POINTS),
                         (WIDE_EAST - WIDE_WEST, WIDE_POINTS)):
        if grid_points(span) != points:
            raise ValueError(f"span {span} does not give {points} grid points at {SPACING_DEG} deg")
    if 2 * TARGET_MARGIN_CELLS >= WIDE_POINTS:
        raise ValueError("the target margin must leave a nonempty interior block")
    offset = TARGET_MARGIN_CELLS * SPACING_DEG
    edges = {
        "south": (WIDE_SOUTH + offset, TARGET_SOUTH),
        "north": (WIDE_NORTH - offset, TARGET_NORTH),
        "west": (WIDE_WEST + offset, TARGET_WEST),
        "east": (WIDE_EAST - offset, TARGET_EAST),
    }
    for name, (derived, target) in edges.items():
        if abs(derived - target) > 1e-9:
            raise ValueError(f"wide {name} edge does not land on the frozen target box")
    return {
        "wide_points": WIDE_POINTS,
        "target_points": TARGET_POINTS,
        "target_margin_cells": TARGET_MARGIN_CELLS,
        "target_margin_deg": offset,
        "interior_selector": ("training/r7_boundary_metrics.boundary_masks(%d, %d, (%d,)) "
                              "'interior' mask" % (WIDE_POINTS, WIDE_POINTS, TARGET_MARGIN_CELLS)),
        "derived_edges": {name: {"wide": wide, "target": target}
                          for name, (wide, target) in edges.items()},
        "identity": "the central %dx%d block of the wide grid is the frozen 65x65 "
                    "27-43N/107-123E box, cell for cell" % (TARGET_POINTS, TARGET_POINTS),
    }


def central_block() -> tuple[slice, slice]:
    """The (rows, columns) slice of the wide grid that equals the target box."""
    containment_evidence()
    start = TARGET_MARGIN_CELLS
    return slice(start, start + TARGET_POINTS), slice(start, start + TARGET_POINTS)


def stamp_cost(stamps: int, field_reads: int = FIELD_READS_PER_STAMP) -> dict:
    """Pre-read decoded cost: every touched chunk decodes whole, crop rides free.

    The spatial chunk is one global field, so the touched-chunk union - and
    therefore the decoded bytes - does not depend on how much of the field is
    retained. The retained (stored) bytes do scale with the ROI, and that is the
    only cost this enlargement adds.
    """
    if stamps < 1 or field_reads < 1:
        raise ValueError("stamps and field_reads must be positive")
    chunk_bytes = SOURCE_DTYPE_BYTES * SOURCE_LATITUDE_POINTS * SOURCE_LONGITUDE_POINTS
    decoded = chunk_bytes * field_reads * stamps
    retained_wide = SOURCE_DTYPE_BYTES * WIDE_POINTS * WIDE_POINTS * stamps
    retained_target = SOURCE_DTYPE_BYTES * TARGET_POINTS * TARGET_POINTS * stamps
    return {
        "stamps": stamps,
        "field_reads_per_stamp": field_reads,
        "global_chunk_bytes": chunk_bytes,
        "decoded_bytes": decoded,
        "retained_bytes_wide": retained_wide,
        "retained_bytes_target": retained_target,
        "retained_growth_factor": retained_wide / retained_target,
        "scope": ("decoded bytes are the uncompressed touched-chunk union; the network "
                  "figure is measured by the downloader, not estimated here"),
    }


def wide_frozen_protocol(*, new_artifact_bytes_cap: int = WIDE_NEW_ARTIFACT_BYTES_CAP,
                         decoded_bytes_cap: int = WIDE_DECODED_BYTES_CAP,
                         stamps: int | None = None) -> dict:
    """Machine-readable frozen plan for the wide instance; canonical digest."""
    if new_artifact_bytes_cap < 1 or decoded_bytes_cap < 1:
        raise ValueError("budget caps must be positive")
    body = {
        "format": "r7-era5-wide-read-plan-v1",
        "frozen_before_any_download": True,
        "scientific_claim": False,
        "region": {"south": WIDE_SOUTH, "north": WIDE_NORTH,
                   "west": WIDE_WEST, "east": WIDE_EAST,
                   "points": [WIDE_POINTS, WIDE_POINTS], "spacing_deg": SPACING_DEG},
        "target_box": {"south": TARGET_SOUTH, "north": TARGET_NORTH,
                       "west": TARGET_WEST, "east": TARGET_EAST,
                       "points": [TARGET_POINTS, TARGET_POINTS],
                       "containment": containment_evidence()},
        "single_factor": ("the wide region is the only change against the registered "
                          "v3 instance: same source/snapshot, same 17 channels, same "
                          "cadence and splits, same frozen target box"),
        "channels_rule": ("the 17-channel order comes from r7_era5.DEFAULT_R7_ERA5_CHANNELS "
                          "through read_plan_frozen.channel_plan and is asserted, never rewritten"),
        "splits": {"train_years": list(TRAIN_YEARS), "val_years": list(VAL_YEARS),
                   "sealed_test_year": SEALED_TEST_YEAR},
        "cadence_hours": CADENCE_HOURS,
        "history_steps": HISTORY_STEPS,
        "lead_hours": list(LEAD_HOURS),
        "max_lead_hours": MAX_LEAD_HOURS,
        "init_hours_utc": list(INIT_HOURS_UTC),
        "normalization_rule": ("mean/std, climatology and process-diagnostic normalization fit "
                              "on train years only; a new instance gets its own normalization "
                              "identity and never reuses the v3 figures"),
        "identity_rule": ("new sources, times or regions require an independent identity; the "
                          "frozen v2/v3 read plan and every archived source pin stay in place"),
        "budget_caps": {"new_artifacts_bytes": new_artifact_bytes_cap,
                        "decoded_source_bytes": decoded_bytes_cap},
        "cost_model": stamp_cost(stamps) if stamps else None,
        "limitations": [
            "byte costs are decoded-source estimates from recorded chunk geometry; "
            "http_bytes stays null until actually measured",
            "an 8-degree half-width is a partial answer to synoptic advection: a system "
            "moving at 10 m/s travels ~2600 km (23 deg) in 72 h, more than this margin",
            "no journal technique and no skill claim is derivable from the acquisition itself",
        ],
    }
    return dict(body, protocol_sha256=canonical_digest(body))
