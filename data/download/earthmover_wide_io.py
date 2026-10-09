"""Read/attest/preflight layer for the wide-region (S3 lateral-context) acquisition.

Split out of ``earthmover_spatial_w1.py`` so neither file crosses the 400-line
target (R-021). This layer owns what the wide read *is*: region validation
against the frozen spatial-namespace layout, unit and level attestation, the
dataset assembly, and the read-only preflight. The downloader keeps the part,
merge and CLI bookkeeping.

Every audited helper is imported from ``earthmover_spatial_d1`` and used
unchanged; the only difference from the frozen path is that the region comes from
``read_plan_wide`` instead of being hardcoded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .earthmover_pilot import SNAPSHOT, SOURCE, DecodedBudget, bounded_selection
from .earthmover_spatial_d1 import (
    PRESSURE_DIMS,
    SPATIAL_PRESSURE_CHUNKS,
    SPATIAL_SURFACE_CHUNKS,
    SURFACE_DIMS,
    _attest_spatial_levels,
    _collect_frames,
    _open_pinned_session,
    d1_fields,
    stamp_plan,
)
from .read_plan_wide import (
    SPACING_DEG,
    TARGET_POINTS,
    WIDE_EAST,
    WIDE_NORTH,
    WIDE_POINTS,
    WIDE_SOUTH,
    WIDE_WEST,
    containment_evidence,
    wide_frozen_protocol,
)
from .season_plan_s1 import _planned_stamps, season_plan


def attest_wide_fields(groups, fields, surface_coordinates, level_count):
    """Metadata attestation of every stored source variable; reads no data.

    Asserts the frozen chunk geometry, float32 payloads, no packing, and that
    every planned channel's unit is in the audited SI set for that channel. A
    difference is refused rather than converted.
    """
    from data.preprocess.r7_preflight import SI_UNITS, _unit, canonical_unit

    arrays, observed_units = {}, {}
    for field in fields:
        array = groups[field["group"]][field["short_name"]]
        expected_dims = SURFACE_DIMS if field["group"] == "single" else PRESSURE_DIMS
        if tuple(array.metadata.dimension_names) != expected_dims:
            raise ValueError(f"unexpected dimensions for {field['short_name']}")
        lengths = {"valid_time": len(surface_coordinates["valid_time"]),
                   "latitude": len(surface_coordinates["latitude"]),
                   "longitude": len(surface_coordinates["longitude"]),
                   "pressure_level": int(level_count)}
        if tuple(array.shape) != tuple(lengths[name] for name in expected_dims):
            raise ValueError(f"field/coordinate shape mismatch for {field['short_name']}")
        if array.shards is not None:
            raise ValueError(f"sharded source requires a different audited read plan: "
                             f"{field['short_name']}")
        if np.dtype(array.dtype) != np.dtype("float32"):
            raise ValueError("this path preserves float32 source payloads without guessed conversion")
        if "scale_factor" in array.attrs or "add_offset" in array.attrs:
            raise ValueError("packed data require an explicit decoding audit")
        chunks = tuple(int(c) for c in array.chunks)
        frozen_chunks = (SPATIAL_SURFACE_CHUNKS if field["group"] == "single"
                         else SPATIAL_PRESSURE_CHUNKS)
        if chunks != frozen_chunks:
            raise ValueError(
                f"{field['short_name']} chunk geometry {chunks} differs from the frozen spatial "
                f"layout {frozen_chunks}; the measured cost model no longer applies")
        observed = str(dict(array.attrs).get("units", ""))
        for name in field["channel_names"]:
            if _unit(observed) not in SI_UNITS[canonical_unit(name)]:
                raise ValueError(f"unit mismatch for {name}: observed {observed!r}")
        observed_units[field["short_name"]] = observed
        arrays[(field["group"], field["short_name"])] = array
    return arrays, observed_units


def validate_wide_namespace(root, times_requested, budget):
    """Assert the wide region, the frozen spatial layout, units and level axis.

    A faithful adaptation of ``earthmover_spatial_d1.validate_spatial_namespace``
    with the region read from ``read_plan_wide`` instead of hardcoded. The
    containment of the frozen target box is asserted here, so a wide read that
    did not actually cover it cannot be published.
    """
    import pandas as pd
    import xarray as xr
    from data.preprocess.grid import regular_latlon_spacing
    from data.preprocess.r7_preflight import SI_UNITS, _unit, canonical_unit

    containment_evidence()
    fields, shared_levels = d1_fields()
    groups = {}
    for group in ("single", "pressure"):
        try:
            groups[group] = root[f"{group}/spatial"]
        except Exception as error:
            raise ValueError(f"spatial namespace {group!r} unavailable: {error}") from error
    coordinates = {}
    for group in ("single", "pressure"):
        coordinates[group] = {
            name: bounded_selection(groups[group][name],
                                    (np.arange(groups[group][name].shape[0]),), budget)
            for name in ("latitude", "longitude", "valid_time")
        }
    for name in ("latitude", "longitude", "valid_time"):
        if not np.array_equal(coordinates["single"][name], coordinates["pressure"][name]):
            raise ValueError("surface/pressure coordinate mismatch")
    sc = coordinates["single"]
    spacing = regular_latlon_spacing(sc["latitude"], sc["longitude"])
    if not np.isclose(spacing, SPACING_DEG, rtol=0, atol=1e-5):
        raise ValueError("source grid must have native 0.25-degree spacing")
    yi = np.flatnonzero((sc["latitude"] >= WIDE_SOUTH - 1e-9)
                        & (sc["latitude"] <= WIDE_NORTH + 1e-9))
    xi = np.flatnonzero((sc["longitude"] >= WIDE_WEST - 1e-9)
                        & (sc["longitude"] <= WIDE_EAST + 1e-9))
    if (len(yi), len(xi)) != (WIDE_POINTS, WIDE_POINTS):
        raise ValueError(f"expected the wide {WIDE_POINTS}x{WIDE_POINTS} region, "
                         f"got {len(yi)}x{len(xi)}")
    latitude = np.sort(sc["latitude"][yi])
    longitude = np.sort(sc["longitude"][xi])
    if not (np.allclose(latitude, np.arange(WIDE_SOUTH, WIDE_NORTH + 0.01, SPACING_DEG),
                        rtol=0, atol=1e-5)
            and np.allclose(longitude, np.arange(WIDE_WEST, WIDE_EAST + 0.01, SPACING_DEG),
                            rtol=0, atol=1e-5)):
        raise ValueError("selected coordinates do not match the wide region")
    # The target box must be the central block of *this* selection, in the
    # selection's own order, not merely the same numbers on paper.
    start = (WIDE_POINTS - TARGET_POINTS) // 2
    interior = slice(start, start + TARGET_POINTS)
    if not np.allclose(latitude[interior], np.arange(27.0, 43.0 + 0.01, SPACING_DEG),
                       rtol=0, atol=1e-5):
        raise ValueError("the central latitude block is not the frozen target box")
    if not np.allclose(longitude[interior], np.arange(107.0, 123.0 + 0.01, SPACING_DEG),
                       rtol=0, atol=1e-5):
        raise ValueError("the central longitude block is not the frozen target box")
    time_attrs = dict(groups["single"]["valid_time"].attrs)
    pressure_time_attrs = dict(groups["pressure"]["valid_time"].attrs)
    for key in ("units", "calendar"):
        if time_attrs.get(key) != pressure_time_attrs.get(key):
            raise ValueError("surface/pressure time encoding mismatch")
    times = pd.DatetimeIndex(xr.coding.times.decode_cf_datetime(
        sc["valid_time"], time_attrs["units"], time_attrs.get("calendar", "standard"),
        use_cftime=False))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("source times must be unique, increasing and valid")
    stamps = stamp_plan(times, pd.DatetimeIndex(times_requested))

    level_array = groups["pressure"]["pressure_level"]
    if str(level_array.attrs.get("units")) != "hPa":
        raise ValueError("explicit hPa pressure coordinate required")
    levels = bounded_selection(level_array, (np.arange(level_array.shape[0]),), budget)
    attestation = _attest_spatial_levels(levels)
    level_index = {int(level): position for position, level in enumerate(levels)}
    for level in shared_levels:
        if level not in level_index:
            raise ValueError(f"required level {level} hPa absent from the source axis")

    arrays, observed_units = attest_wide_fields(groups, fields, sc, len(levels))
    read_count = sum(1 if field["group"] == "single" else len(field["levels"]) for field in fields)
    return {
        "fields": fields, "shared_levels": shared_levels, "stamps": stamps,
        "yi": yi, "xi": xi, "level_index": level_index, "arrays": arrays,
        "latitude": sc["latitude"][yi].astype(np.float32),
        "longitude": sc["longitude"][xi].astype(np.float32),
        "times": times[stamps], "level_attestation": attestation,
        "observed_units": observed_units, "grid_spacing_deg": float(spacing),
        "per_stamp_field_reads": int(read_count),
        "per_stamp_field_bytes": int(read_count * surface_chunk_bytes),
        "roi": {"south": WIDE_SOUTH, "north": WIDE_NORTH, "west": WIDE_WEST,
                "east": WIDE_EAST, "points": [WIDE_POINTS, WIDE_POINTS],
                "spacing_deg": SPACING_DEG},
        "target_box": {"points": [TARGET_POINTS, TARGET_POINTS],
                       "central_block": containment_evidence()},
    }


# Provenance read first-hand from the source's own registry page on 2026-10-09
# (R-050): the license and its URL, and the DOI the page publishes for the
# Copernicus C3S/ECMWF ERA5 products this edition is derived from. The page
# lists both the single-level and the pressure-level products under that DOI,
# and this path reads both, so the citation is recorded verbatim rather than
# attributed to one of the two by guesswork.
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_REGISTRY_URL = "https://registry.opendata.aws/earthmover-era5/"
SOURCE_DOI = "10.24381/cds.adbb2d47"
SOURCE_CITATION_ACCESSED = "2026-10-09"


def source_citation():
    """The published licence, DOI and registry reference for this source."""
    return {
        "license": SOURCE_LICENSE,
        "license_url": SOURCE_LICENSE_URL,
        "doi": SOURCE_DOI,
        "registry": SOURCE_REGISTRY_URL,
        "accessed": SOURCE_CITATION_ACCESSED,
        "note": ("read from the source's own registry page; the page cites the Copernicus "
                 "C3S/ECMWF ERA5 products under this DOI and lists both the single-level and "
                 "the pressure-level products, and this path reads both"),
    }


def _wide_dataset_attributes(snapshot_id):
    """Snapshot provenance for the wide segment; no D1 prose is inherited."""
    citation = source_citation()
    return {
        "source": SOURCE,
        "source_snapshot_id": snapshot_id,
        "license": citation["license"],
        "license_url": citation["license_url"],
        "source_doi": citation["doi"],
        "citation_accessed": citation["accessed"],
        "attribution": ("Copernicus C3S/ECMWF ERA5; NSF NCAR historical archive; "
                        "Earthmover Icechunk edition"),
        "reference": citation["registry"],
        "native_grid_spacing_deg": SPACING_DEG,
        "interpolation": "none",
        "spatial_resampling": "none",
        "target_semantics": "native ERA5 grid; no spatial upsampling",
        "region_points": [WIDE_POINTS, WIDE_POINTS],
        "target_box_points": [TARGET_POINTS, TARGET_POINTS],
        "purpose": ("wide-region lateral-context segment for the S3 climatology campaign; "
                    "the central 65x65 block is the frozen target box; the acquisition "
                    "itself claims no skill and no journal technique"),
    }


def _assemble_wide_dataset(plan_read, frames, snapshot_id):
    """Build the NetCDF from validated frames, units and metadata."""
    import xarray as xr

    data_vars = {}
    for field in plan_read["fields"]:
        values = frames[field["variable"]]
        if field["group"] == "pressure":
            if values.ndim != 4 or values.shape[1] != len(field["levels"]):
                raise ValueError(f"{field['variable']} lost its stored level axis")
            data_vars[field["variable"]] = (("time", "level", "latitude", "longitude"), values)
        else:
            if values.ndim != 3:
                raise ValueError(f"{field['variable']} is not a surface plane")
            data_vars[field["variable"]] = (("time", "latitude", "longitude"), values)
    coords = {
        "time": plan_read["times"],
        "level": np.asarray(plan_read["shared_levels"], dtype=np.int32),
        "latitude": plan_read["latitude"],
        "longitude": plan_read["longitude"],
    }
    dataset = xr.Dataset(data_vars, coords=coords)
    dataset["latitude"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    dataset["longitude"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    dataset["level"].attrs = {"units": "hPa", "long_name": "pressure_level"}
    units_by_variable = {field["variable"]: plan_read["observed_units"][field["short_name"]]
                         for field in plan_read["fields"]}
    for variable, data in dataset.data_vars.items():
        data.attrs["units"] = units_by_variable[variable]
        data.attrs["source_variable"] = variable
    dataset.attrs.update(_wide_dataset_attributes(snapshot_id))
    return dataset


def preflight_wide(report_path=None, *, years=(2018,), seasons=("winter",)):
    """Read-only source audit of the wide region: layout, units, one stamp.

    Writes nothing except an explicitly requested report. It does not certify the
    source for scientific training and it measures no network bytes.
    """
    import pandas as pd

    plan = season_plan(list(years), season_names=tuple(seasons))
    requested = _planned_stamps(plan)[:1]
    started = time.monotonic()
    session, ic, zarr = _open_pinned_session()
    budget = DecodedBudget(limit=512 * 2**20, seconds=900.0, started=started)
    with zarr.config.set({"async.concurrency": 1}):
        root = zarr.open_group(store=session.store, mode="r")
        plan_read = validate_wide_namespace(root, requested, budget)
        first = _collect_frames(plan_read, budget, started + 900.0)
    sample = {name: {"min": float(values.min()), "max": float(values.max()),
                     "finite": bool(np.isfinite(values).all()),
                     "shape": list(values.shape)}
              for name, values in first.items()}
    report = {
        "schema_version": 1,
        "mode": "read-only-preflight",
        "scientific_training_certified": False,
        "scientific_claim": False,
        "source": SOURCE,
        "snapshot_id": session.snapshot_id,
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "read_plan_protocol_sha256": wide_frozen_protocol()["protocol_sha256"],
        "region": plan_read["roi"],
        "target_box": plan_read["target_box"],
        "time_requested": pd.Timestamp(requested[0]).isoformat(),
        "time_found": plan_read["times"][0].isoformat(),
        "grid_spacing_deg": plan_read["grid_spacing_deg"],
        "stored_level_axis_hpa": plan_read["shared_levels"],
        "level_attestation": plan_read["level_attestation"],
        "observed_units": plan_read["observed_units"],
        "per_stamp_field_reads": plan_read["per_stamp_field_reads"],
        "decoded_chunk_reads": int(budget.reads),
        "decoded_charged_bytes": int(budget.used),
        "first_stamp_samples": sample,
        "limitations": [
            "metadata plus one stamp only; full finiteness checks happen during the write",
            "preflight does not certify the source for scientific training",
            "no file is written (an explicit --report path aside)",
        ],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if report_path is not None:
        path = Path(report_path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return report
