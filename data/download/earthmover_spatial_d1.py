"""Bounded Earthmover spatial-namespace extraction of the frozen D1 segment (#63).

Decision 0004 pinned the ``spatial`` namespace (one global field per time;
chunks ``(1, 721, 1440)`` surface and ``(1, 1, 721, 1440)`` pressure) for D1
after measurement showed the ``temporal`` namespace needs ~612 chunks per stamp
for the 65x65 ROI. This module adds ONLY the chunk addressing for that layout:

- scope (ROI, times, channel order, cadence, caps) is read from
  ``read_plan_frozen`` and asserted, never rewritten;
- the bounded per-chunk reading machinery (budget/deadline guards, unique-chunk
  selection, finiteness and CF missing-value checks) is the audited
  ``earthmover_pilot`` code, reused unmodified;
- the publishing contract (level/unit attestation, dataset schema,
  ``failed-no-fallback`` receipts, payload and artifact SHA256) follows
  ``arco_regional_bounded``;
- ``earthmover_pilot``'s temporal 12x12 semantics are untouched.

Preflight mode reads coordinates, metadata and exactly one stamp and writes
nothing (an optional ``--report`` file aside). Write mode downloads the 120
D1 stamps into a fresh NetCDF plus a receipt with the measured network bytes
(/proc/net/dev delta, never a decoded estimate). Failure leaves a
``status='failed-no-fallback'`` receipt and re-raises; synthetic data is never
substituted. D1 is an engineering segment: ``scientific_claim: false``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .arco_regional_bounded import PRESSURE_LEVELS_HPA
from .earthmover_pilot import SOURCE, SNAPSHOT, DecodedBudget, bounded_selection
from .read_plan_frozen import (
    FIRST_STAGE_DECODED_BYTES_CAP,
    GRID_HEIGHT,
    GRID_WIDTH,
    channel_plan,
    d1_request,
    frozen_protocol,
)

# Earthmover stores short ECMWF names; the audited converter and the frozen
# channel plan speak the long ARCO names. This mapping is the declared source
# identity of this path, asserted against channel_plan() in d1_fields().
SOURCE_VARIABLE_SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "geopotential": "z",
    "temperature": "t",
    "specific_humidity": "q",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}
SURFACE_DIMS = ("valid_time", "latitude", "longitude")
PRESSURE_DIMS = ("valid_time", "pressure_level", "latitude", "longitude")
SPATIAL_SURFACE_CHUNKS = (1, 721, 1440)
SPATIAL_PRESSURE_CHUNKS = (1, 1, 721, 1440)
DEFAULT_DEADLINE_SECONDS = 1800.0


def d1_fields():
    """The exact read/store field list, derived from the frozen channel plan.

    Returns one row per stored source variable (Earthmover group and short
    name plus the R7 channel names used for unit checks) and the shared stored
    pressure axis: every audited level any planned channel needs, so all five
    pressure variables keep one consistent axis like arco_regional_bounded.
    """
    rows = channel_plan()
    levels_by_variable: dict[str, set[int]] = {}
    for row in rows:
        variable = row["variable"]
        if variable not in SOURCE_VARIABLE_SHORT:
            raise ValueError(f"no audited Earthmover short name for {variable!r}")
        if row["level_hpa"] is not None:
            levels_by_variable.setdefault(variable, set()).add(int(row["level_hpa"]))
    shared_levels = sorted({level for levels in levels_by_variable.values() for level in levels})
    if not shared_levels or not set(shared_levels) <= set(PRESSURE_LEVELS_HPA):
        raise ValueError("stored level axis must stay inside the audited 13-level set")
    fields = []
    for variable, short in SOURCE_VARIABLE_SHORT.items():
        needed = sorted(levels_by_variable.get(variable, ()))
        group = "pressure" if needed else "single"
        fields.append({
            "group": group, "variable": variable, "short_name": short,
            "levels": list(shared_levels) if needed else [],
            "channel_names": sorted(row["channel"] for row in rows if row["variable"] == variable),
        })
    return fields, shared_levels


def stamp_plan(times, requested):
    """Exact index of every requested stamp; never nearest-neighbour substitution."""
    import pandas as pd
    times = pd.DatetimeIndex(times)
    requested = pd.DatetimeIndex(requested).as_unit(times.unit)
    missing = [str(time) for time in requested if time not in times]
    if missing:
        raise ValueError(f"exact requested timestamps absent from source: {missing[:3]}")
    return times.get_indexer(requested)


def _attest_spatial_levels(level_values):
    """Assert the Earthmover level axis carries the audited hPa value set.

    ``arco_regional_bounded.attest_levels`` pins the ARCO axis in ascending
    source order; this snapshot stores the same 13 audited values in its own
    descending order. The value set must match exactly, the source order is
    recorded verbatim and no conversion or reordering is applied.
    """
    from .arco_regional_bounded import LEVEL_UNITS_ATTESTED, LEVEL_UNITS_EVIDENCE

    values = np.asarray(level_values)
    if values.ndim != 1 or values.size != len(PRESSURE_LEVELS_HPA):
        raise ValueError(
            f"source pressure axis has {values.size} values, expected {len(PRESSURE_LEVELS_HPA)}")
    if sorted(int(v) for v in values) != list(PRESSURE_LEVELS_HPA):
        raise ValueError(
            "source pressure levels differ from the audited hPa set; refusing to guess units")
    return {
        "levels": [int(v) for v in values],
        "units": LEVEL_UNITS_ATTESTED,
        "evidence": LEVEL_UNITS_EVIDENCE,
        "source_order": "descending as stored by this snapshot (audited value set matches)",
        "conversion_applied": False,
    }


def validate_spatial_namespace(root, times_requested, budget):
    """Assert the frozen spatial layout, coordinates, units and level axis.

    Reads coordinates and metadata only (charged to the shared budget).
    Returns the validated indices, the per-stamp read plan and observed units.
    """
    import pandas as pd
    import xarray as xr
    from data.preprocess.grid import regular_latlon_spacing
    from data.preprocess.r7_preflight import SI_UNITS, _unit, canonical_unit

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
    if not np.isclose(spacing, 0.25, rtol=0, atol=1e-5):
        raise ValueError("source grid must have native 0.25-degree spacing")
    south, north, west, east = 27.0, 43.0, 107.0, 123.0
    if (GRID_HEIGHT, GRID_WIDTH) != (65, 65):
        raise ValueError("frozen grid points changed; this path pins 65x65")
    yi = np.flatnonzero((sc["latitude"] >= south - 1e-9) & (sc["latitude"] <= north + 1e-9))
    xi = np.flatnonzero((sc["longitude"] >= west - 1e-9) & (sc["longitude"] <= east + 1e-9))
    if (len(yi), len(xi)) != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"expected the frozen {GRID_HEIGHT}x{GRID_WIDTH} ROI, got {len(yi)}x{len(xi)}")
    if not (np.allclose(np.sort(sc["latitude"][yi]), np.arange(south, north + 0.01, 0.25), rtol=0, atol=1e-5)
            and np.allclose(np.sort(sc["longitude"][xi]), np.arange(west, east + 0.01, 0.25), rtol=0, atol=1e-5)):
        raise ValueError("selected coordinates do not match the frozen ROI")
    time_attrs = dict(groups["single"]["valid_time"].attrs)
    pressure_time_attrs = dict(groups["pressure"]["valid_time"].attrs)
    for key in ("units", "calendar"):
        if time_attrs.get(key) != pressure_time_attrs.get(key):
            raise ValueError("surface/pressure time encoding mismatch")
    times = pd.DatetimeIndex(xr.coding.times.decode_cf_datetime(
        sc["valid_time"], time_attrs["units"], time_attrs.get("calendar", "standard"), use_cftime=False))
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

    arrays, observed_units = {}, {}
    surface_chunk_bytes = int(np.prod(SPATIAL_SURFACE_CHUNKS)) * 4
    for field in fields:
        array = groups[field["group"]][field["short_name"]]
        expected_dims = SURFACE_DIMS if field["group"] == "single" else PRESSURE_DIMS
        if tuple(array.metadata.dimension_names) != expected_dims:
            raise ValueError(f"unexpected dimensions for {field['short_name']}: {array.metadata.dimension_names}")
        lengths = {"valid_time": len(sc["valid_time"]), "latitude": len(sc["latitude"]),
                   "longitude": len(sc["longitude"]), "pressure_level": len(levels)}
        if tuple(array.shape) != tuple(lengths[name] for name in expected_dims):
            raise ValueError(f"field/coordinate shape mismatch for {field['short_name']}")
        if array.shards is not None:
            raise ValueError(f"sharded source requires a different audited read plan: {field['short_name']}")
        if np.dtype(array.dtype) != np.dtype("float32"):
            raise ValueError("this path preserves float32 source payloads without guessed conversion")
        if "scale_factor" in array.attrs or "add_offset" in array.attrs:
            raise ValueError("packed data require an explicit decoding audit")
        chunks = tuple(int(c) for c in array.chunks)
        frozen_chunks = SPATIAL_SURFACE_CHUNKS if field["group"] == "single" else SPATIAL_PRESSURE_CHUNKS
        if chunks != frozen_chunks:
            raise ValueError(
                f"{field['short_name']} chunk geometry {chunks} differs from the frozen spatial "
                f"layout {frozen_chunks}; the measured cost model no longer applies and a new "
                "read plan is required")
        observed = str(dict(array.attrs).get("units", ""))
        for name in field["channel_names"]:
            if _unit(observed) not in SI_UNITS[canonical_unit(name)]:
                raise ValueError(f"unit mismatch for {name}: observed {observed!r}")
        observed_units[field["short_name"]] = observed
        arrays[(field["group"], field["short_name"])] = array
    read_count = sum(1 if field["group"] == "single" else len(field["levels"]) for field in fields)
    plan = {
        "fields": fields, "shared_levels": shared_levels, "stamps": stamps,
        "yi": yi, "xi": xi, "level_index": level_index, "arrays": arrays,
        "latitude": sc["latitude"][yi].astype(np.float32),
        "longitude": sc["longitude"][xi].astype(np.float32),
        "times": times[stamps], "level_attestation": attestation,
        "observed_units": observed_units, "grid_spacing_deg": float(spacing),
        "per_stamp_field_reads": int(read_count),
        "per_stamp_field_bytes": int(read_count * surface_chunk_bytes),
    }
    return plan


def read_stamp(plan, position, budget):
    """One stamp: read every needed (variable, level) chunk once, cropped to the ROI.

    The spatial chunk is one whole global field, so each read decodes exactly
    one chunk and ``bounded_selection`` retains only the frozen ROI cells.
    """
    ti = int(plan["stamps"][position])
    frames = {}
    for field in plan["fields"]:
        array = plan["arrays"][(field["group"], field["short_name"])]
        if field["group"] == "single":
            values = bounded_selection(array, (np.array([ti]), plan["yi"], plan["xi"]), budget)[0]
        else:
            planes = [
                bounded_selection(
                    array,
                    (np.array([ti]), np.array([plan["level_index"][level]]), plan["yi"], plan["xi"]),
                    budget,
                )[0, 0]
                for level in field["levels"]
            ]
            values = np.stack(planes, axis=0)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite values in {field['variable']} at {plan['times'][position]}")
        frames[field["variable"]] = values.astype(np.float32)
    return frames


def _net_recv_bytes():
    total = 0
    with open("/proc/net/dev", encoding="utf-8") as handle:
        for line in handle:
            if ":" in line and not line.strip().startswith("lo"):
                total += int(line.split(":")[1].split()[0])
    return total


def _open_pinned_session():
    import icechunk as ic
    import numcodecs.zarr3  # noqa: F401 - registers the source's zarr3 codecs
    import zarr
    try:
        from icechunk.storage import s3_storage
    except ImportError:  # older icechunk exposes the factory on the top module
        s3_storage = ic.s3_storage
    storage = s3_storage(bucket="earthmover-icechunk-era5", prefix="icechunkV2",
                         region="us-east-1", anonymous=True)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session(snapshot_id=SNAPSHOT)
    if session.snapshot_id != SNAPSHOT:
        raise RuntimeError("snapshot pin not honored")
    return session, ic, zarr


def _requested_times():
    import pandas as pd
    request = d1_request()
    start = pd.Timestamp(request["first_time"])
    stamps = [start + pd.Timedelta(hours=6 * step) for step in range(request["steps"])]
    if len(stamps) != 120 or stamps[-1] != pd.Timestamp("2016-01-30T18:00"):
        raise ValueError("the frozen D1 request must be 120 stamps ending 2016-01-30T18:00")
    hours = sorted({stamp.hour for stamp in stamps})
    if hours != [0, 6, 12, 18]:
        raise ValueError("D1 stamps must cover exactly the four UTC initialization hours")
    return stamps, request


def _collect_frames(plan, budget, deadline):
    frames: dict[str, list[np.ndarray]] = {}
    for position in range(len(plan["stamps"])):
        if time.monotonic() > deadline:
            raise TimeoutError("wall-clock deadline reached during stamp read")
        stamp_frames = read_stamp(plan, position, budget)
        for variable, values in stamp_frames.items():
            frames.setdefault(variable, []).append(values)
    return {variable: np.stack(stack) for variable, stack in frames.items()}


def _channel_payloads(dataset):
    from data.preprocess.r7_era5 import DEFAULT_R7_ERA5_CHANNELS, stack_era5_channels

    state, names, _, _, _ = stack_era5_channels(dataset, DEFAULT_R7_ERA5_CHANNELS)
    if state.shape[1] != 17 or len(names) != 17:
        raise ValueError("the audited converter must yield exactly the 17 planned channels")
    payloads = {}
    for index, name in enumerate(names):
        values = state[:, index]
        payloads[name] = {
            "payload_sha256": hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest(),
            "shape": list(values.shape),
            "finite": bool(np.isfinite(values).all()),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    del state
    return payloads, names


def _write_netcdf(dataset, out_nc):
    encoding = {}
    for variable, data in dataset.data_vars.items():
        shape = tuple(int(v) for v in data.shape)
        chunksizes = (1, shape[1], shape[2], shape[3]) if len(shape) == 4 else (1, shape[1], shape[2])
        encoding[variable] = dict(zlib=True, complevel=1, shuffle=True, chunksizes=chunksizes)
    dataset.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding)


def _dataset_attributes(snapshot_id):
    return {
        "source": SOURCE,
        "source_snapshot_id": snapshot_id,
        "license": "CC-BY-4.0",
        "attribution": ("Copernicus C3S/ECMWF ERA5; NSF NCAR historical archive; "
                        "Earthmover Icechunk edition"),
        "reference": "https://registry.opendata.aws/earthmover-era5/",
        "native_grid_spacing_deg": 0.25,
        "interpolation": "none",
        "spatial_resampling": "none",
        "target_semantics": "native ERA5 grid; no spatial upsampling",
        "purpose": ("frozen D1 engineering segment (30 consecutive days from the "
                    "candidate train years) for window/unit/IO verification; "
                    "no journal technique is claimable from this data"),
    }


def extract_d1(out_nc, receipt_json, *, deadline_seconds=DEFAULT_DEADLINE_SECONDS,
               decoded_budget_bytes=FIRST_STAGE_DECODED_BYTES_CAP):
    """Download the frozen D1 segment into a new NetCDF plus an honest receipt.

    Nothing is written before the byte budget, the deadline and every
    metadata attestation pass. On failure a ``failed-no-fallback`` receipt is
    written and the exception is re-raised; synthetic data is never substituted.
    """
    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    requested, request = _requested_times()
    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    result: dict = {
        "status": "pending",
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "decision": "docs/decisions/0004-d1-source-selection.md",
        "read_plan_protocol_sha256": frozen_protocol()["protocol_sha256"],
        "synthetic_fallback": False,
        "scientific_claim": False,
    }
    try:
        session, ic, zarr = _open_pinned_session()
        budget = DecodedBudget(limit=int(decoded_budget_bytes), seconds=float(deadline_seconds),
                               started=started)
        with zarr.config.set({"async.concurrency": 1}):
            root = zarr.open_group(store=session.store, mode="r")
            plan = validate_spatial_namespace(root, requested, budget)
            estimated = plan["per_stamp_field_bytes"] * len(plan["stamps"])
            budget.check(estimated)
            network_before = _net_recv_bytes()
            frames = _collect_frames(plan, budget, deadline)
            network_bytes = _net_recv_bytes() - network_before
            import xarray as xr
            data_vars = {}
            for field in plan["fields"]:
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
                "time": plan["times"],
                "level": np.asarray(plan["shared_levels"], dtype=np.int32),
                "latitude": plan["latitude"],
                "longitude": plan["longitude"],
            }
            dataset = xr.Dataset(data_vars, coords=coords)
            dataset["latitude"].attrs = {"units": "degrees_north", "long_name": "latitude"}
            dataset["longitude"].attrs = {"units": "degrees_east", "long_name": "longitude"}
            dataset["level"].attrs = {"units": "hPa", "long_name": "pressure_level"}
            units_by_variable = {field["variable"]: plan["observed_units"][field["short_name"]]
                                 for field in plan["fields"]}
            for variable, data in dataset.data_vars.items():
                data.attrs["units"] = units_by_variable[variable]
                data.attrs["source_variable"] = variable
            dataset.attrs.update(_dataset_attributes(session.snapshot_id))
            payloads, channel_names = _channel_payloads(dataset)
            descriptor, temp = tempfile.mkstemp(prefix=".era5-d1-", suffix=".nc", dir=out_nc.parent)
            os.close(descriptor)
            try:
                _write_netcdf(dataset, temp)
                os.link(temp, out_nc)
            finally:
                os.unlink(temp)
            del dataset, frames
    except Exception as exc:
        failure = dict(result)
        failure.update({
            "status": "failed-no-fallback",
            "error": f"{type(exc).__name__}: {exc}",
            "synthetic_fallback": False,
            "scientific_claim": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        with receipt_json.open("x", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2, ensure_ascii=False, allow_nan=False)
        raise
    with out_nc.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    result.update({
        "status": "downloaded-real-source",
        "source_is_real_reanalysis": True,
        "protocol": {
            "d1_request": request,
            "roi": {"south": 27.0, "north": 43.0, "west": 107.0, "east": 123.0,
                    "points": [GRID_HEIGHT, GRID_WIDTH], "spacing_deg": 0.25},
            "channels": channel_names,
            "channel_count": 17,
            "stored_level_axis_hpa": plan["shared_levels"],
            "selection": "exact 6-hourly UTC timestamps; no nearest substitution",
        },
        "timestamps": [stamp.isoformat() for stamp in plan["times"]],
        "init_hour_coverage": {str(hour): int(sum(1 for stamp in plan["times"] if stamp.hour == hour))
                               for hour in (0, 6, 12, 18)},
        "variables": sorted(plan["observed_units"]),
        "channel_units": plan["observed_units"],
        "level_attestation": plan["level_attestation"],
        "shape": [int(v) for v in (len(plan["times"]), len(plan["shared_levels"]),
                                   len(plan["latitude"]), len(plan["longitude"]))],
        "payloads": payloads,
        "decoded_chunk_budget": {
            "cap_bytes": int(decoded_budget_bytes),
            "estimated_field_bytes": int(estimated),
            "charged_bytes": int(budget.used),
            "chunk_reads": int(budget.reads),
            "budget_scope": ("sum of uncompressed touched spatial-chunk sizes (fields plus "
                             "coordinates); NOT the measured network transfer and not peak RAM"),
        },
        "network_body_bytes": int(network_bytes),
        "network_method": ("recv-byte delta over non-loopback /proc/net/dev interfaces "
                           "around the field reads; host-wide counter, same method as "
                           "docs/R7_D1_READ_SPEED.md"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "local_artifact": {
            "path": out_nc.name,
            "bytes": out_nc.stat().st_size,
            "sha256": digest,
        },
        "icechunk_version": ic.__version__,
        "zarr_version": zarr.__version__,
        "limitations": [
            "one 30-day engineering segment from the candidate train years; "
            "no seasonal, multi-year or test claim",
            "D1 verifies windowing/units/IO only; no journal technique is claimable",
            "stored pressure axis keeps the planned 250/500/850 hPa levels, not the "
            "full audited 13-level axis; the kept levels are explicit in this receipt",
            "network bytes are host-wide interface counters during the reads, not an "
            "icechunk per-request accounting",
        ],
    })
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def preflight(report_path=None):
    """Read-only source audit: layout, coordinates, units, one stamp. No download."""
    requested, request = _requested_times()
    started = time.monotonic()
    session, ic, zarr = _open_pinned_session()
    budget = DecodedBudget(limit=512 * 2**20, seconds=900.0, started=started)
    with zarr.config.set({"async.concurrency": 1}):
        root = zarr.open_group(store=session.store, mode="r")
        plan = validate_spatial_namespace(root, requested, budget)
        first = read_stamp(plan, 0, budget)
        sample = {name: {"min": float(values.min()), "max": float(values.max()),
                         "finite": bool(np.isfinite(values).all()), "shape": list(values.shape)}
                  for name, values in first.items()}
    report = {
        "schema_version": 1,
        "mode": "read-only-preflight",
        "scientific_training_certified": False,
        "scientific_claim": False,
        "source": SOURCE,
        "snapshot_id": session.snapshot_id,
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "d1_request": request,
        "stamps_found": int(len(plan["stamps"])),
        "time_first": plan["times"][0].isoformat(),
        "time_last": plan["times"][-1].isoformat(),
        "roi_points": [int(len(plan["yi"])), int(len(plan["xi"]))],
        "grid_spacing_deg": plan["grid_spacing_deg"],
        "stored_level_axis_hpa": plan["shared_levels"],
        "level_attestation": plan["level_attestation"],
        "observed_units": plan["observed_units"],
        "per_stamp_field_reads": plan["per_stamp_field_reads"],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-scope Earthmover spatial-namespace D1 extraction (#63, decision 0004)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true",
                      help="read-only audit of layout/coordinates/units/one stamp")
    mode.add_argument("--write", action="store_true",
                      help="download the 120 D1 stamps into a new NetCDF plus receipt")
    parser.add_argument("--report", help="preflight only: new JSON report path")
    parser.add_argument("--out", help="write only: new NetCDF path")
    parser.add_argument("--receipt", help="write only: new receipt JSON path")
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    parser.add_argument("--max-decoded-gib", type=float, default=FIRST_STAGE_DECODED_BYTES_CAP / 2**30)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.report)
        return
    if not args.out or not args.receipt:
        parser.error("--write requires --out and --receipt")
    receipt = extract_d1(args.out, args.receipt, deadline_seconds=args.deadline_seconds,
                         decoded_budget_bytes=int(args.max_decoded_gib * 2**30))
    print(json.dumps({
        "status": receipt["status"],
        "stamps": len(receipt["timestamps"]),
        "network_bytes": receipt["network_body_bytes"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "sha256": receipt["local_artifact"]["sha256"],
    }))


if __name__ == "__main__":
    main()
