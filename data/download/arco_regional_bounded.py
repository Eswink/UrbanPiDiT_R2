"""Bounded regional extraction from the anonymous public ARCO-ERA5 archive.

Why this exists (#13): the earlier real subset was one 12x12 tile over the first
eight days of January. This module widens the *regional context* (65x65 native
0.25-degree points) and the *seasonal spread* (four season starts in three
separate years) under an explicit, machine-checked budget, because ARCO chunking
makes ROI area nearly free while timestamps are what cost real work.

Hard limits, all enforced before any field value is read:
- exactly one fixed public source, no redirects, no credentials;
- exact-timestamp selection only, never nearest-neighbour substitution;
- a contiguous ROI that lies inside one already-audited source grid;
- an estimated touched-chunk byte cap and a wall-clock deadline;
- no synthetic substitution anywhere: failure is recorded and re-raised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .chunk_budget import estimate_bundle_chunk_bytes
from .wb2_public_probe import describe_dataset
from .wb2_surface_pilot import _exact_time_indices, _roi_slice

# The 6-hourly 13-level analysis-ready distribution. Anonymous read; no key.
SOURCE = "gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
ATTRIBUTION = (
    "Contains modified Copernicus Climate Change Service information 2026; "
    "distribution via ARCO-ERA5."
)
REFERENCE = "https://github.com/google-research/arco-era5"

# Upstream omits units on this axis. Values are asserted against this exact
# integer list, which is the audited ERA5 pressure-level set, and the sibling
# ARCO dataset full_37-1h-0p25deg-chunk-1.zarr-v3 declares the same integers as
# "Hectopascal(hPa)". No numeric conversion is applied; the attestation records
# exactly where the claim comes from.
PRESSURE_LEVELS_HPA = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
LEVEL_UNITS_ATTESTED = "hPa"
LEVEL_UNITS_EVIDENCE = (
    "source omits level units; integer values equal the audited ERA5 13-level set "
    "and match ARCO full_37-1h-0p25deg-chunk-1.zarr-v3, which declares Hectopascal(hPa); "
    "no conversion applied"
)

# (source variable, pressure level or None, R7 channel name, required units)
CHANNELS = (
    ("2m_temperature", None, "t2m", "K"),
    ("10m_u_component_of_wind", None, "u10", "m s**-1"),
    ("10m_v_component_of_wind", None, "v10", "m s**-1"),
    ("mean_sea_level_pressure", None, "mslp", "Pa"),
    ("temperature", 850, "t850", "K"),
    ("temperature", 500, "t500", "K"),
    ("specific_humidity", 850, "q850", "kg kg**-1"),
    ("u_component_of_wind", 850, "u850", "m s**-1"),
    ("u_component_of_wind", 500, "u500", "m s**-1"),
    ("v_component_of_wind", 850, "v850", "m s**-1"),
    ("v_component_of_wind", 500, "v500", "m s**-1"),
)

# The nine source variables the R7 contract needs. geopotential is included even
# though no default R7 channel uses it yet, because the audited converter may
# request z* channels and a second read would cost the same chunk anyway.
SOURCE_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
)
LEVEL_VARIABLES = frozenset({
    "geopotential", "temperature", "specific_humidity",
    "u_component_of_wind", "v_component_of_wind",
})

# Required units for every stored source variable, including geopotential, which
# no default R7 channel consumes yet but the audited converter may request.
SOURCE_UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m s**-1",
    "10m_v_component_of_wind": "m s**-1",
    "mean_sea_level_pressure": "Pa",
    "geopotential": "m**2 s**-2",
    "temperature": "K",
    "specific_humidity": "kg kg**-1",
    "u_component_of_wind": "m s**-1",
    "v_component_of_wind": "m s**-1",
}

SEASON_STARTS = ((1, 1), (4, 1), (7, 1), (10, 1))
YEARS = (2018, 2019, 2020)
# 65x65 native 0.25-degree points: 16 degrees of latitude and longitude, centred
# on the East-Asia domain used by the established R7 CPU/GPU case sets.
ROI_SOUTH, ROI_NORTH, ROI_WEST, ROI_EAST = 27.0, 43.0, 107.0, 123.0
BLOCK_STEPS = 4  # a 24-hour continuous block: three +6h windows per season/year
DEFAULT_MAX_CHUNK_BYTES = 16 * 2**30
DEFAULT_DEADLINE_SECONDS = 1800.0
DEFAULT_MAX_BYTES = 512 * 2**20


class BudgetExceeded(RuntimeError):
    """Raised before reading when a declared bound cannot be honoured."""


@dataclass(frozen=True)
class RegionalPlan:
    """The frozen sampling protocol, written into the receipt verbatim."""

    years: tuple[int, ...] = YEARS
    season_starts: tuple[tuple[int, int], ...] = SEASON_STARTS
    block_steps: int = BLOCK_STEPS
    roi: tuple[float, float, float, float] = (ROI_SOUTH, ROI_NORTH, ROI_WEST, ROI_EAST)
    channels: tuple[tuple[str, int | None, str, str], ...] = CHANNELS
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS

    def validate(self) -> None:
        if not self.years or any(isinstance(y, bool) or not isinstance(y, int) for y in self.years):
            raise ValueError("years must be a nonempty integer sequence")
        if len(set(self.years)) != len(self.years):
            raise ValueError("duplicate years in plan")
        if not self.season_starts:
            raise ValueError("at least one season start is required")
        for month, day in self.season_starts:
            if not 1 <= month <= 12 or not 1 <= day <= 31:
                raise ValueError(f"invalid season start {month}-{day}")
        if isinstance(self.block_steps, bool) or not isinstance(self.block_steps, int) or self.block_steps < 3:
            raise ValueError("block_steps must be an integer >= 3 to form one window")
        south, north, west, east = self.roi
        if not (-90 <= south < north <= 90) or not (-180 <= west < east <= 180):
            raise ValueError("invalid ROI bounds")
        names = [name for _, _, name, _ in self.channels]
        if not names or len(set(names)) != len(names):
            raise ValueError("channel names must be unique and nonempty")
        for bound in (self.max_chunk_bytes, self.deadline_seconds):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)) or bound <= 0:
                raise ValueError("budget bounds must be positive numbers")


def plan_timestamps(plan: RegionalPlan) -> tuple[list[str], dict[str, list[str]]]:
    """Exact 6-hourly UTC timestamps per plan; no rounding, no substitution."""
    import pandas as pd

    by_block: dict[str, list[str]] = {}
    ordered: list[str] = []
    for year in plan.years:
        for month, day in plan.season_starts:
            start = pd.Timestamp(year=year, month=month, day=day)
            block = [start + pd.Timedelta(hours=6 * step) for step in range(plan.block_steps)]
            key = f"{year}-{month:02d}-{day:02d}"
            by_block[key] = [t.isoformat() for t in block]
            ordered.extend(t.isoformat() for t in block)
    if len(set(ordered)) != len(ordered):
        raise ValueError("plan produced duplicate timestamps")
    return ordered, by_block


def attest_levels(level_values: Sequence[float]) -> dict:
    """Assert the source pressure axis is the audited hPa set, exactly."""
    values = np.asarray(level_values)
    if values.ndim != 1 or values.size != len(PRESSURE_LEVELS_HPA):
        raise ValueError(
            f"source pressure axis has {values.size} values, expected {len(PRESSURE_LEVELS_HPA)}"
        )
    if not np.array_equal(values.astype(np.int64), np.asarray(PRESSURE_LEVELS_HPA, dtype=np.int64)):
        raise ValueError(
            "source pressure levels differ from the audited hPa set; refusing to guess units"
        )
    return {
        "levels": list(PRESSURE_LEVELS_HPA),
        "units": LEVEL_UNITS_ATTESTED,
        "evidence": LEVEL_UNITS_EVIDENCE,
        "conversion_applied": False,
    }


def check_channel_units(ds, channels=CHANNELS) -> dict:
    """Source units must match the R7 contract verbatim; never convert.

    Every required source variable is checked, not only the ones a given plan
    happens to map to an R7 channel, because the local file stores them all.
    """
    required = {variable: units for variable, _, _, units in channels}
    observed: dict[str, str] = {}
    for variable in SOURCE_VARIABLES:
        if variable not in ds:
            raise KeyError(f"source is missing required variable {variable!r}")
        found = str(ds[variable].attrs.get("units", ""))
        observed[variable] = found
        if variable in required and found != required[variable]:
            raise ValueError(
                f"{variable} units {found!r} != required {required[variable]!r}; no guessed conversion"
            )
    for variable, expected in required.items():
        if observed.get(variable) != expected:
            raise ValueError(
                f"{variable} units {observed.get(variable)!r} != required {expected!r}; no guessed conversion"
            )
    return observed


def _level_indices(plan: RegionalPlan) -> list[int]:
    """All audited levels, in ascending order.

    ARCO stores one global ``(1, 13, 721, 1440)`` chunk per pressure variable and
    timestamp, so a level subset costs exactly the same to fetch as the whole
    slab. Retaining all 13 levels keeps one shared, consistent pressure axis for
    every variable and lets the audited converter select any channel set
    (including geopotential) without another source read. Level *selection* is
    therefore a downstream concern, not an acquisition concern.
    """
    return list(range(len(PRESSURE_LEVELS_HPA)))


def region_budget(ds, plan: RegionalPlan, block_times: Sequence[str]) -> dict:
    """Estimate touched-chunk bytes for ONE block before any value is read.

    Every stored source variable is estimated, because the extraction reads and
    writes all of them, not only those a plan maps to an R7 channel.
    """
    report = describe_dataset(ds, SOURCE_VARIABLES)
    time_indices = _exact_time_indices(ds["time"].values, list(block_times))
    south, north, west, east = plan.roi
    lat_slice = _roi_slice(ds["latitude"].values, south, north, "latitude")
    lon_slice = _roi_slice(ds["longitude"].values, west, east, "longitude")
    level_indices = _level_indices(plan)
    selections = {}
    for variable in SOURCE_VARIABLES:
        dims = list(ds[variable].dims)
        entry: dict[str, object] = {
            "time": time_indices,
            "latitude": lat_slice,
            "longitude": lon_slice,
        }
        if "level" in dims:
            entry["level"] = level_indices
        selections[variable] = entry
    return estimate_bundle_chunk_bytes(report, selections)


def _decode_blocks(ds, plan: RegionalPlan, block_times: Sequence[str], deadline: float) -> dict:
    """Read one block, cropped to the ROI, preserving the source's own schema.

    Each variable is read once per timestamp and keeps its full native level
    axis: ARCO charges the whole global pressure chunk either way, so subsetting
    levels here would cost the same while making the stored schema differ from
    the source. Channel/level selection stays a converter concern.
    """
    south, north, west, east = plan.roi
    time_indices = _exact_time_indices(ds["time"].values, list(block_times))
    lat_slice = _roi_slice(ds["latitude"].values, south, north, "latitude")
    lon_slice = _roi_slice(ds["longitude"].values, west, east, "longitude")
    lat = np.asarray(ds["latitude"].values, dtype=np.float32)[lat_slice]
    lon = np.asarray(ds["longitude"].values, dtype=np.float32)[lon_slice]
    frames: dict[str, list[np.ndarray]] = {name: [] for name in SOURCE_VARIABLES}
    for step, index in enumerate(time_indices):
        for variable in SOURCE_VARIABLES:
            if time.monotonic() > deadline:
                raise BudgetExceeded("wall-clock deadline reached during block read")
            part = ds[variable].isel(time=index, latitude=lat_slice, longitude=lon_slice).load()
            values = np.asarray(part.values, dtype=np.float32)
            expected = (len(PRESSURE_LEVELS_HPA), lat.size, lon.size) if "level" in part.dims else (lat.size, lon.size)
            if values.shape != expected:
                raise ValueError(f"{variable} cropped frame has shape {values.shape}, expected {expected}")
            if not np.isfinite(values).all():
                raise ValueError(f"{variable} contains nonfinite values at {block_times[step]}")
            if "level" in part.dims:
                seen = np.asarray(part["level"].values, dtype=np.int64)
                if not np.array_equal(seen, np.asarray(PRESSURE_LEVELS_HPA, dtype=np.int64)):
                    raise ValueError(f"{variable} pressure axis changed during read")
            frames[variable].append(values)
            del part, values
    return {
        "frames": {name: np.stack(stack).astype(np.float32) for name, stack in frames.items()},
        "latitude": lat,
        "longitude": lon,
        "times": list(block_times),
    }


def _sha256_bytes(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def _surface_names(plan: RegionalPlan) -> list[str]:
    return [name for _, level, name, _ in plan.channels if level is None]


def _level_names(plan: RegionalPlan) -> list[str]:
    return [name for _, level, name, _ in plan.channels if level is not None]


def _variable_order(plan: RegionalPlan) -> list[str]:
    ordered: list[str] = []
    for variable, _, _, _ in plan.channels:
        if variable not in ordered:
            ordered.append(variable)
    return ordered


def build_regional_dataset(plan: RegionalPlan, blocks: Sequence[dict]):
    """Assemble cropped blocks, reproducing the source's own variable schema.

    All nine required source variables are kept (surface plus the full audited
    pressure axis), so the audited converter can select any channel set — the
    11 R7 process channels included — from this one local file.
    """
    import pandas as pd
    import xarray as xr

    times: list[str] = []
    for block in blocks:
        times.extend(block["times"])
    if len(set(times)) != len(times):
        raise ValueError("duplicate timestamps across blocks")
    latitude = blocks[0]["latitude"]
    longitude = blocks[0]["longitude"]
    for block in blocks:
        if not np.array_equal(block["latitude"], latitude) or not np.array_equal(block["longitude"], longitude):
            raise ValueError("ROI coordinates changed between blocks")
        if set(block["frames"]) != set(SOURCE_VARIABLES):
            raise ValueError("source variable set changed between blocks")
    data_vars = {}
    for variable in SOURCE_VARIABLES:
        planes = [block["frames"][variable] for block in blocks]
        stacked = np.concatenate(planes, axis=0).astype(np.float32)
        if variable in LEVEL_VARIABLES:
            if stacked.ndim != 4 or stacked.shape[1] != len(PRESSURE_LEVELS_HPA):
                raise ValueError(f"{variable} assembled shape {stacked.shape} has no full pressure axis")
            data_vars[variable] = (("time", "level", "latitude", "longitude"), stacked)
        else:
            if stacked.ndim != 3:
                raise ValueError(f"{variable} assembled shape {stacked.shape} is not a surface plane")
            data_vars[variable] = (("time", "latitude", "longitude"), stacked)
    coords = {
        "time": pd.DatetimeIndex(times),
        "latitude": latitude,
        "longitude": longitude,
    }
    coords["level"] = np.asarray(PRESSURE_LEVELS_HPA, dtype=np.int32)
    dataset = xr.Dataset(data_vars, coords=coords)
    dataset["latitude"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    dataset["longitude"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    # The upstream axis carries no units attribute. This records the attestation
    # performed by attest_levels(): values are the audited ERA5 hPa set,
    # cross-checked against a sibling ARCO dataset that declares
    # Hectopascal(hPa). No numeric conversion is applied.
    dataset["level"].attrs = {"units": LEVEL_UNITS_ATTESTED, "long_name": "pressure_level"}
    units_by_variable = dict(SOURCE_UNITS)
    for variable, _, _, units in plan.channels:
        units_by_variable[variable] = units
    for variable, data in dataset.data_vars.items():
        data.attrs["units"] = units_by_variable[variable]
        data.attrs["source_variable"] = variable
    dataset.attrs.update({
        "source": SOURCE,
        "attribution": ATTRIBUTION,
        "reference": REFERENCE,
        "native_grid_spacing_deg": 0.25,
        "interpolation": "none",
        "spatial_resampling": "none",
        "target_semantics": "native ERA5 grid; no spatial upsampling",
        "purpose": (
            "bounded regional real-source subset for the R7 data contract; "
            "engineering data path only, not a forecast-skill benchmark"
        ),
    })
    return dataset


def _describe_units(plan: RegionalPlan) -> dict[str, str]:
    return {name: units for _, _, name, units in plan.channels}


def extract_regional_region(
    out_nc: str | Path,
    receipt_json: str | Path,
    *,
    plan: RegionalPlan | None = None,
    source: str = SOURCE,
) -> dict:
    """Read the bounded regional subset and write NetCDF plus an honest receipt.

    Nothing is written before the byte budget and the unit/level attestations
    pass. On failure a ``failed-no-fallback`` receipt is written and the
    exception is re-raised; synthetic data is never substituted.
    """
    plan = RegionalPlan() if plan is None else plan
    plan.validate()
    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(plan.deadline_seconds)
    ordered_times, by_block = plan_timestamps(plan)
    result: dict = {
        "status": "pending",
        "source": source,
        "access": "anonymous-public-gcs",
        "attribution": ATTRIBUTION,
        "reference": REFERENCE,
        "synthetic_fallback": False,
        "scientific_claim": False,
    }
    try:
        import xarray as xr

        # A local path is used by the offline tests; a remote URI needs the
        # anonymous token. Neither path may fall back to other data.
        remote = "://" in str(source)
        ds = xr.open_zarr(
            source,
            chunks=None,
            storage_options={"token": "anon"} if remote else None,
        )
        try:
            block_keys = list(by_block)
            first_budget = region_budget(ds, plan, by_block[block_keys[0]])
            per_block = int(first_budget["estimated_uncompressed_bytes"])
            worst = {key: region_budget(ds, plan, by_block[key])["estimated_uncompressed_bytes"] for key in block_keys}
            total_estimate = int(sum(worst.values()))
            if total_estimate > int(plan.max_chunk_bytes):
                raise BudgetExceeded(
                    f"estimated touched chunks {total_estimate} bytes exceed cap {plan.max_chunk_bytes}"
                )
            level_report = attest_levels(ds["level"].values)
            units = check_channel_units(ds, plan.channels)
            blocks = []
            for key in block_keys:
                if time.monotonic() > deadline:
                    raise BudgetExceeded("wall-clock deadline reached before next block")
                blocks.append(_decode_blocks(ds, plan, by_block[key], deadline))
            dataset = build_regional_dataset(plan, blocks)
        finally:
            ds.close()
        if dataset.sizes["time"] != len(ordered_times):
            raise ValueError(
                f"assembled {dataset.sizes['time']} timestamps but plan requires {len(ordered_times)}"
            )
        encoding = {}
        for variable, data in dataset.data_vars.items():
            shape = tuple(int(v) for v in data.shape)
            if len(shape) == 4:
                encoding[variable] = dict(
                    zlib=True, complevel=1, shuffle=True,
                    chunksizes=(1, shape[1], shape[2], shape[3]),
                )
            else:
                encoding[variable] = dict(
                    zlib=True, complevel=1, shuffle=True,
                    chunksizes=(1, shape[1], shape[2]),
                )
        dataset.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding)
        level_axis = [int(v) for v in np.asarray(dataset["level"].values)] if "level" in dataset.coords else []
        units_by_name = _describe_units(plan)
        payloads = {}
        for variable, level, name, _ in plan.channels:
            values = np.asarray(dataset[variable].values)
            if level is None:
                if values.ndim != 3:
                    raise ValueError(f"{name} should be a surface plane, got dims {values.shape}")
                sample = values
            else:
                if values.ndim != 4 or level not in level_axis:
                    raise ValueError(f"{name} level {level} missing from the assembled pressure axis")
                sample = values[:, level_axis.index(level)]
            payloads[name] = {
                "payload_sha256": _sha256_bytes(sample),
                "shape": [int(v) for v in sample.shape],
                "finite": True,
                "min": float(np.min(sample)),
                "max": float(np.max(sample)),
                "units": units_by_name[name],
                "source_variable": variable,
                "pressure_hpa": -1 if level is None else int(level),
            }
            del values, sample
        digest = hashlib.sha256(out_nc.read_bytes()).hexdigest()
        result.update({
            "status": "downloaded-real-source",
            "protocol": {
                "years": list(plan.years),
                "season_starts": [list(s) for s in plan.season_starts],
                "block_steps": int(plan.block_steps),
                "roi": {
                    "south": plan.roi[0], "north": plan.roi[1],
                    "west": plan.roi[2], "east": plan.roi[3],
                },
                "roi_points": [int(dataset.sizes["latitude"]), int(dataset.sizes["longitude"])],
                "selection": "exact 6-hourly UTC timestamps; no nearest substitution",
            },
            "timestamps": ordered_times,
            "blocks": by_block,
            "variables": [name for _, _, name, _ in plan.channels],
            "channel_units": units,
            "level_attestation": level_report,
            "interpolation": False,
            "shape": [int(v) for v in dataset.sizes.values()],
            "sizes": {str(k): int(v) for k, v in dataset.sizes.items()},
            "payloads": payloads,
            "source_chunk_budget": {
                "cap_bytes": int(plan.max_chunk_bytes),
                "estimated_uncompressed_bytes": total_estimate,
                "first_block_estimate_bytes": per_block,
                "per_block_estimate_bytes": {k: int(v) for k, v in worst.items()},
                "actual_http_bytes": None,
                "note": (
                    "estimate is an uncompressed touched-chunk bound, not measured HTTP traffic; "
                    "HTTP bodies are not instrumented by this reader"
                ),
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "local_artifact": {
                "path": out_nc.name,
                "bytes": out_nc.stat().st_size,
                "sha256": digest,
            },
            "limitations": [
                "four 24-hour blocks per year in three years; not continuous full-year coverage",
                "native 0.25-degree grid; not a higher-resolution target and never labelled as one",
                "engineering data path only: no forecast skill, process-benefit or GPU claim",
                "level units are attested from value identity, not read from the source attribute",
            ],
        })
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
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded regional ARCO-ERA5 extraction for the R7 data contract."
    )
    parser.add_argument("--out", required=True, help="new NetCDF path")
    parser.add_argument("--receipt", required=True, help="new receipt JSON path")
    parser.add_argument("--max-chunk-gib", type=float, default=16.0)
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args()
    plan = RegionalPlan(
        max_chunk_bytes=int(args.max_chunk_gib * 2**30),
        deadline_seconds=args.deadline_seconds,
    )
    receipt = extract_regional_region(args.out, args.receipt, plan=plan)
    print(json.dumps({
        "status": receipt["status"],
        "shape": receipt["shape"],
        "timestamps": len(receipt["timestamps"]),
        "sha256": receipt["local_artifact"]["sha256"],
    }))


if __name__ == "__main__":
    main()
