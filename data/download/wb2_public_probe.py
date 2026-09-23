"""Metadata-only anonymous probe for the public WeatherBench2 ERA5 Zarr.

No meteorological field values are loaded here. The purpose is to inspect the
real public store's schema and chunk geometry before authorizing a bounded
multivariate subset read.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import xarray as xr

WB2_ERA5_025 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

DEFAULT_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
)


def _small_coordinate_summary(ds: xr.Dataset, name: str) -> dict | None:
    if name not in ds.coords:
        return None
    arr = ds.coords[name]
    result = {"dims": list(arr.dims), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.ndim != 1 or arr.size < 1:
        return result
    indices = sorted(set([0, arr.size // 2, arr.size - 1]))
    values = np.asarray(arr.isel({arr.dims[0]: indices}).values)
    if np.issubdtype(values.dtype, np.datetime64):
        result["samples"] = [str(v.astype("datetime64[ns]")) for v in values]
    else:
        vals = values.astype(np.float64)
        if np.isfinite(vals).all():
            result["samples"] = [float(v) for v in vals]
    return result


def describe_dataset(ds: xr.Dataset, variables: Iterable[str] = DEFAULT_VARIABLES) -> dict:
    """Return JSON-safe schema/chunk metadata without materializing fields."""
    desc = {
        "dimensions": {str(k): int(v) for k, v in ds.sizes.items()},
        "coordinates": {},
        "variables": {},
        "missing_variables": [],
    }
    for name in ("time", "level", "latitude", "longitude"):
        summary = _small_coordinate_summary(ds, name)
        if summary is not None:
            desc["coordinates"][name] = summary

    for name in variables:
        if name not in ds:
            desc["missing_variables"].append(name)
            continue
        arr = ds[name]
        encoding = dict(arr.encoding)
        chunks = encoding.get("chunks")
        preferred = encoding.get("preferred_chunks")
        desc["variables"][name] = {
            "dims": list(arr.dims),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "units": str(arr.attrs.get("units", "")),
            "chunks": None if chunks is None else [int(v) for v in chunks],
            "preferred_chunks": None if preferred is None else {
                str(k): int(v) for k, v in preferred.items()
            },
        }
    return desc


def probe_weatherbench2(source: str = WB2_ERA5_025) -> dict:
    ds = xr.open_zarr(
        source,
        chunks=None,
        storage_options={"token": "anon"},
    )
    try:
        report = describe_dataset(ds)
        report["source"] = source
        report["access"] = "anonymous-public-gcs"
        report["field_values_loaded"] = False
        required = {
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        }
        missing = required.difference(report["variables"])
        if missing:
            raise RuntimeError(f"WeatherBench2 schema missing required variables: {sorted(missing)}")
        if "level" not in report["coordinates"]:
            raise RuntimeError("WeatherBench2 schema has no pressure-level coordinate")
        return report
    finally:
        ds.close()
