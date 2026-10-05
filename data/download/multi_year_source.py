"""Combine verified per-batch ERA5 sources into one confirmation source.

The S3 confirmation instance spans acquisition batches: the S1 2017 four-season
source and the batch-2 2022/2023 source. ``prepare_r7_local`` reads one source,
so the verified batches are concatenated here into one publication-stable
NetCDF with its own receipt.

No synthetic fallback: every input must carry a merge receipt whose
``local_artifact.sha256`` matches the file bytes on disk, the coordinates and
level axis must be identical across batches, and the concatenated axis must be
exactly the sorted union of the inputs' own time axes (no duplicates, no gaps
smoothed over, four UTC init hours). Existing outputs are refused, never
truncated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .earthmover_spatial_d1 import _channel_payloads, _write_netcdf


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified(nc_path, receipt_path):
    """Read one input's merge receipt and re-assert the file hash against it."""
    nc_path, receipt_path = Path(nc_path), Path(receipt_path)
    if not nc_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing source artifact: {nc_path} / {receipt_path}")
    with receipt_path.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("status") != "merged-real-source":
        raise ValueError(f"source {nc_path.name} is not a completed merge: "
                         f"{receipt.get('status')!r}")
    digest = _sha256_file(nc_path)
    if receipt.get("local_artifact", {}).get("sha256") != digest:
        raise ValueError(f"source {nc_path.name} does not match its receipt hash; "
                         "refusing to combine unverified input")
    return receipt, digest


def combine_sources(source_pairs, out_nc, receipt_json, *, expected_stamps):
    """Concatenate verified batch sources along time into one new source.

    ``source_pairs`` are ``(source.nc, merge_receipt.json)`` in chronological
    order. The merged axis must equal the sorted union of every input's own
    recorded stamps, so reordering, duplicates and overlaps are refused rather
    than repaired.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    if not source_pairs:
        raise ValueError("at least one source is required")
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)

    datasets, recorded, all_times = [], [], []
    for nc_path, receipt_path in source_pairs:
        receipt, digest = _load_verified(nc_path, receipt_path)
        dataset = xr.open_dataset(nc_path)
        datasets.append(dataset)
        times = pd.DatetimeIndex(dataset["time"].values)
        all_times.extend(times)
        recorded.append({
            "path": Path(nc_path).name, "receipt": Path(receipt_path).name,
            "stamps": int(len(times)),
            "first_time": str(times[0]), "last_time": str(times[-1]),
            "bytes": Path(nc_path).stat().st_size, "sha256": digest,
        })

    reference = datasets[0]
    time_name = "time"
    for dataset, entry in zip(datasets, recorded):
        if dataset["time"].dims != (time_name,):
            raise ValueError(f"source {entry['path']} has an unexpected time axis")
        for coord in ("latitude", "longitude", "level"):
            if coord not in dataset.coords or coord not in reference.coords:
                raise ValueError(f"source {entry['path']} lacks coordinate {coord!r}")
            if dataset[coord].shape != reference[coord].shape or not np.array_equal(
                    np.asarray(dataset[coord].values), np.asarray(reference[coord].values)):
                raise ValueError(f"source {entry['path']} coordinate {coord!r} differs "
                                 "from the first source; refusing to combine")
        if set(dataset.data_vars) != set(reference.data_vars):
            raise ValueError(f"source {entry['path']} variables differ from the first source")

    merged = xr.concat(datasets, dim=time_name) if len(datasets) > 1 else datasets[0]
    times = pd.DatetimeIndex(merged["time"].values)
    if int(expected_stamps) != sum(len(pd.DatetimeIndex(d["time"].values)) for d in datasets):
        raise ValueError(f"expected {expected_stamps} stamps, sources record a different total")
    if len(times) != int(expected_stamps):
        raise ValueError(f"combined source has {len(times)} stamps, expected {expected_stamps}")
    expected_axis = pd.DatetimeIndex(sorted(set(all_times)))
    if len(expected_axis) != len(all_times):
        raise ValueError("sources overlap: duplicate time stamps")
    if not times.is_unique:
        raise ValueError("sources overlap: duplicate time stamps")
    if not times.equals(expected_axis):
        raise ValueError("combined time axis differs from the sorted union of the sources "
                         "(check the given order is chronological)")
    if not times.is_monotonic_increasing or not times.is_unique or times.hasnans:
        raise ValueError("combined time axis must be unique, increasing and valid")
    if sorted({stamp.hour for stamp in times}) != [0, 6, 12, 18]:
        raise ValueError("combined stamps must cover the four UTC initialization hours")

    attrs = dict(reference.attrs)
    attrs["combined_batches"] = json.dumps(
        [{"path": entry["path"], "sha256": entry["sha256"]} for entry in recorded])
    attrs["purpose"] = ("S3 confirmation source: verified four-season batches combined "
                        "along the time axis for a train/val/test year-split store")
    merged.attrs.update(attrs)
    descriptor, temp = tempfile.mkstemp(prefix=".era5-s3-combine-", suffix=".nc",
                                        dir=out_nc.parent)
    os.close(descriptor)
    try:
        _write_netcdf(merged, temp)
        os.link(temp, out_nc)
    finally:
        os.unlink(temp)
    payloads, channel_names = _channel_payloads(merged)
    for dataset in datasets:
        dataset.close()
    with out_nc.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    result = {
        "status": "combined-real-source",
        "format": "r7-s3-combined-source-receipt-v1",
        "stage": "S3",
        "objective": ("combine the verified per-batch four-season sources into the single "
                      "source the confirmation store is built from"),
        "sources": recorded,
        "n_sources": len(recorded),
        "stamps": int(len(times)),
        "time_first": str(times[0]),
        "time_last": str(times[-1]),
        "union_rule": ("sources are concatenated in the given order along the time axis; the "
                       "combined axis is re-asserted to equal the sorted union of every "
                       "source's own stamps, unique, increasing and on the four UTC init hours"),
        "channels": channel_names,
        "channel_count": len(channel_names),
        "payloads": payloads,
        "local_artifact": {"path": out_nc.name, "bytes": out_nc.stat().st_size,
                           "sha256": digest},
        "synthetic_fallback": False,
        "source_is_real_reanalysis": True,
        "scientific_claim": False,
        "limitations": [
            "each season is a ~30-day sample of one ROI, not a full-season mean",
            "the source contains only the months January/April/July/October",
            "combining is a byte-level reliability step; it makes no scientific claim",
        ],
    }
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True,
                        help="verified source NetCDF paths, in chronological order")
    parser.add_argument("--receipts", nargs="+", required=True,
                        help="each source's merge receipt, in the same order")
    parser.add_argument("--out", required=True, help="new combined NetCDF path")
    parser.add_argument("--receipt", required=True, help="new combined receipt path")
    parser.add_argument("--expected-stamps", required=True, type=int)
    args = parser.parse_args(argv)
    if len(args.sources) != len(args.receipts):
        parser.error("--sources and --receipts must have equal length")
    result = combine_sources(list(zip(args.sources, args.receipts)), args.out,
                             args.receipt, expected_stamps=args.expected_stamps)
    print(json.dumps({key: result[key] for key in
                      ("status", "stamps", "time_first", "time_last", "n_sources",
                       "local_artifact")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
