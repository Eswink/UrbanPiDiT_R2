"""Bounded Earthmover extraction of the M2 two-month (Jan-Feb) segment (#69).

Why this segment exists. Every earlier R7 segment is a single January, so the
train-only ``(month, hour)`` climatology has exactly **four** buckets and each
held-out block sits inside the same 30 days those buckets were averaged from.
The #67 sealed report read that as "the neural arms lose to a zero-parameter
climatology on all 85 cells", which cannot be separated from the data range:
with four buckets the baseline is close to memorising the mean field of the very
month it is scored on.

M2 doubles the bucket count by covering **two** months, and separates the scored
block from the fitted days inside the same months:

- train ``[2016-01-01T00:00, 2016-02-17T00:00)`` 188 stamps - all of January
  (31 days, buckets ``(1,00/06/12/18)``) plus 16 days of February (buckets
  ``(2,00/06/12/18)``), i.e. **8** buckets instead of 4;
- val ``[2016-02-17T00:00, 2016-02-23T00:00)`` 24 stamps - 11 windows at a 72 h
  horizon, where the January re-cut's 8-stamp val block held **none**;
- test ``[2016-02-23T00:00, 2016-03-01T00:00)`` 28 stamps - 15 windows at 72 h,
  and every valid time is still February, so all 8 trained buckets apply and the
  fail-closed ``normalized_climatology`` refusal is not triggered.

The February buckets are fitted on 2016-02-01..02-16 while the scored block is
2016-02-23..02-29, so the baseline is evaluated a week away from the days it was
averaged over rather than on the same days. That is the separation this segment
buys; it is **not** a cross-season or cross-year claim, and the segment is still
one winter in one region.

Acquisition is bounded and honest like D1/B2: scope is read from
``read_plan_frozen`` under the **second-stage** caps (and the second-stage
protocol digest, which never rewrites the D1/B2 first-stage identity), nothing is
written until the byte budget, the deadline and every metadata attestation pass,
and a failure leaves a ``failed-no-fallback`` receipt instead of any synthetic
substitute.

The 240-stamp read is split into contiguous parts so that no single process runs
longer than the project's 30-minute bound; ``merge_m2_parts`` concatenates them,
re-asserts exact 6-hourly continuity and records every part's hash and measured
network bytes, so the merged source is auditable back to each part.
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

from .earthmover_pilot import SOURCE, SNAPSHOT, DecodedBudget
from .earthmover_spatial_b2 import B2_LAST_STAMP, B2_STAMPS, B2_START
from .earthmover_spatial_d1 import (
    _channel_payloads,
    _collect_frames,
    _net_recv_bytes,
    _open_pinned_session,
    _write_netcdf,
    read_stamp,
    validate_spatial_namespace,
)
from .read_plan_frozen import (
    D1_DAYS,
    GRID_HEIGHT,
    GRID_WIDTH,
    SECOND_STAGE_DECODED_BYTES_CAP,
    SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP,
    frozen_protocol,
)

M2_DAYS = 60
M2_START = "2016-01-01T00:00"
M2_LAST_STAMP = "2016-02-29T18:00"
M2_STAMPS = M2_DAYS * 24 // 6
# Contiguous parts, sized so one process stays near the 30-minute bound at the
# measured ~9.4 s/stamp read rate.
M2_PART_STAMPS = 120
DEFAULT_M2_DEADLINE_SECONDS = 1800.0
D1_STAMPS = D1_DAYS * 24 // 6


def m2_request(days=M2_DAYS, start=M2_START, cadence_hours=6):
    """The M2 continuous 60-day segment: explicit init/valid-time coverage."""
    from datetime import datetime, timedelta

    start_time = datetime.fromisoformat(start)
    steps = days * 24 // cadence_hours
    times = [start_time + timedelta(hours=cadence_hours * i) for i in range(steps)]
    init_times = [t for t in times if t.hour in (0, 6, 12, 18)]
    if len(times) != M2_STAMPS:
        raise ValueError(f"the M2 request must be {M2_STAMPS} stamps, got {len(times)}")
    if times[-1] != datetime.fromisoformat(M2_LAST_STAMP):
        raise ValueError(f"the M2 request must end {M2_LAST_STAMP}, got {times[-1].isoformat()}")
    if times[0] != datetime.fromisoformat(M2_START):
        raise ValueError(f"the M2 request must start {M2_START}, got {times[0].isoformat()}")
    return {"days": days, "start": start, "steps": steps,
            "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
            "init_times": len(init_times),
            "season": "Jan-Feb" if start_time.month == 1 and days >= 59 else f"{start_time.month:02d}"}


def _requested_times():
    import pandas as pd

    request = m2_request()
    start = pd.Timestamp(request["first_time"])
    stamps = [start + pd.Timedelta(hours=6 * step) for step in range(request["steps"])]
    if len(stamps) != M2_STAMPS or stamps[-1] != pd.Timestamp(M2_LAST_STAMP):
        raise ValueError(f"the frozen M2 request must be {M2_STAMPS} stamps "
                         f"ending {M2_LAST_STAMP}")
    hours = sorted({stamp.hour for stamp in stamps})
    if hours != [0, 6, 12, 18]:
        raise ValueError("M2 stamps must cover exactly the four UTC initialization hours")
    # The extension must strictly contain both earlier frozen requests, or the
    # earlier segments stop being prefixes and no earlier baseline transfers.
    if stamps.index(pd.Timestamp(B2_LAST_STAMP)) != B2_STAMPS - 1:
        raise ValueError("the M2 request no longer contains B2's 144 stamps as a prefix")
    if stamps[0] != pd.Timestamp(B2_START):
        raise ValueError("the M2 request no longer starts where B2 starts")
    return stamps, request


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
        "purpose": ("M2 two-month engineering segment (60 consecutive days, all of "
                    "January and February 2016, from the candidate train years) whose "
                    "train block spans two months so the train-only month-hour "
                    "climatology has eight buckets instead of four; no journal "
                    "technique is claimable from this data"),
    }


def _byte_caps(decoded_budget_bytes):
    return {"new_artifacts_bytes": SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP,
            "decoded_source_bytes": int(decoded_budget_bytes)}


def _second_stage_protocol():
    """The second-stage frozen plan, whose digest is independent of stage one."""
    return frozen_protocol(stage="second",
                           new_artifact_bytes_cap=SECOND_STAGE_NEW_ARTIFACT_BYTES_CAP,
                           decoded_bytes_cap=SECOND_STAGE_DECODED_BYTES_CAP)


def extract_m2_part(out_nc, receipt_json, *, offset=0, stamps=M2_PART_STAMPS,
                    deadline_seconds=DEFAULT_M2_DEADLINE_SECONDS,
                    decoded_budget_bytes=SECOND_STAGE_DECODED_BYTES_CAP):
    """Download one contiguous M2 part into a new NetCDF plus an honest receipt.

    Nothing is written before the byte budget, the deadline and every metadata
    attestation pass. On failure a ``failed-no-fallback`` receipt is written and
    the exception is re-raised; synthetic data is never substituted.
    """
    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    offset, stamps = int(offset), int(stamps)
    if offset < 0 or stamps < 1 or offset + stamps > M2_STAMPS:
        raise ValueError(f"M2 part [{offset}, {offset + stamps}) is outside the "
                         f"{M2_STAMPS}-stamp segment")
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    all_stamps, request = _requested_times()
    requested = all_stamps[offset:offset + stamps]
    protocol = _second_stage_protocol()
    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    result: dict = {
        "status": "pending",
        "segment": "m2-two-month-january-february",
        "part": {"offset": offset, "stamps": stamps,
                 "first_time": requested[0].isoformat(),
                 "last_time": requested[-1].isoformat()},
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "read_plan_stage": protocol["stage"],
        "read_plan_protocol_sha256": protocol["protocol_sha256"],
        "budget_caps": _byte_caps(decoded_budget_bytes),
        "contains_d1_prefix_stamps": D1_STAMPS,
        "contains_b2_prefix_stamps": B2_STAMPS,
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
            descriptor, temp = tempfile.mkstemp(prefix=".era5-m2-", suffix=".nc", dir=out_nc.parent)
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
            "m2_request": request,
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
            "one 60-day winter segment from the candidate train years; both months "
            "are January and February 2016, so no cross-season, cross-year or "
            "cross-region claim survives",
            "the scored block is separated from the fitted days by one week inside "
            "the same two months, not by a season boundary: the train-only "
            "month-hour climatology still shares the test months by construction "
            "and cannot be made out-of-sample in month at this data range",
            "M2 verifies windowing/units/IO and the bucket count; no journal "
            "technique is claimable from this data",
            "stored pressure axis keeps the planned 250/500/850 hPa levels, not the "
            "full audited 13-level axis; the kept levels are explicit in this receipt",
            "network bytes are host-wide interface counters during the reads, not an "
            "icechunk per-request accounting, and each part measures its own window",
            "the earlier segments are re-read rather than spliced from their archived "
            "artifacts; each acquisition is its own identity",
        ],
    })
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def merge_m2_parts(part_pairs, out_nc, receipt_json, *, expected_stamps=M2_STAMPS):
    """Concatenate contiguous M2 parts into one source, re-asserting continuity.

    ``part_pairs`` is a sequence of ``(netcdf_path, receipt_path)``. The merged
    receipt carries every part's hash, stamp count and measured network bytes, so
    the concatenation is auditable back to the individual downloads instead of
    hiding them behind one file.
    """
    import pandas as pd
    import xarray as xr

    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    if not part_pairs:
        raise ValueError("at least one part is required")
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    parts, datasets, recorded = [], [], []
    for nc_path, rc_path in part_pairs:
        nc_path, rc_path = Path(nc_path), Path(rc_path)
        if not nc_path.is_file() or not rc_path.is_file():
            raise FileNotFoundError(f"missing part artifact: {nc_path} / {rc_path}")
        with rc_path.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        if receipt.get("status") != "downloaded-real-source":
            raise ValueError(f"part {nc_path.name} is not a completed download: "
                             f"{receipt.get('status')!r}")
        if receipt["local_artifact"]["sha256"] != _sha256_file(nc_path):
            raise ValueError(f"part {nc_path.name} does not match its receipt hash; "
                             "refusing to merge unverified input")
        dataset = xr.open_dataset(nc_path)
        datasets.append(dataset)
        recorded.append({
            "path": nc_path.name, "receipt": rc_path.name,
            "stamps": int(dataset.sizes["time"]),
            "first_time": str(pd.Timestamp(dataset["time"].values[0])),
            "last_time": str(pd.Timestamp(dataset["time"].values[-1])),
            "bytes": nc_path.stat().st_size,
            "sha256": receipt["local_artifact"]["sha256"],
            "network_body_bytes": int(receipt.get("network_body_bytes", 0)),
            "elapsed_seconds": receipt.get("elapsed_seconds"),
        })
        parts.append(nc_path)
    merged = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    times = pd.DatetimeIndex(merged["time"].values)
    expected = M2_STAMPS if expected_stamps is None else int(expected_stamps)
    if len(times) != expected:
        raise ValueError(f"merged source has {len(times)} stamps, expected {expected}")
    deltas = times.to_series().diff().dropna().unique()
    if len(deltas) != 1 or pd.Timedelta(deltas[0]) != pd.Timedelta(hours=6):
        raise ValueError(f"merged stamps are not exactly 6-hourly continuous: {deltas[:3]}")
    if not times.is_monotonic_increasing or not times.is_unique or times.hasnans:
        raise ValueError("merged time axis must be unique, increasing and valid")
    if (times[0] != pd.Timestamp(M2_START)
            or times[-1] != pd.Timestamp(M2_LAST_STAMP)):
        raise ValueError(f"merged range {times[0]}..{times[-1]} is not the frozen M2 range")
    attrs = dict(datasets[0].attrs)
    attrs.update(_dataset_attributes(attrs.get("source_snapshot_id", SNAPSHOT)))
    merged.attrs.update(attrs)
    descriptor, temp = tempfile.mkstemp(prefix=".era5-m2-merge-", suffix=".nc", dir=out_nc.parent)
    os.close(descriptor)
    try:
        _write_netcdf(merged, temp)
        os.link(temp, out_nc)
    finally:
        os.unlink(temp)
    for dataset in datasets:
        dataset.close()
    with out_nc.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    with xr.open_dataset(out_nc) as merged_source:
        payloads, channel_names = _channel_payloads(merged_source)
    result = {
        "status": "merged-real-source",
        "segment": "m2-two-month-january-february",
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "merge_rule": ("parts are concatenated in the given order along the time axis; "
                       "the merged axis is re-asserted to be unique, increasing, exactly "
                       "6-hourly and equal to the frozen M2 range"),
        "parts": recorded,
        "n_parts": len(recorded),
        "stamps": int(len(times)),
        "time_first": str(times[0]),
        "time_last": str(times[-1]),
        "network_body_bytes_total": int(sum(entry["network_body_bytes"] for entry in recorded)),
        "network_method": ("sum of each part's own recv-byte delta over non-loopback "
                           "/proc/net/dev interfaces; host-wide counters, not an "
                           "icechunk per-request accounting"),
        "channels": channel_names,
        "channel_count": len(channel_names),
        "payloads": payloads,
        "local_artifact": {"path": out_nc.name, "bytes": out_nc.stat().st_size,
                           "sha256": digest},
        "synthetic_fallback": False,
        "scientific_claim": False,
        "source_is_real_reanalysis": True,
        "limitations": [
            "the merged artifact is a concatenation of separately measured downloads; "
            "the total network figure is the sum of host-wide counter windows, not a "
            "single continuous measurement",
            "one 60-day winter segment from the candidate train years; no "
            "cross-season, cross-year or cross-region claim survives",
        ],
    }
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "segment": "m2-two-month-january-february",
        "source": SOURCE,
        "snapshot_id": session.snapshot_id,
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "m2_request": request,
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
        description="Frozen-scope Earthmover spatial-namespace M2 extraction (#69)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true",
                      help="read-only audit of layout/coordinates/units/one stamp")
    mode.add_argument("--write", action="store_true",
                      help="download one contiguous M2 part into a new NetCDF plus receipt")
    mode.add_argument("--merge", action="store_true",
                      help=f"concatenate parts into the {M2_STAMPS}-stamp M2 source")
    parser.add_argument("--report", help="preflight only: new JSON report path")
    parser.add_argument("--out", help="write/merge: new NetCDF path")
    parser.add_argument("--receipt", help="write/merge: new receipt JSON path")
    parser.add_argument("--offset", type=int, default=0,
                        help=f"write only: first stamp index inside the {M2_STAMPS}-stamp segment")
    parser.add_argument("--stamps", type=int, default=M2_PART_STAMPS,
                        help="write only: number of contiguous stamps in this part")
    parser.add_argument("--parts", nargs="+", help="merge only: part NetCDF paths, in order")
    parser.add_argument("--part-receipts", nargs="+",
                        help="merge only: each part's receipt JSON, same order as --parts")
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_M2_DEADLINE_SECONDS)
    parser.add_argument("--max-decoded-gib", type=float,
                        default=SECOND_STAGE_DECODED_BYTES_CAP / 2**30)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.report)
        return
    if args.merge:
        if not args.parts or not args.part_receipts:
            parser.error("--merge requires --parts and --part-receipts")
        if len(args.parts) != len(args.part_receipts):
            parser.error("--parts and --part-receipts must have the same length")
        if not args.out or not args.receipt:
            parser.error("--merge requires --out and --receipt")
        receipt = merge_m2_parts(list(zip(args.parts, args.part_receipts)),
                                 args.out, args.receipt)
        print(json.dumps({"status": receipt["status"], "stamps": receipt["stamps"],
                          "network_bytes_total": receipt["network_body_bytes_total"],
                          "sha256": receipt["local_artifact"]["sha256"]}))
        return
    if not args.out or not args.receipt:
        parser.error("--write requires --out and --receipt")
    receipt = extract_m2_part(args.out, args.receipt, offset=args.offset, stamps=args.stamps,
                              deadline_seconds=args.deadline_seconds,
                              decoded_budget_bytes=int(args.max_decoded_gib * 2**30))
    print(json.dumps({
        "status": receipt["status"],
        "part": receipt["part"],
        "stamps": len(receipt["timestamps"]),
        "network_bytes": receipt["network_body_bytes"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "sha256": receipt["local_artifact"]["sha256"],
    }))


if __name__ == "__main__":
    main()
