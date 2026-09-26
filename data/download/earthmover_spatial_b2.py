"""Bounded Earthmover extraction of the B2 multiseed segment (#64 B2).

The frozen D1 segment (``earthmover_spatial_d1``) is 120 stamps, and its
val/test blocks are 12 stamps each. A free 72 h rollout needs 14 contiguous
held-out stamps (one history frame plus twelve lead frames), so **no** 72 h
free rollout window exists on D1 at all - ``ZarrRolloutDataset`` refuses the
request outright. #64 B2 requires 6/12/24/48/72 h free rollouts, so the
held-out blocks have to be wider than the segment D1 froze.

This module acquires a 36-day segment (144 stamps) that **contains** D1's
120 stamps as a prefix, so the train range stays byte-identical:

- train ``[2016-01-01T00:00, 2016-01-25T00:00)`` 96 stamps - identical to D1,
  hence identical train-only normalization and climatology buckets;
- val ``[2016-01-25T00:00, 2016-02-01T00:00)`` 28 stamps - holds 15 windows
  at a 72 h horizon where D1's 12-stamp val block held none;
- test ``[2016-02-01T00:00, 2016-02-06T00:00)`` 20 stamps - a block no
  earlier R7 experiment has read, kept sealed by the B2 harness.

The 24 extra stamps are re-read rather than spliced from the D1 archive: one
request, one receipt and one source identity are worth more than the 0.66 GiB
of redundant network, and both costs sit far below the frozen caps.

Scope (ROI, channel order, cadence, level axis, budget caps) is read from
``read_plan_frozen`` and asserted, never rewritten; the bounded per-chunk read
machinery, the metadata guards and the publishing contract are the audited
``earthmover_spatial_d1`` / ``earthmover_pilot`` code, reused unmodified. This
segment is still an engineering slice of one January: ``scientific_claim:
false``, and no seasonal or cross-year result is claimable from it.
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
from .earthmover_spatial_d1 import (
    DEFAULT_DEADLINE_SECONDS,
    _channel_payloads,
    _collect_frames,
    _net_recv_bytes,
    _open_pinned_session,
    _write_netcdf,
    read_stamp,
    validate_spatial_namespace,
)
from .read_plan_frozen import (
    FIRST_STAGE_DECODED_BYTES_CAP,
    GRID_HEIGHT,
    GRID_WIDTH,
    frozen_protocol,
)

B2_DAYS = 36
B2_START = "2016-01-01T00:00"
B2_STAMPS = B2_DAYS * 24 // 6
# D1's frozen request ends here; the extension continues where it stopped.
D1_LAST_STAMP = "2016-01-30T18:00"
B2_LAST_STAMP = "2016-02-05T18:00"


def b2_request(days=B2_DAYS, start=B2_START, cadence_hours=6):
    """The B2 continuous 36-day segment: explicit init/valid-time coverage."""
    from datetime import datetime, timedelta

    start_time = datetime.fromisoformat(start)
    steps = days * 24 // cadence_hours
    times = [start_time + timedelta(hours=cadence_hours * i) for i in range(steps)]
    init_times = [t for t in times if t.hour in (0, 6, 12, 18)]
    if len(times) != B2_STAMPS:
        raise ValueError(f"the B2 request must be {B2_STAMPS} stamps, got {len(times)}")
    if times[-1] != datetime.fromisoformat(B2_LAST_STAMP):
        raise ValueError(f"the B2 request must end {B2_LAST_STAMP}, got {times[-1].isoformat()}")
    if times[0] != datetime.fromisoformat(B2_START):
        raise ValueError(f"the B2 request must start {B2_START}, got {times[0].isoformat()}")
    return {"days": days, "start": start, "steps": steps,
            "first_time": times[0].isoformat(), "last_time": times[-1].isoformat(),
            "init_times": len(init_times),
            "season": "Jan" if start_time.month == 1 else f"{start_time.month:02d}"}


def _requested_times():
    import pandas as pd

    request = b2_request()
    start = pd.Timestamp(request["first_time"])
    stamps = [start + pd.Timedelta(hours=6 * step) for step in range(request["steps"])]
    if len(stamps) != B2_STAMPS or stamps[-1] != pd.Timestamp(B2_LAST_STAMP):
        raise ValueError(f"the frozen B2 request must be {B2_STAMPS} stamps "
                         f"ending {B2_LAST_STAMP}")
    hours = sorted({stamp.hour for stamp in stamps})
    if hours != [0, 6, 12, 18]:
        raise ValueError("B2 stamps must cover exactly the four UTC initialization hours")
    # The extension must strictly contain the frozen D1 request, or the train
    # range cannot stay byte-identical and the D1 comparison stops transferring.
    if stamps.index(pd.Timestamp(D1_LAST_STAMP)) != 119:
        raise ValueError("the B2 request no longer contains D1's 120 stamps as a prefix")
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
        "purpose": ("B2 multiseed engineering segment (36 consecutive January days "
                    "from the candidate train years); its held-out blocks are wide "
                    "enough for a 72 h free rollout; no journal technique is "
                    "claimable from this data"),
    }


def extract_b2(out_nc, receipt_json, *, deadline_seconds=DEFAULT_DEADLINE_SECONDS,
               decoded_budget_bytes=FIRST_STAGE_DECODED_BYTES_CAP):
    """Download the B2 segment into a new NetCDF plus an honest receipt.

    Nothing is written before the byte budget, the deadline and every metadata
    attestation pass. On failure a ``failed-no-fallback`` receipt is written and
    the exception is re-raised; synthetic data is never substituted.
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
        "decision": "docs/decisions/0008-b2-segment-extension.md",
        "read_plan_protocol_sha256": frozen_protocol()["protocol_sha256"],
        "contains_d1_prefix_stamps": 120,
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
            descriptor, temp = tempfile.mkstemp(prefix=".era5-b2-", suffix=".nc", dir=out_nc.parent)
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
            "b2_request": request,
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
            "one 36-day engineering segment from the candidate train years; all of "
            "it January, so no seasonal, multi-year or test claim survives",
            "B2 verifies windowing/units/IO and holds-out width only; no journal "
            "technique is claimable",
            "stored pressure axis keeps the planned 250/500/850 hPa levels, not the "
            "full audited 13-level axis; the kept levels are explicit in this receipt",
            "network bytes are host-wide interface counters during the reads, not an "
            "icechunk per-request accounting",
            "the 120 D1 stamps are re-read rather than spliced from the archived D1 "
            "artifact; the two acquisitions are separate identities",
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
        "b2_request": request,
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
        description="Frozen-scope Earthmover spatial-namespace B2 extraction (#64 B2)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true",
                      help="read-only audit of layout/coordinates/units/one stamp")
    mode.add_argument("--write", action="store_true",
                      help=f"download the {B2_STAMPS} B2 stamps into a new NetCDF plus receipt")
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
    receipt = extract_b2(args.out, args.receipt, deadline_seconds=args.deadline_seconds,
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
