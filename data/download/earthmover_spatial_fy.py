"""Bounded Earthmover extraction of **full-year** regional ERA5 segments (FY).

Derived from ``earthmover_spatial_s1`` without modifying it: the S1 path is frozen
to the four 30-day season windows, while this path tiles every calendar month so
the train span gets uniform full-year coverage. Source, snapshot, ROI, channels,
level axis and every budget/attestation gate are identical to S1 - only the
requested time windows differ (``season_plan_fy``).

Reused verbatim from the audited modules: the pinned session, the namespace/units
attestation, the one-chunk-cropped ROI reads, the NetCDF writer, the dataset
assembly and the part/merge bookkeeping. Nothing is written until every budget,
deadline and attestation passes; failures leave ``failed-no-fallback``, never a
synthetic substitute, and each part stays under the 30-minute process bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from .earthmover_pilot import SOURCE, SNAPSHOT, DecodedBudget
from .earthmover_spatial_d1 import (
    _channel_payloads,
    _collect_frames,
    _dataset_attributes,
    _net_recv_bytes,
    _open_pinned_session,
    _write_netcdf,
    validate_spatial_namespace,
)
from .earthmover_spatial_s1 import _assemble_dataset, _merge_parts, _sha256_file
from .read_plan_frozen import SECOND_STAGE_DECODED_BYTES_CAP, frozen_protocol
from .season_plan_fy import (  # noqa: F401 - re-exported for the CLI and tests
    DEFAULT_DEADLINE_SECONDS, MAX_STAMPS_PER_PART, MAX_YEARS_PER_BATCH, MONTHS,
    STAMPS_PER_DAY, full_year_plan, plan_report, planned_stamps,
)

SEGMENT = "fy-full-year-multi-year"
ROI = {"south": 27.0, "north": 43.0, "west": 107.0, "east": 123.0,
       "points": [65, 65], "spacing_deg": 0.25}


def _second_stage_protocol():
    # Reuse the audited second-stage plan; each batch declares its own byte cap.
    return frozen_protocol(stage="second",
                           new_artifact_bytes_cap=100 * 2**30,
                           decoded_bytes_cap=SECOND_STAGE_DECODED_BYTES_CAP)


def _fy_part_receipt(result, plan, plan_read, payloads, channel_names, *, estimated, budget,
                     network_bytes, elapsed):
    """The measured fields of a successful full-year part, apart from the read loop."""
    result = dict(result)
    result.update({
        "status": "downloaded-real-source",
        "source_is_real_reanalysis": True,
        "protocol": {
            "full_year_plan": plan,
            "roi": ROI,
            "channels": channel_names,
            "channel_count": len(channel_names),
            "stored_level_axis_hpa": plan_read["shared_levels"],
            "selection": "exact 6-hourly UTC timestamps; no nearest substitution",
        },
        "timestamps": [stamp.isoformat() for stamp in plan_read["times"]],
        "init_hour_coverage": {str(hour): int(sum(1 for stamp in plan_read["times"] if stamp.hour == hour))
                               for hour in (0, 6, 12, 18)},
        "channel_units": plan_read["observed_units"],
        "level_attestation": plan_read["level_attestation"],
        "shape": [int(v) for v in (len(plan_read["times"]), len(plan_read["shared_levels"]),
                                   len(plan_read["latitude"]), len(plan_read["longitude"]))],
        "payloads": payloads,
        "decoded_chunk_budget": {
            "cap_bytes": int(budget.limit), "estimated_field_bytes": int(estimated),
            "charged_bytes": int(budget.used), "chunk_reads": int(budget.reads),
            "budget_scope": ("sum of uncompressed touched spatial-chunk sizes (fields plus "
                             "coordinates); NOT the measured network transfer and not peak RAM"),
        },
        "network_body_bytes": int(network_bytes),
        "network_method": ("recv-byte delta over non-loopback /proc/net/dev interfaces "
                           "around the field reads; host-wide counter"),
        "network_measurement_caveat": ("the counter is host-wide, so when parts run "
                                       "concurrently each part's window also counts its "
                                       "siblings' traffic and this figure overstates its own "
                                       "transfer; under a pool it is not additive"),
        "elapsed_seconds": round(elapsed, 3),
        "limitations": [
            "a month window is a calendar-month sample; the train-only month-hour "
            "climatology shares the sampled months and the ROI is the single 27-43N/107-123E box",
            "network bytes are host-wide counters, not icechunk per-request accounting; "
            "the stored pressure axis keeps the planned levels, not all 13 audited levels",
        ],
    })
    return result


def extract_fy_part(out_nc, receipt_json, *, years, months=MONTHS,
                    deadline_seconds=DEFAULT_DEADLINE_SECONDS,
                    decoded_budget_bytes=SECOND_STAGE_DECODED_BYTES_CAP):
    """Download one nonempty subset of the calendar months of ``years``.

    Blocks are stored in the frozen plan order with every stamp recorded.
    """
    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    plan = full_year_plan(years, months=months)
    all_stamps = planned_stamps(plan)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    protocol = _second_stage_protocol()
    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    result: dict = {
        "status": "pending",
        "segment": SEGMENT,
        "part": {"years": plan["years"], "months": plan["months"],
                 "first_time": all_stamps[0].isoformat(), "last_time": all_stamps[-1].isoformat(),
                 "stamps": len(all_stamps), "blocks": plan["blocks"]},
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "read_plan_stage": protocol["stage"],
        "read_plan_protocol_sha256": protocol["protocol_sha256"],
        "budget_caps": {"decoded_budget_bytes": int(decoded_budget_bytes),
                        "seconds": float(deadline_seconds)},
        "synthetic_fallback": False,
        "scientific_claim": False,
    }
    try:
        session, ic, zarr = _open_pinned_session()
        budget = DecodedBudget(limit=int(decoded_budget_bytes), seconds=float(deadline_seconds),
                               started=started)
        with zarr.config.set({"async.concurrency": 1}):
            root = zarr.open_group(store=session.store, mode="r")
            plan_read = validate_spatial_namespace(root, all_stamps, budget)
            estimated = plan_read["per_stamp_field_bytes"] * len(plan_read["stamps"])
            budget.check(estimated)
            network_before = _net_recv_bytes()
            frames = _collect_frames(plan_read, budget, deadline)
            network_bytes = _net_recv_bytes() - network_before
            dataset = _assemble_dataset(plan_read, frames, session.snapshot_id)
            payloads, channel_names = _channel_payloads(dataset)
            descriptor, temp = tempfile.mkstemp(prefix=".era5-fy-", suffix=".nc", dir=out_nc.parent)
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
    result = _fy_part_receipt(result, plan, plan_read, payloads, channel_names,
                              estimated=estimated, budget=budget, network_bytes=network_bytes,
                              elapsed=time.monotonic() - started)
    result["local_artifact"] = {"path": out_nc.name, "bytes": out_nc.stat().st_size,
                                "sha256": digest}
    result.update(icechunk_version=ic.__version__, zarr_version=zarr.__version__)
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def merge_fy_parts(part_pairs, out_nc, receipt_json, *, expected_stamps,
                   require_contiguous=False):
    """Concatenate full-year parts, re-asserting exact timestamp identity.

    ``require_contiguous`` additionally asserts the merged axis is one unbroken
    6-hourly run, which holds when every calendar month of every year is present.
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
    merged, datasets, recorded, planned = _merge_parts(part_pairs)
    times = pd.DatetimeIndex(merged["time"].values)
    if int(expected_stamps) != len(planned):
        raise ValueError(f"expected {expected_stamps} stamps, part receipts plan {len(planned)}")
    if len(times) != int(expected_stamps):
        raise ValueError(f"merged source has {len(times)} stamps, expected {expected_stamps}")
    expected_axis = pd.DatetimeIndex(sorted(set(planned)))
    if len(expected_axis) != len(planned):
        raise ValueError("part receipts overlap: duplicate planned stamps")
    if not times.equals(expected_axis):
        raise ValueError("merged time axis differs from the union of planned stamps")
    if not times.is_monotonic_increasing or not times.is_unique or times.hasnans:
        raise ValueError("merged time axis must be unique, increasing and valid")
    if sorted({stamp.hour for stamp in times}) != [0, 6, 12, 18]:
        raise ValueError("merged stamps must cover the four UTC initialization hours")
    if require_contiguous:
        contiguous = pd.date_range(times[0], times[-1], freq="6h")
        if not times.equals(contiguous):
            raise ValueError(f"merged axis is not one contiguous 6-hourly run "
                             f"({len(times)} stamps vs {len(contiguous)} expected)")
    attrs = dict(datasets[0].attrs)
    attrs.update(_dataset_attributes(attrs.get("source_snapshot_id", SNAPSHOT)))
    merged.attrs.update(attrs)
    descriptor, temp = tempfile.mkstemp(prefix=".era5-fy-merge-", suffix=".nc", dir=out_nc.parent)
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
        "segment": SEGMENT,
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "merge_rule": ("parts are concatenated in the given order along the time axis; the "
                       "merged axis is re-asserted to equal the exact union of every part's "
                       "recorded stamps, unique, increasing and on the four UTC init hours"
                       + ("; the axis is additionally asserted to be one contiguous 6-hourly run"
                          if require_contiguous else "")),
        "require_contiguous": bool(require_contiguous),
        "parts": recorded,
        "n_parts": len(recorded),
        "stamps": int(len(times)),
        "time_first": str(times[0]),
        "time_last": str(times[-1]),
        "network_body_bytes_total": int(sum(entry["network_body_bytes"] for entry in recorded)),
        "network_method": ("sum of each part's own recv-byte delta over non-loopback "
                           "/proc/net/dev interfaces; host-wide counters"),
        "network_measurement_caveat": ("each part's figure is host-wide, so if the parts were "
                                       "downloaded concurrently this sum counts siblings' traffic "
                                       "once per running part and overstates the real transfer; "
                                       "the batch driver's own host sample is the accurate "
                                       "batch-level figure"),
        "channels": channel_names,
        "channel_count": len(channel_names),
        "payloads": payloads,
        "local_artifact": {"path": out_nc.name, "bytes": out_nc.stat().st_size, "sha256": digest},
        "synthetic_fallback": False,
        "scientific_claim": False,
        "source_is_real_reanalysis": True,
        "limitations": [
            "the total network figure sums each part's host-wide counter window, not one "
            "continuous measurement; each part is a calendar month of one ROI",
        ],
    }
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded Earthmover full-year (every calendar month) extraction.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the frozen plan and estimates")
    mode.add_argument("--write", action="store_true", help="download one month part")
    mode.add_argument("--merge", action="store_true", help="concatenate month parts")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--months", type=int, nargs="+", choices=list(MONTHS))
    parser.add_argument("--out", help="write/merge: new NetCDF path")
    parser.add_argument("--receipt", help="write/merge: new receipt JSON path")
    parser.add_argument("--parts", nargs="+", help="merge only: part NetCDF paths, in order")
    parser.add_argument("--part-receipts", nargs="+",
                        help="merge only: each part's receipt JSON, same order as --parts")
    parser.add_argument("--expected-stamps", type=int, help="merge only: required total")
    parser.add_argument("--require-contiguous", action="store_true",
                        help="merge only: assert one unbroken 6-hourly axis")
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    parser.add_argument("--max-decoded-gib", type=float,
                        default=SECOND_STAGE_DECODED_BYTES_CAP / 2**30)
    args = parser.parse_args()
    months = tuple(args.months) if args.months else MONTHS
    if args.plan:
        print(json.dumps(plan_report(args.years, months=months), indent=2, ensure_ascii=False))
        return
    if args.merge:
        if not args.parts or not args.part_receipts or len(args.parts) != len(args.part_receipts):
            parser.error("--merge requires --parts and --part-receipts of equal length")
        if not args.out or not args.receipt or args.expected_stamps is None:
            parser.error("--merge requires --out, --receipt and --expected-stamps")
        result = merge_fy_parts(list(zip(args.parts, args.part_receipts)),
                                args.out, args.receipt, expected_stamps=args.expected_stamps,
                                require_contiguous=args.require_contiguous)
        print(json.dumps({"status": result["status"], "stamps": result["stamps"],
                          "network_bytes_total": result["network_body_bytes_total"],
                          "sha256": result["local_artifact"]["sha256"]}))
        return
    if not args.out or not args.receipt:
        parser.error("--write requires --out and --receipt")
    result = extract_fy_part(args.out, args.receipt, years=args.years, months=months,
                             deadline_seconds=args.deadline_seconds,
                             decoded_budget_bytes=int(args.max_decoded_gib * 2**30))
    print(json.dumps({"status": result["status"], "part": result["part"],
                      "stamps": len(result["timestamps"]),
                      "network_bytes": result["network_body_bytes"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "sha256": result["local_artifact"]["sha256"]}))


if __name__ == "__main__":
    main()
