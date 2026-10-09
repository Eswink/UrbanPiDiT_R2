"""Bounded Earthmover extraction of the wide-ROI (S3 lateral-context) segment.

Source and snapshot are the frozen ones
(``s3://earthmover-icechunk-era5/icechunkV2`` snapshot ``ZFKDHBCTBVHVXM3BQFV0``,
anonymous, CC-BY-4.0). The single difference from the frozen S1/S3 acquisition is
the region: ``read_plan_wide`` declares the 129x129 19-51N/99-131E box, whose
central 65x65 block is asserted to be the frozen 27-43N/107-123E target box.

The region validation, unit/level attestation, dataset assembly and read-only
preflight live in ``earthmover_wide_io``; this module owns the part, merge and CLI
bookkeeping, mirroring ``earthmover_spatial_s1``.

Nothing is written until every budget, deadline and attestation passes; a failure
leaves ``failed-no-fallback`` and re-raises, and synthetic data is never
substituted.
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

from .earthmover_pilot import SNAPSHOT, SOURCE, DecodedBudget
from .earthmover_spatial_d1 import (
    _channel_payloads,
    _collect_frames,
    _net_recv_bytes,
    _open_pinned_session,
    _write_netcdf,
)
from .earthmover_spatial_s1 import _merge_parts
from .earthmover_wide_io import (  # noqa: F401 - re-exported for the CLI and tests
    _assemble_wide_dataset,
    _wide_dataset_attributes,
    preflight_wide,
    source_citation,
    validate_wide_namespace,
)
from .read_plan_wide import (
    SPACING_DEG,
    WIDE_DECODED_BYTES_CAP,
    WIDE_EAST,
    WIDE_NORTH,
    WIDE_POINTS,
    WIDE_SOUTH,
    WIDE_WEST,
    containment_evidence,
    wide_frozen_protocol,
)
from .season_plan_s1 import (  # noqa: F401 - re-exported for the CLI and tests
    DEFAULT_DEADLINE_SECONDS,
    MAX_SEASONS_PER_YEAR,
    MAX_YEARS_PER_BATCH,
    SEASON_MONTHS,
    _planned_stamps,
    plan_report,
    season_plan,
)


def extract_wide_part(out_nc, receipt_json, *, years, seasons,
                      deadline_seconds=DEFAULT_DEADLINE_SECONDS,
                      decoded_budget_bytes=WIDE_DECODED_BYTES_CAP):
    """Download one nonempty subset of the four season blocks of ``years``, wide."""
    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    plan = season_plan(years, season_names=tuple(seasons))
    all_stamps = _planned_stamps(plan)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    protocol = wide_frozen_protocol(stamps=len(all_stamps))
    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    result: dict = {
        "status": "pending",
        "segment": "w1-wide-region-multi-year",
        "part": {"years": plan["years"], "seasons": plan["season_names"],
                 "first_time": all_stamps[0].isoformat(),
                 "last_time": all_stamps[-1].isoformat(),
                 "stamps": len(all_stamps), "blocks": plan["blocks"]},
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "citation": source_citation(),
        "read_plan_format": protocol["format"],
        "read_plan_protocol_sha256": protocol["protocol_sha256"],
        "region": protocol["region"],
        "target_box_containment": protocol["target_box"]["containment"],
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
            plan_read = validate_wide_namespace(root, all_stamps, budget)
            estimated = plan_read["per_stamp_field_bytes"] * len(plan_read["stamps"])
            budget.check(estimated)
            network_before = _net_recv_bytes()
            frames = _collect_frames(plan_read, budget, deadline)
            network_bytes = _net_recv_bytes() - network_before
            dataset = _assemble_wide_dataset(plan_read, frames, session.snapshot_id)
            payloads, channel_names = _channel_payloads(dataset)
            descriptor, temp = tempfile.mkstemp(prefix=".era5-w1-", suffix=".nc", dir=out_nc.parent)
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
    result = _wide_part_receipt(result, plan_read, payloads, channel_names,
                               (out_nc, digest),
                               {"estimated": estimated, "budget": budget,
                                "network_bytes": network_bytes,
                                "elapsed": time.monotonic() - started,
                                "icechunk_version": ic.__version__,
                                "zarr_version": zarr.__version__})
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def _wide_part_receipt(result, plan_read, payloads, channel_names, artifact, measured):
    """The measured fields of one successful wide part, kept apart from the read loop."""
    out_nc, digest = artifact
    estimated, budget = measured["estimated"], measured["budget"]
    network_bytes, elapsed = measured["network_bytes"], measured["elapsed"]
    icechunk_version, zarr_version = measured["icechunk_version"], measured["zarr_version"]
    result = dict(result)
    result.update({
        "status": "downloaded-real-source",
        "source_is_real_reanalysis": True,
        "channels": channel_names,
        "channel_count": len(channel_names),
        "channel_units": plan_read["observed_units"],
        "level_attestation": plan_read["level_attestation"],
        "stored_level_axis_hpa": plan_read["shared_levels"],
        "timestamps": [stamp.isoformat() for stamp in plan_read["times"]],
        "init_hour_coverage": {str(hour): int(sum(1 for stamp in plan_read["times"]
                                                  if stamp.hour == hour))
                               for hour in (0, 6, 12, 18)},
        "shape": [int(v) for v in (len(plan_read["times"]), len(plan_read["shared_levels"]),
                                   len(plan_read["latitude"]), len(plan_read["longitude"]))],
        "payloads": payloads,
        "selection": "exact 6-hourly UTC timestamps; no nearest substitution",
        "decoded_chunk_budget": {
            "cap_bytes": int(budget.limit), "estimated_field_bytes": int(estimated),
            "charged_bytes": int(budget.used), "chunk_reads": int(budget.reads),
            "budget_scope": ("sum of uncompressed touched spatial-chunk sizes (fields plus "
                             "coordinates); NOT the measured network transfer and not peak RAM"),
        },
        "network_body_bytes": int(network_bytes),
        "network_method": ("recv-byte delta over non-loopback /proc/net/dev interfaces "
                           "around the field reads; host-wide counter"),
        "elapsed_seconds": round(elapsed, 3),
        "local_artifact": {"path": out_nc.name, "bytes": out_nc.stat().st_size,
                           "sha256": digest},
        "icechunk_version": icechunk_version,
        "zarr_version": zarr_version,
        "limitations": [
            "one wide segment; no skill, generalization or seasonal claim follows from it",
            "network bytes are host-wide interface counters, not icechunk per-request accounting",
            "the stored pressure axis keeps the planned levels, not all 13 audited levels",
            "an 8-degree half-width is a partial answer to synoptic advection (see read_plan_wide)",
        ],
    })
    return result


def merge_wide_parts(part_pairs, out_nc, receipt_json, *, expected_stamps):
    """Concatenate wide parts, re-asserting exact timestamp identity."""
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
    if int(expected_stamps) != len(planned) or len(times) != int(expected_stamps):
        raise ValueError(f"expected {expected_stamps} stamps, parts plan {len(planned)}, "
                         f"merged {len(times)}")
    expected_axis = pd.DatetimeIndex(sorted(set(planned)))
    if len(expected_axis) != len(planned):
        raise ValueError("part receipts overlap: duplicate planned stamps")
    if not times.equals(expected_axis):
        raise ValueError("merged time axis differs from the union of planned stamps")
    if not times.is_monotonic_increasing or not times.is_unique or times.hasnans:
        raise ValueError("merged time axis must be unique, increasing and valid")
    if sorted({stamp.hour for stamp in times}) != [0, 6, 12, 18]:
        raise ValueError("merged stamps must cover the four UTC initialization hours")
    if [int(v) for v in merged["latitude"].shape] != [WIDE_POINTS] \
            or [int(v) for v in merged["longitude"].shape] != [WIDE_POINTS]:
        raise ValueError("merged parts are not on the wide region grid")
    attrs = dict(datasets[0].attrs)
    attrs.update(_wide_dataset_attributes(attrs.get("source_snapshot_id", SNAPSHOT)))
    merged.attrs.update(attrs)
    descriptor, temp = tempfile.mkstemp(prefix=".era5-w1-merge-", suffix=".nc", dir=out_nc.parent)
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
        "segment": "w1-wide-region-multi-year",
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "read_plan_protocol_sha256": wide_frozen_protocol(stamps=int(expected_stamps))["protocol_sha256"],
        "region": {"south": WIDE_SOUTH, "north": WIDE_NORTH, "west": WIDE_WEST,
                   "east": WIDE_EAST, "points": [WIDE_POINTS, WIDE_POINTS],
                   "spacing_deg": SPACING_DEG},
        "target_box_containment": containment_evidence(),
        "merge_rule": ("parts are concatenated in the given order along the time axis; the merged "
                       "axis is re-asserted to equal the exact union of every part's recorded "
                       "stamps, unique, increasing and on the four UTC init hours"),
        "parts": recorded,
        "n_parts": len(recorded),
        "stamps": int(len(times)),
        "time_first": str(times[0]),
        "time_last": str(times[-1]),
        "network_body_bytes_total": int(sum(entry["network_body_bytes"] for entry in recorded)),
        "network_method": ("sum of each part's own recv-byte delta over non-loopback "
                           "/proc/net/dev interfaces; host-wide counters"),
        "channels": channel_names,
        "channel_count": len(channel_names),
        "payloads": payloads,
        "local_artifact": {"path": out_nc.name, "bytes": out_nc.stat().st_size, "sha256": digest},
        "synthetic_fallback": False,
        "scientific_claim": False,
        "source_is_real_reanalysis": True,
        "limitations": [
            "the total network figure sums each part's host-wide counter window, not one "
            "continuous measurement",
            "the wide region is a development instance; the frozen target box is unchanged and "
            "its climatology is rebuilt from train years only",
        ],
    }
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded Earthmover wide-region extraction (S3 lateral context).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the frozen plan and estimates")
    mode.add_argument("--preflight", action="store_true",
                      help="read-only audit of region/coordinates/units/one stamp")
    mode.add_argument("--write", action="store_true", help="download one season part")
    mode.add_argument("--merge", action="store_true", help="concatenate season parts")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--seasons", nargs="+", choices=sorted(SEASON_MONTHS))
    parser.add_argument("--report", help="preflight only: new JSON report path")
    parser.add_argument("--out", help="write/merge: new NetCDF path")
    parser.add_argument("--receipt", help="write/merge: new receipt JSON path")
    parser.add_argument("--parts", nargs="+", help="merge only: part NetCDF paths, in order")
    parser.add_argument("--part-receipts", nargs="+",
                        help="merge only: each part's receipt JSON, same order as --parts")
    parser.add_argument("--expected-stamps", type=int, help="merge only: required total")
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    parser.add_argument("--max-decoded-gib", type=float, default=WIDE_DECODED_BYTES_CAP / 2**30)
    args = parser.parse_args()
    seasons = tuple(args.seasons) if args.seasons else tuple(SEASON_MONTHS)
    if args.plan:
        report = plan_report(args.years, season_names=seasons)
        report["wide_read_plan"] = wide_frozen_protocol(stamps=report["plan"]["total_stamps"])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.preflight:
        preflight_wide(args.report, years=tuple(args.years), seasons=seasons)
        return
    if args.merge:
        if not args.parts or not args.part_receipts or len(args.parts) != len(args.part_receipts):
            parser.error("--merge requires --parts and --part-receipts of equal length")
        if not args.out or not args.receipt or args.expected_stamps is None:
            parser.error("--merge requires --out, --receipt and --expected-stamps")
        result = merge_wide_parts(list(zip(args.parts, args.part_receipts)), args.out,
                                  args.receipt, expected_stamps=args.expected_stamps)
        print(json.dumps({"status": result["status"], "stamps": result["stamps"],
                          "network_bytes_total": result["network_body_bytes_total"],
                          "sha256": result["local_artifact"]["sha256"]}))
        return
    if not args.out or not args.receipt:
        parser.error("--write requires --out and --receipt")
    result = extract_wide_part(args.out, args.receipt, years=args.years, seasons=seasons,
                               deadline_seconds=args.deadline_seconds,
                               decoded_budget_bytes=int(args.max_decoded_gib * 2**30))
    print(json.dumps({"status": result["status"], "part": result["part"],
                      "stamps": len(result["timestamps"]),
                      "network_bytes": result["network_body_bytes"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "sha256": result["local_artifact"]["sha256"]}))


if __name__ == "__main__":
    main()
