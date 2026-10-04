"""Bounded Earthmover extraction of the S1 four-season, multi-year segments.

Scope is frozen per receipt: source
``s3://earthmover-icechunk-era5/icechunkV2`` snapshot ``ZFKDHBCTBVHVXM3BQFV0``
(anonymous), ROI 27-43N/107-123E at 65x65/0.25 deg, the frozen 17 channels and
the ``SeasonPlan`` from ``season_plan_s1`` (see ``R7_S0_INCUMBENT_GAP_AUDIT.md``).

The audited D1 helpers do the attestation and one-chunk-cropped ROI reads; this
module owns the part/merge bookkeeping. Nothing is written until every budget,
deadline and attestation passes; failures leave ``failed-no-fallback``, never a
synthetic substitute, and parts stay under the 30-minute process bound.
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
    _channel_payloads,
    _collect_frames,
    _dataset_attributes,
    _net_recv_bytes,
    _open_pinned_session,
    _write_netcdf,
    validate_spatial_namespace,
)
from .read_plan_frozen import SECOND_STAGE_DECODED_BYTES_CAP, frozen_protocol
from .season_plan_s1 import (  # noqa: F401 - re-exported for the CLI and tests
    DEFAULT_DEADLINE_SECONDS, DEFAULT_PART_DAYS, MAX_SEASONS_PER_YEAR, MAX_YEARS_PER_BATCH,
    SEASON_DAYS, SEASON_MONTHS, SEASON_STAMPS, STAMPS_PER_DAY, _planned_stamps, plan_report,
    season_plan,
)


def _second_stage_protocol():
    # Reuse the audited second-stage plan; each batch declares its own byte cap.
    return frozen_protocol(stage="second",
                           new_artifact_bytes_cap=100 * 2**30,
                           decoded_bytes_cap=SECOND_STAGE_DECODED_BYTES_CAP)


def _assemble_dataset(plan_read, frames, snapshot_id):
    """Build the NetCDF dataset from validated frames, units and metadata.

    The caller owns budget/deadline bookkeeping; this owns the field layout, the
    per-variable unit attestation and the snapshot attributes.
    """
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
    dataset.attrs.update(_dataset_attributes(snapshot_id))
    return dataset


def _part_receipt(result, plan, plan_read, payloads, channel_names, *, estimated, budget,
                  network_bytes, elapsed):
    """The measured fields of a successful part, kept apart from the read loop."""
    result = dict(result)
    result.update({
        "status": "downloaded-real-source",
        "source_is_real_reanalysis": True,
        "protocol": {
            "season_plan": plan,
            "roi": {"south": 27.0, "north": 43.0, "west": 107.0, "east": 123.0,
                    "points": [65, 65], "spacing_deg": 0.25},
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
        "elapsed_seconds": round(elapsed, 3),
        "limitations": [
            "a season window is a ~30-day sample; the train-only month-hour climatology "
            "shares the sampled months and the ROI is the single 27-43N/107-123E box",
            "network bytes are host-wide counters, not icechunk per-request accounting; "
            "the stored pressure axis keeps the planned levels, not all 13 audited levels",
        ],
    })
    return result


def extract_seasons_part(out_nc, receipt_json, *, years, seasons, deadline_seconds=DEFAULT_DEADLINE_SECONDS,
                         decoded_budget_bytes=SECOND_STAGE_DECODED_BYTES_CAP):
    """Download one nonempty subset of the four season blocks of ``years``.

    Blocks are stored in the frozen season order with every stamp recorded.
    """
    import pandas as pd
    import xarray as xr

    out_nc, receipt_json = Path(out_nc), Path(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    plan = season_plan(years, season_names=tuple(seasons))
    all_stamps = _planned_stamps(plan)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    protocol = _second_stage_protocol()
    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    result: dict = {
        "status": "pending",
        "segment": "s1-four-season-multi-year",
        "part": {"years": plan["years"], "seasons": plan["season_names"],
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
            descriptor, temp = tempfile.mkstemp(prefix=".era5-s1-", suffix=".nc", dir=out_nc.parent)
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
    result = _part_receipt(result, plan, plan_read, payloads, channel_names,
                           estimated=estimated, budget=budget, network_bytes=network_bytes,
                           elapsed=time.monotonic() - started)
    result["local_artifact"] = {"path": out_nc.name, "bytes": out_nc.stat().st_size,
                                "sha256": digest}
    result.update(icechunk_version=ic.__version__, zarr_version=zarr.__version__)
    with receipt_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def _merge_parts(part_pairs):
    """Validate and concatenate the parts; return the merged set and its records."""
    import pandas as pd
    import xarray as xr

    datasets, recorded, planned = [], [], []
    for nc_path, rc_path in part_pairs:
        nc_path, rc_path = Path(nc_path), Path(rc_path)
        if not nc_path.is_file() or not rc_path.is_file():
            raise FileNotFoundError(f"missing part artifact: {nc_path} / {rc_path}")
        with rc_path.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        if receipt.get("status") != "downloaded-real-source":
            raise ValueError(f"part {nc_path.name} is not a completed download: {receipt.get('status')!r}")
        if receipt["local_artifact"]["sha256"] != _sha256_file(nc_path):
            raise ValueError(f"part {nc_path.name} does not match its receipt hash; "
                             "refusing to merge unverified input")
        dataset = xr.open_dataset(nc_path)
        datasets.append(dataset)
        planned.extend(pd.Timestamp(value) for value in receipt["timestamps"])
        recorded.append({
            "path": nc_path.name, "receipt": rc_path.name,
            "stamps": int(dataset.sizes["time"]),
            "first_time": str(pd.Timestamp(dataset["time"].values[0])),
            "last_time": str(pd.Timestamp(dataset["time"].values[-1])),
            "blocks": receipt["part"]["blocks"],
            "bytes": nc_path.stat().st_size,
            "sha256": receipt["local_artifact"]["sha256"],
            "network_body_bytes": int(receipt.get("network_body_bytes", 0)),
            "elapsed_seconds": receipt.get("elapsed_seconds"),
        })
    merged = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    return merged, datasets, recorded, planned


def merge_season_parts(part_pairs, out_nc, receipt_json, *, expected_stamps):
    """Concatenate season parts, re-asserting exact timestamp identity.

    The union is not contiguous, so the merged axis is re-checked against the
    recorded union of planned stamps instead of a fixed-range 6-hourly range.
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
    attrs = dict(datasets[0].attrs)
    attrs.update(_dataset_attributes(attrs.get("source_snapshot_id", SNAPSHOT)))
    merged.attrs.update(attrs)
    descriptor, temp = tempfile.mkstemp(prefix=".era5-s1-merge-", suffix=".nc", dir=out_nc.parent)
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
        "segment": "s1-four-season-multi-year",
        "source": SOURCE,
        "snapshot_id": SNAPSHOT,
        "access": "anonymous-public-s3-icechunk",
        "layout": "earthmover-spatial-namespace (one global field per time)",
        "merge_rule": ("parts are concatenated in the given order along the time axis; the "
                       "merged axis is re-asserted to equal the exact union of every part's "
                       "recorded stamps, unique, increasing and on the four UTC init hours"),
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
            "continuous measurement; each season is a ~30-day sample of one ROI",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded Earthmover four-season multi-year extraction (S1).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the frozen plan and estimates")
    mode.add_argument("--preflight", action="store_true",
                      help="read-only audit of layout/coordinates/units/one stamp; no download")
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
    parser.add_argument("--max-decoded-gib", type=float,
                        default=SECOND_STAGE_DECODED_BYTES_CAP / 2**30)
    args = parser.parse_args()
    seasons = tuple(args.seasons) if args.seasons else tuple(SEASON_MONTHS)
    if args.plan:
        print(json.dumps(plan_report(args.years, season_names=seasons), indent=2, ensure_ascii=False))
        return
    if args.preflight:
        from .earthmover_spatial_d1 import preflight as d1_preflight

        # The audited D1 preflight validates namespace/units/levels on one stamp;
        # the season plan only changes which stamps are requested later.
        d1_preflight(args.report)
        return
    if args.merge:
        if not args.parts or not args.part_receipts or len(args.parts) != len(args.part_receipts):
            parser.error("--merge requires --parts and --part-receipts of equal length")
        if not args.out or not args.receipt or args.expected_stamps is None:
            parser.error("--merge requires --out, --receipt and --expected-stamps")
        result = merge_season_parts(list(zip(args.parts, args.part_receipts)),
                                    args.out, args.receipt,
                                    expected_stamps=args.expected_stamps)
        print(json.dumps({"status": result["status"], "stamps": result["stamps"],
                          "network_bytes_total": result["network_body_bytes_total"],
                          "sha256": result["local_artifact"]["sha256"]}))
        return
    if not args.out or not args.receipt:
        parser.error("--write requires --out and --receipt")
    result = extract_seasons_part(args.out, args.receipt, years=args.years, seasons=seasons,
                                  deadline_seconds=args.deadline_seconds,
                                  decoded_budget_bytes=int(args.max_decoded_gib * 2**30))
    print(json.dumps({"status": result["status"], "part": result["part"],
                      "stamps": len(result["timestamps"]),
                      "network_bytes": result["network_body_bytes"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "sha256": result["local_artifact"]["sha256"]}))


if __name__ == "__main__":
    main()
