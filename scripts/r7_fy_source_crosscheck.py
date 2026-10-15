"""Cross-check the FY reader against the registered store, cell for cell.

Reads one stamp the registered wide instance already contains straight from the
pinned source through the full-year path, and compares it bit for bit with the
store's own copy over the frozen 65x65 target box. This is the check that decides
whether the new path may be used at all: if it reproduces the registered data
exactly, every reading the campaign already produced is directly comparable with
anything the full-year data will produce; if it does not, the path is unusable.

Read-only against both the store and the source; writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SURFACE = {"t2m": "2m_temperature", "u10": "10m_u_component_of_wind",
           "v10": "10m_v_component_of_wind", "mslp": "mean_sea_level_pressure"}
PRESSURE = {"z": "geopotential", "t": "temperature", "q": "specific_humidity",
            "u": "u_component_of_wind", "v": "v_component_of_wind"}
LEVELS = (250, 500, 850)
#: 129 = 65 + 2 * 32, so the frozen target box is the middle 65 cells of the wide grid.
ROI_MARGIN = 32


def _source_plane(dataset, channel):
    if channel in SURFACE:
        return np.asarray(dataset[SURFACE[channel]].values[0], dtype=np.float32)
    variable, level = PRESSURE[channel[:-3]], int(channel[-3:])
    return np.asarray(dataset[variable].values[0, LEVELS.index(level)], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="FY-vs-registered-store cell-for-cell check.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--stamp", default="2017-01-01T06:00")
    parser.add_argument("--report", help="optional new JSON report path")
    args = parser.parse_args()

    import pandas as pd
    import zarr

    from data.download.earthmover_pilot import DecodedBudget
    from data.download.earthmover_spatial_d1 import (
        _collect_frames, _open_pinned_session, validate_spatial_namespace,
    )
    from data.download.earthmover_spatial_s1 import _assemble_dataset

    stamp = pd.Timestamp(args.stamp)
    root = zarr.open_group(args.store, mode="r")
    channels = list(root.attrs["channels"])
    times = pd.DatetimeIndex(pd.to_datetime(root["time_ns"][:], unit="ns"))
    matches = np.flatnonzero(times == stamp)
    if len(matches) != 1:
        raise SystemExit(f"stamp {stamp} is not in the store exactly once: {len(matches)}")
    index = int(matches[0])
    limit = 129 - ROI_MARGIN
    store_block = np.asarray(root["state"][index, :, ROI_MARGIN:limit, ROI_MARGIN:limit])
    print(f"store stamp #{index} = {times[index]}  shape {store_block.shape}")

    started = time.monotonic()
    session, _ic, zarr_module = _open_pinned_session()
    budget = DecodedBudget(limit=16 * 2**30, seconds=900.0, started=started)
    with zarr_module.config.set({"async.concurrency": 1}):
        groups = zarr_module.open_group(store=session.store, mode="r")
        plan_read = validate_spatial_namespace(groups, pd.DatetimeIndex([stamp]), budget)
        frames = _collect_frames(plan_read, budget, started + 900.0)
        dataset = _assemble_dataset(plan_read, frames, session.snapshot_id)

    print(f"\n{'channel':7} {'store':>14} {'source':>14} {'max|diff|':>12} {'identical':>10}")
    per_channel, all_identical = {}, True
    for position, channel in enumerate(channels):
        source_block = _source_plane(dataset, channel)
        if source_block.shape != store_block[position].shape:
            raise SystemExit(f"{channel}: {source_block.shape} vs {store_block[position].shape}")
        difference = float(np.max(np.abs(source_block - store_block[position])))
        identical = bool((source_block == store_block[position]).all())
        all_identical = all_identical and identical
        per_channel[channel] = {"max_abs_difference": difference, "bit_identical": identical,
                                "store_mean": float(store_block[position].mean()),
                                "source_mean": float(source_block.mean())}
        print(f"{channel:7} {store_block[position].mean():>14.4f} {source_block.mean():>14.4f} "
              f"{difference:>12.3e} {str(identical):>10}")

    print(f"\nALL CHANNELS BIT-IDENTICAL: {all_identical}")
    if args.report:
        report_path = Path(args.report)
        if report_path.exists():
            raise SystemExit(f"refusing existing report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "format": "r7-fy-source-crosscheck-v1",
            "store": str(args.store), "stamp": str(stamp), "store_index": index,
            "roi": {"margin": ROI_MARGIN, "points": [65, 65]},
            "channels": per_channel,
            "all_channels_bit_identical": bool(all_identical),
            "scientific_claim": False,
            "limitations": ["one stamp and one ROI; it establishes value-level agreement between "
                            "the two read paths, not correctness of ERA5 itself"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {report_path}")
    if not all_identical:
        raise SystemExit("the FY path does not reproduce the registered store; do not use it")


if __name__ == "__main__":
    main()
