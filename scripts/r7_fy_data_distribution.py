"""Read-only distribution audit of the data the campaign is currently training on.

Answers the questions a data-expansion decision turns on, from the store and the
split manifests alone - nothing is written outside ``--report``:

- what the annual cycle coverage actually is (which months, days and init hours
  each split contains, and therefore which months the model has never seen);
- per-channel physical-unit min/p01/p50/p99/max/mean/std over the frozen ROI;
- the 16 month-hour climatology buckets the skill denominator is built from, with
  their sample counts, so the reader can see how thin each bucket is;
- the stored normalization constants and the years they were fit on.

The store is opened read-only and the test manifest is never read.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

#: Frozen ROI inside the wide grid: 129 = 65 + 2 * 32, so the core box is the
#: middle 65 cells. Asserted against the store's coordinates before use.
ROI_MARGIN = 32
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _roi_slice(latitude, longitude):
    """The frozen 65x65 ROI inside the wide grid, validated against the coordinates."""
    lat, lon = np.asarray(latitude), np.asarray(longitude)
    height, width = len(lat), len(lon)
    if (height, width) != (129, 129):
        raise ValueError(f"expected the wide 129x129 grid, got {height}x{width}")
    ys, xs = slice(ROI_MARGIN, height - ROI_MARGIN), slice(ROI_MARGIN, width - ROI_MARGIN)
    if (ys.stop - ys.start, xs.stop - xs.start) != (65, 65):
        raise ValueError("ROI slice is not 65x65")
    lat_roi, lon_roi = lat[ys], lon[xs]
    if not (np.isclose(np.abs(lat_roi).min(), 27.0, atol=1e-4)
            and np.isclose(np.abs(lat_roi).max(), 43.0, atol=1e-4)):
        raise ValueError(f"ROI latitude span is not 27..43N: {lat_roi.min()}..{lat_roi.max()}")
    if not (np.isclose(lon_roi.min(), 107.0, atol=1e-4)
            and np.isclose(lon_roi.max(), 123.0, atol=1e-4)):
        raise ValueError(f"ROI longitude span is not 107..123E: {lon_roi.min()}..{lon_roi.max()}")
    return ys, xs


def _manifest_stamps(path):
    import pandas as pd

    stamps = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            value = record.get("time") or record.get("init_time") or record.get("timestamp")
            stamps.append(pd.Timestamp(value))
    return pd.DatetimeIndex(stamps)


def _coverage(stamps):
    """Which years/months/days/hours a split contains, and the 16 month-hour buckets."""
    import pandas as pd

    buckets = collections.Counter()
    for stamp in stamps:
        buckets[(int(stamp.month), int(stamp.hour))] += 1
    return {
        "n": int(len(stamps)),
        "years": {int(y): int(c) for y, c in sorted(collections.Counter(
            int(s.year) for s in stamps).items())},
        "months": {MONTH_NAMES[m - 1]: int(c) for m, c in sorted(collections.Counter(
            int(s.month) for s in stamps).items())},
        "days_of_month": {int(d): int(c) for d, c in sorted(collections.Counter(
            int(s.day) for s in stamps).items())},
        "init_hours": {int(h): int(c) for h, c in sorted(collections.Counter(
            int(s.hour) for s in stamps).items())},
        "month_hour_buckets": {f"{MONTH_NAMES[m - 1]}_{h:02d}Z": int(c)
                               for (m, h), c in sorted(buckets.items())},
        "months_present": sorted({MONTH_NAMES[int(s.month) - 1] for s in stamps}),
        "months_absent": [name for index, name in enumerate(MONTH_NAMES)
                          if (index + 1) not in {int(s.month) for s in stamps}],
        "first": str(stamps.min()) if len(stamps) else None,
        "last": str(stamps.max()) if len(stamps) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only distribution audit of the training data.")
    parser.add_argument("--store", required=True, help="cache.zarr of the instance to audit")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--report", required=True, help="new JSON report path")
    parser.add_argument("--label", default="current-instance")
    args = parser.parse_args()

    import zarr

    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit(f"refusing existing report: {report_path}")

    root = zarr.open_group(args.store, mode="r")
    attrs = dict(root.attrs)
    channels = list(attrs["channels"])
    units = list(attrs["units"])
    ys, xs = _roi_slice(root["latitude"][:], root["longitude"][:])
    state = root["state"]

    train = _manifest_stamps(args.train_manifest)
    val = _manifest_stamps(args.val_manifest)
    coverage = {"train": _coverage(train), "val": _coverage(val)}

    # Restrict the per-channel statistics to the frames the trainer actually sees.
    import pandas as pd

    store_times = pd.DatetimeIndex(pd.to_datetime(root["time_ns"][:], unit="ns"))
    wanted = np.isin(store_times.values, train.values)
    if int(wanted.sum()) != len(train):
        raise ValueError(f"train manifest stamps {len(train)} not all present in the store "
                         f"({int(wanted.sum())} matched)")
    rows = np.flatnonzero(wanted)

    stats = {}
    for index, name in enumerate(channels):
        values = []
        for start in range(0, len(rows), 256):
            block = np.asarray(state[rows[start:start + 256], index, ys, xs], dtype=np.float64)
            values.append(block.reshape(-1))
        flat = np.concatenate(values)
        finite = flat[np.isfinite(flat)]
        percentiles = np.percentile(finite, [1, 50, 99])
        stats[name] = {
            "units": units[index], "n": int(finite.size),
            "min": float(finite.min()), "p01": float(percentiles[0]),
            "p50": float(percentiles[1]), "p99": float(percentiles[2]),
            "max": float(finite.max()), "mean": float(finite.mean()), "std": float(finite.std()),
        }

    normalization = {
        "fit_years": [int(y) for y in attrs["normalization_years"]],
        "rule": attrs["normalization"],
        "mean": {name: float(root["normalization_mean"][i]) for i, name in enumerate(channels)},
        "std": {name: float(root["normalization_std"][i]) for i, name in enumerate(channels)},
    }

    result = {
        "format": "r7-fy-data-distribution-v1",
        "label": args.label,
        "store": str(args.store),
        "store_shape": list(state.shape),
        "channels": channels,
        "roi": {"points": [65, 65], "slice": [ROI_MARGIN, 129 - ROI_MARGIN],
                "south": 27.0, "north": 43.0, "west": 107.0, "east": 123.0,
                "note": "the frozen 65x65 box inside the wide 129x129 grid"},
        "coverage": coverage,
        "channel_stats_over_roi_train_frames": stats,
        "normalization": normalization,
        "test_read": False,
        "test_read_policy": "the test manifest is never read by this audit",
        "scientific_claim": False,
        "limitations": [
            "statistics describe the stored split and ROI only; no claim about ERA5 at large",
            "percentiles are computed over every grid cell of every train frame, so they are "
            "dominated by spatial degrees of freedom and are not independent samples",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "train_n": coverage["train"]["n"], "val_n": coverage["val"]["n"],
        "train_months_present": coverage["train"]["months_present"],
        "train_months_absent": coverage["train"]["months_absent"],
        "train_buckets": len(coverage["train"]["month_hour_buckets"]),
        "train_days": len(coverage["train"]["days_of_month"]),
        "init_hours": sorted(coverage["train"]["init_hours"]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
