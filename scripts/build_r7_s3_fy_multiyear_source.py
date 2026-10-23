"""Combine the FY 2000-2022 source with the sealed 2023 four-month block.

The FY source stops at 2022-12-31 and the store builder requires a non-empty test
split, so the round reuses the same sealed 2023 block the annual-coverage round used.
Only the 2023 frames are taken from that file; its 2022 four months duplicate frames
the FY source already carries in full and are dropped rather than repaired.

The merge is asserted, never repaired: both inputs must share latitude, longitude,
level, variables, units and dtype, and the combined axis must be the unique, strictly
increasing, six-hourly concatenation of exactly the frames this module declares.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.download.earthmover_spatial_d1 import _write_netcdf
from data.download.multi_year_source import _assert_compatible

FORMAT = "r7-fy-multiyear-merged-source-v1"
FY_SOURCE = ROOT / "outputs/r7_s3_fy_2000_2022/source.nc"
FY_SHA256 = "b5a8dba01d1b9a1be56e511c61bf69316dd43661c019c54e77b960334756c7df"
FY_STAMPS = 33604
SEALED_SOURCE = ROOT / "outputs/r7_s2_batch2_20222023/source.nc"
SEALED_SHA256 = "44a24ca0096c3802fefc92a5e089c7283812001cfcffd6f88b2e06187607ec62"
SEALED_YEAR, SEALED_STAMPS = 2023, 480
SEALED_SLICE_FIRST, SEALED_SLICE_LAST = "2023-01-01 00:00:00", "2023-10-30 18:00:00"
EXPECTED_STAMPS = FY_STAMPS + SEALED_STAMPS
FIRST_TIME, LAST_TIME = "2000-01-01 00:00:00", "2023-10-30 18:00:00"
FY_LAST_TIME = "2022-12-31 18:00:00"
INIT_HOURS = (0, 6, 12, 18)
STEP = "6h"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_input(path, expected):
    """An input is used only when its bytes still match the frozen acquisition hash."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"missing source artifact: {path}")
    digest = _sha256_file(path)
    if digest != expected:
        raise ValueError(f"source {Path(path).name} has {digest}, expected {expected}")
    return digest


def _stamp(moment):
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _sealed_part(sealed, sealed_times):
    """The 2023 frames only; the file's 2022 four months duplicate the FY source."""
    mask = sealed_times.year == SEALED_YEAR
    if int(mask.sum()) != SEALED_STAMPS:
        raise ValueError(f"sealed {SEALED_YEAR} slice is {int(mask.sum())} frames, "
                         f"expected {SEALED_STAMPS}")
    part = sealed.sel(time=sealed_times[mask])
    part_times = sealed_times[mask]
    if _stamp(part_times[0]) != SEALED_SLICE_FIRST or _stamp(part_times[-1]) != SEALED_SLICE_LAST:
        raise ValueError(f"sealed slice is {_stamp(part_times[0])}..{_stamp(part_times[-1])}, "
                         f"expected {SEALED_SLICE_FIRST}..{SEALED_SLICE_LAST}")
    return part


def _assert_axis(times, fy_count):
    """The FY span must be one contiguous run; the sealed block only has to be on-grid.

    The sealed 2023 block is four months, so the axis carries 59-62 day gaps inside
    2023. Those gaps are the sealed block's shape, not damage: the store builder assigns
    windows per split and the test split stays unread.
    """
    import pandas as pd

    if len(times) != EXPECTED_STAMPS:
        raise ValueError(f"combined source has {len(times)} stamps, expected {EXPECTED_STAMPS}")
    if not times.is_unique or not times.is_monotonic_increasing or times.hasnans:
        raise ValueError("combined time axis must be unique, increasing and valid")
    if _stamp(times[0]) != FIRST_TIME or _stamp(times[-1]) != LAST_TIME:
        raise ValueError(f"combined axis is {_stamp(times[0])}..{_stamp(times[-1])}")
    if sorted({int(stamp.hour) for stamp in times}) != list(INIT_HOURS):
        raise ValueError("combined stamps must cover the four UTC initialization hours")
    fy_span = times[:fy_count]
    gaps = sorted(set(fy_span.to_series().diff().dropna()))
    if gaps != [pd.Timedelta(hours=6)]:
        raise ValueError(f"the FY span is not one contiguous 6-hourly run: {gaps}")
    if _stamp(fy_span[-1]) != FY_LAST_TIME:
        raise ValueError(f"the FY span ends at {_stamp(fy_span[-1])}, expected {FY_LAST_TIME}")
    if times[fy_count] - times[fy_count - 1] != pd.Timedelta(hours=6):
        raise ValueError("the sealed block does not start six hours after the FY span")


def _write(body, out_nc, receipt_json):
    receipt_json.parent.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.mkstemp(prefix=".r7-multiyear-receipt-", suffix=".json",
                                  dir=str(receipt_json.parent))[1]
    Path(descriptor).write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    Path(descriptor).replace(receipt_json)


def _absolute(path):
    """Receipts record repository-relative paths, so resolve before comparing."""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def combine(out_nc, receipt_json):
    import pandas as pd
    import xarray as xr

    out_nc, receipt_json = _absolute(out_nc), _absolute(receipt_json)
    for path in (out_nc, receipt_json):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    fy_digest = _verify_input(FY_SOURCE, FY_SHA256)
    sealed_digest = _verify_input(SEALED_SOURCE, SEALED_SHA256)

    with xr.open_dataset(FY_SOURCE) as fy, xr.open_dataset(SEALED_SOURCE) as sealed:
        fy_times = pd.DatetimeIndex(fy["time"].values)
        if len(fy_times) != FY_STAMPS:
            raise ValueError(f"FY source has {len(fy_times)} frames, expected {FY_STAMPS}")
        part = _sealed_part(sealed, pd.DatetimeIndex(sealed["time"].values))
        entries = [{"path": FY_SOURCE.name}, {"path": SEALED_SOURCE.name}]
        _assert_compatible([fy, part], entries, fy, "time")
        for field, extract in (("units", lambda ds: {n: ds[n].attrs.get("units")
                                                     for n in ds.data_vars}),
                               ("dtype", lambda ds: {n: str(ds[n].dtype)
                                                     for n in ds.data_vars})):
            if extract(fy) != extract(part):
                raise ValueError(f"the two sources disagree on variable {field}")
        attrs = {name: dict(fy[name].attrs) for name in fy.data_vars}
        combined = xr.concat([fy, part], dim="time")
        for name, values in attrs.items():
            combined[name].attrs = values
        times = pd.DatetimeIndex(combined["time"].values)
        _assert_axis(times, FY_STAMPS)
        temp = out_nc.with_name(out_nc.name + ".partial")
        if temp.exists():
            temp.unlink()
        _write_netcdf(combined, temp)
        temp.replace(out_nc)
        months = sorted({int(stamp.month) for stamp in times})
        body = {
            "format": FORMAT, "status": "merged-real-source", "scientific_claim": False,
            "synthetic_fallback": False, "source_is_real_reanalysis": True,
            "inputs": [{"path": str(FY_SOURCE.relative_to(ROOT)), "sha256": fy_digest,
                        "slice": "2000-01-01..2022-12-31", "stamps": FY_STAMPS},
                       {"path": str(SEALED_SOURCE.relative_to(ROOT)), "sha256": sealed_digest,
                        "slice": f"{SEALED_SLICE_FIRST}..{SEALED_SLICE_LAST}",
                        "stamps": SEALED_STAMPS}],
            "stamps": len(times), "time_first": FIRST_TIME, "time_last": LAST_TIME,
            "months": months, "years": [int(times.year.min()), int(times.year.max())],
            "step": STEP,
            "local_artifact": {"path": str(out_nc.relative_to(ROOT)),
                               "bytes": out_nc.stat().st_size, "sha256": _sha256_file(out_nc)},
            "compatibility": {"coordinates_equal": True, "variables_equal": True,
                              "units_equal": True, "dtype": "float32",
                              "levels": int(combined.sizes["level"]),
                              "grid": [int(combined.sizes["latitude"]),
                                       int(combined.sizes["longitude"])]},
            "limitations": [
                "the sealed 2023 block is four months (Jan/Apr/Jul/Oct, days 1-30); it exists only "
                "so the store builder has a non-empty test split, and is never read",
                "the FY span 2000-01-01..2022-12-31 is one contiguous 6-hourly run; the axis carries "
                "59-62 day gaps inside 2023 because the sealed block is four months",
                "the FY source is native 65x65; this merge performs no spatial resampling"],
        }
        _write(body, out_nc, receipt_json)
        return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    body = combine(args.out, args.receipt or (args.out.parent / "merge_receipt.json"))
    print(json.dumps({key: body[key] for key in ("stamps", "time_first", "time_last")}, indent=1))
    print("source.nc", body["local_artifact"]["bytes"], body["local_artifact"]["sha256"])


if __name__ == "__main__":
    main()
