"""Read-only check: a wide-region NetCDF's central block is the frozen target box.

The wide instance's whole scientific value rests on one property - the central
65x65 block of the 129x129 grid must be the *same data* as the registered
27-43N/107-123E target box, not merely the same coordinates. If it is not, the
wide run cannot be scored on the registered climatology and the comparison it
exists to make is void.

This tool compares the overlapping stamps channel by channel against the
registered v3 ``source.nc``: coordinates, per-variable units, the nine stored
source variables at their three levels, and the 17 stacked R7 channels. It
reports bit-identity (``np.array_equal``) and the maximum absolute difference,
writes nothing except an explicitly requested JSON report, opens no network
socket and no CUDA device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _central(values, rows, columns):
    """Select the central (rows, columns) block of the trailing two axes."""
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("a central block needs at least two spatial axes")
    index = (slice(None),) * (array.ndim - 2) + (rows, columns)
    return array[index]


def compare_central_block(wide_path, reference_path, *, rows, columns, channels=17):
    """Compare the wide file's central block against the reference at shared times."""
    import pandas as pd
    import xarray as xr

    wide_path, reference_path = Path(wide_path), Path(reference_path)
    report: dict = {
        "format": "r7-wide-target-block-check-v1",
        "scientific_claim": False,
        "wide_path": wide_path.name,
        "wide_sha256": _sha256_file(wide_path),
        "reference_path": reference_path.name,
        "reference_sha256": _sha256_file(reference_path),
        "central_block": {"rows": [rows.start, rows.stop], "columns": [columns.start, columns.stop],
                          "points": [rows.stop - rows.start, columns.stop - columns.start]},
        "network": 0,
        "gpu": 0,
    }
    with xr.open_dataset(wide_path) as wide, xr.open_dataset(reference_path) as reference:
        wide_times = pd.DatetimeIndex(wide["time"].values)
        reference_times = pd.DatetimeIndex(reference["time"].values)
        positions = reference_times.get_indexer(wide_times)
        missing = [str(wide_times[i]) for i, value in enumerate(positions) if value < 0]
        if missing:
            report["verdict"] = "fail"
            report["error"] = f"{len(missing)} wide stamps absent from the reference: {missing[:3]}"
            return report
        report["stamps"] = int(len(wide_times))
        report["first_time"] = str(wide_times[0])
        report["last_time"] = str(wide_times[-1])

        coordinates = {}
        for name, selection in (("latitude", slice(rows.start, rows.stop)),
                                ("longitude", slice(columns.start, columns.stop))):
            # The wide file carries the margin; the reference *is* the target box,
            # so only the wide side is sliced and the two must then have one shape.
            wide_values = np.asarray(wide[name].values)[selection]
            reference_values = np.asarray(reference[name].values)
            if wide_values.shape != reference_values.shape:
                coordinates[name] = {"shape_match": False,
                                     "wide_shape": list(wide_values.shape),
                                     "reference_shape": list(reference_values.shape)}
                continue
            coordinates[name] = {
                "bit_identical": bool(np.array_equal(wide_values, reference_values)),
                "max_abs_diff": float(np.max(np.abs(wide_values.astype(np.float64)
                                                  - reference_values.astype(np.float64)))),
            }
        report["coordinates"] = coordinates

        report["stored_variables"] = _compare_variables(wide, reference, positions,
                                                          rows, columns)

        report["stacked_channels"] = _compare_stacked(wide, reference, positions, rows,
                                                     columns)
        if report["stacked_channels"].get("count") != channels:
            report["stacked_channels"]["count_matches_declared"] = False

    def _all_identical(section):
        return all(value.get("bit_identical", False)
                   for value in section.values() if "bit_identical" in value)

    def _units_match(section):
        return all(value.get("unit_wide") == value.get("unit_reference")
                   for value in section.values() if "unit_wide" in value)

    report["units_match"] = _units_match(report["stored_variables"])
    report["verdict"] = "pass" if (
        _all_identical(report["coordinates"])
        and _all_identical(report["stored_variables"])
        and report["units_match"]
        and report["stacked_channels"].get("bit_identical") is True
        and report["stacked_channels"].get("count") == channels
    ) else "fail"
    return report


def _compare_variables(wide, reference, positions, rows, columns):
    """Per-stored-variable comparison of the wide central block against the reference."""
    variables = {}
    for name in wide.data_vars:
        if name not in reference.data_vars:
            variables[name] = {"present_in_reference": False}
            continue
        wide_block = _central(wide[name].values, rows, columns)
        # The reference is already the target box: only the wide side is cropped.
        reference_block = np.asarray(reference[name].values[positions])
        if wide_block.shape != reference_block.shape:
            variables[name] = {"shape_match": False,
                               "wide_shape": list(wide_block.shape),
                               "reference_shape": list(reference_block.shape)}
            continue
        difference = np.abs(wide_block.astype(np.float64) - reference_block.astype(np.float64))
        variables[name] = {
            "bit_identical": bool(np.array_equal(wide_block, reference_block)),
            "max_abs_diff": float(np.max(difference)) if difference.size else 0.0,
            "unit_wide": str(wide[name].attrs.get("units", "")),
            "unit_reference": str(reference[name].attrs.get("units", "")),
        }
    return variables


def _compare_stacked(wide, reference, positions, rows, columns, channels=17):
    """Comparison of the 17 stacked R7 channels over the wide central block."""
    from data.preprocess.r7_era5 import DEFAULT_R7_ERA5_CHANNELS, stack_era5_channels

    wide_state, wide_names, _, _, _ = stack_era5_channels(wide, DEFAULT_R7_ERA5_CHANNELS)
    reference_state, reference_names, _, _, _ = stack_era5_channels(reference, DEFAULT_R7_ERA5_CHANNELS)
    if list(wide_names) != list(reference_names):
        return {"names_match": False}
    wide_block = _central(wide_state, rows, columns)
    reference_block = reference_state[positions]
    return {
        "names_match": True, "count": int(len(wide_names)), "channels": list(wide_names),
        "bit_identical": bool(np.array_equal(wide_block, reference_block)),
        "max_abs_diff": float(np.max(np.abs(wide_block.astype(np.float64)
                                          - reference_block.astype(np.float64)))),
    }


def main() -> None:
    from data.download.read_plan_wide import central_block

    parser = argparse.ArgumentParser(
        description="Check that a wide-region NetCDF's central block is the frozen target box.")
    parser.add_argument("--wide", required=True, help="wide-region NetCDF to check")
    parser.add_argument("--reference", required=True,
                        help="registered target-box NetCDF (e.g. the v3 instance source.nc)")
    parser.add_argument("--report", help="new JSON report path")
    parser.add_argument("--rows", type=int, default=None, help="central block row start")
    parser.add_argument("--columns", type=int, default=None, help="central block column start")
    parser.add_argument("--points", type=int, default=None, help="central block side length")
    parser.add_argument("--channels", type=int, default=17)
    args = parser.parse_args()
    if args.rows is None or args.columns is None or args.points is None:
        rows, columns = central_block()
    else:
        rows, columns = slice(args.rows, args.rows + args.points), slice(args.columns, args.columns + args.points)
    report = compare_central_block(args.wide, args.reference, rows=rows, columns=columns,
                                   channels=args.channels)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        path = Path(args.report)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
