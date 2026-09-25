"""Bounded ARCO wb13 read-speed probe for the D1 acquisition estimate (#63).

Measures real elapsed time and real network bytes (via /proc/net/dev) for the
frozen D1 window shape: 17 channels, 65x65 ROI (27-43N/107-123E), 6-hourly
stamps. Read-only against the anonymous public source; the only write is a
JSON receipt inside this repository's outputs directory.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

URL = "gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
LAT = slice(43, 27)
LON = slice(107, 123)
SURF = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
PRES = [
    ("geopotential", 850), ("temperature", 850), ("specific_humidity", 850),
    ("u_component_of_wind", 850), ("v_component_of_wind", 850),
    ("geopotential", 500), ("temperature", 500), ("specific_humidity", 500),
    ("u_component_of_wind", 500), ("v_component_of_wind", 500),
    ("geopotential", 250), ("u_component_of_wind", 250), ("v_component_of_wind", 250),
]
# Receipt stays inside the repo's ignored tool-state directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_DIR = REPO_ROOT / "outputs"
RECEIPT_NAME = "speedtest_arco_wb13.json"


def receipt_path() -> Path:
    """Resolve the receipt path and refuse anything outside outputs/."""
    target = (RECEIPT_DIR / RECEIPT_NAME).resolve()
    if target.parent != RECEIPT_DIR.resolve():
        raise ValueError("receipt must stay inside the outputs directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def net_bytes():
    total = 0
    with open("/proc/net/dev", encoding="utf-8") as handle:
        for line in handle:
            if ":" in line and not line.strip().startswith("lo"):
                total += int(line.split(":")[1].split()[0])
    return total


def read_stamp(dataset, stamp):
    count = 0
    for name in SURF:
        values = dataset[name].sel(time=stamp, latitude=LAT, longitude=LON).values
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {name}")
        count += 1
    for name, level in PRES:
        values = dataset[name].sel(
            time=stamp, level=level, latitude=LAT, longitude=LON
        ).values
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {name}{level}")
        count += 1
    return count


def main():
    started = time.time()
    dataset = xr.open_zarr(URL, chunks=None, storage_options={"token": "anon"})
    print(f"open {time.time() - started:.1f}s", flush=True)

    before = net_bytes()
    clock = time.time()
    read_stamp(dataset, "2016-01-01T06:00")
    warmup = time.time() - clock
    warmup_mib = (net_bytes() - before) / 2**20
    print(f"warmup+1 stamp: {warmup:.1f}s net={warmup_mib:.1f} MiB", flush=True)

    stamps = [
        str(pd.Timestamp("2016-01-01") + pd.Timedelta(hours=24 * i))
        for i in range(1, 5)
    ]
    before = net_bytes()
    clock = time.time()
    for stamp in stamps:
        read_stamp(dataset, stamp)
    elapsed = time.time() - clock
    mib = (net_bytes() - before) / 2**20
    per_stamp = elapsed / len(stamps)
    per_stamp_mib = mib / len(stamps)
    print(
        f"{len(stamps)} stamps: {elapsed:.1f}s net={mib:.1f} MiB "
        f"({per_stamp:.2f}s/stamp, {per_stamp_mib:.1f} MiB/stamp)",
        flush=True,
    )
    d1_minutes = per_stamp * 120 / 60
    print(
        f"-> D1 120 stamps => {d1_minutes:.1f} min, "
        f"{per_stamp_mib * 120 / 1024:.2f} GiB",
        flush=True,
    )
    receipt = {
        "source": "arco-wb13-6h",
        "url": URL,
        "warmup_s": warmup,
        "warmup_mib": warmup_mib,
        "stamps": len(stamps),
        "elapsed_s": elapsed,
        "net_mib": mib,
        "per_stamp_s": per_stamp,
        "per_stamp_mib": per_stamp_mib,
        "d1_minutes": d1_minutes,
        "d1_gib": per_stamp_mib * 120 / 1024,
        "scientific_claim": False,
        "note": "read-speed probe for the frozen D1 window shape; not skill",
    }
    destination = receipt_path()
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1)
    print(f"saved {destination.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
