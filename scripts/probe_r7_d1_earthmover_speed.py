"""Bounded Earthmover icechunk read-speed probe for the D1 estimate (#63).

Uses the repository's production namespace (`{single,pressure}/temporal`) and
the frozen D1 window shape: 17 channels, 65x65 ROI (27-43N/107-123E), 6-hourly
stamps. Measures real elapsed time and real network bytes (/proc/net/dev).
Read-only against the anonymous public snapshot; the only write is a JSON
receipt inside this repository's outputs directory.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = "ZFKDHBCTBVHVXM3BQFV0"
LAT_BAND = (27.0, 43.0)
LON_BAND = (107.0, 123.0)
CONCURRENCY = 1  # production pattern in earthmover_pilot

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_DIR = REPO_ROOT / "outputs"
RECEIPT_NAME = "speedtest_earthmover_temporal.json"


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


def main():
    import icechunk as ic
    import zarr
    from icechunk.storage import s3_storage

    storage = s3_storage(
        bucket="earthmover-icechunk-era5",
        prefix="icechunkV2",
        region="us-east-1",
        anonymous=True,
    )
    clock = time.time()
    repo = ic.Repository.open(storage)
    session = repo.readonly_session(snapshot_id=SNAPSHOT)
    with zarr.config.set({"async.concurrency": CONCURRENCY}):
        root = zarr.open_group(store=session.store, mode="r")
        raw = np.asarray(root["single/temporal/valid_time"][:])
        times = pd.Timestamp("1940-01-01") + pd.to_timedelta(raw, unit="h")
        keep = (
            (times >= pd.Timestamp("2016-01-01T00:00"))
            & (times <= pd.Timestamp("2016-01-30T18:00"))
            & (times.hour % 6 == 0)
        )
        stamps = np.flatnonzero(keep)
        latitude = np.asarray(root["single/temporal/latitude"][:])
        longitude = np.asarray(root["single/temporal/longitude"][:])
        levels = [int(v) for v in root["pressure/temporal/pressure_level"][:]]
        print(f"open+coords {time.time() - clock:.1f}s | stamps={len(stamps)}", flush=True)

        yi = np.flatnonzero(
            (latitude >= LAT_BAND[0] - 1e-9) & (latitude <= LAT_BAND[1] + 1e-9)
        )
        xi = np.flatnonzero(
            (longitude >= LON_BAND[0] - 1e-9) & (longitude <= LON_BAND[1] + 1e-9)
        )
        if (len(yi), len(xi)) != (65, 65):
            raise ValueError(f"expected a 65x65 ROI, got {len(yi)}x{len(xi)}")
        level_index = {level: index for index, level in enumerate(levels)}

        surface = ["t2m", "u10", "v10", "msl"]
        pressure = [
            ("z", 850), ("t", 850), ("q", 850), ("u", 850), ("v", 850),
            ("z", 500), ("t", 500), ("q", 500), ("u", 500), ("v", 500),
            ("z", 250), ("u", 250), ("v", 250),
        ]

        def read_stamp(index):
            count = 0
            for name in surface:
                block = np.asarray(
                    root[f"single/temporal/{name}"].oindex[[index], yi, xi],
                    dtype="f4",
                )
                if not np.isfinite(block).all():
                    raise ValueError(f"nonfinite {name}")
                count += 1
            for name, level in pressure:
                block = np.asarray(
                    root[f"pressure/temporal/{name}"].oindex[
                        [index], level_index[level], yi, xi
                    ],
                    dtype="f4",
                )
                if not np.isfinite(block).all():
                    raise ValueError(f"nonfinite {name}{level}")
                count += 1
            return count

        reads = read_stamp(int(stamps[0]))
        print(f"warmup stamp: {reads} array reads", flush=True)

        probe = [int(v) for v in stamps[::40][:3]]
        before = net_bytes()
        clock = time.time()
        for index in probe:
            read_stamp(index)
        elapsed = time.time() - clock
        mib = (net_bytes() - before) / 2**20
        per_stamp = elapsed / len(probe)
        per_stamp_mib = mib / len(probe)
        print(
            f"{len(probe)} stamps ({reads} reads each): {elapsed:.1f}s "
            f"net={mib:.1f} MiB ({per_stamp:.2f}s/stamp, {per_stamp_mib:.1f} MiB/stamp)",
            flush=True,
        )
        d1_minutes = per_stamp * 120 / 60
        print(
            f"-> D1 120 stamps => {d1_minutes:.1f} min, "
            f"{per_stamp_mib * 120 / 1024:.2f} GiB",
            flush=True,
        )
        receipt = {
            "source": "earthmover-icechunk-temporal-namespace",
            "snapshot": SNAPSHOT,
            "concurrency": CONCURRENCY,
            "roi": {"lat": list(LAT_BAND), "lon": list(LON_BAND), "points": [65, 65]},
            "arrays_per_stamp": reads,
            "probe_stamps": len(probe),
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
