"""Bounded concurrency pilot for the full-year (FY) acquisition.

Measures what the FY downloader actually costs the host before a 32-hour batch is
frozen: for each pool width it runs that many month-parts concurrently, samples
``/proc/net/dev`` once a second, and reports the achieved per-part rate together
with the **host-wide share of the link**.

The share is the reason this script exists. ``earthmover_spatial_fy`` (like S1)
runs one zarr stream per process with ``async.concurrency=1``, so parallelism is
the shell pool; until now the pool width was chosen from throughput alone and the
host cost was never measured. Here a hard guard aborts the whole pool when the
sampled host receive rate stays above ``--max-host-mb-s``, so a neighbour's
traffic cannot be starved by this experiment. The guard only ever signals its own
children (decision 0026 / R-054 keep non-experiment processes untouched).

Read-only against the source; writes only under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONSECUTIVE_BREACHES_TO_ABORT = 3
SAMPLE_SECONDS = 1.0


def _net_counters():
    """Per-interface rx/tx byte counters from /proc/net/dev, loopback excluded."""
    counters = {}
    with open("/proc/net/dev", encoding="ascii") as handle:
        for line in handle.readlines()[2:]:
            name, _, rest = line.partition(":")
            fields = rest.split()
            if len(fields) < 16 or name.strip() == "lo":
                continue
            counters[name.strip()] = (int(fields[0]), int(fields[8]))
    return counters


def _link_bytes_per_second(interface):
    """The nominal link rate of one interface in bytes/s, or None if unknown."""
    try:
        speed = int(Path(f"/sys/class/net/{interface}/speed").read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return None if speed <= 0 else speed * 1_000_000 / 8


class _Children:
    """Registry of this pilot's own children, so the guard can only signal them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._live = []

    def add(self, process):
        with self._lock:
            self._live.append(process)

    def discard(self, process):
        with self._lock:
            if process in self._live:
                self._live.remove(process)

    def terminate_all(self):
        with self._lock:
            live = list(self._live)
        for process in live:
            if process.poll() is None:
                # start_new_session put each child in its own group, so the whole
                # part is signalled without touching any other process.
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)


def _run_part(pool_dir, year, month, deadline_seconds, max_decoded_gib, children):
    """One month part as its own process, exactly as the batch will run it."""
    stem = f"fy{year}{month:02d}"
    out_nc = pool_dir / f"{stem}.nc"
    receipt = pool_dir / f"{stem}.receipt.json"
    command = [sys.executable, "-m", "data.download.earthmover_spatial_fy",
               "--write", "--years", str(year), "--months", str(month),
               "--out", str(out_nc), "--receipt", str(receipt),
               "--deadline-seconds", str(deadline_seconds),
               "--max-decoded-gib", str(max_decoded_gib)]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    children.add(process)
    try:
        stdout, stderr = process.communicate()
    finally:
        children.discard(process)
    return {"part": stem, "year": year, "month": month,
            "returncode": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": (stdout or "").strip()[-2000:], "stderr": (stderr or "").strip()[-2000:]}


def _sampler(stop, samples, interface_totals):
    previous = _net_counters()
    previous_at = time.monotonic()
    while not stop.is_set():
        time.sleep(SAMPLE_SECONDS)
        now, now_at = _net_counters(), time.monotonic()
        span = now_at - previous_at
        rates = {}
        for name, (rx, _tx) in now.items():
            if name in previous:
                delta = max(0, rx - previous[name][0])
                rates[name] = delta / span
                interface_totals[name] = interface_totals.get(name, 0) + delta
        samples.append({"at_seconds": round(now_at - previous_at, 3),
                        "rx_bytes_per_second": rates, "total": sum(rates.values())})
        previous, previous_at = now, now_at


def _summarise(samples):
    if not samples:
        return {"n_samples": 0}
    totals = [entry["total"] for entry in samples]
    by_interface = {}
    for entry in samples:
        for name, rate in entry["rx_bytes_per_second"].items():
            by_interface.setdefault(name, []).append(rate)
    return {
        "n_samples": len(samples),
        "host_rx_mean_bytes_per_second": sum(totals) / len(totals),
        "host_rx_peak_bytes_per_second": max(totals),
        "per_interface_peak_bytes_per_second": {name: max(values)
                                                for name, values in by_interface.items()},
        "per_interface_mean_bytes_per_second": {name: sum(values) / len(values)
                                                for name, values in by_interface.items()},
    }


def run_pool(pool, parts, out_dir, *, deadline_seconds, max_decoded_gib, max_host_mb_s,
             link_cap_bytes_per_second=None):
    """Run ``parts`` with at most ``pool`` concurrent processes, sampling the host."""
    pool_dir = Path(out_dir) / f"pool{pool}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    samples, interface_totals = [], {}
    sampler = threading.Thread(target=_sampler, args=(stop, samples, interface_totals), daemon=True)
    sampler.start()
    guard = {"aborted": False, "breaches": 0, "peak": 0.0}
    children = _Children()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=pool) as pool_executor:
        futures = [pool_executor.submit(_run_part, pool_dir, year, month,
                                        deadline_seconds, max_decoded_gib, children)
                   for year, month in parts]
        while any(not future.done() for future in futures):
            time.sleep(SAMPLE_SECONDS)
            if max_host_mb_s is not None and samples:
                rate = samples[-1]["total"]
                guard["peak"] = max(guard["peak"], rate)
                guard["breaches"] = guard["breaches"] + 1 if rate > max_host_mb_s * 1e6 else 0
                if guard["breaches"] >= CONSECUTIVE_BREACHES_TO_ABORT:
                    guard["aborted"] = True
                    children.terminate_all()
                    break
        results = [future.result() for future in futures]
    stop.set()
    sampler.join(timeout=5)
    wall = time.monotonic() - started
    done = [entry for entry in results if entry["returncode"] == 0]
    stamps_ok = sum(calendar.monthrange(entry["year"], entry["month"])[1] * 4 for entry in done)
    received = sum(interface_totals.values())
    summary = _summarise(samples)
    active = max(summary.get("per_interface_mean_bytes_per_second", {}) or {"": 0},
                 key=summary.get("per_interface_mean_bytes_per_second", {"": 0}).get)
    report = {
        "pool": pool, "parts": [f"{year}-{month:02d}" for year, month in parts],
        "n_parts": len(parts), "n_ok": len(done),
        "wall_seconds": round(wall, 3),
        "host_rx_bytes_during_run": int(received),
        "aggregate_bytes_per_second": (received / wall) if wall else None,
        "sampling": summary,
        "active_interface": active or None,
        "active_interface_link_bytes_per_second": _link_bytes_per_second(active) if active else None,
        "guard": {"max_host_mb_s": max_host_mb_s, "aborted": guard["aborted"],
                  "peak_mb_s": round(guard["peak"] / 1e6, 3)},
        "parts_detail": results,
        "stamps_ok": stamps_ok,
    }
    if link_cap_bytes_per_second:
        report["host_share_of_link_mean"] = (summary.get("host_rx_mean_bytes_per_second", 0)
                                             / link_cap_bytes_per_second)
        report["host_share_of_link_peak"] = (summary.get("host_rx_peak_bytes_per_second", 0)
                                             / link_cap_bytes_per_second)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="FY concurrency pilot with a host-network guard.")
    parser.add_argument("--out-dir", required=True, help="fresh directory for this pilot's parts")
    parser.add_argument("--pools", type=int, nargs="+", required=True,
                        help="pool widths, each run with exactly that many parts")
    parser.add_argument("--first-year", type=int, default=2000)
    parser.add_argument("--report", required=True, help="new JSON report path")
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    parser.add_argument("--max-decoded-gib", type=float, default=16.0)
    parser.add_argument("--max-host-mb-s", type=float, default=60.0,
                        help="abort the pool after 3 consecutive seconds above this host rx rate")
    parser.add_argument("--link-mb-s", type=float, default=1000.0,
                        help="nominal link rate used only to express the host share")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing non-empty pilot directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit(f"refusing existing report: {report_path}")

    # A contiguous run of months from --first-year, consumed in pool order, so no
    # two pool levels ever fetch the same month.
    available = [(args.first_year, month) for month in range(1, 13)]
    reports, cursor = [], 0
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        """Rewrite the report after every level, so an interrupted pilot still yields data."""
        report_path.write_text(json.dumps({
            "format": "r7-fy-concurrency-pilot-v1",
            "out_dir": str(out_dir), "pools": args.pools,
            "max_host_mb_s": args.max_host_mb_s, "link_mb_s": args.link_mb_s,
            "guard_rule": (f"abort the pool after {CONSECUTIVE_BREACHES_TO_ABORT} consecutive "
                           f"{SAMPLE_SECONDS:g}s samples above --max-host-mb-s; only this pilot's "
                           "own child processes are signalled"),
            "reports": reports, "complete": len(reports) == len(args.pools),
            "scientific_claim": False,
            "limitations": ["a single host, a single day and a single source snapshot; the share "
                            "is measured against the nominal link rate, not a reserved allocation"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    publish()
    for pool in args.pools:
        parts = available[cursor:cursor + pool]
        cursor += pool
        if len(parts) < pool:
            raise SystemExit(f"not enough months left for pool={pool}")
        print(f"[pilot] pool={pool} parts={parts}", flush=True)
        result = run_pool(pool, parts, out_dir, deadline_seconds=args.deadline_seconds,
                          max_decoded_gib=args.max_decoded_gib,
                          max_host_mb_s=args.max_host_mb_s,
                          link_cap_bytes_per_second=args.link_mb_s * 1e6 / 8)
        print(f"[pilot] pool={pool} n_ok={result['n_ok']} wall={result['wall_seconds']}s "
              f"agg={result['aggregate_bytes_per_second']} B/s "
              f"host_mean={result['sampling'].get('host_rx_mean_bytes_per_second')} B/s "
              f"aborted={result['guard']['aborted']}", flush=True)
        reports.append(result)
        publish()
        if result["guard"]["aborted"]:
            print("[pilot] host guard tripped; stopping further pool levels", flush=True)
            break
    print(f"[pilot] wrote {report_path}")


if __name__ == "__main__":
    main()
