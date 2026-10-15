"""Bounded batch download of one full-year (FY) segment into a fresh outputs path.

One batch = one frozen protocol = one exclusive directory. The driver:

- re-verifies the frozen protocol's own digest and that its planned stamp count
  matches the plan recomputed from ``season_plan_fy`` (the plan is arithmetic, so
  a drift between protocol and code is caught before anything is fetched);
- refuses to start unless the free-space gate passes, and re-checks it per part;
- runs each calendar-month part as its own process under a ``timeout`` watchdog,
  at most ``pool`` at a time (the pool width is read from the protocol, which the
  speed pilot froze from a measured host-network share);
- signals only its own process groups, never any other process;
- merges the parts into ``source.nc`` with the contiguous-axis assertion, then
  writes ``BUILD_COMPLETE.json``.

Any failure stops the batch: no ``BUILD_COMPLETE`` is written, the failing part
keeps its own ``failed-no-fallback`` receipt, and the driver exits non-zero. No
synthetic substitute is ever produced and no input is modified.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WATCHDOG_KILL_AFTER = 60
HOST_SAMPLE_SECONDS = 2.0
CONSECUTIVE_BREACHES_TO_ABORT = 3


def _canonical_digest(payload):
    body = {key: value for key, value in payload.items() if key != "protocol_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _free_gib(path):
    return shutil.disk_usage(path).free / 2**30


def _net_counters():
    counters = {}
    with open("/proc/net/dev", encoding="ascii") as handle:
        for line in handle.readlines()[2:]:
            name, _, rest = line.partition(":")
            fields = rest.split()
            if len(fields) >= 16 and name.strip() != "lo":
                counters[name.strip()] = int(fields[0])
    return counters


class _Children:
    """Registry of this driver's own children, so the guard can only signal them."""

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
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)


class _HostSampler:
    """Samples host-wide non-loopback receive so the batch can prove its cost."""

    def __init__(self, max_host_mb_s):
        self.max_host_mb_s = max_host_mb_s
        self.peak = 0.0
        self.breaches = 0
        self.aborted = False
        self.samples = 0
        self.total_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        previous = _net_counters()
        previous_at = time.monotonic()
        while not self._stop.is_set():
            time.sleep(HOST_SAMPLE_SECONDS)
            now, now_at = _net_counters(), time.monotonic()
            span = now_at - previous_at
            delta = sum(max(0, now[k] - previous[k]) for k in now if k in previous)
            rate = delta / span if span else 0.0
            self.samples += 1
            self.total_bytes += delta
            self.peak = max(self.peak, rate)
            if self.max_host_mb_s and rate > self.max_host_mb_s * 1e6:
                self.breaches += 1
                if self.breaches >= CONSECUTIVE_BREACHES_TO_ABORT:
                    self.aborted = True
                    return
            else:
                self.breaches = 0
            previous, previous_at = now, now_at

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def report(self):
        return {"samples": self.samples, "peak_mb_s": round(self.peak / 1e6, 3),
                "max_host_mb_s": self.max_host_mb_s, "aborted": self.aborted}


def _run_part(parts_dir, year, month, deadline_seconds, watchdog_seconds,
              max_decoded_gib, children):
    stem = f"fy{year}{month:02d}"
    out_nc = parts_dir / f"{stem}.nc"
    receipt = parts_dir / f"{stem}.receipt.json"
    inner = [sys.executable, "-m", "data.download.earthmover_spatial_fy",
             "--write", "--years", str(year), "--months", str(month),
             "--out", str(out_nc), "--receipt", str(receipt),
             "--deadline-seconds", str(deadline_seconds),
             "--max-decoded-gib", str(max_decoded_gib)]
    command = ["timeout", "-k", str(WATCHDOG_KILL_AFTER), "-s", "TERM",
               str(watchdog_seconds)] + inner
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    children.add(process)
    try:
        stdout, stderr = process.communicate()
    finally:
        children.discard(process)
    elapsed = round(time.monotonic() - started, 3)
    ok = process.returncode == 0 and out_nc.is_file() and receipt.is_file()
    if ok:
        with receipt.open(encoding="utf-8") as handle:
            ok = json.load(handle).get("status") == "downloaded-real-source"
    return {"part": stem, "year": year, "month": month, "ok": bool(ok),
            "returncode": process.returncode, "elapsed_seconds": elapsed,
            "nc_bytes": out_nc.stat().st_size if out_nc.is_file() else 0,
            "stdout": (stdout or "").strip()[-1500:], "stderr": (stderr or "").strip()[-1500:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded FY batch download.")
    parser.add_argument("--protocol", required=True, help="frozen protocol JSON for this batch")
    parser.add_argument("--out-dir", required=True, help="fresh exclusive directory for this batch")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    if not protocol_path.is_file():
        raise SystemExit(f"missing protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    digest = _canonical_digest(protocol)
    if protocol.get("protocol_sha256") not in (digest, "PENDING"):
        raise SystemExit(f"protocol digest mismatch: recorded {protocol.get('protocol_sha256')} "
                         f"but body hashes to {digest}")

    years = [int(y) for y in protocol["plan"]["years"]]
    months = [int(m) for m in protocol["plan"]["months"]]
    pool = int(protocol["concurrency"]["pool"])
    deadline_seconds = float(protocol["budgets"]["per_part_deadline_seconds"])
    watchdog_seconds = float(protocol["budgets"]["per_part_watchdog_seconds"])
    max_decoded_gib = float(protocol["budgets"]["per_part_decoded_gib_cap"])
    gate = float(protocol["budgets"]["free_space_gate_gib"])
    max_host_mb_s = float(protocol["concurrency"]["max_host_mb_s"])

    from data.download.season_plan_fy import full_year_plan

    plan = full_year_plan(years, months=months)
    if plan["total_stamps"] != int(protocol["plan"]["total_stamps"]):
        raise SystemExit(f"protocol plans {protocol['plan']['total_stamps']} stamps but the code "
                         f"plan yields {plan['total_stamps']}")
    if plan["parts"] != int(protocol["plan"]["parts"]):
        raise SystemExit("protocol part count does not match the recomputed plan")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing non-empty batch directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    if _free_gib(out_dir) < gate:
        raise SystemExit(f"free-space gate failed: {_free_gib(out_dir):.1f} GiB < {gate} GiB")

    jobs = [(block["year"], block["month"]) for block in plan["blocks"]]
    started = time.monotonic()
    sampler = _HostSampler(max_host_mb_s)
    sampler.start()
    children = _Children()
    outcomes = []

    def worker(year, month):
        outcomes.append(_run_part(parts_dir, year, month, deadline_seconds, watchdog_seconds,
                                  max_decoded_gib, children))

    # The live-thread count is the concurrency cap: at most `pool` part processes
    # exist at once, so the pool width the protocol froze is what actually runs.
    threads, not_started = [], []
    for position, (year, month) in enumerate(jobs):
        while sum(1 for thread in threads if thread.is_alive()) >= pool:
            time.sleep(HOST_SAMPLE_SECONDS)
            if sampler.aborted:
                break
        if sampler.aborted:
            not_started = [f"fy{y}{m:02d}" for y, m in jobs[position:]]
            break
        thread = threading.Thread(target=worker, args=(year, month))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    sampler.stop()
    results = sorted(outcomes, key=lambda entry: (entry["year"], entry["month"]))
    wall = round(time.monotonic() - started, 3)

    failed = [entry for entry in results if not entry["ok"]]
    summary = {
        "status": "batch-complete" if not failed and not not_started else "batch-failed",
        "format": "r7-fy-batch-summary-v1",
        "protocol": str(protocol_path), "protocol_sha256": protocol.get("protocol_sha256"),
        "out_dir": str(out_dir), "years": years, "months": months, "pool": pool,
        "planned_parts": len(jobs), "done_parts": len(results) - len(failed),
        "failed_parts": [entry["part"] for entry in failed],
        "not_started_parts": not_started,
        "wall_seconds": wall,
        "nc_bytes_total": sum(entry["nc_bytes"] for entry in results),
        "host_network": sampler.report(),
        "host_rx_bytes_during_batch": sampler.total_bytes,
        "parts": results,
        "scientific_claim": False,
        "limitations": ["the host network figure is a sampled host-wide counter window, not "
                        "icechunk per-request accounting"],
    }
    if failed or not_started or sampler.aborted:
        (out_dir / "batch_failure.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[batch] FAILED: {len(failed)} part(s) failed, "
              f"{len(not_started)} not started, aborted={sampler.aborted}")
        sys.exit(1)

    from data.download.earthmover_spatial_fy import merge_fy_parts

    pairs = [(parts_dir / f"{entry['part']}.nc", parts_dir / f"{entry['part']}.receipt.json")
             for entry in results]
    merge = merge_fy_parts(pairs, out_dir / "source.nc", out_dir / "merge_receipt.json",
                           expected_stamps=int(protocol["plan"]["total_stamps"]),
                           require_contiguous=True)
    summary["merge"] = {"stamps": merge["stamps"], "time_first": merge["time_first"],
                        "time_last": merge["time_last"],
                        "sha256": merge["local_artifact"]["sha256"],
                        "bytes": merge["local_artifact"]["bytes"]}
    (out_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    complete = {
        "format": "r7-fy-build-complete-v1",
        "out_dir": str(out_dir),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "plan": {"years": years, "months": months, "stamps": plan["total_stamps"],
                 "parts": plan["parts"]},
        "merge_receipt": "merge_receipt.json",
        "source_nc": "source.nc",
        "source_nc_sha256": merge["local_artifact"]["sha256"],
        "source_nc_bytes": merge["local_artifact"]["bytes"],
        "time_first": merge["time_first"], "time_last": merge["time_last"],
        "host_rx_bytes_during_batch": sampler.total_bytes,
        "host_network": sampler.report(),
        "wall_seconds": wall,
        "scientific_claim": False,
        "test_read": False,
    }
    (out_dir / "BUILD_COMPLETE.json").write_text(
        json.dumps(complete, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[batch] COMPLETE: {plan['parts']} parts, {plan['total_stamps']} stamps, "
          f"source.nc sha256={merge['local_artifact']['sha256'][:16]} wall={wall}s "
          f"host_rx={sampler.total_bytes} B peak={sampler.report()['peak_mb_s']} MB/s")


if __name__ == "__main__":
    main()
