"""Sweep batch sizes from large to small, one fresh process each.

Each probe runs in its own interpreter (see ``r7_s3_batch_ceiling_worker.py`` for
why), so a failed allocation cannot inflate the next measurement. The sweep stops
at the first size that fits and then continues downward only for the sizes the
caller asked for, so the answer is "the largest batch that actually ran", not the
largest batch the caller hoped for.

This is a measurement instrument: ``scientific_claim: false``, no test split, no
frozen-criteria change. It writes one JSON per size plus a summary.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r7_s3_batch_ceiling_worker.py"


def run_one(batch, out_dir, updates, steps, device, timeout, workers=0):
    target = out_dir / f"batch_{batch:03d}.json"
    argv = [sys.executable, str(WORKER), "--batch", str(batch), "--updates", str(updates),
            "--steps", str(steps), "--device", device, "--out", str(target),
            "--workers", str(workers)]
    try:
        done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
        exit_code = done.returncode
    except subprocess.TimeoutExpired:
        tail, exit_code = [f"timeout after {timeout} s"], -9
    record = {"batch_size": batch, "exit_code": exit_code, "tail": tail}
    if target.is_file():
        record.update(json.loads(target.read_text(encoding="utf-8")))
    else:
        record["ok"] = False
        record["error"] = "worker wrote no record"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)

    records = []
    for batch in sorted(set(args.batches), reverse=True):
        print(f"[sweep] batch {batch} ...", flush=True)
        record = run_one(batch, args.out, args.updates, args.steps, args.device, args.timeout,
                         args.workers)
        records.append(record)
        print(f"[sweep] batch {batch}: ok={record.get('ok')} "
              f"peak={record.get('peak_reserved_mib')} MiB "
              f"s/upd={record.get('seconds_per_update_mean')} "
              f"oom={record.get('out_of_memory')}", flush=True)

    ran = [item["batch_size"] for item in records if item.get("ok")]
    oom = [item["batch_size"] for item in records if item.get("out_of_memory")]
    other = [item["batch_size"] for item in records
             if not item.get("ok") and not item.get("out_of_memory")]
    summary = {
        "scientific_claim": False, "test_read": False,
        "probe": "batch ceiling sweep; largest batch that actually ran",
        "largest_that_ran": max(ran) if ran else None,
        "ran": sorted(ran), "out_of_memory": sorted(oom), "other_failures": sorted(other),
        "note": ("each size is a fresh interpreter so peak reserved is that size's own "
                 "high-water mark; data construction is inside the timed region"),
        "measurements": records,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("largest_that_ran", "ran", "out_of_memory", "other_failures")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
