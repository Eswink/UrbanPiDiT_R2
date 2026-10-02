"""Prepare or execute the independent zero-training M3 validation complement offline."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time


def deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("M3 complement is offline: outbound connections forbidden")
    socket.socket.connect = refused
    socket.create_connection = refused


def main(argv=None):
    # Capture run entry before any project/torch/data imports or CPU pin checks.
    started = time.perf_counter()
    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "prepare" and args.original_output is None:
        parser.error("CPU preparation requires --original-output")
    if args.mode == "run" and args.original_output is not None:
        parser.error("run consumes the original output already frozen in the new protocol")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from training.r7_m3_complement_driver import prepare, run_bounded_round
    if args.mode == "prepare":
        result = prepare(args.original_output, args.output, started_perf_counter=started)
        print("prepared-not-run " + result["protocol_sha256"], flush=True)
    else:
        outcome = run_bounded_round(args.output, started_perf_counter=started)
        print("paused" if outcome["paused"] else "descriptive-complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
