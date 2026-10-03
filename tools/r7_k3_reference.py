"""Standalone original #72 K3 reference runner. Prepare is bytes-only; all adds exactly10 CUDA evaluations."""
from __future__ import annotations

import time
_ENTRY_STARTED = time.perf_counter()  # Earliest CPU clock, before stdlib/companion/project imports.
import socket


def _deny_network():
    def refused(*args, **kwargs):
        raise RuntimeError("K3 reference is offline: outbound connections forbidden")
    socket.socket.connect = refused
    socket.socket.connect_ex = refused
    socket.create_connection = refused


_deny_network()  # stdlib offline bootstrap before torch/project/companion imports.
import argparse
import math
import os
from pathlib import Path
import sys


def main(argv=None):
    _deny_network()
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "all", "archive_worker"), required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seed", type=int, choices=(41, 42))
    parser.add_argument("--lead", type=int, choices=(6, 12, 24, 48, 72))
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if args.mode == "archive_worker":
        if args.protocol is None or args.seed is None or args.lead is None or args.deadline is None or not math.isfinite(args.deadline):
            parser.error("archive_worker requires exact protocol/seed/single lead and finite driver hard deadline")
        if args.repo is not None or args.output is not None:
            parser.error("archive_worker cannot override frozen source/output")
        from r7_k3_reference_worker import run_worker
        run_worker(args.protocol, {"seed": args.seed, "lead": args.lead}, args.deadline)
    else:
        if args.repo is None or args.output is None or any(v is not None for v in (args.protocol, args.seed, args.lead, args.deadline)):
            parser.error("prepare/all require repo and a fresh output; worker-only arguments are forbidden")
        if args.mode == "prepare":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from r7_k3_reference_driver import prepare, run_all
        protocol = prepare(args.repo, args.output, _ENTRY_STARTED)
        if args.mode == "all":
            run_all(protocol, _ENTRY_STARTED)
        else:
            print("prepared-not-run; zero weather samples/checkpoint loads/CUDA jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
