"""Independent B/C attempt CLI; preparation is CPU-only and no retry is supported."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def deny_network():
    from scripts.r7_m3_offline import deny_network as shared_denial
    shared_denial()


def main(argv=None):
    # Capture the whole-round clock before importing any project/torch/data code.
    started = time.perf_counter()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    deny_network()
    from training.r7_v2_driver import prepare, run_bounded_round
    from training.r7_v2_protocol import DEFAULT_OUTPUT, local_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run", "all"), default="all")
    parser.add_argument("--stage", choices=("B", "C"), default="B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--configuration", type=Path, help="Explicit C configuration JSON frozen by the main chain after B")
    args = parser.parse_args(argv)
    if args.phase in ("prepare", "all"):
        if not args.gpu_uuid:
            parser.error("a fixed physical --gpu-uuid is required before preparation")
        if args.stage == "C" and args.configuration is None:
            parser.error("C requires --configuration; no method/primary/adaptive choice is guessed")
        configuration = None if args.configuration is None else json.loads(local_path(args.configuration).read_text(encoding="utf-8"))
        sidecar = args.sidecar
        if sidecar is None:
            original = json.loads((ROOT / "outputs/r7_73_process_supervision/protocol.json").read_text(encoding="utf-8"))
            sidecar = local_path(original["sidecar"]["path"])
        protocol = prepare(args.manifests, sidecar, args.output, args.gpu_uuid, stage=args.stage,
                           configuration=configuration, round_started_perf_counter=started, boot_id=boot)
        if args.phase == "prepare":
            print(json.dumps({"status": "prepared-not-run", "output": protocol["output"],
                              "protocol_sha256": protocol["protocol_sha256"], "scientific_claim": False}))
            return 0
    outcome = run_bounded_round(args.output)
    print(json.dumps(outcome, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
