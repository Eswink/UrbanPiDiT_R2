"""M3 bounded three-arm round: CPU prepare first, then one authorized GPU attempt.

Prepare does not run updates or select CUDA. Run never resumes/retries or rewrites
outputs. Authorizations use training.r7_m3_protocol.AUTHORIZATION_SCOPE and exact
bounds; failures/partial/budget_limited and unresolved comparisons stop the round.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_m3_driver import DEFAULT_ESTIMATED_PEAK_MIB, prepare, run_bounded_round


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--manifests", type=Path,
                        default=ROOT / "outputs/r7_m2_segment/store/manifests")
    parser.add_argument("--source-root", type=Path,
                        help="optional store parent; must agree with manifests/cache.zarr")
    parser.add_argument("--sidecar", type=Path, default=ROOT / "outputs/r7_m3_scale_sidecar")
    parser.add_argument("--output", "--out", dest="output", type=Path,
                        default=ROOT / "outputs/r7_73_process_supervision")
    parser.add_argument("--gpu-uuid", help="physical UUID frozen in prepare, never a mutable index")
    parser.add_argument("--estimated-peak-mib", type=int, default=DEFAULT_ESTIMATED_PEAK_MIB)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    if args.source_root is not None and args.source_root.resolve() != args.manifests.resolve().parent:
        parser.error("--source-root must match the existing store parent of --manifests")
    if args.mode == "prepare":
        if not args.gpu_uuid:
            parser.error("--mode prepare requires the one physical --gpu-uuid to freeze")
        protocol = prepare(args.manifests, args.sidecar, args.output, args.authorization,
                           args.gpu_uuid, args.estimated_peak_mib)
        print(json.dumps({"status": "prepared-not-run", "scientific_claim": False,
                          "protocol_sha256": protocol["protocol_sha256"],
                          "output": protocol["output"], "gpu_hours_charged": 0.0}))
    else:
        if args.gpu_uuid is not None:
            parser.error("run reads the UUID only from the frozen protocol; no override")
        outcome = run_bounded_round(args.output, args.authorization)
        print(json.dumps(outcome, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
