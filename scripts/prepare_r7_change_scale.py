"""Local CPU change-scale (d_c / s_c) preflight; writing needs the reviewed identity.

The published module owns the logic; this CLI only carries the explicit read/write
gate, mirroring prepare_r7_process_scale.py and prepare_r7_typed_evidence.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocess.r7_change_scale import (
    change_scale_preflight, publish_change_scale_sidecar,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--train-manifest', required=True, type=Path)
    parser.add_argument('--sidecar-dir', '--out', dest='sidecar_dir', type=Path,
                        help='new disjoint sidecar directory (required with --write)')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--preflight-identity',
                        help='exact SHA256 from the reviewed read-only report')
    args = parser.parse_args(argv)
    if args.write:
        if args.sidecar_dir is None or args.preflight_identity is None:
            parser.error('--write requires --sidecar-dir and --preflight-identity')
        meta = publish_change_scale_sidecar(
            args.store, args.train_manifest, args.sidecar_dir, args.preflight_identity)
        report = {'mode': 'published-change-scale-sidecar', 'scientific_claim': False,
                  'sidecar_dir': str(args.sidecar_dir.resolve()), 'metadata': meta,
                  'change_scale_identity': meta['change_scale_identity']}
    else:
        if args.preflight_identity is not None:
            parser.error('--preflight-identity is only used with --write')
        report = change_scale_preflight(args.store, args.train_manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
