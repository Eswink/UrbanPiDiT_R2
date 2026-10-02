"""Local CPU process-scale preflight; writing requires the reviewed identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocess.r7_process_scale_sidecar import (
    publish_process_scale_sidecar, sidecar_preflight,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--train-manifest', required=True, type=Path)
    parser.add_argument('--manifest-dir', '--out', dest='manifest_dir', type=Path,
                        help='new disjoint sidecar directory (required with --write)')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--preflight-identity', help='exact SHA256 from the reviewed read-only report')
    args = parser.parse_args(argv)
    if args.write:
        if args.manifest_dir is None or args.preflight_identity is None:
            parser.error('--write requires --manifest-dir and --preflight-identity')
        meta = publish_process_scale_sidecar(
            args.store, args.train_manifest, args.manifest_dir, args.preflight_identity)
        report = {'mode': 'published-process-scale-sidecar', 'scientific_claim': False,
                  'manifest_dir': str(args.manifest_dir.resolve()), 'metadata': meta,
                  'sidecar_identity': meta['sidecar_identity']}
    else:
        if args.preflight_identity is not None:
            parser.error('--preflight-identity is only used with --write')
        report = sidecar_preflight(args.store, args.train_manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
