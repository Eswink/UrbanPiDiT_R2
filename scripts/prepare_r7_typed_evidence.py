"""Local CPU typed-evidence preflight; writing requires the reviewed identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocess.r7_typed_evidence_scale import (
    publish_typed_evidence_scale, typed_evidence_preflight,
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
        meta = publish_typed_evidence_scale(
            args.store, args.train_manifest, args.sidecar_dir, args.preflight_identity)
        report = {'mode': 'published-typed-evidence-sidecar', 'scientific_claim': False,
                  'sidecar_dir': str(args.sidecar_dir.resolve()), 'metadata': meta,
                  'typed_evidence_identity': meta['typed_evidence_identity']}
    else:
        if args.preflight_identity is not None:
            parser.error('--preflight-identity is only used with --write')
        report = typed_evidence_preflight(args.store, args.train_manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
