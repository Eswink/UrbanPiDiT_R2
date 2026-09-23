"""Explicit bounded CPU experiment on the verified local real-data artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from training.r7_cpu_study import run_study
    result = run_study(args.source, args.receipt, args.out)
    print(json.dumps({'finished':True, 'scientific_claim':False,
                      'protocol_sha256':result['protocol']['protocol_sha256'],
                      'elapsed_seconds':result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
