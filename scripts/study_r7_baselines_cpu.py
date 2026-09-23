"""Explicit bounded compact-baseline CPU study from the pinned local artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True);ap.add_argument('--receipt',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    from training.r7_baseline_study import run_baseline_study
    r=run_baseline_study(args.source,args.receipt,args.out)
    print(json.dumps(dict(finished=True,scientific_claim=False,elapsed_seconds=r['elapsed_seconds'])))


if __name__=='__main__':main()
