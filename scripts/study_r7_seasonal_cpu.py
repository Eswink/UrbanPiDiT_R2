"""Run the predeclared #50 study from the verified local #49 artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--receipt',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    from training.r7_seasonal_study import run_seasonal_study
    result=run_seasonal_study(args.source,args.receipt,args.out)
    print(json.dumps({'finished':True,'scientific_claim':False,'elapsed_seconds':result['elapsed_seconds'],
        'selected_policies':[{str(c['seed']):c['selection']['selected_policy']} for c in result['controllers']]}))


if __name__=='__main__': main()
