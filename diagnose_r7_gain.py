"""Offline validation oracle diagnostic; never used for deployed forecasts."""
from __future__ import annotations
import argparse
import json
from training.r7_gain_oracle import run_oracle_diagnostic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--max-steps',type=int,default=4)
    ap.add_argument('--max-samples',type=int,default=32)
    ap.add_argument('--step-cost',type=float,default=0.)
    ap.add_argument('--device',default='cpu')
    a=ap.parse_args()
    result=run_oracle_diagnostic(a.manifest,checkpoint=a.checkpoint,output=a.out,max_steps=a.max_steps,
        max_samples=a.max_samples,step_cost=a.step_cost,device_name=a.device)
    print(json.dumps({'n_cases':result['n_cases'],'deployable':False,'scientific_claim':False}))


if __name__=='__main__':
    main()
