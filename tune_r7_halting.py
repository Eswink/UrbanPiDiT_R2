"""Explicit validation-only halting threshold search; never a background job."""
from __future__ import annotations
import argparse
import json
from training.r7_policy_selection import run_policy_search


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--controller',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--gain-thresholds',type=float,nargs='+',default=[0.])
    ap.add_argument('--probability-thresholds',type=float,nargs='+',default=[.3,.5,.7])
    ap.add_argument('--leads',type=int,nargs='+',default=[6,12,24,48,72])
    ap.add_argument('--max-samples',type=int,default=32)
    ap.add_argument('--relative-rmse-tolerance',type=float,default=.01)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--normalized',action='store_true')
    a=ap.parse_args()
    result=run_policy_search(a.manifest,checkpoint=a.checkpoint,controller_checkpoint=a.controller,
        output_dir=a.out,gain_thresholds=a.gain_thresholds,probability_thresholds=a.probability_thresholds,
        lead_hours=a.leads,max_samples=a.max_samples,relative_rmse_tolerance=a.relative_rmse_tolerance,
        device_name=a.device,normalized=a.normalized)
    print(json.dumps({'selected_policy':result['selected_policy'],'scientific_claim':False}))


if __name__=='__main__':
    main()
