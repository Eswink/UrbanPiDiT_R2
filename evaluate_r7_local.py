from __future__ import annotations
import argparse
import json
from training.r7_evaluate import evaluate_local


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out',required=True)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--persistence',action='store_true')
    ap.add_argument('--leads',type=int,nargs='+',default=[6,12,24,48,72])
    ap.add_argument('--step-hours',type=int,default=6)
    ap.add_argument('--max-samples',type=int,default=32)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--normalized',action='store_true')
    ap.add_argument('--reasoning-steps',type=int)
    ap.add_argument('--controller')
    ap.add_argument('--min-steps',type=int,default=1)
    ap.add_argument('--force-full-depth',action='store_true')
    a=ap.parse_args()
    result=evaluate_local(a.manifest,output_dir=a.out,checkpoint=a.checkpoint,lead_hours=a.leads,
        step_hours=a.step_hours,max_samples=a.max_samples,device_name=a.device,
        normalized=a.normalized,reasoning_steps=a.reasoning_steps,controller_checkpoint=a.controller,
        min_reasoning_steps=a.min_steps,force_full_depth=a.force_full_depth)
    print(json.dumps({'n_evaluated':result['n_evaluated'],'scientific_claim':False}))


if __name__=='__main__':
    main()
