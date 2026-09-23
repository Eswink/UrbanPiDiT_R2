"""Explicit checkpoint inference profile; no rentals/downloads/training."""
from __future__ import annotations
import argparse
import json
from training.r7_inference_profile import profile_local


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--controller')
    ap.add_argument('--policy-selection')
    ap.add_argument('--out',required=True)
    ap.add_argument('--force-full-depth',action='store_true')
    ap.add_argument('--reasoning-steps',type=int)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--repetitions',type=int,default=10)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    a=ap.parse_args()
    report=profile_local(a.manifest,checkpoint=a.checkpoint,output=a.out,controller_checkpoint=a.controller,
        policy_selection=a.policy_selection,force_full_depth=a.force_full_depth,reasoning_steps=a.reasoning_steps,
        batch_size=a.batch_size,warmup=a.warmup,repetitions=a.repetitions,device_name=a.device,precision=a.precision)
    print(json.dumps({'device':report['device'],'median_seconds_per_batch':report['median_seconds_per_batch'],
        'scientific_claim':False}))


if __name__=='__main__':
    main()
