from __future__ import annotations
import argparse
import json
from training.r7_calibration_runner import run_calibration


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--train-manifest',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--updates',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--max-steps',type=int,default=4)
    ap.add_argument('--gain-threshold',type=float,default=0.)
    ap.add_argument('--probability-threshold',type=float,default=.5)
    ap.add_argument('--device',default='cpu')
    a=ap.parse_args()
    path,report=run_calibration(a.checkpoint,a.train_manifest,output_dir=a.out,updates=a.updates,
        batch_size=a.batch_size,max_steps=a.max_steps,gain_threshold=a.gain_threshold,
        probability_threshold=a.probability_threshold,device_name=a.device)
    print(json.dumps({'controller':str(path),'forecaster_unchanged':report['forecaster_unchanged'],'scientific_claim':False}))


if __name__=='__main__':
    main()
