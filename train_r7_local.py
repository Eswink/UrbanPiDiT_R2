"""Explicitly capped local experiment. No downloading or automatic rentals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from training.r7_experiment import dataset_identity,canonical_digest
from training.r7_local_runner import run_local_updates


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--synthetic',action='store_true')
    ap.add_argument('--out',required=True)
    ap.add_argument('--updates',type=int,default=10,help='absolute optimizer-update endpoint, including resumed updates')
    ap.add_argument('--resume')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--bf16',action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.manifest:
        identity,ds=dataset_identity(args.manifest)
    else:
        from data.synthetic_atmos import SyntheticAtmosDataset
        options=cfg.get('synthetic',{'channels':cfg['model']['in_channels'],'hw':[16,24],'length':32})
        ds=SyntheticAtmosDataset(**options)
        identity='SYNTHETIC-NOT-WEATHER-TRUTH:'+canonical_digest(options)
    ckpt,report=run_local_updates(ds,kind=cfg['kind'],model_config=cfg['model'],
        data_identity=identity,output_dir=args.out,total_updates=args.updates,
        device_name=args.device,bf16=args.bf16,resume=args.resume,**cfg.get('train',{}))
    print(json.dumps({'checkpoint':str(ckpt),'updates':report['updates_this_run'],
        'scientific_claim':False,'seconds':report['elapsed_seconds']},ensure_ascii=False))


if __name__=='__main__':
    main()
