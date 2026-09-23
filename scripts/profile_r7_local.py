"""Run isolated bounded synthetic shape profiles; not weather-skill tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--steps',type=int,nargs='+',default=[1,2,4,8])
    ap.add_argument('--updates',type=int,default=3)
    ap.add_argument('--bf16',action='store_true')
    args=ap.parse_args()
    import yaml
    config=yaml.safe_load(Path(args.config).read_text())
    if len(args.steps)!=len(set(args.steps)) or any(k<0 for k in args.steps):
        raise ValueError('steps must be distinct nonnegative integers')
    if config['kind']=='native':
        raise ValueError('native has no recursive K; use train_r7_local directly')
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for k in args.steps:
        case=dict(config,train=dict(config.get('train',{}),steps=k,process_weight=0.0))
        path=out/f'K{k}.yaml'
        path.write_text(yaml.safe_dump(case))
        command=[sys.executable,str(ROOT/'train_r7_local.py'),'--config',str(path),
            '--synthetic','--out',str(out/f'K{k}'),'--updates',str(args.updates),'--device',args.device]
        if args.bf16:
            command+=['--bf16']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (out/f'K{k}.log').write_text(result.stdout+'\n'+result.stderr)
        row={'K':k,'returncode':result.returncode,'data':'synthetic shape fixture'}
        if result.returncode==0:
            row['measurement']=json.loads((out/f'K{k}'/f'updates_to_{args.updates:07d}.json').read_text())
        results.append(row)
    (out/'profiles.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    if any(r['returncode']!=0 for r in results):
        raise SystemExit('one or more profiles failed; inspect logs, no fallback/result fabrication')


if __name__=='__main__':
    main()
