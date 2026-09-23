from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REQUIRED=['coarse_history','urban_history','urban_static','urban_baseline','urban_target']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input_dir'); ap.add_argument('output'); args=ap.parse_args()
    root=Path(args.input_dir); rows=[]; bad=[]
    for p in sorted(root.glob('*.npz')):
        try:
            with np.load(p,allow_pickle=False) as z:
                miss=[k for k in REQUIRED if k not in z]
                if miss: raise ValueError(f'缺少 {miss}')
            rows.append({'path':str(p.resolve()),'sample_id':p.stem})
        except Exception as e: bad.append((str(p),str(e)))
    Path(args.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
    print(f'有效样本={len(rows)} 无效样本={len(bad)}')
    for p,e in bad[:20]: print('[坏样本]',p,e)
if __name__=='__main__': main()
