from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, yaml
from model import UrbanPiDiTR2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/r2_v6_4090d.yaml'); a=p.parse_args(); cfg=yaml.safe_load(open(a.config,encoding='utf-8')); m=UrbanPiDiTR2(**cfg['model']);
    total=sum(x.numel() for x in m.parameters()); train=sum(x.numel() for x in m.parameters() if x.requires_grad); print(f'total={total:,} ({total/1e6:.2f}M) trainable={train/1e6:.2f}M')
    for name,mod in [('coarse',m.coarse),('process_init',m.process_init),('reasoner',m.reasoner),('router',m.router),('urban',m.urban),('verifier',m.verifier)]: print(name, sum(p.numel() for p in mod.parameters())/1e6)
if __name__=='__main__': main()
