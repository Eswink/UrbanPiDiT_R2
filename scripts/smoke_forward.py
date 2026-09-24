from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, yaml, torch
from torch.utils.data import DataLoader
from data.synthetic import SyntheticR2Dataset
from model import UrbanPiDiTR2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/r2_v6_smoke.yaml'); a=ap.parse_args(); cfg=yaml.safe_load(open(a.config,encoding='utf-8'))
    d=cfg['data']
    ds=SyntheticR2Dataset(length=4,coarse_hw=tuple(d['coarse_hw']),urban_hw=tuple(d['urban_hw']),
        coarse_steps=d['coarse_steps'],urban_steps=d['urban_steps'],coarse_channels=d['coarse_channels'],
        urban_channels=d['urban_channels'],static_channels=d['static_channels'],
        out_channels=d['out_channels'],anchored_processes=d['anchored_processes'])
    batch=next(iter(DataLoader(ds,batch_size=2)))
    m=UrbanPiDiTR2(**cfg['model']); m.eval()
    with torch.no_grad(): out=m(batch,force_zoom=True,adaptive_reasoning=True)
    n=sum(p.numel() for p in m.parameters())
    print(f'参数量: {n/1e6:.2f}M')
    print('forecast',tuple(out.forecast.shape))
    print('zoom_prob',out.diagnostics.zoom_probability.tolist())
    print('reasoning_steps',out.diagnostics.reasoning_steps)
    print('verifier',out.diagnostics.verifier_score.tolist())
if __name__=='__main__': main()
