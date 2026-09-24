from __future__ import annotations
import sys,argparse,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model import UrbanPiDiTR2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/r2_v6_4090d.yaml')
    a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding='utf-8'))
    m=UrbanPiDiTR2(**cfg['model']); n=sum(p.numel() for p in m.parameters());
    print(f'参数量: {n/1e6:.2f}M')
    print(f'BF16 参数本体: {n*2/1024**2:.1f} MiB')
    print(f'FP32 AdamW 两个状态: {n*8/1024**2:.1f} MiB')
    up=cfg['model'].get('urban_patch',4); cp=cfg['model'].get('coarse_patch',2); ws=cfg['model'].get('window_size',8)
    print(f'Urban 128x128 -> {(128//up)}x{(128//up)}={((128//up)**2)} tokens')
    print(f'Window attention: {ws}x{ws}={ws*ws} tokens/window；不构造全局 token² attention map')
    print('注意：真实峰值显存依赖 PyTorch/CUDA kernel、batch、checkpoint 与数据通道；请用 benchmark_gpu.py 实测。')
if __name__=='__main__': main()
