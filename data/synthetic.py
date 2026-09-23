from __future__ import annotations
import math
from typing import Dict, Any
import torch
from torch.utils.data import Dataset


class SyntheticR2Dataset(Dataset):
    """可重复的多尺度合成数据，用于端到端 smoke/CI，不代表真实气象性能。"""
    def __init__(
        self, length: int = 64, coarse_hw=(16, 16), urban_hw=(64, 64),
        coarse_steps: int = 2, urban_steps: int = 2,
        coarse_channels: int = 12, urban_channels: int = 7,
        static_channels: int = 6, out_channels: int = 7,
        anchored_processes: int = 12, seed: int = 42,
    ):
        self.length = int(length); self.coarse_hw = tuple(coarse_hw); self.urban_hw = tuple(urban_hw)
        self.tc=coarse_steps; self.tu=urban_steps; self.cc=coarse_channels; self.cu=urban_channels
        self.cs=static_channels; self.co=out_channels; self.p=anchored_processes; self.seed=int(seed)

    def __len__(self): return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(self.seed + idx)
        Hc,Wc=self.coarse_hw; Hu,Wu=self.urban_hw
        coarse = torch.randn(self.tc,self.cc,Hc,Wc,generator=g)
        urban = torch.randn(self.tu,self.cu,Hu,Wu,generator=g)
        static = torch.rand(self.cs,Hu,Wu,generator=g)
        baseline = torch.nn.functional.interpolate(
            coarse[-1,:self.co].unsqueeze(0), size=(Hu,Wu), mode='bilinear', align_corners=False
        ).squeeze(0)
        # 构造有物理直觉的局地 residual：静态形态 + 最近城市状态 + 弱噪声。
        morph = static.mean(0,keepdim=True)
        residual = 0.15 * torch.tanh(urban[-1,:self.co]) + 0.10 * (morph - 0.5)
        target = baseline + residual + 0.01*torch.randn(baseline.shape,generator=g)
        process = torch.zeros(self.p)
        process[0] = coarse[-1,0].mean()                   # pressure-gradient proxy
        process[1] = urban[-1,0].mean()                   # advection-like proxy
        process[2] = coarse[-1,1].mean()                  # moisture transport proxy
        process[3] = urban[-1,5:7].abs().mean() if self.cu >= 7 else urban[-1].abs().mean()
        process[4] = coarse[-1,2].std()
        process[5] = coarse[-1].std()
        process[6] = urban[-1].std()
        process[7] = static.mean()
        process[8] = static[0].mean()
        process[9] = static[min(1,self.cs-1)].mean()
        process[10] = static[min(2,self.cs-1)].mean()
        process[11] = static.std()
        # 只有当高分辨率 residual 足够大时，ZOOM 才有价值。
        zoom_target = (residual.abs().mean() > 0.09).float()
        return {
            'coarse_history': coarse.float(), 'urban_history': urban.float(), 'urban_static': static.float(),
            'urban_baseline': baseline.float(), 'urban_target': target.float(),
            'process_targets': process.float(), 'zoom_target': zoom_target,
            'sample_id': f'synthetic_{idx:05d}',
        }
