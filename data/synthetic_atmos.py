from __future__ import annotations
from typing import Dict, Any
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class SyntheticAtmosDataset(Dataset):
    """Deterministic forecast-native synthetic fixture for R7 smoke/CI only."""

    def __init__(
        self,
        length:int=64,
        hw:tuple[int,int]=(16,24),
        history_steps:int=2,
        channels:int=8,
        lead_time_hours:float=6.0,
        grid_spacing_deg:float=0.25,
        seed:int=1234,
    ):
        self.length=int(length)
        self.hw=tuple(hw)
        self.history_steps=int(history_steps)
        self.channels=int(channels)
        self.lead=float(lead_time_hours)
        self.grid=float(grid_spacing_deg)
        self.seed=int(seed)
        if self.history_steps<2:
            raise ValueError('SyntheticAtmosDataset 需要至少 2 个历史时次')

    def __len__(self):
        return self.length

    def __getitem__(self,idx:int)->Dict[str,Any]:
        g=torch.Generator().manual_seed(self.seed+idx)
        H,W=self.hw
        history=torch.randn(
            self.history_steps,self.channels,H,W,generator=g
        )
        previous=history[-2]
        current=history[-1]
        smooth=F.avg_pool2d(
            current.unsqueeze(0),kernel_size=3,stride=1,padding=1
        ).squeeze(0)
        temporal=0.18*(current-previous)
        spatial=0.06*(smooth-current)
        tendency=temporal+spatial
        target=current+tendency+0.005*torch.randn(
            current.shape,generator=g
        )

        latitude=torch.linspace(60.0,25.0,H)
        longitude=torch.arange(W,dtype=torch.float32)*self.grid+100.0
        return {
            'coarse_history':history.float(),
            'atmos_target':target.float(),
            'lead_time_hours':torch.tensor(self.lead,dtype=torch.float32),
            'latitude':latitude.float(),
            'longitude':longitude.float(),
            'grid_spacing_deg':torch.tensor(self.grid,dtype=torch.float32),
            'sample_id':f'synthetic_atmos_{idx:05d}',
        }
