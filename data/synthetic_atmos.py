from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .r7_store import HOUR_NS, init_time_fields, history_offsets_hours_from_ns

# 2016-01-01T00:00Z, the segment every R7 engineering store starts from. The
# synthetic fixture must not invent a *different* calendar convention from the
# real readers, so it uses the same derivation (``init_time_fields``) on a
# synthetic stamp.
SYNTHETIC_EPOCH_NS = int(np.datetime64('2016-01-01T00:00').astype('datetime64[ns]').astype(np.int64))


class SyntheticAtmosDataset(Dataset):
    """Deterministic forecast-native synthetic fixture for R7 smoke/CI only.

    The initialization-time fields are synthetic *inputs* (derived from the epoch
    above plus the sample index), not observations: this fixture exists so the
    space-time pathway can be exercised offline, and nothing in it is a claim
    about real weather.
    """

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
        # Whole-hour steps so the synthetic initialization time stays hour-aligned,
        # which is what the derivation requires of every real store.
        init_ns=SYNTHETIC_EPOCH_NS+int(round(idx*self.lead))*HOUR_NS
        return {
            'coarse_history':history.float(),
            'atmos_target':target.float(),
            'lead_time_hours':torch.tensor(self.lead,dtype=torch.float32),
            'history_offsets_hours':torch.from_numpy(history_offsets_hours_from_ns(
                np.asarray([init_ns - (self.history_steps - 1 - slot) * 6 * HOUR_NS
                            for slot in range(self.history_steps)], dtype=np.int64))),
            'latitude':latitude.float(),
            'longitude':longitude.float(),
            'grid_spacing_deg':torch.tensor(self.grid,dtype=torch.float32),
            'sample_id':f'synthetic_atmos_{idx:05d}',
            **{name:torch.tensor(float(value),dtype=torch.float32)
               for name,value in init_time_fields(init_ns).items()},
        }
