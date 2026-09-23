from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
try:
    import pytorch_lightning as pl
except Exception:
    pl=None
from .synthetic_atmos import SyntheticAtmosDataset
from .schema import validate_forecast_sample
from .r7_store import require_complete_manifest


class ManifestAtmosNPZDataset(Dataset):
    """R7 forecast-native completed manifest + NPZ adapter."""
    def __init__(self,manifest:str|Path):
        self.manifest=Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        require_complete_manifest(self.manifest)
        self.records=[json.loads(line) for line in self.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f'manifest 为空: {self.manifest}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self,idx):
        rec=self.records[idx]
        path=Path(rec['path'])
        if not path.is_absolute():
            path=(self.manifest.parent/path).resolve()
        with np.load(path,allow_pickle=False) as z:
            sample={k:torch.from_numpy(z[k]).float() for k in z.files}
        if any(not torch.isfinite(v).all() for v in sample.values()):
            raise ValueError('nonfinite NPZ sample')
        sample['sample_id']=rec.get('sample_id',path.stem)
        validate_forecast_sample(sample,batched=False)
        return sample


if pl is None:
    R7ForecastDataModule=object
else:
    class R7ForecastDataModule(pl.LightningDataModule):
        def __init__(self,data_cfg:Dict[str,Any],train_cfg:Dict[str,Any]):
            super().__init__()
            self.data_cfg=dict(data_cfg)
            self.train_cfg=dict(train_cfg)

        def setup(self,stage:Optional[str]=None):
            mode=str(self.data_cfg.get('mode','synthetic_atmos'))
            if mode=='synthetic_atmos':
                base=SyntheticAtmosDataset(
                    length=int(self.data_cfg.get('length',96)),
                    hw=tuple(self.data_cfg.get('hw',[16,24])),
                    history_steps=int(self.data_cfg.get('history_steps',2)),
                    channels=int(self.data_cfg.get('channels',8)),
                    lead_time_hours=float(self.data_cfg.get('lead_time_hours',6.0)),
                    grid_spacing_deg=float(self.data_cfg.get('grid_spacing_deg',0.25)),
                    seed=int(self.train_cfg.get('seed',42)))
                n=len(base)
                ntr=max(1,int(n*0.75))
                nv=max(1,int(n*0.125))
                self.train_ds,self.val_ds,self.test_ds=random_split(base,[ntr,nv,n-ntr-nv],
                    generator=torch.Generator().manual_seed(int(self.train_cfg.get('seed',42))))
            elif mode in ('manifest_atmos_npz','manifest_atmos_zarr'):
                if mode=='manifest_atmos_zarr':
                    from .r7_zarr_dataset import ZarrAtmosWindowDataset
                    dataset=ZarrAtmosWindowDataset
                else:
                    dataset=ManifestAtmosNPZDataset
                self.train_ds=dataset(self.data_cfg['train_manifest'])
                self.val_ds=dataset(self.data_cfg['val_manifest'])
                self.test_ds=dataset(self.data_cfg['test_manifest'])
            else:
                raise ValueError(f'未知 R7 data.mode: {mode}')

        def _loader(self,ds,shuffle=False):
            workers=int(self.train_cfg.get('num_workers',0))
            return DataLoader(ds,batch_size=int(self.train_cfg.get('batch_size',1)),shuffle=shuffle,
                num_workers=workers,pin_memory=bool(self.train_cfg.get('pin_memory',True)),persistent_workers=workers>0)

        def train_dataloader(self):
            return self._loader(self.train_ds,True)

        def val_dataloader(self):
            return self._loader(self.val_ds,False)

        def test_dataloader(self):
            return self._loader(self.test_ds,False)
