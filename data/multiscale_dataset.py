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
    pl = None
from .synthetic import SyntheticR2Dataset
from .schema import validate_sample


class ManifestNPZDataset(Dataset):
    """基于 JSONL manifest 的轻量正式数据接口。

    每行最少包含 `path`；NPZ 内键遵循 R2 data contract。
    大规模生产数据可用同一 contract 替换为 Zarr backend。
    """
    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        if not self.manifest.exists(): raise FileNotFoundError(self.manifest)
        self.records=[]
        for line in self.manifest.read_text(encoding='utf-8').splitlines():
            if line.strip(): self.records.append(json.loads(line))
        if not self.records: raise ValueError(f"manifest 为空: {self.manifest}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec=self.records[idx]; p=Path(rec['path'])
        if not p.is_absolute(): p=(self.manifest.parent/p).resolve()
        with np.load(p, allow_pickle=False) as z:
            sample={k: torch.from_numpy(z[k]).float() for k in z.files}
        sample['sample_id']=rec.get('sample_id', p.stem)
        validate_sample(sample, batched=False)
        return sample


if pl is None:
    R2DataModule = object
else:
    class R2DataModule(pl.LightningDataModule):
        def __init__(self, data_cfg: Dict[str, Any], train_cfg: Dict[str, Any]):
            super().__init__(); self.data_cfg=dict(data_cfg); self.train_cfg=dict(train_cfg)
        def setup(self, stage: Optional[str]=None):
            mode=str(self.data_cfg.get('mode','synthetic'))
            if mode=='synthetic':
                base=SyntheticR2Dataset(
                    length=int(self.data_cfg.get('length',96)),
                    coarse_hw=tuple(self.data_cfg.get('coarse_hw',[16,16])), urban_hw=tuple(self.data_cfg.get('urban_hw',[64,64])),
                    coarse_steps=int(self.data_cfg.get('coarse_steps',2)), urban_steps=int(self.data_cfg.get('urban_steps',2)),
                    coarse_channels=int(self.data_cfg.get('coarse_channels',12)), urban_channels=int(self.data_cfg.get('urban_channels',7)),
                    static_channels=int(self.data_cfg.get('static_channels',6)), out_channels=int(self.data_cfg.get('out_channels',7)),
                    anchored_processes=int(self.data_cfg.get('anchored_processes',12)), seed=int(self.train_cfg.get('seed',42)))
                n=len(base); ntr=max(1,int(n*0.75)); nv=max(1,int(n*0.125)); nt=n-ntr-nv
                self.train_ds,self.val_ds,self.test_ds=random_split(base,[ntr,nv,nt],generator=torch.Generator().manual_seed(int(self.train_cfg.get('seed',42))))
            elif mode=='manifest_npz':
                self.train_ds=ManifestNPZDataset(self.data_cfg['train_manifest'])
                self.val_ds=ManifestNPZDataset(self.data_cfg['val_manifest'])
                self.test_ds=ManifestNPZDataset(self.data_cfg['test_manifest'])
            else: raise ValueError(f"未知 data.mode: {mode}")
        def _loader(self, ds, shuffle=False):
            return DataLoader(ds,batch_size=int(self.train_cfg.get('batch_size',1)),shuffle=shuffle,
                num_workers=int(self.train_cfg.get('num_workers',0)),
                pin_memory=bool(self.train_cfg.get('pin_memory',True)),
                persistent_workers=bool(self.train_cfg.get('num_workers',0)>0))
        def train_dataloader(self): return self._loader(self.train_ds,True)
        def val_dataloader(self): return self._loader(self.val_ds,False)
        def test_dataloader(self): return self._loader(self.test_ds,False)
