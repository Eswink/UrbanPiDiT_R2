from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import validate_forecast_sample


class ZarrAtmosWindowDataset(Dataset):
    """On-demand R7 atmospheric windows from a physical-unit Zarr archive."""

    def __init__(self,manifest:str|Path):
        self.manifest=Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.records=[
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"manifest 为空: {self.manifest}")
        self._stores={}

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state=dict(self.__dict__)
        # Zarr handles are reopened independently in DataLoader workers.
        state["_stores"]={}
        return state

    def _store(self,record):
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError(
                "ZarrAtmosWindowDataset requires optional dependency zarr; "
                "install requirements-r7-data.txt"
            ) from exc

        path=Path(record["store_path"])
        if not path.is_absolute():
            path=(self.manifest.parent/path).resolve()
        key=str(path)
        if key not in self._stores:
            self._stores[key]=zarr.open_group(key,mode="r")
        return self._stores[key]

    def __getitem__(self,idx):
        rec=self.records[idx]
        root=self._store(rec)
        state=root["state"]
        history=np.stack(
            [
                np.asarray(state[int(i)],dtype=np.float32)
                for i in rec["history_indices"]
            ],
            axis=0,
        )
        target=np.asarray(
            state[int(rec["target_index"])],dtype=np.float32
        )
        mean=np.asarray(
            root["normalization_mean"][:],dtype=np.float32
        )[:,None,None]
        std=np.asarray(
            root["normalization_std"][:],dtype=np.float32
        )[:,None,None]
        history=(history-mean[None,...])/std[None,...]
        target=(target-mean)/std

        sample={
            "coarse_history":torch.from_numpy(history).float(),
            "atmos_target":torch.from_numpy(target).float(),
            "lead_time_hours":torch.tensor(
                float(rec["lead_time_hours"]),dtype=torch.float32
            ),
            "latitude":torch.from_numpy(
                np.asarray(root["latitude"][:],dtype=np.float32)
            ),
            "longitude":torch.from_numpy(
                np.asarray(root["longitude"][:],dtype=np.float32)
            ),
            "grid_spacing_deg":torch.tensor(
                float(root.attrs["native_grid_spacing_deg"]),
                dtype=torch.float32,
            ),
            "sample_id":rec["sample_id"],
        }
        validate_forecast_sample(sample,batched=False)
        return sample
