from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .schema import validate_forecast_sample
from .r7_store import (init_time_fields, require_complete_manifest, validate_store,
    validate_record, normalization, history_offsets_hours_from_ns)


class ZarrAtmosWindowDataset(Dataset):
    """Read-only windows from a completed version-1 physical-unit R7 archive."""
    def __init__(self, manifest: str|Path):
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        require_complete_manifest(self.manifest)
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.records:
            raise ValueError('empty R7 manifest')
        ids = [r['sample_id'] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate sample ids')
        self._stores = {}

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        result = dict(self.__dict__)
        result['_stores'] = {}
        return result

    def _store(self, record):
        import zarr
        path = Path(record['store_path'])
        if not path.is_absolute():
            path = (self.manifest.parent/path).resolve()
        key = str(path)
        if key not in self._stores:
            root = zarr.open_group(key,mode='r')
            validate_store(root)
            self._stores[key] = root
        return self._stores[key]

    def __getitem__(self, idx):
        record = self.records[idx]
        root = self._store(record)
        indices = validate_record(root,record)
        state = root['state']
        frames = np.stack([np.asarray(state[i],dtype=np.float32) for i in indices])
        mean,std = normalization(root,count=state.shape[1])
        frames = (frames-mean[None,:,None,None])/std[None,:,None,None]
        if not np.isfinite(frames).all():
            raise ValueError('nonfinite atmospheric sample')
        sample = {
            'coarse_history':torch.from_numpy(frames[:-1]), 'atmos_target':torch.from_numpy(frames[-1]),
            'lead_time_hours':torch.tensor(float(record['lead_time_hours'])),
            'latitude':torch.from_numpy(np.asarray(root['latitude'][:],dtype=np.float32)),
            'longitude':torch.from_numpy(np.asarray(root['longitude'][:],dtype=np.float32)),
            'grid_spacing_deg':torch.tensor(float(root.attrs['native_grid_spacing_deg'])),
            'sample_id':record['sample_id'],
        }
        # validate_record pins init_time == the last history timestamp, so the
        # initialization time is the store's own entry for `indices[-2]`.
        sample.update({name:torch.tensor(float(value))
            for name,value in init_time_fields(int(root['time_ns'][indices[-2]])).items()})
        history_ns = np.asarray([root['time_ns'][i] for i in indices[:-1]], dtype=np.int64)
        sample['history_offsets_hours'] = torch.from_numpy(history_offsets_hours_from_ns(history_ns))
        if 'process_diagnostics_raw' in root:
            raw = np.asarray(root['process_diagnostics_raw'][indices[-2]],dtype=np.float32)
            pmean,pstd = normalization(root,'process_normalization',len(raw))
            process = (raw-pmean)/pstd
            if not np.isfinite(process).all():
                raise ValueError('nonfinite input-time process targets')
            sample['process_targets'] = torch.from_numpy(process)
        validate_forecast_sample(sample,batched=False)
        return sample
