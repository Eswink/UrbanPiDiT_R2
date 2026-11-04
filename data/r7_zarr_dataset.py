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
    """Read-only windows from a completed version-1 physical-unit R7 archive.

    Store-constant metadata (the timestamp column and the normalization pairs) is read
    once per store: every read of a zarr array is a full round trip and the window
    builders used to pay it per frame. ``span`` and ``sample_from_frames`` let a
    multi-frame caller read its whole contiguous window in one go; they reuse this
    class's own discovery and validation, so the set of admissible windows is unchanged.
    """
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
        self._aux = {}

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        result = dict(self.__dict__)
        result['_stores'] = {}
        result['_aux'] = {}
        return result

    def _store_key(self, record):
        path = Path(record['store_path'])
        if not path.is_absolute():
            path = (self.manifest.parent/path).resolve()
        return str(path)

    def _store(self, record):
        import zarr
        key = self._store_key(record)
        if key not in self._stores:
            root = zarr.open_group(key,mode='r')
            stamps = validate_store(root)
            self._stores[key] = root
            # validate_store already read the whole timestamp column; keep it rather than
            # re-reading scalars, and read the normalization pair once instead of per sample.
            aux = {'time_ns': np.asarray(stamps,dtype=np.int64),
                   'normalization': normalization(root,count=root['state'].shape[1])}
            if 'process_diagnostics_raw' in root:
                aux['process_normalization'] = normalization(
                    root,'process_normalization',root['process_diagnostics_raw'].shape[1])
            self._aux[key] = aux
        return self._stores[key]

    def aux(self, record):
        """Store-constant metadata for ``record``'s store (timestamps, normalization).

        Filled when the store is first opened, and computed here if a caller supplied the
        store another way (the metadata preflight tests do exactly that).
        """
        root = self._store(record)
        key = self._store_key(record)
        if key not in self._aux:
            stamps = np.asarray(root['time_ns'][:], dtype=np.int64)
            aux = {'time_ns': stamps,
                   'normalization': normalization(root,count=root['state'].shape[1])}
            if 'process_diagnostics_raw' in root:
                aux['process_normalization'] = normalization(
                    root,'process_normalization',root['process_diagnostics_raw'].shape[1])
            self._aux[key] = aux
        return self._aux[key]

    def indices(self, record):
        return validate_record(self._store(record),record,time_ns=self.aux(record)['time_ns'])

    def span(self, record):
        """The contiguous store indices this record needs, or None when not contiguous."""
        indices = self.indices(record)
        return indices if indices == list(range(indices[0],indices[-1]+1)) else None

    def read_normalized(self, record, span):
        """Read ``span`` from this record's store in one call, normalized to float32."""
        frames = np.asarray(self._store(record)['state'][np.asarray(span,dtype=np.intp)],dtype=np.float32)
        mean,std = self.aux(record)['normalization']
        return (frames-mean[None,:,None,None])/std[None,:,None,None]

    def sample_from_frames(self, record, normalized_frames):
        """Assemble one sample from already-read, already-normalized frames."""
        root = self._store(record)
        aux = self.aux(record)
        indices = validate_record(root,record,time_ns=aux['time_ns'])
        frames = np.asarray(normalized_frames,dtype=np.float32)
        if frames.shape[0] != len(indices):
            raise ValueError('frame count does not match the record\'s indices')
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
            for name,value in init_time_fields(int(aux['time_ns'][indices[-2]])).items()})
        history_ns = np.asarray(aux['time_ns'][indices[:-1]], dtype=np.int64)
        sample['history_offsets_hours'] = torch.from_numpy(history_offsets_hours_from_ns(history_ns))
        if 'process_diagnostics_raw' in root:
            raw = np.asarray(root['process_diagnostics_raw'][indices[-2]],dtype=np.float32)
            pmean,pstd = aux['process_normalization']
            process = (raw-pmean)/pstd
            if not np.isfinite(process).all():
                raise ValueError('nonfinite input-time process targets')
            sample['process_targets'] = torch.from_numpy(process)
        validate_forecast_sample(sample,batched=False)
        return sample

    def __getitem__(self, idx):
        record = self.records[idx]
        indices = self.indices(record)
        frames = self.read_normalized(record,indices)
        return self.sample_from_frames(record,frames)
