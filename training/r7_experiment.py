"""Shared model/checkpoint utilities for bounded, local R7 experiments."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch


def make_model(kind, config):
    from model.weather_forecaster_r7 import NativeAtmosForecaster
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    classes = {'native':NativeAtmosForecaster, 'generic':GenericRecursiveWeatherForecaster, 'process':ProcessForecastCoReasoner}
    if kind not in classes:
        raise ValueError(f'unsupported model kind: {kind}')
    return classes[kind](**config)


def select_device(name, bf16=False):
    device = torch.device(name)
    if device.type not in ('cpu','cuda'):
        raise ValueError('single CPU/CUDA devices only')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        torch.cuda.set_device(device)
        if bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 requested but unsupported')
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    return {'python':random.getstate(), 'numpy':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG topology differs')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def dataset_identity(manifest):
    """Bind manifests, coordinates, time, norms and metadata; not all raw chunks.

    Full raw-data provenance must be supplied/audited separately. This fingerprint
    is intentionally bounded and never pretends to authenticate ERA5 observations.
    """
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    ds = ZarrAtmosWindowDataset(manifest)
    roots = {}
    for rec in ds.records:
        key = rec['store_path']
        if key in roots:
            continue
        root = ds._store(rec)
        arrays = ['latitude','longitude','time_ns','normalization_mean','normalization_std']
        arrays += [n for n in ['process_normalization_mean','process_normalization_std'] if n in root]
        roots[key] = {'attributes':dict(root.attrs), 'shape':list(root['state'].shape),
            'arrays':{n:hashlib.sha256(np.asarray(root[n][:]).tobytes()).hexdigest() for n in arrays}}
    result = {'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(), 'stores':roots}
    return canonical_digest(result), ds


def save_exclusive(path, payload):
    """Fully write a temporary checkpoint, then publish without overwriting."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.r7-checkpoint-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp,path)  # fails if another producer already published path
    finally:
        os.unlink(temp)


def load_checkpoint(path, *, expected=None):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format') != 'r7-local-v1':
        raise ValueError('unsupported R7 checkpoint format')
    if expected is not None and checkpoint['signature'] != expected:
        raise ValueError('checkpoint model/data/training identity differs')
    if checkpoint['signature'] != canonical_digest(checkpoint['contract']):
        raise ValueError('checkpoint contract digest mismatch')
    return checkpoint
