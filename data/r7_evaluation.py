"""Exact-time held-out forecast windows, separate from model inference inputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from model.r7_rollout import validate_horizons
from .r7_store import (HOUR_NS, init_time_fields, validate_store, normalization,
    split_time_labels, history_offsets_hours_from_ns)


class RolloutStoreScan:
    """Read-only store facts that do not depend on the requested horizons.

    ``ZarrRolloutDataset`` derives all of this inside its constructor, so a sweep that
    evaluates one lead at a time re-read and re-scanned the entire store once per lead.
    The scan holds exactly the values the window derivation needs and nothing that
    depends on the horizons, so a caller may build it once and hand it to several
    datasets. Every field is a plain array/dict/tuple: the scan pickles like the
    dataset it feeds.
    """

    def __init__(self, store_path, *, split='test', history_steps=2, step_hours=6):
        import zarr
        if split not in ('val', 'test'):
            raise ValueError('rollout evaluation requires a held-out split')
        self.path = str(Path(store_path).resolve())
        root = zarr.open_group(self.path, mode='r')
        raw = validate_store(root)
        for value in (history_steps, step_hours):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('positive integer history/step required')
        self.split = split
        self.history_steps, self.step_hours = history_steps, step_hours
        self.names = tuple(root.attrs['channels'])
        self.units = tuple(root.attrs.get('units', ['unknown'] * len(self.names)))
        self.mean, self.std = normalization(root, count=len(self.names))
        self.latitude = np.asarray(root['latitude'][:], dtype=np.float32)
        self.longitude = np.asarray(root['longitude'][:], dtype=np.float32)
        self.raw = np.asarray(raw)
        self.times = pd.DatetimeIndex(raw.astype('datetime64[ns]'))
        self.index = {int(t): i for i, t in enumerate(raw)}
        self.labels = split_time_labels(root)
        self.allowed_years = (None if self.labels is not None
                              else frozenset(root.attrs['split_years'][split]))

    def permits(self, position):
        """Whether the store timestep at ``position`` belongs to this held-out split."""
        if self.labels is None:
            return self.times[position].year in self.allowed_years
        return self.labels[position] == self.split

    def windows(self, lead_hours):
        """The complete held-out windows at exactly these horizons, in store order."""
        lead_hours = validate_horizons(lead_hours)
        if any(h % self.step_hours for h in lead_hours):
            raise ValueError('horizons must be multiples of the transition step')
        delta = self.step_hours * HOUR_NS
        windows = []
        for i, stamp in enumerate(self.raw):
            if not self.permits(i) or int(stamp) % delta:
                continue
            required = [int(stamp) + k * delta for k in
                        range(-(self.history_steps - 1), max(lead_hours) // self.step_hours + 1)]
            if any(t not in self.index or not self.permits(self.index[t]) for t in required):
                continue
            history = [self.index[int(stamp) - delta * k] for k in reversed(range(self.history_steps))]
            targets = [self.index[int(stamp) + h * HOUR_NS] for h in lead_hours]
            windows.append((history, targets))
        if not windows:
            raise ValueError('no complete held-out rollout windows at requested horizons')
        return windows


class ZarrRolloutDataset(Dataset):
    def __init__(self, store_path, *, split='test', lead_hours=(6, 12, 24, 48, 72), history_steps=2,
                 step_hours=6, scan=None):
        if scan is None:
            scan = RolloutStoreScan(store_path, split=split, history_steps=history_steps,
                                    step_hours=step_hours)
        elif (scan.split != split or scan.history_steps != history_steps
              or scan.step_hours != step_hours):
            raise ValueError('the shared store scan does not match this dataset request')
        self.path = scan.path
        self.history_steps, self.step_hours = scan.history_steps, scan.step_hours
        self.split = scan.split
        self.names = scan.names
        self.units = scan.units
        self.mean, self.std = scan.mean, scan.std
        self.latitude, self.longitude = scan.latitude, scan.longitude
        self.times = scan.times
        self.lead_hours = validate_horizons(lead_hours)
        if any(h % self.step_hours for h in self.lead_hours):
            raise ValueError('horizons must be multiples of the transition step')
        self.windows = scan.windows(self.lead_hours)
        self._root = None

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        result=dict(self.__dict__)
        result['_root']=None
        return result

    def __getitem__(self,index):
        import zarr
        if self._root is None:
            self._root=zarr.open_group(self.path,mode='r')
            validate_store(self._root)
        history,targets=self.windows[index]
        frames=np.stack([np.asarray(self._root['state'][i],dtype=np.float32) for i in history+targets])
        frames=(frames-self.mean[None,:,None,None])/self.std[None,:,None,None]
        if not np.isfinite(frames).all():
            raise ValueError('nonfinite rollout data')
        return {'coarse_history':torch.from_numpy(frames[:len(history)]),
            'rollout_targets':torch.from_numpy(frames[len(history):]),
            'latitude':torch.from_numpy(self.latitude.copy()),'longitude':torch.from_numpy(self.longitude.copy()),
            'lead_time_hours':torch.tensor(float(self.step_hours)),
            'history_offsets_hours':torch.from_numpy(history_offsets_hours_from_ns(
                np.asarray([self.times[i].value for i in history], dtype=np.int64))),
            'init_time':self.times[history[-1]].isoformat(),
            'valid_times':[self.times[i].isoformat() for i in targets],
            # The same derivation the training reader uses, so the two producers
            # cannot drift into different definitions of "initialization time".
            **{name:torch.tensor(float(value)) for name,value in
               init_time_fields(int(self.times[history[-1]].value)).items()}}


def fit_training_climatology(store_path):
    """Train-only monthly/hourly grid-cell mean; not WeatherBench2's climatology.

    Membership follows the store's own declared split semantics: the declared
    half-open train ``split_time_ranges`` when ``split_mode == 'time_ranges'``,
    otherwise the declared train years. On a single-year engineering segment
    the year filter cannot discriminate train from the held-out interior
    windows, so it would fit the climatology on validation/test steps too.
    """
    import zarr
    root=zarr.open_group(str(store_path),mode='r')
    raw=validate_store(root)
    times=pd.DatetimeIndex(raw.astype('datetime64[ns]'))
    labels=split_time_labels(root)
    train=set(root.attrs['split_years']['train'])
    n_selected=0
    means,counts={},{}
    for start in range(0,len(times),int(root['state'].chunks[0])):
        stop=min(len(times),start+int(root['state'].chunks[0]))
        if labels is None:
            chosen=[i for i in range(start,stop) if times[i].year in train]
        else:
            chosen=[i for i in range(start,stop) if labels[i]=='train']
        n_selected+=len(chosen)
        for i in chosen:
            stamp=times[i]
            key=(stamp.month,stamp.hour)
            x=np.asarray(root['state'][i],dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError('nonfinite training climatology input')
            n=counts.get(key,0)+1
            means[key]=x.copy() if n==1 else means[key]+(x-means[key])/n
            counts[key]=n
    if not counts:
        raise ValueError('no training climatology records')
    result={'kind':'train-only-month-hour-grid-mean-v1','training_years':sorted(train),
        'means':means,'counts':counts,'channels':list(root.attrs['channels']),
        'n_selected_steps':n_selected,
        'selection':'declared_train_years' if labels is None else 'declared_train_time_ranges'}
    if labels is not None:
        result['split_mode']='time_ranges'
        result['train_time_ranges']=root.attrs['split_time_ranges']['train']
    return result


def normalized_climatology(climatology,timestamps,mean,std):
    fields=[]
    for stamp in pd.DatetimeIndex(timestamps):
        key=(stamp.month,stamp.hour)
        if key not in climatology['means']:
            raise ValueError(f'missing training climatology bucket {key}; no held-out fallback')
        fields.append((climatology['means'][key]-mean[:,None,None])/std[:,None,None])
    return torch.from_numpy(np.stack(fields).astype(np.float32))
