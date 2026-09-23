"""Exact-time held-out forecast windows, separate from model inference inputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from model.r7_rollout import validate_horizons
from .r7_store import validate_store, normalization, HOUR_NS


class ZarrRolloutDataset(Dataset):
    def __init__(self,store_path,*,split='test',lead_hours=(6,12,24,48,72),history_steps=2,step_hours=6):
        import zarr
        if split not in ('val','test'):
            raise ValueError('rollout evaluation requires a held-out split')
        self.path=str(Path(store_path).resolve())
        root=zarr.open_group(self.path,mode='r')
        raw=validate_store(root)
        self.lead_hours=validate_horizons(lead_hours)
        for value in (history_steps,step_hours):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('positive integer history/step required')
        if any(h%step_hours for h in self.lead_hours):
            raise ValueError('horizons must be multiples of the transition step')
        self.history_steps,self.step_hours=history_steps,step_hours
        self.split=split
        self.names=tuple(root.attrs['channels'])
        self.units=tuple(root.attrs.get('units',['unknown']*len(self.names)))
        self.mean,self.std=normalization(root,count=len(self.names))
        self.latitude=np.asarray(root['latitude'][:],dtype=np.float32)
        self.longitude=np.asarray(root['longitude'][:],dtype=np.float32)
        self.times=pd.DatetimeIndex(raw.astype('datetime64[ns]'))
        index={int(t):i for i,t in enumerate(raw)}
        allowed=set(root.attrs['split_years'][split])
        self.windows=[]
        delta=step_hours*HOUR_NS
        for i,stamp in enumerate(raw):
            if self.times[i].year not in allowed or int(stamp)%delta:
                continue
            required=[int(stamp)+k*delta for k in range(-(history_steps-1),max(self.lead_hours)//step_hours+1)]
            if any(t not in index or self.times[index[t]].year not in allowed for t in required):
                continue
            history=[index[int(stamp)-delta*k] for k in reversed(range(history_steps))]
            targets=[index[int(stamp)+h*HOUR_NS] for h in self.lead_hours]
            self.windows.append((history,targets))
        if not self.windows:
            raise ValueError('no complete held-out rollout windows at requested horizons')
        self._root=None

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
            'init_time':self.times[history[-1]].isoformat(),
            'valid_times':[self.times[i].isoformat() for i in targets]}


def fit_training_climatology(store_path):
    """Train-only monthly/hourly grid-cell mean; not WeatherBench2's climatology."""
    import zarr
    root=zarr.open_group(str(store_path),mode='r')
    raw=validate_store(root)
    times=pd.DatetimeIndex(raw.astype('datetime64[ns]'))
    train=set(root.attrs['split_years']['train'])
    means,counts={},{}
    for start in range(0,len(times),int(root['state'].chunks[0])):
        stop=min(len(times),start+int(root['state'].chunks[0]))
        chosen=[i for i in range(start,stop) if times[i].year in train]
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
    return {'kind':'train-only-month-hour-grid-mean-v1','training_years':sorted(train),
        'means':means,'counts':counts,'channels':list(root.attrs['channels'])}


def normalized_climatology(climatology,timestamps,mean,std):
    fields=[]
    for stamp in pd.DatetimeIndex(timestamps):
        key=(stamp.month,stamp.hour)
        if key not in climatology['means']:
            raise ValueError(f'missing training climatology bucket {key}; no held-out fallback')
        fields.append((climatology['means'][key]-mean[:,None,None])/std[:,None,None])
    return torch.from_numpy(np.stack(fields).astype(np.float32))
