"""Pooled latitude-weighted anomaly correlation against frozen climatology."""
from __future__ import annotations
import torch
from model.r7_rollout import validate_horizons


class RolloutACCAccumulator:
    """Accumulate weighted anomaly dot products before division, per lead/variable.

    Climatology is external, frozen and must use the same normalization as both
    fields. This is pooled ACC, not mean-per-case ACC. Zero-energy anomalies are
    explicitly undefined (NaN), never assigned artificial perfect/zero skill.
    """
    def __init__(self,lead_hours,variables):
        self.lead_hours=validate_horizons(lead_hours)
        self.variables=tuple(variables)
        if not self.variables or len(set(self.variables))!=len(self.variables):
            raise ValueError('unique nonempty variables required')
        shape=(len(self.lead_hours),len(self.variables))
        self.dot=torch.zeros(shape,dtype=torch.float64)
        self.forecast_energy=torch.zeros_like(self.dot)
        self.target_energy=torch.zeros_like(self.dot)
        self.initializations=0

    @torch.no_grad()
    def update(self,prediction,target,climatology,latitude):
        if prediction.ndim!=5 or prediction.shape!=target.shape or prediction.shape!=climatology.shape or min(prediction.shape)<1:
            raise ValueError('equal nonempty [B,L,C,H,W] fields required')
        b,l,c,h,w=prediction.shape
        if (l,c)!=self.dot.shape or not (prediction.device==target.device==climatology.device):
            raise ValueError('horizon/channel/device mismatch')
        if not all(x.is_floating_point() and torch.isfinite(x).all() for x in (prediction,target,climatology)):
            raise ValueError('ACC needs finite floating-point fields')
        lat=torch.as_tensor(latitude,device=prediction.device,dtype=torch.float64)
        if lat.shape==(h,):
            lat=lat[None,:].expand(b,h)
        if lat.shape!=(b,h) or not torch.isfinite(lat).all() or (lat.abs()>90).any():
            raise ValueError('invalid latitude')
        weight=torch.cos(torch.deg2rad(lat)).clamp_min(0)
        sums=weight.sum(-1)
        if (sums<1e-12).any():
            raise ValueError('zero usable area')
        weight=weight/sums[:,None]
        p=prediction.double()-climatology.double()
        t=target.double()-climatology.double()
        def reduce(field):
            return (field.mean(-1)*weight[:,None,None,:]).sum(-1).sum(0).cpu()
        values=[reduce(p*t),reduce(p*p),reduce(t*t)]
        if not all(torch.isfinite(v).all() for v in values):
            raise ValueError('ACC sufficient-statistic overflow')
        self.dot+=values[0]
        self.forecast_energy+=values[1]
        self.target_energy+=values[2]
        self.initializations+=b

    def compute(self):
        if not self.initializations:
            raise RuntimeError('no initialization samples')
        denom=self.forecast_energy.sqrt()*self.target_energy.sqrt()
        value=torch.full_like(denom,float('nan'))
        valid=denom>0
        value[valid]=(self.dot[valid]/denom[valid]).clamp(-1,1)
        return value
