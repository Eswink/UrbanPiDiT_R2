"""Score full domain and complementary edge/interior without changing forecasts."""
from __future__ import annotations
import csv
from pathlib import Path
import torch


def boundary_masks(height,width,margins):
    for v in (height,width):
        if isinstance(v,bool) or not isinstance(v,int) or v<1:
            raise ValueError('positive grid dimensions required')
    margins=tuple(margins)
    if not margins or len(margins)!=len(set(margins)):
        raise ValueError('nonempty unique margins required')
    if any(isinstance(m,bool) or not isinstance(m,int) or m<1 or 2*m>=min(height,width) for m in margins):
        raise ValueError('each positive margin must leave a nonempty interior')
    result=[('full',0,torch.ones(height,width,dtype=torch.bool))]
    for m in sorted(margins):
        interior=torch.zeros(height,width,dtype=torch.bool)
        interior[m:-m,m:-m]=True
        result.extend([(f'interior_{m}',m,interior),(f'edge_{m}',m,~interior)])
    return result


class BoundaryRMSEAccumulator:
    """Equal initialization weights, area mean per region, no mixed-unit score.

    This stratifies SAME forecasts. It is not an input halo or boundary forcing
    experiment. Latitude/grid geometry is fixed across all initialization batches.
    """
    def __init__(self,lead_hours,variables,*,margins=(1,),training_std=None,units=None):
        self.lead_hours=tuple(lead_hours)
        self.variables=tuple(variables)
        if not self.lead_hours or any(isinstance(h,bool) or not isinstance(h,int) or h<1 for h in self.lead_hours) or tuple(sorted(set(self.lead_hours)))!=self.lead_hours:
            raise ValueError('positive ordered unique horizons required')
        if not self.variables or any(not isinstance(v,str) or not v for v in self.variables) or len(set(self.variables))!=len(self.variables):
            raise ValueError('unique nonempty variable names required')
        self.margins=tuple(margins)
        # Validate type/duplicates before knowing H,W; dimensions checked on update.
        if not self.margins or len(set(self.margins))!=len(self.margins) or any(isinstance(m,bool) or not isinstance(m,int) or m<1 for m in self.margins):
            raise ValueError('positive unique margins required')
        c=len(self.variables)
        if training_std is None:
            if units is not None:
                raise ValueError('physical units require training-only standard deviations')
            self.std=torch.ones(c,dtype=torch.float64)
            self.units=('normalized',)*c
        else:
            self.std=torch.as_tensor(training_std,dtype=torch.float64).detach().cpu().clone()
            if self.std.shape!=(c,) or not torch.isfinite(self.std).all() or (self.std<=0).any():
                raise ValueError('positive finite training_std [C] required')
            if units is None or len(units)!=c or any(not isinstance(u,str) or not u for u in units):
                raise ValueError('one explicit physical unit per variable required')
            self.units=tuple(units)
        self.initializations=0
        self.sum_squared_error=None
        self.latitude=None
        self.width=None
        self.regions=None

    @torch.no_grad()
    def update(self,prediction,target,latitude):
        if prediction.ndim!=5 or prediction.shape!=target.shape or min(prediction.shape)<1:
            raise ValueError('equal nonempty [B,L,C,H,W] predictions and targets required')
        b,l,c,h,w=prediction.shape
        if (l,c)!=(len(self.lead_hours),len(self.variables)) or prediction.device!=target.device:
            raise ValueError('horizon/channel/device mismatch')
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError('floating point inputs required')
        lat=torch.as_tensor(latitude,dtype=torch.float64).detach().cpu()
        if lat.shape==(h,):
            lat=lat[None,:].expand(b,h)
        if lat.shape!=(b,h) or not torch.isfinite(lat).all() or (lat.abs()>90).any():
            raise ValueError('latitude must be finite [H] or [B,H] in [-90,90]')
        if not torch.equal(lat,lat[0:1].expand(b,h)):
            raise ValueError('one immutable latitude grid per accumulator required')
        axis=lat[0]
        if h>1 and not ((axis.diff()>0).all() or (axis.diff()<0).all()):
            raise ValueError('latitude must be strictly monotone')
        if self.latitude is not None and (self.width!=w or not torch.equal(self.latitude,axis)):
            raise ValueError('grid changed during boundary accumulation')
        regions=boundary_masks(h,w,self.margins)
        weight=torch.cos(torch.deg2rad(axis)).clamp_min(0)[:,None].expand(h,w)
        full_area=weight.sum()
        if full_area<1e-12:
            raise ValueError('no usable latitude area')
        delta=(prediction.detach().double()-target.detach().double())*self.std.to(prediction.device)[None,None,:,None,None]
        squared=delta.square()
        if not torch.isfinite(squared).all():
            raise ValueError('nonfinite input; no cases silently dropped')
        summaries=[]
        definitions=[]
        for name,margin,mask in regions:
            weights=weight*mask
            area=weights.sum()
            if area<1e-12:
                raise ValueError('empty or zero-area scoring region')
            normalized=(weights/area).to(prediction.device)
            summaries.append((squared*normalized).sum((-1,-2)).sum(0).cpu())
            definitions.append(dict(region=name,margin_cells=margin,n_grid_points=int(mask.sum()),
                full_area_fraction=float(area/full_area)))
        contribution=torch.stack(summaries)
        total=contribution if self.sum_squared_error is None else self.sum_squared_error+contribution
        if not torch.isfinite(total).all():
            raise ValueError('boundary accumulation overflow')
        self.sum_squared_error=total
        self.initializations+=b
        self.latitude=axis.clone()
        self.width=w
        self.regions=definitions

    def compute(self):
        if self.initializations==0:
            raise RuntimeError('no initialization samples')
        return (self.sum_squared_error/self.initializations).sqrt()

    def rows(self):
        values=self.compute()
        return [dict(**definition,lead_hours=lead,variable=name,unit=self.units[j],
                     rmse=float(values[r,i,j]),n_initializations=self.initializations)
            for r,definition in enumerate(self.regions)
            for i,lead in enumerate(self.lead_hours) for j,name in enumerate(self.variables)]

    def write_csv(self,path):
        rows=self.rows()
        with Path(path).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
