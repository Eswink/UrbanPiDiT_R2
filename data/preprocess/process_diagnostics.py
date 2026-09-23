"""Physical-unit diagnostic proxies, not causal labels or closed energy budgets."""
from __future__ import annotations
from typing import Mapping,Sequence
import numpy as np

EARTH_RADIUS_M=6_371_000.0
KAPPA_DRY_AIR=.2854
PROCESS_DIAGNOSTIC_NAMES=(
    'mslp_gradient_strength','divergence_850_rms','vorticity_850_rms',
    'temperature_advection_850_mean','moisture_advection_850_mean','moisture_convergence_850_mean',
    'static_stability_850_500_mean','vertical_wind_shear_850_500_mean',
)
REQUIRED_PROCESS_CHANNELS=('mslp','t850','q850','u850','v850','t500','u500','v500')


def _axis(values,name,minimum=2):
    raw=np.asarray(values)
    if raw.ndim!=1 or raw.size<minimum or raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a numeric 1-D axis with >= {minimum} points')
    x=raw.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite values')
    delta=np.diff(x)
    if len(x)>1 and not (np.all(delta>0) or np.all(delta<0)):
        raise ValueError(f'{name} must be strictly monotone')
    if name=='latitude':
        if np.any(np.abs(x)>90):
            raise ValueError('latitude outside [-90,90]')
    else:
        convention=np.all((x>=-180)&(x<=180)) or np.all((x>=0)&(x<360))
        if not convention or np.ptp(x)>=360:
            raise ValueError('longitude must use one non-wrapped convention without duplicate seam')
    return x


def _coordinates(latitude,longitude):
    lat,lon=_axis(latitude,'latitude'),_axis(longitude,'longitude')
    phi,lam=np.deg2rad(lat),np.deg2rad(lon)
    cosine=np.cos(phi)
    if np.any(np.abs(cosine)<1e-4):
        raise ValueError('process diagnostics do not support near-pole grids')
    return phi,lam,cosine


def _field(value):
    result=np.asarray(value,dtype=np.float64)
    if result.ndim<2 or min(result.shape)<1 or not np.isfinite(result).all():
        raise ValueError('finite fields with nonempty spatial dimensions required')
    return result


def _gradient(field,coordinate,axis):
    return np.gradient(_field(field),coordinate,axis=axis,edge_order=2 if len(coordinate)>=3 else 1)


def spherical_scalar_gradient(field,latitude,longitude):
    phi,lam,cosine=_coordinates(latitude,longitude)
    field=_field(field)
    if field.shape[-2:]!=(len(phi),len(lam)):
        raise ValueError('field/coordinate shape mismatch')
    return (_gradient(field,lam,-1)/(EARTH_RADIUS_M*cosine[:,None]),
        _gradient(field,phi,-2)/EARTH_RADIUS_M)


def _wind_fields(u,v,phi,lam):
    u,v=_field(u),_field(v)
    if u.shape!=v.shape or u.shape[-2:]!=(len(phi),len(lam)):
        raise ValueError('wind/coordinate shape mismatch')
    return u,v


def spherical_divergence(u,v,latitude,longitude):
    phi,lam,cosine=_coordinates(latitude,longitude)
    u,v=_wind_fields(u,v,phi,lam)
    return (_gradient(u,lam,-1)+_gradient(v*cosine[:,None],phi,-2))/(EARTH_RADIUS_M*cosine[:,None])


def spherical_vorticity(u,v,latitude,longitude):
    phi,lam,cosine=_coordinates(latitude,longitude)
    u,v=_wind_fields(u,v,phi,lam)
    return (_gradient(v,lam,-1)-_gradient(u*cosine[:,None],phi,-2))/(EARTH_RADIUS_M*cosine[:,None])


def horizontal_advection(scalar,u,v,latitude,longitude):
    scalar,u,v=_field(scalar),_field(u),_field(v)
    if scalar.shape!=u.shape or scalar.shape!=v.shape:
        raise ValueError('advection fields must have identical shapes')
    dx,dy=spherical_scalar_gradient(scalar,latitude,longitude)
    return -(u*dx+v*dy)


def moisture_flux_convergence(q,u,v,latitude,longitude):
    q,u,v=_field(q),_field(u),_field(v)
    if q.shape!=u.shape or q.shape!=v.shape:
        raise ValueError('moisture flux fields must have identical shapes')
    return -spherical_divergence(q*u,q*v,latitude,longitude)


def _weights(latitude,width):
    lat=_axis(latitude,'latitude',minimum=1)
    if width<1:
        raise ValueError('positive width required')
    weights=np.cos(np.deg2rad(lat))
    if np.any(weights<1e-4):
        raise ValueError('near-pole summaries unsupported')
    return np.broadcast_to(weights[:,None],(len(lat),width))


def area_weighted_mean(field,latitude):
    field=_field(field)
    if field.ndim!=2:
        raise ValueError('regional summary requires a single 2-D field')
    weights=_weights(latitude,field.shape[-1])
    if weights.shape!=field.shape:
        raise ValueError('summary field/latitude shape mismatch')
    return float(np.sum(field*weights)/np.sum(weights))


def area_weighted_rms(field,latitude):
    field=_field(field)
    return float(np.sqrt(max(area_weighted_mean(field*field,latitude),0.)))


def _channel_map(state:np.ndarray,channel_names:Sequence[str])->Mapping[str,np.ndarray]:
    state=_field(state)
    names=list(channel_names)
    if state.ndim!=3 or len(names)!=state.shape[0] or len(set(names))!=len(names):
        raise ValueError('unique channel names must match [C,H,W] state')
    missing=[name for name in REQUIRED_PROCESS_CHANNELS if name not in names]
    if missing:
        raise KeyError(f'缺少 process diagnostic channels: {missing}')
    return dict(zip(names,state))


def compute_process_diagnostic_vector(state,channel_names,latitude,longitude):
    """Eight input-time proxies in physical units before training normalization.

    Stability is theta500-theta850 (K), not Brunt-Vaisala frequency. Shear is
    the magnitude of the 500-minus-850 wind difference (m/s), not a vertical
    derivative. Input-time summaries cannot alone enforce future physical budgets.
    """
    f=_channel_map(state,channel_names)
    dx,dy=spherical_scalar_gradient(f['mslp'],latitude,longitude)
    divergence=spherical_divergence(f['u850'],f['v850'],latitude,longitude)
    vorticity=spherical_vorticity(f['u850'],f['v850'],latitude,longitude)
    at=horizontal_advection(f['t850'],f['u850'],f['v850'],latitude,longitude)
    aq=horizontal_advection(f['q850'],f['u850'],f['v850'],latitude,longitude)
    mq=moisture_flux_convergence(f['q850'],f['u850'],f['v850'],latitude,longitude)
    stability=f['t500']*(1000./500.)**KAPPA_DRY_AIR-f['t850']*(1000./850.)**KAPPA_DRY_AIR
    shear=np.sqrt((f['u500']-f['u850'])**2+(f['v500']-f['v850'])**2)
    mean=lambda x:area_weighted_mean(x,latitude)
    result=np.asarray([mean(np.sqrt(dx*dx+dy*dy)),area_weighted_rms(divergence,latitude),
        area_weighted_rms(vorticity,latitude),mean(at),mean(aq),mean(mq),mean(stability),mean(shear)],dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite diagnostic output')
    return result
