from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

EARTH_RADIUS_M=6_371_000.0
KAPPA_DRY_AIR=0.2854

PROCESS_DIAGNOSTIC_NAMES=(
    "mslp_gradient_strength",
    "divergence_850_rms",
    "vorticity_850_rms",
    "temperature_advection_850_mean",
    "moisture_advection_850_mean",
    "moisture_convergence_850_mean",
    "static_stability_850_500_mean",
    "vertical_wind_shear_850_500_mean",
)

REQUIRED_PROCESS_CHANNELS=(
    "mslp",
    "t850",
    "q850",
    "u850",
    "v850",
    "t500",
    "u500",
    "v500",
)


def _coordinates(
    latitude:np.ndarray,
    longitude:np.ndarray,
)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    lat=np.asarray(latitude,dtype=np.float64)
    lon=np.asarray(longitude,dtype=np.float64)
    if lat.ndim!=1 or lon.ndim!=1:
        raise ValueError("latitude/longitude 必须是一维坐标")
    if lat.size<2 or lon.size<2:
        raise ValueError("latitude/longitude 至少需要 2 个点")
    if np.any(np.diff(lat)==0) or np.any(np.diff(lon)==0):
        raise ValueError("latitude/longitude 不允许重复坐标")

    lat_rad=np.deg2rad(lat)
    lon_rad=np.deg2rad(lon)
    cos_lat=np.cos(lat_rad)
    if np.any(np.abs(cos_lat)<1e-4):
        raise ValueError("当前诊断不支持极点附近 cos(latitude)≈0 网格")
    return lat_rad,lon_rad,cos_lat


def _gradient(
    field:np.ndarray,
    coordinate:np.ndarray,
    axis:int,
)->np.ndarray:
    edge_order=2 if coordinate.size>=3 else 1
    return np.gradient(
        np.asarray(field,dtype=np.float64),
        coordinate,
        axis=axis,
        edge_order=edge_order,
    )


def spherical_scalar_gradient(
    field:np.ndarray,
    latitude:np.ndarray,
    longitude:np.ndarray,
)->tuple[np.ndarray,np.ndarray]:
    """Return zonal/meridional derivatives df/dx, df/dy on a sphere."""
    lat_rad,lon_rad,cos_lat=_coordinates(latitude,longitude)
    f=np.asarray(field,dtype=np.float64)
    if f.shape[-2:]!=(lat_rad.size,lon_rad.size):
        raise ValueError("field 空间尺寸与 latitude/longitude 不一致")

    d_dlambda=_gradient(f,lon_rad,axis=-1)
    d_dphi=_gradient(f,lat_rad,axis=-2)
    metric=EARTH_RADIUS_M*cos_lat[:,None]
    d_dx=d_dlambda/metric
    d_dy=d_dphi/EARTH_RADIUS_M
    return d_dx,d_dy


def spherical_divergence(
    u:np.ndarray,
    v:np.ndarray,
    latitude:np.ndarray,
    longitude:np.ndarray,
)->np.ndarray:
    """Horizontal wind/flux divergence on a spherical lat/lon grid."""
    lat_rad,lon_rad,cos_lat=_coordinates(latitude,longitude)
    u=np.asarray(u,dtype=np.float64)
    v=np.asarray(v,dtype=np.float64)
    if u.shape!=v.shape or u.shape[-2:]!=(lat_rad.size,lon_rad.size):
        raise ValueError("u/v 空间尺寸不一致")

    du_dlambda=_gradient(u,lon_rad,axis=-1)
    v_cos=v*cos_lat[:,None]
    d_vcos_dphi=_gradient(v_cos,lat_rad,axis=-2)
    return (
        du_dlambda+d_vcos_dphi
    )/(EARTH_RADIUS_M*cos_lat[:,None])


def spherical_vorticity(
    u:np.ndarray,
    v:np.ndarray,
    latitude:np.ndarray,
    longitude:np.ndarray,
)->np.ndarray:
    """Vertical component of relative vorticity on a sphere."""
    lat_rad,lon_rad,cos_lat=_coordinates(latitude,longitude)
    u=np.asarray(u,dtype=np.float64)
    v=np.asarray(v,dtype=np.float64)
    if u.shape!=v.shape or u.shape[-2:]!=(lat_rad.size,lon_rad.size):
        raise ValueError("u/v 空间尺寸不一致")

    dv_dlambda=_gradient(v,lon_rad,axis=-1)
    u_cos=u*cos_lat[:,None]
    d_ucos_dphi=_gradient(u_cos,lat_rad,axis=-2)
    return (
        dv_dlambda-d_ucos_dphi
    )/(EARTH_RADIUS_M*cos_lat[:,None])


def horizontal_advection(
    scalar:np.ndarray,
    u:np.ndarray,
    v:np.ndarray,
    latitude:np.ndarray,
    longitude:np.ndarray,
)->np.ndarray:
    d_dx,d_dy=spherical_scalar_gradient(
        scalar,latitude,longitude
    )
    return -(
        np.asarray(u,dtype=np.float64)*d_dx+
        np.asarray(v,dtype=np.float64)*d_dy
    )


def moisture_flux_convergence(
    q:np.ndarray,
    u:np.ndarray,
    v:np.ndarray,
    latitude:np.ndarray,
    longitude:np.ndarray,
)->np.ndarray:
    q=np.asarray(q,dtype=np.float64)
    return -spherical_divergence(
        q*np.asarray(u,dtype=np.float64),
        q*np.asarray(v,dtype=np.float64),
        latitude,
        longitude,
    )


def _weights(latitude:np.ndarray,width:int)->np.ndarray:
    lat_rad,_,cos_lat=_coordinates(
        latitude,np.arange(width,dtype=np.float64)
    )
    del lat_rad
    return np.broadcast_to(cos_lat[:,None],(cos_lat.size,width))


def area_weighted_mean(
    field:np.ndarray,
    latitude:np.ndarray,
)->float:
    x=np.asarray(field,dtype=np.float64)
    w=np.cos(np.deg2rad(np.asarray(latitude,dtype=np.float64)))[:,None]
    if x.shape[-2]!=w.shape[0]:
        raise ValueError("field H 与 latitude 不一致")
    return float(
        np.sum(x*w)/(
            np.sum(w)*x.shape[-1]
        )
    )


def area_weighted_rms(
    field:np.ndarray,
    latitude:np.ndarray,
)->float:
    x=np.asarray(field,dtype=np.float64)
    return float(np.sqrt(max(
        area_weighted_mean(x*x,latitude),
        0.0,
    )))


def _channel_map(
    state:np.ndarray,
    channel_names:Sequence[str],
)->Mapping[str,np.ndarray]:
    state=np.asarray(state,dtype=np.float64)
    if state.ndim!=3:
        raise ValueError("state 必须为 [C,H,W]")
    names=list(channel_names)
    if len(names)!=state.shape[0]:
        raise ValueError("channel_names 长度与 state C 不一致")
    if len(set(names))!=len(names):
        raise ValueError("channel_names 不允许重复")
    missing=[name for name in REQUIRED_PROCESS_CHANNELS if name not in names]
    if missing:
        raise KeyError(f"缺少 process diagnostic channels: {missing}")
    return {name:state[names.index(name)] for name in names}


def compute_process_diagnostic_vector(
    state:np.ndarray,
    channel_names:Sequence[str],
    latitude:np.ndarray,
    longitude:np.ndarray,
)->np.ndarray:
    """Compute eight input-time meteorological diagnostic proxy targets.

    Inputs must be in physical ERA5 units. These values are *diagnostic proxies*,
    not causal ground truth, and must be computed from the current/input state,
    never from the future forecast target unless an explicitly target-side task
    is being studied.
    """
    fields=_channel_map(state,channel_names)
    lat=np.asarray(latitude,dtype=np.float64)
    lon=np.asarray(longitude,dtype=np.float64)

    grad_x,grad_y=spherical_scalar_gradient(
        fields["mslp"],lat,lon
    )
    pressure_gradient=np.sqrt(grad_x*grad_x+grad_y*grad_y)

    divergence=spherical_divergence(
        fields["u850"],fields["v850"],lat,lon
    )
    vorticity=spherical_vorticity(
        fields["u850"],fields["v850"],lat,lon
    )
    temperature_advection=horizontal_advection(
        fields["t850"],
        fields["u850"],fields["v850"],
        lat,lon,
    )
    moisture_advection=horizontal_advection(
        fields["q850"],
        fields["u850"],fields["v850"],
        lat,lon,
    )
    moisture_convergence=moisture_flux_convergence(
        fields["q850"],
        fields["u850"],fields["v850"],
        lat,lon,
    )

    theta850=fields["t850"]*(1000.0/850.0)**KAPPA_DRY_AIR
    theta500=fields["t500"]*(1000.0/500.0)**KAPPA_DRY_AIR
    stability=theta500-theta850

    wind_shear=np.sqrt(
        (fields["u500"]-fields["u850"])**2+
        (fields["v500"]-fields["v850"])**2
    )

    vector=np.asarray([
        area_weighted_mean(pressure_gradient,lat),
        area_weighted_rms(divergence,lat),
        area_weighted_rms(vorticity,lat),
        area_weighted_mean(temperature_advection,lat),
        area_weighted_mean(moisture_advection,lat),
        area_weighted_mean(moisture_convergence,lat),
        area_weighted_mean(stability,lat),
        area_weighted_mean(wind_shear,lat),
    ],dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("process diagnostics 包含非有限值")
    return vector
