from __future__ import annotations

import numpy as np
import pytest

from data.preprocess.process_diagnostics import (
    EARTH_RADIUS_M,
    PROCESS_DIAGNOSTIC_NAMES,
    compute_process_diagnostic_vector,
    horizontal_advection,
    spherical_divergence,
    spherical_scalar_gradient,
    spherical_vorticity,
)


def _grid():
    lat=np.array([-10.0,0.0,10.0],dtype=np.float64)
    lon=np.array([100.0,110.0,120.0,130.0],dtype=np.float64)
    return lat,lon


def test_spherical_scalar_meridional_gradient_uses_metric_distance():
    lat,lon=_grid()
    phi=np.deg2rad(lat)[:,None]
    field=EARTH_RADIUS_M*phi*np.ones((1,len(lon)))
    dx,dy=spherical_scalar_gradient(field,lat,lon)
    np.testing.assert_allclose(dx,0.0,atol=1e-10)
    np.testing.assert_allclose(dy,1.0,rtol=1e-6,atol=1e-6)


def test_spherical_divergence_and_vorticity_have_known_analytic_value():
    lat,lon=_grid()
    phi=np.deg2rad(lat)[:,None]
    lam=np.deg2rad(lon)[None,:]
    cosphi=np.cos(phi)
    k=2.5e-5

    # u = k R cos(phi) lambda, v=0 -> divergence = k.
    u=k*EARTH_RADIUS_M*cosphi*lam
    v=np.zeros_like(u)
    div=spherical_divergence(u,v,lat,lon)
    np.testing.assert_allclose(div,k,rtol=1e-6,atol=1e-10)

    # v = k R cos(phi) lambda, u=0 -> relative vorticity = k.
    vort=spherical_vorticity(
        np.zeros_like(u),
        k*EARTH_RADIUS_M*cosphi*lam,
        lat,lon,
    )
    np.testing.assert_allclose(vort,k,rtol=1e-6,atol=1e-10)


def test_horizontal_advection_sign_is_physical():
    lat,lon=_grid()
    phi=np.deg2rad(lat)[:,None]
    lam=np.deg2rad(lon)[None,:]
    scalar=3.0*EARTH_RADIUS_M*np.cos(phi)*lam
    u=np.full_like(scalar,4.0)
    v=np.zeros_like(scalar)
    adv=horizontal_advection(scalar,u,v,lat,lon)
    np.testing.assert_allclose(adv,-12.0,rtol=1e-6,atol=1e-6)


def test_process_vector_is_finite_and_semantically_ordered():
    lat,lon=_grid()
    H,W=len(lat),len(lon)
    yy,xx=np.meshgrid(
        np.arange(H,dtype=np.float64),
        np.arange(W,dtype=np.float64),
        indexing="ij",
    )
    names=[
        "mslp","t850","q850","u850","v850",
        "t500","u500","v500",
    ]
    state=np.stack([
        100000.0+10.0*xx,
        290.0+0.4*xx,
        0.010+1e-4*xx,
        8.0+0.2*yy,
        2.0+0.1*xx,
        260.0+0.2*xx,
        20.0+0.2*yy,
        8.0+0.1*xx,
    ],axis=0)
    vector=compute_process_diagnostic_vector(
        state,names,lat,lon
    )
    assert vector.shape==(8,)
    assert len(PROCESS_DIAGNOSTIC_NAMES)==8
    assert np.isfinite(vector).all()
    assert vector[0]>0
    assert vector[7]>0


def test_process_vector_requires_physical_channel_contract():
    lat,lon=_grid()
    state=np.zeros((2,len(lat),len(lon)),dtype=np.float32)
    with pytest.raises(KeyError,match="缺少 process diagnostic channels"):
        compute_process_diagnostic_vector(
            state,["mslp","t850"],lat,lon
        )
