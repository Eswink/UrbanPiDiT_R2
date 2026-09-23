import numpy as np
import pytest
from data.preprocess.process_diagnostics import (EARTH_RADIUS_M,spherical_scalar_gradient,
    spherical_divergence,spherical_vorticity,area_weighted_mean,area_weighted_rms,
    horizontal_advection,moisture_flux_convergence)


@pytest.mark.parametrize('lat,lon',[
    ([40,39.75,40],[115,115.25,115.5]),
    ([40,np.nan,39.5],[115,115.25,115.5]),
    ([90,90.25,90.5],[115,115.25,115.5]),
    ([-90,-89.75,-89.5],[115,115.25,115.5]),
    ([40,39.75,39.5],[115,115.25,115]),
    ([40,39.75,39.5],[359.75,0,.25]),
    ([40,39.75,39.5],[115,np.inf,115.5]),
])
@pytest.mark.parametrize('kind',['scalar','divergence','vorticity'])
def test_standalone_functions_reject_invalid_axes(lat,lon,kind):
    x=np.zeros((len(lat),len(lon)))
    with pytest.raises(ValueError):
        if kind=='scalar': spherical_scalar_gradient(x,lat,lon)
        elif kind=='divergence': spherical_divergence(x,x,lat,lon)
        else: spherical_vorticity(x,x,lat,lon)


def test_irregular_monotone_grid_preserves_analytic_derivative():
    lat=np.array([39.,39.4,40.1])
    lon=np.array([115.,115.3,116.])
    field=EARTH_RADIUS_M*np.deg2rad(lat)[:,None]*np.ones((1,3))
    dx,dy=spherical_scalar_gradient(field,lat,lon)
    np.testing.assert_allclose(dx,0.,atol=1e-10)
    np.testing.assert_allclose(dy,1.,rtol=1e-10)
    reverse=spherical_scalar_gradient(field[::-1,::-1],lat[::-1],lon[::-1])
    np.testing.assert_allclose(reverse[1],dy[::-1,::-1],rtol=1e-10)


@pytest.mark.parametrize('summarize',[area_weighted_mean,area_weighted_rms])
def test_region_summary_requires_one_finite_field(summarize):
    lat=[40.,39.75,39.5]
    with pytest.raises(ValueError,match='2-D'): summarize(np.zeros((2,3,4)),lat)
    with pytest.raises(ValueError): summarize(np.full((3,4),np.nan),lat)
    with pytest.raises(ValueError): summarize(np.ones((3,4)),[40.,39.75,40.])
    assert summarize(np.full((3,4),2.),lat)==pytest.approx(2.)


def test_vector_operations_do_not_silently_broadcast_fields():
    x=np.zeros((3,4)); bad=np.zeros((1,4))
    for operation in [horizontal_advection,moisture_flux_convergence]:
        with pytest.raises(ValueError,match='identical'):
            operation(x,bad,x,[40.,39.75,39.5],[115.,115.25,115.5,115.75])
