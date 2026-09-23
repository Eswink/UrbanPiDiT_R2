import numpy as np
import pytest
from test_r7_earthmover_plan import Array, root_fixture
from data.download.earthmover_pilot import DecodedBudget, bounded_selection, extract_dataset


@pytest.mark.parametrize('kwargs', [dict(limit=True),dict(limit=float('inf')),dict(limit=0),
    dict(used=-1),dict(reads=.5),dict(seconds=float('nan')),dict(seconds=float('inf')),
    dict(seconds=0),dict(started=float('nan')),dict(limit=1,used=2)])
def test_invalid_budgets_refused(kwargs):
    with pytest.raises(ValueError): DecodedBudget(**kwargs)


@pytest.mark.parametrize('n', [True, -1, .1, float('nan')])
def test_invalid_charges_refused(n):
    with pytest.raises(ValueError): DecodedBudget().check(n)


@pytest.mark.parametrize('key', ['_FillValue','missing_value'])
def test_finite_cf_sentinel_is_not_a_valid_weather_observation(key):
    a = Array(np.array([[0., -9999.],[2.,3.]], dtype='f4'), (2,2), attrs={key:-9999.})
    with pytest.raises(ValueError,match='CF missing'):
        bounded_selection(a,(np.arange(2),np.arange(2)),DecodedBudget())


def test_zarr_allocation_fill_does_not_invalidate_physical_zeros():
    a = Array(np.zeros((2,2),dtype='f4'), (2,2))
    a.fill_value = 0
    np.testing.assert_array_equal(bounded_selection(a,(np.arange(2),np.arange(2)),DecodedBudget()),a.x)


@pytest.mark.parametrize('mode', ['folded','duplicate','irregular','wrong-spacing','field-shape'])
def test_shared_bad_geometry_fails_before_weather_reads(mode):
    root = root_fixture()
    for group in root.values():
        lat = group['latitude'].x
        if mode == 'folded': lat[[1,2]] = lat[[2,1]]
        elif mode == 'duplicate': lat[2] = lat[1]
        elif mode == 'irregular': lat[2] += .01
        elif mode == 'wrong-spacing':
            lat[:] = np.linspace(42,36.5,12)
            group['longitude'].x[:] = np.linspace(114,119.5,12)
    if mode == 'field-shape': root['single/temporal']['t2m'].shape = (95,12,12)
    with pytest.raises(ValueError): extract_dataset(root)
    assert not root['single/temporal']['t2m'].calls
    assert not root['pressure/temporal']['t'].calls


def test_original_mapping_is_unchanged():
    root = root_fixture()
    ds, report = extract_dataset(root)
    assert report['times_per_year'] == 32 and report['grid_spacing_deg'] == .25
    np.testing.assert_array_equal(ds['t2m'].values, root['single/temporal']['t2m'].x)
    np.testing.assert_array_equal(ds['t500'].values, root['pressure/temporal']['t'].x[:,1])
