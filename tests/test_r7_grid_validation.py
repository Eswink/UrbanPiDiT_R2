import numpy as np
import pandas as pd
import pytest
import xarray as xr
from data.preprocess.grid import regular_latlon_spacing
from data.preprocess.r7_era5 import ERA5ChannelSpec, build_r7_era5_npz_from_dataset
from data.preprocess.r7_era5_zarr import build_r7_era5_zarr_from_dataset


@pytest.mark.parametrize('lat,lon', [
    ([40,39.75,39.5],[115,115.25,115.5]),
    ([39.5,39.75,40],[115.5,115.25,115]),
    ([0,.25,.5],[179.75,180,180.25]),
    ([0,.25,.5],[-180,-179.75,-179.5]),
    ([0,.25,.5],[359.75,359.5,359.25]),
])
def test_valid_axis_order_is_preserved(lat,lon):
    lat,lon=np.asarray(lat),np.asarray(lon)
    before_lat,before_lon=lat.copy(),lon.copy()
    assert regular_latlon_spacing(lat,lon)==.25
    np.testing.assert_array_equal(lat,before_lat)
    np.testing.assert_array_equal(lon,before_lon)


@pytest.mark.parametrize('lat,lon', [
    ([40,39.75,40],[115,115.25,115.5]),
    ([40,40,39.75],[115,115.25,115.5]),
    ([0,.25,.5],[115,115.25,115]),
    ([0,.25,.5],[359.75,0,.25]),
    ([0,.25],[-180,180]),
    ([0,.25],[0,360]),
    ([0,.25],[-181,-180.75]),
    ([90,90.25],[0,.25]),
    ([-90,-90.25],[0,.25]),
    ([0,np.nan],[0,.25]),
    ([0,.25],[0,np.inf]),
    ([[0,.25]],[0,.25]),
    ([0],[0,.25]),
    ([0,.25],[[0,.25]]),
    ([0,.25,.6],[0,.25,.5]),
    ([0,.25,.5],[0,.25,.6]),
    ([0,.5],[0,.25]),
    ([False,True],[0,.25]),
])
def test_invalid_axes_raise(lat,lon):
    with pytest.raises(ValueError):
        regular_latlon_spacing(lat,lon)


@pytest.mark.parametrize('builder',['npz','zarr'])
@pytest.mark.parametrize('latitude',[[40,39.75,40],[90,90.25,90.5],[40,np.nan,39.5]])
def test_both_builders_reject_before_creating_output(tmp_path,builder,latitude):
    times=pd.DatetimeIndex(np.concatenate([
        pd.date_range(f'{year}-01-01',periods=6,freq='6h').values
        for year in [2018,2019,2020]
    ]))
    ds=xr.Dataset({'t':(('time','latitude','longitude'),np.zeros((18,3,3)))},
        coords={'time':times,'latitude':latitude,'longitude':[115,115.25,115.5]})
    kwargs=dict(ds=ds,specs=(ERA5ChannelSpec('t'),),
        manifest_dir=tmp_path/'manifest',split_years={'train':[2018],'val':[2019],'test':[2020]})
    with pytest.raises(ValueError):
        if builder=='npz':
            build_r7_era5_npz_from_dataset(out_dir=tmp_path/'output',**kwargs)
        else:
            build_r7_era5_zarr_from_dataset(store_path=tmp_path/'output',**kwargs)
    assert not (tmp_path/'output').exists()
