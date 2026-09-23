import numpy as np
import pytest
import xarray as xr

from data.download.chunk_budget import ChunkBudgetExceeded
from data.download.wb2_surface_pilot import (
    PILOT_ROI,
    PILOT_TIMES,
    SURFACE_VARIABLES,
    select_surface_pilot,
)


def fixture():
    lat=np.arange(42.0,38.0,-0.25,dtype=np.float32)  # 16 values, ends 38.25
    lon=np.arange(114.0,120.0,0.25,dtype=np.float32) # 24 values
    time=np.asarray(PILOT_TIMES,dtype="datetime64[ns]")
    coords={"time":time,"latitude":lat,"longitude":lon}
    data={}
    units={
        "2m_temperature":"K",
        "10m_u_component_of_wind":"m s**-1",
        "10m_v_component_of_wind":"m s**-1",
        "mean_sea_level_pressure":"Pa",
    }
    for i,name in enumerate(SURFACE_VARIABLES):
        values=np.full((9,16,24),float(i+1),dtype=np.float32)
        da=xr.DataArray(values,dims=("time","latitude","longitude"),coords=coords)
        da.attrs["units"]=units[name]
        # Simulate the real remote WeatherBench2 whole-spatial source chunks.
        da.encoding["chunks"]=(1,721,1440)
        data[name]=da
    return xr.Dataset(data,coords=coords)


def test_surface_pilot_passes_realistic_source_chunk_cap_without_interpolation():
    subset,receipt=select_surface_pilot(fixture())
    assert dict(subset.sizes)=={"time":9,"latitude":16,"longitude":24}
    assert receipt["native_grid_spacing_deg"]==pytest.approx(.25)
    assert receipt["interpolation"] is False
    assert receipt["source_chunk_budget"]["estimated_uncompressed_bytes"]==36*721*1440*4
    assert receipt["source_chunk_budget"]["estimated_uncompressed_bytes"]<192*1024*1024


def test_surface_pilot_refuses_budget_before_field_read():
    with pytest.raises(ChunkBudgetExceeded):
        select_surface_pilot(fixture(),max_chunk_bytes=32*1024*1024)


def test_surface_pilot_rejects_unit_drift():
    ds=fixture()
    ds["mean_sea_level_pressure"].attrs["units"]="hPa"
    with pytest.raises(ValueError,match="units changed"):
        select_surface_pilot(ds)
