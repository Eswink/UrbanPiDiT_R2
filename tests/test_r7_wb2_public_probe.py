import numpy as np
import xarray as xr

from data.download.wb2_public_probe import describe_dataset


def test_describe_dataset_does_not_require_field_materialization():
    time=np.array(["2020-01-01T00","2020-01-01T06"],dtype="datetime64[h]")
    lat=np.array([40.0,39.75],dtype=np.float64)
    lon=np.array([116.0,116.25,116.5],dtype=np.float64)
    level=np.array([500,850],dtype=np.int32)
    ds=xr.Dataset(
        {
            "2m_temperature":(("time","latitude","longitude"),np.zeros((2,2,3),dtype=np.float32)),
            "temperature":(("time","level","latitude","longitude"),np.zeros((2,2,2,3),dtype=np.float32)),
        },
        coords={"time":time,"level":level,"latitude":lat,"longitude":lon},
    )
    ds["2m_temperature"].attrs["units"]="K"
    ds["temperature"].attrs["units"]="K"
    report=describe_dataset(ds,("2m_temperature","temperature","missing"))
    assert report["dimensions"]=={"time":2,"level":2,"latitude":2,"longitude":3}
    assert report["variables"]["2m_temperature"]["units"]=="K"
    assert report["variables"]["temperature"]["dims"]==["time","level","latitude","longitude"]
    assert report["missing_variables"]==["missing"]
    assert report["coordinates"]["level"]["samples"]==[500.0,850.0]
