import h5netcdf
import numpy as np

from data.download.ncar_public_probe import describe_h5netcdf


def test_describe_h5netcdf_reports_chunks_without_loading_fields(tmp_path):
    path=tmp_path/"tiny.nc"
    with h5netcdf.File(path,"w") as nc:
        nc.dimensions={"time":2,"level":2,"latitude":4,"longitude":6}
        t=nc.create_variable(
            "T",("time","level","latitude","longitude"),
            dtype="f4",chunks=(1,1,2,3),compression="gzip"
        )
        t.attrs["units"]="K"
    with h5netcdf.File(path,"r") as nc:
        report=describe_h5netcdf(nc)
    assert report["dimensions"]=={"time":2,"level":2,"latitude":4,"longitude":6}
    meta=report["variables"]["T"]
    assert meta["chunks"]==[1,1,2,3]
    assert meta["compression"]=="gzip"
    assert meta["units"]=="K"
    assert meta["uncompressed_chunk_bytes"]==24
