"""Metadata-only probe of public NSF/NCAR ERA5 NetCDF4 objects.

Uses anonymous S3 and the HDF5 file-object driver. No field arrays are loaded.
The probe exists to decide whether small region/time/level range reads are
bandwidth-safe before any real multivariate extraction is attempted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import math

import h5netcdf
import numpy as np

NCAR_T2M_202401 = (
    "nsf-ncar-era5/e5.oper.an.sfc/202401/"
    "e5.oper.an.sfc.128_167_2t.ll025sc.2024010100_2024013123.nc"
)
NCAR_Z_20240101 = (
    "nsf-ncar-era5/e5.oper.an.pl/202401/"
    "e5.oper.an.pl.128_129_z.ll025sc.2024010100_2024010123.nc"
)


def describe_h5netcdf(nc: h5netcdf.File) -> dict:
    dims={str(k): int(v.size) for k,v in nc.dimensions.items()}
    variables={}
    for name,var in nc.variables.items():
        h5=var._h5ds
        chunks=None if h5.chunks is None else [int(v) for v in h5.chunks]
        variables[str(name)]={
            "dimensions":[str(v) for v in var.dimensions],
            "shape":[int(v) for v in var.shape],
            "dtype":str(var.dtype),
            "chunks":chunks,
            "compression":None if h5.compression is None else str(h5.compression),
            "compression_opts":h5.compression_opts,
            "shuffle":bool(h5.shuffle),
            "fletcher32":bool(h5.fletcher32),
            "units":str(var.attrs.get("units","")),
        }
        if chunks is not None:
            variables[str(name)]["uncompressed_chunk_bytes"]=int(
                math.prod(chunks)*np.dtype(var.dtype).itemsize
            )
    return {"dimensions":dims,"variables":variables}


def probe_object(key: str, *, block_size: int = 1 << 20) -> dict:
    if not key.startswith("nsf-ncar-era5/"):
        raise ValueError("probe_object is restricted to the public nsf-ncar-era5 bucket")
    if block_size < 64 * 1024 or block_size > 8 * 1024 * 1024:
        raise ValueError("block_size must stay between 64 KiB and 8 MiB")
    import s3fs
    fs=s3fs.S3FileSystem(anon=True)
    info=fs.info(key)
    with fs.open(
        key,"rb",block_size=block_size,cache_type="blockcache"
    ) as raw:
        with h5netcdf.File(raw,"r") as nc:
            report=describe_h5netcdf(nc)
    report.update({
        "source":"s3://"+key,
        "access":"anonymous-public-s3",
        "object_size_bytes":int(info["Size"]),
        "probe_block_size_bytes":int(block_size),
        "field_values_loaded":False,
    })
    return report


def probe_ncar_examples() -> dict:
    return {
        "surface_2m_temperature":probe_object(NCAR_T2M_202401),
        "pressure_geopotential":probe_object(NCAR_Z_20240101),
    }
