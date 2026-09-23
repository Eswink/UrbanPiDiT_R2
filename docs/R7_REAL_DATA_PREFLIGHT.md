# Local real-data preflight (#30)

```bash
pip install -r requirements-r7-data.txt
python prepare_r7_local.py --source /path/to/regional-era5.nc --config configs/r7_era5_local.yaml
# Only after inspecting the report, explicitly permit a bounded new output:
python prepare_r7_local.py --source /path/to/regional-era5.nc --config configs/r7_era5_local.yaml --write --store /new/path/era5.zarr --manifests /new/path/manifests --max-raw-gib 2
```

The first command reads coordinates/schema plus one selected frame and prints a
report. It creates no cache, downloads nothing and opens local NetCDF4/HDF5 via
h5netcdf or an existing Zarr directory. Input should already be a regional subset;
this entrypoint deliberately does not authorize a global cloud download.

Canonical units for the 17-channel plan are K, Pa, m s-1, kg kg-1 and m2 s-2.
Pressure-level coordinates need explicit hPa/millibar units. Celsius, hPa-valued
surface pressure and geopotential height in metres are not silently converted or
relabeled. Unknown units/custom channels require an explicit audited mapping.
Reference: ECMWF ERA5 documentation https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
and xarray open_dataset https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html.

The phrase 'native 0.25-degree' in this project means the model learns directly
on the supplied 0.25-degree ERA5 distribution grid, without this pipeline making
pseudo-finer truth. ERA5's original atmospheric analysis is not itself a native
0.25-degree regular-grid numerical simulation; the CDS regular grid is a
regridded distribution product. Do not conflate these meanings in the paper.

Write requires an explicit positive raw-state GiB cap and fresh outputs. The cap
is on uncompressed dynamic arrays, not a guarantee of final disk usage or runtime.
Full finite-field checking occurs while writing; preflight checks only one frame.
No real source authenticity or scientific training readiness is certified by a
successful report. Small local files have full SHA256; larger files are labeled
stat-only, Zarr hashes root metadata only. Scientific provenance still needs a
trusted source/download manifest and immutable raw data.

The proposed split in the example config is a starting configuration, not a claim
that those years have been downloaded. The current conversation attachments are
two screenshots, not an ERA5 cache. Actual real-data/hardware acceptance requires
an accessible local/source cache and measurements from the user's GPU environment.
