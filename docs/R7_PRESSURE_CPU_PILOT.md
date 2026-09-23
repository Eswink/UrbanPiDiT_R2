# Bounded public ERA5 pressure/CPU pilot (#44/#45)

Source: https://registry.opendata.aws/earthmover-era5/ . Anonymous, read-only
Icechunk snapshot `ZFKDHBCTBVHVXM3BQFV0` in the public quarterly ERA5 edition,
NOT a paid subscription. Attribution: Copernicus C3S/ECMWF, NSF NCAR and Earthmover;
catalog license CC-BY-4.0. The source was subset in time/level/space without spatial
interpolation. Retaining its 0.25-degree grid is not a claim to recover finer
physical scales than the underlying reanalysis.

The NCAR metadata probe confirmed that its tested geopotential chunks still span
all 37 levels and the global grid. Instead this pilot uses temporal tiles with
chunks [8736,1,12,12] for pressure variables and [8736,12,12] for surface fields.
Only a Beijing-containing 12x12 tile, 32 six-hourly timestamps in each of 2018,
2019 and 2020, four surface channels and seven pressure-level channels are selected.
The 850/500 hPa levels are selected from real coordinate values, not nearest-level
interpolation. All eight existing process diagnostics are computed from physical
input-time values, with training-only normalization.

Before weather reads, the entire selected chunk plan must fit 192 MiB of decoded
chunk bytes. Each chunk is read once per selected variable-level; per-chunk and
selected-output allocations are capped at 8 MiB. Sharded layouts, missing values,
missing times, coordinate or unit mismatch fail closed. This is a conservative
logical decoded-read budget, NOT instrumented HTTP traffic or measured peak RAM.
The client may read additional Icechunk metadata; network_body_bytes is null.
The abandoned NCAR transport prototype is not part of this implementation.

Explicit command (no schedule):

```bash
pip install -r requirements.txt -r requirements-r7-data.txt 'icechunk==2.2.2' 'pcodec>=0.3'
python scripts/real_r7_pressure_pilot.py --out outputs/my_pressure_pilot
```

Outputs must be new. The pilot executes 20 CPU updates each for native/generic/
process models, then train-only controller fitting, validation-only policy selection
and held-out free-running 6/12/24/72h scoring on six initializations. Source NetCDF,
payload hashes, receipts, checkpoints, normalization metadata and metric tables
are preserved. Negative comparisons are not suppressed. Tests use separately
labeled synthetic fixtures to validate indexing and budget logic.

These are CPU integration experiments on January-only, tiny-region data, not
converged training, SOTA, statistical significance, full East-Asia coverage or
4090D memory/latency acceptance. GPU work remains deferred.
