# Verified tiny real-source proof (#33 / #35)

The bounded workflow completed successfully on source commit
`5f2ddcc9e8e206302b39617ecdacf255017bedc3`:
https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/35858478284
Job: 107172590025. Artifact: 10748357388 (10,655 bytes).

The artifact was downloaded and its ZIP digest independently checked:
`e7f63e7972a95e3a094662b41adbd064ab67755b5f11552ed58fcddf722eacc1`.
The receipts record **21,258,262 response-body bytes across 32 objects**, within
64 MiB/64-object bounds, with source URL/SHA256/ETag/generation per object.
It contains t2m for 00/06/12 UTC on January 1 of 2018, 2019 and 2020, latitude
40.5..39.5, longitude 115.5..117.0 at 0.25 degrees, shape [9,1,5,7], units K.

Actual pipeline: real ARCO reads -> exact ROI crop -> local unit/schema preflight
-> train-only normalization and versioned Zarr -> **two CPU optimizer updates**
-> checkpoint restore -> **one held-out +6h initialization**. The workflow did
not train a useful forecasting model or evaluate a representative test period.

The artifact's NetCDF SHA256 is
`db1edf860ca3ce90b3a0466644b8ea3fea578a67811cf9df8f8cf40f6a4e3d06`.
The exact 315 float32 t2m values have raw little-endian byte SHA256
`5f859932025f2fa889928991225f0310bfa75853e606cf7819f1b684151ffef1`.
Those bytes, original coordinates/times and provenance references are pinned in
`tests/fixtures/r7_arco_t2m.json`. The offline regression verifies the digest and
replays the same software chain without remote downloads. This is a real-grid
fixture, separate from the older optional Beijing station fixtures.

Attribution: ECMWF ERA5 / Copernicus Climate Change Service, distributed via
ARCO-ERA5. Contains modified Copernicus Climate Change Service information 2026.
Original access/attribution instructions: https://github.com/google-research/arco-era5

## What this does not establish

Nine one-variable frames are **not** a multi-year 17-channel atmospheric training
set, a process-label validation dataset, a 72h weather-skill benchmark, SOTA,
GPU peak-memory measurement or wall-clock adaptive-inference speedup. The
full-data and actual-hardware research gates remain open. Artifacts expire;
the pinned bytes/checksums retain the small offline regression, not the whole
upstream source archive. No interpolation created finer-resolution targets.
