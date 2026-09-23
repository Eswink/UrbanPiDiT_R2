# Tiny real-source engineering proof (#33)

The optional workflow runs only when a push commit explicitly includes
`[real-smoke]`. It is not a scheduled task and does not download data for every
normal CI commit. The script is fixed to nine historical 2m-temperature frames
and a 5x7 Beijing-region ROI. No credentials, pressure-level archive, GPU rental
or large training experiment is involved.

Hard bounds: <=64 MiB response bodies, <=64 successfully fetched objects,
<=16 MiB per object/decoded chunk, per-request timeout 20s, download deadline
240s and workflow timeout 10min. Unsupported schemas/codecs/redirects fail.
Surface time chunks are fetched then cropped; this is not arbitrary global data
access. Coordinate/time arrays are decoded from upstream metadata, not guessed.
Every fetched object has URL, byte count, SHA256 and available ETag/generation
recorded. HTTP/schema failures leave explicit receipts, no synthetic fallback.

The optional pipeline applies the same audited local preflight/Zarr builder,
trains a tiny native model for two CPU optimizer updates, restores its checkpoint
and evaluates one +6h held-out initialization. This proves a real-data software
path only. Nine frames of one variable cannot establish forecast skill, process
reasoning benefit, multi-year generalization, SOTA or 24GB GPU feasibility.

Data: ECMWF ERA5 / Copernicus Climate Change Service, distributed by ARCO-ERA5.
Official access/chunking and attribution instructions:
https://github.com/google-research/arco-era5
The analysis-ready 0.25-degree distribution is regridded upstream; this pipeline
does not interpolate it to fabricate finer-resolution target data.
