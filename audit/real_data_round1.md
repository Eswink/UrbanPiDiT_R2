# Real Data Audit Round 1 — Authenticity & Reproducibility

## Initial score: 74/100 — FAIL

### Failure reasons

1. Dynamic fixture was real Beijing hourly meteorology, but `urban_static` was still a synthetic coordinate/radius/station scaffold.
2. Network download failure had been observed, but source status was not yet aggregated into a machine-readable audit.
3. WorldCover path only had a WMS preview downloader; no categorical COG ROI downloader.
4. Provenance did not yet contain a second real geospatial checksum.

## Rework

- Added real Dongcheng administrative-boundary GeoJSON smoke fixture and checksum.
- Rasterized the real polygon into `real_district_mask` and a derived edge channel.
- Added `PROVENANCE.md` and structured `provenance.json` with explicit scientific limitations.
- Added Google ARCO ERA5 downloader with explicit optional-dependency checks.
- Added WorldCover WMS downloader with friendly no-fake-fallback errors.
- Added official-pattern WorldCover categorical COG ROI downloader.
- Added `scripts/try_real_downloads.py` and `audit/real_data_download_status.json`.

## Retest score: 92/100 — PASS

### Evidence

- Dynamic fixture checksum is fixed and verified.
- Static GeoJSON checksum is fixed and verified.
- `dynamic_is_real_observation=true`.
- `static_contains_real_geodata=true`.
- `is_urban_morphology=false` and `scientific_training_ready=false` prevent scientific overclaim.
- Current sandbox network/DNS failures are recorded rather than hidden.
