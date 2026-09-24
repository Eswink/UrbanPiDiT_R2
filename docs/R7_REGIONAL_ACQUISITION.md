# R7 real-data regional acquisition (#13)

Real, anonymous, no-key ERA5 via the public ARCO archive, widened from the
earlier 12x12 / January-only subset to a 65x65 East-Asia domain across four
seasons in three separate years, under an explicit byte budget.

This page records one bounded acquisition. It is an **engineering data path**:
`scientific_claim: false` everywhere. It does not establish forecast skill.

## What changed and why

The earlier real subset was one 12x12 tile over the first eight days of January.
That is genuine ERA5, but it cannot support the process/adaptive/rollout gates
that need regional context and seasonal spread. Two measured facts drove the new
sampling protocol:

1. **ROI area is nearly free; timestamps are what cost.** ARCO stores one global
   `(1, 13, 721, 1440)` chunk per pressure variable and timestamp, so a 65x65
   crop touches the same chunks as a 12x12 crop. Widening the domain therefore
   adds almost no transfer, while every extra timestamp costs a full slab.
2. **Precision matters for budgeting.** Measured per-timestamp cost for all nine
   source variables ranged from ~1.1 s to ~22.6 s across runs, so the module
   enforces a *chunk-byte* budget plus a wall-clock deadline rather than relying
   on a throughput estimate.

The extraction therefore buys **area and seasonal spread, not density**: four
24-hour blocks per year, three years, at full regional width.

## Frozen protocol

| Item | Value |
| --- | --- |
| Source | `gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2` |
| Access | anonymous public GCS, `storage_options={"token": "anon"}`; no credentials |
| Years | 2018, 2019, 2020 |
| Season starts | Jan 1, Apr 1, Jul 1, Oct 1 (four blocks per year, 12 blocks) |
| Steps per block | 4 (one continuous 24-hour sequence at 6-hourly cadence) |
| Timestamps | 48, exact UTC; no nearest-neighbour substitution |
| ROI | 27.0–43.0 N, 107.0–123.0 E → 65x65 native 0.25-degree points |
| Stored variables | 9: `2m_temperature`, `10m_{u,v}_component_of_wind`, `mean_sea_level_pressure`, `geopotential`, `temperature`, `specific_humidity`, `{u,v}_component_of_wind` (pressure variables keep all 13 levels) |
| Interpolation / resampling | none; the target is the native 0.25-degree grid |

Storing the full 13-level pressure axis is deliberate: the source charges the
whole slab anyway, so subsetting at read time would cost the same while making
the stored schema differ from the source. Channel and level selection stay a
converter concern, which also keeps `geopotential` (needed for `z*` channels)
available without a second source read.

## Acquisition result

Implemented in `data/download/arco_regional_bounded.py`; entry point
`python -m data/download/arco_regional_bounded.py` (see `--help`).

```
status            downloaded-real-source
shape             [48, 65, 65, 13]
timestamps        48
local artifact    36,980,288 bytes
SHA256            d3fa1fba6da46ed59a535ce27f7501813afc2c93cb8b745c90e349454a40960a
elapsed           1222.8 s
chunk estimate    12.81 GiB uncompressed touched chunks (cap 16 GiB)
```

The hash in `source_receipt.json` was re-derived from the file bytes during this
round and matches. `synthetic_fallback: false`, `scientific_claim: false`.

**Note on the byte number.** 12.81 GiB is the *uncompressed touched-chunk*
bound, not HTTP traffic and not peak RAM. This reader does not instrument HTTP
bodies, and the receipt says so explicitly
(`source_chunk_budget.actual_http_bytes: null`). A single `temperature`
timestamp object is ~28.3 MiB compressed upstream; the uncompressed bound is the
conservative gate that decision used.

### Pressure-level unit attestation

The `wb13` source omits units on its `level` axis. The extractor **refuses to
guess**:

- it asserts the axis is exactly the audited ERA5 set
  `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` as integer
  identity — a reordered or shifted axis raises;
- the sibling dataset `full_37-1h-0p25deg-chunk-1.zarr-v3` declares the same
  integers with `units: Hectopascal(hPa)`;
- `conversion_applied: false` is recorded, and the evidence string is stored in
  the receipt rather than asserted only in prose.

## Conversion through the audited publication path

Read-only preflight, then an explicitly authorized write to **new** targets:

```bash
python prepare_r7_local.py --source outputs/r7_regional_real/source.nc \
  --config /tmp/r7_preflight_cfg.yaml --report /tmp/r7_real_preflight.json

python prepare_r7_local.py --source outputs/r7_regional_real/source.nc \
  --config /tmp/r7_preflight_cfg.yaml --write \
  --store outputs/r7_regional_real/cache.zarr \
  --manifests outputs/r7_regional_real/manifests --max-raw-gib 0.05
```

Result: 17-channel store, `shape [48, 17, 65, 65]`, 8 complete +6h windows per
split (`train 2018 / val 2019 / test 2020`, chronological and disjoint), eight
process diagnostics enabled, and `BUILD_COMPLETE.json` present with
`build_complete: true`, `schema_version: 1`, `time_unit: ns`.

Verified independently in this round:

| Check | Result |
| --- | --- |
| Full-file source SHA256 | `d3fa1fba…40960a`, matches the receipt |
| Publication contract | `BUILD_COMPLETE.json` + zarr `build_complete=true` |
| Production reader | `ZarrAtmosWindowDataset` loads `[2, 17, 65, 65]` history and `[17, 65, 65]` target at `grid_spacing_deg = 0.25` |
| Normalization source | `normalization_years = [2018]`; means equal 2018-only statistics |
| Target semantics | `native ERA5 grid; physical units retained; no spatial upsampling` |

### Train-only normalization, verified rather than asserted

Atmospheric and process statistics are computed from the 16 training rows only.
Two process channels (`moisture_advection_850_mean`,
`moisture_convergence_850_mean`) have stored `std = 1e-6` while their empirical
std is ~1e-8. That is the documented `eps=1e-6` floor in
`RunningMoments.finish` acting on near-degenerate channels, **not** leakage: the
other six channels reproduce the 2018-only statistics to ~3.6e-8 relative error,
and the floored channels reproduce too once the same floor is applied. Recorded
here because a bare `std` comparison looks like a mismatch until the floor is
accounted for.

### Physical sanity of the real field

All eight diagnostics are finite with non-zero variance. Divergence RMS
(1.9e-5 to 4.6e-5 s⁻¹) and vorticity RMS (3.2e-5 to 9.7e-5 s⁻¹) sit in the
expected synoptic range; an independent finite-difference divergence computed
directly from the 850 hPa wind fields gave ~3.9e-5 s⁻¹. Static stability is
14.96–27.99 K, 850–500 hPa shear 6.78–27.81 m/s, and `t850 > t500` holds at every
grid point.

## Tests

| Suite | Result |
| --- | --- |
| `tests/test_arco_regional_bounded.py` | 13 passed (offline planning/attestation/budget/failure-record, plus the real-subset smoke below) |
| `tests/test_r7_era5_converter_identity.py` | 2 passed (byte-identical rebuild; held-out years cannot move statistics) |
| Full `pytest -q` (GPU present) | **819 passed, 3 skipped, 0 failed** |
| Full `pytest -q` with `CUDA_VISIBLE_DEVICES=""` | 813 passed, 9 skipped, 0 failed |
| `python tools/check_conventions.py` | 34 blocking rules, **0 violations** |

Push CI for this commit: run `36034017115` (event `push`, workflow **R7 CPU CI**),
job `107749511426`, conclusion **success**, all 10 steps green including
`Check repository conventions` and
`Run unit, integration and installed-wheel tests`. Per-test counts from that job
were not retrievable without an authenticated log download (HTTP 403), so the
pass/skip numbers above are the locally measured ones for the same commit
content, not a transcription of the CI log.

CI needs no network: every non-smoke test uses synthetic xarray fixtures. The
real-subset assertions are gated on
`outputs/r7_regional_real/source.nc` being present, which it is **not** in a
clean checkout, so CI skips them and no synthetic stand-in is ever substituted.
The two new suites belong to Acceptance items 1–3 directly.

## Honest limitations

- **Sparse, not continuous.** 48 timestamps, not full-year coverage. Windows per
  split are 8 each. `#13`'s "representative coverage" is improved
  (four seasons x three years x regional width) but is **not** a continuous
  multi-year archive.
- **One ROI.** 65x65 over East Asia only; other regions are untested by data.
- **0.25 degrees is the native analysis-ready grid**, already regridded upstream
  by ARCO. It is never called higher-resolution truth, and nothing here is
  interpolated upward.
- **Engineering only.** No forecast skill, process-benefit, adaptive-halting or
  GPU claim is supported by this artifact.
- **HTTP bytes not measured** for this acquisition; the receipt reports the
  uncompressed touched-chunk bound and explicitly marks actual HTTP as `null`.
- The subset and its cache are runtime artifacts under `outputs/` and are
  deliberately **not** committed; only the code and these records are.

## Reproduce

```bash
.venv/bin/python -c "
from data.download.arco_regional_bounded import RegionalPlan, extract_regional_region
extract_regional_region('outputs/r7_regional_real/source.nc',
                        'outputs/r7_regional_real/source_receipt.json',
                        plan=RegionalPlan(max_chunk_bytes=16*2**30, deadline_seconds=3600.))
"
```

Re-running requires new, empty destinations (`fresh_outputs` refuses existing
paths by design). Upstream attribution: **Contains modified Copernicus Climate
Change Service information 2026; distribution via ARCO-ERA5** —
<https://github.com/google-research/arco-era5>.
