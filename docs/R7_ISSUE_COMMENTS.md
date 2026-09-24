# R7 issue comments: paste-ready drafts

Status: **NOT YET PUBLISHED.** This machine has no `gh` and no token.
`POST /repos/Eswink/UrbanPiDiT_R2/issues/<n>/comments` was re-tested this round
and still returns **HTTP 401 `Requires authentication`**. No comment here has
been posted to GitHub, and no issue has been closed. Nothing in this file should
be read as a published link or a changed issue state.

To publish, a human needs either `gh auth login` or a PAT with `issues:write`.

---

## #13 — [R7.1-DATA] Build production regional ERA5 0.25° training pipeline

Draft comment (paste verbatim, minus this line):

> **Bounded regional real acquisition landed at `f37ceba` — all four Acceptance
> items now verified against real ERA5.**
>
> The earlier real subset was one 12x12 tile over Jan 1-8. It is now a 65x65
> East-Asia domain over four seasons in three separate years, still acquired
> anonymously with no key.
>
> **Frozen protocol** (`data/download/arco_regional_bounded.py`):
> `gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2`,
> years 2018/2019/2020, season starts Jan/Apr/Jul/Oct 1, four 24-hour blocks per
> year, 48 exact 6-hourly UTC timestamps (no nearest substitution), ROI
> 27-43 N / 107-123 E, nine source variables with the full 13-level pressure
> axis retained.
>
> **Why this shape:** ARCO stores one global `(1,13,721,1440)` chunk per pressure
> variable and timestamp, so a 65x65 crop touches the same chunks as a 12x12
> crop. ROI area is nearly free; timestamps are what cost. The acquisition
> therefore buys regional width and seasonal spread rather than density, inside a
> 16 GiB touched-chunk cap and a wall-clock deadline, both enforced before any
> field value is read.
>
> **Artifact:** 36,980,288 bytes, SHA256
> `d3fa1fba6da46ed59a535ce27f7501813afc2c93cb8b745c90e349454a40960a`
> (re-derived from file bytes and matched against the receipt), shape
> `[48, 65, 65, 13]`, 1222.8 s, `status: downloaded-real-source`,
> `synthetic_fallback: false`, `scientific_claim: false`.
>
> **Acceptance:**
> 1. Deterministic and leakage-safe — `tests/test_r7_era5_converter_identity.py`
>    asserts two independent builds produce byte-identical stored windows,
>    identical manifest records and identical statistics, and that perturbing
>    only the held-out years cannot move the training statistics.
> 2. Synthetic xarray fixtures, no network in CI — the 13 tests in
>    `tests/test_arco_regional_bounded.py` are fully offline.
> 3. Optional real smoke — a gated test runs the real-subset physical checks when
>    the file is locally present (it ran here and passed) and skips cleanly in a
>    clean checkout, with no synthetic stand-in.
> 4. Provenance/normalization written alongside manifests — the audited path
>    produced `BUILD_COMPLETE.json` (`build_complete: true`), `zarr_metadata.json`
>    (17 channels, units, `normalization_years: [2018]`,
>    `spatial_resampling: false`) and `source_preflight.json` carrying the
>    full-file SHA256, plus the acquisition receipt.
>
> **Verified independently this round:** publication contract present; the
> production `ZarrAtmosWindowDataset` loads `[2,17,65,65]` history and
> `[17,65,65]` target at `grid_spacing_deg 0.25`; atmospheric and process
> normalization equal 2018-only statistics; eight process diagnostics are finite
> with non-zero variance and physically sane (divergence RMS 1.9e-5 to 4.6e-5
> s^-1 against an independently computed ~3.9e-5 s^-1; `t850 > t500` at every
> grid point).
>
> **Unit attestation, not a guess:** the `wb13` source omits units on its
> `level` axis. The extractor asserts exact integer identity with the audited
> ERA5 13-level set and cross-checks it against
> `full_37-1h-0p25deg-chunk-1.zarr-v3`, which declares `Hectopascal(hPa)`;
> `conversion_applied: false` is recorded in the receipt.
>
> **Not done / limits:** this is four 24-hour blocks per year, not continuous
> full-year coverage; one ROI (East Asia); 0.25 deg is the native analysis-ready
> grid and is never called finer-resolution truth. HTTP bodies were not
> instrumented, so the receipt reports the uncompressed touched-chunk bound and
> marks actual HTTP as null.
>
> **Verification:** full `pytest -q` 819 passed / 3 skipped / 0 failed with GPU
> present, 813 / 9 / 0 with `CUDA_VISIBLE_DEVICES=""`;
> `python tools/check_conventions.py` 34 blocking rules, 0 violations. Push CI
> run `36034017115` (R7 CPU CI), job `107749511426`, conclusion `success`, all 10
> steps green. Full record: `docs/R7_REGIONAL_ACQUISITION.md`.

Closure statement (only after the comment above is posted):

> Closing: all four Acceptance items are verified against real ERA5 at
> `f37ceba`, with the coverage limitation recorded above. Follow-on work that
> needs denser temporal sampling should be tracked as its own bounded protocol
> rather than holding this parent open, since the converter it asked for exists,
> is audited, and is exercised on real regional data.
