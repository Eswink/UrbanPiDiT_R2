# R7 issue comments: paste-ready drafts

Status: **NOT YET PUBLISHED.** This machine has no `gh` and no token.
`POST /repos/Eswink/UrbanPiDiT_R2/issues/<n>/comments` was re-tested in round 1 and once more before round 2
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

---

## #20 — [R7-MEMORY] Bound recursive training activation retention

Draft comment (paste verbatim, minus this line):

> **The multi-seed optimization/trick comparison is now done at `7dfbea6`.**
> This was the one remaining item; the engineering measurements landed earlier
> at `50954c94`.
>
> The earlier sweep measured each configuration once at a single seed, so a
> small delta could not be separated from machine variation that the same record
> documents as up to ~40%. Two changes fix that: every configuration is
> replicated across seeds 41/42/43, and each trick is compared against its
> baseline **within the same seed**, so seed drift cancels instead of inflating
> the effect. Cells also now read the published regional ERA5 R7 store
> (17 channels, 65x65, `BUILD_COMPLETE.json`) through the production reader,
> replacing synthetic 12x12 shapes.
>
> **Protocol:** frozen before any measurement
> (`protocol.json`, digest
> `6c84be201134faa3a0895ebb9c1bfb1cf67f4161dd283fd613ba2a33fc70e7f7`), 72 cells
> = 2 models x K in {1,4,8} x 2 training modes x checkpointing on/off x 3 seeds,
> one process per cell. Run: 72/72 cells, 0 failed, 266.4 s.
>
> **Streamed truncated vs full BPTT** (per-seed paired, MiB allocated):
> `generic` K=8 −393 (sd 125, all seeds same sign), K=4 −145, K=1 +10;
> `process` K=8 −174, K=4 −44 (sign not consistent across seeds), K=1 +9
> (also not consistent). Full BPTT rises 934 -> 1164 -> 1464 MiB over K=1/4/8
> for `generic` while streamed stays 938 -> 957 -> 957 — the flat-K result now
> reproduced on real multivariate data rather than synthetic shapes.
>
> **Activation checkpointing** is the strongest lever: −469 to −627 MiB, a
> consistent 51.8–54.5 % cut of peak, for +25 to +38 ms per step. All six rows
> agree in sign on both metrics.
>
> **Negative results kept:** streamed training is **slower at every K**
> (+12.0 to +98.0 ms); memory savings are paid for in step time. Two `process`
> rows have memory deltas whose sign disagrees across seeds, so at those
> settings the effect is below seed dispersion and this run does not establish
> it. They are reported as unresolved, not as wins.
>
> **Reproducibility:** the sweep was run twice end to end. Peak allocated and
> peak reserved reproduced **bit-identically in all 72 cells** (0.0 MiB max
> difference), while step time did not (max 41.0 ms, mean 8.2 ms; 35/72 within
> 5 ms). Memory is stable enough to pair across seeds; timing is reported with
> dispersion and not as a headline.
>
> Three seeds is a dispersion check, **not** a significance test — no p-values
> and none should be inferred. Not a skill, convergence or final-quality claim.
> Full record: `docs/R7_GPU_MULTISEED.md`; tests:
> `tests/test_r7_multiseed_comparison.py` (17 offline).
>
> **Verification:** `pytest -q` 836 passed / 3 skipped / 0 failed with GPU;
> 830 / 9 / 0 with `CUDA_VISIBLE_DEVICES=""`;
> `python tools/check_conventions.py` 34 blocking rules, 0 violations.

Closure statement (only after the comment above is posted):

> Closing: the CPU acceptance items (gradient ownership, optimizer cadence,
> repeatable finite updates, saved-tensor accounting as a function of K) were
> covered earlier, and the measured 3090 peak allocated/reserved memory and wall
> time for K=1/2/4/8 now include a multi-seed comparison of both training tricks
> on real data. The reported limitations — three seeds as a noise check, and two
> unresolved `process` rows — are recorded rather than smoothed over.
