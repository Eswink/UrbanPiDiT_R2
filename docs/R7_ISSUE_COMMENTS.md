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

---

## #5 — [R7.2] Implement generic TRM-like recursive weather baseline

Draft comment (paste verbatim, minus this line):

> **The acceptance criterion — "comparable parameter/FLOP budget" — is now
> measured rather than assumed, at `bf57fd4`.**
>
> The model, K=1/2/4/6/8 support, deep-supervised drafts and the
> truncated-recursion option already existed; parameter counts were already
> reported per variant. What did **not** exist anywhere in the repository was any
> **FLOP accounting**, so the FLOP half of the criterion had never been checked.
> A repo-wide case-insensitive search for `flop` (excluding the read-only archive)
> returned no hits.
>
> **Result**, at `dim=128, depth=4`, batch 2, reading real 17-channel ERA5 windows
> from the published regional store, with a 5 % tolerance declared before
> measurement:
>
> | metric | generic | process | relative difference |
> | --- | --- | --- | --- |
> | parameters | 1,264,034 | 1,264,419 | +0.030 % |
> | FLOPs K=1 | 5,006,418,176 | 5,006,422,272 | +0.0001 % |
> | FLOPs K=2 | 5,662,100,736 | 5,662,108,928 | +0.0001 % |
> | FLOPs K=4 | 6,973,465,856 | 6,973,482,240 | +0.0002 % |
> | FLOPs K=6 | 8,284,830,976 | 8,284,855,552 | +0.0003 % |
> | FLOPs K=8 | 9,596,196,096 | 9,596,228,864 | +0.0003 % |
>
> The 385-parameter gap is **entirely** the process readout (`process_readout`).
> Backbone (862,225), recursive cell (331,008), correction head (42,897) and draft
> encoder (9,088) are identical across the two models, and both carry a 16-token
> recursive state — which is why the FLOP ratio stays at 1.0000 for every K even
> as absolute cost grows about 1.9x from K=1 to K=8.
>
> **Counting convention is recorded in the report**, because a FLOP number without
> it is not comparable: `FlopCounterMode` over one forward pass; elementwise and
> normalization ops not counted; backward not counted and not assumed equal;
> gradients required (the counter raises under `torch.no_grad()` on these models).
> Attention uses `F.scaled_dot_product_attention`, which contributes no counted
> parameters.
>
> **What this does not say.** A budget match is not evidence of forecast skill; it
> only removes capacity as a confound, which is exactly what #6 needs. Whether
> process state helps remains #6's question, and its three-seed evidence is mixed
> (spatial feedback helps some wind scores but worsens T500). No test split was
> touched.
>
> Full record: `docs/R7_BUDGET_PARITY.md`; tests:
> `tests/test_r7_budget_audit.py` (8 offline). Verification: `pytest -q` 844
> passed / 3 skipped / 0 failed with GPU, 838 / 9 / 0 with
> `CUDA_VISIBLE_DEVICES=""`; `python tools/check_conventions.py` 34 blocking
> rules, 0 violations.

Closure statement (only after the comment above is posted):

> Closing: all four listed tasks exist (parameter-shared recursive cell, K=1/2/4/6/8,
> deep-supervised drafts, truncated-recursion option for bounded VRAM) and the
> accuracy-vs-reasoning-steps reporting is in
> `R7_PRESSURE_CPU_RESULTS.md` / `R7_CPU_REFINEMENT_RESULTS.md`. The acceptance
> criterion of a comparable parameter/FLOP budget with R7.3 is now measured at
> every K, with the counting convention stated.

---

## #6 — [R7.3] Implement Process–Forecast Co-Reasoning Loop

Draft comment (paste verbatim, minus this line):

> **The fair-budget comparison is done at `1bc1eef`. The gate is NOT met, and
> this is now quantified rather than asserted.**
>
> #5 had just established that the arms are budget-comparable (+0.030 %
> parameters, +0.0001–0.0003 % forward FLOPs), so what was missing was a run that
> held budget and seed fixed while reporting **every** variable — with 17 channels
> spanning K, Pa and m/s, an aggregate score would be meaningless.
>
> **Protocol** (frozen before any optimizer step, digest
> `253c5a442934fa82f6d7bd2bbdfa8c1d0ca668175faad5bbfbcf700bf6bec296`): three arms
> (`generic`, `process_no_feedback`, `process_feedback`) x seeds 41/42/43 x
> identical 200-update budget, ablation at K=0 (no recursion) / K=1 / K=3,
> validation split only, +6 h lead, 8 capped initializations, all 17 channels.
> Parameters 92,450 vs 92,547 (a 97-parameter process readout). Data: a new
> contiguous ten-day January block per year at native 0.25°, 65x65, real 17-channel
> ERA5, SHA256 `db7c02191e520a81873ab90b2a35348a43d556fb65a82da0cd60cce03041a2f4`,
> 38 windows per split through the audited converter.
>
> **Result.** Both process arms fail. At K=3, `process_feedback` has a lower mean
> RMSE on 10 of 17 variables — and **15 of those 17 deltas change sign between
> seeds**, so the win/loss split is not evidence. Counting only deltas whose sign
> is consistent across all three seeds:
>
> | depth | arm | established improved | established worsened | unresolved |
> | --- | --- | --- | --- | --- |
> | K=0 | process_feedback | 1 | 6 | 10 |
> | K=0 | process_no_feedback | 1 | 5 | 11 |
> | K=1 | process_feedback | 3 | 5 | 9 |
> | K=1 | process_no_feedback | 1 | 5 | 11 |
> | K=3 | process_feedback | 1 | 1 | 15 |
> | K=3 | process_no_feedback | 2 | 2 | 13 |
>
> `gate_met: false` for every arm and depth.
>
> **Three findings, stated plainly:**
>
> 1. The clearest established effect is a **loss**: `t2m` worsens at every depth
>    for both arms with all seeds agreeing (+0.23 K no-feedback, +0.30 K feedback
>    at K=3).
> 2. **Forecast feedback does not rescue it.** At K=3 the feedback arm has fewer
>    established improvements (1) than the no-feedback arm (2), and adds a
>    worsening (`v250`) the no-feedback arm does not have. On this evidence
>    feedback is not the missing ingredient.
> 3. **Most differences are unresolved.** At K=3, 15/17 (feedback) and 13/17
>    (no-feedback) deltas flip sign across seeds. The honest reading is that the
>    process arms are largely **indistinguishable** from generic at three seeds —
>    not better.
>
> **Correction to earlier framing.** The previously recorded statement that
> spatial feedback helps some wind scores but worsens T500 does not reproduce
> here: this run shows `t500` *improving* under the no-feedback arm (−0.023 K, all
> seeds) and unresolved under the feedback arm at K=3. What does reproduce is the
> general mixed pattern, with `t2m` as the consistent loser.
>
> The sign-stability rule is enforced in code, not just in prose: a comparison
> block marks each delta `unresolved` when the per-seed deltas disagree, and
> `gate_met` is true only if every sign-consistent variable improved. Two tests
> pin exactly the case this run hit — an arm that looks better on average while
> every per-seed delta flips sign must be reported unresolved, not as a win.
>
> `scientific_claim: false`; test split not read; three seeds is a noise-floor
> check, not a significance test; ten January days per year and 200 updates at
> `dim=32, depth=2` is a bounded CPU comparison, not a converged run. Full record:
> `docs/R7_COREASONING_FAIR_BUDGET.md`; tests
> `tests/test_r7_coreasoning_compare.py` (12 offline). Verification: `pytest -q`
> 856 passed / 3 skipped / 0 failed with GPU, 850 / 9 / 0 with
> `CUDA_VISIBLE_DEVICES=""`; `python tools/check_conventions.py` 34 blocking
> rules, 0 violations.

Closure statement (only after the comment above is posted):

> Closing as answered-with-a-negative-result: the four listed tasks exist (feedback
> path, forecast solver/refiner, complete P0,Y0..PK,YK trace, process-proxy
> supervision hooks) and all four ablations (no process state, no forecast
> feedback, no recursion, plus generic) were run under matched budget and seeds.
> The research gate — "process-aware co-reasoning must beat generic R7.2 under
> comparable budget" — is **not supported** by this comparison: the only
> sign-consistent effects are `t2m` worsening for both process arms and `t500`
> improving for the no-feedback arm. Reported as a negative result rather than
> tuned until it turns positive.

---

## #7 — [R7.4] Consistency and marginal-gain guided adaptive halting

Draft comment (paste verbatim, minus this line):

> **The gate is audited at `0f5df17` and found NOT satisfiable as stated. No
> controller threshold was changed and no saving is claimed.**
>
> The gate says adaptive "should approach fixed-Kmax accuracy with meaningfully
> lower average reasoning depth". That assumes fixed-Kmax is the accuracy worth
> approaching. Before re-tuning any controller, I tested the assumption against a
> measured error-versus-depth curve, reusing the K=0/1/3 evaluation from the #6
> run (real regional ERA5, three seeds, validation split only).
>
> **Finding 1 — deeper is not better.** Equal-channel normalized RMSE (the
> objective the training loss minimizes):
>
> | arm | K=0 | K=1 | K=3 | K=3 vs K=0 |
> | --- | --- | --- | --- | --- |
> | generic | 0.31079 | 0.30991 | 0.32117 | +3.3 % |
> | process_feedback | 0.31519 | 0.31484 | 0.31976 | +1.5 % |
> | process_no_feedback | 0.31594 | 0.31589 | 0.32166 | +1.8 % |
>
> Counted per seed, `process_no_feedback` is worse at K=3 than K=0 in **all three
> seeds**; the others in two of three. The fixed-Kmax reference is therefore
> **not** the best available accuracy on this data. Per-variable, the trend is
> genuinely split: 5 variables consistently improve with depth, 5 consistently
> worsen, 7 are unresolved.
>
> **Finding 2 — no shallower depth passes the per-variable tolerance.** Against
> the K=3 reference, mirroring `select_validation_policy` (every variable must
> meet the tolerance separately, no aggregation):
>
> | tolerance | K=0 | K=1 | any depth feasible | blocker |
> | --- | --- | --- | --- | --- |
> | 0 % | 9/17 | 9/17 | no | `z250` (1.190×) |
> | 1 % | 11/17 | 12/17 | no | `z250` (1.190×) |
> | 5 % | 15/17 | 16/17 | no | `z250` (1.190×) |
> | 10 % | 16/17 | 16/17 | no | `z250` (1.190×) |
>
> This is the mechanism behind the previously reported "strict policy fell back to
> full depth": `z250` sits 14–19 % above the reference at every shallower depth,
> so no tolerance below 19 % admits a reduction, and at 10 % one variable vetoes it.
>
> **Combined:** `gate_satisfiable: false`, for two independent reasons, either of
> which suffices.
>
> **A caveat I am recording rather than hiding:** `z250` is the blocker, but its
> own numbers are seed-unstable (per-seed K=0 RMSEs 257.7 / 235.7 / 188.2; K=3−K=0
> deltas −48.0 / +6.6 / −67.3, not same-sign). So the reliable statement is that
> the per-variable tolerance rule is unsatisfiable here — the specific blocking
> variable is not itself a robust finding. Both are stated because either alone
> would mislead.
>
> **What this does not establish:** that adaptive halting is impossible. A
> controller that reduces depth per-variable wherever depth is harmful could
> still help, but that is a different objective from the one the gate states and
> would need the gate rewritten before it can be tested. I did not relax the
> tolerance and did not drop the blocking variable to manufacture a pass.
>
> `scientific_claim: false`; test split not read; three seeds is a noise-floor
> check, not a significance test; substrate is a bounded CPU comparison (200
> updates, dim=32, depth=2, ten January days per year, +6 h lead). Full record:
> `docs/R7_HALTING_GATE_AUDIT.md`; tests
> `tests/test_r7_halting_gate_audit.py` (9 offline). Verification: `pytest -q`
> 859 passed / 9 skipped / 0 failed with `CUDA_VISIBLE_DEVICES=""`;
> `python tools/check_conventions.py` 34 blocking rules, 0 violations.

Closure statement (only after the comment above is posted):

> Closing as answered-with-a-negative-result: the listed tasks exist (halt/gain
> head, CONTINUE/STOP trained from observed marginal gain, per-sample halting with
> Kmax fallback, compute-penalty warm-up; the consistency auxiliary loss is
> optional and not implemented), and the research gate was audited against real
> data. The gate is **not supported**: the fixed-Kmax reference is not the
> accuracy ceiling, and the per-variable tolerance admits no depth reduction.
> Reaching an adaptive benefit would require restating the objective, not
> re-tuning thresholds inside this one.

---

## #8 — [R7.5] Build 6–72h rollout and East-Asia evaluation protocol

Draft comment (paste verbatim, minus this line):

> **The acceptance sentence is satisfied at `b7f45ab`: evaluation scripts now
> produce paper-ready metric tables without changing training code.** Three of the
> issue's scope bullets are explicitly **not** done (listed below).
>
> The rollout machinery, RMSE accumulator and ACC accumulator already existed and
> were covered by synthetic tests. What was missing was a script that drives
> already-trained checkpoints through a 6/12/24/48/72 h autoregressive rollout on
> the held-out year and emits comparable tables.
>
> **`scripts/rollout_r7_metric_tables.py`** evaluates only:
>
> - **No training code touched** — enforced by an AST-level test asserting no
>   optimizer, `backward`, `step`, `run_local_updates` or
>   `calibrate_controller_step` is referenced at all; the check runs on parsed
>   code, not on prose.
> - **Model generations cannot be mixed** — a checkpoint whose recorded
>   `model_code_sha256` differs from the live `model/` digest aborts, with an
>   instruction to replay via the archived `code.zip` instead of bypassing the
>   identity check.
> - **Datasets cannot be mixed** — one identical `data_identity`, matching channel
>   order and units, or the run aborts.
> - **Held-out split enforced**, **same-data persistence baseline included** by
>   default on the same manifest/leads/case cap, **no cross-variable average**
>   (K, Pa, m/s are not summable), **undefined ACC written as `undefined`, never 0**.
>
> **Real measurement** — held-out 2020 block of the #6 store (native 0.25°, 65x65,
> 17 channels, real ERA5; `manifest_sha256 1a8636ed752b0438…`,
> `data_identity 9c714f6189bb6b4b…`, train-only climatology over 2018), 4
> initializations per model, 200-update checkpoints, K=3:
>
> RMSE (extract, physical units):
>
> | model | variable | 6h | 12h | 24h | 48h | 72h |
> | --- | --- | --- | --- | --- | --- | --- |
> | generic | t2m (K) | 3.72081 | 4.26451 | 4.42684 | 5.56105 | 5.61225 |
> | generic | t500 (K) | 1.37051 | 2.27627 | 3.34248 | 3.58531 | 3.27549 |
> | process_no_feedback | t2m (K) | 4.09721 | 4.65464 | 3.96513 | 5.39556 | 6.50987 |
> | process_no_feedback | t500 (K) | 1.15669 | 1.84994 | 2.63029 | 3.80466 | 4.10074 |
> | persistence | t2m (K) | 4.35171 | 5.07542 | 2.17555 | 2.90139 | 3.86503 |
>
> ACC: generic t2m 0.576725 / 0.466857 / 0.311284 / **−0.016203** / 0.336861 at
> 6/12/24/48/72 h (negative at 48 h means no better than train-only climatology).
>
> **What these numbers do not say, stated plainly:**
>
> 1. **Persistence looks deceptively strong at 24 h** (2.18 K, better than both
>    models) — this is **not** a bug and **not** a finding. With 4 initializations
>    the 6/12/24 h targets land on different valid times of day (12:00, 18:00,
>    06:00+1d), so each lead scores a different subset of the diurnal cycle. The
>    case set is too small and too unevenly sampled for cross-lead ranking.
> 2. **`process_no_feedback` beats generic on t500 at every lead but loses on t2m
>    except at 24 h** — consistent with the #6 result that the process arms trade
>    variables rather than dominate.
> 3. **Four initializations is smoke scale**, not journal evidence.
>
> **NOT done** (scope bullets, not acceptance): accuracy–compute Pareto, average
> reasoning depth versus accuracy, weather-complexity vs reasoning-depth
> diagnostics, and extreme-event metrics. The script records `reasoning_steps` but
> does not sweep it. I am listing them rather than quietly dropping them.
>
> `scientific_claim: false`. Verification: `pytest -q` 869 passed / 9 skipped / 0
> failed with `CUDA_VISIBLE_DEVICES=""`; `python tools/check_conventions.py` 34
> blocking rules, 0 violations. Full record: `docs/R7_ROLLOUT_TABLES.md`.

Closure statement (only after the comment above is posted):

> Closing on the acceptance criterion: evaluation scripts produce paper-ready
> metric tables without changing training code, verified on real regional ERA5
> over 6/12/24/48/72 h with RMSE, ACC, units, provenance and a same-data
> persistence baseline. The un-done scope bullets (Pareto, complexity diagnostics,
> extreme events) are recorded in the linked document rather than being represented
> as complete.
