# R7 conversation-driven task queue

#64 B2 and B3 both completed on 2026-09-27; both are **negative/mixed** and are
recorded that way (`docs/R7_B2_MULTISEED.md`, `docs/R7_B3_SCALE.md`). B2's
pre-registered skill gate is NOT met and no criterion was relaxed to change
that; B3's apparent capacity win is not seed-consistent, so it is reported as
unresolved rather than as a win. Cumulative GPU budget after B3: **2.269 of
4.0 GPU-h**.

Updated after verified issue #13 regional acquisition at `f37ceba`, the #20
multi-seed comparison at `7dfbea6`, the #5 budget-parity audit at `bf57fd4`, the #6 fair-budget comparison at `1bc1eef`, the #7 halting-gate audit at `0f5df17`, the #8 rollout tables at `b7f45ab`, the #9 access audit at `6dc388d`, and the #1 cumulative gate status at `ffe501e`;
on 2026-09-25 the eight verdicts were finalized, the closing commit carrying
`Closes #N` for all eight was pushed on `r7/weather-reasoning` (`e5be0c0`), and
after the hook policy change (decision 0003) the fast-forward to `main`
**landed**: `92a8c4d..2d7da05` at 2026-09-25T10:11Z — **all eight issues are
CLOSED on GitHub** (#13/#20/#5/#6/#7/#8/#9/#1, closed 10:11:00–10:11:03Z) and
**PR #12 shows merged** (fast-forward, no merge commit). Merges remain gated
by `guard_destructive_git` (user authorization required); non-force pushes to
main are allowed by decision 0003. No force, no release, no paid GPU, no
uncontrolled data mirror. Experimental workflows stay tag-gated and skipped
by design.

## Accepted work in this iteration

Corrections recorded under #62 (2026-09-26): the #8 close sentence in the
historical table below was stale (the issue closed with the 2026-09-25 main
fast-forward) and now reads CLOSED; wording errors in
R7_ROLLOUT_TABLES.md, R7_COREASONING_FAIR_BUDGET.md and R7_GPU_BRINGUP.md
are fixed in place with per-file correction logs; no measured value changed.


| Issue | State | Verification |
| --- | --- | --- |
| #53 variable-wise error/update geometry | DONE | actual35891600411; CI35891600361 |
| #54 fixed200->800 original-checkpoint control | DONE | actual35892145587; CI35892145656 |
| #55 same-budget continuous250day source | DONE | actual35893397412; test-only repairCI35894113212 |
| #56 pinned offline continuous-data800update control | DONE | actual35895446235; CI35895446093 |
| #57 optional spatial solver plus400update matched ablation | DONE (engineering/experiment) | actual35896740916; CI35896740912; defaultFalse retained |
| #58 explicit case selection/common-case retrospective audit | DONE | actual35898091953; CI35898091928 |
| #60 seed-identity comparator, fail-closed gates, historical re-aggregation (S0-A) | DONE (acceptance chain) | local 909 passed/3 skipped; conventions 34/0; audit 27/27 records, max mean diff 0.0, 0/34 directions changed; see R7_SEED_IDENTITY_COMPARATOR.md |
| #61 DDP verified reasoning depth, resume contract, padded samplers (S0-B) | DONE (CPU+2×3090; real-region throughput BLOCKED on #63 D1) | local 939 passed/3 skipped; CUDA: eval K observed 4 with steps=10, resume 111/111 weight hashes identical, streamed+DDP refused; see R7_DDP_K_CONTRACT.md |
| #62 GPU audit pack, doc corrections, climatology skill baseline (S0-C) | DONE (publish pack = user decision) | local tests incl. 9 new (5 ACC skill + 4 audit pack); real pack 562 files/0 credentials; tables rebuilt from pack only (104 rows, 30 oom/skip flagged); retained_truncated memory cells added; see R7_GPU_AUDIT_PACK.md |
| #63 ERA5 v2 frozen read plan, cost table, offline replay, eps floor audit (S1) | DONE (engineering segment): D1 acquired via Earthmover `spatial` (decision 0004) — 120/120 stamps, measured 3.31 GiB network / 20.0 min (budget 3.5 GiB / 30 min), source SHA256 `2ec0d444…3020e`; store published (BUILD_COMPLETE, windows 94/10/10 via decision-0005 time-range splits); offline replay all checks pass. Read-plan/cost-table/floor-audit parts unchanged. D2 + 2021 test-seal audit still need separate authorization | local 971 passed/9 skipped (6 CUDA-only cases skip without GPU); protocol sha 3bf7ab2cce103545 unchanged; see R7_D1_ACQUISITION.md |
| #64 B0 train-only learnability on real D1 windows (S2) | DONE (B0 gate): 4 minimal arms (native_window/unet/afno_small/generic) x 1/8/32 fixed windows — 12/12 show interpretable loss drop (1-window x2.8-2.9, 32-window x1.1-1.2); 17/17 channels improve; gradients reach encoder+solver in all 12; history/lead/denormalization checks clean. Protocol frozen c008b59c…, seed 41, 200 updates, 0.156 GPU-hours. Learnability only — not generalization; B1 unlocked | see R7_B0_LEARNABILITY.md |
| #64 B1 strong-baseline name-vs-implementation audit + prereg draft (S2) | DONE (audit segment; **no training launched**, 0.0 GPU-hours): all 6 baselines checked against their names — the 4 neural families are genuinely what they claim (Swin-like finite-boundary mask proven by single-token perturbation staying inside its window; AFNO proven to have a real Fourier mixer + softshrink, not a plain FFT layer). Correct factory entry documented: `make_model(kind, {...,'architecture':arch})` is 3-field; `make_model('unet',…)` was never a valid call. 1–5M candidates measured at 2.79–2.83M (1.49% spread). **Two blocking findings, neither fixed (awaiting authorization D-1..D-4)**: (1) `fit_training_climatology` filters by YEAR, so on the single-year D1 store it degrades to "all 120 steps" and leaks the val/test days into the baseline itself (+6.9% flattered on t2m, +2.9% mslp, +6.0% v850); (2) the same year-based check in `validate_record`/`ZarrRolloutDataset` makes D1 val/test unreadable (0/10 and 0/10), so B1 cannot start until fixed | see R7_B1_BASELINE_AUDIT.md |
| #64 D-2 split-time-range reader fix (S2) | DONE at `140c0ea`: ownership moved into one helper `r7_store.split_time_labels` (None ⇒ year mode unchanged, else per-step labels from the store's own `split_time_ranges`); `parse_split_time_ranges` lifted to `preprocess/contracts` so writer and readers share one implementation. On the real D1 store `validate_record` val/test **0/10 → 10/10** (train 94/94 unchanged), `ZarrRolloutDataset` val/test now 10 windows at lead 6h, train-only climatology **120 → 96** steps (24/bucket). Year mode proven **bit-for-bit identical** against the HEAD modules on three real multi-year stores (same pass counts, window counts, and fp64 climatology means by SHA256). `verify_r7_d1_store.py` now calls the readers instead of re-deriving ownership, so the replay covers this defect class. ADR 0006 | 990 passed/3 skipped; 34 blocking rules 0 violations; see R7_B1_D1_COMPARISON.md |
| #64 D-3/D-4 climatology-as-baseline + TRM difference table (S2) | DONE at `f0bd647`: `evaluate_local` scores `rmse_climatology`/`mse_skill` on exactly its scored cases (self-comparison gives skill exactly 0); `rollout_r7_metric_tables.py` adds a third wide table and refuses a skill cell whose case count disagrees with `n_evaluated`. D-4 documents the per-item difference vs TRM at upstream `c01103738605` (MIT): ours backprops the whole chain where upstream runs `H_cycles-1` in `no_grad` and hard-detaches the carry, ours adds per-draft deep supervision, and ours has **no halt head in the model** (the adaptive stop lives in the separate `r7_halting` controller) — labels stay `generic`, never `TRM` | see R7_TRM_DIFFERENCE_TABLE.md |
| #64 D-1 data-scope decision (S2) | DONE at `a295334`: **B1 stays on D1, no D2 expansion**. Cost table from measured Earthmover unit cost (8.78 s / 25.15 MiB per timestep): the disk product is never binding (a full year is 0.23 GiB) while network/decoded bytes are — the smallest seasonal extension (60 d) already costs 35.1 min, past the frozen 30 min/experiment cap, and a full year decodes 110 GiB against the 64 GiB cap. Abandoned options and their cost recorded; **no seasonal claim is testable** on a January-only segment | ADR 0007 |
| #64 B2 controlled multiseed confirmatory comparison (S2) | **DONE** at `94828da` (**negative result, reported as pre-registered**): #64 B2 needs ≥3 pre-declared seeds, a frozen protocol, a per-variable/per-lead skill check and free 6/12/24/48/72 h rollouts. The frozen D1 segment **cannot host a 72 h rollout at all** — its held-out blocks are 12 stamps wide and a 72 h rollout needs 14 contiguous ones, so the window count is 0, not merely small. Per decision 0008 this acquired a 36-day segment (144 stamps, 22.6 min, 10.6/64 GiB decoded) whose **train split is byte-identical to D1** (same state SHA, same normalization arrays, same 94 windows) while val grows to **15 windows at 72 h**. Three seeds (41/42/43) ran under one protocol digest `244af00b0834948091506f20f6d3d26a093cff8b142a5f10b084e4b95c6643e2`, derived independently by each process, 800 updates × 5 arms × 3 seeds = **1.157 GPU-h**. The digest check fired on a post-hoc edit to the frozen criteria made during the session; the edit was reverted rather than the digest accepted. **Gate NOT met**: both required arms (`native_window`, `generic`) lose to the zero-parameter train-only climatology on t2m at 6 h (generic also at 12 h); neither arm loses to persistence. **Seed spread grows from ~3% at 6 h to 20–23% median at 72 h (up to 65% in one cell)**, so single-seed long-lead rankings are not trustworthy — the empirical basis for the #60 comparator's same-sign requirement | 1046 passed/3 skipped; CI run 36276587108 success; see R7_B2_MULTISEED.md |
| #64 B3 capacity scale-up to 18M (S2) | **DONE** at `eba90e0` (mixed result): three arms at 18.20M/18.12M/18.12M (**6.47–6.49×** B2, hard 15–20M gate, spread 0.459%, 30M never approached), same store/schedule/800-update budget/checkpoint rule/case sets as B2, **capacity the only declared change**; two seeds, one independent process per 3090 (no DDP). 0.547 GPU-h, 968 MiB peak of 24 GiB, 6/6 arms decayed monotonically (ratio 0.519–0.599), no early stop — so "18M was unstable" is not an available excuse. **The capacity hypothesis does not survive seed resolution**: the one cell that looked like a win (generic t2m 6 h, mean 2.7787 vs climatology 2.8029) is **not seed-consistent** (seed 41 wins by −0.060, seed 42 loses by +0.012), and its margin is smaller than the seed noise in that cell; t2m 24–72 h **worsens** with capacity (41 of 85 cells improve, i.e. about half); all arms still beat persistence in roughly half the long-lead cells but climatology in **0/17** at 48 h and 72 h. No "ours wins" claim is made anywhere | 1058 passed/3 skipped; CI run 36278531741 success; see R7_B3_SCALE.md |
| #64 B1 controlled six-baseline comparison on real D1 (S2) | **DONE** at `fa8dd68` (three runs; main table protocol digest `1b1c176702f23da4edffd8feeea9d2d7` frozen before the first step; 1,476 s ≈ 0.41 GPU-h): 4 neural arms inside ±5% of 2.8M (measured 2,789,903–2,831,433, spread **1.483%**) + persistence and climatology at **0** params. Four required tables emitted (parameters / FLOPs / wall time / case counts) plus a joined RMSE+skill table (408 rows). Test **sealed** in all three runs — every evaluation was on `val`, `test_read: false`. Independent recomputation of persistence t2m 6h from the store matches the harness bit-for-bit (5.0978 K). **Main negative result: in t2m all four neural arms lose to the zero-parameter train-only climatology at every lead**, and a 800-update control (17–18 epochs) shows t2m 6h/12h improving 13.1–30.1% while t2m 24h/48h *worsen* — so the 200-update long-lead cells were flattered by the zero-initialised residual head sitting near persistence, and "simply under-trained" is ruled out for t2m. Repeat runs reproduced rankings (0/68 flips) but weights are **not** bitwise identical (GPU float nondeterminism); evidence in `outputs/r7_b1_d1_run2/` and `outputs/r7_b1_budget_control/` | 999 passed/3 skipped; CI run 36260593536 success; see R7_B1_D1_COMPARISON.md |
| #65 (a) pre-diagnostic, four items (S3) | **DONE** at `f292b1a` (CPU, 0 GPU, read-only checkpoint `sha256 1d39da75…0627e7` verified before and after): (1) the `eps=1e-6` floor is active on exactly **2 of 8** proxies — moisture advection (raw std 7.59e-9) and convergence (1.18e-8) — leaving normalized std **0.0076 / 0.0118**, i.e. 132×/85× signal lost; but the proxies are **supervision targets only** (`forecast_inputs` never forwards `process_targets`), so the defect **cannot** make any input channel constant — the campaign's hypothesis is corrected on that point; (2) at the auxiliary weight B2/B3 actually ran (**0.0**) the process readout receives **zero** gradient, and the arm gap is smaller than the generic seed spread in **80/85** cells; (3) K0–K4 trace: K=1 gives the entire gain, K=3/K=4 start worsening, adjacent-correction cosine rises 0.864→0.939→0.982; (4) **attribution: B2/B3's "process vs generic" is not evidence about process structure** — once the single non-aligned projection is forced equal the two arms are the same function bit-for-bit (locked by `tests/test_r7_process_arm_equivalence.py`) | 1083 passed/3 skipped; 34 blocking rules 0 violations; see R7_65_PREDIAGNOSTIC.md |
| #65 (b) C1 process supervision at engaged auxiliary weights (S3) | **DONE** at `f292b1a` (mixed, leans negative): 4 arms × 3 seeds × 800 updates, **12/12 no early stop**, 3,886 s ≈ **1.08 GPU-h**, protocol digest `d0a59c9b21ffe1c3…` frozen before step 1. Compute alignment is **proven, not assumed**: +0.0206% parameters, FLOPs ratio **1.000002**, identical case counts (15/19/23/25/26). No arm improves the primary variable with a consistent sign; **aux=0.1 worsens t2m at the trained lead in all three seeds** (+0.05…+0.35 K), and its training loss improves fastest while validation worsens — consistent with fitting two floor-degraded moisture labels | 1138 passed/3 skipped; see R7_65_C1_PROCESS_SUPERVISION.md |
| #65 (c) C2 feedback routing + C3 depth/compute (S3) | **DONE** at `c0dcf85` (C2 negative, C3 mixed): C2 (4 arms × 3 seeds, 3,926 s ≈ 1.09 GPU-h) — turning on the solver's spatial feedback worsens **28 of 85** cells (t2m@48h worse in all 3 seeds by +1.12…+1.22 K); the reasoner-feedback-off arm is the only mildly positive change (14 improved / 8 worsened, t2m@12h consistent −0.08…−0.16 K) but it also costs **10.7% fewer forward FLOPs**, so it is an efficiency observation; **current defaults are retained**. C3 — the solver-feedback switch is FLOP-identical (elementwise addition is uncounted), test-time deepening helps only at 6h and reverses at 24/72h so **free scaling is not supported**, and the independently trained K=1 arm costs **27.9%** fewer FLOPs with a mixed result (18 improved / 14 worsened / 53 unresolved) | 1138 passed/3 skipped; see R7_65_C2_C3_FEEDBACK_AND_DEPTH.md |
| #65 (d) a harness defect found and fixed by C3 | **DONE** at `c0dcf85`: the first C3 run's "independently trained K=1" arm declared `default_reasoning_steps=1` but was handed the phase-wide `steps=4`, so it trained at K=4 — FLOPs matched the K=4 arm bit-for-bit and the accuracy difference appeared only in the 5th significant digit. Invalid run **retained** at `outputs/r7_65_c3_invalid_k1_config/`. Each arm now declares its own depth in the frozen protocol, the harness asserts `contract["steps"]` after training and fails closed, and `tests/test_r7_65_ablation_harness.py` (9 cases) includes a counterproof that the shallow arm must be strictly cheaper | 1138 passed/3 skipped |
| #66 gate self-check (S4) | **BLOCKED — gate not met** (0 new GPU; judged on C3's existing artifacts): G1 (fixed K=1/2/4 frontier exists) and G2 (an independently trained strong K1 exists) are **satisfied**, but G3 is not — the optimal-K ordering is seed-consistent **only at 6h**, and there the best K **is** the most expensive K=4 (gain **0.00%**); at 12/24/48/72h the ordering is inconsistent across seeds. The fixed trained K=1 already costs 27.9% less, so any adaptive policy's best case is matching it minus controller overhead. An oracle (future-truth) shows per-sample optimal K does vary (+6.9…+13.0%), reported as the **non-deployable** upper bound it is. Per #66's own failure exit, no controller is trained and no complexity is added. Missing conditions M1–M4 stated explicitly | see R7_66_GATE_AUDIT.md |
| #67 (a) publication protocol, frozen before reading test (S5) | **DONE** at `6259c78`: task definition, main-table membership, metric conventions (incl. the ACC-climatology/not-recentred caveat and "positive ACC ≠ positive MSE skill"), ≥3 fixed seeds reported separately, day-block paired resampling (week blocks are impossible inside 7 val days), four-season grouping recorded as **impossible** on this data, train-only extreme thresholds, no precipitation skill claimed, external pretrained models restricted to a separate reference table, and the deliverable/sealing checklist | frozen at `6259c78` **before** test was read |
| #67 (b) sealed test report (S5) | **DONE** at `f653897`: the report now exists on a **January re-cut** store (new `data_identity` `4055fc7e30fd4677ff30a58137466ff5ef9e8eda0da7cc6946205c26fb031bca`). Re-cut: train `[01-01,01-25)` **byte-identical** to the frozen chain (train manifest hashes equal, 94 windows), val `[01-25,01-27)` and test `[01-27,02-01)`, so both held-out blocks need only the `(1,00/06/12/18)` buckets the train-only climatology actually has. Source is the already-frozen `source.nc` (sha256 `2ba504fe…`), **no new download**; store 0.030 GiB. Because `data_identity` covers `split_time_ranges`, all **15 arm-seed cells were retrained** (measured 1.08 GPU-h) — this is a **new experiment** and the report does not claim to confirm the frozen chain. Test now carries all five leads: **18/17/15/11/7** windows at 6/12/24/48/72h. **Result: every neural arm loses to the zero-parameter train-only climatology on all 85 cells (better 10 / worse 75 / unresolved 0)**; at 6h t2m the best arm is still **9.09×** climatology and mslp **244×**. `process` differs from `generic` with a seed-consistent sign only at 6h (+0.37…+0.85 K, generic better); at 12/24/48/72h the sign flips across seeds | `read_count: 1`; 75 evaluation dirs; 1151 passed/3 skipped; 34 blocking rules 0 violations; see R7_67_SEALED_TEST_REPORT.md |
| #67 (c) the protocol's two unexecuted analyses (S5) | **DONE** at `f653897`, both run on the sealed test report: (1) **day-block paired resampling** (2000 resamples; the *day* is the resampling unit because same-day initializations are not independent) per arm-pair × seed × lead — 60 cells; `process_vs_generic` has **6/15** intervals containing zero and the direction reverses across seeds at every lead past 6h; `afno_small_vs_native_window` is **9/15**. (2) **full/interior/boundary stratification** for all 75 arm-seed-lead cells: at the trained lead the edge is **6.6%** worse than full, while at 48/72h it is ~8% *better*, and `edge_1` vs `edge_2` disagree in direction at 24h — all three reported rather than the convenient one. Seed variation and weather-sampling variation are reported separately, never pooled into one N | scripts `analyze_r7_67_paired_blocks.py` (12 tests) + `analyze_r7_67_sealed_report.py`; see R7_67_SEALED_TEST_REPORT.md §3–§4 |
| #64 remainder: 1→2→4 lead-time curriculum (S2) | **DONE** at `208943d` (**negative**): 2 arms × 3 seeds × 800 updates, **6/6 no early stop**, 1,411 s ≈ **0.39 GPU-h**, protocol digest `7bebc8b6c2269b6f…`. Parameters and FLOPs are **bit-identical** between arms and the stage budget sums exactly to the control's, so the only declared difference is the target distribution. The curriculum worsens t2m with a consistent sign at **all three short leads** (+1.58…+1.61 K at the trained 6h lead, ≈+56%) and the 85-cell table is **5 improved / 58 worsened / 22 unresolved**; the single consistent gain is at 48h (−1.55…−2.55 K), making the trade real but **net negative** under the pre-registered rule. Mechanism is structural: it gives up half its +6h updates, so a retest must **add** budget rather than reallocate. This axis is prediction lead time, kept distinct from the internal reasoning depth K | 1138 passed/3 skipped; see R7_64_CURRICULUM.md |
| #68 transport-layer boundary (parallel) | **DONE** at `f5ce02f`: `data/download/http_pinned.py` pins the connection to the addresses that were validated (one resolution shared by check and connect), validates scheme/host/**port** and the resolved addresses per redirect hop, declares the proxy policy instead of inheriting `*_proxy`, and keeps SNI plus certificate verification on the default verifying context. `http_public.py` now re-exports it, so the existing downloaders gain the stronger behaviour without changing their imports (a test asserts both modules expose the same objects and that neither downloader calls `urlopen` directly). 37 offline cases in `tests/test_http_pinned.py`, whose headline case scripts a name that answers publicly first and privately second — a pre-resolution check passes that, the pinned connection refuses it and **no socket is ever created**. The triage doc states the boundary precisely: this is an address constraint, **not** an allowlist, not content inspection, and not immunity from an attacker-controlled public origin | 1138 passed/3 skipped; 34 blocking rules 0 violations; see R7_SECURITY_SCAN_TRIAGE.md |

**2026-09-27 campaign closure.** All six campaign issues carry a terminal verdict and were
closed through `Closes #N` on `main` (fast-forward `3bb9001..90307e7`, the only git path that
works here): **#64, #65, #68 DONE; #66, #67 BLOCKED with their missing conditions written down;
#59 closed as the epic now that every child is terminal.** No issue was closed on the strength
of a cancelled or queued run, and no closure implies the process-recursion hypothesis was
validated — the campaign's scientific outcome is negative-to-mixed. CI on the closing commit:
**run 36309161879 success**; local verification at the same tree **1138 passed / 3 skipped**,
34 blocking convention rules **0 violations**.

**2026-09-27 campaign artifact accounting (new-artifact cap 16 GiB).** C1+C2+C3 wrote
5.758 GiB unpruned, taking the cumulative footprint to **16.03 GiB — over cap**. Resolved by
narrowing storage, not evidence: `tools/prune_r7_run_checkpoints.py` removes only `update_*.pt`
files that are not each arm's recorded `selected_checkpoint`, keeps every evaluation directory /
CSV / protocol / result JSON, is dry-run by default, and writes a receipt with path+bytes+sha256
per deletion (9 tests in `tests/test_prune_r7_run_checkpoints.py`). Applied to C1/C2/C3
(**3,731.4 MiB reclaimed, 30 endpoints kept**) and to the curriculum run (836.5 MiB, 12 kept).
Verified after pruning: all 6 curriculum `final_checkpoint` paths still exist and appear in the
receipt's `kept` list. Campaign artifacts now total **2.273 GiB**, i.e. cumulative **12.543 /
16 GiB**. The invalid C3 run is deliberately **not** pruned — the tool's fail-closed check
refuses it, and its checkpoints are the physical evidence for the harness defect above.
GPU-hours this campaign (all measured from each run's own `training_report.json`, not estimated):
C1 1.08 + C2 1.09 + C3 0.50 + curriculum 0.39 + the #67 re-cut retrain 1.08 ≈ **4.14 GPU-h**, all
on the local 2×3090; no paid resource was used.

**Final CI binding.** Every `ci.yml` run below was pulled from the workflow's own run list, and
each is `completed / success`:

| run id | SHA | what that SHA is |
| --- | --- | --- |
| `36292366591` | `f4a552b` | campaign start |
| `36308047201` | `208943d` | curriculum + C2/C3 docs |
| `36308658711` | `0484927` | ADR 0009 |
| `36308913373` | `9720c27` | artifact accounting |
| `36309161879` | `90307e7` | issue-closing commit |
| `36309375783` | `93832fc` | closure record |
| `36310343924` | `ecdae0d` | user-authored budget-caps commit |
| `36312388200` | `6b8578c` | **covers code-bearing `f653897`** — the sealed-test report and its analysis scripts were pushed together with `6b8578c`, and `f653897` is its ancestor |
| `36312611169` | `77f2718` | change manifest + epoch curves |
| `36313302904` | `1548c3c` | SHA-range/run-id record |
| `36313635208` | `5383914` | CI run table + measured GPU-hours |
| `36313941380` | `8822165` | per-tip CI table |

**Why this list terminates instead of chasing its own tip.** The last commit that changes any
**non-doc** file is `f653897`; every commit after it touches only `docs/**`. A documentation-only
commit cannot alter behaviour, so the CI conclusions above cover the campaign's code regardless of
how many further doc commits follow, and no later tip needs its own row for the code claim to
hold. Readers who want the newest documentation commit's run can read it off the `ci.yml` run
list directly; what is guaranteed here is the part that carries scientific weight.

Job detail for `36313302904`: job `108603223124` (`pytest`), every step green — *Check repository
conventions*, *Compile active modules and check whitespace*, *Run unit, integration and
installed-wheel tests*. On every SHA the 17 experiment workflows report `skipped`, which is their
tag-gated design rather than a failure. Local verification on the same tree: **1151 passed /
3 skipped**, 34 blocking convention rules **0 violations**.




Latest full code CI: **578passed,3old-fixture-skipped,2existingLightningwarnings**
in49.25s, job107306975262, plus compile/whitespace/installed-wheel checks.
No new tests were skipped. No new implementation is waiting for verification.

Detailed evidence, limits and mixed results:
[R7_CPU_REFINEMENT_RESULTS.md](R7_CPU_REFINEMENT_RESULTS.md).
Per-archive hashes: [R7_CPU_ITERATION_ARTIFACTS.json](R7_CPU_ITERATION_ARTIFACTS.json).

## GPU bring-up on 2×3090 (engineering, not science)

Measured at `50954c94eb0aa15fa61cb19440543b40c6c38326`. Full record:
[R7_GPU_BRINGUP.md](R7_GPU_BRINGUP.md). Every artifact `scientific_claim: false`.

| Item | Result |
| --- | --- |
| Single-GPU sweep | 2 models × K=1/2/4/8 × {full BPTT, streamed} × ckpt on/off, 32/32 OK at 1.25M and 18.07M params, one process per cell |
| Streamed vs full BPTT | streamed **flat on both metrics** across K=1…8 (46.00 MiB reserved at 1.25M; 380.00 MiB at 18.07M); full BPTT grows (394 → 426 MiB reserved at 18.07M) |
| Measurement artefact | whole-sweep-in-one-process inflates `reserved` via allocator carryover (process K=1: 428 vs true 396 MiB); per-process measurement required |
| Checkpointing cost | 21 % (1.25M) to 70 % (18.07M) more step time to bound the peak |
| GPU checkpoint resume | bitwise identical weights, optimizer state and loss sequence |
| DDP smoke (first in repo) | 2 ranks, loss vs single-GPU reference max delta 3.1e-06, sampler covers dataset once, 1 checkpoint, resume bitwise identical |
| DDP negative result | **7 % slower** than single-GPU at this scale on SYS/PCIe (no NVLink); does not add per-model memory |
| OOM boundary | 178.2M params / batch 48 / 128×128 / K=8; only streamed+checkpointing survived in every degradation variant |
| Real 11-channel ERA5 | **not on this disk** (`data/*` = `.gitkeep` only) — memory numbers use synthetic shapes |

Engineering child items for #20 are covered. The **multi-seed optimization
comparison is not done**, and #5/#6/#7/#8 stay open — engineering passing is
not a scientific result.

## Prior work is not pending

Issues47–52 had already completed: source fill-value/budget correctness,
four-season source/replay, multiseed study, compact strong baselines, original
checkpoint/cache restoration and delayed-benefit diagnostics. Do not recreate
them from old chat summaries. Earlier forecast, controller, streamed backward,
strict data contracts, chronological normalization, rollout, ACC, profiling,
policy selection and boundary scoring remain implemented.

## Available real input

Eleven physical ERA5 channels on one12x12 native0.25-degree tile.
3000six-hour timestamps:250days each in2018/2019/2020, not3full years.
998one-step windows per chronological split, eight real input-process proxies.
NetCDF19,052,672bytes, SHA256
`0609fa38c1d88b82b985a15f93dd502c7eb7031f5d2b7452bbe971bf936dd9c1`.
Decoded source read charge180,142,968bytes remains under192MiB; this is not HTTP
traffic/RAM. Local replay needs no new cloud-source access.

## Remaining research gates

| Parent/task | State | Evidence/next useful action |
| --- | --- | --- |
| #13 representative data | DONE verdict final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Bounded regional acquisition completed at `f37ceba`: 3 years x 4 seasons, 65x65 East-Asia, 48 exact 6-hourly timestamps, 9 variables, source SHA256 `d3fa1fba6da46ed59a535ce27f7501813afc2c93cb8b745c90e349454a40960a`. Converted through the audited publication path to a 17-channel store (8 windows per split, `BUILD_COMPLETE.json`, train-only statistics). Coverage is four 24-hour blocks per year, NOT continuous full-year; see [R7_REGIONAL_ACQUISITION.md](R7_REGIONAL_ACQUISITION.md). |
| #5/#6 recurrence/process benefit | #5 DONE, #6 DONE (negative) verdicts final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | #5's parameter/FLOP parity is measured: +0.030% parameters and +0.0001–0.0003% forward FLOPs across K=1/2/4/6/8 on real ERA5 — [R7_BUDGET_PARITY.md](R7_BUDGET_PARITY.md). #6's fair-budget comparison is done and the gate is NOT met: at K=3, 15 of 17 deltas flip sign between seeds, the only established effects are t2m worsening (+0.30 K, all depths and seeds) and t500 improving under the no-feedback arm, and forecast feedback does not rescue it — [R7_COREASONING_FAIR_BUDGET.md](R7_COREASONING_FAIR_BUDGET.md). The earlier 'feedback hurts T500' framing did not reproduce. Keep defaults and all negatives. |
| #7 adaptive benefit | DONE (negative audit) verdict final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Gate audited and found not satisfiable as stated: K=3 raises the equal-channel normalized objective 1.5–3.3% over K=0 (worse in all seeds for process_no_feedback), so fixed-Kmax is not the accuracy ceiling; and no shallower depth passes the per-variable tolerance at any tolerance ≤10% because `z250` stays 14–19% above the reference. This explains the earlier full-depth fallback. Controller thresholds were NOT changed and no savings are claimed — see [R7_HALTING_GATE_AUDIT.md](R7_HALTING_GATE_AUDIT.md). |
| #8 journal evaluation | acceptance DONE verdict final; CLOSED on GitHub via the 2026-09-25 main fast-forward; Pareto/complexity/extremes NOT done | `scripts/rollout_r7_metric_tables.py` produces paper-ready 6/12/24/48/72 h per-variable RMSE and ACC tables from frozen checkpoints without touching training code (AST-enforced), refusing to mix model generations or datasets and including a same-data persistence baseline — see [R7_ROLLOUT_TABLES.md](R7_ROLLOUT_TABLES.md). Explicitly NOT done: accuracy–compute Pareto, weather-complexity vs depth diagnostics, and extreme-event metrics. Test separation preserved (train2018/val2019/test2020). |
| #20 memory/resource acceptance | DONE verdict final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Multi-seed paired comparison on real 17-channel ERA5: 72/72 cells, 3 seeds, both tricks, at [R7_GPU_MULTISEED.md](R7_GPU_MULTISEED.md). Streamed stays flat in K (−393 MiB at K=8 vs full BPTT, all seeds agreeing) but is slower at every K; checkpointing cuts 51.8–54.5 % of peak for +25–38 ms. Memory reproduced bit-identically in all 72 cells across two runs; step time did not. No rental used and no fabricated measurement. |
| #9 finer-resolution expert | BLOCKED verdict final with reopen condition recorded; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Access audited against live endpoints: HRRR is open but CONUS+Alaska only (not co-located with 107–123 °E); HRCLDAS exposes no key-free endpoint and the CMA portal is a registration route; SMBFD has no open download; WeatherBench2/ARCO contain no km-scale East-Asia product; NOAA PSL has only US km-scale products. **Reopen when** a key-free co-located high-resolution dynamic truth source becomes available: a CMA data-service account with HRCLDAS/CLDAS rights, an open km-scale East-Asia store, or a domain pivot to CONUS (open 3 km HRRR). No interpolation or synthetic truth produced — see [R7_URBAN_EXTENSION_BLOCKED.md](R7_URBAN_EXTENSION_BLOCKED.md). |
| #1 / PR12 scientific release | negative cumulative gate status final; CLOSED on GitHub 2026-09-25T10:11Z (ff 92a8c4d..2d7da05, decision 0003) | Cumulative G1–G4 status in [R7_ROADMAP_GATE_STATUS.md](R7_ROADMAP_GATE_STATUS.md): **no gate is positively supported** — G1/G2 not supported, G3 not satisfiable as stated, G4 shows the model losing to train-only climatology at 48 h. The five things the sequence DID establish are listed there. Verdict finalized as an answered negative-result set per the 2026-09-25 authorization; PR #12 will show as merged once the authorized fast-forward carries the branch tip to main (no merge commit, ff only); a release still requires explicit human authorization. |

## Execution discipline

TODO -> IN_PROGRESS -> VERIFY -> DONE; BLOCKED names a real dependency.
During each independent CI/CPU workflow, advance another issue and return to
actual results. Close only after acceptance. Artifact/publication review is useful
work, but don't add speculative modules merely to avoid saying an experiment
has mixed results. No background continuation or re-enabled timer.

The planned controls53–58 have completed, including repairs and provenance
audits. Remaining scientific questions are not all external blockers; any new
study needs its own fixed hypothesis, bounded budget and case selection before
execution, rather than extending the completed endpoints until a desired win.
