# R7 parent roadmap: cumulative gate status (#1)

Issue #1 states four research gates and a dependency order. This page is the
cumulative answer after the #13 → #20 → #5 → #6 → #7 → #8 sequence, with every
claim bound to a commit and a CI run.

`scientific_claim: false`. This is a status synthesis, not a new experiment.

## Gate-by-gate status

| Gate | Statement | Status | Evidence |
| --- | --- | --- | --- |
| **G1** | Fixed recursive reasoning improves over K=1 | **NOT SUPPORTED** | K=1 vs K=0/3 on real ERA5: `generic` normalized RMSE 0.31079 (K=0), 0.30991 (K=1), 0.32117 (K=3). Deeper is *worse*. [R7_HALTING_GATE_AUDIT.md](R7_HALTING_GATE_AUDIT.md) |
| **G2** | Process-aware recursion improves over generic | **NOT SUPPORTED** | Fair-budget comparison, 3 seeds: 10 of 17 variables nominally better but **15 of 17 deltas flip sign across seeds**; the only sign-consistent effects are `t2m` worsening (+0.30 K) and `t500` improving under the no-feedback arm. Budget parity holds (+0.030 % params, +0.0001–0.0003 % FLOPs). [R7_COREASONING_FAIR_BUDGET.md](R7_COREASONING_FAIR_BUDGET.md) |
| **G3** | Adaptive halting approaches Kmax accuracy at materially lower average K | **NOT SATISFIABLE AS STATED** | The Kmax reference is not the accuracy ceiling, and no shallower depth passes the per-variable tolerance at any tolerance ≤10 % (`z250` stays 14–19 % above reference). [R7_HALTING_GATE_AUDIT.md](R7_HALTING_GATE_AUDIT.md) |
| **G4** | Competitive on accuracy/parameter/compute Pareto vs strong same-data baselines | **PARTIALLY, NEGATIVELY** | Rollout tables exist with a same-data persistence baseline over 6–72 h; generic t2m RMSE 3.72 → 5.61 K and **ACC −0.016 at 48 h**. Persistence looks better at 24 h because the 4 initializations sample different times of day. Not a Pareto study — the Pareto bullet is explicitly un-done. [R7_ROLLOUT_TABLES.md](R7_ROLLOUT_TABLES.md) |

**No gate is positively supported.** Three are unsupported or unsatisfiable as
written, and the fourth is measured but shows the model losing to climatology at
48 h on this case set.

## What the sequence did establish

These are real, bounded, reproducible results — just not the ones the roadmap hoped for:

1. **The data pipeline works on real data.** A 65x65 native-0.25° East-Asia
   extraction (48 and 120 timestamps, SHA256 recorded per artifact), converted
   through the audited publication path to 17-channel stores with
   `BUILD_COMPLETE.json` and train-only statistics. (#13)
2. **The training budget is now measured, not assumed.** 72/72 GPU cells at three
   seeds: streamed truncated training is flat in K (−393 MiB at K=8 vs full BPTT,
   all seeds agreeing) but slower at every K; activation checkpointing cuts
   51.8–54.5 % of peak for +25–38 ms per step. **Memory reproduced bit-identically
   in all 72 cells across two independent runs; step time did not** (max 41 ms
   spread). (#20)
3. **The baselines are genuinely comparable.** `generic` and `process` differ by
   +0.030 % parameters and +0.0001–0.0003 % forward FLOPs across K=1/2/4/6/8, so
   G2's comparison is not confounded by capacity. This was previously *unchecked*:
   no FLOP accounting existed anywhere in the repository. (#5)
4. **The evaluation harness produces paper-ready tables** from frozen checkpoints
   without touching training code, refusing to mix model generations or datasets,
   with a same-data persistence baseline and explicit units. (#8)
5. **Deep reasoning trades variables rather than improving them.** At K=3 vs K=0,
   5 variables consistently improve, 5 consistently worsen, 7 are unresolved.

## Why the parent stays open

`#1`'s own dependency order says the extension resumes "only after R7.1–R7.5 are
stable". They are not stable in the sense the roadmap meant: R7.3 and R7.4 have
produced **negative** findings, not a working core to extend. Closing #1 would
require either the hypotheses being demonstrated or an explicit decision to
publish the negative result set — and that is a **human decision**, not an
automatic one.

## What would change the picture

The negative results are all bounded by the same substrate: 200 optimizer updates,
`dim=32, depth=2`, ten January days per year, 38 windows per split, 3 seeds. Three
concrete ways forward, in increasing cost:

1. **More data and longer training** on the same protocol. The current stores are
   ten-day January blocks; the pipeline supports any bounded window, and the
   measured cost model is ~25 s/timestamp for all nine source variables regardless
   of ROI size.
2. **Restate G3.** The measured evidence says depth hurts the normalized objective
   while helping some variables, so "approach Kmax" is the wrong target; a
   per-variable halting objective would need to be declared before it is tested.
3. **Raise seed count** to separate real small effects from seed noise. The
   sign-stability rule currently labels 13–15 of 17 variables unresolved per arm,
   which is a statement about power, not about the models.

## Machine-readable artifacts

| Item | Location |
| --- | --- |
| Real extraction (120 timestamps) | `outputs/r7_coreasoning_data/source.nc`, SHA256 `db7c02191e520a81873ab90b2a35348a43d556fb65a82da0cd60cce03041a2f4` |
| Real extraction (48 timestamps) | `outputs/r7_regional_real/source.nc`, SHA256 `d3fa1fba6da46ed59a535ce27f7501813afc2c93cb8b745c90e349454a40960a` |
| Fair-budget comparison | `outputs/r7_coreasoning_v2/coreasoning_result.json`, protocol digest `253c5a442934fa82f6d7bd2bbdfa8c1d0ca668175faad5bbfbcf700bf6bec296` |
| GPU multi-seed | `outputs/r7_multiseed_real_v3/multiseed.json`, protocol digest `6c84be201134faa3a0895ebb9c1bfb1cf67f4161dd283fd613ba2a33fc70e7f7` |
| Rollout tables | `outputs/r7_rollout_tables/` |
| Halting audit | `outputs/r7_halting_audit/report.json` |
| Budget parity | `outputs/r7_budget_audit/report.json` |

These live under `outputs/` and are deliberately untracked (R-012); the code,
protocol digests and tables are in version control and reproduce them.

## Honest limitations of this synthesis

- **Every gate rests on a bounded CPU/GPU substrate**, not a converged run.
- **Seed counts are 3** (or 1 for the rollout tables), which is a noise check, not
  a significance test. No p-values are claimed anywhere.
- **Validation cases were inspected in earlier iterations**, so they are not a
  pristine held-out benchmark; the rollout tables do use the untouched test split,
  but only 4 initializations.
- **G2's "process does not help" finding is bounded by this budget** and does not
  prove process state is useless in general.
- **#9 is BLOCKED on external data access** ([R7_URBAN_EXTENSION_BLOCKED.md](R7_URBAN_EXTENSION_BLOCKED.md)),
  so the optional extension contributes nothing to G1–G4.
