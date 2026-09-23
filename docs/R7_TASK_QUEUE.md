# R7 conversation-driven task queue

Updated after exact-SHA engineering acceptance at
`6ba682021d502523138e735d9798c1418c4abffe` (2026-09-23).
No scheduled automation. No automatic merge/release, paid GPU use or large data downloads.
The current conversation can execute multiple independent tasks; a single waiting
workflow is never treated as the entire project being blocked.

## Completed in this iteration

| Issue | State | Commit | Exact R7 CPU CI | Result |
|---|---|---|---|---|
| #36 validation-only frozen halting-policy selection | DONE | fc80e0cc1ca0a7408021bb0626ea3949763bcad3 | 35864456724 / job 107192403363 | 273 passed, 3 skipped |
| #37 same-forecast full/interior/edge scoring | DONE | 69b94ce925dd78df2e6305a5aed8a2dbfccb1ff6 | 35865390327 / job 107195559964 | 287 passed, 3 skipped |
| #38 isolated fixed/adaptive inference profiling | DONE (engineering) | e3555298251e9511021ab0cdc68995b06fd11f7c | 35866192926 / job 107198269800 | 299 passed, 3 skipped |
| #39 retrospective delayed-gain oracle diagnostic | DONE (engineering) | 6ba682021d502523138e735d9798c1418c4abffe | 35866881415 / job 107200631291 | 314 passed, 3 skipped |

Each run also passed active-module compilation, whitespace and installed-wheel
checks. Two known Lightning logging warnings remain. The three skips are only
old untracked local Beijing/UCI fixtures; new tests are not skipped. Tiny actual
ERA5 t2m regression from #35 runs offline; this is separate from those skips.

## Existing implemented foundations

Strict geographic/time/channel contracts, non-destructive versioned Zarr
publication and train-only normalization are already implemented (#23/#24 and
data children). Streamed recursive training with local checkpoint/resume and
profiling is implemented (#21/#25). Exact-time free-running evaluation,
train-only climatology/ACC, compact baseline adapters, frozen controller fitting
and paired comparison helpers are available. See the corresponding docs and
closed engineering issue receipts; these are not pending implementation tasks.

## Remaining open parent gates

| Issue / task | State | Concrete blocker | Resume condition / next action |
|---|---|---|---|
| #13 production multivariate ERA5 validation | BLOCKED (external data / run authorization) | No multi-year, multivariate cache or authorized target storage/run supplied; #35 is a genuine but tiny t2m-only fixture | Provide cache location, variables/levels/units, coverage and provenance, or explicitly approve a bounded acquisition/storage plan; then run preflight and local real-data acceptance |
| #20 actual single-4090D resource acceptance | BLOCKED (hardware access) | Current execution container is CPU-only; no authorized GPU host is connected | Run the existing bounded K=1/2/4/8 profile on the user's GPU with the frozen realistic shape/config, or connect an authorized host; inspect actual peak allocated/reserved memory and synchronized time |
| #5 generic recursion and #6 process-loop research gates | BLOCKED (data / trained checkpoints / approved compute) | No comparable trained multivariate baselines or controlled multi-seed pilot results | After #13/#20, freeze data split/normalization and equal training budgets, run matched models and K ablations; report negative as well as positive results |
| #7 adaptive scientific acceptance | BLOCKED (trained model / held-out data / hardware) | Controller engineering is verified; real validation-selected policy and held-out accuracy/latency evidence are absent | Fit on training only, use #36 on validation, freeze policy, run held-out free trajectories, #38 timing and #39 diagnostic; any consistency/look-ahead redesign follows evidence rather than speculation |
| #8 full journal evaluation | BLOCKED (experimental artifacts / protocol decisions) | Evaluation tools exist, but seasonal/extreme/multi-seed results and representative boundary tests require a frozen real dataset and trained models | Evaluate identical initialization sets; use paired bootstrap and #37 full+edge+interior reports; predeclare event definitions and do not hide boundary failures |
| #9 optional finer-resolution expert | BLOCKED (external scientific labels + core benchmark gate) | Genuine co-located high-resolution dynamic targets are not supplied | Obtain/audit target source and core benchmark evidence before developing the extension; interpolated ERA5 is never fine-grid truth |
| #1 research epic / PR #12 scientific release | BLOCKED (parent gates) | Engineering progress does not establish the thesis hypotheses | Reopen the execution queue when data/hardware/permission gates resolve; PR remains Draft, no main merge |

## Scheduling and verification discipline

States: TODO -> IN_PROGRESS -> VERIFY -> DONE. BLOCKED means a named dependency,
not lack of initiative. A workflow in VERIFY records its exact SHA/run, required
checks and resume condition in its issue. During it, implement/review an
independent task; do not stack half-finished changes just to keep CI busy.
Before writes, reread the remote ref and preserve other work. No force push.
After each completed issue rescan the queue and inspect relevant regressions.

At this handoff no newly implemented issue is left awaiting CI. The remaining
open parents require external experimental inputs/authorization. More speculative
model features are not a substitute for obtaining those inputs. Concrete bugs
found in a later review still become executable issues and take priority.

See [R7_EXPERIMENT_HANDOFF.md](R7_EXPERIMENT_HANDOFF.md) for the next bounded run.
