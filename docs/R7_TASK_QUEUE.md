# R7 conversation-driven task queue

Updated 2026-09-23 after the actual public pressure-data CPU experiment and offline replay.
No timer, automatic main merge/release, paid GPU use or uncontrolled data mirror.
User explicitly defers GPU testing and authorizes public-data acquisition plus small CPU tests.
No longer describe the entire project as blocked because a GPU or private dataset is absent.

## Latest completed real-data / CPU milestone

| Issue | State | Actual evidence |
|---|---|---|
| #41 NCAR public metadata decision | DONE | 69d73259..., probe35873661154; tested pressure layout remained a global 37-level slab, so not adopted |
| #45 anonymous temporal ERA5 source | DONE | a9c605de..., probe35874572228; pinned snapshot ZFKDHBCTBVHVXM3BQFV0, single-level spatial tiles |
| #44 real pressure/surface CPU pilot | DONE | 4cee1f85..., real run35875707823; eleven channels, 96 times, 12x12 native0.25-degree tile; all eight process proxies; CPU train/val/test |
| #46 verified local-source replay | DONE | 4ba62b1c..., actual replay35877069975 with outbound sockets prohibited; source SHA unchanged, no cloud-source dependency |
| #40 bounded public multivariate CPU parent | DONE (bounded integration) | Its requested channel bundle and finite CPU/held-out/proxy acceptance criteria are met; this is not final scientific acceptance |

Latest code verification: 4ba62b1c603bcf8c5dda888dce4afe3bf8e24efe,
CPU CI35877070004 / job107235638164: **359 passed, 3 skipped, 2 warnings**.
The skips are only existing untracked local Beijing/UCI fixtures. New extraction,
replay, corruption and CLI tests ran. Compilation/whitespace/wheel checks passed.
A deadline TEST incorrectly assumed monotonic time started at zero long ago;
a5fe3e41... replaced it with an explicit clock and pre-read expiry assertion.
No production time/budget check was weakened and no failed test was hidden.

Source NetCDF: 629,696 bytes, SHA256
`13fb72807d3f6988b0fcf2b6b2f130f890207242916018556fb85553686b9a12`.
Full original artifact10757057891/run35875707823: 3,026,772 bytes, ZIP SHA256
`bca9e915630ebb6f73df33f5a0d0c569915a1b5d44c174d73ca49a0615d8ff66`.
Receipts, per-field hashes, original model checkpoints, per-variable/horizon
RMSE/ACC and validation selection are retained. The source is genuine ERA5
reanalysis, not direct station observations. Dataset access is anonymous, no paid
subscription. Decoded chunk accounting is not an HTTP traffic or RAM measurement.

**Measured limitation:** validation selected force-full-depth K3; the tiny trial
has not shown adaptive compute saving. Process K3 does not uniformly outperform
generic recursion/K1. This negative evidence must survive reporting.
Instructions/tables: [R7_PRESSURE_CPU_RESULTS.md](R7_PRESSURE_CPU_RESULTS.md).

## Previously completed foundations

#36 validation-only frozen policy selection (fc80e0cc..., CI35864456724,273passed),
#37 same-forecast full/interior/edge scoring (69b94ce9..., CI35865390327,287passed),
#38 isolated inference profiling (e3555298..., CI35866192926,299passed),
#39 retrospective delayed-gain diagnostic (6ba68202..., CI35866881415,314passed).
Each above had three optional local-fixture skips; all remain in the current suite.

Native/generic/process models, finite geographic/time/channel/unit contracts,
non-destructive versioned Zarr publication, train-only normalization, streamed
truncated training with encoder gradients, checkpoint/resume, exact-time rollout,
climatology/ACC, compact baseline adapters and paired-comparison helpers exist.
Do not recreate them or substitute synthetic smoke for the newly available real pilot.

## Remaining research tasks (not certified by the CPU milestone)

| Task | State | Next action / actual dependency |
|---|---|---|
| #13 production multivariate dataset | IN_PROGRESS | Small-data acquisition/physical-unit acceptance now works. A representative multi-season/larger-domain dataset still needs a bounded plan; the Jan1-8 excerpts are not three full training years. |
| #5/#6 same-data research gates | IN_PROGRESS | Tiny native/generic/process CPU results are available, but 20 updates, one seed and six correlated test initializations cannot establish convergence or superiority. Next controlled experiment should assess training budget/sample diversity, not add speculative modules. |
| #7 adaptive scientific gate | IN_PROGRESS | Actual train/calibrate/validation-select/test path executed but selected full K3. Diagnose data/training/gain calibration before claiming savings; true GPU speed measurements remain deferred. |
| #8 journal evaluation | IN_PROGRESS | The real tiny pilot has 6/12/24/72h RMSE/ACC; representative seasons/extremes, multi-seed uncertainty and boundary interventions are still missing. |
| #20 4090D resource acceptance | BLOCKED (hardware, explicitly deferred) | User cannot access the card yet. Keep existing profiling commands; no request to rent a GPU and no claim CPU counts measure VRAM. |
| #9 optional finer-resolution expert | BLOCKED (scientific targets/core gate) | Genuine co-located finer-resolution dynamic labels and core experimental evidence are still required. |
| #1 / PR#12 research hypotheses | IN_PROGRESS | Maintain Draft, do not label data-pipeline success as SOTA or a completed paper. |

## Execution discipline

TODO -> IN_PROGRESS -> VERIFY -> DONE. Mark asynchronous tasks VERIFY with their
exact SHA/run and resume condition; work independently during CI. Do not force
push, overwrite others or close research gates from engineering tests. Scheduled
work remains disabled. Record precise new blockers rather than claiming all paths
are blocked merely because GPU hardware is unavailable. The current requested
small real-data CPU milestone is complete; no newly implemented code is waiting
for validation. No larger training or download job is running in the background.
