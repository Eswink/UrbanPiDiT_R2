# R7 CPU refinement iteration: verified results for issues #53–#58

This is a bounded real-data research iteration, not a scientific release.
All six engineering/experimental children were accepted after actual workflows.
No GPU, paid rental, scheduler, test-set tuning or automatic main merge was used.

## Verified execution records

| Issue | Implementation | Full CI | Actual experiment/audit |
| --- | --- | --- | --- |
| #53 error/correction geometry | `1ce88009532eec597c3cf78457bfe4e6ce7da445` | 35891600361, 478 passed / 3 skipped | 35891600411 |
| #54 fixed 200-to-800 optimization control | `163e88a60f28bf0a4e31a97396130439a08606b1` | 35892145656, 502 passed / 3 skipped | 35892145587 |
| #55 continuous data within unchanged source cap | `d4830b1f5eff6715cd3cf08659fda923ee8f742a` | repaired test at `d02634acb1d5c3ed8f2897dc3d85cafac707cc44`, CI35894113212 success | 35893397412 |
| #56 continuous-data 800-update control | `f1d395f5ab1e4b69e805cbbb095e7a4acf6c9743` | 35895446093 success | 35895446235 |
| #57 opt-in spatial solver | `7d32687d3383e4d227ba750d33ee3c127ab8a82a` | 35896740912 success | 35896740916 |
| #58 explicit/common-case alignment | `a4d5eab9c0969169ae385e649dd2a3bbb1c56f63` | 35898091928, 578 passed / 3 skipped | 35898091953 |

Final code CI35898091928, job107306975262: **578 passed, 3 skipped,
2 warnings in 49.25 seconds**, plus compilation, whitespace and installed-wheel
checks. The skips are old untracked Beijing/UCI fixtures; new tests ran.
The warnings are the existing Lightning unit-test logging-without-Trainer warnings.

The #55 test initially mixed pandas datetime64[us] `.asi8` with nanosecond
Timedelta.value. It was corrected to explicit `as_unit("ns")`, with ns/us/ms
regressions. No source values, extraction caps or experimental scores changed.

## 1. Real data expanded without increasing decoded source reads

Source: anonymous Earthmover Icechunk ERA5, snapshot `ZFKDHBCTBVHVXM3BQFV0`,
`s3://earthmover-icechunk-era5/icechunkV2`. Exact provider 0.25-degree tile,
12 x 12, 39.25–42 N / 114–116.75 E; no additional spatial interpolation.
This is reanalysis, not direct station observation.

The named `continuous-250d` profile retains 1000 six-hourly timestamps starting
January 1 in each of 2018/2019/2020: **3000 total timestamps** and **998 one-step
windows in each chronological split**. End dates are September 7 at18UTC in
2018/2019 and September 6 at18UTC in leap2020. These are not three full years.

Eleven fields: t2m/u10/v10/mslp/t850/t500/q850/u850/u500/v850/v500.
All eight physical input-time process proxies and train-only normalization run.
The old January and four-season profile defaults/pins are unchanged.

- Raw NetCDF: **19,052,672 bytes**.
- Source SHA256: `0609fa38c1d88b82b985a15f93dd502c7eb7031f5d2b7452bbe971bf936dd9c1`.
- Decoded charged bytes: **180,142,968**, exactly unchanged from the previous
  four-season extraction, under the same192MiB cap.
- Why: additional selected times remain within the same8736-hour source chunks;
  the code verifies that property for every variable/level before weather reads.
- This is decoded logical accounting, NOT measured HTTP traffic, peak RAM or VRAM.
- Acquisition + preparation +20-update integration:68.338seconds, excluding
  installation/workflow startup. CPU2threads, no test evaluation.

Reuse locally, with the original paired receipt:
```bash
python -m data.download.continuous_pilot_replay \
  --source /path/to/era5_continuous250d.nc \
  --receipt /path/to/receipt.json \
  --out outputs/fresh_continuous_cache
```
The verifier requires the exact audited file pin and all variable/time/space/unit
checks. A new cache gets its own true identity; old normalization identities are
never pasted onto changed data.

## 2. More optimization does not fix every recursive error

#54 resumes nine original four-season runs (generic/process/process-no-aux,
seeds41/42/43) from200 to800 optimizer updates. It preserves optimizer/RNG,
batch2,LR2e-4,K3 and objectives. No new architecture or threshold selection.
The endpoint800 was fixed before the run and not extended until a win.

For Process, three-seed mean validation +6h:
- T2m:200K3 5.594853K ->800K3 4.407987K.
- T500:200K3 1.303789K ->800K3 1.442146K;800K1 is1.184223K.
- MSLP at800:K1 203.301059Pa versusK3 211.468801Pa.

Thus some learning progresses while deeper corrections still degrade selected
variables. These are pilot observations, not proof of convergence or a cause.
Execution371.948seconds excluding setup/retrieval,9 x600 new updates.

#53 decomposes each actual update:
`MSE_after - MSE_before = 2*mean(error*correction) + mean(correction²)`.
At K2->K3, wrong-direction updates were common for T500 and MSLP; consecutive
corrections had cosine means near0.99. The oracle damping diagnostic used truth
retrospectively and was NEVER applied to deployed forecasts.

## 3. More temporal data: paired within-study controls

#56 trains fresh generic/process/process-no-aux from three fixed seeds for
800updates each on998training windows. Fresh starts are required because the
data and normalization changed. There is no old checkpoint identity override.
It evaluates the same24validation cases within this study at6/12/24/72h,
with both same-checkpoint K1 andK3. No controller fit or test use.

| Model at800updates | +6h T2m K | +6h MSLP Pa | +6h T500 K |
| --- | ---: | ---: | ---: |
| Generic K3 | 4.174097 | 228.972133 | 1.019690 |
| Process K1 | 4.688453 | 233.101369 | 0.991436 |
| Process K3 | 4.544389 | 235.682551 | 1.003683 |

All variables, horizons and seeds remain in the artifact. Process does not
uniformly beat generic or shallow inference. Runtime587.061seconds excluding
setup/retrieval,9 x800updates; CPU2threads. This is not model-only latency.

## 4. Case-set discrepancy found and audited, not hidden

The first-six-complete rule yields different initialization sets for sparse
seasonal snippets versus continuous histories. Their24-case aggregate results
cannot be treated as a paired data-quantity experiment.

#58 retained the **21 timestamp-intersection cases** and both three-case
exclusion lists. Seasonal-only:Apr/Jul/Sep2 at12UTC. Continuous-only:
Apr/Jul/Sep1 at00UTC. Nothing was filtered by forecast error.

On those SAME21cases,800updates,K3,three-seed means:
| Model/field | Seasonal | Continuous |
| --- | ---: | ---: |
| Generic T2m K | 4.301289 | 3.983440 |
| Generic T500 K | 1.475463 | 1.048380 |
| Generic MSLP Pa | 229.723093 | 214.804597 |
| Process T2m K | 4.447228 | 4.263834 |
| Process T500 K | 1.361127 | 1.039178 |
| Process MSLP Pa | 214.961550 | 217.262718 |

These exploratory results vary by variable. Coverage, normalization and sample
order all changed; they do not isolate one cause. The comparator checks original
model-source equality and resource/checkpoint ownership, not raw truth-array
equivalence between source files. It never trains or recomputes a forecast.
A target-free explicit-case selector is now available for future comparisons.

## 5. Small architecture iteration, with a fair generic control

#57 adds **optional `spatial_solver_feedback=True`**, no new trainable parameters:
`Decode(C + projected_global_process_summary + aligned_draft_tokens)`.
Previously the draft reached the reasoner, but the correction decoder saw only
fixed spatial context plus a globally pooled latent summary.

Both generic and process models receive the same mechanism. DefaultFalse is
bitwise equivalent to the old decoder equation. The global/spatial versions
respectively retain generic83,222 andprocess83,319parameters.

A fixed400-update,3-seed,24-validation-case experiment trained all four variants
(generic/process x off/on):12runs,4800updates,440.824seconds excluding setup.
Do not compare400-update scores against the earlier800-update scores as a fair
architecture contest. Every variable and lead is retained.

Process K3, +6h three-seed mean RMSE:
| Field | Original pooled solver | Spatial draft solver | Change |
| --- | ---: | ---: | --- |
| U10 m/s | 1.299816 | 1.252527 | 3.64% lower |
| V10 m/s | 1.695001 | 1.630204 | 3.82% lower |
| MSLP Pa | 260.877509 | 256.505033 | 1.68% lower |
| T2m K | 5.152417 | 5.133995 | 0.36% lower |
| T500 K | 1.002459 | 1.054561 | 5.20% higher/worse |

Generic spatial feedback also helps selected wind variables. Therefore the
experiment does NOT establish process-token superiority. T500 regression prevents
a uniform improvement claim; default remainsFalse, candidate stays opt-in.

Fixed forward, streamed truncated gradient, checkpointing/odd-grid/BF16 and
adaptive active-subset pathways are tested. If process forecast feedback is off,
both draft routes are off. No future labels or learned damping were added.

Model-source digest legitimately changed. Use archived code.zip for older
checkpoints; do not bypass integrity checks. New baselines/candidates were all
trained under the same new implementation.

## Artifacts and integrity

All archives were downloaded and independently SHA256-verified. Per-variable
source hashes, original checkpoint digests, protocols and complete summary tables
were also recomputed/checked as relevant. See `R7_CPU_ITERATION_ARTIFACTS.json`.

No result establishes SOTA, causal reasoning, full-year/domain generalization,
GPU latency/VRAM or adaptive savings. The earlier controller fallbacks remain
valid negatives. No controller was recalibrated in issues53–58.

## Next research decision boundary

The completed controls distinguish training amount, temporal coverage, and one
minimal spatial input route. They do not justify a blind test-set sweep or
automatically changing the default solver. The next useful experiment should
freeze a broader, temporally spaced validation case set and a small explicit
hypothesis (e.g. auxiliary-loss conflict), rather than repeatedly optimizing
these already inspected24cases. A larger domain/final benchmark needs a separately
bounded protocol. GPU acceptance remains deferred, not a blanket CPU blocker.
