# UrbanPiDiT-R² R7 Research Plan

Status: active on branch `r7/weather-reasoning`.

## 1. Research objective

R7 targets a **small-model, weather-native latent reasoning system** that can be trained primarily on one RTX 4090D 24GB GPU.

The central research question is:

> Can a parameter-shared small weather model improve forecast quality by repeatedly updating a meteorological Process State and a forecast draft, and learn when additional reasoning is worth its compute cost?

R7 is not intended to compete with global foundation models by parameter count or pretraining budget. The intended contribution is the **accuracy / parameter / test-time-compute Pareto frontier** on a controlled regional 0.25° benchmark.

## 2. Main hypotheses

H1. Parameter-shared recursive computation improves over a same-size one-pass forecaster.

H2. A semi-structured meteorological Process State improves over generic latent recursion under a comparable parameter/FLOP budget.

H3. Feeding the current forecast draft back into process reasoning improves iterative refinement.

H4. Marginal-gain / consistency guided halting approaches fixed-Kmax accuracy with lower average reasoning depth.

H5 (optional). The same reasoning state can route selected regions to the existing UrbanExpert when scientifically valid high-resolution targets are available.

## 3. Core method

### 3.1 Process–Forecast Co-Reasoning

Maintain two recurrent states:

- Process state P_k: anchored meteorological diagnostics + free latent reasoning tokens.
- Forecast state Y_k: the current forecast draft.

Recurrence:

\[
P_{k+1}=U(P_k, C, E(Y_k))
\]

\[
Y_{k+1}=Y_k+S(C,P_{k+1},E(Y_k))
\]

where C is encoded atmospheric context, U is the shared Process Reasoner, S is the shared Forecast Solver, and E is a lightweight forecast encoder.

### 3.2 Adaptive reasoning

During training, fixed/random reasoning depth provides per-step forecast errors:

\[
G_k=L(Y_k,Y^*)-L(Y_{k+1},Y^*)
\]

The halt controller learns whether another step has positive marginal value. A later consistency head may also compare process diagnostics with the forecast tendency.

Inference uses per-sample STOP / CONTINUE decisions with a hard Kmax.

## 4. Model scale

4090D-first target:

- total parameters: 15–30M
- BF16 mixed precision
- activation checkpointing
- physical batch 1–2 + gradient accumulation
- local/window attention only
- no global pixel-token N² attention
- recurrent parameters shared across reasoning steps
- truncated/detached recursive supervision is allowed for memory control

Initial Kmax: 4; scale experiments: 1/2/4/6/8 after correctness is established.

## 5. Forecast task

### Phase A benchmark
Regional native ERA5 0.25° atmospheric transition:

- history: t-6h, t
- base transition: +6h
- final rollout: 6/12/24/48/72h
- chronological year split; never random time split

Candidate variables (final list fixed after data-size benchmark):

Surface:
- 2m temperature
- 10m u/v wind
- mean sea-level pressure
- total precipitation (with appropriate transform/loss)

Pressure levels:
- 850 hPa: Z/T/Q/U/V
- 500 hPa: Z/T/Q/U/V
- 250 hPa: Z/U/V

The first engineering fixture may use fewer channels; paper experiments should use a multi-variable state.

### Domain
Prefer an East-Asia regional benchmark compatible in spirit with WeatherBench-style regional evaluation. A smaller North-China ROI may be used for fast ablation, but the main paper table should not rely on a single-city domain.

## 6. Anchored process diagnostics

Initial candidate proxies computed directly from weather fields:

- pressure/geopotential gradient magnitude
- horizontal divergence
- relative vorticity
- temperature advection
- moisture advection
- moisture convergence
- static-stability proxy
- vertical wind shear

These are diagnostic proxies, not claimed causal ground truth.

Use 8 anchored tokens + 8 free latent tokens as the first R7 configuration.

## 7. Training stages

### Stage 0 — correctness
Issues #2–#3.
Fix geospatial attention boundaries, reasoning trace, intermediate supervision and per-sample adaptive execution.

### Stage 1 — forecast-native backbone
Issue #4.
Implement native atmospheric transition prediction and remove external urban baseline from the core forecast task.

### Stage 2 — generic recursive baseline
Issue #5.
Implement a TRM-like generic recurrent forecaster with no meteorological Process State.

### Stage 3 — process–forecast co-reasoning
Issue #6.
Feed forecast drafts back into Process Reasoning and compare against the generic recursive baseline.

### Stage 4 — adaptive halting
Issue #7.
Train gain/consistency-based STOP/CONTINUE with compute-cost warm-up.

### Stage 5 — rollout and journal evaluation
Issue #8.
6–72h rollout, parameter/FLOP/latency reporting, reasoning-depth analysis and paper-ready ablations.

### Stretch — adaptive spatial resolution
Issue #9.
Only after the core paper is stable and real fine-resolution dynamic targets are available.

## 8. Mandatory baselines

At minimum:

- persistence
- compact U-Net
- ConvLSTM
- Swin-style forecaster
- AFNO/FourCastNet-lite style forecaster
- fixed-depth transformer
- generic parameter-shared recursive model (R7.2)
- fixed-depth Process Reasoning
- adaptive Process–Forecast Co-Reasoning (ours)

All main-table neural baselines must use the same train/validation/test data and variable set.

## 9. Mandatory ablations

| Process state | Forecast feedback | Shared recursion | Adaptive halt | Purpose |
|---|---|---|---|---|
| no | no | no | no | one-pass baseline |
| yes | no | no | no | process representation |
| no | yes | yes | no | generic recursive baseline |
| yes | no | yes | no | recurrent process reasoning |
| yes | yes | yes | no | co-reasoning |
| yes | yes | yes | yes | full method |

Additional:
- anchored-only vs free-only vs anchored+free
- K=1/2/4/6/8
- process auxiliary loss on/off
- detached/truncated recursion vs full BPTT where feasible

## 10. Success gates

G1: fixed recursive reasoning improves most core variables over K=1.

G2: process-aware co-reasoning beats generic recursion at comparable budget.

G3: adaptive reasoning reaches near-Kmax accuracy with substantially lower average K.

G4: R7 lies on a competitive accuracy/parameter/compute Pareto frontier against same-data strong baselines.

G5: gains remain visible beyond +6h during autoregressive rollout.

## 11. Reporting and interpretation

Primary plots:

- RMSE/ACC vs lead time
- accuracy vs reasoning steps
- accuracy vs measured inference compute
- histogram/map of adaptive reasoning depth
- reasoning depth vs vorticity/divergence/pressure-gradient/precipitation complexity
- process-token diagnostic calibration
- failure cases where extra reasoning does not help

Do not claim natural-language Chain-of-Thought. Use:
- latent process reasoning
- recurrent test-time computation
- Process–Forecast Co-Reasoning
- adaptive reasoning depth

## 12. Resource policy

4090D is the development and ablation platform. Rental GPUs are used only after:
- data pipeline is fixed,
- all smoke/unit tests pass,
- hyperparameters are substantially frozen,
- the final run demonstrably requires more VRAM/throughput.

No expensive GPU time is used for debugging data or model correctness.

## 13. Stage-4 engineering contract (issue #19)

The first gain policy is an opt-in wrapper; the fixed-depth forecaster remains unchanged.
See [R7_ADAPTIVE_HALTING.md](R7_ADAPTIVE_HALTING.md) for the implementation, tests and usage.
Fit this small policy on training data with a frozen forecaster first, before experimenting
with joint optimization or compute-penalty warm-up. Stream teacher states and retain only
compact features/errors. Only observed consecutive-draft transitions receive gain labels;
the final draft receives no invented continuation target. Adaptive inference must whitelist
initialization-time inputs and execute only active samples, never read future labels.

Passing synthetic/CPU tests closes engineering children, not the scientific gates in #7.
One-step gain is a greedy proxy, not optimal long-horizon value. Add held-out calibration,
look-ahead/oracle controls and measured latency/VRAM before making efficiency claims.
A gain controller is not a physical consistency verifier; input-time process diagnostics
must not be compared as if they were future-time truth.

Source inspection also identified #20: detach_between_steps truncates gradient dependencies
but retaining all graph-carrying drafts still retains per-step activations. Do not describe
existing end-to-end training as constant-memory or24GB-validated without measurements.

## 14. Evidence-driven CPU iteration (issues53–58)

GPU access is currently deferred by the user. Small public-data CPU experiments
are authorized, not automatic paid rentals or unbounded archive downloads.
The continuous250day profile supplies3000real timestamps and998windows per split;
this remains a single small tile, not the representative final paper domain.

The iteration completed fixed optimization and data-coverage controls, then tested
an optional aligned draft route directly into the shared correction decoder.
`spatial_solver_feedback=False` remains the default; enabling it adds existing
draft tokens to the solver context with no new parameters. Both generic and
process baselines receive the same mechanism. Fixed/streamed/adaptive paths
are synchronized and gradient/execution equivalence-tested.

Real validation results are mixed: wind improvements in the400-update spatial
ablation do not establish uniform improvement, and T500 worsened. Keep this
candidate opt-in. Do not claim the process auxiliary objective or adaptive
controller has met the scientific gates. No controller retuning or test
evaluation occurred in these controls.

Future data-profile comparisons must use explicitly identical initialization
sets, or a clearly labeled timestamp-only common-case audit that retains excluded
IDs and both normalization/data identities. Old sparse/continuous default
selectors yielded21common cases out of24; aggregate24-case comparisons were not
a paired data-quantity experiment.

See `R7_CPU_REFINEMENT_RESULTS.md` and its artifact manifest for complete runs,
budgets, negative outcomes and reproducibility limits. Prior model checkpoints
require their archived source-code digest; never rewrite signatures after a
new optional architecture is introduced.
