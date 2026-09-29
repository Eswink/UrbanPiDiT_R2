# R7 Evidence Candidate Brief

> This is an evidence index, not a scientific verdict. Priority is a human triage field.
> No entry authorizes training, data access, GPU use, or a change to a frozen criterion.

Records: 12; human-review candidates: 1

## b2-multiseed-negative

- Outcome class: `negative`; candidate state: `candidate`
- Human triage priority: `90` (not a scientific score)
- Evidence: `docs/R7_B2_MULTISEED.md` (SHA256 `4bc8827e35e5996593294ff1386e62faacd6ba4a550c36f7f5aafdfa14a68ee1`)
- Evidence commit: `07fc05c3d66ce4719afaff808c178a855910567b`; experiment commit: `94828da846dc43a2aac76183c4ea0a9e49b5009b`
- Protocol SHA256: `244af00b0834948091506f20f6d3d26a093cff8b142a5f10b084e4b95c6643e2`; data identity: `50517d945e8c6e2dd2c48fa4a57041cdfb67588f98cfe7aa6edd684754cd6961`
- Reason: Three-seed evidence supports retaining the negative result and considering a separately authorized broader-data question rather than changing defaults.
- Limitations:
  - All evaluations were on val; test_read is false.
  - The segment is January 2016 with four climatology buckets, not cross-season evidence.
  - Three seeds provide consistency evidence, not a significance test.
  - The bounded 800-update runs are not convergence claims.
- CI run: `36276587108`
- Excluded from runnable candidates: Candidate for human review only; follow-up still requires a new frozen protocol and explicit authorization.
- Recorded metrics (not recomputed): gate_met=false, gpu_hours=1.157, lead_hours=[6, 12, 24, 48, 72], seeds=3, test_read=false

## rw-b-subtraction-round-cannot-attribute

- Outcome class: `unresolved`; candidate state: `needs-review`
- Human triage priority: `79` (not a scientific score)
- Evidence: `docs/R7_72_RW_B_SUBTRACTION.md` (SHA256 `5de5b165cb3a4a6fd0820265bc137a24c2dcf9ee0f92de8d384654b81aae92fb`)
- Evidence commit: `926d2fef81a5750fcb9b960f6fe0415706fe0d91`; experiment commit: `926d2fef81a5750fcb9b960f6fe0415706fe0d91`
- Protocol SHA256: `58fc74b7a7aaa513197d85f836684cd55851013b3c7f8519f184649837b357d4`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: The pre-declared subtraction round: the registered negative control RW-B-(a) is degenerate by construction - with identical weights it is bitwise identical to RW-A in the forward pass, the training loss and all 131 shared gradients, receiving zero gradient on the solver side - so its same-sign 'worsening' of 3e-05..8e-05 K is float noise on an identity comparison, not a confound. The frozen branch rule fired as written (stop-confounded-control) and no attribution is reported. The primary arm's own reading (48 h +0.785 K, 72 h +1.174 K, both seeds, worsened) would have pointed at the gate+anchored proposal had the control been admissible, and the previous round's registered deltas reproduced to ~2e-04 K under a changed model-code digest.
- Limitations:
  - The round's negative control is invalid by construction: this is a defect of the control arm's design, recorded rather than worked around.
  - Two seeds: sign agreement is a consistency check, not significance; no threshold was introduced or relaxed.
  - The arms are not capacity-matched, so no pair here is a compute-controlled contrast.
  - The leave-one-out arms remove a piece at training time; the reference arm RW-A was retrained under this round's digests, so the two rounds' numbers are never pooled.
  - The correction probe covers 8 validation windows per seed: proportions over those windows, not a distribution estimate.
  - Validation split only; the test split stayed sealed and was never opened.
  - One winter segment of one year in one region: no seasonal or cross-region claim.
- Excluded from runnable candidates: Cannot attribute at this budget: the control arm cannot distinguish anything, so no mechanism conclusion may be drawn from this round. The next admissible step is the pre-declared falsifiable hypothesis in the evidence document, or the turn to forecast state / training objective / data regime that stop condition 3 requires.
- Recorded metrics (not recomputed): arms=4, branch=stop-confounded-control, forward_flops_ratio_rw_b_vs_rw_a=1.2097, gpu_hours=0.4128, gpu_hours_evaluation=0.0368, gpu_hours_training=0.376, negative_control_shared_weights_relative_diff=0.00016, parameters_rw_b=3283157, previous_round_reproduced_max_abs_delta=0.00018, primary_no_recurrence_t2m_48h=0.7854, primary_no_recurrence_t2m_72h=1.1739, round_reference_t2m_48h=1.0651, round_reference_t2m_72h=1.5805, rw_b_shared_weights_relative_diff=1.74, seeds=2, test_read=false, thresholds_added=0, updates_per_arm=400

## rw-b-bounded-round-negative

- Outcome class: `negative`; candidate state: `needs-review`
- Human triage priority: `78` (not a scientific score)
- Evidence: `docs/R7_72_RW_B_PILOT.md` (SHA256 `7d23503f06b967045aa94545c95e56451d04275826c19278b5346cb1e71421e3`)
- Evidence commit: `d8c828d8d34a97096465b3c46308dd4ddc889b52`; experiment commit: `d8c828d8d34a97096465b3c46308dd4ddc889b52`
- Protocol SHA256: `6f48874296352d660c2a5d07c3de7c3bceac16619e54adec4a505ce8ff5d9ed9`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: The first bounded real comparison of RW-B: the registered primary RW-B - RW-A on t2m is not supported (12/24 h improve by 0.08 K, 6/48/72 h worsen by 0.12/1.07/1.58 K, every lead sign-consistent across both seeds, modal reading 'worsened'), so the local gated solver state is a negative result at this budget and capacity and must not be extended before the mechanism is attributed.
- Limitations:
  - Two seeds: sign agreement is a consistency check, not significance; no threshold was introduced.
  - The arms are not capacity-matched, so the negative result is not attributed to the per-position gate, the X_t-anchored proposal or Z; decision 0023's matched-Generic prerequisite is still unmet.
  - Validation split only (22 windows, 11-22 per lead depending on lead); the test split was never opened for method selection.
  - One bounded 400-update run on one winter segment in one region: not a convergence or cross-season comparison.
  - The K=1/2/4 depth probe runs on a K=3 checkpoint, and the correction probe covers 8 windows: diagnostics, not deployable rules.
  - The round spent 0.8356 GPU-h instead of 0.42 because a first attempt died in a new probe after training and evaluation had completed; that attempt's numbers are retained but were not used.
  - The deep security scan run at the end of the round is inconclusive and none of its 27 findings (all inside the read-only legacy_v531_full snapshot) was triaged.
- Excluded from runnable candidates: Negative result at one bounded budget: not a candidate for extension. The next admissible action is a predeclared subtraction of the gate/proposal, Z and the role markers, which needs its own authorization.
- Recorded metrics (not recomputed): arms=4, forward_flops_ratio_vs_rw_a=1.2097, gpu_hours=0.8356, gpu_hours_single_clean_attempt=0.4154, parameters_added_vs_rw_a=314898, primary_leads_supported=2, primary_leads_worsened=3, seeds=2, t2m_delta_48h=1.065, t2m_delta_72h=1.5804, test_read=false, thresholds_added=0, updates_per_arm=400

## rw-b-subtraction-probe

- Outcome class: `audit`; candidate state: `needs-review`
- Human triage priority: `76` (not a scientific score)
- Evidence: `docs/R7_72_RW_B_SUBTRACTION.md` (SHA256 `5de5b165cb3a4a6fd0820265bc137a24c2dcf9ee0f92de8d384654b81aae92fb`)
- Evidence commit: `926d2fef81a5750fcb9b960f6fe0415706fe0d91`; experiment commit: `926d2fef81a5750fcb9b960f6fe0415706fe0d91`
- Protocol SHA256: `not recorded`; data identity: `not recorded`
- Reason: The 0 GPU-h leave-one-observation probe over the archived RW-B checkpoints: removing the gate+anchored proposal collapses the step-1 correction to 0.20-0.24x the RW-A magnitude in both seeds, while removing the cross-step recurrence of Z turns the error/correction cosine positive from step 1 in both seeds without reproducing the 1.8x magnitude. The same probe measures that the focus arm's correction_head never received a gradient, which is why the decision-0021 round retrains the leave-one-out arms from scratch rather than reusing this row.
- Limitations:
  - Validation split only, 8 windows, one lead: a diagnostic, never a skill number.
  - No threshold is introduced; the readings are the three the bounded round defined.
  - Each row is an inference-time removal from a checkpoint trained with the piece on, so it is not the training-time contrast the bounded round measured.
  - The gate+proposal-removed row routes the checkpoint through a correction head this arm never trained; the probe reports that measurement instead of leaving it implicit.
  - The archived rows were measured on CUDA and this replay on CPU, so agreement is float32 reduction order (max |delta| 7e-06 on one seed and one flipped boolean on the other, reported).
- Excluded from runnable candidates: Read-only diagnostic over archived checkpoints: it trains nothing and produces no forecast skill, so it cannot be indexed as a candidate model result.
- Recorded metrics (not recomputed): checkpoints_hashed=32, checkpoints_replayed=4, cpu_seconds=6.2, focus_arm_correction_head_moved_by_training=false, gpu_hours=0.0, recurrence_removed_cosine_positive_both_seeds=true, seeds=2, step1_magnitude_ratio_gate_removed_mean=0.22, step1_magnitude_ratio_recurrence_removed_seed41=1.48, step1_magnitude_ratio_recurrence_removed_seed42=3.04, test_read=false, thresholds_added=0, validation_windows=8

## b1-d1-controlled-comparison

- Outcome class: `unresolved`; candidate state: `needs-review`
- Human triage priority: `75` (not a scientific score)
- Evidence: `docs/R7_B1_D1_COMPARISON.md` (SHA256 `ff693be93c6daa48e053fe8b8946ee5be389eaf3e5dec3c380236f32c61385f5`)
- Evidence commit: `d70dfd4638efbcc87854153182950de33753ac51`; experiment commit: `a295334ed4964adf815f0246977a9af854e964be`
- Protocol SHA256: `1b1c176702f23da4edffd8feeea9d2d7d8ee42f882d8553faece20c1ecb98e14`; data identity: `588cfc8b95e2fade8937b1b5936d6386f4824a5ff5b20bb1e620ca7a147d4f3a`
- Reason: Controlled validation-only baseline comparison is reproducible enough to inform the next frozen study, but is not a multi-seed skill verdict.
- Limitations:
  - All evaluations used val; test_read is false.
  - D1 is a January engineering re-split, not the v2 2021 test candidate.
  - Single-seed and four-lead evidence cannot establish a model winner.
  - Seasonal claims are not measured.
- CI run: `36258059039`
- Excluded from runnable candidates: Do not launch directly from this index; any follow-up needs a new frozen protocol and authorization.
- Recorded metrics (not recomputed): gpu_hours=0.41, ranking_flips=0, test_read=false, train_windows=94, val_windows=10

## rw-b-local-solver-state

- Outcome class: `engineering-positive`; candidate state: `needs-review`
- Human triage priority: `75` (not a scientific score)
- Evidence: `docs/R7_72_RW_B.md` (SHA256 `96fc5f10f598aa27f43e19cdaef59fc033bb8da914b2143fdb11517926c5e172`)
- Evidence commit: `f4a571b370691162126f207b7d984761ec7ec57f`; experiment commit: `f4a571b370691162126f207b7d984761ec7ec57f`
- Protocol SHA256: `not recorded`; data identity: `not recorded`
- Reason: RW-B is implemented as real model code behind two default-off switches, its boundaries are asserted with counterproofs, and the step now has a single implementation across the fixed, streamed and adaptive paths. The evidence document carries a dated update block: the bounded train/val comparison it recorded as not executed was run in the following round, and its verdict is registered separately.
- Limitations:
  - No training was run, so this record carries no RMSE, no skill and no accuracy-compute frontier.
  - The gate is a stabilization candidate, not a guarantee of monotone improvement or physical correctness.
  - No VRAM or wall-clock measurement is made; the persisted solver state increases per-step working memory.
  - The role markers are shown to be reachable, not shown to be useful.
  - The matched-Generic prerequisite for any process-structure claim remains unmet, so decision 0023 is not yet satisfied.
- CI run: `36572082730`
- Excluded from runnable candidates: Engineering completion only: the bounded train/val comparison needs the decision-0021 authorization at execution time and was not run.
- Recorded metrics (not recomputed): forward_flops_ratio=1.2097, gpu_hours=0.0, new_tests=37, parameter_ratio_added=0.1061, parameters_added=314898, test_read=false, tests_passed=1456

## b1-baseline-audit

- Outcome class: `audit`; candidate state: `needs-review`
- Human triage priority: `70` (not a scientific score)
- Evidence: `docs/R7_B1_BASELINE_AUDIT.md` (SHA256 `6cc58ac3977e908d03d1c92387cc06253f0e88528b057262955ba7e8c731db99`)
- Evidence commit: `cce0bf0e9decfdca6f94f3ddb2c6dbe5f512cf98`; experiment commit: `e8834d58aeee9494bc6dcde077fcac1b2724cef8`
- Protocol SHA256: `not recorded`; data identity: `not recorded`
- Reason: Static baseline and mechanism audit completed without launching the formal B1 comparison.
- Limitations:
  - No formal B1 training comparison was run in this record.
  - The audit cannot establish forecast skill or an architecture winner.
  - A protocol digest and data identity are not recorded for this audit-only segment.
- Excluded from runnable candidates: Not a runnable experiment record; missing protocol and data identity.
- Recorded metrics (not recomputed): candidate_parameter_band=1-5M, formal_training_started=false, gpu_hours=0.0

## e0-pre-registered-diagnostics

- Outcome class: `audit`; candidate state: `needs-review`
- Human triage priority: `70` (not a scientific score)
- Evidence: `docs/R7_E0_DIAGNOSTICS.md` (SHA256 `72d13994c5c4c4df9c2a84b93a4d1bbf5b21a4aafe84dd1d06eb76a6bba04072`)
- Evidence commit: `f4a571b370691162126f207b7d984761ec7ec57f`; experiment commit: `d8aff68e06357ddf8036d8e2971b92dd26adb96a`
- Protocol SHA256: `not recorded`; data identity: `not recorded`
- Reason: The two diagnostics rounds two and three pre-registered and never ran were executed on validation only, at 0 GPU-h, with the checkpoints replayed by the model revision that trained them; one archived claim is contradicted.
- Limitations:
  - Validation-only, one 2016 winter segment, three seeds per arm: a stability observation, never a pooled estimate.
  - Round two and round three have different protocol digests and different model_code_sha256, so their deltas are compared but never merged.
  - The correction geometry is re-measured by re-running the forward pass, not replayed from a published number.
  - The 0.25-0.29 K band reproduces at t2m 48h; the t2m 72h limb is not sign-stable in round three and must not be read as mechanism evidence.
  - No threshold, significance level or confidence interval is introduced.
- CI run: `36572082730`
- Excluded from runnable candidates: Read-only diagnostic, not a runnable experiment: it trains nothing and produces no forecast skill, so it cannot be indexed as a candidate model result.
- Recorded metrics (not recomputed): checkpoints_replayed=12, cpu_seconds=111.1, gpu_hours=0.0, test_read=false, thresholds_added=0, validation_windows=22

## c1-process-supervision-mixed

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `65` (not a scientific score)
- Evidence: `docs/R7_65_C1_PROCESS_SUPERVISION.md` (SHA256 `9f5cf0e695cb60003d537c86c87298a8c1d5558d753ded624ef9b79855eb7f89`)
- Evidence commit: `77f271843f5b6a03193f5e32aa6264b1787688f4`; experiment commit: `f292b1a219f9b201f09b7d2e283a71cf2221c763`
- Protocol SHA256: `d0a59c9b21ffe1c336fd02aec93e39d1c40e5426679be3a94974f74b205a8e16`; data identity: `50517d945e8c6e2dd2c48fa4a57041cdfb67588f98cfe7aa6edd684754cd6961`
- Reason: Auxiliary process supervision has a mixed, lead-dependent result and no established process win.
- Limitations:
  - Validation-only evaluation with test_read false.
  - The process labels include floor-degraded proxies.
  - Three seeds are a consistency check, not a significance test.
  - No C1 by C2 full factorial was run.
- Excluded from runnable candidates: No stable positive process effect; defaults and scope should not change from this record alone.
- Recorded metrics (not recomputed): auxiliary_weights=[0.0, 0.01, 0.1], gpu_hours=1.08, process_win_established=false, seeds=3, test_read=false

## c3-depth-mixed

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `60` (not a scientific score)
- Evidence: `docs/R7_65_C2_C3_FEEDBACK_AND_DEPTH.md` (SHA256 `8b8a1f73815c1990686afa6474c2954a2cdfe0cb2ab81fb0c82ca3932cb9d189`)
- Evidence commit: `77f271843f5b6a03193f5e32aa6264b1787688f4`; experiment commit: `c0dcf85e95e5d16c29ee2bdf0d7a0ad5994bd501`
- Protocol SHA256: `4b1d9103a5b4c3e3d79a00bca5bf81564a747b60e83714a062c5728ea00e5498`; data identity: `50517d945e8c6e2dd2c48fa4a57041cdfb67588f98cfe7aa6edd684754cd6961`
- Reason: Depth changes trade accuracy and compute by lead; free scaling is not supported and the independent shallow arm is mixed.
- Limitations:
  - Validation-only evaluation with test_read false.
  - K6/K8 were not measured.
  - Three seeds are a consistency check, not a significance test.
  - The invalid first K1 harness run is retained but excluded from conclusions.
- Excluded from runnable candidates: Mixed depth trade-off; requires a separately frozen hypothesis before any follow-up.
- Recorded metrics (not recomputed): free_scaling_supported=false, gpu_hours=0.5, independent_k1_result=mixed, phase=c3, seeds=3, test_read=false

## b3-capacity-mixed

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `55` (not a scientific score)
- Evidence: `docs/R7_B3_SCALE.md` (SHA256 `dc1feb1dec8a541381f710257d2723bf2edbba9f9302b47a998bd17055059e49`)
- Evidence commit: `f8b360c9ae090d98da257c580748f29566a96bc9`; experiment commit: `eba90e0834f6bf0ed88875acf667bcfe0f0649b9`
- Protocol SHA256: `not recorded`; data identity: `50517d945e8c6e2dd2c48fa4a57041cdfb67588f98cfe7aa6edd684754cd6961`
- Reason: Capacity expansion produced a seed-inconsistent isolated improvement and does not support a default capacity change.
- Limitations:
  - Two seeds do not constitute a significance test.
  - All evaluation remained on val within the January engineering segment.
  - The fixed 800-update budget cannot separate capacity from optimization duration.
  - No recursive versus non-recursive conclusion is in scope.
- Excluded from runnable candidates: Mixed and seed-inconsistent; the report only preserves a short protocol summary, so no immediate runnable candidate is justified.
- Recorded metrics (not recomputed): capacity_multiple_vs_b2=6.48, gpu_hours=0.547, seed_consistent_gain=false, seeds=2, test_read=false

## c2-feedback-negative

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `45` (not a scientific score)
- Evidence: `docs/R7_65_C2_C3_FEEDBACK_AND_DEPTH.md` (SHA256 `8b8a1f73815c1990686afa6474c2954a2cdfe0cb2ab81fb0c82ca3932cb9d189`)
- Evidence commit: `77f271843f5b6a03193f5e32aa6264b1787688f4`; experiment commit: `c0dcf85e95e5d16c29ee2bdf0d7a0ad5994bd501`
- Protocol SHA256: `9a5ce6c50365626449aee83d936389f8499331c15582c0f984f126f0d7d13eff`; data identity: `50517d945e8c6e2dd2c48fa4a57041cdfb67588f98cfe7aa6edd684754cd6961`
- Reason: Feedback routing was tested under matched declared conditions and the current defaults were retained.
- Limitations:
  - Validation-only evaluation with test_read false.
  - FLOP parity does not imply wall-time parity for every path.
  - The experiment covers one January engineering segment.
  - No external seasonal or yearly claim is supported.
- Excluded from runnable candidates: Negative result with no surviving default-changing hypothesis in the tested scope.
- Recorded metrics (not recomputed): defaults_changed=false, gpu_hours=1.09, phase=c2, seeds=3, test_read=false, worsened_cells=28
