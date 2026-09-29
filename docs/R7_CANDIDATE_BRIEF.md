# R7 Evidence Candidate Brief

> This is an evidence index, not a scientific verdict. Priority is a human triage field.
> No entry authorizes training, data access, GPU use, or a change to a frozen criterion.

Records: 7; human-review candidates: 1

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
