# R7 Evidence Candidate Brief

> This is an evidence index, not a scientific verdict. Priority is a human triage field.
> No entry authorizes training, data access, GPU use, or a change to a frozen criterion.

Records: 16; human-review candidates: 1

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

## n1-cost-v2-coresidency-preparation

- Outcome class: `audit`; candidate state: `blocked`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_N1_COST_V2_PREPARATION.md` (SHA256 `e86aa5f938533a9f6f94103aef486d538766fe00cfda690d7de490f248bb1c1d`)
- Evidence commit: `90e2c839bfafeeb97c9b20b3f4ffa2040676b4c0`; experiment commit: `b227020ca3d4932a765a6cd7a79e26fc659cdeaf`
- Protocol SHA256: `not recorded`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: Default co-residency policy and per-cell fresh-process v2 implementation are prepared and engineering-verified. Decision-0021 execution-time AskUserQuestion received no answer, so no GPU probe or thirty-evaluation cost attempt was launched.
- Limitations:
  - No named v2 authorization response or receipt; no GPU experiment, probe result, cost attempt, terminal tables or cost_views exists.
  - CPU/fake CUDA and exact-commit CI do not verify real allocator residues, peak memory, replay or budget feasibility.
  - Fourteen skipped clean-clone tests are not passing tests; local test counts are not remote CI log counts.
  - Original v1 failure and all original scientific evidence and pins remain unchanged; N1 stays paused and N2d is only proposed.
  - Co-residency memory queries are not reservations and neighbors can affect wall-clock measurements.
  - Mimosa scanner_enobufs is inconclusive, not security clearance; no security or credential configuration changed.
- CI run: `36855190840`
- Excluded from runnable candidates: Awaiting named probe-plus-30-val co-residency execution authorization. Engineering tests and free-memory observations do not replace actual CUDA cost evidence or permission.
- Recorded metrics (not recomputed): campaign_status=paused, clean_clone_passed=1988, clean_clone_seconds=173.22, clean_clone_skipped=14, current_node=N1, engineering_ci_main_steps_success=9, evaluations_completed=0, evaluations_planned=30, gpu_execution_authorized=false, gpu_experiment_executed=false, gpu_hours=0.0, independent_evaluation_cost_complete=false, original_branch=cannot-distinguish, targeted_tests_passed=359, test_read=false, training_updates=0

## n1-cost-v2-evaluation-supplement-success

- Outcome class: `audit`; candidate state: `needs-review`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_N1_COST_SUPPLEMENT_V2.md` (SHA256 `7ce659dfa6e244b754383a8b42bc5ce8e16eacb6b136a24f97bad63cd9e01873`)
- Evidence commit: `3d7a8e2e52941407cfb882780413b8b99d7e5a3c`; experiment commit: `b714458ed2ed1e29282d024ae5b72bf2c4839215`
- Protocol SHA256: `877d0cabef976b2896ab3ffd817e78ba6f92ac03e218c6920c1c0ea32a9a510f`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: The explicitly authorized one-shot v2 cost audit succeeded: P1 reproduced a releasable torch process workspace family, so conditional P2 did not trigger; thirty fresh subprocesses replayed the original six checkpoints and five validation leads at zero allocator baselines. Four terminal cost views are complete, independently artifact-verified and fully charged. The original 48h/72h cannot-distinguish reading, paused N1 and proposed-only N2d are unchanged.
- Limitations:
  - This is validation-only cost replay with zero training updates, no new arms/seeds/cases and test_read false; two seeds on one winter segment do not establish significance, convergence or seasonal skill.
  - All thirty RMSE, ACC and climatology-skill CSV files are byte-identical to their originals; provenance differs only in elapsed_seconds. This does not establish future bit-reproducible GPU execution, allocator peaks or wall-clock measurements; original training remains config-reproducible.
  - Evaluation memory and elapsed cover the whole archived evaluate_local call including loading, IO and metrics, not isolated forward latency/memory, device-wide usage or the original run's historical peaks.
  - Parameters, FLOPs, training throughput and six training-memory rows reference the original archive, not newly recomputed or retrained values; thirty independent evaluation-memory rows are newly measured.
  - P1 supports a releasable torch workspace family only. The exact v1 residual holder and historical byte counts remain unrecorded; the private release API is diagnostic-only, never used for accepted evaluations.
  - Shared-headroom queries are not reservations. Ninety-one startup/spawn/bracket observations recorded no external PID, so this is not empirical performance under an actively co-resident neighbor load.
  - Independent read-only verification checks persisted identities, process records, tables, snapshots and budgets, not a second GPU run or complete runtime syscall proof of absence of networking, test tensor reads or neighbor signals.
  - Original N1 plus failed v1 plus v2 totals 0.49326015495695175 GPU-h, above the historical 0.45 figure. The separate v2 audit authorization caps v2 at 0.25 GPU-h and does not rewrite the old cap; every interval remains charged to the campaign.
  - Cost completion does not repair the original scientific design limitations, satisfy matched-Generic or revise the unresolved primary contrast; N1 remains paused and N2d is a proposal only.
  - Mimosa pre-commit/pre-push scans were inconclusive (scanner_enobufs), not security clearance; dependencies, model/data identities, credentials and security configuration are unchanged.
- CI run: `36855190840`
- Excluded from runnable candidates: Accepted cost-audit artifacts, not a runnable model candidate, scientific mechanism verdict, goal-completion declaration or permission to advance N1. The one-shot authorization is consumed; historical prepared/failed records and original scientific evidence remain unchanged.
- Recorded metrics (not recomputed): all_allocator_baselines_zero=true, attempt_sha256=e172da59a39ceb9b456e2e910b0e616e67876b8a1e51df34e2d530e1f1cae9e0, attempt_status=success, authorization_scope=n1-evaluation-cost-supplement-v2-with-residue-probe, authorization_sha256=0770b4f80ff1b95b0223d10cf80b498c524ffc626e1ece91f10ae3199db35edf, automatic_retry=false, byte_identical_csv_counts={"acc.csv": 30, "climatology_skill.csv": 30, "rmse.csv": 30}, campaign_remaining_gpu_hours_exact=19.79553984504305, campaign_status=paused, campaign_used_gpu_hours_exact=4.204460154956952, case_evaluations=528, checkpoint_identities=6, conditional_p2_triggered=false, cost_views_sha256=42606c5782d6f25d90e1686aefbf3f52d74fbf75423655d81c44ee935fbb98db, current_node=N1, device_policy=shared-headroom, distinct_launch_ids=30, distinct_worker_pids=30, evaluations_completed=30, evaluations_missing=0, evaluations_registered=30, flops_table_rows=3, gpu_hours=0.12826335332563354, gpu_hours_cap=0.25, gpu_phase_seconds=461.74807197228074, independent_artifact_assertions=8592, independent_evaluation_cost_complete=true, independent_verification_sha256=40699952b68fedabfa159ec8ca52ee4dab20ffeaf7c11ddcaa06d9202a08b0f2, measurement_code_zip_sha256=b14b3252f1b440c942ce220ae1e6707c9c9213cfdbc20a82423e294f4c2cdae0, memory_table_rows=36, minimum_free_mib=2048, minimum_observed_free_mib=23713, next_node_proposal=N2d, observed_external_pids=[], original_branch=cannot-distinguish, original_n1_plus_v1_plus_v2_gpu_hours=0.49326015495695175, p1_after_clear_allocated_bytes=0, p1_after_clear_reserved_bytes=0, p1_before_clear_allocated_bytes=8519680, p1_before_clear_reserved_bytes=20971520, p1_classification=torch-releasable-workspace-family, p1_diagnostic_seconds=0.9619096238166094, parameter_table_rows=3, peak_allocated_bytes_range=[39590400, 45923840], peak_reserved_bytes_range=[46137344, 77594624], probe_sha256=b94aa9383e453635bce5a67edaa38696e816a630ef6dda23f8a09b7474f134b9, reproducibility_level=byte-identical validation metric CSV replay; original training config-reproducible; no future bitwise GPU cost guarantee, result_sha256=7b758e2d428f9a256e8b60fb97535c479b9114b9098aa20d0d3aa34b5b076b0c, rmse_cells=510, run_files_unchanged=407, selected_device_observations=91, selected_update=400, test_read=false, thresholds_added=0, training_throughput_table_rows=6, training_updates=0, whole_wall_seconds=463.4213050529361, whole_wall_seconds_cap=1200.0

## n1-evaluation-cost-supplement-failed

- Outcome class: `audit`; candidate state: `blocked`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md` (SHA256 `4b350357af20ce95bc9c2e31b7411f83108fda4dc4366e681b61e0cbecdac9fc`)
- Evidence commit: `33d57d67fc8b247262e807aa74510cb5828792f8`; experiment commit: `1e03f82067e429185a4dc41b6d76da0020d6dece`
- Protocol SHA256: `1ce9222321bfe6d799b0f86d7bc0ff4de127d451edaa0e5e8a45ca5a4a3ffc22`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: The named one-shot validation-only cost supplement failed after the first of thirty evaluations. Seed41/RW-A/6h has an independent zero-baseline allocated/reserved peak and exact original RMSE/case replay; the second evaluation was refused before its call because the allocator baseline was nonzero. The failed interval is fully charged, no automatic retry occurred, and the original unresolved scientific result is unchanged.
- Limitations:
  - Only one of thirty evaluations completed; no worker success summary, result.json, terminal four cost tables or cost_views.json was produced.
  - The second-lead allocated/reserved baseline values were not logged; the underlying allocator/object cause is unverified and no additional CUDA diagnostic or rerun occurred.
  - Memory scope is the whole archived evaluate_local call including model loading, IO and metrics, not isolated model-forward memory or the original run's historical peak.
  - Exact RMSE, per-case MSE and identity replay is verified only for the retained seed41/RW-A/6h cell, not the missing twenty-nine cells.
  - Parameters, FLOPs and training throughput were to reference the original archive; zero optimizer updates or new selection were performed.
  - Occupancy checks cannot exclude transient competition between observations; device UUID was bound to the exclusive child environment.
  - The original 48h/72h primary remains cannot-distinguish; this failure does not revise a scientific criterion or authorize campaign advancement.
  - The one-shot authorization has been used; the remaining budget is arithmetic only, not automatic retry permission.
  - Mimosa commit/push scans have been inconclusive (scanner_enobufs), not security clearance; credentials and security configuration are unchanged.
- CI run: `36718455666`
- Excluded from runnable candidates: Failed cost evidence is retained for audit and budget accounting, not accepted full cost delivery or a runnable scientific candidate. Twenty-nine evaluation peaks and the terminal four cost tables remain missing; any repair/retry needs a separate decision and named authorization.
- Recorded metrics (not recomputed): attempt_sha256=662dd5ab2862bd4b0a1cfc585366bfe4b29a0e57edffd6f7f03715f405a35479, attempt_status=failed, automatic_retry=false, campaign_status=paused, current_node=N1, evaluations_completed=1, evaluations_missing=29, evaluations_registered=30, first_cell_allocated_baseline_bytes=0, first_cell_arm=process_spacetime_rwa, first_cell_cases=22, first_cell_cost_measurement_sha256=b654d17d812d3353e887b392505248f9bbdf5d8796c397b2dccb51ab85dcf270, first_cell_elapsed_seconds=11.9927125191316, first_cell_exact_numeric_replay=true, first_cell_lead_hours=6, first_cell_peak_allocated_bytes=39590400, first_cell_peak_reserved_bytes=46137344, first_cell_reserved_baseline_bytes=0, first_cell_rmse_variables=17, first_cell_seed=41, gpu_hours=0.005956627869357666, gpu_hours_cap=0.09, gpu_phase_seconds=21.443860329687595, independent_evaluation_cost_complete=false, measurement_code_zip_sha256=b7f20e7b8adda117c394deb0751fcd1683b536404d5cfa18e121d61c39b742c0, original_branch=cannot-distinguish, original_plus_supplement_gpu_hours=0.3649968016313182, test_read=false, thresholds_added=0, training_updates=0, whole_wall_seconds=22.372659532353282

## n1-pivot-audit-frozen-z-unresolved

- Outcome class: `audit`; candidate state: `needs-review`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_N1_PIVOT_AUDIT.md` (SHA256 `e5448064f1c4612ce30026ababb3b1b5a725048da566f72be4eec7cec92b8106`)
- Evidence commit: `33ee4702b22bfac5db303164886c1a57a3406a59`; experiment commit: `c4e7e83a5deaaade1caa21fe82faa064e75b3b72`
- Protocol SHA256: `e19ef488be60136364702b1df389e5f58be7ebf3845487289139e10d30231e01`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: N1 four zero-GPU audit sections plus the authorized frozen-random-Z falsification arm: frozen Z - RW-A is seed-disagreeing at both t2m 48h and 72h, so the preregistered reading is cannot-distinguish and the round stops with N2d proposed only. The contemporaneous RW-B reference still worsens; frozen Z - RW-B improves at the two long leads but cannot replace the unresolved primary.
- Limitations:
  - Two seeds and 400 updates on one winter regional ERA5 train/val segment; no significance, convergence, cross-season or causal verdict.
  - Frozen random Z changes trainable capacity and backward FLOPs; original cell forward still runs, so this is not perfectly matched compute or capacity.
  - The primary 48h and 72h seed signs disagree; their means are null and the secondary carrier contrast does not settle necessity of learned Z.
  - Training is config-reproducible only, not bit-reproducible; no GPU retraining or additional model evaluation was performed after the stop.
  - The three predeclared pairs contain complete final metadata and exactly paired cases, but the finalizer does not enforce the full secondary set for future runs.
  - Validation has no per-case deadline check; this run stayed below both caps but a general hard-stop guarantee remains unverified.
  - All 30 evaluation peak-memory values inherit the last training peak; independent evaluation allocated/reserved peaks are missing, so the memory view is not fully accepted.
  - The merged whole_round_elapsed_seconds field contains GPU phase time; true whole-wall time is taken from attempt.json without rewriting the archive.
  - D1 objective/exposure rankings share structural evidence, not independent statistical support or a causal ranking; no next-node experiment is authorized.
  - Mimosa pre-commit/pre-push scans were inconclusive (scanner_enobufs), not security clearance; no security or credential configuration was changed.
- CI run: `36691526554`
- Excluded from runnable candidates: Stopped under the frozen unresolved rule, not a runnable model candidate or permission to add seeds, rerun an arm, change a criterion, read test or enter another node. The execution is recorded, not the goal declared complete.
- Recorded metrics (not recomputed): archived_comparison_metadata_replay=byte-identical, arms=3, branch=cannot-distinguish, case_counts_by_lead=[22, 21, 19, 15, 11], cells_per_pair=85, code_zip_sha256=5fd26146af2a7d11016fb769d67390f5daa23a73620de9ae83e2e6cc38a35a0a, deadline_per_validation_case_guarded=false, depth0_table_rows=255, engineering_local_clone_passed=1795, engineering_local_clone_skipped=14, evaluations=30, finalizer_full_set_guarded=false, gpu_hours=0.3590401737619605, gpu_hours_cap=0.45, gpu_phase_seconds=1292.544625543058, independent_evaluation_peak_measured=false, next_node_proposal=N2d, paired_comparison_sha256=d7a345c17923cd49d1b03f2cbea767c95579e5d18a043965bf29fb697c222a64, primary_long_lead_means_K=[null, null], primary_long_lead_outcomes=["unresolved", "unresolved"], primary_t2m_48h_seed_deltas_K=[0.06312165146909354, -0.19323128240116816], primary_t2m_72h_seed_deltas_K=[0.32588177204935054, -0.15142658867975278], reproducibility_level=config-reproducible training; byte-identical metadata comparator replay, rmse_cells=510, seeds=[41, 42], stop_required=true, test_read=false, thresholds_added=0, training_runs=6, updates_per_arm=400, whole_wall_seconds=1297.8983452636749

## rw-b-subtraction-round-cannot-attribute

- Outcome class: `unresolved`; candidate state: `needs-review`
- Human triage priority: `79` (not a scientific score)
- Evidence: `docs/R7_72_RW_B_SUBTRACTION.md` (SHA256 `e6557c9ada40151d40977bf061d5c1da7907f907d68859f604204a62b57cdc4e`)
- Evidence commit: `5dc479e00225b075542ad3f73eb3952bbce045e7`; experiment commit: `5dc479e00225b075542ad3f73eb3952bbce045e7`
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
- Evidence: `docs/R7_72_RW_B_SUBTRACTION.md` (SHA256 `e6557c9ada40151d40977bf061d5c1da7907f907d68859f604204a62b57cdc4e`)
- Evidence commit: `5dc479e00225b075542ad3f73eb3952bbce045e7`; experiment commit: `5dc479e00225b075542ad3f73eb3952bbce045e7`
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
