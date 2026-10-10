# R7 Evidence Candidate Brief

> This is an evidence index, not a scientific verdict. Priority is a human triage field.
> No entry authorizes training, data access, GPU use, or a change to a frozen criterion.

Records: 68; human-review candidates: 1

## s3-budget-curve

- Outcome class: `mixed`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_BUDGET_CURVE.md` (SHA256 `7e2c0d37efadcbd0d4175e1a738a47831d929511b16f5c2971faf3a58fbbe48d`)
- Evidence commit: `2ae7990cf6651264ce00986e7f06ffe0636d9658`; experiment commit: `2559c1fadeb390294475102ea8f8556d61343139`
- Protocol SHA256: `08eaf4768dd4fc06a2bed26080cf1f37ddced51e277799fa58e419561f0fa434`; data identity: `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09`
- Reason: S3-BC budget-curve screen: the second and final declared dose of the update-budget factor on the v2 instance (l6 at 1600 updates vs the pinned 400-update control). The pre-registered primary reads supported at 6h and 12h on all seeds, with t2m seed-mean skill rising monotonically across the 400/800/1600 doses at 6/12/24h (0.272/0.498/0.595, -0.077/0.196/0.333, -0.116/0.120/0.226); training loss still descends through 1600. However the u10/v10/mslp pre-screen fails 17 cells at 48h/72h (worse than the 13 at 800 updates), so the frozen conjunction rule registers no-advance and the pre-declared budget-response reading resolves toward data: on the single-year train set the budget is saturating. Whole round 4937.6 s vs planned 5400 / hard 10800 with zero overrun; code fenced at 3995267; test unread.
- Limitations:
  - screening only: one instance (2017 train / 2022 val, one ROI, 17 channels), no significance, convergence or SOTA claim
  - the control readings come from the registered S3-D3 incumbent run (400 l6 updates) on the same store and are pinned by SHA256, not retrained inside this protocol
  - the candidate deliberately spends 4x the control's training updates and FLOPs; the verdict prices that extra training and is not a compute-matched comparison
  - the 800-update reading compared against is the registered S3-UB round; this round does not re-run it
  - three seeds are consistency evidence, not a significance test
  - K4 is an inference-depth probe on the same K4-trained checkpoint, not an independent model
  - GPU runs are co-resident; latency/memory observations include neighbor load
  - validation split only; the test split stays sealed until the S4 preregistered read
  - this is the final declared budget dose on the v2 instance; further budget questions move to the batch-3 expanded instance
- Excluded from runnable candidates: 混合筛选结果：第二剂（也是 v2 实例最后一剂）更新预算因子。主格 t2m/full 6h/12h 三 seed 同号 supported（6h −0.6427/−0.7454/−0.9084 K、12h −0.6195/−0.7667/−0.9339 K，24h 亦全负），近三 lead 的 skill 单调改善（seed 均值 6h +0.595、12h +0.333、24h +0.226）；但 u10/v10/mslp 守门在 48h/72h 有 17 个正 cell（比 800 更新的 13 个更差），按冻结合取规则不进 S4。预声明的预算响应读数据此判为「单年数据下预算饱和」：下一能力投资是数据（batch-3 扩年在并行获取）而非继续加预算。不构成任何科学声明。
- Recorded metrics (not recomputed): budget_response=t2m improves monotonically at 6/12/24h through 1600 updates while the long-lead gate cost grows 13->17 cells; on one train year the budget is saturating and the next capability investment is data, candidate_updates=1600, control_updates=400, correction_evidence=docs/R7_S3_V3_NUMERICAL_ERRATA.md, delta_t2m_full_rmse_vs_incumbent={"41": {"12": -0.9339, "6": -0.9084}, "42": {"12": -0.7667, "6": -0.7454}, "43": {"12": -0.6195, "6": -0.6427}}, dose=2, elapsed_seconds_total=4937.6, first_spawn_to_last_reap_seconds=4884.4, gate_failures=17, gate_failures_by_lead={"48": 8, "72": 9}, gpu_hours=1.3716, hard_cap_seconds=10800.0, mode=l6, network_bytes=0, planned_seconds=5400.0, positive_seed_cells_of_51={"12": 51, "24": 51, "48": 7, "6": 51, "72": 2}, primary_verdict={"12": "supported", "6": "supported", "overall": "supported"}, seeds=[41, 42, 43], soft_overrun_seconds=0.0, t2m_seed_mean_skill_12h=0.333, t2m_seed_mean_skill_24h=0.226, t2m_seed_mean_skill_6h=0.595, test_read=false, updates_ratio=4.0

## s3-case-interleaving-pilot

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_CASE_INTERLEAVING_PILOT.md` (SHA256 `7043fa608de5a74d1ecb2b10c52475419e5f937278460248a8dd6c65d85a63bb`)
- Evidence commit: `9dde48de1666de88035e2c677f20c70ede33ed48`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `6efae97d11db3d4fa23e0d4be49771010804cede9ccf6909a9ed20ad5bbdf877`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Complete equal-exposure ordering intervention: two arms start from identical original0 weights and fresh AdamW/seed41,80 batch1 updates each, exactly20 exposures per original seasonal train case; blocked four20case blocks versus interleaved fourcase cycles. Both fit allfour train objectives, and interleaved improves302/340 train final cells, but predeclared necessary development support is false: t2m6/12/24/48 is worse than blocked; only72 strictly beats both blocked/shared0,48/72climatology skill still negative; u10 allfive leads regress versus shared0. Fixed endpoint80/new16pairs and unchanged common0 scorebytes, all17x5x3 and947saved file/math/process/cost facts independently checked. Negative development support stops this ordering instance; no automatic dose/seed/fullval expansion or scientific acceptance.
- Limitations:
  - Single seed and four repeatedly fitted training/four already-exposed development cases; no annual generalization, simultaneous intervals, formal candidate or scientific acceptance.
  - Artificial blocked pressure condition is not historical bulk randperm regime or a proved worst case; order includes global LR position, AdamW memory and endpoint recency, not isolated forgetting causality.
  - Ordinary new-contract loading retains full source/data/case/activity/RNG identities; no old optimizer/RNG/cursor restore, no digest bypass.
  - Native FP32 objective and FP64 decomposition preserve residuals; configuration-only, not universal bitwise CUDA training.
  - Original CPU, prelaunch, hard-limit, helper calendar/publication false-green, runner and document-review failures remain failed; new bounded stages never retroaccept them.
  - Independent terminal hashes four newcheckpoint bytefiles but does not loadweights or reread rawsource/store/weather; runtime prepare/training/scoring separately qualify those inputs.
  - Whole supervisorGPU-h includes metadata, botharms, evaluation, reading, finalinventory and reaping; CPU preparation/review and archive copies separately reported with zero GPU.
  - Shared-store split isolation follows frozen train-only dataset/case paths, not a field-index OSsandbox; socket/import/owned guards are cooperative.
- CI run: `37664236520`
- Excluded from runnable candidates: Predeclared necessary development support false; only t2m72 improves both controls while other primary/guard cells regress. Stop this ordering instance, no extra seeds/dose/fullval, remain S3/test0/r0.
- Recorded metrics (not recomputed): all16_train_eval_final_hash_equal=true, arms=["blocked", "interleaved"], bundle_manifest_sha256=05f71e52741b266fab3206ec14f6c0a4d72059c2544fe9e436ee249b14ba87ad, cleanup_reserve_seconds=180, confirmation_r_consumed=0, development_cases=4, development_defined_cells=20, development_necessary_support=false, development_no_worse_both_cells=5, document_supplement_receipt_sha256=b602d2c79e84961ed6466a7bd20af0266b81472fdbd9f9a0062805ed9c9c3c3e, elapsed_seconds_total=2464.990661121905, exposures_per_case_per_arm=20, external_pins_sha256=922a87d5877ec24c0e2306c19c3ade7713f6878fa62e22891e4d4309f2087ebe, gpu_hours=0.6848, hard_cap_seconds=14400, hard_overrun_seconds=0, historical_case80_or_control0_recharged=false, inner_elapsed_seconds=2461.9399664048105, known_reserved_peak_bytes=2491416576, lead_hours=[6, 12, 24, 48, 72], minimum_free_bytes=4638900224, model_parameter_count=3286037, optimizer_updates_per_arm=80, optimizer_updates_total=160, owned_train_reserved_peak_bytes=2434793472, physical_steps=12, planned_seconds=5400, prelaunch_ack_sha256=373e62545f4a476746b62a772f87976fa6c88babb926f2a7828b6f20749de25e, qualified_prelaunch_tests=49, reasoning_steps=4, regions=["full", "interior_1", "edge_1"], result_sha256=24a35359b2c8d0fa701655ae2c45c28d7a0cd2c76e470ee194d68d44845ab723, scoring_pairs=16, seed=41, shared0_additional_model_computations=0, shared0_reused_pairs=8, soft_overrun_seconds=0, source_bundle_sha256=2f029b0518c33be3e42b144cea69eea094c9ccf7748c6edc17ec09aaec738ba3, source_network_requests=0, terminal_allowed_files=947, terminal_helper_manifest_sha256=ff88929bbde22765c0aa3d514ae6ab7a78c7137b5ecb801700e0ffb259116da6, terminal_receipt_sha256=44a6e844662790b38740c4158a74bc274907137057b541b4b5cc4fab13a405c2, terminal_regression_tests=72, terminal_result_sha256=0cca2beacc0e88cbb78b6b37f4cc874d23f44e7cdb3c458d52b47cc621c22cfd, test_read=false, train_cases=4, train_final_improved_vs0={"blocked": 241, "interleaved": 302}, train_final_worsened_vs0={"blocked": 99, "interleaved": 38}, val_final_improved_vs0={"blocked": 132, "interleaved": 151}, val_final_worsened_vs0={"blocked": 208, "interleaved": 189}, val_t2m_rmse_blocked_K=[1.8448277936449216, 1.9908937525090833, 2.5884560322006123, 3.769138460189457, 4.464696428043901], val_t2m_rmse_interleaved_K=[1.8653973872943392, 2.075465740327451, 3.0144025891756923, 4.007737849921529, 3.9709113546608408], val_t2m_rmse_shared0_K=[1.9374453033540768, 2.041813066488836, 2.6784736372973894, 3.744068532894296, 4.411827847107436], val_t2m_skill_interleaved=[0.43348718877140124, 0.2152684073101787, 0.28814062084120495, -0.530063570237739, -0.7893378003788789]

## s3-climatology-anomaly-anchor

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_CLIMATOLOGY_ANOMALY_ANCHOR.md` (SHA256 `b50e4f968d7b6e80f0d1b770d2860f46136d102556d1bd688f3e4c26889b4a2f`)
- Evidence commit: `ef7302a03d3f2e1b3301780616052bab61ce1f0d`; experiment commit: `abe5439d3444a89ee569aef3172588dc92ec9ca8`
- Protocol SHA256: `d5ecdc89b4b570ffd875f616fb161fc390ea4e5f52e0d352420a3cc85c5b2a21`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: One predeclared train-only climatology-anchor package screen completed exactly80 fresh AdamW updates (seed41, four 2021 train cases x20 exposures, original K4/full12-step/FP32/deep-supervision objective) plus original four 2022 development cases scored for candidate0/80, with oldshared0/oldinterleaved80 reused by saved byte pins. Necessary development support is false: t2m 48/72h improve versus both controls while 6/12/24h and several guard variables regress (20 defined gate cells, 7 no-worse-both). The saved-record engineering readback then completed on the v6 chain (admission 1500/1500, 1,946,575,475B, three original hash passes equal, 13 future ENDs equal, 0 additional model executions, no failure record) verifying the saved facts against the immutable model without any scientific acceptance. Negative single-instance screen; no extra seed/dose/fullval, remain S3, test unscored r=0.
- Limitations:
  - One seed and four repeatedly fitted training / four already-exposed development cases; no annual generalization, simultaneous intervals, formal candidate or scientific acceptance.
  - The package screen changes output prior, proposal prior and feedback representation together; no isolated mechanism attribution or exact X-minus-C compensation proof.
  - Saved-facts engineering readback only: no weather/source arrays, no checkpoint or table tensor deserialization, no GPU, no network, no model/table execution; opaque subtraction content and unrecorded OS actions are not proved.
  - Historical arithmetic residuals between native FP32 and FP64 reconstructions remain; configuration-only, not universal bitwise reproducibility.
  - Retained failures (attempt03 preflight, attempt04 GLOBAL<=1500, attempt05 qualifier six-arg call, blocked v5 chain F-RV6-01) are preserved and never retro-accepted; the v6 chain does not ratify them.
  - The 1500 labels/paths boundary was exercised at exactly 1500/1500; the caps were not and must not be relaxed.
  - Frozen online and cumulative read caps remain external governance; the whole-round conservative GPU-hours includes metadata, evaluation, archive copies and reaping overhead, not literal kernel time.
- CI run: `37789073741`
- Excluded from runnable candidates: Predeclared necessary development support false for the train-only climatology-anchor package screen (t2m 6/12/24h and guard variables regress versus both controls; only 48/72h improve). Stop this instance; no extra seeds/dose/fullval; remain S3 with test unscored r=0. The completed saved-record readback is engineering verification of saved facts, not scientific ratification.
- Recorded metrics (not recomputed): ack5_sha256=cc2511a764f7a5d61c1e490075bc8fc034a052e04c6b1ee592ff0ac9d1248639, ack7_sha256=1969a5128790c7118a399eab2c31d148190da80b7f1f1e125ddaabae700b66f3, archived_new_code_zip=e42e9bf67df954846669491db4c8ff9733b6b28a6ef55f7da15d6d187cd0bab7, attempt_sha256=b5e7625ff6fe937c371a8cd8bcf5a18ac35c4007908ce44e34bd9c09928d9e5f, checkpoint_loads_in_readback=0, cleanup_reserve_seconds=180, confirmation_r_consumed=0, development_defined_cells=20, development_no_worse_both_cells=7, development_strict_better_both_cells=7, development_support=false, document_review_receipt_sha256=cde3032ad8dfe65ddadba5a24b8eb092481d80836f5c8e3d469791234595c6eb, elapsed_seconds_total=2468.1777711212635, endpoint_updates=[0, 80], external_pins_sha256=ad05dc5b0d7c95a6dfaf8245c320351712ec9a2aa483fd18415729692d14832c, extra_outer_cleanup_seconds=360, gpu_hours=0.6859, hard_cap_seconds=14400, hard_overrun_seconds=0, network_requests_source=0, new_model_digest=d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab, new_pairs=16, new_training_digest=6e4363d5707ea4935d5449c800e4fecefa99bf553b42a747dd4dad37f8d548db, optimizer_updates=80, physical_steps=12, planned_seconds=5400, prelaunch_ack_sha256=53b60fee941ca15e18f94d1b39ac2cca9c121bd14223ce6ac82b9637654540cf, producer_bundle_sha256=dd519f6c4f72bef27b94f2974c34b61464c4cbb278306ff6eee94f09248e1ecf, producer_manifest_sha256=9100abb886b18c3fe7ab2b54cbaeef074c862fd72ebec67f1307a64cf6c4007d, readback_additional_model_executions=0, readback_archive_manifest_sha256=4f0148dfebe7c818bf92d4abe6e079bdb457d225cda889a69d637bdd7748830a, readback_audit_fit_calls=0, readback_clock_resets=0, readback_completion_sha256=c06cfe68af17b1f9098a38996b41c23dc094230f9f126cddc43d1759b9c3237d, readback_elapsed_seconds=40.22008040826768, readback_future_all13_FIRST_END_equal=true, readback_global_labels=1500, readback_global_paths=1500, readback_parent_exit=0, readback_parent_whole_elapsed_seconds=40.59019962884486, readback_receipt_sha256=dd6204554b3057328545ab84efff09359a464ef5e4f17a7ebd245fdf3a31371f, readback_source_FIRST_END_equal=true, readback_tensor_deserializations=0, readback_three_original_hash_passes=true, readback_total_read_bytes=1946575475, reasoning_steps=4, result_sha256=7f83e7c7d2d9144c06f56082cffd6c2018be21fed0c91202b53bc2a042d762c2, retained_blocked_chains=1, retained_failed_attempts=3, reused_control_pairs=16, rv6_block_ack_sha256=803862c02813dc839c4f1c3fa4185cd531612909f367ea3c09150b6d16216788, scoring_calls=384, scoring_draft_cells=16320, seed=41, soft_overrun_seconds=0, source_audit_sha256=a3b940ec89b11dca8a433c2574b9f586782d7488302d5e19c37f75dcaa87ed6d, t2m_candidate80_mse=[8.201087281145625, 5.079027133109043, 17.952668386510403, 13.708107694174556, 8.769223385478101], t2m_lead_hours=[6, 12, 24, 48, 72], t2m_oldinterleaved80_mse=[3.479707412524547, 4.3075580392729735, 9.086622969629119, 16.061962673693643, 15.768136986574392], t2m_oldshared0_mse=[3.753694303488771, 4.169000598484543, 7.174221025697108, 14.018049179009246, 19.464224952512634], table_queries_including_training=1344, test_read=false, train_cases=4, v6_source_freeze_sha256=8b7565ed845c30012222c3af902a8cf4437cac87c815f430769904346052a404, val_cases=4, weather_reads_in_readback=0, whole_outer_receipt_sha256=0e45dac5f6c2f809090baeeb4d2beb35503f6c2ec4a17253b6ff46e89795afd3

## s3-fixed-case-objective-response

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md` (SHA256 `fbfd83ee84ceecdcaac8f93250e564fde9e0ac57c3d71f87b0ae4cb3f68e0742`)
- Evidence commit: `6b2e3db541797708dc1ebb916b39ea7d9c02b91a`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `699ee5e5fbb67369185ea7824d9e5f33a14b3dce3cf9226dac3facffd0a0fa0f`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: One predeclared January2021 in-sample train case completed exactly80 fresh AdamW updates under the original all17 normalized deep-K/K4/12-step full-BPTT objective and clip1. Endpoints0/20/80 total1.29634428024292/0.774080753326416/0.5330567359924316; every12-step loss decreases across endpoints. Full external contract, seed41 native RNG and original per-parameter activity anchors,80-row arithmetic/schedule/activity, three standard checkpoint semantics and saved whole-process/cost facts checked. This finite response is descriptive, not generalization, climatology skill, convergence or clipping causality.
- Limitations:
  - Single in-sample cached case and seed41; normalized objective is not physical weather RMSE or climatology skill.
  - New80 cosine schedule and fresh AdamW, not seamless resume of original200 schedule; parent optimizer/cursor/RNG not imported.
  - No proof of generalization, total capacity, underfitting, convergence or clipping harm; no scientific/candidate threshold added.
  - Non-atomic private exclusive checkpoint publication retains partial files on failure; ordinary loader and original guards unchanged.
  - All interrupted/negative/over-hard/reserve-failed engineering reviews remain failed; new bounded completion never ratifies old stages.
  - Saved-fact/CPU semantic checks are not global OS or GPU lifetime tracing; gate timestamps absent; reproducibility at most configuration-only.
- CI run: `37543875564`
- Excluded from runnable candidates: Descriptive train-only response diagnostic; not runnable candidate, S4, independent confirmation or scientific PASS.
- Recorded metrics (not recomputed): active_parameters=141, attempt_sha256=43d18c3e493dd1496ef9de3c18a4e08457a4b634844d3b34bddbb0c833751650, checkpoint_completion_receipt_sha256=b2c0276818f4aa188b4afe82c686c31e87a334c375e463dfe5eacbd4661d5ad2, checkpoint_sha256=["298eccd8b8c4b9648c3214e7d860c2ea8dbd7a5dc620bc15931762c5dd42b6b0", "c982db31832d498f1ff08d833efc2e222402ab18833115fb699da1371f6eb846", "98cbd2d51bb5ce7c3b55d544b7b234e91397d06c02d2ed328201dce6ab35c9e9"], confirmation_r_consumed=0, elapsed_seconds_total=1547.848237130791, endpoint_per_step_losses=[[0.19521258771419525, 0.30220484733581543, 0.35116615891456604, 0.41650983691215515, 0.4893997311592102, 0.5351357460021973, 0.6573091149330139, 0.7724584341049194, 0.8372243642807007, 0.7294480800628662, 0.7245166897773743, 0.711090087890625], [0.1793542206287384, 0.2653624415397644, 0.2694452106952667, 0.295052707195282, 0.33120232820510864, 0.3478480577468872, 0.3579563498497009, 0.34115666151046753, 0.3599538207054138, 0.29754096269607544, 0.24072617292404175, 0.2878812253475189], [0.1504697948694229, 0.1961665004491806, 0.1869969666004181, 0.20198631286621094, 0.2431214153766632, 0.2563667893409729, 0.2488587349653244, 0.21179020404815674, 0.2569209337234497, 0.24375811219215393, 0.1668146699666977, 0.15523090958595276]], endpoint_total_losses=[1.29634428024292, 0.774080753326416, 0.5330567359924316], endpoint_updates=[0, 20, 80], final_document_review_sha256=b1ae94ff2c5971224f7787d59c52cf58a0b8f9d4b44d4c6b00cea80bbfe4af6c, gpu_hours=0.43, hard_cap_seconds=7200.0, hard_overrun_seconds=0.0, independently_assigned_files=807, known_reserved_peak_bytes=2491416576, last_before_step_loss=0.5335893630981445, network_bytes=0, optimizer_state_imported=false, optimizer_updates=80, owned_cuda_reserved_peak_bytes=2434793472, physical_steps=12, planned_seconds=3600.0, preclip_gt1_rows=42, reasoning_steps=4, required_free_bytes=4638900224, result_sha256=a59614df732c0f090e26798bcc2fa084252a7ab26ab7e606939d0e483840e3b7, seed=41, soft_overrun_seconds=0.0, support_sha256=7e4540b42debde4befdda247edcbc4c18b81af25c0fa862aa5f6d297bf75ba86, test_read=false, total_decline_percent_0_to_80=58.88000247183234, train_cases=1, unused_parameters=14, val_read=false, wrapper_sha256=3073ea784dbd76a44c1f88adb1329c66ce120194235e18120425029467da97f8

## s3-fixed-case-objective-response-readback

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_FIXED_CASE_OBJECTIVE_RESPONSE.md` (SHA256 `fbfd83ee84ceecdcaac8f93250e564fde9e0ac57c3d71f87b0ae4cb3f68e0742`)
- Evidence commit: `6b2e3db541797708dc1ebb916b39ea7d9c02b91a`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `6b9be74b035a19f47407eb74158e342ec15e7c4f37bfe0f4271c6eb9920801a6`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: All three predeclared real endpoints0/20/80 were independently recomputed from the unchanged archived training_long_rollout on the same cached train case, keeping train/all-trainableFP32/enable_grad/dropout0. Native typed exact complete12-loss/total/state-after/flags/inventory/recipe/RNG projections have zero differences without tolerance, optimizer or backward. External115 parent pins/full new80 contract/all692 archive and1532 saved-file/process/gate/cost facts checked. Whole1467.480233s costs0.4078GPU-h independently of the original training; prior terminal hard900 overrun retained, fresh byte-identical auditor qualified only assigned saved facts.
- Limitations:
  - Exact saved endpoint projection in this configuration is not universal bit-reproducible GPU training.
  - One in-sample train case, no val/test/weather skill, climatology fit or scientific confirmation.
  - Independent saved-file auditor did not reread weather/source/model tensors or rerun forward; actual execution qualified those inputs.
  - Old scope-negative and hard/reserve failures remain negative; fresh bounded same-helper completion applies only to its declared assignment.
  - Python offline/import/owned guards are not an OS sandbox; saved signal/headroom facts do not establish global lifetime intervention absence.
- CI run: `37543875564`
- Excluded from runnable candidates: Configuration reproducibility and assigned saved-facts audit only; not candidate or scientific acceptance.
- Recorded metrics (not recomputed): attempt_sha256=e1e23874d804bf3a7ce255b7f9633970809f917592eaeecb2832fd7e69218a8e, confirmation_r_consumed=0, difference_counts_by_endpoint=[0, 0, 0], driver_sha256=4df86db1f6ea8c4a7f7d4b6e0e4c42259e3c8d4e64fda80419e231e7b07fe517, elapsed_seconds_total=1467.4802325293422, endpoint_updates=[0, 20, 80], final_document_review_sha256=b1ae94ff2c5971224f7787d59c52cf58a0b8f9d4b44d4c6b00cea80bbfe4af6c, fresh_terminal_audit_seconds=5.029937160201371, gpu_hours=0.4078, hard_cap_seconds=3600.0, hard_overrun_seconds=0.0, historical_parent_training_cost_recharged=false, known_reserved_peak_bytes=2491416576, network_bytes=0, numeric_difference_count=0, optimizer_created=false, optimizer_updates=0, original_terminal_hard_overrun_seconds=48.49717978853732, owned_cuda_reserved_peak_bytes=2409627648, parent_assigned_files=807, parent_files_pinned=115, physical_steps=12, planned_seconds=1800.0, readback_assigned_files=725, reasoning_steps=4, required_free_bytes=4638900224, result_sha256=663e085a2c27514beb9af6eb5740264361307101876333047e372c3077585372, seed=41, soft_overrun_seconds=0.0, support_sha256=f4408d2ed919998c04f901b92fac3ac800c9de2aa35452f62767ce1bb22f685d, terminal_completion_receipt_sha256=fa8d67c2d52f41224140decbacd68effc46111c3fa52bf1b91afb67ec651fd62, test_read=false, train_cases=1, val_read=false

## s3-gradient-mixed-suite-resource-deviation

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` (SHA256 `627b324d60f08f442592a3430113f088bb356d4dccf1b6431cbbadc5f69bbdba`)
- Evidence commit: `6c3bb69696b5d4855da81e9b15eb9f5dba3acbe9`; experiment commit: `aa9e4ac0c43f5f1295233d9b7446a34803736ea3`
- Protocol SHA256: `not recorded`; data identity: `not recorded`
- Reason: Default full pytest executed six CUDA-conditional engineering tests and4009passed/3skipped, but no dedicated startup UUID/headroom or GPU-study protocol receipt was retained. Truthfully registered as engineering test success with resource-compliance evidence incomplete; entire observed interval conservatively charged. Explicit CUDA-hidden bounded CPU repeat4003passed/9skipped does not retrospectively fix the initial gap.
- Limitations:
  - No dedicated GPU startup gate/protocol was retained for this default full pytest run; do not claim R-054 or bounded-study compliance for it.
  - Entire observed pytest interval conservatively charged; raw GPU utilization and startup shell time before JUnit were not independently measured.
  - Six CUDA-conditional engineering tests ran; passing does not establish scientific weather skill or general GPU bitwise training reproducibility.
  - No evidence of neighbor intervention was found, but global lifetime absence of signals was not traced.
  - A separate bounded CUDA-hidden CPU verification is underway; it does not retroactively fix the original execution gap.
- CI run: `37559461860`
- Excluded from runnable candidates: Engineering test accounting with resource gate/protocol evidence gap; not compliant GPU-study or scientific evidence.
- Recorded metrics (not recomputed): conservative_ceil_seconds=1259, cpu_repeat_passed=4003, cpu_repeat_skipped=9, cpu_repeat_whole_seconds=1206.472975098528, cuda_condition_tests_executed=6, dedicated_startup_gate_receipt=false, deviation_receipt_sha256=07023eb8d925ec0ba926ec4065a174b6ba9d67215bd986a9f8a4b505bbfbcc17, elapsed_seconds_total=1258.3494682312012, gpu_hours=0.3497, passed=4009, protocol_frozen=false, reported_pytest_seconds=1257.87, skipped=3, warnings=6

## s3-long-rollout-feasibility

- Outcome class: `engineering-positive`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_LONG_ROLLOUT_FEASIBILITY.md` (SHA256 `90fa24c695159971961138d4e84c9a2ade3d3266a9b5f3768c219253651766ae`)
- Evidence commit: `5abc3da3c8aaae3af76ad8d68f5a987db213b0cc`; experiment commit: `66836d29dd257a11b0c946c8af2622b35042a5ce`
- Protocol SHA256: `78b96197cf0021f36ea326e351b3ac5e7762d753b95da00595751f53d6f27ee5`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: One train-only FP32 K4 twelve-step full-BPTT feasibility sample from registered seed41 parent1600 completed with finite loss/gradient; no optimizer update or val/test score. Independently verified source/data/window/archive/checkpoint/scope/cost identities. Whole round1605.013031s, not7.454s measured GPU section. Co-resident reservedpeak2.244GiB requires4346MiB subsequent spawn headroom. Not scientific screening or candidate acceptance.
- Limitations:
  - One train sample/no optimizer update, not200updates/three-seed validation or scientific confirmation
  - Four30-day seasonal blocks peryear are not complete-year coverage; no test scored
  - External actualC non-Python configuration remains a pinned separate dependency, absent fromcodezip; derivedchunks notexhaustively rehashed
  - Highest config-reproducible, not bitwise training; supported aten FLOPs omitoperations; CUDA peak belongs to ownedworker
  - Independent audit soft600 exceeded268.864359s underhard1200; costs retained
- CI run: `37500335847`
- Excluded from runnable candidates: Engineering feasibility only; no weather skill, no S4/test/goal acceptance. Separate compatible screen still required.
- Recorded metrics (not recomputed): audit_soft_overrun_seconds=268.864359, audit_wall_seconds=868.864359, bf16=false, elapsed_seconds_total=1605.0130310487002, forward_backward_flops=393859201536, gpu_hours=0.4458, gradient_norm=40.788814544677734, hard_cap_seconds=3600, hard_overrun_seconds=0.0, loss=1.2537332773208618, measured_gpu_section_seconds=7.454262489452958, network_bytes=0, next_spawn_required_free_bytes=4557111296, owned_cuda_allocated_peak_bytes=2274339328, owned_cuda_reserved_peak_bytes=2409627648, physical_steps=12, physical_weights=[1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5], planned_seconds=1800, reasoning_steps=4, soft_overrun_seconds=0.0, test_read=false, train_windows_excluded=220, train_windows_input=2360, train_windows_usable=2140, workers_exit0_reaped_no_signals=3

## s3-long-rollout-feasibility-replay

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_LONG_ROLLOUT_FEASIBILITY.md` (SHA256 `90fa24c695159971961138d4e84c9a2ade3d3266a9b5f3768c219253651766ae`)
- Evidence commit: `5abc3da3c8aaae3af76ad8d68f5a987db213b0cc`; experiment commit: `66836d29dd257a11b0c946c8af2622b35042a5ce`
- Protocol SHA256: `8f7e2f6ce24dc58b5f06f351bb5d6393c18dd47035910db87ff1c0b8f5a6c3a8`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Archived exact66836d2 one train sample forward/backward replay completed in a separate fresh output. SampleID/loss/all12steplosses/gradientnorm/supportedFLOPs exactly equal original; source/data identities rebound inside each owned worker. Both workersexit0/reaped/nosignals. Independent terminal28checks agree. Whole846.060163s fully charged. No optimizer/val/test scoring or new scientific evidence.
- Limitations:
  - Single train sample, not training/evaluation/three-seed weather confirmation
  - Observed exact arithmetic not cross-platform bitwise reproducibility; actualC externalconfiguration limit retained
  - Terminal audit deliberately inherits previousfullsource/window/checkpointinventoryaudit, notindependent runtime tracing
- CI run: `37500335847`
- Excluded from runnable candidates: Identity/configuration reproducibility audit only, not a runnable candidate or scientific PASS.
- Recorded metrics (not recomputed): elapsed_seconds_total=846.0601630983874, exact_field_count=5, gpu_hours=0.235, hard_cap_seconds=3600, hard_overrun_seconds=0.0, network_bytes=0, owned_cuda_reserved_peak_bytes=2409627648, per_step_loss_count=12, planned_seconds=1800, replay_driver_sha256=7e71f0409e015963c8a48e532b28fb796c8c9511d761f3fb310bcb4fe012c2c0, required_free_bytes=4557111296, restoration_accepted_sha256=3b3680057e75cbceda14f359124961faf4b9f83e0b6ee6befed86a85331cd152, soft_overrun_seconds=0.0, terminal_audit_checks=28, terminal_audit_wall_seconds=85.756, test_read=false

## s3-long-rollout-screen

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_LONG_ROLLOUT_SCREEN.md` (SHA256 `aeec87e66c0b60a285878dd49a847654e8655625118f079f3a607bc0c5f9ef4b`)
- Evidence commit: `d1dcd46b8deb9591cc5e1108afef3c4b46249750`; experiment commit: `66836d29dd257a11b0c946c8af2622b35042a5ce`
- Protocol SHA256: `a1631d9b87721c2260a28c5862ef26ae7245fdd5f7c30cd404cb05708a8d10ea`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Complete three-seed FP32 K4 twelve-step full-BPTT screen at200 freshAdamW updates from registered ownBD1600parents. Original development primary6/12h allstrictnegative and controlgate0/45positivecells; all255RMSEcells belowD3400control, frozen decision advance-to-S4-freeze means only independentfreeze-package eligibility. Absolute t2m48/72hclimatologyskill remainsnegativein everyseed; parentrelativegate16/45positive6/12cells. NoS4/test/scientificacceptance. Independent state/765RMSErow/case/cost audit and archivedseed41exactreplay support recordedconfiguration, notgoalcompletion.
- Limitations:
  - Development support vsD3400control only; everyseedt2m48/72h remainsworsethantrainonlyclimatology,72h15/17variablesnegative
  - No unseencompleteyear orseason/yearsimultaneousintervals, nointerior/edgeconfirmation and2023testnot scored
  - Notcompute-matched; parent1600L6plus200longupdates andnewwindowselection, no cleanmechanismcausalclaim
  - Parent-relative16gatepositivecells retained; highestconfig-reproducible, notbitwiseGPUtraining
  - Source/norm/windowqualificationinheritedfromregisteredprobe; baselinefitfieldsnotregeneratedinreadonlyaudit; externalactualCconfignotinarchive
  - Wholepublishedwallincludespreparation/loading/reading;publicationtail/outerexitnotindependentlytimed; auditfailedpartialandsoftoverrunretained
- CI run: `37500335847`
- Excluded from runnable candidates: Needs independent development/freeze-package review, not final confirmation-ready candidate. Absolute48/72hclimatology failure, incomplete-year data and missing simultaneous statistics prohibit goal/S4 test acceptance.
- Recorded metrics (not recomputed): audit_unique_wall_seconds_before_replay=2262.372, bf16=false, candidate_lower_rmse_vs_control_cells=255, decision=advance-to-S4-freeze, delta_t2m_rmse_k={"41": {"12": -0.546425404894, "6": -0.432457996959}, "42": {"12": -0.736492184981, "6": -0.763970550907}, "43": {"12": -0.498234, "6": -0.436941976038}}, elapsed_seconds_total=8994.237921394408, gate_failures=0, gate_passed=true, gpu_hours=2.4984, hard_cap_seconds=12600, hard_overrun_seconds=0.0, lr=2e-05, mode=long_rollout, network_bytes=0, next_spawn_required_free_bytes=4582277120, owned_cuda_reserved_peak_bytes=2434793472, parent_relative_gate_failures=16, parent_updates=1600, per_seed_seconds=3600, physical_steps=12, physical_weights=[1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5], planned_seconds=6300, positive_climatology_skill_cells_of51={"12": 51, "24": 51, "48": 22, "6": 51, "72": 6}, primary_verdict=supported, reasoning_steps=4, seeds=[41, 42, 43], soft_overrun_seconds=2694.2379213944077, t2m_skill_vs_climatology_rounded={"41": [0.587811, 0.312714, 0.285808, -0.414955, -0.967931], "42": [0.587259, 0.313789, 0.267697, -0.360455, -0.788045], "43": [0.597834, 0.32985, 0.327613, -0.226734, -0.609788]}, test_read=false, updates_per_seed=200, validation_case_records_three_arms=20448, warmup=10, workers_exit0_reaped_no_signals=6

## s3-long-rollout-screen-replay

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_LONG_ROLLOUT_SCREEN.md` (SHA256 `aeec87e66c0b60a285878dd49a847654e8655625118f079f3a607bc0c5f9ef4b`)
- Evidence commit: `d1dcd46b8deb9591cc5e1108afef3c4b46249750`; experiment commit: `66836d29dd257a11b0c946c8af2622b35042a5ce`
- Protocol SHA256: `e1ca165ceda0c3dc06303bcf51cd1bc95a2f58666d2dc2f3f6b8cddb37daa789`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Archived exact66836d2 seed41fivefullvalleads regenerated:15CSVhashesidentical,2272casesand38624physicalMSEscalars exact,all3savedreadingscanonicaldigest matchedarchivedcollector. Mandatoryindependentpins, safeverifiedbytearchiveimports, ownedboundedworkersandknownpeakguards. Whole1935.462220s softoverrun135.462220s,ceil1936secondsfullycharged0.5378GPUh. Independent153checksconfirm directnumbers; auditoptionaldoctypefinalizationhardoverrun3.402sreportednotbudgetcompliant. No training/test/newscienceclaim.
- Limitations:
  - Actualforecastreplayseed41only; remainingseedsrecordedmetric/stateaudit, no training replay
  - ObservedexactCSV/configurationnotcrossplatformbitwisetrainingreproducibility or scientificconfirmation
  - Collector publishes recomputeddigestnotfullpayload; independentterminalauditdidnotreruncollector
  - Coreauditterminalchecks569.297sunderhard600butoptionalnumericdocfinalization603.402sovershot3.402s; notallbudgetcompliant
  - ExternalactualCconfiguration/sourcefield/chunkqualification limitsretained
- CI run: `37500335847`
- Excluded from runnable candidates: Identity/configuration reproducibility audit only; no newscientificcandidate orPASS. Auditbudgetexceptionretained.
- Recorded metrics (not recomputed): audit_checks=153, audit_final_wall_seconds=603.402, audit_hard_overrun_seconds=3.402, audit_soft_overrun_seconds=303.402, collector_canonical_sha256=aa65dac1117c0ba6f109a70cf383a825f95bbeda3918f179cefc4c47da5f09cc, conservative_ceil_seconds=1936, elapsed_seconds_total=1935.4622196508572, exact_case_count=2272, exact_csv_count=15, gpu_hours=0.5378, hard_cap_seconds=3600, hard_overrun_seconds=0.0, mse_difference_count=0, network_bytes=0, physical_mse_scalar_count=38624, planned_seconds=1800, replay_driver_sha256=495fdff90c55237131faa350f966da7dc37201b8301fc32a22ad347a85259ff1, restoration_accepted_sha256=734a6746555c15c91f93e2c21565c7393975cf2da57bc0e29d55770c450e642c, soft_overrun_seconds=135.46221965085715, test_read=false

## s3-objective-forecast-diagnostic

- Outcome class: `mixed`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_OBJECTIVE_FORECAST_DIAGNOSTIC.md` (SHA256 `ff02acac4fb87160bbb1eafd82bdbd1465fd4e724b5c5ded1afec0ac55a35a58`)
- Evidence commit: `0befd262f971318cacbcae627d62a16a6e5723b4`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `26a17743b50c159b4b5e1e70703c89c3a9dd0ee56aaf4efebe4e687663d2a59d`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Complete no-update0/20/80xoriginal four2021train+four2022development paired objective/final-score diagnostic under original archived deepK and ordinary eval paths. Fitted January objective falls58.8800% with83/85 final cells improved at80, but all other3train and4val objectives increase; val t2m allfive pooled leads worsen and48/72 climatology skill remains negative. All24 train/evalfinal tensor hashes exactly agree in this run. Independently checked saved24pairs/288 native losses/24480 draft cells/1530 aggregate cells per path, allsource/protocol/case/state/count/mse/acc/gate/cost receipts,754file first/end hashes, no mismatch. Full whole-cost0.5256GPU-h, not candidate or generalization.
- Limitations:
  - Eight already exposed representative cases and one seed are descriptive; no independent annual confirmation, statistical significance or scientific acceptance.
  - Single-case fitting response with cross-case negative response does not identify memory, capacity, optimization or seasonal-interference cause.
  - NativeFP32 objective versus FP64 decomposition retains arithmetic residuals, not universal bit reproducibility; configuration-only.
  - Independent terminal auditor read only allowed saved bytes, not true weather/source/store/checkpoints or new forward; runtime workers separately qualified real inputs.
  - Original33FALSE_ACCEPT, reviewer environment failure, helper schema failure and provider errors remain preserved; new limited qualification does not relabel prior runs.
  - Whole supervisorGPU-h includes metadata/checkpoint/field/scoring/inventory/reaping overhead; CPU preparation/review separately reported, not added as fabricated GPU cost.
  - Python network/import/owned guards are accidental-use protection; signals/gates do not establish global absence of neighbor interference.
- CI run: `37543875564`
- Excluded from runnable candidates: No development support for single-case80 as a performance candidate; fitted-case response only, othertrain/val majority negative. Remain S3 and test/r unchanged.
- Recorded metrics (not recomputed): backward_calls=0, bundle_manifest_sha256=6ca336511c8eda4d2d04f8bd34806387db1af0cc3dd0546b487ca5f649714254, cleanup_reserve_seconds=180, confirmation_r_consumed=0, elapsed_seconds_total=1891.3772793579847, endpoint_updates=[0, 20, 80], external_pins_sha256=31b7d5d8a4afc2c24cc4728c65391e8c91022e1156914ebba441027544a70959, fitted_case_80_improved_full_cells=83, fitted_case_80_worsened_full_cells=2, gpu_hours=0.5256, hard_cap_seconds=10800, hard_overrun_seconds=0, historical_case80_or_readback_recharged=false, inner_attempt_elapsed_seconds=1888.921116472222, known_reserved_peak_bytes=2491416576, lead_hours=[6, 12, 24, 48, 72], minimum_free_bytes=4638900224, objective_change_0_to80_by_case=[-0.5888000247183234, 0.5623749620971819, 0.14178963349940266, 0.1825658410765092, 0.38310849583872586, 0.935123141808105, 0.08912729696596933, 0.744223936074957], optimizer_created=false, optimizer_updates=0, other_train_80_improved_full_cells=73, other_train_80_worsened_full_cells=182, owned_reserved_peak_bytes=2409627648, pairs=24, physical_steps=12, planned_seconds=3600, prelaunch_receipt_sha256=600a7abf8b1e98f05a7b19ffc8e4a73e88cf2c6b9360ec5187122261a4999aff, preparation_gpu_hours=0, reading_receipt_sha256=5bdb162152b736beba1436e377df2a7a3acc94ceee1862c2b1979330ef5922f4, reasoning_steps=4, regions=["full", "interior_1", "edge_1"], result_sha256=ca8919139c6999d9a7c2532ee91e2e2eae6dd37fb830f424ab5d5a12999e9d9a, seed=41, soft_overrun_seconds=0, source_network_requests=0, terminal_allowed_files=754, terminal_receipt_sha256=c60add04a8cd1952b637524fb8b0383b0183dd6a8dfd07c3adec2ec7f83eff0e, terminal_result_sha256=62df597aab9c1c5bf7db19fd51bc34251c433d92533cd08a392f851b79ab2f4d, test_read=false, train_cases=4, train_eval_final_max_abs=0, val_80_improved_full_cells=99, val_80_worsened_full_cells=241, val_cases=4, val_read=true, val_t2m_80_improved_case_lead_cells=2, val_t2m_80_worsened_case_lead_cells=18, val_t2m_rmse_update0_K=[1.9374453033540768, 2.041813066488836, 2.6784736372973894, 3.744068532894296, 4.411827847107436], val_t2m_rmse_update80_K=[1.9826350170853602, 2.096059658504984, 3.331547886686013, 5.5678146122562415, 7.367494364417267], val_t2m_skill_update0=[0.3888808281097974, 0.2405101800727535, 0.4379609958099575, -0.3353602427262241, -1.2087627404722974], val_t2m_skill_update80=[0.3600404092851308, 0.19961806378928681, 0.13047149565592742, -1.953115244876117, -5.159586763509309]

## s3-rollout-dose

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_ROLLOUT_DOSE.md` (SHA256 `0568acef6c1d0ad127e38b6a1a22c8aada583b9886924b740bf9b3233613c77c`)
- Evidence commit: `73459af78b8b2484a3291ef43439470d09600b00`; experiment commit: `7e384dbeb6a7061a6cfa1a3e1898d44863f442eb`
- Protocol SHA256: `655fec891491dd42c0d2bc840160bf6f72ceacf13b091d5bc105cd94cc7a74c9`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: S3 rollout-dose screen: the registered 200-update rollout screen had already annealed its cosine schedule to the 0.1x floor, so this round changes one factor only - 800 rollout updates instead of 200 at the identical parent, 12-step weights, LR 2e-5, warmup 10, FP32 K4, batch 1 and clip 1. The frozen decision functions read primary supported (t2m/full 6h and 12h, all three seeds negative vs the pinned v3-D3 400-update control) and the u10/v10/mslp gate 0/45 positive cells, decision advance-to-S4-freeze - the same frozen form as the registered 200-update screen. The absolute climatology gate is still not met: t2m/full seed-mean skill 6h +0.5816, 12h +0.3151, 24h +0.3668, 48h -0.1191, 72h -0.4309, and the linear-rescaling ceiling ACC^2 is only +0.1355 and +0.0366, so no further dose alone can clear 48/72h. The dose still pays: versus the 200-update screen the 24/48/72h t2m RMSE drops 0.1536/0.3248/0.4653 K while 6h/12h stay flat within 0.026 K, and the 48h/72h anomaly correlation rises 0.329->0.367 and 0.153->0.189, so the gain is pattern, not only amplitude; all 17 variables improve at 24/48/72h. The attempt's frozen reading worker failed on the shared collector's parent-pin check after the model identity change (all five earlier phases succeeded, worker reaped, no signals, failure preserved and charged in full); the readings come from a read-only tool that delegates to the same frozen verdict functions and records the deviation. The registered v3-BD 1600 parent was reused through an audited model-only migration: exported under the archived revision 66836d2 with that revision's own loader, then installed under the current code after re-checking source SHA pins, exported state digests, tensor keys/shapes/dtypes and module semantics, with a bit-identical FP32 forward on a fixed synthetic batch under both revisions. An independent agent recomputed 14/14 claims from raw artifacts with no discrepancy. Cost 0.4808 (failed probe) + 0.4706 (probe) + 3.3251 (screen) = 4.2765 GPU-h, cumulative 27.5905. Test never read, r=0, S4 not started.
- Limitations:
  - Development full-region screening on one instance (one ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim.
  - The attempt's frozen reading worker failed on the shared parent-pin check after the model identity change; all five earlier phases succeeded and the failure is preserved and charged in full. The verdicts come from a read-only tool that delegates to the same frozen decision functions and records its one deviation.
  - Three seeds are consistency evidence, not a significance test; no simultaneous intervals, no interior/edge stratification and no test access.
  - The registered v3-BD parent was reused through an audited model-only migration (no optimizer, cursor or RNG); the forward equivalence is shown on a fixed synthetic batch, not on real weather.
  - Dose measured at two points only (200 and 800); the long-lead wall remains pattern correlation, whose linear-rescaling ceiling ACC^2 is still below zero at 48/72h.
- Excluded from runnable candidates: Frozen reading phase failed after the model identity change, so the verdict comes from a read-only re-derivation; the round also does not meet the absolute 48/72h train-only climatology gate.
- Recorded metrics (not recomputed): all17_relative_mse_change_negative_at_24_48_72h=true, batch_size=1, bf16=false, candidate_updates=800, clip=1.0, confirmation_r_consumed=0, cumulative_gpu_hours=27.5905, decision=advance-to-S4-freeze, elapsed_seconds_total=11970.334048915654, forward_bit_identical_across_revisions=true, gate_cells=45, gate_most_positive=-0.10440126896160548, gate_positive_cells=0, gpu_hours=3.3251, hard_cap_seconds=18000, hard_overrun_seconds=0.0, independent_recheck_claims_confirmed=14/14, lead_hours=[6, 12, 24, 48, 72], lr=2e-05, migration_archived_model_code_sha256=3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217, migration_current_model_code_sha256=d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab, migration_export_sha256=d1d9886ae5d4d6f783b0f8936f5fe92e2cd82ef8856f017835c71ea48db15328, migration_receipt_sha256=23daf794d8f18709b3321e8f21486b83ed44df0eb4ddc07e4ca3e122349a694a, model_parameter_count=3286037, network_requests=0, per_seed_seconds=4200, physical_steps=12, physical_weights=[1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5], planned_seconds=9000, primary_verdict=supported, probe_attempt01_gpu_hours=0.4808, probe_attempt02_gpu_hours=0.4706, readings_sha256=e8b594d08089c2921cad25402ec0c827e10b4ffa74daa7a01a818db9b202c1eb, reasoning_steps=4, round_gpu_hours_total=4.2765, seeds=[41, 42, 43], soft_overrun_seconds=2970.3340489156544, source_network_bytes=0, t2m_acc_seed_mean=[0.795345, 0.665149, 0.652526, 0.36711, 0.189429], t2m_acc_squared_ceiling=[0.632574, 0.442423, 0.42579, 0.13477, 0.035883], t2m_rmse_delta_800_minus_200_K=[0.025441, 0.007629, -0.153554, -0.324838, -0.465331], t2m_rmse_delta_vs_control_K_seed_mean=[-0.519, -0.586, -0.789, -1.431, -1.736], t2m_sigma_ratio_seed_mean=[1.021, 1.022, 0.895, 0.871, 0.873], t2m_skill_seed_mean=[0.581624, 0.315123, 0.366812, -0.119139, -0.430893], test_read=false, val_cohorts=[472, 468, 460, 444, 428], warmup=10

## s3-same-case-gap-diagnostic

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_SAME_CASE_GAP.md` (SHA256 `82f73cc6b103ce79ea96be9df16aac2ecc5ea02d1a01fe3410eea208b6eb6135`)
- Evidence commit: `09447bc194562f4dac80868237122084ee5745dc`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `8b336951185b092ad30b05f4ba2eb7aa6b21828ae8601378b22dba0a1c7ca216`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Complete metadata-selected20in-sampletrain+4developmentval same-case seed41 parentBD1600 versus long200 endpoint without optimizer. t2m allfivelead poolederrors improve versusparent, but48/72h stillworse than unchangedtrainonlyclimatology inbothtrainandval andeverytrainyear/month group. This excludes a simpleval-only explanation inthesesamples, notcausalunderfitting/clipping/generalizationproof. Strictidentity/independentNumPy/arithmetic/costchecks plus exactarchivednumericreplay supportrecorded configuration. Failedfsumrelative5e-14 audit retained, separate descriptive supplement quantifiescancellation withoutrelaxingpredicate. NoS4/test/scientificacceptance.
- Limitations:
  - One seed,20in-sampletrain/4developmentval cases, notcompletecohorts/year/seasons/statisticalconfirmation; no cross-variable unitaverage
  - Training-fit climatology self-reference optimistic; negative48/72h train does not isolate optimization/capacity/clipping cause
  - Audit01FAILED at relative8.415e-14 above unchanged5e-14; fsum5082numericdifferences characterized separately; production-order independentNumPyexact
  - No optimizer/gradient/causalintervention; sameenvironment exactnumericreplay is config-reproducible maximum
  - Full source hashes include testyear sourcebytes for identity only, no testmanifest/state scoring; noOSlifetime/no globalneighborsignal proof
- CI run: `37543875564`
- Excluded from runnable candidates: Descriptive diagnostic only, no newcandidate/primarygate/PASS. Longleadclimatologyfailure persists in-sample; requiresseparatelyfrozen independentresearch.
- Recorded metrics (not recomputed): audit01_and_supplement_publication_seconds=12.288973688147962, audit01_frozen_relative_tolerance=5e-14, audit01_rejected_relative_difference=8.415017050200539e-14, audit01_status=failed, elapsed_seconds_total=1751.4211974898353, fsum_nonidentical_numeric_values=5082, gpu_hours=0.4865, hard_cap_seconds=3600, hard_overrun_seconds=0.0, independent_numpy_numeric_leaves_exact=51484, independent_numpy_typed_leaves_exact=19500, known_reserved_peak_bytes=2434793472, measurement_sha256=81c6ab827817288df87def06118720b5159793c2d5e9ae0f742a4a5842e4dd75, network_bytes=0, optimizer_updates=0, owned_cuda_reserved_peak_bytes=92274688, physical_steps=12, planned_seconds=1800, reasoning_steps=4, required_free_bytes=4582277120, seed=41, selection_sha256=b6b0e23231271fe71deb07ab6e6f91f5f7bb653acefb5214c5e6755f2c629706, soft_overrun_seconds=0.0, source_bytes=540856239, test_read=false, train_candidate_t2m_climatology_skill=[0.45583912506887153, 0.18056699451525046, -0.05087983750121831, -0.7792325083200121, -1.0025621480519484], train_candidate_t2m_rmse_k=[1.9017984166798423, 2.2081563250269003, 3.0120473748695358, 3.9933551476466054, 4.09989100393555], train_cases=20, train_climatology_t2m_rmse_k=[2.5781063292305513, 2.439344224166281, 2.9382264811838765, 2.9937917223879063, 2.897205557578146], train_parent_t2m_climatology_skill=[0.43251914537980996, 0.12229366438316197, -0.19067360963892235, -1.933083652459449, -3.7648619281704017], train_parent_t2m_rmse_k=[1.9421216611313996, 2.2853236080676687, 3.2061337256773257, 5.1272419033713, 6.324183603984531], val_candidate_t2m_climatology_skill=[0.38888082810979735, 0.2405101800727532, 0.4379609958099574, -0.33536024272622406, -1.2087627404722974], val_candidate_t2m_rmse_k=[1.937445303354077, 2.041813066488836, 2.6784736372973894, 3.744068532894296, 4.411827847107436], val_cases=4, val_climatology_t2m_rmse_k=[2.4783720141866024, 2.34290673181801, 3.572762508520549, 3.239996705812388, 2.968547645553845], val_parent_t2m_climatology_skill=[0.37113228973190343, 0.1141600029069016, 0.26305973019208045, -2.046285119071694, -6.672093090384134], val_parent_t2m_rmse_k=[1.9653782447404624, 2.2051221057739907, 3.067046023057042, 5.654963982111222, 8.222444593234775], workers_exit0_reaped_no_signals=4

## s3-same-case-gap-replay

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_SAME_CASE_GAP.md` (SHA256 `82f73cc6b103ce79ea96be9df16aac2ecc5ea02d1a01fe3410eea208b6eb6135`)
- Evidence commit: `09447bc194562f4dac80868237122084ee5745dc`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `e18034c17d15f82cfba8372d5a8379f1ec2b3843f48b5101e2b4bebdc2be5835`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Archivedexact61e46bd same24train/val cases regenerated with bothpinnedseed41 endpoints and oncefitunchanged2400/16x150climatology. Fulloriginal/recomputedpayloads retained; canonicalnativefloatJSON projectionexact,numericdifferences0. OnlyelapsedandtwoCUDApeak fields excluded. Threeownedworkers exit0/reaped/nosignals, knowntrainingpeakfloor+margin upheld; freshindependentdeadline. Whole2844.160051s softover1044.160051 hardover0,full0.7900GPUhcharged; independentterminalnumericrestorationgate acceptednot scientificconfirmation.
- Limitations:
  - Same environment exact nativefloat numeric equality, not rawfloat-bit or crossdevice GPUtraining guarantee
  - Inheritedsource/checkpoint/mathaudit explicit; terminal didnotrerunGPU/weatherfield/climatefit
  - Archivedonlyimports verified;externalactualCmetadata livehashpinnednotfullyselfcontainedZIP
  - WholeCPUprepare/loading/readingchargedandsoftoverrunpreserved;Pythonread/networkguardnotOSsandbox; noindependentnonblockingOSbootstrapwatchdog
- CI run: `37543875564`
- Excluded from runnable candidates: Identity/configuration reproducibility audit only, no independenttest/weatherconfirmation/newcandidate orscientificPASS.
- Recorded metrics (not recomputed): elapsed_seconds_total=2844.160050935112, fullresult_sha256=647274a1f22b273f22eb406c6c547554036219436a77762c8e6b2ecfe755fb57, gpu_hours=0.79, hard_cap_seconds=3600, hard_overrun_seconds=0.0, metadata_cases_exact=24, network_bytes=0, numeric_difference_count=0, physical_mse_cells_per_payload=6120, physical_rmse_cells_per_payload=6120, planned_seconds=1800, prelaunch_audit_seconds=292.499, projection_sha256=936b7d265f22912c21fecda8943dc13149b391667a806910df29e01dc03aa406, replay_wrapper_sha256=62a0d93f5c0a34d9f98fad7f241943de4eadf655fff343f560009a527d47a802, restoration_accepted_sha256=1a8a20d5d03bf8cd154368e8c157d39b5f41567184c2d6b82dbfe8c71b753201, soft_overrun_seconds=1044.160050935112, terminal_audit_seconds=226.115, test_read=false

## s3-train-gradient-components

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` (SHA256 `627b324d60f08f442592a3430113f088bb356d4dccf1b6431cbbadc5f69bbdba`)
- Evidence commit: `6c3bb69696b5d4855da81e9b15eb9f5dba3acbe9`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `44b634ed478d398754dfa87bebd176cdd060048385f04e2f7f8713a827c4ddc9`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Four predeclared train2021 seasonal same-case FP32/K4/full12 component gradients and actual total completed without optimizer, state change or held-out readers. Independent saved-fact identity/arithmetic/accounting qualification supports descriptive metrics; long-lead component norms dominate but cosine directions vary, not proof of clipping harm or underfitting. Exact representative replay failed and is registered separately, never relaxed.
- Limitations:
  - One endpoint and four in-sample cases; normalized all17 training objective is not physical weather skill.
  - No causal convergence, underfitting or clipping claim; no scientific/candidate threshold added.
  - Config-reproducible maximum. Representative replay has161 numeric-statistic differences and failed exact matching.
  - Saved-fact audit is not raw-gradient/tensor provenance or global OS tracing; original gate receipts lack timestamps.
  - All preparation failures/soft overruns and separate test-resource deviation remain registered.
- CI run: `37543875564`
- Excluded from runnable candidates: Descriptive diagnostic only, not candidate or scientific acceptance; exact gradient restoration failed.
- Recorded metrics (not recomputed): active_total_gradient_tensors=141, cancellation_ratios=[0.66086863824846, 0.785620448927343, 0.8683714333895012, 0.8252848521038937], clip1_factors_descriptive=[0.06488679349422455, 0.07552441209554672, 0.1645030826330185, 0.08896627277135849], elapsed_seconds_total=2465.08890417777, gpu_hours=0.6847, hard_cap_seconds=3600, hard_overrun_seconds=0.0, independently_audited_files=721, network_bytes=0, next_spawn_required_free_bytes=4638900224, objective_losses=[1.29634428024292, 1.2238487005233765, 0.6454952955245972, 1.1550655364990234], optimizer_updates=0, owned_cuda_reserved_peak_bytes=2491416576, parameter_tensors=155, physical_steps=12, planned_seconds=1800, reasoning_steps=4, representative_exact_restoration=false, result_sha256=a9c7da6d1fe28c177f952ba61423b436430f6e90198f31edb86bdf654fbac11a, seed=41, selection_sha256=649c20bfa508f83a75b5cb905ac1352d3585509b63c8135a75f4d38688a0b9a4, soft_overrun_seconds=665.08890417777, test_read=false, total_gradient_norms_fp64=[15.41145594365589, 13.240752040249031, 6.078913099940799, 11.24021256746771], train_cases=4, unused_total_gradient_tensors=14, val_manifest_read=false

## s3-train-gradient-replay-attempt01

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` (SHA256 `627b324d60f08f442592a3430113f088bb356d4dccf1b6431cbbadc5f69bbdba`)
- Evidence commit: `6c3bb69696b5d4855da81e9b15eb9f5dba3acbe9`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `9e8480f7dbaa9b9196a4bbbfc88b046de9fecda22d527caaab24c0b889d2a571`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Code-archive fixture manifest copying was rejected by strict basename guard before actual fields or model qualification. Failed attempt retained; later Python-only extraction repair did not revive it.
- Limitations:
  - Failed restoration is not a pass; all cost charged and failed root retained.
  - No comparator tolerance relaxation, field exclusion expansion, in-place retry or attempt03.
  - No independent test or scientific confirmation; exact identity/arithmetic does not establish raw GPU gradient bits.
- CI run: `37543875564`
- Excluded from runnable candidates: Failed-restoration; no numerical acceptance marker, candidate or scientific PASS.
- Recorded metrics (not recomputed): elapsed_seconds_total=1.8343665357679129, failure_sha256=c7b8fc4410cc4e128f2a5ae96c746bc2066b682bf99d90e333b86d954af8825d, gpu_hours=0.0005, hard_cap_seconds=3600, hard_overrun_seconds=0.0, network_bytes=0, numeric_difference_count=0, planned_seconds=1800, recorded_worker_signals=[[]], recorded_workers_reaped=true, restoration_accepted=false, soft_overrun_seconds=0.0, status=failed-restoration, test_read=false, val_manifest_read=false

## s3-train-gradient-replay-attempt02

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md` (SHA256 `627b324d60f08f442592a3430113f088bb356d4dccf1b6431cbbadc5f69bbdba`)
- Evidence commit: `6c3bb69696b5d4855da81e9b15eb9f5dba3acbe9`; experiment commit: `61e46bd78d4d7ce49da7a87d574e40c6c18a1c99`
- Protocol SHA256: `6610be1d74747317ef091dc2a74c293ff2bc88e0be1b0e0c541597caefee1b17`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Two owned workers regenerated January case under pinned code/source/checkpoint/state/recipe, but full native JSON exact restoration failed on161 numeric leaves. Losses, state, RNG, masks and FP32 clip stats exact; Gram-derived metrics and residual norm differ. Independent characterization preserves negative exact restoration, no tolerance or attempt03.
- Limitations:
  - Failed restoration is not a pass; all cost charged and failed root retained.
  - No comparator tolerance relaxation, field exclusion expansion, in-place retry or attempt03.
  - No independent test or scientific confirmation; exact identity/arithmetic does not establish raw GPU gradient bits.
- CI run: `37543875564`
- Excluded from runnable candidates: Failed-restoration; no numerical acceptance marker, candidate or scientific PASS.
- Recorded metrics (not recomputed): elapsed_seconds_total=1569.81766172871, failure_sha256=5b994e3511f2501ad51aac1e44bee48635dfc3680830f3cb8b10ce09772ced2f, gpu_hours=0.4361, hard_cap_seconds=3600, hard_overrun_seconds=0.0, network_bytes=0, numeric_difference_count=161, planned_seconds=1800, recorded_worker_signals=[[], []], recorded_workers_reaped=true, restoration_accepted=false, soft_overrun_seconds=0.0, status=failed-restoration, test_read=false, val_manifest_read=false

## s3-v3-numerical-errata

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_V3_NUMERICAL_ERRATA.md` (SHA256 `87db1c93aab2f4ddd48250afbdd6742e7d4fe8a676b816a1bba8dfb50a559b8e`)
- Evidence commit: `440d92232d7b0549531ac4cb0048f1e5a67f10b8`; experiment commit: `dd2139e8acaee758bc928709f5e5db0f8ec3db00`
- Protocol SHA256: `not recorded`; data identity: `not recorded`
- Reason: Independent read-only audit corrects climatology-skill vs incumbent-improvement counts, seed42 cross-instance direction, convergence and co-resident causality overstatements; recipe/negative verdict unchanged. Old evidence pages and outputs preserved.
- Limitations:
  - read-only recorded artifact audit, no trained model replay or independent weather confirmation
  - cross-instance differences descriptive; no new causal or convergence claim
- Excluded from runnable candidates: Numerical correction and identity audit only; no candidate or scientific PASS.
- Recorded metrics (not recomputed): audited_file_identities=286, gpu_hours=0.0, lower_rmse_vs_incumbent_cells_of_51={"12": 51, "24": 50, "48": 22, "6": 51, "72": 14}, pin_mismatches=0, pooled_rmse_max_relative_difference=6.82e-13, positive_skill_vs_climatology_cells_of_51={"12": 51, "24": 50, "48": 4, "6": 51, "72": 0}

## s3-v3-rollout-ft-attempt01

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT01.md` (SHA256 `fb0c0742c243f3dae4326748ab97daa7a39892d671ae8b7d76d29733fee23003`)
- Evidence commit: `440d92232d7b0549531ac4cb0048f1e5a67f10b8`; experiment commit: `dd2139e8acaee758bc928709f5e5db0f8ec3db00`
- Protocol SHA256: `8e8fc195b5a3e58420a6d4101d5f00a035c3a4eee753b043fcebb90273d0381d`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Failed incomplete rollout continuation: seed41/42 finished200 updates and all val leads; seed43 only20/40/60 checkpoints, no endpoint/evaluation. Cooperative deadline overshot7200s to9473.920853s. No three-seed verdict/decision/complete; partial8gate violations on completed two seeds. All costs charged2.6316GPU-h; original artifacts unchanged.
- Limitations:
  - two seeds only, no pre-registered three-seed verdict; diagnostic partial results not PASS
  - blocking cause unknown; cooperative deadline late
  - whole-round GPU-h-equivalent includes failure, not active utilization measurement
  - config-reproducible maximum; no bit-reproducibility claim
- CI run: `37437569014`
- Excluded from runnable candidates: Failed partial attempt. No S4 promotion, in-place resume, or combining seeds with a future attempt.
- Recorded metrics (not recomputed): elapsed_seconds_total=9473.920853041112, gpu_hours=2.6316, hard_cap_seconds=7200, hard_overrun_seconds=2273.920853041112, network_bytes=0, partial_gate_failure_cells_on_two_completed_seeds=8, planned_seconds=4200, seed43_latest_saved_update=60, seeds_completed=[41, 42], soft_overrun_seconds=5273.920853041112, status=failed-incomplete, test_read=false

## s3-v3-rollout-ft-attempt02

- Outcome class: `mixed`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md` (SHA256 `3148ca389bde923fa905f4ebb316db55bd139e7efad53733db65c4229f9a6ac8`)
- Evidence commit: `9eed6aea30e64b21b0a11c5f85120fecaed180a4`; experiment commit: `8466c2def5df04276cc70db530d667ba9b674d44`
- Protocol SHA256: `f02fec53283154aed8c123435f3f415ca31b653dd86df35e6152a5922581ae37`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Complete three-seed FP32 short rollout continuation of registered v3-BD1600 endpoints:200 two_step updates with fresh AdamW, LR2e-5,K4. Primary6/12h supported on all seeds; same-data zero-tolerance u10/v10/mslp gate still13/45 positive cells (48h5,72h8), so unchanged conjunction is registered-negative. Parent gate17->13, not cleared. t2m climatology skill positive6/12/24h but allnegative48/72h. Whole round4222.618677s, soft5400/hard10800 unexceeded;1.1729GPU-h. Test unscored; no old-seed combining. Independent artifact audit and archived code replay validate identity and numbers, not scientific acceptance.
- Limitations:
  - Development full-region point estimates on one ROI and four30-day season blocks per year, not full-year confirmation
  - Same-data400-update control pinned, not retrained; fieldedrecipe costs5.000889xcontrol FLOPs, not compute-matched
  - Highest config-reproducible; no bitwise training, significance or convergence claim
  - t2m48/72h remains worse than climatology and gate remains negative
  - Original failed attempt01 costs preserved separately
- CI run: `37468877656`
- Excluded from runnable candidates: Frozen conjunction negative: primary supported but13gate failures; short two-step dose stopped, noS4/test/goal acceptance.
- Recorded metrics (not recomputed): candidate_vs_parent_gate_failures=3, decision=registered-negative, elapsed_seconds_total=4222.618676601909, evaluation_loop_seconds=2337.532753434032, fielded_flop_ratio_vs_400_control=5.000889131555414, fine_tune_updates=200, gate_failures={"by_lead": {"48": 5, "72": 8}, "by_seed": {"41": 3, "42": 5, "43": 5}, "total": 13}, gate_passed=false, gpu_hours=1.1729, hard_cap_seconds=10800, hard_overrun_seconds=0.0, lambda12=0.5, lower_rmse_vs_incumbent_cells_of_51={"12": 51, "24": 51, "48": 32, "6": 51, "72": 18}, lr=2e-05, mode=two_step, network_bytes=0, parent_gate_failures=17, parent_updates=1600, planned_seconds=5400, positive_skill_vs_climatology_cells_of_51={"12": 51, "24": 51, "48": 5, "6": 51, "72": 0}, primary_verdict=supported, seed_count=3, seed_mean_t2m_skill={"12": 0.3314290588573388, "24": 0.2738378251991728, "48": -0.5574060415928729, "6": 0.5969316038321609, "72": -1.337276897688223}, soft_overrun_seconds=0.0, t2m_skill_vs_climatology={"41": {"12": 0.319684013168843, "24": 0.27046129443648403, "48": -0.4948879469079752, "6": 0.5915128000455058, "72": -1.1015964750729772}, "42": {"12": 0.3303053093514684, "24": 0.24523729026654306, "48": -0.6943867545963651, "6": 0.5962876919072397, "72": -1.7119073294075333}, "43": {"12": 0.3442978540517051, "24": 0.30581489089449143, "48": -0.4829434232742782, "6": 0.6029943195437373, "72": -1.1983268885841585}}, test_read=false, training_seconds=295.6816603280604

## s3-v3-rollout-ft-replay

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_V3_ROLLOUT_FT_ATTEMPT02.md` (SHA256 `3148ca389bde923fa905f4ebb316db55bd139e7efad53733db65c4229f9a6ac8`)
- Evidence commit: `9eed6aea30e64b21b0a11c5f85120fecaed180a4`; experiment commit: `8466c2def5df04276cc70db530d667ba9b674d44`
- Protocol SHA256: `a0a9b50088223f5917d6681fccc22bc99d87982aca7830d8991d4ddf87f6933e`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: Archived exact8466c2d seed41 five full-val lead replay:15CSV hashes identical, per-case MSE maxdifference0.0; archived three-seed collector exact JSON representation hash matches. Two preforecast wrapper failures retained; numerical-sort/string-sort attribution independently verified and oneULP/decision mutations rejected. Successful1375.732820s plusfailed8.888093/8.885239s all charged0.3871GPU-h. Not a scientific confirmation or new candidate.
- Limitations:
  - Seed41 only actual forecast replay; otherseeds artifact-audited, no retraining
  - Exact observed bytes on same machine do not establish cross-platform bitwise reproducibility
  - Failed wrapper attempts and immutable originalrun outputs preserved
- CI run: `37468877656`
- Excluded from runnable candidates: Identity/configuration reproducibility audit only, not new scientific evidence or candidate.
- Recorded metrics (not recomputed): archived_collector_json_sha256=3b6b9af8a0cb343685713ec817bf22def9675461e3e0b2a12bc8363aa8a6c7b7, case_max_relative_mse_difference=0.0, elapsed_seconds_failed_replays=[8.888092823326588, 8.88523946981877], elapsed_seconds_successful_replay=1375.7328200042248, exact_csv_count=15, failed_preforecast_replays=2, gpu_hours=0.3871, hard_cap_seconds_each=3600, hard_overrun_seconds=0.0, network_bytes=0, planned_seconds_each=1800, soft_overrun_seconds=0.0, successful_replays=1, test_read=false

## s3-wide-interior-supervision

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_WIDE_INTERIOR_SUPERVISION.md` (SHA256 `2d07aa5a95238f734d757a9248bd8c86d078a4ff36531b07cb8673f6c3f29b39`)
- Evidence commit: `63e533f457d6ed7a8715b42e95077e589a61f918`; experiment commit: `7ef43cbdc3a72d4942c018b6817384584b2bc3f3`
- Protocol SHA256: `6fc50d049d484182f90ea51bf6e86d9b3cc146324ae183d360a7926c399c5fc7`; data identity: `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12`
- Reason: S3 wide 129x129 input with supervision confined to boundary_masks(129,129,(32,))['interior_32'] (4225 cells = the frozen central 65x65 box = the registered narrow arm's own grid), same registered 800-update long-rollout recipe, same migrated v3-BD 1600 parent, seeds 41/42/43, scored on the same interior_32 region against the pinned narrow arm (index record s3-rollout-dose, not retrained). This arm removes the previous round's supervision-dilution confound: its seed-mean t2m delta (arm minus narrow, K) is negative at every lead (-0.0348 at 6h, -0.0401 at 12h, -0.0069 at 24h, -0.0528 at 48h, -0.0445 at 72h) and 3/3 seeds are better at 6h and 12h, while the previous full-grid arm was positive at every lead. Skill improves at every lead against both the narrow arm and the previous wide arm (interior_32 t2m skill means 0.5944/0.3341/0.3700/-0.0868/-0.3996 vs narrow 0.5816/0.3151/0.3668/-0.1191/-0.4309). So the previous null is largely explained by supervision dilution. However the absolute climatology gate still fails at 48/72h (only seed 43 reaches +0.0140 at 48h; 72h is negative for all three seeds), so no S4 freeze package is entered and the arm is promoted to a dose test.
- Limitations:
  - development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim
  - the narrow control is a pinned registered reference (index record s3-rollout-dose), not retrained inside this protocol; the arms did not share identical host load
  - the outer 129x129 ring is unconstrained during training because supervision is restricted to interior_32; only interior_32 is scored
  - the migrated v3-BD 1600 parent was trained on the narrow instance under full-grid supervision; the optimization landscape difference is not separated
  - the absolute climatology gate still fails at 48/72h (only seed 43 is positive at 48h)
  - three seeds are consistency evidence, not a significance test; GPU runs are co-resident
  - validation split only; the 2023 test split stays sealed and r is not consumed
- CI run: `38022897803`
- Excluded from runnable candidates: 开发筛选轮：对照是已登记臂的 pinned 复用（不在本协议内重训练），三 seed 只是一致性证据，没有显著性检验；收益幅度小且 48/72h 绝对气候态门未过，收益来源（域外上下文 / 监督噪声 / 梯度预算）未分离，故标 needs-review 而非 candidate，也不进 S4 冻结包。
- Recorded metrics (not recomputed): checkpoint_sha256={"41": "590b2cc70cb441b97c4a062b310434037207bce861fae5cd20950d83e89adb85", "42": "b292865e029f9a7b3efc980f9253f7c38d8dd24f9602f7e449b6bd90dd7d94ba", "43": "8377f63d8ab8093ff9dabed3d9184c791c57f9ca49e40b74fdda96d288ac6ab7"}, code_zip_sha256=2e60dfd60449eef4af2b7947b6ff37d189a27124328f56eacdcfabebd71b2351, delta_range_t2m_k={"12": [-0.0515, -0.0259], "24": [-0.0204, 0.0138], "48": [-0.1044, 0.0111], "6": [-0.0459, -0.0141], "72": [-0.0948, 0.0152]}, disk_gib=4.6, gpu_hours=3.4655, hard_overrun_seconds=0.0, interior_t2m_skill_mean={"12": 0.3341, "24": 0.37, "48": -0.0868, "6": 0.5944, "72": -0.3996}, mean_delta_minus_narrow_t2m_k={"12": -0.0401, "24": -0.0069, "48": -0.0528, "6": -0.0348, "72": -0.0445}, model_code_sha256=d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab, narrow_t2m_skill_mean={"12": 0.3151, "24": 0.3668, "48": -0.1191, "6": 0.5816, "72": -0.4309}, network_bytes=0, previous_wide_full_grid_delta_mean={"12": 0.0147, "24": 0.0189, "48": 0.0121, "6": 0.0013, "72": 0.0231}, region_points=[129, 129], reproducibility=config-reproducible; archived code.zip replays the source identity, GPU training is not bitwise reproducible, seeds=[41, 42, 43], seeds_better_than_narrow={"12": 3, "24": 2, "48": 2, "6": 3, "72": 2}, soft_overrun_seconds=3475.851618, supervision_cells=4225, supervision_mask_sha256=4a2a576bf948b92924720fa28a44c128a61445dd29b4906805930430e0a0919a, test_read=false, training_code_sha256=8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974, updates=800, val_cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, wall_seconds_total=12475.851618

## s3-wide-region-full

- Outcome class: `engineering-positive`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_WIDE_REGION_FULL.md` (SHA256 `44a19a4b5b76c24527cd6c76eb8ad79375d1b8df363dbf18bea5ad058b01b467`)
- Evidence commit: `123097b0a13d4f330782025e274b3f6c739b4c3f`; experiment commit: `123097b0a13d4f330782025e274b3f6c739b4c3f`
- Protocol SHA256: `be50afbc100091d55aee7117baaf623927bfded8a27fed4246da29fb1a1cf8c5`; data identity: `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12`
- Reason: S3 wide-region (lateral-context) full acquisition, v4 store and wide train-only climatology. The S3 rollout-dose round left the 48/72h wall as pattern correlation (t2m/full ACC 0.367/0.189, linear-rescaling ceiling ACC^2 only +0.1355/+0.0366), and at 10 m/s a synoptic system travels ~2600 km (23 deg) in 72h, more than the frozen 16-degree box, so 'the box lacks the information the long leads need' is the recorded, still-untested hypothesis. This round changes one factor only - the read region, 65x65 (27-43N/107-123E) to 129x129 (19-51N/99-131E) - whose central 65x65 block is asserted to be the frozen target box. It acquires the 28 (year, season) parts the registered v3 instance covers (train 2017-2021, val 2022, test 2023; 3360 stamps), merges them into source.nc (SHA256 6bc9a1a2...), publishes a v4 store [3360,17,129,129] with the same splits and window counts as v3 (2360/472/472), and scores the train-only climatology and persistence on val 2022 with boundary_margins=(32,), whose interior_32 region is exactly the frozen 65x65 box (4225 points). The wide instance's interior_32 t2m climatology and persistence RMSE match the registered v3-D2 values to ~1e-10, so the single-factor comparison's skill denominator is the registered one. Along the way the committed wide acquisition path was found unrunnable - four defects left by the R-021 module split, including a receipt writer that referenced its caller's local and crashed after a part had fully downloaded - and was repaired with two regression guards (a scope walk that reports all seven pre-fix undefined names, and a receipt-composition test). Two parts (winter_2020, winter_2022) hit the frozen 1800 s per-part deadline, wrote no artifact, and were re-fetched under the _r2 name; both failures are kept. CPU only: 0 GPU-hours, the test manifest is built but never opened.
- Limitations:
  - data acquisition, a published store and a parameter-free reference only: no model was trained or scored, and no skill claim follows
  - the wide region is a development instance; the frozen 16-degree target box is unchanged
  - network_body_bytes is a host-wide /proc/net/dev counter over each part's read window, so under pool 4 each part's figure includes sibling traffic (~4x inflated); the per-part 'within 25% of the pilot' criterion could not be evaluated as written and is recorded as a deviation, with the batch total measured separately
  - two parts overran the frozen 1800 s per-part deadline at pool 4; both wrote no artifact, are preserved as failed-no-fallback, and were re-fetched under the _r2 name
  - the acquisition ran at the measured-safe pool of 4; six concurrent readers of this icechunk store deadlock with frozen CPU and idle sockets
  - the batch network total (~101 GB) combines measured host-wide windows with part-window bounds; it is an estimate, not one continuous measurement
  - the store build reused the region-agnostic v3 build config, so the split contract is identical to v3 by construction
  - no test-split byte was read; the 2023 manifest is built but sealed
- CI run: `37974234521`
- Excluded from runnable candidates: 工程/数据制度轮：它获取数据、发布 store 并给出参数自由基线，不训练、不评分任何模型，因此不构成候选，也不进入 S4。宽区是否带来长 lead 收益由下一轮单因素 train/val 判定。
- Recorded metrics (not recomputed): batch_wall_seconds=12223.0, climatology_interior32_t2m_rmse_k={"12": 3.488909, "24": 3.437473, "48": 3.353443, "6": 3.517116, "72": 3.318361}, d2_baselines_seconds=2461.6, disk_peak_gib=6.6, gpu_hours=0.0, interior_32_points=4225, network_bytes_estimate=101000000000, network_hard_cap_bytes=161061273600, parts_new=27, parts_retried=2, parts_reused=1, parts_total=28, persistence_interior32_t2m_rmse_k={"12": 5.92348, "24": 3.163328, "48": 4.179511, "6": 4.547019, "72": 4.726945}, pool=4, source_bytes=2003552683, stamps_per_part=120, stamps_total=3360, store_build_seconds=168.8, store_shape=[3360, 17, 129, 129], test_read=false, v3_d2_interior_match_max_abs_diff=6e-10, validation_part_seconds=1218.5

## s3-wide-region-pilot

- Outcome class: `engineering-positive`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_WIDE_REGION_PILOT.md` (SHA256 `98606358b669c8d6896259903f08ff15c5497a9b69534f154aa18a1fc32f612a`)
- Evidence commit: `6eeaffd99a3d6e7ce5fdfc4a2447948cc66edb1b`; experiment commit: `8ace116a9187c8c88ed15f1110de5d6ab4848a3e`
- Protocol SHA256: `bfdcfa892a551f33c4d8530cbc02d58a43dd1aa7f7a9c25ff186bdfce1419526`; data identity: `not recorded`
- Reason: S3 wide-region (lateral-context) acquisition pilot. The S3 rollout-dose round left the 48/72h wall standing as pattern correlation: t2m/full ACC 0.367/0.189 and a linear-rescaling ceiling ACC^2 of only +0.1355/+0.0366, so no further dose can clear the train-only climatology gate; at 10 m/s a synoptic system travels ~2600 km (23 deg) in 72h, more than the frozen 16-degree target box, which makes 'the box does not contain the information the long leads need' a falsifiable hypothesis. This round changes one factor only - the read region, from 27-43N/107-123E at 65x65 to 19-51N/99-131E at 129x129 (8 deg on every side, 3.94x the area) - with the source, snapshot, 17 channels, 0.25 deg grid, 6h cadence, 2017-2021/2022 split, model spec and archived initialization all unchanged. The wide grid's central 65x65 block is asserted to be the frozen target box, so the registered climatology and val cohorts stay valid and boundary_masks(129,129,(32,)).interior_32 selects exactly that box; a test asserts the mask and the central block are elementwise equal, and the pilot run confirms it empirically by comparing against the registered v3 source.nc. One part (2018 winter, 120 stamps) was downloaded against the live pinned snapshot: network 3,565,096,475 B versus the registered target-box part's 3,554,716,077 B (ratio 1.0029), decoded charged bytes 9,482,837,880 and chunk reads 2287 both exactly equal, wall 1223.141 s versus 1267.312 s, stored 70,207,133 B versus 18,977,715 B. The reason is the source layout: one whole-globe field per time in one chunk, so any ROI read decodes the same global chunk and enlargement costs only storage. The frozen criteria pass 18/18 with zero deviations, and the central block is bit-identical to the registered v3 source.nc over all 120 stamps for nine stored variables, the 17 stacked channels, both coordinates and every unit. The preflight caught a real error before any download - the cost model used 17 field reads per stamp (the channel count) where the source has 19 (4 surface variables plus 5 pressure variables at 3 levels) - and the protocol records that pre-download revision; three defects in the new checking tools were also found and fixed, with negative controls added for each criterion. No model, gate, contract or frozen evidence changed, no synthetic substitute was used, the test year was never read and r stays 0. Cost 0 GPU-h, 3.565 GB network, 70.2 MB retained; cumulative GPU-h unchanged at 27.5905.
- Limitations:
  - One part, one year, one season: a cost and identity pilot, not a seasonal or climatological sample.
  - The measured near-equality of the network cost is a single observation, not a cost model.
  - An 8-degree half-width is only a partial answer to synoptic advection: 72h at 10 m/s needs ~23 deg.
  - No training, no scoring, no skill claim; the 2023 test year stays sealed and r stays 0.
  - The registered climatology and incumbent for a wide instance are not rebuilt yet, so no wide-instance comparison exists.
- CI run: `37897495917`
- Excluded from runnable candidates: No model was trained or scored and no skill number exists: this is a cost/identity feasibility pilot for a data-regime change, so it cannot be a candidate and it does not test the region hypothesis itself.
- Recorded metrics (not recomputed): area_ratio=3.938698224852071, artifact_bytes_ratio_vs_target_box_part=3.699451330152234, artifact_sha256=b05f216353c2d71abfd649d46b603522fba33851565aa06bafd01775881b12fc, chunk_reads=2287, chunk_reads_ratio_vs_target_box_part=1.0, ci_run_id_correction=an earlier draft of this record named 37898041952 for 7bfca85; the run for that commit is 37899661687. The wrong id was caught before it was relied on and is corrected here rather than left as an unverifiable pin., ci_runs=[{"commit": "d3a7a1b5ded3749f60a2c7e5d6d29cda735e4a31", "conclusion": "success", "run_id": 37897495917, "scope": "the acquisition code and the evidence page"}, {"commit": "7bfca857fbded17e1ef82c18c8b0f6a045ee029f", "conclusion": "success", "run_id": 37899661687, "scope": "the CI binding and the review corrections"}, {"commit": "d4038361935907f6c3a5ca880ef8453890e375db", "conclusion": "success", "run_id": 37901785259, "scope": "the source licence/DOI field and the CI run-id correction (current HEAD)"}, {"commit": "356195d405be084176ad2e1a4536a44c9a776710", "conclusion": "cancelled", "run_id": 37901721391, "scope": "superseded by the concurrency group when the next push arrived; a cancelled run is not a pass and is recorded as such, and the same tree is verified by the d403836 run above"}], confirmation_r_consumed=0, criteria_check_sha256=16db2ee282324bfef53fc41cd2be75cc8b971b47bbbb984f812dd381046fbc03, criteria_passed=18, criteria_total=18, criteria_verdict=pass, cumulative_gpu_hours=27.5905, decoded_charged_bytes=9482837880, decoded_charged_bytes_ratio_vs_target_box_part=1.0, disk_retained_bytes=70207133, download_elapsed_seconds=1223.141, elapsed_seconds_ratio_vs_target_box_part=0.9651459151337636, gpu_hours=0.0, hard_overrun_seconds=0.0, hard_watchdog_seconds=1860, independent_recheck={"agent": "general-purpose read-only verifier, own scripts, raw artifacts", "claims_confirmed": "C1-C6, C8(tests pass), C9, C10", "claims_refuted": ["C7 partial: the download path imports torch transitively through read_plan_frozen -> training.r7_experiment.canonical_digest, so the 0 GPU-h claim rests on no CUDA device being opened, not on torch being absent"], "doc_corrections_from_review": ["test-case count corrected to 45 (17 + 8 + 20) from an earlier '43'/'23'", "the forward full-batch wall estimate corrected to ~8.2 h from ~8.6 h"], "mutation_control": "setting TARGET_MARGIN_CELLS=30 in a /tmp copy fails 5 geometry/protocol tests and changes the read-plan digest d5618cbf -> 7c4137a6", "not_verifiable": ["cumulative GPU-h ledger arithmetic", "the host-wide network counter itself", "positive proof that the test manifest was never opened"]}, network_bytes=3565096475, network_bytes_ratio_vs_target_box_part=1.0029201764009126, pilot_result_sha256=4845099f7367d6fe54a965a5cdba811ccb41952dea507b04cde16bb5d35b03ad, planned_seconds=1800.0, read_plan_protocol_sha256=d5618cbf9709a43883666d65fd6729f9e048f1103bee4415e33a777b5fe035a5, receipt_sha256=77cedd699135c4e7c9c94a787ea0249b1d3eaac8dfe4907997d32bcbfc0628b6, round_gpu_hours_total=0.0, shape=[120, 3, 129, 129], soft_overrun_seconds=0.0, source_citation={"accessed": "2026-10-09", "doi": "10.24381/cds.adbb2d47", "license": "CC-BY-4.0", "note": "added after the pilot receipt was written, so it applies to subsequent parts only", "registry": "https://registry.opendata.aws/earthmover-era5/"}, stamps=120, target_block_check_sha256=b39ebc9709acd5e01810f5a6eeb3e16ddf5fd0d373838291d964a05bae763979, target_block_coordinates_identical=true, target_block_stacked_channels_identical=true, target_block_stored_variables_identical=true, target_block_verdict=pass, target_box={"east": 123.0, "margin_cells": 32, "north": 43.0, "points": [65, 65], "south": 27.0, "west": 107.0}, test_read=false, wide_region={"east": 131.0, "north": 51.0, "points": [129, 129], "south": 19.0, "west": 99.0}

## s3-wide-single-factor

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `99` (not a scientific score)
- Evidence: `docs/R7_S3_WIDE_SINGLE_FACTOR.md` (SHA256 `5db6bb7ea510d0f6c5ac1bb232049587c0bc953013954f1ae6d7e6d55e3b91b6`)
- Evidence commit: `dcd1a86c88d97bc231508ab8c1ebd0a2161557d2`; experiment commit: `13f3c440557b4e0d2d7ef5bcdb1be1985a8aeb5f`
- Protocol SHA256: `d016eb7a152efa60cae7bfb42ca877456a5204fe205d921c710bad42fe31d324`; data identity: `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12`
- Reason: S3 wide-region single-factor train/val: the registered 800-update long-rollout recipe applied unchanged to the wide 129x129 store (the only changed factor) and scored on the frozen central 65x65 box via boundary_masks(129,129,(32,))['interior_32'], against the registered narrow 65x65 rollout-dose arm (index record s3-rollout-dose, pinned by rmse_csv_sha256, not retrained). The wide arm is neutral-to-worse at every lead: seed-mean t2m delta (wide minus narrow, K) is +0.0013 (6h), +0.0147 (12h), +0.0189 (24h), +0.0121 (48h), +0.0231 (72h), and only 2/3 seeds are better at 48/72h. The absolute climatology gate still fails at 48/72h and the wide arm is slightly worse there (skill -0.1271/-0.4485 vs narrow -0.1191/-0.4309). The reading therefore does NOT support 'the frozen 16-degree box lacks the information long leads need'. It does not refute the hypothesis either: this arm supervises the full 129x129 grid, so supervision area is 3.94x the narrow arm and information gain is confounded with supervision dilution. No S4 freeze package is entered.
- Limitations:
  - development screening on one wide instance (129x129, one ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim
  - the narrow control is a pinned registered reference (index record s3-rollout-dose), not retrained inside this protocol; the two arms did not share identical host load
  - full 129x129 supervision confounds information gain with supervision dilution (3.94x supervision area); separating them needs a same-supervision-domain arm, which is the next round
  - the migrated v3-BD 1600 parent was trained on the narrow instance and reused on the wide store because the model spec is region-agnostic
  - scoring is interior_32 only; the outer ring's error is not scientifically read
  - the absolute climatology gate still fails at 48/72h for both arms (negative skill)
  - validation split only; the 2023 test split stays sealed and r is not consumed
- CI run: `38005892925`
- Excluded from runnable candidates: 开发筛选轮：对照是已登记臂的 pinned 复用（不在本协议内重训练），三 seed 只是一致性证据，没有显著性检验；它给出的是 null 读数与一个已记录的混淆（全监督稀释），不是科学判定，也不构成对假设的反证，故不进候选、不进 S4 冻结包。
- Recorded metrics (not recomputed): checkpoint_sha256={"41": "0174f1328876d5b62167d7ad738a3b4e6ec7cb9845be22cdb7cafc402bc1e42e", "42": "0082b41e23a268e8a3f9df2bee11ce51107024ae4eee47c280586460deabfc3c", "43": "b1bee658ac12035c056a77c964819ee8b919f95ab9329ca01d4e8e414a39b39f"}, climatology_interior32_t2m_rmse_k={"12": 3.488909, "24": 3.437473, "48": 3.353443, "6": 3.517116, "72": 3.318361}, code_zip_sha256=fa5752f60b00817614299b04ea5b0edea5456337b6b0a83df3068f9ad64337ab, delta_range_t2m_k={"12": [-0.0101, 0.0317], "24": [0.0012, 0.034], "48": [-0.0124, 0.0585], "6": [-0.0132, 0.0192], "72": [-0.065, 0.1428]}, disk_gib=4.6, gpu_hours=3.5339, hard_overrun_seconds=0.0, interior_32_points=4225, mean_delta_wide_minus_narrow_t2m_k={"12": 0.0147, "24": 0.0189, "48": 0.0121, "6": 0.0013, "72": 0.0231}, model_code_sha256=d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab, narrow_interior_t2m_skill_mean={"12": 0.3151, "24": 0.3668, "48": -0.1191, "6": 0.5816, "72": -0.4309}, network_bytes=0, region_points=[129, 129], reproducibility=config-reproducible; archived code.zip replays the source identity, GPU training is not bitwise reproducible, seeds=[41, 42, 43], seeds_better_than_narrow={"12": 1, "24": 0, "48": 2, "6": 2, "72": 2}, soft_overrun_seconds=3721.940569, test_read=false, training_code_sha256=6e4363d5707ea4935d5449c800e4fecefa99bf553b42a747dd4dad37f8d548db, updates=800, val_cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, wall_seconds_total=12721.940569, wide_interior_t2m_skill_mean={"12": 0.3081, "24": 0.3581, "48": -0.1271, "6": 0.5812, "72": -0.4485}

## s3-ub-update-budget

- Outcome class: `mixed`; candidate state: `not-candidate`
- Human triage priority: `98` (not a scientific score)
- Evidence: `docs/R7_S3_UB_UPDATE_BUDGET.md` (SHA256 `14a68802012047debacb6f5b20dce63740f500f59888360f4e7179ab4fd3cc00`)
- Evidence commit: `9ff4f47abe85c31efe6b03dbe2b6eb1c5fc5180c`; experiment commit: `994a0e82f9f810962192ff65ca0340c70aa38ddb`
- Protocol SHA256: `a625b9d46f3722445c44983651e177084289c5ad0bc2faa9ed391e5c467ed653`; data identity: `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09`
- Reason: S3-UB update-budget screen: the S0 observation that the 400-update l6 incumbent's loss is still descending was tested directly by running the identical recipe to 800 updates against the pinned S3-D3 control. The pre-registered primary reads supported at 6h and 12h with all three seeds the same sign (and at all five leads for t2m), and t2m skill vs the same-data climatology becomes positive at 24h for every seed -- the first time a model point estimate has been positive beyond 6h in this direction. However the u10/v10/mslp pre-screen fails 13 cells at 48h/72h, so under the frozen conjunction rule the endpoint does not advance to the S4 freeze and this round is registered as mixed/no-advance. Whole round 3963.6 s vs planned 5400 / hard 10800 with zero overrun; code fenced at 994a0e8; test unread. Node soft budget 2.5 GPU-h is now exceeded by 0.5470 (3.0470 used) and recorded as overrun per decision 0030.
- Limitations:
  - screening only: one instance (2017 train / 2022 val, one ROI, 17 channels), no significance, convergence or SOTA claim
  - the control readings come from the registered S3-D3 incumbent run (400 l6 updates) on the same store and are pinned by SHA256, not retrained inside this protocol
  - the candidate deliberately spends 2x the control's training updates and FLOPs; the verdict prices that extra training and is not a compute-matched comparison
  - three seeds are consistency evidence, not a significance test
  - K4 is an inference-depth probe on the same K4-trained checkpoint, not an independent model
  - GPU runs are co-resident; latency/memory observations include neighbor load
  - validation split only; the test split stays sealed until the S4 preregistered read
- Excluded from runnable candidates: 混合筛选结果：主格（t2m/full 6h/12h，l6_800 − incumbent_l6_400）三 seed 全部同号为负（supported，6h −0.3985/−0.5144/−0.6170 K、12h −0.3487/−0.5105/−0.6191 K，且 24/48/72h 亦全负），但 u10/v10/mslp 守门预审 48h/72h 共 13 个正 cell 未过；按冻结合取规则（primary 与 gate 须同时通过）登记为不进入 S4 的负面轮。不构成任何科学声明。
- Recorded metrics (not recomputed): candidate_updates=800, control_updates=400, delta_t2m_full_rmse_vs_incumbent={"41": {"12": -0.6191, "6": -0.617}, "42": {"12": -0.5105, "6": -0.5144}, "43": {"12": -0.3487, "6": -0.3985}}, elapsed_seconds_total=3963.6, first_spawn_to_last_reap_seconds=3911.6, gate_failures=13, gate_failures_by_lead={"48": 6, "72": 7}, gpu_hours=1.101, hard_cap_seconds=10800.0, mode=l6, network_bytes=0, planned_seconds=5400.0, positive_seed_cells_of_51={"12": 51, "24": 51, "48": 8, "6": 51, "72": 0}, primary_verdict={"12": "supported", "6": "supported", "overall": "supported"}, seeds=[41, 42, 43], soft_overrun_seconds=0.0, t2m_skill_vs_climatology_12h={"41": 0.1844, "42": 0.2101, "43": 0.1923}, t2m_skill_vs_climatology_24h={"41": 0.0774, "42": 0.1248, "43": 0.1579}, t2m_skill_vs_climatology_6h={"41": 0.4929, "42": 0.5017, "43": 0.4983}, test_read=false, updates_ratio=2.0

## s3-d4-rc-candidate

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `97` (not a scientific score)
- Evidence: `docs/R7_S3_D4_RC_CANDIDATE.md` (SHA256 `467e9591fa81a3d0731da9ba3a9536ad59a46529a507f56977635fae9ed7cabe`)
- Evidence commit: `1610b5485958fcb4613267dac7ea691004121893`; experiment commit: `c46981e139b820d35f14d183723fc7be859b7abc`
- Protocol SHA256: `c078f80432416e236a1eb4e560c48e19f262055f82042df912ff2aaf8b6864af`; data identity: `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09`
- Reason: S3-D4 R-C lead-coverage candidate single-factor screen: same spec/initialization/data/eval path as the registered S3-D3 incumbent, sole difference the training objective (control l6 at 400 updates vs candidate two_step l6+0.5*l12 at FLOP-matched 200 updates). The pre-registered primary reads worsened at both 6h and 12h with all three seeds the same sign (+0.92...+1.46 K), and the u10/v10/mslp gate pre-screen fails with 34 positive cells, so the candidate stops per the frozen rule. Whole round 3050.1 s vs planned 5400 / hard 10800 with zero overrun; code fenced at c46981e1; test unread. This independently reproduces the B-stage negative for extending the training objective to +12h on the v2 instance.
- Limitations:
  - screening only: one instance (2017 train / 2022 val, one ROI, 17 channels), no significance, convergence or SOTA claim
  - the control readings come from the registered S3-D3 incumbent run on the same store and are pinned by SHA256, not retrained inside this protocol
  - FLOP-matching follows the measured two_step/l6 objective ratio 2.0018; equal updates is deliberately NOT the comparison, so the update allocator differs
  - three seeds are consistency evidence, not a significance test
  - K4 is an inference-depth probe on the same K4-trained checkpoint, not an independent model
  - GPU runs are co-resident; latency/memory observations include neighbor load
  - validation split only; the test split stays sealed until the S4 preregistered read
- Excluded from runnable candidates: 筛选级负结果：预注册主格（t2m/full 6h/12h，two_step_200 − incumbent_l6_400）三 seed 全部同号为正，即把训练目标扩到 +12h 的候选比同 FLOP 的 l6 控制更差（6h +0.9202/+1.0247/+0.9263 K、12h +1.3219/+1.4587/+1.2365 K）；u10/v10/mslp 守门预审 34 个正 cell 也不通过。按冻结决定文本候选停当轮（candidate-stops-registered-negative），不进 S4 冻结包，不构成任何科学声明。
- Recorded metrics (not recomputed): candidate_mode=two_step, candidate_updates=200, control_mode=l6, control_updates=400, delta_t2m_full_rmse_vs_incumbent={"41": {"12": 1.3219, "6": 0.9202}, "42": {"12": 1.4587, "6": 1.0247}, "43": {"12": 1.2365, "6": 0.9263}}, elapsed_seconds_total=3050.1, first_spawn_to_last_reap_seconds=3001.2, flop_parity_relative_error=0.0008891315554136186, gate_failures=34, gpu_hours=0.8472, hard_cap_seconds=10800.0, lambda12=0.5, network_bytes=0, planned_seconds=5400.0, primary_verdict={"12": "worsened", "6": "worsened"}, seeds=[41, 42, 43], soft_overrun_seconds=0.0, test_read=false

## s3-v3-budget-dose

- Outcome class: `mixed`; candidate state: `not-candidate`
- Human triage priority: `97` (not a scientific score)
- Evidence: `docs/R7_S3_V3_BUDGET_DOSE.md` (SHA256 `c20ce54405edeeb20ef88f9ff16117cc9d93d1757a67b5e1105f32d98cbaa4bd`)
- Evidence commit: `7ec1c326301a8a914a4f0c25e9b8b9493cfcb14f`; experiment commit: `828ca548934224a64478661e026d1e034ba74640`
- Protocol SHA256: `83002900073b35a74691c50660cd1547d8556787adbd044649c1255a891f7e1e`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: v3 budget-dose screen: l6 x1600 updates on the five-train-year instance (train 2017-2021, val 2022) against the pinned v3-D3 400-update control, val-only, three seeds, FLOP ratio 4.0. Primary pre-registered cell supported: t2m/full RMSE deltas 6h -0.4328/-0.7607/-0.4444 K and 12h -0.5351/-0.7199/-0.5158 K (all seeds same sign, candidate lower); 24h also all-negative. Gate pre-screen failed with 17 positive u10/v10/mslp cells (24h 1, 48h 7, 72h 9) -- the same net count as the registered v2-BC reading at the same dose on the single-year train (17 cells at 48/72 h), so the pre-declared data-response branch reads 'persist': the gate remains nonpassing at this dose; cross-instance count is not a controlled causal test of training volume. Frozen conjunction rule -> registered-negative; the endpoint does not advance to the S4 freeze package. Candidate t2m skill vs the v3 five-year climatology: 6h +0.588/+0.586/+0.601, 12h +0.307/+0.306/+0.338, 24h +0.199/+0.127/+0.299, 48h/72h negative; positive climatology-skill cells51/51/50/4/0; lower-RMSE cells versus D3 51/51/50/22/14 (different quantities). Whole round 6883.9 s vs planned 6300 (soft overrun 583.9 s recorded) and hard 12600 untouched; conservative 1.9122 GPU-h; co-resident GPU0 with passed read-only gates (seed41 evaluation slower; neighbor attribution not established); zero network; test unread.
- Limitations:
  - screening only: one instance (2017-2021 train / 2022 val, one ROI, 17 channels); no significance, convergence or SOTA claim
  - the control comes from the registered v3-D3 run pinned by SHA256, not retrained inside this protocol
  - the candidate spends 4x the control's updates and FLOPs; not a compute-matched comparison
  - the dose is not epoch-matched (about 0.7 epochs on v3 vs 3.4 on v2)
  - three seeds are consistency evidence, not a significance test
  - GPU runs are co-resident; latency includes neighbor load
  - validation split only; the test split stays sealed until the S4 preregistered read
- Excluded from runnable candidates: 混合筛选：primary supported、全45格守门17个正cell，冻结合取仍registered-negative，不进S4；跨实例总格数相同不证明因果、收敛或退化幅度相同。勘误record:s3-v3-numerical-errata；原证据digest保留。
- Recorded metrics (not recomputed): cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, correction_evidence=docs/R7_S3_V3_NUMERICAL_ERRATA.md, decision=registered-negative, elapsed_seconds_total=6883.938784704544, first_spawn_to_last_reap_seconds=6673.923021165654, flop_ratio=4.0, gate_failures={"by_lead": {"24": 1, "48": 7, "72": 9}, "by_variable": {"mslp": 6, "u10": 5, "v10": 6}, "total": 17}, gate_passed=false, gpu_hours=1.9122, hard_cap_seconds=12600, initialization_matches_archived_actual_c_bitwise=true, lower_rmse_vs_incumbent_seed_cells_of_51={"12": 51, "24": 50, "48": 22, "6": 51, "72": 14}, network_bytes=0, planned_seconds=6300, positive_seed_cells_of_51={"12": 51, "24": 50, "48": 4, "6": 51, "72": 0}, primary_verdict=supported, seed_mean_skill={"12": 0.3171049340369213, "24": 0.20822701432390175, "48": -0.9812139278071352, "6": 0.5915221403365957, "72": -2.490233292375998}, soft_overrun_seconds=583.9387847045437, t2m_rmse_delta_k={"41": {"12": -0.5351, "24": -0.3237, "48": 0.1379, "6": -0.4328, "72": 1.1889}, "42": {"12": -0.7199, "24": -0.4513, "48": -0.2162, "6": -0.7607, "72": 0.6898}, "43": {"12": -0.5158, "24": -0.6301, "48": -0.7578, "6": -0.4444, "72": -0.5936}}, t2m_skill_vs_v3_climatology={"41": {"12": 0.3073, "24": 0.1992, "48": -1.2675, "6": 0.588, "72": -3.507}, "42": {"12": 0.3059, "24": 0.1267, "48": -1.206, "6": 0.5861, "72": -2.7871}, "43": {"12": 0.3381, "24": 0.2988, "48": -0.4701, "6": 0.6005, "72": -1.1766}}, test_read=false, updates_candidate=1600, updates_control=400, v2_bc_gate_failures_same_dose={"by_lead": {"48": 8, "72": 9}, "total": 17}

## s3-d3-incumbent-retrain

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `96` (not a scientific score)
- Evidence: `docs/R7_S3_D3_INCUMBENT.md` (SHA256 `b5193cd0ea69f19ddb2ea8ee53f3ade8ed46748aa631c3c82c62a7216968d592`)
- Evidence commit: `7d31556c26a14187b5ef5e0cb131670cd79e7401`; experiment commit: `55ab0b685182061531eb19754546435c2dd31d55`
- Protocol SHA256: `e4d521bb2e57c89a697d5ee37a6d2f505c3b7e25e06c2b6e8db149373d6467ec`; data identity: `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09`
- Reason: S3-D3 same-data incumbent retrain on the v2 confirmation instance: three pre-declared seeds were asserted bitwise-equal to the archived actual-C seeded initialization on CPU, trained 400 L6 updates on the 2017 train split through the worker's fine_tune path (frozen endpoint, no validation selection), and scored on the same per-lead val cohorts D2 used. t2m/full skill vs the same-data train-only climatology is positive at 6h for every seed (+0.21/+0.28/+0.33) and negative at 48-72h, reproducing the S0 shape on this instance; seed43 reaches +0.0037 at 12h. Whole round 3955.3 s vs planned 5400 / hard 10800 with zero overrun and zero network; test never opened. These RMSE CSVs are the pinned control the D4 R-C screen pairs against.
- Limitations:
  - screening control only: one instance (2017 train / 2022 val, one ROI, 17 channels); no significance, convergence or SOTA claim
  - three seeds are consistency evidence, not a significance test
  - K4 is an inference-depth probe on the same K4-trained checkpoint, not an independent model
  - the incumbent is retrained, not imported: actual-C numbers on the old dev store are a different instance and are not reused as this instance's control
  - GPU runs are co-resident (a 10.4 GiB neighbour appeared mid-round); latency observations include neighbor load and never signal it
  - validation split only; the test split stays sealed until the S4 preregistered read
- Excluded from runnable candidates: Same-data control rebuild only: retrains the actual-C process incumbent on the v2 train split to produce the comparison readings later candidate screens pair against. It enters no verdict, is not itself a candidate, and makes no climatology-superiority claim; the D4 screen and the S4 gate have their own protocols.
- Recorded metrics (not recomputed): cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, elapsed_seconds_total=3955.3, first_spawn_to_last_reap_seconds=3936.4, gpu_hours=1.0987, hard_cap_seconds=10800.0, initial_state_sha256={"41": "a79ea47fa098bfc4dce651e2ea6c8fd18d996e4ff88a0e8503832645a29f1c47", "42": "ec5bb5ef581e1cfe4923b938f60635fd5c79aaebd0f3f1fb37f01524ea3a26d0", "43": "844bd23402cabb3f0cfb961efa4ba2e854207a06c51b51d32fba89e6e358efdf"}, initialization_matches_archived_actual_c_bitwise=true, mode=l6, network_bytes=0, planned_seconds=5400.0, positive_seed_cells_of_51={"12": 49, "24": 47, "48": 6, "6": 51, "72": 0}, seeds=[41, 42, 43], soft_overrun_seconds=0.0, t2m_full_skill_vs_same_data_climatology={"41": {"12": -0.1655, "24": -0.2907, "48": -1.7115, "6": 0.2143, "72": -2.6343}, "42": {"12": -0.0697, "24": -0.1088, "48": -1.3401, "6": 0.2754, "72": -2.0742}, "43": {"12": 0.0037, "24": 0.0526, "48": -1.2948, "6": 0.3262, "72": -2.2304}}, test_read=false, train_windows_usable=468, updates_per_seed=400

## s3-v3-d3-incumbent

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `96` (not a scientific score)
- Evidence: `docs/R7_S3_V3_D3_INCUMBENT.md` (SHA256 `f992d5b449b160758ad9d16041d2e4c7a0ec120ac531e179783987257d7f0619`)
- Evidence commit: `984e85af39b070134c150cc4b16b62b4ef69229c`; experiment commit: `db0b463ae68b428d1beeea384dd5ea81dc7f5d22`
- Protocol SHA256: `9912a67ff3f3c1ae35f743cb956aab80fea43c39ecc61eecf32760d9e0621bbe`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: v3 incumbent control retrained on the five-year train split: three seeds reproduced the archived actual-C initialization bitwise on CPU, trained 400 L6 updates to the frozen endpoint and scored the full per-lead val cohorts (472/468/460/444/428) against the five-year train-only climatology. t2m/full skill is positive at 6h for every seed (+0.4148/+0.2610/+0.4248) and at 12h for two of three (+0.0285/-0.0805/+0.0756); 24h is one positive; 48-72h remain negative. Descriptively the near-lead shape improves on the v2 single-year control (6h seeds41/43 higher but seed42 lower, 12h two positive vs one), consistent with the data-expansion direction but not a controlled comparison. Whole round 5404.5 s vs planned 5400 (soft overrun 4.5 s recorded per decision 0030) and hard 10800 untouched; 1.5012 conservative GPU-h; co-resident GPU0 with read-only gates; zero network; test unread. These per-lead CSVs are the pinned control the v3 budget-dose screen pairs against.
- Limitations:
  - screening control only: one instance (2017-2021 train / 2022 val, one ROI, 17 channels); no significance or SOTA claim
  - three seeds are consistency evidence, not a significance test
  - K4 is an inference-depth probe on the same K4-trained checkpoint, not an independent model
  - cross-instance (v2 vs v3) comparisons are descriptive, not controlled
  - GPU runs are co-resident; latency/memory observations include neighbor load
  - validation split only; the test split stays sealed until the S4 preregistered read
- Excluded from runnable candidates: Same-data control rebuild only: it enters no candidate verdict and makes no climatology-superiority claim. The v3 budget-dose screen and the S4 gate have their own protocols.
- Recorded metrics (not recomputed): cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, correction_evidence=docs/R7_S3_V3_NUMERICAL_ERRATA.md, elapsed_seconds_total=5404.5, final_loss={"41": 0.06114, "42": 0.08334, "43": 0.13406}, first_spawn_to_last_reap_seconds=5331.8, gpu_hours=1.5012, hard_cap_seconds=10800, initialization_matches_archived_actual_c_bitwise=true, mode=l6, network_bytes=0, planned_seconds=5400, positive_seed_cells_of_51={"12": 50, "24": 45, "48": 2, "6": 51, "72": 0}, seed_count=3, soft_overrun_seconds=4.5, t2m_skill_vs_v3_climatology={"41": {"12": 0.0285, "24": 0.0218, "48": -1.1453, "6": 0.4148, "72": -2.1142}, "42": {"12": -0.0805, "24": -0.1359, "48": -1.4017, "6": 0.261, "72": -2.0213}, "43": {"12": 0.0756, "24": -0.0418, "48": -1.0692, "6": 0.4248, "72": -1.7365}}, test_read=false, updates_per_seed=400

## s0-incumbent-gap-audit

- Outcome class: `audit`; candidate state: `needs-review`
- Human triage priority: `95` (not a scientific score)
- Evidence: `docs/R7_S0_INCUMBENT_GAP_AUDIT.md` (SHA256 `e761df5aa5f8e99c861524bb5d7ecb3e9f1d7869b2ccb059c9c103a1eee413f5`)
- Evidence commit: `64a155ddbe9849af6459a34eb1c9aac45fb27fb6`; experiment commit: `562e526afc5fdbdc99053a25ab20036b66a2a8bc`
- Protocol SHA256: `ea0efb80e1b11eb7813fae74b4ee04ca67d145d16519eb8aa442af8d8851ad80`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: Establishes the campaign's fair starting point: the actual-C incumbent only exceeds the two-month train-only climatology at t2m/6h, so the main-model performance ladders (#77/#78/#79) start from a measured negative gap at 12-72 h rather than from a package-improvement claim.
- Limitations:
  - Mechanical identity and gap arithmetic only; no significance, SOTA or scientific verdict.
  - Single two-month 2016 winter segment in one region; no cross-season or cross-year claim.
  - The incumbent beats train-only climatology only at t2m/6h; every 12-72 h lead is negative.
  - K1/K2/K4 are inference-depth probes of one K4-trained checkpoint, not equal-compute models.
  - The process vs matched_generic separation remains unresolved and is not re-adjudicated here.
- Excluded from runnable candidates: Read-only identity and gap audit of an already-frozen archive; it produces no new candidate and must not be launched as a run. Any follow-up training needs a new frozen protocol.
- Recorded metrics (not recomputed): cells_recomputed=6885, gpu_hours=0.0, incumbent=process/K4/400updates, max_relative_deviation=6.629694519880788e-16, network_requests=0, positive_seed_cells_of_255=118, t2m_12h_skill_seed_mean=-0.1911, t2m_6h_skill_seed_mean=0.2073, t2m_72h_skill_seed_mean=-3.4061, test_read=false

## s3-batch3-20182021-acquisition

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `95` (not a scientific score)
- Evidence: `docs/R7_S3_BATCH3_ACQUISITION.md` (SHA256 `277bd3df1668b9a0ade36610d41ec003620d1644d9ba6317f043836b8d0fe0e3`)
- Evidence commit: `7a225038122fa7383d2924a4ac30d2a1540b01a0`; experiment commit: `7a225038122fa7383d2924a4ac30d2a1540b01a0`
- Protocol SHA256: `df9cc22b75f608ef4493c8165ffc65ae66d8d5adb7b9aed8c925c92e790772c5`; data identity: `not recorded`
- Reason: Batch-3 four-season regional ERA5 acquisition (2018-2021) completed for the expanded v3 train split: 16/16 parts downloaded real under the frozen per-part budgets with zero failures, network 57,084,203,564 bytes (98.4% of the 58 GB plan, 53.2% of the 100 GiB hard cap), and the merged 1920-stamp source 97d29bca... with exact-union timestamp re-validation. The per-part watchdog (timeout -k 60 -s TERM 1860) and the 16 GiB decode cap inherited from the batch-2 fixes were never triggered across all 16 parts - consistent with the fixes but not an independent validation of the original failure root causes. This is the registered response to the S3-UB/S3-BC budget readings: on one train year the update budget saturates, so the next capability investment is data. No part was deleted, no resume is claimed, no synthetic substitute exists, and test (2023) was never opened.
- Limitations:
  - one ROI (27-43N/107-123E) at 0.25 degrees; each season block is a 30-day sample, not a full season
  - acquisition only: no store is built here, no model is trained or scored, and no scientific claim is made
  - the per-part watchdog was never actually triggered, so its effectiveness is preventive, not demonstrated
  - the batch did not reproduce either batch-2 failure mode, so it cannot confirm those root causes were eliminated
  - the network readings are host-level recv-byte deltas and include minor non-process traffic (same method as batch-2)
- Excluded from runnable candidates: Data acquisition only: this record produces no model result and no skill number. It feeds the v3 confirmation instance build (train 2017-2021 / val 2022 / test 2023); any scientific comparison requires its own frozen protocol under the S4 gate.
- Recorded metrics (not recomputed): download_seconds_successful_parts=20709.0, gpu_hours=0.0, hard_cap_seconds=43200, merged_source_bytes=309074574, network_bytes=57084203564, network_hard_cap_bytes=107374182400, parts_failed_kept=0, parts_successful=16, per_part_deadline_seconds=1800.0, per_part_decoded_gib_cap_final=16.0, planned_network_bytes=58000000000, planned_seconds=21600, stamps=1920, synthetic_fallback=false, test_read=false

## s3-confirmation-instance-v2

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `95` (not a scientific score)
- Evidence: `docs/R7_S3_CONFIRMATION_INSTANCE.md` (SHA256 `95f6df52854deba8becec592519826f5c4586e61e40562596975a07f38818570`)
- Evidence commit: `8fa6e0cb9a83bb4450d312ef1af9ffba19809520`; experiment commit: `8fa6e0cb9a83bb4450d312ef1af9ffba19809520`
- Protocol SHA256: `f4160989b309e29b5785ca90d9c1d131bc4b0ad21109e84c4b08daadb67499b0`; data identity: `e01828e9988ba8baed8d29c7e1c2a06553700e6fe3ee6f56f6a22e31bad76a0e`
- Reason: S3 confirmation instance published as the corrected v2 build: two verified four-season batches (2017 from S1, 2022/2023 from batch-2) combined into one 1440-stamp source and built into a year-split store (train 2017 / val 2022 / test 2023) with train-only normalization and process diagnostics. The v2 build exists because the v1 audit was degraded to a file-stat-only source fingerprint by the old 64 MiB hash cap (decision 0039 removed it), which structurally blocked the process-scale sidecar the S4 incumbent training contract requires; the weather bytes are byte-identical to v1 and only the audit scope changed. All three train-only sidecars (change-scale, process-scale, typed-evidence) are published and bind one data identity. No model training, scoring or test read.
- Limitations:
  - three years of one ROI (27-43N/107-123E) at 0.25 degrees, 17 channels; season blocks are 30-day samples, not full seasons
  - store preparation only: no model result and no scientific claim; the test split is never scored
  - the fingerprint audit attests source byte identity, not the scientific adequacy of the source
  - the v1 defective build remains on disk; any formal run must point explicitly at the v2 root (mixed identities are refused by the loaders)
  - sidecar counts differ by definition (change-scale 476 train pairs; process-scale 480 history+target frames; typed-evidence 476 pairs over 4.02M field values), all bound to the same data identity
- Excluded from runnable candidates: Data preparation and identity correction only: this record is not a runnable candidate and produces no skill number. The fair confirmation on this instance requires its own frozen protocol under the S4 gate.
- Recorded metrics (not recomputed): combined_source_bytes=231865649, combined_source_sha256=e0b51616a7c31f29b9832221e74522f8b5cad4b36b72b0d7586d140787e3b42d, gpu_hours=0.0, hard_cap_seconds=3600, network_bytes=0, planned_seconds=1200, raw_gib_cap=3.0, raw_state_gib=0.3853, sidecar_identities={"change_scale": "6fd9e77456de6378b2c3116a16614541d771c81c2e24871e267c178cce311465", "process_scale": "b28300145a2d9785099fdce1358b9f142470bddb54683eb092c2e3e531d7cd7e", "typed_evidence": "8760894aa4c83ba38761111d044e37518d5b4d1a82860bd443a9b0cb25bc4495"}, stamps=1440, test_read=false, test_windows=472, train_windows=472, v1_defect_build_kept=true, v2_wall_seconds_approx=123, val_windows=472, weather_bytes_identical_to_v1=true

## s3-confirmation-instance-v3

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `95` (not a scientific score)
- Evidence: `docs/R7_S3_CONFIRMATION_INSTANCE_V3.md` (SHA256 `1449a1c5f8ed222d360d44fbd058f2f6d826c87b84f515d2d95cdc82d29a5915`)
- Evidence commit: `b9f1ae90dec3f9d049b876469c822a69f80800dd`; experiment commit: `b9f1ae90dec3f9d049b876469c822a69f80800dd`
- Protocol SHA256: `2151c91e79865541a5235854e077cc206e6865346e52ccdeb32e03a965d2461a`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: S3 confirmation instance v3 published: three verified four-season batches (2017 S1, 2018-2021 batch-3, 2022/2023 batch-2) combined into one 3360-stamp source (bc2ff9cf..., byte-deterministic on re-run) and built into a year-split store (train 2017-2021 / val 2022 / test 2023) with train-only normalization and process diagnostics. This is the registered response to the S3-UB/S3-BC budget readings: on one train year the update budget saturates, so the next capability investment is data. All three train-only sidecars (change-scale f76c373d..., process-scale 0e87fe40..., typed-evidence 6f9bedbf...) are published, load back through their strict loaders, and bind one data identity 2564eeaf.... Whole build about 509 s vs planned 1200 / hard 3600, zero network, zero GPU, test never scored.
- Limitations:
  - one ROI (27-43N/107-123E) at 0.25 degrees, 17 channels; season blocks are 30-day samples, not full seasons
  - store preparation only: no model result and no scientific claim; the test split is never scored
  - the fingerprint audit attests source byte identity, not the scientific adequacy of the source
  - v2 and v3 instance trees coexist; any formal run must point explicitly at its own root (mixed identities are refused)
  - five train years vs one does not by itself establish that more data improves pattern skill; that is the v3 screens' hypothesis
- Excluded from runnable candidates: Data preparation only: this record is not a runnable candidate and produces no skill number. The controls, the budget-dose screen and (if a candidate survives development) the unseen-year confirmation round run on this instance under their own frozen protocols.
- Recorded metrics (not recomputed): build_wall_seconds_approx=509, combine_determinism_replay_seconds=41.1, combined_source_bytes=540856239, combined_source_sha256=bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8, gpu_hours=0.0, hard_cap_seconds=3600, network_bytes=0, planned_seconds=1200, raw_gib_cap=3.0, raw_state_gib=0.8990317583084106, sidecar_identities={"change_scale": "f76c373da82073e627cb2bed9300b1e6146aafe9c4a15dec368e06e9d455d595", "process_scale": "0e87fe406764663c3cc59289bbc3c04a8e18adbc4d99a3e9cf59ac52cf8545a4", "typed_evidence": "6f9bedbf79237c3724358377f49ec7a6d571bfd7e3b0a1a69ef4bc231008dbde"}, stamps=3360, test_read=false, test_windows=472, train_windows=2360, val_windows=472

## s3-v3-d2-baselines

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `95` (not a scientific score)
- Evidence: `docs/R7_S3_V3_D2_BASELINES.md` (SHA256 `ea1e26478ca881c1d0f6e8da7d81d5c0cc4965483fce365776889a7fc5cba583`)
- Evidence commit: `984e85af39b070134c150cc4b16b62b4ef69229c`; experiment commit: `984e85af39b070134c150cc4b16b62b4ef69229c`
- Protocol SHA256: `a14dd1b2900ba964337a21dd06debffcd47ab8435a0a8412ee31a96ccffb1ed8`; data identity: `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`
- Reason: v3 baselines rebuilt on the expanded confirmation instance (train 2017-2021 / val 2022 / test 2023): the train-only climatology now fits five train years (2400 steps, 16 buckets x 150) and persistence reruns on the unchanged 2022 val split; both scored through the identical evaluate_local path on full per-lead cohorts 472/468/460/444/428. t2m climatology RMSE 3.5171/3.4889/3.4375/3.3534/3.3184 K, persistence 4.5470/5.9235/3.1633/4.1795/4.7269 K; persistence skill vs the five-year climatology is positive only at 24h (+0.1531). Whole round 992.4 s vs planned 1800 / hard 3600 with zero overrun, zero network, zero GPU; test never opened. Corrected by s3-v3-numerical-errata: climatology t2m is lower at 6/12/24 h, higher at 48/72 h; exact6h difference -0.0232417992 K. Code-at-freeze is not recorded in D2 protocol.
- Limitations:
  - parameter-free reference scoring only: no model result and no skill claim
  - one ROI, five train years plus one eval year, 17 channels; val is 2022 only
  - the climatology is a train-only (month,hour) grid-cell mean, not a WeatherBench2 reproduction
  - the v2/v3 numerical comparison is descriptive across instances, not a controlled comparison
  - actual executing commit not independently established; experiment_commit retains original registration pointer, not a verified freeze identity
- Excluded from runnable candidates: Data preparation (baselines) only: this record is not a runnable candidate and produces no model skill number. The v3 incumbent control and the budget-dose screen consume it under their own frozen protocols.
- Recorded metrics (not recomputed): climatology_n_selected_steps=2400, climatology_training_years=[2017, 2018, 2019, 2020, 2021], cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, correction_evidence=docs/R7_S3_V3_NUMERICAL_ERRATA.md, elapsed_seconds_total=992.4, gpu_hours=0.0, hard_cap_seconds=3600, network_bytes=0, persistence_skill_t2m={"12": -1.8825, "24": 0.1531, "48": -0.5533, "6": -0.6714, "72": -1.0291}, planned_seconds=1800, t2m_rmse_climatology_k={"12": 3.4889, "24": 3.4375, "48": 3.3534, "6": 3.5171, "72": 3.3184}, t2m_rmse_persistence_k={"12": 5.9235, "24": 3.1633, "48": 4.1795, "6": 4.547, "72": 4.7269}, test_read=false

## s3-batch2-20222023-acquisition

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `94` (not a scientific score)
- Evidence: `docs/R7_S3_BATCH2_ACQUISITION.md` (SHA256 `3ed6e91182e1047e9649c5ed278212f41f1600b0d65110a53e3c34ce89f89c37`)
- Evidence commit: `c791317a3a006ddb71e62363976a8d6a0842bf62`; experiment commit: `c791317a3a006ddb71e62363976a8d6a0842bf62`
- Protocol SHA256: `7847fa1e9ebe94cf7d5d2fd138a973024844dd1be249563d5d5821d9e3cf88cb`; data identity: `not recorded`
- Reason: Batch-2 four-season regional ERA5 acquisition (2022/2023) completed for the confirmation instance: 8/8 parts downloaded real under the frozen per-part budgets, network 28,773,423,423 bytes (99.9% of plan, 55.8% of hard cap), and the merged 960-stamp source 44a24ca0... with exact-union re-validation. Two genuine failures are kept as failed-no-fallback with no synthetic substitute: winter_2022 refused by the frozen 8 GiB per-part decode cap (amended to 16 GiB, the value S1 actually passed under, before any successful part; retried as a NEW artifact name) and winter_2023 hung past its 1800 s deadline (executor terminated its own process, wrote an honest audit receipt because the downloader's failure path does not cover SIGTERM, and added a per-part timeout watchdog; retried as winter_2023_r2). No part file was ever deleted; no resume is claimed.
- Limitations:
  - two years of one ROI at 0.25 degrees; each season block is a 30-day sample, not a full season
  - acquisition only: no store is built here, no model is trained or scored, and no scientific claim is made
  - winter_2023's hang has no stack-level root cause (the process was terminated); single-stream blocked-read is a hypothesis consistent with the isolated per-connection rate-limit readings, not a controlled reproduction
  - the transport-concurrency speedup note is a cross-window working handoff, not registered evidence; actual throughput in this batch was serial single-stream
- Excluded from runnable candidates: Data acquisition only: this record produces no model result and no skill number. It feeds the S3 confirmation instance build; any scientific comparison requires its own frozen protocol under the S4 gate.
- Recorded metrics (not recomputed): download_seconds_successful_parts=9832.8, gpu_hours=0.0, hard_cap_seconds=28800, network_bytes=28773423423, network_hard_cap_bytes=51539607552, parts_failed_kept=2, parts_successful=8, per_part_deadline_seconds=1800.0, per_part_decoded_gib_cap_final=16.0, planned_network_bytes=28800000000, planned_seconds=10800, stamps=960, synthetic_fallback=false, test_read=false

## s3-d2-same-data-baselines

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `93` (not a scientific score)
- Evidence: `docs/R7_S3_D2_BASELINES.md` (SHA256 `6498f5e1952b43a43056532ea303f416b1e19e3733ac833fcf2ce3d9f43adcec`)
- Evidence commit: `4a3c309ce609590d6435fc158f38b2bae96c9b55`; experiment commit: `4a3c309ce609590d6435fc158f38b2bae96c9b55`
- Protocol SHA256: `dba134239c254f1cd909e7b4f86631ef10ee79746b2acb3552b84a6b94d38ca3`; data identity: `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09`
- Reason: S3-D2 same-data reference rebuild on the v2 confirmation instance: the train-only (month,hour) climatology (fitted on 2017 train only, 480 steps, complete 16-bucket coverage, fail-closed) and persistence scored through the identical evaluate_local path and per-lead full val cohorts (472/468/460/444/428) a later incumbent/candidate uses, so their RMSEs are same-case and same-unit comparable. Zero GPU, zero network, test never opened; 553.1 s against planned 1800 / hard 3600 with zero overrun. Climatology identity was byte-identical across the two arms' independent runs.
- Limitations:
  - parameter-free reference scoring only; it produces no model result and no skill claim
  - the climatology is a train-only (month, hour) grid-cell mean, not a WeatherBench2 reproduction
  - one ROI, three years, 17 channels; val is 2022 only
  - persistence skill vs climatology at 24h is a property of this 30-day val block, not a physical conclusion
- Excluded from runnable candidates: Parameter-free reference scoring only: (month,hour) train-only climatology and persistence produce no model result, no skill claim and enter no verdict. They are the same-case denominators a later incumbent/candidate comparison needs; the model-side comparison requires its own frozen protocol under the S4 gate.
- Recorded metrics (not recomputed): bucket_count=16, bucket_max_count=30, bucket_min_count=30, climatology_fit_steps=480, climatology_fit_years=[2017], cohorts={"12": 468, "24": 460, "48": 444, "6": 472, "72": 428}, elapsed_seconds_total=553.1, gpu_hours=0.0, hard_cap_seconds=3600.0, network_bytes=0, persistence_skill_vs_climatology_t2m_full={"12": -1.8507000588003528, "24": 0.16054305008166647, "48": -0.5544940980001842, "6": -0.6495262105328024, "72": -1.0373428186765081}, planned_seconds=1800.0, soft_overrun_seconds=0.0, t2m_full_climatology_rmse_K={"12": 3.5083358322351277, "24": 3.4525900928001483, "48": 3.352207924793912, "6": 3.5403577941696134, "72": 3.3116810336238727}, t2m_full_persistence_rmse_K={"12": 5.923479932137085, "24": 3.163328065604577, "48": 4.179511140488608, "6": 4.547018959975189, "72": 4.726945088063722}, test_read=false

## s1-four-season-2017-dev-store

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `92` (not a scientific score)
- Evidence: `docs/R7_S1_FOUR_SEASON_ACQUISITION.md` (SHA256 `cea9ea13d99b5acaa6a753dde1fe6a008f330702a8c5a2b223b77733f2f08be1`)
- Evidence commit: `6814c460cd11f16ab6837d8f18261e37e0edb353`; experiment commit: `6814c460cd11f16ab6837d8f18261e37e0edb353`
- Protocol SHA256: `5abdea4b00c05a643e1f5706250acf4d9bedafde530663516ef1a79c7328ddd1`; data identity: `894b8d1b6c08d49f93255558fdacb1290ace690de43757d84369a91b3b28e02c`
- Reason: One-year four-season real ERA5 segment (2017, ROI 27-43N/107-123E, 17 channels, 480 six-hourly stamps) downloaded in four season parts under decision 0038, merged with exact-union timestamp re-validation, and published as a dev store with train-only normalization and process diagnostics. The store carries a corrected time-range split whose scored October buckets are fully covered by train; the originally intended month-disjoint plan was mechanically unscorable under the fail-closed train-only climatology and was rejected before any build or score.
- Limitations:
  - single year and single region; this is a dev screening instance, not a confirmation dataset
  - val/test are 6-15 day windows inside one season block, not complete seasons
  - the train-only climatology so far covers four sampled months of one year
  - no cross-region, cross-year or significance claim; the test split was not opened
- Excluded from runnable candidates: Data preparation only: the dev store screens #77/#78 mechanisms and never produces a scientific claim or a runnable candidate. The fair confirmation requires the separate multi-year acquisition (unseen years 2022/2023) described in the record limitations.
- Recorded metrics (not recomputed): download_seconds=4604.432, gpu_hours=0.0, network_bytes=14390323242, network_hard_cap_bytes=25769803776, parts=4, planned_network_bytes=14504924706, stamps=480, store_build_seconds=16.83, test_read=false, test_windows=22, train_windows=340, val_windows=34

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

## s2-78-change-scale-screening

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `90` (not a scientific score)
- Evidence: `docs/R7_S2_78_CHANGE_SCALE.md` (SHA256 `22b9f2a834ea19ddfcd9587646c1e4fa247a068dc99185fa98111564d46253b1`)
- Evidence commit: `833d6bbf65885d1ef9f5d888bac5329d0dcdb5c8`; experiment commit: `833d6bbf65885d1ef9f5d888bac5329d0dcdb5c8`
- Protocol SHA256: `84dafc498c69c4ef0cd0e16b4dea5cbe0e2e9fd6915ee782e6977ca62997e074`; data identity: `894b8d1b6c08d49f93255558fdacb1290ace690de43757d84369a91b3b28e02c`
- Reason: #78 R-A 变化尺度重参数化解码单因素筛选：同参数量/FLOPs/初始化的 identity vs normalized_change_scale 两臂（Y = X_t + (d_c/s_c)*r_c，loss 不变），三 seed、400 更新、val-only。接线探针证明 ratio 确实进入前向（≤2.6e-5），但预注册主格两 lead 三 seed 同号为正（worsened），按冻结决定文本该机制在本预算/实例被证伪；全 85 cell 如实报告，长 lead 的改善多数格仅作描述不作主张。证据：outputs/r7_78_change_scale_pilot_v2/paired_comparison.json。
- Limitations:
  - 单 dev store（2017 四季、单区域），val-only；val/test 是十月内 6–15 天窗口，非完整季节。
  - 400 更新/seed 的有界筛选，无收敛或显著性主张；三 seed 只作一致性证据。
  - 主格两 lead 三 seed 同号为正（worsened）；24–72h 的多数 improved 格是描述性、未预注册，不并入判定。
  - v1 三 seed 在训练前被接线探针拒绝（harness 缺陷，7 s/seed），已修复并 v2 重跑；两轮均计费。
  - test_read: false；未改任何冻结判据；R-B（差值尺度 loss）未因此轮做任何主张。
- Excluded from runnable candidates: 筛选级负结果：预注册主格（t2m 6h/12h，change_scale−identity）三 seed 全部同号为正，即变化尺度重参数化使 val RMSE 恶化（6h +0.1105、12h +0.1668 seed 均值）；按冻结决定文本被证伪。不进入正式确认实例候选，不构成任何科学声明。
- Recorded metrics (not recomputed): arms=2, cells_improved=30, cells_unresolved=44, cells_worsened=11, change_scale_ratio_max=0.747278094291687, change_scale_ratio_min=0.07220174372196198, delta_seed_mean_12h=0.16681554533105677, delta_seed_mean_6h=0.1105266114221289, forward_backward_flops=46282984704, forward_flops=15465592704, gpu_hours=0.37003, gpu_hours_total=0.3759, gpu_hours_v1=0.0059, gpu_hours_v2=0.37003, parameters_per_arm=2948771, primary_cells={"12h": "worsened", "6h": "worsened"}, probe_max_relative_error=2.64e-05, seeds=3, soft_overrun_seconds=0.0, updates_per_arm=400

## s2-78-rb-loss-screening

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `90` (not a scientific score)
- Evidence: `docs/R7_S2_78_RB_LOSS.md` (SHA256 `4f6bd885f64d3b91e368290d71459464a0594cb5baba07812241690da73dd231`)
- Evidence commit: `1c4b366dd5c2740fa41f2087f8fddc38b1bb79a2`; experiment commit: `1c4b366dd5c2740fa41f2087f8fddc38b1bb79a2`
- Protocol SHA256: `4bee95613783c2950b9800cbcb70b9b0712afb78fb4f31ed6b8d97f3a20fad73`; data identity: `894b8d1b6c08d49f93255558fdacb1290ace690de43757d84369a91b3b28e02c`
- Reason: #78 R-B 变化尺度加权损失单因素筛选（R-A 的另轮）：同参数量/FLOPs/初始化的两臂，解码均保持 identity，只把训练目标换成 w_c=(1/runtime_ratio_c)^2（归一化均值 1）的逐通道加权，三 seed、400 更新、val-only。接线探针三 seed 相对误差 0.0（权重确实只改定价），但预注册主格两 lead 三 seed 同号为正（worsened），按冻结决定文本该目标在本预算/实例被证伪；全 85 cell 如实报告。证据：outputs/r7_78_rb_loss_pilot/paired_comparison.json。
- Limitations:
  - 单 dev store（2017 四季、单区域），val-only；val/test 是十月内 6–15 天窗口，非完整季节。
  - 400 更新/seed 的有界筛选，无收敛或显著性主张；三 seed 只作一致性证据。
  - 主格两 lead 三 seed 同号为正（worsened）；48h 的多数 improved 格是描述性、未预注册，不并入判定。
  - 权重口径单一（w_c=(1/runtime_ratio)^2 归一化均值 1，train-only sidecar），未试其他口径。
  - test_read: false；未改任何冻结判据；与 R-A 的机制差异已在协议内写定（解码不变、只改目标权重）。
- Excluded from runnable candidates: 筛选级负结果：预注册主格（t2m 6h/12h，loss_change_scale−loss_identity）三 seed 全部同号为正，即按变化尺度加权的训练目标使 val RMSE 恶化（6h +0.3101、12h +0.4806 seed 均值）；按冻结决定文本被证伪。加权目标不进入正式确认实例候选，不构成任何科学声明。
- Recorded metrics (not recomputed): arms=2, cells_improved=21, cells_unresolved=50, cells_worsened=14, delta_seed_mean_12h=0.48057, delta_seed_mean_6h=0.310095, forward_backward_flops=46282984704, forward_flops=15465592704, gpu_hours=0.3533, loss_weight_max=6.778827422832301, loss_weight_min=0.06328276508370698, parameters_per_arm=2948771, primary_cells={"12h": "worsened", "6h": "worsened"}, probe_max_relative_error=0.0, seeds=3, soft_overrun_seconds=0.0, updates_per_arm=400

## s2-79-typed-evidence-screening

- Outcome class: `engineering-positive`; candidate state: `needs-review`
- Human triage priority: `90` (not a scientific score)
- Evidence: `docs/R7_S2_79_TYPED_EVIDENCE.md` (SHA256 `002f319f21981b8853c72253cb21a3f67cd5621150cd89b170d020b7487fdbec`)
- Evidence commit: `3ad1d97284a934bfeb4fdb2a7f081d6cd032d300`; experiment commit: `3ad1d97284a934bfeb4fdb2a7f081d6cd032d300`
- Protocol SHA256: `bd093a1591e9f5f35e456c833ed7bc760ff313ee85c7c866bca5795aa5cba6e6`; data identity: `894b8d1b6c08d49f93255558fdacb1290ace690de43757d84369a91b3b28e02c`
- Reason: #79 类型诊断证据三臂（A 现役 V2 / B 同容量同信息无类型融合 / C 类型路由），aux=0、三 seed、400 更新、val-only。接线逐 seed 核验（off 路径位级不变、4×4 到达矩阵严格对角、B/C 参数量与 FLOPs 相等）。注册主格以容量匹配配对为准：B→C 6h/12h 三 seed 同号为负，supported；整体 C−A 两 lead unresolved，故仅允许作为下一确认实例的设计输入。证据：outputs/r7_79_typed_evidence_pilot/paired_comparison.json。
- Limitations:
  - 单 dev store（2017 四季、单区域），val-only；val 是十月内 6–15 天窗口，非完整季节。
  - 400 更新/seed 的有界筛选，无收敛或显著性主张；三 seed 只作一致性证据。
  - B→C 归因 supported 只说明同容量同信息下类型结构优于无类型融合；C 相对 incumbent 与 climatology 均无优势（三臂 t2m skill ≈ −0.11）。
  - 短 lead 改善/长 lead 恶化的分界仅描述性、未预注册，不并入判定。
  - test_read: false；未改任何冻结判据；未做 R-B（差值尺度 loss）。
- Excluded from runnable candidates: 筛选级正结果：预注册的容量匹配归因配对 typed_routing − generic_fusion 在 t2m 6h/12h 三 seed 全部同号为负（seed 均值 6h −0.053297、12h −0.053297），按冻结决定文本 supported；相对 incumbent 的整体配对（C−A）6h/12h 均 unresolved，故本读数不构成超过现状或气候态的主张，只允许把 typed evidence 带进下一确认实例的设计冻结讨论。
- Recorded metrics (not recomputed): arms=3, cells_overall_improved=11, cells_overall_unresolved=63, cells_overall_worsened=11, cells_typed_improved=13, cells_typed_unresolved=64, cells_typed_worsened=8, delta_seed_mean_12h=-0.053297, delta_seed_mean_6h=-0.053297, forward_flops_evidence=16776992640, forward_flops_incumbent=15465592704, gpu_hours=0.7129, gpu_hours_training=0.498, parameters_per_arm_evidence=3097571, parameters_per_arm_incumbent=2948771, primary_cells={"12h": "supported", "6h": "supported"}, primary_pair=typed_routing - generic_fusion, probe_off_path_identity=true, probe_typed_arrival_diagonal=true, secondary_pair_12h=unresolved, secondary_pair_6h=unresolved, seeds=3, soft_overrun_seconds=0.0, updates_per_arm=400

## s2-77-pe-band-screening

- Outcome class: `negative`; candidate state: `not-candidate`
- Human triage priority: `88` (not a scientific score)
- Evidence: `docs/R7_S2_77_PE_BAND.md` (SHA256 `7474941004626874f0d21c0edeaa37b0add8e3ec5aca01a4cfe21e97b3f949ff`)
- Evidence commit: `74c22791cebac3a5eb3cd3dfdead5f3de518d7d2`; experiment commit: `74c22791cebac3a5eb3cd3dfdead5f3de518d7d2`
- Protocol SHA256: `503df8e43dc2cf56e472b196c7d1f157d77a8b46c6a4d577201487f5485e1c80`; data identity: `894b8d1b6c08d49f93255558fdacb1290ace690de43757d84369a91b3b28e02c`
- Reason: #77 位置编码频带单因素筛选：同参数量/FLOPs/初始化的 legacy vs nyquist_band 两臂，三 seed、400 更新、val-only。机制探针确认开关生效（16→192 live channels），但预注册主格在两 lead 均符号不一致（unresolved），按冻结决定文本该机制在本预算/实例上未成为杠杆；全 85 cell 如实报告，不做方向追认。证据：outputs/r7_77_pe_band_pilot_v2/paired_comparison.json。
- Limitations:
  - 单 dev store（2017 四季、单区域），val-only；val/test 是十月内 6–15 天窗口，非完整季节。
  - 400 更新/seed 的有界筛选，无收敛或显著性主张；三 seed 只作一致性证据。
  - 主格两格均 unresolved（每 seed 增量符号不一致）；18/14/53 improved/worsened/unresolved 不并入判定。
  - v1 轮 writer 缺陷已被共享合并门拒绝、产物保留并计费；v2 为注册轮（修复见 74c2279）。
  - test_read: false；未改任何冻结判据、未扩臂、未换开关。
- Excluded from runnable candidates: 筛选级负结果：预注册主格（t2m 6h/12h，band−legacy）三 seed 符号不一致，按冻结规则无 seed 均值、无方向判定；机制探针确认开关生效但未转化为验证收益。不进入正式确认实例候选，不构成任何科学声明。
- Recorded metrics (not recomputed): arms=2, cells_improved=18, cells_unresolved=53, cells_worsened=14, forward_backward_flops=46282984704, forward_flops=15465592704, gpu_hours=0.3297, gpu_hours_total=0.6498, gpu_hours_v1=0.3201, gpu_hours_v2=0.3297, legacy_dead_channels=68, legacy_live_channels=16, nyquist_dead_channels=0, nyquist_live_channels=192, parameters_per_arm=2948771, primary_cells={"12h": "unresolved", "6h": "unresolved"}, seeds=3, soft_overrun_seconds=0.0, updates_per_arm=400

## m3-process-supervision-budget-limited

- Outcome class: `audit`; candidate state: `blocked`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_73_PROCESS_SUPERVISION.md` (SHA256 `3eb2ea3472b6595c146ab6b5fe51180d5e72d9cd5852b91c2afd7bfbc145cb53`)
- Evidence commit: `ed1a03414fe2bccd7911beffcc69da6358ba1f17`; experiment commit: `d6c98cf1c33eca5885772c473805af3ef0ad62ba`
- Protocol SHA256: `404cf32b8ee8f6c3ff192d46c1de6765abe4ae3fa72967469af800a774fde15d`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: M3 train-only dimensioned scaling and three timed supervision sources are engineered and CPU/CI verified. The one authorized real round completed all six400-update trainings but only seven of thirty evaluations before the continuous deadline killed the eighth owned worker. Partial119 RMSE cells and costs are independently audit-valid, not an accepted full three-arm experiment; no forecast contrast verdict, retry, automatic advancement or goal completion.
- Limitations:
  - Offline stdlib audit of frozen records/bytes, not a rerun of training, evaluation, initialization or backward.
  - CPU pairing, parameters, FLOPs and gradient norms are checked as pinned recorded evidence, not independently remeasured.
  - Zarr compressed fields/chunks and checkpoint payloads are not decoded; raw train statistics and checkpoint internal contracts are not reexecuted.
  - Selected checkpoint bytes/report signatures are checked; pinned worker code checks payload contracts. No independent OS/GPU attestation is claimed.
  - ACC values are validated for coverage/status/identity/skill consistency, not reconstructed from forecast tensors.
  - FLOPs omit supported-counter-excluded elementwise and normalization arithmetic and are not streamed-training FLOPs.
  - One winter segment, two seeds, validation only: descriptive signs are not significance, convergence, causality or generalization.
  - Source is opaquely hashed; no sealed test manifest/name contents or held-out fields are opened.
  - Code/data/protocol-pinned reproducibility and recorded bitwise initial pairing only; GPU result bitwise reproducibility is not established.
  - Engineering success is separate from D5 full coverage; no self-goal completion, next-node progression or issue closure.
  - Only7/30evaluations and119/510RMSE cells completed; future_draft_aux and all seed42 forecast scoring remain missing.
  - The execution-phase cap held at1790.1153s, but1805.1086s whole wall exceeded30minutes because pre-execution CPU identity checks were outside the hard deadline. No cap/criterion/archive was revised.
  - MetPy was not installed/run; analytic and NumPy oracle tests do not claim MetPy execution.
  - Terminal merged/paired/four cost tables were not generated; CPU parameters/FLOPs and sixtraining/seveneval raw costs cannot be called accepted full cost delivery.
  - No new full security scan conclusion; credentials and security configuration unchanged.
- CI run: `36973907623`
- Excluded from runnable candidates: Blocked on23 missing evaluations, complete two-seed comparisons and terminal four cost views. Whole wall1805.1086s exceeded the frozen30min wording. One-shot authorization is consumed; budget remainder is not permission to complete/retry.
- Recorded metrics (not recomputed): arms=3, attempt_sha256=525545bc6900f05c0d11f27c2b12bd5db9af39fa3740d52220e7e156009c60ef, attempt_status=failed, automatic_retry=false, budget_limited=true, campaign_remaining_gpu_hours_exact=19.29828558477297, campaign_status=budget_limited, campaign_used_gpu_hours_exact=4.701714415227032, case_evaluations_completed=131, current_node=N2a, engineering_exact_clone_passed=2267, engineering_exact_clone_skipped=14, engineering_full_local_passed=2272, engineering_full_local_skipped=9, engineering_verified=true, evaluations_completed=7, evaluations_missing=23, evaluations_planned=30, execution_attempt_sha256=1c5f86f0ec527b628bbdd21f709d2d17e34cee886aec71d47ade3514d69d50d1, full_experiment_accepted=false, gpu_hours=0.49725426027008024, gpu_hours_cap=1.0, gpu_phase_seconds=1790.1153369722888, independent_audit_status=incomplete-audit-valid, independent_receipt_sha256=73afca69d42de3ae8d837b19916f1fc9070cc61bd111782a604682aa00adfe45, independent_run_files_unchanged=109, paired_comparison_complete=false, reasoning_steps=4, reproducibility_level=config-reproducible training; bitwise initialization paired only; independent persisted-artifact verification, not GPU replay, rmse_cells_completed=119, rmse_cells_planned=510, seeds=[41, 42], sidecar_active_channels=8, sidecar_identity=4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d, sidecar_train_frames=188, terminal_four_cost_views_complete=false, test_read=false, thresholds_added=0, training_runs_completed=6, training_runs_planned=6, updates_per_arm=400, whole_wall_cap_seconds=1800.0, whole_wall_seconds=1805.1085775829852, whole_wall_within_cap=false

## m3-validation-complement-complete-paused

- Outcome class: `audit`; candidate state: `blocked`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_73_VALIDATION_COMPLEMENT.md` (SHA256 `17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`)
- Evidence commit: `1de3a672d859b42efa7a5fac3293c2832df4a6ee`; experiment commit: `011ab4cdc3b4fa9f6671a22962a2cf3227998c66`
- Protocol SHA256: `74d05ab36a5c15451893e20f9cadcaecdac62170717af61a3e7524af57509dd5`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: Independent zero-training supplementation completed all missing23 archived-code validation evaluations. Original6 training+7 evaluation sources are unchanged; full30 evaluations/510RMSE/528case/3pairs85 cells and four cost views are independently engineering accepted. Frozen unresolved exit triggered scientific pause.
- Limitations:
  - Two seeds on one winter segment; not significance, convergence, generalization or causal process contribution.
  - Driver base011ab4c differs from original archived forecast evaluator base d6c98cf; old zip80 byte identity governs, not a clean old commit claim.
  - Original failed attempt and all full costs remain unchanged.
  - Metadata comparator replay exact, not a second GPU forecast evaluation or bitwise GPU reproducibility claim.
  - Stage artifact manifest/cost pending unchanged; final round sealed only by actual sidecar receipt.
  - Parameters/FLOPs and training throughput quote original measured evidence; no new training/CPU profile measurement.
  - Independent stdlib verification reads opaque checkpoint/source and CSV/per-case metadata, not arrays or complete runtime syscall attestation.
  - No new data publication, download, credentials, main push or issue closure. The user disabled Mimosa; the executor did not edit security settings and the local configuration change is excluded from registration commits.
  - Mimosa hook reports12high/1low in read-only legacy with incomplete coverage; no independent exploit or full project security clearance.
  - Current targeted CPU recheck and metadata/hash audit do not cover MetPy, a new GPU replay or a full real-17-channel CPU train-to-evaluate integration. Sensitive BF16 arithmetic tests do not establish unquantized-FP32 equivalence.
- CI run: `37036869969`
- Excluded from runnable candidates: Complete audit artifacts, not a runnable candidate or scientific success. N3/N4/N5 remain blocked by the original any-unresolved exit. D6 proceeds after the user disabled Mimosa; no official finding disposition or L3 security clearance is claimed.
- Recorded metrics (not recomputed): L3_security_clearance=false, advance_next_node=false, aggregate_rows=255, all_new_baselines_zero=true, attempt_status=paused, campaign_status=paused, case_evaluations=528, cells_per_pair=85, current_node=N2a, engineering_full_cpu_passed=2395, engineering_full_cpu_skipped=9, evaluations=30, evidence_commit_available=true, executor_changed_security_configuration=false, final_round_sealed=true, four_cost_views_complete=true, gpu_hours=0.17899193391850632, gpu_phase_seconds=644.3709621066228, hard_cap_seconds=3600, independent_seal_sha256=0810665b861a3f2206b53556a6878b179e68f2923e560e11c7ef906af6797a72, independent_verification_sha256=576a72b26efd4148ddc9a6ffc59347c8e3bddb0b979968f6f9fcd5167ca2ab79, new_attempt_sha256=401b270fdfe378dfd19e4cc29e7cb9309e2349aaf689ef862ca7b05b5a380e5e, new_distinct_worker_pids=23, new_evaluations=23, new_files_sealed=276, official_disposition_received=false, original_evaluations=7, original_failed_preserved=true, original_files_unchanged=109, original_gpu_hours=0.49725426027008024, pair_totals={"future_draft_aux - aux_off": {"improved": 19, "unresolved": 21, "worsened": 45}, "future_draft_aux - input_aux": {"improved": 3, "unresolved": 38, "worsened": 44}, "input_aux - aux_off": {"improved": 21, "unresolved": 28, "worsened": 36}}, paired_comparison_sha256=5e845badb28086a32ef016167b3efc48114fe303780fd3d13898b2fa035c2d68, planned_seconds=1800, postrun_targeted_passed=221, postrun_targeted_skipped=0, registration_ci_available_at_index_write=false, registration_full_cpu_log_sha256=2ec0741610d13c681dc27c9ff68e818975ee1ca9d56f379aabfb7eff0c534dab, registration_full_cpu_passed=2395, registration_full_cpu_seconds=230.96, registration_full_cpu_skipped=9, registration_full_cpu_warnings=2, registration_status_at_index_write=evidence-frozen-index-recorded-ci-pending, reproducibility_level=code/data/protocol-pinned; original training config-reproducible; exact metadata comparator replay only, rmse_cells=510, scientific_halt=true, soft_overrun_seconds=0.0, test_read=false, total_m3_gpu_hours=0.6762461941885866, training_runs_reused=6, training_updates=0, user_disabled_mimosa=true, whole_wall_seconds=700.5363800507039

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

## n3-autoregressive-attempt-failed

- Outcome class: `audit`; candidate state: `blocked`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_74_AUTOREGRESSIVE_ATTEMPT.md` (SHA256 `0f25cba37ac14f5c52fca21d337a0fbf2771d7f652bde3deefff2643abda0cdc`)
- Evidence commit: `42ecb890ce8f8f1d2925f8841a74859b2decd8f7`; experiment commit: `616b029dce569b92bd08295737981512180a1ad3`
- Protocol SHA256: `9f76e6e3e0eb0d7e27d1aaaa8d59ff7616241ff50e13c30b26be7f13f5e62b02`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: The newly authorized independent N3 route under decisions 0030/0032 completed six trainings/1600 optimizer updates and thirty validation evaluations in 36 successful fresh CUDA workers, but the original CLI exited one when aggregate loss auditing recombined FP32 losses in binary64. Attempt status remains failed and finalized false; complete worker inventory is not a complete accepted experiment. Independent review accepts only the full original continuous wall/GPU costs, including failure and cleanup. Original failed attempts and M3 scientific pause are preserved; that historical pause does not block this separately authorized route or get rewritten by it. Decision 0036's independently frozen zero-GPU statistics completion remains pending, with no repeat of the six trainings or thirty GPU evaluations.
- Limitations:
  - All six trainings and thirty evaluation workers succeeded, but original aggregate failed/CLI exit one/finalized false/outcome null are retained. jobs_completed 36 and partial false describe worker inventory only, never full experiment acceptance.
  - The binary64 L6 + 0.5*L12 check incorrectly compared FP32-derived losses at physical-metric 1e-9/1e-12 tolerances. It rejected 99/88 of the 200 rollout records per seed; operation-by-operation FP32 composition matches all 400 exactly. This frozen metadata diagnosis is not a relaxed weather criterion or a newly accepted aggregate.
  - stage_result, paired, merged and artifact_manifest were not produced. Independent full statistics, units, sufficient statistics, official pairing, UTC grouping and candidate selection remain unaccepted; no favorable partial CSV numbers are a verdict.
  - The full original 0.8526056814201487 GPU-h and 3200.912431293167 wall seconds are accepted costs only. Summed successful-worker seconds are smaller and are not substituted for the continuous GPU interval; final receipt serialization follows the snapshot, not a measured exact CLI-return time.
  - The independent cost audit checks persisted clocks, identities, allocator baselines, process records and opaque hashes, not weather/model tensor decoding, a GPU rerun or full syscall attestation. Its two initial clock-parser rejections remain preserved; revision02 uses the true cost anchors and records the 0.325-microsecond reap/sample discrepancy.
  - 361 original files/3089276264 bytes are sealed and unchanged in the frozen review. This index registration only reads frozen pages/git blobs and copies their pointers; it does not read actual output, weather/source or checkpoint payloads.
  - Two seeds, one winter region, 185 train windows and a small developed validation set cannot establish significance, convergence, causal process contribution, cross-season/generalization or SOTA. The sealed test was not read.
  - Each lead uses its full 22/21/19/15/11 validation cohort, not a common 72h subset. Persistence/train-only climatology and negative/undefined raw metrics are retained; no accepted pooled weather contrast is claimed.
  - All arms use new optimizers and same-seed M3 aux_off/RW-A/K4 parents, normalized initial/all-five-draft supervision and full internal/physical graphs. The equal-compute arm is a 400-update control, not a strict equal-FLOPs or wall-time guarantee.
  - Measured FlopCounterMode forward/backward values omit unsupported elementwise/normalization arithmetic; worker elapsed time is not isolated model latency or a device-wide memory measure.
  - The second history retains the first prediction, not future truth; physical transitions use Gregorian +6h while internal K stays at one valid time. Known-context query switches remain false, so B is not the decision-0035 local-solar/history-offset/source-position contrast.
  - The genuine original #72 reference is RW-A/K3, not this M3 K4 parent. Its compatible archived bridge is checked but the reference was not run; old 365.25/cumulative-lead semantics differ from B Gregorian/fixed +6h and cannot be called calendar matched or bitwise score replay.
  - Code/data/protocol-bound numerical reproducibility is the ordinary training claim, not bitwise GPU training. The separate small FP32/BF16 probe does not replace this weather comparison.
  - Original M3 failed/complete-supplement-paused/any-unresolved/advance-false records remain intact. Independent-route authorization is new and does not retroactively change their scientific halt.
  - Campaign 24 GPU-h is a historical bookkeeping base, not an authorization ceiling; every failed interval is charged. A later zero-GPU statistics supplement must not charge the original GPU interval twice or revive/retry the original attempt.
  - CI is anonymous Jobs API engineering success observed 2026-10-03, not authenticated pytest-log download or later registration CI. No independent C/adaptive verdict, six issue closures, main advancement or final goal completion is claimed.
  - No new data writing/publication/download, dependency installation, credential/security change, paid resource or neighbor-process operation is part of this record.
- CI run: `37133487340`
- Excluded from runnable candidates: Blocked on independent full numerical/statistical acceptance in new outputs, not missing worker computation. The original failed attempt, protocol, checkpoint/CSV bytes and earlier failed audits remain unchanged; only original wall/GPU cost is accepted, not aggregate, weather comparison, candidate selection or full cost-table delivery. No finalize on the failed original, no GPU rerun, no M3 exit-rule change and no fabricated registration CI. The new independent-route authorization is separate from historical M3 pause and this index grants no further execution permission.
- Recorded metrics (not recomputed): all_allocator_baselines_zero=true, archived_source_members=93, arms=["continue_l6", "rollout_l6_l12", "equal_compute_l6"], artifact_manifest_produced=false, attempt_path=outputs/r7_74_autoregressive_20261003_attempt01/attempt.json, attempt_sha256=1868e0a5904d9916b19845475a203c4eaf7aa84d37ecb8e51563bf993bbb0664, attempt_status=failed, automatic_retry=false, binary64_max_abs_difference_by_seed={"41": 2.9802322387695312e-08, "42": 1.4901161193847656e-08}, binary64_rejections_by_seed={"41": 99, "42": 88}, budget_limited=false, campaign_accounting_remaining_gpu_hours_display=18.2195, campaign_used_gpu_hours_display=5.7805, campaign_used_gpu_hours_exact=5.780427976370344, case_counts_by_lead=[22, 21, 19, 15, 11], case_evaluations=528, cli_exit_code=1, code_zip_sha256=d3d10a74c77303849d38e06a44fa2fcb06d2ac8da0c7ddc4646b4218e0cb8bd8, device_policy=shared-headroom, device_uuid=GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b, distinct_worker_pids=36, engineering_ci_receipt_path=outputs/r7_v2_remaining_acceptance_20261003/engineering_ci_acceptance.json, evaluations_completed=30, evaluations_missing=0, evaluations_planned=30, execution_attempt_sha256=da98745af2d0453b87e5d6aa071c5cdd64efdce062b052266805e49dde6501c3, failure_reason=Binary64 aggregate objective auditing rejected FP32 operation-by-operation loss records; original attempt is failed, not finalized, finalized=false, first_binary64_combination=0.2679187208414078, first_rejected_l12=0.22569331526756287, first_rejected_l6=0.15507206320762634, first_rejected_loss=0.2679187059402466, fp32_operationwise_exact_matches=400, fp32_operationwise_max_abs_difference=0.0, full_experiment_accepted=false, gpu_hours=0.8526056814201487, gpu_phase_seconds=3069.380453112535, gpu_rerun_performed_for_registration=false, hard_cap_seconds=10800, historical_campaign_baseline_gpu_hours=24.0, independent_acceptance_scope=original continuous wall/GPU costs only; aggregate and full experiment not accepted, independent_cost_accepted=true, independent_cost_audit_gpu_hours=0.0, independent_cost_audit_seconds_rounded=5.3582, independent_cost_review_path=outputs/r7_v2_remaining_acceptance_20261003/b_independent_cost_review/revision02/audit_result.json, independent_cost_review_sha256=c00dd04ca68152a9be57f41a433e1234657abd6db6139a132ab9e131d3f900db, independent_file_hash_seconds_rounded=8.1985, independent_statistics_acceptance=pending-separately-frozen-zero-gpu-complement, initial_failed_cost_audits_preserved=true, jobs_completed=36, l6_forward_backward_flops=23265272448, l6_forward_flops=7774613952, lead_hours=[6, 12, 24, 48, 72], loss_cause_path=outputs/r7_v2_remaining_acceptance_20261003/b_fp32_loss_cause.json, loss_cause_sha256=f07a6712fbdca5906c0dd08fa15d4f7d3de6bcb6fec923eac40d07c48bb2f631, m3_original_pause_preserved=true, merged_result_produced=false, model_code_sha256=0cc9c16e7a12bf23150fb04e13acb72f548f8cb461ad64df1d45123b387a223e, original_bytes_sealed=3089276264, original_failed_preserved=true, original_files_sealed=361, original_files_unchanged=361, outcome=null, output_dir=outputs/r7_74_autoregressive_20261003_attempt01, owned_unreaped=false, paired_comparison_complete=false, parent_arm=M3 aux_off/RW-A/K4, partial=false, planned_seconds=5400, precision_probe_gpu_hours=0.04711594580465721, prior_campaign_gpu_hours_exact=4.880706349145538, protocol_file_sha256=2d937850fb817d715219050c9418811f8d3dbf56a4d48b3e13fe253afd9557dc, reasoning_steps=4, regions=["full", "interior", "edge_2"], registration_ci_available_at_index_write=false, registration_status_at_index_write=evidence-frozen-index-recorded-awaiting-separate-registration-ci, reproducibility_level=code/data/protocol-bound numerical reproducibility; no bitwise ordinary GPU training, GPU replay or accepted aggregate claim, rollout_loss_records=400, scale_sidecar_identity=4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d, seeds=[41, 42], soft_overrun_seconds=0.0, source_pins_path=outputs/r7_v2_remaining_acceptance_20261003/b_failed_source_pins.json, source_pins_sha256=e0151b6345453b581e44088e7fdb7eaa29af24a56a0701b033782b2b0317dc84, source_tree_sha256=5de6b16dccb9e59116116017a9f5286ac0576c8d7db270d0f6f8fe7b9c1cf2c9, stage_result_produced=false, successful_workers_elapsed_seconds=3011.8308975147083, terminal_four_cost_views_complete=false, test_read=false, thresholds_added=0, train_windows=185, training_runs_completed=6, training_runs_planned=6, training_updates=1600, two_step_forward_backward_flops=46587416832, two_step_forward_flops=15549227904, updates_per_arm_per_seed={"continue_l6": 200, "equal_compute_l6": 400, "rollout_l6_l12": 200}, variables=17, whole_wall_seconds=3200.912431293167, workers_successful=36

## n3-b-statistics-complement-negative

- Outcome class: `mixed`; candidate state: `not-candidate`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_74_STATISTICS_COMPLEMENT.md` (SHA256 `7ea1b5b7f6f9f2f22f1a4b39b59750112486f35bbee745d835b90db78ac18806`)
- Evidence commit: `1615795d70117af608787ad6de15d2e5640b901c`; experiment commit: `adb28667759e8d1aabd3bc0bf7bc07f7b2bbd591`
- Protocol SHA256: `076f28dfe88cd6e600d0ed37089f40b8e0562008e5d254facb99f93bb9a449e9`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: A separately frozen, zero-GPU complement completed and independently verified all original failed B metadata. Rollout beats the 200-update L6 comparator on all four primary cells but loses to the 400-update L6 control on all four. The unchanged frozen rule selects L6 for independent C and terminates only the rollout hypothesis. Original B remains failed/finalized false and complement01 remains inventory-rejected.
- Limitations:
  - Metadata reaggregation only, with no new optimizer updates, forward passes, checkpoint tensor loads, weather/test arrays or GPU use.
  - Original failed B and the first complement inventory FAIL remain immutable; matching numerics did not rehabilitate complement01.
  - Two seeds on one winter-region developed validation set provide descriptive signs, not significance or generalization.
  - Exact physical sufficient statistics, units and row hashes verified; original ordinary GPU training remains identity-bound numerical, not bitwise.
  - Three overlapping regions and repeated baseline references are metric-row exposures, not independent weather events.
  - Climatology undefined ACC and negative model/reference metrics are retained; serialized near-zero signs are distinct from producer diagnostic labels.
  - 400-update L6 is an update-count control, not strict equal FLOPs or wall time; all measured costs are retained.
  - The full original 0.8526056814201487 GPU-h is already charged in its own ledger row; this record adds zero.
  - Independent audit is persisted-artifact verification, not a second GPU run or operating-system attestation.
  - Original #72 K3 calendar differs from B Gregorian; M3 K4 parents cannot substitute for the genuine reference.
  - New engineering full suite originally failed five stale input fixtures; its later repair and exact CI remain separately recorded, never inferred from this result.
  - No new data publication/download, dependencies, credentials, security configuration, paid resources or neighbor signals.
- Excluded from runnable candidates: Complete metadata/statistical delivery, not successful original B, rollout support, scientific improvement, runnable candidate or final goal completion. Genuine K3, new-package precision and three-arm C remain separate required work.
- Recorded metrics (not recomputed): acceptance_scope=metadata-statistical-complement-only, aggregate_rows=765, automatic_retry=false, case_evaluations_original=528, complement01_accepted=false, complement02_accepted=true, csv_tables_verified=11, derived_files=113, evaluations_new=0, evaluations_reused=30, explicit_exclusions=2, gpu_hours=0.0, hard_cap_seconds=1200, independent_acceptance_sha256=0f965f2e7fd8dff509637835866737f3961003eb56b434704abf515676bf4e8b, independent_audit_soft_overrun_seconds=0.0, independent_audit_whole_seconds=442.5259756054729, independent_closure_sha256=d3e493da84432fef9316a2475baa06c569f61511e1c4f587755fd84ce5dea11f, manifest_pins=111, metric_rows=1530, negative_acc_model_case_rows=5656, negative_skill_model_case_rows=11255, original_failed_preserved=true, pair_cells_each=255, pair_totals={"equal_minus_continue": {"improved": 77, "unresolved": 141, "worsened": 37}, "rollout_minus_continue": {"improved": 113, "unresolved": 109, "worsened": 33}, "rollout_minus_equal": {"improved": 83, "unresolved": 102, "worsened": 70}}, planned_seconds=600, primary_rollout_minus_continue={"12h_seed41": -0.2551141788589866, "12h_seed42": -0.04350118569222117, "6h_seed41": -0.08880950883362715, "6h_seed42": -0.0011212856314508635}, primary_rollout_minus_equal={"12h_seed41": 0.040405678414765056, "12h_seed42": 0.1309677380286569, "6h_seed41": 0.15415675114328176, "6h_seed42": 0.12114250057941867}, reproducibility_level=identity-bound numerical(metadata), not GPU bitwise, scientific_state=negative_or_mixed, selected_mode=l6, soft_overrun_seconds=0.0, source_attempt_finalized=false, source_attempt_status=failed, source_files=361, source_gpu_hours_already_charged=0.8526056814201487, test_read=false, training_runs_reused=6, training_updates_new=0, undefined_climatology_case_reference_rows=26928, whole_wall_seconds=52.027686852030456

## n3-b-utc-statistics-metadata-complement

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_74_UTC_STATISTICS.md` (SHA256 `b4fd436993648d34544d260c6140f2043abd061d00c079dded9bcea4b4e52d4e`)
- Evidence commit: `61db2725f6fda6abf986d94b7dbd0c6dd0bf12a4`; experiment commit: `1615795d70117af608787ad6de15d2e5640b901c`
- Protocol SHA256: `9a82c1357750b41b9a930879543757dfe6ca537228ea862b2f5ba5d6bc57c5e8`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: Official corrected B-only UTC CLI actually produced20400 rows/400 groups with no empty group or cell. New qualified independent review bound248 files, checked all80784 source rows and independently recomputed all20400 formula rows/400 exact group identities. Acceptance is accepted-official-utc-metadata-evidence-only; owner old whole-package NOT ACCEPTED/full-log-prefix mismatch and all original B/B01/UTC failures, negative_or_mixed/l6 remain unchanged. Zero new GPU, no C/M1 coverage.
- Limitations:
  - experiment_commit1615795 is held_HEAD provenance only, not a clean source commit: exact four-file corrected runtime map aa212882 and helper archive govern; original B archived source base616b/codezipd3d10a74 is distinct and unchanged.
  - Owner whole-package remains NOT ACCEPTED for OWNER_REQUIRED_ARTIFACT_FULL_LOG_PIN_MISMATCH: required475-byte prefix1408f889 differs from final878-byte full log49afc3bd; explanation and new closure do not satisfy the old required full-file pin.
  - Original source B remains failed/finalized false/negative_or_mixed/selected_mode l6, complement01 inventoryFAIL and first actual UTC failed0rows/0groups retained; B02 accepted metadata complement remains separately indexed at1615795/7ea1.
  - Fresh independent stdlib/math.fsum physical sufficient-statistic formulas, not producer helper or owner self-review oracle; exact null/blank/status semantics, negatives and tiny binary64 climatology signs preserved.
  - 20400 cells overlap by variable/region and duplicate init/valid UTC axes; not independent weather events, significance, causality or generalization. Two developed winter-region validation seeds only.
  - Accepted248-file/80784-row audit does not claim repeat all361 opaque original B file hashes, tensor/weather oracle, GPU replay or cross-platform bitwise; registration reads page/receipt identities only.
  - CLI147.193569 seconds, owner final949.854352/over49.854352 and independent final1340.701317/over440.701317 each use own900soft1800hard budget; pre-serialization snapshots and earlier owner712.884945 are not final whole cost.
  - Original B0.8526056814201487 GPU-h and old precision0.04711594580465721 are already charged once; UTC registration adds zero and never repeats source training or evaluation.
  - Exact parent523c819 CI FAILED step9, all12 necessary steps not passed. No new exact CI, current-package precision, C, M1, adaptive evaluation, issue closure or final goal acceptance is implied.
  - No new checkpoint/weather/test payloads, GPU/network, data writes/downloads, dependencies or credentials/security changes in this registration.
- Excluded from runnable candidates: Official UTC metadata evidence only, not an accepted old owner whole-package, successful original B, model candidate, science PASS, C/M1 execution, adaptive verdict, final goal or closure. Independent new full-C and current-package precision remain pending.
- Recorded metrics (not recomputed): acceptance_scope=accepted-official-utc-metadata-evidence-only, actual_pass=false, all_12_required_ci_steps_passed=false, attempt_status=actual-statistics-complete, bound_files=248, c_actual_jobs_completed=0, c_prepared=false, case_counts_by_lead=[22, 21, 19, 15, 11], cli_inputs=133, cli_publication_seconds=147.193569, complement01_accepted=false, corrected_runtime_four_file_sha256=aa212882688bf75eca273f60de9178f2fdceeaf40562afa996690ff87ff7be59, empty_cells=0, empty_groups=0, evaluations_new=0, gpu_hours=0.0, group_identity_canonical_sha256=f67ae82c5f4cb1d0cd81d0460a273ae4cb83542d0a1a076af73e3308b5a0915c, groups=400, hard_cap_seconds=1800, held_head_provenance=1615795d70117af608787ad6de15d2e5640b901c, independent_audit_hard_cap_seconds=1800, independent_audit_planned_seconds=900, independent_audit_soft_overrun_seconds=440.701317, independent_audit_whole_seconds=1340.701317, independent_final_seal_sha256=cecf4b1e999c3744a609630c4308b3fe3ed53a8ad18edb2501c5b2417d3e7aa3, independent_metadata_accepted=true, independent_receipt_path=/tmp/r7_utc_b_corrected_independent_20261003_4Qji7Sw2/qualified/verification_receipt.json, independent_receipt_sha256=594a11816066c28998586343f18ba29d91ac30b3aa512b030d7dea3d1ef3f611, lead_hours=[6, 12, 24, 48, 72], m1_actual_jobs_completed=0, metric_rows=20400, model_k=4, negative_acc_rows={"climatology": 0, "model": 2552, "persistence": 620}, negative_skill_rows={"climatology": 808, "model": 6166, "persistence": 2352}, new_package_precision_run=false, original_b_code_zip_sha256=d3d10a74c77303849d38e06a44fa2fcb06d2ac8da0c7ddc4646b4218e0cb8bd8, original_b_source_base_commit=616b029dce569b92bd08295737981512180a1ad3, owner_failure_reason=OWNER_REQUIRED_ARTIFACT_FULL_LOG_PIN_MISMATCH, owner_final_full_log_bytes=878, owner_final_full_log_sha256=49afc3bdc0edbb222ccc505b4ace901b9e432294035b7de326f791b1d464def7, owner_old_whole_package_accepted=false, owner_required_full_log_sha256=1408f8899cb7a98c04b3065753bf2b488b7b10e275f251655c13b696b470c861, owner_required_prefix_bytes=475, parent_engineering_ci_commit=523c819b8d42a3f43163eb53509f76136d8038f4, parent_engineering_ci_run_id=37155293949, parent_engineering_ci_status=completed/failure, persistent_qualified_directory=outputs/r7_v2_remaining_acceptance_20261003/utc_corrected_independent_final_evidence_supplemental04_addendum/qualified, planned_seconds=900, preparation_inputs=210, protocol_file_sha256=b1ef1991718c92beb94785d8c8e87eadfd973be0012282f7a20826c654a67ab3, qualified_copy_map_path=outputs/r7_v2_remaining_acceptance_20261003/repair_evidence_utc_independent_addendum_copy_mapping.json, qualified_copy_map_sha256=3047f6c054112211ad166a30f0c0d97e0b54405d78d399c527e8669f911e5dab, raw_copy_map_sha256=502bdfe1ef8012d93918093e67dd041433c7452ced0a46e78b430951c01b9c00, regions=["full", "interior", "edge_2"], registration_ci_available_at_index_write=false, reproducibility_level=identity-bound independent scalar numerical metadata replay; not GPU bitwise, scientific_gate_evaluated=false, scientific_state=negative_or_mixed, seeds=[41, 42], selected_mode=l6, soft_overrun_seconds=49.854352, source_b_finalized=false, source_b_gpu_hours_already_charged=0.8526056814201487, source_b_status=failed, source_identity_sha256=2988eb85ece3a9db88deb4989a45320fc637faa1a27058bd7401c95e6bd0db32, source_rows=80784, test_read=false, training_updates_new=0, typed_row_canonical_sha256=401327f121b4e50cf47b09ac0380598da43c2d371c98b16d85a793d016f24205, undefined_climatology_acc_rows=4080, utc_metrics_csv_sha256=c51fbd1d7befff2e4042fe4b412b476f68c586da02df5b06d42d1c3cfe16c443, variables=17, whole_wall_seconds=949.854352, workers_receipts_verified=36, workers_timing_sidecars_verified=36

## n3-original-k3-reference-metadata-complement

- Outcome class: `audit`; candidate state: `not-candidate`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_72_ORIGINAL_K3_REFERENCE.md` (SHA256 `c77f510fff619cee4e105dd14e98a94ec14e91347587d07bb284fa92cf3c614b`)
- Evidence commit: `61db2725f6fda6abf986d94b7dbd0c6dd0bf12a4`; experiment commit: `616b029dce569b92bd08295737981512180a1ad3`
- Protocol SHA256: `8c42e5031e88c04f4c9514236d31bb4bc89da5b1fb96aa86bb9700ab63d8bdbc`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: Actual genuine #72 RW-A/K3 selected400 reference completed zero trainings and ten evaluations. A new independent final closure accepted all136 artifact inventory,1530 metric rows,170 full-model cells and24 opaque inputs only as accepted-reference-evidence-metadata-complement-only. Original success/finalized true and reference complete-not-accepted remain unchanged. Recorded continuous0.04818810004533993 GPU-h is charged once; this registration adds no GPU run.
- Limitations:
  - experiment_commit616b is the declared source_base_commit, not a complete historical or clean execution commit: actual source_commit is null, held before/after HEAD1615795, runtime identity is exact seven companion files plus the compatible archived sourcebridge; no fake full #72 historical commit.
  - Old365.25 feature denominator, accumulated lead, fixed init day/hour and full/interior_2/edge_2 semantics are retained; not Gregorian or architecture-matched paired C and no cross-calendar gain is inferred.
  - Original reference_result remains complete-not-accepted and attempt success/finalized true; no original whole-owned frozen manifest or acceptance marker was manufactured.
  - Independent metadata/scalar reconstruction only; checkpoints are opaque hashes,131 state keys/2968259 entries are evaluator-reported, not independent tensor recount or GPU/weather replay.
  - All negative and undefined values retained, including262 model negative skill/100 negative ACC,280 persistence negative skill/68 negative ACC,54 near-zero climatology negative skill and510 undefined climatology ACC.
  - Two seeds, one developed winter-region validation cohort and overlapping regions are descriptive, not significance/generalization/SOTA; config/source/data/protocol-pinned numerical reference, not GPU bitwise.
  - 136-file/24-input acceptance and numerical formula claims belong to the cited independent closure; registration verifies frozen page Git bytes and cited receipts, not a repeat complete artifact audit.
  - Exact parent523c819 CI actually FAILED step9; local historical CPU3467passed/9skipped/3warnings and older exact engineering CI never turn it green or accept current-package precision/C.
  - No new data/checkpoint/weather/test reads, training/evaluation, GPU/network, dependencies, security/credentials, main changes or closure in this governance registration.
- Excluded from runnable candidates: Reference metadata evidence, not a runnable candidate, scientific gain, Gregorian/calendar/architecture-matched paired C, successful original acceptance, final goal completion or issue closure. New exact engineering CI and current-package precision are still required before C.
- Recorded metrics (not recomputed): acceptance_scope=accepted-reference-evidence-metadata-complement-only, actual_files_independently_verified=136, all_12_required_ci_steps_passed=false, archived_model_code_sha256=11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476, attempt_status=success, c_actual_jobs_completed=0, c_prepared=false, case_counts_by_lead_per_seed=[22, 21, 19, 15, 11], case_evaluations=176, companion_code_zip_sha256=f7861557055e2047cf101fd00f477080e1d40f0ac13eb904a37c80e373211a16, companion_runtime_sha256=9596dde84055329e1a815ef0244f4678d1329268bdc92822800f3486df1ae76f, evaluations=10, finalized=true, full_model_cells=170, gpu_hours=0.04818810004533993, gpu_phase_seconds=173.47716016322374, hard_cap_seconds=3600, held_head_after=1615795d70117af608787ad6de15d2e5640b901c, held_head_before=1615795d70117af608787ad6de15d2e5640b901c, independent_audit_gpu_hours=0.0, independent_audit_hard_cap_seconds=1800, independent_audit_planned_seconds=900, independent_audit_soft_overrun_seconds=0.0, independent_audit_whole_seconds=719.8785105217248, independent_closure_receipt_path=/tmp/r7_original_k3_actual_audit_wjar1kjk/closure_receipt.json, independent_closure_receipt_sha256=226121c83175db96d7598f67e81ce0b6a9d1127384fb1e9d9f4643e5f0b85cd7, independent_metadata_accepted=true, lead_hours=[6, 12, 24, 48, 72], metric_rows=1530, metrics_csv_sha256=965ce69e30728be58d51bdbb11743acf3fad8b0e5e72987c484f1727a14f4591, negative_acc_rows={"climatology": 0, "original_k3": 100, "persistence": 68}, negative_skill_rows={"climatology": 54, "original_k3": 262, "persistence": 280}, new_package_precision_run=false, original_reference_status=complete-not-accepted, parent_engineering_ci_commit=523c819b8d42a3f43163eb53509f76136d8038f4, parent_engineering_ci_run_id=37155293949, parent_engineering_ci_status=completed/failure, persistent_independent_audit_dir=outputs/r7_v2_remaining_acceptance_20261003/original_k3_actual_audit_evidence_supplemental04, planned_seconds=1800, protocol_file_sha256=c0e57aa1842a4c9459c00a7c606797d9caccc0e4223546ad020c29160e4b6fdf, raw_copy_map_path=outputs/r7_v2_remaining_acceptance_20261003/repair_evidence_supplemental04_copy_mapping.json, raw_copy_map_sha256=502bdfe1ef8012d93918093e67dd041433c7452ced0a46e78b430951c01b9c00, reasoning_steps=3, regions=["full", "interior_2", "edge_2"], registration_ci_available_at_index_write=false, registration_gpu_hours=0.0, reproducibility_level=identity-bound numerical(metadata); not GPU bitwise, scientific_gate_evaluated=false, seeds=[41, 42], selected_update=400, soft_overrun_seconds=0.0, source_base_commit=616b029dce569b92bd08295737981512180a1ad3, source_commit=null, source_identity_sha256=9874210ef6259de7196dc605219344ecc6f2c12126633670e535702b690bda36, sourcebridge_code_zip_sha256=18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95, sourcebridge_source_tree_sha256=e2a4e562a592e6b6f92c7d18a4d0fb236e95ebaf676464fccebe9acbe707f29b, test_read=false, training_runs=0, training_updates=0, undefined_acc_rows={"climatology": 510, "original_k3": 0, "persistence": 0}, unique_opaque_inputs=24, variables=17, whole_wall_seconds=176.6783818155527

## n3-v2-precision-gpu-acceptance

- Outcome class: `engineering-positive`; candidate state: `needs-review`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_V2_PRECISION_ACCEPTANCE.md` (SHA256 `982bd632ccd9e4d74b3ef04216484ec2ff94f511b2f36f8755140003852a1f30`)
- Evidence commit: `42ecb890ce8f8f1d2925f8841a74859b2decd8f7`; experiment commit: `616b029dce569b92bd08295737981512180a1ad3`
- Protocol SHA256: `0829da0251449a840d256878ff5d2687bfcb7681db39c585df6a2f202255f4ba`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: The frozen real-train FP32/BF16 engineering probe completed eight optimizer updates in two fresh CUDA workers, with all four acceptance flags true and same-precision checkpoint-body weights/optimizer/RNG/loss resume equality. Continuous GPU cost is fully charged. Original M3 failed/paused records, failed CPU logs and the first independent failed audit are preserved. The newly authorized independent N3 route under decisions 0030/0032 is not blocked by the historical M3 pause and does not change its exit rule. B remains an original failed aggregate; separately frozen zero-GPU statistics completion is pending, without repeating GPU training or evaluation.
- Limitations:
  - Two real train windows, 17 channels, 65x65, batch two and K4; no validation skill comparison, convergence, generalization or SOTA claim, and the sealed test was not read.
  - Eight updates count both precision-specific uninterrupted-two and intentional-resume-two branches. Bitwise acceptance is same-seed/same-endpoint within each precision, never FP32 versus BF16 or an ordinary B/C training guarantee.
  - Checkpoint-body save-at-one/resume-to-two is accepted; it is not a claim that this GPU probe executed fine_tune interruption orchestration. The latter has separate CPU subprocess counterproofs.
  - Deterministic algorithms, TF32 false and CUBLAS workspace :4096:8 apply to this small same-device/software probe only; ordinary B/C GPU training remains code/data/config-bound numerical reproducibility, not bitwise training.
  - L12-only gradients and target-poison checks were executed and recorded by the workers. Poison forecast tensors and gradient arrays were not separately saved; independent review checks archived code/receipts, not a new GPU forward/backward.
  - Loss/L6/L12/grad_norm equality comes from branch receipts; checkpoints do not contain loss, so the CPU checkpoint comparison does not recompute those losses.
  - Independent CPU review compared 131 model tensors, 123 optimizer states and RNG, and preserved 29 artifact hashes/bytes. This registration only reads frozen documents and git blobs, not those actual artifacts or tensors.
  - The first independent checker added a current-status-equals-frozen-status constraint and rejected new documentation. Its failed audit is retained; the later identity-only review verifies frozen runtime/source/archive identity without rewriting the failure or relaxing the experiment contract.
  - Timing records contain exited/cleaned strings but not raw exit codes or the last reap timestamp; the archived driver accepts exit zero and takes its final snapshot after owned reap. Closeout serialization itself follows its cost snapshot.
  - Table worker seconds and MiB peaks are rounded reported measurements, not isolated model-forward latency/memory or a cross-device performance guarantee. Shared-headroom observations are not reservations or neighbor-load benchmarking.
  - The frozen engineering history retains the initial 10 failed/2614 passed/9 skipped CPU run and concurrent-source identity rejection; stable 3107 passed/9 skipped/2 warnings and final 160 targeted passes are historical, not tests rerun by this registration. Skips and two Lightning warnings remain.
  - CI identity is from anonymous Jobs API observations on 2026-10-03, not downloaded pytest logs. Seventeen unrequested tag-gated experiment workflows being skipped is not experimental acceptance or registration CI.
  - The original M3 failed attempt, completed supplementation pause and any-unresolved exit remain unchanged. The new N3 independent-route authorization does not turn those old records into scientific success.
  - No accepted B aggregate/statistics, independent C, adaptive verdict, issue closure, main advancement or final goal completion is claimed. The four synthetic bridge cases do not extend this real probe to calendar-enabled old experiments.
  - No new data download/publication, dependency installation, credential or security-configuration change is part of this record.
- CI run: `37133487340`
- Excluded from runnable candidates: Engineering prerequisite only, not a scientific or runnable model candidate, weather comparison, full B experiment acceptance or goal completion. Historical failed records and M3 pause remain unchanged; independent-route authorization is new, not a rewrite of those records. Independent B statistics acceptance is pending in separate outputs, with no GPU rerun. This index entry grants no execution permission and does not invent registration CI.
- Recorded metrics (not recomputed): acceptance_flags_true=4, all_allocator_baselines_zero=true, all_same_precision_resume_checks_equal=true, archived_source_members=85, attempt_path=outputs/r7_v2_precision_probe_20261003_attempt01/attempt.json, attempt_sha256=8ecd15ea505f43235544d576b214ddc648e2a76c143ed6928334bd9fad094ada, attempt_status=success, b_independent_statistics_acceptance=pending, b_original_aggregate_status=failed, batch_size=2, bf16_peak_allocated_mib_rounded=631.8799, bf16_peak_reserved_mib=674, bf16_worker_seconds_rounded=79.1489, budget_limited=false, calendar_phase_atol=2e-06, closeout_sha256=4db58b6a6b8070f9468154e7432e2b3cea67277872dacd8fc3fa346895a4c29e, closeout_wall_seconds=242.43781951908022, device_policy=shared-headroom, device_uuid=GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b, engineering_ci_receipt_path=outputs/r7_v2_remaining_acceptance_20261003/engineering_ci_acceptance.json, finalized=true, fp32_peak_allocated_mib_rounded=987.6782, fp32_peak_reserved_mib=1134, fp32_worker_seconds_rounded=84.0124, full_internal_and_physical_bptt_recorded=true, future_target_poison_forecast_unchanged_recorded=true, gpu_hours=0.04711594580465721, gpu_phase_seconds=169.61740489676595, gpu_rerun_performed_for_registration=false, grid_shape=[65, 65], hard_cap_seconds=1800, independent_acceptance_sha256=bd167cb754eb179cc457f08f60841b336f46edbaca684fe8bda9319c8099aa86, independent_artifact_pins_sha256=062d27579b98cf50231a429c21ec66ab7f68fa7c8cdb567f639b11eb9c9f09e8, independent_artifacts_unchanged=29, independent_review_dir=outputs/r7_v2_remaining_acceptance_20261003/precision_independent_review, intentional_resume_updates_per_precision=2, m3_original_failed_preserved=true, m3_original_pause_preserved=true, model_code_sha256=0cc9c16e7a12bf23150fb04e13acb72f548f8cb461ad64df1d45123b387a223e, original_failed_audit_preserved=true, output_dir=outputs/r7_v2_precision_probe_20261003_attempt01, parent_arm=M3 aux_off seed41 update400, planned_seconds=900, precisions=["FP32", "BF16"], reasoning_steps=4, registration_ci_available_at_index_write=false, registration_status_at_index_write=evidence-frozen-index-recorded-awaiting-separate-registration-ci, reproducibility_level=same-device/software same-precision checkpoint-body bitwise resume in this small probe; ordinary GPU training code/data/config-bound numerical reproducibility only, resume_model_tensors=131, resume_optimizer_states=123, soft_overrun_seconds=0.0, test_read=false, thresholds_added=0, train_windows=2, training_updates=8, uninterrupted_updates_per_precision=2, updates_per_precision=4, variables=17, whole_wall_seconds=242.43521373253316, workers_completed=2, workers_failed=0

## v2-actual-c-confirmation

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_C_ACTUAL_CONFIRMATION.md` (SHA256 `bc9cdfe3d1582cfa91335dc49a7e2a0becd23446641fb3cc2283b0c683497a51`)
- Evidence commit: `858eddbf7b917ac158689c5f7de150fca1152e47`; experiment commit: `562e526afc5fdbdc99053a25ab20036b66a2a8bc`
- Protocol SHA256: `ea0efb80e1b11eb7813fae74b4ee04ca67d145d16519eb8aa442af8d8851ad80`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: 独立冻结协议的 actual C 三臂确认（old_ours/process/matched_generic × seed 41/42/43）144/144 job 完成、1333 文件全 pin、独立审阅接受（aa44b8cd…）；package 相对 old_ours 在冻结 primary 6/6 cell 严格改善，但 process vs matched_generic 为 unresolved（1e−5 K 量级差正负不稳）→ 不能归因过程语义独立贡献；adaptive 四门 evaluated:true、gate_met:false、控制器不训练。
- Limitations:
  - 三个 seed、一个冬季区域、单次运行；描述性一致性，不是显著性，也不构成泛化证据。
  - package 改善不能拆归因到某一 reader/query/过程语义；matched_generic 同结构同输入对照仍解析为不能区分。
  - accuracy_cost_tradeoff 门未过；adaptive 保持 not-started，K 数是推理轮次，不代表延迟优势。
  - 指标含负值与 undefined（AC 对 climatology 的 normalize 语义），全部保留未过滤；可复现等级为 identity-bound numerical replay，非 GPU 逐位一致。
  - UTC 标量补链（67,320 rows/1,320 groups）为 metadata 层，历史 old whole-owner 日志失败保留不回修。
- Excluded from runnable candidates: 不是 SOTA、显著性、过程语义归因成功或目标完成；单区域单次运行且 adaptive 未启动；不授权控制器训练、新数据或最终泛化测试。
- Recorded metrics (not recomputed): adaptive_gate_met=false, arms=3, gpu_hours=3.155003245259221, hard_limit_seconds=21600, jobs_completed=144, jobs_planned=144, package_improved_cells=6, primary_cells=6, process_minus_generic_resolved=unresolved, seeds=3, soft_overrun_seconds=884.2877719895914, whole_seconds=11684.287771989591

## v2-actual-m1-completion

- Outcome class: `mixed`; candidate state: `needs-review`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_M1_ACTUAL_AND_UTC.md` (SHA256 `8816bb4ccbd4d2bee34af45924bc2dcc6ee0cef88a9bae11a57639dc633b3efe`)
- Evidence commit: `858eddbf7b917ac158689c5f7de150fca1152e47`; experiment commit: `562e526afc5fdbdc99053a25ab20036b66a2a8bc`
- Protocol SHA256: `c5df8cc63c8860e21f8feb6d2ec658d6218e82e2bb87923b8cc5b93974ca2d58`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: actual M1 单输入因素补全 24/24 job（4 fresh scratch 训练 + 20 K4 评估）实跑成功，accept_own_output verify 通过（c5df8cc6…）；primary 4/4 cell 相对 old_ours 严格改善，但 process−generic 在 4e−6 K 量级 → 单因素 unresolved，#71 无实际改善不得 DONE-positive；M1-only UTC 标量链 400 groups/20,400 cells 经独立 oracle（635cd90c…）通过。
- Limitations:
  - 两个种子、单冬季段；合法复用 C 的 old_ours 41/42 十个 K4 评分（不导入权重），未新增第三 seed。
  - M1 相对旧 ours 的改善主要来自 known-context 输入容量；过程语义独立贡献 unresolved。
  - UTC 统计为标量 metadata 重算（独立公式），不是天气场/tensor oracle，也不改任何 GPU 数值身份。
  - round-4 运算符先放收据冲突 sealed as failed 如实保留；可复现等级为 config/numerical identity bound，非 GPU 逐位。
  - scientific_claim: false；test_read: false；未扩臂、未读封存 test、未改判据。
- Excluded from runnable candidates: 不是科学增益、SOTA 或 #71 DONE-positive 依据；单因素读数为 unresolved；不授权过程语义追认实验、最终泛化测试或任何新数据。
- Recorded metrics (not recomputed): gpu_hours=0.8757688270136714, jobs_completed=24, jobs_planned=24, m1_utc_cells=20400, m1_utc_groups=400, process_minus_generic_resolved=unresolved, reused_c_evaluation_jobs=10, soft_overrun_seconds=0, whole_seconds=3543.913658151403

## v2-new-package-precision-acceptance

- Outcome class: `engineering-positive`; candidate state: `needs-review`
- Human triage priority: `80` (not a scientific score)
- Evidence: `docs/R7_V2_NEW_PACKAGE_PRECISION_ACCEPTANCE.md` (SHA256 `8fec7a972a77605b51fbf979faa034fefcc98652eac6e2531f2e4d6378adf66e`)
- Evidence commit: `858eddbf7b917ac158689c5f7de150fca1152e47`; experiment commit: `562e526afc5fdbdc99053a25ab20036b66a2a8bc`
- Protocol SHA256: `edc83fb64df15ef9b1925758a029251c25dbc213b3ebca3e1b9b87b0a9c46f68`; data identity: `ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07`
- Reason: 新工程包 FP32/BF16 精度探针实跑 8 实际优化更新（两精度各 4）与 4 checkpoint decode，获独立 metadata-only 接受（accepted-actual-new-package-precision-metadata-only，接受文件 8503adfe…）；resume 逐位相等、L12-only 第一步与 process_reader 梯度 finite>0、poison/calendar 断言通过。
- Limitations:
  - 独立接受是 metadata/source/inventory 校验与 CPU 端点对比；GPU forward 未被独立重算，不构成 forecast-skill、收敛或任何科学声明。
  - 0.0544421515867321 GPU-h 与旧 precision 0.04711594580465721 是不同对象，各记一次；本探针是 actual C 的工程前置之一而非其必要条件。
  - CI green 绑定尚未对本次登记取得（本记录 ci_run_id/ci_commit 为 null）；精确 CI 以收尾提交的工作分支运行单独绑定。
  - scientific_claim: false；test_read: false；未新增训练臂、未读封存 test、未改判据。
- Excluded from runnable candidates: 工程精度前置，不是天气收益、可运行候选、科学成功或目标完成；GPU forward 未独立重算；不构成任何新运行授权。
- Recorded metrics (not recomputed): actual_updates=8, checkpoint_decodes=4, gpu_hours=0.0544421515867321, gpu_phase_seconds=195.99174571223557, soft_overrun_seconds=0, whole_seconds=273.1497834455222

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
