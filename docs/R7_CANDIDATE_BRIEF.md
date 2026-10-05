# R7 Evidence Candidate Brief

> This is an evidence index, not a scientific verdict. Priority is a human triage field.
> No entry authorizes training, data access, GPU use, or a change to a frozen criterion.

Records: 36; human-review candidates: 1

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
