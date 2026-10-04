# v2-l6-confirmation-current：独立 L6 三臂确认与必要 M1 单输入因素闭合

<!-- round-node: N4 -->

**状态：active → 实证完成（2026-10-04）。C 已实跑并独立接受（144/144 job，protocol `ea0efb80…`）；M1-only 补全已实跑并 verify（24/24 job）；两侧 UTC 标量统计链完成（C 67,320 rows/1,320 groups；M1 20,400 cells/400 groups）；adaptive 四门已评价、`gate_met:false`，控制器不训练。** 本轮由实际 N3/B02 的 `negative_or_mixed / selected_mode:l6` 出口派生，适用0030/0032，不复用旧 prepared N4 长文为实际执行。`scientific_claim:false`；不自宣最终 goal 完成。剩余工作：N5 登记与六 issue 决定（见 §8）。

## §0 Objective（可直接粘贴，单段；实测 1179 字符，≤4000）

> 在 docs/goals/v2-l6-confirmation-current.md 按0030/0032连续推进实际N4：D1先核N3/B02已登记出口、真正原K3与官方UTC受限独立接受，修复精确523c819/CI37155293949的真实失败并取得新工程包精确CI全部12必要步骤success、独立新包FP32/BF16/resume/poison精度接受，未齐不启动C；D2冻结独立C协议/代码/数据/初始化/window/产物身份，引用原R7_V2_CONFIRMATION_PREREGISTRATION不改primary或容忍，三臂old_ours/process/matched_generic×seed41/42/43，同seed同scratch anchor、新优化器、各400 L6 updates/Ktrain4/fullBPTT/selected400，共9训练与135同checkpoint K1/2/4评估，完整17变量×5lead×3region、185训练窗口与22/21/19/15/11完整val，负值undefined、四成本与实际时钟全部保留；D3独立核实际144job整包后才评价固定process的四项adaptive冻结门，当前unevaluated不是failed，不以K数代latency或加复杂度练到赢；D4仅实际C独立接受后复用old_ours seed41/42的10个K4评分和相同初始化/配方身份，4项fresh scratch M1-only Process/Generic各400 L6训练与20项K4评估补#71单输入因素缺口，禁RW-A/direct-query/source-position/roles/local-Z混入，不重训旧baseline或续训其权重；D5登记完整正混负/身份/成本/未做与逐要求审计，工程、实验、科学及后续结题分开，旧M3 any-unresolved/paused/advancefalse、B failed/B01FAIL、UTC owner整包拒绝与K3 complete-not-accepted不改。新包工程900软1800硬、C10800软21600硬、后续M1 5400软10800硬均最早CPU入口至清理整轮计时，软超继续留overrun，仅硬限因时长截断，失败全额记账，新修复另冻协议新输出。普通决策与节点自主不逐次问，账本无总GPU-h许可上限；每节点对表，默认共驻每spawn只读UUID/余量、不信号邻居，实验离线，test不读、不新增数据/依赖/solver家族/seed、不改冻结结果/判据/测试/用户安全配置。本轮不操作main、issue关闭或其它保留权限，不以文档登记当实验进展或宣称SOTA；不要自行宣布目标完成。

## §1 现状、由来与精确身份

- 登记起点 `61db2725f6fda6abf986d94b7dbd0c6dd0bf12a4`，父 `523c819b8d42a3f43163eb53509f76136d8038f4`，工作分支 `r7/weather-reasoning`。主计划 `docs/goals/main-model-v2-campaign.md` §8 已转 current=N4/previous=N3；实际上一轮是 `docs/goals/v2-remaining-stages-exploration.md:2` 的 N3 marker，其§8已追加“下一动作：N4…”，不倒改历史。
- N3出口来自 `docs/R7_74_STATISTICS_COMPLEMENT.md` §6/§8，index `n3-b-statistics-complement-negative`，evidence commit `1615795d70117af608787ad6de15d2e5640b901c` / SHA256 `7ea1b5b7f6f9f2f22f1a4b39b59750112486f35bbee745d835b90db78ac18806`。两步 rollout 四 primary 胜200L6却皆劣400L6，冻结规则选 L6，仅终结该假设；原 B failed/finalized:false、B01 inventoryFAIL 保留。
- 真正 #72/K3 与官方 B UTC 已受限独立接受并登记两新 audit/not-candidate 记录，冻结页见 `docs/R7_72_ORIGINAL_K3_REFERENCE.md:294` 与 `docs/R7_74_UTC_STATISTICS.md:184`。K3 保持旧365.25/accumulated lead/interior_2，不是 Gregorian/architecture-matched paired C；原 complete-not-accepted 与 success/finalized:true 不改。UTC owner old whole-package NOT ACCEPTED/full-log-prefix mismatch 与官方独立 metadata 接受分层，negative/undefined 全保留。
- 实际精确 parent CI37155293949/attempt1/job111297340351 completed/failure，测试step9 failed、post17 skipped，**全部12必要步骤未通过**。回执 `outputs/r7_v2_remaining_acceptance_20261003/engineering_523c819_ci_network_recovery_20261003T215523Z_6cb51964/verification_receipt.json` SHA256 `a2e054d5903f4f48a55eb046780de803efcebdc5d9e6629c617e2f0def05ab5f`；远端失败名称/计数未取得，不能用本地3467pass或旧工程CI追认green。
- 主链只读诊断初轮17failed/220passed/1warning、pytest153.34秒/whole157.486秒来自 basetemp 嵌入 evidence root 的 nested-old-output harness 缺陷；不是远端归因，不改 guard。纠正独立clean523/Python3.12全量实际3462passed/14skipped/6warnings、917.94秒pytest/whole926.8922247411683秒、1800软3600硬/overrun0/RC0/sourcechanged[]/ownedreaped；protocol `1475fd57bfc678932c80cdf892cdb9cf43b58eddac5668d0a93c03339d62677f`、tree `5b6699bb78249d6251770c5d5b78e582b9ba341b55183bf001700ff11b475956`，回执 `/tmp/r7_clean_full_523_cpu_20261003_gn645i1v/attempt.json`。14skip为6CUDA+8缺M2/D1/optional fixtures，不算通过；6warning三junit record_property+duplicatezip+两Lightning保留。另建独立Python3.11.15/torch2.14.1+cpu精确523 full已实际3462passed/14skipped/6warnings/0fail/0error、pytest906.85秒/child+reap909.57秒/attempt whole909.92秒/exit0、425sourcepins前后不变、tree `cb81953cebc6bc032ac619cfa144c0da28101656`；六warning为三record_property+duplicatezip+两Lightning，主链stdlib JUnit确认两环境skip identities精确同14，不算通过，不拿child/pytest时间代whole。两本地绿不接受实际remote523失败或新包CI，remote cause未复现/确认；本登记未执行这些诊断。
- 新 C `prepare/run`、完整144jobs、新包GPU precision、实际 M1 与 adaptive 尚未执行。旧 `docs/goals/n4-m5-confirmation.md` 是2026-10-02 prepared历史，不作本轮目标或沿用其预算/授权门。
- M1历史#71对照全是 Process 路径，不能作为旧/当前Process/matched-Generic完整单输入因素归因。临时 owner `/tmp/r7_m1_current_20261003_af95660c/engineering_receipt.json` SHA256 `be529b82e284cd9355dfbdf32be241aa4f436ae341e8f1329c2fccf7414b24cf`，whole1816.6227561449632秒>hard1800，`engineering_accepted:false`；独立 `/tmp/r7_m1_review_20261003_wrw_xxug` 审阅进行中，不把临时准备当实际M1或C前置。

## §2 交付物清单 D1–D5

| # | 交付物 | 可核证据形态 |
| --- | --- | --- |
| D1 | 新包完整工程资格；历史失败与当前CI分开 | exact新SHA/CI run/job及12必要步骤；独立新包FP32/BF16/resume/poison回执、protocol/digest/全程成本；不复用旧precision为新包通过 |
| D2 | C新独立冻结并实际9train/135eval | protocol.json及digest、代码commit/zip、source/data/preflight/BUILD_COMPLETE、185-window与同seed初始化映射、selected400/完整144worker inventory；17变量/五lead/三区域/K1/2/4及零训练baseline全部指标 |
| D3 | 实际C全包独立审计、四成本与adaptive读法 | 原始逐case充分统计、配对/单位/negative/undefined、params/FLOPs/训练评估/真实isolated latency/allocated-reserved；独立receipt/seal、四项门逐seed/lead与evaluated/start_training/oracle_deployable:false |
| D4 | 后续M1-only缺失控制，C独立接受先行 | C old_ours两seed10个K4scores身份复用，4fresh scratch trains/20eval新协议及整包独立接受；相同anchor/优化器/L6/all-draft/K4/窗口/案例，单输入因素信息白名单与反证、UTC另冻CPU协议 |
| D5 | 每节点登记、账本、对表与逐要求审计 | 冻结页/E/index/canonical brief/精确CI/产物digest、whole与GPU一次记账、全部失败/overrun/未做/复现等级；下一节点实证清单，不自判最终goal |

## §3 判据与证据来源（只引用，不新写阈值）

- 唯一科学前瞻冻结来源 `docs/R7_V2_CONFIRMATION_PREREGISTRATION.md` §1–4，原字节不改：primary t2m6/12h、0.0退化容忍、同seed/lead/K/region配对，完整17通道/五lead/三区域/K1/2/4及坏case；固定process四项adaptive门。当前 **unevaluated**，不是failed gate；合法negative/mixed/cannot-distinguish不是缺实验成功。
- L6选择只读 `docs/R7_74_STATISTICS_COMPLEMENT.md` 与旧N3预声明；Generic同结构、同已知输入、无过程语义遵 `docs/decisions/0023-main-model-first-baseline-freeze.md`。C包级改善不能拆称某一reader/query或过程因果作用。
- 初始化/训练：seed41/42/43，每seed三臂共享fresh scratch anchor及完整target映射；新优化器、初始/all-draft深监督、Ktrain4/fullBPTT、同样400 L6 updates、只选update400。不续训旧权重，不事后换seed、端点或有利checkpoint。135=9×5lead×3K，三region是每次评分而非另训；全144尚未跑。
- M1只接 `docs/R7_MAIN_MODEL_V2_DESIGN.md` §4与 `docs/plans/0004-r7-main-model-v2.md` #71验收；当前C整体包不能自动代单输入因素。启用 spacetime_inputs/known_context_inputs，仅确定性时间/经度/历史offset输入，保留旧pooled reader/solver，不加RW-A/local-Z/direct draft query/source-position/roles或过程aux。新增输入容量须披露；旧Ours与当前M1-only Process/对应Generic各须真实训练证据，旧Ours仅合法复用实际C已独立接受的两seed评分与初始化/配方身份，不新增旧Generic臂或改名分数。
- 运行/治理 `docs/decisions/0030-autonomous-execution-and-direction-boundary.md`、`docs/decisions/0032-independent-autoregressive-exposure-route.md`、`docs/rules/ci-and-verification.md`、`docs/rules/gpu-resources.md`；skip/queued/cancelled/failed不算通过。M3 any-unresolved/paused/advancefalse不改，不作为该独立路线总前置。

## §4 实施顺序与依赖（不跳步）

1. governance gates与身份对表后，主链先完成实际523 CI失败归因/合法活跃修复、精确新包CI和独立新包precision接受；工程未齐停止相关GPU动作，不重旧失败、不逐次问普通许可。
2. D1齐备才执行C最早prepare入口：独立排他新输出、同boot whole anchor、冻结协议/源码归档/数据只读资格/初始化与完整144job计划；冻结完成仍不算实跑。
3. 一次C按seed41/42/43×old_ours/process/matched_generic各400 L6更新、K4/selected400执行；同checkpoint135项K1/2/4验证，全部17变量、五lead、full/interior/edge_2、22/21/19/15/11病例与零训练persistence/train-climatology；K1/K2标非独立训练。
4. 全库存、时钟、配对、单位与四成本独立接受，实际C本身**不依赖M1验收**；此后才评价adaptive四门，缺C证据不写失败或默认通过。若门成立才另冻控制器协议/数值预算，禁止为赢扩复杂度。
5. C独立接受后，资格核其old_ours seed41/42的10个K4score、scratch anchor/配方/窗/单位/病例，**仅复用score，不从selected400权重续训**。另冻M1-only新协议、4fresh scratch Process/Generic400L6train、20K4eval、185窗、完整cohorts与实际成本，独立反证/整包核验，UTC另立CPU协议。
6. 每节点实际登记与对表后接N5的必要反证/逐issue证据审计；普通方向内动作连续，不等待逐轮触发。本轮不实施issue/main操作，最终完成由用户/运行时独立校验。

## §5 预算与停止条件（数值先冻结，whole不重置）

| 独立轮 | planned_seconds软 | hard_cap_seconds硬 | 起止与范围 |
| --- | ---: | ---: | --- |
| 新包工程/precision | 900 | 1800 | 最早CPU入口先于imports/pins，准备/worker/聚合/最后owned reap与发布；新协议，不沿用旧成功 |
| C实际确认 | 10800 | 21600 | 最早prepare入口至9train/135eval/汇总/清理/最后封印，全144job |
| 后续必要M1-only | 5400 | 10800 | C独立接受资格核起至4train/20eval/完整审计与清理；旧baseline10scores只复用 |

软超继续，记录 `soft_overrun_seconds=max(0,whole-planned)`；只有硬上限因时长截断，记budget_limited/failed且全额成本保留。最早同boot anchor不得在CUDA或reload重置；独立审阅有自己的先冻数值预算，不能用inner快照替代final whole。真实错误/身份/资源/不完整集合停止对应attempt，保留原失败；修复另立协议新输出，不复活旧轮。无总GPU-h授权上限，24仅会计基数。GPU默认共驻，每spawn只读UUID/headroom，不对邻居发信号；本次治理登记0GPU。遇保留权限只停相关动作，不执行付费、新数据、发布、独占、main合并或破坏性操作。

## §6 与planner草稿的差异

本轮路由和数值范围来自用户规定及冻结判据，不外委科学标准。本次登记子智能体没有planner/Agent工具，使用主链已冻结纠正方案；主链已实际委派planner并拒绝错误草案，普通治理整理降级自规划；工作态 `/tmp/r7_governance_registration_audit_20261003_NRQYhoBF/plan.json` 已过同一check_planner_plan契约，verified:true，不作实验或完成证据。M1采用主链纠正草案 `/tmp/r7_m1_completion_corrected_plan_20261003.json`，不沿用被拒的信息/loss/初始化/cohort偏移；临时owner硬超与独立审阅未终态不伪接受。

## §7 明确不做

- 不改原M3暂停/failed、B失败/B01FAIL、K3原candidate或UTC owner required-log拒绝；不重训/评估旧B或重复旧precision成本。
- 不修改冻结primary/容忍/adaptive页或既有结果/源码归档，不删弱测试；不将更多文档/CI/metadata接受当天气收益。
- 不读封存test或据曝光test选择，不读写新天气数据、不下载/发布、不装依赖、不改用户配置/凭据/安全hook。
- 不新增solver家族、额外seed、adaptive训练练到赢，不将C全包误当M1单因素、K3误称Gregorian同构paired。
- 不操作main/issue关闭、不force/merge、付费/独占/邻居信号、cron/守护或会话后自建续跑；不自宣SOTA、科学PASS或最终goal complete。

## §8 进度块

- **状态**：实证完成（2026-10-04）。D1–D4 全部执行：新包 precision 独立接受（E-245）；actual C 144/144 job 独立接受（E-246）；actual M1 24/24 job verify 通过（E-247）；两侧 UTC 标量统计链完成；adaptive 四门 evaluated、`gate_met:false`、控制器不训练。
- **已登记（2026-10-04）**：`docs/R7_C_ACTUAL_CONFIRMATION.md`、`docs/R7_M1_ACTUAL_AND_UTC.md`、`docs/R7_V2_NEW_PACKAGE_PRECISION_ACCEPTANCE.md` 冻结于 `858eddbf7b917ac158689c5f7de150fca1152e47`；E-245–E-247；账本新行 0.0544/3.1550/0.8758，逐行显示总 **9.9139**、余 **14.0861**（精确总 9.913830300275308 / 余 14.086169699724692）。
- **C 读数**：package 相对 old_ours 在 primary 6/6 cell 严格改善；process vs matched_generic unresolved（1e−5 K 量级差正负不稳）→ 不归因过程语义；adaptive 门 accuracy_cost_tradeoff 未过。
- **M1 读数**：M1 版相对 old_ours 4/4 cell 严格改善；process−generic 4e−6 K 量级 → 单因素 unresolved；#71 无实际改善，按原条款不得 DONE-positive。
- **六项recheck**：①actual N3§8 next N4与主state/current目标一致，旧N3 marker不变；②新行各记一次、旧B/precision不重记、四历史notes保持；③上一轮B02/K3/UTC记录evidence_commit可达；④本轮实现0030/0032/L6与冻结preregistration；⑤冻结页/源/tests/用户配置不改，primary/容忍/adaptive判据未动；⑥实验均先冻协议、离线、共驻只读余量。
- **实际验证**：full pytest 与 conventions/campaign/index 门禁见 N5 登记轮（`docs/goals/v2-issue-closeout.md` §7）；官方 CI 以 858eddb+登记提交的精确 run 为准。
- **未做/下一动作**：N5 登记与六 issue 决定已按 `docs/goals/v2-issue-closeout.md` 执行；#71 过程语义独立贡献的重开条件、#75 之后的最终泛化测试均未做，仅提议。

### Objective副本（与§0同段；实测 1179 字符）

> 在 docs/goals/v2-l6-confirmation-current.md 按0030/0032连续推进实际N4：D1先核N3/B02已登记出口、真正原K3与官方UTC受限独立接受，修复精确523c819/CI37155293949的真实失败并取得新工程包精确CI全部12必要步骤success、独立新包FP32/BF16/resume/poison精度接受，未齐不启动C；D2冻结独立C协议/代码/数据/初始化/window/产物身份，引用原R7_V2_CONFIRMATION_PREREGISTRATION不改primary或容忍，三臂old_ours/process/matched_generic×seed41/42/43，同seed同scratch anchor、新优化器、各400 L6 updates/Ktrain4/fullBPTT/selected400，共9训练与135同checkpoint K1/2/4评估，完整17变量×5lead×3region、185训练窗口与22/21/19/15/11完整val，负值undefined、四成本与实际时钟全部保留；D3独立核实际144job整包后才评价固定process的四项adaptive冻结门，当前unevaluated不是failed，不以K数代latency或加复杂度练到赢；D4仅实际C独立接受后复用old_ours seed41/42的10个K4评分和相同初始化/配方身份，4项fresh scratch M1-only Process/Generic各400 L6训练与20项K4评估补#71单输入因素缺口，禁RW-A/direct-query/source-position/roles/local-Z混入，不重训旧baseline或续训其权重；D5登记完整正混负/身份/成本/未做与逐要求审计，工程、实验、科学及后续结题分开，旧M3 any-unresolved/paused/advancefalse、B failed/B01FAIL、UTC owner整包拒绝与K3 complete-not-accepted不改。新包工程900软1800硬、C10800软21600硬、后续M1 5400软10800硬均最早CPU入口至清理整轮计时，软超继续留overrun，仅硬限因时长截断，失败全额记账，新修复另冻协议新输出。普通决策与节点自主不逐次问，账本无总GPU-h许可上限；每节点对表，默认共驻每spawn只读UUID/余量、不信号邻居，实验离线，test不读、不新增数据/依赖/solver家族/seed、不改冻结结果/判据/测试/用户安全配置。本轮不操作main、issue关闭或其它保留权限，不以文档登记当实验进展或宣称SOTA；不要自行宣布目标完成。
