# 目标：S3 匹配算力重估（含训练年份覆盖轴重开）与 S4 就绪

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-fy-year-coverage-matched", "driver": "scripts/study_r7_s3_fy_year_coverage_matched.py", "output_stem": "r7_s3_fy_year_coverage_matched_20261026", "control_record": "s3-fy-annual-coverage-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_fy_year_coverage_matched.py"], "planned_seconds": 259200.0, "hard_cap_seconds": 518400.0, "code_archive": "outputs/r7_s3_fy_year_coverage_matched_code.zip", "commit_paths": ["scripts/study_r7_s3_fy_year_coverage_matched.py", "scripts/r7_s3_fy_year_coverage_matched_support.py", "tests/test_r7_s3_fy_year_coverage_matched.py", "docs/goals/s3-year-coverage-matched-reeval.md", "docs/goals/s3-matched-exposure-reeval.md", "docs/goals/main-model-climatology-campaign.md", "docs/decisions/0062-r1-year-coverage-matched-compute-round.md", "docs/decisions/README.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): implement the R1 matched-compute year-coverage re-run (updates 2400->16071, batch 2)", "register_paths": ["docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-matched-exposure-reeval.md"], "register_message": "docs(r7): register the R1 matched-compute year-coverage reading as an evidence record"} -->

本文件是 goal 模式的目标长文源。objective 见 §0；交付物见 §2；判据见 §3；
实施顺序（含 workflow 编排）见 §4；预算与停止见 §5；进度块见 §8。
**本轮（R1，Track 1 第 1 轮）的机器可读 spec 就是本文件顶部的 `round-spec` 块**，
`campaign-state.current_round_goal` 指向本文件；R1 的人读 brief 是
`docs/goals/s3-year-coverage-matched-reeval.md`（只引用决策 0060/0061，不重写 ADR）。

**状态：active（2026-10-25 起）。** 设计依据是 `docs/plans/0026-matched-exposure-reeval-and-s4.md`
（计划是不可变记录，本长文是它的活文档执行源）；**本长文对交付形状做了一处更正，见 §6**。
`scientific_claim: false`、`test_read: false`（Track 0 的评分按 §4.1 单独预声明后例外）。

---

## 0. objective（可直接粘贴，单段，实测 2572 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning执行目标长文docs/goals/s3-matched-exposure-reeval.md（先完整读它，细节与判据以它为准）。继承状态：S3节点、index90、账本累计247.356641GPU-h、2023test未评分r0、S4未启动；从tools/r7_round.py state的下一动作恢复，不重复已登记轮次；cap/remaining仅会计、无总GPU-h上限。本轮用workflow编排：先加载dynamic-workflows技能（未加载时CreateWorkflow拒跑），从.zcode/workflow-drafts/r7-campaign-round.dwf.ts派生.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts，maxRounds硬上限5、stage超时从34200000ms抬到约129600000ms（36h）、world.run命令名保持编译期字面量且运行值落盘（文件即参数）、自跑tools/r7_round.py commit-guard（提交前）与verify（登记后）不依赖hook兜底、只推工作分支r7/weather-reasoning（禁main/force/merge/关issue）；它是有界、可恢复的批量编排器而不是守护进程，被打断按ResumeWorkflowRun续；已知边界（实测）：后台workflow仍在跑时轮末goal完成校验可能不触发（是defer不是失败，因此没有校验器读数）、卡片可见性只从运行API观测、探针未覆盖GPU轮/freeze/commit/登记。两条轨：Track0先补2016再进S4——2016是唯一预声明的完整未见年份（1464帧，可用初值6/12/24/48/72h=1461/1460/1458/1454/1450），新建排他目录放只含2016帧的小store（装入FY candidate的2017–2021 normalization_mean/std）+逐字节复制FY该臂train.jsonl作同级文件（身份门才放行）+指向小store的test.jsonl，两臂各一份，不声明多个年份；然后冻S4评分协议（r=1、alpha_1=0.025）对FY candidate三份已冻结checkpoint评分，按D4组成（总集/各年/四季组×17变量×5lead×full/interior/edge）报读数与u10/v10/mslp相对同数据incumbent的变化，声明"气候态分母期与test期不同代"这一代价，若test覆盖仍不满足docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111则如实记录停在门前、不放宽判据。Track1按序跑匹配算力重估队列5轮（R1训练年份覆盖5y→22y、R2气候态锚异常反馈、R3全局混合、R4位置保留、R5训练信号族），新制度=65×65多年store（2000–2021train/2022val）×full_grid（4225点）×每臂≥1epoch，batch=2、updates×batch≥32142⇒updates=16071；每轮=新ADR（四段齐全、Consequences含负面后果）+新brief（§0–§8、预声明出口(a)–(d)+桥(e)、任一假即永久关闭该支）+过七gate+冻protocol.json+跑+四件套登记（§8进度块含下一动作第一条命令、账本行、index record、campaign-state）；只读复用store/两套清单/48桶气候态/迁移父，不重建store、不新增下载。R1的override ADR必须如实写：按决策0057的成文定义只改算力不满足"不是同一族"（store/网格/监督域/数据跨度四项全不变），因此是"用户具名授权推翻冻结条款"、取代docs/goals/s3-year-coverage.md:74的"不调2400updates"；写明负面后果（削弱"预声明出口不可翻案"纪律、为按算力重测开先例）并把先例范围限定在该轴、不授权重开depth/width/SAM/zero-base/EMA/长lead权重/rollout剂量；科学依据写成可证伪预测（该轮(a)(b)(c)已全真、(d)只栽在6/12/24h短lead，而短lead正是0.075epoch最吃亏处）。预算：用户已显式授权不够就自主加——现行per-seed43200s对≥1epoch必然不足（每seed约22.7h≈81900s），按决策0055先例另立预算决策（只改时间、不动科学门、用本轮首次实测支撑、不在运行中改硬限），名义起点per-seed129600/planned259200/hard518400秒（1:2:4），执行时按实测重定；注意per_seed_seconds在复用白名单内⇒改预算后旧attempt的seed不能复用必须重跑。授权：用户下放方向内实验设计、架构/方法、训练信号/优化、数据源/区域/年份/变量、GPU训练评估、seed/更新/规模、时长预算、普通决策/排障、下一实验与节点推进权；时长/GPU-h自主增加并简报。禁止：改冻结门、伪造PASS、删弱测试、绕身份、事后放宽已冻结判据、换名重开已关轴、不动data/raw|interim|processed与归档快照、不合成数据、不读test（除Track0冻结评分）、不main合并/force/关issue/删已有产物、GPU默认共驻按R-054只读核余量且不对邻居进程发任何信号、不paid/rental/exclusive、不cron/守护/会话外续跑。只在三种情况停：触保留授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在§8写"为何停+恢复第一条命令"。每步给可核查证据（命令+输出+digest），估计值不得写成实测值；结束按AGENTS.md汇报改动/验证/接口/数据影响/兼容性/依赖/未做/下一项。不要自行宣布目标完成。

---

## 1. 现状与前置

- **节点**：S3。`current_round_goal` 原为 `docs/goals/s3-year-coverage.md`（该轮已按预声明出口
  `mixed`/`not-candidate`、`(d)` 假 ⇒ 永久关闭年份覆盖支）。本长文是**新制度下**的后续目标。
- **为什么要有"匹配算力"这个新制度**：`docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md:117-119` 自己写
  「固定 2,400 updates 时候选臂只看到约 **7%** 的 epoch（32,131 窗）、控制臂约 33%」，
  因此「一个**负**结果**不能**区分『多出的 17 年没有额外价值』与『updates 太少、用不上多出的年份』」。
  决策 0057 据此把使能制度定为「**数据 × 匹配算力**」的合取。
- **新制度**：65×65 多年 store（`outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`，
  `[34084,17,65,65]`，train 2000–2021 / val 2022）× `full_grid`（4,225 点）× **每臂 ≥1 epoch**。
- **batch = 2**（用户已定）。实测依据：`docs/plans/0025-*.md:416-421` 的
  `seconds_per_update` = 1.839/2.351/3.360/5.620、`peak_reserved` = 2,322/4,528/8,950/18,092 MiB
  （batch 1/2/4/8）；**batch 8 不过共驻门槛（−661 MiB）**。`updates × batch ≥ 32,142`
  ⇒ batch 2 时 **updates = 16,071**。
- **年份轴重开的程序冲突**（详见 §4.2）：按决策 0057 `:26-28` 的成文定义，
  "不是同一族"的实质内容是"换 store / 网格 / 监督域 / 数据跨度"四项，而年份轴**只改算力**，
  四项全不变 ⇒ 必须走**用户具名授权推翻冻结条款**的 ADR。
- **Track 0 的数据前提已实测**：多年 store 的 `time_ns` 含 2016 完整年份（1,464 帧，可用初值
  1,461/1,460/1,458/1,454/1,450）；FY candidate 从未见过 2000–2016（train 2017–2021 / val 2022 /
  归一化 2017–2021）。**不需要下载**。
- **workflow 链路已实测可行**：`docs/goals/workflow-in-goal-probe.md` 与
  `docs/R7_WORKFLOW_IN_GOAL_PROBE.md`（2026-10-11，run id
  `dwfrun-8b989f84-1043-408e-bd07-4690677135fc`、终态 `completed`、用时 1m07s，六环全通）。

---

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | Track 0 的 2016 小 store 与同级 `train.jsonl`（两臂各一份） | 排他 `outputs/` 路径 + `BUILD_COMPLETE` + 两臂 `dataset_identity` 实测等于 checkpoint 记录的 `data_identity` |
| D2 | 2016 的 S4 评分协议（先冻结） | `protocol.json` 的 `protocol_sha256` + r=1 / `alpha_1 = 0.025` 登记 + "分母期与 test 期不同代"写进 `limitations` |
| D3 | S4 首次评分读数 | 按 D4 组成的逐组读数（总集 / 各年 / 四季组 × 17 变量 × 5 lead × full/interior/edge）+ u10/v10/mslp 相对同数据 incumbent 的同组同 lead 变化 |
| D4 | Track 1 五轮各自的 ADR | `docs/decisions/NNNN-*.md`，四段齐全、**Consequences 含负面后果**（R-033）+ `docs/decisions/README.md` 索引行 |
| D5 | Track 1 五轮各自的 round brief | `docs/goals/<slug>.md`，过 `check_goal_brief.py --brief`，带预声明出口 (a)–(d) + 桥 (e) |
| D6 | 五轮各自的冻结协议与逐 seed 回执 | `protocol.json` digest + `seed<N>_receipt.json` + `window_sha256` 与协议逐字节一致 |
| D7 | 五轮各自的读数与判决 | 证据页（`docs/R7_*.md`）+ 出口逐条数字 + `outcome_class`（candidate / negative / null / mixed） |
| D8 | 每轮的交接四件套 | 主计划 §8 进度块（含下一动作第一条命令）+ §7 账本行 + `docs/R7_EVIDENCE_INDEX.jsonl` record + `<!-- campaign-state: ... -->` |
| D9 | workflow 运行产物 | run id + 终态 + 每轮相位记录；被打断时写明 `ResumeWorkflowRun` 的恢复命令 |
| D10 | 本长文的 §8 进度块 | 状态机：已完成 / 未做 / 下一动作 |

---

## 3. 判据与证据来源

- **科学门**只引用 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与各轮 brief 自己冻结的出口；
  **不新增阈值、不放宽任何已冻结判据**。
- **年份轴重开的正当性**：`docs/decisions/0057-supersede-0056-queue-is-executable.md`、
  `docs/decisions/0058-anchor-anomaly-feedback-reeval-admission.md`、
  `docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md`（关闭读数与它的 limitation）、
  `docs/goals/s3-year-coverage.md:68-74`（被取代的关闭条款）。
- **Track 0 的数据与身份**：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111`
  （"至少覆盖一个完整未见年份和全部四季"）、`docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md`
  （FY candidate 的冻结身份与读数）。
- **workflow 的边界**：`docs/R7_WORKFLOW_IN_GOAL_PROBE.md` §1（四条边界）、
  `docs/decisions/0044-workflow-as-round-executor.md`（自跑闸门 / 只推工作分支 / 每轮一臂 /
  有界不自宣）、`docs/plans/0020-campaign-driver-workflow.md`（驱动形态与成本）。
- **机械门**：`python tools/check_conventions.py`（37 条阻断）、
  `python tools/check_goal_brief.py --brief <每轮 brief>`、`python tools/check_campaign_state.py`、
  `python tools/verify_r7_evidence_index.py`。
- **不算通过的**：skip / cancelled / queued / 被硬上限截断的 attempt；`report()` 与 CI 绿都不是科学结论。

---

## 4. 实施顺序（依赖，不跳步）

### 4.0 workflow 编排（本长文的执行载体）

按 `docs/decisions/0044` 与 `.agents/skills/parallel-research-workflow/SKILL.md` 的
「campaign 驱动 workflow」节：

- **先加载 `dynamic-workflows` 技能**（未加载时 `CreateWorkflow` 拒跑）；
- 从 `.zcode/workflow-drafts/r7-campaign-round.dwf.ts` 派生
  `.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts`（草稿，git-ignored）；
- **`maxRounds` 硬上限 5**；**stage 超时从 `34,200,000 ms` 抬到约 `129,600,000 ms`（36 h）**
  —— 现行模板的 9.5 h 装不下 22.7 h 的 seed 阶段；
- 一轮 = 侦察 → 提可证伪假设 → 派生单因素驱动 → 独立复核（≤3）→ 冻结并等 CI → 逐 stage GPU
  → 读数 → 四件套登记 → 交接；
- `world.run` 的**命令名保持编译期字面量**，运行值落盘（顶部那行 `round-spec` 注释块，文件即参数；
  正文里不得再出现第二个 `round-spec` 注释块，否则 `round_spec()` 会拒绝该文件）；
- **自跑确定性闸门**：提交前 `tools/r7_round.py commit-guard <paths>`，登记后 `verify`；
  **不依赖 hook 兜底**（子会话很可能不跑本仓 hook，E-187 推测）；
- **只推 `r7/weather-reasoning`**；禁 main / force / merge / 关 issue；
- **有界、可恢复，不是守护进程**：进程退出即 `interrupted`，按 `ResumeWorkflowRun`
  或 `args.resumeFromRound` 续；状态在 §8 与账本里，不在 workflow 里；
- **四条已知边界**（`docs/R7_WORKFLOW_IN_GOAL_PROBE.md` §1）：① 后台 workflow 仍在跑时
  **轮末 goal 完成校验可能不触发**（是 `defer`，**不是失败、也不等于通过**）；
  ② 卡片"可见"只从运行 API 观测，UI 渲染未核；③ 探针**未覆盖** GPU 轮编排 / freeze / commit /
  四件套登记 —— 所以本长文必须自跑闸门；④ 探针不验证无人值守多轮驱动。

### 4.1 Track 0 —— 先补 2016 再进 S4（0 GPU-h 训练，最高优先级）

1. 新建排他目录，放三样：① 只含 2016 帧的小 store（约 0.42 GB raw），
   **装入 FY candidate 的 2017–2021 `normalization_mean/std`**
   （`data/r7_store.py:113-120` 从 store 读、无外部覆盖参数 ⇒ 只能由 store 携带）；
   ② **逐字节复制** FY 该臂的 `train.jsonl` 作同级文件 —— 身份门
   （`training/r7_evaluate.py:121-123`）比的是**同级 `train.jsonl`** 的 `dataset_identity`，
   逐字节复制才恰好等于 checkpoint 记录的 `data_identity`（已实测：treatment `c17494ab…`、
   control `0469312978…`）；③ 指向小 store 的 `test.jsonl`（2016）。**两臂各一份。**

   > **⚠ 2026-10-26 更正（执行时实测，见决策 0059）：上面第 1 条的字面机制不可实现。**
   > ① `dataset_identity`（`training/r7_experiment.py:89-103`）除清单字节外还绑 **store 的
   > `state` 形状与 `time_ns` / 归一化数组** ⇒ 换 store 后逐字节复制 `train.jsonl` **不足以**
   > 让身份门放行；② `validate_store`（`data/r7_store.py:145-146`）强制
   > `normalization_years == split_years['train']`，而 `chronological_splits`
   > （`data/preprocess/contracts.py:63-64`）要求 train < val < test ⇒
   > **「2016 目标帧 + 2017–2021 归一化」在同一份 store attrs 里不可同时为真**；
   > ③ 标准构建器（`data/preprocess/r7_era5_zarr.py:166-167`）还要求 train/val/test
   > 各自非空 ⇒ 只含 2016 帧的 store **也造不出来**。
   > **改用决策 0059 的机制**：不新建 store；2016 帧取自已冻结的**多年 store**
   > （`outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`），由**显式 init-time 集合**钉住；
   > 驱动**显式**断言 checkpoint 的 `data_identity` 等于 `dataset_identity(FY 该臂 train.jsonl)`
   > （真校验，不绕过）；评分时**显式**施加 FY 的 2017–2021 归一化数组（从 FY store 读、按 sha256 钉住）；
   > 窗口选择按显式 init-time 集合做，**只**跳过 `validate_record` 里那条无法表达的
   > 「跨 split 年份」子句（`data/r7_store.py:179-183`），并在协议里逐字声明。
2. **预声明恰好一个年份（2016）**：D4 用 max-statistic 同时区间，cell 越多临界值越宽
   ⇒ 1 年 = 30 primary cell，17 年 = 110。2016 是紧邻训练窗之前的完整年份，代差最小。
3. 把"**气候态分母期（2017–2021）与 test 目标期（2016）不同代**"冻进协议并写进 `limitations`。
4. 冻 S4 评分协议（**r = 1、`alpha_1` = 0.025**），对 FY candidate 三份已冻结 checkpoint 评分，
   按 D4 组成报读数。**评分消耗 r=1 曝光、不可逆**，执行前把 r/alpha 登记做全。
5. 若 test 覆盖仍不满足 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111`，
   **如实记录并停在门前，不放宽判据**。
6. 这是 D3 暂缓的"重建 store"路径，**用户已显式授权启用**；不做任何下载。

### 4.2 Track 1 —— 匹配算力重估队列（5 轮，每轮一个臂）

| 轮 | 轴 | 依据 | 为什么现在能问 |
| --- | --- | --- | --- |
| **R1** | 训练年份覆盖 5y → 22y | **新 override ADR（见下）** | (a)(b)(c) 已全真、(d) 只栽在 6/12/24h 短 lead；短 lead 正是 0.075 epoch 最吃亏处 |
| **R2** | 气候态锚异常反馈 | 决策 0058 已准入；brief `docs/goals/s3-anchor-anomaly-feedback-reeval.md` 已在；锚表已发布 `outputs/r7_s3_fy_anchor_table_20261025_attempt01/`（`table_sha256 18448ae3df2b260e…`） | 五 lead 3/3 seed 配对全改善（72h −0.321 K），只栽绝对门 2/3 |
| **R3** | 全局混合 | 决策 0057 队列 | null，页内明写"更强操纵/更多数据下未测"；吃算力不吃显存（实测 8.955 GiB） |
| **R4** | 位置保留 | 决策 0057 队列 | null 来自**弱操纵**：`position_query` 模长 0.0074–0.0125 vs 同层投影 ~8.02，差 3 个数量级 |
| **R5** | 训练信号族（空间梯度 / dense 12 步 / final-only / tendency） | 决策 0057 队列 | mixed（有配对改善但绝对门未过）、模型零改动 |

**每轮的程序（硬约束）**：新 ADR（四段齐全、Consequences 含负面后果）→ 新 brief（§0–§8、
预声明出口 (a)–(d) + 桥 (e)、任一假即永久关闭该支）→ 过七 gate（`AGENTS.md:26`）→
**只读复用** store / 两套清单（`manifests_control` 7,291、`manifests_treatment` 32,142）/
48 桶气候态 `reference.json` / 迁移父（**不重建 store、不新增下载**）→ 冻 `protocol.json`（R-006）
→ 跑（每轮一个臂、GPU 由统一父调度按共驻余量**串行**、**不对邻居进程发任何信号**）→ 四件套登记。

**R1 的 override ADR 必须写明**：

- 如实写：按决策 0057 的成文定义，只改算力**不满足**"不是同一族"
  （store / 网格 / 监督域 / 数据跨度四项全不变），因此走的是**"用户具名授权推翻冻结条款"**，
  取代 `docs/goals/s3-year-coverage.md:74` 的"不调 2400 updates"；
- 写明**负面后果**：削弱"预声明出口不可翻案"的纪律；为按算力重测开先例；
- 把先例范围**限定在年份覆盖这一条轴**，明确**不授权**重开 depth / width / SAM / zero-base
  decoder / plain EMA / 长 lead 权重 / rollout 剂量；
- 科学依据写成**可证伪预测**：该轮 (a)(b)(c) 已全真、(d) 只栽在 6/12/24h 短 lead，
  而短 lead 正是 0.075 epoch 最吃亏处 ⇒ 预期补上 ≥1 epoch 后 (d) 翻转。

**队列外**（方向明确的负结果，不因"更多数据"翻案）：SAM（15/15 一致更差）、
zero-base decoder（72h +1.37 K）、plain EMA、长 lead 权重、rollout 剂量、depth、width。

---

## 5. 预算与停止条件

- **预算必须重定**：现行 per-seed 43,200 s 是按"两臂 × 2,400 updates"标定的；
  ≥1 epoch 下每臂训练约 10.49 h + 评分约 0.88 h ⇒ **每 seed 约 22.7 h ≈ 81,900 s**。
  按决策 0055 的先例**另立预算决策**（只改时间、不动科学门、用本轮首次实测支撑、
  **不在运行中改硬限**）。名义起点：**per-seed 129,600 s（36 h）/ planned 259,200 s（72 h）/
  hard 518,400 s（144 h）**，比例 1:2:4；**执行时按实测重定**。
- **两个机制点**：① `per_seed_seconds` 在 `REUSE_PROTOCOL_FIELDS` 白名单内
  （`tools/r7_round_support.py:74`）⇒ 改预算后**旧 attempt 的 seed 不能复用、必须重跑**；
  ② workflow 的 stage 超时必须抬到 36 h（见 §4.0）。
- **时长口径**：软预算超了继续并记 `soft_overrun_seconds`；**只有硬上限因时长截断**，
  记 `budget_limited` / `failed` 并全额记账，**不算科学通过**。无总 GPU-h 上限，账本只记账。
- **成本量级（外推，须以实测替换）**：batch 2 下 6 臂（2 臂 × 3 seed）**68.2 GPU-h**（下界；
  按 `attempt03` 实测的 2.265 s/update 约 **84 GPU-h**）；双卡两并发 + 一随后 ⇒ 墙钟约 34–45 h。
- **停止条件（仅三种，决策 0043）**：触保留授权边界（付费 / 租卡 / 独占 / 方向改变 /
  main 合并 / 破坏性 / 无法安全克服的外部硬依赖）；达已冻结的科学停止条件；
  确无可安全独立推进的工作。**停下必须在 §8 写「为何停 + 恢复第一条命令」。**

---

## 6. 与计划 0026 的差异（形状更正）

`docs/plans/0026-matched-exposure-reeval-and-s4.md` 把 goal 与 workflow 写成**两件交付物**
（新 goal 长文 + 让另一窗口"派生 workflow"的说明），并把提示词拆成 Part A / Part B。
**这个形状是错的**：仓库早已把 workflow 安排在 goal 内 ——
`docs/goals/main-model-climatology-campaign.md:12` 的 objective 里就写着
「**用workflow**：先加载 dynamic-workflows 技能（未加载时 CreateWorkflow 拒跑）…
**多轮编排用 `.zcode/workflow-drafts/r7-campaign-round.dwf.ts`**…脚本自设轮数硬上限、
自跑 commit-guard/verify 不依赖 hook、只推工作分支」，而
`docs/goals/workflow-in-goal-probe.md` + `docs/R7_WORKFLOW_IN_GOAL_PROBE.md`（2026-10-11）
已经**实测过这条链路可行**（run id 真实、终态 `completed`）。

**更正**：workflow 编排写进**本长文的 §0 objective（内联）与 §4.0**，不再外挂一份说明；
最终提示词就是 §0 那段 objective。

**0026 的其余内容全部保留有效**：§1 的程序冲突、§2 Track 0、§3 队列、§4 预算、§5 每轮程序、
§8 明确不做、§9 风险，本长文逐条沿用。0026 作为计划**保持不动**（计划是不可变记录）。

---

## 7. 明确不做

- 不动 `data/raw|interim|processed` 与归档快照；不读 test 除非进入 §4.1 的冻结评分。
- 不 main 合并 / force / 关 issue / 删已有产物；不合成数据。
- 不跳过或弱化测试与断言；**禁止事后放宽已冻结判据**；**禁止换名重开已关轴**。
- GPU 默认共驻、只读核余量、**不对邻居进程发任何信号**。
- 不重开 depth / width / SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量。
- 不 paid / rental / exclusive；不 cron / 守护 / 会话外续跑。
- 不自行宣布科学通过、不称泛化 / SOTA、不自行宣布最终 goal 完成。

---

## 8. 进度块（状态机）

- **状态**：active（2026-10-25 建立）。最终 goal 状态不由本文件宣布。
- **已完成**：
  - 本长文建立（§0 objective 实测 2572 字符，过 `check_goal_brief.py --brief` 的 G-01..G-08）。
  - 前置核实（只读）：workflow-in-goal 链路已实测可行（2026-10-11 探针）；
    2016 在多年 store 内在位；batch 1/2/4/8 的实测速率与显存已在案；年份轴 override 的程序冲突已写明。

### 2026-10-26 — 开工轮：§4.1 机制更正 + workflow 脚本派生（本窗口）

**做了什么（全部只读 + 派生一个草稿脚本 + 两份文档）**：

1. **§4.1 的字面机制实测不可实现，已更正并留痕。** 见上文 §4.1 的 ⚠ 更正框与
   **决策 0059**（`docs/decisions/0059-s4-2016-instance-store-contract.md`，含 R-033/R-036 通过）。
   三条实测证据（读码，`file:line` 见决策）：
   - `dataset_identity`（`training/r7_experiment.py:89-103`）除清单字节外还绑 store 的
     `state` 形状 + `time_ns` + 归一化数组 ⇒ 逐字节复制 `train.jsonl` **不足以**过身份门；
   - `validate_store`（`data/r7_store.py:145-146`）+ `chronological_splits`
     （`data/preprocess/contracts.py:63-64`）⇒「2016 目标帧 + 2017–2021 归一化」**不可同时为真**；
   - 标准构建器（`data/preprocess/r7_era5_zarr.py:166-167`）要求三 split 各非空 ⇒
     只含 2016 帧的 store **造不出来**。
   **更正后的机制**：不新建 store；2016 帧取自已冻结多年 store，显式 init-time 集合钉住；
   驱动显式断言 checkpoint 身份、显式施加 FY 归一化、只跳过无法表达的年份归属子句。

2. **workflow 脚本已派生**：`.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts`
   （从 `r7-campaign-round.dwf.ts` 派生，草稿、git-ignored）。按 §4.0 冻结的两处形状差异：
   `maxRounds` 硬上限 **5**；stage 超时 `34,200,000 → 129,600,000 ms`（36 h）。
   `world.run` 命令名保持编译期字面量、运行值落盘（文件即参数）。
   **commit-guard 语义由 `round-guard` 覆盖**（它内部就对 round-spec 的 `commit_paths` 跑
   `check_conventions.py --quiet --paths ...`）；`commit-guard` 子命令要求把路径写在 argv 里、
   与「argv 必须是编译期字面量」冲突，故不直接调用它。登记后照跑 `verify`。只推工作分支。

3. **本窗口没有跑任何 GPU 轮、没有读 test、没有提交任何登记。**

### 2026-10-26 — Track 0 D1'/D2 完成、D3 已启动（同一窗口）

- **D1'（实例，决策 0059）**：不新建 store；2016 目标帧取自已冻结多年 store
  `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`（`[34084,17,65,65]`，
  `source.nc` sha256 `1db839554cce30cd…`），由**显式 case 集合**钉住。
  **实测**：2016 = 1,464 帧、`2016-01-01 00:00 → 2016-12-31 18:00`、6 h 步长；
  可用初值 6/12/24/48/72 h = **1,462 / 1,461 / 1,459 / 1,455 / 1,451**
  （长文 §1 原先记的 1,461/1,460/1,458/1,454/1,450 每个 lead 少 1，**以实测为准**）。
- **D2（协议先冻）**：`outputs/r7_s4_fy_2016_20261026_attempt01/protocol.json`，
  `protocol_sha256 = 0dd100dcb4436a848bcd1b4fa91fdd8709f2a99a03e686b750b765572be2db1c`；
  人读版 `docs/R7_S4_FY_2016_CONFIRMATION_PROTOCOL.md`。
  **r = 1、`alpha_1` = 0.025 已登记**（§4.1 第 4 条：先登记后评分）；
  「气候态分母期 2017–2021 与 test 目标期 2016 **不同代**」写进 `limitations`。
  归一化按 sha256 钉住（mean `e15c3f26…`、std `afc998a3…`）；
  气候态 = FY 的 48 桶 train-only（`reference.json` sha256 `4c89c709…`），
  **实测该文件不含 48 桶均值表** ⇒ 驱动从 FY store 的声明 train 年份重拟合并核对
  `kind`/`training_years`/`n_selected_steps`（7,304）。六份 checkpoint 的 sha256 与
  `data_identity` 全部入协议。
- **D3（评分）已完成（本窗口）**：`outputs/r7_s4_fy_2016_20261026_attempt01/reading01/readings.json`
  （sha256 见证据页），1,451 例、全部 5 lead 共享同一病例集、四季齐全、`non_finite_paths = 0`。
  证据页 `docs/R7_S4_FY_2016_CONFIRMATION_RESULT.md`，index record
  `s4-fy-2016-first-confirmation`（`outcome_class = unresolved`、`candidate_state = not-candidate`），
  账本行 2.0572 GPU-h（累计 **249.413841**），master §8 已加进度块，campaign-state 已更新。
  **读数**：t2m 相对同数据 incumbent 的 MSE 变化 **−1.6%/−1.5%/−2.7%/−4.3%/−6.4%**（candidate 全部更优）；
  candidate 的 t2m skill 0.703/0.533/0.551/0.233/0.038（全 >0），incumbent 72 h 为 **−0.027**。
  **判决未宣**（如实记录、不放宽）：① primary 的估计量歧义（协议写「分组均值之比」、冻结工具算
  「per-case 均值」，48/72 h 符号相反）需先预注册；② 守门按字面未达（同时区间上界 ≤0 只有 6/15）。
  驱动 `scripts/study_r7_s4_fy_2016_confirmation.py`（CI-safe 测试 14 项全过）；`r7_round.py verify` 三校验器全过。
- **Track 1 的编排已启动**：`.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts`
- **D4 family 的表达力（实测限制，已写进协议与读数）**：
  `training/r7_simultaneous_stats.simultaneous_paired_interval` 要求同一 family 的
  **所有 cell 共享同一组 case key**，而四季组是对 2016 病例的**划分** ⇒
  「总集 + 各年 + 四季组同处一个 family」**不可表达**；family 因此按**分组**实现
  （每组 5 个 lead），并在读数里逐字记录。
- **本窗口仍未做**：Track 1 五轮（ADR / brief / 驱动 / 协议 / 运行）、预算决策、
  R1 的 override ADR、四件套登记 —— 均未执行。

- **Track 1 的两个前置决策已写好**（本窗口，供 R1 直接引用，**不要再重复写**）：
  - **决策 0060**（`docs/decisions/0060-matched-exposure-budget-correction.md`）：预算更正，
    `per_seed 129,600 s / planned 259,200 / hard 518,400`（1:2:4），只改时间不动科学门，
    执行时按 R1 首次实测重定（另立更正决策，不在运行中改硬限）。
  - **决策 0061**（`docs/decisions/0061-r1-year-coverage-override.md`）：R1 的 override ADR，
    按决策 0057 成文定义如实写「只改算力不满足不是同一族」⇒ 用户具名授权推翻
    `docs/goals/s3-year-coverage.md` 的「不调 2400 updates」；负面后果与先例范围限定都在里面。
  - ⇒ R1 的 brief 只需**引用** 0060/0061，不必再各写一份；`round-spec` 的
    `planned_seconds`/`hard_cap_seconds` 取 0060 的值。
- **Track 1 的编排已启动**：`.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts`
  编译干净并已在后台运行，run id **`dwfrun-442a17eb-ba11-4af9-9b56-6460f32a1e0b`**
  （`maxRounds=5`、`dryRun=false`、`resumeFromRound=0`）。它是**有界、可恢复的批量编排器**，
  不是守护进程：进程退出即 `interrupted`，按 `ResumeWorkflowRun` 续；
  已知边界见 §4.0 四条（后台运行时轮末 goal 校验可能不触发＝`defer` 不是失败；
  卡片可见性只从运行 API 观测；探针未覆盖 GPU 轮/freeze/commit/登记）。
- **未做**：Track 1 五轮的 brief / 驱动 / 协议 / 运行 / 四件套登记；R1 的 brief 与驱动 —— 均未执行。
- **下一动作（可执行的第一条命令）**：
  `.venv/bin/python tools/r7_round.py state`；
  随后**先预注册 S4 primary 的估计量**（「分组均值之比」还是「per-case 均值」）与守门的读法
  （守门字面未达是否算未过），再决定 S4 是否宣布判决；**不**在没有预注册的情况下挑一个估计量报通过。
  Track 1 的 R1–R5 已由 `.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts`
  （`maxRounds=5`、`dryRun=false`）编排，预算/override 已由决策 0060/0061 给出。
- **为何停 + 恢复**：本窗口停下是因为**本轮预算用尽**，不是触到保留授权边界、也不是达科学停止条件；
  剩余工作全部**可安全独立推进**，只是没做完。
  恢复第一条命令＝ `.venv/bin/python tools/r7_round.py state`。

### 2026-10-26 — R1（Track 1 第 1 轮）实现完成、GPU 未开跑

- **本轮 spec 落在本文件顶部**：`round-node: S3` + 一行 `round-spec`；
  `campaign-state.current_round_goal` 已指向本文件。**R1 的人读 brief** 是
  `docs/goals/s3-year-coverage-matched-reeval.md`（只引用 0060/0061，不重写 ADR）。
- **顺带修掉两处会让本轮无法开工的既有缺陷**（如实记录，不是本轮的科学内容）：
  ① `campaign-state` 块被上一窗口写成了**多行 JSON**，而 `tools/r7_round_support.campaign_state()`
  的正则**不带 `re.S`** ⇒ `r7_round.py state / round-guard / freeze` 全部报
  「expected exactly one campaign-state block, found 0」（`tools/check_campaign_state.py` 带 `re.S`，
  所以 `verify` 当时是绿的、掩盖了这个不一致）；已改回单行。
  ② 本文件 §4.0 正文里那处把注释语法写成**字面示例**的地方，会被
  `ROUND_SPEC_RE` 当成第二个 spec 块（`round_spec()` 拒绝多块）⇒ 已改成不含注释语法的措辞。
- **交付（CPU）**：驱动 `scripts/study_r7_s3_fy_year_coverage_matched.py`（578 行，R-051 通过）
  + 纯数据 support 模块 `scripts/r7_s3_fy_year_coverage_matched_support.py`（56 行）
  + CI-safe 测试 `tests/test_r7_s3_fy_year_coverage_matched.py`（34 项全过，不依赖 `outputs/`，
  含 batch-2 的 CPU smoke 与反证）+ 本轮 ADR **决策 0062**（不重新授权，只写执行设计）。
- **唯一改动因子**：年份 5 → 22（月份固定 12）；相对已关闭那一轮**只改算力**
  （updates 2,400 → 16,071、batch 1 → 2）。出口 (a)–(d) 逐字沿用 `docs/goals/s3-year-coverage.md:68-79`。
- **未做**：`freeze` / `stage prepare` / `stage archive` / `stage-seeds` / `reading` / 四件套登记 —— 全部未执行；
  **本轮 0 GPU-h、未读 test、无科学结论**。
- ~~**下一动作**（已执行）~~：`round-guard` → 重建 code zip → `freeze` → `stage
  prepare/archive/seed41/seed42/seed43` 均已在 2026-10-26…28 期间执行（见下一条进度块）。
  登记提交前必须复算 `docs/rules/size-thresholds.md` 的 `<!-- measured: ... -->` 标记
  （`test_size_report_counts_match_the_checker` 按提交集合核对）。

### 2026-10-28 — R1 GPU 轮：seed41/42 完成、seed43 因清单口径漂移失败并经归档重放补跑（本窗口）

**实测结果（三个 seed 的 `stage_ledger.jsonl`）**：seed41 `success` 89,519.6 s、
seed42 `success` 88,362.4 s、seed43 `failed` **20.76 s**（`recorded_at` 2026-10-28T04:09:42+00:00）。

**失败根因（机械，不是科学）**：`validate_protocol`
（`scripts/study_r7_s3_fy_year_coverage_matched.py:275-280`）断言
`set(execution_files_sha256) == set(execution_files())`，而 `execution_files()` 经
`wide.execution_files()` 落到 `scripts/study_r7_s3_v3_rollout_ft_worker.py:31-37`，
它对 `training/`、`model/`、`data/` 做**活体 `rglob("*.py")`**。冻结时 **195** 个文件，现在是 **197**：

```
frozen entries: 195 / missing: [] / content-changed: 0
ONLY NOW: ['data/download/urban_static_sources.py', 'data/preprocess/urban_static_aggregate.py']
ONLY FROZEN: []
```

这两个文件是**另一条研究线**（城市静态协变量轮，决策 0063/0064）在 commit `7065f19`
（2026-10-27 14:37:32 +0800）引入的；`grep` 实测**没有任何冻结清单内的文件 import 它们**。
时间线解释了为什么只有 seed43 撞上：seed41（2026-10-26T02:45Z）与 seed42（2026-10-27T03:36Z）
通过校验都**早于** `7065f19` 落地（06:37Z）。⇒ **本轮没有任何科学内容变化。**

**补跑方式 = 决策 0065：用归档 `code.zip` 在冻结的绝对路径下重放 seed43。**
不改 `validate_protocol`（AGENTS.md 禁止改比对逻辑）、不重新冻结（`protocol_sha256` /
`code_zip_sha256` 都在 0045 白名单里 ⇒ 会逼三个 seed 全重训，约 74 GPU-h）。
实现用**非特权 user+mount namespace** 把归档树 bind 到仓库真实路径，再把真实 `outputs/` 与
`.venv/` bind 回去（只读，不移动/删除/修改任何真实文件）。实测证据：

```
# 归档完备性
frozen entries 195 / missing from zip 1 / hash mismatches 0
（唯一缺的是 outputs/.../protocol.json，归档器按设计不含 outputs/）
# 重放下 validate_protocol
execution_files(): 195
VALIDATE_PROTOCOL: PASS protocol 15a92e3230c5   # 与 prepare 记录的同一 digest
```

**seed43 已启动并确认在训（实测）**：`stage-plan seed 43` → `skip: false`；
worker pid 3557334 存活；`training/control/update_0000020.pt` 已写出；
`fine_tune_started.json` 的 `contract.data_identity = 71bb33e01a9d44024be9895c`
**等于**协议里 control 臂的 `train_data_identity`；`batch_size = 2`；`scientific_claim: false`。
与 seed41 的 contract 逐字段 diff，**只有 seed 专属字段不同**
（`seed`、`initialization/parent_checkpoint`、`initial_weights_sha256`、`output_dir`）。
速率实测 **2.554 s/update**（update 40 @ 102.2 s）⇒ 每臂约 **11.4 h**、两臂约 **22.8 h**，
在 36 h 的 per-seed 上限内；冻结 `deadline_perf_counter` 尚余约 **86 h**。

**纪律留痕**：本条**不是**"无异常完成"，而是"外部提交造成的清单口径漂移 + 归档重放补跑"，
证据页与 index record 必须照此写明。**教训**（R2–R5 采用，R1 不改）：`execution_files()`
对三个包目录做活体 rglob 是过宽的身份表面，后续轮应改为**显式文件清单**。

**本窗口仍未做**：seed43 读数（约 23 h 后）、R1 的 `reading` 阶段与四件套登记、R2–R5。

- **下一动作（可执行的第一条命令）**：等 seed43 拿到 receipt（约 23 h，实测速率 2.554 s/update），
  然后 `.venv/bin/python tools/r7_round.py stage reading`；随后
  `.venv/bin/python tools/r7_round.py verify` 与四件套登记（证据页 / index record / 账本行 /
  campaign-state）。**seed43 的补跑必须在归档重放的 namespace 内执行**（决策 0065）——
  真实工作树里那 2 个无关模块会让 `validate_protocol` 再次失败；任何新的 commit 往
  `training/ model/ data/` 加 `.py` 也会让它再次失败，所以这条窗口有时限。
