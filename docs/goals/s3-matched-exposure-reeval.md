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
  **判决（决策 0066，2026-10-28）**：① primary 的估计量裁定为**分组均值之比**（协议写的就是它、
  「per-case 均值」，48/72 h 符号相反）需先预注册；② 守门按字面**未达**，且**比初版记录更差**：
  全 6 族 90 格中点估计只有 78/90 ≤0（**12 个 >0**，集中在 DJF/JJA/SON 的 u10/v10）、
  区间上界只有 25/90 ≤0；另「per seed ≤ 0」子句因读数只存 seed 平均序列而**未测**。
  **2026-10-28 更正**：初版只统计了 total set 一族（15 格）却写成"全部"、且计数笔误，
  已在证据页 §5.1 留档（`readings.json` 未变，只是对它的读法写错）。
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
- ~~**下一动作**（已执行，2026-10-28）~~：S4 的估计量与守门读法**已由决策 0066 预注册并裁定** ——
  primary = **分组均值之比**（`training/r7_climatology_skill.py:120-126`），守门是**独立合取项**且
  按协议字面**未达** ⇒ 依 `R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:71` 记为
  「t2m 有获益但**未达终极 package 门**」。不再以"估计量歧义"为由悬置；**不**重算 primary。
  Track 1 的 R1–R5 由 `.zcode/workflow-drafts/r7-matched-exposure-round.dwf.ts` 编排，
  预算/override 由决策 0060/0061 给出，seed43 的补跑方式由决策 0065 给出。
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

### 2026-10-28 — S4 判决已定（决策 0066）+ seed43 重放中（本窗口收尾）

- **S4 判决已判定**（决策 0066）：primary 估计量 = **分组均值之比**；守门是独立合取项且
  **未达** ⇒ **「t2m 有真实获益，但未达终极 package 门」**。`outcome_class` 由 `unresolved`
  改为 `mixed`。同时更正了证据页上**自己**的一处口径错误（只统计 total set 15 格却写成"全部"，
  实为 90 格：上界 25/90、点估计 78/90 其中 12 个 >0），并登记「per seed ≤0」子句因只存
  seed 平均而**未测**。`readings.json` 未变。
- **seed43 重放中**（决策 0065）：实测 update 660/16,071、**2.357 s/update** ⇒ control 臂约
  10.1 h、两臂约 20 h 后完成。`fine_tune_started.json` 的 `data_identity` 与协议 control 臂一致；
  与 seed41 的 contract 逐字段 diff 仅 seed 专属字段不同。worker pid 3557334 存活。
- **R1 的 34 项测试全过**（`pytest tests/test_r7_s3_fy_year_coverage_matched.py -q` → `34 passed`）；
  `r7_round.py verify` 三校验器全过；`check_conventions.py` 37 条阻断规则 0 失败。
- **本窗口 0 新增 GPU-h 的登记**：seed43 仍在跑，账本行与 index record 要等它的 receipt，
  **不能**用估计值预填。

- **为何停 + 恢复第一条命令**：停在这里**不是**触到保留授权边界、也**不是**达科学停止条件 ——
  是本轮时长预算用尽，而**唯一的剩余前置**是 seed43 的 GPU 时间（约 20 h）。
  它已在后台运行，**没有**可安全独立推进的并行工作（R2 的驱动会在 `training/ model/ data/`
  之外新增文件，但那不会影响正在跑的 namespace 内副本；然而 R2 的**开轮**依赖 R1 的读数先登记，
  按 §4.2「按序跑」不能越过）。恢复第一条命令：
  `.venv/bin/python tools/r7_round.py state`（然后查 seed43 receipt 是否已落盘；
  **若要补跑必须在决策 0065 的 namespace 重放内执行**）。

### 2026-10-28（续）— seed43 训练期间的对表审计（只读，无 GPU 新增）

在等 seed43 的约 19 h 里，对**已经存在**的交付物逐条对表，全部为**只读核查**：

- **R1 的 brief 已补齐到 §0–§8**（原先只有 §0–§6，且 §6 已过时 —— 它还写着
  `freeze`/各 stage "均未执行"）。现在逐条记入实测：seed41 **89,519.6 s**、seed42
  **88,362.4 s** 均 success；四份 receipt 的 `updates = 16,071`、`batch_size = 2`、
  `train_data_identity` = control `71bb33e01a9d…` / treatment `81307008b572…`，与冻结协议一致。
  `check_goal_brief.py --brief` 仍 `failures=0`。
- **冻结协议自洽性（实测）**：`protocol_sha256` 用 `canonical_digest` **重算一致**
  （`15a92e3230c5…`）；`code_zip_sha256` 与 `code.zip` 的实际 sha256 **逐位相同**
  （`6a85ac39…`）。
- **`reading` 阶段的输入齐备性（实测）**：`_readings` 需要的 3 seed × 2 臂 receipt 与
  5 lead × 2 臂的 `boundary_rmse.csv` 中，**只缺 seed43 的两份 receipt**，其余全在。
  `stage-plan reading` 返回 `skip: false, already_running: false`。
- **D3 的读数组成完整性（实测）**：17 变量 × 5 lead × 3 区域 × 6 分组 × 2 臂 = **3,060 格**
  全部存在，无缺格。
- **`size-thresholds.md` 的 `<!-- measured: ... -->` 标记当前与 checker 一致**
  （`test_size_report_counts_match_the_checker` → `1 passed`），登记时不会因它变红。
- **workflow 脚本的形状要求已逐条核对**：`maxRounds` 硬上限 5（`Math.min(..., 5)`）、
  stage 超时 **129,600,000 ms**、`push-round` 只推工作分支、登记后跑 `verify`；
  `commit-guard` 语义由 `round-guard` 覆盖（脚本内已写明理由）。
- **seed43 的时限核算（实测速率 2.358 s/update）**：control 臂余 **9.47 h**、两臂余 **18.93 h**，
  而 per-seed 上限 36 h、round 剩余 **85.14 h** ⇒ **两边都够**。

**本段 0 GPU-h、未读 test、无科学结论**；seed43 仍在前述 namespace 重放内运行。

### 2026-10-28（续 2）— R2 的只读前置核实（为队列不空转）

seed43 仍约 19 h，`reading` 无法在本窗口执行。利用等待期把**R2 的前置**逐条实测，
全部**只读**、**0 GPU-h**、不改任何已冻结产物：

- **R2 的锚表（R2 唯一要新建的产物，已存在）**：`outputs/r7_s3_fy_anchor_table_20261025_attempt01/`
  的 `BUILD_COMPLETE`（`table_sha256 18448ae3…`）、`anchor_spec.json`（18 键，等于
  `model/climatology_anchor_r7.py::_SPEC_KEYS`）、`anchor_table.pt` 三者**互相一致**；
  把 `anchor_table.pt` 的**原始张量字节**重新 sha256，得到 `18448ae3…`，与声明**逐位相同**。
  形状 `[48, 17, 65, 65]`、`n_selected_steps = 32,144`、48 个桶的计数非零。
  `_validate_spec(spec)` **PASS**（说明模型加载器会接受它）。
- **发现并解掉一个会烧掉 GPU 时间的真问题**：`model` 加了锚之后**多出两个 buffer**
  （`backbone.climatology_anchor.table` / `.ready`），而**父检查点没有这两个键** ⇒
  照 R1 的 `load_state_dict(..., strict=True)` 直接加载**必然失败**（实测报
  `climatology load requires both table and ready buffers`）。
  仓库里已有正确解法：`scripts/r7_s3_anchor_isolation_parent.py::expand_parent_state`
  —— 先 `install(表)` 再把两个新 buffer 补进 state。**用 R2 的 48 桶表 + 种子 43 的父**实测：
  缺失键**恰好**等于 `ANCHOR_KEYS`、父里**没有多余键**、补齐后 `strict=True` **PASS**。
- **四条恒等反证（R2 的正/负控制）实测全过**：`bit_identical_outside_anchor = true`
  （127 个共享键逐位相同）、`unanchored_max_abs = 0.0`、`feedback_is_read = true`
  （`feedback_delta_max_abs = 4.07e-4` ≠ 0）、`zeroed_table_delta_max_abs = 0.0`。
  这正是 R2 brief §3 要求的 ①②③④，说明「只差一个布尔」这条单因子前提在实现上成立。
- **R2 的 batch 阻塞项已不再阻塞**：decision 0058 §7 把「batch 取值」列为唯一待定项，
  而本轮的目标长文已把 batch 定为 **2**（`updates = 16,071`）。
- **仍未做（且不在本窗口做）**：R2 的驱动、测试、`protocol.json` 冻结、GPU 运行与四件套登记 ——
  按 §4.2「按序跑」，R2 **必须等 R1 的读数登记之后**才开轮。

### 2026-10-28（续 3）— seed43 在跑期间的只读复核（按要求逐项）

- **`stage-plan seed 43`**：`already_running = true`、`running_pids = [3557334]`、
  `already_done = false`、`skip = false` ⇒ 确认是新起的进程在跑，不是待修的失败。
- **逐字节核对冻结清单**（`protocol.json` 的 `execution_files_sha256`，全部 195 项，
  用 `hashlib.sha256` 读当前工作树比对）：**missing = 0、changed = 0**、
  即 **195/195 逐位一致**。
- **集合层面**（`execution_files()` 现算 vs 冻结）：frozen **195**、current **197**，
  **新增 2、删除 0**，新增项为 `data/download/urban_static_sources.py` 与
  `data/preprocess/urban_static_aggregate.py`（城市静态协变量轮 `7065f19` 引入，
  与 R1 无关、无冻结文件 import 它们）。这正是 12:09 UTC 那次 exit 1 的**唯一**原因。
  ⇒ 不变，但**不重启 seed43、不改冻结清单**（决策 0065：走归档重放）。
- **`stage_ledger.jsonl` 五条（原样）**：prepare `success` 632.02 s；archive `success` 5.82 s；
  seed41 `success` 89,519.56 s；seed42 `success` 88,362.38 s；
  **seed43 `failed` exit 1、20.76 s @ 2026-10-28T04:09:42Z**（旧尝试，已被取代，
  须原样写进结果页的 limitations 与成本表）。
- **seed43 当前实测进度**：control 臂 update **2,080 / 16,071**、**2.354 s/update**、
  进程存活（pid 3557334, 状态 `Sl`）；control 臂约 **9.1 h**、两臂约 **18.3 h**。
- **决策 0067 已写**（proposed）：执行文件清单不得对 `training/model/data` 做活体 `rglob`。
  依据是本轮的实测结构 —— 195 项里 **182 项（93%）来自 glob**，仅 13 项具名。

### 2026-10-28（续 4）— 证据页写作前的两处前提更正（实测，防把错数字写进结果页）

在准备结果页时，对任务书里引用的两个数字做了核对，**两个都与仓库实测不符**，如实记录：

**① `+0.0258` 在仓库里找不到。** 全库 `grep`（`docs/` + 该 attempt 的 json + 证据索引
metrics）**没有任何** `0.0258`。FY 2400 轮**控制臂**（4 个月 `control_h72` 清单）的 72 h t2m
skill 实测是三 seed **−0.018861 / −0.001317 / +0.021495**（见
`docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md:54-58` 与 index record `control_skill_72h`），
**不是** +0.0258。最接近的量是 `delta_6h` 的 seed41 = **−0.025918**，那是**另一个量**、
另一个 lead、另一个符号。⇒ 结果页**不写** +0.0258。

**② `+0.1527` 属于本轮的 NEW 控制臂，不属于 FY 2400 控制臂。**
实测（从 `seed41_control_receipt.json` 的 72 h `boundary_rmse.csv` 与本轮 48 桶分母
`3.926891` 反算）：本轮控制臂 seed41 72 h = **+0.152708**、seed42 = **+0.162656**
（seed43 待出）。而 FY 2400 控制臂同一 lead 是 **−0.018861 / −0.001317 / +0.021495**。
两者是**不同的臂**（不同清单：4 个月 vs 2017–2021 五年；不同 updates：2,400 vs 16,071），
**不能**把它们写成同一条"算力增益"曲线的两端而不加说明。

**因此结果页的"算力增益对照"必须照这个形状写**（两列并列、明确标注是两条不同的臂）：

| 臂 | train 清单 | updates | seed41 72 h skill | seed42 | seed43 |
| --- | --- | --- | --- | --- | --- |
| FY 轮的 4 个月控制臂 | `control_h72`（2,140 窗） | 2,400 | **−0.018861** | **−0.001317** | **+0.021495** |
| 本轮的 5 年控制臂 | `manifests_control`（7,291 窗） | 16,071 | **+0.152708** | **+0.162656** | 待出 |
| 本轮的 22 年候选臂 | `manifests_treatment`（32,131 窗） | 16,071 | **+0.179874** | **+0.194920** | 待出 |

（上表后两行的数字是**实测**，从已落盘 receipt 的 `boundary_rmse.csv` 反算，命令与中间量已在本条记录。
候选 vs 控制的"算力增益"**不是**本轮的科学量 —— 本轮的预声明出口是 (a)–(d)，见 §4.2 与 R1 brief。）

**③ epoch 混杂已精确复算（实测）**：`updates × batch = 16,071 × 2 = 32,142`；
候选臂可用窗 **32,131** ⇒ **1.00 epoch**；控制臂可用窗 **7,291** ⇒ **4.41 epoch**。
这正是"同一算力、不同暴露"的固有权衡，必须写进结果页的 limitations。

### 2026-10-28（续 5）— 本轮收尾：停在这里的原因与恢复链路

- **已完成（本窗口）**：① 按要求做的只读复核（`already_running=true` + pid、
  195/195 逐位一致、集合差 2 新增 0 删除）；② **决策 0067**（执行文件清单缺陷，proposed）；
  ③ 两处**前提更正**（`+0.0258` 在库中不存在；`+0.1527` 属于本轮新控制臂而非 FY 2400 控制臂）；
  ④ 结果页 `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md` 的**已实测部分**
  （协议/归档 digest、stage_ledger 五条成本表、epoch 混杂 1.00 vs 4.41、bridge (e) 参考值与带宽、
  候选 seed41 的 17 变量 × 5 lead 表）；未定格全部标 `待 seed43` / `待 readings.json`，**未填估计值**。
- **未做（全部因同一个前置）**：`stage reading`、出口 (a)–(d) 判定、四件套登记、
  以及任务书的第四步（切到静态协变量轮 + 把 glob 缺陷机械化）。
  前置 = **seed43 的 receipt**，实测 update **2,500 / 16,071**、**2.353 s/update**、
  进程存活（pid 3557334）、两臂约 **17.7 h**。
- **为何停**：不是触保留授权边界、也不是达科学停止条件 —— 是**本轮时长用尽**，
  而剩余工作**全部**等同一个 GPU 结果；R2 按 §4.2「按序跑」必须在 R1 登记之后才能开轮，
  静态协变量轮同理。**没有**可安全独立推进的下一步（再做只会重复已完成的核对）。
- **恢复链路（按序，第一条命令在最前）**：
  1. `.venv/bin/python tools/r7_round.py state` —— 确认
     `outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/seed43_receipt.json` 是否已落盘；
  2. **若未落盘且无 worker 存活**：必须在**决策 0065 的 namespace 重放内**补跑
     （真实工作树里那 2 个无关模块会让 `validate_protocol` 再次失败）；
     查存活用 `.venv/bin/python tools/r7_round.py stage-plan seed 43`；
  3. `.venv/bin/python tools/r7_round.py stage reading` —— 同样需在重放内；
  4. 读 `readings.json` 的 `exit_conditions` 与 `bridge_e`，**只按它填**出口表与桥表，
     不得手算、不得跨 attempt 混算；任一 (a)–(d) 假 ⇒ 按预声明**永久关闭**「训练年份覆盖」支；
  5. `.venv/bin/python tools/r7_round.py round-guard` → `verify`；
  6. 四件套：index record、账本行（先 `ledger-plan <新 GPU-h>` 算）、campaign §8、campaign-state；
     **注意** 结果页与四件套一旦进入跟踪集合会改变
     `docs/rules/size-thresholds.md` 的 `<!-- measured: ... -->`，登记前必须复算；
  7. 最后才把 `campaign-state.current_round_goal` 指向静态协变量轮。

### 2026-10-28（续 6）— 用两个已完成 seed 把结果页填到只剩 seed43

在等 seed43 的时间里，把结果页从"骨架"推进到"只差一列"，全部用**实测**：

- **用官方读取器交叉验证**：改用 `wide._region_variable_rmse`（`_readings` 自己用的函数）
  从各 receipt 的 `boundary_rmse.csv` 读 t2m RMSE，再用本轮的 48 桶分母换算 skill。
  结果与我先前的独立反算**逐位一致**（seed41 control 72 h `+0.152708` 等），
  说明结果页 §5 的数字不是手算走偏。
- **出口 (a)(b)(c) 在 seed41/42 上已 2/2 为真**（实测）：
  (a) 72 h skill **+0.179874 / +0.194920**；(b) 72 h F−N ΔRMSE **−0.058419 / −0.069908**；
  (c) 48 h skill **+0.356924 / +0.361391**。
- **(d) 是唯一有风险的一条，已把算术摆出来**：6 h 两个 seed 的 F−N 都是**正**的
  （**+0.006961 / +0.013395**），12 h 也是（**+0.004625 / +0.022064**）；
  要让三 seed 均值 ≤0，seed43 必须在 6 h ≤ **−0.020356**、12 h ≤ **−0.026689**（24 h ≤ **+0.003425**）。
  **⚠ 这三个阈值于 2026-10-28 更正**：原文写的是 −0.030534 / −0.040034 / +0.005137，
  错在用 `−3 × 两 seed 均值` 代替 `−(d₄₁ + d₄₂)`；偏差恰好等于两 seed 均值本身。
  24 h 的错向是**放宽**（会夸大 seed43 的余量）。已在结果页 §2.1 用驱动自己的
  `evaluate_exit` 逐 1e-9 验证正确阈值并留档。**判据实现从未出错，错的只是这张说明表。**
  ⇒ **决策 0061 的可证伪预测（「补上 ≥1 epoch 后 (d) 翻转」）在 6/12 h 上尚未显示翻转**：
  已到手的两个 seed 都**没有**让 F 在短 lead 上优于 N。**但这不是判决** ——
  seed43 完全可能落进那个区间，判决仍只认 `readings.json` 的 `evaluate_exit`。
- **17 变量 × 5 lead 表**已扩成逐 seed 三列，seed41/42 填满、seed43 留 `待 seed43`（85 行）。
- **本段 0 GPU-h、未读 test、无判决**。seed43 实测 update **2,740 / 16,071**、
  **2.354 s/update**、进程存活（pid 3557334，经历过一次 `Dl` 磁盘等待后恢复为 `Rl`），
  两臂约 **17.5 h**。

### 2026-10-28（续 7）— 登记用的成本算术（先算好，登记时直接用）

**已实测的 GPU 时间**（`stage_ledger.jsonl` 逐条秒数，只算 seed 阶段）：

| 阶段 | 秒 | 折算 |
| --- | --- | --- |
| seed41（success） | 89,519.564216 | 24.866546 GPU-h |
| seed42（success） | 88,362.383506 | 24.545107 GPU-h |
| seed43（**failed** exit 1） | 20.757882 | 0.005766 GPU-h |
| **小计（已发生）** | **177,902.705604** | **49.417418 GPU-h** |

- prepare 632.015875 s + archive 5.818272 s = 637.834147 s = **0.177176 h**，是 **CPU**
  （prepare 自己声明"metadata prepared in bounded CPU worker"），**不计入 GPU-h**，
  但在成本表里单列。
- **待落盘**：seed43 重放的两臂（control + treatment）。**不得**用估计值预填账本行；
  等 receipt 到齐后按同一条 `stage_ledger.jsonl` 的真实 `elapsed_seconds` 相加。
- **登记时的算式**（`ledger-plan` 已实测可用）：
  `new_gpu_h = 49.417418 + seed43_replay 的实测小时`；
  当前账本 `previous_total_gpu_h = 249.413841`（73 行）、`cap_gpu_h = 20.0`、
  `remaining_gpu_h = −229.413841`（**cap/remaining 仅会计，无总上限**）。

### 2026-10-28（续 8）— `stage reading` 的读前预检（只读，防止 17 h 后才发现缺件）

在等 seed43 的时间里，把 `_readings()` **实际会读的每一项**都静态核了一遍，
避免 GPU 时间花完才在 `reading` 阶段崩掉：

- **`_readings` 依赖清单（逐项实测存在）**：`body['protocol_sha256']`、
  `body['factor']['name']`、`body['climatology_reference']['leads'][lead]['t2m_rmse']`（5 个 lead）、
  `body['bridge_e']['reference']`、`D2_REFERENCE` 文件、`SEEDS`/`ARMS`/`REGION='full'`；
  以及每个 seed×两臂 receipt 的 `train_data_identity` 与 5 个 lead 的 `boundary_rmse.csv`。
  ⇒ **唯一缺件是 seed43 的两份 receipt**（预期，仍在跑）。**PASS**。
- **`evaluate_exit` 在已完成的两个 seed 上的实际输出**（用驱动自己的函数，
  `seeds_order=(41,42)`，输入由官方 `_region_variable_rmse` 读取）：
  `a=true, b=true, c=true, d=false`，`seed_mean_delta_6_12_24 = {6: +0.010178, 12: +0.013345, 24: −0.001712}`。
  这三个均值与我先前独立反算的 **+0.010178 / +0.013345 / −0.001712 逐位一致**
  ⇒ 我的读法与驱动的判据实现**同源**，不是"自己算给自己看"。
- **`bridge_verdict` 的接线也验过**（喂 41/42 真实控制臂 RMSE）：它能正常产出
  `control_mean / reference_mean / difference / band / passed`；它此刻显示不通过是**正常的**——
  seed43 位置喂的是占位值，**不得**把这次调用结果当桥的结论。
- **这两个函数在占位输入下会给出 `a=b=c=d=false` 与 `verdict=close-...`**：
  这提醒 `reading` 之后必须确认 seed43 的 receipt 真的落盘、且三个 seed 都在，
  再读 `exit_conditions`；**不得**在缺件状态下把 `false` 当判决。
- **本段 0 GPU-h、未写 attempt 目录、未读 test、无判决**。

### 2026-10-28（续 9）— 重放机制从会话 `/tmp` 固化为仓库工具（本窗口唯一新增代码）

**发现的问题（自查）**：seed43 正在决策 0065 的 namespace 重放内训练，但那个重放是用
**一次性的 `/tmp` 手工命令**搭起来的，没有留下可复现的脚本 —— 我这一窗口重建它时
只能从 `/proc/3557334/mountinfo` 反向读出当时的挂载布局
（`/tmp/k_o`、`/tmp/k_v` + 归档树 bind 到 `/data/esw/UrbanPiDiT_R2`）。
这意味着：**干净克隆无法复现本轮的补救手段**，违反 R-035/R-037 的"新克隆是否需要它来理解/复现"判据。
而且紧随其后的 `stage reading`（约 17 h 后）**必须在同一 overlay 内**才能通过
`validate_protocol`，否则会在挂载布局上手写出错。

**处置**：新增 `tools/r7_archived_replay.py`（220 行，R-051 通过、目标带内不增 R-019b 命中）、
配套反证 `tests/test_r7_archived_replay.py`（117 行、9 项全过）。三个子命令：

```
.venv/bin/python tools/r7_archived_replay.py prepare --code-zip outputs/<round>/code.zip
.venv/bin/python tools/r7_archived_replay.py verify  --out outputs/<round>
.venv/bin/python tools/r7_archived_replay.py run -- .venv/bin/python tools/r7_round.py stage reading
```

**实测（本窗口，全部只读或全新 `/tmp` 树）**：

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 端到端重放验证 | `verify --tree /tmp/r7_archived_replay/tree --out outputs/r7_s3_..._attempt01` | `execution_files(): 195` + `VALIDATE_PROTOCOL: PASS 15a92e3230c5` |
| 拒绝覆盖在跑的树 | `prepare --tree /tmp/r7_r1_replay_probe/root` | exit **3**，报 `mounted by live pid(s) [3557323, 3557331, 3557334]` |
| `reading` 阶段计划 | `run -- ... stage-plan reading` | `already_done: false`、`already_running: false`、`skip: false` |
| 反证 | `pytest tests/test_r7_archived_replay.py -q` | **9 passed**（含"捕获移到 overlay 之后必须报错"） |
| 不污染活清单 | `execution_files()` 在真实工作树 | **197**，且 `tools/`、`tests/` **不在**清单内 ⇒ 新文件不改变任何在跑轮次的身份 |

**纪律**：本段**未**改 `validate_protocol`、**未**动任何冻结产物、**未**重新冻结、
**未**触碰在跑的 seed43（pid 3557334，`tree_in_use` 的拒绝就是证据）。
决策 0065 已补 Decision 第 6 条登记这个工具与其三个保护。

**未做**：seed43 读数（实测 3240/16071、2.354 s/update，两臂尚需约 17 h）、
R1 的 `reading` 与四件套登记、R2–R5。

- **下一动作（可执行的第一条命令）**：等 `seed43_treatment_receipt.json` 落盘后
  （约 17 h，实测 2.354 s/update），执行

  ```
  .venv/bin/python tools/r7_archived_replay.py run -- .venv/bin/python tools/r7_round.py stage reading
  ```

  再 `.venv/bin/python tools/r7_round.py round-guard` 与 `verify`，然后按 `readings.json`
  的 `evaluate_exit` 输出填四个出口与 `bridge_e`，完成四件套登记。

### 2026-10-28（续 10）— `round-guard` 查出一条我上一窗口写下的红测试，已修并留反证

**这条失败不是 seed43 造成的，也不是本窗口改动造成的**：它自证据页创建那次提交
（`e52ab29`）起就一直是红的，只是此前没人跑 `round-guard`。

**缺陷**：`tests/test_r7_s3_fy_year_coverage_matched.py::test_the_round_spec_splits_the_two_commits`
有一条断言 `assert not (ROOT / "docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md").exists()`
——它要求证据页在工作树里**永远不存在**。这是**快照**，不是不变量，而且在**最不该翻面的时刻翻面**：
登记提交的职责正是创建该页，所以这一轮的测试套件会在登记提交上必然变红。

**实测**：`round-guard` 报 `"passed": false`、`1 failed, 153 passed`。

**修正（未削弱）**：不变量改为「每个**仅登记**产物只属于 register 提交」——
证据页、`R7_EVIDENCE_INDEX.jsonl`、`R7_CANDIDATE_BRIEF.md` 必须在 `register_paths` 里、
且不得出现在 `commit_paths` 里。**不做全集合互斥**：campaign 与 goal 两份**活文档**
本来就合法地同时出现在两次提交里（实测这两条路径的交集就来自它们）。
历史侧也直接查：按 `commit_message` 定位实现提交，`git show --name-only` 不得列出证据页。

**反证（实测）**：把证据页塞进 `commit_paths` 后断言立即失败 ⇒ 判定未被放宽。
`tests/test_r7_s3_fy_year_coverage_matched.py` **34 passed**；`round-guard` **154 passed, passed=true**。

### 2026-10-28（续 11）— 17×5 表的核对方式固化（我曾误读一次列义，已查清）

**我自己的错误（如实记录）**：核对 §6 的 17 变量 × 5 lead 表时，我先把它当作
「control/treatment 两列」去比，得到 **149 处不一致**，一度以为表是错的。
读表头才发现 §6 **只有候选 F（treatment）臂**，第 3、4 列是 **seed 41 / seed 42**。
按正确列义重比：**85/85 全部对上**（容差 1e-4）。⇒ **表是对的，我的核对读法错了**。

**处置**：新增 `scripts/read_r1_seed_rows.py`（84 行，R-051 通过）——用仓里自己的读取函数
（`study_r7_s3_wide_single_factor.py:186` 的 `_region_variable_rmse`）从某个 seed 的
逐 lead `boundary_rmse.csv` 直接打印证据页该列的行，seed43 的整列**不必手抄**。
缺 lead 或缺变量时**报错退出**（exit 2），不打印默认值 —— 未跑完的 seed 必须让该列
**可见地缺失**，绝不能用编造的数字填满。反证在 `tests/test_read_r1_seed_rows.py`
（5 项，含 region 过滤与两条拒绝路径，纯合成 fixture、CI 可用）。

**实测**：`--seed 42 --arm treatment` 的输出与证据页第 42 列逐位一致
（如 `t2m/6h = 2.2634`、`t2m/72h = 3.5235`）。

**规模计数未变**：R-019b **1131**、R-021 **129**（与 `docs/rules/size-thresholds.md`
的 `<!-- measured: ... -->` 标记一致）⇒ 本轮新增的 `tools/`、`tests/`、`scripts/` 文件
都在 400 行目标带内且无 >120 字符行，**标记无需更新**。

**本段 0 GPU-h、未读 test、未改任何冻结产物、未触碰在跑的 seed43。**

- **下一动作（不变）**：等 `seed43_treatment_receipt.json` 落盘后
  ```
  .venv/bin/python tools/r7_archived_replay.py run -- .venv/bin/python tools/r7_round.py stage reading
  ```

### 2026-10-28（续 12）— 决策 0067 机械化；本窗口收尾（停在 GPU 上，非停工）

**背景**：任务 2/3 全部阻塞在 seed43 的两份 receipt 上（还需约 18–22 h）。这段时间里
本窗口只做**不再触及 GPU 且不改变任何冻结物**的预备工作，全部有实测。

**① 决策 0067 从"只能靠人读"变成可运行实现。** 新增
`tools/r7_explicit_inventory.py`（179 行）+ `tests/test_r7_explicit_inventory.py`（161 行、12 项全过）。
实测（真实 R1 驱动）：

| 口径 | 文件数 | 说明 |
| --- | --- | --- |
| R1 冻结的 glob 清单 | **182**（来自 training/model/data）+ 13 具名 = 195 | 正是事故来源 |
| 本工具算出的 import 闭包 | **89** | 不泄漏 `tools/`、`tests/`、归档 |
| 运行时实际导入的仓内模块 | **15** | 与闭包交叉核对：**0 遗漏** |

**反证（决策 0067 第 5 条明确要求的那一条）**：在同一棵树上新增一个无关
`data/*.py` ⇒ 闭包 **不变**；同一棵树上跑 R1 的 glob 形状 ⇒ **184 → 185**。
两半都实测，因此"缺陷是真的"与"修复有效"都被证明，而不是只断言一半。
另有反向反证：新增一条真实传递 import **必须**让清单变化（否则检查是瞎的）。
两个故障方向都覆盖：漏写闭包模块（身份缺口，危险方向）与多写不可达模块。
**未做**：接进 `check_conventions.py` 作阻断规则——要先定强制点，故决策仍是 proposed。

**② 本次会话所有新增都在活清单之外。** `execution_files()` 只扫
`training/`、`model/`、`data/`；本窗口新增的 `tools/r7_archived_replay.py`、
`tools/r7_explicit_inventory.py`、`scripts/read_r1_seed_rows.py` 与三个测试文件
**都不在其中**（实测：真实工作树仍是 197，与本窗口开始时相同）⇒ **seed43 的校验不会被本窗口的提交打断**。
相反，`scripts/read_r1_seed_rows.py` 落在 `scripts/`——若它是**被 glob 覆盖**的目录就会打崩 seed43，
这正是决策 0067 要消灭的脆弱性。

**③ seed43 的存活与时限（实测）**：

- pid 3557334 存活（`Sl`），属 **`setsid` 独立 session**（SID 3557323）⇒ 会话结束**不会**杀掉它。
- 进度 **4140 / 16071** control，2.354 s/update（已用 9748.9 s）。
- **worker deadline 尚余 33.4 h**，round deadline 尚余 **83.6 h**；按两个已完成 seed 的
  实测均值（89,941 s/seed）估计，seed43 还需 **约 22.1 h** ⇒ 余量充足。

**本窗口改动汇总**（6 个提交，全部只推 `r7/weather-reasoning`）：
重放工具固化、`round-guard` 红测试修复（含反证）、17×5 读取器、登记清单、
闭包清单机械化、决策 0065/0067 补记。

**为何停在这里**：不是遇到障碍，而是**任务 2 的唯一输入（seed43 的两份 receipt）在物理上
还要约 18–22 h 才能生成**。继续"推进"只能是伪造读数或提前宣布结果，两者都被明令禁止；
提前切 `current_round_goal` 到静态协变量轮也会**破坏** `r7_round.py` 对当前轮的解析，
使待办的 `reading` 与登记无法执行。

- **恢复的第一条命令**（≈22 h 后；先确认 receipt 已落盘且无存活 worker）：
  ```
  cd /data/esw/UrbanPiDiT_R2 && ls -la outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/seed43_*_receipt.json && .venv/bin/python tools/r7_round.py stage-plan seed 43
  ```
  预期 `"already_done": true`、`"already_running": false`；随后按
  `docs/R1_REGISTRATION_CHECKLIST.md` 逐步执行（第 1 步即重放内的 `stage reading`）。

### 2026-10-28（续 13）— 登记前把证据页的每个已填数字都独立复核了一遍（查出两处真错）

seed43 仍需约 18 h（4520/16071、2.354 s/update；worker 余量 33.0 h）。等待期间做的是
**对即将登记的页面做逐项复核**，而不是继续写新内容。三项发现，前两项是我的错：

**① (d) 的 seed43 阈值算错了（已修）。** 原文写 −0.030534 / −0.040034 / +0.005137，
用的是 `−3 × 两 seed 均值`；(d) 要求的是 `d₄₁ + d₄₂ + d₄₃ ≤ 0`，正确阈值是 `−(d₄₁+d₄₂)`
= **−0.020356 / −0.026689 / +0.003425**。偏差恰好等于两 seed 均值本身（`3m − 2m = m`）。
前两个原文**更严格**，第三个（24 h）原文**更宽松** —— 会让读者以为 seed43 余量更大，
这是危险方向。用驱动自己的 `evaluate_exit` 在边界上逐 1e-9 验证（其余两 lead 设为 −10 以隔离）：
三个 lead 在正确阈值处均 `d=True`、阈值 +1e-9 均 `d=False`。
**判据实现从未出错，错的只是说明表**，已在结果页 §2.1 留档。

**② §5 缺 FY 候选臂，导致"2400 vs 16,071"的对照其实做不出来（已修）。** 原文只列了
FY **控制**臂。从已登记 index record 的 `skill_72h` 读出 FY **候选**臂实测
**+0.012121 / +0.032076 / +0.064427**（seed 均值 **+0.036208**）。补成四臂表后，
候选↔候选（清单族相同）的对照才成立。

**③ 任务书的前提经实测不成立，页面按实测写。** `+0.0258` 在全库不存在
（`docs/`、本轮 attempt 的 json、证据索引 metrics 均无）；`+0.1527` 属于**本轮 5 年控制臂**
（16,071 轮），不是 2400 轮的任何臂。FY 2400 轮候选臂 seed41 的 72 h skill 实测 **+0.012121**。
**不照抄该前提。**

**④ 其余数字全部逐位对上（实测）**：

| 页内数字 | 独立复核方式 | 结果 |
| --- | --- | --- |
| §3 五个 lead 的参考均值/带宽 | 直读 `protocol.json` 的 `bridge_e.reference` | 5/5 逐位相同 |
| §4 候选 1.00 / 控制 4.41 epoch | `updates×batch/usable_windows` | `32142/32131=1.0003`、`32142/7291=4.4084` ✓ |
| §4 窗口 32,131 / 7,291 | `train_windows.usable_windows` | 逐位相同 |
| §6 17×5 表 85 行 | `read_r1_seed_rows.py` 对 `boundary_rmse.csv` | 85/85 一致 |
| §7 195/195 冻结清单 | `hashlib` 逐字节（当场重跑） | `ok=195, missing=0, changed=0` |

**⑤ 任务 4 的切换仍被自己的 brief 门控。** `docs/goals/s3-urban-static-covariates.md`
§0 写明"**启动那一刻**才把 `campaign-state.current_round_goal` 指向本长文"，§1 记用户指定
"等 R1 跑得差不多再启动"。R1 未登记就切会让 `r7_round.py` 解析到错误轮次、
使待办的 `reading`/登记无法执行 ⇒ **不切**。该轮 brief 已 prepared，启动即可。

**本段 0 GPU-h、未读 test、未改任何冻结判据、未触碰在跑的 seed43。**

- **下一动作（不变）**：等 `seed43_treatment_receipt.json` 落盘 → 按
  `docs/R1_REGISTRATION_CHECKLIST.md` 第 0–6 步执行（第 1 步为重放内的 `stage reading`）。

### 2026-10-28（续 14）— 桥 (e) 的两个已完成 seed 已实测不通过；并预演了 `_readings` 全链路

**① `_readings` 全链路预演（结构性验证，合成数据、绝不写真实目录）。**
在 `/tmp` 建一个只含 protocol + 六份 receipt 的副本，seed43 的两份**借用 seed41 的评估目录**
（刻意的合成，仅测管道），跑官方 `_readings`：

- 三个 seed 都被正确索引（`['41','42','43']`），`exit_conditions` 与 `bridge_e` 都产出；
- `evaluate_exit` 在该合成输入上返回 `a=b=c=true, d=false`、`verdict="close-…"`；
- 副本使用后**已删除**；真实 attempt 目录**未被写入**（`readings.json` 仍不存在，实测）。

⇒ 18 h 后那一步不会因结构缺陷失败。**合成数据不产生任何读数。**

**② 桥 (e) 在 41/42 上已实测不通过（这是真实数字，不是预演）。**
控制臂 5-lead t2m RMSE seed 均值 vs 已登记 FY 参考：

| lead | 本轮 N 均值（41/42 实测） | FY 参考 | 带宽 | 差值 | 过 |
| --- | --- | --- | --- | --- | --- |
| 6 h | 2.234882 | 2.286343 | 0.114317 | −0.051461 | ✓ |
| 12 h | 2.736178 | 2.867620 | 0.143381 | −0.131442 | ✓ |
| 24 h | 2.488209 | 2.666144 | 0.133307 | −0.177935 | **✗** |
| 48 h | 3.177350 | 3.415757 | 0.170788 | −0.238407 | **✗** |
| 72 h | 3.604002 | 3.825636 | 0.191282 | −0.221634 | **✗** |

**性质（协议原文，不得改写）**：`bridge_e` 声明 `scientific: false`，其 `rule` 写
「a failure is **identity/engineering unresolved, not a verdict**」⇒ **不进入 (a)–(d)**，
不构成对年份覆盖的支持或反对。
**但它确立了一件重要的事**：本轮控制臂在 24/48/72 h 上**比 FY 参考低 0.18–0.24 K**
⇒ 任何"拿本轮数减 FY 数"的跨 attempt 算术在这些 lead 上**不成立**。
这正是任务书前提 `+0.0258→+0.1527` 所犯的错误类型。

**反解**（实测 `3(ref±band)−2×两seed均值`）：要让桥回到带宽内，seed43 的 RMSE 必须
≥ 2.6221 / 3.3802 / 3.6951 K（24/48/72 h）—— 即**比两个已完成的 seed 都差**（实测三个都成立）。
**这不是预告**：桥的结论仍只由 `readings.json` 的 `bridge_verdict` 给出。

**③ 归档树与当前工作树逐位相同（实测）。** 在依赖"重放里算的数 = 我算的数"之前先验证：
194 个冻结条目在归档树与当前工作树 `hasattr` 比对，**0 差异、0 缺失**
（`archive != frozen digest: 0`、`archive != current worktree: 0`）。
⇒ `stage reading` 在重放内会看到与我测量时**完全相同**的驱动。

**本段 0 GPU-h、未读 test、未改判据、未写真实 attempt 目录。**

- **下一动作（不变）**：等 receipt → 按 `docs/R1_REGISTRATION_CHECKLIST.md` 第 0–6 步执行。

### 2026-10-28（续 15）— 用真实校验器预演登记的两步机械约束（并固化为测试）

seed43 在 5080/16071（2.357 s/update），receipt 仍不存在的。等待期间把**登记提交会不会红**
这件事先用校验器本身试了一遍，而不是等人到现场再发现：

**① `data_identity` 是本轮的一个判断点，不是可以照抄的字段。**
校验器要求 `candidate_state == "candidate"` 时 `data_identity` **非空**，但本轮
`protocol.json` 的**顶层** `data_identity` 实测是 **`None`**（两臂各带自己的清单身份，
没有单一数据集身份）。已登记的两条 candidate 记录都来自**单臂**轮次，约定不可直接照抄。
处置：记**候选臂**（treatment）身份 `81307008…`，两臂身份都放进 `metrics`
（`train_data_identity_control 71bb33e0…` / `…_treatment 81307008…`）。

**② 用真实校验器在 `/tmp` 探针记录上验过这条约定可过，并做了两个反证**：

| 反证 | 结果 |
| --- | --- |
| `data_identity = null` | `ERROR: candidate requires data_identity`（拒绝）|
| 错误的 `evidence_sha256` | `ERROR: evidence_sha256 does not match …`（拒绝）|
| 上述约定（候选臂身份 + 双身份进 metrics）| `PASS evidence index: 93 records` |

⇒ 探针能过不是校验器宽容，是它真的在查。探针索引已删除，真实索引仍是 **92** 条。

**③ 账本行有两条硬约束，在 master plan 的临时副本上实测**：

- **index record 必须先于账本行**。先插账本行、记录尚未入索引时，校验器立即报
  `FAIL C-03 … cites unknown record 'r1-dryrun-probe'`。
- **记录的 `metrics.gpu_hours` 必须等于账本行的 GPU-h**（容差 1e-4，
  `record_gpu_hours` 只读 `metrics.gpu_hours` 或顶层 `gpu_hours`）⇒ 数字必须写进记录本身。
- **C-02 的累计/余量算术在预演中自动成立**，不需要人工核对别的字段。

两条都已固化为测试（`tests/test_check_campaign_state.py` 新增 2 项，62 项全过），
并对第一条做了反证：**记录存在时同一账本产生 0 条 C-03 失败** ⇒ 该失败确实关于登记顺序。

**④ 单因子断言实测通过**：`require_single_factor` 对冻结的两臂（未做任何剔除）通过，
两臂**只**在 `('train_data_identity', 'train_manifest', 'train_windows')` 上不同
⇒ "唯一改变的是训练清单"有了直接证据，可写进记录的 `reason`。

**本段 0 GPU-h、未读 test、未写真实 attempt 目录、未改任何判据。**

- **下一动作**：等 `seed43_treatment_receipt.json` 落盘 → 按
  `docs/R1_REGISTRATION_CHECKLIST.md` 第 0–6 步（4a→4b 顺序）完成四件套登记；
  登记完成后，把 `campaign-state.current_round_goal` 切到
  `docs/goals/s3-urban-static-covariates.md`，在 **节点 S3** 内推进静态协变量接线轮。

> **本行措辞受 C-04 约束（2026-10-28 实测）**：`check_campaign_state.py` 要求
> **上一轮** brief 最后那个「下一动作」里出现 master plan 的 `current_node`。
> 在临时 master 副本上把 `previous_round_goal` 指向本文件、`current_node=S3` 后，
> 原措辞「等 receipt → 清单第 0–6 步」**不含 `S3`**，校验器立刻报
> `FAIL C-04 ... next action does not name the master plan's current node S3`。
> ⇒ 本行补上 **S3**。这不是为过门而改文字：登记之后的下一步**本来就**是在 S3 节点内
> 启动静态协变量轮，写出来是如实的。
