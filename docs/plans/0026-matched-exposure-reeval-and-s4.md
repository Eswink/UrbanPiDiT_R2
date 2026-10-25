# 0026 — 匹配算力重估计划（含训练年份覆盖轴重开）与 S4 就绪

- **日期**：2026-10-25
- **状态**：计划已定稿并归档；**代码与实验尚未执行**（按 §6 在另一窗口执行）
- **范围**：S3 节点内的「匹配算力重估」队列（5 轮）+ 2016 补年份后的 S4 首次评分
- **依据**：决策 0030 / 0043 / 0044 / 0045 / 0047 / 0055 / 0057 / 0058；
  `docs/plans/0025-training-speed-observability-axis-reeval.md`；`docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md`；
  `docs/goals/s3-anchor-anomaly-feedback-reeval.md`；`AGENTS.md` 的硬约束与七 gate
- **本文件不含任何科学结论**；`scientific_claim: false`、`test_read: false`（Track 0 的评分按 §2 单独预声明）

---

## §0 摘要

两条轨，先做便宜且能达最终门的那条：

| 轨 | 内容 | 训练 GPU-h | 性质 |
| --- | --- | --- | --- |
| **Track 0** | 补 2016 完整未见年份 → S4 首次评分 | **0** | 确认（消耗 r=1） |
| **Track 1** | 匹配算力重估队列，5 轮、每轮一个臂 | 每轮约 68–84 GPU-h | 探索（每轮预声明出口） |

**本次交付**：本文件（提交归档）+ §10 的最终提示词（goal objective + workflow 派生说明）。

---

## §1 必须先写进案的冲突（本计划最敏感的一处）

用户要求重开"22 年数据轴"。按仓库自己的成文标准，这条路**比想象中难走**：

- **决策 0057**（`docs/decisions/0057-supersede-0056-queue-is-executable.md:26-28`）把"不是同一族"的
  **实质内容**定义为"换 **store / 网格 / 监督域 / 数据跨度**"四项。
- 年份覆盖轴重开**只改算力**（updates / batch）：store（同一个 65×65 多年 store）、
  网格（65×65）、监督域（`full_grid` 4,225 点）、数据跨度（2000–2021）**四项一个都不变**
  ⇒ **不满足**该定义。
- **决策 0058**（`docs/decisions/0058-anchor-anomaly-feedback-reeval-admission.md:40-42,71-72`）
  明文**禁止**"旧臂欠训练，所以训练久一点再问"这一动机；
  `docs/goals/s3-anchor-anomaly-feedback-reeval.md:143` 还把"年份覆盖（0054）"列入不重开清单。
- 该轮的**关闭条款本身**（`docs/goals/s3-year-coverage.md:74`）就写了"**不调 2400 updates**"。

⇒ 因此走**"用户具名授权推翻冻结条款"的 ADR**，而**不是**"不是同一族"的论证。
ADR 必须写明负面后果（R-033 硬要求）：**这会削弱"预声明出口不可翻案"的纪律**，
并为其他已关轴按算力重测开先例。

**科学上它仍然站得住**（ADR 的正当性来源）：该轮 **(a)(b)(c) 已全真、(d) 只栽在 6/12/24h 短 lead**
（`docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md:98-103`：6h 三 seed 全正
`+0.009267/+0.003540/+0.005199`，seed 均值 `+0.006002`），而短 lead 正是 0.075 个 epoch
最吃亏处（该页 `:117-119` 自己写"负结果**不能**区分『没有额外价值』与『updates 太少用不上』"）。

⇒ **"补上匹配算力后 (d) 会翻转"是一个可证伪的新预测**，不是换名重问。
ADR 里必须把这条预测写死，并**把先例范围限定在年份覆盖这一条轴**，
明确**不授权**重开 depth / width / SAM / zero-base / EMA / 长 lead 权重 / rollout 剂量。

---

## §2 Track 0 — 先补 2016 再进 S4（0 GPU-h 训练，最高优先级）

按 campaign state 的 2026-10-24 续查留痕执行（下列各项均已在案实测过）：

1. **2016 是完整未见年份**：多年 store
   `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr` 的 `time_ns` 实测 34,084 帧、
   2000-01-01 00:00 → 2023-10-30 18:00；2016 实测 1,464 帧，可用初值 6/12/24/48/72h =
   **1,461/1,460/1,458/1,454/1,450**；对 FY candidate 真正未见
   （其 train 2017–2021 / val 2022 / 归一化 2017–2021）。
2. **必须预声明恰好一个年份**：D4 用 max-statistic 同时区间（`training/r7_simultaneous_stats.py`），
   cell 越多临界值越宽 ⇒ 1 年 = 30 primary cell，17 年 = 110。**2016 是紧邻训练窗之前的那个完整
   年份**，把"气候态分母期与 test 期不同代"这个已声明代价压到最小。
3. **建小 store**（约 0.42 GB raw）：只含 2016 帧 + 装入 FY candidate 的 2017–2021
   `normalization_mean/std`（`data/r7_store.py:113-120` 从 store 读、**无外部覆盖参数**
   ⇒ 只能由 store 携带）。
4. **身份门放行已实测**：`training/r7_evaluate.py:121-123` 比的是**同级 `train.jsonl`** 的
   `dataset_identity`（`training/r7_experiment.py:89-104`）；逐字节复制 FY 该臂的 `train.jsonl`
   即恰好等于 checkpoint 记录的 `data_identity`（实测 treatment `c17494ab…`、
   control `0469312978…` 与协议完全相同）。两臂各需一份。
5. **必须声明的代价**：气候态分母期（2017–2021）与 test 目标期（2016）**不同代**（ERA5 有趋势），
   会改变 t2m skill 这条在测什么 —— 冻进协议并作为限制写明。
6. 这是 **D3 暂缓的"重建 store"路径**，用户已显式授权启用；**不做任何下载**。
7. **冻 S4 评分协议**（**r = 1、alpha_1 = 0.025**），对 FY candidate 三份已冻结 checkpoint 评分，
   按 D4 组成（总集 / 各年 / 四季组 × 17 变量 × 5 lead × full/interior/edge）+
   u10/v10/mslp 相对同数据 incumbent 的同组同 lead 变化。
   **评分消耗 r=1 曝光、不可逆**，执行前把 r 与 alpha 登记做全；
   若 test 覆盖仍不满足 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md:111`
   的"至少一个完整未见年份和全部四季"，**如实记录并停在门前，不放宽判据**。

---

## §3 Track 1 — 匹配算力重估队列（5 轮，恰好用满 workflow 的 5 轮上限）

新制度（`docs/decisions/0057:41-43`、`0058:33-39`）：
**65×65 多年 store（2000–2021 train / 2022 val）× `full_grid`（4,225 点）× 每臂 ≥1 epoch**。

| 轮 | 轴 | 依据 | 为什么现在能问 |
| --- | --- | --- | --- |
| **R1** | 训练年份覆盖 5y → 22y | **本次新 ADR（override，见 §1）** | (a)(b)(c) 已全真、(d) 只栽短 lead；补算力后可证伪地预期翻转 |
| **R2** | 气候态锚异常反馈 | 决策 0058 已准入；brief `docs/goals/s3-anchor-anomaly-feedback-reeval.md` 已在；锚表已发布 `outputs/r7_s3_fy_anchor_table_20261025_attempt01/`（`table_sha256 18448ae3df2b260e…`） | 五 lead 3/3 seed 配对全改善（72h −0.321 K），只栽绝对门 2/3 |
| **R3** | 全局混合（H-C） | 决策 0057 队列 | null，页内明写"更强操纵/更多数据下未测"；吃算力不吃显存（实测 8.955 GiB，低于 depth/width） |
| **R4** | 位置保留（H-B） | 决策 0057 队列 | null 来自**弱操纵**：`position_query` 模长 0.0074–0.0125 vs 同层投影 ~8.02，差 3 个数量级 |
| **R5** | 训练信号族（空间梯度 / dense 12 步 / final-only / tendency） | 决策 0057 队列 | mixed（有配对改善但绝对门未过）、模型零改动 |

**队列外**（方向明确的负结果，不因"更多数据"翻案）：SAM（15/15 一致更差）、
zero-base decoder（72h +1.37 K）、plain EMA、长 lead 权重、rollout 剂量、depth、width。

**命名注意**：`H-D` 在案上有两个含义（ADR 0048 的宽度 vs 架构序列的气候态锚反馈），
本计划内一律写全名。

---

## §4 预算：batch 2 与新预算常数

**batch = 2（用户已拍板）**。实测依据（`docs/plans/0025-*.md:416-421`）：
`seconds_per_update` = **1.839 / 2.351 / 3.360 / 5.620**（batch 1/2/4/8），
`peak_reserved` = **2,322 / 4,528 / 8,950 / 18,092 MiB**；**batch 8 不过共驻门槛（−661 MiB）**。
选 2 的理由：注册 LR 2e-5 是按 batch 1 定的，batch 2 的偏离只有一半，
且不必引入"重调 LR"这个新自由参数。

**新制度要求 `updates × batch ≥ 32,142`** ⇒ batch 2 时 **updates = 16,071**
（候选臂 1 个 epoch、控制臂 2.2 个）。

**预算必然不足**（`docs/decisions/0058:104-106` 已预判）：现行 per-seed 43,200 s（12 h）是按
"两臂 × 2,400 updates"标定的；≥1 epoch 下每臂训练约 10.49 h + 评分约 0.88 h
⇒ **每 seed 约 22.7 h ≈ 81,900 s**。

⇒ 必须**另立预算决策**（照决策 0055 的先例：只改时间、不动科学门、用本轮自己的首次实测支撑、
不在运行中改硬限）。名义起点：**per-seed 129,600 s（36 h）、planned 259,200 s（72 h）、
hard 518,400 s（144 h）**，比例 1:2:4；**执行时须按本轮首次实测重定**，并写进新 ADR 与证据页。

**两个必须改的机制点**：

- `per_seed_seconds` 在 `REUSE_PROTOCOL_FIELDS` 白名单内（`tools/r7_round_support.py:74`）
  ⇒ 改预算后**旧 attempt 已完成的 seed 不能复用**，必须重跑（正确：曝光变了）。
- workflow 模板的 stage 超时是 `34,200,000 ms`（9.5 h），**装不下 22.7 h 的 seed 阶段**
  ⇒ 派生脚本时必须抬到约 **129,600,000 ms（36 h）**。

**成本量级（外推，须以实测替换）**：batch 2 下 6 臂（2 臂 × 3 seed）**68.2 GPU-h**（下界；
用 `attempt03` 实测的 2.265 s/update 则约 **84 GPU-h**）；双卡两并发 + 一随后
⇒ 墙钟约 **34–45 h**。5 轮总量很大 —— 这正是需要 workflow 自设轮数硬上限的原因。

---

## §5 每轮的程序（硬约束，逐条）

1. **新 ADR**（`docs/decisions/NNNN-<slug>.md`，四段齐全、**Consequences 必含负面后果**，R-033）
   + 更新 `docs/decisions/README.md`；`python tools/check_conventions.py --rule R-033 --rule R-036` 通过。
2. **新 brief**（`docs/goals/<slug>.md`，骨架照 `.agents/skills/goal-loop/SKILL.md:64-70` 的 §0–§8；
   `tools/check_goal_brief.py --brief` 的 G-01..G-08 通过），带**预声明出口 (a)–(d) + 桥 (e)**，
   格式照 `docs/goals/s3-year-coverage.md:68-79`：全真 → 提请独立复核；任一假 → 永久关闭该支。
3. **过七 gate**（`AGENTS.md:26`）；**每轮只跑一个臂**、GPU 由统一父调度按共驻余量**串行**
   （R-054，**不对邻居进程发任何信号**）。
4. **先冻 `protocol.json` 再跑**（R-006）；**只读复用** store + 两套清单
   （`manifests_control` 7,291 / `manifests_treatment` 32,142）+ 48 桶气候态 `reference.json`
   + 迁移父 ⇒ **不重建 store、不新增下载**。
5. **四件套**（`docs/goals/main-model-climatology-campaign.md:19-23`）：
   §8 进度块（含下一动作第一条命令）+ 账本行 + `docs/R7_EVIDENCE_INDEX.jsonl` record
   + `<!-- campaign-state: ... -->` 更新。
6. **workflow 自跑确定性闸门**（决策 0044）：提交前 `tools/r7_round.py commit-guard <paths>`、
   登记后 `verify`；只推 `r7/weather-reasoning`；**禁 main / force / merge / 关 issue**。

---

## §6 交付物与执行顺序

1. **本次（计划模式）**：`docs/plans/0026-<slug>.md`（提交归档，含 §10 的最终提示词）。
2. **下一窗口按提示词执行**：
   - (a) 建 `docs/goals/s3-matched-exposure-reeval.md`（§0–§8，objective 单段 ≤4000 code point、
     交付物内嵌、结尾"不要自行宣布目标完成"）；
   - (b) 写 R1 的 override ADR（§1 的四条要求）；
   - (c) 加载 dynamic-workflows 技能，从 `.zcode/workflow-drafts/r7-campaign-round.dwf.ts`
     派生新脚本（`maxRounds = 5`、stage 超时抬到 36 h、文件即参数）；
   - (d) 跑 Track 0（2016 → S4）与 Track 1 的 5 轮。
3. **停下条件**（决策 0043 封闭三条）：触保留授权边界 / 达已冻结的科学停止条件 /
   确无可安全独立推进的工作。停下必须在 §8 写"为何停 + 恢复第一条命令"。

---

## §7 验证

- `check_conventions` 37 条阻断 0 失败；`check_goal_brief.py` G-01..G-08 通过；
  `check_campaign_state.py` C-01..C-06 通过。
- Track 0：身份门实测复现（两臂 `train.jsonl` 的 `dataset_identity` 等于 checkpoint 记录的
  `data_identity`）+ r/alpha 登记完整。
- Track 1 每轮：`window_sha256` 与冻结协议逐字节一致；预声明出口逐条给数字；
  `tools/r7_round.py verify` 通过。
- 所有成本只写实测值；外推必须标注为外推。

---

## §8 明确不做

- 不动 `data/raw|interim|processed` 与归档快照；不读 test 除非进入 Track 0 的冻结评分。
- 不 main 合并 / force / 关 issue / 删已有产物。
- 不合成数据；不跳过或弱化测试与断言；**禁止事后放宽已冻结判据**。
- GPU 默认共驻、只读核余量、**不对邻居进程发任何信号**。
- 不重开 depth / width / SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量。

---

## §9 风险

- **R1（高）**：年份轴 override 会**削弱"预声明出口不可翻案"的纪律**。
  缓解：ADR 把先例范围**限定在这一条轴**，写明不授权其他轴按算力重测。
- **R2（高）**：预算从 12 h 抬到 36 h/seed，5 轮总量约 350–420 GPU-h。
  缓解：workflow 自设 5 轮硬上限 + 每轮预声明出口 + 硬上限截断即记失败 attempt（**不算科学通过**）。
- **R3（中）**：workflow 模板的 stage 超时（9.5 h）装不下新 seed（22.7 h）
  ⇒ 派生时必须抬高，否则会误判超时。
- **R4（中）**：Track 0 消耗 r=1 且不可逆。缓解：2016 已按"cell 最少、代差最小"预声明；
  评分前把 r/alpha 登记做全。
- **R5（中）**：5 轮 × 3 seed = 30+ 个新比较 ⇒ 必须预先声明出口并守 r/alpha 纪律。

---

## §10 最终提示词（本次交付的第二件产物，供拷入另一窗口）

```text
在 /data/esw/UrbanPiDiT_R2（分支 r7/weather-reasoning）执行已归档的计划
docs/plans/0026-matched-exposure-reeval-and-s4.md。先完整读它，再按 §6 的顺序做。
不要自行宣布目标完成。

【先核状态】git rev-parse HEAD；.venv/bin/python tools/r7_round.py state；
确认没有别的 attempt 在跑（tools/r7_round.py orphans）。

【Track 0：先补 2016 再进 S4（0 GPU-h 训练，最高优先级）】
按计划 §2 逐条做：2016 是完整未见年份（1,464 帧，可用初值 1,461/1,460/1,458/1,454/1,450）；
新建排他目录放 ① 只含 2016 帧的小 store（装入 FY candidate 的 2017–2021
normalization_mean/std）② 逐字节复制 FY 该臂的 train.jsonl 作同级文件（身份门才放行）
③ 指向小 store 的 test.jsonl；两臂各一份。预声明恰好一个年份（2016），不要声明多个。
把"气候态分母期与 test 期不同代"冻进协议并写成限制。
然后冻 S4 评分协议（r=1、alpha_1 = 0.025）对 FY candidate 三份 checkpoint 评分，
按 D4 组成报读数；若 test 覆盖仍不满足协议第 111 行，如实记录并停在门前，不放宽判据。

【Track 1：匹配算力重估队列（5 轮，每轮一个臂）】
新制度：65×65 多年 store（2000–2021 train / 2022 val）× full_grid（4,225 点）× 每臂 ≥1 epoch。
batch = 2（用户已定）；updates × 2 ≥ 32,142 ⇒ updates = 16,071。
队列：R1 训练年份覆盖 5y→22y（需新 override ADR，见下）；R2 气候态锚异常反馈
（决策 0058 已准入，brief 与锚表都已在）；R3 全局混合；R4 位置保留；R5 训练信号族。
每轮：新 ADR（四段齐全、Consequences 含负面后果）→ 新 brief（§0–§8、预声明出口 (a)-(d)+桥 (e)）
→ 过七 gate → 冻 protocol.json → 跑 → 四件套登记。只读复用 store/两套清单/48 桶气候态/迁移父，
不重建 store、不新增下载。

【R1 的 override ADR 必须写明（这是本次最敏感的一处）】
- 如实写：按决策 0057 的成文定义，只改算力**不满足**"不是同一族"
  （store/网格/监督域/数据跨度四项全不变），因此走的是"用户具名授权推翻冻结条款"这条路，
  取代 docs/goals/s3-year-coverage.md:74 的"不调 2400 updates"。
- 写明负面后果：削弱"预声明出口不可翻案"的纪律；为按算力重测开先例。
- 把先例范围**限定在年份覆盖这一条轴**，明确不授权重开 depth/width/SAM/zero-base/EMA/
  长 lead 权重/rollout 剂量。
- 科学依据写成可证伪预测：该轮 (a)(b)(c) 已全真、(d) 只栽在 6/12/24h 短 lead，
  而短 lead 正是 0.075 epoch 最吃亏处 ⇒ 预期补上 ≥1 epoch 后 (d) 翻转。

【预算：用户已显式授权"不够就自主加"】
现行 per-seed 43,200 s 对 ≥1 epoch 必然不足（每 seed 约 22.7 h ≈ 81,900 s）。
按决策 0055 的先例另立预算决策（只改时间、不动科学门、用本轮首次实测支撑、
不在运行中改硬限）；名义起点 per-seed 129,600 s / planned 259,200 s / hard 518,400 s
（1:2:4），执行时按实测重定。
注意：per_seed_seconds 在复用白名单内 ⇒ 改预算后旧 attempt 的 seed 不能复用，必须重跑。

【workflow】
先加载 dynamic-workflows 技能（未加载时 CreateWorkflow 拒跑），再从
.zcode/workflow-drafts/r7-campaign-round.dwf.ts 派生新脚本：maxRounds 硬上限 5；
**stage 超时从 34,200,000 ms 抬到约 129,600,000 ms（36 h）**——否则装不下 22.7 h 的 seed 阶段；
world.run 的命令名保持编译期字面量、运行值落盘（文件即参数）；
自跑 tools/r7_round.py commit-guard（提交前）与 verify（登记后），不依赖 hook 兜底；
只推 r7/weather-reasoning（禁 main/force/merge/关 issue）；报告与 CI 绿都不是科学结论。
它是**有界、可恢复的批量编排器，不是守护进程**；被打断按 ResumeWorkflowRun 续。

【纪律】不动 data/raw|interim|processed 与归档快照；不读 test 除非 Track 0 的冻结评分；
不 main 合并/force/关 issue/删已有产物；不合成数据；不跳过或弱化测试与断言；
禁止事后放宽已冻结判据；GPU 默认共驻、按 R-054 只读核余量、不对邻居进程发任何信号；
外部检索走 arxiv-researcher / web-researcher 两条通道。
每一步都要给可核查证据（命令 + 输出 + digest），估计值不得写成实测值。
收尾按 AGENTS.md 报告：改动 / 实跑过的验证 / 接口变化 / 数据与结果影响 / 兼容性风险 /
依赖影响 / 未做的事 / 下一项。
```

---

## 实际结果

- **完成情况**：本次交付的两件产物均已完成——本计划文件与 §10 的最终提示词。
  **计划内的代码、ADR、brief、workflow 与实验（Track 0 / Track 1）全部未执行**：
  它们按 §6 排在另一窗口执行。
- **与计划的差异**：无（本文件即计划定稿；执行尚未开始）。
- **验证**：本次只做只读探查（三路 Explore + 关键文件复核），未跑任何训练或评估；
  所有引用数字均标注来源文件与行号；实测与外推已分开标注。
- **遗留**：① 年份覆盖轴的 override ADR 是本次最敏感的一处——它**削弱**"预声明出口不可翻案"的
  纪律，执行时必须在 ADR 里写明并把先例范围限定在该轴（见 §1、§9 R1）；
  ② §4 的成本（68.2–84 GPU-h/轮）与预算常数均为外推，须由本轮首次实测替换；
  ③ Track 0 消耗 r=1 且不可逆（见 §2.7）。
