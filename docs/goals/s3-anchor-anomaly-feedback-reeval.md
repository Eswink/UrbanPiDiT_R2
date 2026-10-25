# S3：W3 队列① 气候态锚异常反馈的单因素重估轮（`draft` vs `draft − C_valid`）——65×65 多年实例 × `full_grid` × 每臂 ≥1 epoch

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-anchor-anomaly-feedback-65x65", "driver": "scripts/study_r7_s3_anchor_feedback_65x65.py", "output_stem": "r7_s3_anchor_feedback_65x65_20261025", "control_record": "s3-fy-annual-coverage-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_anchor_feedback_65x65.py"], "commit_paths": ["scripts/study_r7_s3_anchor_feedback_65x65.py", "scripts/r7_s3_anchor_feedback_support.py", "scripts/r7_s3_fy_anchor_table_build.py", "scripts/r7_s3_fy_annual_parent.py", "tests/test_r7_s3_anchor_feedback_65x65.py", "docs/goals/s3-anchor-anomaly-feedback-reeval.md", "docs/decisions/0058-anchor-anomaly-feedback-reeval-admission.md", "docs/decisions/README.md", "docs/goals/main-model-climatology-campaign.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): implement the 65x65 climatology-anchor anomaly-feedback reevaluation round", "register_paths": ["docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md"], "register_message": "feat(r7): register the 65x65 anchor anomaly-feedback reevaluation run as a new evidence record", "code_archive": "outputs/r7_s3_anchor_feedback_65x65_code.zip"} -->

状态 **准入已定（决策 0058）、设计已冻结在本文档；锚表/驱动/测试尚未派生、协议未冻、GPU 未开跑**（2026-10-25）。
唯一主计划 `docs/goals/main-model-climatology-campaign.md`；承接 **决策 0057**（计划 0025 §4.2 队列可执行）
与 **决策 0058**（本条的开轮准入论证）。

## §0 Objective（单段）

> 在 docs/goals/s3-anchor-anomaly-feedback-reeval.md 接续 S3/index（见 campaign-state）、2023 test 未评分 r0、S4 未启动；计划 0025 §4.2 的「已关闭轴重估」队列第 1 条＝**气候态锚异常反馈**（旧臂 `record:s3-anchor-isolation-result`：F−N 五个 lead 三 seed 全为配对改善、72h −0.321199 K，但 F 的 72h 绝对 skill 三 seed 未全正 ⇒ 按预声明关闭该族）；决策 0057 判定队列可执行并要求本条 ADR 论证「与已关闭旧臂不是同一族」，决策 0058 给出该论证：**新制度 = 同一个 65×65 多年 store（train 2000–2021 / val 2022 / test 2023 封存）× `full_grid`（整张 65×65 = 4,225 点）× 每臂 ≥1 epoch**，其「不是同一族」的实证核心是**旧臂的关闭判据（72h 绝对门）是实例性质**——在新实例上无锚的 FY 轮两臂 72h t2m skill 已 3/3 为正（treatment +0.047693/+0.050579/+0.092226；control +0.025785/+0.047884/+0.076949），而旧实例同位置对照为 −0.174635/−0.277247/−0.156169；本轮据此在**新制度**上重问这个机制：协议内**两臂**（N = `anomaly_feedback=False`、F = `True`），两臂装**同一张**从 FY store train split 派生的 **48 桶**（12 月 × 4 synoptic hour）train-only 锚表、都设 `anchor_output_prior=False`、`new_params=0`，其余（store、归一化、模型 spec K=4/depth=4/dim=192、迁移父、物理权重、seeds 41/42/43、val 2022 清单、48 桶气候态分母、updates、batch）逐字相同，唯一因子仍是一个布尔；预声明出口 (a) F 的 72h t2m skill 三 seed 全 >0、(b) F−N 72h ΔRMSE 三 seed 全 <0、(c) F 48h skill 三 seed 全 >0、(d) F−N 6/12/24h seed 均值 ΔRMSE ≤0，全真只提请独立复核、任一假则永久关闭「气候态锚异常反馈」这一支（不扫表粒度/不减锚/不加 seed/不调 updates/不换名重开）；桥 (e) 为恒等/工程核对（N 的 update-1 `loss` 与已登记 FY 候选臂逐位相同；表清零 ⇒ F ≡ N；两臂表 digest 相同），不作科学判定；**batch 取值由用户按计划 0025 §5 探针决定，执行者不自行拍板**，且 `updates × batch ≥ 32,142` 的暴露约束把二者绑死，故 `protocol.json` 必须在用户定 batch 之后才冻；本轮不读 test、不改 `model/**`、不新增下载、不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成；不 paid/rental/exclusive/main/release/issueclose/force/mirror/破坏性/cron，不改科学合同/冻结证据/用户配置。

（objective 实测字符数由 `tools/check_goal_brief.py` 在提交前核验。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 GPU-h 与 index 见 campaign-state。
- 起点证据：`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（旧臂，已按预声明永久关闭）、
  `docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md` 与其 index record（`record:s3-fy-annual-coverage-result`，
  新实例上无锚两臂的读数来源）、`docs/plans/0025-training-speed-observability-axis-reeval.md` §4/§5、
  决策 0057 与 0058。
- **为什么是这一条**：计划 §4.2 把它排第 1，理由是它是「**唯一有正向长 lead 线索的轴**」——
  旧臂 F−N 在 15/15 个 seed×lead 单元同号为负、72h 均值 −0.321199 K；关闭它的是**绝对门**
  （F 的 72h skill 2/3 正），不是配对效应。**本轮不重开旧臂**，而是在**新制度**上问一个**新问题**。
- **与旧臂的实质差异（三处 + 一处，逐条如实声明）**：
  1. **实例/网格**：129×129（`outputs/r7_s3_wide_instance_v4_20261009_attempt01`）→ 65×65
     （`outputs/r7_s3_fy_multiyear_20261023_attempt01`，`state [34084,17,65,65]`）。
  2. **监督域**：`interior_32`（4,225 格，`mask_sha256 4a2a576b…`）→ `full_grid`（整张 65×65，
     4,225 点）。**这不是选择**：`interior_32` 在 65×65 上只剩 1×1。
  3. **数据跨度**：2017–2021（2,140 可用窗）→ 2000–2021（32,131 可用窗）。
  4. **锚表粒度**：16 桶（4 个月）→ **48 桶（12 个月）**。旧 store 的 train/val 只覆盖
     Jan/Apr/Jul/Oct；新 store 覆盖 12 个月、val 是 2022 全年，16 桶表会让 val 里 8 个月的查询
     触发 `TrainClimatologyAnchor` 的缺桶报错。**如实承认这是新假设的又一处不同**。
- **「不是同一族」的实证核心**：旧臂的关闭判据是 **72h 绝对门**，而该门是**实例的性质**：
  在新实例上，**无锚**的 FY 轮两臂 72h t2m skill 已 3/3 为正（treatment
  +0.047693/+0.050579/+0.092226、control +0.025785/+0.047884/+0.076949），而旧实例同位置对照 N 是
  −0.174635/−0.277247/−0.156169。⇒ 同一机制在新实例上面对的是一个**已经可过**的绝对门。
- **不得使用的动机**：**禁止**写「旧臂只跑了 1.12 epoch / 欠训练」。0057 已实测旧臂
  `updates/usable_windows = 2400/2140 = 1.12` epoch；本轮的开轮理由是上面的三条差异 + 绝对门性质，
  与训练充分性无关。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3、决策 0058 |
| D2 | **65×65、48 桶 train-only 气候态锚表**（新派生数据集，身份齐全） | `outputs/r7_s3_fy_anchor_table_<date>/`：`anchor_spec.json`（`r7-train-climatology-anchor-v1`）＋`anchor_table.pt`＋`build_receipt.json`＋`BUILD_COMPLETE.json`＋只读 preflight 报告 |
| D3 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D4 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{control,candidate}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D5 | 配对读数与预声明判定 | `readings.json`（F−N、`full` 区域 skill、桥 (e)、出口 (a)–(d)） |
| D6 | 登记与精确 CI | `docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；本轮**不触碰**任何阈值，也**不宣布**通过。

- **协议内控制臂 N**：锚**已装**（`climatology_anchor_spec=<48 桶 FY train-only 表>`）但
  `anchor_output_prior=False`、`anomaly_feedback=False`。**两个消费者都关 ⇒ 锚不被任何前向路径读取**，
  故 N 在**初始化与 forward** 上必须与**无锚**的注册模型逐位相同（除新增的两个 persistent buffer
  `backbone.climatology_anchor.{table,ready}`，它们不参与前向）。
- **候选臂 F**：与 N 同表、同 `anchor_output_prior=False`，**只把 `anomaly_feedback=True`**
  （`model/process_step_r7.py:108` 的 `feedback_draft = draft − climate`）。**唯一因子。**
- **新制度（逐字冻结）**：store `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`
  （`[34084,17,65,65]`，train 2000–2021 / val 2022 / test 2023 封存）；两臂训练清单 = 该 store 的
  **treatment** 清单（2000–2021 全 12 个月，**32,142** 窗，freeze 时由
  `preflight_long_rollout_windows` 写入协议）；val 清单 = 2022 全年（**1,458** 窗，两臂逐字节相同）；
  **每臂 ≥1 epoch**，即 `updates × batch ≥ 32,142`。
- **评分域**：整张 65×65，`boundary_margins=(1,)`，读 `boundary_rmse.csv` 的 `full` 区域
  （4,225 点）；**不用** `interior_32`（在 65×65 上只剩 1×1）。
- **气候态分母**：48 个 `(month, hour)` 桶，在 store 的 train 年份（2000–2021）上拟合；
  缺桶即报错，不允许 held-out fallback。**与旧臂的 2017–2021 4 个月分母不可比。**
- **其余逐字沿用注册配方**：模型 spec K=4、depth=4、dim=192；迁移父 = FY 轮的 `r7_s3_fy_annual_parent`
  pins；LR 2e-5、warmup 10、物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、FP32、clip 1、
  seeds 41/42/43；**本轮不改 `model/**`**（`anomaly_feedback` / `anchor_output_prior` /
  `climatology_anchor_spec` 三个开关均已存在），故**无需父重迁移**。
- **正/负控制（逐 seed 断言，失败即 `raise`）**：
  ① N 的初始化与 forward ≡ 无锚模型（`unanchored_max_abs == 0`）；
  ② 把锚**表清零**（`ready` 仍 True）⇒ `climate = 0` ⇒ `draft − 0 = draft` ⇒ **F 的前向 ≡ N 的前向**
  （`zeroed_table_delta_max_abs == 0`）；
  ③ 两臂 `table_sha256` 逐 seed 相同（不同即 `raise`）；
  ④ `feedback_is_read == true`（候选臂的反馈路径确被读取）。
- **预声明出口（冻结后不放宽）**：
  - (a) F 的 **72h** t2m skill 在 seed 41/42/43 **全部 > 0**（旧臂正是栽在这一条）；
  - (b) F−N 的 **72h** t2m ΔRMSE **三 seed 全部 < 0**（负值 = F 更优；旧臂在这一条上是 3/3 通过的）；
  - (c) F 的 **48h** t2m skill **三 seed 全部 > 0**；
  - (d) F−N 在 **6/12/24h** 的 **seed 均值** ΔRMSE **≤ 0**（F 不得伤短 lead）。
  全真 → **只提请独立复核**，不自动进 S4；任一 (a)–(d) 假 → **永久关闭「气候态锚异常反馈」这一支**：
  不扫表粒度、不减锚、不加 seed、不调 updates、不与已关轴组合、**不换名重开**。
- **桥 (e)，非科学判定**：① N 臂 **update-1** 的 `loss` 必须与已登记 FY 候选臂
  （`record:s3-fy-annual-coverage-result`）记录的 update-1 `loss` **逐位相同**——这见证「N 就是 FY 臂
  加一个惰性锚」；`gradient_norm` 一并记录（跨 attempt 实测最后几位可不同，不作逐位断言）。
  ② 两臂装同一张表、三个恒等反证成立（见上）。
  失败只记 `identity/engineering unresolved`，**不进入 (a)–(d)**。
- **batch 与暴露（本轮最关键的待定项）**：`updates × batch ≥ 32,142` 把 batch 与 updates 绑死，
  故 **batch 取值必须先定**（计划 0025 §5 决策点 + 用户保留）。**batch 8 已实测不过共驻门槛**
  （需 24,748 MiB，探针开始时空闲 24,087 MiB ⇒ −661 MiB），**不在选项内**。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

### §3.1 并行准入七 gate（逐条，照 `AGENTS.md` 与决策 0042）

| gate | 判定 | 依据 |
| --- | --- | --- |
| 假设独立 | **过** | 唯一因子是 `anomaly_feedback` 一个布尔，与队列其余三条（H-C 全局混合 / H-B 位置保留 / 训练信号族）及已关闭的年份覆盖轴互不重叠；本轮不与任何候选臂共用机制。 |
| 产物隔离 | **过** | 全部产物写**全新排他** `outputs/r7_s3_anchor_feedback_65x65_<date>_attemptNN/` 与全新的 48 桶锚表路径；不写任何既有 attempt 目录，不删既有产物。 |
| 计算资源 | **过（但必须串行）** | GPU 由统一父调度按共驻余量**串行**，每阶段 spawn 前只读余量门；本轮**不得**与另一条 GPU 轮次并发。资源量级已在 §4 用实测外推列明（batch 1/2/4 = 103.8/68.2/50.3 GPU-h）。 |
| 数据实例 | **过** | 只**只读**读 FY store 的 treatment train 清单与 val 清单，以及 pinned 迁移父检查点；锚表从同一 store 的 train split 只读派生。不写 `data/`，不建新 store，不新增下载。 |
| 可变状态 | **过** | 本轮的可变状态全部限制在自己的 attempt 目录内（checkpoint、`training_report.json`、`progress.jsonl`、receipts、`readings.json`）；与其它轮次**无共享可变状态**、无共享缓存。 |
| 可独立验证 | **过** | `protocol_sha256` + `code.zip` SHA + 逐 seed×两臂 receipt（含 `expanded_state_sha256` / `anchor_table_sha256`）+ `readings.json` + 逐 lead `boundary_rmse.csv` 齐备；可复现等级为**协议级可复现**（GPU 训练非逐位可复现）。 |
| 多重比较 | **过（受纪律约束）** | 一条轴、一个候选臂、三个 seed、四个**预声明**出口 (a)–(d)，冻结后不放宽；不做事后挑 lead、不扫表粒度、不加 seed；test 未读（r=0），val 只用于预声明出口。 |

⇒ 七 gate 全过；本轮按「每轮只跑一个臂、GPU 串行」执行。

## §4 数字预算与停止

| batch | ≥1 epoch 的 updates | 实测 s/update | 训练 h/臂 | +评分 h/臂 | 合计 h/臂 | 6 臂 GPU-h | 墙钟 h（双卡 3 seed） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32,142 | 1.839 | 16.42 | 0.88 | 17.30 | 103.8 | 51.9 |
| 2 | 16,071 | 2.351 | 10.49 | 0.88 | 11.37 | 68.2 | 34.1 |
| 4 | 8,036 | 3.360 | 7.50 | 0.88 | 8.38 | 50.3 | 25.1 |

- 上表是**外推**：训练项用计划 0025 §5 探针在**同一 store、同一 spec** 上实测的 `s/update`；
  评分项用 `attempt03` 实测每臂五 lead 均值 3,159 s；墙钟按决策 0047 双卡、3 seed（两并发 + 一随后）
  ≈ 1.5 波 × 2 臂。同一条 treatment 清单上 `attempt03` 的实测训练速率是 **2.265 s/update**（batch 1，
  含每臂装载/发现开销），比探针高 **23%** ⇒ 上表训练项是**下界**。
- **预算常数**：现行 per-seed 43,200 s 是按「两臂 × 2400 updates」标定的，对本轮的 ≥1 epoch 暴露
  **必然不足**；须照决策 0053/0055 先例**另立一份预算决策**（先冻后跑、不在运行中改）。
  **具体常数在用户定 batch 之后、与 `protocol.json` 一起冻结**；本文件先记规则与外推，不拍数字。
- 每阶段 spawn 前做**只读** `gpu_gate`；`required = max(estimated_peak, owned_peak) + 512 MiB + 6 GiB`；
  **禁止**对任何非本实验进程发送信号（R-054 / 决策 0026）。失败 attempt 原样保留、全额计费、
  不做 in-place resume。软超继续并记 overrun，hard 截止即停 attempt 并全额留失败记录。
- 墙钟不是科学判据；GPU-h 按 spawn 到 reap 连续计费。

## §5 明确不做

不改 `model/**`（三个开关均已存在，故无父重迁移、无 `model_code_sha256` 变化）；
不新增下载、不重建 store（锚表只从**既有** FY store 的 train split 派生）；
不碰 `data/raw|interim|processed` 与归档快照；不复用旧臂的 16 桶 4 个月锚表；
**不重开旧臂**（不重跑 129×129、不重训已登记臂）；不扫表粒度、不减锚、不加 seed、不调 updates；
**batch 不在本轮由执行者拍板**，且**batch 变更不与本轮同轮做**（计划 §5）；
不重开容量轴（0048）、H-A/H-B/H-C、PE/scale/typed/ordering/rollout-dose/EMA/dense/final-only/
spatial-gradient/zero-base decoder/gradient-surgery/tendency-input/SAM/全年覆盖（月份轴）/年份覆盖（0054）；
不读 test；不 main/force/merge/发版/关 issue；不删 `outputs/`；不合并另一窗口的改动。

## §6 进度与交接

- **已完成（2026-10-25）**：① 决策 0057（取代 0056，队列可执行）；② 决策 0058（本条准入论证：
  新制度逐字定义、「不是同一族」三条论证、48 桶锚表、预声明出口与恒等桥、batch 1/2/4 成本表、
  禁写「旧臂欠训练」动机）；③ 本 brief；④ 只读核对 FY store 身份（`[34084,17,65,65]`、
  train 2000–2021、48 桶气候态 `n_selected_steps` 32,144）；⑤ 只读核对桥 (e) 的可实现性：
  `attempt02` 与 `attempt03` 的 treatment 臂 update-1 `loss` **逐位相同**
  （seed41 `1.2447926998`、seed42 `2.0360622406`、seed43 `0.6930788159`）。
- **待做（顺序固定）**：
  1. **用户定 batch**（计划 §5 决策点；`updates × batch ≥ 32,142` 把二者绑死）；
  2. 按 batch 定预算常数并**另立预算决策**（照 0053/0055 先例）；
  3. 派生并合格化 **65×65、48 桶**锚表（D2，只读 preflight 后 `--write` 到全新排他路径）；
  4. 派生驱动（**R-051 600 行硬限**：FY 轮的 store/manifest/气候态/出口/登记管路 +
     旧臂的 `arm_spec`/`anchor_block`/`expand_parent_state` 机制，超出上限的部分拆到
     `scripts/r7_s3_anchor_feedback_support.py`）与 `tests/test_r7_s3_anchor_feedback_65x65.py`；
  5. 独立复核、`round-guard`/`commit-guard`、提交推工作分支并等 CI 绿；
  6. `freeze` → `stage prepare/archive/stage-seeds/reading` → 四件套登记。
- **CI 安全**：测试**不得**依赖 `outputs/`；纯函数在 tmp 路径上构造，`sha256_file`/`dataset_identity`
  用 stub；在干净 worktree（无 `outputs/`）里实跑确认。
- **第一条命令（用户定 batch 之后）**：`.venv/bin/python tools/r7_round.py round-guard` →
  重建 code zip → `.venv/bin/python tools/r7_round.py freeze` →
  `stage prepare/archive/stage-seeds/reading`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作（S3）

- **当前唯一阻塞项**：**batch 取值**（用户决定）。其余 CPU 前置（锚表、驱动、测试）可随后进行。
- 与 S4 的关系：本轮**不推进到 S4**；FY 轮的 candidate 仍等用户验收。
- 预声明出口见 §3：**(a)–(d) 全真 → 提请独立复核；任一假 → 永久关闭「气候态锚异常反馈」支**。
- `test` 未读、r=0；不宣布科学通过、不称泛化/SOTA、不自宣最终 goal 完成。
