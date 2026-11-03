# S3：Track 1 R2 —— 气候态锚异常反馈（`draft` vs `draft − C_valid`）在**匹配算力**下的单因素重估——65×65 多年实例 × `full_grid` × 每臂 ≥1 epoch

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-anchor-anomaly-feedback-65x65", "driver": "scripts/study_r7_s3_anchor_feedback_65x65.py", "output_stem": "r7_s3_anchor_feedback_65x65_20261101", "control_record": "s3-fy-year-coverage-matched-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_anchor_feedback_65x65.py"], "planned_seconds": 259200.0, "hard_cap_seconds": 518400.0, "code_archive": "outputs/r7_s3_anchor_feedback_65x65_code.zip", "commit_paths": ["scripts/study_r7_s3_anchor_feedback_65x65.py", "scripts/r7_s3_anchor_feedback_support.py", "scripts/r7_s3_anchor_feedback_parent.py", "scripts/r7_s3_anchor_isolation_parent.py", "scripts/r7_s3_fy_anchor_table_build.py", "tests/test_r7_s3_anchor_feedback_65x65.py", "tests/test_r7_s3_fy_anchor_table_build.py", "docs/goals/s3-anchor-anomaly-feedback-reeval.md", "docs/goals/main-model-climatology-campaign.md", "docs/decisions/0074-anchor-feedback-r2-execution-design.md", "docs/decisions/README.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): implement the R2 anchor-feedback matched-compute round (updates 16071, batch 2)", "register_paths": ["docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-matched-exposure-reeval.md"], "register_message": "docs(r7): register the R2 anchor-feedback matched-compute reading as an evidence record"} -->

状态 **已完成（2026-11-03，两臂 × 三 seed 训练与评分全部跑完，`stage reading` exit 0）**；预声明 (a)–(d) 全真，只提请独立复核。
本文件是 R2 的**人读 brief 与机器可读 `round-spec`**（`campaign-state.current_round_goal` 指向它）；
每轮的硬约束程序来自 `docs/goals/s3-matched-exposure-reeval.md` §4.2。
承接 **决策 0058**（准入与「不是同一族」论证）、**决策 0074**（执行设计）、**决策 0060**（预算）、
计划 0026 §3（Track 1 队列：R1 已关闭年份覆盖支，R2 = 本轮）。

## §0 Objective（单段）

> 在 docs/goals/s3-anchor-anomaly-feedback-reeval.md 接续 S3/Track 1；计划 0026 §3 队列的 R2＝**气候态锚异常反馈**（旧臂 `record:s3-anchor-isolation-result`：F−N 五个 lead 三 seed 全为配对改善、72h −0.321199 K，但 F 的 72h 绝对 skill 三 seed 未全正 ⇒ 按预声明关闭该族；关闭判据是**绝对门**，而该门是**实例性质**——在新实例上无锚的 R1 两臂 72h t2m skill 已 3/3 为正 +0.1799/+0.1949/+0.2046）；本轮在**新制度**上重问这个机制：同一个 65×65 多年 store（train 2000–2021 / val 2022 / test 2023 封存）、**两臂都训练在 treatment 清单**（32,142 窗）、**batch 2 / 16,071 updates**（`16,071×2=32,142 ≥` 32,142 ⇒ 每臂 ≥1 epoch，batch 由用户在计划 0026 §4 拍板）、`full_grid`（4,225 点）；协议内**两臂**（N=`anomaly_feedback=False`、F=`True`）装**同一张**已发布的 **48 桶**（12 月 × 4 synoptic hour）train-only 锚表（`table_sha256 18448ae3df2b260e…`）、都设 `anchor_output_prior=False`、`new_params=0`，其余（归一化、主干 K=4/depth=4/dim=192、迁移父、物理权重、seeds 41/42/43、LR 2e-5、warmup 10、clip 1、val 2022 清单、48 桶气候态分母）逐字相同，唯一因子仍是一个布尔；模型修订在 R1 之后移动（`896eabf` 城市静态开关，默认关、off 路径实测无差异），故先在 `a38be16c…` 下重跑**模型只迁移**（155 张量逐 seed `state_digest` 与 FY 钉逐位相同），本轮**不改 `model/**`**；预声明出口 (a) F 的 72h t2m skill 三 seed 全 >0、(b) F−N 72h ΔRMSE 三 seed 全 <0、(c) F 48h skill 三 seed 全 >0、(d) F−N 6/12/24h seed 均值 ΔRMSE ≤0，全真只提请独立复核、任一假则**永久关闭「气候态锚异常反馈」支**（不扫表粒度/不减锚/不加 seed/不调 updates/不与已关轴组合/不换名重开）；桥 (e) 为**同算力下的身份核对**：N 的第 1 个 update `loss` 须与已登记 R1 treatment 臂逐位相同（41 `1.1786479949951172`、42 `1.4581815004348755`、43 `0.8535277843475342`；`gradient_norm` 只记录不断言；R1 自己的桥因比较不同算力而 4/5 超带宽，故本轮改比同算力），失败只记 identity/engineering unresolved、不进入 (a)–(d)；预算取决策 0060（per-seed 129,600 s / planned 259,200 s / hard 518,400 s，R1 实测每 seed 88,362–89,520 s）；每轮一个臂、GPU 由统一父调度按共驻余量串行、每阶段 spawn 前只读 gpu_gate、不对邻居进程发任何信号；不读 2023 test（r=0、test_read=false、scientific_claim=false）；软超继续并记 soft_overrun，仅 hard 截止截断 attempt；不在运行中改硬限；不新增下载、不重建 store、不 main 合并 / force / 关 issue、不合成数据、不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成。

（objective 实测字符数由 `tools/check_goal_brief.py` 在提交前核验。）

## §1 起点与唯一改动因子

- 起点工作分支 `r7/weather-reasoning`；节点 S3；账本累计见 campaign-state
  （cap 20.0 / remaining 为负仅会计，无总 GPU-h 上限，决策 0030/0038）。
- 起点证据：`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（旧臂，已按预声明永久关闭）、
  `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md`（R1：同 store、同算力的实测与桥失败）、
  `docs/plans/0026-matched-exposure-reeval-and-s4.md` §3/§4、决策 0057/0058/0060/0074。
- **唯一因子**：`anomaly_feedback` 一个布尔（N=`False`，反馈=绝对草稿；F=`True`，反馈=`draft − C_valid`，
  `model/process_step_r7.py:108`）。两臂装同一张表、都关输出/proposal 先验、`new_params=0`。
- **与旧臂的实质差异（0058 的三条 + 锚表粒度）**：实例/网格（129×129 → 65×65）、
  监督域（`interior_32` → `full_grid`）、数据跨度（2,140 → 32,131 可用窗）、锚表（16 桶 4 个月 → 48 桶 12 个月）、
  算力（2,400 updates batch 1 → 16,071 updates batch 2）。旧臂读数对本轮**没有约束力**，反之亦然。
- **为什么现在是可问的**：旧臂的关闭判据是 72 h **绝对门**，而该门是实例性质——R1 在新实例上
  **无锚**两臂的 72 h t2m skill 为 **+0.1799 / +0.1949 / +0.2046**（48 h +0.3569/+0.3614/+0.3788），
  而旧实例同位置对照是 −0.1746/−0.2772/−0.1562。同一机制在新实例上面对的是**已经可过**的绝对门。
- **不得使用的动机**：**禁止**写「旧臂只跑了 1.12 epoch / 欠训练」（决策 0057 已实测旧臂 1.12 epoch，
  0058 明禁该动机）。本轮的开轮理由是上面的制度差异，与训练充分性无关。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮目标与冻结设计决定 | 本文件 §3、决策 0058/0074 |
| D2 | 48 桶 train-only 锚表（已发布）＋当前修订父迁移 | `outputs/r7_s3_fy_anchor_table_20261025_attempt01/`（`table_sha256 18448ae3…`）；`outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/`（`migration_receipt.json`，155 张量 `state_digest` 与 FY 钉逐位相同）；`scripts/r7_s3_anchor_feedback_parent.py` 的 `state_equality_evidence()` |
| D3 | 新驱动/support/父钉/测试 | `scripts/study_r7_s3_anchor_feedback_65x65.py`（584 行）、`scripts/r7_s3_anchor_feedback_support.py`、`scripts/r7_s3_anchor_feedback_parent.py`、`tests/test_r7_s3_anchor_feedback_65x65.py`（27 项，不读 `outputs/`） |
| D4 | 新协议/新输出/新 code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D5 | **两臂**三 seed 真实训练与评分 | `seed{41,42,43}_{control,candidate}_receipt.json`、每 seed 每臂五 lead 的 `boundary_rmse.csv` |
| D6 | 配对读数与预声明判定 | `readings.json`（F−N、`full` 区域 skill、桥 (e)、出口 (a)–(d)） |
| D7 | 登记与精确 CI | `docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`；本轮**不触碰**任何阈值，也**不宣布**通过。

- **新制度（逐字冻结）**：store `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`
  （`[34084,17,65,65]`，train 2000–2021 / val 2022 / test 2023 封存）；**两臂**训练清单 = 该 store 的
  **treatment** 清单（**32,142** 窗，freeze 时由 `preflight_long_rollout_windows` 写入协议）；
  val 清单 = 2022 全年（**1,458** 窗）；**每臂 ≥1 epoch**：`16,071 × 2 = 32,142 ≥ 32,142`。
- **评分域**：整张 65×65，`boundary_margins=(1,)`，读 `boundary_rmse.csv` 的 `full` 区域（4,225 点）；
  **不用** `interior_32`（在 65×65 上只剩 1×1）。
- **气候态分母**：48 个 `(month, hour)` 桶，在 store 的 train 年份（2000–2021）上拟合；缺桶即报错。
- **其余逐字沿用注册配方**：主干 K=4、depth=4、dim=192；LR 2e-5、warmup 10、
  物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、FP32、clip 1、seeds 41/42/43；**本轮不改 `model/**`**。
- **父迁移（本轮唯一新增的数据侧前置）**：模型修订在 R1 冻结后移动（`896eabf` 城市静态开关，
  默认关、off 路径实测无差异；`6f21f71` 只改 `data/preprocess/`），FY 轮父钉按设计无法加载。
  已在当前修订 `a38be16c…` 下用 `scripts/migrate_r7_parent_state.py` 重跑：**155 个张量逐 seed
  `state_digest` 与 FY 钉逐位相同**，仅容器内嵌代码身份不同；`semantics_equal: false` 与
  tendency 迁移同类、已接受。**不放宽任何身份门。**
- **正/负控制（逐 seed 断言，失败即 `raise`）**：① N 的初始化与前向 ≡ 无锚模型；
  ② 表清零 ⇒ `draft − 0 = draft` ⇒ F 前向 ≡ N 前向；③ 两臂 `anchor_table_sha256` 相同；
  ④ 反馈确被读取；⑤ 两臂 `expanded_state_sha256` 相同。
- **预声明出口（冻结后不放宽）**：
  - (a) F 的 **72h** t2m skill 在 seed 41/42/43 **全部 > 0**（旧臂正是栽在这一条）；
  - (b) F−N 的 **72h** t2m ΔRMSE **三 seed 全部 < 0**（负值 = F 更优；旧臂在这一条上 3/3 通过）；
  - (c) F 的 **48h** t2m skill **三 seed 全部 > 0**；
  - (d) F−N 在 **6/12/24h** 的 **seed 均值** ΔRMSE **≤ 0**（F 不得伤短 lead）。
  全真 → **只提请独立复核**，不自动进 S4；任一 (a)–(d) 假 → **永久关闭「气候态锚异常反馈」这一支**：
  不扫表粒度、不减锚、不加 seed、不调 updates、不与已关轴组合、**不换名重开**。
- **桥 (e)，非科学判定（决策 0074 重设计）**：R2 控制臂 N 的**第 1 个 update** `loss` 必须与
  已登记 R1 treatment 臂（`record:s3-fy-year-coverage-matched-result`）的 update-1 `loss` **逐位相同**
  （41 `1.1786479949951172`、42 `1.4581815004348755`、43 `0.8535277843475342`）。依据：同 store、
  同清单、同 seed、同 batch、同父权重、同 LR 表；数据顺序由 `torch.randperm(..., generator=
  torch.Generator().manual_seed(contract["seed"] + epoch))` 决定，不依赖被 fork 的模型构造 RNG；
  N 的前向被锚轮实测为与无锚模型逐位相同。`gradient_norm` 一并记录、**不作逐位断言**。
  **R1 自己的桥比较的是不同算力（16,071-update 控制臂 vs 2,400-update 登记臂），5 个 lead 里
  4 个超带宽——那量的是算力，不是身份；本轮改比同算力的第 1 个 update。**
  失败只记 `identity/engineering unresolved`，**不进入 (a)–(d)**，且须先定位再继续训练。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

### §3.1 并行准入七 gate（逐条，照 `AGENTS.md` 与决策 0042）

| gate | 判定 | 依据 |
| --- | --- | --- |
| 假设独立 | **过** | 唯一因子是 `anomaly_feedback` 一个布尔，与队列其余（年份覆盖已关、H-C/H-B/训练信号族）互不重叠；本轮不与任何候选臂共用机制。 |
| 产物隔离 | **过** | 全部产物写**全新排他** `outputs/r7_s3_anchor_feedback_65x65_<date>_attemptNN/` 与父迁移的全新路径；不写任何既有 attempt 目录，不删既有产物。 |
| 计算资源 | **过（必须串行）** | GPU 由统一父调度按共驻余量**串行**，每阶段 spawn 前只读余量门；本轮**不得**与另一条 GPU 轮次并发。量级：R1 实测 73.62 GPU-h / 三 seed 88,362–89,520 s。 |
| 数据实例 | **过** | 只**只读**读 treatment train 清单与 val 清单、已发布的锚表与迁移父；不写 `data/`，不建新 store，不新增下载。 |
| 可变状态 | **过** | 本轮可变状态全部在自己的 attempt 目录内；与其它轮次无共享可变状态、无共享缓存。 |
| 可独立验证 | **过** | `protocol_sha256` + `code.zip` SHA + 逐 seed×两臂 receipt（含 `expanded_state_sha256` / `anchor_table_sha256` / update-1 loss）+ `readings.json` + 逐 lead `boundary_rmse.csv`；协议级可复现（GPU 训练非逐位可复现）。 |
| 多重比较 | **过（受纪律约束）** | 一条轴、一个候选臂、三个 seed、四个**预声明**出口 (a)–(d)，冻结后不放宽；不做事后挑 lead、不扫表粒度、不加 seed；test 未读（r=0）。 |

⇒ 七 gate 全过；本轮按「每轮只跑一个臂、GPU 串行」执行。

## §4 数字预算与停止

| 项 | 冻结值 | 来源 |
| --- | ---: | --- |
| updates × batch | 16,071 × 2 = 32,142 | 计划 0026 §4（**用户拍板 batch = 2**） |
| 每 seed 软预算 | 129,600 s | 决策 0060 |
| round planned（软） | 259,200 s | 决策 0060 |
| round hard | 518,400 s | 决策 0060 |
| 峰值估计 | 5,596,409,024 B（5,337 MiB） | R1 实测 5,582,618,624 B + 48 桶表 13,790,400 B |
| 共驻门槛 | 估计 5,337 + margin 512 + 邻居 6,144 = **11,993 MiB** | 同上 |
| R1 实测参照 | 每臂训练 37,996–42,254 s、每臂评分 2,554–2,665 s、每 seed 88,362–89,520 s、六臂 73.624654 GPU-h | `outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/` |

- 成本量级与 R1 相同（约 **74 GPU-h**、双卡墙钟约 **35–40 h**）；这是**一条轴**的代价，
  队列还有 R3/R4/R5。若本轮首次实测与 0060 冲突，照 0055/0060 先例**另立预算决策**，不在运行中改硬限。
- `per_seed_seconds` 在 `REUSE_PROTOCOL_FIELDS` 白名单内 ⇒ 预算常数一变，旧 attempt 的 seed
  **不能复用**，必须重跑（正确：曝光变了）。
- 每阶段 spawn 前做**只读** `gpu_gate`；**禁止**对任何非本实验进程发送信号（R-054 / 决策 0026）。
  失败 attempt 原样保留、全额计费、不做 in-place resume。软超继续并记 overrun，hard 截止即停。
- 墙钟不是科学判据；GPU-h 按 spawn 到 reap 连续计费。

## §5 明确不做

不改 `model/**`（三个开关均已存在，故本轮自身无 `model_code_sha256` 变化；父迁移是**既有工具在
当前修订下重跑**，不改比对逻辑）；不新增下载、不重建 store（锚表只从既有 FY store 的 train split 派生，
已发布）；不碰 `data/raw|interim|processed` 与归档快照；不复用旧臂的 16 桶 4 个月锚表；
**不重开旧臂**（不重跑 129×129、不重训已登记臂）；不扫表粒度、不减锚、不加 seed、不调 updates；
不重开容量轴（0048）、H-A/H-B/H-C、PE/scale/typed/ordering/rollout-dose/EMA/dense/final-only/
spatial-gradient/zero-base decoder/gradient-surgery/tendency-input/SAM/全年覆盖（月份轴）/
**年份覆盖（已在匹配算力下第二次关闭，R1）**；不读 test；不 main/force/merge/发版/关 issue；
不删 `outputs/`；不合并另一窗口的改动。

## §6 进度与交接

- **已完成（2026-11-01）**：① 决策 0074（R2 执行设计）+ 决策 README 索引；
  ② 当前修订下的父迁移（`outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/`，
  155 张量、`state_digest` 与 FY 钉逐位相同、`semantics_equal: false` 同类已接受）+
  `scripts/r7_s3_anchor_feedback_parent.py`（`state_equality_evidence()` 实测通过）；
  ③ `r7_s3_anchor_isolation_parent.anchor_payload(run=…)` 增加可选路径参数（缺省不变，宽表实测
  `da2f28c9…` 16 桶、FY 表 `18448ae3…` 48 桶都能加载）；
  ④ 驱动 `scripts/study_r7_s3_anchor_feedback_65x65.py`（584 行，R-051 内）+ support
  `scripts/r7_s3_anchor_feedback_support.py`（154 行，冻结文本/桥钉值/单因子守卫）；
  ⑤ `tests/test_r7_s3_anchor_feedback_65x65.py` **30 passed**（不读 `outputs/`，`frozen_constants`
  把锚 payload 作为参数、合成表在测试内构造）；⑥ `frozen_constants` 用真实 48 桶表实测通过
  （factor/closed_round/changed/recipe epochs_floor/budgets/bridge 钉值/48 桶 anchor block）；
  ⑦ **干净检出复现 CI 并修掉两处真缺陷**（`git worktree` 到无 `outputs/` 的树）：
  (a) `frozen_constants` 的 `climatology_anchor` 块漏改、仍走全局 `anchor_payload()`——本地有
  `outputs/` 所以不暴露，干净树上必炸；已改为使用传入的 payload；(b) 执行清单原来**继承 wide 轮的
  202 项 glob**，违反**决策 0067**（R2–R5 必须用具名清单），且其中一个条目
  `outputs/r7_v2_comparison_20261004_attempt01/protocol.json` 是**未跟踪**的本地文件，干净树上不存在；
  已改为 `tools/r7_explicit_inventory.explicit_execution_files(入口)` 的**本轮 import 闭包 91 项**，
  归档 protocol 是本地 `outputs/` 产物、刻意不进清单（它提供的 spec 由 `spec_canonical_digest`
  在值层面钉住）。**身份表面由 202 项收窄到 91 项，这是 0067 的目标态，不是规模回归**；
  sys.modules 交叉核对：运行时实际导入的 18 个仓内模块 **0 遗漏**；0067 第 5 条的反证
  （新增无关 `data/*.py` 不改变清单）已进测试。
- **读前预检（2026-11-01，seed 阶段运行中）**：用临时目录 + 合成 receipt/CSV **只读预演了
  `_readings` 全链路**（`outputs/r7_s3_anchor_fb_reading_rehearsal_20261101/`，脚本
  `r2_reading_rehearsal.py` + `rehearsal_stdout.json`）：receipt 字段名、逐 lead 评估目录布局、
  `boundary_rmse.csv` 的 `full,t2m` 行格式、`delta_candidate_minus_control`、`exit_conditions`、
  `bridge_e` 全部按预期消费，无缺件。合成 RMSE 相同 ⇒ 出口读数无意义，仅证明**管线**通。
  另实测磁盘投影：R1 的同规模 attempt 实占 **227 GiB**（4,824 个 checkpoint），
  本轮同量级，`/data` 余 488 GiB，够。
- **待做（顺序固定）**：`round-guard`/`commit-guard` → 提交推工作分支并等 CI 绿 →
  重建 code zip → `freeze` → `stage prepare/archive/stage-seeds/reading` → 四件套登记。
- **CI 安全**：测试**不得**依赖 `outputs/`（已用合成锚 payload 满足）；在干净 worktree 里实跑确认。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作（S3）

**2026-11-03 更新：本轮已跑完并登记，预声明 (a)–(d) 全真，只提请独立复核。**
证据 `docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`、索引 `record:s3-anchor-feedback-65x65-result`
（`needs-review` / `engineering-positive`）、账本累计 401.377595 GPU-h、`campaign-state` 同步。
实耗 77.4371 GPU-h（训练 70.4065 + 评分 4.7009 + reading 0.002183；planned 72.0 h，软超 0）。

- **已完成**：`round-guard` → `commit-guard` → 推工作分支 → code zip 重建
  （`8908258f…`、commit `c48c57f`）→ `freeze`（`4a3fe1e8…`）→ `stage prepare`（重封 `89832cce…`）→
  `stage archive` → `stage-seeds`（三 seed 两臂）→ `stage reading`（exit 0）→ 四件套登记。
- **本轮不推进到 S4**：全真只提请独立复核；S4 按决策 0066 已出判决，不改。
- **下一条动作**：W3 队列的 **R4**（位置保留 H-B）执行设计——先按决策 0079 定两臂初始化路线
  （① 迁移 H-B 已训练 query ⇒ `|q|≈0.008` 非零起点；② 新初始化 ⇒ 零起点；**不是同一道题**），
  再立执行设计 ADR、改 `model/**` 承载冻结 `scale`（字段集是精确断言，`process_forecast_r7.py:152-157`）、
  按 0075 重绑父钉。**R4 的 `model/**` 改动现在可以落地了**：R2 的 seed+reading 已全部结束。
- 恢复第一条命令：`.venv/bin/python tools/r7_round.py state`。
- 与 S4 的关系：本轮**不推进到 S4**；S4 已按决策 0066 出过判决，本轮不改它。
- 预声明出口见 §3：**(a)–(d) 全真 → 提请独立复核；任一假 → 永久关闭「气候态锚异常反馈」支**。
- `test` 未读、r=0；不宣布科学通过、不称泛化/SOTA、不自宣最终 goal 完成。
