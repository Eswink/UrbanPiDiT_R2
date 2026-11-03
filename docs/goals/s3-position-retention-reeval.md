# S3 Track 1 R4：位置保留（H-B 的 pooled vs spatial 聚合）在**零起点 + 冻结前向敏感度**下的单因素重估——65×65 多年实例 × `full_grid`

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-position-retention-65x65", "driver": "scripts/study_r7_s3_position_retention_65x65.py", "output_stem": "r7_s3_position_retention_65x65_20261103", "control_record": "s3-anchor-feedback-65x65-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_position_retention_65x65.py", "tests/test_r7_s3_position_retention_65x65_router.py"], "planned_seconds": 259200.0, "hard_cap_seconds": 518400.0, "code_archive": "outputs/r7_s3_position_retention_65x65_code.zip"} -->

状态 **设计已冻结（决策 0078 准入 / 0079 问法收窄 / 0080 执行设计 / 0081 常数定值），驱动与测试未写、协议未冻、GPU 未开跑**（2026-11-03）。
本文件是 R4 的**人读 brief 与机器可读 `round-spec`**。

## §0 Objective（单段）

> 在 docs/goals/s3-position-retention-reeval.md 接续 S3/Track 1；W3 队列（计划 0025 §4.2）的第 4 条—— **R3 已由决策 0077 否决（不开轮）**，故顺位到此。本轮在**新制度**（65×65 多年 store、`full_grid` 4,225 点、 treatment 清单 32,142 窗、batch 2 / 16,071 updates）上重问旧臂 H-B（`record:s3-typed-spatial-result`） 的**类型化证据 patch 位置聚合**假设：控制臂 B=`pooled`（对 P 个 patch 位置取**均匀均值**）， 候选臂 C=`spatial`（用**一个共享可学**的 `[dim]` query 对 P 做 softmax 注意力），**两臂共用同一个 冻结常数 `position_scale = 8000.0`**（决策 0081，跑前在真实 val 帧上实测定值），其余逐字沿用注册配方； 唯一因子是**聚合方式**。本轮的问法按决策 0079 收窄为「**在给定的前向敏感度下**，位置加权是否改变长 lead 读数」—— `scale` **不**承担修正优化器病态之责（0079 实测 AdamW 约掉整体梯度缩放，`scale` 是前向敏感度旋钮、 不是学习幅度旋钮；`position_query` 初始化为**精确 0**）。初始化按决策 0080 走**路线②**：两臂在 `isolated_stream()` 下**新初始化**、**零起点**（`|q|==0`），**不**迁移 H-B 的 9 个训练后张量； typed-evidence 侧车**必须**用 65×65 store 重算（已发布 `outputs/r7_s3_typed_evidence_fy65_20261103_attempt01/`， `typed_evidence_identity a61d22aa…`），**不得**沿用 H-B 的 129×129 payload（实测错配解码风速场偏 14–31%、 `v500` 符号翻转）。预声明出口 (a) C 的 72h t2m skill 三 seed 全 >0、(b) C−B 的 72h ΔRMSE 三 seed 全 <0、 (c) C 的 48h skill 三 seed 全 >0、(d) C−B 在 6/12/24h 的 seed 均值 ΔRMSE ≤0；全真**只提请独立复核**， 任一假则**永久关闭「位置保留」这一支**（不扫 `scale`、不加 seed、不换聚合族、不改表粒度）。 桥 (e) 为四条可断言的恒等式：两臂初始 `|position_query|==0`、同 seed 两臂 `pathway_init_sha256` 相等、 `position_scale` 逐字相同、query 清零 ⇒ softmax 权重**精确均匀**且聚合落在 B 的均值的 float32 累加界内（决策 0082）； 失败只记 `identity/engineering unresolved`，**不进入** (a)–(d)。 预算取决策 0060（per-seed 129,600 s / planned 259,200 s / hard 518,400 s，R2 实测每 seed 88,362–89,520 s）； 每轮只跑一个臂、GPU 由统一父调度按共驻余量串行、每阶段 spawn 前只读 `gpu_gate`、不对邻居进程发任何信号； 不读 2023 test（`r=0`、`test_read=false`、`scientific_claim=false`）；软超继续并记 `soft_overrun`， 仅 hard 截止截断 attempt；不在运行中改硬限；不新增下载、不重建 store、不 main 合并 / force / 关 issue、 不合成数据、不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成。

（objective 实测字符数 **1733**，由 `tools/check_goal_brief.py` 在提交前核验。）

## §1 起点与唯一改动因子

- **起点证据**：`docs/R7_S3_TYPED_SPATIAL_RESULT.md`（旧臂，已按预声明关闭）、
  决策 0078（准入）/ 0079（问法收窄 + 机制否证）/ 0080（执行设计）/ 0081（常数定值）；
  新制度基准 `docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md`（R2，同为 65×65 多年、`full_grid`）。
- **唯一因子**：`typed_evidence_mode` 的聚合方式——B=`typed_routing`（pooled 均匀均值）vs
  C=`typed_spatial`（softmax over P，带冻结 `position_scale`）。两臂都**启用** typed evidence、
  都带**同一个** `position_scale`、都从**零起点**新初始化。
- **参数差异（已声明，不声称对齐）**：C 恰多 `typed_evidence.position_query`（**+192 参数**，`+0.13%`），
  与旧臂同样的已声明差异继续存在。`position_scale` 是**常数不是参数**（不进 `state_dict`）。
- **与旧臂 H-B 的实质差异**：实例（129×129 → 65×65）、监督域（`interior_32` → `full_grid`）、
  数据跨度（2,140 → 32,142 窗）、算力（2,400 updates batch 1 → 16,071 batch 2）、
  锚表/payload（129×129 侧车 → 65×65 侧车）、**起点**（H-B 从带 typed 的注册父扩张 → 本轮零起点）、
  **前向敏感度**（H-B 的 `scale≡1`，实测熵恰 1.000000 → 本轮 8000）。
  旧臂读数对本轮**没有约束力**，反之亦然；**不得**写「比 H-B 更好/更差」。
- **不得使用的动机**：**禁止**写「旧臂只跑了 1.12 epoch / 欠训练」（决策 0057/0058 明禁）。
  本轮开轮理由是 0078 §Context ② 的**原页自证的弱操纵**与 0079/0081 的**幅度修正**。
- **必须如实写明的**：`scale` 是人造的、`|q|` 从 0 起步、本轮**不**回答「优化器自然会学到多大 query」。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 设计与准入 | 本文件 §3；决策 0078/0079/0080/0081 |
| D2 | 65×65 typed-evidence 侧车（已发布） | `outputs/r7_s3_typed_evidence_fy65_20261103_attempt01/`（`typed_evidence_identity a61d22aa…`、`BUILD_COMPLETE`、`data_identity 81307008…`） |
| D3 | 常数定值探针 | `outputs/r7_s3_position_scale_probe_20261103_attempt01/scale_probe.json`（SHA256 `e34ccedc…`） |
| D4 | 父迁移（新修订） | `outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`（3 seed、155 张量 `state_digest` 与 R2 迁移逐位相同） |
| D5 | 新驱动/support/测试 | `scripts/study_r7_s3_position_retention_65x65.py`、support、`tests/test_r7_s3_position_retention_65x65.py` |
| D6 | 新协议/输出/code.zip | `preparation_protocol.json`、`prepared_protocol.json`、`code.zip` + `archive_receipt.json` |
| D7 | 两臂三 seed 真实训练与评分 | `seed{41,42,43}_{control,candidate}_receipt.json`、每 seed 每臂五 lead `boundary_rmse.csv` |
| D8 | 配对读数与预声明判定 | `readings.json`（C−B、`full` skill、桥 (e)、出口 (a)–(d)） |
| D9 | 登记 | `docs/R7_S3_POSITION_RETENTION_RESULT.md`、index record、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结；科学门形式只引 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`，本轮不触碰任何阈值、不宣布通过）

**判据来源**：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`（科学门形式）、`docs/plans/0025-training-speed-observability-axis-reeval.md` §4.2（队列位置）、决策 0078/0079/0080/0081；本轮不新增、不放宽任何阈值。

- **新制度**：store `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`（`[34084,17,65,65]`，
  train 2000–2021 / val 2022 / test 2023 封存）；两臂训练清单 = treatment **32,142** 窗；
  val 2022（1,458 窗）；`16,071 × 2 = 32,142 ≥ 32,142` ⇒ 每臂 ≥1 epoch。
- **评分域**：整张 65×65，`boundary_margins=(1,)`，读 `boundary_rmse.csv` 的 `full`（4,225 点）。
- **气候态分母**：48 桶 train-only（与 R2 同一份，`climatology_reference_sha256 096c384a…`）。
- **其余逐字沿用注册配方**：K=4、depth=4、dim=192、LR 2e-5、warmup 10、
  物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、FP32、clip 1、seeds 41/42/43。
- **typed evidence（本轮新增通路）**：`typed_evidence_mode` 两臂分别为 `typed_routing`/`typed_spatial`；
  payload = **65×65 侧车的 `field_*`** + **store 自己的 `normalization_*` 作为 `denorm_*`**；
  **两臂都带 `position_scale: 8000.0`**。两臂在 `isolated_stream()` 下新初始化，
  `|q|==0`（`typed_spatial` 的 `position_query` 是 `torch.zeros`）。
- **父**：`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`（新 `model_code_sha256`
  `59340710…` 下重跑迁移，155 张量 `state_digest` 与 R2 迁移**逐位相同**）。
- **正/负控制（逐 seed 断言，失败即 `raise`）**：① 两臂共享张量 `pathway_init_sha256` 相等；
  ② 两臂 `expanded_state_sha256` 相同、`anchor_table_sha256` 相同；
  ③ `zero_query_is_pooled`（**决策 0082 改写**；原文「`zero_query_max_abs == 0`」数学上不可满足）：
  拆成 (i) `uniform_weights_exact`——C 的 query 清零后 softmax 权重**逐位等于** `1/P`（精确断言）；
  与 (ii) 该聚合与 B 的 `value.mean(dim=1)` 之差 `≤ 2·P·eps_f32·max|v|`，实测值与界一并记录
  （`zero_query_max_abs` 是**测量值**，不是判据）；
  ④ `position_scale` 两臂逐字相同；⑤ 两臂初始 `|position_query| == 0.0`。
- **跑前熵判据（决策 0081）**：以真实 val 批次实测的归一化熵落在 **`[0.90, 0.99]`** 为准；
  **`熵 < 0.5` 视为退化成 one-hot，该轮读数不得用于判定原假设**。`scale=8000` 在实测 `|q|` 范围
  与三个 init seed 上都留在带内。**运行中不改 `scale`、不扫。**
- **预声明出口（冻结后不放宽）**：
  - (a) C 的 **72h** t2m skill 在 seed 41/42/43 **全部 > 0**；
  - (b) C−B 的 **72h** t2m ΔRMSE **三 seed 全部 < 0**；
  - (c) C 的 **48h** t2m skill **三 seed 全部 > 0**；
  - (d) C−B 在 **6/12/24h** 的 **seed 均值** ΔRMSE **≤ 0**。
  全真 → **只提请独立复核**，不自动进 S4；任一假 → **永久关闭「位置保留」这一支**：
  不扫 `scale`、不加 seed、不换聚合族、不改表粒度、不换名重开。
- **桥 (e)，非科学判定（按决策 0080 §Decision 4 重写、0082 修 ③）**：四条恒等式（见 §3 正/负控制 ①③④⑤），
  其中 ③ 按 0082 拆成「权重精确均匀」+「聚合在 float32 累加界内」，两者都逐 seed 断言。
  失败只记 `identity/engineering unresolved`，**不进入** (a)–(d)。
  **本轮的桥与旧臂的桥不可比**：H-B 的桥断言「迁移张量 SHA 等于归档」，本轮的桥断言「零起点 + 共享初始化」。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

### §3.1 并行准入七 gate

| gate | 判定 | 依据 |
| --- | --- | --- |
| 假设独立 | **过** | 唯一因子是聚合方式，且**操纵强度已被修正**（`scale` 8000 使熵离开 1.0）；与 R2 的锚反馈是不同因子，与已关闭的 H-C 是不同机制。0078 已逐条论证「不是同一族」 |
| 产物隔离 | **过** | 全部产物写**全新排他** `outputs/r7_s3_position_retention_65x65_<date>_attemptNN/` |
| 计算资源 | **过（必须串行）** | 照 R2 实测（训练 70.4065 + 评分 4.7009 ≈ 75.1 GPU-h / 6 臂）；双卡共驻、每卡峰值 5.213 GiB ≪ 11.7 GiB；本臂**模型零新增大张量**（+192 参数） |
| 数据实例 | **过** | 只读 treatment/val 清单、已发布侧车、迁移父；不写 `data/`、不建 store、不新增下载 |
| 可变状态 | **过** | 本轮可变状态全在自己的 attempt 目录内 |
| 可独立验证 | **过** | `protocol_sha256` + `code.zip` SHA + 逐 seed×两臂 receipt（含 `pathway_init_sha256`/`position_scale`/`zero_query_max_abs`）+ `readings.json` + 逐 lead CSV |
| 多重比较 | **过（受纪律约束）** | 一条轴、一个候选臂、三个 seed、四个**预声明**出口；冻结后不放宽；**不扫 `scale`**；test 未读（r=0） |

## §4 数字预算与停止

- 训练：照 R2 实测 70.4065 GPU-h（6 臂）；评分 4.7009 GPU-h；合计 ≈ **75.1 GPU-h**。
- `planned_seconds 259,200`（72 h 墙钟）/ `hard_cap_seconds 518,400`（144 h）/ `per_seed 129,600`；
  软超继续并记 `soft_overrun`，仅 hard 截止截断。
- 停止条件：预声明出口全部判定完即停；任一假即关闭该支，不追加 seed/剂量/常数扫描。

## §5 明确不做

- **不扫 `scale`**、不改 `position_scale` 常数、不以「换常数」重试。
- 不迁移 H-B 的训练后 query（路线①已由 0080 否决）。
- 不做 `interior_32`（65×65 上只剩 1×1）、不改评分域、不改气候态分母。
- 不加 seed、不改 updates、不与已关轴组合、不换名重开。
- 不读 test、不宣布科学通过、不称 SOTA/泛化、不自宣最终 goal 完成。

## §6 进度与交接

- **已完成（2026-11-03）**：① 决策 0078（准入）/ 0079（问法收窄 + 机制否证）/
  0080（执行设计：路线② + 侧车重算 + `position_scale` 字段 + 桥重写）/ 0081（常数 8000）；
  ② 65×65 typed-evidence 侧车**已发布**（`typed_evidence_identity a61d22aa…`，loader 校验通过）；
  ③ `model/typed_evidence_r7.py` 加 `position_scale`（默认 `1.0`，**实测逐位不变**、**零新参数**）、
  `model/process_forecast_r7.py` 的字段集断言改为「七项 + 可选 `position_scale`」；
  ④ `model_code_sha256` 由 `a38be16c…` 移到 **`59340710…`**；
  ⑤ 父钉按 0075 重绑：`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`，
  3 seed、155 张量 `state_digest` 与 R2 迁移**逐位相同**、源 checkpoint SHA 相同；
  ⑥ `scale` 探针实测（`scale_probe.json`，SHA `e34ccedc…`），指出 0078 的 `[1e3,2e3]` 仍近均匀。
- **冻结前更正（2026-11-03，决策 0082）**：写测试时发现 0080 §Decision 4 的桥 (e)④
  「零 query 聚合**逐位等于** `value.mean(dim=1)`（`zero_query_max_abs == 0`）」**数学上不可满足**——
  `_position_attention` 先乘 `1/P` 再求和，`mean(dim=1)` 先求和再除 `P`，实测差 `2.98e-08`（64 seed 最坏）。
  驱动里该判据写作 `if difference != 0.0: raise`，**会在第一个 seed 训练前中止整轮**。
  **本轮当时尚未冻结、尚未开跑**，故在冻结前改写：拆为「权重逐位等于 `1/P`（精确）」+
  「聚合差 ≤ `2·P·eps·max|v|`（实测占用 `2.4e-05`）」，两者都逐 seed 断言。ADR 0082 记录其原因、
  实测与三个被否决的备选。
- **待做（顺序固定）**：`round-guard` → 提交推工作分支等 CI 绿 →
  重建 code zip → `freeze` → `stage prepare/archive/stage-seeds/reading` → 四件套登记。
- **CI 安全**：测试**不得**依赖 `outputs/`。
- **第一条命令**：`.venv/bin/python tools/r7_round.py round-guard`。
- **本轮尚未开跑**：0 GPU-h；无读数、无结论。

## §7 下一动作

- 写 `scripts/study_r7_s3_position_retention_65x65.py`（参照 R2 驱动）+ support + 测试。
- 与 S4 的关系：本轮**不推进到 S4**；S4 已按决策 0066 出过判决，不改它。
- 预声明出口见 §3：**(a)–(d) 全真 → 提请独立复核；任一假 → 永久关闭「位置保留」支**。
