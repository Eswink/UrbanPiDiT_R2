# 0025 — 训练提速、可观测性修复与已关闭轴的重估

- **日期**：2026-10-23
- **状态**：计划已定稿并归档；**W1/W2 已落地并逐项验证、§5 探针已实测、W3 队列经决策 0056 判定不开轮**（2026-10-24，见文末「实际结果」）
- **范围**：S3 节点内的工程提速 + 可观测性修复 + 已关闭单因素轴的重估队列
- **依据**：本轮实测（`outputs/r7_s3_fy_year_coverage_20261023_attempt02` 的逐阶段回执）；
  决策 0048/0050/0054/0055；`docs/goals/s3-year-coverage.md`；`AGENTS.md` 的硬约束与七 gate
- **本文件不含任何科学结论**；`scientific_claim: false`、`test_read: false`

---

## §0 摘要

三件互不重叠的事，按依赖顺序排：

| 工作流 | 内容 | 科学风险 | 身份代价 |
| --- | --- | --- | --- |
| **W1** | 去掉重复的验证/构造开销（约 47% 墙钟） | **零**（登记数字逐位不变） | 改 `training_code_sha256` / `execution_files_sha256` |
| **W2** | 补上训练进度与阶段台账 | **零** | W2a/W2b 改身份；W2c 零代价 |
| **W3** | 重估已关闭的单因素轴 | **高**（需新 ADR + 新 brief + 预声明出口） | 新实例、新协议 |

**本次交付**：本文件（提交归档）+ §10 的改写提示词（用户在另一窗口执行）。

---

## §1 硬时序约束（最先读这条）

编辑 `training/**`、`model/**`、`data/**` 或 `scripts/study_r7_s3_*.py`，会让正在跑的
`attempt02` 的 `reading` 阶段以 **「execution source changed」** 失败：

- `tools/r7_round_stages.py` 的 `worker()` 在分发任何阶段前先调 `validate_protocol(output)`；
- 该函数在 `scripts/study_r7_s3_fy_year_coverage.py:332-336` 逐文件比对 `execution_files_sha256`；
- 而 `execution_files()`（同文件 `:95-96`，经 `wide.execution_files()`）**通配了整个
  `training/` + `model/` + `data/` 目录树**（排除 legacy）加上各 study 入口脚本。

**结论**：W1/W2a/W2b 的代码改动**全部排在 attempt02 收尾之后**（预计 2026-10-24 09:00 左右）。
**例外**：`tools/` 不在 `execution_files()` 内 ⇒ 改它零身份代价，是唯一可以立刻动的区域（W2c）。

---

## §2 W1 — 提速

### W1a 删掉丢弃式校验（最大单项，约 25% 墙钟）

**落点**：`data/r7_long_rollout_dataset.py:63`

```python
target = dict(record, target_index=target_index, target_time=target_time, lead_time_hours=lead)
# Validate every available train target, even after an earlier gap.
validate_record(root, target)          # ← 返回值被丢弃
targets.append(target)
```

**为什么是纯浪费**：`:69-77` 构造 `pins` 与 `identity` 时只取 `target_index` / `target_time`，
**从不使用 `validate_record` 的返回值**。而 `target_index` 来自 `lookup.get(stamp_ns)`、
`target_time` 来自同一个 `stamp_ns` 的 `pd.Timestamp(...)` ⇒ 这个校验在构造上近乎同义反复。

**规模**：每窗调用 1（`:37`）+ 12（`:63`）= 13 次；治疗臂 32,142 窗 ⇒ 约 **41.8 万次**。

**实测依据**：控制臂 7,291 窗的清单校验实测约 **26 min**（`attempt02` 16:08→16:34）
⇒ 每调用约 0.27 ms；外推治疗臂 41.8 万次 ≈ **113 min**。
删掉 12/13 的调用 ⇒ 控制臂约 2 min、治疗臂约 9 min，**每 seed 省约 125 min**。

**保留**：`:37` 的调用必须留下——它返回 `indices`（`:40-41` 使用），且做 manifest↔store 的绑定校验。

**验收判据（硬）**：改完后对**现有** manifest 重跑 discovery，
`window_sha256` / `usable_windows` / `excluded_sample_ids` / `windows` pins **必须逐字节不变**。
先在 CPU 上跑通并留下 digest 比对证据，才允许进任何新 attempt。
**不通过就回退，不得"更新 digest 让它过"。**

**身份代价**：该文件在 `training_code_digest` 的 `SOURCE_FILES` 内
（`training/r7_long_rollout_runner.py:34-38`）⇒ `training_code_sha256` 会变；新 attempt 记录新值即可。

### W1b 评估的每-lead 重建（约 25% 墙钟）

**落点**：`training/r7_evaluate.py:87, 94-103, 165, 170`

```python
reader=ZarrAtmosWindowDataset(manifest)                 # :87  每次重建
for r in reader.records: validate_record(root, r)       # :95-96  1,458 条
ds=ZarrRolloutDataset(store, split=..., lead_hours=...) # :100  枚举全部 34,084 时次
ds.windows=[w for w in ds.windows if ... in allowed]    # :103  再过滤到 1,458
clim=fit_training_climatology(store)                    # :165/:170  与 lead 无关
```

`ZarrRolloutDataset.__init__`（`data/r7_evaluation.py:14-54`）在 `:46` 遍历**整个 store 的时次**，
与 `lead_hours` 无关地重建一遍。而驱动在 `scripts/study_r7_s3_fy_year_coverage.py:462-467`
**每个 lead 调一次** `evaluate_local` ⇒ 每臂 5 次、每 seed 10 次。

**修法**：把 reader、store 句柄、基础枚举、气候态提到**每臂一次**，5 个 lead 复用；
只让 `lead_hours` 影响窗口选择。

**验收判据**：同 checkpoint、同 val，修前/修后 5 个 lead 的
`rmse.csv` / `boundary_rmse.csv` / `climatology_skill.csv` **逐字节一致**。

**身份代价**：`training/r7_evaluate.py` **不在** `SOURCE_FILES` ⇒ 只改 `execution_files_sha256`，
**不改 `training_code_sha256`**。

### W1c `validate_protocol` 反复哈希 5.49 GB `source.nc`

**落点**：`scripts/study_r7_s3_fy_year_coverage.py:341` 的 `sha256_file(SOURCE)`，
每轮被调 5 次（archive + 3 seed + reading）。改为 prepare 时算一次并记进协议，
后续只比对记录值 + 校验文件 size/mtime。小项，零风险。

### W1d（默认不做，先测后定）

seed 阶段复用 prepare 已冻结的 `arm["train_windows"]["window_sha256"]`。
但 W1a 落地后 discovery 本身应变成秒级 ⇒ **先实测 W1a 之后的耗时，只有仍显著才引入缓存**。
不为没测量过的东西做优化。

### W1 预期收益

每 seed 约 507 min → 约 270 min（**约 −47%**），轮墙钟约 19 h → 约 10 h。
**这是估计，必须由 W1a/W1b 落地后的实测替换；不得把估计写成实测。**

---

## §3 W2 — 可观测性

### W2a 训练进度旁路（不改 `training_report.json`）

**落点**：`training/r7_long_rollout_runner.py` 的 `_run_updates`（append 点 `:265`、
checkpoint 分支 `:268`）。

**修法**：每 N 个 update 往 `folder / "progress.jsonl"` **追加一行**
（`update` / `epoch` / `lr` / `loss` / `l6` / `l12` / `gradient_norm` + 时间戳 + 已用秒）。

**不要动** `training_report.json` 的结构与写时机——它是训练结束时的快照
（`:396` `_write_json`，用排他创建），改它会改审计契约。

**顺带修掉一个真缺陷**：失败路径 `:398-403` 只往 `failed_attempt.json` 写一行原因，
**已累积的 `losses` 是 `_run_updates` 的局部变量，随异常一起丢失**。
旁路 jsonl 天然解决"硬截断/异常时整条曲线全丢"。

### W2b 阶段级输出（让 `seed<N>.log` 不再为空）

**落点**：`scripts/study_r7_s3_fy_year_coverage.py` 的 `worker()` / `_seed` / `_train_arm`
（`:566` / `:480` / `:439`）——目前**全函数无一处 `print`**，所以
`tools/r7_round_stages.py:335` 写出的 `prepare.log` / `archive.log` / `seed*.log` 是 **0 字节**
（实测确认）。在阶段边界与每臂/每 lead 起止加 `print(..., flush=True)`，
输出会自动进 `seed<N>.log`。

**身份代价**：该文件在 `execution_files` 内 ⇒ 改协议 hash；**不改 `training_code_sha256`**。

### W2c 恢复阶段台账（零身份代价）

**现状**：`<attempt>.driver.log` 只由旧路径 `run_attempt` 打印
（`scripts/study_r7_s3_long_rollout.py:395` 的 `phase_done` 行），
而当前 `r7_round.py stage` 路径**从不执行它** ⇒ 当前 attempt 连阶段台账都没有。

**修法**：在 `tools/r7_round_stages.py` 的 `run_stage` 里补写 `stage_ledger.jsonl`。
`tools/` 不在 `execution_files` ⇒ **零身份代价，可以立刻做**。

### W2d 风格留痕

`docs/rules/python-implementation.md:5` 把"0 处 logging"记为既有风格（E-133）。
本计划要引入进度输出 ⇒ 应记一条规则/证据更新，说明从"无输出"改为
"结构化旁路 jsonl + 阶段 print"，避免风格与规则脱节。

---

## §4 W3 — 已关闭轴的重估

### §4.1 先澄清一个命名碰撞

`H-D` 在案上有**两个不同含义**：

- **ADR 0048 的 `H-D`** = 加宽主干 `dim` 192→256（容量轴；已关闭；**本次排除**）；
- **架构/训练信号序列的 `H-D`** = 气候态锚异常反馈 `draft − C_valid`
  （`docs/R7_S3_ANCHOR_ISOLATION_RESULT.md:1`）。

本计划内一律写全名，**禁止单用 `H-D`**。

### §4.2 队列

按"关闭类型 × 是否加显存"分层。显存数字取自各页 §6 实测
（基座注册 2400 剂量臂约 8.36 GiB）。

| 序 | 轴 | 关闭类型 | 显存增量 | 重估理由 |
| --- | --- | --- | --- | --- |
| 1 | 气候态锚异常反馈（锚隔离） | **非 null**：五 lead 3/3 seed 配对全改善，72h −0.321 K；只栽在绝对门 2/3 | **+0**（`new_params=0`） | 唯一有正向长 lead 线索的轴 |
| 2 | 全局混合（H-C） | **null**，页内明写"更强操纵/更多数据下未测" | +0.6 GiB（**低于** depth/width） | 吃算力不吃显存；表达力型机制最贴合"数据变大"前提 |
| 3 | 位置保留（H-B） | **null（弱操纵）**：`position_query` 模长 0.0074–0.0125 vs 同层投影 ~8.02，**差 3 个数量级** | +0.2 GiB | 这是**操纵强度缺陷**，不是数据问题 |
| 4+ | 训练信号族：空间梯度损失 / dense 12 步监督 / final-only 草稿监督 / 显式 tendency 输入 | **mixed**（有配对改善但绝对门未过） | ≈+0（模型零改动） | 有真实效应、零显存 |
| **排除** | SAM（15/15 一致更差）/ zero-base decoder（72h +1.37 K）/ plain EMA（方向明确）/ 长 lead 权重 / rollout 剂量 / depth / width | **方向明确的负结果** | — | 不是 null ⇒ "更多数据"不构成翻案理由 |

### §4.3 使能制度不是"更多数据"，而是"数据 × 匹配算力"

案上原文：

- `docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md:175`：「the budget is fixed at 2400 updates:
  the treatment sees about one third of an epoch while the control sees about one,
  so a positive result prices **data diversity at this budget**, not data volume at every budget」；
- `docs/goals/s3-year-coverage.md:14,22-25` 把这一点写成待检验的未决问题；
- 决策 0050 代价 4：「2000–2016 未使用；『更多年份』是后续独立因子，不能与本轮合并解释」。

**探查确认：案上没有任何文档授权"因数据变大而重开旧轴"。** 因此重估的正当性必须由
**新 ADR** 明确论证："在 exposure-matched（每臂 ≥1 epoch）的新制度下，
该假设与已关闭的旧臂**不是同一族**"。

配套：batch/exposure 必须先由 §5 的探针定下来，否则"更多数据"仍会被摊薄成 7.5% 个 epoch。

### §4.4 程序（硬约束，逐条）

1. **新 ADR**（`docs/decisions/NNNN-<slug>.md`，四段齐全，**Consequences 必须含负面后果**，R-033）
   + 更新 `docs/decisions/README.md` 索引；`python tools/check_conventions.py --rule R-033 --rule R-036` 通过。
2. **新 round brief**（`docs/goals/<slug>.md`）带**预声明出口 (a)–(d) + 桥 (e)**，
   格式照 `docs/goals/s3-year-coverage.md:68-79`：全真 → 提请独立复核；任一假 → 永久关闭该支。
3. 过并行准入**七 gate**（`AGENTS.md:26`：假设独立 / 产物隔离 / 计算资源 / 数据实例 /
   可变状态 / 可独立验证 / 多重比较）；**每轮只跑一个臂**，GPU 由统一父调度按共驻余量串行。
4. test 全程封存、`r`/alpha 累积纪律照科学合同；先冻 `protocol.json` 再跑（R-006）。
5. **禁止**"换名重开"与"事后放宽已冻结判据"（`AGENTS.md:32`、各结果页"不换名重开"）；
   ADR 必须显式论证"不是同一族"。

---

## §5 batch / 显存探针（先测后定）

**复用先例**：`scripts/probe_r7_b2_throughput.py`（短跑 + 外推，`PROBE_UPDATES=20`，
报 `seconds_per_update` / `peak_allocated_mib` / `peak_reserved_mib`）与
`scripts/bench_r7_ddp_smoke.py`（per-GPU batch）。

**测什么**：在**当前 65×65 多年 store + 当前 spec**（K=4 / depth=4 / dim=192 / 12 步 BPTT）上，
测 **batch ∈ {1, 2, 4, 8}** 的：

- 每 update 秒数；
- `peak_reserved_bytes`；
- 相对共驻门槛的余量（`estimated_peak_bytes = 4096 MiB` + `headroom_margin_bytes = 512 MiB`
  + 邻居预留 6 GiB，见 `scripts/study_r7_s3_fy_year_coverage.py:70,296-298`）。

**边界**：有界探针、不写任何科学产物；按 R-054 只读核余量、**不对邻居进程发任何信号**；
产物落全新排他 `outputs/r7_s3_batch_probe_<date>/`。

**决策点**：拿到实测后才定 batch 取值与是否作为新配方常量——**由用户决定，执行者不自行拍板**。
**batch 与轴重估不要同轮**，否则混淆。

---

## §6 执行顺序

1. **本次**：写本文件（提交归档）+ §10 的改写提示词。
2. **attempt02 收尾前**（可并行、零身份代价）：只读准备、探针协议冻结、ADR/brief 草稿、
   W2c（`tools/` 侧）。
3. **attempt02 收尾后**：W1a → 验 `window_sha256` 不变 → W1b → 验三张 csv 逐字节一致
   → W2a/W2b → 新 attempt 重跑基线（记录新 `training_code_sha256`）。
4. **§5 探针** → 用户定 batch。
5. **W3 队列逐条**：ADR → brief → 冻协议 → 跑 → 登记四件套。

---

## §7 验证

- `python tools/check_conventions.py`（37 条阻断）保持 0 失败。
- W1a 的 `window_sha256` 逐字节不变证据（CPU，现有 manifest）。
- W1b 的三张 csv 逐字节一致证据（同 checkpoint、同 val）。
- W1/W2 落地后的**前/后每 seed 墙钟实测对比**。
- 每次登记后自跑 `tools/r7_round.py verify`。
- 探针报告只写实测值，不写估计值。

---

## §8 明确不做

- 不动 `data/raw|interim|processed`、归档快照、val 2022 / test 2023（**不读 test**）。
- 不 main 合并、不 force、不关 issue、不删任何已有产物。
- 不重开 depth、width、以及方向明确的负结果（SAM / zero-base decoder / plain EMA /
  长 lead 权重 / rollout 剂量）。
- 不在 attempt02 收尾前编辑 `training/**`、`model/**`、`data/**`、`scripts/study_r7_s3_*.py`。
- 不自行宣布科学通过、不称泛化 / SOTA。

---

## §9 风险

- **R1（高）**：W1a 若改变 discovery 的结果（哪怕一个窗口）就会改 `window_sha256`
  ⇒ 必须逐字节比对；不通过就回退，**不得"更新 digest 让它过"**。
- **R2（中）**：提速后每 seed 墙钟缩短，而 `per_seed_seconds` / `planned` / `hard` 是协议常量
  ⇒ 新 attempt 需按实测重定预算（照决策 0055 的先例：用实测定、不拍脑袋）。
- **R3（中）**：H-B 的重估实为"操纵强度修复"而非数据问题 ⇒ ADR 必须写清它与另三条理由不同，
  否则会被读成"换名重开"。
- **R4（中）**：4+ 条轴 × 3 seed = 12+ 个新比较 ⇒ 必须预先声明出口并守住 `r`/alpha 纪律。
- **R5（低）**：`training_report.json` 若被误改成流式写，会改审计契约 ⇒ 明确只加旁路 jsonl。

---

## §10 改写提示词（本次交付的第二件产物，供拷入另一窗口）

```text
在 /data/esw/UrbanPiDiT_R2（分支 r7/weather-reasoning）执行已归档的计划
docs/plans/0025-training-speed-observability-axis-reeval.md。先完整读它，再按 §6 的顺序做。

先核状态：git rev-parse HEAD、tools/r7_round.py state、以及
outputs/r7_s3_fy_year_coverage_20261023_attempt02 是否已收尾（阶段台账/回执齐）。

硬时序：编辑 training/**、model/**、data/** 或 scripts/study_r7_s3_*.py 会让
attempt02 的 reading 阶段以 "execution source changed" 失败。attempt02 收尾前只许动
tools/ 下的东西（W2c）与只读准备；其余代码改动全部排在它收尾之后。

W1 提速（每个改动都要给出验收证据）：
- W1a 删 data/r7_long_rollout_dataset.py:63 那个返回值被丢弃的 validate_record
  （12×/窗，约 41.8 万次调用；:37 必须保留）。验收：对现有 manifest 重跑 discovery，
  window_sha256 / usable_windows / excluded_sample_ids / windows pins 逐字节不变。
  不通过就回退，不得更新 digest 让它过。
- W1b 把 training/r7_evaluate.py 里每个 lead 重建的 ZarrAtmosWindowDataset、
  ZarrRolloutDataset（data/r7_evaluation.py:46 枚举全部 34,084 时次）、
  fit_training_climatology 提到每臂一次、5 个 lead 复用。验收：同 checkpoint 同 val，
  5 个 lead 的 rmse.csv / boundary_rmse.csv / climatology_skill.csv 逐字节一致。
- W1c scripts/study_r7_s3_fy_year_coverage.py:341 的 sha256_file(SOURCE)（5.49 GB）
  每轮算 5 次，改成 prepare 算一次记进协议、后续比对记录值 + size/mtime。
- W1d 默认不做：W1a 落地后先实测 discovery 耗时，仍显著才考虑 seed 阶段复用。

W2 可观测性：
- W2a training/r7_long_rollout_runner.py 的 _run_updates 每 N 个 update 追加一行到
  folder/progress.jsonl。不要动 training_report.json 的结构与写时机。失败路径
  (:398-403) 目前丢已累积的 losses，用 jsonl 一并解决。
- W2b scripts/study_r7_s3_fy_year_coverage.py 的 worker()/_seed/_train_arm 加
  print(..., flush=True)，让 prepare.log/archive.log/seed*.log 不再是 0 字节。
- W2c 在 tools/r7_round_stages.py 的 run_stage 补写 stage_ledger.jsonl（零身份代价）。
- W2d 按 docs/rules/python-implementation.md:5 的"0 处 logging"风格记一条更新。

W1/W2 完成后：重跑一次基线 attempt，记录新的 training_code_sha256，并报出前/后
每 seed 墙钟的实测对比。

§5 batch 探针：复用 scripts/probe_r7_b2_throughput.py 的形态，在当前 store 与当前
spec 上测 batch ∈ {1,2,4,8} 的 seconds_per_update 与 peak_reserved_bytes，并算出
相对共驻门槛（estimated_peak 4096 MiB + margin 512 MiB + 邻居预留 6 GiB）的余量。
产物落全新排他 outputs/r7_s3_batch_probe_<date>/，只写实测值。测完把数字报给我，
batch 取值由我定，不要自行拍板。

W3 轴重估：按 docs/plans/0025-training-speed-observability-axis-reeval.md §4 的队列
逐条做，每条都要新 ADR（四段齐全、Consequences 含负面后果，R-033）→ 新 brief 带预声明
出口 (a)-(d)+桥(e) → 过七 gate → 冻 protocol.json → 跑 → 登记四件套。
每轮只跑一个臂。ADR 必须显式论证"与已关闭旧臂不是同一族"。
注意 H-D 有两个含义（0048 的宽度 vs 锚反馈），文中一律写全名。
排除项（SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量 /
depth / width）不在队列内：它们是方向明确的负结果，不是 null。

纪律：不动 data/raw|interim|processed 与归档快照；不读 test（test_read=false）；
不 main 合并 / force / 关 issue / 删已有产物；不合成数据；不跳过或弱化任何测试与断言，
禁止事后放宽已冻结判据；GPU 默认共驻、按 R-054 只读核余量、不对邻居进程发任何信号；
外部检索走 arxiv-researcher / web-researcher 两条通道（主链路 WebSearch/WebFetch 被 R-049 拒绝）。
每一步都要给可核查证据（命令 + 输出 + digest），不要把估计值写成实测值。
收尾按 AGENTS.md 报告：改动 / 实跑过的验证 / 接口变化 / 数据与结果影响 / 兼容性风险 /
依赖影响 / 未做的事 / 下一项。
```

---

## 实际结果

**执行时间**：2026-10-24（分支 `r7/weather-reasoning`）。**代码 commit**：`b613f8e`（W1/W2 实现）、
`ee54be0`（W3 的 ADR 0056）、`b902c94`（W1c 回归守卫测试）。**本节所有数字均为实测**，
估计与实测分开标注；未跑的一律写「未做」。

### W1 提速 —— 四项都落地，逐项有验收证据

| 项 | 改动 | 验收证据 | 结果 |
| --- | --- | --- | --- |
| W1a | 删 `data/r7_long_rollout_dataset.py:63` 被丢弃的 `validate_record`（`:37` 保留） | 对**现有** manifest 重跑 discovery，与 `attempt02/protocol.json` 冻结值逐字段比对 | **PASS**：两臂的 `window_sha256` / `usable_windows` / `excluded_sample_ids` / `windows` pins **全部逐字节相同**（控制 `b778893f…`、治疗 `e8f22c30…`） |
| W1b | `RolloutStoreScan` + `EvaluationPreparation`：reader / store 扫描 / 训练气候态每臂一次、5 lead 复用 | ① 数据集层：新路径与原 `__init__` 的窗口推导逐 lead 比对；② 端到端：同 checkpoint（`c80ccba5…`）同 val 重评 5 lead，与 `attempt02` 回执里的 CSV 哈希比对 | **PASS**：① 五个 lead 的 `windows` 逐字节相同且 cohort = 冻结 `VAL_COHORTS`（1458/1457/1455/1451/1447）；② 5 lead × 3 张 CSV（`rmse` / `boundary_rmse` / `climatology_skill`）**15/15 逐字节相同** |
| W1c | 5.49 GB `source.nc` 只在 prepare 哈希一次；后续阶段比对记录值 + size/mtime | 归档阶段耗时对比；AST 回归守卫（`tests/test_r7_s3_fy_year_coverage.py`） | **PASS**：archive 阶段 **20.09 s → 6.36 s**（≈ 一次热哈希 14.7 s）；守卫断言 prepare 仍哈希、后续阶段不再读 |
| W1d | **未做**（按计划「先测后定」） | W1a 后实测 discovery：控制 **113 s**、治疗 **464 s**（合计 577 s，W1a 前约 6,664 s） | 实测已降至原 ~8.7%；本轮**未**引入 seed 阶段缓存 |

### W2 可观测性 —— 四项都落地并在真实轮次上验证

| 项 | 改动 | 证据 |
| --- | --- | --- |
| W2a | `_run_updates` 每 20 个 update 追加一行 `progress.jsonl`；`training_report.json` 结构与写时机**未动**；失败路径记 `progress_file` / `progress_updates` | 真实轮次 `attempt03/seed41|42/training/*/progress.jsonl` 存在且字段齐全；新增 2 个测试（含「失败时曲线只剩 sidecar 且可追溯」） |
| W2b | 驱动 `worker()/_seed/_train_arm` 加 `print(..., flush=True)` | `prepare.log` **0 → 156 B**、`archive.log` **0 → 133 B**、`seed42.log` **0 → 1,605 B**（含每臂/每 lead 起止） |
| W2c | `tools/r7_round_stages.py` 的 `run_stage` 补写 `stage_ledger.jsonl`（零身份代价） | `attempt03/stage_ledger.jsonl` 逐阶段一行，含 `device` 与 `elapsed_seconds`；新增 1 个测试（含「追加而非截断」） |
| W2d | `docs/rules/python-implementation.md` 记「仍然 0 处 `logging`，但长跑必须留结构化进度」；`CHANGELOG.md` 记本轮规则侧变化 | E-133 继续成立（未新增 `logging`） |

### 回归与纪律

- `pytest tests/`：**5,155 passed / 3 skipped / 0 failed**（3 个 skip 是既有的「干净检出无真实数据 fixture」，
  非本轮引入）。`tools/check_conventions.py`：**37 条阻断规则 0 失败**。`tools/r7_round.py verify`：**passed**。
- 一次真实的行为回归被既有测试抓住并**修好而不是放宽**：`test_evaluator_deadline_checks_before_setup_climatology_and_samples`
  断言「气候态拟合之前必须有一次 deadline 检查」。W1b 初版把拟合移进构造函数，破坏了这条契约；
  改为**惰性拟合**（`EvaluationPreparation.climatology(deadline)`）后该测试原样通过，契约未被放宽。

### 基线重跑（`attempt03`，同轮同 store 同 seed）

| 阶段 | `attempt02` | `attempt03` | 变化 |
| --- | ---: | ---: | ---: |
| prepare | 7,421.6 s | **601.2 s** | **−91.9%** |
| archive | 20.1 s | **6.4 s** | **−68.3%** |
| seed 41 | 29,583 s (8.22 h) | **18,026 s (5.01 h)** | **−39.1%** |
| seed 42 | 30,243 s (8.40 h) | **17,503 s (4.86 h)** | **−42.1%** |
| seed 43 | 29,801 s (8.28 h) | **18,434 s (5.12 h)** | **−38.1%** |
| reading | 23.8 s | 未跑（见下） | — |
| **process 合计** | **97,093.7 s = 26.970475 GPU-h** | **54,570.1 s = 15.158375 GPU-h** | **−11.8121 h（−43.8%）** |
| 墙钟（首 spawn → 末 reap） | 59,803 s (16.61 h) | **35,939 s (9.98 h)** | **−39.9%** |

- 三个 seed 的 soft/hard overrun 均为 **0**、状态全部 `success`；`attempt03` 的 seed 43 落在
  `GPU-9d1624af…`。省下的时间来自原先**不可见**的 setup（W1a 的 discovery、W1b 的每-lead 重建、
  W1c 的重复哈希），不是把计算变少——见下条「不变的量本身也是证据」。

- **新 `training_code_sha256`**：`a03f9ba653cdf205…`（`attempt02`）→ **`96a68d6eb29767f2…`**（`attempt03`）。
  `model_code_sha256` 保持 `60de3b2dc5de2871…`（未改 `model/**`）；`execution_files_sha256` 由 193 项 → 194 项
  （新增 `scripts/r7_s3_fy_year_coverage_climatology.py`）。
- **新协议 `protocol_sha256`**：`d8739550e0c86ac4…`（`attempt02` 为 `fdfb56aa3b9a0bc5…`）。
- **身份逐字不变**：两臂 `train_data_identity`、两臂 `window_sha256`、`climatology_reference`、
  `model_code_sha256` 与 `attempt02` **完全相同** ⇒ W1a/W1b/W1c 没有改动任何被冻结的数据身份。
- **不变的量本身也是证据**：回执里 `report["elapsed_seconds"]`（纯更新循环）与逐 lead 的
  `elapsed_seconds`（纯评分循环）前后几乎相同（控制训练 4,472 → 4,551 s、评分循环 2,669 → 2,733 s），
  说明省下的时间**全部**来自原先不可见的 setup，而不是把计算变少。
- **`attempt03` 的 reading 阶段故意未跑**：该支已按预声明出口永久关闭，本轮只为测墙钟与记录新代码身份，
  **不重算 (a)–(d) 出口、不登记新证据记录**。

### §5 batch / 显存探针（实测，非估计）

`outputs/r7_s3_batch_probe_20261024_attempt01/`（`batch_probe.json` 原始读数 +
`batch_probe_margins.json` 的余量算术）。当前 store + 当前 spec（K=4/depth=4/dim=192/12 步 BPTT），
每档 25 个 update、丢弃前 5 个预热后取稳态：

| batch | seconds_per_update | 相对 batch=1 | peak_reserved | 门槛要求 | 余量（对探针开始时实测空闲 24,087 MiB） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **1.839** | 1.000× | 2,322 MiB | 10,752 MiB | **+13,335 MiB（过）** |
| 2 | 2.351 | 1.278× | 4,528 MiB | 11,184 MiB | **+12,903 MiB（过）** |
| 4 | 3.360 | 1.828× | 8,950 MiB | 15,606 MiB | **+8,481 MiB（过）** |
| 8 | 5.620 | 3.057× | 18,092 MiB | 24,748 MiB | **−661 MiB（不过）** |

门槛口径照轮次冻结值：`required = max(estimated_peak 4096 MiB, measured_peak) + margin 512 MiB + 邻居预留 6 GiB`。
**吞吐**（样本/秒）随 batch 上升但次线性：0.544 / 0.851 / 1.190 / 1.423 ⇒ batch 8 只有 batch 1 的 2.6 倍吞吐
却要 7.8 倍显存。**batch 取值由用户决定，执行者不自行拍板**；探针只写实测值。
（探针自报的 `gate_margin_mib` 用的是**运行后**的空闲内存、当时仍持有本次预留，故不是共驻余量；
`batch_probe_margins.json` 已按**运行前**实测空闲重算并标注这一更正。）

### W3 轴重估 —— 队列未开轮，理由是计划自己写的准入论证按实测不成立

> **更正（2026-10-25，决策 0057/0058）：本节结论已作废。** 决策 **0057** 取代 **0056**：
> §4.3 的使能制度是「**数据 × 匹配算力**」的**合取**，0056 只核了算力那一半（旧臂 1.12 epoch
> 这一实测保留、结论作废），§4.4.5 禁的是「换名重开」而非制度变更，§4.3 要求的是**去做论证**
> 而非「论证不顺就永久不开」。⇒ §4.2 队列**按序可执行**；**① 气候态锚异常反馈已按决策 0058 开轮**
> （brief：`docs/goals/s3-anchor-anomaly-feedback-reeval.md`，新制度 = 65×65 多年 × `full_grid`
> × 每臂 ≥1 epoch，锚表改 48 桶，预声明出口 (a)–(d) + 恒等桥 (e)）。下面这段**原样保留**，
> 以便看到判据是怎样被读窄又怎样被更正的。

**决策 0056**（`docs/decisions/0056-closed-axis-reevaluation-admission.md`）记录了完整论证，要点：

- §4.3 把使能制度写成「**数据 × 匹配算力**」，§4.4 要求新 ADR 论证「在 exposure-matched（每臂 ≥1 epoch）
  的新制度下，该假设与已关闭的旧臂**不是同一族**」，并明写「**论证不成立则不得开轮**」。
- **「匹配算力」不是差别**：实测六条已关闭臂的冻结 `protocol.json`——`input_windows=2360`、
  `usable_windows=2140`、`updates=2400`、`batch_size=1` ⇒ **每臂 1.12 个 epoch**，且 `region`
  （129×129、`interior_32` 4,225 格、`mask_sha256 4a2a576b…`）逐字相同。它们**本来就在 exposure-matched 制度下跑过**。
- **「更多数据」在这些臂的实例上不存在**：多年数据只有 **65×65**（`[34084,17,65,65]`）；
  wide 实例实测 `(3360,17,129,129)`，**没有 129×129 多年源**（§8 亦禁新下载）。要「加数据」只能换到 65×65，
  那**同时**改网格（129→65）与监督域（`interior_32`→`full_grid`）⇒ 与已关闭臂**不可直接比较**，
  属 §4.4 第 5 条禁止的「换名重开」形态。
- ⇒ **四条队列（气候态锚异常反馈 / H-C / H-B / 训练信号族）本轮都不开轮，0 GPU-h**；
  §4.2 的排除项（SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量 / depth / width）继续排除。
  **未写新 brief、未冻新 protocol.json、未跑、未登记**——因为开轮的前置条件（ADR 论证）不成立。
- **留给用户的决定**（属总体研究方向，决策 0030）：若要在 65×65 多年实例上重问这些机制，
  必须**按新假设立项**（新 ADR 论证它是**不同**的假设），并如实接受「与已关闭臂不可比较」这一代价。
- **本条结论已被 0057 推翻（见上方更正框）**：0058 正是按「新假设立项」做的——它论证了
  在 65×65 多年 × `full_grid` × ≥1 epoch 的新制度下，① 与已关闭旧臂**不是同一族**（实例/网格/
  监督域/数据跨度全变；旧臂的关闭判据 72h 绝对门在新实例上无锚臂已 3/3 过；暴露 1.12 → ≥1 epoch），
  并如实声明与旧臂**不可直接比较**这一代价。② 尚未开轮，**batch 取值仍是用户的决定**（计划 §5）。

### 与计划的差异

1. **W1d 未做**（计划本就默认不做）：W1a 后 discovery 已降到 577 s，未引入 seed 阶段缓存。
2. **R-051 迫使一次拆分**：`scripts/study_r7_s3_fy_year_coverage.py` 落地时**正好卡在 600 行硬限**上，
   而 W2b 要求往里加 `print`。按既有先例拆出 `scripts/r7_s3_fy_year_coverage_climatology.py`
   （D2 气候态参考 + 其五个 lead 常量），驱动 600 → 591 行；被拆走的常量由驱动 import 回来，
   `driver.<name>` 的既有引用（含测试）不变；新模块已加入该轮 `ENTRY_FILES`。
   **未改阈值、未新增或放宽冻结例外清单**（`docs/rules/CHANGELOG.md` 留痕）。
3. **W1b 的实现与计划描述有一处不同**：计划说把三样都「提到每臂一次」。实测三者代价差三个数量级——
   训练气候态 **~730 s**、`validate_record` 循环 **~19 s**、store 扫描 **~2 s**——因此**气候态改为惰性拟合**
   （为保住 deadline 契约），另两样在 `EvaluationPreparation.__init__` 里一次建好。
4. **§5 探针的余量口径更正**：探针自报的 `gate_margin_mib` 用运行后空闲内存，会把自己的预留算进去；
   已按运行前实测空闲重算并存为 `batch_probe_margins.json`（原始读数不动）。
5. **重跑用了新的 code archive**：轮次 round-spec 里 `code_archive` 指向 `attempt02` 的归档
   （ZIP comment 是旧 commit `14fb768`），归档阶段会拒绝 commit 不匹配的包，而覆盖它会毁掉
   `attempt02` 的可重放归档。故新建 `outputs/r7_s3_fy_year_coverage_code_w1w2_20261024.zip`
   （1,586 条目、comment `b613f8e…`），**未动任何既有产物**；该 zip 的构建脚本与其余验证脚本
   一并留存在 `outputs/r7_w1w2_verification_20261024_attempt01/`。

### 遗留

- ① W3 的四条轴**未**开轮（决策 0056），是否另立新假设由用户决定；
  ② `attempt03` 的 reading 阶段故意未跑，故 `attempt03` 不是一个「完成」的轮次，只是一次工程计时重跑。
