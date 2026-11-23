<!-- round-spec: {"arm": "s3-typed-evidence-in-protocol", "driver": "scripts/study_r7_s3_in_protocol.py", "output_stem": "r7_s3_typed_evidence_in_protocol_20261120", "control_record": "s3-typed-evidence-budget-partial", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_in_protocol.py", "tests/test_r7_79_typed_evidence.py", "tests/test_r7_process_tensor_diagnostics.py", "tests/test_r7_simultaneous_stats.py"], "commit_paths": ["scripts/r7_s3_in_protocol_protocol.py", "scripts/r7_s3_in_protocol_preflight.py", "scripts/r7_s3_in_protocol_verdict.py", "scripts/r7_s3_in_protocol_assembly.py", "scripts/study_r7_s3_in_protocol.py", "scripts/r7_s3_joint_criterion.py", "tests/test_r7_s3_in_protocol.py", "docs/goals/s3-typed-evidence-in-protocol.md", "docs/plans/0035-typed-evidence-three-arm-in-protocol.md", "docs/decisions/0116-typed-evidence-in-protocol-and-no-truncation-budget.md"], "commit_message": "feat(r7): typed evidence in-protocol on three arms -- mechanism-partitioned families and a probed envelope", "register_paths": ["docs/R7_S3_TYPED_EVIDENCE_IN_PROTOCOL_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-typed-evidence-in-protocol.md"], "register_message": "docs(r7): register the in-protocol typed-evidence readings", "planned_seconds": 210240.0, "hard_cap_seconds": 660000.0, "code_archive": "outputs/r7_s3_typed_evidence_in_protocol_code.zip"} -->

# S3 — typed evidence 的 in-protocol 三臂重训

本文件是**唯一执行源**；机器可读的 `round-spec` 在上方。

## Objective（objective）

> 以 goal 模式执行 `docs/goals/s3-typed-evidence-in-protocol.md`：在 r7/weather-reasoning 分支上，把 #79 的容量匹配三臂（A incumbent / B generic_fusion / C typed_routing）在**同一协议内**从已注册 1600-update 父权重重训 32,132 updates（4 epochs，batch 4，seed 41/42/43），回答此前任何预算都没回答过的 `C − B`，判据是**按机制划分**的两族：族 P（`t500,t850,u850,v850` × 5 lead，20 格，主判据）与族 S（其余 13 通道 × 5 lead，65 格），度量为量纲无关的 Δskill，族内一个 max-statistic 同时区间（α=0.05），下限 0.001，并要求通过格跨 ≥2 lead 且三 seed 同号；`B − A` 只作描述性读数（A 少 148,800 参数，必须同时申报）。**不改 model/ 与 training/**，零父迁移、零 code identity 变更、不做 pinned 复用、不做安慰剂臂、不做 16-epoch 档位。**无截断是本轮硬要求**：包络按 treated 实测速率 1.47 s/upd 的 2.05 倍定为 310,000 s，每个 seed 先跑三臂各 400 updates 的探针用实测速率外推，若需超过剩余包络 60% 即中止并记 refused-insufficient-envelope，任何非正常结束必须写 partial_receipt。跑完即停，不做无人值守续跑。

（字符：本 objective 单段，已实测在 4000 code points 以内。）

## 上下文与开轮依据

前一阶段的事实与取舍全在 `docs/plans/0035-typed-evidence-three-arm-in-protocol.md` 与
`docs/decisions/0116-typed-evidence-in-protocol-and-no-truncation-budget.md`。执行前必读这两份，
其中 ADR 的 D6 说明了**为什么本轮不做安慰剂臂**，以及由此产生的、必须写进 limitations 的归因缺口。

## 判据（训练前冻结进 protocol.json，此后不得改动）

判据的完整依据与取舍见 `docs/decisions/0116-typed-evidence-in-protocol-and-no-truncation-budget.md`
（族划分、度量、α、下限、D6 的归因缺口）与
`docs/plans/0035-typed-evidence-three-arm-in-protocol.md`（成本、实测验证、否决的替代方案）。
本节的规则与那两份文档冲突时，以本文件为准；本文件一旦冻结，其取值不得在看过读数后改动。

### 两个族（按机制划分，不按观测量）

| 族 | 通道 | 格数 | 角色 |
|---|---|---:|---|
| **P** `physical-pathway-inputs` | `t500, t850, u850, v850` | 20 | **主判据** |
| **S** `other-channels` | 其余 13 个通道 | 65 | 次要 |

族 P 的四个通道由 `model/r7_process_tensor_diagnostics.TYPED_EVIDENCE_CHANNELS` 声明，
`typed_local_fields()` 实际消费它们——分区在任何读数之前就已固定。

### 度量与统计

`Δskill_i = (mse_对照,i − mse_处理,i) / D[L,c]`，`D` 是该格气候态 MSE（`rmse_climatology` 平方）。
**正值 = 处理臂更好。** 逐 case 除以固定常数是单调缩放，改不了符号或区间是否含零。

同时区间：`training/r7_simultaneous_stats.simultaneous_paired_interval`，
α=0.05、block_length=28 cases、4,000 次重采样、族内一个 max-statistic 临界值。

### 通过规则（P 与 S 各自适用）

1. ≥1 格的 Δskill 同时区间在**改善方向**排除零；
2. 这些格跨 **≥2 个不同 lead**；
3. 每个通过格的三 seed 均值**同号**；
4. 最大通过格的 seed 均值 `Δskill ≥ 0.001`。

### 主对比与描述性对比

- **`C − B`（a）**：唯一可归因的对照（参数 3,434,837 ≡ 3,434,837，forward FLOPs 逐位相等）。
- **`B − A`（b）**：**描述性**。A 为 3,286,037，少 **148,800 参数（+4.5%）**。
  任何引用必须同时写这个数，**不得**声称归因于物理内容。
- **(c) 声明范围**：两族都过 → `type-structure-and-pathway`；只有 (a) 过 →
  `type-structure-only`；都不过 → `no-increment-from-this-pathway`。(c) 是范围规则，
  不把测出的假变成通过。
- **(d) 跨剂量**：本轮只有一个剂量，16-epoch 档位不授权（段 B 已判 `saturated`），记 `null`。
- **(G) 动力学护栏**：任一处理臂 `z500@72h` 相对各自 seed 的 incumbent 恶化 ≤5%；
  且任一臂不得在 85 格中 >50% 上比 A 更差。
- **(F) 预检**：六条全绿（off 臂位级等价、四类到达矩阵严格对角、未来标签不可读、
  B≡C 容量、FLOPs 逐位相等、包络覆盖 treated 需求）。

**未测一律 `null`，不得记 `false` 也不得记 `true`。** 这是 `da1ed00` 那次「正则把空 body
默认成 false、卡片与读数相反」的直接修正。

## 无截断纪律（本轮硬要求）

1. **包络按 treated 速率推**：1.47 s/upd（高于六个实测值：generic 1.4043/1.4312/1.4271，
   typed 末段边际 1.4565/1.4618/1.4618）→ 单 seed 需 151,302 s → ×2.05 = **310,000 s**；
   硬上限 660,000 s 覆盖两波。
2. **探针早停**：每 seed 先跑三臂各 400 updates，用实测速率外推；若剩余包络 60% 装不下，
   立刻中止并记 `refused-insufficient-envelope`。**不允许**烧十小时再被截断。
3. **截断必留收据**：任何非正常结束写 `seed{N}_{arm}_partial_receipt.json`，含
   `updates_reached` / `seconds_per_update` / `gpu_hours_in_failed_attempt` / `progress_file`。
4. **真实截断点**是 `tools/r7_round_stages.py:292` 的 `min(deadline, now() + per_seed)`，
   不是 `hard_cap_seconds`。调后者无效，这一点写在这里防止执行者改错地方。

## 交付物

1. `outputs/r7_s3_typed_evidence_in_protocol_20261120_attempt01/` — 9 份 arm receipt、
   3 份 probe 记录、`readings.json`、`archive_receipt.json`；任何失败另有 partial receipt。
2. `docs/R7_S3_TYPED_EVIDENCE_IN_PROTOCOL_RESULT.md` — 证据页。必须含：
   族 P 与族 S 的**完整 85 格表**（差、区间、三 seed 单值、是否排除零）；
   `C − B` 与 `B − A` 分开陈述；`B − A` 旁写 `+148,800 参数`；
   探针实测 s/upd 与包络余量；实际 GPU-h 与账本行；limitations（含 D6 的归因缺口、
   交集丢 11 个 case、单区域单分辨率、2023 sealed）。
3. `docs/R7_EVIDENCE_INDEX.jsonl` 新行（`record_id: s3-typed-evidence-in-protocol`）。
4. `docs/goals/main-model-climatology-campaign.md` 账本行 + `campaign-state` 前移。

## 预算

- 软预算 **210,240 s**（58.4 h 墙钟）；超了继续并记 `overrun`。
- 硬包络 **660,000 s**；只有它截断，且截断不算通过。
- 预计实际 ≈117 GPU-h / ≈84 h 墙钟（3 seed 在 2 卡上两波）。
- 无总 GPU-h 上限，账本只记账。

## 停止条件

- 探针判 `refused-insufficient-envelope` → 立即停，写 partial receipt，不改包络重试。
- 任一 `(f)` 子句为假 → 不开始训练。
- 任一族以 `null` 结束（臂缺失）→ `readings.json` **不得写出**，如实记未测。
- 三 seed 全部完成且 `readings.json` 写出 → 立即停，不做续跑。
- 禁止 cron / 守护 / 会话外续跑。

## 进度

- **状态**：**`complete`（读数完成，等待独立复核）**。2026-11-20 冻结、2026-11-24 读数落盘。
- **已完成**：41 条反证（+4 条 driver 契约反证 = 45）；`(f)` 六条全绿；段 B `--dry-check`
  族 P 20/20、族 S 65/65、共用 1,447 case；attempt02 的 9/9 臂全部到达 32,132 updates 并评 5 lead，
  **0 份 partial receipt**；`readings.json` 已写出；证据页、索引、账本、`campaign-state` 已登记。
- **结果**：**(a) `C − B` = `false`**（族 P 0/20、族 S 0/65 在改善方向排除零，
  另有 13 格在恶化方向排除零）；**(b) `B − A` = `true` 但仅描述性**（+148,800 参数，无安慰剂臂）；
  **(c) = `no-increment-from-this-pathway`**；**(d) = `null`**；**(g) = `true`**。
  详见 `docs/R7_S3_TYPED_EVIDENCE_IN_PROTOCOL_RESULT.md`。
- **成本**：**115.2743 GPU-h**（训练 106.7570 + 评分 7.2172 + 探针 0.9206 + attempt01 失败 0.3795）。
- **未做**：16-epoch 档位（**不授权**）、安慰剂臂（ADR 0116 D6）、2023 test（`r = 0`）、
  129×129 wide 实例。
- **待修**：`scripts/r7_s3_in_protocol_assembly.py:93` 的 `epochs` 字段（实测 4.0001、字段记 1.0000311）；
  该文件在本轮冻结集内，故留到下一轮开轮前修。

## 纪律

- 不改 `model/**`、`training/**`、`data/**`；零父迁移；零 code identity 变更。
- 物理量只作**前向输入证据**，不加任何 loss 惩罚项（C1/C2 已关闭）。
- 每轮只跑一个臂；GPU 由统一父调度按共驻余量串行；spawn 前只读核对；
  **禁止**对非本实验进程发送任何信号。
- 提交前自跑 `tools/r7_round.py commit-guard`；登记后自跑 `tools/r7_round.py verify`；
  不依赖 hook 兜底；只推 `r7/weather-reasoning`；禁 main / force / merge。
- 四件套 + 账本行 GPU-h 与 record 的 `metrics.gpu_hours` 对齐到 1e-4。
- 统一写 `readings.json`（不是 `diagnostics.json`），含五键；`source_sha256` /
  `data_identity` 为 `null` 必须 raise。
- 产物必须带 `scientific_claim: false` 并如实写 `limitations`。
- `report()` 与 CI 绿都不是科学结论。
