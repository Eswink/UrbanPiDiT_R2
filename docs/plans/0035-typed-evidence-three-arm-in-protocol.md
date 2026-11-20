# 计划 0035 — 三臂 in-protocol 重训：把「物理类型结构」测出来，并首次给出 32k 的 run-to-run 桥

## Context

用户显式授权 **GPU 训练时间无硬上限**，要求**严格避免训练未完成就截断**。上一轮
`s3-typed-evidence-budget` 的三个 `typed` 臂全被每 seed 35 h 包络截断，29.32% 的段 A 算力花在
从未评分的臂上（`docs/R7_S3_TYPED_EVIDENCE_BUDGET_RESULT.md` §6），并且留下了一个从未回答的问题。

### 开轮前必须面对的四个事实

1. **`C − B` 在任何预算下都没测过。** 400 updates 只测了 A→C 与 B→C 的方向（`#79`）；
   2,400 updates 测出 `|Δrmse| ≤ 3.2e-6 K`（`s3-typed-spatial-result`）；32,132 updates 时
   typed 臂在到达任何 lead 评分前被截断。这是整条物理线唯一悬空的问题。
2. **`B − A` 无法归因。** `generic` 比 `incumbent` 多 **148,800 参数（+4.5%）**，「物理场进入
   前向」与「参数更多」在同一个 delta 里。上一轮把这一点当作物理证据，不成立。
3. **上一轮冻结的 (b) 判据选错了变量。** 它判 `t2m` 且要求 `−0.005 K`，实测只有
   `−0.0016…−0.0032 K`（门槛的 1/3）。
4. **`hard_cap_seconds` 形同虚设。** 真正的截断点是 `tools/r7_round_stages.py:292` 的
   `limit = min(deadline, now() + per_seed)`；把 hard cap 调成 30 天什么都不修。而 `tools/`
   **不在** `training_code_digest()` 的哈希集里。

### 段 B 在开轮前就给出了本轮最重要的机制信息

`scripts/r7_s3_joint_criterion.py --dry-check` 在**上一轮已存在的 15 份 CSV / provenance** 上
装配族并求同时区间（0 训练、纯 CPU），结论（`exploratory`）：

- 族 P（`t500,t850,u850,v850` × 5 lead，20 格）：**10 格的 Δskill 同时区间排除零，全部为正，零个为负**
- 族 S（其余 13 通道 × 5 lead，65 格）：**29 格排除零，全部为正，零个为负**
- 最强不在物理输入通道，而在 **z500（72h +0.69%、48h +0.66%）与 z250（72h +0.51%）**；
  族 P 内最强是 `t500`（48h +0.46%）与 `v850`（24h +0.45%）
- 每族共用 **1,447** 个 case（五 lead 交集，丢掉 11 个只在短 lead 出现的 case），
  block_length 28、1,420 个 blocks

**这解释并纠正了上一轮 (b) 为何判 False**：它只看 `t2m`，而效应落在模式顶部的动力学与
`t500/t850` 上。同时它把本轮的阈值动机写清了——但正因为阈值是看着这些数字定的，
段 B **不构成确认**，只有本轮新训的读数能检验它。

### 为什么本轮不做安慰剂臂

安慰剂（同参数、同 FLOPs、物理场逐 cell 置换）是分离「物理内容」与「容量」的唯一干净办法，
静态协变量轮（决策 0113）正因缺安慰剂而读不出结论。但机制查证后它是硬约束：

- 场在 `model/typed_evidence_r7.py:270` 的 `field_maps(anchor, batch)` 内部由
  `typed_local_fields()` **现算**；`evidence()` / `token_evidence()` / `inject()` / `forward()`
  整条链上没有任何形参接受外部场。
- 唯一能逐臂换值的通道是 state-dict，而 8 个标准化量（`field_scale/field_mean/field_std/
  denorm_*/field_active`）全是 `persistent=False`，**不进 checkpoint**
  （`model/typed_evidence_r7.py:151-164`）。
- 因此安慰剂必须改 `model/` ⇒ `model_code_sha256` 变 ⇒ 本轮即将产出的 6 个 32k checkpoint
  与 3 个父 checkpoint 全部不可加载（`training/r7_experiment.py:128`）。

**顺序论证**：安慰剂轮只需训 D 一个臂（`B/C/A` 可 pinned），约 42 GPU-h。所以「先三臂（≈115）
→ 若 C−B 有信号再补 D（≈42）」总账与「一次四臂（≈156）」相同，但**若 C−B 是空的，可以在花
那 42 之前停下**。先做便宜的决定性问题。

---

## 方案

三个臂，全部在本轮协议内从同一父权重重训，**不改 `model/` 任何文件、不做父迁移、不做 pinned 复用**。

| 臂 | `typed_evidence_mode` | 参数量 | 本轮性质 |
|---|---|---:|---|
| **A** incumbent | `None` | 3,286,037 | **正对照**：同代码/数据/parent/seed 重训 |
| **B** generic | `generic_fusion` | 3,434,837 | 物理场进入，无类型路由 |
| **C** typed | `typed_routing` | 3,434,837 | 每类场独立投影 |

预算 **32,132 updates / batch 4 / seed 41,42,43**，与已注册 p4000 剂量点一致。

### 判据（训练前冻结进 `protocol.json`，训练后不得改动）

**族按机制划分，不按观测量** —— 这是避免「挑了最好看的变量」的唯一办法：

- **族 P（主判据，20 格）**：物理通路声明的消费通道 `t500,t850,u850,v850` × 5 lead。
  这四个通道由 `TYPED_EVIDENCE_CHANNELS`（`model/r7_process_tensor_diagnostics.py:158-165`）
  预先确定，与任何读数无关。
- **族 S（次要，65 格）**：其余 13 个通道 × 5 lead。
- **量纲无关度量**：`Δskill_i = (mse_对照,i − mse_处理,i) / D[L,c]`，`D` 是该格的气候态 MSE。
  `D` 与臂无关且固定，逐 case 除以它是单调缩放，改不了符号也改不了区间是否排除零，
  而 17 个跨 K/m·s⁻¹/Pa/kg·kg⁻¹/m²s⁻² 的变量于是可比。
- **多重比较**：`training/r7_simultaneous_stats.simultaneous_paired_interval`
  （max-statistic，一个临界值覆盖整族，α=0.05，block_length=28 cases=7 天）。
- **通过规则**（P/S 各自适用）：① ≥1 格区间在改善方向排除零；② 通过格跨 **≥2 个 lead**；
  ③ 通过格三 seed **同号**；④ 区间差 `≥ 0.001`（量纲无关下限）。
- **主对比与描述性对比严格分开**：`C − B` 可归因（容量/FLOPs 逐位相同）；`B − A` 只作
  描述性工程读数，**必须**同时写 `+148,800 参数（+4.5%）`。
- **护栏 G**：`z500@72h` 相对 A 恶化 ≤5%，且任一臂不得在 85 格中 >50% 上比 A 更差。
- **结构 F**：`(f)` 预检六条全绿。
- **(d) 跨剂量方向记 `null`**：16-epoch 档位不授权；段 B 探针已在同一代码身份内判
  `saturated`（`tail_delta=+0.041507`，末 5 点 3.3350→3.3765）。

### 「无硬上限」的诚实落地

字面意义的无上限做不到而不改 `training/`（`_validate_options` 的 `isfinite` 拒绝 + driver
拒绝 `None` + `subprocess.run(timeout=)`），改 `training/` 会再引入一个
`training_code_sha256` 变更。本轮改为三件结构上的事：

1. **包络按 treated 实测速率推**（修缺陷 18）：取 1.47 s/upd（高于全部六个实测值：
   generic 1.4043/1.4312/1.4271，typed 末段边际 1.4565/1.4618/1.4618），
   单 seed 需 3 × (32,132 × 1.47 + 3,200) = **151,302 s = 42.03 h**，
   取 2.05× = **310,000 s**；硬上限 660,000 s 覆盖两波。
2. **机械可测的下界**：`envelope()` 从 treated 常数重算需求，测试断言
   `PER_SEED_SECONDS ≥ 2.0 × need`，并另有一条测试断言「按 incumbent 速率算出的包络
   < 本轮需求」——那正是上一轮的错误，现在它会失败。
3. **探针早停**（新，本轮核心）：每 seed 先跑三个臂各 **400 updates**，用实测速率外推；
   若剩余包络的 **60%** 都装不下，立即中止并记 `refused-insufficient-envelope`。
   成本二十分钟，而不是十小时。
4. **截断必留收据**（修缺陷 20）：任何非正常结束写 `seed{N}_{arm}_partial_receipt.json`，
   含已完成 updates、实测 s/upd、`progress.jsonl` 回指，账本不可能静默少计。

---

## 要创建的文件

| 文件 | 性质 |
|---|---|
| `docs/plans/0035-typed-evidence-three-arm-in-protocol.md` | 本文件 |
| `docs/decisions/0116-typed-evidence-in-protocol-and-no-truncation-budget.md` | ADR：三臂、族划分、无截断预算、正面论证为何不开安慰剂 |
| `docs/goals/s3-typed-evidence-in-protocol.md` | goal brief，唯一执行源 |
| `scripts/r7_s3_in_protocol_protocol.py` | 注册表、预算常数、`freeze`/`prepare`/`validate`（558 行） |
| `scripts/r7_s3_in_protocol_preflight.py` | (f) 六条子句 + 机制断言（333 行） |
| `scripts/r7_s3_in_protocol_verdict.py` | 族 P/S 同时区间 + 判定（533 行） |
| `scripts/study_r7_s3_in_protocol.py` | driver：`probe`/`seed`/`reading`/`archive`（562 行） |
| `scripts/r7_s3_in_protocol_assembly.py` | 装配层：receipts → 判定输入（122 行） |
| `scripts/r7_s3_joint_criterion.py` | 段 B：零训练装配自检 |
| `tests/test_r7_s3_in_protocol.py` | 41 条反证 |

## 复用而非新建

- `training/r7_long_rollout_runner.py:499` `fine_tune_long_rollout(...)` — 不改
- `training/r7_evaluate.py:121` `evaluate_local(..., deadline=)` — 不改
- `training/r7_simultaneous_stats.py:54` `simultaneous_paired_interval(...)` — max-statistic
- `training/r7_paired_comparison.py:35` `paired_rmse_bootstrap(...)` — 六字段校验 + 逐 case mse
- `training/r7_experiment.py:31` `model_code_digest()` / `:128` digest 校验 — 只读核对
- `tools/r7_round_stages.py` 调度、`tools/r7_round_support.py` `sha256_file`
- `scripts/r7_s3_typed_budget_protocol.py` 的 `registered_spec()` 与 `payload()` 形态

## 实测验证（本轮已跑，非计划）

| 检查 | 结果 |
|---|---|
| `pytest tests/test_r7_s3_in_protocol.py -q` | **41 passed** |
| `scripts/r7_s3_joint_criterion.py --dry-check` | 族 P 20/20 格、族 S 65/65 格装配成功，共用 1,447 case |
| `scripts/r7_s3_in_protocol_protocol.py --preflight-only` | `f_preflight_passed: true`，六条全绿；A=3,286,037，B≡C=3,434,837 |
| `tools/check_conventions.py` | 37 条阻断 0 失败（R-051/R-052 通过） |

## 成本

| 段 | 内容 | GPU-h |
|---|---|---:|
| B | 零训练，复用上一轮 15 份 CSV/provenance | **0** |
| Probe | 3 臂 × 3 seed × 400 upd | ≈1.5 |
| A/B/C × 3 seed | 3 臂 × 3 seed × 32,132 upd + 5 lead 评分 | ≈117 |
| 墙钟 | 3 seed 在 2 卡 = 2 波 × ≈42 h | ≈84 h（3.5 天） |

软预算 210,240 s（58.4 h 墙钟，超了记 overrun）；硬包络 660,000 s。

## 明确不做

不改 `model/**` 与 `training/**`（零父迁移、零 code identity 变更）；不做安慰剂臂；不做 pinned
臂复用（三臂全在协议内，`C−B` 是 in-protocol）；不做 16-epoch 档位；不重开 0048 容量轴 / EMA /
长 lead 权重 / C1 / C2 / Stage R / 低秩带 / 平衡带 / 地转投影 / case 自适应；不重建 store；
不新增或重下数据；不读 2023 test（`r = 0`）；不 cron / 守护 / 会话外续跑；不改已冻结判据。

## 实际结果

`NOT EXECUTED YET`
