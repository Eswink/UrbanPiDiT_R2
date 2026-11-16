# 0034 — 物理证据进前向 × 训练预算：让模型的物理机制在足够训练下兑现

- **日期**：2026-11-16
- **状态**：planned
- **工作分支**：`r7/weather-reasoning`
- **准入裁决**：决策 0115
- **执行源**：`docs/goals/s3-typed-evidence-budget.md`（唯一）
- **毕业论文题目**：基于深度学习的物理信息混合天气预测研究与应用

## 实际结果

`NOT EXECUTED YET`

---

## Context

用户对上一轮定位给出明确否决：「不接受，请重新搞新的创新出来，必须追求正面结果并且创新」，
随后追加硬约束：「**必须使模型拥有物理信息相关**，不管是在模型中还是在后处理中，**必须要有物理信息的贡献**」。
同时要求「最好不要创新方法和机制，需要的是模型的收益增量」。

这三条共同排除了「纯评估侧构造」与「纯证伪型论文」。本计划记录本轮如何同时满足它们，
以及为什么落在 `typed evidence × 训练预算` 上。

### §1 九轮物理路线的关闭状态（本轮新增的实测证据）

此前七轮（C1 / C2 / Stage P / Stage R / Stage C / Stage A / 城市静态）已在各自轮次内按预声明关闭。
本轮为了寻找「物理信息有真实贡献」的可行位置，又做了七项判决性测试，全部只读、0 GPU-h，
**结果全部为负**——但它们把可行位置逼到了一个具体点。

| 路线 | 本轮实测的关闭依据 |
| --- | --- |
| 物理失衡 → 误差大小 | 逐 lead `corr(imb, err)` 最大 **0.2676**，lead 12 为 **−0.0164**；pooled 0.4544 是 lead-趋势伪相关（lead 内去均值后 **0.1604**，lead 间均值 **0.8076**） |
| 物理失衡 → 误差空间图样 | case 内中位 Spearman **0.048–0.074**；**打不过 `\|∇z500\|`**（0.067 vs **0.131**，Wilcoxon p=3.8e−40）；剔除 `\|∇z500\|`+谱噪声后残差只剩 **+0.010** |
| 谱增益校正（Stage R 同族） | `f + g(f − box5 f)` 在 `g ≤ −1` 时**恒等于 box5**（恒等式残差 **0.000e+00**）；相干上限 ρ²=0.033→0.0087，**已达 87–98%**，headroom 近零 |
| 低秩平滑带 | 有效维数 `D_eff`=24.2–210.6（噪声对照 690.6），与相干长度 ρ=**−0.949**；但随 K 增大**退化到平坦常宽带**（空间结构仅值 6% 宽度），被块 b=4 支配 |
| 平衡子空间带 | `R²_bal`=**0.110**；`δz` 仅占归一化误差方差 **6.0%**（风场 94%）；`ws_ag` 最好 **6.22×** 真值（阈值 2×），且效率的「通过」是靠把松弛推到 +33.5 m/s 买来的 |
| case 自适应缩放 | **oracle** σ̂ 的 `P(all)`=0.005，**低于**基线 0.058；可部署代理最优 ρ=**0.272**（阈值 0.5） |
| **地转投影后处理** | 20/20 组合 RMSE **全升**（+4.4%…+183.0%）；`ws_ag_proj ≡ λ·ws_ag_orig` 是**构造恒等式**，不能当证据；matched box5 在全部 20 组上反而改善（−0.08%…−4.69%），**投影比纯平滑器差 10–156 个百分点** |

### §2 地转投影失败反推出的机制（本轮最重要的新证据）

失败本身给出了模型失衡的因果定位：

| 实测 | 值 | 含义 |
| --- | --- | --- |
| `corr(v_a, v_g 误差)` | **−0.79…−0.96** | 非平衡残差 `v_a` 在**补偿**一个错误的 `v_g`，不是噪声 |
| 预报 `\|v\|` / 真值 `\|v\|` | 0.84–0.97× | 风本身**偏弱** |
| 预报 `\|v_g\|` / 真值 `\|v_g\|` | 1.12–2.04× | 地转风**偏强** |
| 预报 z 的 f13 细尺度噪声 / 真值 | **2.56×** | 位势高频**过噪** |
| 预报风场 f13 / 真值 | 0.44–0.89× | 风场细尺度**不过噪**，反而更光滑 |
| 最优 `λ_LS`（oracle） | 0.87–1.00 | RMSE 要求**几乎不动** `v_a`；物理 `λ*`=0.22–0.62 方向相反 |
| 给真值 z 的完美 `v_g` | lead 6 的 250 hPa 仍 **+43.2%** | 事后地转重建的天花板**低于预报本身** |

**结论**：失衡的可修部分是 **z 的高频误差**，且必须在**前向**里修；
事后投影会拆掉模型自己建立的补偿项。这条把「物理信息该放在哪里」从猜测变成推断。

### §3 可行的物理通道：#79 typed evidence（唯一有正面先例的前向物理装置）

`docs/R7_S2_79_TYPED_EVIDENCE.md:15-17` 的注册读数：

| 项 | 值 |
| --- | --- |
| 归因配对 | `typed_routing − generic_fusion`，t2m，6h/12h |
| 判定 | **双 lead `supported`**（三 seed 同号为负） |
| 增量 | 6h **−0.05330 K**；12h **−0.05330 K**（逐 seed −0.0518/−0.0546/−0.0535） |
| 容量匹配 | 参数量 **3,097,571**、FLOPs **16,776,992,640** 逐位相等 |

机制核验逐条留痕（同文件 §2）：off 臂首步状态与旧实现**位级相同（差 0.0）**、扰动任一类型只移动其对应槽位
（**4×4 到达矩阵严格对角**）、毒化未来标签对 forecast 是 no-op、`typing_evidence_mode=None` 与不传参位级一致。

四个物理场由纯 torch 球面算子算出（`model/r7_process_tensor_diagnostics.py:158-165`）：
`divergence_850`、`vorticity_850`、`temperature_advection_850`、`static_stability_850_500`；
所需四个通道 `t500, t850, u850, v850` 全在 store 的 17 通道内。

**两个缺口**：
1. 生产配置里 `typed_evidence_mode` **未设**，即该通路在主线模型上是关闭的。
2. #79 只跑了 **400 updates**（`scripts/study_r7_79_typed_evidence.py:41`），
   而 Stage A 的对照臂是 4,000 updates。**该通路从未在任何 ≥1 epoch 的预算下被评过。**

### §4 训练预算缺口

仓内唯一四条剂量阶梯（同父、同数据、同 val、同分母，15/15 格点单调）：

| point | updates | epochs | t2m 6h | 72h | skill@72h |
| --- | ---: | ---: | ---: | ---: | ---: |
| p0025 | 2,008 | 0.25 | 2.2862 | 3.7293 | +0.0981 |
| p1000 | 8,033 | 1.00 | 2.2474 | 3.6006 | +0.1593 |
| p4000 | 32,132 | 4.00 | 2.1109 | 3.4446 | +0.2305 |
| p1600 | 128,528 | 16.00 | **1.8605** | **3.3765** | **+0.2607** |

- 饱和判定 `unsaturated`（Δ=**+0.030124** > 0.01），产物 `outputs/r7_s3_saturation_verdict_20261107.json`
- 规则要求的后续「最终 run 取 16–64 epoch」（`docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md:59`）
  **从未执行**（同文件 `:148` 自记）
- 所有已关闭的物理对照臂都跑在 **0.075–0.5 epoch**（容量 2,400；EMA 2,400；谱几何 4,000；导数匹配 32,132）

⇒ 「物理机制无效」这个结论的适用范围**只有不到半个 epoch**。

### §5 两处数字纠正（此前推理作废）

1. **3.14 K 是 48h 值，不是 72h**，且是全仓 48h **最强一档**（`docs/R7_S3_ANCHOR_FEEDBACK_65X65_RESULT.md:95`）。
   72h 的 t2m RMSE 全仓分布：min 3.3207 / median 3.663 / max 3.934。据此判断「模型弱」的两处推理作废。
2. **`ws_ag` 的 5.15→12.22 m/s 不可复现**。已登记口径（逐 case 中位数再取中位数）是
   **4.731→10.697**（cal）/ **4.750→10.738**（`diagnostics.json`），真值 **2.997→2.974** 平坦。
   方向、量级、单调性一致，但引用时必须用登记值。**任何「投影降低 ws_ag」的读数都是构造恒等式**，
   必须配 RMSE 才有意义。

---

## 方案

**唯一改动是「物理证据通路是否开启 / 以何种方式融合」与「训练预算」。网络结构、数据、评估代码逐字不动。**

### 段 A（主结果）：typed evidence 在 4-epoch 预算下的容量匹配三臂

| 臂 | `typed_evidence_mode` | 物理场 | 类型路由 | 参数量 |
| --- | --- | --- | --- | --- |
| **A** incumbent | `None`（未设） | 否 | — | 3,286,037 |
| **B** generic | `generic_fusion` | 是（四场融合到一个投影） | 否 | 3,434,837 |
| **C** typed | `typed_routing` | 是（每场独立投影） | 是 | 3,434,837 |

B 与 C 的参数量与 FLOPs **逐位相等**，这是唯一能把增量归因给**物理类型结构**的对照；
`B − A` 回答「物理证据本身是否有增量」，不区分类型路由。

预算 **32,132 updates @ batch 4（= 4 epochs）**，三 seed（41/42/43），同父
（`r7_s3_position_retention_parent_migrated_20261103_attempt01`，1600 步端点，
`state_digest 7c416ccb…`，当前 digest `59340710…` 下的模型迁移）同数据同 loss。

**为什么是 4 epoch 而不是 2,400 updates**：#79 的 400 updates 与 Stage A 的 4,000 updates 都不能回答
「物理证据在模型接近收敛时是否仍有增量」。4 epoch 是仓内已被证明有实质 skill 的档位
（p4000 的 72h skill +0.2305），且成本（≈31 GPU-h）可控。

### 段 B（0 训练，先跑，为 A 定预算）：已有 checkpoint 的剂量与推理深度

1. **组内剂量曲线**：p1600 的 13 个中间 checkpoint（每 10,000 步：5000/15000/…/125000）+ 端点
   128,528，leads {6, 24, 72}，**只有端点被评过分**（25 个从未评分）。
   判据：末 5 点是否严格上升。
2. **推理深度扫描**：p1600 端点，`K ∈ {2, 4, 6, 8}`，leads 全部 5 个。
   `evaluate_local` 的裸 `reasoning_steps` 不需要 controller（只有
   `policy_selection`/`validation_thresholds`/`force_full_depth` 才要求）。
   判据：存在 `K*≠4` 使 ≥3 lead 增益 > +0.005 且 CI 排除零；**优先 `K*<4`**（同时降推理成本）。

段 B 直接决定段 A 是否需要 16-epoch 档位，且**0 训练成本**。

### 预声明判据（冻结在跑之前）

| 判据 | 要求 | 失败意味着 |
| --- | --- | --- |
| **(a)** | `C − B` 在 ≥2 lead 上同号为负（t2m RMSE 更低），三 seed 一致，block bootstrap 95% CI 排除零 | 类型结构无贡献 |
| **(b)** | `B − A` 在 ≥2 lead 上同号为负 | 物理证据本身无增量 |
| **(c)** | 若 (a) 真而 (b) 假，**只能**声称类型结构，不得声称物理增量 | —— |
| **(d)** | 同臂在 4 epoch 与 16 epoch 上增量方向一致 | 4-epoch 结论不可外推 |
| **(e)** | z500@72h 相对 A 臂恶化 ≤ **5%** | t2m 好看但动力学变差 |
| **(f)** | 预检全真：三臂 trunk 位级相同、off 臂首步差 0.0、4×4 到达矩阵严格对角、`test_read: false` | 单位/接线事故 |

**(a) 的判定语义**：`C−B` 的符号必须在同一 (lead) 上三 seed 一致才计。
单 seed 反转只报不判——这是平衡轮 `(b)` 能靠噪声通过的教训（`scripts/r7_balance_exits.py:44-47`）。

**无对照即 `null`**：任一臂缺失时该子句返回 `None`（未测），**绝不是 `False` 也不是 `True`**。
这是 `da1ed00` 那次「正则把空 body 默认成 false、卡片与读数相反」的直接修正。

---

## 三个可能出现的结果，各自的论文含义

| 结果 | 含义 |
| --- | --- |
| (a) 真 + (b) 真 | **最佳**：物理证据提供增量，且类型结构进一步加成。可写「物理信息在前向路径中的可归因贡献」 |
| (a) 真 + (b) 假 | 物理证据总量无增益，但类型路由有增益。可写「物理量必须按其物理类别路由」，仍需诚实说明总量无增益 |
| (a) 假 + (b) 假 | 负结果，但**首次具备可引用性**——因为它跑在 4 epoch 而不是 0.075 epoch。必须写「在足够训练预算下物理证据通路仍无增量」 |

**不承诺正面结果**，但三档都有可写的内容，且 (a)/(b) 的判据不因结果而改。

---

## 要创建的文件

| 文件 | 性质 |
| --- | --- |
| `docs/plans/0034-typed-evidence-under-budget.md` | 本文件 |
| `docs/decisions/0115-typed-evidence-under-adequate-budget.md` | ADR |
| `docs/goals/s3-typed-evidence-budget.md` | 唯一执行源，`round-spec` + objective ≤4000 字符 |
| `scripts/r7_s3_typed_budget_protocol.py` | 协议构建器 + (a)–(f) 预检 |
| `scripts/r7_typed_budget_verdict.py` | (a)–(e) 求值器，无对照返回 `None` |
| `scripts/study_r7_s3_typed_budget.py` | driver：`freeze`/`readings`/`worker`/`DRIVER_REQUIRED_NAMES` |
| `scripts/r7_s3_dose_probe.py` | 段 B 的 0 训练探针 |
| `.zcode/workflow-drafts/r7-typed-evidence-budget.dwf.ts` | workflow |
| `tests/test_r7_s3_typed_budget.py` | 七条反证 |

**复用而非新建**（全部已核实存在）：

- `model/typed_evidence_r7.py` — 三种 mode 已实现；`isolated_stream` 保证开/关不移动既有权重流
- `model/r7_process_tensor_diagnostics.py:166` `typed_local_fields(...)`
- `scripts/r7_s3_position_retention_support.py` — `MODES`/`ARM_FACTOR_KEYS`/`require_single_factor`/
  `arm_spec`/sidecar 身份 pin（**65×65 sidecar**：`r7_s3_typed_evidence_fy65_20261103_attempt01`，
  `data_identity 81307008…`，与 store 一致）
- `scripts/r7_s3_typed_spatial_parent.py` — `expand_parent_state`（父权重位级安装 + 新增键白名单 + 阳性对照）、
  `positive_control`（共享张量位级相同、零 query 等于 pooled、投影置零后证据恰为零）
- `training/r7_long_rollout_runner.py:499` `fine_tune_long_rollout(...)`（597/600 行，**不改**）
- `training/r7_evaluate.py:121` `evaluate_local(..., checkpoint=, reasoning_steps=)`
- `training/r7_paired_comparison.py:35` `paired_rmse_bootstrap(...)`
- `training/r7_simultaneous_stats.py:54` `simultaneous_paired_interval(...)`
- `tools/r7_round_support.py` / `tools/check_round_receipts.py:83` / `tools/r7_explicit_inventory.py`

**硬约束**：

- 不改 `model/**` 的任何**已有行为**（三臂都用既有 mode；若确需新增代码路径必须走 ADR 并接受父迁移）
- 不改 `training/r7_evaluate.py`、不改 `training/r7_long_rollout_runner.py`（597/600）
- 不重建 store、不新增/重下数据、不改 `data_identity`
- 不读 2023 test、不 import 归档快照、不合成数据

## 测试要含的七条反证

1. **off 臂位级等价**：`typed_evidence_mode=None` 与不传参的前向输出逐位相同（沿用 #79 的 `test_off_path_is_bitwise_the_previous_model`）
2. **容量匹配断言**：B 与 C 参数量/FLOPs 逐位相等，A 严格更少，否则 harness 拒绝
3. **槽位隔离**：扰动任一物理场类型只移动其对应槽位（4×4 到达矩阵严格对角）
4. **无未来信息**：毒化 `atmos_target`/`process_targets` 对 forecast 是 no-op
5. **判据必须先能失败**：给一条 `C−B` 符号不一致的假读数，断言 (a) 返回 `False`
6. **无读数返回 `None`**：空 readings 上每个 clause 是 `null`（钉住 `da1ed00` 反向事故）
7. **`worker(phase="reading")` 端到端**：断言 `readings.json` 真被写出且含
   `scientific_claim / test_read / protocol_sha256 / exit_conditions / seeds` 五键
   （平衡轮用 7.2 s 的 `NameError` 换来的教训）

## Verification

1. `pytest tests/test_r7_s3_typed_budget.py -q` → 七条反证全过
2. `pytest tests/test_r7_typed_evidence.py tests/test_r7_process_tensor_diagnostics.py tests/test_r7_paired_comparison.py -q` → 不回归
3. `python tools/check_conventions.py` → 37 条阻断 0 失败（R-051 ≤600 / R-052 ≤200）
4. `python tools/check_goal_brief.py --brief docs/goals/s3-typed-evidence-budget.md` → failures=0
5. `python tools/check_campaign_state.py --campaign docs/goals/main-model-climatology-campaign.md` → exit 0
6. `python tools/r7_round.py state` → `current_round_goal` 指向新 brief
7. `python tools/r7_round.py round-guard` → 0
8. workflow 零成本编译：`CreateWorkflow{path, args:{maxRounds:0, dryRun:true}}`
9. **实跑前置**：`python scripts/r7_s3_typed_budget_protocol.py --preflight-only` 打印三臂参数量/FLOPs、
   trunk 位级一致性、到达矩阵，全绿才允许开跑
10. `git diff --check`

## 成本

| 段 | 内容 | 估算 |
| --- | --- | --- |
| B-1 | 13 ckpt × 3 lead × 1447 case | 5.4 GPU-h |
| B-2 | 4 K 值 × 5 lead × 1447 case | 2.9 GPU-h |
| A | 3 臂 × 1 seed × 32,132 upd = 96,396 upd @ 1.1536 s/upd | **30.9 GPU-h** |
| A（16 epoch，**条件启动**） | 3 臂 × 128,528 upd | 123 GPU-h |

**本轮只授权段 B + 段 A 的 4-epoch 档位**，合计约 **39 GPU-h**。
16-epoch 档位仅在 B-1 显示末 5 点严格上升时才开，且需具名预算授权。

软预算 **32 GPU-h** / 硬上限 **48 GPU-h**（段 B + A4ep）。

## 风险与负面后果

1. **C−B 可能为假**。#79 的先例是 400 updates、三臂绝对 skill 均为负（≈−0.11）。
   在 4 epoch 下它可能放大，也可能被更强的基线吞掉。
2. **`generic_fusion` 的参数匹配不等于计算图等价**。#79 已核实 FLOPs 逐位相等，但
   generic 模式在每 patch 位置融合再池化，typed 模式分类型池化后再投影。
   二者信息相同、容量相同，但**归因对象是「融合发生在池化前还是后」**，不是「有无物理信息」。
   这一点必须在证据页写明。
3. **z500@72h 护栏可能失败**。p4000→p1600 的 z500@72h 已从 427.35 升到 445.78（+4.3%），
   预算增加本身就在恶化这项指标。若 (e) 为假，则即使 (a)(b) 真，结论也必须限定在 t2m。
4. **4 epoch 仍可能不足以让 typed evidence 兑现**。若 (a)(b) 都假而 B-1 显示未饱和，
   正确动作是开 16-epoch 档位，**不是**改判据。
5. **GPU 共驻**：延迟与内存观测含邻居负载；每次 spawn 前必须只读核对余量。

## 明确不做

不新增任何物理 loss 项（C1/C2 已按预声明关闭）；不重开 Stage R / 容量 / EMA / 长 lead 权重 /
低秩带 / 平衡子带 / 地转投影 / case 自适应（本轮全部已实测关闭）；不改 `model/**` 已有行为；
不改 `training/r7_long_rollout_runner.py`；不重建 store；不新增或重下数据；不读 2023 test；
不 `cron`/守护/会话外续跑；不改冻结判据；不扫 `lr`/`batch`/`clip`（另立计划）。
