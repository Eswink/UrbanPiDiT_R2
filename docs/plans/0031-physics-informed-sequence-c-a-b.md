# 0031 — 物理信息主线阶梯 C→A→B：β_j 归一化与「预训练期」制度

- **日期**：2026-11-12
- **状态**：**定稿归档**（archived）；Step C **已执行**，Step A/B **未执行**，见文末 `## 实际结果`
- **分支**：`r7/weather-reasoning`；起点 HEAD `487dc6a`
- **`scientific_claim: false`**、`test_read: false`、`r = 0`
- **上游**：`docs/plans/0030-thesis-physics-informed-mainline.md`（前一版主线）、
  `docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`（Stage P 读数）、决策 **0110**（区域分块分解）、
  **0111**（Stage Q 预声明跳过，其 §136-137 留了「新的理由」开口）
- **一手文献**：`arXiv:2509.17601`（FastNet，2025-09-22，已读正文 §4.1.1–4.2.2、§5）
- **唯一执行源**：`docs/goals/s3-physics-informed-sequence.md`

---

## §0 摘要：为什么是「阶梯」而不是一个新名词

毕业论文题目是「基于深度学习的物理信息混合天气预测研究与应用」。前四轮几何全部为负：

| 轮次 | 形式 | 结果 | 成本 |
| --- | --- | --- | --- |
| C1 | 无目标约束束 | 真值被罚 10.2×，8/8 例方向反 | **0 GPU-h**（前置否决） |
| C2 | 导数域目标匹配 | 15/15 lead×seed 全负 | 66.99 GPU-h |
| Stage P | 谱几何前置 | P0 过、P1 判 `skip-q`、P2 三方向全不过 | 0 GPU-h |
| Stage R | 后处理振幅校正 | 85/85 (变量, lead) 全劣化 | 0 GPU-h |

**本计划不再发明新算子。** 一手文献检索找到一个与已关闭分支**精确互补**的先例，而且它拆成三个可分离的**设置差异**：

### §0.1 `arXiv:2509.17601`（FastNet）的三条逐字依据

**① 式 (8)，变量间 AMSE 量级归一化**（§4.1.2）：

> `β_j = [⟨AMSE_j(x^{d_0}, x^{d_0+τ})⟩_year]^{-1}`
>
> 「geopotential spectra span more than six orders of magnitude on the O96 grid, whereas
> specific humidity spans only three. **Without this adjustment, the MSH loss tends to
> overweight v-winds and humidity while underweighting temperature and geopotential.**」

**② 加入时机是最关键的差异**（§4.2.1）：

> 「the model was **pre-trained** with horizontal gradients in the loss function, rather than
> adding them only in fine-tuning. This approach best suppressed artefacts … **By contrast,
> introducing gradients solely at fine-tuning led to rapid RMSE degradation, while artefact
> reduction appeared only after extensive training (~30% of the pre-training schedule).**」

**③ 导数场归一化**（§4.2.1）：

> 「All derived fields are normalised to zero mean and unit variance so their contribution to
> the loss is comparable to that of the raw fields.」

### §0.2 第 ② 条**解释了**我们自己的负结果

我们的空间梯度实验是「从**没有**该项的父模型微调」，实测：15 个 (seed, lead) 单元中 14 个方向正确、
效应极小、**72 h 绝对 skill 三 seed 全负**（−0.171537 / −0.273893 / −0.152612）。
这正是 FastNet 描述的 **fine-tuning-only 失效模式**。文献**预测**了它。

### §0.3 封锁条款的逐字范围（决定 B 是否合法）

`docs/R7_EVIDENCE_INDEX.jsonl:83` 与 `docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md:43` 逐字：

> 不扫 λ、不加 seed、不调 2400 updates、不与 anchor/feedback/architecture/容量/K 轴组织组合、不换名重开

**「预训练期加入」与「零均值单位方差归一化」不在这五条内。** 但那轮确实是 fine-tuning-only，
所以「在预训练期加梯度」是一个**未测制度**，不是被禁的那一支。B 仍须**新 ADR 明确取代**与
fine-tuning-only 绑定的那部分，并如实写明原结论仍然成立、只是适用域被文献收窄。

---

## §1 起点（可核查）

- HEAD `487dc6a`、工作树干净
- `campaign-state`：`used_gpu_h 571.500957`、`remaining −551.500957`（**仅会计**，决策 0030/0038）
- 证据索引 **109** 条；账本 89 行
- `current_round_goal` 原指 `docs/goals/s3-physics-informed-campaign.md`
- 2023 test **未评分**、`r = 0`；S4 未启动

**基础设施已就绪，不需新建模块**：

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| `amse_analogue` | `training/r7_spectral_geometry.py:250` | 可微、`pred=target` 精确 0、**本轮修掉一个 `coh=-1` 奇点** |
| `spectral_geometry` 轴 | `training/r7_objective_axes.py:31` | 已注册，`:52-54` 已接 `spectral_geometry_term` |
| `band_powers` | `training/r7_spectral_geometry.py:127` | **对 17 通道直接求和**——这正是 §2.2 要修的 |
| 长 rollout runner | `training/r7_long_rollout_runner.py` | **597/600 行（R-051），不得改动** |

---

## §2 Step C 已执行：两个 0 GPU-h 测量

### §2.1 C1 —— 缺口的**衰减率**

Stage P 判 `skip-q` 的依据是「缺口单调下降」，但单调性不含速率信息。本轮拟合
`gap ~ N^(-α)`（4 端点 × 6 带 × 5 lead）：

| band | α | 拟合残差 |
| ---: | ---: | ---: |
| 0 | 0.10805 | 1.16e-01 |
| 1 | 0.10761 | 4.24e-02 |
| 2 | 0.05051 | 1.50e-02 |
| 3 | 0.04685 | 1.22e-02 |
| 4 | 0.04140 | 2.62e-02 |
| 5 | 0.08258 | 1.13e-01 |
| **细带均值（判据口径）** | **0.05111** | **2.57e-02** |
| 12-lead 口径 | 0.05723 | 2.06e-02 |

**⇒ 把细带缺口再减半需要 `0.5^(-1/0.05111) = 775,102×` 训练量**，即约
`128,528 × 775,102 ≈ 9.96×10^10` updates ≈ **3.07×10^7 GPU-h**（按 1.1096 s/update）。

**这条推翻了 `skip-q` 的实质含义而保留其形式**：缺口确实随训练缩小（满足 §3.1 的判据文字），
但缩小的速率使它在**任何可及预算**内都消不掉。这是决策 0111 §136-137 所说的「新的理由」。

### §2.2 C2 —— β_j 重加权后的量级

`band_powers` 对 17 通道直接求和，所以 Stage P 的缺口是一个**功率加权混合**，产物里没有逐变量信息。
按 FastNet 式 (8) 在**归一化空间**（`normalization_mean/std`，即损失所见的空间）计算：

| 变量 | mean AMSE | β_j | 相对份额 |
| --- | ---: | ---: | ---: |
| z250 | 4.8243e+00 | 0.207 | **39.84%** |
| z500 | 9.4588e+00 | 0.106 | **20.32%** |
| t500 | 1.6583e+01 | 0.060 | 11.59% |
| t850 | 3.1881e+01 | 0.031 | 6.03% |
| mslp | 4.0383e+01 | 0.025 | 4.76% |
| z850 | 5.2985e+01 | 0.019 | 3.63% |
| v500 / v10 / u10 | 3.5e+02 … 4.2e+02 | 0.003 / 0.003 / 0.002 | 0.55% / 0.54% / 0.46% |

**β_j 跨度 = 86.75×**（归一化空间）。**物理空间对照 = 1.16×10¹¹×**——这个对照证明
「逐变量谱权重必须与损失同空间」，否则 11 个数量级的跨度会完全重标定该项。

**这精确复现了 FastNet 的描述**：不加 β 时，无权重池化给每个变量 1/17（5.88%），而位势场的真实
贡献应在 39.84% + 20.32% = **60.16%**，风场应各 <1%。我们的 Stage P 缺口因此被风场与湿度的
噪声主导，而它本该主要由位势与温度的谱保真度决定。

### §2.3 顺带修掉一个会让损失返回 `inf` 的训练缺陷

`_terms` 把 `2·max(P,Q)·(1-coh)` 有理化成 `2·M·(P·Q−cross²)/(P·Q·(1+coh))` 以获得精确零。
在 **`coh = −1`（完全反相）** 处分母为零、分子是浮点残差，于是返回 **`inf`**——而正确的极限是
`4·max(P,Q)`，即**最大**惩罚。

实测（真实 6 h `v10` 对，store frame 1464）：`P=802.585`、`Q=93.396`、`cross=−273.785`
⇒ `coh=−1.0`，有理化形式 `inf`，字面值 **3210.34**。band 0（DC，单 bin）必然触发。

**朴素修法（在 `coh ≤ −1` 处置零）会把「完全反相」报成「完美匹配」，比 inf 更糟。**
修法：`1+coh` 不为正时用字面形式，其余保留有理化形式（以维持精确零性质）。
反证测试 `test_anti_phase_band_matches_the_literal_limit` 钉住 3210.34，
`test_exact_zero_is_preserved` 钉住 `amse(x,x) == 0.0` 的**精确**相等。

**这条的影响域要说准**：它不改变 Stage P 已登记的 `amse_at_truth = 0.0` 与
`amse_after_coarsening = 549.5046`（那两个调用未触及 `coh = −1`），但**任何逐帧、逐变量的大量
计算都会命中它**——本轮 1459 个真值帧里有 108 处（前 201 对中）。所以它对 A 轮是**阻断级**的：
不修就无法在训练中稳定求值。

---

## §3 阶梯：C（已做）→ A → B

成本从低到高，**任一级出现有效果就停在那里**。

### Step A — AMSE + β_j 归一化（约 12 GPU-h）

**唯一改动**：给 `spectral_geometry` 轴加 FastNet 式 (8) 的 `β_j`。从 p1600 端点微调
（AMSE 原文自己就是微调：¼° GraphCast fine-tune，约 26.2 GPU-days）。

**预声明出口**（任一为假即关此支）：

| 判据 | 要求 |
| --- | --- |
| (a) | 细带 `amp_ratio` 相对同 seed off 臂**升高** |
| (b) | 72 h t2m skill 三 seed 配对 Δ **全 ≥ 0** |
| (c) | 24/48 h skill 配对 Δ 三 seed **全 ≥ 0** |
| (d) | off 路径逐位等价（由 dispatcher 的 `None` 早退 + 反证测试支撑，**不由读数自证**——0109 的教训） |
| (e) | 预检：`β_j` 在真实 batch 上**非全相等**，且 `amse` 在全部 8 case × 12 step 上**有限** |

3 seed × 2 臂 × 4,000 updates ≈ 11.7 GPU-h（按 20 估）。

### Step B — 梯度项改到**预训练期**（约 40–77 GPU-h，仅当 A 失败才做）

**三处同时改**（FastNet §4.2.1 的全部差异）：① 预训练期加入；② 导数场零均值单位方差；
③ 权重沿用逐变量权重。

**必须新 ADR 明确取代**旧关闭条款中与 fine-tuning-only 绑定的部分，并逐字引用 FastNet §4.2.1
的 ablation 作为设置差异的依据，同时如实写明：这是**未测制度**，不是对原结论的翻案；
原结论（fine-tuning-only 有害）**仍然成立且被文献解释**。

预声明出口同 A 的 (a)–(e)，再加 **(f) 72 h 绝对 skill ≥ 0**。

---

## §4 成本表

| Step | 内容 | GPU-h | 累计 |
| --- | --- | ---: | ---: |
| **C** | 缺口衰减律 + β_j（已执行） | **0** | **0** |
| **A** | AMSE + β_j，2 臂 × 3 seed × 4,000 updates | ≈11.7（按 20 估） | ≈20 |
| **B** | 梯度项预训练期，从迁移父 | ≈40–77 | ≈60–97 |

**参照**：C2 一轮 66.99 GPU-h 换来一个否定结论。本阶梯的 A 只要 12 GPU-h，且自带能否开 B 的判据。

---

## §5 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1 / C2 | 0084 §6、0094、0109 已按预声明关闭 |
| **重开 fine-tuning-only 的空间梯度支** | 其结论成立，且已被 FastNet §4.2.1 解释 |
| 扫 λ、加 seed、调 2400 updates、与 anchor/feedback/architecture/容量/K 组合 | 封锁条款逐字禁止 |
| 引入 `pywt` / `pyshtools` / `ducc0` / `torch_harmonics` | `β_j` 与单位方差归一化不需要 SHT |
| 改 `training/r7_long_rollout_runner.py` | 597/600 行（R-051） |
| 改 Stage R 的 `per_case_ratio` schema | 改它会改已登记 digest，须另立决策 |
| 读 2023 test | `r = 0` |
| 承诺一定有正面结果 | 若 C 判 `close-line`、或 A 的 (a)–(c) 任一为假，如实记负并停 |

---

## §6 风险（如实列）

1. **C2 的 β_j 是 prequential 近似**：FastNet 用「给定年份的连续 6 h 真值对」的 AMSE 均值；本仓
   的 AMSE 是**分带**形式而非球谐阶，所以 β_j 是 *AMSE-analogue 的归一化*，不是论文那个量。
   命名纪律：不得称「复现 FastNet 式 (8)」，称 **β_j AMSE-analogue 归一化**。
2. **单年（2001）估 β_j**：论文报告年际方差极小，本仓未测年际方差，是一个未量化假设。
3. **A 的 dose 未定**：AMSE 是全损失替换还是附加项、权重多少，是 A 轮 ADR 必须**冻结前**写定的。
   本轮不预设。
4. **A 只有 4,000 updates**：若 AMSE 的作用需要更长的适应期，4,000 步可能不够。这与 FastNet 的
   微调日程（36,525 + 4×7,305 步）比是短的；A 的 (a) 判据（细带振幅比升高）正是为此设的
   **直接作用点**检查，比 skill 更早能看到信号。
5. **`_terms` 奇点修复改变了 `training_code_sha256`**：A 轮的父迁移与代码身份必须重做，不能沿用
   Stage P 的身份。
6. **n = 1 模型、单实例、单分辨率**：C1/C2/R 的适用域限制同样适用。

---

## §7 workflow 与提示词

执行形态是**一份 goal 长文 + 一个派生 workflow**，两者合在 objective 里
（照 `docs/goals/s3-throughput-and-physics-campaign.md:18` 的形态）。

- `docs/goals/s3-physics-informed-sequence.md`，头部 `<!-- round-node: S3 -->` + `<!-- round-spec: {...} -->`
- `.zcode/workflow-drafts/r7-physics-informed-sequence.dwf.ts`，从 `r7-campaign-round.dwf.ts` 派生
- **必须用 `CreateWorkflow {path, args}`，不要用 `AmendWorkflow`**——上一轮实测：`AmendWorkflow`
  丢 `args` 让 `dryRun` 静默变 true
- 零成本编译校验：`CreateWorkflow` + `args: {maxRounds: 0, dryRun: true}`

---

## 实际结果（Actual Results）

**Step C 已执行；Step A 与 Step B 未执行。**

### 已发生

- **`scripts/r7_s3_gap_decay_and_beta.py`** 已写并实跑（0 GPU-h，约 6 min CPU）：
  - `alpha_fine_five_lead = 0.05111`（残差 2.57e-02）、`alpha_fine_twelve_lead = 0.05723`（残差 2.06e-02）
  - 逐带 α ∈ [0.04140, 0.10805]，全部 `identifiable = true`
  - **`updates_to_halve_gap = 775,102`** ⇒ ≈3.07×10⁷ GPU-h
  - `beta_spread_normalized = 86.75`、`beta_spread_physical_control = 1.16×10¹¹`
  - `verdict = open-a`
- **`tests/test_r7_s3_gap_decay_and_beta.py`** 23 项全过（含奇点反证、精确零反证、两点不可识别反证、
  排序陷阱反证）
- **修掉一个训练缺陷**：`training/r7_spectral_geometry.py::_terms` 在 `coh = −1` 处返回 `inf`；
  实测命中 108/201 个真值对。修复后该带返回 3210.34（字面极限），`amse(x,x)` 仍精确 0。
  **改了 `training_code_sha256`。**

### 未发生

- **Step A 未执行**：`β_j` 已算出但**未接进训练**；无 protocol、无 code.zip、无读数
- **Step B 未执行**：未写 ADR 0113，未碰梯度项
- **GPU-h = 0**；未读 2023 test

### 未验证

- β_j 是否真的改善细带振幅（A 的 (a)）
- 2001 单年估 β_j 的年际稳定性（未测）
- `_terms` 修复对 Stage P 已登记读数的影响（**预期为零**，但那两个调用路径未被重跑验证）

**下一步的判据**：A 轮开轮并给出 (a)–(e) 的读数之后，本文件的状态改为「A 已测」。
在此之前，**任何引用本文件作为「β_j 归一化有效」的证据都是错的**——它只被证明**可计算且量级正确**。
