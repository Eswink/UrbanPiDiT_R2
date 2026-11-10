# 0030 — 毕业论文主线：物理信息混合预报（谱几何 → 平衡约束 → 正面结果）

- **日期**：2026-11-11
- **状态**：**定稿归档**（archived）；本计划**尚未执行**，见文末 `## 实际结果`
- **分支**：`r7/weather-reasoning`；起点 HEAD `ddd9104e2aa072ea4ba15cdfc8c3bea3f7206694`
- **`scientific_claim: false`**、`test_read: false`、`r = 0`；本计划不含任何科学结论
- **上游**：`docs/plans/0029-physics-operator-candidates.md`（候选清单与 J1–J4 判据）、
  `docs/R7_PHYSICS_LOSS_LITERATURE_REVIEW.md`（一手文献检索，2026-11-10）、
  决策 0084 §6（族判别）、0094（C1 前置否决）、0109（C2 永久关闭）
- **唯一执行源**：`docs/goals/s3-physics-informed-campaign.md`

---

## §0 摘要：为什么离开「再加一个物理约束项」

物理信息的两个方向都已有了结果，**一个方向错、一个无效**：

| 轮次 | 形式 | 实测结果 | 状态 | 成本 |
| --- | --- | --- | --- | --- |
| **C1** | **无目标**约束束（散度幅度、比湿非负、露点 ≤ T） | 分析场（真值）被罚是模型的 **10.2×**（8/8 例，比值 6.503–15.481） | 前置否决，**未开轮** | **0 GPU-h** |
| **C2** | **有目标**导数域匹配（+72 h 的散度/涡度差） | 72 h t2m skill 配对差 **−0.011634 / −0.016342 / −0.013380**，均值 −0.013785；**15/15** 个 (lead, seed) 配对差全负 | 按预声明**永久关闭** | 66.993549 GPU-h |

⇒ 继续做物理信息，**不能只是换一个物理名词**。换名词属于「换名重开」，明令禁止（决策 0084 §6、0109）。

**本计划的判断是：C1 与 C2 的失败指向的是同一件事，而这件事恰好给出了出路。**

---

## §1 起点（可核查）

- 分支 `r7/weather-reasoning`，HEAD `ddd9104`，工作树干净
- `campaign-state`：`used_gpu_h 571.500957`、`remaining_gpu_h −551.500957`（**仅会计字段**，决策 0030/0038 后无总 GPU-h 上限）
- 证据索引 **106** 条；账本 **89** 行
- `current_round_goal` 现指 `docs/goals/s3-throughput-physics-c2.md`（C2 已关闭，指针待改）
- 2023 test **未评分**、`r = 0`；S4 未启动
- 未处置的三件（处置权在用户与复核者）：R2 四出口全真但 `needs-review`、无独立复核；R4 停在 1,440/16,071 无读数（重开＝重跑约 77 GPU-h）；FY candidate 等验收是否进 S4

**起点的健康数据**：65×65 多年实例 `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr`，
shape `[34084,17,65,65]`，train 2000–2021 / val 2022 / test 2023 封存；
channel 序 `t2m,u10,v10,mslp,z850,t850,q850,u850,v850,z500,t500,q500,u500,v500,z250,u250,v250`。

---

## §2 三条已实测事实，与一条把它们连起来的论证

### §2.1 三条事实

| # | 事实 | 数值 | 出处 |
| --- | --- | --- | --- |
| 1 | **模型场比真值平滑 2.9 倍** | `mean\|∇·(u,v)\|` 模型 **8.5227e-06** vs 分析场 **2.5063e-05** s⁻¹；p95 2.3913e-05 vs 7.2998e-05（3.1×） | `docs/R7_S3_PHYSICAL_CONSTRAINTS_C1_PREFLIGHT_RESULT.md` §3 |
| 2 | **无目标方程约束方向反** | 散度项对分析场惩罚是模型的 **10.2×**，8/8 例一致（6.503–15.481） | 同上 §2 |
| 3 | **有目标导数匹配一致退化** | 15/15 (lead, seed) 配对差全负，均值 **−0.013785**；幅度仅 0.91×–1.28× 控制臂极差 | `docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md` §2 |

### §2.2 统一论证（本计划的核心，也是 J4 族判别的答案）

**这三条不是三个独立的失败，是同一条病的三个侧面。**

1. **事实 1 是病。** 真实大气的逐层散度**不是**近零的（`mean 2.5e-05 s⁻¹`），而模型场比真实场**更平滑**。
   即模型的问题是**小尺度能量不足**，不是「违反物理定律」。
2. **事实 2 是病 1 的机制解释。** 任何惩罚「粗糙度」的无目标项，其下降方向都是把预测推向**更平滑**的场
   ——而这正是已被关闭的空间梯度损失的失效模式（最优解是常数场）。所以 C1 的失败不是「选了错的物理量」，
   是**这一类项的下降方向与真值相反**：模型已经比真值平滑，再罚粗糙度只会更远离真值。
3. **事实 3 是「只在一个派生量上匹配目标」的剂量问题。** C2 方向对（真值处精确为 0），但 dose =
   `total/L_mse` = 0.2889–0.4109，约束项一开始就占基础损失的 **29–41%**，与 skill 抢梯度，
   且只约束**最后一步的、派生的**量。

⇒ **出路不是加项，是改目标函数的几何**：让损失函数对**尺度的相对权重**不再是 MSE 隐含的那个。
这正是 `arXiv:2501.19374v2` 的路径：MSE 的平滑是**最优解性质**（解析证明 `σ_X = ρ < 1`），
而 AMSE 通过**打断振幅与相干性的耦合**来修正它，**parameter-free**、零当且仅当 `x = y`、
在 `x = y` 处与 MSE 同 Taylor 展开。

### §2.3 一条必须修的文献误引（本计划的第一项事实修正）

`plan 0029` §4.1 与 `docs/R7_PHYSICS_LOSS_LITERATURE_REVIEW.md` §4.2 都断言
「**本仓是 global 65×65**，所以球谐理论上适用」。**这是错的。**

`docs/DATA_SPEC.md:36` 钉死：ROI `27.0–43.0°N / 107.0–123.0°E`、0.25°、`[65, 65]`
—— 一个**区域 patch**，不是全球网格。球谐变换（SHT）在区域块上**不良定义**。

一手核对（arXiv MCP，`2501.19374v2` §4.2，访问 2026-11-10）逐字：

> Implementing this in a training cycle requires fast computation of spherical harmonic transforms.
> This is simple enough for **global** latitude/longitude grids, but it might be **difficult for
> local-area models** without a regular global grid structure. In these cases, we think that the basic
> intuition behind (6) might still apply through other multiscale decompositions such as **wavelet
> lifting**, provided suitable equivalents to scale-dependent variance and correlation could be found.
> The multiscale decomposition is critical in some form, however, since the method takes advantage of
> the approximate independence of scale-separated modes. **Without such a decomposition (e.g. applying
> the adjustment of (6) globally, without the harmonic transform), the model might be able to "cover up"
> a lack of fine-scale variability by over-emphasizing coarser scales.**

同一段还引 **Han et al. (2024)**：一个 **local-area** 模型在长 lead 上出现「diminished forecast
activity」（一个与 blurring 相关的整体度量），即**区域模型同样有病 1**。

⇒ 两条结论：**(a)** 直接把 AMSE 套到 `rfft2` 上属于论文自己点名的反模式；
**(b)** 但不能因此放弃谱几何——它是唯一有解析证明的物理尺度信息。
正确做法是给出**在区域 patch 上良定义、且带尺度分解**的近似，并**显式声明近似**。见 ADR **0110**。

### §2.4 可用工具（实测，不是设想）

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 平面 `rfft2` | **已有**（模型内部结构，不是损失） | `model/r7_baselines.py:117` |
| DCT-II `dctn` | **已有**（scipy 1.18.1） | venv 实测 |
| `pyshtools` / `ducc0` / `torch_harmonics` | **未安装** | venv 实测 |
| `pywt`（wavelet） | **未安装** | venv 实测 |
| 球面散度/涡度/梯度/平流算子 | **已有，可复用** | `model/r7_process_tensor_diagnostics.py:127,133,139,145` |
| 轴分发点（唯一入口） | **已有，45 行，有余量** | `training/r7_objective_axes.py:41` |
| 长 rollout runner | **597/600 行，已满** | `training/r7_long_rollout_runner.py`（R-051） |

**最后一行是硬约束**：谱损失的接线**不能**加进 `r7_long_rollout_runner.py`，必须另开
`training/r7_spectral_geometry.py`，从 `r7_objective_axes.py` 那一个分发点进入。

---

## §3 三个 Stage 与各自的开轮判据

沿用 `plan 0029` §2 的 J1–J4，但**按 §2.2 的论证重新解释 J4**：谱几何**不是**「又一个物理约束」，
它替换 MSE 的几何，因此与已关闭的三支都不同族。

| 判据 | 内容 | 本计划里的用法 |
| --- | --- | --- |
| **J1**（方向） | `penalty(analysis) ≤ penalty(model)`，逐例一致 | **Stage P2**：静力平衡/整层散度/湿物理三项逐方向测；不过即不开该方向 |
| **J2**（判别力） | 守卫必须**能失败**（K2：C2 attempt07 的死因） | **Stage P0**：构造「只在粗 band 加能量」的假场，断言损失**必须变差**；`x = y` 必须**精确 0** |
| **J3**（剂量） | `dose = penalty/L_mse` ∈ 预声明带 `[1e-3, 1e-1]` | **Stage P2** 逐方向报 dose；**Stage Q** 的 dose 在冻结前实测 |
| **J4**（族判别） | 最优点集合必须不同 | 谱几何：`pred = target` 精确 0、最优解是**目标谱**而非常数场；与空间梯度损失（最优解常数场）、C1（无目标、方向反）、C2（单一派生量、剂量过大）都不同 |

### §3.1 Stage P — 零 GPU 前置探针（0 GPU-h）

| 编号 | 测什么 | 判据 | 不过怎么办 |
| --- | --- | --- | --- |
| **P0** | 分解合法性：`rfft2(norm='ortho')` 的 Parseval 精确性、**接缝能量占比**、**DCT-II 交叉校验** | Parseval 到浮点容差；接缝能量**报出**（不设阈值，但必须在记录里）；DCT 与 rfft2 **结论一致** | **停**，不开 Q；把理由写进进度块 |
| **P1** | 振幅/相干性：4 端点 × 8 train case × 17 变量 × 5 lead 的 `amp_ratio(k)` 与 `coh(k)` | 缺口 `1 − amp_ratio(k)` 随训练量**是否缩小** | **缩小 ⇒ Q 跳过**（AMSE 动机弱）；不变/扩大 ⇒ 开路 |
| **P2** | J1 方向门：静力平衡、整层散度积分、湿物理（`q≥0`、`q≤q_sat`） | `penalty(analysis) ≤ penalty(model)` **逐例一致**，且 dose ∈ `[1e-3,1e-1]` | 逐方向记录「过/不过」，只对**过**的方向考虑开轮 |

**P0 的 J2 反证是设计核心，不是形式**：C2 attempt07 正是因为守卫「按构造恒零」而把一个 bug 写成物理结论
（决策 0105）。所以 P0 必须构造一个**必须为正**的反例（粗 band 加能量、细 band 清零的假场），
并让它在错误实现下真的非零。

**P1 的四个端点**（同一 treatment 清单、同一 recipe，只有 updates 不同）：

| 点 | epoch | updates | 72 h t2m skill |
| --- | ---: | ---: | ---: |
| p0025 | 0.25 | 2,008 | +0.098082 |
| p1000 | 1 | 8,033 | +0.159293 |
| p4000 | 4 | 32,132 | +0.230544 |
| p1600 | 16 | 128,528 | +0.260668 |

**这个阶梯本身就是一条独立证据**：若缺口随 ×64 的训练量**不缩小**，说明它是损失函数几何决定的
（与 AMSE 的解析论证一致），而不是训练不足；若缩小，则 AMSE 的动机弱，应当停。

### §3.2 Stage Q — 谱几何损失 on/off（≈12 GPU-h）

**关键设计判断：Q 轮从 p1600 端点微调，不是从迁移父从零训。**

理由：AMSE 原论文自己就是**微调**（¼°/13 层 GraphCast fine-tune，约 26.2 GPU-days），不是从零训。
所以「从 p1600 端点微调 4,000 updates」不仅便宜 20 倍（1.95 GPU-h/臂 vs 40.33 GPU-h/臂），
而且**更贴原文的方法学**。

- 2 臂（MSE 续训 / 谱几何损失续训）× 3 seed（41/42/43）× 4,000 updates
- 每臂 ≈ 4,000 × 1.1096 s + 2,588 s 评分 ≈ 7,028 s ≈ 1.95 GPU-h
- 合计 ≈ **11.7 GPU-h**（含 prepare/archive，按 20 估上限）
- **预声明出口**：72 h t2m skill 三 seed 配对 Δ **全 ≥ 0** 才算正面；任一为负则如实记为 negative，
  **不调权重、不换项、不加 seed、不换名重开**（照 C2 的关闭纪律，决策 0109）

### §3.3 Stage R — 后处理振幅校正（≈5–15 GPU-h，纯评估、零训练）

若 P1 证实 `amp_ratio ≈ coh`，则「按相干性反演振幅」可以做成**后处理校正**：

- 用 **train 年**（2000–2021）拟合校正系数，在 **2022 val** 报告（**不碰 2023 test**）
- 用**同一已注册 D2 分母**（`sha256 096c384a…`、`n_selected_steps 32144`、train-only 2000–2021）
- 不改任何权重、不开训练轮
- **预声明出口**：五 lead 的 skill 与 RMSE 变化，**不劣化 24/48/72 h** 为通过条件

这是一条**方向由解析式给出、不是调参**的改进路径，而且**成本最低**。

---

## §4 成本表

| Stage | 内容 | GPU-h | 口径 |
| --- | --- | ---: | --- |
| **P** | 前置探针（P0/P1/P2） | **0** | 纯 CPU 只读，按 C1 前置同一方法 |
| **Q** | 谱几何 on/off，2 臂 × 3 seed × 4,000 updates | **≈11.7**（按 20 估） | 从 p1600 端点微调 |
| **R** | 后处理振幅校正 | **≈5–15** | 纯评估，无训练 |
| **合计** | | **≈17–35** | 对比 C2 的 66.99 GPU-h |

**参照**：从零 16 epoch 的 Q 轮会是 6 臂 × 40.3329 = **242 GPU-h**。微调路线省 20 倍。

---

## §5 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1 的任何变体（无目标约束幅度） | 方向反已实测 8/8（决策 0094） |
| 重开 C2（导数域目标匹配）的任何形式 | 三 seed 一致退化，永久关闭（决策 0109） |
| 引入 `pywt` / `pyshtools` / `ducc0` / `torch_harmonics` | 改依赖即改 `training_code_sha256` 与执行闭包；且区域网格上 SHT 装也没用 |
| 单独用 `rfft2` 不加尺度分解 | 正是论文 §4.2 点名的反模式（会「cover up」细尺度不足） |
| 把 `rfft2` 近似称作 AMSE | 必须称 *AMSE-analogue on a limited-area grid* |
| 引入 legacy `physical_consistency.py` | 打包了已关闭/未授权的 FFT、梯度、日变化、拖曳项（决策 0092 明文） |
| 在 P1 出结果前开 Q 轮 | 缺口若已随训练缩小，AMSE 动机消失，开了就是白烧 |
| 读 2023 test | `r = 0` |
| 把「物理更正确」当结果 | 物理正确性是**设计理由**，不是**结果证据** |
| 承诺训练臂一定涨 skill | 探针若显示缺口已缩，本计划的主线就**只剩证伪章**，如实登记 |

---

## §6 workflow 编排与提示词

执行形态是**一份 goal 长文 + 一个派生 workflow**，两者合在 objective 里（照
`docs/goals/s3-throughput-and-physics-campaign.md:18` 的实际形态，不是两份独立提示词）。

- **goal 长文**：`docs/goals/s3-physics-informed-campaign.md`，头部 `<!-- round-node: S3 -->` +
  `<!-- round-spec: {...} -->`；`round-spec` **初值指向 P 轮**（`device_mode: cpu`、`seeds: []`），
  由 workflow 每轮改写为 Q/R 轮
- **workflow**：`.zcode/workflow-drafts/r7-physics-informed-round.dwf.ts`，从
  `.zcode/workflow-drafts/r7-campaign-round.dwf.ts` 派生（**不是** `r7-throughput-physics-round.dwf.ts`
  ——那个是 GPU-seed 形状）
- 硬约束（决策 0044）：自跑 `round-guard`/`commit-guard`/`verify` 不依赖 hook；只推 `r7/weather-reasoning`；
  有界，`report()` 与 CI 绿都不是科学结论
- 零成本编译校验：`CreateWorkflow` + `path` + `args: {maxRounds: 0}`

---

## §7 风险（如实列）

1. **最贵的一环仍未验证**：谱几何损失在真实训练 batch 上是否可用（`moved` 里是否恒有
   `physical_targets`、N 是否恒等于 rollout 的 N），要等 Q 轮第一次冻结才在真实路径上被检验
   —— 这与 C2 attempt07 的教训同类。缓解：Q 轮冻结前的 preflight 必须走**部署配对**，不是 `target` 对 `target`。
2. **`rfft2` 的周期接缝是真实近似**：区域块左右/上下周期延拓会引入人工不连续。P0 必须**量化**它；
   若接缝能量占比不可忽略，结论的可信度要打折，并写进 limitations。
3. **方域径向分带是选择**：band 边界由我定，会被质疑为调参。缓解：**冻结**并写进协议，且
   DCT-II 交叉校验必须得出同一结论。
4. **P1 可能显示缺口已缩**：那时本计划的主线只剩证伪章，正面结果要靠 R 或下一期的应用章。
   **这是预先写下的可能结局，不是事后解释。**
5. **n = 1 模型、单实例、单分辨率**：C1/C2 的适用域限制同样适用，不得外推。
6. **4 个端点是独立 GPU 运行之差**，带未量化运行间方差，不是显著性结论。

---

## 实际结果（Actual Results）

**NOT EXECUTED YET.** 本计划于 2026-11-11 归档，**尚未执行任何 Stage**。

- **已发生（只读核查）**：
  - 起点 HEAD `ddd9104`、工作树干净；
  - `docs/DATA_SPEC.md:36` 确认为**区域**网格 27–43°N/107–123°E（**推翻了 0029 §4.1 与文献页 §4.2
    的「global 65×65」断言**，见 §2.3）；
  - venv 实测：`scipy`（含 `dctn`）与 `numpy` 在，`pyshtools`/`ducc0`/`torch_harmonics`/`pywt` **不在**；
  - `model/r7_baselines.py:117` 是全仓唯一的 `rfft2` 用法，且是模型内部结构不是损失；
  - `training/r7_long_rollout_runner.py` 已 **597/600 行**（R-051），谱损失必须另开模块；
  - arXiv MCP 一手核对 `2501.19374v2` §4.2，取回 local-area 限制与反模式原文（见 §2.3）。
- **未发生**：P0/P1/P2 全部**没有跑**；Q 与 R **没有开轮**；**0 GPU-h**；没有冻结任何协议；
  没有修改任何模型/训练代码；没有读 2023 test。
- **未验证**：§3.1 P1 的缺口是否随训练量缩小；§3.3 R 的校正能否不劣化 24/48/72 h；
  §7.1 的接线风险（`moved` 里 `physical_targets` 是否恒在）。
- **本计划的下一步判据**：P0 跑完并给出分解合法性表之后，状态改为「已测量」，
  并按 P1 的缺口趋势决定 Q 开或跳过。在此之前，**任何引用本计划作为「AMSE 在本模型上有效」的证据都是错的**。
