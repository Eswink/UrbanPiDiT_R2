# 0112 — 阶梯准入 C→A→B：β_j 归一化、`skip-q` 判据缺陷、以及「预训练期」作为未测制度

- **日期**：2026-11-12
- **状态**：proposed —— Step A 的读数出来之后才转 accepted
- **范围**：`training/r7_spectral_geometry.py`（`_terms` 奇点修复 + 待加的 `beta_weights`）、
  `training/r7_objective_axes.py`（已有 `spectral_geometry` 轴）、
  `scripts/r7_s3_gap_decay_and_beta.py`（Step C 探针）。
  **本条 Step C 已实跑、0 GPU-h；Step A/B 未开轮。**
- **依据**：`docs/plans/0031-physics-informed-sequence-c-a-b.md`（本条的归档计划）、
  `docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`（Stage P 读数）、
  `arXiv:2509.17601`（FastNet，一手，2026-11-12 读取）、决策 **0110**（区域分块分解）、
  **0111**（Stage Q 预声明跳过，§136-137 留「新的理由」开口）、
  `docs/R7_EVIDENCE_INDEX.jsonl:83`（空间梯度关闭条款逐字）

## Context

**① 前一版主线的三级判据全部触发，但其中一级的判据本身有缺陷。**

Stage P 判 `skip-q` 的依据是「细带缺口随训练量单调下降」（`0.27871 → 0.25531 → 0.23296 →
0.22690`）。单调性是真的，但它**不含速率信息**。本轮把速率测了：拟合 `gap ~ N^-α` 得
`α = 0.05111`（细带均值，残差 2.57e-02，逐带 ∈ [0.04140, 0.10805] 全部可识别）。
把缺口再减半需要 `0.5^(-1/α) = 775,102×` 训练量 ≈ **3.07×10⁷ GPU-h**。

⇒ **缺口形式上是「随训练缩小」，实质上是「任何可及预算都消不掉」。** `skip-q` 的文字判据被满足，
而它想表达的结论（欠训练是主因）被否定。这正是决策 0111 §136-137 保留的开口：

> 若将来出现**新的**理由（例如 P1 的端点统计被发现有问题、或改用不同端点集合），
> 那时立 ADR 是正确做法——**本决定不禁止**。

**这不是「事后放宽判据」**：`skip-q` 的判定**保持不变**，Q 仍然记为 `predeclared-skip`、仍然不产生
`readings.json`/`protocol.json`/`code.zip`。本条改变的是**下一级**的准入依据，不是回溯改写 Q。

**② 一手文献给出了一个与已关闭分支精确互补、且拆成可分离设置差异的先例。**

`arXiv:2509.17601`（FastNet）§4.1.2 式 (8)：

```
β_j = [⟨AMSE_j(x^{d_0}, x^{d_0+τ})⟩_year]^{-1}
```

逐字理由：「geopotential spectra span more than six orders of magnitude on the O96 grid, whereas
specific humidity spans only three. **Without this adjustment, the MSH loss tends to overweight
v-winds and humidity while underweighting temperature and geopotential.**」

本轮实测本仓的对应量（归一化空间，即损失所见空间）：**β_j 跨 86.75 倍**，
其中 `z250` 占 39.84%、`z500` 占 20.32%（合计 **60.16%**），而 `v10`/`u10`/`v500` 各仅约 0.5%。
无权重池化给每个变量 1/17 = 5.88%。**方向与论文描述一致：位势被严重欠加权、风场被严重过加权。**

**③ 我们的负结果被文献解释，并由此产生一个新制度。**

`docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md` 的实测：15 个 (seed, lead) 单元中 14 个方向正确、
效应极小、**72 h 绝对 skill 三 seed 全负**。FastNet §4.2.1 逐字：

> 「the model was **pre-trained** with horizontal gradients in the loss function, rather than
> adding them only in fine-tuning. This approach best suppressed artefacts … By contrast,
> introducing gradients solely at fine-tuning led to rapid RMSE degradation, while artefact
> reduction appeared only after extensive training (~30% of the pre-training schedule).」

我们那轮**正是** fine-tuning-only。所以它是文献预测的失效模式，而不是「梯度项无用」。

**④ 封锁条款的逐字范围必须核。**

`docs/R7_EVIDENCE_INDEX.jsonl:83` 与 `docs/R7_S3_SPATIAL_GRADIENT_LOSS_RESULT.md:43`：

> 不扫 λ、不加 seed、不调 2400 updates、不与 anchor/feedback/architecture/容量/K 轴组织组合、不换名重开

**「预训练期加入」与「零均值单位方差归一化」不在这五条内。** 但它们与 fine-tuning-only 的结论
绑在一起，所以 B 仍须本条之外的**新 ADR** 明确取代那部分，不得由本条代劳。

## Decision

**采用阶梯 C→A→B，逐级门控；任一级出现有效果即停在那里。**

### D1 —— Step C 作为 A 的准入前置（已执行）

`β_j` 与衰减率是两个**可失败**的测量。判据：

- 细带 `α` 不可识别 ⇒ `verdict = unresolved`，不开 A
- `updates_to_halve_gap ≤ 1000` ⇒ `verdict = close-line`，说明 `skip-q` 实质正确，不开 A
- `updates_to_halve_gap > 1000` ⇒ `verdict = open-a`

实测 `775,102 > 1000` ⇒ **`open-a`**。

**阈值 1000 的来源要说清**：它是「把缺口减半所需的训练量倍数」这一读出量的分界，不是扫出来的
参数。1000× 于 128,528 updates 已经是 1.28×10⁸ updates ≈ 4×10⁴ GPU-h，远超本仓任何可及预算。

### D2 —— Step A 的唯一改动是 `β_j` 归一化

不给 AMSE 加新说明项、不改分带、不加可调权重。**`β_j` 是 parameter-free 的**（由真值自身的 AMSE
导出，无自由参数），这一点与决策 0110「AMSE 的价值在于 parameter-free」一致。

`β_j` 必须在**归一化空间**计算。实测对照：物理空间跨 **1.16×10¹¹ 倍**、归一化空间跨 **86.75 倍**。
用物理空间会把整项重标定 11 个数量级，而损失看到的是归一化场。

### D3 —— 修掉 `_terms` 的 `coh = −1` 奇点（已执行）

`_terms` 为获得精确零把 `2·max(P,Q)·(1−coh)` 有理化为
`2·M·(P·Q−cross²)/(P·Q·(1+coh))`。在 `coh = −1` 处分母为零、分子是浮点残差，返回 **`inf`**。
正确极限是 `4·max(P,Q)`（**最大**惩罚）。

实测（真实 6 h `v10` 对，store frame 1464）：`P=802.585`、`Q=93.396`、`cross=−273.785`
⇒ `inf`，字面值 **3210.34**。本轮 1459 个真值帧中命中 **108/201**（前 201 对）。

**修法**：`1+coh` 不为正时用字面形式，其余保留有理化形式（维持精确零）。
**否决**「在 `coh ≤ −1` 处置零」：那会把「完全反相」报成「完美匹配」，比 `inf` 更糟。

### D4 —— Step B 只在 A 失败之后，且须新 ADR

B 的三处改动（预训练期、单位方差归一化、逐变量权重）**同时**施加，照 FastNet §4.2.1。
但 B 触及封锁条款的适用域，**本条不授权 B**：B 必须有自己的 ADR 0113，逐字引用 FastNet §4.2.1
的 ablation，并写明「原结论（fine-tuning-only 有害）仍然成立，只是被文献指出适用域」。
**不做 λ 扫描、不加 seed、不改 2400 updates、不与其它已关轴组合。**

## Consequences

**正面**

- Step C 用 **0 GPU-h** 给 A 提供了两个独立的准入依据：衰减率（`α = 0.051`）与量级失衡（`β_j` 跨 86.75×）。
  C1 的前置否决省了约 64 GPU-h，本轮的 C 结构相同但更便宜（6 min CPU）。
- 修掉了一个会让训练损失返回 `inf` 的真实缺陷。它不是「可能有问题」：在真实数据上 108/201 命中。
- `β_j` 是**无自由参数**的，满足决策 0110 对「parameter-free」的坚持，也不进入「扫参数」的禁区。
- 三处文献依据都是**逐字读到的正文**（§4.1.2、§4.2.1、§5），不是摘要转述。

**负面（如实列，R-033）**

1. **`β_j` 不是论文那个量。** FastNet 的 `β_j` 是**球谐阶** AMSE 的逆；本仓在区域 patch 上只有
   平面 `rfft2` 分带（决策 0110）。所以它是 **β_j AMSE-analogue 归一化**，**不得**称「复现 FastNet 式 (8)」。
   **代价**：论文里 1,250 km → 160 km 那类数字不能搬来当自己的结果。
2. **单年（2001）估 `β_j`。** 论文报告年际方差极小，本仓**未测**年际方差。这是一个未量化假设。
   **代价**：若年际漂移大，`β_j` 本身会带一个未记录的偏差。
3. **修 `_terms` 改了 `training_code_sha256`。** A 轮的父迁移与代码身份必须重做，不能沿用 Stage P 的身份。
   **代价**：多一次父迁移的开销，且 A 与 Stage P 不再是同一代码身份。
4. **A 只有 4,000 updates。** FastNet 的微调日程是 36,525 + 4×7,305 步，我们短得多。
   **代价**：若 AMSE 需要更长适应期，4,000 步的阴性可能是剂量不足造成的假阴性——
   这正是 C1 栽过的坑（`s_^-2` 量纲事故让轴完全不能作用）。缓解：出口 (a) 判**细带振幅比**这个
   AMSE 的直接作用点，比 skill 更早能看到信号；且 (e) 预检要求 `β_j` 非全相等、`amse` 全有限。
5. **本条不保证 A 会赢。** 若 (a)–(c) 任一为假，A 按预声明关闭，如实记负。**这可能意味着整条
   物理信息线在 C2/P/R/A 四连负之后结束**，论文的正面结果只能靠证伪章与机制发现。
6. **D1 的 1000× 阈值是一个读出量分界，不是统计检验。** 它足够粗（实测值 775,102 远超它），
   但不能用来声称「α 显著大于零」。
7. **本条不授权开 B。** B 需要 ADR 0113，且只在 A 失败后。

## 与其它决策的关系

- **不冲突 0111**：Q 仍记 `predeclared-skip`，Q 的判定与产物状态**一字未改**。本条改变的是**下一级**
  的准入依据，并明确引用 0111 自己留下的「新的理由」开口。
- **受约束 0110**：`β_j` 是 AMSE-analogue 的归一化，不是球谐形式；命名纪律沿用 0110 的三项近似声明。
- **受约束 0084 §6**：族判别论证仍须可检验。本条的 J4 落在「`β_j` 是 parameter-free 且由真值导出」
  这一点上，而不是「换了个物理名词」。
- **封锁条款**：本条**不解除** `不扫 λ、不加 seed、不调 2400 updates、不与其它已关轴组合、不换名重开`。

## 恢复第一条命令

```bash
.venv/bin/python tools/r7_round.py state
```
