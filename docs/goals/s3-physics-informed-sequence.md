# S3：物理信息主线阶梯 C→A→B —— β_j AMSE-analogue 归一化与「预训练期」制度

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-physics-informed-sequence-a", "driver": "scripts/study_r7_s3_spectral_geometry_a.py", "output_stem": "r7_s3_spectral_geometry_a_20261112", "control_record": "s3-spectral-geometry-stage-p", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_spectral_geometry_a.py", "tests/test_r7_s3_gap_decay_and_beta.py", "tests/test_r7_spectral_geometry.py"], "commit_paths": ["scripts/r7_s3_spectral_geometry_a_protocol.py", "scripts/study_r7_s3_spectral_geometry_a.py", "scripts/r7_s3_gap_decay_and_beta.py", "tests/test_r7_s3_spectral_geometry_a.py", "tests/test_r7_s3_gap_decay_and_beta.py", "training/r7_spectral_geometry.py", "docs/goals/s3-physics-informed-sequence.md", "docs/plans/0031-physics-informed-sequence-c-a-b.md", "docs/decisions/0112-sequence-admission-beta-normalisation.md"], "commit_message": "feat(r7): Stage A -- beta_j AMSE-analogue normalisation on/off, three seeds", "register_paths": ["docs/R7_S3_SPECTRAL_GEOMETRY_A_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-physics-informed-sequence.md"], "register_message": "docs(r7): register Stage A beta_j readings", "planned_seconds": 115200.0, "hard_cap_seconds": 230400.0, "code_archive": "outputs/r7_s3_spectral_geometry_a_code.zip"} -->

本文件是**唯一执行源**；机器可读的 `round-spec` 在上方。
设计依据是不可变记录 `docs/plans/0031-physics-informed-sequence-c-a-b.md`（全部实测数字、成本表与负面后果在那里）。
准入裁决是决策 **0112**；上游读数见 `docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`（Stage P，0 GPU-h）与
决策 **0110**（区域分块分解）、**0111**（Stage Q 预声明跳过，其 §136-137 留了「新的理由」开口）。
`campaign-state.current_round_goal` 必须指向本文件，否则 `tools/r7_round.py` 会解析到别的轮次并冻结错协议。

**状态：prepared（2026-11-12 建立，未启动 Step A）。** `scientific_claim: false`、`test_read: false`、`r = 0`。
**Step C 已执行完毕（0 GPU-h）**，其读数见 §1.2；Step A 未启动；Step B 不在本 objective 授权范围内。

---

## §0 Objective（单段，实测 3,574 字符）

> 在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 执行目标长文 docs/goals/s3-physics-informed-sequence.md（先完整读它，判据与相位以它为准；设计与实测数字见 docs/plans/0031-physics-informed-sequence-c-a-b.md，准入裁决见决策 0112）。毕业论文主线：基于深度学习的物理信息混合天气预测。起点：HEAD 487dc6a、used_gpu_h 571.500957、index 109 条、账本 89 行。阶梯 C→A→B，逐级门控，任一级出现有效果就停在那里。Step C 已执行完毕（0 GPU-h）：缺口衰减 alpha=0.05111（细带 5-lead，残差 2.57e-02，逐带 0.04140-0.10805 全部可识别），把缺口减半需 775102 倍训练量约 3.07e7 GPU-h，故缺口形式上随训练缩小但实质上任何可及预算都消不掉；beta_j 归一化空间跨度 86.75 倍（z250 占 39.84%、z500 占 20.32% 合计 60.16%，v10/u10/v500 各约 0.5%，而无权重池化给每个变量 1/17=5.88%），物理空间对照跨度 1.16e11 倍，证明逐变量谱权重必须与损失同空间；verdict=open-a。顺带修掉 training/r7_spectral_geometry.py::_terms 在 coh=-1 处返回 inf 的真实缺陷（实测真实 6h v10 对 P=802.585 Q=93.396 cross=-273.785，有理化形式 inf，字面极限 3210.34，本轮 1459 帧中命中 108/201），修法是 1+coh 不为正时用字面形式、其余保留有理化形式以维持 amse(x,x) 精确为 0，否决在 coh<=-1 置零（那会把完全反相报成完美匹配）。Step A 的交付物 D1 Step A 冻结协议（含 beta_j 实测块与 (e) 预检）· D2 代码身份归档（带 commit 注释）· D3 逐 seed receipt（两臂各一份，含父身份与 s/update）· D4 十七变量 x 五 lead 读数（含 exit_conditions）· D5 证据页与四件套登记（index record、账本行、campaign 8 节、campaign-state）· D6 判定点：逐条写 (a)-(e) 真或假并决定 B 开或整条线停。若 Step C 的 verdict 不是 open-a（重跑后 alpha 不可识别或 updates_to_halve_gap 不大于 1000），则不执行 A、直接写判定点并停。本 objective 执行 Step A（Step B 仅在 A 失败后且须新 ADR 0113，本 objective 不授权）。Step A 的唯一改动：给已有的 spectral_geometry 轴加决策 0112 的 beta_j AMSE-analogue 归一化（FastNet 式 8 的精神，arXiv:2509.17601 4.1.2 逐字理由：geopotential spectra span more than six orders of magnitude whereas specific humidity spans only three，without this adjustment the MSH loss tends to overweight v-winds and humidity while underweighting temperature and geopotential），从 p1600 端点微调 4000 updates（AMSE 原文自己就是微调），2 臂 x 3 seed，batch 4，注册 recipe 逐字不变，beta_j 在归一化空间由 train 年真值连续 6h 对算出并冻结进 protocol。预声明出口任一为假即永久关闭该支：(a) 细带 amp_ratio 相对同 seed off 臂升高；(b) 72h t2m skill 三 seed 配对差全不小于 0；(c) 24/48h skill 配对差三 seed 全不小于 0；(d) off 路径逐位等价，由 dispatcher 的 None 早退加反证测试支撑，不由读数自证（0109 的教训：c=True 是字面量、伪造读数仍 all=True）；(e) 预检 beta_j 在真实 batch 上非全相等且 amse 在全部 8 case x 12 step 上有限。不调 lambda、不扫 beta、不加 seed、不改 4000 updates、不与其它已关轴组合、不换名重开。执行方式：用 workflow 编排，先加载 dynamic-workflows 技能（未加载时 CreateWorkflow 拒跑），从 .zcode/workflow-drafts/r7-campaign-round.dwf.ts 派生 .zcode/workflow-drafts/r7-physics-informed-sequence.dwf.ts，必须用 CreateWorkflow 传 path 加 args 启动而不要用 AmendWorkflow（上一轮实测 AmendWorkflow 丢 args 让 dryRun 静默变 true），maxRounds 自设硬上限、world.run 命令名保持编译期字面量且运行值落盘到 round-spec、自跑 tools/r7_round.py round-guard 与 commit-guard 与 verify 不依赖 hook 兜底、只推工作分支 r7/weather-reasoning；先 CreateWorkflow 传 path 加 args maxRounds 0 dryRun true 做零成本编译校验；它是有界可恢复的批量编排器不是守护进程，被打断按 ResumeWorkflowRun 续。四个坑必须避开：桥不得要求零 query 聚合逐位等于 mean(dim=1)（实测差 2.98e-08，照决策 0082 改写）；code.zip 必须带 commit 注释否则 archive 拒绝启动；跑 stage 前先跑 tools/r7_round.py state 核对 current_round_goal；Step C 轮的 round-spec 必须存在但绝不许调 freeze。接线硬约束：不得改 training/r7_long_rollout_runner.py（已 597/600 行 R-051），beta 归一化加在 training/r7_spectral_geometry.py 并从 training/r7_objective_axes.py 那一个分发点进入。禁止：重开 C1/C2、重开 fine-tuning-only 的空间梯度支（其结论成立且已被 FastNet 4.2.1 预言）、扫 lambda、加 seed、改 2400 或 4000 updates、与其它已关轴组合、引入 pywt 或 pyshtools 或 ducc0 或 torch_harmonics、改 Stage R 的 per_case_ratio schema、读 2023 test、重建 store、动 data/raw|interim|processed 与归档快照、合成数据、改冻结判据、事后放宽出口、删弱测试、绕 checkpoint 与数据身份校验、paid 与 rental 与 exclusive 与 cron 与守护与会话外续跑、对邻居进程发任何信号、main 合并与 force 与关 issue、自宣科学通过或 SOTA 或最终 goal 完成。只在三种情况停：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在 8 节写为何停加恢复第一条命令。每步给可核查证据（命令加输出加 digest），估计值不得写成实测值。不承诺 A 一定有正面结果：若 (a)-(c) 任一为假，如实记负并停，那可能意味着物理信息线在四连负之后结束、正面结果只能靠证伪章与机制发现。结束按 AGENTS.md 汇报改动、实验与失败、技能与代理使用、验证与精确 CI、成本与 issue 状态、未做项与下一项。

---

## §1 现状与起点

- 工作分支 `r7/weather-reasoning`；起点 HEAD `487dc6a`（工作树干净）
- 节点 S3；`used_gpu_h 571.500957`、`remaining −551.500957`（**仅会计**，决策 0030/0038 后无总 GPU-h 上限）
- 证据索引 **109** 条、账本 **89** 行
- 2023 test **未评分**、`r = 0`；S4 未启动

### §1.1 为什么需要这个阶梯

前四轮物理信息几何**全部为负**：

| 轮次 | 形式 | 结果 | 成本 |
| --- | --- | --- | --- |
| C1 | 无目标约束束 | 真值被罚 10.2×，8/8 例方向反 | 0 GPU-h（前置否决） |
| C2 | 导数域目标匹配 | 15/15 (lead, seed) 全负 | 66.99 GPU-h |
| Stage P | 谱几何前置 | P0 过、P1 判 `skip-q`、P2 三方向全不过 | 0 GPU-h |
| Stage R | 后处理振幅校正 | 85/85 (变量, lead) 全劣化 | 0 GPU-h |

本阶梯**不发明新算子**。一手文献 `arXiv:2509.17601`（FastNet）给出三个**可分离的设置差异**，
见 `docs/plans/0031-physics-informed-sequence-c-a-b.md` §0.1。

### §1.2 Step C 已执行的读数（0 GPU-h，`scientific_claim: false`）

| 量 | 实测 | 含义 |
| --- | --- | --- |
| 细带缺口衰减 `α`（5-lead） | **0.05111**（残差 2.57e-02） | 减半需 **775,102×** 训练量 ≈ 3.07×10⁷ GPU-h |
| 逐带 `α` | 0.04140 – 0.10805，全部可识别 | 无一带显示「可训练消除」 |
| 12-lead 口径 `α` | 0.05723（残差 2.06e-02） | 换 lead 集不改变结论 |
| `β_j` 跨度（归一化空间） | **86.75×** | z250 占 39.84%、z500 占 20.32%（合计 60.16%） |
| `β_j` 跨度（物理空间，对照） | 1.16×10¹¹× | 权重必须与损失同空间，否则重标定 11 个数量级 |
| `verdict` | **`open-a`** | Step C 的两个门都通过 |

**Step C 同时修掉一个训练缺陷**：`training/r7_spectral_geometry.py::_terms` 在 `coh = −1` 处返回
`inf`（真实 6 h `v10` 对：`P=802.585`、`Q=93.396`、`cross=−273.785`；字面极限 **3210.34**；
1459 帧中命中 108/201）。修法：`1+coh` 不为正时用字面形式，其余保留有理化形式以维持
`amse(x,x)` **精确为 0**。**否决**「在 `coh ≤ −1` 置零」——那会把完全反相报成完美匹配。

### §1.2b 执行 Step A 时纠正的三条前提（2026-11-12 实测）

上一条与 0112 代价③都写「这改了 `training_code_sha256`，Step A 必须重做父迁移」。
**实测不成立**，三条一并更正：

| 主张 | 实测 | 证据 |
| --- | --- | --- |
| `_terms` 修复改了 `training_code_sha256` | **没有**。`training/r7_spectral_geometry.py` 不在 runner 的 `SOURCE_FILES`（17 个）里，带/不带该修复两侧 digest **都是** `e767411883ce…` | `git stash push -- training/r7_spectral_geometry.py` 前后各跑一次 `training_code_digest()` |
| 父迁移必须重做 | **不需要**。`model_code_sha256` 仍是 `5934071062…`（与 `parent_pins.MODEL_CODE_SHA256` 相等），三个父 checkpoint 文件 SHA 与 pin 相等，`load_checkpoint` 全部通过并报 `updates=1600` | 逐 seed 跑 `pinned_parent` + `load_checkpoint` |
| `spectral_geometry` 轴接线完整 | **不完整**：`spectral_geometry_term` 返回**裸 tensor**，而 `_update` 执行 `result.loss + constraint["total"]` ⇒ 第一次 update 就 `IndexError: too many indices for tensor of dimension 0`。Q 轮被跳过，所以这个接口不匹配从未暴露 | 直接调用实测；修法与反证测试见 §1.3 与 `tests/test_r7_s3_spectral_geometry_a.py::test_the_axis_matches_the_runner_contract` |

改动留痕落在 `execution_files_sha256` 上（它覆盖 spectral 模块），不落在
`training_code_sha256` 上。**父迁移未做，因为它是无对象的操作**——为一条错误前提再跑一次迁移
只会产生一个与现有 pin 逐位相同的新目录。

**第四条纠正（来自一手文献）**：`arXiv:2509.17601` 式 (10) 是
`Σ_j s_j w_j β_j Σ_k γ_k AMSE_{j,k}` 的**纯求和**——不除以 J、不除以 Σβ_j，且**没有**并行的
逐格点 MSE 项。所以 `β_j` 归一的实现是 `Σ_j β_j AMSE_j`，不是「MSE + 小修正」。
本仓 runner 无法移除 MSE（597/600 行，R-051），只能让 MSE 变成残差；实测剂量 58.0–95.2
使这个残差占 1.03%–1.69%，故 on 臂训练的就是论文 §4.1 的 MSH 制度本身。**这条在冻结前写定**
（`dose_decision`），不能事后从结果反读。

### §1.3 基础设施（复用，不新建）

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| `amse_analogue` | `training/r7_spectral_geometry.py` | 可微、精确零已守住、奇点已修 |
| `spectral_geometry` 轴 | `training/r7_objective_axes.py:31` | 已注册，`:52-54` 已接 `spectral_geometry_term` |
| `band_powers` | `training/r7_spectral_geometry.py:127` | **对 17 通道直接求和**——这正是 β_j 要修的 |
| 长 rollout runner | `training/r7_long_rollout_runner.py` | **597/600 行（R-051），不得改动** |

---

## §2 交付物清单

### Step C（已完成）

| # | 交付物 | 状态 |
| --- | --- | --- |
| **C-D1** | 缺口衰减律（逐带 × 逐 lead 的 α 与残差，两种 lead 集） | **已产出** `/tmp/stepc3.json` 的 `c1` 段 |
| **C-D2** | β_j 逐变量表（归一化 + 物理对照） | **已产出** 同文件的 `beta` 与 `beta_physical_control` |
| **C-D3** | 判定点 `open-a / close-line / unresolved` | **已产出** `verdict = open-a` |
| **C-D4** | 23 项反证测试 | **已过** `tests/test_r7_s3_gap_decay_and_beta.py` |

### Step A（本 objective 的执行范围）

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| **D1** | Step A 冻结协议，含 β_j 实测块与 (e) 预检 | `outputs/<stem>_attempt01/protocol.json` |
| **D2** | 代码身份归档（**带 commit 注释**） | `outputs/r7_s3_spectral_geometry_a_code.zip` |
| **D3** | 逐 seed receipt（两臂各一份，含父身份与 s/update） | `seed{41,42,43}_receipt.json` |
| **D4** | 十七变量 × 五 lead 读数（含 `exit_conditions`） | `readings.json` |
| **D5** | 证据页与四件套登记 | `docs/R7_S3_SPECTRAL_GEOMETRY_A_RESULT.md` + index record + 账本行 + master §8 + campaign-state |
| **D6** | **判定点**：逐条写 (a)–(e) 真或假，并决定 B 开或整条线停 | 结果页 §判定 |

---

## §3 判据与证据来源（**全部预声明，不得事后调整**）

本节每条判据指向已归档的写定地，**不在本文件新写阈值**：
`docs/plans/0031-physics-informed-sequence-c-a-b.md` §3（阶梯与出口）、
`docs/decisions/0112-sequence-admission-beta-normalisation.md`（D1–D4 的裁决）、
`docs/decisions/0110-limited-area-multiscale-decomposition.md`（分解与三项近似）、
`docs/decisions/0109-c2-derivative-matching-closed-on-three-seed-regression.md`（(d) 不得是字面量）。

### §3.1 Step A 的预声明出口（**任一为假即永久关闭该支**）

| 判据 | 要求 |
| --- | --- |
| **(a)** | 细带 `amp_ratio` 相对**同 seed** off 臂**升高**（AMSE 的直接作用点） |
| **(b)** | 72 h t2m skill 三 seed 配对 Δ **全 ≥ 0** |
| **(c)** | 24/48 h skill 配对 Δ 三 seed **全 ≥ 0** |
| **(d)** | off 路径**逐位等价**——由 dispatcher 的 `None` 早退 + 反证测试支撑，**不由读数自证** |
| **(e)** | 预检：`β_j` 在真实 batch 上**非全相等**，且 `amse` 在全部 8 case × 12 step 上**有限** |

**关于 (d) 的特别要求**：0109 记录了 C2 的 `c = True` 是字面量、`d` 只读两个自写常量，
伪造三 seed skill 为 0.5 仍 `all=True`。本轮的 (d) **必须**由「`axis is None` 早退」这条代码路径
加一个会失败的测试来支撑，**不得**写成常量。

**关于 (a) 的方向**：(a) 是「升高」，因为 AMSE 的作用是抬高细带振幅。若 (a) 为假，说明 β_j 归一化
没有作用到它声称的作用点，那无论 skill 如何都不该开 B。

### §3.2 剂量与族判别

- **剂量**：`β_j` 是 **parameter-free** 的（由真值自身 AMSE 导出，无自由参数），所以不设权重带。
  但协议必须记录 `amse_total / L_mse` 的实测比值，供 §4.3 的 J3 对照。
- **族判别**：`β_j` 归一化**不加新项**、不改分带、不改 AMSE 形式，只改**变量间的相对权重**。
  因此它不是「又一个物理约束」，与 C1/C2/空间梯度都不同族。这一点必须在协议里写明。

---

## §4 实施顺序（依赖，不跳步）

1. **先跑 `tools/r7_round.py state`** 核对 `current_round_goal` 指向本文件
2. 在 `training/r7_spectral_geometry.py` 加 `beta_weights`（**不改** `amse_analogue` 的默认行为：
   默认 off 时逐位不变）
3. 写 `scripts/study_r7_s3_spectral_geometry_a.py`，派生自 `study_r7_s3_derivative_matching_c2.py`
4. **重做父迁移**（`_terms` 修复改了 `training_code_sha256`）
5. 跑 (e) 预检 → 冻 `protocol.json` → `archive` → `stage-seeds`（3 seed × 2 臂）
6. 读数 → (a)–(d) 求值 → 四件套登记 → **D6 判定点**

---

## §5 预算与停止

| Stage | `planned_seconds`（软） | `hard_cap_seconds`（硬） | GPU-h |
| --- | ---: | ---: | ---: |
| **C** | 已执行（约 6 min CPU） | — | **0** |
| **A** | 115,200（32 h/seed） | 230,400（64 h） | ≈12（按 20 估） |
| **B** | **不在本 objective 授权内** | — | ≈40–77 |

软超预算**继续**并记 `soft_overrun_seconds`；**只有**硬上限因时长截断，记 `budget_limited`/`failed` 并全额记账。

**停止条件（只有三种）**：
1. 触保留授权边界（paid / rental / exclusive / 新数据下载 / main 合并 / 破坏性）；
2. 达已冻结的科学停止条件（(a)–(e) 任一为假 ⇒ 关此支；Step B 需要 ADR 0113 而本 objective 不授权）；
3. 确无可安全独立推进的工作。

停时必须在本文件 §8 写「**为何停 + 恢复第一条命令**」。

---

## §6 进度与交接

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```

**下一动作**（初始）：在 `training/r7_spectral_geometry.py` 加 `beta_weights`，写 Step A 驱动与测试，
`round-guard` 过关后重做父迁移，再冻协议。

---

## §7 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1 / C2 | 0084 §6、0094、0109 已按预声明关闭 |
| 重开 **fine-tuning-only** 的空间梯度支 | 其结论成立，且已被 FastNet §4.2.1 逐字预言 |
| 扫 λ、扫 β、加 seed、改 4,000 updates、与其它已关轴组合、换名重开 | 封锁条款 `docs/R7_EVIDENCE_INDEX.jsonl:83` 逐字禁止 |
| 开 Step B | 需 ADR 0113，且只在 A 失败之后 |
| 引入 `pywt` / `pyshtools` / `ducc0` / `torch_harmonics` | β_j 与单位方差归一化不需要 SHT |
| 改 `training/r7_long_rollout_runner.py` | 597/600 行（R-051） |
| 改 Stage R 的 `per_case_ratio` schema | 改它会改已登记 digest，须另立决策 |
| 读 2023 test | `r = 0` |
| 把 β_j 称作「复现 FastNet 式 (8)」 | 那是球谐阶 AMSE 的逆；本仓是区域分带 **AMSE-analogue** |
| 承诺 A 一定有正面结果 | (a)–(c) 任一为假即如实记负并停 |

---

## §8 进度块

**状态**：`Step A complete`（2026-11-13）。**判定：(b)(c) 为假 ⇒ 永久关闭 `spectral_geometry` 支。**

### Step A 已执行（21.825 GPU-h 总计：有效 15.999 + 浪费 5.826）

| 判据 | 真/假 | 读数 |
| --- | --- | --- |
| **(a)** 细带 `amp_ratio` 相对同 seed off 臂升高 | **真** | +0.125128 / +0.113816 / +0.116573（三 seed 全部 > 0） |
| **(b)** 72 h t2m skill 三 seed 配对 Δ 全 ≥ 0 | **假** | −0.066368 / −0.057539 / −0.062297 |
| **(c)** 24/48 h skill 配对 Δ 三 seed 全 ≥ 0 | **假** | 24 h 三 seed 全负（−0.0257 ～ −0.0323）、48 h 三 seed 全负（−0.0462 ～ −0.0512） |
| **(d)** off 路径逐位等价 | 支撑在位 | `training/r7_objective_axes.py:43` 的 `None` 早退 + `test_off_arm_matches_the_registered_arithmetic` 等两条反证测试实跑 `2 passed` |
| **(e)** 预检：β_j 非全相等且 amse 全有限 | **真** | 跨度 86.75497080940221×；8 case × 12 step 全有限；剂量 58.016–95.217 已在冻结前写定 |

**⇒ `verdict = permanently-close-the-spectral-geometry-branch`。** 交付物 D1–D6 全部产出：
冻结协议（`protocol_sha256 75897e51…`）、代码身份归档（zip comment = `9fb7a19`）、
逐 seed receipt（三份，各含父身份与 s/update）、十七变量 × 五 lead 读数（含 `exit_conditions`）、
证据页 `docs/R7_S3_SPECTRAL_GEOMETRY_A_RESULT.md`（`sha256 84c0699e…`，含 §6.4 的四次修复记录）、index record
`s3-spectral-geometry-a-negative`、账本行、master §8、campaign-state。

**这个负结果的性质要说准**：不是「项没起作用」。**(a) 三 seed 一致升高、臂间差比 seed 间离散大一个
数量级**，说明 β_j 归一化精确作用在了它声称的作用点上；而同一改动让 **9/9 个 (lead, seed) 单元 skill
全部下降，降幅随 lead 单调增大**（24 h ≈ −0.029 → 48 h ≈ −0.049 → 72 h ≈ −0.062）。
剂量 58–95 使 on 臂训练的就是论文 §4.1 的 MSH 制度本身，所以这是对该目标的直接检验，不是剂量不足。

**顺带修掉两个会让训练崩溃的真实缺陷**（都不是「可能有问题」）：
`spectral_geometry_term` 返回裸 tensor 而 runner 执行 `constraint["total"]`（该轴从未端到端跑过）；
`hermitian_weights`/`radial_band_index` 建在 CPU 而 `spectrum_x` 在 `cuda:0`（attempt01 两 worker
各 ~8,600 s 后崩）。**教训**：(e) 预检在 CPU 上跑，能证明「量级正确」但**不能**证明「能在 GPU 上跑」。

### 登记之后的四次修复（都不改读数，但都推翻过一句话）

产物登记（`cdfc94e`…）完成后，在干净树上跑全量测试与查 CI，又暴露四件事。读数一字未动，
但其中三件各自推翻了本轮某个先前的说法，所以记在这里而不是只写在提交里：

| 提交 | 修的是什么 | 推翻了什么 |
| --- | --- | --- |
| `e40562e` | 测试拆分遗留一条断言在父文件里（`NameError`）；C2 的 `AXES` 成员钉自 `0440993` 起就红 | `ce03b06` 自己写的「no assertion dropped」 |
| `99a3bd7` | evidence record 的 `beta_spread_max_over_min` 里装的其实是**与 Step C 的最大相对差** 4.71e-06，真跨度是 86.755×；页新增 §6.4 | 记录里一个字段的名实不符（页与全部文字表述一直是对的） |
| `e5f78e3` | CI 长期零绿的四个成因；其中一个是真的科学完整性缺陷 | 「全量测试绿 ⇒ CI 应该绿」这个未检验的假设 |

**`e5f78e3` 里那个真缺陷值得单独记**：`scripts/r7_s3_operator_j1_probe.py` 的
`_real_fields()` 文档承诺 `list[dict] | None`，却在降级路径返回 `None, None`。调用方解包一个值，
于是 `cases` 变成**二元组** `(None, None)`——**真值**——记录便写下 `"measured": True` 与一个空的
case 清单，下一行 `[entry['case'] for entry in cases]` 抛 `TypeError`。**它既没降级成功，也没说
自己没降级。** 这正是决策 0105 的 attempt07 失效模式在一个新位置。已修并反证（把 `curve.STORE`
指向不存在的路径，断言 `cases is None` 且 `measured: False`）。

**CI 的事实**（不是推断）：本仓最近 **100 次 `R7 CPU CI` run 的成功数为 0**（90 failure / 10
cancelled，总量 976）。这不是本轮引入的，也不是「这次特殊」——它是一个长期状态，本轮只是第一次
去查它。四个成因里 2 个是「测试硬依赖 git-ignored 的 `outputs/`」（按 `test_real_data_pipeline.py`
的既有约定改成带理由的 skipif，且把渲染器的缺文件降级分支**单独**留成无条件测试），1 个是
`scipy` 被**生产代码**延迟导入却未在任何 requirements/pyproject 声明（Q-011 那一类，已声明），
1 个是上面的 j1 缺陷。**四个都修了不等于 CI 一定绿**——那是关于下一次 run 的预测，不是实测。

**冻结身份未被触碰**：改依赖前先核过 `execution_files_sha256` 覆盖
`training/r7_spectral_geometry.py`（`c4953b2b…`）与 `training/r7_objective_axes.py`（`212ff51f…`），
两者哈希仍与冻结值相等，所以依赖修复只落在 `requirements.txt` / `pyproject.toml`，
Step A 的 96 文件执行闭包与 `model_code_sha256` 全部不变。

### 为什么停

**达已冻结的科学停止条件**：brief §3.1 预声明「(a)–(e) 任一为假即永久关闭该支」，实测 (b)(c) 为假。
Step B 不在本 objective 授权内——它须新 ADR 0113，且只在 A 失败之后；是否开 B 留待用户读过证据页后决定。

**未做**：Step B（未授权）；2023 test 未读（`r = 0`）。

**下一动作（S3）**：用户阅读 `docs/R7_S3_SPECTRAL_GEOMETRY_A_RESULT.md` 后决定是否以 ADR 0113 开
Step B（预训练期 + 单位方差归一化 + 逐变量权重，≈40–77 GPU-h），或按本质证负结果把物理信息线收束为
证伪章与机制发现。**在此之前不新开任何实验轮**。

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```
