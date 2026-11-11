# S3：物理信息混合预报主线（谱几何 → 平衡约束 → 正面结果）—— 65×65 多年实例

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-physics-informed-p", "driver": "scripts/r7_s3_spectral_geometry_probe.py", "output_stem": "r7_s3_spectral_geometry_20261111", "control_record": null, "seeds": [], "device_mode": "cpu", "test_path": ["tests/test_r7_s3_spectral_geometry_probe.py", "tests/test_r7_s3_operator_j1_probe.py"], "commit_paths": ["scripts/r7_s3_spectral_geometry_probe.py", "scripts/r7_s3_operator_j1_probe.py", "tests/test_r7_s3_spectral_geometry_probe.py", "tests/test_r7_s3_operator_j1_probe.py", "docs/goals/s3-physics-informed-campaign.md", "docs/goals/main-model-climatology-campaign.md", "docs/plans/0030-thesis-physics-informed-mainline.md", "docs/decisions/0110-limited-area-multiscale-decomposition.md", "docs/decisions/README.md", "docs/plans/README.md", "docs/plans/0029-physics-operator-candidates.md", "docs/R7_PHYSICS_LOSS_LITERATURE_REVIEW.md"], "commit_message": "feat(r7): Stage P spectral-geometry probe -- limited-area rfft2 bands plus DCT cross-check", "register_paths": ["docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-physics-informed-campaign.md"], "register_message": "docs(r7): register Stage P spectral-geometry and J1 readings", "planned_seconds": 10800.0, "hard_cap_seconds": 21600.0, "code_archive": null} -->

本文件是**唯一执行源**；机器可读的 `round-spec` 在上方。
设计依据是不可变记录 `docs/plans/0030-thesis-physics-informed-mainline.md`（统一论证、三个 Stage、成本表与负面后果在那里）。
分解方法由决策 **0110** 裁定。上游证据：`docs/R7_S3_PHYSICAL_CONSTRAINTS_C1_PREFLIGHT_RESULT.md`（C1，0 GPU-h 前置否决）、
`docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md`（C2，66.993549 GPU-h 永久关闭）、
`docs/R7_PHYSICS_LOSS_LITERATURE_REVIEW.md`（一手文献，及其 §7 更正）。
`campaign-state.current_round_goal` 必须指向本文件，否则 `tools/r7_round.py` 会解析到别的轮次并冻结错协议。

**状态：prepared（2026-11-11 建立，未启动）。** `scientific_claim: false`、`test_read: false`、`r = 0`。

---

## §0 Objective（单段，实测 3,828 字符）

> 在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 执行目标长文 docs/goals/s3-physics-informed-campaign.md（先完整读它，判据与相位以它为准；设计与实测数字见 docs/plans/0030-thesis-physics-informed-mainline.md，分解方法见决策 0110）。毕业论文主线：基于深度学习的物理信息混合天气预测。起点：HEAD ddd9104、used_gpu_h 571.500957、index 106 条、账本 89 行、current_round_goal 原指 C2 brief。三个已关闭结论不可重开：C1 无目标约束束（真值被罚 10.2 倍、8/8 例方向反，0 GPU-h 否决，决策 0094）、C2 导数域目标匹配（三 seed 配对差全负 -0.0116/-0.0163/-0.0134、15/15 lead×seed 全负，66.993549 GPU-h，永久关闭，决策 0109）、空间梯度损失（最优解是常数场）。统一论证（本轮核心，也是 J4 答案）：这三条不是三个失败而是同一病的三个侧面——模型场比真值平滑 2.9 倍（mean|div(u,v)| 8.5227e-06 对 2.5063e-05 s^-1，p95 2.3913e-05 对 7.2998e-05）；任何惩罚粗糙度的无目标项都推向更平滑即远离真值，这是 C1 的机制死因；只在单一派生量上匹配目标则剂量与 skill 抢梯度（C2 dose 0.2889-0.4109）。出路不是加项而是改损失函数的几何：标准 MSE 的最优解是 sigma_X=rho<1（E[MSE]=sigma_X^2+1-2 sigma_X rho 对 sigma_X 求极小），即振幅不足是最优解性质而非训练不足（arXiv:2501.19374v2 解析证明加实测，GraphCast 有效分辨率 1250km 到 160km）。事实修正：本仓 65x65 是区域 patch 不是全球网格（docs/DATA_SPEC.md:36：27.0-43.0N/107.0-123.0E、0.25 度、[65,65]），球谐在区域块上不良定义且 pyshtools/ducc0/torch_harmonics/pywt 均不在 venv；同一手论文 4.2 节写明该法对 global 网格简单但对 local-area 模型困难，并点名不加尺度分解会 cover up 细尺度不足这一反模式。所以一律称 AMSE-analogue on a limited-area grid，不得称 AMSE。按三 Stage 推进，全部预声明、一次跑到底。P（0 GPU-h 纯 CPU 只读）：P0 分解合法性硬门——rfft2 norm=ortho 的 Parseval 精确性、强制报出周期接缝能量占比（不设阈值，设阈值等于事后调参）、scipy.fft.dctn type=2 独立重算且两法结论必须一致否则结论作废；J2 反证必须能失败（构造只在粗 band 加能量、细 band 清零的假场，断言损失必须变差；且 pred=target 必须精确为 0），这是 C2 attempt07 把 bug 写成物理结论的教训（决策 0105）；P0 任一不过即停、不开 Q 并把理由写进进度块。P1 振幅与相干性：4 个训练端点（p0025 2008 updates、p1000 8033、p4000 32132、p1600 128528，72h skill +0.098082/+0.159293/+0.230544/+0.260668）乘 8 个 train case（1000/5000/9000/13000/17000/21000/25000/29000）乘 17 变量乘 5 lead，逐带算 amp_ratio(k)=sqrt(PSD_pred/PSD_target) 与 coh(k)，判据是缺口 1-amp_ratio 随训练量是缩小还是不变；缩小则动机弱、Q 跳过并写理由，不变或扩大则开路；绝不许设计成只会说好话。P2 J1 方向门：静力平衡（R/g*ln(850/500)=15.53 m/K、R/g*ln(500/250)=20.29 m/K，先核 z* 是位势高度还是位势、t* 是 K 还是摄氏度，单位错就是 C1 的 s^-2 事故重演）、整层散度积分（预期失败，作证伪章第二根柱子）、湿物理（q>=0、q<=q_sat，只在反归一化后的物理单位判），判据是 penalty(analysis)<=penalty(model) 逐例一致且 dose 在 [1e-3,1e-1] 内，逐方向记过或不过。Q（约 12 GPU-h，2 臂乘 3 seed 乘 4000 updates）：从 p1600 端点微调不是从迁移父从零训，因为 AMSE 原论文自己就是微调（四分之一度 GraphCast，约 26.2 GPU-days），每臂 4000 乘 1.1096 秒加 2588 秒评分约 1.95 GPU-h，比从零 16 epoch 的 242 GPU-h 省 20 倍且更贴原文方法学；出口是 72h t2m skill 三 seed 配对差全不小于 0 才算正面，任一为负则如实记 negative，不调权重不换项不加 seed 不换名重开。R（约 5-15 GPU-h，纯评估零训练）：若 P1 证实 amp_ratio 约等于 coh，则按相干性反演振幅做后处理校正，用 train 年 2000-2021 拟合、2022 val 报告、同一已注册 D2 分母（sha256 096c384a、n_selected_steps 32144、train-only），不碰 2023 test，出口是五 lead skill 与 RMSE 不劣化 24/48/72h。执行方式：用 workflow 编排，先加载 dynamic-workflows 技能（未加载时 CreateWorkflow 拒跑），从 .zcode/workflow-drafts/r7-campaign-round.dwf.ts 派生 .zcode/workflow-drafts/r7-physics-informed-round.dwf.ts（不是 r7-throughput-physics-round 那个 GPU-seed 形状），maxRounds 自设硬上限、world.run 命令名保持编译期字面量且运行值落盘到 round-spec、自跑 tools/r7_round.py round-guard 与 commit-guard（提交前）与 verify（登记后）不依赖 hook 兜底、只推工作分支 r7/weather-reasoning；先 CreateWorkflow 传 path 加 args maxRounds 0 做零成本编译校验；它是有界可恢复的批量编排器不是守护进程，被打断按 ResumeWorkflowRun 续。四个坑必须避开：桥不得要求零 query 聚合逐位等于 mean(dim=1)（实测差 2.98e-08，照决策 0082 改写）；code.zip 必须带 commit 注释否则 archive 拒绝启动；跑 stage 前先跑 tools/r7_round.py state 核对 current_round_goal；P 轮 round-spec 必须存在（round-guard 靠它跑测试）但绝不许调 freeze/stage/stage-seeds，驱动必须对 prepare/archive/seed 三个子命令显式抛错并有反证测试。接线硬约束：谱损失不得加进 training/r7_long_rollout_runner.py（已 597/600 行，R-051），必须另开 training/r7_spectral_geometry.py 并从 training/r7_objective_axes.py 那一个分发点进入。禁止：读 2023 test、重建 store、新建 129x129 源、动 data/raw|interim|processed 与归档快照、合成数据、改冻结判据、事后放宽出口、换名重开、删弱测试、引入 pywt 或 pyshtools 或 ducc0 或 torch_harmonics、绕 checkpoint 与数据身份校验、paid 与 rental 与 exclusive 与 cron 与守护与会话外续跑、对邻居进程发任何信号、main 合并与 force 与关 issue、自宣科学通过或 SOTA 或最终 goal 完成。只在三种情况停：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在 8 节写为何停加恢复第一条命令。每步给可核查证据（命令加输出加 digest），估计值不得写成实测值。结束按 AGENTS.md 汇报改动、实验与失败、技能与代理使用、验证与精确 CI、成本与 issue 状态、未做项与下一项。

---

## §1 现状与起点（带 file:line）

- 工作分支 `r7/weather-reasoning`；起点 HEAD `ddd9104`（工作树干净）
- 节点 S3；`used_gpu_h 571.500957`、`remaining −551.500957`（**仅会计**，决策 0030/0038 后无总 GPU-h 上限）
- 证据索引 **106** 条、账本 **89** 行
- 2023 test **未评分**、`r = 0`；S4 未启动

### §1.1 三条已实测事实（本轮的出发点）

| # | 事实 | 数值 | 出处 |
| --- | --- | --- | --- |
| 1 | 模型场比真值**平滑 2.9 倍** | `mean\|∇·(u,v)\|` 模型 **8.5227e-06** vs 分析场 **2.5063e-05** s⁻¹ | C1 前置 §3 |
| 2 | 无目标方程约束**方向反** | 真值被罚 **10.2×**，8/8 例（6.503–15.481） | C1 前置 §2 |
| 3 | 有目标导数匹配**一致退化** | 15/15 (lead, seed) 全负，均值 **−0.013785** | C2 结果 §2 |

### §1.2 统一论证

三条是同一条病的三个侧面：模型**小尺度能量不足**（事实 1）；任何惩罚粗糙度的**无目标**项都把预测推向更平滑、即远离真值（事实 2 是事实 1 的机制解释）；只在**单一派生量**上匹配目标则剂量与 skill 抢梯度（事实 3）。

⇒ **出路不是加项，是改损失函数的几何**（尺度的相对权重）。

### §1.3 一条必须遵守的事实修正

`docs/DATA_SPEC.md:36`：ROI **27.0–43.0°N / 107.0–123.0°E**、0.25°、`[65, 65]` —— **区域 patch**，不是全球网格。
球谐变换在区域块上**不良定义**；`pyshtools`/`ducc0`/`torch_harmonics`/`pywt` 四个包**都不在 venv**（实测）。
一手论文 `arXiv:2501.19374v2` §4.2 自己写明该法对 **local-area** 模型困难，并点名「不加尺度分解」是反模式。

⇒ 一律称 ***AMSE-analogue on a limited-area grid***，**不得**称 AMSE。方法见决策 **0110**。

### §1.4 代码面硬约束（实测）

| 项 | 状态 | 位置 |
| --- | --- | --- |
| 唯一 `rfft2` 用法 | 模型内部结构，**不是损失** | `model/r7_baselines.py:117` |
| DCT-II `dctn` | 可用（scipy 1.18.1） | venv |
| 球面散度/涡度/梯度/平流 | 可复用 | `model/r7_process_tensor_diagnostics.py:127,133,139,145` |
| 轴分发点（唯一入口） | 45 行，有余量 | `training/r7_objective_axes.py:41` |
| 长 rollout runner | **597/600 行，已满** | `training/r7_long_rollout_runner.py`（R-051） |

⇒ Q 轮的谱损失必须另开 `training/r7_spectral_geometry.py`，**不得**加进 runner。

---

## §2 交付物清单

### Stage P（0 GPU-h，纯 CPU 只读）

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| **D1** | 分解合法性表：`rfft2(norm='ortho')` 的 Parseval 精确性、**周期接缝能量占比**、DCT-II 交叉校验一致性 | `outputs/<stem>_attempt01/p0_decomposition.json` |
| **D2** | J2 反证（**必须能失败**）：粗 band 加能量/细 band 清零的假场必须使损失变差；`pred = target` 必须精确 0 | 同上 `p0_guards.json`，含正反两向 |
| **D3** | 振幅/相干性表：4 端点 × 8 train case × 17 变量 × 5 lead 的 `amp_ratio(k)`、`coh(k)`、逐带缺口 | `p1_amplitude_coherence.json` |
| **D4** | J1 方向门表：静力平衡 / 整层散度积分 / 湿物理，逐方向 `penalty(analysis)` vs `penalty(model)` + dose | `p2_j1_gates.json` |
| **D5** | **判定点**：逐方向写「过/不过 J1」、P1 缺口趋势、Q 开或跳过的结论与理由 | `p3_decision.md` |

### Stage Q（约 12 GPU-h，2 臂 × 3 seed × 4,000 updates）

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| **D6** | 谱几何损失 on/off 三 seed 配对读数（72 h t2m skill 与五 lead 全变量） | `readings.json`，含预声明 `exit_conditions` |
| **D6b** | 冻结协议 | `protocol.json`，含 dose 实测块与 preflight |
| **D6c** | 代码身份归档 | `outputs/r7_s3_spectral_geometry_loss_code.zip`（**带 commit 注释**） |

### Stage R（约 5–15 GPU-h，纯评估零训练）

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| **D7** | 后处理振幅校正：train 拟合 / 2022 val 报告，五 lead skill 与 RMSE 变化 | `posthoc_amplitude_correction.json` |
| **D8** | 结果页与登记 | `docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md` + 四件套 |

---

## §3 判据与证据来源（**全部预声明，不得事后调整**）

本节的每条判据都指向一个已归档的写定地，**不在本文件新写阈值**：
`docs/plans/0030-thesis-physics-informed-mainline.md` §3（三 Stage 的判据与出口）、
`docs/decisions/0110-limited-area-multiscale-decomposition.md`（分解方法与三项近似）、
`docs/plans/0029-physics-operator-candidates.md` §2（J1–J4 的原始定义）、
`docs/decisions/0094-c1-veto-ratio-is-cross-lead.md`、`0105-c2-objective-pairs-72h-forecast-with-6h-analysis.md`、
`0109-c2-derivative-matching-closed-on-three-seed-regression.md`（前置否决 / 守卫必须有判别力 / 关闭纪律）。
**剂量带 `[1e-3, 1e-1]` 来自 `docs/plans/0029-physics-operator-candidates.md` §2 的 J3**，本轮不重定义。

### §3.1 Stage P 的闸门

| 门 | 判据 | 不过的处置 |
| --- | --- | --- |
| **P0-a** | `rfft2(norm='ortho')` 满足 Parseval 到浮点容差 | **停**，不开 Q |
| **P0-b** | 强制**报出**接缝能量占比（**不设阈值**——设阈值等于事后调参） | 报出即可继续，数字进 limitations |
| **P0-c** | DCT-II 与 rfft2 的**结论必须一致**；相左则说明结论由边界假设驱动 | **结论作废** |
| **P0-d** | J2 反证**必须能失败**：假场使损失变差、`pred=target` 精确 0 | **停**（守卫无判别力） |
| **P1** | 缺口 `1 − amp_ratio(k)` 随训练量**缩小 ⇒ Q 跳过**；不变或扩大 ⇒ 开路 | 写进 D5 并如实说明 |
| **P2** | 逐方向 `penalty(analysis) ≤ penalty(model)` **且逐例一致**，dose ∈ `[1e-3, 1e-1]` | 逐方向记「过/不过」 |

**P0-d 为什么是硬门**：C2 attempt07 的守卫用 `terms(target, target)` 把 target 与自己比，恒零、与部署配对无关，于是把一个 bug 写成了物理结论（决策 0105）。只有构造一个**必须为正**的反例，守卫才有判别力。

**P1 为什么可以否掉 Q**：若缺口随 ×64 训练量缩小，说明它是训练不足而非损失函数的几何性质，AMSE 的动机消失。**这条是本轮唯一能阻止白烧 GPU 的机制，必须真的能触发。**

### §3.2 Stage Q 的出口

| 判据 | 要求 |
| --- | --- |
| **(a)** | 72 h t2m skill 三 seed **配对差全 ≥ 0**（同 seed、同父、只有损失不同） |
| **(b)** | on 臂逐步记录逐带 `PSD`/`coh` 与 dose，**严格为正** |
| **(c)** | off 路径与注册算术**逐位等价**（由 dispatcher 的 `None` 早退 + 反证测试支撑，**不由读数自证**——0109 的教训） |
| **(d)** | 无判据被改动（`protocol_sha256` 自校验、`test_read is False`） |

**(a) 为假 ⇒ 如实记 negative，不调权重、不换项、不加 seed、不换名重开**（照决策 0109 的关闭纪律）。
**(c)(d) 必须是真判据而不是字面量**：0109 记录了 C2 的 `c = True` 是常量、`d` 只读两个自写常量，伪造读数仍 all=True。本轮**不得重犯**。

### §3.3 Stage R 的出口

同一已注册 D2 分母（`sha256 096c384a…`、`n_selected_steps 32144`、train-only 2000–2021）下：
**五 lead 的 skill 与 RMSE 不劣化 24/48/72 h** 为通过条件。用 train 拟合、2022 val 报告，**不碰 2023 test**。

---

## §4 实施顺序（依赖，不跳步）

1. **P0** → 过全部四门才继续；任一不过 ⇒ 写 D5、登记、停。
2. **P1 与 P2 并行**（互不依赖，都是 CPU 只读）。
3. **D5 判定点** → 决定 Q 开或跳过。
4. Q 若要开：先写 ADR（族判别 + 冻结的 dose 带）→ 冻 `protocol.json` → archive → 三 seed 两臂 → 读数。
5. **R** 只在 P1 证实 `amp_ratio ≈ coh` 时开。
6. 四件套登记（`docs/R7_EVIDENCE_INDEX.jsonl`、账本行、master §8、`campaign-state`）。

---

## §5 预算与停止

| Stage | `planned_seconds`（软） | `hard_cap_seconds`（硬） | GPU-h |
| --- | ---: | ---: | ---: |
| **P** | 10,800（3 h 墙钟） | 21,600（6 h） | **0** |
| **Q** | 115,200（32 h/seed） | 230,400（64 h） | ≈12（按 20 估） |
| **R** | 28,800（8 h） | 57,600（16 h） | ≈5–15 |

软超预算**继续**并记 `soft_overrun_seconds`；**只有**硬上限因时长截断，记 `budget_limited`/`failed` 并全额记账。

**停止条件（只有三种）**：
1. 触保留授权边界（paid / rental / exclusive / 新数据下载 / main 合并 / 破坏性）；
2. 达已冻结的科学停止条件（P0 任一门不过、P1 缺口已缩、Q 的 (a) 为假）；
3. 确无可安全独立推进的工作。

停时必须在本文件 §8 写「**为何停 + 恢复第一条命令**」。

---

## §6 进度与交接

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```

**下一动作**（初始）：写两个探针脚本与 CI-safe 测试，跑 `round-guard`，然后执行 P0。

---

## §7 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1 的任何变体 | 方向反已实测 8/8（0094） |
| 重开 C2 的任何形式 | 三 seed 一致退化，永久关闭（0109） |
| 引入 `pywt` / `pyshtools` / `ducc0` / `torch_harmonics` | 改依赖即改 `training_code_sha256` 与执行闭包；区域网格上 SHT 装也没用 |
| 单独用 `rfft2` 不加尺度分解 | 正是论文 §4.2 点名的反模式 |
| 把 `rfft2` 近似称作 AMSE | 必须称 *AMSE-analogue on a limited-area grid* |
| 引入 legacy `physical_consistency.py` | 打包了已关闭/未授权项（0092 明文） |
| 在 P1 出结果前开 Q 轮 | 缺口若已随训练缩小，开了就是白烧 |
| 把谱损失加进 `training/r7_long_rollout_runner.py` | 已 597/600 行（R-051） |
| 读 2023 test | `r = 0` |
| 把「物理更正确」当结果 | 物理正确性是设计理由，不是结果证据 |
| 承诺训练臂一定涨 skill | P1 若显示缺口已缩，主线只剩证伪章，如实登记 |

---

## §8 进度块

**状态**：**Stage P 已测量并登记（2026-11-11）**；`scientific_claim: false`、`test_read: false`、`r = 0`、**0 GPU-h**

- **已完成**：本长文与 `round-spec`；plan 0030；ADR 0110；三处事实修正；`training/r7_spectral_geometry.py`
  （区域分带 AMSE-analogue，接入 `training/r7_objective_axes.py` 唯一分发点）；两个探针接到真实数据；
  **P0 四门全过**；**P1 判 `skip-q`**；**P2 三个方向全不过**；结果页 `docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`；
  证据索引 106 → **107** 条（`record:s3-spectral-geometry-stage-p`）
- **未做**：**Stage Q 未启动**（P1 判 `skip-q`）；**Stage R 未启动**；未冻结任何协议；未改任何训练/模型代码
  （新增的是损失模块，未被任何已注册轮使用）；未读 2023 test
- **入口**：`.venv/bin/python tools/r7_round.py state` 核对 `current_round_goal` 指向本文件
- **下一动作**：Stage P 已收口；下一轮的方向选择留给用户/复核者——候选是
  （a）把 `skip-q` 作为证伪章定稿、转向 Stage R 一类零训练后处理；
  （b）在**新的**预声明下扩大 P1 到三 seed 以量化端点间方差；
  （c）转向 plan 0030 之外的近邻方向（守恒/连续方程/CRPS）。
  **不得**因本轮不通过而换名重开 C1/C2（0094 / 0109）。

### 停在这里的理由（三选一中的第 2 条：达已冻结的科学停止条件）

按 §3.1 的**预声明**规则，P1 细带缺口随训练量单调缩小（0.32346 → 0.29482 → 0.26783 → 0.25637）
⇒ **Stage Q 必须跳过**。这是本目标长文写死的机制（"P1 是本轮唯一能阻止白烧 GPU 的机制，必须真的能触发"），
它触发了。同时 P2 三个方向全部不过 ⇒ 没有任何物理算子方向可开轮。

**恢复第一条命令**（若要继续）：

```bash
.venv/bin/python tools/r7_round.py state
```

**未做的事要写清**：P1 只覆盖单 seed 的 4 个端点，端点间差异含未量化运行间方差；
接缝占比 0.9921 使该分解的绝对数值带未量化偏差；band 边界未扫；
`skip-q` 只否掉这 4 个端点上的 Q 动机，**不得外推**为"AMSE-analogue 无效"。
