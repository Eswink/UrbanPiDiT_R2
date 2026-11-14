# S3：物理平衡诊断 —— 把「RMSE 好但物理不可信」从案例描述变成大样本统计量

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-balance-diagnostics", "driver": "scripts/study_r7_s3_balance_round.py", "output_stem": "r7_s3_balance_diagnostics_20261114", "control_record": "s3-urban-static-ablation-negative", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_balance_diagnostics.py", "tests/test_r7_balance_verdict.py", "tests/test_r7_process_tensor_diagnostics.py", "tests/test_r7_spectral_geometry.py"], "commit_paths": ["training/r7_balance_diagnostics.py", "scripts/r7_s3_balance_protocol.py", "scripts/r7_s3_balance_preflight.py", "scripts/r7_balance_verdict.py", "scripts/r7_balance_exits.py", "scripts/r7_balance_field_dump.py", "scripts/study_r7_s3_balance_round.py", "scripts/study_r7_s3_balance_diagnostics.py", "tests/test_r7_balance_diagnostics.py", "tests/test_r7_balance_verdict.py", "docs/goals/s3-balance-diagnostics.md", "docs/plans/0033-balance-diagnostics.md", "docs/decisions/0114-balance-diagnostics-as-evaluation.md"], "commit_message": "feat(r7): balance diagnostics -- four physical balances as an evaluation axis", "register_paths": ["docs/R7_S3_BALANCE_DIAGNOSTICS_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-balance-diagnostics.md"], "register_message": "docs(r7): register the balance diagnostics readings", "planned_seconds": 14400.0, "hard_cap_seconds": 28800.0, "code_archive": "outputs/r7_s3_balance_diagnostics_code.zip"} -->

本文件是**唯一执行源**；机器可读的 `round-spec` 在上方。
设计依据是不可变记录 `docs/plans/0033-balance-diagnostics.md`（文献核对、P0 实测读数与全部负面后果在那里）。
准入裁决是决策 **0114**。
`campaign-state.current_round_goal` 必须指向本文件，否则 `tools/r7_round.py` 会解析到别的轮次并冻结错协议。

**状态：prepared（2026-11-14 建立）。P0 预检与诊断模块已执行，P2–P5 未执行。**
`scientific_claim: false`、`test_read: false`、`r = 0`。
**GPU-h = 0**；无 `protocol.json`、无 `code.zip`、无 `diagnostics.json`；(a)–(e) 全部未判定。
**模型场一次都还没看过**——已有的全部读数都来自真值分析场。

---

## §0 Objective（单段，实测 3887 字符）

> 在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 执行目标长文 docs/goals/s3-balance-diagnostics.md（先完整读它，判据与相位以它为准；文献核对、P0 实测读数与全部负面后果见 docs/plans/0033-balance-diagnostics.md，准入裁决见决策 0114）。毕业论文主线：基于深度学习的物理信息混合天气预测研究与应用。起点：HEAD 78cd65a、used_gpu_h 613.790428、model_code_sha256 59340710623c（不得改动）。前七轮物理信息全负且全部在损失侧或输入侧：C1 8/8 方向反、C2 15/15 全负 66.99 GPU-h、Stage P 判 skip-q、Stage R 85/85 全劣化、Stage C 减半需 775102 倍训练量、Stage A 振幅三 seed 升 +0.114 到 +0.125 但 skill 9/9 全降 21.83 GPU-h、城市静态三臂消融 (a) 假按预声明关闭 20.46 GPU-h。七轮共同结构是只改「按什么算分」或「模型能看到什么」，没有一轮问过「模型算出来的场在物理上是否可信」。本轮改问这个问题：不训练任何模型，只做评估侧诊断。依据是 2026-11-14 一手读正文的 arXiv:2604.01454（Pasquini 等，目前唯一系统评估区域 DL 模型物理平衡的工作，评估静力/地转/梯度风/连续方程四个平衡加谱分析，结论是 despite its relatively good performance in terms of RMSE, Bris significantly disrupts several atmospheric balances，但它的样本是一个风暴一个 lead 一个 443x443 子域即 n=1），以及 arXiv:2309.08473（Bonavita，逐字：Pangu-Weather forecasts do not have the fidelity and physical consistency of physics-based models and their advantage in accuracy on traditional deterministic metrics of forecast skill can be at least partly attributed to these peculiarities）。我们的增量：全年 2022 val 的 1447 个初始时刻乘 5 个 lead 乘 3 seed 乘 2 组受控臂，把物理不平衡从案例描述变成可统计的分布。四个诊断量全部照文献不发明：D1 静力偏差 delta_HB 等于 (dPhi 减 dPhi_HB) 除 dPhi，dPhi_HB 等于 负 R 乘 T_v 除 p 乘 delta_p（式 1 到 3，T_v 等于 T 乘 1 加 0.61q，层对 850-500 用虚拟温度、500-250 只能用干温度因为 store 没有 t250 和 q250 且必须分开报告）；D2 地转偏差 ws_ag 等于 v 减 v_g 的范数，v_g 等于 f 分之一 k 叉乘 grad Phi，只在 Rossby 数小于 0.1 的格点上（式 4 到 6，f 等于 2 Omega sin phi 在活跃代码中此前零出现必须新写，赤道处 |f| 小于 1e-5 必须 raise 而不是除零）；D3 柱积分散度 D850 加 D500 加 D250；D4 谱，std(Phi_f1 减 Phi_f13) 作细尺度噪声代理加径向振幅谱。不做梯度风平衡（式 7 到 8 需要气旋中心识别与 N M 不超过 5 的径向拟合，区域 65x65 上气旋未必在域内，列为未做的限制）。硬单位纪律：store 的 z 通道是位势 m**2 s**-2 不是米，delta_HB 无量纲且计算中绝不允许除以 g（scripts/r7_s3_operator_j1_probe.py 记录过一次 224 倍事故：符号写反让 36.6 m 变成 8193 m）。交付物：D1 field-dump 脚本把 15 个 checkpoint 的预报场落盘（不改 training/r7_evaluate.py 也不改 r7_long_rollout_runner.py，后者已 597/600 行）· D2 预检报告（单位检查、f 在三个纬度的量级、Ro 小于 0.1 的逐层覆盖率、65x65 全有限、delta_HB 的 +2.6% 系统偏置作为差值基线记录）· D3 冻结协议含 static 段与 family_argument · D4 code.zip 且必须带 commit 注释 · D5 逐 seed 逐臂 receipt · D6 diagnostics.json（四诊断量乘 5 lead 乘 3 臂组乘 3 seed）· D7 证据页与四件套登记 · D8 判定点逐条写 (a) 到 (e)。预声明出口：(a) 三臂 delta_HB 中位数在三 seed 上一致地互不相同；(b) 背离，至少一个 lead 变量上 RMSE 更好的臂平衡偏差更差，这是主判据且是 Bonavita 论点的可证伪表述；(c) 三 seed 平衡偏差极差小于臂间差；(d) 真值分析场的偏差显著小于模型场作为阳性对照，防止诊断量恒真；(e) 预检全真。(a) 与 (b) 失败形态不同必须分开报：(a) 假说明诊断量是死的对任何扰动无反应，(b) 假说明它活了但和 RMSE 说同一件事即冗余；只有 (a)(b) 都真才可称物理平衡是独立的评价轴。臂间比较只在同组内成立（Stage A 的 on/off 与城市静态的 control/treatment/placebo 不同父配方，跨组只比量级不比差值）。预算：重推理 15 checkpoint 约 6.5 GPU-h，软预算 14400 秒硬上限 28800 秒；场必须落盘后诊断可零成本重跑。执行方式：用 workflow 编排，先加载 dynamic-workflows 技能，从 .zcode/workflow-drafts/r7-urban-static-ablation.dwf.ts 派生 r7-balance-diagnostics.dwf.ts，必须用 CreateWorkflow 传 path 加 args 启动而不要用 AmendWorkflow（实测丢 args 让 dryRun 静默变 true），先传 maxRounds 0 dryRun true 做零成本编译校验；判定读数必须用 json.load 结构化读取而绝不用正则匹配自报文本（上一轮实测一条 judge 命令返回空 body 让正则把全部子句默认成 false，卡片与读数相反，见提交 da1ed00）；自跑 round-guard 与 commit-guard 与 verify 不依赖 hook 兜底；只推 r7/weather-reasoning；跑任何 stage 前先跑 tools/r7_round.py state 核对 current_round_goal。禁止：重开 C1 C2 Stage P R C A、重开 tendency 输入、重开物理诊断量输入、重开城市静态协变量、训练任何模型、改 model 目录任何文件（会改 model_code_sha256 让 15 个 checkpoint 全部不可加载）、改 training/r7_evaluate.py 或 training/r7_long_rollout_runner.py、做梯度风平衡、引入 pywt 或 pyshtools 或 ducc0（区域 patch 上 SHT 不良定义）、重建 store、新增或重下数据、改 data_identity、扫参数、加 seed、与其它已关轴组合、读 2023 test、动 data/raw|interim|processed 与归档快照、合成数据、事后放宽出口、删弱测试、绕 checkpoint 与数据身份校验、paid 与 rental 与 exclusive 与 cron 与守护与会话外续跑、对邻居进程发任何信号、main 合并与 force 与关 issue、自宣科学通过或 SOTA 或最终 goal 完成。只在三种情况停：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在 8 节写为何停加恢复第一条命令。每步给可核查证据（命令加输出加 digest），估计值不得写成实测值。不承诺正面结果：若 (a)(b)(c)(d) 任一为假，如实记负并停；特别地若 (b) 为假（平衡更好的臂 RMSE 也更好）那会否证 Bonavita 论点在区域小模型上的适用性，这仍是有价值的结论但论文叙事必须改，且不得事后调整判据来避免它。本轮是评估不是改进，即使全部出口为真也不产生任何 skill 提升。结束按 AGENTS.md 汇报改动、实验与失败、技能与代理使用、验证与精确 CI、成本与 issue 状态、未做项与下一项。

---

## §1 现状与起点

| 项 | 值 | 来源 |
| --- | --- | --- |
| 工作分支 | `r7/weather-reasoning` | `git status` |
| 起点 HEAD | `78cd65a`（工作树干净） | `git log -1` |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324` | 实测 |
| 数据实例 | `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr` | `[34084,17,65,65]` |
| val manifest | `outputs/.../store/manifests_treatment/val.jsonl`（451,980 B，2022 全年） | —— |
| 可用 checkpoint | static ablation 9 + Stage A 6 = **15**，3 seed | —— |
| `used_gpu_h` | `613.790428`（**仅会计**） | campaign-state |
| 2023 test | **未评分**，`r = 0` | —— |

### §1.1 七轮负结果的统一诊断

| 轮次 | 改了什么 | 结果 | 成本 |
| --- | --- | --- | --- |
| C1 | 无目标约束束 | 真值被罚 10.2×，8/8 方向反 | 0 GPU-h |
| C2 | 导数域目标匹配 | 15/15 (lead, seed) 全负 | 66.99 GPU-h |
| Stage P | 谱几何前置 | P1 判 `skip-q`、P2 三方向全不过 | 0 GPU-h |
| Stage R | 后处理振幅校正 | 85/85 (变量, lead) 全劣化 | 0 GPU-h |
| Stage C | 缺口衰减律 + β_j | α=0.05111，减半需 775,102× 训练量 | 0 GPU-h |
| Stage A | β_j AMSE-analogue 归一化 | 振幅升 +0.114…+0.125，skill 9/9 全降 | 21.83 GPU-h |
| 城市静态 | 三臂输入消融 | (a) 假 ⇒ 按预声明关闭 | 20.46 GPU-h |

**共同结构：七轮全都在改「按什么算分」或「模型能看到什么」，没有一轮问过「模型算出来的场在物理上是否可信」。**

Stage A 是最尖锐的证据：它把细带振幅精确抬高了 12%，而 skill 因此全面下降。
**一个能精确执行指令却让结果变差的方法，说明「振幅」不是瓶颈**——但七轮里没有一次测量去问
「被抬高的那部分振幅，是不是物理上本来就不该有」。

### §1.2 文献空白（2026-11-14 一手核实）

`arXiv:2309.08473`（Bonavita）摘要逐字：

> "Pangu-Weather forecasts, and possibly those of similar ML models, **do not have the fidelity and
> physical consistency of physics-based models** and their advantage in accuracy on traditional
> deterministic metrics of forecast skill can be at least partly attributed to these peculiarities."

`arXiv:2604.01454`（Pasquini et al. 2026）是目前唯一系统评估**区域** DL 模型物理平衡的工作。
它评估四个平衡（§3.1 静力、§3.2 地转、§3.3 梯度风、§3.4 连续方程）+ §4.6 谱分析，结论逐字：

> "despite its relatively good performance in terms of RMSE, Bris struggles to capture important
> mesoscale features of the event and that **it significantly disrupts several atmospheric balances**.
> This unrealistic disruption is mainly linked to the fine-scale noise evidenced in its output fields"

**它的 n = 1**：一个风暴、一个 lead、一个 443×443 子域。

### §1.3 我们的增量

- 文献：1 个风暴 × 1 个 lead × 1 个子域
- 我们：**1447 个初始时刻 × 5 个 lead × 3 seed × 2 组受控臂**
- 并且有一组天然受控扰动：Stage A 的 on 臂振幅被精确抬升 12% 而 skill 全降

⇒ 第一次能问：**被抬高的那 12% 振幅，是否正是物理上不该有的部分？**

### §1.4 一个必须承认的工程事实

`find outputs -path "*/evaluation/*" -type f ! -name "*.csv" ! -name "*.json"` → **0**。
`training/r7_boundary_metrics.py:99` 的 `sum((-1,-2))` 把空间轴坍缩；`training/r7_rollout_metrics.py:75`
逐字写着 `# Only compact [L,C] sufficient statistics survive update()`。
⇒ 任何空间型指标都必须重推理。**这就是本轮 ≈6.5 GPU-h 的全部成本来源。**

---

## §2 四个诊断量（照文献，不发明）

| 量 | 公式 | 出处 |
| --- | --- | --- |
| `δ_HB` | `(dΦ − dΦ_HB)/dΦ`，`dΦ_HB = −R·T̄_v/p̄·Δp`，`T_v = T(1+0.61q)` | `2604.01454` 式 (1)–(3) |
| `ws_ag` | `‖v − v_g‖`，`v_g = (1/f)k̂×∇Φ`，限 `Ro<0.1` | 式 (4)–(6) |
| 柱积分散度 | `D850 + D500 + D250` | §3.4 |
| 谱 | `std(Φ_f1 − Φ_f13)` + 径向振幅谱 | §4.6 |

**单位纪律（硬）**：`z*` 是**位势 `m**2 s**-2`**，不是米。`δ_HB` 无量纲且**计算中绝不允许除以 `g`**。
`scripts/r7_s3_operator_j1_probe.py:70-90` 记录过一次真实的 **224×** 事故。

**层对限制**：`850–500` 有 q，用虚拟温度；`500–250` **只能用干温度**（store 没有 `t250`/`q250`）。
两者必须分开报告，不得合并成「平均静力偏差」。

**不做梯度风平衡**：需要气旋中心识别 + `N,M≤5` 径向拟合；区域 65×65 上气旋未必在域内。
这是未做的限制，不是不需要。

---

## §3 预声明出口（任一为假即如实记负并停）

| 判据 | 要求 | 失败意味着 |
| --- | --- | --- |
| **(a)** | 三臂 `δ_HB` 中位数在三 seed 上一致地互不相同 | 诊断量是**死的** |
| **(b)** | **背离**：至少一个 (lead, 变量) 上，RMSE 更好的臂平衡偏差更差 | 诊断量活了但与 RMSE **冗余** |
| **(c)** | 三 seed 平衡偏差极差 < 臂间差 | 信号被 seed 离散淹没 |
| **(d)** | 真值/分析场偏差显著小于模型场（阳性对照） | 诊断量恒真（0109 教训） |
| **(e)** | 预检：单位、`f` 量级、`Ro<0.1` 逐层覆盖率、65×65 全有限 | 单位事故 |

### §3.1 (b) 的判定语义

(b) 不是「诊断量有效」，而是「诊断量提供了 RMSE 没有的信息」。
**只有 (a)(b) 都真，才可以说物理平衡是一个独立的、有信息的评价轴。**

若 (b) 为假（平衡更好的臂 RMSE 也更好），那会**否证** Bonavita 论点在区域小模型上的适用性。
这仍是有价值的结论，但论文叙事必须改，且**不得事后调整判据来避免它**。
出口 (a)–(e) 的完整定义在 `docs/plans/0033-balance-diagnostics.md` §4。

### §3.2 臂间比较只在同组内成立

Stage A 的 on/off 与城市静态的 control/treatment/placebo **来自不同父配方**。
跨组只能比较「诊断量的量级」，**不能比较 delta**。

### §3.3 判据的证据来源

判定所用的每一个数字都必须能在下列不可变记录里找到出处，不得由执行者自报：

- 出口的定义与失败形态：`docs/plans/0033-balance-diagnostics.md` §4，以及
  `docs/decisions/0114-balance-diagnostics-as-evaluation.md` §D4
- 诊断量的公式与逐字文献引文：`docs/plans/0033-balance-diagnostics.md` §0.2、§3
- 预检的实测读数与偏置来源：`docs/plans/0033-balance-diagnostics.md` §2
- 224× 单位事故的原始记录：`docs/decisions/0114-balance-diagnostics-as-evaluation.md` §D2
- 已关闭七轮的判定与封锁条款：`docs/decisions/0084-direction-expansion-throughput-curve-physics.md`、
  `docs/decisions/0109-c2-derivative-matching-closed-on-three-seed-regression.md`、
  `docs/decisions/0112-sequence-admission-beta-normalisation.md`、
  `docs/decisions/0113-urban-static-ablation-and-placebo.md`

---

## §4 预算与停止

| 项 | 值 |
| --- | --- |
| 重推理 | 15 checkpoint × 5 lead × 1447 case ≈ **6.5 GPU-h** |
| 软预算 / 硬上限 | **14,400 s / 28,800 s** |
| 场落盘 | **必须**；落盘后诊断可零成本重跑 |

**这一条是对上一轮「场从未落盘」缺陷的直接修正。**

---

## §5 交付物清单

| # | 交付物 | 验收标准 | 落盘位置 |
| --- | --- | --- | --- |
| **D1** | field-dump 脚本 | 重推理并落盘场；不改 `r7_evaluate.py` 与 `r7_long_rollout_runner.py` | `scripts/r7_balance_field_dump.py` |
| **D2** | 预检报告 | 出口 (e) 全绿；`δ_HB` 的 +2.6% 系统偏置作为差值基线记录 | `outputs/<stem>/preflight.json` |
| **D3** | 冻结协议 | 含 `family_argument`，`protocol_sha256` 落盘，训练前写定 | `outputs/<stem>/protocol.json` |
| **D4** | 代码身份归档 | `code.zip` **必须带 commit 注释** | `outputs/r7_s3_balance_diagnostics_code.zip` |
| **D5** | 逐 seed 逐臂 receipt | 15 份，含 checkpoint sha 与 s/update | `outputs/<stem>/seed{N}/{arm}/receipt.json` |
| **D6** | `diagnostics.json` | 4 诊断量 × 5 lead × 5 臂 × 3 seed | `outputs/<stem>/diagnostics.json` |
| **D7** | 证据页 + 四件套登记 | index record、账本行、campaign §8、`campaign-state` | `docs/` |
| **D8** | 判定点 | 逐条写 (a)–(e) 真或假，附实测数字 | 证据页 §判定点 |

## §6 实施顺序

1. **P0** 预检（**已执行**，0 GPU-h）—— 出口 (e)
2. **P1** 诊断模块（**已写**，25 条测试全过）
3. **P2** field-dump 脚本
4. **P3** 15 checkpoint 重推理 → 场落盘
5. **P4** 诊断计算 → `diagnostics.json`
6. **P5** 判定 (a)–(e)，**用 `json.load` 而非正则**
7. **P6** 证据页 + 四件套登记

---

## §7 判据与证据来源

- 出口定义与失败形态：本文件 §3；设计与全部负面后果：`docs/plans/0033-balance-diagnostics.md` §4、§7
- 准入裁决与族判别：`docs/decisions/0114-balance-diagnostics-as-evaluation.md` §D1–D6
- 文献依据（公式与逐字引文）：`docs/plans/0033-balance-diagnostics.md` §0.2–§0.3
- 224× 单位事故的原始记录：`scripts/r7_s3_operator_j1_probe.py:70-90`
- 空间轴坍缩的位置：`training/r7_boundary_metrics.py:99`、`training/r7_rollout_metrics.py:75`
- 已关闭七轮的判定：`docs/decisions/0084-*.md`、`0094-*.md`、`0109-*.md`、`0111-*.md`、`0112-*.md`、`0113-*.md`

---

## §8 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1/C2/Stage P/R/C/A | 全部按预声明关闭 |
| 重开 tendency 输入 / 物理诊断量输入 / 城市静态 | 0049、`record:s3-typed-spatial-result`、0113 |
| **训练任何模型** | 本轮纯评估；15 个 checkpoint 已存在 |
| 改 `model/**` | 会改 `model_code_sha256`，让 15 个 checkpoint 全部不可加载 |
| 改 `training/r7_evaluate.py` / `r7_long_rollout_runner.py` | 316 行 / **597/600 行（R-051）** |
| 做梯度风平衡 | 需气旋中心识别；区域域内未必有气旋。列为未做的限制 |
| 引入 `pywt`/`pyshtools`/`ducc0` | 区域 patch 上 SHT 不良定义（0110） |
| 重建 store / 新增或重下数据 / 改 `data_identity` | 只读 |
| 扫参数 / 加 seed / 与其它已关轴组合 | 冻结前定死 |
| 读 2023 test | `r = 0` |
| `cron` / 守护 / 会话外续跑 / 对邻居进程发信号 | 硬约束 |
| 承诺正面结果 | 见 §3；本轮是评估不是改进 |

---

## §9 进度块

| 时间（UTC） | 阶段 | 状态 | 证据 |
| --- | --- | --- | --- |
| 2026-11-14 | 规划 + P0 预检 + P1 模块 | **完成** | `docs/plans/0033-…`、`docs/decisions/0114-…`、`training/r7_balance_diagnostics.py`、25 条测试 |
| —— | P2 field-dump | **未开始** | —— |
| —— | P3 重推理 | **未开始** | —— |
| —— | P4 诊断计算 | **未开始** | —— |
| —— | P5 判定 (a)–(e) | **未开始** | —— |
| —— | P6 登记 | **未开始** | —— |

**P0 已实测的三个读数**（0 GPU-h，真值分析场）：`δ_HB` 中位数 **+0.021…+0.029**（系统偏置，
来源已查清：反推隐含上层气压 498.90 hPa vs 真值 500）；地转 `Ro<0.1` 覆盖率 **6.8%–15.8%**；
`z500` f13 细尺度噪声 **39.97**。

**GPU-h = 0。2023 test 未读，`r = 0`。模型场一次都还没看过。**

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```

---

## §9 进度块（2026-11-14 会话）

| 时间（UTC） | 阶段 | 状态 | 证据 |
| --- | --- | --- | --- |
| 2026-11-14 | 规划 + 文献 + P0 + P1 + P2 冒烟 | **完成** | 见下 |
| —— | P3 正式重推理（1447 case × 15 checkpoint） | **未开始** | —— |
| —— | P4–P6 | **未开始** | —— |

**本轮会话实际完成的**：

- `training/r7_balance_diagnostics.py`（275 行）：12 个函数，纯 torch，AST 扫描证明不读文件
- `tests/test_r7_balance_diagnostics.py`：**25 条全过**（六条反证 + 三条真实数据钉子 + 物理符号验证）
- `scripts/r7_balance_field_dump.py`（约 190 行）：重推理并落盘物理单位场，端到端跑通 6 case
- `scripts/study_r7_s3_balance_diagnostics.py`（269 行）：四诊断量 + (a)–(e) 判定，结构化读而非正则
- `scripts/r7_s3_balance_protocol.py`（242 行）：**`--preflight-only` 五条子句全绿**
- `docs/plans/0033-…`、`docs/decisions/0114-…`、本文件
- workflow `.zcode/workflow-drafts/r7-balance-diagnostics.dwf.ts`，**零成本编译校验通过**

**P2 冒烟的决定性读数**（6 case × 5 lead，CPU，非正式）：

模型 `ws_ag` 随 lead **4.58 → 9.75 m/s（2.1×，单调）**，真值场 **平**（2.34–2.77，无趋势）；
模型 `z500` f13 噪声 **34.4 → 57.3（1.7×）**，真值**下降**（29.5 → 21.4）；比值 1.86× → **4.17×**。
⇒ **(d) 阳性对照通过**：失衡是模型产生的，不是数据的性质。
静力 `δ_HB` **无分辨力**（模型与真值都在 2.7–2.9%，差 ≤0.0022）。

**修掉一个真实缺陷**：首次落盘把归一化空间当物理单位，`δ_HB` 报 −51.12、真值 std 7088；
反归一化后 `z500` 均值 54757/54826、`t2m` 276.2/275.8 K。

**GPU-h = 0。2023 test 未读，`r = 0`。(a)(b)(c) 仍是未测——只有 control 单臂。**

**为何停**：P3 需要 15 个 checkpoint × 1447 个 case 的正式重推理（≈6.5 GPU-h），
本轮先证明整条链在 6 个 case 上跑通并给出可解释的读数，再交给正式轮。
**恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`

---

## §9 进度块（2026-11-14 第二轮会话：把轮次做成可执行的）

**状态：implementation committed, round not yet frozen; GPU-h = 0。**

### 修掉的阻塞缺陷（prepared 状态的 round 跑不起来）

`round-spec` 的 `driver` 原本指向 `scripts/study_r7_s3_balance_diagnostics.py`，而该文件
只是一个 `--fields/--out` 诊断计算器。`tools/r7_round.py freeze` 会 `load_driver()` 并要求
`freeze`、`SEEDS`、`PER_SEED_SECONDS`、`PLANNED_SECONDS`、`HARD_CAP_SECONDS` 与
`--worker <prepare|archive|seed|reading>` 入口（`tools/r7_round_stages.py:158-161`），
这些一个都不存在 ⇒ **`freeze` 必然失败，P3 一步都跑不了**。
另外 `evaluate_exits` 把 `placebo_available=False` 写死，即使 15 个臂全部落盘，(a)(b)(c)
也会返回 `null`。两处都已修。

**修正内容**（`round-spec` 的 `driver` 改为 `scripts/study_r7_s3_balance_round.py`）：

| 文件 | 行数 | 作用 |
| --- | ---: | --- |
| `scripts/r7_s3_balance_preflight.py` | 219 | 出口 (e) 的五条单位/覆盖率子句 |
| `scripts/r7_s3_balance_protocol.py` | 422 | 15 臂注册表 + `freeze`/`prepared_protocol`/`validate_protocol` |
| `scripts/r7_balance_verdict.py` | 252 | 逐臂表（四诊断量 + RMSE），(a)–(e) 的输入 |
| `scripts/r7_balance_exits.py` | 356 | (a)–(e) 判定，全部是显式比较 |
| `scripts/study_r7_s3_balance_round.py` | 225 | `--worker` 驱动（每 seed 一 worker，5 臂） |
| `tests/test_r7_balance_verdict.py` | 370 | 反证测试 |

`scripts/study_r7_s3_balance_diagnostics.py` 改为薄预览层，复用同一套 verdict 模块：
它原先自带一份 `placebo_available=False` 的判定副本，两条判定路径正是「卡片与读数相反」
那类缺陷的温床。

### 实测证据（每条都可复核）

| 项 | 读数 | 方法 |
| --- | --- | --- |
| 15 个 checkpoint 严格加载 | `model_code_sha256 = 59340710623c…` 不变，15/15 ok | `clause_arm_registry` |
| 出口 (e) | **6/6 子句全绿** | `scripts/r7_s3_balance_protocol.py --preflight-only` |
| 真实落盘速率 | **1.09–1.18 s/case**（12 case 两次计时） | 实测 |
| 诊断 CPU 速率 | **0.263 s/case** | 实测 |
| **RMSE 交叉核对** | 1447 case × 7 变量 vs 已登记 `boundary_rmse.csv`，**最差相对差 8.5e-9** | 见下 |
| 并行 reading | **16.5×**（163.2 s → 9.9 s，15 臂 × 40 case），结果逐位相同 | `table_of_arms(workers=15)` |
| 端到端冒烟 | 5 臂 × 6 case，落盘 + 判定全通 | —— |

**RMSE 交叉核对是 (b) 的地基。** 重新推导的 `case_rmse` 与那一臂自己登记的
`boundary_rmse.csv`（region=full, lead=72h, `n_initializations = 1447`）逐变量对比：

| 变量 | 本轮推导 | 已登记 | 相对差 |
| --- | ---: | ---: | ---: |
| `t2m` | 3.662812 | 3.662812 | 8.3e-10 |
| `z500` | 450.197087 | 450.197087 | 1.4e-10 |
| `u500` | 7.275873 | 7.275873 | 8.5e-09 |
| `mslp` | 499.444134 | 499.444133 | 2.2e-09 |

最差 8.5e-9，是 float32 舍入量级。这一条同时证明三件事：反归一化正确、落盘覆盖的正是
已登记的 1447 个 case、以及 (b) 用的 RMSE 与登记评估报的是同一个数。

### 测试抓到的五个真实缺陷（全部在实现里，已修）

1. `case_rmse` **返回 MSE 而名字说 RMSE**（测试 `test_rmse_reproduces_…` 抓到）。
2. 汇总时对已被开方的值再开一次根 ⇒ 低估总误差；改为 `sqrt(mean(rmse²))`。
3. **键名不一致**（端到端冒烟抓到，单元测试全部漏掉）：`case_metrics` 产出
   `forecast:hydrostatic:850-500`，判定侧查 `hydrostatic:850-500` ⇒ 正式轮会**全部子句 null**。
   修法是让生产者与消费者共用 `model_key()`/`truth_key()`，并补一条走生产者建表的端到端测试。
4. **(b) 能靠噪声通过**：数百个比较里「至少一个反转」几乎必然成立；改为**三 seed 一致**才计，
   单 seed 反转只报不判。
5. **部分落盘会出结论**（冒烟抓到）：只有 seed41 的 5 臂时 (a) 报 `false`（"诊断量是死的"）、
   (b) 报 `true`，而冻结判据写的是三 seed。加了完整性闸门后，部分落盘一律 `null`（未测）。
   这正是「未测 ≠ 假」的技术实现。

### 如实记账

- **GPU-h = 0**；未冻结 `protocol.json`、未建 `code.zip`、无 `diagnostics.json`。
- **(a)(b)(c) 仍是未测**。冒烟只跑了一个 seed 的 5 臂，按上面第 5 条现在正确地返回 `null`。
- 冒烟里 `(d) = true`（25/25 行模型比真值更不平衡且更噪），`(e) = true`；这两个不依赖三 seed 对照。
- **一条必须记下的观察**：`stage_a/off/seed41` 与 `urban_static/control/seed41` 的权重
  最大差 **3.76e-6**（五个臂的权重 digest 两两不同，但这两臂数值上近乎同一个模型）。
  两臂来自不同配方却如此接近，说明当轮实际训练几乎没把这两条配方分开；本轮**如实记录**，
  不据此推断任何结论。

**为何停**：实现已落盘并通过 45 条测试与 37 条阻断规则，但正式 P3 需要先 commit 实现、
按当前 HEAD 构建 `code.zip`，再由 workflow 冻结并开 GPU。这三步是同一个动作序列，
下一步就是执行它。
**恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`
