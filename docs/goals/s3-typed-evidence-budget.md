# S3：物理证据进前向 × 训练预算 —— 在足够训练下检验 typed evidence 的可归因贡献

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-typed-evidence-budget", "driver": "scripts/study_r7_s3_typed_budget.py", "output_stem": "r7_s3_typed_evidence_budget_20261116", "control_record": "s3-typed-spatial-result", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_typed_budget.py", "tests/test_r7_79_typed_evidence.py", "tests/test_r7_process_tensor_diagnostics.py"], "commit_paths": ["scripts/r7_s3_typed_budget_protocol.py", "scripts/r7_typed_budget_verdict.py", "scripts/study_r7_s3_typed_budget.py", "scripts/r7_s3_dose_probe.py", "tests/test_r7_s3_typed_budget.py", "docs/goals/s3-typed-evidence-budget.md", "docs/plans/0034-typed-evidence-under-budget.md", "docs/decisions/0115-typed-evidence-under-adequate-budget.md"], "commit_message": "feat(r7): typed physical evidence under an adequate budget -- capacity-matched three arms", "register_paths": ["docs/R7_S3_TYPED_EVIDENCE_BUDGET_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-typed-evidence-budget.md"], "register_message": "docs(r7): register the typed-evidence budget readings", "planned_seconds": 172800.0, "hard_cap_seconds": 259200.0, "code_archive": "outputs/r7_s3_typed_evidence_budget_code.zip"} -->

本文件是**唯一执行源**；机器可读的 `round-spec` 在上方。
设计依据是不可变记录 `docs/plans/0034-typed-evidence-under-budget.md`（全部实测读数与负面后果在那里）。
准入裁决是决策 **0115**。
`campaign-state.current_round_goal` 必须指向本文件，否则 `tools/r7_round.py` 会解析到别的轮次并冻结错协议。

**状态：planned（2026-11-16 建立）。段 A 与段 B 均未执行。**
`scientific_claim: false`、`test_read: false`、`r = 0`。**GPU-h = 0**。

---

## §0 Objective（单段，实测字符数见文末标记）

> 在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 执行目标长文 docs/goals/s3-typed-evidence-budget.md（先完整读它，判据与相位以它为准；全部实测读数与负面后果见 docs/plans/0034-typed-evidence-under-budget.md，准入裁决见决策 0115）。毕业论文硬约束：物理信息必须在模型前向或后处理里有真实贡献，且不发明新机制。起点：HEAD 4476f63、used_gpu_h 619.436628、model_code_sha256 59340710623c（不得改动）。九轮物理全负且位置共同点是事后或损失侧：C1 约束束对分析场惩罚是模型 10.2 倍 8/8 方向反、C2 导数匹配 15/15 全负 66.99 GPU-h、Stage P skip-q、Stage R 振幅校正 85/85 全劣化、Stage C 缺口半衰需 775102 倍训练量、Stage A 振幅升 12% 但 skill 9/9 全降 21.83 GPU-h、城市静态三臂 (a) 假 20.46 GPU-h、平衡诊断 (a) 假且 (b)(d) 真 5.65 GPU-h。本轮又做七项只读判决测试全部为负：物理失衡与误差大小的逐 lead 相关最大 0.2676 lead12 为 -0.0164；与误差空间图样中位 Spearman 仅 0.048 到 0.074 且打不过梯度幅值 0.067 对 0.131 剔除后残差仅 0.010；谱增益算子在 g 小于等于负一时恒等于 box5 平滑残差 0.000e+00 且相干上限 rho 平方 0.033 到 0.0087 已达 87 到 98 百分比；低秩平滑带随 K 增大退化到平坦常宽带且被块 b 等于 4 支配；平衡子空间带 R 平方只有 0.110 且 z 误差只占归一化误差方差 6.0 百分比；case 自适应缩放 oracle 的 P(all) 0.005 低于基线 0.058；地转投影 20/20 组合 RMSE 全升 4.4 到 183.0 百分比且 ws_ag 投影后等于 lambda 乘原值是构造恒等式。第七项的失败给出关键机制证据：corr(v_a, v_g 误差) 等于 -0.79 到 -0.96 说明非平衡残差在补偿一个错误的 v_g，而 v_g 偏强 1.12 到 2.04 倍的根源是 z 的 f13 细尺度噪声是真值 2.56 倍而风场 f13 反而低于真值 0.44 到 0.89 倍，所以失衡的可修部分是 z 的高频误差且必须在前向修，事后投影会拆掉模型自己建立的补偿。唯一有正面先例的前向物理装置是 #79 typed evidence：四个局地物理场 850 hPa 散度 850 hPa 涡度 850 hPa 温度平流 850 到 500 hPa 静力稳定度由纯 torch 球面算子从 X_t 与模型自己的初稿 Y_0 算出经 patch 编码加到前四个 anchored 证据槽在第一步递归之前，已在参数量 3097571 与 FLOPs 16776992640 逐位相等的三臂对照下拿到 6h 与 12h 双 lead supported 三 seed 同号 t2m 减 0.0533 K，机制核验逐条留痕 off 臂首步状态与旧实现位级相同差 0.0 扰动任一类型只移动对应槽位 4x4 到达矩阵严格对角毒化未来标签对 forecast 是 no-op。但它只跑过 400 updates 且生产配置里该通路关闭。本轮的主判据就是把它放到 4 epochs。段 A 三臂：A incumbent 为生产配置 typed_evidence_mode 未设；B generic_fusion 四物理场经一个融合投影；C typed_routing 每场独立投影。B 与 C 参数量与 FLOPs 逐位相等，所以只有 C 减 B 能把增量归因给物理类型结构；B 减 A 回答物理证据本身是否有增量。预算 32132 updates 等于 4 epochs 乘 batch 4，三 seed 41 42 43，父为 r7_s3_position_retention_parent_migrated_20261103_attempt01 的 1600 步端点 state_digest 7c416ccb 当前 digest 下的模型迁移，同数据同 loss 同物理权重向量 (1,.5,0,.5,0,0,0,.5,0,0,0,.5)。段 B 先跑且 0 训练：p1600 的 13 个中间 checkpoint 每 10000 步加端点 128528 在 lead 6 24 72 上的组内剂量曲线，只有端点曾评分；以及 p1600 端点 reasoning_steps 取 2 4 6 8 在全部 5 lead 上的推理深度扫描，evaluate_local 的裸 reasoning_steps 不需要 controller。段 B 决定段 A 是否需要 16 epoch 档位而 16 epoch 本轮不授权。判据冻结：(a) C 减 B 在至少 2 个 lead 上同号为负且三 seed 一致且 block bootstrap 95 百分比 CI 排除零；(b) B 减 A 在至少 2 个 lead 上同号为负；(c) 若 (a) 真而 (b) 假只能声称类型结构不得声称物理增量；(d) 同臂在 4 与 16 epoch 上增量方向一致；(e) z500 在 72h 相对 A 臂恶化不超过 5 百分比因为 p4000 到 p1600 的 z500 72h 已从 427.35 升到 445.78；(f) 预检全真包括三臂 trunk 位级相同、off 臂首步差 0.0、4x4 到达矩阵严格对角、test_read 为 false。任一臂缺失时该子句必须是 null 而不是 false 也不是 true，这是 da1ed00 那次正则把空 body 默认成 false 导致卡片与读数相反的直接修正。交付物：D1 协议构建器含 (a) 到 (f) 预检 · D2 三臂契约与容量匹配断言 · D3 段 B 零训练探针 · D4 冻结协议含 protocol_sha256 · D5 code.zip 带 commit 注释 · D6 逐 seed 逐臂 receipt · D7 readings.json 含 scientific_claim test_read protocol_sha256 exit_conditions seeds 五键 · D8 证据页与四件套登记 · D9 判定点逐条写 (a) 到 (f)。执行方式：用 workflow 编排，先加载 dynamic-workflows 技能，从 .zcode/workflow-drafts/r7-balance-diagnostics.dwf.ts 派生 r7-typed-evidence-budget.dwf.ts，必须用 CreateWorkflow 传 path 加 args 启动而不要用 AmendWorkflow（实测丢 args 让 dryRun 静默变 true），先传 maxRounds 0 dryRun true 做零成本编译校验；判定读数必须用 json.load 结构化读取而绝不用正则匹配自报文本；自跑 round-guard 与 commit-guard 与 verify 不依赖 hook 兜底；只推 r7/weather-reasoning；跑任何 stage 前先跑 tools/r7_round.py state 核对 current_round_goal。禁止：新增任何物理 loss 项、重开 C1 C2 Stage P R C A、重开城市静态协变量、重开容量轴、重开 EMA 与权重平均、重开长 lead 权重、重开低秩带与平衡带与地转投影与 case 自适应、改 model 目录任何已有行为、改 training/r7_evaluate.py 或 training/r7_long_rollout_runner.py、重建 store、新增或重下数据、改 data_identity、扫 lr 或 batch 或 clip、加 seed、与其它已关轴组合、读 2023 test、动 data/raw|interim|processed 与归档快照、合成数据、事后放宽出口、删弱测试、绕 checkpoint 与数据身份校验、paid 与 rental 与 exclusive 与 cron 与守护与会外续跑、对邻居进程发任何信号、main 合并与 force 与关 issue、自宣科学通过或 SOTA 或最终 goal 完成。只在三种情况停：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在进度节写为何停加恢复第一条命令。每步给可核查证据（命令加输出加 digest），估计值不得写成实测值。不承诺正面结果：若 (a)(b) 都为假，如实记负并写明这首次具备可引用性因为跑在 4 epoch 而不是 0.075 epoch；若 (a) 真而 (b) 假，只能写物理量必须按其类别路由不得写物理信息有用；若 (e) 为假，结论必须限定在 t2m 并写明 z500 动力学指标反向。不得事后调整判据。结束按 AGENTS.md 汇报改动、实验与失败、技能与代理使用、验证与精确 CI、成本与 issue 状态、未做项与下项。

字符数：**3,995**（`len(paragraph)` 实测；上限 4,000）

---

## §1 现状与起点

| 项 | 值 | 来源 |
| --- | --- | --- |
| 工作分支 | `r7/weather-reasoning` | `git status` |
| 起点 HEAD | `4476f63` | `git log -1` |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324` | 实测 |
| 数据实例 | `outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr` `[34084,17,65,65]` | —— |
| typed-evidence sidecar | `outputs/r7_s3_typed_evidence_fy65_20261103_attempt01`，`data_identity 81307008…`，`active [True×4]` | 实测 |
| 父 checkpoint | `r7_s3_position_retention_parent_migrated_20261103_attempt01/seed{41,42,43}/parent_update_0001600.pt` | 实测，当前 digest 下 |
| val manifest | `outputs/.../store/manifests_treatment/val.jsonl`（2022 全年） | —— |
| `used_gpu_h` | `619.436628`（**仅会计**，`cap_gpu_h`=20，`remaining`=−599.436628） | campaign-state |
| 2023 test | **未评分**，`r = 0` | —— |

### §1.1 三臂

| 臂 | `typed_evidence_mode` | 物理场 | 类型路由 | 参数量 |
| --- | --- | --- | --- | --- |
| **A** incumbent | 未设 | 否 | — | 3,286,037 |
| **B** generic | `generic_fusion` | 四场经一个融合投影 | 否 | 3,434,837 |
| **C** typed | `typed_routing` | 每场独立投影 | 是 | 3,434,837 |

B 与 C 逐位相等 ⇒ 只有 `C − B` 能归因给物理类型结构。
`B − A` 的 **+148,800 参数（+4.5%）** 是既有通路的固有成本，**必须如实申报**：
(b) 为真时只能说「开启物理证据通路后更好」，**不能**说该增量已排除容量解释。

### §1.2 为什么是 4 epochs

| point | updates | epochs | t2m 6h | 72h | skill@72h |
| --- | ---: | ---: | ---: | ---: | ---: |
| p0025 | 2,008 | 0.25 | 2.2862 | 3.7293 | +0.0981 |
| p1000 | 8,033 | 1.00 | 2.2474 | 3.6006 | +0.1593 |
| **p4000** | **32,132** | **4.00** | 2.1109 | 3.4446 | **+0.2305** |
| p1600 | 128,528 | 16.00 | 1.8605 | 3.3765 | +0.2607 |

15/15 单调；饱和判定 `unsaturated`。已关闭的物理对照臂全部在 **0.075–0.5 epoch**。

---

## §2 判据（冻结，任一为假即如实记负）

出口定义、失败形态与每条判据的文献/实测依据在
`docs/plans/0034-typed-evidence-under-budget.md` §「预声明判据」与
`docs/decisions/0115-typed-evidence-under-adequate-budget.md` §D5，本表是它们的不动摘要。

| 判据 | 要求 | 失败意味着 |
| --- | --- | --- |
| **(a)** | `C − B` 在 ≥2 lead 上同号为负 + 三 seed 一致 + block bootstrap 95% CI 排除零 | 物理类型结构无贡献 |
| **(b)** | `B − A` 在 ≥2 lead 上同号为负 | 物理证据本身无增量 |
| **(c)** | 若 (a) 真而 (b) 假，**只能**声称类型结构 | —— |
| **(d)** | 同臂在 4 / 16 epoch 增量方向一致 | 4-epoch 结论不可外推 |
| **(e)** | z500@72h 相对 A 臂恶化 ≤ **5%** | t2m 好看但动力学变差 |
| **(f)** | 预检全真 | 接线/单位事故 |

**判定语义**：

- 单 seed 反转只报不判。三 seed 一致才计，这是平衡轮 `(b)` 能靠噪声通过的教训。
- **任一臂缺失时该子句返回 `null`（未测），不是 `False` 也不是 `True`**。
  这是 `da1ed00` 那次「正则把空 body 默认成 false、卡片与读数相反」的直接修正。
- (c) 不是「免于报负」的出口：它限制的是**能声称什么**，不改变 (a)(b) 各自的真假。

### §2.1 必须先承认的相似性

`record:s3-typed-spatial-result` 关闭的是 `typed_routing → typed_spatial` 的聚合方式，
判据是 2,400 updates 下 B→C `|Δrmse| ≤ 3.2e-6 K`。本轮 `C − B` **复用同一对臂**，
**只把预算从 400/2,400 提到 32,132**。差别是**剂量**，不是机制：

- 若本轮为负 ⇒ 「typed evidence 在 0.075–4 epoch 全域无效」才成为可引用结论
- 若本轮为正 ⇒ 只在 4 epoch 成立，**不得**回溯为「之前那轮错了」

### §2.2 判据的证据来源

判定用的每个数字必须能在下列记录里找到出处，不得由执行者自报：

- 判据定义与失败形态：本文件 §2、`docs/plans/0034-typed-evidence-under-budget.md` §「预声明判据」
- 九轮关闭依据：`docs/decisions/0092-*.md`、`0094-*.md`、`0109-*.md`、`0111-*.md`、`0113-*.md`、`0114-*.md`
- 七项判决性测试的读数：`docs/plans/0034-typed-evidence-under-budget.md` §1–§2
- #79 的先例读数：`docs/R7_S2_79_TYPED_EVIDENCE.md:15-17`、§2
- 预算阶梯：`docs/R7_S3_THROUGHPUT_PHYSICS_B_P1600_RESULT.md:59,148`
- 族判别：`docs/decisions/0115-typed-evidence-under-adequate-budget.md` §D6

---

## §3 预算与停止

| 段 | 内容 | 估算 |
| --- | --- | --- |
| B-1 | 13 ckpt × 3 lead × 1447 case | 5.4 GPU-h |
| B-2 | 4 K 值 × 5 lead × 1447 case | 2.9 GPU-h |
| A | 3 臂 × 3 seed × 32,132 upd = 289,188 upd @ 1.1536 s | **92.7 GPU-h** |
| A（16 epoch） | **条件启动，本轮不授权** | 123 GPU-h |

> 注：`round-spec` 的 `planned_seconds`/`hard_cap_seconds`（172,800 / 259,200）是 §3 的
> 96 / 144 GPU-h 软硬预算按两卡折算成的挂钟。若本轮只跑到一个 seed 的前置档，账本按实测登记，
> 不得把未跑的部分写进 GPU-h。

软预算 **96 GPU-h** / 硬上限 **144 GPU-h**（段 B + A 三 seed 4-epoch）。

> **执行前修正 1（2026-11-16，0 GPU-h，未冻结任何协议）**：挂钟包络原写作 115,200 / 172,800 s，
> 是按**一轮**而不是一个 **seed** 设的。一个 seed worker 串行跑**三臂**、每臂再评 5 个 lead，
> 实测需 3 × (32,132 × 1.1536 + 2,682.3) = 119,254 s = **33.13 h**；16 h 的 per-seed 包络会在
> `_check_deadline` 处**截断训练臂**（不是优雅失败）。三 seed 在两卡上要两波，全轮不得超过硬上限
> 36 h/波 ⇒ 单 seed 包络上限 36 h。现取 **per_seed 126,000 s (35 h)**，两界都满足，比实测需求
> 留 5.6% 余量；同时把 `MEASURED_SECONDS_PER_EVALUATION` 由 2,588.1 修正为**2,682.3 s**
> （原值未覆盖全部 5 个 lead；现值为 lead 6/12/24/48/72 实测 elapsed 之和）。两处都由新增
> 反证测试钉住，并用「改回旧值必须变红」验证过。9 臂 × 11.04 h = **99.3 GPU-h** 会超软预算，
> 按决策 0030 记 overrun 继续，不截断。

> **执行前修正 2（2026-11-16，0 GPU-h）**：`evaluate_local` 只在传入 `boundary_margins` 时才写
> `boundary_rmse.csv`，而探针与 driver 都没传 —— `reading` 阶段会在**全部 GPU-h 花完之后**才
> `FileNotFoundError`（`outputs/r7_s3_typed_evidence_budget_probe.json` 已录到这个失败）。
> 现统一传 `BOUNDARY_MARGINS=(1,)`、按 `REGION="full"` 读取；`full` 是不分割的全网格，
> 与 `rmse.csv` 实测一致到 1 ULP（t2m lead24：2.665020267467307 vs …67），
> 因此**不改变任何判据数字**，只是让既有读取器真的有文件可读。

**只在三种情况停**：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作。
停时在 §5 写为何停 + 恢复第一条命令。

---

## §4 交付物清单

| # | 交付物 | 验收标准 | 落盘位置 |
| --- | --- | --- | --- |
| **D1** | 协议构建器 + (a)–(f) 预检 | `--preflight-only` 全绿才允许开跑 | `scripts/r7_s3_typed_budget_protocol.py` |
| **D2** | 三臂契约与容量匹配断言 | B/C 参数量与 FLOPs 逐位相等，A 严格更少 | 同上 |
| **D3** | 段 B 零训练探针 | 13 checkpoint × 3 lead；K ∈ {2,4,6,8} × 5 lead | `scripts/r7_s3_dose_probe.py` |
| **D4** | 冻结协议 | 含 `protocol_sha256`、父 pin、sidecar 身份 | `outputs/<stem>/protocol.json` |
| **D5** | 代码身份归档 | `code.zip` **必须带 commit 注释** | `outputs/r7_s3_typed_evidence_budget_code.zip` |
| **D6** | 逐 seed 逐臂 receipt | 9 份，含 checkpoint sha、s/update、expansion | `outputs/<stem>/seed{N}_{arm}_receipt.json` |
| **D7** | `readings.json` | 含 `scientific_claim / test_read / protocol_sha256 / exit_conditions / seeds` 五键 | `outputs/<stem>/readings.json` |
| **D8** | 证据页 + 四件套登记 | index record、账本行、campaign §8、`campaign-state` | `docs/` |
| **D9** | 判定点 | 逐条写 (a)–(f) 真/假/null，附实测数字 | 证据页 §判定点 |
| **D10** | 部分读数工具（本轮新增） | 从逐臂 receipt 装配 (b)/(e)，(a) 恒 `null`；空集不崩 | `tools/r7_s3_typed_budget_partial_reading.py`（190 行，`tools/` 不进冻结身份，已测断言） |

**命名纪律**：内容写 `readings.json`（**不是** `diagnostics.json`）。平衡轮两者命名不一致，
导致该轮的 `readings.json` receipt 从未出现、stage 永远显示未完成
（`tools/r7_round_stages.py:57`；`docs/R7_S3_BALANCE_DIAGNOSTICS_RESULT.md:176-180`）。本轮不得重犯。

---

## §5 进度块

| 时间（UTC） | 阶段 | 状态 | 证据 |
| --- | --- | --- | --- |
| 2026-11-16 | 规划 + 七项判决性只读测试 | **完成** | `docs/plans/0034-…`、`docs/decisions/0115-…` |
| 2026-11-16 | (f) 预检六条 | **完成，全真** | `--preflight-only`：`f_preflight_passed: true`；incumbent 3,286,037 / generic 3,434,837 / typed 3,434,837 |
| 2026-11-16 | 执行前缺陷修复（共 10 处） | **完成，0 GPU-h** | 见 §3 修正块；每处配反证测试（改回旧值必红，已实测） |
| 2026-11-16 | 单 lead 冒烟（路径验证） | **完成** | t2m@6h = **1.8605 K**，与 p1600 登记值逐位一致 |
| 2026-11-16 | 真实 2-update 训练冒烟 | **完成** | 2 updates / 216.1 s，loss 1.28813→1.17719，checkpoint 落盘 |
| 2026-11-16 | 协议冻结 + 代码归档 | **完成** | `outputs/r7_s3_typed_evidence_budget_code.zip` `afa0b98f…`（comment = `99ca045`，1,896 条目） |
| 2026-11-16 | 段 B-1 组内剂量曲线 | **42/42 lead 完成** | `.../segment_b_probe_scoring/dose/`，14 ckpt × 3 lead |
| 2026-11-16 | 段 B-2 推理深度扫描 | **完成 20/20 lead** | `.../segment_b_depth.json`，`gpu_hours 3.3461`，`error: null` |
| 2026-11-17 | 轮次冻结（attempt01） | **完成** | `outputs/r7_s3_typed_evidence_budget_20261116_attempt01/preparation_protocol.json`，`protocol_sha256 bc506eb1…`；`per_seed 126000 s / planned 172800 / hard_cap 259200` |
| 2026-11-17 | 段 A 启动 | **训练中** | `attempt01`：prepare ✓ archive ✓（`code_commit 49e359d`）；seed 41/42 并发于双卡，各 ~9.3 GB / 25% |
| 2026-11-17 | 段 A incumbent 两臂 | **完成** | `seed4{1,2}_incumbent_receipt.json`：35,866.8 / 35,828.7 s，**1.1162 / 1.1150 s/upd**，5 lead 齐全（1458/1457/1455/1451/1447），`new_parameters = 0` |
| 2026-11-18 | 第十八处缺陷：每 seed 包络装不下三臂 | **发现并留证（§5.5）** | treated 臂实测 **1.3886 / 1.4266 s/upd（+24.4% / +27.9%）**⇒ 三臂需 **37.29 / 37.96 h** vs 冻结包络 35.00 h ⇒ **typed 臂三个 seed 全部必然截断** |
| 2026-11-18 | 第十九处缺陷：装配器身份字段静默 null | **发现并留证（§5.5）** | `readings()` 不写 `source_sha256`/`data_identity`，`build_record` 用 `.get()` ⇒ 空身份记录可过校验器（实测 exit 0） |
| —— | 判定 (a)–(f) | **本轮判 null（未测）** | typed 臂必然缺失 ⇒ 冻结 `readings()` 拒写（§5.2b 守卫）⇒ **不产出 `readings.json`** |
| —— | 登记 | **待段 A 进程退出后做** | 段 B 12.7809 + 段 A 实测 GPU-h；`outcome_class` 记 `unresolved` |

**GPU-h = 段 B 实测 12.7809（尚未入账）。2023 test 未读，`r = 0`。段 A 已在跑并产出两份 incumbent receipt。**

### §5.1 段 B-2 判读（`json.load` 结构化读取，非正则）

来源：`outputs/r7_s3_segB_probe_20261116/segment_b_depth.json`
的 `decision` 键，字段名逐字引用。

| 读出字段 | 值 |
| --- | --- |
| `decision.depth_candidates` | **`[]`**（无深度合格） |
| `decision.cheaper_depth_candidates` | **`[]`** |
| `decision.dose_curve_status` | `"untested"`（该文件是 `--depth-only` 运行，不评分剂量曲线——**不是** saturated 判定） |
| `decision.sixteen_epoch_is_warranted` | `false` |
| `gpu_hours` | `3.3461`（实测） |
| `error` | `null` |

**逐深度读数**（`depth_sweep.per_depth[K].delta_vs_k4`，t2m，负值=优于 K=4）：

| K | 6h | 12h | 24h | 48h | 72h | 达标 lead 数（≥0.005 K） | 是否候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | **−0.0178** | **−0.0233** | −0.0011 | +0.0273 | +0.0559 | **2** | **否**（需 ≥3） |
| 4 | 0 | 0 | 0 | 0 | 0 | — | 基准 |
| 6 | +0.0376 | +0.0568 | +0.0515 | +0.0543 | +0.0438 | 0 | 否 |
| 8 | +0.1206 | +0.1912 | +0.2393 | +0.2884 | +0.3009 | 0 | 否 |

**判读（按 §4 预声明规则，不事后放宽）**：
- **没有任何深度在 ≥3 个 lead 上比 K=4 好 ≥0.005 K**，故 `depth_candidates` 为空。
- **K=2 是最接近的一个，但不能记为候选**：它只在 6h/12h 两个 lead 上更优，而在 48h/72h
  **反向变差**（+0.0273/+0.0559）。它同时便宜（`K<4` ⇒ 更少递归步、更低推理成本），
  但预声明明确禁止把「更便宜但更差」记成候选——故只作**观察**记录，不计入结论。
- K=6 与 K=8 在**全部 5 个 lead 上一致变差**，且随 K 单调恶化（K=8 达 +0.30 K），
  即在该 checkpoint 上加深推理**没有**收益。
- 因此本轮**不引入推理深度变更**，生产配置维持 K=4。

### §5.2 段 B-1 剂量曲线读数（14 个 checkpoint × 3 lead，已完成 42/42）

来源：`outputs/r7_s3_segB_probe_20261116/segment_b_probe.json`，
`json.load` 结构化读取；`gpu_hours 9.4348`（实测），`error: null`。

**身份核验**：`dose_identity.passed = True`，14 个 checkpoint 的
`distinct_identities` **只有一条** `59340710…f5d9324`，即全部 14 点在**同一个代码身份**
下加载 ⇒ 这确实是一条轨迹内的曲线，不是跨版本拼接。

| 读出字段 | 值 |
| --- | --- |
| **`decision.dose_curve_status`** | **`"saturated"`** |
| `decision.sixteen_epoch_is_warranted` | **`false`** |
| `decision.depth_candidates` | `[]` |
| `decision.cheaper_depth_candidates` | `[]` |
| `dose_curve.value` | `False` |
| `dose_curve.still_falling` | `False` |
| `dose_curve.tail_delta` | **+0.041507**（正值 = 末点比末 5 点之首更差） |

**末 5 点 t2m@72h（`dose_curve.tail_rmse_72h`）**：
`3.3350, 3.3324, 3.3239, 3.3538, 3.3765`

**判读（唯一依据是上面的结构化字段，不是对整条曲线做平滑观感）**：
末 5 点**不是严格下降**——先小幅下降到 3.3239，随后两点连续**回升**到 3.3765，
净变化 **+0.0415**。按 §4 的预声明规则，这判为 **`saturated`**，
`sixteen_epoch_is_warranted` 因此为 **`false`**。

**16-epoch 档位：数据不支持，本轮不授权，且不进入下一项任务的首选。** 这与
`docs/plans/0034-…` §4 引用的「15/15 单调、`unsaturated`」不同：那是以**四个独立阶梯档位**
（p0025/p1000/p4000/p1600）跨两次 `training_code_sha256` 比较得出的；本轮是**单条轨迹内**
14 点、单一代码身份的结果，方差来源更少，故优先采信本轮的 `saturated`。
两者差异本身是有信息的：**跨档位的表观单调**在单轨迹内**不成立**于末段。

**全曲线 t2m@72h（14 点，供核对）**：

| updates | 6h | 72h |
| ---: | ---: | ---: |
| 5,000 | 2.2650 | 3.6972 |
| 15,000 | 2.1570 | 3.5053 |
| 25,000 | 2.0614 | 3.4408 |
| 35,000 | 2.0040 | 3.4604 |
| 45,000 | 1.9580 | 3.4041 |
| 55,000 | 1.9272 | 3.3415 |
| 65,000 | 1.8990 | 3.4262 |
| 75,000 | 1.8818 | 3.3552 |
| 85,000 | 1.8763 | 3.3713 |
| 95,000 | 1.8690 | 3.3350 |
| 105,000 | 1.8702 | 3.3324 |
| 115,000 | 1.8708 | 3.3239 |
| 125,000 | 1.8720 | 3.3538 |
| 128,528 | 1.8605 | 3.3765 |

注意 6h 在约 95,000 步之后**也不再下降**（1.8690→1.8702→1.8708→1.8720，端点 1.8605 仅是
单点回摆），与 72h 的饱和判读方向一致。

**段 B 实测成本（不是估算）**：`EvaluationPreparation` 一次 29.7 s、train-only 气候态拟合
**722 s / 117 GB 读**（修复前每个 checkpoint 重建一次，18 次）；单 lead 229–238 s，
单 checkpoint（3 lead）1571–2357 s。B-1 合计 **9.4348 GPU-h**、B-2 合计 **3.3461 GPU-h**，
段 B 总计 **12.7809 GPU-h**（高于 §3 的 8.3 h 估算，实测偏高 54%——按实测登记，不按估算）。

**工具链副作用（如实记）**：探针产物目录原名 `…_20261116_attempt01_probe` 会命中
`current_attempt_dir` 的 `{stem}_attempt*` glob 并**抢在真正的 attempt 目录之前**，
使 `r7_round.py stage prepare` 读到不存在的 `protocol.json`。本轮改在**两段 B 跑完之后**
再 `freeze`（届时探针目录改名），以免移动正在写入的路径。这是编排顺序问题，不触及任何判据。

### §5.2b 段 A 排程与 reading 前置（只读核对，0 GPU-h）

| 核对项 | 实测 | 结论 |
| --- | --- | --- |
| 硬上限是否容得下两波 | 硬上限余 **71.66 h**；wave1 = 3 臂 × 33.22 h = **33.22 h**；seed43 在 +33.22 h 拿到 **68.22 h** 包络（需 33.22 h） | ✓ **不会截断**；三 seed 全部在硬上限内 |
| `readings()` 依赖的 protocol 键 | `protocol_sha256 / updates / batch_size / closed_dose / independent_arm_argument / limitations / preflight / train_data_identity / val_manifest` 全在冻结协议内 | ✓ 无 KeyError 风险 |
| 评估目录布局 | 写端 `seed{N}/evaluation/{arm}/lead_XXXh`；读端取 `receipt["evaluations"][lead]["dir"]` | ✓ 两端同一路径 |
| `verify` 三个校验器 | conventions `failing=0`、campaign_state `failures=0`、evidence index `113 records` | ✓ 登记前基线为绿 |

**第十二处缺陷（本轮新发现，0 GPU-h 修复）**：`readings.json` 本身就是 `reading` 阶段的
receipt，而 `write()` 拒绝覆盖、`stage_plan` 以 receipt 存在即 skip。因此**在臂未跑满时**执行
`stage reading` 会把不完整的表**永久冻结**并让该阶段显示已完成，之后每次都会 skip，
缺失的臂再也拿不到分数。现已改为：`complete` 为假时**抛错且不落 receipt**，并配反证测试
（关掉守卫会让测试写出文件并失败）。

### §5.2c 第十三处缺陷：判据 (a) 原本**没有**计算置信区间（0 GPU-h 修复）

冻结判据 (a) 是「C−B 在 ≥2 lead 同号为负 **+ 三 seed 一致 + block bootstrap 95% CI 排除零**」。
实测发现求值器 `r7_typed_budget_verdict.py` 只实现了前两条：**整个模块没有任何区间代码**，
而它自己的 docstring 还写着「the interval is what carries significance」。若照原样跑完段 A，
(a) 会在**只靠符号**的情况下判真，哪怕重采样区间跨零——这正是本轮明令禁止的读法。

修正（均已实跑验证，非仅改代码）：

| 项 | 内容 |
| --- | --- |
| 区间计算 | 新增 `pair_interval()`，从每个 lead 的 `provenance.json` 取逐初始化 MSE，调用仓内既有 `paired_rmse_bootstrap`（按初始化时间分块、95%） |
| 数据通路 | `readings()` 把每个 lead 的 `evaluation_dirs` 随臂表带出，求值器据此定位 provenance |
| 判据收紧 | 一个 lead 只有在**区间排除零**时才计入；(a) 的 `intervals` 字段逐条留痕 |
| null 纪律 | 区间算不出来 ⇒ 该子句为 **`null`**（未测），**不是 `False`**；缺失不确定性不得读成已测负结果 |

**真实数据验证**（不是 fixture）：拿段 B 的 K=8 对 K=4、lead 72h 跑该函数，得
Δ = **+0.3009 K**、区间 **[0.1927, 0.4267]**（排除零）、1,447 次初始化、1000 次重采样，
与 B-2 深度扫描报的 +0.3009 一致 ⇒ 通路正确。

**反证**：把「区间排除零」这一条件从 (a) 剥掉后，null 纪律测试立刻变成 `value: True`
（只靠符号通过）并失败 ⇒ 该条件是**承重**的，不是装饰。

**同时修正的 fixture 缺陷（由这些测试暴露）**：旧判据 fixture 只写 RMSE 数字、没有 provenance，
加区间后必然得到 `null`；而按 seed 键套 `evaluation_dirs` 会让每个区间静默报「not available」。
fixture 现按 **lead 键**写 `evaluation_dirs`，与 driver 实际写出的结构一致。

### §5.2d 第十四次发现：**冻结后被自己改破**（1.7352 GPU-h 已损失，如实记）

**事故**：冻结 attempt01（`code_commit 49e359d`）之后，我为修第十三处缺陷继续修改了
`scripts/r7_typed_budget_verdict.py` 与 `scripts/study_r7_s3_typed_budget.py`，两者都在
`execution_files_sha256` 的钉住清单里。于是 `validate_protocol()` 失败：
`execution source changed: scripts/r7_typed_budget_verdict.py`。

**影响面（精确）**：`worker()` 对**每个 seed 与 reading 阶段**都调用 `validate_protocol`。
seed 41/42 在改动**之前**已通过校验并在进程内运行，所以它们不受影响；但 **seed 43 与
`reading` 阶段会直接失败**——后者会在全部 GPU-h 花完之后才失败。

**处置**：立即停掉本轮的 seed worker（**只停本轮自己的进程**，pid 经 `pgrep` 精确匹配
`study_r7_s3_typed_budget.py --worker seed`，未触碰任何邻居进程），在训练到约 **2,700/32,132**
时终止，然后重冻结。

| 项 | 值 |
| --- | --- |
| 损失 | **1.7352 GPU-h**（两个 worker 的 `elapsed_seconds` 之和，实测；非估算） |
| 新冻结 | `code_commit f0da923`，`protocol_sha256 2630b9c8…`，archive `cc58c88c…` |
| 重跑校验 | `validate_protocol()` **PASS**（`protocol_sha256 76e1c214…`、`source_sha256 1db83955…`、`f_preflight true`） |
| 重跑 | seed 41/42 已重新启动 |

**教训（写进规程）**：冻结与「改执行源」互斥。任何触及 `execution_files_sha256` 清单内文件的
修复，必须在**冻结之前**完成；冻结之后若仍需修，正确动作是**重冻结**（新 attempt），
而不是在旧 attempt 上继续改代码——本次正是靠 `validate_protocol` 在 2,700 步时暴露，
而不是在 99 GPU-h 之后。

### §5.2e reading 阶段端到端干跑（真实数据，0 GPU-h，未触碰 attempt 目录）

在段 A 仍在训练时就证明了 `reading` 全链路可用：用**已存在的段 B 深度扫描产物**（四个深度臂
各有全部 5 个 lead、每次 1,447–1,458 个真实 case）冒充三臂，在 `/tmp` 下构造 9 份 receipt，
调用真实的 `driver.readings()` + `verdict.write()`。**这不是科学结果**（三臂故意取同/近 checkpoint，
对照无意义），只验证机器。

实测输出：

| 检查项 | 结果 |
| --- | --- |
| `complete` / `missing` | `True` / `[]` |
| 五键 `scientific_claim/test_read/protocol_sha256/exit_conditions/seeds` | **全在** |
| `seeds` / `region` | `[41, 42, 43]` / `full` |
| **判据 (a)** | `True`；`n_clearing=2`、`n_interval_excluding_zero=2`；lead 12 区间 **[−0.029568, −0.017142]**、`excludes_zero=True`、`n_initializations=1457` |
| 判据 (b) | `False`（该 fixture 下 B 未优于 A） |
| 判据 (c) | `True`，`permitted_claim` 按 (a) 真/(b) 假降级（scope 规则正常工作） |
| **判据 (d)** | **`None`** —— 无 16-epoch 读数，按 Objective 必须为 null，实测为 null |
| 判据 (e) | `True`（z500@72h 未恶化超 5%） |
| `verdict` | `mixed-type-structure-only` |
| 回读 `readings.json` | 五键仍在 |

`t2m@6h = 1.8605` 再次与 p1600 登记值一致，确认全程为真实数据。
**顺带发现的约束**：B-1 剂量曲线只评了 leads {6,24,72}，所以干跑必须改用深度扫描（5 lead）；
这也确认 `readings()` 对缺失 lead 会走 `missing` 分支而不是静默跳过。

### §5.2f 登记契约预检（0 GPU-h，避免在 99 GPU-h 之后才发现格式错）

**证据索引校验器会拒绝伪造的 commit id**。用一份「形状正确但值占位」的记录试跑
`tools/verify_r7_evidence_index.py`，得到三条 ERROR：

```
evidence_sha256 does not match docs/goals/s3-typed-evidence-budget.md
evidence_commit looks like a short SHA zero-padded to 40 characters, which is a fabricated commit id
experiment_commit looks like a short SHA zero-padded to 40 characters, which is a fabricated commit id
```

⇒ 登记时**必须**填真实 40 位 commit 与**实际文件**的 SHA256，不能写占位符；这也意味着
证据页必须先定稿提交、再用**该提交的 hash** 与文件 sha256 生成记录。

**账本算术已核对**（`r7_round.py ledger-plan 12.7809`）：

| 项 | 值 |
| --- | --- |
| `rows_before` | 94 |
| `previous_total_gpu_h` | 619.436628 |
| `new_row_gpu_h`（段 B 实测） | **12.7809** |
| `new_total_gpu_h` | **632.217528** |
| `cap_gpu_h` / `remaining_gpu_h` | 20.0 / −612.217528（只记账） |

段 A 完成后，其 GPU-h 必须按**实测**（9 臂的 `elapsed_seconds` 之和）另行加行，
不得把估算写进账本。索引记录的 `metrics.gpu_hours` 必须与账本行对齐到 1e-4。

**枚举值**（校验器允许集）：`outcome_class ∈ {engineering-positive, audit, negative, mixed,
unresolved}`；`candidate_state ∈ {candidate, blocked, needs-review, not-candidate}`；
`scientific_claim` 必须为 `false`；`limitations` 必须是非空字符串列表。

### §5.2g 排程余量（实时核对，决定本轮能否跑完）

| 项 | 实测值 |
| --- | --- |
| 当前 s/update（两 seed 取较差者） | **1.1448** |
| 单臂（32,132 upd + 5 lead 评分） | 10.96 h |
| **wave 1**（3 臂串行，seed 41/42 并发） | **32.89 h** |
| **两波总需**（含 seed 43） | **65.78 h** |
| 硬上限剩余 | **71.63 h** |
| **余量** | **5.85 h** |
| 可承受的最慢 s/update | **1.2541**（即再慢 **9.5%** 就会截断） |
| seed 43 实际拿到的包络 | 67.89 h（对其 32.89 h 需求） |
| 磁盘 | 835 GB 可用；attempt 现占 88 MB（每臂 5 lead × 1,447 case） |

**判读**：按当前速率**能跑完**，但余量只有 9.5%。GPU 共驻若显著抬高延迟就可能越过 1.2541，
届时按冻结规则是 `budget_limited`/`failed` 且**全额记账**——不会被当作通过。
这一点在开跑时并不显式，属于本轮实测得到的运行约束，如实登记于此。
若最终被硬上限截断，正确动作是**新 attempt、新协议**重跑缺的 seed，**不是**放宽判据或拼接读数。

**上表之后的重算（§5.2g 定稿时用的速率已过时，这里更正，不是改判据）**：

| 项 | §5.2g 原值 | 实测重算 |
| --- | --- | --- |
| s/update（两 seed 取较差） | 1.1448 | **1.1422** |
| 单臂（32,132 upd + **五 lead 合计**评分 2,682.3 s） | 10.96 h | **10.94 h** |
| wave（3 臂串行） | 32.89 h | **32.82 h** |
| 剩余墙钟 | 71.63 h | **70.97 h** |
| 还需工作 | 65.78 h | **64.76 h**（wave1 已烧 0.88 h） |
| **余量** | 5.85 h | **+6.21 h** |
| 可承受的最慢 s/update | 1.2541 | **1.2254**（余量 **+7.3%**） |

两点更正说明，避免把口径当成结论：

1. `MEASURED_SECONDS_PER_EVALUATION = 2_682.3` 是**五个 lead 的合计**
   （231.7+308.0+426.6+711.1+1,004.9 s），**不是**每 lead 值。第一次重算我按 5× 计过，
   把余量算成 −11.66 h——那是我自己的口径错误，已更正，如实留在这里。
2. 两个 seed worker 在**两张卡上并发**，故两波 = 2 × wave，不是 9 × 单臂串行。
   按 9 臂逐个串行会算出「上限 0.8157 s/update、缺口 −28.6%」，同样是口径错误。

⇒ 结论不变（能跑完），但**余量从 9.5% 降到 7.3%**——GPU 共驻变慢超过 7.3% 就会截断。

**随训练推进的再次复算（update ≈ 4,440，速率已改善）**：

| 项 | 值 |
| --- | --- |
| 实测 s/update（seed41/42 全程增量口径） | **1.1303 / 1.1265** |
| 单臂（32,132 upd + 5-lead 评分 2,682.3 s） | **10.83 h** |
| wave（3 臂串行） | **32.50 h** |
| 剩余墙钟 | 70.46 h |
| 还剩工作（wave1 余量 + wave2） | 63.60 h |
| **余量** | **+6.86 h** |
| 可承受的最慢 s/update | **1.2584**（余量 **+11.3%**） |

速率比早期估计**更快**（1.1422 → 1.1303），故余量从 7.3% 回升到 **11.3%**。
**这一格的算式我先后写错过两次**（一次把 5-lead 合计当每 lead、一次把两卡并发当九臂串行，
还有一次把 `burned` 重复扣减得出 −92.8 s/update 的荒谬值），最终口径为
`arm = (remaining_wall + burned)/6`，即两波各 3 臂、wave1 已burned ⇒ 上表。三个错误版本
都留在此处，以免把某一次的口径当成结论。

### §5.2h 登记四件套的精确契约（0 GPU-h 预读，避免在 99 GPU-h 后返工）

三套校验器各自的硬性要求，逐项读出（不是猜）：

| 组件 | 要求（来源） |
| --- | --- |
| **证据索引记录** | `format="urbanpidit-r7-evidence-record-v1"`；`record_id` kebab-case；`evidence_path` 必须仓库相对且**文件存在**；`evidence_sha256` 必须与**该文件实际 sha256** 一致；`evidence_commit`/`experiment_commit` 必须是**真实 40 位 SHA**（零填充的短 SHA 会被判为 fabricated）；`outcome_class ∈ {engineering-positive, audit, negative, mixed, unresolved}`；`candidate_state ∈ {candidate, blocked, needs-review, not-candidate}`；`priority ∈ [0,100]` 整数；`scientific_claim` **必须 false**；`limitations` 非空字符串列表；`metrics` 必须是标量/列表构成的 JSON 对象 |
| **账本行** | 表格行必须匹配 `| name | gpu | cum | evidence |`；`gpu` 与 `cum` 数值，累计容差 **1e-3**；行内 `record:<id>` 必须指向**已存在**的索引记录；`metrics.gpu_hours` 与账本行对齐到 **1e-4** |
| **campaign-state** | 必含 8 键：`current_node / previous_node / current_round_goal / previous_round_goal / previous_round_evidence / cap_gpu_h / used_gpu_h / remaining_gpu_h`；累计容差 1e-3；`current_round_goal` 必须指向真实 brief 且通过 `check_goal_brief` |
| **campaign §8** | 行内 `record:<id>` 引用必须可解析；文档指针必须是真实 `docs/*.md` |

**因此登记顺序被钉死**：① 先定稿并提交证据页 → ② 用**该提交的 40 位 SHA** 与文件 sha256 生成
索引记录 → ③ 账本加行（GPU-h 取实测、与记录对齐 1e-4）→ ④ 更新 campaign §8 与 campaign-state
→ ⑤ `commit-guard` 再提交 → ⑥ `verify` 三套校验器全绿。
任何一步用占位符都会在第 ② 步被拒。

### §5.2i 无断点续跑：本轮最大的剩余运行风险（只读核对）

`training/r7_long_rollout_runner.py` 自己写明：

```
"A new optimizer imports verified model weights only; no failed-attempt resume is supported."
"resume_permitted": False
```

即 **worker 若中途死亡，该臂只能从父 checkpoint 重跑，不能续跑**。`checkpoint_every=20` 会在
盘上留下中间 checkpoint，但 runner 拒绝据其恢复（这是刻意的科学约束：新优化器只接受已验证的
模型权重，避免把失败尝试的半程状态混进结果）。

**敞口量化**（单 seed 三臂串行；臂单价 10.96 h 实测）：

| 死亡发生处 | 损失 |
| --- | --- |
| 臂 1（incumbent） | **32.88 h** |
| 臂 2（generic） | 21.92 h |
| 臂 3（typed） | 10.96 h |
| **期望损失/次** | **21.92 h** |

66 h 墙钟内共 6 个臂进程（每 wave 2 seed × 3 臂顺序执行），因此这一项的期望敞口**大于**
§5.2g 的 9.5% 速率余量，是本轮最大的剩余运行风险。

**若真发生，正确动作（现在写定，避免事后临时决定）**：
- 死亡的 seed **不带**其已完成臂进入结论；该 seed 在新 attempt 下**整 seed 重跑**；
- 若只剩一个 seed 缺失且硬上限不允许重跑满一波，则按冻结规则把该 seed 记为
  `budget_limited`/`failed`，**该 seed 不进对照**，判据按「缺失臂 ⇒ `null`」处理；
- **不得**用 `checkpoint_every` 留下的中间 checkpoint 冒充完成态，**不得**跨 attempt 拼接读数，
  **不得**因缺臂而放宽 (a)/(b) 的容差或改判据。

本轮**不**为这一风险新增恢复机制：加断点续跑会改动 `training/**` 行为（会改
`training_code_sha256`，等于换协议），属于本轮明令不做的事。

### §5.2j 训练→评估链路端到端验证（真实 checkpoint，0 训练 GPU-h）

此前只验证过「段 B 的既有 checkpoint 能被评分」，但没验证过**本轮训练路径产出的 checkpoint
能被 `evaluate_local` 读回**。这是九臂各训 11 h 之后才会踩到的最后一环，现在提前打通：

| 步骤 | 实测 |
| --- | --- |
| 用本轮 `_expand` + 真实合约训练出的 checkpoint | `/tmp/arm_step_out/typed/update_0000002.pt`，**163 个模型键** |
| 代码身份 | `model_code_sha256` 与当前树**一致** |
| 合约回读 | `kind=process`、`typed_evidence_mode=typed_routing`、`data_identity=81307008…` |
| `evaluate_local` 评分 | `n_evaluated: 200`、`boundary_rmse.csv` **已写出**、**200 条 provenance 初始化行** |
| 读数 | t2m **2.1509**、z500 **126.95** |

⇒ 训练产出的 checkpoint → 评分 → boundary CSV → provenance 逐 case 行 → 判据输入，
整条链路在**真实文件**上跑通。这也再次证明读数的 (a) 区间所需的逐初始化行确实由该路径产生。

（该 checkpoint 只训了 2 updates，故 RMSE 不具科学意义；此步只验证机制。）

### §5.2k 主干同一性：三臂的因果归因前提被**直接**验证（0 GPU-h）

判据 (a)(b) 的整个解释力压在一句话上：**incumbent / generic / typed 从同一套主干权重出发**。
若 pathway 构建挪动了任何主干张量，那么 (a)(b) 的差值里就同时包含「处理效应」和「初始化差异」，
两件事再也分不开。preflight 只对 off 通路断言了这一点；这里把三臂、每个共享张量都直接查一遍。

实测（`/tmp/trunk_identity.py`，直接对 `proto.parent_pin(seed)` 的真实父权重跑 `driver._expand`）：

| seed | incumbent 新增键 | incumbent == parent（逐张量） | generic 主干 155 张量 | typed 主干 155 张量 | generic vs typed 共享 159 张量 | pathway 键交集 |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | `[]` | **True** | **逐位相同**，`max_abs_diff 0.000e+00` | **逐位相同**，`0.000e+00` | 0 个不同 | 仅 `typed_evidence.conv/norm` 4 个 |
| 42 | `[]` | **True** | 同上 | 同上 | 0 个不同 | 同上 |
| 43 | `[]` | **True** | 同上 | 同上 | 0 个不同 | 同上 |

三点结论，逐条对应一个可能被质疑的地方：

1. **incumbent 新增零键、且是父权重逐张量原样**——不存在「off 通路其实也建了点什么」的可能。
2. **generic/typed 与父共享的 155 个张量全部逐位相同（`torch.equal`，不是容差比较）**——
   处理只发生在 pathway 块内部，主干未被任何随机初始化或重排触碰。
3. **两臂的 pathway 键交集恰好是共享编码器的 4 个张量**（`typed_evidence.conv.{weight,bias}`、
   `.norm.{weight,bias}`），**不含投影块**；而共享 159 张量中 0 个不同 ⇒ 两臂的**唯一**差异
   是投影块的具体形状/内容。这正是 §5.2 里 `shared_encoder_sha256` 检查所依赖的那个不变量的
   独立复核：`_seed` 断言「两臂共享编码器同一、投影块不同」，此处从权重层面确认它成立。

⇒ (a) 的「类型结构」与 (b) 的「物理证据」两个归因**不是靠推断**，而是靠逐位相同的主干成立的。

### §5.2l 第十六处缺陷：`epochs` 元数据少了 batch 因子（**不修，记 errata**）

价值层审计（形状审计查不到的那一类）发现两处把 `epochs` 算成 `updates / 32131`：

| 位置 | 代码 | 记录值 | 正确值 |
| --- | --- | --- | --- |
| `scripts/study_r7_s3_typed_budget.py:339` | `int(body["updates"]) / 32_131` | **1.000031** | **4.000124** |
| `scripts/r7_s3_typed_budget_preflight.py:174` | `proto.UPDATES / 32_131` | **1.000031** | **4.000124** |

正确算式是 `updates × batch / usable_windows = 32132 × 4 / 32131`，这是本仓的既定口径
（`tests/test_r7_s3_curve_large_points.py:37` 逐字钉住 `32132 * 4 / 32131`）。**独立佐证**：
已注册的 `p4000` 档与本轮剂量完全相同，`r7_s3_curve_point_large_protocol.point_blocks("p4000")`
记录 `epochs = 4.000124`——与本轮正确值逐位相同。缺的正是 ×4。

**影响面（已逐项核对，是惰性的）**：

- 判据 (a)–(e) **不读**该字段（`grep` 全空）；
- (f) 子句的 `passed` 只由 `updates / batch_size / seeds / closed_round` 算出，
  **不含** `epochs`（实测 `"epochs" in str(clause["passed"]) == False`）；
- 三套校验器（`check_round_receipts` / `verify_r7_evidence_index` / `r7_round`）**都不读**它。

⇒ 该字段不影响任何判定，但会永久留在 `protocol.json` 与 `readings.json` 里，**必须在证据页
的 `limitations` 里点名**，否则日后有人引用「1.000031 epoch」就会把本轮说成 1/4 的剂量。

**为什么不修（这是刻意的，不是省事）**：两个文件都在 `execution_files_sha256` 里（84 个冻结源之一），
而 `validate_protocol` 在**每个** stage 开头逐文件比对。现在改它们会让 `reading` 阶段——
也就是**全部 99 GPU-h 花完之后**——抛 `execution source changed`。这正是第十四处的原样重演。
重冻结则要换 `protocol_sha256`，而已启动的 6 个臂的 `fine_tune_started.json` 写的是
`contract.protocol_sha256 = 76e1c214…`，与新的 body 不符，等价于从头重跑，且 `archive_receipt`
会因「receipt 存在即跳过」而留下**旧代码 zip 配新 digests** 的断裂链条。

**处置**：不动冻结源、不重冻结；本节的数字为权威算式，`epochs` 字段按 errata 处理，
并在证据页 `limitations` 与账本 evidence 里各写一次。判据与容差**未**改动。

### §5.2m receipt 契约与真实写者逐字段对齐（0 GPU-h）

`readings()` 消费的字段此前只在合成 fixture 上验证过。这里把 `_train_arm` 真正写下的 receipt
与 `readings()` / `pair_interval()` 的读取点逐个对上，并用**段 B 的真实产物**取 schema：

| receipt 字段（`_train_arm` 写入） | 真实来源 | 读取方 | 对齐 |
| --- | --- | --- | --- |
| `evaluations[str(lead)].dir` | `seed{seed}/evaluation/{arm}/lead_{lead:03d}h` | `_region_rmse(dir/boundary_rmse.csv)` | ✓ |
| `evaluations[str(lead)].n_evaluated` | `provenance["n_evaluated"]` | 非空守卫 | ✓ |
| `evaluations[str(lead)].boundary_rmse_csv_sha256` | `sha256_file(destination/"boundary_rmse.csv")` | 归档 | ✓ |
| `evaluations[str(lead)].provenance_sha256` | `sha256_file(destination/"provenance.json")` | 归档 | ✓ |
| `parameters` / `seconds_per_update` / `typed_evidence_mode` | 模型与 report | `readings()` 表格 | ✓ |

`boundary_rmse.csv` 实测列（段 B 真实文件）：
`region, margin_cells, n_grid_points, full_area_fraction, lead_hours, variable, unit, rmse,
n_initializations` ⇒ `_region_rmse(path, "full", "t2m"/"z500")` 的键都在，且 `region == "full"`
的行确实存在（17 变量 × 5 lead × 3 region 全齐）。

`provenance.json` 实测结构：顶层 15 键，`initializations` 每条含
`{cumulative_reasoning_steps, init_time, mse, valid_times}`。`pair_interval()` 重组的正是
`init_time / valid_times / mse` 三个字段并交给 `paired_rmse_bootstrap` ⇒ **区间所需的数据确实由
`evaluate_local` 写出**，不是假设。段 B 真实文件含 **1,455 条**逐初始化行（lead 24 h）。

⇒ `_train_arm` → receipt → `readings()` → 区间，字段层面无缺口。

### §5.2n 区间与写盘在**真实文件**上跑通（0 GPU-h，不是 fixture）

上一节只核对了字段「在不在」。这一节把 `pair_interval()` 依赖的两个拒绝路径真正跑一遍——
它们每一个都会把判据 (a) 变成 `null`，而且是在**全部 99 GPU-h 花完之后**才暴露：

**① `paired_rmse_bootstrap` 的元数据契约**。它要求 6 个字段在两臂间**逐字相等**
（`evaluation_manifest_sha256 / channels / units / lead_hours / step_hours / split`），
另要求 `n_evaluated == len(initializations)`；任一不符即 `ValueError`。实测段 B 的
`provenance.json` **29 个顶层键**，上述 6 个 + `n_evaluated` **全部存在**：

| 字段 | 实测值 |
| --- | --- |
| `channels` | 17 个变量（t2m, u10, v10, mslp, z850, …） |
| `units` | 与 `channels` 等长 |
| `lead_hours` / `step_hours` / `split` | `[24]` / `6` / `val` |
| `evaluation_manifest_sha256` | `f8133e79…` |
| `n_evaluated` | 1455 == `len(initializations)` |

**② 真跑一次**（两个真实 checkpoint、真实 1,447 个初始化、真实 72 h）：

```
paired_rmse_bootstrap(a, b, block_days=7, replicates=1000, seed=42, confidence=.95)
→ SUCCESS: replicates=1000, block_days=7, n_initializations=1447
→ t2m@72h: rmse_A_minus_B = +0.05594, interval = [-0.02824, +0.116997]
```

**这一条同时是一个反证**：`k2` 与 `k4` 是同一 checkpoint 的两个深度，本不该有差异，
而区间**正确跨零**（`-0.028 … +0.117`）。若区间代码写错成恒排除零，(a) 就会在无差异处为真。
冻结参数 `block_days=7 / replicates=1000 / confidence=0.95` 与 `paired_rmse_bootstrap` 的默认值
**逐字一致**，本轮未传任何覆盖值。

**③ `verdict.write()` 拒绝覆盖**（`path.open("x")`），配合 §5.2 的 completeness 守卫：
不完整的读数既不会写出，也不会被后续 attempt 顶掉。

⇒ 判据 (a) 的区间、五键写盘、以及「不完整即拒写」三条路径都在真实产物上验证过。

### §5.2o 调度器与磁盘：两件决定「能否跑完」的运行事实（只读核对，0 GPU-h）

**① 谁在推进 seed 43**。只有 seed 41/42 有 worker 进程时，容易误判为「第三 seed 没人管」。
实测父进程链：

```
timeout 259200  →  tools/r7_round.py stage-seeds (pid 395250)  →  study_r7_s3_typed_evidence...py --worker seed --seed 41 / 42
```

`run_seeds()`（`tools/r7_round_stages.py:362`）的行为已逐条读出：
- `pending` 取出所有**尚无 receipt** 的 seed；
- 每张卡最多一个本轮 worker（决策 0047），第三个 seed 在**任一卡空出时立即启动**；
- 每个 seed 的截止是 `min(round_deadline, now + per_seed_seconds)`（`:292`），
  即 seed 43 拿到的是**完整 35 h 包络**，不会被前面两波耗掉的时间压扁；
- `headroom-blocked` 只**只读等待**（R-054），绝不对邻居进程发信号。

⚠️ 一个容易看错的守卫（**`scripts/study_r7_s3_typed_budget.py:398`**，不在 scheduler 里）：
`if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS: raise`
看起来像「剩余时间多于单 seed 包络就拒绝」，实际是**拒绝一个比冻结包络更宽松的 deadline**
——防止调用方把截止时间声明得超出冻结值。方向是对的，但读反了会得出「seed 43 永远起不来」
的错误结论。已按语义核对：`run_stage` 传的是 `min(round_deadline, now + per_seed)`
（`tools/r7_round_stages.py:292`），恒 `≤ now + per_seed`，故该守卫必然通过。

**seed 43 的时间账（按 `perf_counter` 冻结值与实测速率实算，不是估计）**：

| 项 | 值 |
| --- | --- |
| round deadline | `3433567.3` |
| wave1 结束（3 臂 × 10.94 h） | `3293033.0` |
| seed 43 的 limit = `min(deadline, pop + 126000)` | `3419033.0` ⇒ 窗口 **35.00 h** |
| seed 43 需要 | **32.82 h** ✓（余 2.18 h） |
| seed 43 结束 | `3411183.4` |
| 留给 `reading` | **6.22 h** ✓ |

⇒ 第三 seed 有 2.18 h 余量、reading 有 6.22 h，两处都不依赖乐观估计。
（这些绝对 `perf_counter` 值随轮次起点不同而变；判读只用**差值**。）

**② 磁盘与孤儿守卫**。`/data` 3.6 T，已用 2.6 T，**可用 834 GiB**（77%）；attempt 现占 **88 MB**
（两臂各 37 MB，尚无 checkpoint）。

**checkpoint 体积已实测，不是估的**：父 checkpoint `parent_update_0001600.pt` = **12.6 MiB**
（13,228,745 bytes）。`checkpoint_every=5000` ⇒ 每臂 5000/10000/…/30000 + 端点 = **7 个**，
九臂共 63 个 ⇒ **0.78 GiB**。评测目录按每 lead 实测 `elapsed_seconds ≈ 392 s` 量级、
provenance 1,455 行，远小于 checkpoint 总量。⇒ 段 A 全部落盘**不到 1 GiB**，834 GiB 毫无压力。
（此处原写「数十 GB 量级」，是未测量的估计；实测后已更正。）

`tools/r7_round.py stage-plan seed 41` 实测返回 `already_running: true` + 5 个存活 PID
⇒ 重复启动会被拒而不是双开抢 GPU；`orphans` 子命令报告各 stage 的 `running/pids`。
两处都只读，不向任何进程发信号（R-054）。

**③ 共驻实况（只读快照，2026-11-16T21:48Z）**。余量结论依赖「GPU 共驻不会显著抬高延迟」，
故直接看卡：

| GPU | 显存 | 利用率 | 计算进程 |
| --- | --- | --- | --- |
| 0 `GPU-408ad137…` | 9308 / 24576 MiB | 67% | **仅** pid 395258（seed41），9262 MiB |
| 1 `GPU-9d1624af…` | 9287 / 24576 MiB | 27% | **仅** pid 395260（seed42），9262 MiB |

**两张卡上都只有本轮自己的一个 worker，没有邻居进程**（`nvidia-smi --query-compute-apps`
只有这两行）。显存占用 9.26 GiB / 24.6 GiB ⇒ 余量充裕；`estimated_peak_bytes` 的门槛远未触及。
这解释了速率为何从 1.1422 改善到 1.1303：**当前没有共驻竞争**。

⚠️ 但**不得**把这读成「余量可以吃掉」：利用率是瞬时量，不是速率证据；两个 seed 的
`update` 与 `elapsed_seconds` 仍在同步前进（4960 / 1.56 h，两者相同），这才是速率证据。
邻居随时可能出现，而 R-054 禁止我们干预——所以 §5.2g 的 11.3% 余量仍是必须守住的量。

### §5.2p 最后一步之前的所有断言都已在真实冻结协议上预演（0 GPU-h）

这些断言**只在训练全部跑完后才执行**——`_seed` 的路径一致性检查在 3 臂 × 10.94 h 之后、
`readings()` 的字段解引用在 9 臂之后。也就是说，任何一条不成立都要在**99 GPU-h 花完**才暴露。
逐条预演：

| 断言 | 位置 | 何时才执行 | 预演结果 |
| --- | --- | --- | --- |
| 两 treated 臂共享编码器同一 | `_seed` | 每 seed 三臂跑完（32.8 h） | **PASS**，三 seed 各自 `540c0471…` / `d50442a8…` / `ef4b81af…` |
| 两 treated 臂投影块**不同** | `_seed` | 同上 | **PASS**（若相同则路由开关没建出不同图） |
| incumbent 无共享编码器 | `_seed` | 同上 | `None`（符合：off 臂不带通路） |
| `readings()` 的解引用键齐全 | `readings()` | 9 臂之后（99 h） | **24/24 顶层键全部存在**，含 `train_windows.{window_sha256,excluded_sample_ids}`、`preflight.clauses.{capacity…,forward_arithmetic…}` |
| receipt 五键 + 结构完整 | `check_round_receipts --require-reading` | 登记时 | **PASS**（`reading ok=True`；`scientific_claim=False`、`test_read=False`、`protocol_sha256=76e1c214…`、`exit_conditions` 五子句、`seeds=[41,42,43]`） |
| 现有 stage 的 receipt 合法 | `check_round_receipts` | — | **PASS**，`checked=2, failed=[]`（`prepare` + `archive`） |

**其中第三条反向排除了缺陷 5 的重演**：缺陷 5 是「pathway 初始化检查哈希了全部新增键」，
generic 5 键 vs typed 8 键导致 digest 必然不同、在 33 h/seed 之后抛错。现在的检查换成了
**可达的**共享编码器不变量，并**正向**断言投影块不同——上表第二行就是这条正向断言的实测。

⇒ 从「训练结束」到「登记完成」之间没有未预演的断言了。

### §5.2q (e) 护栏的两个常数已追到已注册产物（0 GPU-h）

判据 (e) 的注释声称「p4000 → p1600 的 z500@72h 已从 427.35 升到 445.78」。这两个数此前只是
注释里的引用，本次直接到已注册的 `readings.json` 核对：

| 档位 | 产物 | `leads.72.z500` 实测 |
| --- | --- | --- |
| p4000 | `outputs/r7_s3_throughput_physics_b_p4000_20261105_attempt02/readings.json` | **427.346** |
| p1000 | `outputs/r7_s3_throughput_physics_b_p1000_20261104_attempt01/readings.json` | 444.74 |
| p1600 | `outputs/r7_s3_throughput_physics_b_p1600_20261106_attempt01/readings.json` | **445.777** |

⇒ `Z500_72H_REFERENCE = 427.35` 与「445.78」两个常数**都是已注册读数的四舍五入**，不是估的。
且剂量阶梯不是单调的（p1000 444.74 > p1600 445.777 ≈ 同水平），说明 z500@72h 在 8033 updates
之后基本走平——这恰恰是 (e) 存在的理由：**t2m 能继续变好而动力学指标不动或反向**。

**但 (e) 的判定对象不是这两个常数**。`clause_e()` 有三列，只有一列参与判定：

| 列 | 内容 | 是否判定 |
| --- | --- | --- |
| `relative_change` | 该臂 vs **同 seed 的 incumbent 臂** | **判定**（≤ +5%） |
| `against_reference` | 该臂 vs 427.35 | 仅信息（混入了预算漂移，不是臂效应） |
| `within_tolerance` | 上列布尔汇总 | 判定 |

`tests/test_r7_s3_typed_budget.py:203` 逐字钉住这一点：*"The guard is an arm-to-arm test; the
427.35 reference is informational only."* ⇒ 即使用 427.35 当分母，也不会把本轮的臂效应和
p4000→p1600 的既有漂移混为一谈。

### §5.2r 两次被作废的尝试留下了可核对的失败证据（0 GPU-h）

`outputs/` 下有 `r7_s3_typed_evidence_budget_voided_20261117b/`（不在 `output_stem` 的
`{stem}_attempt*` glob 内，故不会被 `current_attempt_dir` 选中）。它不是垃圾，是**缺陷 11 的
现场**：

```
seed41.log:  File ".../study_r7_s3_typed_budget.py", line 234, in _arm_contract
               "data_identity": body["train_data_identity"], "source_sha256": body["source_sha256"],
             KeyError: 'source_sha256'
```

`stage_ledger.jsonl` 记 `prepare/archive success`、`seed 41/42 failed`。

**两个冻结协议的差异已实测**（这正是缺陷 11 被修好的证据）：

| 项 | voided（63f23c46…） | live（76e1c214…） |
| --- | --- | --- |
| `source_sha256` | **缺失** ⇒ `KeyError` | **存在**（`1db83955…`，由 `sha256sum` 实算） |
| `train_windows` | 两键 | 两键（同） |
| 文件 sha256 | 不同 | 不同 |

⇒ 现在跑的 6 个臂用的是**已修复**的冻结协议。这条写在这里是为了让「为什么有两个 attempt 目录」
有据可查，而不是靠记忆。

### §5.2s 登记装配器与两处自造缺陷（含反证测试）

最后一步（证据页 + index record + 账本行 + campaign §8）此前只能靠手写，而它发生在**全部
99 GPU-h 花完之后**。新增 `scripts/r7_s3_typed_budget_register.py`（191 行，纯只读装配）把
能从产物读出的东西全部读出来，不手抄：

| 装配项 | 来源 | 为什么不能手抄 |
| --- | --- | --- |
| GPU-h | 臂 receipt 的 `elapsed_seconds` + 每个 `evaluations[lead].elapsed_seconds`，另加段 B 两个探针的 `gpu_hours` | **没有任何 stage receipt 记累计 GPU-h**（实测 `prepared_protocol.json` 只有 `planned/hard_cap/per_seed`，`archive_receipt.json` 只有 3 键） |
| (a)–(f) | `readings.json` 的 `exit_conditions` / `exits.values` / `preflight` | 二次计算等于给判定开后门 |
| `protocol_sha256` / `source_sha256` / `data_identity` | `readings.json` | 手抄即可能漂移 |

**段 B 的 GPU-h 尚未入账**（已核对：账本 94 行里没有本轮任何一行），实测
`segment_b_probe.json = 9.4348` + `segment_b_depth.json = 3.3461` = **12.7809 GPU-h**，
必须与段 A 合并在同一行登记。

**我在这个新文件里自己造了两处缺陷，两处都已修并留反证**：

1. **(f) 判据的形状**。`checks.capacity` 与 `checks.forward_arithmetic` 是**带 `passed` 键的
   dict**，我写成 `entry is True` ⇒ 绿着的 preflight 会被报成 (f) 假。这正是「把形状判错、
   于是卡片与读数相反」那一类。改为逐条读 `entry["passed"]`，且**缺失返回 `None`（未测）
   而不是 `False`**；反证测试覆盖 `{"passed":True}/{"passed":False}/True/{}/None` 五种输入。
2. **浮点尾巴**。段 B 求和得 `12.780899999999999`，与账本要求的 **1e-4 对齐**在末位就会差
   1e-16。已按 1e-4 量化。

另有两处「拒绝而非记账」的守卫，都用反证测试钉住：
- receipt 缺 `elapsed_seconds` ⇒ `RuntimeError`（**不得记 0**，否则账本少记）；
- `readings.json` 缺任一子句 ⇒ `RuntimeError`。

**code.zip 与冻结身份的一致性已核对**：注释 `f0da923796f2…` == `protocol.json` 的
`code_commit`，zip 实测 sha256 == 冻结的 `code_zip_sha256`（`cc58c88c…`）。测试里那条
`pytest.skip("archive is stale … built from f0da923, HEAD is b980703")` **是预期行为**：
HEAD 因文档提交前进，而归档只须匹配**冻结**身份，不需要匹配 HEAD。

### §5.2t 段 B 判读以 `json.load` 复核（0 GPU-h，非正则）

按 brief 的强制要求，两个探针文件的 `decision` 都**结构化读取**，并逐层展开：

```
segment_b_probe.json  → decision.dose_curve_status = "saturated"
                        decision.depth_candidates   = []
                        gpu_hours = 9.4348, scientific_claim=False, test_read=False
segment_b_depth.json  → decision.depth_candidates   = []
                        decision.dose_curve_status  = "untested"
                        gpu_hours = 3.3461, r = 0, scientific_claim=False, test_read=False
```

`depth` 文件里 `dose_curve_status = "untested"` **不是缺陷**：该文件测的是深度、不是剂量，
空值纪律（未测 ≠ 假）在这里正确生效。剂量判定只取 `segment_b_probe.json` 那一个。

**深度扫描逐档回读**（`depth_sweep.per_depth`，K=4 为参照，5 个 lead 齐全）：

| 深度 | 6h | 12h | 24h | 48h | 72h | 达 ≥0.005 K 更优的 lead 数 |
| --- | --- | --- | --- | --- | --- | --- |
| K=2 | **−0.01783** | **−0.02327** | −0.00110 | +0.02726 | +0.05594 | **2**（6h、12h） |
| K=4 | 0 | 0 | 0 | 0 | 0 | —（参照） |
| K=6 | +0.0376 | +0.05681 | +0.0515 | +0.05431 | +0.04385 | 0（全面更差） |
| K=8 | +0.12063 | +0.19116 | +0.23932 | +0.28838 | +0.30092 | 0（全面更差，且随 lead 放大） |

`depth_sweep.value = false`，`depth_candidates = []`。逐档 `n_clearing` 实测：K=2 **2**、
K=4 **0**、K=6 **0**、K=8 **0**。探针自带的判定规则逐字为：

> "a depth is a candidate only when it improves at least three leads by the declared floor
> against the trained K=4; a cheaper candidate is reported separately because a lower depth
> also lowers inference cost"

**独立复算**：直接从 `depth_tables` 的两个 RMSE 相减，与 `delta_vs_k4` 逐位相同
（如 lead 6：`1.84269 − 1.86052 = −0.017832`），且 `better = delta < 0`、
`clears_0.005 = delta ≤ −0.005` 的判定与 `n_clearing` 一致 ⇒ 读数没有被符号或阈值口径污染。

**判读（按 brief 的规则，不作美化）**：
- K=2 在 6h/12h 优于 K=4 超过 0.005 K，**但 48h/72h 反而更差**（+0.027、+0.056），
  故只清 2 个 lead，未达「≥3 个 lead」⇒ **不记为候选**。若把它写成「更便宜也更好」，
  就是 brief 明令禁止的「把更便宜但更差记成候选」。
- K=6/K=8 单调更差且 72h 放大到 +0.30 K ⇒ **加深不是提精度的手段**，K=4 是当前前沿的合理点。
- 因此本轮**没有**更优深度，段 A 沿用 `steps=4` 不变。

### §5.2u 登记的**两半**都用真实工具验过，不是等跑完再试

登记是最后一公里，且不可重来（`readings.json` 与 index 都拒绝覆盖/追加错行）。两半各自实测：

**① index record**。用**真实 digest 与真实 40 位 commit** 造了一条意图记录，直接喂仓内校验器：

```
evidence_path   = docs/goals/s3-typed-evidence-budget.md
evidence_sha256 = cc7a39204b1c502a…  （对该文件实际 sha256）
evidence_commit = ba85bf3c8793…      （当时 HEAD，完整 40 位，非零填充）
experiment_commit = f0da923796f2…    （冻结 commit）
protocol_sha256 = 76e1c214…  source_sha256 = 1db83955…  data_identity = 81307008…
→ tools/verify_r7_evidence_index.py --index …  ⇒  PASS evidence index: 1 records   exit 0
```

这一条同时排除了**已记录过的 fabrication 陷阱**：校验器会把「短 SHA 补零到 40 位」判为
fabricated commit id，故必须用 `git rev-parse HEAD` 的完整值。

**② 账本行**。用仓内真实正则 `LEDGER_ROW_RE` 匹配置图行：

```
| S3 typed evidence budget: … | 12.7809 | 632.217528 | `docs/goals/…md` (record:s3-…) |
→ matched: True；name/gpu/cum 三组均解析；RECORD_REF_RE 取出 record_id
```

**③ 累计算术**。`tools/r7_round.py ledger-plan 12.7809`：

```
rows_before 94, previous_total 619.436628, new_row 12.7809
→ new_total 632.217528, remaining_gpu_h −612.217528
```

`remaining_gpu_h` 只须等于 `cap_gpu_h − used_gpu_h`（`tools/check_campaign_state.py:173-178`），
**负值是既定事实而非错误**：起点就已 −599.436628（brief §2 明确「只记账，无总 GPU-h 上限」）。
故本轮登记不会因累计为负被拒。

⚠️ **段 B 的 12.7809 是已实测、尚未入账的**；段 A 的 GPU-h 只能等 receipt 出来再算
（`segment_a_hours()` 会累加训练 + 每 lead 评分秒数）。我另外用 `ledger-plan 112.0` 试算过一次
「若段 A 是某个整数会怎样」，那**只是算术演练，任何估算值都不得写进账本**。

### §5.2v 配对前提：两臂评测的是**同一组初始化**（不是同样多）

`paired_rmse_bootstrap` 的拒绝条件是 `set(ra) != set(rb)`，即要求**集合相同**；仅仅
`n_evaluated` 相等并不够（两个同样大的不同集合会被拒，且拒绝发生在 99 GPU-h 之后）。
用段 B 的真实 provenance 直接比集合：

| lead | K=2 | K=4 | K=6 | K=8 | 与 K=4 **集合完全相同** |
| --- | --- | --- | --- | --- | --- |
| 6h | 1458 | 1458 | 1458 | 1458 | **True** |
| 24h | 1455 | 1455 | 1455 | 1455 | **True** |
| 72h | 1447 | 1447 | 1447 | 1447 | **True** |

（其余 lead 12/48 的 `n_evaluated` 同为 1457/1451，见 `depth_tables`。）

这同时解释了 lead 之间的条数递减：长 lead 需要更多未来帧，可用窗口更少（1458→1447）。
**段 A 的三臂用同一 val manifest、同一 `max_samples`、同一 chronological selection**，
因此同样会落在相同集合上：冻结协议只有一个 `val_manifest`
（`…/store/manifests_treatment/val.jsonl`，`val_manifest_sha256 = f8133e79…`），
`_arm_contract` **不**让每臂各带一份 manifest——所以三个臂无法指向不同的评测集。
`_train_arm` 里 `EvaluationPreparation(...)` 只构建一次、被五个 lead 复用，也不引入分歧源。

⚠️ 一处措辞更正：`readings()["checks"]["val_manifest_sha256"]` 是**整轮的一个值**，
不是「三臂各一个值且相同」——它只能证明该轮声明的 manifest 未变，不能单独证明三臂同集。
三臂同集的真正保证来自「合约里只有一份 manifest」，以及 `_train_arm` 对每个 lead 都用
同一个 `prepared` 对象。

### §5.2w 区间在**单 lead** 形态上跑通（段 A 的真实形态）

段 A 的 `_train_arm` 对每个 lead 单独开目录（`lead_006h … lead_072h`），每个
`provenance.json` 的 `lead_hours` 是**单元素列表**。段 B 的 depth 扫描也是这个形态，故可直接实测：

```
a = update_0128528_k2/lead_012h/provenance.json   (lead_hours = [12], 17 channels)
b = update_0128528_k4/lead_012h/provenance.json
paired_rmse_bootstrap(a, b, block_days=7, replicates=1000, seed=42, confidence=.95)
→ rows = 17，leads present = [12]，n_initializations = 1457
→ t2m@12h: delta = −0.023272, interval = [−0.0296, −0.0171]  （排除零）
```

三点确认：
1. **17 行**（17 个变量）而不是更少 ⇒ `channels` 与 `units` 等长、无一被丢弃；
2. `pair_interval()` 用 `int(item["lead_hours"]) == int(lead)` 取行，实测该行**确实存在**；
3. `n_initializations = 1457` 与 lead 12 的 `n_evaluated` 一致 ⇒ 没有静默交集。

⇒ 判据 (a) 所需的区间在**段 A 的实际目录形态**上可算，不是只在「五 lead 合并」形态上可算。

### §5.2x 第十七处缺陷：(a) 假 + (b) 真 的措辞会与读数相反（**不修，记 errata**）

`clause_c()` 只有三个分支：两个都为真、primary（a）为真、以及**兜底**。当
**(a) 假而 (b) 真**时走兜底，输出：

```
permitted_claim = "no-increment-from-this-pathway"
forbidden_claim = "any positive claim about the typed-evidence pathway"
```

但 (b) 真**就是**「B − A 清了」——通路确有增量。于是禁止语与实测读数相反。
这正是 `da1ed00` 那一类（卡片与读数相反），只是换了个位置。

**这个组合可达，不是假想**：`A=2.30, B=2.28, C=2.29` ⇒ `B−A=−0.02`（b 真）、
`C−B=+0.01`（a 假），两个 RMSE 之间没有任何矛盾。brief §7 与计划 0034 的表格只列了三档
（(a)T(b)T、(a)T(b)F、(a)F(b)F），**恰好没列这一格**，所以它既没被文案覆盖，也没被测试覆盖
（`grep clause_c tests/` 只命中另一个轮的 `r7_balance_verdict`）。

**影响面（已逐条界定）**：
- `verdict` 字符串**不受影响**：它由 (a)/(b) 直接算出，此情形为
  `negative-no-type-structure-increment`，措辞准确（否定的是**类型结构**）。
- 受影响的是 `permitted_claim` / `forbidden_claim` 两个**文案字段**。
- 无任何外部代码消费这两字段（`grep permitted_claim` 只命中本模块与测试）。

**为什么不修**：`scripts/r7_typed_budget_verdict.py` 在 `execution_files_sha256` 里
（digest `b4bc4f7c…`）。改它 ⇒ `validate_protocol` 在 `reading` 阶段抛
`execution source changed`，即**全部 99 GPU-h 花完之后**才失败。

**⚠️ 我自己在这次排查中真的改了它一次，随后立刻回滚，如实记录**：
编辑后实测 `frozen b4bc4f7c… vs actual 859ef128…` 不匹配；`git checkout` 复原后重新核对
`MATCH: True`，并跑 `validate_protocol` 得 **PASS**、`protocol_sha256` 仍 `76e1c214…`、
10 个 worker 存活。**没有任何 GPU-h 损失**，因为失败点（reading 阶段）尚未到达。
这是第十四处的**原样重演**，说明「先查文件是否冻结、再动手」这一条在压力下仍会漏——
解决办法只有一条：**任何对 `scripts/`/`training/`/`model/` 的编辑前先查
`execution_files_sha256`**。

**本轮若真落到 (a)F+(b)T**，正确写法（人工，不靠代码字段）：写「通路（融合投影）对
incumbent 有增量，但按类型路由没有再进一步」；**不得**写「通路无增量」，也**不得**写
「类型路由带来增益」。该分支已作为**下一轮**的修复项登记，本轮不改冻结代码。

**回滚后的完整性复核（同一会话内实测，证明这次误改没有留下任何痕迹）**：

```
scripts/r7_s3_typed_budget_protocol.py --preflight-only
  f_preflight_passed: True
  all six clauses true: True
  parameters: incumbent 3286037 / generic 3434837 / typed 3434837
validate_protocol(attempt01) : PASS, protocol_sha256 = 76e1c214…
model_code_sha256            : 59340710623c…（= 冻结 pin）
存活 worker                   : 10
```

### §5.3 段 A 实跑合约核对（只读，取自 `fine_tune_started.json`）

第一个已启动的训练臂（`seed41/training/incumbent`）在**第一条 update 之前**写下的冻结合约，
逐字段核对（`json.load`）：

| 字段 | 实测值 | 与要求 |
| --- | --- | --- |
| `data_identity` | `81307008…335af0` | ✓ 与 sidecar pin 一致 |
| `source_sha256` | `1db83955…b32f5` | ✓ 与 `sha256sum source.nc` 实测一致 |
| `total_updates` | **32,132** | ✓ = 4.000 epochs @ batch 4 |
| `batch_size` | **4** | ✓ |
| `steps` / `lr` / `clip` | `4` / `2e-05` / `1.0` | ✓ 未扫、未改 |
| `autoregression.window_sha256` | `e8f22c30…` | ✓ long-rollout 发现（缺陷 8 的修正） |
| `autoregression.excluded_sample_ids` | **11** 条 | ✓ 同上 |
| `autoregression.supervision` | `{kind: full_grid, description: …}` | ✓ 完整两键（缺陷 10 的修正） |
| `autoregression.physical_weights` | `[1.0,0.5,0.0,0.5,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5]` | ✓ **逐字**沿用 |

`fine_tune_started.json` 在**任何 update 之前**落盘，满足 R-006「先冻结 protocol 再训练」。
臂顺序为 `incumbent → generic → typed`，每 seed 串行三臂，seed 41/42 并发于两卡。

### §5.4 本轮为什么在此暂停（会话边界，非科学结论）

按 brief §9 的三类停止条件，这里命中**第三类**：*确无可安全独立推进的工作*——
段 A 还需约 63.6 h 墙钟（两个 wave：wave1 在跑、wave2 尚未开始），而可做的零 GPU-h
前置工作已全部做完并留证（§5.1–§5.2t，共 20 节）。**不是**触授权边界，**不是**达科学停止条件。

**当前实测状态（`date -u` = 2026-11-16T21:43Z 时的读数，随训练前进）**：

| 项 | 实测 |
| --- | --- |
| `seed41/training/incumbent` | `update 4660 / 32132`（**14.5%**），loss 0.8779 |
| `seed42/training/incumbent` | `update 4680 / 32132`（**14.6%**），loss 1.2547 |
| seed43 | 未启动（`stage-seeds` 会在任一卡空出时启动，见 §5.2o） |
| `seed4{1,2,3}_receipt.json` | **三份都还不存在** |
| `readings.json` | 不存在（reading 阶段 `already_done: false`，未运行） |
| 已完成阶段 | `prepare` ✅、`archive` ✅（`check_round_receipts` `checked=2, failed=[]`） |
| 段 B | 42/42 剂量 + 20/20 深度，**12.7809 GPU-h**（9.4348 + 3.3461），**尚未入账** |
| 账本 | 94 行，`used_gpu_h 619.436628`，**本轮未加行** |
| `current_round_goal` | `docs/goals/s3-typed-evidence-budget.md` ✅ |
| 速率余量 | 1.1303 s/update ⇒ **+11.3%** 余量（§5.2g 复算） |

**未完成、且不在本会话可推进范围内的**（brief 交付物清单对照）：

| 交付物 | 状态 |
| --- | --- |
| D1 协议构建器含 (a)–(f) 预检 | ✅ 完成，`f_preflight_passed: true` 六条全真 |
| D2 三臂契约与容量匹配断言 | ✅ 完成（incumbent 3,286,037；generic ≡ typed 3,434,837，逐位相等） |
| D3 段 B 零训练探针 | ✅ 完成（42/42 + 20/20，`json.load` 判读见 §5.2t） |
| D4 冻结协议含 `protocol_sha256` | ✅ 完成（`76e1c214…`） |
| D5 `code.zip` 带 commit 注释 | ✅ 完成（注释 `f0da923`，sha256 `cc58c88c…`，与冻结逐位一致） |
| D6 逐 seed 逐臂 receipt | ⏳ **2/9**（incumbent×2 已落盘；generic×2 在跑；typed 必被截断） |
| D7 `readings.json` 五键 | ❌ **不可产出**（冻结守卫拒绝不完整读数，`complete=false` 即抛错） |
| D8 证据页 | ✅ **已写并提交** `docs/R7_S3_TYPED_EVIDENCE_BUDGET_RESULT.md` |
| D8b 四件套登记 | ⏳ **阻塞**：见下方「为什么现在不能登记」 |
| D9 判定点逐条写 (a)–(f) | ✅ **[a]=null, [b]=待定, [c]=null, [d]=null, [e]=待定, [f]=true**（证据页 §1） |
| D10 部分读数工具 | ✅ `tools/r7_s3_typed_budget_partial_reading.py`（190 行 + 2 条反证） |

**为什么现在不能登记（这是硬约束，不是我拖延）**：

1. `tools/verify_r7_evidence_index.py` 把 `evidence_sha256` 与证据页**硬绑定**——实测把
   sha 改一位即 `exit 1 / ERROR evidence_sha256 does not match`。而 seed43 结束后证据页
   **必然要改**（(b) 从「待定」变成真值），所以**此刻写的记录在 seed43 后一定失效**。
2. 账本行若引 `record:<id>`，`tools/check_campaign_state.py:203` 要求
   `metrics.gpu_hours` 与行内数值对齐 **1e-4**；而段 A 的 GPU-h 要等 seed43 结束才完整。
   现在写只能写 12.7809（段 B）+ 21.358（两个 incumbent），**缺 generic 与截断臂**，
   等于把不完整的成本记成完整。
3. `scripts/r7_s3_typed_budget_register.py` 的 `main()` **实测只打印装配结果、不写账本与
   index**（`inspect.getsource` 已确认），且它 `_read(readings.json)` —— 而该文件本轮不存在。

⇒ 登记的前置是 **seed43 完成 + 证据页定稿**。这不是可绕过的步骤，绕过去等于伪造机器核数。

**恢复的第一条命令**（不改判据、不重冻结、不新建 attempt）：

```bash
cd /data/esw/UrbanPiDiT_R2 && .venv/bin/python tools/r7_round.py state
```

然后按顺序（`stage-seeds` 若仍在跑则它会自己推进，**不要**再起第二个）：

```bash
# 1) 等三份 receipt（absence 即未完成；不要用中间 checkpoint 冒充）
ls outputs/r7_s3_typed_evidence_budget_20261116_attempt01/seed4{1,2,3}_receipt.json
# 2) 生成 readings.json（拒绝覆盖；不完整会拒写）
.venv/bin/python tools/r7_round.py stage reading
# 3) 用 json.load 读 (a)–(f)，(d) 必须为 null
.venv/bin/python -c "import json;b=json.load(open('outputs/r7_s3_typed_evidence_budget_20261116_attempt01/readings.json'));print(json.dumps(b['exits']['values'],indent=2));print(b['verdict'])"
# 4) 装配 GPU-h 与子句（段 B 12.7809 必须与段 A 同列一行）
.venv/bin/python scripts/r7_s3_typed_budget_register.py --attempt outputs/r7_s3_typed_evidence_budget_20261116_attempt01
```

**若 `stage-seeds` 已退出且 receipt 不齐**：先 `tools/r7_round.py orphans` 只读确认无存活 worker，
再 `stage-seeds` 续跑剩余 seed；**不得**删 `failed_attempt.json` 复活已失败的臂（runner 明确拒绝），
也**不得**跨 attempt 拼接读数。

**本轮到此没有、也不得有**：任何 (a)–(f) 的数值判定、任何 `readings.json`、任何登记记录。
段 B 的 `saturated` 与 `depth_candidates = []` 是**探针读数**，不是段 A 的判据结论。

---

### §5.5 第十八处缺陷：**每 seed 的 35 h 冻结包络装不下三臂**，typed 臂在三个 seed 上都不可能完成（2026-11-17T17:5xZ 实测）

这一节推翻了交接单给出的「(A)(B)(C) 三选一」框架：**三个选项的前提都不成立**。
瓶颈不是 seed43，而是**每 seed 的包络本身对三臂不够**，且这一点与是否启动 seed43 无关。

#### 根因：冻结的 `MEASURED_SECONDS_PER_UPDATE` 是 **incumbent 速率**，而 treated 臂慢 24–28%

冻结协议按 `per_seed = 3 × (32132 × 1.1536 + 2682.3) = 119,249 s = 33.12 h` 定下
`PER_SEED_SECONDS = 126,000 s = 35.00 h`（余量 5.6%）。该算式假设**三臂同速**。
实测不是：physical-evidence 通路给 generic/typed 臂带来真实的前向开销。

| 臂 | seed41 实测 s/upd | seed42 实测 s/upd | 相对 incumbent | 来源 |
| --- | ---: | ---: | ---: | --- |
| incumbent | **1.1162** | **1.1150** | — | `seed4{1,2}_incumbent_receipt.json` 的 `elapsed_seconds/32132` |
| generic | **1.3886** | **1.4266** | **+24.4% / +27.9%** | `seed4{1,2}/training/generic/progress.jsonl` 后半段增量口径 |

**这条开销读数本身是工程发现**（brief 要求记进证据页 limitations/成本段）：
物理证据通路（四场 patch 编码 + 融合投影）使前向慢 **24–28%**，与 §1.1 申报的
+148,800 参数（+4.5%）是两件事——参数只涨 4.5%，**时间涨 24–28%**，因为开销在
四场球面算子与逐 patch 编码上，不在参数量的线性项上。

#### 每 seed 三臂的真实成本与包络缺口（实测，非估算）

`incumbent` 用 receipt 实测（训练 + 五 lead 评分）；`generic/typed` 用 progress 增量速率，
评分按同比例放大（评分也要跑同一条前向通路）。

| seed | incumbent | generic | typed（代理=generic 速率） | 三臂合计 | 包络 35.00 h | 缺口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 10.690 h | 13.298 h | 13.298 h | **37.287 h** | 35.00 h | **+2.287 h** |
| 42 | 10.668 h | 13.648 h | 13.648 h | **37.964 h** | 35.00 h | **+2.964 h** |

**最有利界（反证用）**：即使 typed 臂按 **incumbent 速率**跑（不可能——它带 +24~28% 的通路开销，
且这是它的**下界**），seed41 仍需 10.690 h、seed42 需 10.668 h，而 generic 跑完后包络只剩
**10.619 h / 10.332 h** ⇒ 缺口仍为 **+0.072 h / +0.336 h**。

**⇒ typed 臂在这两个 seed 上无论如何都跑不完。** 不是「可能赶不上」，是**必然被
`_check_deadline` 截断**（`training/r7_scheduled_runner.py:140-142` 抛
`RuntimeError("scheduled runner deadline exceeded")`，runner 随即写 `failed_attempt.json`
并 `resume_permitted: False`，`training/r7_long_rollout_runner.py:588-598`）。

#### 为什么 (A)(B)(C) 都不产生**可发表的**读数

先更正本节初稿里的一个错误判断（我先写成「(b) 也算不出来」，冻结代码实测后推翻）：
**(b) 在算术上并不需要 typed 臂**——`B − A` 只用 generic 与 incumbent。用冻结
`r7_typed_budget_verdict.evaluate()` 做过代码路径探针（数值是占位的，**不是结果**）：

```
（3 seed 都有 inc+gen、typed 为空、无 provenance 目录）
b_generic_minus_incumbent → n_clearing = 5   （符号与增益门槛都过了）
b_generic_minus_incumbent → value = None     （只因区间算不出而 null）
(b) reason: the paired deltas clear the sign and gain floors, but no clearing lead's
            block-bootstrap interval could be computed, so the clause is untested
```

⇒ 在**真实**产物上（provenance 目录齐全）三 seed 的 inc+gen **足以让 (b) 给出真值**。
所以 (A) 的收益不是零。(A) 真正的阻塞点是**发表路径**，不是算术：

| 阻塞 | 位置 | 后果 |
| --- | --- | --- |
| `readings()` 逐臂硬取九份 receipt | `study_r7_s3_typed_budget.py:313` | 缺 `seed41_typed_receipt.json` 直接 `FileNotFoundError`，**整个 `readings.json` 产不出来** |
| `complete` 为假即拒写 | 同文件 `:391-395` | 即便绕过上一条，`missing` 非空 ⇒ 抛错、**不落 receipt** |
| brief 禁止手工拼表判读 | §2.2、Objective | `(b)` 只能由 `readings.json` 结构化读出；手拼的 inc+gen 表**不可采信** |

于是三个选项的真实收益：

| 选项 | 算得出什么 | 能发表什么 | 代价 |
| --- | --- | --- | --- |
| (A) 等卡后启动 seed43 | 3 seed 的 (b) 可算；**但 `readings.json` 仍写不出**（缺 3 份 typed） | **无**（手拼表不被允许） | **+35 GPU-h / +48 h 墙钟**（seed43 结束于 +48.26 h，仅剩 1.9 h 给 reading） |
| (B) 只跑 seed41/42 收尾 | 只有 2 seed ⇒ (b) 也是 `null` | 无 | 0 |
| (C) 两卡拆 seed43 的臂 | —— | —— | 需改 `_seed()` 串行结构，它在 `execution_files_sha256` 里 ⇒ **禁止** |

**(A) 花 35 GPU-h 换来一个不能发表的 (b)，且本轮主判据 (a) 仍是 `null`。** 这是本轮
不启动 seed43 的唯一理由——不是「没时间」，是**买不到可发表的东西**。

**并且 `readings()` 在真实 attempt 上直接抛错**（不是返回不完整表）：

```
driver.readings(attempt01, protocol) → FileNotFoundError: seed41_generic_receipt.json
```

（该文件在 seed41 的 generic 臂写 receipt 之前必然缺失；三臂跑完后才是 `complete`。）
配合 §5.2b 的「不完整即拒写」守卫 ⇒ **本轮无法产出 `readings.json`，因此无法产出任何
(a)–(f) 判定，也无法登记。**

#### 与 §5.2o「seed43 有 2.18 h 余量」的关系（那格是错的，此处更正）

§5.2o 算出「seed43 需 32.82 h、窗口 35.00 h、余 2.18 h」。那是用 **incumbent 速率 1.1536**
乘以三臂得到的——**与冻结协议同一个错误假设**。用 treated 臂实测速率重算，seed43 需
**37.29 h > 35.00 h**，**没有余量，反而缺 2.29 h**。§5.2o 的算术本身没算错，但它继承了一个
**被实测推翻的输入**。这一条现在写定，避免日后有人引用「2.18 h 余量」以为第三 seed 能跑完。

#### 墙钟可行性（两种排程口径，都不可行）

| 口径 | 需求 | 冻结上限 | 结论 |
| --- | ---: | ---: | --- |
| seed 串行（当前 driver 结构，2 卡两波） | 37.96 + 37.29 = **75.25 h** | 72.00 h | **不可行**（超 3.25 h） |
| 臂级打包（需改 driver 才可能） | 112.54/2 = **56.27 h** | 72.00 h | 可行但**需改冻结代码 ⇒ 禁止** |
| GPU-h 总量 | 112.54 | 144 | 不超（但 GPU-h 不是瓶颈） |

⇒ 即使不改代码，**seed 串行口径下总墙钟也超硬上限**。这不是余量不足，是结构性不可行。

#### 处置（本轮不做，记入下一轮前置）

1. **本轮不追加任何 GPU 运行**：typed 臂在任何 seed 上都跑不完，启动 seed43 只会多烧
   ~35 h 并同样留下一个被截断的 typed 臂，**不改变任何子句的值**。
2. **不延长 deadline、不改判据、不重冻结**（延长 = 改冻结协议，本轮全部产物作废）。
3. 让 seed41/42 把 generic 跑完（它们会自然结束），然后**如实**把本轮登记为
   `inconclusive` + `budget_limited`：段 B 的 12.7809 GPU-h + 段 A 实测的 inc/generic 六臂
   GPU-h 全额入账，`readings.json` **不产出**（冻结守卫拒绝不完整读数），(a)–(f) 全部
   `null`，证据页写清「typed 臂因每 seed 包络不足被截断，截断不算通过」。
4. **下一轮前置（必须在新 ADR 里改）**：`PER_SEED_SECONDS` 必须按**最慢臂**（treated，
   实测 1.39–1.43 s/upd）而非 `MEASURED_SECONDS_PER_UPDATE`（incumbent）来定；
   或把 `per_seed` 改成**每臂**锚定而非每 seed。这是设计缺陷，不是运行事故。

#### 为什么现有的防回归测试没拦住（它结构上是盲的）

`tests/test_r7_s3_typed_budget.py:380-399`
（`test_the_per_seed_envelope_exceeds_what_one_seed_actually_does`）正是为这一类错误写的，
但**它拦不住这一种**，因为它把**同一个错误常数**喂给断言：

```
per_arm = UPDATES * MEASURED_SECONDS_PER_UPDATE + MEASURED_SECONDS_PER_EVALUATION
        = 32132 * 1.1536 + 2682.3 = 39,749.8 s
assert PER_SEED_SECONDS > 3 * per_arm      # 126000 > 119,249.3 ✓ 余 +1.875 h
```

`MEASURED_SECONDS_PER_UPDATE = 1.1536` 是 **incumbent 速率**（`measured_on: batch 4 on the
p4000 rung`）。测试用 `proto.MEASURED_SECONDS_PER_UPDATE` 而不是独立测得的 treated 速率，
所以**冻结值与测试值同源** ⇒ 结构上恒绿，看不见 treated 臂慢 24–28% 这件事。

用实测速率喂进去立刻变红：

| 口径 | per_arm | ×3 arms | vs `PER_SEED_SECONDS` |
| --- | ---: | ---: | ---: |
| 冻结/测试同源（incumbent 1.1536） | 11.04 h | 33.13 h | **+1.875 h 通过** |
| 实测 generic/seed41（1.3886） | 13.30 h | 39.90 h | **−17,629 s 不足** |
| 实测 generic/seed42（1.4266） | 13.65 h | 40.95 h | **−21,400 s 不足** |

**教训（比缺陷本身更重要）**：一个断言如果**从被测对象自己的常数推导期望值**，它只能
防「常数被误改」，不能防「常数与现实不符」。每 seed 包络的守卫必须有一个**独立测得**的
treated 臂速率作为输入。下一轮改 `PER_SEED_SECONDS` 时必须同时补这条独立反证
（把 treated 速率写成 incumbent 速率，测试必须变红）。

#### 交接单里两条零 GPU 指令在本轮**不可执行**（已逐条实测，不是推断）

交接单第四节要求「修缺陷 17」与「按冻结提交重建 code.zip」。两条都会破坏本轮证据，故拒绝执行：

**① 缺陷 17（`clause_c` 的 (a)F+(b)T 措辞）——不能在本轮修。**

| 核查项 | 实测 |
| --- | --- |
| `scripts/r7_typed_budget_verdict.py` 是否在冻结清单 | **是**，`execution_files_sha256` 84 项之一 |
| 冻结 digest | `b4bc4f7c83bc4bc5…` |
| 磁盘现状 | `b4bc4f7c83bc4bc5…` ⇒ **MATCH** |
| `outputs/…_code.zip` 内副本 | `b4bc4f7c83bc4bc5…` ⇒ **MATCH** |
| `validate_protocol(attempt01)` | **PASS**，`protocol_sha256 76e1c214…` |

改它 ⇒ `reading` 阶段 `validate_protocol` 抛 `execution source changed`，即**在全部 GPU-h
花完之后失败**——§5.2d（第十四处）与 §5.2x（第十七处）已各自原样发生过一次。
交接单要求的「补一条反证测试」也可行性为零：`docs/rules/EVIDENCE.md:54,171` 记录本仓
**0 处 xfail**（E-027/E-064），所以只能写成 xfail 的「已知缺陷测试」按规约不允许存在。
**⇒ 缺陷 17 保持 errata 状态（§5.2x 已完整记录），修在下一轮，与本轮冻结解耦。**

**② code.zip「按冻结提交重建」——它已经就是按冻结提交建的，重建反而会破坏身份。**

| 核查项 | 实测 |
| --- | --- |
| zip comment | `f0da923796f2eeb74b7c53e3e95b7876b80ccbb8` |
| `protocol.json` 的 `code_commit` | 同值 ⇒ **一致** |
| zip 实测 sha256 | `cc58c88c…` |
| 冻结 `code_zip_sha256` | `cc58c88c…` ⇒ **一致** |
| 84 个冻结源在 zip 内的副本 | **全部 MATCH**（无一项漂移） |

交接单说「与 HEAD 不符」——**这是对的，但不是缺陷**：HEAD 已前进到 `df949ae`（文档提交），
而归档只须匹配**冻结身份** `f0da923`，不需要匹配 HEAD（§5.2s 已写定）。
重建到 HEAD 会让 `code_zip_sha256` 与 `protocol.json` 脱钩，`_archive` 阶段会直接拒收。
**⇒ 那条 skip 是预期行为，不是待修项。** 已复跑确认：37 passed / 1 skipped，
skip 理由逐字为 `the archive is stale: built from f0da923796f2, HEAD is df949ae559ab`。

#### 第十九处缺陷（本轮新发现，0 GPU-h，**不改本轮代码**）：装配器会把两个身份字段读成 `null`

`scripts/r7_s3_typed_budget_register.build_record()` 从 `readings` 里取三个身份字段：

```python
"protocol_sha256": readings.get("protocol_sha256"),
"source_sha256":   readings.get("source_sha256"),      # ← readings() 从不写这个键
"data_identity":   readings.get("data_identity"),      # ← readings() 也不写这个键
```

而 `study_r7_s3_typed_budget.readings()` 的 18 个顶层键里**没有** `source_sha256` 和
`data_identity`（实测：`inspect.getsource` 抽出键名比对）。于是 `build_record` 会**静默**
产出 `{"source_sha256": null, "data_identity": null}`。

**它现在还没炸，是因为 `build_record` 目前没有任何调用方**（`grep -rn build_record` 只命中
定义处）：`main()` 只把 `segment_b/segment_a/clauses` 打到 stdout 供人读，不生成 index 记录。
但一旦下一轮把它接进登记流程，`tools/verify_r7_evidence_index.py` 的行为是：

- `candidate_state != "candidate"` 时：`_digest_or_none(None)` **通过** ⇒ **静默放行**（已实测：
  造一条 `source_sha256=null/data_identity=null` 的记录喂真校验器，`exit 0 / PASS 1 records`）
- `candidate_state == "candidate"` 时：`for key in ("protocol_sha256","data_identity")` ⇒ 只有
  `data_identity` 会被拦，`source_sha256` **仍然漏过**

⇒ 一条声称 `negative`/`audit` 的记录可以带着**空身份**通过校验。这不是本轮的判据问题
（本轮产不出 `readings.json`，也登记不了），是**下一轮接登记前必须补的守卫**。
正确做法：`readings()` 补写 `source_sha256`/`data_identity`，或让 `build_record` 在缺键时
`raise` 而不是 `.get()`。两者都改 `scripts/`，**本轮冻结内不得动**。

#### 第二十处缺陷（本轮新发现，0 GPU-h）：被截断的臂不写 receipt ⇒ 账本会系统性漏记它的 GPU-h

`segment_a_hours()` 只 glob `seed*_*_receipt.json`（`scripts/r7_s3_typed_budget_register.py:62`）。
而被 `_check_deadline` 截断的臂走 `except BaseException` 分支，**只写 `failed_attempt.json`，
不写 receipt**（`training/r7_long_rollout_runner.py:588-598`；`training_report.json` 只在
成功路径写）。⇒ 被截断的 typed 臂**烧掉的真实 GPU 时间不会进账本**。

这正是 AGENTS.md「不得用估计值、不得漏记账」的对立面——不是记账出错，是**记账器看不见它**。

**补算通路（存在且是实测值）**：`progress.jsonl` 每条含 `elapsed_seconds`（实测键：
`elapsed_seconds, epoch, gradient_norm, l12, l6, loss, lr, recorded_at, update`），
所以被截断臂的真实秒数 = 该臂 `progress.jsonl` 最后一条的 `elapsed_seconds`。
登记时必须把这一项**显式加进账本**并在证据页写明「该值取自 `progress.jsonl` 而非 receipt，
因为截断臂不产出 receipt」。**不得**因为 `segment_a_hours()` 没读到就记 0。

#### 本轮的成本敞口（含一笔无法避免的浪费，如实记）

单位一致口径（秒→小时），数字标注为**外推**（基于实测速率，不是实测值）：

| 项 | GPU-h | 性质 |
| --- | ---: | --- |
| 段 B（B-1 + B-2） | **12.7809** | **实测**（9.4348 + 3.3461） |
| 段 A incumbent ×2 | **21.358** | **实测**（receipt 的 `elapsed_seconds` + 五 lead 评分） |
| 段 A generic ×2（将完成） | ≈ 27.0 | 外推（实测速率 × 剩余 update + 按比例评分） |
| 段 A typed ×2（**必被截断**） | ≈ **20.9** | 外推；**不可回收** |
| **本轮合计** | ≈ **59.5** | 外推 |

**这 20.9 GPU-h 是必输的**：typed 臂一旦启动就会跑到包络尽头才抛
`scheduled runner deadline exceeded`，而 R-054 禁止对非本实验进程发信号、
brief §5 也禁止实现「冻结/终止自动化」。**我不会为了省钱去 kill 它们**——
那是本轮明令禁止的动作，且会破坏 `failed_attempt.json` 的诚实记录。

**能省的只有 seed43 那约 37 h**（三个 seed 都缺 typed ⇒ 第三份 `null`，无可发表收益）。
这是本轮唯一采取的止损动作。

#### 更正（本节初稿有一处实质错误，用冻结代码复核后推翻）

初稿写「(A) 花 35 GPU-h 换来一个不能发表的 (b)，所以不启动 seed43」。**前半对、后半错**：

- **对的部分**：主判据 (a) 确实是 `null`，这一点没变。
- **错的部分**：我把 (b) 也说成买不到。**(b) 只用 `generic − incumbent`，不需要 typed 臂。**
  三个 seed 的 inc+gen 齐全时，`_clause_pair` 能算出真值（符号 + 增益门槛 + 区间三者都只需要这两臂）。
  初稿的探针只喂了"无 provenance 目录"的占位表，于是区间算不出、`(b)` 塌成 `null`——
  我把**探针的缺陷**误读成了判据的性质。

**更正后的 (A) 核算（实测速率，秒/小时单位已对齐）**：

| 项 | 值 |
| --- | --- |
| seed43 拿到包络 | `min(round 剩余 36.92 h, per_seed 35.00 h)` = **35.00 h** |
| seed43 incumbent + generic 需求（两 seed 较差速率） | 10.69 + 13.65 = **24.34 h** ⇒ 余 **+10.66 h** |
| inc+gen 完成时刻 | round 剩余 **12.57 h**（不必等 typed 截断就能算 (b)） |
| seed43 总烧 | 24.3 h（有效）+ 10.7 h（typed 空转）= **35.0 h** |

⇒ **(A) 在时间上可行，且能给出 (b) 的三 seed 实测真值。** 这是本轮唯一还能拿到的科学产出。

#### 但 (A) 有一个刚发现的硬阻塞：重启 `stage-seeds` 会把已完成的 41/42 整体重跑

`stage_receipt(output, "seed", seed)` 找的是 **`seed{seed}_receipt.json`**，而该文件**只在
`_seed()` 三臂全部成功后**才写（`study_r7_s3_typed_budget.py:400`）。typed 臂必败 ⇒
`seed41_receipt.json` / `seed42_receipt.json` **永远不会出现**（实测 `stage_plan` 当前
`skip: False, already_running: True`）。因此：

- 若现有 `stage-seeds` 进程在 typed 失败后置 `stop=True` 退出（`r7_round_stages.py:440-441`），
  再手动重启 `stage-seeds`，它会把 seed41/42 视为**没跑过**，从父 checkpoint **整 seed 重跑**
  三臂 ⇒ **白烧 21.4 GPU-h 已完成的 incumbent+generic**，且 runner 明确不支持续跑
  （`resume_permitted: False`，§5.2i）。
- 所以 **(A) 不能靠重启 `stage-seeds` 实现**；必须用一条**只跑 seed43 单 seed** 的调用，
  且该调用必须绕过 `stage-seeds` 的"pending 由 receipt 决定"逻辑。

**这一条是本轮真正的操作风险，也是我不在本次会话内启动 seed43 的原因**：可行路径需要
一个不重跑 41/42 的启动方式，而现有工具（`stage-seeds` / `stage seed <n>`）都按 seed 级
receipt 判存在。**在有明确的单-seed 启动方式并经只读验证之前，启动 seed43 会触发 41/42 重跑。**

#### 第二十一处缺陷（本轮新发现，0 GPU-h）：`_clause_pair` 在空 seed 列表上除零

写部分读数工具时实测踩到：

```
verdict._clause_pair({incumbent:{},generic:{},typed:{}}, [], [6,12,24,48,72], "t2m", ...)
→ ZeroDivisionError: division by zero
```

原因在 `scripts/r7_typed_budget_verdict.py:128-129`：`mean = sum(measured) / len(measured)`，
当 `have_b == []` 时 `len(measured) == 0`。

**在冻结的 `evaluate()` 里不会触发**——它先检查 `seeds`/`leads` 非空，且九臂齐全时
`have_b` 恒为三个 seed。**但任何部分读数调用方都会踩到**，而这正是本轮唯一还能拿到读数的
路径。新增的 `tools/r7_s3_typed_budget_partial_reading.py` 已加空集守卫（返回
`value: None` + 理由，符合 null 纪律），且**不修改冻结模块**（它不能改，见 §5.5①）。

#### 本轮的执行决定：采用 **(A)**，seed43 待卡空出后以 `stage seed 43` 启动

复核后确认交接单的推荐是**对的**，我草稿里「不启动 seed43」的结论建立在两个错误上
（把 (b) 误判为需要 typed 臂；把探针缺陷误读为判据性质），两处都已在上面更正。

**决定：启动 seed43，只跑它，用 `stage seed 43` 而非 `stage-seeds`。**

理由是 `stage seed <n>` 按 seed43 自己的 receipt 判存在（实测 `already_done: false`,
`already_running: false`, `skip: false`），**不会触碰 seed41/42**；而 `stage-seeds` 会把
41/42 视为未跑过（它们的 seed 级 receipt 因 typed 必败而永不出现）⇒ 白烧 21.4 GPU-h 重跑。
这一条是本次决策里最关键的技术判断。

**预算核算（外推，基于实测速率；不是实测值）**：

| 项 | GPU-h | 说明 |
| --- | ---: | --- |
| 段 B | 12.7809 | 实测 |
| 段 A seed41/42 | ≈ 71.1 | inc 21.4 实测 + gen 27.0 + **typed 空转 22.7** |
| 段 A seed43 | ≈ 35.0 | inc 10.7 + gen 13.7 + **typed 空转 10.6** |
| **本轮合计** | **≈ 118.9** | 软 96 / 硬 144 |

⇒ **超软预算约 22.9 GPU-h（+24%），但在硬上限内。** 按决策 0030，软预算超支**记 overrun
继续**、不截断；账本按实测登记，不得写成估算值。

**必须如实申报的成本**：typed 臂的空转总量 **≈ 33.4 GPU-h** 换不到任何子句——它在三个 seed
上都会被 `_check_deadline` 截断。这部分**不可回收**（R-054 禁对进程发信号，brief §5 禁终止
自动化，且我会刻意不实现任何冻结/终止逻辑）。

**seed43 的收益与不收益**（不得夸大）：

| 子句 | seed43 后 |
| --- | --- |
| **(b)** `generic − incumbent` | **三 seed 齐全 ⇒ 可判真值**（可用新增部分读数工具从逐臂 receipt 算出） |
| (a) `typed − generic` | **`null`**（typed 无 receipt ⇒ 未测，不是假） |
| (c) 声明范围 | **`null`**（依赖 (a)） |
| (d) 4 vs 16 epoch | **`null`**（无 16-epoch 读数） |
| (e) z500@72h 护栏 | **三个 seed 都可判**（只依赖 inc+gen） |

⇒ 本轮的最终结论形态是：**(b) 有真值，(a) 为 null，verdict 记 `untested-no-arm-contrast`
或按 (b) 单独报告**。**绝不**把 (b) 为真写成「typed evidence 有类型结构贡献」——
那需要 (a)，而 (a) 本轮未测。

#### 启动条件与命令（卡空出后，约 12.3 h）

seed41/42 的 generic 完成后会**立刻**启动各自的 typed 臂并跑到包络尽头，所以卡要到
`pop + 126,000 s` 才空出（当时实测距启动约 **12.3 h**）。到点后执行：

```bash
cd /data/esw/UrbanPiDiT_R2
# 0) 只读确认 seed43 可启动且 41/42 已退出
.venv/bin/python tools/r7_round.py stage-plan seed 43
.venv/bin/python tools/r7_round.py orphans
# 1) 只启动 seed43（不要用 stage-seeds：那会重跑 41/42）
.venv/bin/python tools/r7_round.py stage seed 43
# 2) 约 24.4 h 后 inc+gen 完成；用新增工具取 (b)（不写 readings.json，不改冻结源）
.venv/bin/python tools/r7_s3_typed_budget_partial_reading.py \
    --attempt outputs/r7_s3_typed_evidence_budget_20261116_attempt01
```

**不要**在卡被占时就调用 `stage seed 43`：`run_stage` 遇 `headroom-blocked` **立即返回**
（只读等待是 `run_seeds` 的行为，单 stage 没有循环）。已实测当前两卡 gate 均为
`passed=False`（`required_bytes 16.0 GiB` vs `free_bytes 13.6 GiB`），所以此刻调用会被安全拒绝、
**不会双开**；但也没有意义，等它真能通过时再调。

**一条必须点明的风险**：`run_stage` 的 gate 是**逐卡余量**判定，**不看 `run_seeds` 的
`free` 列表**。也就是说，只要某张卡的空闲够 16.0 GiB，`stage seed 43` 就会在**那张卡上**
再起一个 worker——即使该卡已有本轮的 worker 在跑。当前不会发生（余量不够），但**卡空出后
必须确认目标是真空卡**，而不是"余量够了的共驻卡"。判据：`orphans` 无 41/42 的存活 PID。

#### seed43 的启动窗口很窄（1.92 h），但**会话内等待是允许的**

| 项 | 时刻（相对首次记录时，`perf_counter` 差值） |
| --- | --- |
| 卡空出（41/42 的 typed 臂跑到各自包络尽头） | **+12.28 h** |
| 完整 35 h 包络的**最后**启动时刻（`round_deadline − per_seed`） | **+14.20 h** |
| **⇒ 完整包络的启动窗口** | **+12.28 … +14.20 h，仅 1.92 h 宽** |
| 只要 (b)（inc+gen 24.37 h）的最后启动时刻 | +24.83 h（窗口 12.55 h） |

**⚠️ 更正（我先前的判断是错的）**：我曾写「会话内无法跨越这 12 h，所以本轮停止」。
这是把**会话内阻塞等待**误当成了 §8 禁止的行为。§8 禁止的是
`cron` / 守护 / **会话外续跑**——即在会话结束后仍自动执行。在本会话内用轮询**阻塞等待**
并不被禁止，而且本轮已经这样推进过（`19 × 30 s` 的只读轮询循环，累计推进了约 2 h，
见 §5.5 的等待记录）。因此 seed43 **应当在本会话内等到窗口、然后启动**，而不是停止。

**仍然禁止的是**：写一个 12 h sleeper 脚本在会话**之外**替会话执行 stage。我写过并
**自我否决、已删除**（那才是被禁的形态：它等待一个绝对时刻、然后替会话继续跑，
本质是会话外续跑 + 守护）。**判定线**：命令在本轮对话里阻塞等待 ⇒ 允许；
进程在对话结束后仍活着并自主推进 ⇒ 禁止。

#### 本轮的执行决定（先写定，避免既成事实后补理由）

| 决定 | 内容 | 依据 |
| --- | --- | --- |
| **不启动 seed43** | 三个 seed 的 typed 臂都必然被包络截断，seed43 的 35 h 换来的是第三份 `null` | 本节算术；§2「任一臂缺失 ⇒ `null`」 |
| **不延长 deadline** | 延长 = 改冻结协议 ⇒ 本轮全部产物作废 | brief §5(A) 尾注、§3 |
| **不重冻结、不改冻结源** | `scripts/**` 在 `execution_files_sha256` 内，改则 reading 阶段必失败 | §5.2d、§5.2x、本节①② |
| **不向任何进程发信号** | 含本轮自己的两个 worker：R-054 禁止终止自动化，`stage-seeds` 会让它们在冻结包络上自己退出 | R-054、决策 0030 |
| **seed43 不会被自动拉起** | `run_seeds` 在任一 seed `passed=False` 时置 `stop=True`（`tools/r7_round_stages.py:440-441`），seed41/42 的 typed 臂失败后循环退出，seed43 留在 `pending` 不启动 | 代码实测 |
| **收尾登记为 `inconclusive` + `budget_limited`** | 段 B 12.7809 + 段 A 实测 GPU-h 全额入账；`readings.json` 不产出；(a)–(f) 全 `null` | §2、`readings()` 的 completeness 守卫 |

**预期终态**：seed41/42 在各自 `pop + 126,000 s` 处抛
`RuntimeError: scheduled runner deadline exceeded`，runner 写 `failed_attempt.json`
（`resume_permitted: False`），`stage-seeds` 记录 `failed` 并停止。两份 `generic` 逐臂
receipt 会先落盘（generic 臂约 2.3 h 后完成），它们是**本轮唯一完整的 treated 臂五 lead 读数**。

#### 复现命令（0 GPU-h）
```bash
cd /data/esw/UrbanPiDiT_R2 && .venv/bin/python - <<'PY'
import json, time
att="outputs/r7_s3_typed_evidence_budget_20261116_attempt01"
p=json.load(open(f"{att}/protocol.json")); UPD=int(p["updates"]); PER=p["per_seed_seconds"]
now=time.perf_counter(); print("envelope_h", PER/3600, "round_remaining_h", (p["deadline_perf_counter"]-now)/3600)
for s,lim in ((41,3300669.726970281),(42,3300669.76412676)):
    inc=json.load(open(f"{att}/seed{s}_incumbent_receipt.json"))
    inc_s=inc["elapsed_seconds"]+sum(v["elapsed_seconds"] for v in inc["evaluations"].values())
    r_inc=inc["elapsed_seconds"]/UPD
    rows=[json.loads(l) for l in open(f"{att}/seed{s}/training/generic/progress.jsonl") if l.strip()]
    u,e=rows[-1]["update"],rows[-1]["elapsed_seconds"]
    u0,e0=rows[len(rows)//2]["update"],rows[len(rows)//2]["elapsed_seconds"]
    r=(e-e0)/(u-u0)
    ev=sum(v["elapsed_seconds"] for v in inc["evaluations"].values())*(r/r_inc)
    print(f"seed{s} r_inc={r_inc:.4f} r_gen={r:.4f} overhead=+{(r/r_inc-1)*100:.1f}%")
    print(f"   inc={inc_s/3600:.3f}h gen={((UPD)*r+ev)/3600:.3f}h typ(same)={(UPD*r+ev)/3600:.3f}h"
          f" total={(inc_s+2*(UPD*r+ev))/3600:.3f}h vs {PER/3600:.2f}h")
    print(f"   floor(typed at incumbent rate)={inc_s/3600:.3f}h; envelope left after generic="
          f"{(lim-now-((UPD-u)*r+ev))/3600:.3f}h")
PY
```

#### 本轮此时的状态（不写结论）

| 项 | 实测 |
| --- | --- |
| round 剩余墙钟 | 50.18 h（`deadline_perf_counter - perf_counter()`） |
| 每 seed 包络剩余 | 13.26 h（`pop + 126000 - now`），**这才是紧的界** |
| seed41/42 generic 进度 | 27,6xx / 27,0xx of 32,132；预计 2.6–2.9 h 后完成 |
| typed 臂 | **0/3 启动，且按实测速率不可能完成** |
| `readings.json` | 不产出（冻结守卫拒绝不完整读数） |
| (a)–(f) | **全部 `null`**（未测，不是假） |
| 段 B GPU-h | **12.7809**（9.4348 + 3.3461）实测，尚未入账 |
| 段 A GPU-h | 六臂实测值将在 seed41/42 退出后从 receipt 求和入账 |

**这不是科学结论，是运行约束的实测记录。** typed evidence 在 4 epoch 剂量下的可归因贡献
**本轮未测**，原因是预算包络装不下，**不是机制无效**。

### §5.7 登记已完成，剩余只有 seed43（2026-11-17T21:0xZ 交接）

**本轮已登记**（不是全部完成，是「已收尾部分」）：

| 项 | 值 |
| --- | --- |
| 索引记录 | `record:s3-typed-evidence-budget-partial`，`outcome_class: unresolved`、`candidate_state: not-candidate`、`metrics.gpu_hours 61.1668` |
| 账本行 | **61.1668 GPU-h**（= 段 B 12.7809 实测 + 段 A 4 臂 48.3859 实测），累计 **680.603428** |
| campaign-state | `used_gpu_h 680.603428`、`remaining −660.603428`（`check_campaign_state` `failures=0`） |
| 证据页 | `docs/R7_S3_TYPED_EVIDENCE_BUDGET_RESULT.md`（sha256 `af87c971…`） |
| 索引/brief 校验 | `PASS evidence index and brief: 114 records` |
| 提交 | `9ceb350`（已推 `r7/weather-reasoning`） |

**本轮已到手的实测读数（两 seed，不止是过程证据）**：

| lead | Δ(generic−incumbent) seed41 | seed42 | 均值 |
| ---: | ---: | ---: | ---: |
| 6h | −0.000850 | −0.001533 | −0.001192 |
| 12h | −0.000656 | −0.002099 | −0.001378 |
| 24h | −0.006009 | −0.001912 | −0.003961 |
| 48h | −0.007070 | *+0.000613* | −0.003228 |
| 72h | −0.004520 | *+0.002942* | −0.000789 |

⇒ `concordant_leads=[6,12,24]`、**`n_clearing=0`**（无一达 −0.005 K 门槛）⇒ **(b) 在两 seed 上为 `False`**。
(e) 两 seed 均过（z500@72h **−0.41% / −0.49%**，实为改善）。

**仍未完成、且只差 seed43 的**：(b) 的三 seed 终值。冻结定义要三 seed 一致，
**不得**用两 seed 的 `False` 冒充终值，也不得因「反正不过门槛」就省掉种子——那是事后改判据。

### §5.9 本轮读数完成并已登记（2026-11-19T08:4xZ）

**seed43 已启动并跑完 inc+gen**：`stage seed 43` 于窗口内（+9.94 h 后）执行，
`seed43_incumbent_receipt.json`（10.9555 GPU-h，`new_parameters=0`）与
`seed43_generic_receipt.json`（13.5677 GPU-h）均已落盘，5 lead 齐全。
**41/42 的 typed 进度与 receipt 未被触碰**（未用 `stage-seeds`，避免了 21.4 GPU-h 的整 seed 重跑）。

**三个 seed 的 (b) 终值 = `False`**（`n_concordant=3`、**`n_clearing=0`**）：

| lead | seed41 | seed42 | seed43 | 均值 |
| ---: | ---: | ---: | ---: | ---: |
| 6h | −0.000850 | −0.001533 | −0.003272 | **−0.001885** |
| 12h | −0.000656 | −0.002099 | −0.005594 | **−0.002783** |
| 24h | −0.006009 | −0.001912 | −0.000147 | **−0.002690** |
| 48h | −0.007070 | *+0.000613* | −0.003059 | **−0.003172** |
| 72h | −0.004520 | *+0.002942* | −0.003188 | **−0.001589** |

⇒ 五个 lead 的均值全部在 **−0.00317…−0.00159**，**无一达预声明的 −0.005 K 门槛**；
15 个 seed×lead 单元中 13 个为负（方向偏向 generic，幅度系统性低于门槛）。**门槛未调低。**

**(a)/(c)/(d) = `null`**（typed 臂 0/3）。**(e) = `null`** 但三 seed 单臂数字全过
（z500@72h：−0.4067% / −0.4876% / −0.5230%，实为改善）——`clause_e` 遍历 `(GENERIC, TYPED)`，
typed 缺失即 `None`（实测）。

**已登记**：

| 项 | 值 |
| --- | --- |
| 索引记录 | `record:s3-typed-evidence-budget-three-seed`（`mixed` / `not-candidate` / `106.2251`），共 **115** 条 |
| 账本行 | **106.2251 GPU-h**（段 B 12.7809 + 段 A 93.4442），累计 **786.828528** |
| campaign-state | `used_gpu_h 786.828528`、`remaining −766.828528`（`failures=0`） |
| 证据页 | `docs/R7_S3_TYPED_EVIDENCE_BUDGET_RESULT.md`（sha256 `cfb8d169…`） |
| 校验 | `PASS evidence index and brief: 115 records`；`r7_round.py verify` **passed: True** |

**成本（全部实测）**：段 A 93.4442 = 6 份 receipt **72.9091**（inc×3 + gen×3）+ 截断 typed 臂
**20.5351**（无 receipt，取各 `progress.jsonl` 末条：seed41 10.2747 / seed42 10.1115 /
seed43 0.1490）。**seed43 的 typed 臂在登记时仍在跑，故 20.5351 是其下界。**
超软预算 96 约 **10.2**（记 overrun），在硬 144 内。

**仍未完成（本轮确定不做的）**：typed 臂在三个 seed 上都会跑到各自包络尽头被截断；
按 R-054 不发信号，其最终 GPU-h 需在进程自行退出后再补一次入账（增量部分）。

**下一项任务**：等 seed43 typed 臂自行退出后，用
`.venv/bin/python - <<'PY'` 的 `progress.jsonl` 末条补算其最终秒数，追加一行账本增量
（**不重写** `f1ad6ce` 那一行），并同步 `metrics.gpu_hours`。

| 项 | 实测（`perf_counter` 差值） |
| --- | --- |
| 卡空出 | **+9.94 h** |
| 完整 35 h 包络的最后启动时刻 | **+11.86 h** |
| **可用窗口** | **1.92 h 宽**，从 +9.94 h 开始 |
| 现状 | seed41/42 的 typed 臂在跑（1,200 / 800 of 32,132 updates），将各自跑到包络尽头被截断 |

**⚠️ 更正**：这里原先写「命中第三类停止条件，因为会话内无法跨越 10 h」。**判断有误**——
会话内阻塞等待是允许的，被禁的是 cron/守护/会话外续跑（见 §5.5 末的判定线）。
所以本轮**继续在本会话内等到窗口并启动 seed43**。

**待执行的启动步骤**（窗口内，约 +9.94 h 后）：

```bash
cd /data/esw/UrbanPiDiT_R2 && .venv/bin/python tools/r7_round.py stage-plan seed 43
```

确认 `already_running:false` 且 `skip:false` 后：

```bash
.venv/bin/python tools/r7_round.py stage seed 43
```

⚠️ **绝不用 `stage-seeds`**：它按 `seed{N}_receipt.json` 判存在，而该文件要三臂全成功才写；
typed 必败 ⇒ 41/42 会被视为未跑过而**整 seed 重跑**，白烧 21.4 GPU-h。

seed43 的 inc+gen 完成（约 24.4 h 后）即可取三 seed (b)：

```bash
.venv/bin/python tools/r7_s3_typed_budget_partial_reading.py \
    --attempt outputs/r7_s3_typed_evidence_budget_20261116_attempt01
```

届时须**追加**一条新索引记录（证据页会变、旧记录的 `evidence_sha256` 会失配，实测会
`ERROR evidence_sha256 does not match`）并补一行账本，**不得**改写已提交的 `9ceb350` 记录。

按 brief §9，这里命中**第二类 + 第三类**：

- **第二类（已冻结判据下的确定终局）**：typed 臂在三个 seed 上都被每 seed 的 35 h 包络截断
  （§5.5），所以 **(a) = (c) = (d) = `null`** 是冻结规则的确定结论，不是待跑。
- **第三类（确无可安全独立推进的工作）**：零 GPU-h 前置已全部做完（§5.1–§5.5，21 处缺陷）；
  下一步是启动 seed43，而它的窗口是 **+12.28 … +14.20 h（1.92 h 宽）**，
  **会话内无法跨越，且 cron/守护/会话外续跑被明令禁止**（§5.5 末）。
  我写过 12 h sleeper 启动器并**自我否决、已删除**，不以合规包装违规。

**不是**触授权边界，不申请新授权。

#### 决策已定案：(A)

| 项 | 决定 |
| --- | --- |
| seed43 | **启动**，用 `stage seed 43`（**不是** `stage-seeds`，那会重跑 41/42 共 21.4 GPU-h） |
| 窗口 | 卡空出 +12.28 h 后的 **1.92 h** 内（最迟 +24.83 h，但那时 typed 会更早截断） |
| 收益 | **(b) 三 seed 真值**；**(a)(c)(d) 全 `null`**；(e) 可判 |
| 成本 | 段 A 累计 ≈ **106.1 GPU-h**，本轮 ≈ **118.9**；超软 96 约 22.9（记 overrun），在硬 144 内 |
| typed 空转 | ≈ **33.4 GPU-h** 换不到任何子句，不可回收（R-054 禁终止） |
| 本轮终局 | `verdict` 记 `untested-no-arm-contrast`；(b) 单独按实测报；**绝不**写成"类型结构有贡献" |

#### 恢复时的第一条命令（卡空出后的 1.92 h 窗口内）

```bash
cd /data/esw/UrbanPiDiT_R2 && .venv/bin/python tools/r7_round.py stage-plan seed 43
```

确认 `already_running:false` 且 `skip:false` 后：

```bash
.venv/bin/python tools/r7_round.py stage seed 43
```

seed43 的 inc+gen 完成（约 24.4 h 后）即可取 (b)，**不必等 typed 截断**：

```bash
.venv/bin/python tools/r7_s3_typed_budget_partial_reading.py \
    --attempt outputs/r7_s3_typed_evidence_budget_20261116_attempt01
```

#### 登记（seed43 结束后，或决定放弃 seed43 时）

```bash
# 汇总实测 GPU-h：receipt 可见部分 + 截断臂的 progress.jsonl 末条（第二十处缺陷的补算）
# outcome_class=unresolved / candidate_state=non-candidate
# limitations 必写：epochs 字段 errata（§5.2l）、包络缺陷（§5.5）、
#                   截断臂不写 receipt（§5.5 第二十处）、typed 臂未测
.venv/bin/python scripts/r7_s3_typed_budget_register.py --attempt outputs/r7_s3_typed_evidence_budget_20261116_attempt01
```

⚠️ 登记前必须先补 §5.5 第十九处（装配器把 `source_sha256`/`data_identity` 读成 `null`），
否则 index 记录会带空身份通过校验（已实测 `exit 0`）。

#### 本轮到此没有、也不得有

- 任何 (a)–(f) 的**捏造**判定：(a)(c)(d) 是 `null`（未测），(b)(e) 只有在 inc+gen 齐全的
  seed 上才是真值
- `readings.json`（冻结守卫拒绝不完整读数，这是**正确行为**；部分读数走新工具的独立输出）
- 任何 `scientific_claim: true`；任何「typed evidence 有效/无效」的结论
- 任何对存活进程的信号、任何终止/冻结自动化、任何 cron/守护/会话外续跑
- 任何把 (b) 为真读成"物理类型结构有贡献"的写法（那需要 (a)）

**本轮已确立的可发表事实是工程性的**：physical-evidence 通路带来 **+24.4% / +27.9%**
的实测前向开销，使原定每 seed 35 h 包络装不下三臂；typed evidence 在 4 epoch 剂量下的
可归因贡献**本轮未测**（不是无效）。

---

## §6 明确不做

| 不做 | 理由 |
| --- | --- |
| 新增任何物理 loss 项 | C1 分析场被罚 10.2×（0094）、C2 15/15 全负（0109） |
| 重开 Stage P/R/C/A、城市静态、容量、EMA、长 lead 权重 | 全部按预声明关闭 |
| 重开低秩带 / 平衡带 / 地转投影 / case 自适应 | 本轮七项判决测试已逐条实测关闭 |
| 改 `model/**` 已有行为 | 会改 `model_code_sha256`，父与既有 checkpoint 全部不可加载 |
| 改 `training/r7_evaluate.py` / `training/r7_long_rollout_runner.py` | 316 行 / **597/600 行（R-051）** |
| 扫 `lr` / `batch` / `clip` | 另立计划；`clip=1.0` 让 100% update 被裁剪，是独立问题 |
| 重建 store / 新增或重下数据 / 改 `data_identity` | 只读 |
| 读 2023 test | `r = 0` |
| `cron` / 守护 / 会话外续跑 / 对邻居进程发信号 | 硬约束 |
| 承诺正面结果 | 见 §2；三档结果都有可写内容，但判据不因结果而改 |
