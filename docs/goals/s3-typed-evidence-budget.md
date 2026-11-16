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
| —— | 判定 (a)–(f) | **未开始** | —— |
| —— | 登记 | **未开始** | —— |

**GPU-h = 段 B 实测（见下）。2023 test 未读，`r = 0`。段 A 一个 update 都还没跑。**

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

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```

**当前监视命令**：

```bash
find outputs/r7_s3_segB_probe_20261116 \
  -name boundary_rmse.csv | wc -l   # 42 (dose) + 20 (depth) 全部完成为止
```

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
