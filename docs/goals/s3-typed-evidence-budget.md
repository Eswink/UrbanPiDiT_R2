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

**② 磁盘与孤儿守卫**。`/data` 3.6 T，已用 2.6 T，**可用 835 GB**（77%）；attempt 现占 **88 MB**
（两臂各 37 MB，尚无 checkpoint）。`checkpoint_every=5000` ⇒ 每臂 6 个中间 checkpoint，
九臂全部 checkpoint 在数十 GB 量级，835 GB 充裕。这一条写在这里是为了让「剩余空间」有实测
出处，而不是假定。

`tools/r7_round.py stage-plan seed 41` 实测返回 `already_running: true` + 5 个存活 PID
⇒ 重复启动会被拒而不是双开抢 GPU；`orphans` 子命令报告各 stage 的 `running/pids`。
两处都只读，不向任何进程发信号（R-054）。

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
| D6 逐 seed 逐臂 receipt | ⏳ **0/9**（2 个在跑，7 个未开始） |
| D7 `readings.json` 五键 | ⏳ **不存在** |
| D8 证据页与四件套登记 | ⏳ **未写**（装配器已就绪并自测：§5.2s） |
| D9 判定点逐条写 (a)–(f) | ⏳ **未写**（求值器已就绪，(d) 将恒为 `null`） |

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
