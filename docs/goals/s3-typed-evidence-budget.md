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
| —— | 段 B-1 组内剂量曲线 | **未开始** | —— |
| —— | 段 B-2 推理深度扫描 | **未开始** | —— |
| —— | 段 A 三臂 4-epoch | **未开始** | —— |
| —— | 判定 (a)–(f) | **未开始** | —— |
| —— | 登记 | **未开始** | —— |

**GPU-h = 0。2023 test 未读，`r = 0`。段 A 一个 update 都还没跑。**

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
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
