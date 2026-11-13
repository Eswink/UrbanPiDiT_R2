# S3：城市静态协变量的三臂消融 —— 把「用但未消融」的文献空白做成一次可证伪测量

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-urban-static-ablation", "driver": "scripts/study_r7_urban_static_ablation.py", "output_stem": "r7_s3_urban_static_ablation_20261113", "control_record": "s3-urban-static-covariates-wiring", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_urban_static_ablation.py", "tests/test_r7_static_covariates_wiring.py"], "commit_paths": ["scripts/r7_urban_static_ablation_protocol.py", "scripts/r7_urban_static_placebo.py", "scripts/study_r7_urban_static_ablation.py", "tests/test_r7_urban_static_ablation.py", "docs/goals/s3-urban-static-ablation.md", "docs/plans/0032-urban-static-covariate-ablation.md", "docs/decisions/0113-urban-static-ablation-and-placebo.md"], "commit_message": "feat(r7): urban static covariates -- three-arm ablation with a spatial placebo", "register_paths": ["docs/R7_S3_URBAN_STATIC_ABLATION_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-urban-static-ablation.md"], "register_message": "docs(r7): register the urban static three-arm ablation readings", "planned_seconds": 115200.0, "hard_cap_seconds": 230400.0, "code_archive": "outputs/r7_s3_urban_static_ablation_code.zip"} -->

本文件是**唯一执行源**；机器可读的 `round-spec` 在上方。
设计依据是不可变记录 `docs/plans/0032-urban-static-covariate-ablation.md`（文献核对、成本表与全部负面后果在那里）。
准入裁决是决策 **0113**；接线决策是 **0069**（已 `accepted`），数据决策是 **0063**。
`campaign-state.current_round_goal` 必须指向本文件，否则 `tools/r7_round.py` 会解析到别的轮次并冻结错协议。

**状态：prepared（2026-11-13 建立，未启动）。** `scientific_claim: false`、`test_read: false`、`r = 0`。
**本轮 0 GPU-h**；无 `protocol.json`、无 `code.zip`、无 `readings.json`；(a)–(f) 全部未判定。
在此之前，任何引用本文件作为「静态协变量有效/无效」的证据都是错的。

---

## §0 Objective（单段，实测 3940 字符）

> 在 /data/esw/UrbanPiDiT_R2 的 r7/weather-reasoning 执行目标长文 docs/goals/s3-urban-static-ablation.md（先完整读它，判据以它为准；文献核对见 docs/plans/0032-urban-static-covariate-ablation.md，准入见决策 0113）。毕业论文主线：基于深度学习的物理信息混合天气预测。起点：HEAD 8b5e8c0、used_gpu_h 593.325957、model_code_sha256 59340710623c…（与父 pin MODEL_CODE_SHA256 相等，故父可直接加载）。前六轮物理信息全负且全在损失侧：C1 8/8 方向反、C2 15/15 全负 66.99 GPU-h、Stage P 判 skip-q、Stage R 85/85 全劣化、Stage C 减半需 775102 倍训练量、Stage A 振幅三 seed 升 +0.114 到 +0.125 但 skill 9/9 全降 21.83 GPU-h；Stage A 已排除剂量不足，因直接作用点三 seed 一致移动。输入侧：显式 tendency 输入已由决策 0049 关闭（线性冗余被预先承认），4 个物理诊断量作输入已由 record s3-typed-spatial-result 测过（B 到 C delta rmse 绝对值不超过 3.2e-6 K，float32 噪声级）。故本轮测最后一个未测的输入侧方向：城市静态地表协变量。文献依据（2026-11-13 一手读正文）：GraphCast 2212.12794 3.3 节静态输入只有 land-sea mask、surface geopotential、sin cos 经纬度；Pangu 2211.02556 III-B 只有 topography/land-sea/soil type 三 mask；ClimaX 2301.10343 Table 9 只有 LSM 与 Orography；Aurora 2405.13063 8.1 只有三个 static variables；AIFS 2406.01465 Table 1 只有 LSM、orography、次网格地形标准差与坡度、insolation、经纬度、时间编码；FourCastNet 2202.11214 与 FengWu 2304.02948 无静态输入。城市建成区比例、建筑高度、人口密度、土地覆被占比在这七篇里零命中；且这七篇无一篇做过有静态对无静态的消融（GraphCast 消融 multi-mesh、FengWu 消融 replay buffer、Aurora 消融数据多样性），故五篇用静态特征的全部是用但未消融，这是可直接引用的文献空白。本轮设计三臂：control 不构造静态通路，treatment 用发布的 8 层真实静态场，placebo 用同一份 8 层场逐格空间置换且复用同一份发布投影。placebo 必须逐变量多重集与 treatment 逐位相同（排序后比较），只破坏空间自相关，因为打乱通道标签会改变每个 token 的码分布而逐格置换保持边缘分布不变。三臂共享同一份代码、同一个 model_code_sha256、同一份父权重、同一个 seed、同一批更新，差异只在两个 buffer 的取值上；禁止改 model 目录任何文件：model_code_digest 覆盖 model 下全部 py 而 load_checkpoint 要求它等于 checkpoint 记录的 digest，改代码会让三 seed 注册父全部不可加载（wiring 轮已栽过），故臂差异必须经 fine_tune_long_rollout 的 parent_weights state dict 实现，可复用 tools/reconcile_r7_static_parent.py 的 _published_buffers 但不得改它。field 与 projection 都是 register_buffer 不在 parameters 里，故三臂 parameter_delta 恒为 0，容量混杂被结构性排除，必须逐臂实测断言而不得写成字面量。交付物：D1 预检报告（8 层 65x65 全有限、static_identity 通过、三臂 parameter_delta 实测为 0、placebo 逐变量多重集与 treatment 逐位相同、三 seed 三臂 strict 加载通过）· D2 冻结协议（含 static 段与 family_argument）· D3 code.zip 身份归档且必须带 commit 注释否则 archive 拒绝启动· D4 逐 seed 逐臂 receipt（含父身份与 s/update）· D5 五 lead 读数（6 12 24 48 72 小时 val 2022 boundary_rmse.csv 含 exit_conditions）· D6 证据页与四件套登记 · D7 判定点：逐条写 (a) 到 (f) 真或假。预声明出口任一为假即永久关闭该支：(a) 72 小时 t2m skill 三 seed 配对差 treatment 减 control 全不小于 0；(b) 24 与 48 小时配对差三 seed 全不小于 0；(c) 6 与 12 小时配对差不劣于 -0.005；(d) control 臂逐位等价，由 static_conditioning 为 False 时 pathway 为 None 的代码路径加会失败的反证测试支撑，不由读数自证（0109：c=True 是字面量、伪造读数仍 all=True）；(e) 预检全真；(f) 归因判据 treatment 减 placebo 在 72 小时不小于 0.001 K，阈值来源是 wiring 轮 on off 差落在 +0.0014 到 +0.0043 K 的 120 updates 噪声上界。若 (a) 真而 (f) 假本轮记为负，写明效应与空间结构无关、等价于位置码，因模型已有 position_features 的 sin cos lat 与 lon 四谐波，这是必须排除的替代解释。预算冻结前写定：3 臂 x 3 seed x 4000 updates 从 pin 的 p1600 父微调，batch 4，recipe 逐字沿用，软预算 115200 秒、硬上限 230400 秒；不扫权重（buffer 固定）、不加 seed、不改 4000 updates、事后不放宽判据。执行方式：用 workflow 编排，先加载 dynamic-workflows 技能（未加载时 CreateWorkflow 拒跑），从 .zcode/workflow-drafts/r7-campaign-round.dwf.ts 派生 r7-urban-static-ablation.dwf.ts，必须用 CreateWorkflow 传 path 加 args 启动、不要用 AmendWorkflow（实测丢 args 让 dryRun 静默变 true），maxRounds 自设硬上限、world.run 命令名保持编译期字面量、自跑 round-guard 与 commit-guard 与 verify 不依赖 hook 兜底、只推 r7/weather-reasoning；先传 maxRounds 0 dryRun true 做零成本编译校验；它是有界可恢复编排器不是守护进程，被打断按 ResumeWorkflowRun 续。跑任何 stage 前先跑 tools/r7_round.py state 核对 current_round_goal。禁止：重开 C1 C2 Stage P R C A、重开 fine-tuning-only 空间梯度支、重开 tendency 输入、重开物理诊断量输入、修改 model 目录任何文件、把 projection 改成可训练、重建 store、新增或重下数据、改 data_identity、用多期数组做时变轮、扫权重、加 seed、改 4000 updates、与其它已关轴组合、换名重开、读 2023 test、动 data/raw|interim|processed 与归档快照、合成数据、事后放宽出口、删弱测试、绕 checkpoint 与数据身份校验、paid 与 rental 与 exclusive 与 cron 与守护与会话外续跑、对邻居进程发任何信号、main 合并与 force 与关 issue、自宣科学通过或 SOTA 或最终 goal 完成。只在三种情况停：触授权边界、达已冻结科学停止条件、确无可安全独立推进的工作；停时必须在 8 节写为何停加恢复第一条命令。每步给可核查证据（命令加输出加 digest），估计值不得写成实测值。不承诺本轮一定有正面结果：若 (a)(b)(c) 任一为假或 (a) 真而 (f) 假，如实记负并停；那时输入侧与损失侧两侧主要方向都已测过，论文的正面结果只能靠证伪章与机制发现，这一点动手前就承认，不得等读数出来再改叙事。结束按 AGENTS.md 汇报改动、实验与失败、技能与代理使用、验证与精确 CI、成本与 issue 状态、未做项与下一项。

---

## §1 现状与起点

| 项 | 值 | 来源 |
| --- | --- | --- |
| 工作分支 | `r7/weather-reasoning` | `git status` |
| 起点 HEAD | `8b5e8c0`（工作树干净） | `git log -1` |
| `model_code_sha256` | `59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324` | `training.r7_experiment.model_code_digest()` 实测 |
| 父 pin `MODEL_CODE_SHA256` | `59340710…` —— **与当前相等** | `scripts/r7_s3_position_retention_parent.py:24` |
| 父 checkpoint | `outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/seed{41,42,43}/parent_update_0001600.pt` | 同上 `:21`、`:26-30` |
| `static_identity` | `8671ed7c…` | `model/static_conditioning_r7.py:54` |
| `used_gpu_h` | `593.325957`（**仅会计**，决策 0030/0038） | campaign-state |
| 2023 test | **未评分**，`r = 0` | —— |

### §1.1 六轮负结果的统一诊断

| 轮次 | 形式 | 结果 | 成本 |
| --- | --- | --- | --- |
| C1 | 无目标约束束 | 真值被罚 10.2×，8/8 方向反 | 0 GPU-h |
| C2 | 导数域目标匹配 | 15/15 (lead, seed) 全负 | 66.99 GPU-h |
| Stage P | 谱几何前置 | P1 判 `skip-q`、P2 三方向全不过 | 0 GPU-h |
| Stage R | 后处理振幅校正 | 85/85 (变量, lead) 全劣化 | 0 GPU-h |
| Stage C | 缺口衰减律 + β_j | α=0.05111，减半需 775,102× 训练量 | 0 GPU-h |
| Stage A | β_j AMSE-analogue 归一化 | (a) 真 +0.114…+0.125，但 skill 9/9 全降 | 21.83 GPU-h |

**共同结构：六轮全部改了「按什么算分」，没有一轮改「模型能看到什么」。**

Stage A 花钱买来的排除：直接作用点（细带振幅比）已经三 seed 一致地移动了，
所以 (a) 的阴性**不可能**再用「剂量不足」解释。这个排除本轮可以直接复用。

### §1.2 输入侧三条分支的状态

| 分支 | 状态 | 依据 |
| --- | --- | --- |
| 显式 tendency 输入 | **关闭** | 决策 0049:55 逐字：「显式 tendency 的**线性冗余**被预先承认」 |
| 4 个物理诊断量作输入 | **已测，无差别** | `record:s3-typed-spatial-result`：B→C `\|Δrmse\| ≤ 3.2e-6 K` |
| **静态地表协变量** | **接线已通，从未科学规模消融** | 决策 0069 `accepted`；本轮 |

### §1.3 文献空白（2026-11-13 一手核对，逐字引文见计划 0032 §0.3）

**七篇主流神经天气模型的静态输入里，城市/土地覆被零命中。**
**且这七篇里没有任何一篇消融过静态输入本身。**

| 论文 | 静态输入 | 含城市/LULC | 是否消融 |
| --- | --- | --- | --- |
| GraphCast `2212.12794` §3.3 | land-sea mask、geopotential at the surface、sin/cos lat/lon | 否 | 未消融（消融 multi-mesh §7.3.1） |
| Pangu `2211.02556` §III-B | topography / land-sea / soil type mask | 否 | 未消融 |
| FourCastNet `2202.11214` §2.2 | **无** | 否 | —— |
| FengWu `2304.02948` §2.1 | **无** | 否 | 未消融（消融 replay buffer §4.4） |
| ClimaX `2301.10343` App. Table 9 | LSM、Orography | 否 | 未消融 |
| Aurora `2405.13063` App. §8.1 | geopotential at surface、land-sea mask、soil-type | 否 | 未消融（消融数据多样性 §15） |
| AIFS `2406.01465` §3 Table 1 | LSM、orography、次网格地形 σ 与坡度、insolation、经纬度、时间编码 | 否 | 未消融 |

⇒ 五篇使用静态特征的论文全部是「**使用但未消融**」。

---

## §2 三臂设计

| 臂 | `field`（8 层） | `projection` | 语义 |
| --- | --- | --- | --- |
| **control** | 不构造（`static_conditioning=False`） | —— | 17 通道基线 |
| **treatment** | 发布的 8 层真实静态场 | 发布投影 | 真实城市/土地覆被信息 |
| **placebo** | 同一份 8 层场**逐格空间置换**（固定种子） | **同一份**发布投影 | 同等码空间，空间结构被破坏 |

**placebo 为什么必须空间置换**：逐格置换保持每个协变量的**边缘分布完全不变**（同一个多重集），
只破坏空间自相关。打乱通道标签会改变每个 token 的码分布——那就不是合格的 placebo。

**为什么不能只做两臂**：`StaticConditioning.projection` 是
`randn(16, 192, seed=0)/√16` 的**固定随机矩阵**（`model/static_conditioning_r7.py:178-184`）。
任何空间变化的 8 场输入都会给出互不相同的逐 token 码，而模型**已有**
`position_features()` 的 `sin/cos(lat)`、`sin/cos(lon)` 四个谐波
（`model/spacetime_conditioning_r7.py:217`）。**两臂的阳性无法排除「多一张位置查找表」。**

---

## §3 预声明出口（任一为假即永久关闭该支）

| 判据 | 要求 | 为何存在 |
| --- | --- | --- |
| **(a)** | 72 h t2m skill 三 seed 配对 Δ(treatment − control) **全 ≥ 0** | 主结论 |
| **(b)** | 24/48 h skill 配对 Δ 三 seed **全 ≥ 0** | 中期一致性 |
| **(c)** | 6/12 h skill 配对 Δ **≥ −0.005** | 短期合理容差，冻结前写定 |
| **(d)** | control 臂**逐位等价** | 由代码路径 + 会失败的反证测试支撑，**不由读数自证**（0109） |
| **(e)** | 预检：8 层全有限、`static_identity` 通过、三臂 `parameter_delta == 0`、**placebo 逐变量多重集与 treatment 逐位相同**、三 seed 三臂 strict 加载通过 | 防止 placebo 不是真 placebo |
| **(f)** | **归因**：Δ(treatment − placebo) 在 72 h **≥ 0.001 K** | **本轮科学核心**。见 §3.1 |

### §3.1 (f) 的方向必须说清

(f) 是**归因**判据，不是**成功**判据。

- **(a) 真而 (f) 假 ⇒ 本轮记为负**，结论：「静态协变量的效应与空间结构无关，等价于位置码」。
- 这仍是一个可发表的**机制负结果**，比一个无法归因的正结果有价值。

**0.001 K 阈值来源**：wiring 轮 on/off Δ 落在 `+0.0014 … +0.0043 K`，那是 **120 updates 的噪声上界**。
0.001 K 取在噪声底上——要求 treatment 至少比 placebo 好过一个噪声底，不是要求显著量。

---

## §4 剂量与预算（冻结前写定）

| 项 | 值 |
| --- | --- |
| 臂 × seed × updates | **3 × 3 × 4,000** |
| 起点 | pin 的 p1600 父（`ENDPOINT = 1600`） |
| batch | 4 |
| recipe | 逐字沿用注册配方 |
| 软预算 / 硬上限 | **115,200 s / 230,400 s**（Stage A 同口径） |
| 估计成本 | ≈11–26 GPU-h（按 20 估） |

**为什么 4,000 updates 够**：Stage A 用同一预算让直接作用点产生三 seed 一致的移动。
若本轮 (a) 阴性，**不能再以「训练太短」解释**。

---

## §5 交付物清单

| # | 交付物 | 验收标准 | 落盘位置 |
| --- | --- | --- | --- |
| **D1** | 预检报告 | 出口 (e) 全绿：8 层 65×65 全有限、`static_identity` 通过、三臂 `parameter_delta` **实测**为 0、placebo 逐变量多重集与 treatment 逐位相同、3 seed × 3 臂 `strict=True` 加载全通过 | `outputs/<stem>/preflight.json` |
| **D2** | 冻结协议 | `protocol.json` 含 `static` 段与 `family_argument`，`protocol_sha256` 落盘，**训练开始前**写定 | `outputs/<stem>/protocol.json` |
| **D3** | 代码身份归档 | `code.zip` **必须带 commit 注释**，否则 archive 拒绝启动 | `outputs/r7_s3_urban_static_ablation_code.zip` |
| **D4** | 逐 seed 逐臂 receipt | 3 seed × 3 臂各一份，含父 checkpoint sha256、父 endpoint updates、s/update 实测 | `outputs/<stem>/seed{N}/{arm}/receipt.json` |
| **D5** | 五 lead 读数 | 6/12/24/48/72 h，val 2022，`boundary_rmse.csv` 行数逐臂相等，含 `exit_conditions` | `outputs/<stem>/**/boundary_rmse.csv` |
| **D6** | 证据页 + 四件套登记 | `docs/R7_S3_URBAN_STATIC_ABLATION_RESULT.md`；index record、账本行、campaign §8、`campaign-state` 四件齐 | `docs/` + `docs/R7_EVIDENCE_INDEX.jsonl` |
| **D7** | 判定点 | 逐条写 (a)–(f) **真或假**，附实测数字；若 (a) 真而 (f) 假，明确记负并写明等价于位置码 | 证据页 §判定点 |

## §6 实施顺序

1. **P0 预检（0 GPU-h）** —— D1，出口 (e) 的全部内容，必须先全绿
2. **P1 冻结协议** —— D2
3. **P2 归档代码身份** —— D3
4. **P3 训练** —— D4，3 臂 × 3 seed，`fine_tune_long_rollout(parent_weights=...)`
5. **P4 评分** —— D5，五 lead，val 2022
6. **P5 判定点** —— D7，逐条写 (a)–(f) 真或假
7. **P6 证据页 + 四件套登记** —— D6

---

## §6 判据与证据来源

- 出口定义与阈值来源：本文件 §3；设计与全部负面后果：`docs/plans/0032-urban-static-covariate-ablation.md` §2.3、§6
- 准入裁决与族判别：`docs/decisions/0113-urban-static-ablation-and-placebo.md` §D1–D5
- 文献空白的一手核对：`docs/plans/0032-urban-static-covariate-ablation.md` §0.3（逐字引文与章节号）
- 两关闭分支的可检验差别：决策 `docs/decisions/0049-*.md:55`（线性冗余）、
  `docs/R7_S3_TYPED_SPATIAL_RESULT.md`（`|Δrmse| ≤ 3.2e-6 K`）
- 父可加载的实测证据：`outputs/r7_static_parent_reconcile_20261108_attempt02/receipt.json`
- 静态产物身份与「派生统计量非真值」定性：`docs/rules/data-and-artifacts.md`、
  `docs/decisions/0063-*.md`

---

## §7 明确不做

| 不做 | 理由 |
| --- | --- |
| 重开 C1 / C2 / Stage P / R / C / A | 全部按预声明关闭（0084 §6、0094、0109、0111、0112） |
| 重开 fine-tuning-only 空间梯度支 | 结论成立，已被 FastNet §4.2.1 解释 |
| 重开显式 tendency 输入 | 决策 0049，线性冗余已承认 |
| 重开 4 个物理诊断量作输入 | `record:s3-typed-spatial-result` 已测，float32 噪声级 |
| **修改 `model/**` 任何文件** | 会改 `model_code_sha256` ⇒ 父再次不可加载。三臂靠 state dict 区分 |
| 把 `projection` 改成可训练 | 会让 `parameter_delta > 0`，重新引入容量混杂 |
| 重建 store / 新增或重下数据 / 改 `data_identity` | 静态场已在 `attempt01/published/`，只读 |
| 用多期数组做时变轮 | 那不再是「时不变条件」，属另一个轴 |
| 扫权重 / 加 seed / 改 4,000 updates | 冻结前定死，事后不放宽 |
| 读 2023 test | `r = 0` |
| `cron` / 守护 / 会话外续跑 / 对邻居进程发信号 | 硬约束 |
| 承诺一定有正面结果 | 若 (a)–(c) 任一为假或 (a) 真而 (f) 假，如实记负并停 |

---

## §8 进度块

| 时间（UTC） | 阶段 | 状态 | 证据 |
| --- | --- | --- | --- |
| 2026-11-13 | 规划 | **完成** | 本文件、`docs/plans/0032-…`、`docs/decisions/0113-…` |
| 2026-11-13 | 实现落盘 | **完成** | 四个文件（见 §8.1）；32 条反证测试通过 |
| 2026-11-13 | P0 预检（e） | **完成，全绿** | §8.2：6/6 子句实测为真 |
| —— | P1 冻结协议 | **未开始** | —— |
| —— | P2 代码身份归档 | **未开始** | —— |
| —— | P3 训练 | **未开始** | —— |
| —— | P4 评分 | **未开始** | —— |
| —— | P5 判定点 | **未开始** | —— |
| —— | P6 登记 | **未开始** | —— |

### §8.1 已落盘的实现（2026-11-13）

三个实现文件，外加一个为满足 R-051（单文件 ≤600 行，**阻断**）拆出的支撑模块：

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `scripts/r7_urban_static_ablation_protocol.py` | 579 | 常量、三臂 spec、parent 加载、freeze/prepare/validate、`--preflight-only` 门 |
| `scripts/r7_urban_static_placebo.py` | 294 | placebo 构造与**全部测量**（多重集、token 衰减、Moran、被否决候选）、`family_argument`、arm state 构造 |
| `scripts/study_r7_urban_static_ablation.py` | 352 | 三臂 × 三 seed 训练与五 lead 评分、(a)–(f) 判定 |
| `tests/test_r7_urban_static_ablation.py` | 485 | 32 条反证测试 |

**验证（实测命令与输出）**：

- `.venv/bin/python tools/r7_round.py round-guard` → `"passed": true`，**187 passed**
  （`test_path` 还包含 `tests/test_r7_static_covariates_wiring.py` 与 `tests/test_check_conventions.py`）
- `.venv/bin/python tools/check_conventions.py --quiet --paths <7 条>` → `blocking rules=37 failing=0`
- 三次**注入缺陷**的反证（每次跑完立即恢复原文件）：
  1. placebo 改成打乱通道标签 → **3 条测试转红**（多重集、contrast、通道独立）
  2. control 臂也构造 pathway → **3 条测试转红**（含 (d) 的载体）
  3. (a) 真而 (f) 假仍判 advance → **1 条测试转红**

### §8.2 (e) 预检：6/6 子句实测为真

产物写在 `/tmp/urban_static_preflight.json`（**不入库**，本轮 0 GPU-h、无 attempt 目录）：

| 子句 | 实测值 |
| --- | --- |
| 8 层 × 65×65 全有限 | `true`，shape `[8, 65, 65]` |
| `static_identity` 校验 | `8671ed7c…`，通过 |
| 三臂 `parameter_delta` **实测** 0 | 三臂均 `3,286,037` 参数 / 155 parameter tensors / 155–157 state tensors；两 buffer 不在 parameters |
| placebo 逐变量多重集与 treatment 逐位相同 | `true`，8/8 变量排序后 `torch.equal` |
| 3 seed × 3 臂 `strict=True` 加载 | `true`，9/9 |
| treatment 与 placebo **只差 field** | `differing_keys == ["backbone.static_conditioning_pathway.field"]` |

### §8.3 一个必须写进结论的实测混杂（**动手时发现，不是读数后**）

**逐格置换保住了 raw 多重集，但没有保住 token 分布。** `static_features` 先做 2×2 patch 平均、再中心化，
所以逐格置换把每个 token 的特征朝全局均值收缩：

| 量 | 结果 |
| --- | --- |
| 原始多重集（排序后逐位比较） | **8/8 逐位相同** |
| per-token 特征标准差（placebo / treatment） | **0.83×** |
| 192 维投影码标准差（placebo / treatment） | **0.69×** |
| 空间自相关（四邻域 Moran 型均值） | treatment **0.74** → placebo **−0.0001** |

⇒ placebo 与 treatment 差**两件事**：没有空间结构 **且** token 幅度更低。因此：

- (f) 真 ⇒ 只能说「相干的、全幅度的静态项胜过被打散且衰减的版本」，**不能**单独说「效应来自空间结构」。
- 已写进 `proto.LIMITATIONS`、`frozen_constants()["decision"]`、
  `preflight.placebo_contrast.confound_statement`，并由
  `test_the_placebo_contrast_reports_its_own_attenuation` 钉住（若未来让它变「干净」，该测试转红）。
- 备选构造（2×2 / 4×4 块置换）也测了，记为 `rejected_constructions`：token 幅度保持更好
  （0.78–0.79）但**残留空间结构**（Moran 0.30–0.38）。看到这些数字后**没有**改构造——
  那会变成按结果挑 placebo。

### §8.4 与 brief 的一处数值差异，如实记录

brief 冻结 `planned 115,200 s / hard 230,400 s`。**这两个数字未被修改**；但 `per_seed_seconds`
由三臂估计推导（3× 后取整到小时 = **64,800 s**），而不是照抄 brief 的 115,200 s：

- brief 的 115,200 s 沿用 Stage A（**两臂**）的 per-seed 口径；本轮三臂，per-seed 估计 = 21,079.5 s。
- 照抄会让 per-seed 上限是自身估计的 5.5×，同时 `stage-seeds` 的单次调用上限反而更可能截断。
- 因此按 Stage A 自己的方法（3× 估计、取整到小时）推导 per-seed 上限；brief 的软/硬**总**预算数字保持原样。
- `test_the_budget_envelope_is_the_frozen_one` 断言：两个 brief 数字不变、per-seed > 估计、倍数 ≥ 3。

**GPU-h = 0。2023 test 未读，`r = 0`。无 `protocol.json`、无 `code.zip`、无 `readings.json`；
(a)–(f) 中只有 (d) 的代码路径与 (e) 有证据，(a)(b)(c)(f) 全部未判定。**

**恢复第一条命令**：

```bash
.venv/bin/python tools/r7_round.py state
```
