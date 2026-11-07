# S3 Stage C2：导数域目标匹配 on/off（散度 + 涡度），三 seed 配对

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-derivative-matching-c2", "driver": "scripts/study_r7_s3_derivative_matching_c2.py", "output_stem": "r7_s3_derivative_matching_c2_20261108", "control_record": null, "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_derivative_matching_c2.py"], "planned_seconds": 115200.0, "hard_cap_seconds": 230400.0, "code_archive": "outputs/r7_s3_derivative_matching_c2_code.zip", "commit_paths": ["scripts/study_r7_s3_derivative_matching_c2.py", "scripts/r7_s3_derivative_matching_c2_protocol.py", "training/r7_derivative_matching.py", "training/r7_objective_axes.py", "training/r7_long_rollout_runner.py", "tests/test_r7_s3_derivative_matching_c2.py", "docs/goals/s3-throughput-physics-c2.md", "docs/decisions/0095-stage-c2-derivative-domain-target-matching.md", "docs/decisions/README.md"], "commit_message": "feat(r7): run Stage C2 derivative-domain target matching on/off, three seeds", "register_paths": ["docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-throughput-physics-c2.md"], "register_message": "docs(r7): register Stage C2 derivative-matching readings"} -->

本文件是 Stage C2 的**唯一执行源**；机器可读的 `round-spec` 在上方。
上游是不可变记录：`docs/goals/s3-throughput-and-physics-campaign.md`（Stage C2）、
`docs/plans/0028-throughput-physics-campaign.md`、决策 **0084** §6、**0094**（C1 前置否决的实测与方法）、
**0095**（本轮设计）、**0086/0087**（守卫与 checkpoint）。

**`scientific_claim: false`、`test_read: false`、`r = 0`。**

---

## §1 Objective（这一轮问的是什么）<!-- 实测 1206 字符 -->

> 在 `/data/esw/UrbanPiDiT_R2` 的 `r7/weather-reasoning` 上跑 S3 Stage C2：判据与相位以本文件 `docs/goals/s3-throughput-physics-c2.md` 为准、设计依据是决策 0095 与 0084 §6。单因子 on/off、 三 seed 配对（41/42/43）、treatment 清单（32,131 窗）、batch 固定 4、注册 recipe（K=4/depth=4/ dim=192/LR 2e-5/warmup 10/12 物理步全 BPTT），**恰好 32,132 updates（4 epoch）**，从 `59340710…` 修订下的迁移父出发。on 臂在注册目标上加**唯一一项**：最后一个 6 h 预报的导数域目标匹配 `w_div·mean(((∇·pred−∇·target)/s_div)²) + w_vort·mean(((∇×pred−∇×target)/s_vort)²)`， `w = 0.1/0.1`、`s_div = 2.5665e-05`、`s_vort = 3.5089e-05 s^-1`；off 臂**不含**该键， 其契约与注册路径逐字相同。预声明出口 (a)–(d)：72 h t2m skill 相对**同代码身份** off 臂的 配对均值不退化、两项匹配在 on 臂逐步记录且严格为正、off 路径逐位等价、无判据被改； **任一假即永久关闭「导数域目标匹配」这一支**，不调权重、不换项、不加 seed。桥 (e) 逐 seed 断言父身份（`model_code_sha256` + 155 张量 `state_digest`）。**前置检验已在驱动内实跑通过** （调用模块本身，非重实现）：4 个 train case 的 `pred = target` **精确 0.0**、on 臂 dose `0.5702/0.4920/0.4707/0.4300`（在预声明带 `[0.1, 10]` 内）、由分析场重导出的尺度与声明值相符。 预算按 Stage B 第四点**实测** 1.1096 s/update 重算：训练 `35,654 s` + 评分 `2,588.1 s` = `38,242 s ≈ 10.62 GPU-h`/arm，两臂 × 三 seed ⇒ **约 63.7 GPU-h**（**估计**）； 包络 `per_seed = planned = 115,200 s`（32 h）、`hard = 230,400 s`。 `scientific_claim: false`、`test_read: false`、`r = 0`，不读 2023 test、不重建 store、 不改任何冻结判据、不自宣科学通过或 goal 完成。

## §2 冻结的配方（与 Stage B 第三点逐字相同，唯一差别是 on 臂的那一项）

- **实例**：`outputs/r7_s3_fy_multiyear_20261023_attempt01`（65×65 多年，17 通道）
- **数据**：treatment manifest，**32,131 可用窗**、11 条年界排除
- **训练**：`32,132 updates × batch 4 = 128,528` 样本 ⇒ **4.000124 epoch**
- **K=4、depth=4、dim=192、LR 2e-5、warmup 10、weight_decay 1e-4、clip 1.0、FP32**
- **物理权重** `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`（12 个 6 h 物理步，全 BPTT）
- **评分**：整张 65×65（`boundary_margins=(1,)`、`full` 4,225 点），五个 lead，全部 17 个变量
- **气候态分母**：48 桶 train-only 的 D2 reference；**只有 t2m 有注册分母**
- **父**：`outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01/`，逐 seed 断言
- **checkpoint**：每 5,000 步 + 端点（决策 0087）
- **同步守卫**：全模型有限性每 100 步 + 每 checkpoint + 端点（决策 0086）

## §3 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 本轮冻结协议 | `outputs/r7_s3_derivative_matching_c2_20261108_attempt01/protocol.json`（含 `preflight` 实测块） |
| D2 | 代码身份归档 | `outputs/r7_s3_derivative_matching_c2_code.zip`（带 commit 注释） |
| D3 | 逐 seed receipt | `seed{41,42,43}_receipt.json`（两臂各一份，含父身份与 s/update） |
| D4 | 十七变量 × 五 lead 读数 | `readings.json`（含 `exit_conditions`） |
| D5 | 证据页 | `docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md` |
| D6 | 四件套登记 | index record / 账本行 / campaign §8 / `campaign-state` |

## §4 判据：预声明出口（冻结后不放宽）

**判据上游**：`docs/goals/s3-throughput-and-physics-campaign.md` §3 与决策
`docs/decisions/0095-stage-c2-derivative-domain-target-matching.md`；本文件不发明新阈值。

- **(a)** 72 h t2m skill 相对**同代码身份**的 off 臂**不退化**（三 seed 配对差的均值 ≥ 0）；
- **(b)** 两项导数匹配在 on 臂**逐步记录且严格为正**（否则「匹配没生效」）；
- **(c)** off 路径**逐位**等于注册实现（由 dispatcher 的早期 `None` 与其反证测试保证）；
- **(d)** 无判据被改动、`test_read is False`、`scientific_claim: False`。

**(a)–(d) 全真 ⇒ 提请独立复核。** **任一假 ⇒ 永久关闭「导数域目标匹配」这一支**，
不调权重、不换项、不加 seed，如实登记。

**桥 (e)**：父身份逐 seed 断言（`model_code_sha256` + 155 张量 `state_digest`）。
失败记 `identity/engineering unresolved`，**不进入** (a)–(d)。

## §5 预算与停止

| 项 | 值 | 依据 |
| --- | --- | --- |
| 训练时间估计 | **35,654 s** | `32,132 × 1.1096`（Stage B 第四点**实测**） |
| 评分余量 | **2,588.1 s** | 第四点五 lead 的**实测**耗时 |
| 合计 | **38,242 s ≈ 10.62 GPU-h**/arm | 上两者之和 |
| 两臂 × 三 seed | **约 63.7 GPU-h** | **估计**；四点 s/update 极差 5.7% ⇒ 实际可能 60–68 |
| `per_seed_seconds` | 115,200 s（32 h） | `ceil(3 × 38,242 / 3600) × 3600` |
| `planned_seconds` | 115,200 s | 与包络相同 |
| `hard_cap_seconds` | 230,400 s（64 h） | 包络的两倍 |
| 峰值显存估计 | 9,126 MiB | Stage B 第二点**实测** reserved peak |

软预算超了继续并记 `soft_overrun`；硬上限截断 attempt（记 `budget_limited`、全额记账、**不算通过**）。

**停止三条**：触保留授权边界；达已冻结停止条件；确无可安全独立推进的工作。

## §6 明确不做

- 不重建 store、不新建 129×129 源、不动 `data/raw|interim|processed` 与归档快照
- 不读 2023 test（`r = 0`、`test_read: false`）
- 不改本轮任何判据；不事后放宽 (a)–(d)
- **不扫 λ**、不把两项拆开单独跑（分离实验不在本轮）
- 不重开 SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量
- 不 paid / rental / exclusive / cron / 守护 / 会话外续跑；不对邻居进程发任何信号
- **不把 C2 说成「无目标的物理信息」**：它读 target，学到的是「像目标的导数场」
- 不自宣科学通过或 goal 完成

## §7 进度与交接

**（执行者在此追加进度块；停时必须写「为何停 + 恢复第一条命令」。）**

- 2026-11-08 建立：状态 `prepared`，未冻结、未开跑，0 GPU-h。

- **2026-11-08 `attempt02` 作废（0 GPU-h）**：`seed` 阶段在**第一次更新之前**就失败——
  `_parent_model` 调用了裸 `arm_spec()`，而它在驱动/协议拆分后只存在于协议模块（`NameError`）。
  **无读数、无 GPU 时间**。修法：改调 `proto.arm_spec()`，并新增一条**全局名解析**反证测试
  （反汇编两个模块里每个函数、断言它加载的每个全局都存在）——这类缺陷对 `import` 与任何
  「不执行该函数」的测试都是不可见的，前两次都是这样漏到 GPU 阶段的。

- **2026-11-08 `attempt03` 作废（0 GPU-h）**：`seed` 阶段在**第一次更新之前**失败——
  `mask must be [H,W] matching the prediction grid`。根因是**我**照搬了 wide 129×129 配方里的
  `interior.supervision_mask()`，而该掩码在 wide store 上是 65×65、在**本 store 上只有 1×1**，
  本 store 的注册配方是 **full grid**（Stage B 的驱动根本不传 mask，其报告写 `supervision.kind
  == "full_grid"`）。**无读数、无 GPU 时间**。修法：不传 mask，并**断言**训练报告必须是 `full_grid`，
  使掩码无法在其它身份检查仍全通过的情况下悄悄回来。

- **2026-11-08 `attempt04` 开跑（当前）**：协议 `ac61e3f…`、archive 通过、`prepare` 578.6 s
  （含驱动内前置检验：dose `0.5702/0.4920/0.4707/0.4300`、`zero_at_target` 全 `0.0`、
  导出尺度与声明相符）。两卡各一 seed 并行（seed41/42 先跑 **off** 臂），
  实测 9,308 / 9,287 MiB、util 25%/68%、loss 0.95 → 0.69。

  **早期实测速率超出预算（如实记录）**：跑起来后 13/12 条进度记录的实测 s/update 为
  **1.226 / 1.275（均值 1.250）**，据此外推每臂 `32,132 × 1.250 + 2,588 = 42,765 s ≈ 11.88 GPU-h`、
  六臂 **≈ 71.3 GPU-h**——**高于**本 brief §5 声明的 60–68 区间（那区间由 Stage B 第四点的
  1.1096 s/update 推出）。两个数都记下：1.1096 是**第四点实测**、1.250 是**本轮早期实测**，
  差异约 12.6%，**超出**四点 s/update 极差（1.1096–1.174，5.7%），说明本轮两臂同跑争用使单臂变慢。
  **判据不受影响**（(a) 是配对比较，两臂同速变慢）；**包络仍够**：每 seed 两臂
  `2 × 42,765 = 85,530 s < per_seed 115,200 s`，不会被时长截断。

## §8 下一动作（S3）

冻结后依次 `prepare` → `archive` → `seed 41` → `seed 42` → `seed 43` → `reading`，然后做四件套登记。
**跑完后下一动作（S3）**：按 §4 的 (a)–(d) 读 `readings.json` 的 `exit_conditions`；
全真则提请独立复核，任一假则永久关闭该支并如实登记——**两种结果都必须原样登记**。

- **2026-11-08 等待期间推进的两项非阻塞交付（均在 C2 执行闭包之外，不改任何训练/模型代码）**：
  1. **Stage F 成本菜单**：`docs/R7_S3_STAGE_F_COST_MENU.md`。按**每个 batch 各自的实测速率**
     重算（batch 4 = **1.1469 s/update**，四次独立单臂长跑均值，极差 5.8%；batch 6/8/10/12 用
     Stage A A6 实测），并把「epoch → updates 的分母随 batch 变」显式算进去。**25 行表格全部
     独立复算核对**（发现并修正 2 处四舍五入：42.7→42.63、168.3→168.38）。**不含任何推荐**——
     选哪个配置是用户的决定。一处口径分离：C2 并发的 1.250 s/update **不进入**速率依据，
     因为它测的是争用（1.09×），不是卡的固有速率。
  2. **Stage E1 决策 0096**：把 E1 拆成 E1a（dense-12）/ E1b（final-only）两轮、各单因子三 seed，
     并**显式申报约 500 GPU-h 的超支**（是 E 段 64 GPU-h/轮预算的 3.9 倍，根因是 E 段按
     45,575 updates 估算而本轮按 Stage B 的 16 epoch = 128,528 updates 执行）。同时预先声明：
     **E1b 的旧读数已是 `mixed` 而非 null，若本轮仍 mixed 则该支永久关闭**，不得事后改叙为
     「还需更多算力」。

  **两项为何现在做而不是等**：它们都是文档/算术，不在 C2 的 94 项执行闭包里，
  不会让 C2 的 `reading` 身份失效。**D1 接线与 E1 实跑则必须等**——`validate_protocol` 会校验
  `model_code_digest()` 与 `training_code_digest()`，现在改 `model/**` 或 `training/**`
  会让 C2 已冻结的身份失效。

- **2026-11-08 第三次 CI 红（`0f72dca`）已修，`0f026c1` 绿（run `42399805771`）**：  干净工作树 **5,569 passed / 27 skipped / 0 failed** 独立复核。
  根因：`scripts/r7_s3_derivative_matching_c2_protocol.py` **404 行**，越过 R-021 的 400 行目标带 4 行，
  使 `size-thresholds.md` 记的 135 与实测 136 不符。
  **处置与前面的 R-020 案例不同，理由必须写明**：该文件的 SHA256 已被在跑的
  `attempt04` 钉进 `execution_files_sha256`（`3f1d8e37…`），`validate_protocol` 会逐文件校验。
  **我先试了就地缩小**（改注释与 docstring 到 400 行），随即实测到
  **`pinned 3f1d8e37… vs actual 4369e449… MATCH: False`** —— 那会让约 500 GPU-h 的在跑轮次身份失效。
  **立刻用 `git checkout` 恢复被钉的字节**，复核：**94 个钉住文件全部匹配、
  `model_code_digest` 与 `training_code_digest` 均与协议相符**。
  然后按 `size-thresholds.md` **自身的既有惯例**记录计数（135→136）并写明理由与恢复条件，
  **不动阈值、不动规则级别、不加冻结例外**；该文件在 C2 跑完后应缩回 ≤400 行、计数回到 135。
  **这一条是「先动手、再验证」的一次真实失误，已如实登记**；正确的顺序应是在改任何
  `execution_files_sha256` 成员之前先核对它是否被在跑轮次钉住。

- **2026-11-08 审计项「Stage D1 接线未做」的核对结果：部分成立、部分错误。**
  审计方依据是 `training/*.py` 对 `static_conditioning`/`static_identity` **零引用**——
  这条观察对，但推论错：开关经 **spec 字典**流通，`training/r7_experiment.make_model`
  原样 `**options` 展开，所以「零引用」是**设计**。已写成决策 **0097**（`proposed`、0 GPU-h），
  关键实测：`make_model('process', spec)` 收该键、参数增量 **0**（off/on 均 3,286,037）、
  off 不构造张量、on 加项 `(1,1089,192)` 范数 **182.7212** 含非零、`named_parameters()`
  中静态量 **`[]`**、两个缓冲区两次构造**逐位相同**、`field` 与 `load_static_fields()`
  **逐位相同**。**真正的缺口**是 on 臂多的 **2 个 persistent buffer**
  （`field` (8,65,65) / `projection` (16,192)）：`strict=True` **两向都失败**、
  `strict=False` **静默丢弃**（实测 `missing 0 / unexpected 2`），而注册父的静态键为 **`[]`**
  ⇒ **注册父不能直接用于任何 on 臂**。同时**证伪目标长文一条预测**：那句「这会改
  `model_code_sha256` 故必须重做父迁移」**从写下之日起就不成立**（开关代码 `896eabf`
  2026-10-29 早于父迁移 2026-11-03，改 `False`→`True` 不动 `model/` 任何字节）；
  缺的是**一次确定性缓冲区对账**（0 GPU-h），不是代码身份迁移。**未执行**——
  C2 的 94 项闭包含全部 17 个 `training_code_digest` 源文件，等 `reading` 之后再动。
  本条不改 C2 的任何判据或身份。
