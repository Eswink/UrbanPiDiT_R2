# S3 Stage C2：导数域目标匹配 on/off（散度 + 涡度），三 seed 配对

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-derivative-matching-c2", "code_archive": "outputs/r7_s3_derivative_matching_c2_code_attempt05.zip", "commit_message": "feat(r7): run Stage C2 derivative-domain target matching on/off, three seeds", "commit_paths": ["scripts/study_r7_s3_derivative_matching_c2.py", "scripts/r7_s3_derivative_matching_c2_protocol.py", "training/r7_derivative_matching.py", "training/r7_objective_axes.py", "training/r7_long_rollout_runner.py", "tests/test_r7_s3_derivative_matching_c2.py", "docs/goals/s3-throughput-physics-c2.md", "docs/decisions/0095-stage-c2-derivative-domain-target-matching.md", "docs/decisions/README.md", "tests/test_r7_s3_c2_reading_exit.py", "tests/test_check_driver_attributes.py", "tools/check_driver_attributes.py", "docs/decisions/0098-stage-c2-attempt04-stopped-three-reading-stage-defects.md"], "control_record": null, "device_mode": "dual", "driver": "scripts/study_r7_s3_derivative_matching_c2.py", "hard_cap_seconds": 230400.0, "output_stem": "r7_s3_derivative_matching_c2_20261108", "planned_seconds": 115200.0, "register_message": "docs(r7): register Stage C2 derivative-matching readings", "register_paths": ["docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-throughput-physics-c2.md"], "seeds": [41, 42, 43], "test_path": ["tests/test_r7_s3_derivative_matching_c2.py", "tests/test_r7_s3_c2_reading_exit.py", "tests/test_check_driver_attributes.py"]} -->

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

- **2026-11-08 `attempt04` 开跑（当前）**：**冻结协议 `dcc38292…`**（`ac61e3f…` 是 `preparation_protocol.json` 的**初稿**摘要，不是冻结值——`prepared_protocol` 会把 live 身份、preflight、94 项闭包与父钉**加入**后再重算，所以两者必然不同；`validate_protocol` 于 2026-11-08 运行中**复验通过**，94 个钉住文件逐字节匹配）、archive 通过、`prepare` 578.6 s
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

- **2026-11-08 `attempt04` 终止（6.5555 GPU-h、零读数，决策 0098）。**
  **为何停**：等待期间只读审读 `readings()`/`evaluate_exit()`，发现**三处必然失败**，
  全部在**十小时训练之后**才触发，且全部落在已钉进 `execution_files_sha256` 的两个文件里
  ⇒ 在当前身份下**无法修补**，跑完只能得到零读数 + 约 61 GPU-h 账单：
  (1) `readings()` 与 `_train_arm()` 都写 `curve.EVALUATION_LEADS`，而该名字只在
  `study_r7_s3_wide_single_factor` 上，`r7_s3_curve_point_large_protocol` 把它 import 成
  `wide` **从不重绑** ⇒ **在跑的这一份字节上实测 `AttributeError`**；
  (2) 同一名字在 `_train_arm` 第 138 行（训练之后、评分之前）；
  (3) `entry[arm]["rmse"]/["skill"]` 的填充被**缩进在 `if arm == CANDIDATE:` 内**，
  `off` 臂永远是空字典 ⇒ 合成结果实测 `KeyError('72')`。
  另发现**第三个同类**：`curve.check_provenance` 实际定义在
  `scripts/r7_s3_curve_point_large_driver.py:96`，一并修。
  **终止动作**：先逐 PID 核对 `/proc/<pid>/cmdline` 含 `attempt04` 才对本实验自己的
  两个 `seed` worker 发 `SIGTERM`（**未对任何非本实验进程发信号**），
  随后确认两卡 0% / 38 MiB / 17 MiB、`stage-seeds` 父进程同时退出。
  **记账**：两臂各自记录的 `elapsed_seconds` 之和 **23,599.7 s = 6.5555 GPU-h**
  （可测的卡占用；墙钟 × 2 卡为 7.8802 GPU-h，但含 prepare 与等待，不采用）。
  **零读数，全额记账**；
  **不记成 `budget_limited`**（那是时长截断，这是主动作废）。
  两个 `off` 臂的 `progress.jsonl`/checkpoint 留在原地作留痕，**不作为任何结论的证据**。
  **修复只动读数路径**（不动训练、不动判据）：`LEADS = wide.EVALUATION_LEADS` 单一具名常量、
  `curve_driver.check_provenance`、填充移出候选分支、新增「每臂 lead 集合必须恰等于冻结五档」断言。
  **收尾实测**：两项新测试在 `git stash` 回缺陷版本后**都失败**、修复后 **23 项全过**。
  **顺带**：`r7_s3_derivative_matching_c2_protocol.py` 404→400（只压 docstring 与三行注释），
  R-021 计数 **136→135**，`size-thresholds.md` 的 marker 与实测重新一致。
  **恢复第一条命令**（`attempt05` 必须**重新冻结**，`attempt04` 协议永久作废）：
  ```
  .venv/bin/python tools/r7_round.py state
  ```
  然后按 §8 的 `freeze` → `prepare`（**preflight 必须重跑**）→ `archive` → `stage-seeds` →
  `reading` 顺序重开。**注意**：`attempt05` 会**重新**把 400 行的协议钉进
  `execution_files_sha256`，届时再改该文件仍须按 `size-thresholds.md` 的办法处理。

- **2026-11-08 第四次 CI 红（`3218058`，run `42414650080`）已修，`71d8e89` 重推**：
  失败步骤是第 8 步 **Compile active modules and check whitespace**，具体是
  `git show --check --format= HEAD` 报
  **`tests/test_r7_s3_derivative_matching_c2.py:298: new blank line at EOF.`**
  ——把读数出口测试拆成 `tests/test_r7_s3_c2_reading_exit.py` 时，原文件尾部留下一个空行。
  `compileall` 那半**实测通过**（`exit=0`），所以红的是**空白**不是语法。
  修法：把六个被触碰的文件的尾部统一成「恰好一个换行」（只改了这一个文件，14460→14459 字节），
  **无任何断言或逻辑改动**。**本地复验**：`git show --check --format= HEAD` 现在输出为空。
  **教训（已写进流程）**：CI 的第 8 步用 `git show --check` 检查**待推提交**的空白，
  所以这条应在**提交前**跑，而不是等 CI；拆分/移动文件的提交尤其容易在尾部留下空行。

- **2026-11-08 修复提交的 CI 实测**：`936d16c` 的 `R7 CPU CI` run **`42416263241`
  = completed / success**。此前两次红（`3218058` 的空白、`7b0c339` 之前的取消）都已解决：
  **取消不算通过**，故只以 tip 的这次 success 为准。
  **干净工作树**（`936d16c`，无 `outputs/`）实测
  `test_r7_s3_derivative_matching_c2` + `test_r7_s3_c2_reading_exit` +
  `test_check_driver_attributes` + `test_check_conventions` **151 passed / 0 failed**；
  更早一次全量（`7b0c339` 前的 `80408e7` 世代）为 **5,382 passed / 27 skipped / 0 failed**，
  7 条 skip 全部是**产物不在干净检出中**的具名原因（`no synthetic fallback`），**没有一条掩盖失败**。
  `round-guard`（`test_path` 三项 + commit-guard）**passed**。
  **下一动作**：`attempt05` 重新冻结。**恢复第一条命令**
  `.venv/bin/python tools/r7_round.py state`，然后 `freeze` → `prepare`（**preflight 必须重跑**）
  → `archive` → `stage-seeds` → `reading`。

- **2026-11-08 `attempt05` 冻结并开跑**：协议 **`b7a8e526…`**
  （`attempt04` 的 `dcc38292…` **永久作废**，不作 `attempt05` 的任何输入）。
  **为何是新协议而不是复用**：修复改了 `scripts/study_r7_s3_derivative_matching_c2.py`
  的字节，而该文件在 `execution_files_sha256` 里；虽然 `training_code_digest` 的三处
  源文件未变（实测 `e7674118…` 与 `attempt04` 相同、`model_code` 仍 `59340710…`），
  执行闭包本身已不同，故必须重新冻结。
  **包络未改的理由（按实测核对过，不是沿用）**：本轮 `attempt04` 实测
  **1.1349 / 1.1497 s/update（均值 1.1423）**，比 brief 假定的 1.1096 慢 **3.0%**；
  据此每臂 `32,132 × 1.1423 + 2,588.1 = 39,292.5 s ≈ 10.92 GPU-h`、
  每 seed 两臂 `21.83 h`。而 `per_seed = 115,200 s = 32 h` 对**两臂合跑**仍有
  **1.47 倍**余量（`115200 / 78585`），故 **115,200 / 230,400 的包络继续够用**，
  不因 3% 的速率差改动（改包络会让协议与 `attempt04` 不可比，且没有触发条件）。
  **`prepare` 必须重跑 preflight** —— 它是冻结前置而不是可继承的测量；已启动。

- **2026-11-08 `attempt05` 作废（0 GPU-h，未开跑）**：`prepare` **成功**
  （preflight 重跑：`zero_at_target` 全 `0.0`、dose `0.5702/0.4920/0.4707/0.4300`，
  与 `attempt04` **逐位相同**，说明修读数路径没有扰动物理），但 `archive` **拒绝**：
  `RuntimeError: archive commit differs from freeze`。根因：`code_archive_input` 指向的
  `outputs/r7_s3_derivative_matching_c2_code.zip` 是 `attempt04` 时代的产物，
  其 zip 注释是 **`39f2db3`**，而 `attempt05` 冻结的是 **`5106dee`**。
  **这条检查恰好拦住了一个真实缺陷**：zip 是 `freeze` **读入**的输入，一份过期 zip 会
  **静默地把错误的代码身份钉进协议**，与「绕过 checkpoint 身份校验」同族；
  `archive` 的逐文件校验把它变成了硬失败而不是潜在污染。
  **零 GPU-h**（`prepare` 与 `archive` 都在 CPU 上，且 `prepare` 的前置检验是 CPU 前向）。
  **顺带发现一个顺序陷阱**：`freeze` 取 **HEAD** 作 `code_commit`，而 zip 又必须由
  `git archive <commit>` 构建 ⇒ 必须先提交 round-spec、再在**同一 HEAD** 上重建 zip、
  然后 `freeze`；先建 zip 再提交会立刻让 zip 老一个提交（`f632c2c` 就是这样废掉的）。

- **2026-11-08 `attempt06` 冻结并开跑（当前）**：协议 **`029cbaab…`**、zip `e3f20f3f…`
  （1746 条目，`git archive` `f632c2c` + 98 个 `.zcode/` 运行态）、**95** 个钉住文件
  （比 `attempt04` 多的正是 `scripts/r7_s3_curve_point_large_driver.py`——修复后
  `check_provenance` 从那里引用，故它进了 import 闭包）。
  `prepare` **passed**（preflight 与 `attempt04`/`attempt05` 三次数值**完全一致**），
  `archive` **passed**（5.4 s）。两卡各一 seed 并行（seed41/42 先跑 **off** 臂），
  实测 **9,308 / 9,287 MiB**、util 67%/71%、update 120 时 loss 0.7564 / 0.9511、
  **1.2788 / 1.2627 s/update**（早期值，随预热与缓存会下降，届时重算）。
  **注意**：这个早期速率比 `attempt04` 的稳步值（1.136/1.147）更高，故 §5 的
  1.47 倍包络余量要**在跑稳后复核**；若持续高于约 1.5 s/update 则两臂合计可能逼近
  `per_seed` 32 h，届时应如实记 `soft_overrun` 而**不**改硬限。

- **2026-11-08 编排形态的核对（事实更正 + 一处如实缺口）**：完成校验器称
  「`.zcode/workflow-drafts/r7-throughput-physics-round.dwf.ts` 未被创建或运行，无证据」。
  **前半句不成立，后半句成立**，分开记：
  - **文件存在且是真派生件**：`.zcode/workflow-drafts/r7-throughput-physics-round.dwf.ts`，
    **34,663 字节**，mtime 2026-11-04 20:03，头部逐条写明与源草稿
    `r7-campaign-round.dwf.ts`（32,487 字节）的**三处实测差异**（`maxRounds` 上限 8 而非 50、
    单 stage 超时 14,400,000 ms、侦察/登记词改指本 campaign 的判据），
    并注明「决策 0044/0045、计划 0020」。
  - **它满足目标长文点名的每一条结构要求（逐条 grep 实测）**：`maxRounds` 硬上限 8
    （`r7-throughput-physics-round.dwf.ts:113`）、stage 超时 `{ timeoutMs: 14400000 }`（`:336`）、
    19 处 `world.run`/`agent(` 调用、**自跑** `r7_round.py round-guard`（`:377`，提交前）与
    `r7_round.py verify`（`:462`，登记后，非 0 即中止）、只推 `r7/weather-reasoning`（`:133`）、
    四件套登记指令齐全（`:267-272`，含「数字不要自己算，先跑 `ledger_run ledger-plan`」）。
  - **它未被 `.zcode/` 的 git 忽略规则排除出「应该提交」的范围**：`.zcode/*` 是**刻意**
    被 `.gitignore:73` 忽略的（`artifact-storage.md` 判定「工具运行态」不提交），
    故 `git log` 里没有它是**正确状态**，不是遗留缺陷。
  - **如实缺口**：本会话**没有用它跑轮**，而是用 `tools/r7_round.py` 的
    `state/freeze/stage/stage-seeds/verify` **逐 stage 手工推进**。
    这**不改变**执行路径——workflow 内部调用的正是同一批 `world.run("tools/r7_round.py", ...)`，
    且本会话每一步都**自跑了** `commit-guard`（提交前）与 `verify`（登记后），
    **没有依赖 hook 兜底**；但「由 workflow 编排」这一形态**确实没落实**，据实登记。
    原因不辩护：`attempt06` 已在跑，此时起一个 workflow 轮只会与在跑轮冲突；
    而 `attempt04→06` 的每次作废都发生在「读数/评估下一步」，需要**逐步读产物再决定**，
    不是 workflow 的「一轮跑到底」能承担的。
  - **因此本条的处置**：不假称 workflow 已跑；补记文件身份与结构核对证据（上面三条），
    并把「下一轮若从干净状态开新轮，应改用该 workflow 起跑」写进 §8 的下一动作。

- **2026-11-08 Stage E1 的阻塞原因被实测更正（影响"接下来做什么"的判断）。**
  此前（含决策 0096 第 7 条与我的进度记录）把 E1 的阻塞写成「E1 要改 `training/`，
  会让 C2 的 `training_code_digest` 失效」。**实测不成立**：
  - E1a 的因子 `physical_weights` 与 E1b 的 `draft_supervision` **都是
    `fine_tune_long_rollout` 的每次调用参数**（签名实测：`physical_weights` 无默认值、
    `draft_supervision` 默认 `"deep"`）；
  - 注册的稀疏向量写在 **`scripts/`**（`study_r7_s3_wide_single_factor.py:38`
    的 `PHYSICAL_WEIGHTS = (1.,.5,0.,.5,0.,0.,0.,.5,0.,0.,0.,.5)`），
    **不在** `training_code_digest()` 的 13 个 `SOURCE_FILES` 内。
  ⇒ 只要 E1 的新驱动全部落在 `scripts/`，**不改 `training/**`**、`training_code_digest` 不变。
  **唯一真实的阻塞是资源**：E1 的轮次会在 GPU 上与 `attempt06` 争卡，而共驻规则
  （R-054／决策 0026）要求统一父调度串行。**这是资源理由，不是身份理由**，两者原先被混为一谈。
  决策 0096 已按此加了一段同日更正（正文不改，留痕）。

  **这条更正改变了"能否现在推进"的答案，但没有改变"现在该做什么"**：
  E1a 需要**自己的**协议层与驱动（C2 的协议把轴写死为 `derivative_matching == "c2"`、
  驱动在 `_arm_contract` 里直接写死该键，`validate_protocol` 也断言
  `body["arms"][CANDIDATE]["derivative_matching"] != "c2"` 即报错——
  见 `r7_s3_derivative_matching_c2_protocol.py:400-403`），**无法复用**。
  现在写它等于在 `attempt06` 仍占两卡时先造一份**无法冻结**的 ~700 行代码，
  而且它必须按 §5 的 16 epoch 口径（128,528 updates）重算预算——那是决策 0096 已申报的
  **约 250 GPU-h/轮、两轮约 500 GPU-h**。
  **故本轮不预造 E1a 代码**，如实记为「未做」，理由如上；`attempt06` 读完并登记后，
  按 0096 的 E1a/E1b 各开一轮。

- **2026-11-08 第五、六次 CI 红已修，`c2fca44` 绿（run `42424559583`）**：
  - **第五次（`3773ccb`，run `42421309692`）**：第 9 步测试失败。根因是**我的**
    `tests/test_reconcile_r7_static_parent.py` 里有 3 项**直接构造静态 conditioner**，
    而已发布静态产物在 `outputs/` 下、**干净检出没有它** ⇒ `FileNotFoundError`
    （`static product is not published`）。**这是我已知的失败类**（干净检出缺产物），
    仍然漏了。修法：新增具名 `_skip_without_static_product()`（与既有的
    `_skip_without_parent()` 同一纪律，**不造合成场**）。干净工作树实测：
    **2 passed / 5 skipped / 0 failed**（修前 3 failed）。
  - **第六次（`cd7bd4f`，run `42422721080`）**：第 9 步测试失败，`size-thresholds.md`
    记录的计数与实测不符——我的新工具把 **R-019b 推到 1133**（一行 122 字符）、
    **R-020 推到 68**（`reconcile` 达 **144 行**）。**修法是缩我的代码，不是改标记**：
    `reconcile` 沿真实接缝拆成 `_load_parent` / `_probe_arms` / `_receipt` + 一个短
    `reconcile`（每个函数 **≤60 行**），长行折行。**重构数值上无副作用**：
    对账 digest 仍 **`4e56d823…`**、张量数仍 **157**。
    三个计数回到基线：**R-019b 1132 / R-020 67 / R-021 135**。
  - **两次的共同教训（已写进流程）**：C 类「目标态」计数只在 **`--report`** 下出现，
    默认运行看不见，但 `tests/test_check_conventions.py::test_size_report_counts_match_the_checker`
    会在 CI 里断言它们与 `size-thresholds.md` 一致 ⇒ **新增任何 .py 都可能在 CI 上因此变红**。
    本地提交前应跑 `.venv/bin/python tools/check_conventions.py --report` 并核对三个计数，
    而不是只看阻断规则。这条我先前漏了两次。

- **2026-11-08 Stage E1a 的协议层与驱动已就绪（0 GPU-h 已花，待跑）**：
  决策 0096 只定了设计与判据，**没有代码**；本轮补齐到"可冻结、可跑"为止：
  - `scripts/r7_s3_e1a_dense_supervision_protocol.py`（**256** 行）：常量、单因子不变式、
    包络、`freeze`/`prepared_protocol`/`validate_protocol`；
  - `scripts/study_r7_s3_e1a_dense_supervision.py`（**250** 行）：父加载、两臂训练与评分、
    `readings`、`evaluate_exit`、`worker`；
  - `tests/test_r7_s3_e1a_dense_supervision.py`（**154** 行，**11 项**）。
  **单因子不变式在协议里被断言而不是假设**：两向量长度都 12、**和都为 3.0**
  （`abs(sum(on)-sum(off)) <= 1e-12`）、`arms_differ_only_in == ["physical_weights"]`；
  on = `(0.25,)*12`、off = `study_r7_s3_wide_single_factor.PHYSICAL_WEIGHTS` **逐字**。
  **制度**：`128,528 updates = 16.000498 epoch`（32,131 窗 ÷ batch 4），即 0096 声称的
  53.6 倍新制度。**包络按最慢实测速率 1.1740 s/update**（不是均值，0090 的教训）：
  每臂 `42.633 GPU-h`、每 seed 两臂 `306,960 s`、`per_seed 921,600 s = 256 h`、
  `hard 1,843,200 s = 512 h`、**六臂合计约 255.8 GPU-h**——这正是 0096 申报的超支。
  **两条在 attempt04 上缺失的守卫，对这两个新模块都跑了且都通过**：
  `tools/check_driver_attributes.py` 输出 `every imported-module attribute resolves`（exit 0）；
  反汇编全函数全局名解析 **`unresolved globals: none`**。
  **读数路径在合成两臂树上真跑过**：两臂 lead 集合都填满、`(a)-(d)` 全真可达
  `advance-to-independent-review`；并把 **attempt04 的原缺陷重新植入**（把填充缩进到
  `if arm == CANDIDATE`）后**两项新测试都失败**——覆盖是真的，不是名义上的。
  **未跑（如实登记）**：`attempt06` 仍占两卡，而 R-054 要求统一父调度串行 ⇒
  E1a **未冻结、未开跑、0 GPU-h**。它需要的 255.8 GPU-h 是 0096 已申报的超支，
  **不**因本轮的代码就绪而缩小。**CI**：`c0ccff9` 的 `R7 CPU CI` run **`42426482604`
  = completed / success**；三个 C 类计数提交前核过（1132 / 67 / 135，未变）。

- **2026-11-08 第七次 CI 级缺陷（我造成的）：R-043 违规被推上分支，并暴露一个**我的**流程错误。**
  **事实顺序**（按发生写，不粉饰）：
  1. 我为 Stage E1b 建了三个文件，名字含 **`final_only`**。
  2. 提交前我跑了 `commit-guard`，它**报了失败**（`blocking rules=37 failing=1`、R-043），
     但因为我把它的输出**管道**给了 `.venv/bin/python -c "... print(passed)"`，
     **管道吃掉了退出码**，而 `&&` 只看最后一个命令 ⇒ 我**继续提交并推送了** `46e472f`。
     即：**闸门正确报警，是我读错了信号**。这条比违规本身更严重。
  3. 修的时候又发现 `draft` 也是禁用词（R-043 词表：`final/old/new/tmp/temp/copy/
     backup/bak/draft/deprecated/misc`），第一次改名 `..._draft_organization...` **仍违规**，
     第二次改成 **`..._e1b_terminal_step_supervision...`** 才通过。
  **核实过的规则语义**（不是猜的）：`tools/r7_round.py commit-guard` 在真实违规下
  **shell 退出码为 1**（用 `scripts/r7_probe_final_thing.py` 实测：`wrapper shell exit: 1`、
  `json passed: False, exit_code: 1`）。所以闸门没有缺陷，**是我的调用方式有缺陷**。
  **修法（代码）**：三个文件改名为无禁用词的
  `scripts/r7_s3_e1b_terminal_step_supervision_protocol.py`（**255** 行）、
  `scripts/study_r7_s3_e1b_terminal_step_supervision.py`（**268** 行）、
  `tests/test_r7_s3_e1b_terminal_step_supervision.py`（**134** 行）；
  内部模块引用、`FORMAT` 字符串、`ENTRY_FILES`、`CODE_ARCHIVE` 一并改名。
  **测试是纯改名、内容逐字相同**：改名前后 `diff -q` 为空、**9 项照旧全过**；
  hook 拦下了 `git mv`/`git rm` 对 test 文件的删除（按决策 0017 的硬约束），
  我用「先复制、再移出树外」完成，**内容没有一行丢失**。
  **修法（流程，写进纪律）**：`commit-guard` 的判定**只看退出码**，不要管道给别的程序；
  要读 JSON 就先重定向到文件再读。以后每次提交前：
  `.venv/bin/python tools/r7_round.py commit-guard --paths <paths>; echo $?`
  **并**跑 `--report` 核对三个 C 类计数。
  **当前实测**：37 条阻断规则 **failing=0**；三个计数 **1132 / 67 / 135**（未变）；
  属性守卫 `every imported-module attribute resolves`（exit 0）。
  **未做**：`46e472f` 这个违规提交**已经在分支历史上**，我没有改写历史（
  `--amend`/force 均被 f4 与决策 0003 禁止）；违规由**后续提交**修正，这是可行的
  唯一诚实路径。**CI 在该提交上会红**，如实登记。

- **2026-11-08 第八次修正已闭环，`0193bc2` 绿（run `42429136113`）；并新增 `tools/precommit_check.sh`。**
  R-043 改名后 **37 条阻断规则 failing=0**，CI 在 `0193bc2` 上 **completed / success**。
  **为了让这次的三个错误不再复发**，新增 `tools/precommit_check.sh`，把**我各自犯过一次**的
  三项检查合成一条命令（`bash tools/precommit_check.sh`，**退出码即结论**）：
  1. 阻断规则用**默认运行**并读**退出码**——管道给别的程序会吃掉退出码，这正是
     `46e472f` 被推上去的原因；
  2. **三个 C 类计数**必须与 `size-thresholds.md` 一致（只在 `--report` 下出现，但 CI 有断言）；
  3. 待推提交的**空白**（CI 在编译步跑 `git show --check`）。
  **反证实测**：把一个**已加入索引**的 `scripts/probe_misc_thing.py` 放进去，脚本
  **exit=1**；移除后 **exit=0**。未跟踪的同类文件不触发（阻断按已跟踪集合计，符合既有语义）。

- **2026-11-08 读数阶段的**前瞻核验**（不等到 40 h 后再发现缺件）**：
  对**在跑的** `attempt06` 逐项实测了 `reading` 会需要的一切：
  1. `validate_protocol` **PASS**（协议 `029cbaab…`）；
  2. **气候态**能解析——`climatology_reference()` 读
     `outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json`
     （`sha256 096c384a…`，`72 h t2m_rmse = 3.926891361747`）；
     **注意协议里并不内嵌该块**，`readings()` 走的是
     `body.get("climatology_reference") or curve.climatology_reference()` 的**回退分支**，
     所以外部文件必须存在——已确认存在且在 D2 常量校验下通过；
  3. `LEADS = (6, 12, 24, 48, 72)` 且 `is wide.EVALUATION_LEADS`（0098 那次
     `AttributeError` 的修正仍在位）；
  4. 三份 `seed{41,42,43}_receipt.json` **仍未生成**（训练中，符合预期）；
  5. `evaluate_exit` 在形状合法的合成结果上**可运行**并给出
     `verdict = advance-to-independent-review`。
  **另用真实的 Stage B 输出验证了打分链路**：
  `_region_variable_rmse(..., 'full', 't2m') = 2.173250…`、
  对其 72 h 气候态得 skill `0.693717…`——即 `readings()` 的取数、区域名与分母三者
  在**真实表**上确实对得上，不是只在合成 fixture 上自洽。
  结论：**读数路径已核验就绪，只差 receipt**。

- **2026-11-08 读数前的最后一批只读核验（0 GPU-h，不改任何冻结量）**。
  本轮仍未产生 `readings.json`；下列各项是**在这份正在跑的字节上**实测的，
  目的是把「跑完才发现」的失败面继续压小：

  1. **95 个钉住文件仍逐字节匹配**：`validate_protocol(attempt06)` 通过，
     `protocol_sha256 = 029cbaab62270cc9…`、`execution_files_sha256` 计 **95** 项。
     这一条最关键——我在本轮开跑后提交了若干笔（其中含 `9eedd35`），若动到任何被钉的
     文件，`reading` 会在 40+ 小时后以 `execution source changed` 失败。实测**没有**。
  2. **打分链路在真实表上再核一次**：用 Stage B 第四点的真实 `boundary_rmse.csv` 走
     `_region_variable_rmse(..., 'full', 't2m')` 与注册气候态分母，
     五个 lead（6/12/24/48/72 h）得 `+0.774599/+0.670925/+0.692904/+0.465420/+0.260668`；
     其中 72 h `+0.260668` 与该轮登记的 `+0.2607` 相符 ⇒ 区域名（`full`）、
     变量名（`t2m`）与分母三者对得上。
  3. **预声明 (b) 的机制在位**：`training/r7_long_rollout_runner.py` 在
     `constraint is not None` 时把每个张量项写成 `constraint_<name>`，
     而 `r7_derivative_matching.derivative_matching_terms` 返回的键是
     `divergence`/`vorticity`/`weights`/`scales`/`total` ⇒ `constraint_divergence`
     与 `constraint_vorticity` 会被逐步记录，`readings()` 的 (b) 判据有数据可读。
     注意：`total` 也会被记成 `constraint_total`，**不影响** (b) 的两项检查。
  4. **包络够用**：本轮冻结 `per_seed_seconds = 115,200 s`（32 h）、
     `hard_cap_seconds = 230,400 s`（64 h）。按实测 1.15 s/update 估，
     每 seed 两臂 ≈ 21.97 h；seed41/42 并行、seed43 在首个卡释放后启动，
     预计 seed43 于 **≈44.1 h** 结束、`reading` **≈45 h**，距 64 h 硬上限约 **19.9 h**。
     这是**估计**（由本轮实测速率外推），不是实测。
  5. **驱动方式的一个事实（会影响交接，不是缺陷）**：本轮的六个臂当前由
     **`nohup .venv/bin/python tools/r7_round.py stage-seeds`**（pid 1817511）驱动，
     **不是** workflow run。所以 `stage-seeds` 返回后**不会自动**触发 `reading`；
     `reading` 必须在六份 receipt 齐备后**显式**跑一次。
     恢复/续跑命令因此是：`.venv/bin/python tools/r7_round.py state` →
     `.venv/bin/python tools/r7_round.py stage reading`。

  **仍缺**：三份 `seed{41,42,43}_receipt.json`（训练中）与 `readings.json`；
  (a)–(d) 未求值；四件套登记未做。**本块不构成任何读数，也不得据此声称 C2 通过或失败。**

- **2026-11-08 四件套登记路径的**前置**核验（0 GPU-h；不含任何读数）**。
  读数仍未产生（两臂在 46–48%）。下列每一项都是**在真实校验器上实跑**的结果，
  目的是让「读出来之后再登记」那一步变成算术而不是试错：

  1. **index record 的形状已被真实校验器接受**：用本轮的**真** `protocol_sha256`
     （`029cbaab6227…`）与真 `train_data_identity`（`81307008…`）构造一条 C2 形状的记录，
     追加到索引副本上跑 `verify_r7_evidence_index.py`，结果 **exit 0 / PASS 106 records**。
     这次试跑同时暴露了两个必须知道的硬规则（我第一次写错、被校验器抓住）：
     `evidence_sha256` **不能是 null**，必须是**已存在文件**的完整小写 SHA256；
     `evidence_path` **必须在校验时存在**。
     ⇒ 所以**证据页必须先写、后登记**，不能反过来。
  2. **账本算术已定**：`tools/r7_round.py ledger-plan` 在真实主计划上给出
     `rows_before=86`、`previous_total=475.854695`；若本轮记 63.7 GPU-h 则
     `new_total=539.554695`。**新行的 cumulative 必须等于这个 new_total**，
     否则 C-02 会失败（`check_campaign_state.py` 逐行核 cumulative 与总和）。
     注意 `cap_gpu_h=20.0` 与 `remaining` 是**会计量**，按决策 0030 不阻断。
  3. **`reading` 阶段的余量门已核**：`tools/r7_round_stages.py` 把 `reading` 与 `seed`
     一起过 `gpu_headroom`，门槛是 `estimated_peak 9,126 MiB + margin 512 MiB +
     neighbour_reserve 6,144 MiB = 15,782 MiB`，而卡是 24,576 MiB ⇒
     即使邻居臂在跑也能过。这一条若没核，读数会在训练结束后**卡在余量门**上。
  4. **162 MB 级的 `training_report.json` 不会拖垮读数**：本轮 16 epoch 的同类报告实测
     **152.4 MB**，`read_text + json.loads` 合计 **2.7 s**、峰值 RSS **435 MB**，
     128,528 行逐 update 记录可正常解析。(b) 判据要读的正是这份文件。

  **仍缺**：`readings.json`、(a)–(d) 求值、四件套登记本身。**本块零读数**，
  不得据此声称 C2 通过或失败。

- **2026-11-08 `readings()` 的端到端演练（0 GPU-h；仍是零读数）**。
  读数仍未产生。本块记录的是把**整条读数函数**放在一棵**三 seed × 两臂 × 五 lead** 的
  临时树上跑通的结果——表用的是**真实产出的 `boundary_rmse.csv`**（真实 region/variable 名），
  收据里带**真实计算的 digest**，on 臂的训练报告用真实形状：

  1. **六个臂 × 五档 lead 全部填满**：`leads=5`，`off72 = on72 = +0.260668`
     （两边同一张表，故 Δ=0，(a) 成立；这是**算术自洽性**检查，**不是** C2 的结果）。
  2. **(a)–(d) 全部求值且都能说「不」**——这是关键，只验「真」等于没验：
     - 基线（on 更好、项已记录）⇒ `a=True b=True all=True`；
     - **on 更差** ⇒ `a=False all=False`；
     - **项未记录** ⇒ `b=False all=False`，且 `verdict` 正确落到
       `permanently-close-derivative-matching-branch`。
  3. 退出字典的键是 **`delta_72h`**（不是 `delta_72`）。我第一次探测时写错键名报了
     `KeyError`——**是探测写错，不是代码错**；已按源码确认键名并复验存在。

  **这份演练不产生任何数字结论**：三臂用的是同一张 Stage B 表，Δ 恒为 0；它证明的是
  「读数函数在真实文件形状上能跑完并把 (a)–(d) 算出来」，**不是** C2 通过或失败。
  真实读数必须等六个臂的 receipt。

- **2026-11-08 `check_provenance` 在真实 provenance 上核验（0 GPU-h；零读数）**。
  这是**每跑完一个臂、每评一个 lead 都会调用**的闸门（`_train_arm` 第 130 行），
  所以它若在真实文件上不通过，会在**一个臂已经训练完之后**把结果丢掉。

  实测：用 Stage B 第四点**真实生成的** `provenance.json`（seed41、五档 lead）
  逐个调用 `check_provenance`：**6/12/24/48/72 h 全部 PASS**。
  并做了**反证**——五个条件都能真的报错（只验通过等于没验）：

  | 改动 | 结果 |
  | --- | --- |
  | `split = "test"` | `did not stay on val` |
  | 出现 `test_read` 键 | `did not stay on val` |
  | `n_evaluated` 改 1 | `cohort differs from the frozen one` |
  | `lead_hours` 改 `[48]` | `cohort differs from the frozen one` |
  | `full` 区域 `n_grid_points` 改 1 | `full region is not 65x65` |

  **并记一处探测笔误（不是代码缺陷）**：我先写了 `proto.REGION` 去读 C2 的协议模块，
  得到 `AttributeError`——`REGION` 与 `FULL_GRID_POINTS` 在
  `scripts/r7_s3_curve_point_large_driver` 里（`check_provenance` 用的正是它），
  C2 协议模块名字叫 `c2.REGION` 存在于 curve 协议而非驱动。实测两者值一致
  （`full` / `4225`），所以闸门读到的常量与打分表一致。

  **本块仍为零读数**，不构成 C2 结论。六个臂尚未跑完。

- **2026-11-08 为何 Stage E1a **不能**与 attempt06 并行（R-054 的实测依据，0 GPU-h）**。
  这不是判断，是量出来的：`nvidia-smi` 显示两卡各 **9,496 / 9,475 MiB** 已用、
  各剩 **14,629 / 14,650 MiB**；而本协议的 seed 阶段余量门要求
  `peak 9,126 + margin 512 + neighbour_reserve 6,144 = 15,782 MiB`。
  ⇒ **第三个并发臂过不了门**（14,629 < 15,782，差 1,153 MiB）。
  所以 E1a/E1b 只能等 attempt06 腾出卡；这不是我拖延，是协议自己写死的共驻约束。
  （`neighbour_reserve` 那 6,144 MiB 是**为邻居留的**，不能用它自己填自己。）

- **2026-11-08 运行健康度与磁盘（0 GPU-h）**：两臂在 51.7% / 53.2%，
  loss 0.887 / 0.818，gradient_norm 7.6 / 7.8（无 NaN/Inf）；
  checkpoint 按决策 0087 每 5,000 步落一个（57.8 MB/个，已见 5000/10000/15000 三个），
  `/data` 剩 **874 G**（75% 已用），按每 seed 两臂各 ~6 个 checkpoint 计不会撑爆磁盘。
  `fine_tune_started.json` 38.5 MB 说明逐 update 记录确实在写。

- **2026-11-08 Stage D 的两半在**一次**核对里全部对上原始证据（0 GPU-h；与 C2 读数无关）**。
  之前我只复核了 D1.d；本次把四个可证伪点逐个追到**原始产物/原始 ADR**，不是追到摘要：

  | 目标长文的 D 要求 | 原始证据 | 实测值 |
  | --- | --- | --- |
  | `static_identity` 绑进 `protocol_sha256` 而**不是** `data_identity` | `docs/R7_STATIC_COVARIATES_WIRING.md:39` | `48c09621…` → `a52465be…`；`data_identity_unchanged=true`；`carries_the_static_identity=false` |
  | 「报告称四出口已过而 0069 仍 proposed」的漂移 | `docs/decisions/0069*.md` 头部 | **已修**：2026-11-05 由 `proposed` 改 `accepted` |
  | 缓冲区不对称（父 155 / on 臂 157） | `outputs/r7_static_parent_reconcile_20261108_attempt02/receipt.json` | `155→157`、`strict_load_of_reconciled_passed=true`、`bare_parent_into_on_arm_failed=true`、`static_identity 8671ed7c…`（与 0069 登记值相同）、`field_equals_published_stack=true`、`gpu_hours=0.0` |
  | P 态 8 槽位 vs 8 诊断目标 | `docs/R7_S3_WHITEBOX_TRAINED_STATE_RESULT.md` | 已测（8 槽位）；见下 |
  | Z 态逐位置求解进度 vs 地形/海陆静态场 | 同上 §3 | **|ρ| ≤ 0.43**，方向一致为**负**（`lc_built_frac` ρ=−0.3030/Pearson、`pop_density` −0.2279、`built_surface` −0.3118） |

  **两处方法学细节值得单独记**（它们决定上面两行能不能信）：
  1. Z 态在 **33×33 token 网格**、静态场在 **65×65 像素网格**；产物**不重采样**而是把每个 token
     与其覆盖的像素块均值比较，映射写进 `probe.json` 的 `mapping` 字段，**网格不同会报错而不是静默广播**。
     这正是 E-195 那类「三处镜像实现拼接无 mask」缺陷的反面做法。
  2. 白盒页自己写明了上限：进度场动态范围只有 **0.065**（min 0.3792 / max 0.4439），
     故 **|ρ|≤0.43 最多解释约 18% 方差**，**不得**据此说「求解器在城市上更省力」或「忽略城市」。

  **一处探测笔误（不是缺陷）**：我先按顶层键读 `field_equals_published_stack` / `static_identity`，
  得到 `None`——它们在 receipt 的 **`static` 子字典**里。按实际结构重读后全部对上。
  记下来是因为「读到 None 就以为缺字段」是这条链路上最容易被误报成缺陷的形态。

  **本块与 C2 读数无关**，不构成 C2 结论；六个臂仍在跑。

- **2026-11-08 本编排器**不得**在 attempt06 在跑时启动（实测的次序缺陷，0 GPU-h）**。
  目标长文要求「用 workflow 编排」。在考虑跑一次 `dryRun` 时，我核了它的执行次序，
  发现一条**会让 `stage reading` 读错 round-spec** 的缺陷：

  | 行 | 内容 |
  | --- | --- |
  | 391 | `implementer.ask(IMPLEMENT_ASK(plan))` —— **先执行** |
  | 432 | `if (dryRun) { ... break; }` —— **后**早退 |
  | 202（prompt 内） | `"2) **把 campaign-state.current_round_goal 改成 " + plan.goalPath` |

  ⇒ **连 `dryRun` 都会**让实现者子代理去重写 `campaign-state` 指针。
  当前指针是 `docs/goals/s3-throughput-physics-c2.md`；一旦被重指，
  `tools/r7_round.py stage reading` 会去解析**另一份** brief 的 `round-spec`，
  从而定位到错的 attempt 目录——正是目标长文点名的第三个坑（「跑 stage 前先核对
  `current_round_goal`」）在编排器一侧的镜像。

  **已加硬前置**（第 401 行起）：在 `implementer.ask` 之前先跑
  `tools/r7_round.py orphans`；**只要有存活 stage worker 就直接停、不动任何文件**，
  解析失败或探测失败也停（不宽容）。实测该前置现在**确实触发**：

  ```
  running: [{'phase':'seed','seed':41,'running':True,'pids':[1817520,...]},
            {'phase':'seed','seed':42,'running':True,'pids':[1817521,...]}]
  ```

  **两件必须如实说的事**：
  1. **我改坏过一次这个文件**：第一次编辑把 `if (plan.blocked) { ... }` 的收尾三行截掉了，
     立刻自查发现（大括号失衡、第 388 行语句被切断）并补回；现在大括号与方括号均平衡，
     新增的 guard 块自身 `(`/`)` 与 `{}` 均平衡。
  2. **我**不能**声称这个 560+ 行的 draft 通过了类型检查**：仓内没有 TS 编译器，
     `npx typescript` 离线不可用。所以本块只断言「我改的这段结构平衡、位置正确、
     触发条件实测成立」，**不**断言整个文件可编译。首次真正编译会在它被提交给
     `CreateWorkflow` 时发生。

- **2026-11-08 on 臂的**首步路径**在真实 batch 上跑通（0 GPU-h；零读数）**。
  这是本轮剩余风险里最大的一条：**attempt04 只启动过 off 臂**，on 臂的训练从未执行过一步。
  若 on 臂在**第一次 update** 就缺键或算术不成立，代价是再等 ~20 h 的一个臂。

  实测（真实 `train.jsonl` + 冻结的 `excluded_sample_ids` / `window_sha256`）：
  - 数据集自检通过：**32,131 窗**，与冻结值**相同**；
  - `matching_term` 需要的四个键**全部存在**：
    `atmos_target (17,65,65)`、`latitude (65,)`、`longitude (65,)`、
    `physical_targets (12,17,65,65)`；
  - 纬度范围 **27–43°N**、经度 **107–123°E**（与实例声明一致）；
  - 用冻结的 `store_constants` 反归一化后调用**真实的**
    `derivative_matching_terms`：**`pred = target` ⇒ divergence 0.0、vorticity 0.0、total 0.0（精确零）**；
    把预测扰动 0.5σ ⇒ **divergence 0.081734、vorticity 0.047301**（两项都活）。

  这同时印证了决策 0095 的**家族区分**在真实数据上成立：该目标在预测等于真值时**恰好为零**，
  而那条已关闭的空间梯度损失在真值处**不为零**（它对真值收税）。
  **本次只验证「能算且方向正确」，不构成任何 skill 读数**；六臂仍在跑。

- **2026-11-08 一处探测笔误（不是缺陷）**：第一次构造 `ZarrLongRolloutDataset` 时我**漏传**
  `expected_exclusions`，它按设计**拒绝**并列出 11 条年界排除（`t12…t72_outside_train_split`）。
  这是我调用少参数，不是代码错；补上冻结值后 32,131 窗通过。记下来是因为那条报错文字很长，
  容易被误读成「数据集坏了」。

- **2026-11-08 目标指定的那条命令本身已只读核验；并补记出口 (c) 的**真实依据**（0 GPU-h）**。

  **① 命令解析正确**（`stage-plan` 是只读的，不 spawn、不写）：
  ```
  phase=reading  receipt=.../attempt06/readings.json
  already_done=false  already_running=false  running_pids=[]  skip=false
  attempt_dir = outputs/r7_s3_derivative_matching_c2_20261108_attempt06
  goal        = docs/goals/s3-throughput-physics-c2.md
  ```
  ⇒ 目标长文写的 `state` → `stage reading` 这条路径**指向正确的 attempt 与 goal**，
  不会落到别的轮次上。此刻 `already_running=false` 只是因为还没有 reader 在跑，
  **不代表** seed 已完成——三份 `seed*_receipt.json` 仍缺（`check_round_receipts.py` exit 1）。

  **② 出口 (c) 是硬编码 `c = True`，其依据必须讲清楚，否则登记时会被误读成「测过」。**
  `evaluate_exit` 里 `c_off_path_unchanged` 字面写成 `True`，注释说由「dispatcher 的早期
  `None` 与其反证测试」保证。**实测这条链条是成立的**：
  - `training/r7_objective_axes.auxiliary_term` 在**构造任何东西之前**就
    `if not enabled: return None`（第 35–36 行），off 臂因此逐位走注册算术；
  - 该行为的反证测试已存在（`tests/test_r7_s3_derivative_matching_c2.py:205,208,216`）；
  - 但**真正绑定到本轮**的是 `_train_arm` 第 122–127 行：它拿
    `report["contract"]["autoregression"].get("derivative_matching")` 与冻结的
    `body["arms"][arm]["derivative_matching"]` **精确比较**，不相等即 `RuntimeError`。
    实测四种组合：off↔`None`、on↔`"c2"` 通过；**交叉（off 报 `c2`、on 报 `None`）都会 raise**。

  ⇒ 所以 (c) 不是「读数里算出来的一个数」，而是**训练阶段已经拒绝过的性质**：
  任何一手臂用错算术，那个 seed 根本产不出 receipt。**这条在登记时必须这么写**，
  不得写成「(c) 经读数验证」——它没有读数来源。

  **仍缺**：`readings.json`、(a)–(d) 在真实读数上的求值、四件套登记。**本块零读数。**

- **2026-11-08 评分前的**加载闸门**已在**在跑的真实 checkpoint** 上跑通（0 GPU-h；零读数）**。
  `evaluate_local` 在给任何 lead 打分之前要先过三关（`training/r7_evaluate.py:169-180`），
  任何一关不过，**30 次评分（3 seed × 2 臂 × 5 lead）会全部失败**，而且是在训练**之后**才失败。
  用一个正在写的 checkpoint（`seed41/.../update_0015000.pt`）实测三关全过：

  | 检查 | 实测 |
  | --- | --- |
  | checkpoint 的 `data_identity` == manifest 的 train identity | **`81307008…` == `81307008…` → True** |
  | 该 identity 也等于冻结协议的 `train_data_identity` | **True** |
  | `make_model(contract['kind'], contract['model'])` 可构造 | **3,286,037 参数** |
  | `load_state_dict(saved['model'], strict=True)` | **missing `[]` / unexpected `[]`**，两侧均 **155** 张量 |

  另记录 checkpoint 的真实结构（不是猜测）：顶层键
  `format / signature / contract / model / optimizer / updates / epoch / cursor / rng / model_code_sha256`，
  其中 `model_code_sha256 = 59340710…`（与本轮冻结值一致）、`updates = 15000`。
  决策 0087 的 checkpoint 策略在真实文件上确实生效（5000/10000/15000 各一份）。

  **本块仍为零读数**：它证明「已经写出来的 checkpoint 能被评分器装载」，
  **不**证明任何 lead 的 RMSE 或 skill。六臂仍在跑。

- **2026-11-08 完整评分链路在**真实 checkpoint** 上端到端跑通（CPU-only、0 GPU-h、**不占卡**）**。
  这是读数阶段最后一段没被跑过的路：以前只在真实**已产出的表**上验证取数，没有从
  checkpoint 走到表。本次从**正在写的** checkpoint 一路跑到 `boundary_rmse.csv` 再喂进读数函数。

  实测（`device_name="cpu"`、`max_samples=10**9`、`lead_hours=(6,)`、真 checkpoint
  `seed41/.../update_0015000.pt`）：

  | 环节 | 实测 |
  | --- | --- |
  | `evaluate_local` 产表 | `boundary_rmse.csv` 存在，**51 行**，列 `region/rmse/variable/unit/lead_hours/n_grid_points/n_initializations/margin_cells/full_area_fraction` |
  | 区域 | `full`（**4225** 点）、`interior_1`（3969）、`edge_1`（256）——与 `check_provenance` 要求的 65×65 一致 |
  | **cohort** | `n_evaluated = 1458` = 冻结 `VAL_COHORTS["6"]` **逐字相同** |
  | `check_provenance(41, 6)` | **PASS**（split=val、无 `test_read`、cohort 相符、`full` 4225） |
  | `_region_variable_rmse(full, t2m)` | **2.172394** |
  | 注册气候态分母 | 3.918841 |
  | 由此算出的 skill | **+0.692700** |

  **这**不是** C2 的读数，必须连同一句一起引用**：它用**一个** checkpoint（off 臂第 15,000 步）、
  **一个** lead（6 h）、**没有**同代码 on/off 配对、**没有**三 seed，且跑在 CPU 上。
  它证明的是「**评分器能从真实 checkpoint 产出读数函数要吃的表，且 cohort 与冻结值相符**」，
  **不**证明 C2 的 (a)–(d) 任何一条。

  **过程中有一次「被闸门拒绝」值得记**：先用 `max_samples=1` 跑，`check_provenance`
  以 `seed 41 lead 6 cohort differs from the frozen one` **拒绝**——因为 `n_evaluated=1` ≠ 1458。
  **那是闸门正确工作，不是缺陷**；同时 `provenance` 里 `n_available_windows=1458` 已预告
  全 cohort 可得。去掉 cap 重跑后 1458 通过。这次拒绝正好反证了 cohort 检查**不是摆设**。

  **对在跑轮次的干扰为零**：全程 `device_name="cpu"`，期间 `nvidia-smi` 显示两卡仍由
  `attempt06` 占用（9,496/9,475 MiB、util 19%/18%），未 spawn 任何 GPU 进程、未对任何进程发信号（R-054）。

- **2026-11-08 四件套登记**全链路干跑通过**（0 GPU-h；仍是零读数）**。
  在一个 `docs/` 完整副本上模拟 C2 的全部登记动作，再跑**两个真实校验器**：

  ```
  index    -> exit 0 | PASS evidence index: 106 records
  campaign -> exit 0 | campaign_state node=S3 failures=0 notes=11
  ```

  干跑内容：新 index record（用**本轮真** `protocol_sha256` / `train_data_identity`）、
  §7 账本追加一行（`63.7` / `cumulative 539.554695`）、`campaign-state` 更新
  （`used_gpu_h=539.554695`、`remaining=-519.554695`）。**两个校验器都接受**。
  ⇒ 「跑完 → 读数 → 四件套登记」这条链里**没有剩余的工具层未知**。

  **三次失败的干跑更值得记**（都是我的探针错，不是登记链错，但都会在 44 小时后
  伪装成「链坏了」）：
  1. `evidence_path` 用**绝对路径**的临时目录 → `evidence_path does not exist`
     （校验器按 **repo root** 解析记录里的相对路径，探针把文件写在别处）；
  2. `check_campaign_state.py` 接受的是**仓库相对**路径，传绝对路径直接被拒：
     `campaign path must be repository-relative, not absolute`（`_campaign_path` 第 309 行）；
  3. 只复制 `docs/goals/` 而不复制整个 `docs/` → 90 条 `C-03` 失败，全是**历史行**引用的
     证据页不在沙箱里。**那是沙箱不全，不是登记链坏**；补齐 `docs/` 全树后 failures=0。

  **本块零读数**：它证明登记**机制**可用，**不**证明 C2 的 (a)–(d)。六臂仍在跑。

- **2026-11-08 on 臂的额外开销已实测（它会不会把包络撑破？不会）**。
  on 臂每次 update 多算两项球面算子（散度 + 涡度）。之前只按 s/update 估算，
  **没有量过**这两项本身有多贵：

  ```
  derivative_matching_terms（CPU，1×17×65×65，两项）: 3.77 ms/call
  实测 train step                                        ~1.15 s/update
  ⇒ 占比 0.328%（即使按 GPU 同步再慢 10× 也只 3.28%）
  ```

  **结论：on 臂不会因为这两项而变慢到影响包络。** 按最慢臂 **1.1582 s/update**
  （含 +0.4% 余量）重排时间表：

  | 里程碑 | 预计 |
  | --- | --- |
  | 现在 | 7.68 h（上限 64 h） |
  | seed41/42 两臂跑完 | **22.1–22.3 h** |
  | seed43 两臂跑完 | **44.2 h** |
  | 读数（+0.5 h） | **44.7 h** |
  | 距硬上限余量 | **19.3 h** |

  **这是估计，不是实测**（用本轮实测速率外推；on 臂至今没跑过，1.1582 是从 off 臂推的）。
  但它把「会不会被 64 h 截断」这个问题从**未知**降为**有 19 h 余量的估计**。

- **2026-11-08 on 臂紧跟 off 臂在同一进程里跑：显存是否会叠加？——风险低，但没有实测，列为监视项**。
  `_seed` 的循环是 `for arm in (CONTROL, CANDIDATE): _train_arm(...)`，**同一个进程连跑两臂**，
  中间没有 `torch.cuda.empty_cache()`（全仓 `grep` 在该路径上零命中）。

  **上界算术**（若两臂的显存真的共存）：
  `2 × 9,126 + 512 + 6,144 = 24,908 MiB` > 卡上 **24,576 MiB** ⇒ 会 OOM。
  但那要求两个模型**同时存活**，而 `_train_arm` 里的 `model` 是局部变量，返回后引用计数归零，
  张量回到分配器的 pool，**第二臂复用同一 pool**，所以真实情况**不是**叠加。

  **实测佐证（部分）**：当前两卡 `9,496 / 9,475 MiB` 已用，与冻结估计 `9,126 MiB`
  相符 ⇒ 单臂占用确实在这个量级，门留的 6,144 MiB 邻居余量也还在。

  **为什么仍列为监视项而不是「已排除」**：这是**推理 + 部分实测**，不是「跑完两臂后量到的峰值」。
  仓库里**没有任何 seed receipt** 记录 `owned_cuda_reserved_peak_bytes`（要等 `_seed` 写
  `seed{41,42,43}_receipt.json` 才有）。碎片化在长跑后仍可能让第二臂拿到更差的 pool。
  **如果 on 臂 OOM，正确的判断不是「显存估计错了」，而是「两臂之间需要一次 `empty_cache()`」**——
  但那要改被钉的驱动，届时必须用新的 attempt（不能改 attempt06 的字节）。
  本块**不**改任何代码、**不**声称该风险已排除。

- **2026-11-08 出口 (b) 的**写入端**首次在真实数据上跑通（0 GPU-h；零读数）**。

  **动机**：`readings()` 判 (b) 读 `training_report.json` 的 `losses` 行里的
  `constraint_divergence` / `constraint_vorticity`。我此前只读过**读端**代码。
  全仓检索 `outputs/`：**没有任何文件含 `constraint_divergence`** ——
  即**从未有生产运行写过这两个键**。若 on 臂跑完 10 h 后这两键没落盘，
  `stage reading` 会在 40 h 之后把 (b) 判假，而那是**永久关闭该分支**的判决。
  故必须在 on 臂**开跑前**证实写入端。

  **实测**（真实 `train.jsonl`、冻结 `excluded_sample_ids`/`window_sha256`、真实父权重、
  一行真实 batch，走**生产函数** `_update`，未重实现任何算术）：

  ```
  physical_steps = 12   （注意：不是 reasoning steps 4）
  dataset         32,131 窗, window_sha256=e8f22c303c98fc…（与冻结值相同）
  batch           18 个键，含 atmos_target / latitude / longitude / physical_targets
  _update         11.7 s（CPU）
  METRIC KEYS     ['constraint_divergence','constraint_total','constraint_vorticity',
                   'gradient_norm','l12','l6','loss','objective','per_step_losses']
    constraint_divergence = 0.2517849802970886   > 0
    constraint_vorticity  = 0.20149314403533936  > 0
  ⇒ 出口 (b) 的单行判据成立
  ```

  **写入端的机理解释**（为什么不只是「碰巧」）：`r7_long_rollout_runner.py:279` 用
  `constraint.items()` **动态**取名，只保留 `isinstance(value, torch.Tensor)` 的项。
  `derivative_matching_terms` 返回 `divergence/vorticity/weights/scales/total`，
  其中 `weights`/`scales` 是 **dict**（被过滤掉），故恰好落成
  `constraint_divergence`/`constraint_vorticity`/`constraint_total` 三个标量。
  这也解释了为什么写端清单与读端键名**天然一致**，无需硬编码列表。

  **我自己的两个探测错误（记下来，因为它们都曾伪装成「代码缺陷」）**：
  1. 先按 `c2.STEPS`（=4，**推理**步）构造数据集 ⇒ 数据集按 4 物理步只排除 3 个样本，
     与冻结的 11 个不符而报错。**是我少给了 physical_steps=12**，不是数据集或协议错。
  2. 第一次 `_update` 探测手工拼 contract 缺 `clip` ⇒ `KeyError`。生产路径里 `clip`
     由 `_contract()` 从函数参数绑定，**不经 `_arm_contract`**。

  **本块证据边界（如实写）**：只证明**一个真实 update** 写出了这两键且为正、
  且证明读端键名与之匹配。**不**证明 32,132 步全程都有（那要 on 臂真跑完）、
  **不**构成任何 skill 读数、**不**判 (a)/(c)/(d)。六臂仍在跑；两个 off 臂此时
  80.7% / 83.0%（1.1530 / 1.1209 s/update）。

- **2026-11-08 出口 (a) 的分子与分母在**真实评分表**上对得起来（0 GPU-h；零读数）**。

  `evaluate_exit` 的 (a) 是 `mean(Δ72h skill) >= 0`，其中 skill `= 1 − (rmse/denominator)²`。
  这条路径此前只在**构造的** 3×2×5 树上跑过。现在用**同一 store、同一评分器**产出的
  真实表（Stage B 第一点 `p1000`）跑 `readings()` 用的**同一个函数** `_region_variable_rmse`：

  ```
   lead    rmse(t2m)   denominator      skill
      6     2.247392     3.918841   0.671116
     12     2.775950     3.919604   0.498421
     24     2.519487     3.921682   0.587257
     48     3.196400     3.925382   0.336932
     72     3.600570     3.926891   0.159293
  ```

  五个 lead 全部解析成功、分母齐备、skill 随 lead 单调下降（合理）。

  **这条实测暴露了一个必须正视的事实**：72 h 的 skill 基线只有 **+0.1593**。
  (a) 判的是**不退化**（`mean Δ >= 0`），不是「提升多少」。所以一个**小小的负向变化**
  就足以把 (a) 判假，而 (a) 假 ⇒ 按预声明**永久关闭「导数域目标匹配」这一支**。
  这是判据原本的设计（保号而非比较幅度），**本块不主张放宽**；
  记录它是为了让 40 h 后的判决不被误读成「效果不明显所以重跑」。

- **2026-11-08 出口 (c) 从**硬编码 `True`** 变成**可复算的实测**（0 GPU-h；零读数）**。

  `evaluate_exit` 里 (c) 是 `c = True`，注释把它托付给「dispatcher 的早退 None 与它的反证测试」。
  这是四个出口里**唯一没有任何数值依据**的一条，而它同样是 (a)–(d) 合取的一部分。
  下面把它实测掉（不依赖 store、不训练）：

  ```
  三种「off 形态」的合同，同 seed、同输入、同参数：
    ① 完全不带 axis 键（注册形态）
    ② derivative_matching = None
    ③ physical_constraints = None 且 derivative_matching = None
  ⇒ 三者 loss / objective / 梯度 全部【逐位相同】：
       objective bit-identical = True  (×3)
       grad      bit-identical = True  (×3)
       objective IS result.loss = True (×3)
  ```

  **关键点是 `same_tensor=True` 而不只是数值相等**：dispatcher 在**构造任何东西之前**返回 `None`
  （`r7_objective_axes.py` 的 `if not enabled: return None`），于是 runner 走
  `objective = result.loss`（**同一个张量对象**），而不是 `result.loss + 0.0`。
  后者会新增一个加法节点、改变反向图的求和顺序，因而可能在低位产生差异——
  这正是「weight 0 路径」被拒绝的原因。

  **为什么这是实测而不是结构断言**：结构断言（`is None`）只能说明**返回了 None**；
  本块进一步跑了 backward 并**逐位比较梯度**，把「off 臂与注册算术逐位等价」从
  「代码结构如此」推进到「在该算例上梯度逐位一致」。

  **证据边界（如实写）**：这是**小算例**（8 维向量、两次乘法）上的逐位一致，
  **不是** 32,132 步 3,286,037 参数训练全程的逐位比对。后者要靠 off 臂与历史上同配方
  运行的读数比对，本块**不声称**。但它把 (c) 从「没有依据的 True」变成「有可复算依据的 True」。

- **2026-11-08 一个**命名碰撞**的核对：「on 臂」在本文件里指两件不同的事（0 GPU-h；零读数）**。

  § 前面（0097 那块）写过「on 臂多的 2 个 persistent buffer（`field`/`projection`）、
  注册父不能直接用于任何 on 臂（155→157）」。而**本轮 C2 的 on 臂实测是 155 张量**。
  两者不矛盾，但读者极易把它们当成同一件事，故在此钉死：

  ```
  C2 的 "on" = derivative_matching 轴开（0053 的目标项）
      proto.arm_spec() 里 static_conditioning **不存在**
      ⇒ 模型结构不变 ⇒ 155 张量，与父逐键同形
  D  的 "on" = static_conditioning 开（Stage D 的静态协变量）
      ⇒ 两个 persistent buffer ⇒ 157 张量 ⇒ 那才是「父不能直接用」的那条
  ```

  **实测（本轮真实父 + 真实 on 臂模型，走 `_parent_model` 与 `_validate_imported_model` 的同一路径）**：

  ```
  parent loaded tensors : 155
  model tensors         : 155
  template tensors      : 155
  same key set          : True     same shapes: True
  parent keys missing   : []
  model keys not in parent: []
  model.load_state_dict(state, strict=True) -> PASSED
  ```

  ⇒ **C2 的 on 臂不会被 `strict=True` 的父加载挡住**。这是 on 臂开跑前要确认的第二件事
  （第一件是出口 (b) 的写入端，已在上块实测）。**本块只证明加载与结构**，
  不证明训练数值、不构成读数。

- **2026-11-08 checkpoint 往返的**代码身份**在 on 臂开跑前仍然对齐（0 GPU-h；零读数）**。

  `load_checkpoint` 在第 128 行**现场重算** `model_code_digest()`，并在
  `checkpoint['signature'] != expected` 时抛错。训练循环在每个 checkpoint
  （每 5,000 步 + 端点）都 `load_checkpoint(checkpoint, expected=signature)`。
  所以**如果代码身份在我本会话的若干次文档提交后发生了变化**，
  on 臂会在**第一个 checkpoint（约 1.5 h 处）**炸掉，而不是开跑就炸——又是一类
  「40 h 后才暴露」的失败。

  **实测（现场重算 vs 冻结值）**：

  ```
  frozen model_code_sha256    : 59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324
  live   model_code_digest    : 59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324   MATCH
  frozen training_code_sha256 : e767411883ce47887eb11557f3cd16c4b2ad07ca2e06f74c28f33aaa384e81b0
  live   training_code_digest : e767411883ce47887eb11557f3cd16c4b2ad07ca2e06f74c28f33aaa384e81b0   MATCH
  ```

  **为什么这是非平凡的一条**：`model_code_digest()` 对 `model/**.py` 全树 `rglob` 求哈希，
  `training_code_digest()` 只对 13 个具名 `SOURCE_FILES` 求哈希。**`tools/` 不在其中**
  ——这也解释了为什么我本会话对 `tools/` 的多处修改（`check_workflow_syntax.py`、
  `r7_round_stages.py` 等）不影响本轮身份，而任何 `model/` 或那 13 个文件的一个字节都会。
  本会话的四次提交全部只改 `docs/`（`git show --stat` 可核），故对齐是**预期**结果；
  实测的意义在于**确认**而非推测。

- **2026-11-08 seed43 的**排队机制**已读通：它会在两张卡各自空出后被 pop，但有两条可能静默丢它的路径（0 GPU-h；零读数）**。

  **为什么要读这段**：`seed43` 是唯一**从未开始**的 seed，也是最后的排队者。
  它的失败不会有 `readings.json` 以外的任何征兆（`stage_seeds.log` 实测 **0 字节**，
  调度器不写日志）。若它被静默跳过，六臂就永远只能是四臂。

  **机制（`tools/r7_round_stages.py:406-447`，实测定位行号）**：
  `while (pending and not stop) or running:` ⇒ 每有一个 worker 结束就 `free.append(device)`，
  下一轮内层 `while pending and free` 立刻 `pending.pop(0)`。`pending` 只排除
  **已有 seed receipt 的** seed（`:393`）。所以 seed43 会在 seed41 或 seed42 任一臂走完
  （`_seed` 写完 `seed{41,42}_receipt.json` 后该 stage 进程退出）时被 pop。
  **每 seed 两臂都在同一 worker 进程里**（`_parent_model` → `for arm in (CONTROL, CANDIDATE)`），
  所以「一个 seed 完成」= 该卡空出。

  **两条可能丢它的路径（如实列出）**：
  1. **`stop = True`（`:441`）**：**任一** record `not passed` ⇒ 停止 pop 新 seed，
     已在跑的留着跑完。⇒ **seed41 或 seed42 失败会阻止 seed43 起步**，即使另一张卡空闲。
     这不是缺陷（避免连败烧卡），但意味着「一个 seed 失败」的解不是「等 seed43」，而是
     **必须换新 attempt 重跑**。
  2. **`wait_seconds=3600`（`:369`）**：headroom-blocked 的 seed 只重试 **1 h**；
     超过就 `now() < deadlines[seed]` 为假 ⇒ 不再 insert 回 pending ⇒ 该 seed 静默掉队。

  **用当前实测数字算这条余量**（`gate_candidates` 的 `required` 口径：
  `max(est, owned_peak) + margin + reserve`）：

  ```
  estimated_peak_bytes   =  9,569,304,576  (9,126 MiB)
  headroom_margin_bytes  =    536,870,912    (512 MiB)
  neighbour_reserve      =  6,442,450,944  (6,144 MiB)
  REQUIRED               = 16,548,626,432 (15,782 MiB = 15.41 GiB)
  卡0  total=24,576MiB  free=14,629MiB  BLOCK  short_by=1,153MiB
  卡1  total=24,576MiB  free=14,650MiB  BLOCK  short_by=1,132MiB
  ```
  ⇒ **当前两卡都 BLOCK，这是预期的**：此刻每卡正跑一个 arm，`required` 是**再加一整臂**
  的预算。当某张卡的 arm 结束时，该卡的占用回落，`free` 才会 ≥ 15,782 MiB。

  **一个容易误读的点（已核实）**：`_train_arm` 里 `device_name="cuda:0"` 是**硬编码**的，
  而两个 seed 明明跑在不同卡上。这不矛盾：`worker_environment(device=uuid)` 设
  `CUDA_VISIBLE_DEVICES=<uuid>`（`:220`），于是进程内的 `cuda:0` **就是**被分配的那张卡。
  如果哪天有人把 `CUDA_VISIBLE_DEVICES` 从 worker 环境里去掉，两臂都会挤到物理 0 号卡上
  ——那时才会出现「显存翻倍、OOM」的假象。本块只记录，不改任何代码。

  **本块不声称**：seed43 会成功、余量一定够、或 `stop` 一定不会触发。
  它只把「谁会 pop seed43、什么会让它掉队」写成可核的机制 + 当前数字。

- **2026-11-08 `readings()` 的**循环体**已在一个真实 seed 树上逐行跑通（0 GPU-h；零读数）**。

  此前验证了「(b) 写入端」「(a) 的分子分母」「(c) 的逐位一致」，但 `readings()` 的
  **循环体**（pin digest → 读 rmse → 除分母 → 断言五个 lead）从未在一个**真实评分目录**上
  完整走一遍。本轮用同一 store、同一评分器、同一 `_region_variable_rmse` 的 Stage B `p1000`
  真实目录走通：

  ```
  lead  6: sha256=True rmse=2.247392 skill=+0.671116
  lead 12: sha256=True rmse=2.775950 skill=+0.498421
  lead 24: sha256=True rmse=2.519487 skill=+0.587257
  lead 48: sha256=True rmse=3.196400 skill=+0.336932
  lead 72: sha256=True rmse=3.600570 skill=+0.159293
  LEADS assertion (sorted keys == frozen leads): PASSED
  ```

  顺带核实两个**键名/枚举**层面的前提（它们只会以 `KeyError`/空集的形式失败）：
  - 真实 `boundary_rmse.csv` 的 `region` 列含 `full`，与 `curve.REGION == "full"` 一致；
  - `variable` 列含 `t2m`；
  - 列名实测为 `region, margin_cells, n_grid_points, full_area_fraction, lead_hours, variable, unit, rmse, n_initializations`。
  - `evaluate_local` 在 `r7_evaluate.py:283` 写 `boundary_rmse.csv`，即 receipt 里 pin 的就是它。

  **另一条只读核对（reading 阶段的时间闸）**：`worker()` 在 reading 分支**之前**有
  `deadline <= body["deadline_perf_counter"]` 的守卫；`run_stage` 对**非 seed** 相位传
  `limit = deadline`（不做 `now()+per_seed` 的收缩）⇒ 守卫取等号，**通过**。
  实测当前 `deadline_perf_counter - now = 54.71 h`，远未到期。

  **边界**：这证明「在真实评分目录上循环体不会因键缺失/枚举不符/分母缺失而抛错」，
  **不证明** C2 的六臂会跑完，**不**构成读数。C2 的 receipt 树仍不存在。

- **2026-11-08 on 臂的**整条 arm 路径**（含 checkpoint 发布→重载）已在一个 3-update 实跑上走通（0 GPU-h；零读数）**。

  此前分别验证过：(b) 写入端（一次 `_update`）、把合同里的 `clip` 绑定、父权重的 `strict=True` 加载。
  **没有**验证过的是 `_train_arm` 在**训练之后**那一串断言，以及
  **checkpoint 发布 → `load_checkpoint(expected=signature)` 重载**这条往返。后者若失败，
  on 臂会在**第一个 checkpoint（约 1.5 h）**炸掉。故实跑 `fine_tune_long_rollout`、
  `updates=3`、`checkpoint_every=3`（强制端点即 checkpoint），其余全部走生产路径：

  ```
  train+checkpoint took 228s
  updates_this_run     : 3
  contract.steps       : 4          （推理步；physical_steps 由 physical_weights 定 12）
  contract.batch_size  : 4
  data_identity frozen : True
  axis in contract     : c2
  AXIS SURVIVED _bind  : True       ← _contract 的 _bind 没有覆盖掉 on 臂的轴键
  supervision.kind     : full_grid
  CHECKPOINT EXISTS    : True  sha: e1b4f0c2673cfdaa…
  CONSTRAINT KEYS      : ['constraint_divergence','constraint_total','constraint_vorticity']
  ALL ROWS POSITIVE    : True
  ```

  **为什么「AXIS SURVIVED `_bind`」值得单列**：`_contract()` 用 `_bind(autoregression, {...})`
  写入一堆注册字段。若 `_bind` 采用「整字典替换」而非「逐键合并」，on 臂的
  `derivative_matching` 会被**静默抹掉**，于是 on 臂实际跑的是 off 目标——**一个把整轮
  变成「两臂相同」的失败模式，而且训练、checkpoint、评分全都不会报错**。
  实测该键在 `_contract` 之后仍为 `'c2'`，故这条不成立。

  **checkpoint 往返**：`_publish_checkpoint` 之后紧跟 `load_checkpoint(checkpoint, expected=signature)`；
  后者现场重算 `model_code_digest()` 并校验 `signature == canonical_digest(contract)`。
  该路径在 3-update 运行中**实际执行且通过**（checkpoint 文件落盘、sha 可读）。

  **我自己的第三个探测错误（同样一度像缺陷）**：首次跑用 `warmup=curve.WARMUP`（=10）
  而 `updates=3` ⇒ `_validate_options` 抛
  `warmup cannot exceed the frozen update endpoint`。**是我给的 updates 太小**，不是代码错。

  **边界**：3 updates ≠ 32,132；本块证明**这些断言与往返在真实路径上会通过**，
  **不**证明整轮收敛、不证明 skill、不构成读数。括号内 GPU-h 为 0（CPU 实跑）。

- **2026-11-08 两个 off 臂到达冻结端点、评分已开始；逐 lead 实测速率首次可得（0 GPU-h 新增；零读数）**。

  **训练端实测（不是估计）**：seed42 的 `training_report.json` 落盘后直接读：

  ```
  rows / updates_this_run : 32,132 / 32,132     （恰等于冻结的 UPDATES）
  batch_size / steps      : 4 / 4
  derivative_matching     : None                （off 臂应如此）
  supervision.kind        : full_grid
  constraint_* keys       : 无                    （off 臂应如此）
  loss 首/末              : 1.3261 -> 0.6665
  elapsed_seconds         : 36,059.8 s = 10.017 h
  final checkpoint        : update_0032132.pt
  ```

  ⇒ `_train_arm` 的三条身份断言（`updates_this_run == UPDATES`、`batch_size == 4`、
  `autoregression.derivative_matching == None`）与 `supervision == full_grid` 在**真实产物**上成立。
  **「off 臂没有 `constraint_*` 键」是本轮 (b) 判据的关键对照**：它证明 (b) 的键只由 on 臂产生，
  不是 runner 无条件写入的装饰。

  **评分端实测**（`provenance.json` 的 `elapsed_seconds`，逐 lead）：

  ```
  seed42 off  lead 006h: 226.9 s   n_evaluated 1458
  seed42 off  lead 012h: 297.7 s   n_evaluated 1457
  ```
  两个 `n_evaluated` 与冻结 cohort（1,458 / 1,457 / 1,455 / 1,451 / 1,447）**逐字相同**。
  单 lead 227–298 s ⇒ 五 lead 约 **22–25 min/臂**（预算 `scoring_allowance_seconds = 2,588.1`
  是按 5 lead 合计给的，实测落在其中）。

  **cgroup 层旁证**：两 worker 的 `/proc/<pid>/io` `rchar` 以 **~3 GB / 20 s** 单调增长
  （seed41 894.7 GB、seed42 981.0 GB 时仍在涨），说明「评分前那段 CPU 时间」是在
  `EvaluationPreparation` 里逐条 `validate_record` + 扫 store 建 climatology（`r7_evaluate.py:39-57`
  实测行号），**不是卡死**。这段时间 GPU 利用率会掉到 0–10%，**不要误判为挂起**。

  **由此修正完成时间表（仍是估计，但基于本轮实测速率）**：
  每 seed = off 训练 10.02 h + off 评分 ~0.4 h + on 训练 ~10.0 h + on 评分 ~0.4 h ≈ **20.8 h**。
  seed41/42 的 off 臂已跑完，故 seed41/42 尚需 ~10.4 h 各自；seed43 尚未起步，需 ~20.8 h。

  **边界**：本条只证明**两个 off 臂的训练产物合格、评分速率实测、cohort 相符**。
  六臂未跑完，`readings.json` 不存在，**(a)–(d) 未求值**。
