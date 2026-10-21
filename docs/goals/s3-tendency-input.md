# S3：显式 6 h tendency 输入（单因素模型输入表示）——检验 72 h interior_32 t2m skill 与外生 (a)–(e) 出口

<!-- round-node: S3 -->

<!-- round-spec: {"arm": "s3-tendency-input", "driver": "scripts/study_r7_s3_tendency_input.py", "output_stem": "r7_s3_tendency_input_20261021", "control_record": "s3-anchor-isolation-result", "reference_record": "s3-wide-interior-dose", "seeds": [41, 42, 43], "device_mode": "dual", "test_path": ["tests/test_r7_s3_tendency_input.py", "tests/test_r7_s3_tendency_input_protocol.py"], "code_archive": "outputs/r7_s3_tendency_input_code.zip", "commit_paths": ["model/coarse_encoder.py", "model/weather_forecaster_r7.py", "model/process_forecast_r7.py", "scripts/r7_s3_tendency_input_parent.py", "scripts/r7_s3_tendency_input_support.py", "scripts/r7_s3_tendency_input_text.py", "scripts/study_r7_s3_tendency_input.py", "tests/test_r7_s3_tendency_input.py", "tests/test_r7_s3_tendency_input_protocol.py", "docs/goals/s3-tendency-input.md", "docs/goals/main-model-climatology-campaign.md", "docs/decisions/0049-explicit-tendency-input-round.md", "docs/decisions/README.md", "docs/rules/size-thresholds.md"], "commit_message": "feat(r7): add the explicit tendency-input single-factor driver [model-digest-change]", "register_paths": ["docs/R7_S3_TENDENCY_INPUT_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-tendency-input.md"], "register_message": "feat(r7): register the tendency-input run as a new evidence record"} -->

状态 **完成：mixed / not-candidate；预声明出口 (a)(b) 未满足，支已永久关闭；证据已登记**（2026-10-21）。唯一主计划
`docs/goals/main-model-climatology-campaign.md`；本轮承接 2026-10-21 §8 重新激活块与决策
`0049-explicit-tendency-input-round.md`，只改模型输入表示，不改数据、父、预算、损失或任何已关轴。

## §0 Objective（可粘贴；单段）

> 在 docs/goals/s3-tendency-input.md 接续 S3/index86/165.012856 GPU-h、2023 test 未评分 r0、S4 未启动：在冻结的 2400-update wide/interior_32 long-rollout recipe 上只改一个模型输入表示因子 tendency_input False→True——候选 encoder 输入从 [X_{t-1},X_t]（34 通道）变为 [X_{t-1},X_t,D_t]（51 通道，D_t=X_t-X_{t-1}，同归一化输入空间），新增一个独立的 17→dim 卷积 patch（kernel=stride=patch_size=2，bias=False，权重零初始化），参数增量 17×192×2×2=13,056 并作为已声明的容量混杂如实报告；模型其余层、persistence decoder base、数据/source、损失、2400 updates、LR 2e-5、warmup 10、K=4、depth=4、dim=192、batch 1、clip 1、deep K-axis supervision、物理权重 (1,.5,0,.5,0,0,0,.5,0,0,0,.5)、seed 41/42/43、评分域 interior_32 与气候态分母全部逐字不变；因为模型代码改变，model_code_sha256 变为 60de3b2d…，已用 committed migrate_r7_parent_state.py 在新 revision 下重跑父迁移到全新排他目录（receipt 73c019dd…），逐 seed 迁移 state digest 仍为 7c416ccb…/857d8804…/61fdac58…，扩张态只多出已发布的 anchor buffers 与零 tendency patch；对照 record 为 pinned s3-anchor-isolation-result（无 tendency input、注册反向、final optimizer weights，不重训），外部参照为 pinned s3-wide-interior-dose；预声明出口（不新增阈值）：(a) 候选 72h interior_32 t2m skill 三 seed 全 >0；(b) 同 seed 候选−pinned 控制 72h ΔRMSE 三 seed 全 <0；(c) 候选 48h skill 三 seed 全 >0；(d) 候选−控制在 6/12/24h 的 seed 均值 ΔRMSE ≤0；(e) 桥：pinned 控制对 pinned s3-wide-interior-dose 的 15 个 seed×lead RMSE 差落在 −4.8e-6…+1.8e-5 K；全真→candidate_state needs-review 并提请独立复核，不自动进 S4；任一 (a)–(d) 假→永久关闭显式 tendency-input 支（不扫 tendency 定义/缩放/符号、不加 seed、不改 updates/LR/warmup/K/depth/dim、不与已关轴组合、不换名）；(e) 假只写 identity_unresolved=true，是工程身份失败而非科学判定；null/inert 仅在最终 tendency_patch 权重恰好全零或每个配对 Δ 在记录精度下恰好 0.0 时成立，小而非零的效应按 (a)–(d) 判读、不得改标 null；预算 per-seed 21600 s、planned 43200 s（软）、hard 86400 s，软超继续并记 overrun，硬截断即 attempt 失败不算科学通过；test 不读、r=0、S4 不启动、scientific_claim=false。

（objective 字符数：1642；单段。）

## §1 起点与实质差异

- 起点工作分支 `r7/weather-reasoning`；累计 **165.012856 GPU-h**、index **86**（gradient-surgery 轮已登记）。
- 起点证据：`docs/R7_S3_GRADIENT_SURGERY_RESULT.md`（mixed/negative，支关闭）、
  `docs/R7_S3_ANCHOR_ISOLATION_RESULT.md`（配对控制 N，registered gradient）、
  `record:s3-wide-interior-dose`（pinned 外部参照 A）、
  `docs/goals/main-model-climatology-campaign.md` §8 的 2026-10-21 重新激活块。
- **为什么选这个因子**：主计划的模型输入/表示方向。显式 tendency 是 GraphCast/FuXi/Pangu/Stormer/
  NeuralGCM 一类系统里常见的物理量；本轮把它作为**输入**与两个历史帧一起送入 encoder，而不是
  作为输出参数化或损失项。一手检索状态见 §3；检索只作线索、不作证据。
- **单因素边界**：控制是 pinned 旧模型/旧训练代码读数（`s3-anchor-isolation-result` 的 control 臂），
  候选是新模型代码三 seed；两臂共享同一父 state、同一数据、同一 2400 updates 与同一评分域，
  但 `model_code_sha256` 与 training-code 版本不同，wall-clock 与 run-to-run 方差也不同
  （limitations 明写）。
- **不是已关轴的重命名**：不改容量（dim/depth）、不改架构开关、不改损失/监督/权重、不改训练
  反向或数据制度；只把一个新的零初始化线性输入通路接到 encoder patch 上。

## §2 交付物

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮冻结设计（本文件 §3） | 本文件、`docs/decisions/0049-explicit-tendency-input-round.md` |
| D2 | `tendency_input` 模型开关（encoder 独立 17→dim patch，bias=False，零初始化；RNG 隔离） | `model/coarse_encoder.py`、`model/weather_forecaster_r7.py`、`model/process_forecast_r7.py` |
| D3 | 新 revision 的父迁移 pin + 零 patch 扩张与逐位控制 | `scripts/r7_s3_tendency_input_parent.py` |
| D4 | 新驱动 + 支持模块 + prose 模块（逐字保留预算/权重/seed/父/评分域/气候分母） | `scripts/study_r7_s3_tendency_input.py`、`scripts/r7_s3_tendency_input_support.py`、`scripts/r7_s3_tendency_input_text.py` |
| D5 | 新 CI-safe 测试（合成树/monkeypatch/反证，不读 `outputs/`） | `tests/test_r7_s3_tendency_input.py`、`tests/test_r7_s3_tendency_input_protocol.py` |
| D6 | 新协议 / 新输出 / code.zip | `preparation_protocol.json`、`protocol.json`、`code.zip` + `archive_receipt.json` |
| D7 | 候选臂三 seed 训练与评分、配对读数 | `seed{41,42,43}_receipt.json`、每 seed 五 lead 的 `boundary_rmse.csv`、`readings.json` |
| D8 | 登记与精确 CI | `docs/R7_S3_TENDENCY_INPUT_RESULT.md`、index record、brief、账本行、§8、campaign-state |

## §3 判据与设计决定（本轮冻结）

科学门与验收形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与 D4 口径；本轮
**不新增**任何数值阈值，也不宣布通过。

- **唯一改动因子**：`tendency_input` 从 `False`（控制，pinned 注册模型）改为 `True`（候选）。
  - 候选 encoder 输入：`[X_{t-1}, X_t, D_t]`（51 通道），`D_t = X_t − X_{t-1}`，同一归一化空间。
  - 新增 `tendency_patch = Conv2d(17, dim, kernel_size=stride=patch_size=2, bias=False)`，
    权重零初始化；构造在重绕 RNG 流中，**不改**任何既有参数的初始化。
  - 前向：`z = patch(concat(X_{t-1}, X_t)) + tendency_patch(pad(D_t))`；线性 padding 对差分
    与对每帧一致。
  - 参数增量：`17 × 192 × 2 × 2 = 13,056`（3,286,037 → 3,299,093）；这是**已声明的容量混杂**，
    没有 parameter-matched 控制臂。
- **保持不变**：模型其余层与开关、persistence decoder base、数据/source/split、损失定义、
  2400 updates、LR 2e-5、warmup 10、K=4、depth=4、dim=192、batch 1、clip 1、deep K-axis
  supervision、物理权重、seed 41/42/43、评分域 `interior_32`、气候态分母、五 lead 与 17 变量
  报告口径。
- **父与身份**：`model_code_sha256` 变为 `60de3b2d…`；用 committed
  `scripts/migrate_r7_parent_state.py` 在全新排他目录
  `outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02` 重跑模型迁移（receipt
  `73c019dd…`）。逐 seed 迁移 state digest 必须仍是 `7c416ccb…/857d8804…/61fdac58…`；扩张态
  必须只新增 `backbone.climatology_anchor.table`、`backbone.climatology_anchor.ready` 与
  `backbone.encoder.tendency_patch.weight`，且 tendency patch 恰为零；扩张 digest 为
  `7105700b…/b1695349…/691058ad…`，anchor init 仍为 `de0e76…`。迁移 receipt 的
  `semantics_equal=false` 是预期的：当前 `CoarseEncoder`/`NativeAtmosForecaster` 即使开关为
  False 也带 `tendency_input` 属性，模块语义 digest 因此至少发生这一处变化；tensor
  keys/shapes、strict load 与迁移 state digest 全部命中。
- **控制臂 N（pinned，不重训练）**：`record:s3-anchor-isolation-result` 的 control 臂
  （`outputs/r7_s3_anchor_isolation_20261015_attempt01/`），spec **逐字等于**本轮
  `control_spec()`（无 `tendency_input`、无 `decoder_base_mode`），`model_code_sha256=3fe747b2…`、
  `training_code_sha256=8ad77d93…`、final optimizer weights。候选 spec 必须恰等于
  `dict(control_spec(), tendency_input=True)`。
- **正/负控制（测试逐条断言，失败即 raise）**：
  1. 同 seed 下开/关两臂的全部既有 state tensor 逐位相等；只多出
     `backbone.encoder.tendency_patch.weight`，形状 `(dim,17,2,2)`、bias=None、权重全零。
  2. 初始化时两臂的 forecast / initial_forecast / draft_forecasts / final_correction 逐位相等；
     把 patch 权重改为非零后输出**不相等**（证明支路确实被读取）。
  3. 用 forward hook 捕获分支输入，断言它逐位等于 `X_t − X_{t-1}`（含 padding 情形），
     并用 encoder 自己的 patch/blocks/norm 重放得到逐位输出；反符号得到不同输出。
  4. `history_steps=1`、非布尔开关被拒绝；`off` 路径不带新键、不带前向分支。
  5. CPU smoke：合成 fixture 的 2-update `fine_tune_long_rollout` 接受
     `tendency_input=True`；contract/model 开关不一致被拒绝；闭轴组合与篡改 contract 被拒绝。
- **预声明出口（不新增阈值）**：
  - (a) 候选 72h `interior_32` t2m skill 在 seed 41/42/43 **全部 >0**；
  - (b) 同 seed 候选−控制 72h ΔRMSE **三 seed 全部 <0**（负值 = 候选更优）；
  - (c) 候选 48h skill **三 seed 全部 >0**；
  - (d) 候选−控制在 6/12/24h 的 **seed 均值** ΔRMSE **≤0**；
  - (e) 桥：pinned 控制 N 的 15 个 seed×lead RMSE 对 pinned A 的差落在 `−4.8e-6…+1.8e-5 K`。
  **同时满足 (a)–(e) → 提请独立复核（不自动进 S4）**；任一 (a)–(d) 不满足 → **永久关闭**
  显式 tendency-input 支：不扫 tendency 定义/缩放/符号、不加 seed、不改
  updates/LR/warmup/K/depth/dim、不与已关轴组合、不换名重开。
- **Null/inert**：只有两种精确情形记 null——最终 `tendency_patch` 权重恰好全零，或每个
  配对 Δ 在记录精度下恰好 0.0。因为 `D_t` 是两个既有帧的线性函数，支路不增加信息，null 是
  真实可能结果；**不**加 seed、不调 tendency 定义以制造效应。小而非零的效应按 (a)–(d) 判读。
- **工程身份门**（只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据）：
  `model_code_sha256 == 60de3b2d…`；父 migration receipt `73c019dd…` 与逐 seed checkpoint/state/
  expanded/anchor init digest 匹配；控制/参照 pins 匹配且未重训；候选 spec == 控制 spec +
  一个开关；数据/source/windows/supervision/气候态分母 pins 匹配；GPU assignment/gates 合法；
  test 2023 未读；r=0；S4 未启动；`scientific_claim=false`。
- **证据页 `docs/R7_S3_TENDENCY_INPUT_RESULT.md` 必须写**：逐 seed 72h skill 与对控制 delta、
  48h skill、6/12/24h seed 均值 delta、桥、全 17 变量 × 5 lead、最终 tendency patch 摘要
  （numel/nonzero/all_zero/abs_max/l2_norm）与 update-0 诊断、checkpoint/parent/protocol/code
  身份、实际 GPU-h/wall/peak/disk/network，以及 limitations：跨代码候选-vs-pinned 控制、
  +13,056 参数容量混杂、零初始化使 update-0 等价、单一 tendency 定义/尺度、线性信息冗余导致的
  null 可能、三 seed 为一致性证据非显著性、单实例/ROI/季节、GPU 共驻与不可逐位复现。
- **报告口径**：五个 lead（6/12/24/48/72h）与全 17 变量的 `interior_32` RMSE 都照报；
  不跨单位平均、不挑赢家。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

- **执行预算（冻结）**：`per_seed_seconds=21600`、`planned_seconds=43200`（软预算）、
  `hard_cap_seconds=86400`（硬上限，为计划的 2 倍）。软超预算继续并记 overrun，硬截止即停
  attempt 并全额保留 `failure.json`；失败 attempt 不在原目录 resume、不缩短重试。
  这是**单臂**候选 envelope；控制与参照均 pinned 不重训。
- 只训练 **候选一个臂 × 三 seed**，由 `stage-seeds` 在双卡上按决策 0047 调度；每阶段 spawn
  前只读 `gpu_gate`，**禁止**对非本实验进程发送任何信号（R-054）。
- 计划成本约 3 ×（2400 updates + 五 lead 评分）≈ 6–8 GPU-h；tendency patch 只增加一次
  `dim×17×2×2` 卷积/前向，开销按回执逐阶段记账（GPU-h/overhead 照报）。
- **前置**：决策 `0049` 与 §8 重新激活块必须在 freeze 前已提交；`test` 未读、r=0、S4 不启动。
- **停止条件**：任一 (a)–(d) 不满足即永久关闭该支；null/inert 同样关支（按结果类型登记）；
  工程失败 / 身份门 / hard cap 只算 attempt 失败，修复仅限已确认工程原因，不改因子或判据。

## §5 明确不做

不重训练 N 或 A、不重跑已登记轮次、不读 test、不扫 tendency 定义/缩放/符号、不加 seed、不改
2400 updates/LR/warmup/K/dim/depth/监督域/物理权重、不 reset 输出头、不改 change_scale 或
output scale、不引入 parameter-matched 控制臂、不引入随机顺序 RNG、不与 anchor/feedback/
architecture/容量/K 轴组织/损失加权/EMA/空间梯度/梯度手术组合、不换名重开、不改
`data/raw|interim|processed` 或归档快照、不合成替代数据、不改科学合同/冻结证据/判据、
不 main 合并/发版/关 issue、不 paid/rental/exclusive/force/mirror/破坏性/cron。

## §6 进度与交接

- **已定**：设计（§3）；登记名 `docs/R7_S3_TENDENCY_INPUT_RESULT.md` / record
  `s3-tendency-input-result`；output stem `r7_s3_tendency_input_20261021`。
- **已完成（2026-10-21，实现轮）**：
  1. `model/coarse_encoder.py` 新增 `tendency_input`：独立 `Conv2d(17, dim, 2, stride=2,
     bias=False)`、零初始化、RNG 重绕；`model/weather_forecaster_r7.py` 与
     `model/process_forecast_r7.py` 透传；`model_code_sha256` 变为 `60de3b2d…`。
  2. committed 迁移工具在全新排他目录 `outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02`
     下重跑：receipt `73c019dd…`；逐 seed 迁移 state digest 不变。
  3. `scripts/r7_s3_tendency_input_parent.py`：pin 迁移/扩张 digest，`expand_parent_state` 逐位
     安装父 state、anchor table 与零 tendency patch，并跑合成输入表示正/负控制。
  4. 新驱动/支持/prose 模块与新 CI-safe 测试（合成树/monkeypatch/反证，不读 `outputs/`）。
  5. 本长文、决策 0049 与 campaign-state 指针已写。
  6. `.venv/bin/python tools/r7_round.py round-guard` **passed**（两个新测试文件 +
     `tests/test_check_conventions.py` 共 175 passed；`commit_guard` 对 14 个实现路径
     exit 0）。另以 CPU 只读方式对三 seed 跑了真实迁移 + `expand_parent_state`：参数
     3,299,093、扩张 digest `7105700b…/b1695349…/691058ad…`、anchor `de0e76…` 全部命中。
- **已完成（2026-10-21，运行轮）**：
  1. `freeze` 成功：attempt `outputs/r7_s3_tendency_input_20261021_attempt01/`，protocol
     `f25f3eb7…`、code.zip `9b75245d…`（1477 entries，ZIP comment == code commit
     `5ef6c46…`）；prepare/archive 各 success。
  2. `stage-seeds` 双卡完成 seed41/42/43（2400 updates + 五 lead，`exit_code=0`，无
     `failure.json`）；`reading` success，`readings.json` `968bf0fb…`。
  3. **结果**：72 h `interior_32` t2m skill 三 seed 全负（−0.174997/−0.276587/−0.155372）；
     72 h 候选−控制 ΔRMSE 非全负（+0.000555/−0.000969/−0.001230 K）；48 h skill 三 seed
     全正（+0.061523/+0.028408/+0.113681）；6/12/24 h seed 均值 Δ 全 ≤0
     （−0.000840/−0.001171/−0.002572 K）；桥 15/15 一致。⇒ (a)(b) false、(c)(d)(e) true、
     `predeclared_exit_met=false`、`identity_unresolved=false`；**支已按预声明永久关闭**。
     这不是 null：三 seed 的 `tendency_patch` 全部 13056 个元素非零，L2 0.130…0.151。
  4. **成本**：process 合计 28,099.457 s = **7.805404837 GPU-h**；墙钟 20,548.289 s；
     soft/hard overrun 0；网络 0；磁盘 19.57 GiB；累计 165.012856 → **172.818261 GPU-h**。
  5. **登记**：证据页 `docs/R7_S3_TENDENCY_INPUT_RESULT.md`、index record
     `s3-tendency-input-result`（`mixed`/`not-candidate`）、brief、账本行、§8、campaign-state。
  6. **CI 如实**：实现 commit `15ba518` 的 CI 被同分支并发推送取消；`97463bf` 修复并发会话
     引入的 FY 反例与尺寸标记；最终 code commit `5ef6c46…` 的 `R7 CPU CI` **39571181164
     completed/success**。并发提交把我们新增的 driver/tests 带进了 `ee5b99f`，文件内容未变，
     已在证据页披露。
- **未做（如实）**：未读 2023 test；未启动 S4；未做 parameter-matched 控制臂；未扫 tendency
  定义/缩放/符号；未把结果称为泛化或 SOTA。
- **下一动作（S3）**：支已关闭；下一轮必须先完成所需的一手检索，提出一个新的、与已关闭轴
  不重叠且可独立检验的因子，写新 round brief 并预声明出口后再 freeze/GPU。
- **恢复第一条命令**：`.venv/bin/python tools/r7_round.py state`。
- 不宣布科学通过、不称泛化/SOTA。
