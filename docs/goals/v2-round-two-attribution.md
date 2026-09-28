# 目标：V2 第二轮——四臂 × 三种子，把「时空输入」与「位置化读写」拆开并做容量控制

本文件是 goal 模式（`/goal <objective>`）的**长文目标源**。§0 是可粘贴的单段 objective，
§2 是逐条核对的交付物，§5 是预算与停止条件。

## 0. objective（可直接粘贴，单段）

> 本轮目标：在 r7/weather-reasoning 上做主模型 V2 第二轮——用四臂 × 三种子把上一轮的两个混淆拆开，回答「位置依赖本身是否超出容量带来改善」（起点 SHA 开工时用 git rev-parse HEAD 记录）。细节与判据见 docs/goals/v2-round-two-attribution.md，不需要用户再授权。 四臂（同一 M2 store、同一 400-update 协议族、同一个新的 protocol_sha256、全部从零训练）：A process_pooled（两开关全关）、B process_spacetime_only（只开 spacetime_inputs）、C process_spacetime_rwa（两个都开）、D process_rwa_capacity_control（开 spacetime_inputs 与 positional_process_readout，但读取的 query 按位置池化因而位置无关；与 C 同一模块、同一参数、算力实测后如实报告）。种子 41/42/43，共 12 个 run。 交付物（逐条核对，未做就写未做）： D1 新增「池化 query」构造开关（默认关、bool 校验与既有开关同风格），默认路径不得产生任何新参数。 D2 逐位等价测试必须实跑并通过（tests/test_r7_switched_path_equivalence.py，23 digest 覆盖三族 forward/rollout/自适应/streamed BPTT 的全参数梯度）。 D3 四臂臂配对实测：跨四臂共享张量逐位相同、锚臂 state_dict 应用到其余三臂、applied/ignored 计数写进产物——是实测不是假定。 D4 池化被钉住：池化时各输出位置逐位相同、非池化时有位置差异；训练后位置探针 D 的 spread 约 0、C 显著非 0。 D5 一轮有界实验：写入新输出目录（不得写进 round one 的 outputs/r7_71_72_m1_rwa/），协议在每个 seed 第一步前冻结、12 个 run 的 protocol_sha256 相同、只读 val、test 封存。 D6 配对比较用 #60 比较器、depth 键 0，至少给出 B−A、C−B、C−D、D−B 四对，逐 seed 同号才算 improved/worsened，其余 unresolved，负面照写。 D7 证据文档（新建，不覆盖 round one 那份）+ 实测 GPU-h + 四臂参数/FLOPs 实测值 + 重跑稳定性观察；C−D 若未获支持就明说。 D8 提交（治理层进版本控制）并核对 CI；skipped 的实验 workflow 是 commit-message 标签门控，不算失败。 纪律：不用 test 做方法选择（test 已被读过）；不新增阈值、不放宽已冻结判据；C−D unresolved 或反向就写「位置依赖本身未获支持」，不得换变量或网格去凑赢面；不做第五臂、不做 RW-B/M3/M4/M5；不新增数据下载；不合并 main、不 force push、不租 GPU、不写 issue。 预算与停止：本轮 ≤1.0 GPU-h（第二批授权 ≤24 GPU-h，已用 1.389），单次实验 ≤30 min；超出即停并如实记录。逐位等价或池化钉住失败、12 个 run 的 protocol_sha256 不一致、check_conventions 阻断违规时停止并报告。 结束时报告：启动 SHA、改动面逐文件、实跑过的验证与结果、protocol_sha256 与实测 GPU-h、每对配对的 improved/worsened/unresolved、未做的事、下一项的第一个具体动作。不要自行宣布目标完成。

## 1. 现状（已核实，开工前请再核一遍）

- 分支 `r7/weather-reasoning`，HEAD `14c7a24`（已推送，与 `origin/r7/weather-reasoning` 同）。
  开工时用 `git rev-parse HEAD` 记录真实起点。
- 上一轮（round one）已把两个**默认关**的开关做进主模型：
  `spacetime_inputs`（`model/weather_forecaster_r7.py:24-27,44,59`）与
  `positional_process_readout`（`model/process_readout_r7.py:35`，接线在
  `model/process_forecast_r7.py:142-161`）。
- round one 的有界实验（M2 双月段 store，seed 41/42，400 updates，val only）：
  参数 2,799,779 → 2,968,259（**+168,480 / +6.02%**），前向 FLOPs 12.8438e9 → 13.9046e9（**+8.26%**）；
  两臂共享 115 个张量**逐位相同**；实测 **0.1846 GPU-h**；
  #60 比较器 val 上 **42 improved / 11 worsened / 32 unresolved**（85 格）。
  产物在 `outputs/r7_71_72_m1_rwa/`（`seed<N>/training/<arm>`、`seed<N>/evaluation/<arm>`、
  `seed<N>/protocol.json`，协议字段名是 **`protocol_sha256`**），**是本轮不可覆盖的证据**。
- round one 自己记下的三条硬局限（`docs/R7_71_72_M1_AND_RWA.md` §7）：
  ①两个开关一起开，分别贡献 unresolved；②两臂参数/算力不对齐，赢面不能归因于机制；
  ③只有 2 个 seed（不是显著性）。
- 本轮就是把这三条按仓库既有纪律逐条消掉，**不引入新机制**。

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 新增「池化 query」**构造**开关（默认关，bool 校验与既有开关同风格）；默认路径**不得**产生任何新参数 | 改动 + 默认路径参数计数对比 |
| D2 | `tests/test_r7_switched_path_equivalence.py` 仍**逐位相同**（23 digest，覆盖三族 forward/rollout/自适应/streamed BPTT 的全参数梯度），且必须**实跑**而不是声称 | 测试名与结果 |
| D3 | **四臂臂配对实测**：跨四臂共享张量逐位相同、锚臂 `state_dict` 应用到其余三臂、`applied/ignored` 计数如实 | 协议/结果里的 `arm_pairing` 字段 |
| D4 | 池化必须被钉住：池化时**各输出位置逐位相同**、非池化时位置间有差异；训练后位置探针 D 的 spread ≈ 0、C 显著非 0 | 两条测试 + 探针数字 |
| D5 | 一轮有界实验：**新输出目录**（不得写进 round one 的目录）、协议在每个 seed 第一步前冻结、12 个 run 的 `protocol_sha256` **相同**、只读 val、test 封存 | 产物目录 + `protocol_sha256` |
| D6 | #60 比较器配对（**depth 键 0**，与 round one 一致）：至少 B−A、C−B、**C−D**、D−B 四对，逐 seed 同号才算 improved/worsened，其余 unresolved，负面照写 | `merged_result.json` 的 comparisons |
| D7 | 证据文档（新建，不覆盖 round one 那份）：起点 SHA、协议 `protocol_sha256`、实测 GPU-h、四臂参数/FLOPs 实测、逐变量每时效 RMSE 与胜负计数、C−D 的判定（若未获支持就明说）、重跑稳定性、`scientific_claim: false`、`limitations`、未做的事、下一项 | 文件路径 |
| D8 | 提交（治理层进版本控制）并核对 CI；17 条实验 workflow skipped 是 commit-message 标签门控，**不算失败** | commit SHA + run id |

## 3. 判据与证据来源

- 判据只有 **#60 比较器的逐 seed 同号配对**（`training/r7_coreasoning_compare.compare`，
  `depth=COMPARATOR_DEPTH=0`），**不新增任何阈值**；`science_criteria_refs` 只列文档指针。
- 证据来源：`docs/R7_71_72_M1_AND_RWA.md`（round one 的数与局限）、
  `docs/R7_B2_MULTISEED.md`（协议冻结 + 同号规则 + unresolved 语义）、
  `docs/R7_67_PUBLICATION_PROTOCOL.md`（test 已读过的纪律）、
  `docs/rules/testing.md`、`docs/rules/ci-and-verification.md`。
- **test 全程封存**：M2 的 test 已被读过，**不得**用于方法选择，也不得重称未见。
- 「算通过」的最小定义：D1–D4 的测试实跑通过、D5 的协议与 val-only 在产物里可查、
  D6 的配对有明确定论（含 unresolved）。**skip 不算通过，被取消/排队的 run 不算通过。**
- 若 C−D 在三个种子上 **unresolved 或方向相反**：结论写「位置依赖本身未获支持」，
  **不得**换变量、换网格、换时效去凑一个赢面，也不得放宽判据。

## 4. 实施顺序（依赖顺序，不可跳步）

1. **recon**：记录开工 SHA；确认 round one 产物仍在且不被改动；确认
   `test_r7_switched_path_equivalence.py` 当前通过。
2. **D1 池化开关**：在 `PositionalProcessReadout` 增加构造参数（默认关），池化时
   **不要**把位置编码混进 query——取 `query_norm(context)` 的按位置均值后广播，
   使读取与位置无关；`ProcessForecastCoReasoner` 透传该开关（与既有 bool 校验同风格）。
3. **D2 等价性**：实跑逐位等价测试（新增开关默认关后仍须 23 digest 全同）。
4. **D4 池化钉住**：先写测试（池化→各位置逐位相同；非池化→位置间有差异），
   再训练；训练后补位置探针（D 的 spread ≈ 0、C 显著非 0）。
5. **D6 前先把臂配对做严**：四臂同 seed 构造、共享张量逐位相同、锚臂权重拷进其余三臂，
   把 `applied/ignored` 计数写进产物；**这是解释四臂表的前提**。
6. **协议冻结**：四臂定义、种子 41/42/43、400 updates、优化器/验证节奏/案例集、
   `test_read: false` 全部写进 `protocol.json`，在第一臂第一次 `optimizer.step()` 之前落盘并回读校验。
7. **跑实验**：新输出目录（建议 `outputs/r7_71_72_round_two/`），一卡一进程、每个 seed 一个进程；
   跑完 finalize 汇总；记录墙钟与**实测 GPU-h**（各 run `elapsed_seconds` 求和 ÷ 3600）。
8. **参数/FLOPs 实测**：用 `training/r7_budget_audit.py:43 count_parameters` / `:47 count_forward_flops`
   报四臂实测值，含 D 与 C 的差（池化把 q 投影从 N 个位置降到 1 个位置，FLOPs 会略低，如实写）。
9. **配对比较**：`compare(table, baseline=..., depth=0)`，逐对报告 improved/worsened/unresolved；
   `beats_baseline_everywhere` 为 false 时不得写成全面更好。
10. **重跑稳定性**：A/C 在两个共有 seed（41/42）上与 round one 的排序对照（round one 没有 seed 43，
    如实说明）；GPU 浮点非确定性下**不要求**权重逐位一致。
11. **证据文档 + 提交 + 核 CI**，报告按 AGENTS.md 的九项。

## 5. 预算与停止

| 项 | 值 | 依据 |
| --- | --- | --- |
| 本轮自设上限 | **≤1.0 GPU-h** | 12 run × ~0.046 GPU-h（round one 实测 0.1846/4 run） |
| 第二批授权 | ≤24 GPU-h，已用 **1.389**（M2 1.204 + round one 0.1846）→ 余 ≈22.6 | `docs/goals/full-auto-campaign.md`，**不得自行扩大** |
| 单次实验墙钟 | ≤30 min（每 seed 一个进程，四臂顺序训，约 11–12 min） | 同上 |
| 新产物 / decoded | ≤100 GiB / ≤256 GiB（本阶段） | 同上 |

停止条件（任一触发即停并如实记录）：

1. 本轮 GPU 用量达 1.0 GPU-h（或开始逼近 22.6 余量）；
2. 单次实验超 30 min；
3. 逐位等价测试（D2）失败——说明新开关动了默认路径；
4. 池化钉住（D4）失败——说明 D 臂不是位置无关，C−D 的解释失效；
5. 12 个 run 的 `protocol_sha256` 不一致；
6. `python tools/check_conventions.py` 出现阻断违规；
7. 需要任何未授权动作（新数据下载、租 GPU、合并 main、force push、写 issue）时，
   停止该分支并写「需要用户授权」。

**开工前置（本机卫生，非判据）**：`tests/fixtures/r7_equivalence_recipe.py`（round one 遗留、
未跟踪、hook 拒绝代理删除）若仍在，本机 `check_conventions` 会报 **R-044 一条阻断命中**，
此时**不得**宣称「conventions 干净」，如实写「有一处待用户清理的未跟踪遗留文件，CI 不受影响」。
`.zcodeignore` 保持未跟踪是项目既定状态，不要动它。

## 6. 与 planner 草案的差异（我改了什么，为什么）

1. **输出目录**：草案把 12 个新 run 写进 round one 的 `outputs/r7_71_72_m1_rwa/`。
   那是已发表的证据，且 `training/r7_experiment.py:106-109` 的 `save_exclusive` 对已存在路径
   **抛错**（不会静默覆盖）→ 改为**新建**输出目录，round one 目录只读。
2. **协议字段名**：草案写 `protocol_digest`；实际产物里是 **`protocol_sha256`**（已从
   `outputs/r7_71_72_m1_rwa/seed41/protocol.json` 读出核对）。
3. **比较器 depth**：草案写 `depth=3`，那是把 `REASONING_STEPS=3` 混了进来。
   round one 用的是 `COMPARATOR_DEPTH = 0`（`scripts/study_r7_71_72_spacetime_rwa.py:72,531,535`），
   且比较器 `on_incomplete="raise"` 会 fail-closed → 必须用 0。
4. **产物布局**：草案写 `seed_<arm>_<seed>/`；实际是 `seed<N>/training/<arm>` 与
   `seed<N>/evaluation/<arm>`。
5. **池化位置**：草案在 `query = query_norm(context) + position_encoding(...)` **之后**池化；
   改为在**位置编码之前**取均值并广播，使控制臂的读取与位置无关（否则位置编码的均值被混入，
   控制臂仍位置无关但语义不干净）。两种都必须由 D4 的两条测试钉住。
6. **「D 与 C 的 FLOPs 差异 <1%」是断言不是判据**：改为实测并如实报告
   （池化会把 q 投影的输入位置数从 N 降到 1，前向 FLOPs 会略低于 C）。
7. **臂配对要覆盖四臂**：草案只说「同 seed 构造」；改为**实测**四臂共享张量逐位相同 +
   锚臂 `state_dict` 应用到其余三臂 + `applied/ignored` 计数（round one 的等价性论证正是靠这条，
   不能因为臂多了就退回假设）。
8. **`--help` 之类的弱验证**换成可机械核对的产物字段（目录/字段/digest 一致性）。
9. **稳定性对照**：草案只比 A/C 的 seed 41/42；如实说明 round one 只有 41/42，无 seed 43 可比。

## 7. 明确不做

- 不做 2×2 全因子的第五臂（`spacetime_inputs=False + readout=True`）——交互项留到 C−D 有定论之后；
- 不做 RW-B（门控 + 局部状态）、M3/M4/M5；
- 不复用 round one 的 checkpoint（新协议 `protocol_sha256` 不同，禁止把不同 digest 的格子拼进一张表）；
- 不读 test、不新增判据/阈值、不放宽已冻结判据；
- 不新增数据下载、不扩基线家族、不合并 main / force push / 租 GPU / 写 issue；
- 不为了让报告好看而改测试或挑变量。

## 8. objective（可直接粘贴）

> 本轮目标：在 r7/weather-reasoning 上做主模型 V2 第二轮——用四臂 × 三种子把上一轮的两个混淆拆开，回答「位置依赖本身是否超出容量带来改善」（起点 SHA 开工时用 git rev-parse HEAD 记录）。细节与判据见 docs/goals/v2-round-two-attribution.md，不需要用户再授权。 四臂（同一 M2 store、同一 400-update 协议族、同一个新的 protocol_sha256、全部从零训练）：A process_pooled（两开关全关）、B process_spacetime_only（只开 spacetime_inputs）、C process_spacetime_rwa（两个都开）、D process_rwa_capacity_control（开 spacetime_inputs 与 positional_process_readout，但读取的 query 按位置池化因而位置无关；与 C 同一模块、同一参数、算力实测后如实报告）。种子 41/42/43，共 12 个 run。 交付物（逐条核对，未做就写未做）： D1 新增「池化 query」构造开关（默认关、bool 校验与既有开关同风格），默认路径不得产生任何新参数。 D2 逐位等价测试必须实跑并通过（tests/test_r7_switched_path_equivalence.py，23 digest 覆盖三族 forward/rollout/自适应/streamed BPTT 的全参数梯度）。 D3 四臂臂配对实测：跨四臂共享张量逐位相同、锚臂 state_dict 应用到其余三臂、applied/ignored 计数写进产物——是实测不是假定。 D4 池化被钉住：池化时各输出位置逐位相同、非池化时有位置差异；训练后位置探针 D 的 spread 约 0、C 显著非 0。 D5 一轮有界实验：写入新输出目录（不得写进 round one 的 outputs/r7_71_72_m1_rwa/），协议在每个 seed 第一步前冻结、12 个 run 的 protocol_sha256 相同、只读 val、test 封存。 D6 配对比较用 #60 比较器、depth 键 0，至少给出 B−A、C−B、C−D、D−B 四对，逐 seed 同号才算 improved/worsened，其余 unresolved，负面照写。 D7 证据文档（新建，不覆盖 round one 那份）+ 实测 GPU-h + 四臂参数/FLOPs 实测值 + 重跑稳定性观察；C−D 若未获支持就明说。 D8 提交（治理层进版本控制）并核对 CI；skipped 的实验 workflow 是 commit-message 标签门控，不算失败。 纪律：不用 test 做方法选择（test 已被读过）；不新增阈值、不放宽已冻结判据；C−D unresolved 或反向就写「位置依赖本身未获支持」，不得换变量或网格去凑赢面；不做第五臂、不做 RW-B/M3/M4/M5；不新增数据下载；不合并 main、不 force push、不租 GPU、不写 issue。 预算与停止：本轮 ≤1.0 GPU-h（第二批授权 ≤24 GPU-h，已用 1.389），单次实验 ≤30 min；超出即停并如实记录。逐位等价或池化钉住失败、12 个 run 的 protocol_sha256 不一致、check_conventions 阻断违规时停止并报告。 结束时报告：启动 SHA、改动面逐文件、实跑过的验证与结果、protocol_sha256 与实测 GPU-h、每对配对的 improved/worsened/unresolved、未做的事、下一项的第一个具体动作。不要自行宣布目标完成。
>
> 实测 1620 字符（上限 4000）。
