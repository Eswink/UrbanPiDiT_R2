# 目标：V2 第三轮——给时空输入（M1）补容量控制臂

本文件是 goal 模式的目标长文源。§0 是可粘贴的单段 objective；交付物见 §2；预算与停止见 §5。
结构可用 `python tools/check_goal_brief.py --brief docs/goals/v2-round-three-m1-attribution.md` 机械核对。

## 0. objective（可直接粘贴，单段 1801 字符，实测 ≤4000）

> 本轮目标：给主模型的时空输入（M1）补容量控制臂——round two 测得 B−A 在 t2m 上是 −1.0…−2.0 K 的大效应，但 B 只比 A 多 19,488 参数且没有容量控制，本轮要把它拆成「信息」与「容量/偏置」两部分（起点 SHA 开工时用 git rev-parse HEAD 记录）。细节与判据见 docs/goals/v2-round-three-m1-attribution.md，不需要用户再授权。 本轮允许使用 goal-loop 技能（§9 进度与停止纪律）与 planner 委派（方案设计）；两者只做纪律与设计，不产生证据、不做科学判定。 四臂（同一 M2 store、同一 400-update 协议族、同一个新的 protocol_sha256、全部从零训练，seed 41/42/43，共 12 个 run）：A process_pooled（全关）、B process_spacetime_only（开 spacetime_inputs，真实字段）、E process_spacetime_constant（模块在场但字段是同形状零张量）、P process_spacetime_shuffled（字段确定性批内错配）。于是 B−E 是「信息」、E−A 是「模块容量/偏置」、B−A 是两者之和。 交付物（逐条核对，未做就写未做）： D1 新增 constant 与 shuffled 两个模式，默认 fields。 D2 替换必须发生在模型内部：dataset 与 rollout 仍携带真实字段，DECLARED_MODEL_INPUTS 与输入路径一致性测试不得改变。 D3 默认路径逐位不变：23 digest 等价测试实跑通过（含反证）。 D4 改动前先在当前树冻结 spacetime_inputs=True 路径的 forward 与反向 digest，改动后断言逐位相同（回归钉住）。 D5 预登记 primary：第一次 optimizer.step() 之前把 primary（t2m 五个时效；B−A、B−E、E−A、P−A 四对）与判定文字写进 protocol.json 并冻结 digest，跑后不得修改。 D6 12 个 run 的 protocol_sha256 全同、只读 val、test 封存、实测 GPU-h 与四臂参数/FLOPs（E 与 B 必须逐张量相同、算力相同）。 D7 比较器出各对计数，逐 seed 同号才算 improved/worsened，全 17 变量 × 5 时效照报。 D8 证据文档 docs/R7_71_72_ROUND_THREE.md：primary 判定、全变量汇总、重跑稳定性（A/B 与 round two 对照）、scientific_claim: false、limitations、未做的事、下一项。 D9 提交（治理层 git add、提交信息带 [model-digest-change]）并核对 CI；skipped 的实验 workflow 是标签门控，不算失败。 纪律：不用 test 做方法选择（test 已被读过）；不新增或放宽任何既有阈值；若 E 或 P 复现了 B−A 的主要收益，如实写「该收益主要是容量/偏置能力」，不得换变量或时效去凑一个仍支持「信息解释」的读法；不做 C/D 臂、不做交互项第五臂、不覆盖 round one / round two 的产物与文档；不新增数据下载；不合并 main、不 force push、不租 GPU、不写 issue。 预算与停止：本轮 ≤0.9 GPU-h（第二批授权 ≤24 GPU-h，开工前自行从证据文档重算余量，round two 后约 22.09），单次实验 ≤30 min；超出即停并如实记录。23 digest 等价或输入路径一致性失败、fields 回归钉住失败、12 个 run 的 protocol_sha256 不一致、check_conventions 阻断违规时停止并报告。 结束时报告：启动 SHA、改动面逐文件、实跑过的验证与结果、protocol_sha256 与实测 GPU-h、primary 判定与四对计数、未做的事、下一项的第一个具体动作。不要自行宣布目标完成。

## 1. 现状（已核实，开工前请再核一遍）

- 分支 `r7/weather-reasoning`。**起点 SHA 以开工时 `git rev-parse HEAD` 为准**；
  写定本文件时本地 HEAD 是 `c126385`，已推送的 tip 是 `107b2cd`（多出的两个提交是
  goal-loop 技能/工具，不碰 `model/`）。
- 上一轮（round two，四臂三种子、M2 双月段、400 updates、只读 val、0.5253 GPU-h）的核心数字：
  - 参数：A 2,799,779；B（只开 `spacetime_inputs`）2,819,267；C = D 2,968,259，且 C 与 D 前向 FLOPs 完全相同。
  - t2m（val、物理 K、三 seed 配对均值 delta，负=前者更好）：
    **B−A −1.045 / −1.271 / −1.373 / −1.963 / −1.061**（6/12/24/48/72h，除 72h 外同号）；
    C−B −0.026 / −0.010 / +0.002 / −0.225 / −0.522；D−B −0.026 / −0.007 / +0.008 / −0.224 / −0.516；
    **C−D −0.001 / −0.003 / −0.006 / −0.002 / −0.006**（量级 1e-3 → 位置依赖未获支持）。
  - 聚合（85 格、逐 seed 同号）：B−A 36/3/46；C−B 8/16/61；D−B 7/17/61；C−D 8/13/64。
  - 读取**模块的存在**在 t2m 长时效上带来 0.22–0.52 K 的一致改善（C−B ≈ D−B 到小数第三位，
    说明这来自容量/结构而非位置），但对另外 16 格有害；净计数为负。
- **本轮要补的缺口**：B−A 的 1.0–2.0 K 是整场战役唯一的大效应，但 B 只比 A 多 19,488 参数（4 个张量），
  **没有容量控制**；且 round two 没有预登记 primary 变量/时效，只能按全部 85 格同等计数判读。
- 证据文档：`docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`（§7 计数与逐变量、§9 局限）；
  round one：`docs/R7_71_72_M1_AND_RWA.md`。两个 `outputs/r7_71_72_*` 目录**只读，不得覆盖**。

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | **两个控制臂模式**：`constant`（同形状零字段）与 `shuffled`（确定性批内错配），默认 `fields` | 改动 + 测试 |
| D2 | **替换发生在模型内部**：dataset / rollout 仍携带真实字段，`DECLARED_MODEL_INPUTS` 与输入路径一致性测试**不得改变** | `tests/test_r7_input_path_consistency.py` 实跑 |
| D3 | **默认路径逐位不变**（23 digest 等价测试实跑通过，含反证） | 测试名与结果 |
| D4 | **`fields` 模式逐位回归钉住**：改动**前**先在当前树冻结 `spacetime_inputs=True` 路径的 forward/反向 digest，改动后断言相同 | 新测试 + 冻结值出处 |
| D5 | **预登记 primary**：第一次 `optimizer.step()` 之前把 primary（t2m 五个时效；B−A、B−E、E−A、P−A 四对）与判定规则写进 `protocol.json` 并冻结 digest，跑后不得修改 | `protocol.json` + digest |
| D6 | **12 个 run**（4 臂 × seed 41/42/43）：`protocol_sha256` 全同、只读 val、test 封存、实测 GPU-h、四臂参数/FLOPs | 产物目录 + 实测数字 |
| D7 | **比较器输出**：B−A、E−A、P−A、B−E、E−P（外加可选的 A−A 自检），逐 seed 同号，**全 17 变量 × 5 时效照报** | `paired_comparison.json` |
| D8 | **证据文档** `docs/R7_71_72_ROUND_THREE.md`：primary 判定、全变量汇总、重跑稳定性（A/B 与 round two 对照）、`scientific_claim: false`、`limitations`、未做的事、下一项 | 文件路径 |
| D9 | 提交（治理层 `git add`，`[model-digest-change]` 标签）并核对 CI；skipped 的实验 workflow 是标签门控，不算失败 | commit SHA + run id |

## 3. 判据与证据来源

- 判据只有 **#60 比较器的逐 seed 同号规则**（`depth=0`）与协议里**跑前冻结**的 primary 判定文字；
  **不新增、不放宽任何既有阈值**。
- 判据来源（文档指针）：`docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`（上一轮计数与局限）、
  `docs/R7_B2_MULTISEED.md`（协议冻结 + 同号 + unresolved 语义）、
  `docs/R7_67_PUBLICATION_PROTOCOL.md`（test 已读过的纪律）、`docs/rules/testing.md`、
  `docs/rules/ci-and-verification.md`。
- **只读 val，test 全程封存**（test 已被读过，不得用于方法选择，也不得重称未见）。
- **判定必须在跑前写成文字**（写进 `protocol.json` 并冻结）：primary 变量 = t2m，
  三对对比 = B−A（信息+容量）、**B−E（信息）**、**E−A（模块容量/偏置）**、P−A（容量+方差），
  五个时效全部报；并写清"E 复现 B−A 的多大比例算容量解释"。全 17 变量照旧全部报告，
  聚合计数与 primary 判定**分开写**。
- 若 E 或 P 复现了 B−A 的主要收益：如实写"该收益主要是容量/偏置能力"，
  **不得**换变量、换时效去凑一个仍支持"信息解释"的读法。

## 4. 实施顺序（依赖顺序，不可跳步）

1. **recon**：`git rev-parse HEAD`；确认 round one / round two 产物与文档未被改动；
   确认当前 23 digest 等价测试与输入路径一致性测试通过；**从证据文档重算第二批已用 GPU-h 与余量**并写进协议。
   每完成一步用 goal-loop 技能的约定把结果登记进 §9 进度块（每轮更新；**执行者不宣布完成**）。
2. **冻结 `fields` 回归 digest**（在改任何模型代码之前）：用当前树构造小模型跑
   `spacetime_inputs=True` 的 forward/反向，记下 digest 作为回归基准。
3. **实现 `constant` / `shuffled` 两个模式**：替换点在 `SpacetimeConditioning.forward` 内部
   （`model/spacetime_conditioning_r7.py`，`require_spacetime_fields` 之后）；`shuffled` 用**确定性批内错配**
   （`roll(1, dims=0)` 一类，batch ≥ 2 可用；batch < 2 报错），不要写成"batch 必须等于 2"的脆弱形式。
4. **把模式接通全链**：`model/weather_forecaster_r7.py` → `model/process_forecast_r7.py`
   （本轮四臂都是 `kind="process"`，模式必须经 `ProcessForecastCoReasoner.__init__` 传到 backbone）
   与 `model/recursive_weather_r7.py`（generic 族保持可构造）；`spacetime_inputs=False` 路径不受影响。
5. **测试**：新增模式行为测试（常量与 fields 不同、错配与 fields 不同、非法 mode 报错、
   三种 mode 参数张量与总量完全相同、`DECLARED_MODEL_INPUTS` 三 mode 不变，含反证）+
   `fields` 回归 digest 断言 + 实跑 23 digest 等价与输入路径一致性。
6. **harness**：复制 round two 的 harness 为 `scripts/study_r7_71_72_round_three.py`，
   ARMS = A/B/E/P，输出目录 `outputs/r7_71_72_round_three/`（**新目录**），
   在协议里冻结 primary 与判定文字；`limitations` 更新为"控制臂只测容量与配对、不测位置依赖"。
7. **跑实验**：4 臂 × 3 seed = 12 run，一卡一进程；协议在第一次 `optimizer.step()` 前落盘并回读；
   记录墙钟与实测 GPU-h（`elapsed_seconds` 求和 ÷ 3600）。
8. **参数/FLOPs 实测**：四臂各自 `count_parameters` / `count_forward_flops`；E 与 B 必须逐张量相同、算力相同。
9. **比较器**：跑上述各对，逐 seed 同号；全 85 格照报；`beats_baseline_everywhere` 为 false 不得写成"全面更好"。
10. **重跑稳定性**：A 与 B 与 round two 在共有 seed（41/42/43）上的排序对照；GPU 浮点非确定性下不要求权重逐位一致。
11. **证据文档 + 提交 + 核 CI**；R-009 基线若因新增测试而变化，同步 `tools/check_conventions.py` 与
    `docs/rules/CHANGELOG.md`。

## 5. 预算与停止

| 项 | 值 | 依据 |
| --- | --- | --- |
| 本轮自设上限 | **≤0.9 GPU-h** | 12 run × ~0.045（round two 实测 0.5253 同形） |
| 第二批授权 | ≤24 GPU-h；已用 **≈1.914**（M2 1.204 + round one 0.1846 + round two 0.5253）→ 余 ≈22.09 | `docs/goals/full-auto-campaign.md`，**开工前自行重算**，不得自行扩大 |
| 单次实验墙钟 | ≤30 min | 同上 |
| 新产物 / decoded | ≤100 GiB / ≤256 GiB（本阶段） | 同上 |

停止条件（任一触发即停并如实记录）：本轮 GPU 用量达 0.9 GPU-h 或逼近 22.09 余量；
单次超 30 min；23 digest 等价（D3）或输入路径一致性（D2）失败；`fields` 回归钉住（D4）失败；
12 个 run 的 `protocol_sha256` 不一致；`check_conventions.py` 出现阻断违规；
需要任何未授权动作（新数据下载、租 GPU、合并 main、force push、写 issue）时停止该分支并写明需要授权。

**开工前置（本机卫生）**：`tests/fixtures/r7_equivalence_recipe.py`（未跟踪、hook 拒绝代理删除）若仍在，
本机 `check_conventions` 会报 **R-044 一条阻断命中**，此时**不得**宣称 conventions 干净；如实记录，不绕过。

## 6. 与 planner 草案的差异（我改了什么，为什么）

1. **草案漏了模式的下游接线**：它只列 `model/weather_forecaster_r7.py`，但本轮四臂都是 `kind="process"`，
   模式必须经 `ProcessForecastCoReasoner.__init__`（`model/process_forecast_r7.py`）传到 backbone，
   generic 族（`model/recursive_weather_r7.py`）也要能构造 —— 否则四臂连模型都建不起来。
2. **错配臂的守卫从"batch 必须等于 2"改为"确定性错配、batch < 2 报错"**：验证/评估路径也会建 batch，
   写死 batch=2 会让控制臂在评估时静默或直接崩；`roll(1, dims=0)` 对任意 batch ≥ 2 都成立且确定。
3. **草案的 open_question #2 是自问自答**：它问"是否需要第六臂分离模块容量与输入信息"——
   **E 就是这个臂**（模块在场、字段为零）：B−E = 信息，E−A = 模块容量/偏置。不需要第六臂。
4. **补"开工前重算预算余量"**：草案的 budget_notes 给了数字，但没有要求执行者自己从证据文档重算；
   本项目明确要求核对最新余量、不得误用被取代的旧上限。
5. **A−A 自检降为可选**：比较器对同一臂自比可能被拒绝，不能作为必过项。
6. **明确"全 17 变量照报"与 primary 判定分开写**：这是 round two 缺的那一条（没有预登记 primary），
   本轮必须补上，且 primary 判定文字必须**跑前冻结**。
7. 我交到校验器的是**我压缩重写后的 JSON**（结构/字段与原草案一致，含上述更正），
   不是 planner 的原始回复；校验器报告 `verified: true`、`fence_stripped: true`（它又加了围栏）。

## 7. 明确不做

- 不做 C/D 臂（位置化读写，round two 已回答）、不做交互项第五臂、不做 RW-B、不做 M3/M4/M5；
- 不新增数据下载（复用 `outputs/r7_m2_segment/store/manifests`）、不扩基线家族；
- 不读 test、不新增或放宽阈值、不事后换 primary 变量/时效；
- 不覆盖 round one / round two 的产物与文档；不合并 main、不 force push、不租 GPU、不写 issue。

## 8. objective（可直接粘贴）

> 本轮目标：给主模型的时空输入（M1）补容量控制臂——round two 测得 B−A 在 t2m 上是 −1.0…−2.0 K 的大效应，但 B 只比 A 多 19,488 参数且没有容量控制，本轮要把它拆成「信息」与「容量/偏置」两部分（起点 SHA 开工时用 git rev-parse HEAD 记录）。细节与判据见 docs/goals/v2-round-three-m1-attribution.md，不需要用户再授权。 本轮允许使用 goal-loop 技能（§9 进度与停止纪律）与 planner 委派（方案设计）；两者只做纪律与设计，不产生证据、不做科学判定。 四臂（同一 M2 store、同一 400-update 协议族、同一个新的 protocol_sha256、全部从零训练，seed 41/42/43，共 12 个 run）：A process_pooled（全关）、B process_spacetime_only（开 spacetime_inputs，真实字段）、E process_spacetime_constant（模块在场但字段是同形状零张量）、P process_spacetime_shuffled（字段确定性批内错配）。于是 B−E 是「信息」、E−A 是「模块容量/偏置」、B−A 是两者之和。 交付物（逐条核对，未做就写未做）： D1 新增 constant 与 shuffled 两个模式，默认 fields。 D2 替换必须发生在模型内部：dataset 与 rollout 仍携带真实字段，DECLARED_MODEL_INPUTS 与输入路径一致性测试不得改变。 D3 默认路径逐位不变：23 digest 等价测试实跑通过（含反证）。 D4 改动前先在当前树冻结 spacetime_inputs=True 路径的 forward 与反向 digest，改动后断言逐位相同（回归钉住）。 D5 预登记 primary：第一次 optimizer.step() 之前把 primary（t2m 五个时效；B−A、B−E、E−A、P−A 四对）与判定文字写进 protocol.json 并冻结 digest，跑后不得修改。 D6 12 个 run 的 protocol_sha256 全同、只读 val、test 封存、实测 GPU-h 与四臂参数/FLOPs（E 与 B 必须逐张量相同、算力相同）。 D7 比较器出各对计数，逐 seed 同号才算 improved/worsened，全 17 变量 × 5 时效照报。 D8 证据文档 docs/R7_71_72_ROUND_THREE.md：primary 判定、全变量汇总、重跑稳定性（A/B 与 round two 对照）、scientific_claim: false、limitations、未做的事、下一项。 D9 提交（治理层 git add、提交信息带 [model-digest-change]）并核对 CI；skipped 的实验 workflow 是标签门控，不算失败。 纪律：不用 test 做方法选择（test 已被读过）；不新增或放宽任何既有阈值；若 E 或 P 复现了 B−A 的主要收益，如实写「该收益主要是容量/偏置能力」，不得换变量或时效去凑一个仍支持「信息解释」的读法；不做 C/D 臂、不做交互项第五臂、不覆盖 round one / round two 的产物与文档；不新增数据下载；不合并 main、不 force push、不租 GPU、不写 issue。 预算与停止：本轮 ≤0.9 GPU-h（第二批授权 ≤24 GPU-h，开工前自行从证据文档重算余量，round two 后约 22.09），单次实验 ≤30 min；超出即停并如实记录。23 digest 等价或输入路径一致性失败、fields 回归钉住失败、12 个 run 的 protocol_sha256 不一致、check_conventions 阻断违规时停止并报告。 结束时报告：启动 SHA、改动面逐文件、实跑过的验证与结果、protocol_sha256 与实测 GPU-h、primary 判定与四对计数、未做的事、下一项的第一个具体动作。不要自行宣布目标完成。
>
> 实测 1801 字符（上限 4000）。

## 9. 进度（每轮更新；不由执行者宣布完成）

- **状态**：active
- **已完成**：无（本轮尚未开工）
- **未做**：D1–D9 全部
- **下一动作**：recon —— 记录 `git rev-parse HEAD`、确认 round one / round two 的产物与文档未被改动、
  实跑 23 digest 等价与输入路径一致性测试、从证据文档重算第二批 GPU-h 余量
- **已知前置**：`tests/fixtures/r7_equivalence_recipe.py`（未跟踪、本会话删不掉）在本机造成一条 R-044 阻断命中，
  需用户在 ZCode 之外删除；未删除时不得宣称 conventions 干净
