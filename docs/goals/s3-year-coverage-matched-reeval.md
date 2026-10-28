# S3：训练年份覆盖轴在**匹配算力**下的重估（5 年 → 22 年，月份固定 12 个月）——65×65 实例

<!-- round-node: S3 -->

状态 **CPU 侧实现已完成、GPU 未开跑**（2026-10-26）。本文件是 R1 的**人读 brief**；
机器可读的 `round-spec` 在目标长文 `docs/goals/s3-matched-exposure-reeval.md` 顶部
（`campaign-state.current_round_goal` 指向它），本文件不重复一份。
`docs/goals/s3-matched-exposure-reeval.md` §4.2 的「每轮程序」是本轮的硬约束来源。

## §0 Objective（单段）

> 在 docs/goals/s3-year-coverage-matched-reeval.md 接续 S3；训练年份覆盖支已按预声明出口关闭（`record:s3-fy-year-coverage-result`：(a)(b)(c) 真、(d) 假，只栽在 6/12/24 h 短 lead），而该轮自己写明候选臂只看到约 7% 的 epoch，故「多出的 17 年没有额外价值」与「算力太少、用不上多出的年份」不可分；本轮按决策 0061 的**用户具名授权**只补算力、重开一次：唯一改动因子仍是训练清单覆盖的年份（2017–2021 五年 → 2000–2021 二十二年，月份固定 12 个月），唯一相对已关闭那一轮改变的是**算力**（updates 2,400 → 16,071、batch 1 → 2，即 `updates × batch = 32,142 ≥` 候选臂 32,142 窗 ⇒ 每臂 ≥1 epoch），store（outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr，[34084,17,65,65]）、网格（full_grid 4,225 点）、监督域、数据跨度（2000–2021 train / 2022 val）、2000–2021 train-only 归一化、48 桶 train-only 气候态分母（outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json）、迁移父（scripts/r7_s3_fy_annual_parent.py，model_code_sha256 60de3b2d…）、spec（K=4/depth=4/dim=192）、seeds 41/42/43、LR 2e-5、warmup 10、clip 1、物理权重、2022 全年 val（1,458 窗）逐字不变；预声明出口逐字沿用 docs/goals/s3-year-coverage.md:68-79 冻结的 (a)–(d)，不放宽、不新增阈值：全真 → 只提请独立复核/验收、不自动进 S4，任一假 → 在**算力已匹配**的条件下永久关闭「训练年份覆盖」这一支（不扫年份子集、不调 updates/LR/warmup/K/depth/dim、不加 seed、不加参数匹配臂、不与已关轴组合、不换名重开，且此后不再以「算力不够」为由重开）；桥 (e) 只比控制臂与已登记 FY 候选臂（record:s3-fy-annual-coverage-result）的 5-lead t2m RMSE seed 均值，用于发现身份/工程错误，失败只记 identity/engineering unresolved，不进入 (a)–(d)；本轮的**科学配对控制臂在本轮协议内重跑**（同 store/归一化/气候态/算力），已登记臂因归一化与气候态分母不同不可作配对控制；每轮一个臂、GPU 由统一父调度按共驻余量串行、每阶段 spawn 前只读 gpu_gate、不对邻居进程发任何信号；不读 2023 test（r=0、test_read=false、scientific_claim=false）；软预算超了继续并记 soft_overrun，仅 hard 截止截断 attempt（记 budget_limited/failed、全额记账、不算科学通过）；不在运行中改硬限；预算常数取决策 0060（per-seed 129,600 s / planned 259,200 s / hard 518,400 s）；不新增下载、不重建 store、不改 model/、不 main 合并 / force / 关 issue、不合成数据、不宣布科学通过、不称泛化/SOTA、不自行宣布最终 goal 完成。

（objective 实测 **1641** 字符、单段单行，由 `tools/check_goal_brief.py` 的
`_first_paragraph` 口径在提交前核验：`briefs=1 failures=0 advisories=0`。）

## §1 起点与唯一改动因子

- 起点：工作分支 `r7/weather-reasoning`；节点 S3；账本累计 249.413841 GPU-h（cap 20.0 / remaining −229.413841
  仅会计，无总 GPU-h 上限，决策 0030/0038）。
- 起点证据：`docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md`（`record:s3-fy-year-coverage-result`，
  `outcome_class: mixed`、`candidate_state: not-candidate`；关闭读数 6 h +0.006002 K / 12 h +0.004365 K /
  24 h −0.018799 K）。
- **唯一改动因子**：训练清单覆盖的**年份**（2017–2021 五年 → 2000–2021 二十二年），月份固定 12 个月。
  两臂同 store、同归一化、同 48 桶气候态分母、同 spec、同迁移父、同 seed、同 val、同 LR/warmup/clip/
  物理权重/updates/batch，只差训练清单（控制 7,291 窗 / 候选 32,142 窗）。
- **相对已关闭那一轮的唯一改动是算力**：`updates` 2,400 → **16,071**、`batch` 1 → **2**
  （`16,071 × 2 = 32,142 ≥` 候选臂窗数 ⇒ 每臂 ≥1 epoch）。按决策 0057 的成文定义，
  「不是同一族」的实质内容是换 store / 网格 / 监督域 / 数据跨度，这四项**全不变**，
  因此本轮走的是**决策 0061 的「用户具名授权推翻冻结条款」**，不是新制度、也不是新假设。
- **一手依据（本轮检索，arXiv MCP）**：`2602.22962`《Scaling Laws of Global Weather Models》
  实测固定算力下把资源给更多训练数据优于给更大模型；`2403.13808` 固定算力预算下数据多样性↑ → 性能↑；
  `2512.22814` 指出再分析训练集对可预报性慢模态的采样有限。
  **反方依据**：GraphCast（`2212.12794`）报「加 1959–1978 这 20 年几乎无影响」但**未给数字**。
  故这是一个有区分力的可证伪检验；「数据越多越好」这类不可证伪写法不接受。
- **可证伪预测**：该轮 (a)(b)(c) 已全真、(d) 只栽在 6/12/24 h 短 lead，而短 lead 正是 0.075 epoch
  最吃亏处 ⇒ 预期补上 ≥1 epoch 后 (d) 翻转。若 (d) 仍假，则「多出的 17 年没有额外价值」
  在算力已匹配的条件下成立，年份覆盖轴**第二次**关闭。

## §2 交付物清单

| 编号 | 交付物 | 可核查证据 |
| --- | --- | --- |
| D1 | 本轮 round-spec（机器可读） | `docs/goals/s3-matched-exposure-reeval.md` 顶部的 `round-spec` 块；`campaign-state.current_round_goal` 指向它 |
| D2 | 本轮驱动 | `scripts/study_r7_s3_fy_year_coverage_matched.py`（R-051 ≤600 行）+ `scripts/r7_s3_fy_year_coverage_matched_support.py` |
| D3 | CI-safe 测试 | `tests/test_r7_s3_fy_year_coverage_matched.py`（不依赖 `outputs/`；含 CPU smoke 与反证） |
| D4 | 本轮设计决策 | `docs/decisions/0062-r1-year-coverage-matched-compute-round.md`（引用 0060/0061，不重写其授权） |
| D5 | 冻结协议与逐 seed 回执 | `preparation_protocol.json` / `prepared_protocol.json` / `protocol.json` 的 `protocol_sha256`、`seed{41,42,43}_{control,treatment}_receipt.json` |
| D6 | 配对读数与预声明判定 | `readings.json`（F−N、skill、桥 (e)、出口 (a)–(d)） |
| D7 | 登记四件套 | 证据页 `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md`、`docs/R7_EVIDENCE_INDEX.jsonl` record、`docs/R7_CANDIDATE_BRIEF.md`、master 长文 §8 + 账本行 + `campaign-state` |

## §3 判据与设计决定（本轮冻结；科学门只引用冻结文档）

科学门形式只来自 `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` 与
`docs/goals/s3-year-coverage.md:68-79`；本轮**不触碰**任何阈值，也**不宣布**通过。

- **预声明出口（逐字沿用，冻结后不放宽）**：
  - (a) F 的 72h t2m skill 在 seed 41/42/43 **全部 > 0**；
  - (b) F−N 的 72h t2m ΔRMSE **三 seed 全部 < 0**；
  - (c) F 的 48h t2m skill **三 seed 全部 > 0**；
  - (d) F−N 在 6/12/24h 的 **seed 均值** ΔRMSE **≤ 0**。
  全真 → 只提请独立复核/验收，不自动进 S4；任一假 → 在算力已匹配的条件下**永久关闭**
  「训练年份覆盖」这一支。
- **桥 (e)，非科学判定**：控制臂 N 的 5-lead t2m RMSE seed 均值与已登记 FY 候选臂
  （`record:s3-fy-annual-coverage-result`，已在 `docs/R7_EVIDENCE_INDEX.jsonl` 核对存在）的 seed 均值
  比较；差值不得超过该登记臂在对应 lead 的 seed 极差或均值的 5%（取较大者）。归一化与气候态分母都不同，
  桥只用于发现身份/工程错误，失败只记 `identity/engineering unresolved`，不进入 (a)–(d)。
- **配对控制臂在本轮内重跑**：已登记的同因子臂用的是**另一套归一化与气候态分母**，
  且决策 0045 的复用白名单含 `per_seed_seconds`（决策 0060 已改），故它**不可复用**、也不可作配对控制；
  `control_record` 只作桥 (e) 的身份/工程参照。
- **评分域**：65×65 整张网格，`boundary_margins=(1,)`，读 `boundary_rmse.csv` 的 `full` 区域（4,225 点）。
- **气候态分母**：48 个 `(month, hour)` 桶，train-only（2000–2021）拟合；缺桶即报错，不允许 fallback。
- **正/负控制**：两臂由同一 seed、同一父状态、同一 spec 构造，初始化逐位相同；协议差异必须**恰好**等于
  `{train_manifest, train_data_identity, train_windows}`（测试断言）；无新增参数，故无参数量混杂。
- **test 策略**：`test.jsonl` 从不读取；`test_read=false`、`scientific_claim=false`。

## §4 预算与停止

| 项 | 冻结值 |
| --- | --- |
| 训练次数 | 2 臂 × 3 seed = 6 |
| 每 seed 软预算 | 129,600 s（决策 0060） |
| round planned（软） | 259,200 s（决策 0060） |
| round hard（宽松硬上限） | 518,400 s（决策 0060） |
| updates × batch | 16,071 × 2 = 32,142（≥ 候选臂 32,142 窗 ⇒ 每臂 ≥1 epoch） |
| 峰值估计 / margin / reserve | 4,096 MiB / 512 MiB / 6 GiB |
| 评分窗口（全 65×65） | 6h 1,458；12h 1,457；24h 1,455；48h 1,451；72h 1,447 |

- 软预算超了继续并记 `soft_overrun_seconds`；**只有 hard 截止因时长截断 attempt**，
  记 `budget_limited` / `failed` 并全额记账，**不算科学通过**。
- 每阶段 spawn 前做**只读** `gpu_gate`；失败 attempt 原样保留、全额计费、不做 in-place resume。
- 若 R1 首次实测 `s/update` 明显偏离 2.351 s，**另立更正决策**，**不在运行中改硬限**（决策 0060 第 2 条）。
- **停止条件（仅三种，决策 0043）**：触保留授权边界；达已冻结的科学停止条件；确无可安全独立推进的工作。
  停下必须在 master 长文 §8 写「为何停 + 恢复第一条命令」。

## §5 明确不做

不扫年份子集；不调 updates / LR / warmup / K / depth / dim；不加 seed；不加参数匹配臂；
不与已关轴（depth / width / SAM / zero-base decoder / plain EMA / 长 lead 权重 / rollout 剂量 /
容量轴 / H-A..H-D / dense 12 步 / final-only / 空间梯度 / 显式 tendency / PE / scale / typed /
ordering / anchor / rollout-dose / 月份轴）组合；**不换名重开**；
不新增下载、不重建 store、不改 `model/`；不复用旧 attempt 的 seed（决策 0045 + 0060）；
不读 test；不 main 合并 / force / 关 issue；不删 `outputs/`；不合成数据。

## §6 进度与交接

- **已完成（CPU，2026-10-26）**：驱动 + support 模块 + CI-safe 测试 + 本轮 ADR + 本 brief；
  `round-spec` 写入目标长文顶部、`campaign-state.current_round_goal` 指向它。
- **已完成（GPU，2026-10-26…28）**：`round-guard` → code zip → `freeze` → `stage prepare` →
  `stage archive` → `stage seed 41` → `stage seed 42`。实测：
  seed41 `success` **89,519.6 s**、seed42 `success` **88,362.4 s**；两 seed 的四份 receipt
  均记录 `updates = 16,071`、`batch_size = 2`，且 `train_data_identity` 分别为
  control `71bb33e01a9d…` / treatment `81307008b572…`，与冻结协议逐字节一致。
- **seed43 失败过一次并已按决策 0065 重放补跑**：首次以 **20.76 s** exit 1，
  `RuntimeError: incomplete execution source inventory`。根因**不是**源码漂移：`execution_files()`
  对 `training/`、`model/`、`data/` 做**活体 `rglob("*.py")`**，而另一条研究线的 commit `7065f19`
  新增了 2 个模块（195 → 197）；195 个冻结条目**逐位未变、无删改**。补跑方式 = 用归档
  `code.zip` 在**冻结的绝对路径**下重放（不改比对逻辑、不重新冻结），实测重放下
  `validate_protocol` 以**同一 digest** `15a92e3230c5…` 通过。
- **未做**：`stage seed 43` 的 receipt（重放中）、`stage reading`、四件套登记
  （证据页 / index record / 账本行 / campaign-state）。
- **登记提交前必须复算** `docs/rules/size-thresholds.md` 的 `<!-- measured: ... -->` 标记
  （`tests/test_check_conventions.py::test_size_report_counts_match_the_checker` 按**提交集合**核对，
  新证据页与四件套一旦进入跟踪集合就会改变计数）。

## §7 为什么这一轮仍然不是科学结论

- 三个 seed 的一致性证据**无显著性**；`scientific_claim: false`、`test_read: false`（`test.jsonl` 从未读取）。
- 本轮的价格（决策 0061 已写明）是**削弱「预声明出口不可翻案」的纪律**并**为按算力重测开先例**；
  先例范围限定在**年份覆盖**这一条轴，不授权重开 depth/width/SAM/zero-base/EMA/长 lead 权重/rollout 剂量。
- 若 (d) 仍假，买到的只是一个**更强的负结果**（算力已匹配），届时按关闭条款登记 `not-candidate`。
- seed43 是**在归档重放下完成**的（决策 0065），证据页必须照此写明，不得写成"无异常完成"。

## §8 下一动作（第一条命令）

```
.venv/bin/python tools/r7_round.py state
```

随后：确认 `outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/seed43_receipt.json` 已落盘 ⇒
`.venv/bin/python tools/r7_round.py stage reading` ⇒ 按预声明出口 (a)–(d) 逐条判数字 ⇒
`.venv/bin/python tools/r7_round.py verify` ⇒ 四件套登记（证据页 / index record / 账本行 /
campaign-state）。
**任何 seed43 的重跑都必须在决策 0065 的 namespace 重放内执行** —— 真实工作树里那 2 个无关模块
会让 `validate_protocol` 再次失败；且任何新 commit 往 `training/ model/ data/` 加 `.py` 也会。
