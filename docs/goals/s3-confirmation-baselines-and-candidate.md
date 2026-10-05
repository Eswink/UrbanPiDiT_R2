# s3-confirmation-baselines-and-candidate：S3 节点同数据基线、incumbent 重训与候选冻结

<!-- round-node: S3 -->

**状态：active（2026-10-05）。本文件是 S3 节点的目标长文。数据实例（v2）已发布并登记；
本轮的产物是**同数据公平基线与 incumbent 控制 + 可证伪的候选筛选结论 + S4 冻结条件**，
不是新的科学确认。**

## §0 Objective（可粘贴；单段，实测 1207 字符）

> 在主计划 docs/goals/main-model-climatology-campaign.md 的 S3 节点内，把已发布的三年确认实例（outputs/r7_s3_confirmation_2017_2022_2023_v2/，train 2017 / val 2022 / test 2023，源 e0b51616…，change-scale/process-scale/typed-evidence 三 sidecar 齐）推进到 S4 就绪。初始状态已核：实例 v2 已登记（docs/R7_S3_CONFIRMATION_INSTANCE.md，索引记录 record:s3-confirmation-instance-v2）；S2 四项机制筛选（#77、#78 R-A、#79、#78 R-B）已全部登记并交接。本轮做四件事：①在实例上重建 train-only (month,hour) 气候态与 persistence，记录身份与窗口覆盖（CPU，零 GPU，只读 train）；②按冻结协议在同数据上重训 actual C 的 process incumbent（3 预声明 seed 41/42/43、400 更新、L6、val-only）作为公平控制，协议先冻结 planned_seconds 软预算与 hard_cap_seconds 宽松硬上限；③以"一次一因子"筛选训练配方/lead 覆盖类候选（S0 证据指向"L6 深监督却评估 72h"与"400 更新末段 loss 仍未收敛"两条假设，R-C 型），只认该轮预注册的 val t2m 主格符号规则与 u10/v10/mslp 相对同数据 incumbent 的守门预审；④把脱颖的候选连同 K、配置、checkpoint、cases/端点、统计法（时间块同时区间用 training/r7_simultaneous_stats.py）一起冻结，写出 S4 预注册协议草案。test（2023）全程不读；数据或基线身份改变必须重训、不能混用旧值；负结果与失败保留并登记，不反复看 test 练到赢、不改冻结判据、不删弱化测试；GPU 默认共驻、每次启动前只读核 UUID/余量、不信号邻居；实验离线（R-028），准备期可联网；硬截止或真实错误停止该 attempt 并全额记账，下一独立协议可自主增加时长。判据权威见 docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md 与本文件（docs/goals/s3-confirmation-baselines-and-candidate.md）；本目标为实际端到端研究，不回改历史终态、不自行宣告 goal complete。停止条件：选中候选完成冻结并具备 S4 预注册草案，或本轮硬上限截断，或触发保留授权（付费/独占/main 合并）。

## §1 承接与现状

- 主计划：`docs/goals/main-model-climatology-campaign.md`（§4 节点表、§7 账本、§8 进度块）。
- 科学合同：`docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`（S4 门全文；`alpha_r=0.05/(r(r+1))`）。
- 实例：`docs/R7_S3_CONFIRMATION_INSTANCE.md`（v2 修正构建；v1 缺陷构建保留不删）。
- S2 已登记读数（全部回主线动作明确，本节点不重开）：
  - #77 频带：主格 unresolved → screening-negative（`docs/R7_S2_77_PE_BAND.md`）。
  - #78 R-A 变化尺度：双格 worsened（`docs/R7_S2_78_CHANGE_SCALE.md`）。
  - #79 类型诊断：B→C 归因 supported、相对 incumbent 两主格 unresolved（`docs/R7_S2_79_TYPED_EVIDENCE.md`，candidate_state=needs-review）。
  - #78 R-B 加权损失：双格 worsened（`docs/R7_S2_78_RB_LOSS.md`）。
- S0 留待检验的两条机制假说：训练目标只有 L6 深监督却评估到 72h；400 更新后 loss 仍在下降。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 ✅ | 三年确认实例 v2（store + 三 sidecar，test 未读） | `docs/R7_S3_CONFIRMATION_INSTANCE.md`；`outputs/r7_s3_confirmation_2017_2022_2023_v2/` |
| D2 | 实例上的 train-only 气候态与 persistence 重建（身份、桶覆盖、窗口计数） | 审计工具输出 + 本轮证据页（CPU） |
| D3 | 同数据 incumbent（process、L6、400 更新、3 seed）冻结协议与 val 读数 | `outputs/r7_s3_*_incumbent/` protocol + seed receipts |
| D4 | R-C 型候选（训练配方/lead 覆盖）单因素筛选协议与 val 读数 | `outputs/r7_s3_*_*/paired_comparison.json` + 协议 SHA |
| D5 | S4 冻结包：候选、K、配置、checkpoint、cases/端点、统计法与预注册草案 | 本轮证据页 + `docs/` 冻结草案 |
| D6 | 证据索引记录 + canonical brief 重渲染 + 账本/进度同步 | `docs/R7_EVIDENCE_INDEX.jsonl`、`docs/R7_CANDIDATE_BRIEF.md`、主计划 §7/§8 |

## §3 判据与证据来源

> 每轮筛选的判据是**该轮冻结协议**里预注册的主格（变量/lead/配对/方向规则），决定文本在
> 第一步之前写进 protocol.json 并逐字执行；判定只认每 seed 增量同号，符号不一致即 unresolved
> 且不许求均值。全变量全 lead 如实报告、不并入判定、不事后改选主格。参考
> `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md` §3–§5 与本文件 §0。

**S3 的判定纪律（写作时冻结）**：

1. **同数据重建不可省**：气候态/归一化/变化尺度/proxy 只 fit 2017 train；incumbent 与所有
   候选都在 v2 实例上重训，旧 dev store 或 v1 的任何数值不得带入比较。
2. **incumbent 控制先行**：候选筛选的对照必须是**同数据、同预算**的 process incumbent；
   与 climatology 的差距表在 incumbent 读数齐备后重算，不用 S0 的旧数字冒充。
3. **R-C 主格（预注册）**：对照开关是训练目标配方（incumbent = 现有 L6 深监督；候选 = 预注册
   的 lead 覆盖/深监督配方，具体文本在该轮 protocol.json 逐字写定）；读 val t2m 在 6h/12h
   （必要时含 24h）的 seed 配对增量；supported 需主格 lead 全 seed 同号且候选更低。若某 lead
   数据缺失、或同号但方向为正，如实写 worsened/unsupported。
4. **守门预审**：u10/v10/mslp 相对同数据 incumbent 的每 seed 相对 MSE 变化在 5 lead/full 上
   全部 ≤0（容忍 0.0）才允许进入 S4 冻结候选；预审不通过则记录并返回独立开发。
5. **test 不读**：全轮 val-only；任何 test JSONL 打开或被评分的记录都占 S4 的确认序号 r。
6. **数值法与统计**：S4 的同时区间用 `training/r7_simultaneous_stats.py`（max-statistic
   成对时间块 bootstrap）；S3 只做点估计筛选，不引入显著性主张。

## §4 执行顺序（每步先跑校检器对表）

1. **D2 基线与身份（CPU，零 GPU）**：在新实例上核算 train-only 气候态桶覆盖与 persistence
   参考；产出可复核的计数/覆盖率读数与 digest，写入本轮证据页草案。
2. **D3 incumbent 重训（GPU）**：先写 `outputs/r7_s3_*_incumbent/protocol.json`（含
   planned/hard 秒数、3 seed、400 更新、L6、val-only、冻结的判据文本），再逐 seed 训练+评估；
   合并读数，登记成本。
3. **D4 R-C 候选筛选（GPU）**：写该轮 protocol.json（机制差异、阳性/负控制、停止出口、回主线
   动作），两臂同参数量/FLOPs/初始化配对，val-only 三 seed；按 §3.3 判读。
4. **D5 冻结**：候选在 val 上主格不 worse 且守门预审通过时，冻结 K/配置/checkpoint/cases/
   端点/统计法与 S4 预注册草案（含 r=1 的 0.025 alpha 与时间块长度）；不通过则如实记录，
   候选停当轮、开发侧给出下一独立假设或转训练信号/优化/数据制度分析。
5. **D6 登记**：每轮进 `docs/R7_EVIDENCE_INDEX.jsonl`；更新 canonical brief、主计划 §7/§8；
   精确 SHA CI。

## §5 预算与停止条件

| 项 | 值 |
| --- | --- |
| 本轮新增软预算 | 2.5 GPU-h（累计软预算 4.5 = S2 的 2.0 + 本轮 2.5；超软继续并记 overrun） |
| 本轮新增硬上限 | 6.0 GPU-h（累计硬上限 12.0 = S2 的 6.0 + 本轮 6.0；累计用量达 12.0 时截断本轮 attempt，不科学通过） |
| 每 seed deadline | 各轮协议内冻结（建议 5400 s 量级，按实例规模核） |
| CPU 预算 | D2 与协议/测试准备不设 GPU；墙钟按工具实际记录 |
| 停止条件 | 候选完成冻结 + S4 草案就绪；或硬截止；或需保留授权（付费/独占/main 合并） |

## §6 不做清单

不读封存 test（S3 全程 val-only）；不改已冻结判据/端点/案例集；不删弱化测试；不覆盖既有
outputs；不合成数据；不付费/租卡/独占；不 main 合并/force/--mirror；不建常驻/cron；
不把 #77/#78/#79 的旧 dev 读数当本实例结论；不用 S0 的旧 gap 表冒称同数据比较。

## §7 进度块

- **状态**：`active`
- **已完成**：
  - batch-2 四季 2022/2023 获取（8/8 part、网络 28,773,423,423 字节、两次失败保留；
    `docs/R7_S3_BATCH2_ACQUISITION.md`，索引记录 `record:s3-batch2-20222023-acquisition`）。
  - D1 实例 v2（`docs/R7_S3_CONFIRMATION_INSTANCE.md`；源 `e0b51616…`、store 472/472/472、
    三 sidecar `6fd9e774…`/`b2830014…`/`8760894a…`、data identity `e01828e9…`；v1 缺陷构建保留；
    test 未读）；决策 0039（源指纹完整化）与回归测试随 `8fa6e0c` 提交。
  - D2 同数据气候态/persistence（`docs/R7_S3_D2_BASELINES.md`，索引记录
    `record:s3-d2-same-data-baselines`；逐 lead 全覆盖、气候态 480 步/16 桶各 30；553.1s、
    overrun 0、网络 0、GPU 0；test 未读）。
  - D3 同数据 incumbent 重训（`docs/R7_S3_D3_INCUMBENT.md`，索引记录
    `record:s3-d3-incumbent-retrain`；三 seed 初始化逐位复现 actual C、400 L6 updates×3；
    整轮 3955.3s、planned 5400/hard 10800、overrun 0、GPU-h 1.0987；t2m 6h skill
    +0.2143/+0.2754/+0.3262，seed43 12h +0.0037、24h +0.0526；test 未读）。
  - D4 R-C lead-coverage 候选筛选（`docs/R7_S3_D4_RC_CANDIDATE.md`，索引记录
    `record:s3-d4-rc-candidate`；two_step×200 更新 vs D3 l6×400，FLOP 匹配比 2.0018；
    **注册负结果 worsened**：6h +0.9202/+1.0247/+0.9263 K、12h +1.3219/+1.4587/+1.2365 K
    三 seed 同号为正，守门预审 34 个正 cell；候选按冻结规则停当轮，不进 S4 冻结包；
    整轮 3050.1s、overrun 0、GPU-h 0.8472；test 未读）。
- **未做**：D5 S4 冻结包（当前无通过 GM 筛选的候选）；#79 typed-evidence 的 needs-review
  处置（留作 S3 内可选候选，须新协议才可再筛）。
- **下一动作**：R-C 杠杆（two_step 目标）已在本实例注册为负、不再重开同因素；按 D4 页 §7
  推测启动下一独立单因素假设——update 预算筛选（l6 800 更新 vs D3 冻结的 400 更新 l6 控制），
  直接检验 S0 的「400 更新末段 loss 仍在降＝可能停早了」；新协议预注册后运行，通过则 D5 冻结
  S4 包，不通过则继续给出下一独立假设。节点软预算 2.5 内余量约 0.55 GPU-h，超出即记 overrun。
