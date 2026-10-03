# main-model-v2-campaign：主模型 V2 的 campaign 主计划与每轮对表

**状态：campaign级主计划（活文档）。当前N3/active：用户0032明确终结已交付的M3辅助监督假设，独立autoregressive-exposure路线接续B/C/D/E。N2a原failed与完整补测any-unresolved/paused/advance=false全部保留，不伪称旧出口放行；新路线不再以该辅助假设暂停作总前置。不自宣最终goal完成或科学增益。**

本文件是「主计划 + 每轮 recheck」机制的**唯一权威**：节点图（§2）、每轮开工前必须走的对表清单（§3）、
预算账本（§7）与进度块（§8）都在这里。每一轮的目标长文是它的**派生物**，不是平行的第二处真相——
`docs/R7_TASK_QUEUE.md` 对 V2（#70–#75）这条线**不再维护**（它停更于 2026-09-28，CI 数字已过期），
保留原位只作历史。机器可查部分由 `tools/check_campaign_state.py` 执行（决策 0025）。

## §0 Objective（主计划路由；实测 379 字符）

> 按 docs/goals/main-model-v2-campaign.md 与 docs/goals/v2-remaining-stages-exploration.md 在0030/0032用户既定路线内连续执行N3可微两步/matched-Generic与必要探索、N4三臂三seed确认、N5反证/充分证据结题及有条件关闭。M3辅助监督假设终结，原failed、完整补测any-unresolved/paused/advance=false及成本保留；新独立autoregressive-exposure路线不以该暂停作总前置。每节点前后对表、先冻结protocol与整轮planned/hard、登记代码/数据/产物digest和精确CI，普通排障/预算/节点自主不逐轮问。账本只记账，无总GPU-h闸门，软超继续记overrun，硬截断或真实错误停本attempt，修复另立协议新输出。不读封存test、不改旧判据/归档、共驻不信号邻居，保留付费/新数据/发布/独占/main合并/破坏性权限。负面只终结对应假设，独立任务继续；科学增益/工程完成/实验结题/关闭分报，不自宣SOTA或最终goal完成。

## §1 现状（带 file:line，2026-09-30 核对）

- 分支 `r7/weather-reasoning`；机制设计时HEAD `4bf92e9`，N1实验代码 `c4e7e83`、工程标记修复 `efa410b`、
  最终证据页提交 `33ee4702b22bfac5db303164886c1a57a3406a59`。open issue **#70–#75**沿用本日开工时
  匿名API核对，收尾未再次查询issue；本轮未执行关闭。
- **N0 已终结**：M2-B 的 RW-B 在登记主端点上 negative（`docs/R7_72_RW_B_PILOT.md:84-94`），
  且在换了 `model_code_sha256` 与 `protocol_sha256` 之后**复现到 1.8e-4 K**（`docs/R7_72_RW_B_SUBTRACTION.md` §5）——
  这不是一次性噪声。减法轮的分支判定是 `stop-confounded-control`：负控制按构造退化成 RW-A 本身，
  **归因未做**（同文件 §4.4）。
- **下一动作已被冻结文档写死**：`docs/goals/main-model-v2-rw-b-subtraction.md:206-208`（停止发明新模块 →
  记录可反驳假设 → 转向重查 forecast state / training objective / data regime）；
  `docs/plans/0004-r7-main-model-v2.md:323-325` 在同样出口上多列一条 **autoregressive exposure**，
  `:335` 另有「连续两次定向改动无收益：停止加开关，重新分析预测状态/边界/数据量」——两条都已命中。
- **四个嫌疑对象的既有线索**（N1 的审计对象，全部已登记）：
  - forecast state：E-195（递推键无来源角色标记与 mask）、`docs/R7_MAIN_MODEL_V2_DESIGN.md` §3.2（同一缺口与候选处置）、
    焦点臂 `correction_head` **从未收到梯度**（E-202）、「门学会关闭」（E-206）。
  - training objective：`training/r7_streaming.py:126-128` 的损失权重轴是**内部 K 步**而非物理时效；
    训练 target 只有 **+6h**（`outputs/r7_m2_segment/store/manifests/train.jsonl` 实测 186 条 `lead_time_hours` 全为 6）；
    `PROCESS_WEIGHT = 0.0`（`training/r7_rw_b_subtraction_protocol.py:31`）⇒ 过程监督从未参与训练（#65 预诊断）；
    损失在归一化空间等权、报告在物理单位逐变量（`training/r7_halting.py:16` vs `training/r7_rollout_metrics.py:69`）。
  - data regime：M2 段 train 186 / val 22 / test 26，逐 lead 窗口 22/21/19/15/11；72h 的 seed spread 20–23%
    （`docs/R7_B2_MULTISEED.md`）；气候态是 8 桶弱基线；归档产物的 `climatology_skill.csv` **仍是归一化单位**
    （决策 0010：保留原样，读历史文件要乘 `normalization_std`）。
  - autoregressive exposure：内部 K 步是自条件（草稿反馈）而物理自由 rollout 从未被训练
    （计划 0004 `:325`；与 #64 curriculum 的「换训练时效＝零和再分配」同向）。
- **机制设计时的缺口**：GPU账本此前是手写散文，无脚本核「主计划当前节点」与「上一轮下一动作」。
  本文件+`tools/check_campaign_state.py`（决策0025）已落地并接入CI，不再称它尚不存在。
- **N1本轮读数**：`docs/R7_N1_PIVOT_AUDIT.md`四块审计支持解释候选；冻结Z−RW-A的48/72h
  均seed41正/seed42负，主分支cannot-distinguish；只提议N2d停止。30val评估/510RMSE cell核齐，
  0.3590401737619605GPU-h未超cap；独立evaluation峰值缺失等工程局限如实记录，不据工程CI判科学结论。
- **N1 补测失败与修复设计（2026-10-01）**：一次具名授权的独立 evaluation 成本补测在 30 项里完成 1 项后
  失败（第二项零基线守卫拒绝：`training/r7_n1_cost_replay.py:268-273`；失败字节未记录、原因未确认。
  见 `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md`、`outputs/r7_n1_cost_failed_independent_review.json`）。
  0 GPU-h 只读审阅把范围收窄到**进程级 allocator 残渣**（推测，未取证）：评测器与归档逐字节相同、
  checkpoint CPU 载入、仓内无 CUDA 全局缓存；修复设计＝**每次评估一个全新进程**，零基线由构造保证
  （已考虑但未采用「调用私有释放 API」方案）。修复轮长文 `docs/goals/n1-cost-supplement-repair.md`
  （prepared，未执行；执行那一刻按决策 0021 重新取具名授权）。
- 工程态：37 条阻断规则 0 违规；门禁是 `R7 CPU CI` 的八步（新增 campaign 检查后为九步，
  历史证据页里的「八步」是它们当时的实测，**不回溯改写**）。

## §2 交付物：节点图

每个节点是一个**可独立启停的循环步**，含依据、输入、冻结前置/出口与证据；决策 0030 后，
执行者可在用户既定方向内自主选择允许节点并连续推进，不需逐节点新授权。节点之外的新总体
研究方向仍不得自行开跑；普通排障与编排不等于改变方向（§3 第 6 条）。旧节点预算是历史规划，
新实验时长/预算按 0030 预先写定，不作为沿用旧总额度的许可门。

| 节点 | 名称 | 依据（冻结文档） | 预声明门禁 / 证伪 | 预算 | 出口 |
| --- | --- | --- | --- | --- | --- |
| N0 | M2-B / RW-B 实现与有界对照 | `docs/goals/main-model-v2-rw-b-round.md`、`docs/R7_72_RW_B_PILOT.md` | 已执行：登记主端点 negative，非证伪 | 已用 0.8356 | **已终结**（negative，两轮复现） |
| N1 | 转向审计 + 一条可证伪臂 | `docs/goals/main-model-v2-rw-b-subtraction.md:206-208`、`docs/plans/0004-r7-main-model-v2.md:323-325` | 0 GPU-h 审计四块；臂：Z 换冻结随机张量，与 RW-B 配对（沿用 #60 比较器、depth 0、逐 seed 同号三态）；两种结局的读法在长文 §3 预先写定 | ≤0.45 GPU-h | 四方向证据强度排序 + 下一节点提议（N2a–N2d 之一） |
| N2a | M3（#73 尺度修复 + 三类时刻语义）的新补全 | issue #73、计划 0009、决策 0028/0030/0031 | **规定交付已完成**：尺度/三时刻/反证、30项完整覆盖与登记齐；原单次失败保留，negative/mixed允许验收，不新增事后完成门 | 已实耗0.6762461941885866 GPU-h | **交付完成，科学paused**：any-unresolved出口保留，不能等同forecast成功或N3放行 |
| N2b | 目标对齐修正（训练目标 vs 报告时效） | 计划 0004 `:325`、#64 curriculum 证据 | 必须先有 N1 的目标审计结论；改动要预注册、不得新增阈值与端点 | 届时重定 | 目标对齐后的重测 |
| N2c | 数据/评估支撑修正（val 跨度、窗口数、气候态） | 决策 0008/0010、`docs/R7_B2_MULTISEED.md` | 只改评估与数据支撑，不改判据；任何扩大数据需用户授权 | 0–0.2 GPU-h | 可分辨性结论 |
| N2d | 停止（在当前数据与算力下不可分辨） | `docs/goals/main-model-v2-rw-b-subtraction.md` §5 停止条件 3 | 审计给出「不可分辨」的量化读法 | 0 | 如实记录并停 |
| N3 | M4（#74 两步可微自回归与 matched-Generic） | issue #74、计划0009、决策0023/0030/0032、新remaining-stages目标 | N2a交付已登记且用户另选独立exposure主线；旧辅助paused不作总前置，两轴/可微/同结构/合法同父验收不变 | B初轮planned5400/hard10800秒，协议先冻结 | M4实现、两seed两臂与equal-compute控制；负面终结此假设，不自动扩unroll |
| N4 | M5（#75，三臂最小确认闭环与有条件 adaptive） | issue #75、计划 0009、决策 0023/0030 | 确认轮须已有 matched-Generic、冻结评估与可运行候选；adaptive **默认不启动**，仅冻结有效前沿门满足后另立协议 | 执行前自主写定 planned/hard | 完整确认与如实正/混合/负结论、adaptive 决定 |
| N5 | #71/#72 追加反证、六 issue 裁定与有条件关闭 | 计划 0009、决策 0030、新接续目标 D4 | N2a/N3/N4 的工程/实验记录和必要控制齐备，逐 issue 证据支持 verdict；不因关闭需要放宽验收 | 0 GPU-h | 登记 DONE/NEGATIVE/BLOCKED，证据齐备者按具名范围关闭；最终 goal 完成另裁定 |

**常设禁止项**：不新增 solver 部件；不改已冻结判据、阈值、端点或案例集；不重跑或改写归档产物；
不读封存 test；不无授权下载数据或租 GPU；不 force push、不合并。main/issue 关闭仅限 N5 在
计划 0009/决策 0030 的具名范围与证据/精确 CI/非 force ff 条件齐备后进行，禁止提前关闭或借此
宣称科学成功。普通节点推进已下放，不需用户逐次放行，但不绕冻结前置与停止线；
skip/queued/cancelled/partial/failed 不算通过，最终 goal 完成不得自判。

## §3 判据与每轮 recheck 清单

**每轮开工前必须逐条走完，并把这六条的结论写进该轮 §8：**

1. **节点一致性**：本文件 §8 的 `current_node` 与上一轮长文 §8 的「下一动作」必须指向同一节点。
2. **账本算术**：§7 账本逐行求和 = 已用、会计基数 `cap_gpu_h` − 已用 = 会计余量（容差 1e-3）；
   每行必须有证据指针。决策 0029 后账本只记账，历史基数 24 不再是总上限，余量不作为停止闸门。
3. **上一轮登记**：上一轮长文已在 `docs/R7_EVIDENCE_INDEX.jsonl` 里有一条 `evidence_path` 指向它的记录，
   且该记录的 `evidence_commit` 在 git 里存在；新长文过 `tools/check_goal_brief.py`。
4. **本轮 objective 派生**：写明它实现的是哪个节点、引用哪些冻结文档；禁止项照抄 §2 的常设清单。
5. **判据未被改动**：`docs/rules/` 与已冻结证据页在本轮开工前没有未记录的改动。
6. **不一致先停实验**：第 1–5 条对不上，执行者在既定方向内自主核实/修复机械漂移并登记依据、
   重新对表；不编造账本或改冻结判据。需改变总体方向或触保留授权边界时才交用户决定，
   普通排障、节点选择不恢复逐次询问。

**机器可查部分**：第 1、2、3 条（含「无机器支撑的账本行如实列出」）由
`.venv/bin/python tools/check_campaign_state.py` 执行，退出码 1 表示硬漂移；
第 4、5、6 条是人工动作（第 5 条用 `git log`/`git diff` 核对）。工具不判断科学方向——
它只能证明"没有跑偏到与主计划矛盾"，**不能**证明方向正确。

判据来源：本文件 §2 的冻结文档指针、`docs/rules/ci-and-verification.md`（运行资格与门禁）、
`docs/decisions/0030-autonomous-execution-and-direction-boundary.md`（实验/普通决策/节点常设下放，继承 0029 时长契约）、
`docs/decisions/0023-main-model-first-baseline-freeze.md`（基线冻结与 matched-Generic）、
`docs/decisions/0025-campaign-master-plan-and-per-round-recheck.md`（本机制本身）。

## §4 实施顺序（每轮固定动作）

1. 读本文件 §8 的 `current_node` 与上一轮 §8 的下一动作 → 跑 `tools/check_campaign_state.py`（§3 第 1–3 条）。
2. 从当前节点**派生**该轮目标长文 `docs/goals/<slug>.md`（骨架见 `.agents/skills/goal-loop/SKILL.md`），
   过 `check_goal_brief.py`，把 `<!-- round-node: N? -->` 与 §8 一起写好。
3. 执行该轮（0 GPU-h 审计先做；实验、普通决策与推进按决策 0030 自主，先冻结软/硬时长与范围、
   核共驻门槛并重读账本；账本只记账，保留项仍须具名授权）。
4. 收尾：证据页 + E 条目 + 索引记录 + CI 绑定 + 更新本文件 §8（本节点证据/账本/下一动作），
   独立审阅核证据，不把其变成新授权许可。
5. **连续推进**：节点出口/前置齐备后，执行者自主更新本文件 state、上一轮 next-action 与新长文标记，
   重跑对表再进入下一允许节点，不逐轮等用户触发。冻结停止条件仍先停相关探索并如实登记；
   不做后台续跑、cron 或脱离当前 goal 的守护任务，最终 goal 完成仍独立裁定。

## §5 预算与停止条件

- 批次授权：**无总 GPU-h 上限**（决策 0030，继承 0029）；账本为记账，不是额度闸门。单次实验在该轮长文与
  冻结协议中声明 `planned_seconds`（软）/`hard_cap_seconds`（宽松硬上限，默认约 2× 计划）。
- 节点内预算由执行者自主决定，按整轮墙钟控制；超软预算继续等待并记 `soft_overrun_seconds`，
  仅硬上限因时长截断，记 `budget_limited`/`failed` 并全额记账。执行前重读账本（§7）作审计，
  不沿用历史 24 GPU-h/≤30 min 为现行闸门；历史节点数值、冻结协议与证据不回溯改写。
- **停止条件**（0032前瞻路线）：真实错误、身份/资源/集合不满足停止对应attempt并全额留证；
  不可分辨、不能归因或负面终结对应假设，不机械加seed/算力/unroll，独立B/C/D/E任务继续。
  新探索须有可反驳不同机制、最小探针与独立预登记，不回改旧判据。需要新数据、付费/租GPU、
  发布--write、独占、main合并或破坏性操作只停相关动作交用户；main分歧不强推或自行合并。
  账本不足不是停止条件；普通排障与节点0030自主，最终goal完成仍独立裁定。
- 目标状态 `active / paused / budget_limited / complete`；执行者可更新节点/工作进度并按证据推进，
  **不得自行将最终 goal 标 complete 或宣告科学成功**。

## §6 明确不做

- 不新增 solver 部件；不把「12/24h 的小改善」当加码理由；不把 role 标记的小幅改善写成机制声明。
- 不重跑、不覆盖、不改写任何归档产物与历史证据页（`outputs/` 下的一律只读）。
- 不把三个 seed 写成显著性；不为让报告好看而放宽判据或弱化测试。
- 不建立定时任务或后台续跑（用户 2026-09-23 决定，`docs/R7_MANUAL_ITERATION.md:3-6`）；
  harness goal 只作 best-effort（决策 0024），不作为完成判定。

## §7 账本（记事，非闸门；每行必须有证据指针）

| 轮次 | 实测 GPU-h | 累计 | 证据 |
| --- | --- | --- | --- |
| M2 段（桶 4→8） | 1.204 | 1.204 | `docs/R7_69_BUCKET_EXPANSION.md:23`（15 份 `training_report.json` 求和 4333.9 s） |
| V2 第一轮（M1 + RW-A） | 0.1846 | 1.3886 | `docs/R7_71_72_M1_AND_RWA.md:14` |
| V2 第二轮（四臂 × 三种子） | 0.5253 | 1.9139 | `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`；长文 `docs/goals/v2-round-two-attribution.md:149` |
| V2 第三轮（M1 容量控制） | 0.5489 | 2.4628 | `docs/R7_71_72_ROUND_THREE.md:14` |
| RW-B pilot（四臂 × 两种子） | 0.8356 | 3.2984 | `docs/R7_72_RW_B_PILOT.md` + 索引记录 `record:rw-b-bounded-round-negative`（`gpu_hours = 0.8356`，机器可核） |
| RW-B 减法轮（四臂 × 两种子） | 0.4128 | 3.7112 | `docs/R7_72_RW_B_SUBTRACTION.md` + 索引记录 `record:rw-b-subtraction-round-cannot-attribute`（`gpu_hours = 0.4128`，机器可核） |
| N1冻结随机Z审计（三臂 × 两种子） | 0.3590 | 4.0702 | `docs/R7_N1_PIVOT_AUDIT.md` + 索引记录 `record:n1-pivot-audit-frozen-z-unresolved`（`gpu_hours = 0.3590401737619605`；账本四位小数舍入，差≤1e-4） |
| N1独立evaluation成本补测（failed，1/30） | 0.0060 | 4.0762 | `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md` + 索引记录 `record:n1-evaluation-cost-supplement-failed`（`gpu_hours = 0.005956627869357666`；失败全额计费，四位显示舍入） |
| N1 v2独立evaluation成本补测（success，30/30，含P1） | 0.1283 | 4.2045 | `docs/R7_N1_COST_SUPPLEMENT_V2.md` + 索引记录 `record:n1-cost-v2-evaluation-supplement-success`（`gpu_hours = 0.12826335332563354`；含P1与全部启动/间隔/清理，四位显示舍入） |
| M3过程监督三臂（budget_limited，6训练/7评估） | 0.4973 | 4.7018 | `docs/R7_73_PROCESS_SUPERVISION.md` + 索引记录 `record:m3-process-supervision-budget-limited`（`gpu_hours = 0.49725426027008024`；失败全额计费，逐行显示舍入） |
| M3独立val补全（23/23，完整但科学paused） | 0.1790 | 4.8808 | `docs/R7_73_VALIDATION_COMPLEMENT.md` + 索引记录 `record:m3-validation-complement-complete-paused`（`gpu_hours = 0.17899193391850632`；只记新补测成本，原失败独立行保留，逐行显示舍入） |
| N3可微两步FP32/BF16精度探针（success，8实际updates） | 0.0471 | 4.9279 | `docs/R7_V2_PRECISION_ACCEPTANCE.md` + 索引记录 `record:n3-v2-precision-gpu-acceptance`（`gpu_hours = 0.04711594580465721`；continuous首spawn至末owned reap，独立工程结果不作科学支持） |
| B同父两步训练（6train/30eval成功，聚合failed） | 0.8526 | 5.7805 | `docs/R7_74_AUTOREGRESSIVE_ATTEMPT.md` + 索引记录 `record:n3-autoregressive-attempt-failed`（`gpu_hours = 0.8526056814201487`；原failed全额连续计费，不以全worker成功冒称聚合通过） |
| **合计已用** | **5.7805** | — | 24 − 5.7805 = **余 18.2195 GPU-h** |

说明（如实）：决策 0029 后账本**只记账、不设总上限**；`cap_gpu_h=24.0` 是历史会计基数，
`used_gpu_h` 为实际累计消耗，`remaining_gpu_h` 为基数减累计的会计差额，三字段仍供 C-02 算术对表，
不是执行许可或停止闸门，未来差额为负也不因此停实验。新补测0.17899193391850632h全额记账，
显示新行0.1790/总4.8808/余19.1192；精确总4.880706349145538/余19.11929365085446，沿用逐行舍入。
旧历史行不改。用户关闭Mimosa后证据A已真实冻结，新补测index记录与账本record指针齐；
登记B的实际提交/精确CI另在§8追加，不把成功提交当安全扫描通过。
**账本不是从证据索引机械累加的**——索引覆盖其中七行（四历史行仍暂未有索引记录），
所以 `check_campaign_state.py` 做的是「逐行算术 + 证据指针存在性 + 有索引者数值一致」，
没有索引支撑的行会被**列出来**而不是被当成已核。把索引补成全量账本是将来可做的一件事，本轮不做。

## §8 进度块

<!-- campaign-state: {"current_node": "N3", "previous_node": "N2a", "current_round_goal": "docs/goals/v2-remaining-stages-exploration.md", "previous_round_goal": "docs/goals/n2a-m3-validation-complement.md", "previous_round_evidence": "docs/R7_73_VALIDATION_COMPLEMENT.md", "cap_gpu_h": 24.0, "used_gpu_h": 5.7805, "remaining_gpu_h": 18.2195, "status": "active", "next_node_proposal": null, "budget_mode": "accounting-only", "route_decision": "0032"} -->

- **N3当前工程窗口（2026-10-03；非预报结果）**：matched Generic、旧身份显式权重导入、exact185窗口与
  原initial/allK深监督的可微两步接线已落；source只读preflight通过，无新数据/发布。独立归档父两seed×
  两合成网格final/all5drafts严格位等价，109产物hash重核无差；首次完整CPU2614pass/10fail/9skip不算通过。
  输入/Generic/calendar定向169pass；模型语义/strict query/roles禁反馈、累计K与零训练基线/硬截止分类
  已活跃修复且有独立直接反证。final模型0cc9c16e…a223e旧父桥接109pins再验，query234/driver77/
  results81/最终runner-driver160各自通过。完整CPU3107pass/9skip/2warning，767.02s；六CUDA与三
  optional fixture跳过不算通过，EOF空行唯一变化有bytes proof与160项再验。计划0014记录实际限制。
  C前瞻primary/0容忍/adaptive门在R7_V2_CONFIRMATION_PREREGISTRATION写定，尚无C协议或天气结果。
  精确CI、真实FP32/BF16工程探针、B6训练30val和C9训练135val及六issue关闭均未做；0新增GPU-h，
  原M3失败/paused/advancefalse及四历史账本notes不变，不能以本条工程进度代科学通过。

- **状态**：`paused`（2026-09-30 N1主48/72h均unresolved，停止条件3已触发；节点保持N1待审阅）
- **已执行**：N1四块零GPU审计、D3机械复算、一次具名授权D2（6run各400/30val评估）、证据页/E-207–E-212；
  索引audit/needs-review与canonical brief登记。D1排序目标/暴露/数据/状态，前两项非独立统计证据。
- **实耗**：N1为0.3590401737619605GPU-h，账本记0.3590；原已用3.7112，精确加和4.070240173761961，
  精确余19.92975982623804，显示4.0702/19.9298不扩大授权。GPU1292.5446s/whole1297.8983s均未超cap。
- **限制与未做**：独立eval峰值缺失、validation deadline内部覆盖缺口、future finalizer全集合拒绝未实现；
  本次终态完整核齐不是future guard已修。M3/M4/M5/matched-Generic/确认轮未动用；目标完成未独立裁定。
  不新增seed/臂，不为补证重跑，test始终封存；禁止项保持。
- **登记工程验证**：提交`6b7875b8da3e047f3bddf35a830deee67064be07`的CI36699197294已终态success、
  九主步骤成功；精确干净clone1795passed/14skipped/2warnings（162.16s），skip不算通过。
  一手URL/访问日期/响应digest与判定边界见本轮长文进度，不改被冻结证据页。
- **成本补测准备（2026-09-30；当时未执行）**：独立只读验收确认停止合规，但独立eval内存缺失仍阻断完整验收。
  用户回复「允许跑实验，反正就是把它搞得完整。」要求补齐，不是豁免缺项；只准备原六checkpoint×五val时效、
  0训练更新的独立计量，拟≤0.09GPU-h/≤600s，新输出且旧归档不改。具名范围/预算/失败停止问题未收到回答，
  没有补测授权回执或GPU运行，账本未增加；准备记录见`docs/R7_N1_SUPPLEMENTAL_COST.md`。
  原主结果与N2d提议不变；不以工程准备或既有实验授权代替新的执行许可。
- **具名成本补测执行（2026-09-30）**：用户AskUserQuestion答「授权上述一次补测 (Recommended)」，
  原30val/0训练/≤324s/≤600s/新输出/失败即停；用户手动腾出GPU1，设备/归档身份重核后执行计量commit1e03f82。
  first seed41/RW-A/+6h零基线、peak39590400/46137344B、22case/17RMSE精确重放；second lead开始前非零
  allocator基线触发failed，1/30完成，没有result/终态四成本表，不自动重试。失败GPU21.443860329687595s
  全额计0.005956627869357666h，whole22.372659532353282s；N1原+失败0.3649968016313182≤0.45。
  campaign精确已用4.076196801631319/余19.923803198368685，显示4.0762/19.9238不扩授权。
  原归档/证据页/科学判据不改；独立只读核验部分记录有效、完整成本验收BLOCKED，失效基线数值/原因未知。
- **失败审阅与修复准备（2026-10-01，0 GPU-h）**：只读复核失败页哈希、复核 JSON、三提交与
  CI 36737308495（`9d8d2b6`，九步全绿）全部对上；审阅确认失败发生在第二项评估**调用之前**
  （守卫是该子步骤首个 CUDA 触点）、评测器与归档逐字节相同（`95fff1be…`）、checkpoint CPU 载入、
  仓内无持有 CUDA 张量的全局缓存；**推测**＝进程级 allocator 残渣（失败字节未记录，无法取证）。
  修复设计＝**每次评估一个全新进程**（零基线由构造保证）；修复轮长文
  `docs/goals/n1-cost-supplement-repair.md` 已备好（prepared、未执行），执行那一刻按决策 0021 重新取具名授权。
  账本未增行（本轮无 GPU 消耗）；48/72h cannot-distinguish 与 N2d 提议不变。提交 `c56cbe3` 的
  CI 36838072212 九主步骤全绿（2026-10-01 匿名 API 只读核对；本提交自身 CI 尾记录只进本轮进度）。
- **修复轮扩围：GPU 共驻政策（2026-10-01，0 GPU-h）**：用户决定本机 GPU 实验改为**默认共驻**——不等待整卡
  空闲、不要求腾卡、不干扰他人作业，显存余量够就开始；该政策作为修复轮的 **D0** 落地（决策 0026 +
  `docs/rules/` 细则 + `AGENTS.md` 一行：默认共驻、启动前只读余量门槛、禁止对非本实验进程发任何信号、
  不实现冻结/终止自动化、独占须先取具名授权）。修复轮长文
  `docs/goals/n1-cost-supplement-repair.md` 已就地扩围：旧「一卡独占/等腾卡」表述废止，v2 守卫由
  「他人进程即拒」改为「余量门槛」；判据、零基线要求与容差全部不放宽（长文 §1.2/§3.2）。
  只读观测（2026-10-01 17:46）：GPU0 free 18922 MiB、GPU1 free 13561 MiB，均远高于本补测峰值 40–50 MiB。
  **状态块、账本、节点均未改动**（本轮无 GPU 消耗）；仍为 prepared，等用户触发后按该轮 §4 依次落
  D0 → D2 → D3（执行那一刻取具名授权，共驻）。扩围提交 `d8dc6fd`，CI `36845897605` 九主步骤全绿
  （2026-10-01 匿名 API 只读核对）。
- **修复轮启动（2026-10-01，0 GPU-h）**：用户触发扩围objective，起点7064ff1，六项对表通过
  （campaign failures=0 notes=4，四条旧账本无索引支撑如实保留）；v1三项冻结哈希未变。D0已落地决策0026、
  R-054、AGENTS约束与索引/E-216；政策反证2passed、相关门禁自测348passed。D2正在修订为30个逐项新进程、
  余量门槛与拒绝字节/snapshot先写后抛已实现；最终准备定向359passed/23.31s、193计量/探针/政策反证，
  独立复核缺口修复后补startupclaim/有界reap/30launch证据/失败诊断。未取本轮GPU授权、未执行CUDA，
  账本/节点/状态块不变。
  planner请求失败按能力降级自规划，草案校验verified=true；不外委科学判据。
- **修复轮准备验证与执行阻塞**：D0/D2提交`b227020ca3d4932a765a6cd7a79e26fc659cdeaf`的
  CI`36855190840` completed/success九主步骤全绿；精确clone1988passed/14skipped/2warnings、173.22s，
  skip不算通过、远端计数未取得。准备回执/响应/日志`outputs/r7_n1_cost_v2_preparation/`；证据页
  `docs/R7_N1_COST_V2_PREPARATION.md`。2026-10-01执行那一刻按决策0021问具名探针+30val共驻授权，
  **未收到回答**；不是拒绝或许可，无回执、无GPU运行/成本表、实耗0 GPU-h，账本与状态块不增加或推进。
  一手API URL/访问日期/精确SHA见修复轮进度；Mimosa scanner_enobufs无安全结论。
- **准备态登记CI尾核**：证据页`90e2c83`、索引`d9abdc02251c1f4714762d1f19f3b1aaa33311f7`，audit/blocked
  与canonical brief15条齐。CI`36857019490`对索引SHA completed/success九主步骤绿，登记后定向359passed；
  一手API URL/访问2026-10-01/response digest见修复轮进度及`outputs/r7_n1_cost_v2_preparation/registration_*`。
  prompt-to-artifact审计明确D0/D2工程已验证、D1/D3/D4因未答授权仍缺；账本0新增、N1paused不动。
- **修复轮具名执行（2026-10-01）**：用户随后明确答「明确授权，刚刚没有看到」，仅确认之前AskUserQuestion
  的一次具名范围；回执`outputs/r7_n1_cost_v2_authorization.json` SHA0770b4f8…先冻结。执行起点b714458，
  八计量源码逐字节等于准备b227020，原30val/六checkpoint/source只读重核，新输出排他创建。
  P1纯torch清理后8519680/20971520B，私有clear诊断后0/0，预声明三态①；条件P2不触发。
  30新进程独立baseline0/0、正峰值、30RMSE文件逐字节等于原归档、528案例/510RMSE格精确语义重放；
  四成本表3/3/6/36行及cost_views齐。attempt/result success，GPU461.74807197228074s=0.12826335332563354h，
  whole463.4213050529361s均未超cap，无失败/重试/退出未确认。共驻门槛minfree23713MiB，未观测邻居PID。
- **修复轮账本与科学边界**：新行0.1283，精确累计4.204460154956952/余19.79553984504305，显示4.2045/19.7955。
  原N1+v1+v2共0.49326015495695175h，超过原科学轮0.45h历史值；v2是另授≤0.25h成本审计，不回改或冒称
  合计≤0.45h。原证据/判据/归档不变，独立成本齐不改cannot-distinguish；N1paused/current_node=N1，
  N2d仅提议。证据页`docs/R7_N1_COST_SUPPLEMENT_V2.md`；独立只读复核/最终登记CI另记本轮进度。
- **修复轮独立产物复核（2026-10-01）**：独立stdlib校验8592断言、407运行文件前后hash不变；81输入pin、
  原code.zip 953/953成员逐字节等于原c4e7e83、118源pin/model digest、八计量源对准备/执行commit核齐。
  30RMSE/ACC/climatology_skill各CSV逐字节同原，provenance仅elapsed_seconds不同，30PID/launch/claim/exit与
  91UUID/余量/时序观察齐；P1三态①、P2条件未触发，四表逐单元/预算/三项v1冻结hash核齐。
  回执`outputs/r7_n1_cost_v2_acceptance/independent_verification.json` SHA256`40699952b68fedabfa159ec8ca52ee4dab20ffeaf7c11ddcaa06d9202a08b0f2`。
  这是产物工程接受，不是科学gate/目标完成或新授权；没有GPU重放、syscall全程审计或真实邻居负载性能验证。
  证据页冻结提交`3d7a8e2e52941407cfb882780413b8b99d7e5a3c`、页SHA256`7ce659dfa6e244b754383a8b42bc5ce8e16eacb6b136a24f97bad63cd9e01873`；
  audit记录`n1-cost-v2-evaluation-supplement-success`单独登记，保留旧prepared/failed记录；canonical brief16条。
  最终登记CPU定向555passed/24.23s、37阻断0违规、campaign0失败4历史notes/goal0失败，登记CI待核。
- **修复轮实际登记CI绑定（2026-10-01）**：索引/账本登记提交`39370892f52d85ce021022e90ad76738f0775dd3`
  的CI`36865544410` head_sha精确匹配、completed/success九主步骤全绿；555passed/24.23s是本地CPU计数，
  远端pytest日志计数未取得。[run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36865544410)、
  [jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/36865544410/jobs?per_page=100)，匿名curl访问2026-10-01；
  回执`outputs/r7_n1_cost_v2_acceptance/registration_verification.json` SHA256`672f94750cd6e26ab16709d897b0bf0900ba5601e968eba21c8fb69b9e75f6f9`。
  逐交付物84项实证审计D0–D5齐有相应证据（P2条件不触发），回执`deliverable_audit.json` SHA256
  `9f234f8d541109ff5ef04a7197c1a6ea28bdf62f96eb20b0a404ed8d036e2d67`；407运行文件/v1 pins/原科学归档不变。
  本条进度尾提交与该精确登记CI SHA区分，不递归改证据页/index digest，不新增GPU或新授权，不自宣目标完成。
  CI回写后定向48passed/0.40s、37阻断/whitespace通过，84项审计回执再次只读验证一致。
- **旧目标再次触发后的只读核对（2026-10-01，起点5fe2ff4，0新增GPU-h）**：原始用户共驻扩围目标、执行时
  具名提问、随后20:03明确授权、20:05回执和20:06启动顺序已核，历史一次范围已用，不把重复的旧独占目标当新许可。
  本次193CPU反证passed（最终归档3.00s），84项交付物重验与另一个5226项相关产物校验exit0；81输入、六checkpoint、
  118活跃源/八计量源、31子进程与30零baseline、snapshot/三类CSV/四表/预算均核，407运行文件与v1三pins未变。
  准备/登记/交付CI归档精确SHA与九主步骤核齐，交付5fe2ff4的run36867579327 success；本次未联网或触发workflow。
- **校验范围限定**：原8592项历史checker全953-member/git blob读取会展开无关real_smoke/test.jsonl（历史UCI
  真实站点工程smoke，非本轮M2封存test）。不将此前「test manifest未读」泛化为所有test清单的完整证明；本次未展开
  任何test清单、不重跑原8592，独立命名5226项只核相关契约与不透明源/zip hashes。不据此推断M2 test用于评估，
  也不声称全历史syscall/未来逐位GPU成本证明；full snapshot不含allocation-history frames，v1具体残渣仍不能回填。
  新侧回执`outputs/r7_n1_cost_v2_readonly_followup_5fe2ff4/followup_verification.json`及精确checker/output SHA见修复长文。
  只追加两活文档进度尾，冻结证据/index/运行目录、账本4.2045/余19.7955、N1paused/cannot-distinguish/N2d提议均不改。
  37阻断、goal/index/campaign仍0失败（四历史notes），追加后相关48项CPU测试通过（0.27s）；本次进度尾未提交/未跑新CI，
  原精确交付CI不冒称覆盖新增文字。
- **验收边界等待（2026-10-01，0新增GPU-h）**：当前objective仍写「一卡独占」及「不读test」，独立目标校验
  未认可已授权共驻与历史无关smoke清单身份读取足以自行解除两字面边界。AskUserQuestion两题请用户明确共驻修订版
  是否用于验收、历史real_smoke/test.jsonl字节读取是否为禁令违反或仅本次具名审计例外；**未收到回答**，不是拒绝/
  接受，不能推定例外。等待回执`outputs/r7_n1_cost_v2_readonly_followup_5fe2ff4/acceptance_boundary_questions_unanswered.json`。
  原证据、授权、账本与科学读法不改，不新增GPU或test读取，也不为通过而改冻结判据。
- **用户明确授权/下放决策后裁定（2026-10-02，0新增GPU-h）**：用户直接回复「均显示授权，并且允许下方决策权
  给你，所有实验均可安排」。执行者据此选择按已授权shared-headroom共驻修订版验收，并仅接受已发生的无关
  real_smoke/test.jsonl归档/git身份字节读取为本次具名审计例外。用户授权与执行者具体选择分别记明，不把自动通知/
  未答回包当确认，不伪称历史独占或读取前已有例外许可；原未答记录保留，当前两项验收不再待裁定。
  决策0027/E-222/索引/CHANGELOG及新侧回执`outputs/r7_n1_cost_v2_acceptance_resolution_20261002/acceptance_resolution.json`
  SHA256`46ac8ba78a30f05898e18f4fa9276743d92c627767574eb8d1cf6389f3ef10eb`已登记。原407运行文件、授权/失败/冻结证据/
  evidence index、账本4.2045/余19.7955不改；不新增GPU重跑/test读取或放宽判据。本轮没有因总括实验安排权
  自动选择新实验，后续仍要具名协议/预算/失败规则及节点/停止门；N1paused/cannot-distinguish/N2d仅提议不变。
- **裁定登记验证（2026-10-02）**：相关CPU合集359passed/23.11s，全部37阻断/ADR/goal/campaign/index/空白通过，
  campaign四历史notes仍如实保留；407运行文件及12原证据SHA未变。新侧`final_verification.json`绑定两项裁定及D1–D5。
  新治理/进度未commit/push或运行新CI，不将原精确CI冒称覆盖新文字；预算/节点/科学读法均不变。
- **下一动作（仅提议）**：交用户/独立目标判定审阅当前授权裁定及既有成本审计证据；不自行宣告目标完成。
  用户已下放实验安排权，但本轮不自动进入下一节点或新增任何GPU运行。
- **M3 具名启动（2026-10-02，当前新增 0 GPU-h）**：用户本轮明确触发 N2a，仅 P-A/P-B/P-C 和三臂×两 seed×400 updates，
  ≤1.0 GPU-h；禁止自动进入下一节点或关闭 issue。起点 `5fe2ff4`，上一轮 six-file package 独立提交
  `e7755ae11d5aeb78038c2363c823c78639c691fa`；CI `36967236450` completed/success 九主步骤全绿，
  匿名 run/jobs API 访问 2026-10-02（响应临时路径 `/tmp/m3_adr_ci_runs_branch_20261002.json` 和 jobs 同名）。
  当前节点推进 N2a/active，是本次用户触发而非自动推进；N1 原科学结论、账本与四历史 notes 不改。
  当时新代码与新目标尚未提交/实跑，不用上述 CI 冒称覆盖它们；具名协议/sidecar 发布/工程门禁先于 GPU。
- **M3 工程门禁与一次执行启动（2026-10-02）**：工程提交 `d6c98cf1c33eca5885772c473805af3ef0ad62ba`
  的CI `36973907623` completed/success九主步骤全绿；本机2272passed/9skipped、精确clone2267passed/
  14skipped，skip不算通过。尺度sidecar identity4fed1c78…已排他发布，真实train186/188帧8proxy全active；
  三类时刻与固定train inverse入contract，model/source/data不改。CPU prepare冻结protocol404cf32b…、
  codezip18595abc…80相关源；按用户本轮具名书面授权启动3arm×2seed×400/K4，GPU1 UUID绑定、共驻门槛、
  1800s连续计费截止、失败即停不重试。终态尚未取得，账本暂未增，不拿工程CI代替GPU或科学验收。
  本条是活文档进度，未进被冻结实验code.zip，不改原证据；保持N2a，不自动进入N3或关闭issue。
- **M3一次尝试终态（2026-10-02）**：failed/budget_limited/partial，6train各400/selected400，7/30eval、
  119/510RMSE与131/528case；第八项seed41/input_aux/+24h父deadline timeout，自有child已terminate/reap。
  不再spawn/重试/生成完整paired/终态四成本，D5未达。GPU连续1790.1153369722888s=0.49725426027008024h，
  whole1805.1085775829852s超30min文字5.1086s，截止只约束GPU阶段留下CPU前置时间缺口，如实记预算门未完全兑现。
  新失败ledger0.4973，逐行显示4.7018/余19.2982；精确4.701714415227032/余19.29828558477297，余额不授权retry。
  独立stdlib审计7.54s核109files前后不变/9pins/80source+commit/6train7val/14launch/cost，receipt73afca69…
  为incomplete-audit-valid/full_experiment_accepted=false/D5false/budget_within_caps=false。证据页R7_73_PROCESS_SUPERVISION，
  实際登记E/index/CI另记尾；N2a转budget_limited，不推进N3、不关闭issue或自宣goal完成。
- **M3实际失败登记CI**：冻结页ed1a034/pageSHA3eb2ea34…，E-223–E-226/0028/索引audit-blocked与
  canonical brief17条/失败账本登记提交`055ee9f475d18c71fb3d03f5648463e1da560fda`；CI36978855723
  head_sha精确匹配completed/success九主步骤绿，本地445passed/70.52s（远端计数未取）。一手API访问2026-10-02，
  登记核验SHA570a2d81…、逐需求最终审计SHAb8785366…，109运行文件/80源/page仍不变。
  工程CI不替代D5缺口，登记仅审计失败，goal不自宣完成；本条尾核与登记SHA区分，不回改冻结页/index。
- **下一动作**：仅审阅本次失败证据与完整对照缺口；新的具名授权/预算决策前，不补跑、重训、扩围或进入N3。
- **实验下放治理落地（2026-10-02，0 新增 GPU-h）**：先归档计划 0009/0010 与三份待执行长文，独立提交
  `5058610744377e6729f12e49b7066b544a9e56aa`，CI `36993386517` 精确 SHA 匹配、completed/success、
  九主步骤全绿；匿名 run/jobs API 访问日期 2026-10-02。按计划 0010 D1–D10 落地
  `docs/decisions/0029-standing-experiment-delegation.md`（accepted），0021 标记 `superseded by 0029`，
  同步 AGENTS、CI/GPU 细则、两份 SKILL 与本主计划。state 只新增 `budget_mode=accounting-only`，
  cap/used/remaining 保持 24.0/4.7018/19.2982，`current_node=N2a`、`status=budget_limited` 与原 M3
  失败结论不改。此轮只实施治理，不执行新实验；**新规则下的实验调度留待新窗口**。届时按计划 0009
  与本主计划准备 N2a 补全、N3、N4、N5 与关闭轮，在每轮执行前写死 planned/hard 并冻结协议；
  旧 prepared 长文里的 0021/总额度/≤30 min 不是现行授权闸门，须在该轮执行记录中明确适用 0029。
  除保留项外不再逐次询问；不回溯改旧协议/归档，不自动推进节点、关闭 issue 或宣告目标完成。
- **治理精确 CI 尾核（2026-10-02）**：决策/规则/技能提交
  `c4d45cfb807cd953e3701219db50276dd4131f1e` 的 CI `36996404191` head_sha 精确匹配、
  completed/success、九主步骤全部 success，含 Check campaign state 与完整 unit/integration/wheel
  测试步骤。匿名 run/jobs API 访问 2026-10-02，响应 SHA 与逐项核验见计划 0010 文末；未取远端
  pytest 计数，本地三组治理测试 152 passed/22.53s 与补正后 152 passed/22.14s 不冒称远端计数。
  只读交付核对 176 项无缺项，独立审阅指出的旧结构示例误用风险已用适用声明限定；没有 GPU 实验、
  新数据、账本追加、节点推进或科学判定。本尾核与 D8 限定另作后续提交并核自己的 CI，不用 c4d45cf
  的绿冒称覆盖新增文字；M3 的 budget_limited/partial 与 23 项缺评估仍保留，不宣告 goal 完成。
- **0030 自主执行授权扩围与交接（2026-10-02，0 新增 GPU-h）**：用户明确下放实验、推进和普通
  决策权，只负责总体实验方向。新 `0030-autonomous-execution-and-direction-boundary.md` 取代0029，
  完整继承时长/会计契约，取代0025逐轮用户触发；现行AGENTS/技能/本主计划同步，可在同一goal
  按冻结前置/出口与每节点对表连续推进，独立审阅不是新许可，最终goal完成仍不可自宣。
  本轮只落治理与计划0011/新目标 `docs/goals/v2-autonomous-completion-and-closeout.md`，
  当前节点/状态/账本/原M3失败与旧归档均不改，prepared交接不当作实验。N2a补测→N3→N4→N5/关闭
  留待新窗口；六issue关闭仅0030具名证据/CI/非force ff条件齐备后，无main合并/force许可。
  本次起点256ef25；实际新验证/精确CI另登记计划0011，不冒称旧绿覆盖新文字。
- **0030治理/goal交接CI尾核**：提交 `b961b3b3084a47ccee8a3bdf2b22b46887055a0e` 的
  CI37024597813精确SHA completed/success九主步骤全部成功；匿名API访问2026-10-02，回执与
  响应SHA/1129项交付审计见计划0011。新goal2027字符/5goal结构0失败，入跟踪后152passed/22.19s；
  独立8文件审阅实质PASS且不作推进许可。991原文件、历史state/账本/进度不变，无GPU/训练/关闭。
  本尾核与术语补正另提交并核自己的CI，不自宣最终goal；下一窗口从新goal的N2a补测协议准备起步，
  后续在既定方向内自主普通决策/实验/推进，不逐轮等用户触发。
- **0030接续执行开工（2026-10-02，0新增GPU-h）**：起点`caea510f25cf7eb77bd65638773c561e8e92c25c`，
  两无关未跟踪项保留。指定四目标/计划/0030通读、campaign0fail/4notes、当前goal0fail、rules diff为空。
  新长文 `n2a-m3-validation-complement.md` 派生N2a，软1800/硬3600秒、0训练/只补23缺val；
  current_round_goal切新页，旧attempt/旧goal/证据保持失败只读，账本不增。只读归档审计与工程实现已委派。
  原M3 `descriptive_outcome` 的any-unresolved暂停出口保持，补齐覆盖不先验授权N3；普通推进权不取消冻结停止线。
  规划主链修正错误指针/种子互配/旧页回改/余额门/complete建议后，JSONverified=true；目标794字符。
  目标/规划/campaign/index CPU门禁93passed/0.48s，未跑补测或完整新工程CI。
- **N2a补测最终工程（GPU尚未启动）**：独立原109文件/24checkpoint/六selected400/七val身份审计完成，
  原失败全额保持；新五源码/两测试落地，归档隔离、同boot整轮anchor与两处写前路径反证齐。
  最终定向123passed/13.04s，完整CPU2395passed/9skipped/2warnings/233.74s（skip不算通过）。37阻断/
  campaign/两当前goal/index/brief/compile/空白exit0，43旧源码/目标及两无关文件hash不变；账本仍未增。
- **N2a实际补全（2026-10-02）**：工程/执行`011ab4cdc3b4fa9f6671a22962a2cf3227998c66`的
  CI37036869969精确九步success后，只一次prepare/run23val；canonical74d05ab3…09dd5、新zip e44ef233…bb43。
  0训练更新，原六训练/七val+新23=30eval/510RMSE/528case，三pair85/255aggregate齐；所有新eval成功、
  23独立零基线进程，25owned进程正常退出。新GPU644.3709621066228s=0.17899193391850632h，
  whole700.5363800507039s含prepare4.9935/间隔/CPU前后置/聚合清理，overrun0；原失败成本与109字节不改。
- **N2a冻结出口实际触发**：future−off=19/45/21、input−off=21/36/28、future−input=3/44/38
  （改善/恶化/unresolved）；原any-unresolved暂停，attempt paused/完整覆盖，非partial/budget_limited。
  归档比较器metadata精确重放一致，未二次GPU重跑。不改冻结出口，N3/N4/N5前置BLOCKED；六issue匿名
  访问2026-10-02仍open，无Closes/main。账本新成本与索引正在登记，不以stage快照代最终whole。
- **最终封印与报告复核**：stdlib独立67380检查/4.3066s实际exit0，276新/109旧rawhash前后不变、
  source/protocol/code/三来源/四指标/per-case/13表/所有owned退出/真实连续时钟核齐；旁侧seal
  0810665b…97a72/verification576a72b2…2ab79 accepted-engineering，原科学暂停不变。有限报告复核关键
  数值一致，首处driver011与原归档evaluator身份区分已修。补测后定向221passed/17.50s，0skip。
- **版本化登记BLOCKED**：单独证据commit的PreToolUse被Mimosa L3拒绝，报告只读legacy12high/1low、
  覆盖不完整；整条提交命令未执行、HEAD保持011ab4c，未改归档/安全hook或绕过。证据页/E/目标与
  全额账本本地保存，新增行0.1790/总4.8808/余19.1192；新index尚无可达evidence_commit，故canonical
  17条保持、新record仅旁侧pending草稿、登记CI未取得，新增第五note不冒称机器核数。
- **阻塞终态验证**：最新完整CPU2395passed/9skipped/2warnings/223.13s，定向221passed/17.50s；
  37阻断/两当前goal/index17/compile/空白exit0，campaign0fail5notes（新账本未登记note保留）。
  原109/新276/43开工pins及两无关文件hash再核不变，七文档本地暂存未commit；HEAD与origin工作分支
  仍011ab4c、main仍dafd22e，不用工程CI冒称覆盖新证据/账本。
- **下一项受阻的具体动作**：安全闸门与只读归档冲突须用户裁定后才可冻结证据commit/正式index与
  精确登记CI。N2a另因原科学出口暂停，N3/N4/N5前置BLOCKED；不额外seed/训练/阈值改动闯关，
  不关闭issue或自行宣布最终goal完成。
- **安全登记阻塞分诊（2026-10-03，0新增GPU-h）**：按获批计划0012保存原始拒绝/警告的SQLite只读投影，
  三类事件不混：工程scanner_enobufs、证据commit12high/1low强制拒绝、Stop两failed/零scanned的ETIMEDOUT。
  12处归档报告有限静态输入/调用链与九源码hash齐，常量反证不自动判误报；没有执行归档或全项目深扫。
  366跟踪归档/109旧/276新seal/43科学pins重核无差异，安全配置未改；相关CPU532passed/38.34s/0skip，
  首次仓内tmp使conventions反证24failed/508passed，日志保留，只改tmp到仓外后同集合通过。
  官方安装包未提供已证实L3归档例外或非降级纠正接口，按硬停点保持BLOCKED，不重试commit/push、
  不改ignore/插件或外发；诊断 `outputs/r7_m3_registration_diagnostic_20261003T024214Z/`，分诊正文
  `docs/R7_SECURITY_SCAN_TRIAGE.md`。未进入A/B/C，canonical17/五notes及账本不变；精确登记CI仍无。
  当前工程依赖是官方支持处置并实际门禁放行；政策范围变化需另有明确授权和接口，科学暂停仍独立。
- **用户关闭Mimosa后的D6恢复（2026-10-03，0新增GPU-h）**：用户先答「我对其进行明确授权」，再说明
  「我已经关闭了Mimosa，可以继续进行」。只读核用户与仓库本地插件开关false，执行者未改安全设置，
  本地配置差异不提交。没有官方finding处置/L3 clearance、新深扫、升级或支持工单外发；原拒绝与
  覆盖缺口保留，不再把此前官方支持前置当作当前用户关闭工具后的登记条件。只继续正常D6/git/SSH/CI，
  仍拒绝即停不换路，不训练/评估/推进/main/关闭，N2a/paused与冻结出口不变。
- **真实证据冻结与本地正式登记**：A=`1de3a672d859b42efa7a5fac3293c2832df4a6ee`只含补测证据页，
  blob SHA256`17c446253af0e6df838cf60ef02800b7d64c6cce0490b981ddfc5e22789b4a85`；index单增
  `m3-validation-complement-complete-paused` audit/blocked记录，旧17条原文不变、canonical brief18。
  账本新补测行增加record数值绑定，不重复GPU-h，四历史notes保留；完整CPU2395passed/9skipped/
  2warnings、230.96s，skip非通过。原109/新276/366归档/43科学pins恢复前不变；B提交/精确登记CI待
  实际绑定，不以37036869969代替。本次回执仓外 `/tmp/r7_m3_registration_resume_u8t215bd/`。
- **D6正式登记CI已实核（2026-10-03）**：B=`da4939e613eb1a0b39f6303142ffd7aa6ab8657c`正常SSH推
  工作分支，CI37108664102/job111162098631精确SHA completed/success九主步骤成功；匿名
  [run API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37108664102)与
  [jobs API](https://api.github.com/repos/Eswink/UrbanPiDiT_R2/actions/runs/37108664102/jobs?per_page=100)
  访问2026-10-03，核验回执SHA256`b9184dc513cddea1d9c8748b09e98b2720194668ead9ed64a396007055e161bb`。
  正式index18/canonical brief与账本新record齐，campaign0fail4历史notes；532定向/37.67s/0skip、
  完整2395/9skip/2warnings230.96s均为本地计数，远端日志计数未取。原failed与全部成本不改。
- **本次收尾审计**：31项原目标对表仅D6从工程缺失改已登记，17核对全过，回执SHA256
  `7ec1a3a03966da8af2f6fcb3b473b2efb61722252ef9f2e9a0a16301ac27134a`；旧109/新276/366归档/43科学pins
  和用户配置/两无关文件不变，359活跃Python只AST、不编译归档。12文档独立复核无必修，不是最终
  goal判定。用户开关false不证明扫描全停，仓外草稿Write增量拒绝与纯只读候选修正另见计划0012，
  未改设置或claim安全。C只进度/plan尾另核自身CI，不递归改冻结证据/index；main refs未动。
- **当前研究出口**：D6登记工程阻塞解除，不取消any-unresolved/N2a-paused，N3/N4/N5依赖仍BLOCKED。
  不加算力闯关、不关闭issue或自行宣布goal完成；下一研究动作仅独立审阅negative/mixed证据与总体
  方向，当前不执行新节点、GPU、数据或关闭。
- **N2a交付验收与全权委托接续（2026-10-03，计划0013）**：用户问「N2a是否已经已经完成，如果未完成，
  则计划完成，并且明确所有授权全部下放给你。」规定D1–D6及补测D0–D4已有真实交付；原目标允许
  negative/mixed结构正确验收，因此不再列缺23val/D6或要求辅助臂取得正增益。A/B/C及精确B/C九步CI
  已存在；C=`20230567eb4272b9fd70a73e0f6e5dc8af34ed63`/CI37109369425的仓外回执SHA256
  `2a451777c4d4a2791fd0adfbd9201380427df2b1ec18ea2ec9517d0aa152f1dc`，不以旧时点pending文字推新缺失。
  全部授权下放作为0030接续确认，本请求所需执行/修复/实验安排/预算/验证/普通决策自主承担，
  不逐项问。执行者据已齐事实选择0新增GPU-h六文档验收留痕，不是因授权不足而停。
  **交付完成≠科学成功≠可推进**：原failed/whole超cap、complete-paused、any-unresolved/advancefalse均
  保留，machine state/current_node/status与账本不改，N3–N5/main/关闭不执行，整个goal不自裁complete。
  MetPy意向/额外真实17chCPU端到端/更强analytic/三seed不事后升N2a完成硬门；未做与弱覆盖保留。
  新计划验收矩阵见 `docs/plans/0013-n2a-delivery-acceptance-and-delegation.md`，本次文字自身CI另核。
- **独立主线接续（2026-10-03，起点edcc335）**：用户明确终结M3辅助监督假设，保留其旧出口，
  选择aux_off或合法RW-A母体进入独立autoregressive-exposure路线；0032/新remaining-stages长文
  记录授权、交付、来源和停止条件，current=N3/previous=N2a。旧科学页、协议、失败、全部paused
  与advancefalse不改；旧§8历史文字按其写入时点保留，不据此再次制造总前置。
  初次campaign0fail4notes，新B5400软/10800硬、新C10800软/21600硬整轮预声明，0新增GPU-h。
  父身份、代码接线及D4反证独立委派，尚无新实验结果或精确新CI，不把文档准备当研究进展。
- **N3真实执行与原失败登记（2026-10-03）**：616b029精确主CI37133487340九principal及三post成功；真实旧包FP32/BF16小探针共8updates成功，连续0.04711594580465721 GPU-h。B六训练1600updates和三十val均success，但训练objective的binary64重构拒FP32导出损失，原attempt failed/finalizedfalse，连续0.8526056814201487 GPU-h/whole3200.912431293167s/overrun0；400两步loss按真实FP32乘加严格一致。原361pins/全成本独立核验，只接受原失败事实与成本，不宣称天气比较已接受。
- **A冻结与正式索引追加**：42ecb890ce8f8f1d2925f8841a74859b2decd8f7只冻结precision与B失败两证据页；实际Git blob字节SHA982bd632…a1f30/0f25cba3…a0cdc。正式index18→20仅追加两record、旧49755-byte前缀SHA3e0b83aa…8c0f不变，brief canonical；两账本行绑定record，0.8997216272248059新GPU-h不重复计。精确累计5.780427976370344、显示5.7805/会计余18.2195，无总许可上限。登记提交与其精确CI尚待，不以616工程绿冒登记绿。
- **独立反证修复进行中**：0035活跃known/source接口集成后220定向通过，但独立118pass/3fail/2deselected揭露B3活跃子集anchor未切片、合法20分钟offset浮点表示误拒，以及旧与新RW-B均存在BF16 fixed/streamed cast-cache梯度兼容差异。原FAIL审阅/40.84s软超/未达1800硬限保留；两真实缺陷修活跃实现，BF16不放宽断言，C仍FP32 fullBPTT。统计补全38pass之后另有三个独立闸门反例（提前失败未封印、source双seal身份不充分、最后hash越硬限），旧FAIL不改、真实统计尚未消费；修后新独立CPU600/1200，不重B GPU。
- **当前具体动作**：完成统计补全闸门修复与独立窄复验后实际零GPU重聚合/选择/登记；两活跃模型缺陷复验、稳定新版fullsuite/精确CI后实跑原K3参照、新包精度、独立C9train/135eval及adaptive。UTC仅真实逐case充分统计派生。六issue尚未结题或关闭，main refs未动，不自判科学收益或最终goal完成。
