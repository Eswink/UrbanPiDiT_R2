# main-model-v2-campaign：主模型 V2 的 campaign 主计划与每轮对表

**状态：campaign 级主计划（活文档）。当前节点 N1（转向审计）。最近完成的一轮：RW-B 减法轮（不能归因）。**

本文件是「主计划 + 每轮 recheck」机制的**唯一权威**：节点图（§2）、每轮开工前必须走的对表清单（§3）、
预算账本（§7）与进度块（§8）都在这里。每一轮的目标长文是它的**派生物**，不是平行的第二处真相——
`docs/R7_TASK_QUEUE.md` 对 V2（#70–#75）这条线**不再维护**（它停更于 2026-09-28，CI 数字已过期），
保留原位只作历史。机器可查部分由 `tools/check_campaign_state.py` 执行（决策 0025）。

## §0 Objective（可粘贴；实测 365 字符）

> 把主模型 V2（#70–#75）的后续推进做成「主计划 + 每轮对表」的可审计循环：主计划是 docs/goals/main-model-v2-campaign.md 的节点图、对表清单与账本；每轮开工前必须对表（节点一致性、账本算术、上一轮登记、冻结判据未被改动），任一条对不上就停下问用户，不得自选新方向；当前节点 N1 是转向审计（forecast state / training objective / data regime / autoregressive exposure，0 GPU-h）加一条预声明可证伪臂（≤0.45 GPU-h），它的目标长文是 docs/goals/main-model-v2-pivot-audit.md；N1 的出口是四方向证据强度排序与下一节点提议，执行者只可提议、不自行宣布完成。

## §1 现状（带 file:line，2026-09-30 核对）

- 分支 `r7/weather-reasoning`，HEAD `4bf92e9`；open issue 恰为 **#70–#75**（六个，2026-09-30 匿名 API 核对）。
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
- **本机制目前在仓库里不存在**：GPU 账本此前是手写散文、无脚本可核；`tools/check_goal_brief.py` 只查结构、
  非阻断、不进 CI；没有任何工具比对「主计划当前节点」与「上一轮 §8 下一动作」。
  本文件 + `tools/check_campaign_state.py`（决策 0025）就是补上这一层。
- 工程态：37 条阻断规则 0 违规；门禁是 `R7 CPU CI` 的八步（新增 campaign 检查后为九步，
  历史证据页里的「八步」是它们当时的实测，**不回溯改写**）。

## §2 交付物：节点图

每个节点是一个**可独立启停的循坏步**，含：依据、输入、预声明门禁/证伪、禁止项、预算、出口。
节点之外的任何方向都不得自行开跑（§3 第 6 条）。

| 节点 | 名称 | 依据（冻结文档） | 预声明门禁 / 证伪 | 预算 | 出口 |
| --- | --- | --- | --- | --- | --- |
| N0 | M2-B / RW-B 实现与有界对照 | `docs/goals/main-model-v2-rw-b-round.md`、`docs/R7_72_RW_B_PILOT.md` | 已执行：登记主端点 negative，非证伪 | 已用 0.8356 | **已终结**（negative，两轮复现） |
| N1 | 转向审计 + 一条可证伪臂 | `docs/goals/main-model-v2-rw-b-subtraction.md:206-208`、`docs/plans/0004-r7-main-model-v2.md:323-325` | 0 GPU-h 审计四块；臂：Z 换冻结随机张量，与 RW-B 配对（沿用 #60 比较器、depth 0、逐 seed 同号三态）；两种结局的读法在长文 §3 预先写定 | ≤0.45 GPU-h | 四方向证据强度排序 + 下一节点提议（N2a–N2d 之一） |
| N2a | 进 M3（#73 尺度修复 + 三类时刻语义） | issue #73、`docs/plans/0004-r7-main-model-v2.md` M3 行 | 只在 N1 审计把 M3 列为**最强**支持时开跑；门禁＝M3 自己的验收（scale/eps floor 与三类时刻的定向测试 + 一轮有界对照） | 届时重定（≤1.0 GPU-h 量级） | M3 验收 |
| N2b | 目标对齐修正（训练目标 vs 报告时效） | 计划 0004 `:325`、#64 curriculum 证据 | 必须先有 N1 的目标审计结论；改动要预注册、不得新增阈值与端点 | 届时重定 | 目标对齐后的重测 |
| N2c | 数据/评估支撑修正（val 跨度、窗口数、气候态） | 决策 0008/0010、`docs/R7_B2_MULTISEED.md` | 只改评估与数据支撑，不改判据；任何扩大数据需用户授权 | 0–0.2 GPU-h | 可分辨性结论 |
| N2d | 停止（在当前数据与算力下不可分辨） | `docs/goals/main-model-v2-rw-b-subtraction.md` §5 停止条件 3 | 审计给出「不可分辨」的量化读法 | 0 | 如实记录并停 |
| N3 | M4（#74 两步可微自回归，区分内部 K 与预测时效） | issue #74 | 只在 N2 出口放行后开跑 | 届时重定 | M4 验收 |
| N4 | M5（#75，最小实验闭环与有条件自适应） | issue #75 | **默认不启动**：adaptive 前置 gate 不成立，且决策 0023 的 matched-Generic 对照仍缺失 | — | — |

**常设禁止项**（适用于所有节点，除非本文件显式改写）：不新增 solver 部件；不改已冻结判据、阈值、端点或案例集；
不重跑或改写归档产物；不读封存 test；不下载数据、不租 GPU；不动 main、不 force push、不合并、不关闭 #70–#75；
不自动进入下一节点；skip/queued/cancelled 不算通过。

## §3 判据与每轮 recheck 清单

**每轮开工前必须逐条走完，并把这六条的结论写进该轮 §8：**

1. **节点一致性**：本文件 §8 的 `current_node` 与上一轮长文 §8 的「下一动作」必须指向同一节点。
2. **账本算术**：§7 账本逐行求和 = 已用、24 − 已用 = 余量（容差 1e-3）；每行必须有证据指针。
3. **上一轮登记**：上一轮长文已在 `docs/R7_EVIDENCE_INDEX.jsonl` 里有一条 `evidence_path` 指向它的记录，
   且该记录的 `evidence_commit` 在 git 里存在；新长文过 `tools/check_goal_brief.py`。
4. **本轮 objective 派生**：写明它实现的是哪个节点、引用哪些冻结文档；禁止项照抄 §2 的常设清单。
5. **判据未被改动**：`docs/rules/` 与已冻结证据页在本轮开工前没有未记录的改动。
6. **不同意就停**：第 1–5 条任一条对不上 ⇒ **停下问用户**，不得自行选择新方向或自行改写判据。

**机器可查部分**：第 1、2、3 条（含「无机器支撑的账本行如实列出」）由
`.venv/bin/python tools/check_campaign_state.py` 执行，退出码 1 表示硬漂移；
第 4、5、6 条是人工动作（第 5 条用 `git log`/`git diff` 核对）。工具不判断科学方向——
它只能证明"没有跑偏到与主计划矛盾"，**不能**证明方向正确。

判据来源：本文件 §2 的冻结文档指针、`docs/rules/ci-and-verification.md`（运行资格与门禁）、
`docs/decisions/0021-experiment-authorization-channel.md`（实验授权）、
`docs/decisions/0023-main-model-first-baseline-freeze.md`（基线冻结与 matched-Generic）、
`docs/decisions/0025-campaign-master-plan-and-per-round-recheck.md`（本机制本身）。

## §4 实施顺序（每轮固定动作）

1. 读本文件 §8 的 `current_node` 与上一轮 §8 的下一动作 → 跑 `tools/check_campaign_state.py`（§3 第 1–3 条）。
2. 从当前节点**派生**该轮目标长文 `docs/goals/<slug>.md`（骨架见 `.agents/skills/goal-loop/SKILL.md`），
   过 `check_goal_brief.py`，把 `<!-- round-node: N? -->` 与 §8 一起写好。
3. 执行该轮（0 GPU-h 的审计先做；需要 GPU 的那一步在执行那一刻按决策 0021 取授权并重读账本）。
4. 收尾：证据页 + E 条目 + 索引记录 + CI 绑定 + 更新本文件 §8（当前节点 / 账本 / 下一动作）。
5. **本轮结束即停**：给出「recheck 结果 + 下一轮提示词」，等用户触发；不做后台续跑、不重开定时器
   （`docs/R7_MANUAL_ITERATION.md:3-6` 的用户决定）。

## §5 预算与停止条件

- 批次授权：第二批 **≤24 GPU-h**，自主分配（`docs/goals/full-auto-campaign.md:39`）；单次实验 ≤30 min。
- 节点内自设上限见 §2；执行前必须重读账本（§7），不得沿用其它文档里的旧数字。
- **停止条件**（满足任一即停并向用户报告）：预算用尽或账本不足；本轮需要新判据；
  本轮结论为「不可分辨」或「不能归因」（→ 提议 N2d）；需要新数据、租 GPU、合并 main 或任何破坏性操作。
- 目标状态 `active / paused / budget_limited / complete`（`docs/goals/README.md:22`）；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不新增 solver 部件；不把「12/24h 的小改善」当加码理由；不把 role 标记的小幅改善写成机制声明。
- 不重跑、不覆盖、不改写任何归档产物与历史证据页（`outputs/` 下的一律只读）。
- 不把三个 seed 写成显著性；不为让报告好看而放宽判据或弱化测试。
- 不建立定时任务或后台续跑（用户 2026-09-23 决定，`docs/R7_MANUAL_ITERATION.md:3-6`）；
  harness goal 只作 best-effort（决策 0024），不作为完成判定。

## §7 账本（第二批 ≤24 GPU-h；每行必须有证据指针）

| 轮次 | 实测 GPU-h | 累计 | 证据 |
| --- | --- | --- | --- |
| M2 段（桶 4→8） | 1.204 | 1.204 | `docs/R7_69_BUCKET_EXPANSION.md:23`（15 份 `training_report.json` 求和 4333.9 s） |
| V2 第一轮（M1 + RW-A） | 0.1846 | 1.3886 | `docs/R7_71_72_M1_AND_RWA.md:14` |
| V2 第二轮（四臂 × 三种子） | 0.5253 | 1.9139 | `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`；长文 `docs/goals/v2-round-two-attribution.md:149` |
| V2 第三轮（M1 容量控制） | 0.5489 | 2.4628 | `docs/R7_71_72_ROUND_THREE.md:14` |
| RW-B pilot（四臂 × 两种子） | 0.8356 | 3.2984 | `docs/R7_72_RW_B_PILOT.md` + 索引记录 `record:rw-b-bounded-round-negative`（`gpu_hours = 0.8356`，机器可核） |
| RW-B 减法轮（四臂 × 两种子） | 0.4128 | 3.7112 | `docs/R7_72_RW_B_SUBTRACTION.md` + 索引记录 `record:rw-b-subtraction-round-cannot-attribute`（`gpu_hours = 0.4128`，机器可核） |
| **合计已用** | **3.7112** | — | 24 − 3.7112 = **余 20.2888 GPU-h** |

说明（如实）：**账本不是从证据索引机械累加的**——索引只覆盖其中两行（其余四轮没有索引记录），
所以 `check_campaign_state.py` 做的是「逐行算术 + 证据指针存在性 + 有索引者数值一致」，
没有索引支撑的行会被**列出来**而不是被当成已核。把索引补成全量账本是将来可做的一件事，本轮不做。

## §8 进度块

<!-- campaign-state: {"current_node": "N1", "previous_node": "N0", "current_round_goal": "docs/goals/main-model-v2-pivot-audit.md", "previous_round_goal": "docs/goals/main-model-v2-rw-b-subtraction.md", "previous_round_evidence": "docs/R7_72_RW_B_SUBTRACTION.md", "previous_round_predates_mechanism": true, "cap_gpu_h": 24.0, "used_gpu_h": 3.7112, "remaining_gpu_h": 20.2888} -->

- **状态**：`active`（N1 尚未开工；上一轮 N0 已终结为 negative）
- **已完成**：N0（RW-B 实现 + 有界对照 + 减法归因轮，结论 negative 与"不能归因"，见 §1）
- **未做**：N1 的四块审计与那条可证伪臂；M3/M4/M5；确认轮（≤2.5 GPU-h）未动用
- **下一动作**：N1 开工 —— 先跑 `tools/check_campaign_state.py`，再执行
  `docs/goals/main-model-v2-pivot-audit.md` 的 D1（0 GPU-h 四块审计），
  完成后再按决策 0021 取 D2（那条臂）的授权
