# main-model-v2-campaign：主模型 V2 的 campaign 主计划与每轮对表

**状态：campaign 级主计划（活文档）。当前节点仍N1；本轮主长lead不可分辨，已paused，只提议N2d停止，不宣告目标完成。**

本文件是「主计划 + 每轮 recheck」机制的**唯一权威**：节点图（§2）、每轮开工前必须走的对表清单（§3）、
预算账本（§7）与进度块（§8）都在这里。每一轮的目标长文是它的**派生物**，不是平行的第二处真相——
`docs/R7_TASK_QUEUE.md` 对 V2（#70–#75）这条线**不再维护**（它停更于 2026-09-28，CI 数字已过期），
保留原位只作历史。机器可查部分由 `tools/check_campaign_state.py` 执行（决策 0025）。

## §0 Objective（可粘贴；实测 365 字符）

> 把主模型 V2（#70–#75）的后续推进做成「主计划 + 每轮对表」的可审计循环：主计划是 docs/goals/main-model-v2-campaign.md 的节点图、对表清单与账本；每轮开工前必须对表（节点一致性、账本算术、上一轮登记、冻结判据未被改动），任一条对不上就停下问用户，不得自选新方向；当前节点 N1 是转向审计（forecast state / training objective / data regime / autoregressive exposure，0 GPU-h）加一条预声明可证伪臂（≤0.45 GPU-h），它的目标长文是 docs/goals/main-model-v2-pivot-audit.md；N1 的出口是四方向证据强度排序与下一节点提议，执行者只可提议、不自行宣布完成。

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
| N1冻结随机Z审计（三臂 × 两种子） | 0.3590 | 4.0702 | `docs/R7_N1_PIVOT_AUDIT.md` + 索引记录 `record:n1-pivot-audit-frozen-z-unresolved`（`gpu_hours = 0.3590401737619605`；账本四位小数舍入，差≤1e-4） |
| N1独立evaluation成本补测（failed，1/30） | 0.0060 | 4.0762 | `docs/R7_N1_COST_SUPPLEMENT_ATTEMPT.md` + 索引记录 `record:n1-evaluation-cost-supplement-failed`（`gpu_hours = 0.005956627869357666`；失败全额计费，四位显示舍入） |
| **合计已用** | **4.0762** | — | 24 − 4.0762 = **余 19.9238 GPU-h** |

说明（如实）：**账本不是从证据索引机械累加的**——索引覆盖其中四行（其余四轮没有索引记录），
所以 `check_campaign_state.py` 做的是「逐行算术 + 证据指针存在性 + 有索引者数值一致」，
没有索引支撑的行会被**列出来**而不是被当成已核。把索引补成全量账本是将来可做的一件事，本轮不做。

## §8 进度块

<!-- campaign-state: {"current_node": "N1", "previous_node": "N1", "current_round_goal": "docs/goals/n1-cost-supplement-repair.md", "previous_round_goal": "docs/goals/main-model-v2-pivot-audit.md", "previous_round_evidence": "docs/R7_N1_PIVOT_AUDIT.md", "cap_gpu_h": 24.0, "used_gpu_h": 4.0762, "remaining_gpu_h": 19.9238, "status": "paused", "next_node_proposal": "N2d"} -->

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
- **下一动作**：D2过定向、精确clone全量与工程CI后，在执行那一刻按决策0021取具名授权（探针+30val、
  共驻、≤0.25GPU-h/≤1200s、失败全额计费即停不重试）。未获授权不执行GPU；N1保持paused，
  cannot-distinguish与N2d提议不变，不推进节点或宣告目标完成。
