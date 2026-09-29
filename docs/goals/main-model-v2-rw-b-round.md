# main-model-v2-rw-b-round：主模型 V2 的第二阶段（RW-B 局部门控求解状态）

**THIS FILE IS A PROMPT TEMPLATE. GOAL MODE HAS NOT BEEN STARTED.** ← 已过期；原句保留以标明出身。

事实上：本文件**已被作为目标启动**——`/goal` 于 2026-09-29 21:46（本地）起跑第 1 轮，主回合在
2026-09-30 00:02 正常结束（该轮结论见 §8 与 `docs/R7_72_RW_B_PILOT.md`）。轮末的自动完成校验
（迭代 1）在客户端侧**悬挂 1773 秒且没有任何超时**，00:31 由用户暂停中止，目标转 `paused`；
成因、证据与恢复步骤见 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` §2，约定见决策 0024。
因此下一段「不是已启动的 goal、也不是执行记录」只描述它**启动前**的身份，现在不成立：
本文件是活的目标源，§8 是它的进度块。

本文件是**下一轮 goal 模式**的长文目标源，供人工审阅后手动启动。它**不是**已启动的 goal，
也不是一份已执行工作的记录；本轮（2026-09-29 Plan Mode）只归档计划、设计与本提示词。

配套文档：计划 `docs/plans/0004-r7-main-model-v2.md`；设计契约
`docs/R7_MAIN_MODEL_V2_DESIGN.md`；外部台账 `docs/R7_MAIN_MODEL_V2_REFERENCES.md`；
决策 `docs/decisions/0023-main-model-first-baseline-freeze.md`。

---

## §0 Objective（可粘贴；实测 990 字符）

> 实现并验证 UrbanPiDiT-R7 主模型 V2 的第二阶段：先做两轮预注册但未执行的诊断（E0），再落地 RW-B（局部门控空间求解状态 Z：共享 step 收敛、role 标记、锚定 X_t 的提案、逐位置门控），为 #72/#73/#74/#75 提供真实模型代码与有界证据。判据与边界见 docs/goals/main-model-v2-rw-b-round.md。交付物：D1 E0 诊断（只用 val、只读已归档 checkpoint、0 GPU-h，回答「读取是否把 solver context 拉偏」与「三轮 E−A 与二轮 C−B/D−B 的 0.25–0.29 K 是否跨运行稳定」）；D2 RW-B 真实模型代码（新增 model/local_solver_state_r7.py，接线 model/process_forecast_r7.py、model/recursive_weather_r7.py、model/r7_halting.py、training/r7_streaming.py；新开关 local_solver_state 默认关闭且与前实现逐位等价；三处 step 收敛为一个共享函数）；D3 定向测试（梯度非零、门控边界、K=0/1/2/4、odd grid、poison future 字段、resume、BF16、fixed/streamed/adaptive 等价）；D4 至少一轮有界真实 M2 train/val 对照（协议在第一次优化器更新前冻结；逐变量 ×6/12/24/48/72h 全表 + 三态计数 + 参数量/FLOPs/显存/墙钟四表；2 seed × 400 updates；≤1.0 GPU-h；执行前按决策 0021 取授权）；D5 证据文档 docs/R7_*.md 与适用时的验证回执；D6 如实写明未做与 limitations，negative 原样保留。禁止：新增 baseline family、重新规划已冻结路线、下载新数据（除非计划写明的触发条件满足）、改动 main、force push、关闭 #70–#75、自动进入下一轮。预算 Pilot ≤1.0 GPU-h（确认轮另立 ≤2.5 GPU-h）；停止条件为预算用尽，或 E0 判定必须先停并向用户报告。不要自行宣布目标完成。

启动方式（复制下面这一行，不需要其它前缀）：

```
/goal 实现并验证 UrbanPiDiT-R7 主模型 V2 的第二阶段：先做两轮预注册但未执行的诊断（E0），再落地 RW-B（局部门控空间求解状态 Z：共享 step 收敛、role 标记、锚定 X_t 的提案、逐位置门控），为 #72/#73/#74/#75 提供真实模型代码与有界证据。判据与边界见 docs/goals/main-model-v2-rw-b-round.md。
```

## §1 现状（带 file:line，2026-09-29 核对）

启动后**必须先自己重读**，不得直接采信本节数字（HEAD 可能已前进）。

- 分支 `r7/weather-reasoning` HEAD `e1d5a7d73aefbc8f1402da484214f14c988e8146`；
  `main` `dafd22e0e59dac12c1ce2a3cc16f614e1ce8d776`；open issue 恰为 #70–#75。
- **#71/M1 已实现且已跑三轮**（`model/spacetime_conditioning_r7.py`；证据
  `docs/R7_71_72_M1_AND_RWA.md`、`docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md`、
  `docs/R7_71_72_ROUND_THREE.md`）。第三轮读数：收益**信息主导**（占 B−A 的 87–104%）。
- **#72 RW-A 已实现**（`model/process_readout_r7.py::PositionalProcessReadout`）。第二轮判决：
  位置依赖本身**未获支持**（C−D = 8 改善/13 恶化/64 未决）。
- **RW-B / M3 / M4 / M5 未实现**（二轮 §10、三轮 §13 明文列出）。
- 两轮各留一个**预注册但未执行**的「下一项第一个具体动作」：二轮 §11（诊断 correction 幅度/符号）、
  三轮 §13（0.25–0.29 K 效应的跨轮稳定性）。这两件是本轮的 E0，**不是可选项**。
- 已知陷阱：`total_updates` 不在 runner 契约里而学习率是它的函数（resume 端点陷阱）；
  协议 digest 不在 checkpoint 里；`model/**` 任一字节变化都会改 `model_code_sha256`。
- 本机 conventions 在 HEAD 上是干净的（37 条阻断、0 违规）。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | E0 诊断：二轮 §11 + 三轮 §13 的预注册动作 | 一份只读诊断记录（命令、输入 checkpoint 的 sha256、逐 seed 数字、结论与反证）；**0 GPU-h、只读 val、不新增阈值** |
| D2 | RW-B 真实模型代码 | `model/local_solver_state_r7.py`（新增）+ 四处接线的 diff；`local_solver_state` 默认关闭时与前实现**逐位相同**的测试证据 |
| D3 | 共享 step 收敛 | 同一个 step 函数被 `forward` / streamed / adaptive 三处调用；三路数值等价测试 |
| D4 | 定向测试 | 新增 `tests/test_r7_local_solver_state.py` 等；`pytest -q` 的 pass/skip 计数 |
| D5 | 一轮有界真实对照 | `protocol.json`（第一步更新前冻结，含 `protocol_sha256`）+ 逐 seed 结果 + 逐变量 ×6/12/24/48/72h 全表 + 三态计数 + 参数量/FLOPs/显存/墙钟四表 + 实测 GPU-h |
| D6 | 证据文档与登记 | `docs/R7_*.md`；`docs/R7_EVIDENCE_INDEX.jsonl` 记录（若适用）；回执（若该运行有对应 profile） |
| D7 | 诚实收尾 | 「未做的事」与 limitations 明写；negative 原样保留；CI run id 与本轮 SHA 绑定 |

## §3 判据与证据来源

- **工程判据**：`python tools/check_conventions.py` 37 条阻断 0 违规；`pytest -q` 无失败
  （skip 需逐条给出原因）；`git show --check` 干净。判据来源见 `docs/rules/ci-and-verification.md`。
- **接口判据**：新开关关闭时与前实现**逐位等价**（对照 `tests/test_r7_switched_path_equivalence.py`
  的既有做法）；打开时 `forward` / streamed / adaptive 三条路径数值一致；未来字段 poison 后
  forward 与 halting 选择逐位不变；门控梯度非零（**允许门控为 0 就是允许切断必要梯度**）。
- **实验判据**：协议在**第一次优化器更新之前**以排他方式写入 `protocol.json` 并记录 digest；
  主端点（建议 t2m 6/12h）在运行前冻结并写明允许的退化；案例与 seed 按 #60 比较器显式配对；
  单位/案例集不一致 fail closed；**三位 seed 只给一致性，不引入任何显著性阈值**。
- **科学判据（本轮不要求达到）**：≥3 固定 seed、精确配对案例、公平信息预算、公平算力报告、
  代表性验证、冻结的最终 test、不确定度、负结果保留。见 `docs/plans/0004-r7-main-model-v2.md`
  的 Scientific gates 一节。
- **运行资格**：`queued` / `cancelled` / `skipped` / `partial` 的 run **不算通过**；
  实验 workflow 的 commit 标签只是触发意图，**不是授权**（决策 0021）。

## §4 实施顺序（按机制依赖，不跳步）

1. **fresh-read**：重读 GitHub（#70–#75 全文与评论、分支头、最近提交）、
   `docs/plans/0004-r7-main-model-v2.md`、`docs/R7_MAIN_MODEL_V2_DESIGN.md`、
   `docs/R7_71_72_ROUND_TWO_ATTRIBUTION.md` §11、`docs/R7_71_72_ROUND_THREE.md` §13。
   **已经冻结的路线不重新规划**；发现与本提示词冲突时以仓库/GitHub 现状为准并如实记录。
2. **D1/E0（先做，0 GPU-h）**：执行两个预注册诊断。若诊断本身需要**新判据**，先停下报告用户，
   不得自行新增阈值。
3. **D2/D3（实现）**：按设计契约实现 RW-B 与共享 step。新参数必须在 `isolated_stream()` 下
   **最后**构造以保持「关 = 逐位相同」；实测参数/FLOPs 增量并写进结果（目标 ≤ 现有主模型 25%，
   超出则先缩 hidden 再冻结）。**`model/**` 变化后提交信息必须带 `[model-digest-change]`**，
   并在 `docs/rules/CHANGELOG.md` 记录 digest 变化。
4. **D4（测试）**：与实现同轮完成，不等 CI 反馈。CI 在跑时推进**另一个**模块的离线测试，
   不要开无关的治理 issue 填空。
5. **D5（有界实验）**：执行那一刻按决策 0021 取授权（写明范围/预算/产物与证据/失败与 skip 处理），
   并**重读第二批次 GPU 预算账本**（不得沿用旧数字）。协议先冻结，再训练。
6. **D6/D7（收尾）**：写证据文档、登记、绑定 CI run id、写清未做与 limitations。

## §5 预算与停止条件

- **Pilot**：≤2 seed、每臂 ≤400 updates、单轮自设上限 **≤1.0 GPU-h**（与二/三轮同规格）。
- **确认**：只有 Pilot 支持后才补 ≥3 seed，上限 **≤2.5 GPU-h**（本轮可不动用）。
- 单次实验 ≤30 min，超时须拆分；不新下载数据（0 GiB）。
- **停止条件**（满足任一即停并向用户报告，不要自行扩大范围）：
  1. 预算用尽或账本显示不足；
  2. E0 判定必须先停（例如需要新判据、或诊断显示应先解决 seed42 异号）；
  3. RW-B 在工程正确的前提下仍为 negative —— 最多执行**一次**预声明变体，然后停止发明新模块，
     记录可反驳假设并转去重新检查 forecast state / training objective / data regime；
  4. 需要新数据、新 GPU 租用、合并 main 或任何 destructive 操作。
- 目标状态是 `active / paused / budget_limited / complete`，**执行者只可建议，不得自宣完成**。

## §6 与 planner 草案的差异

本轮 Plan Mode 未委派 planner 子智能体（任务边界由既有 issue #70–#75 与本仓三轮证据直接决定），
因此无「与 planner 草案的差异」可记。相对**用户提示词**的三处刻意偏离写在其计划归档
`docs/plans/0004-r7-main-model-v2.md`：M1/RW-A 改为「已完成+已判决」、goal 文件名改 kebab-case、
issue 评论落仓为草稿（API 写 401）。

## §7 明确不做

- 不新增或扩展 baseline family；不重跑 U-Net / AFNO / window 的容量或配方搜索；
  已冻结的基线实现与调参不再动（决策 0023）。
- 不重新规划已经冻结的路线；不重做第一/二/三轮已判决的消融（C1/C2/C3、M1、RW-A 的位置依赖）。
- 不把 `spatial_solver_feedback=True` 重开当作新方法（C2 已有负结果）。
- 不下载新数据；不读封存 test 做方法选择；不把旧 test 重新包装成 pristine。
- 不改 `data/raw|interim|processed`；不修改、移动或重命名归档快照。
- 不修改 main、不 force push、不合并、不 release、不租 GPU、不创建定时任务。
- 不关闭 #70–#75；不新增「为了让报告好看」的阈值或弱化测试。
- 不把三个 seed 写成显著性；不把「训练 loss 下降」写成任务收益。

## §8 进度块

- **状态**：`paused`（D1–D4 与 **D5 已完成**；2026-09-30 00:31 由用户暂停——轮末的自动完成校验
  悬挂 1773 秒且客户端侧无超时，见 `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` §2。执行者不自行宣布
  goal 完成；恢复或续跑前先按决策 0024 换掉 `new-provider-4` 这条校验路由）
- **已完成**：
  - **D1/E0**：两个预注册诊断实跑（0 GPU-h，CPU 111.1 s，只读 val）。记录
    `docs/R7_E0_DIAGNOSTICS.md`、`docs/rules/EVIDENCE.md` 的 E-196/E-197、驱动
    `scripts/study_r7_e0_{correction_replay,paired_stability}.py`、产物 `outputs/r7_e0_diagnostic/`。
    读数：读取**放大**修正幅度（C/B 能量比 1.31，三 seed 同号）但**不**使其与误差反相关；
    三轮 §13 的「t2m 72h 三 seed 同号」被推翻（seed42 反号，比较器自己已标 unresolved），
    48h 那一段跨两轮复现（−0.2235…−0.2468 K）。
  - **D2/D3**：`model/local_solver_state_r7.py`（Z 的 ConvGRU 式局部更新、逐位置门控、
    锚定 X_t 的提案、门控 token→native 展开）与 `model/process_step_r7.py`（**唯一一份 step**，
    由 fixed forward / streamed backward / adaptive 三处调用）。两个新开关默认关闭且与前实现逐位等价
    （既有 23 digest 测试未改动即通过，另有显式 `False` 与不写参数的逐张量相等 + 前向 `torch.equal`）。
    实测代价：**+314,898 参数（+10.61%，目标 ≤25% 达成）**、前向 FLOPs ×1.2097。
  - **D4**：`tests/test_r7_local_solver_state.py`（27 项）+ `tests/test_r7_shared_step_paths.py`（10 项），
    含边界反证、poison、odd grid、K=0/1/2/4、resume、BF16、三路一致、结构性防漂移。
  - **D5（本轮新完成）**：授权于 2026-09-29 在执行那一刻经决策 0021 通道取得（用户在
    `AskUserQuestion` 里选择「按上述范围跑」）。一轮有界真实对照：**2 seed × 4 臂 × 400 updates**，
    同 M2 store、同一协议族、全部从零训练；臂 = 池化参照读 / RW-A 加性位置读 / RW-B /
    RW-B+role 标记。协议在第一次 `optimizer.step()` 前以 `'x'` 冻结并回读重算 digest
    （`6f488742…`，8 个 run 全同）；只读 val；test 封存。
    **判决：登记主端点未获支持。** t2m 逐 seed 同号：12h **−0.079**、24h **−0.075** K（supported），
    6h **+0.118**、48h **+1.065**、72h **+1.580** K（worsened）；模态读数 worsened，
    改善比恶化小一个数量级。两条登记证伪条件**均未触发**（6h 恶化但 12h 支持；五个时效全部
    sign-consistent），所以严格读法是「模态不支持且长时效受损」，**不是**「被证伪」。
    role 对照（RW-B+roles − RW-B）五个时效两 seed 全同号为改善，但只有 0.006–0.028 K，
    且 48h 的聚合计数反向（2 改善/10 恶化）——**不是**「role 标记有用」的证据。
    实测 **0.8356 GPU-h**（自设上限 1.0，未超；含一次**失败尝试**的 0.4203，见下）；
    参数量 RW-B 相对 RW-A **+314,898（+10.61%）**、前向 FLOPs **×1.2097**、
    训练峰值 allocated **+27.9 MiB**、墙钟 ×1.068。产物 `outputs/r7_72_rw_b_pilot/`
    （protocol、逐 seed、merged、paired、arm/training/memory/rmse/case/depth_probe 六张表）；
    证据文档 `docs/R7_72_RW_B_PILOT.md`；台账 E-198/E-199/E-200；
    证据索引 `rw-b-bounded-round-negative`（outcome_class `negative`，priority 78）。
  - **D6/D7**：`docs/R7_72_RW_B.md` 加了带日期的更新块（D5 已执行、指向新文档，不改写原文）；
    `docs/R7_72_RW_B_PILOT.md` 写明 limitations、未做的事与下一动作；R-009 基线
    859/2114 → **885/2180**、`size-thresholds.md` 标记同步（两处引用文档同步更新）。
- **未做**：
  - **机制归因**：四臂不等容量（RW-A/RW-B 各自加参数与算力），因此**不能**把负结果归因到
    「门控+锚定提案 / Z / role 标记」中的任何一项；决策 0023 的「匹配 Generic」前置**仍未满足**。
    拆解对照（三个部件各自单独开关）**未做**——这是下一项的第一个具体动作。
  - 未跑第三 seed；确认轮（≤2.5 GPU-h）**未动用**；E2/E3/E4 未做；M3/M4/M5 未做。
  - 未读 test（全程封存）、未下载数据、未租 GPU、未合并 main、未关 #70–#75。
  - **未重跑 mimosa 深度安全扫描**：commit/push hook 报 `scanner_enobufs`（扫描未完成），
    因此**不得**声称本轮通过了完整安全审计。
- **CI**：本轮涉及四个提交。`d8c828d` 的 run **`36587955914` 失败**（第 6 步证据索引：
  改了已登记的证据页而未同步 digest）；`36292c8` 的 run **`36588423695` 失败**（第 8 步 pytest：
  `size-thresholds.md` 的机器核对标记只在干净检出上对不上——本机那四个文件还是未跟踪、命中被容忍）；
  `b4407c4` **没有独立 run**（与 `de1e87d` 同一次推送，只对 head SHA 触发）；
  `de1e87d` 的 run **`36591213197` = success**（八步全绿）。18 条 workflow 中 17 条实验 workflow
  按 commit-message 标签门控 **skipped**（设计行为，不算失败也不当通过）。
  两次失败**都是真实原因**且都在本地复现后才修，随后用 `git clone`（不是 `git archive`，
  多个等价性测试要靠 git 重放冻结修订）复现干净检出验证。
  收尾提交 `9103876`（本文件的进度块 + 索引 digest + brief）的 run **`36592121065` = success**。
- **D5 授权快照**（决策 0021 的通道，**本轮已获授权并已执行**）：本地单卡顺序；
  臂 = 旧 mean-Ours / RW-A / RW-B / RW-B+roles；seed 41/42；400 updates/臂；
  预算 ≤1.0 GPU-h；协议在第一次 `optimizer.step()` 前以 `'x'` 排他冻结；只读 val、test 封存；
  失败/超时/skip 即停并如实记录。
- **轮末自动校验（2026-09-30 00:02–00:31，本地）**：迭代 1 的完成校验调用在客户端侧**悬挂 1773 秒**
  （`model_usage` 行：`status='cancelled'`、`cancelled_by_user=1`、`input_tokens=0`），由用户暂停中止，
  目标转 `paused`。它**不影响本轮结论**（结论由本块与 `docs/R7_72_RW_B_PILOT.md` 承载），
  但说明这一轮不能靠自动校验收尾；成因、26 次校验调用的路由分布与恢复步骤见
  `docs/R7_ZCODE_GOAL_VERIFIER_ABORTS.md` §2，约定见决策 0024。
- **下一动作**：**给 RW-B 做减法而不是加法**——在同一 400-update 协议族下，把
  `local_solver_state` 拆成三个各自单独可开关的部件（(a) 门控+锚定提案、(b) Z 的递推、
  (c) role 标记），用**预声明的单一对比**回答「负结果来自哪一件」；修正几何（幅度 ×1.8、
  余弦在第 2–3 步转正、第 3 步 60% 恶化）指向 (a) 或 (b)，但本轮数据**不能区分**。
  在做出这个拆解之前：**不得**再给 solver 加新部件，**不得**把这轮 12/24h 的改善当作继续加码的理由，
  也**不得**把角色标记的小幅一致改善写成机制声明。执行前按决策 0021 重新取授权并重读账本
  （本轮结余 **20.701 GPU-h**：24 − 2.463 − 0.8356）。若拆解对照仍为负，按 §5 停止条件 3
  停止发明新模块，转去重新检查 forecast state / training objective / data regime。

每推进一步就更新本块（一行「已完成/未做/下一动作」），不要留给下一轮补写。
