# n3-m4-autoregressive-rollout：M4（#74）真实 2 步可微自回归与 matched-Generic V2 接线

<!-- round-node: N3 -->

**状态：prepared（2026-10-02）。本文件是待执行的目标长文；N2a/N3/N4 由用户 2026-10-02 的
「做完并全部关闭」指示一次性放行。执行者在执行那一刻按决策 0021 记录具名范围，先跑
`tools/check_campaign_state.py` 对表，再按 §4 执行。**

本文件是主计划 `docs/goals/main-model-v2-campaign.md` 的节点 **N3** 的目标长文（节点门禁原文为
「只在 N2 出口放行后开跑」，该门禁由用户 2026-10-02 指示解除）。实现顺序见
`docs/plans/0009-r7-v2-completion-and-closeout.md`（A→B→C）。

## §0 Objective（可粘贴；实测 1448 字符）

> 本轮目标：执行 docs/goals/n3-m4-autoregressive-rollout.md 的 M4 轮——训练真正的草稿修正与自回归误差恢复，并顺带把 #75 需要的 matched-Generic V2 接线做出来。0 GPU-h 部分：先补齐 matched-Generic V2 支持（给 GenericRecursiveWeatherForecaster 开放与 Process V2 同款的 local_solver_state 结构开关，复用既有 local_solver_state_r7 组件、不新增 solver 部件；写「同结构、同已知输入、无过程语义」的参数差契约测试，这是决策 0023 对任何「过程结构有用」声明的前置对照，也是下一轮 #75 的前置）；再实现可微的 2 步物理 rollout：从同一已学的 +6h 父 checkpoint（默认 outputs/r7_72_rw_b_subtraction/seed*/training/process_spacetime_rwa/update_0000400.pt）warm-start，把模型自己的 Y_(t+6) 而非 GT 写回 history 求 Y_(t+12)，loss = L6 + λ·L12（λ=0.5 在执行前冻结、写进 protocol），每步只读当时 history 与确定性时空 feature；必须使用真正的可微展开而非 model/r7_rollout.py:72 那个 @torch.no_grad() 的评估 rollout；字段区分 reasoning_steps（内部 K，不推进时间）与 rollout_steps（物理 +6h，每次 +6h，不用 H=12 替换单步 embedding）。定向测试：2 步 forward 计数、第二步输入 hash 等于第一步预测、改真实第二步 target 只改 loss 不改 forward、每物理步正确更新时间、内部 K 不改时间、缺精确 t+12 窗口明确拒绝、梯度从第二步回到第一步、同配置 resume/finite/BF16、streamed 截断与物理步截断分轴报告。GPU 部分：两臂有界 warm-start 实验（同父 checkpoint、同额外更新预算下，继续 L6 训练 vs 真实 L6+λ·L12 两步 rollout），2 个预固定 seed，先冻结额外更新数与时限，如实记录 rollout 多出的一次前向/反向、算力不相等；第一轮只做 2 步、不扩到 4/8/12 步、不做 72h unroll。判据见 docs/goals/n3-m4-autoregressive-rollout.md §3，冻结来源 docs/R7_MAIN_MODEL_V2_DESIGN.md §4 与 docs/decisions/0023-main-model-first-baseline-freeze.md。禁止：改已冻结判据/阈值/端点/案例集、重跑或覆盖 outputs/ 归档、重跑 #64 curriculum、读封存 test、下载数据、租 GPU、动 main、force push、合并、关闭 #70–#75、自动推进节点。预算 ≤1.0 GPU-h（主计划 §5 第二批之内，开工重读账本）；停止条件为主计划 §5 任一条、需要新判据或新数据、或任一评估失败。不要自行宣布目标完成。

## §1 现状（2026-10-02 只读核对）

- 起点：主计划 §8 `current_node=N1`、`status=paused`；账本已用 4.2045 / 余 19.7955 GPU-h。本轮由用户
  授权一次性进入 N3（不再经 N2 出口放行）。
- **两个时间轴（冻结语义）**：`k` = 内部过程推理轮次（`reasoning_steps`，同一 valid time 重估，**不推进
  物理时间**）；`h` = 物理预报转移（每个转移固定 +6h，`model/r7_rollout.py` 的 rollout 轴）。字段名必须
  区分二者，**不得混用**（`docs/R7_MAIN_MODEL_V2_DESIGN.md:83-100`）。
- **现存 rollout 是评估专用**：`model/r7_rollout.py:72` 的 `autoregressive_rollout` 带 `@torch.no_grad()`
  且有 `model.eval()` 守卫（`:95-96`），**正是设计 §4 第 5 条禁止拿来当训练的东西**。
- **训练是单物理步**：`training/r7_local_runner.py:22-42`（loss 在 `:34`，用 `batch['atmos_target']`）；
  调度器 `training/r7_scheduled_runner.py:248-447`，其验证 rollout `score_validation` 也是
  `@torch.no_grad()`（`:77`）；M2 train manifest 每行 `lead_time_hours: 6`（标量）。
- **父 checkpoint 已存在**：`outputs/r7_72_rw_b_subtraction/seed{41,42}/training/process_spacetime_rwa/`
  下 `update_0000400.pt`（+6h、RW-B，约 35 MB）。本轮从它 warm-start，**不必先重训父模型**。
- **matched-Generic 前置缺失**：`GenericRecursiveWeatherForecaster`（`model/recursive_weather_r7.py:123-203`）
  支持 `spatial_solver_feedback` / `spacetime_inputs` / `spacetime_field_mode` / `source_role_markers`，
  但**挂不上** `local_solver_state`（无 `solver_init`/`solver_cell`/`solver_gate`/`proposal_head`），
  因此「matched Generic V2」当前**不可构造**（plan 0004:198 列为待办）。
- **不是重跑 #64 curriculum**：旧实验是 400 次直接 +6h、200 次直接 +12h、200 次直接 +24h 的**目标重分配**
  （`scripts/study_r7_64_curriculum.py`、`docs/R7_64_CURRICULUM.md`）；本轮测的是另一个机制——把模型
  自己的预测状态喂回去做可微 rollout 训练。旧负结果保留，但不据此宣布所有多步训练无效。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | matched-Generic V2 接线（0 GPU-h） | `model/recursive_weather_r7.py` 开放 `local_solver_state`（复用 `model/local_solver_state_r7.py` 组件）；`GenericRecursiveWeatherForecaster` 与 `ProcessForecastCoReasoner` 在「同结构、同已知输入、无过程语义」下的参数差契约测试；`[model-digest-change]` 显式标记 |
| D2 | 可微 2 步物理 rollout（0 GPU-h） | 新的可微展开实现（**独立于** `model/r7_rollout.py:72` 的评估 rollout）；warm-start 接线；`L6 + λ·L12`；`reasoning_steps` / `rollout_steps` 分离；不截断即真展开，若为显存截断则**明确声明版本**并报告优化目标/梯度差异 |
| D3 | 定向测试（0 GPU-h） | `tests/test_r7_m4_autoregressive_rollout.py`（§3 清单，含反证） |
| D4 | 两臂有界实验（GPU ≤1.0 GPU-h） | `outputs/r7_74_autoregressive/`：protocol（先冻结 digest）、逐 seed 结果、合并结果、paired comparison、四张表；**算力不相等如实记录**；`scientific_claim:false` + `limitations` |
| D5 | 证据页与登记 | `docs/R7_74_AUTOREGRESSIVE.md` + E 条目 + `docs/R7_EVIDENCE_INDEX.jsonl` 记录 + 账本行 + 主计划 §7/§8 回写 + CI 绑定 |

## §3 判据与预声明读法

判据全部引自冻结文档，**不新增阈值、不放宽任何既有判据**：

- **两轴分离**（`docs/R7_MAIN_MODEL_V2_DESIGN.md:83-100`）：内部 K 不改时间；每个物理转移固定 +6h，
  **不用 H=12 替换单步 `lead_time_hours` embedding**；`reasoning_steps` 与 `rollout_steps` 不混用。
- **可微展开**（同文件 `:99-100`）：**训练不得使用 `no_grad` 的评估 rollout**；内部 K 的 streamed 截断
  与物理步截断是**两轴**，分别报告。
- **验收**（issue #74 正文，匿名 API 已核）：2 步 forward 数、第二步输入 hash 与第一步 prediction 一致；
  改真实第二步 target **只改 loss 不改 forward**；每个物理步正确更新时间、内部 K 不改时间；缺精确 t+12
  窗口**明确拒绝**、不跨 split/缺测拼接；针对小解析模型验证梯度/可微/截断的预期区别；同配置 resume、
  finite、BF16/显存通过；两臂 × 两固定 seed 的有界 warm-start 实验，先冻结额外更新数与时限，候选支持后
  补第三 seed；报全 17 变量 / +6 到 72h 及原 #72 checkpoint；**+6 显著恶化而只 48h 改善不称总体胜出**。
- **档位**：`L6 + λ·L12`，λ=0.5 **执行前冻结**并写进 protocol；**始终保留 L6 锚**，每个训练 window 仍给
  +6h 监督，不以纯长 lead 任务替换；第一轮**只做 2 步**。
- **算力不相等**：rollout 多了一次前向/反向；**明确记录**并补可负担的 equal-compute 对照或在结论里限定，
  **不称同 updates = 同算力**（plan 0004:221）。
- **matched-Generic（决策 0023）**：任何「过程结构有用」的声明都必须带**同结构、同已知输入、无过程语义**
  的 Generic 臂（`docs/decisions/0023-main-model-first-baseline-freeze.md:47-50`）；本轮把它做出来并验证
  可构造，供下一轮 #75 使用。
- **三态读法**（预声明）：若 2 步 rollout 无验证收益 → 如实写 negative，不扩到 4/8/12 步（plan 0004:327）；
  不根据已曝光 M2 test 选 λ 或扩更新。

## §4 实施顺序（不跳步）

1. **对表**：跑 `tools/check_campaign_state.py`（退出 0）、重读账本、核父 checkpoint 与旧 rollout 指针未变。
2. **D1（0 GPU-h）**：matched-Generic V2 接线 + 参数差契约测试 + `[model-digest-change]` 标记。
3. **D2/D3（0 GPU-h）**：可微 2 步 rollout + 定向测试全绿（含反证）；工程 CI 绿。
4. **D4（GPU）**：执行那一刻按决策 0021 记录具名范围（范围 = 2 臂 × 2 seed × 预声明额外更新数；
   父 checkpoint = 上一步选定的 +6h process_spacetime_rwa/update_0000400.pt；预算 ≤1.0 GPU-h；共驻余量
   门槛；失败即停全额计费不重试）；先冻结 `protocol.json` digest 再训练；只读 val。
5. **D5 登记**：证据页、E 条目、索引记录、账本行、主计划 §7/§8 回写、CI 绑定；**只提议下一节点**。

## §5 预算与停止条件

- 预算 **≤1.0 GPU-h**（主计划 §5 第二批之内；开工重读账本，当时余 19.7955）；单次实验 ≤30 min（共驻
  余量门槛）。
- 配额记录：本轮 + 后续 C 与应急的合计目标 ≤8.0 GPU-h（见 plan 0009）；若将超出自设额度，**先通知用户**。
- **停止条件**（满足任一即停并报告）：预算用尽或账本不足；需要新判据/新数据/租 GPU/合并 main；结论为
  「不可分辨」；任一评估失败（即停、全额计费、不重试）。
- 目标状态 `active / paused / budget_limited / complete`；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不改已冻结判据、阈值、端点或案例集；不重跑 #64 curriculum；不用 `no_grad` 评估 rollout 当训练。
- 不重跑、不覆盖、不改写任何归档产物与历史证据页（`outputs/` 一律只读）。
- 不读封存 test；不下载数据；不租 GPU；不动 main；不 force push；不合并；**本轮不关闭 issue**。
- 不新增 solver 部件（matched-Generic 只复用既有组件并开同款开关）；不扩到 4/8/12 步；不做 72h 训练
  unroll；不建立定时任务或后台续跑。

## §7 进度块

- **状态**：`prepared`（2026-10-02）：本文件为执行前长文；未实现、未跑实验、未登记、未开 GPU。
- **前置**：用户 2026-10-02 授权一次性放行 N2a/N3/N4；N3 的「只在 N2 出口放行后开跑」门禁由该指示解除
  （记录见 `docs/plans/0009-r7-v2-completion-and-closeout.md`）。父 checkpoint 已存在，无需先跑 M2。
- **下一动作**：进入执行窗口，按 §4 顺序实现 D1–D3，再按 §5 取具名授权跑 D4。

## §8 下一动作（仅提议）

M4 完成后**只提议**进入 N4（M5 最小确认闭环，`docs/goals/n4-m5-confirmation.md`）；matched-Generic V2
在本轮已可构造，是 N4 的前置之一。不自行宣告 M4 成功或自动跳步。