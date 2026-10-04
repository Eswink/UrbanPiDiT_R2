# n4-m5-confirmation：M5（#75）最小确认闭环与有条件自适应

<!-- round-node: N4 -->

**状态：prepared（2026-10-02）。本文件是待执行的目标长文；N2a/N3/N4 由用户 2026-10-02 的
「做完并全部关闭」指示一次性放行。执行者在执行那一刻按决策 0021 记录具名范围，先跑
`tools/check_campaign_state.py` 对表，再按 §4 执行。**

本文件是主计划 `docs/goals/main-model-v2-campaign.md` 的节点 **N4** 的目标长文（节点门禁原文为
「**默认不启动**：adaptive 前置 gate 不成立，且决策 0023 的 matched-Generic 对照仍缺失」；其中
matched-Generic 前置在 N3 已可构造，故本轮按用户指示开跑）。实现顺序见
`docs/plans/0009-r7-v2-completion-and-closeout.md`（A→B→C）。

## §0 Objective（可粘贴；实测 1213 字符）

> 本轮目标：执行 docs/goals/n4-m5-confirmation.md 的 M5 轮——用最小实验闭环给 V2 主模型一个有界、可复核的结论，并决定 adaptive 是否启动。0 GPU-h 部分：运行前冻结评估协议（primary 端点建议 t2m 的 +6/+12h 并写明容许退化，全部 17 变量 × 6/12/24/48/72h 照报、长 lead 失败不隐藏；单位正确的 RMSE / climatology MSE skill / ACC；不同物理单位不直接平均、正 ACC 不等于正 MSE skill；case/seed 按 #60 显式配对、全域/边界分别报告；K1/2/4 同 checkpoint 探针必须标「非独立训练 K1」；参数 count、forward/backward counted ops 定义、总 update/样本/GPU-h、真实 forward latency 与训练 step time 分开记录；旧 GPU/CI counts 不复制成新值）；把这三臂 runner 接好：旧 Ours / matched Generic V2（同结构、同已知输入、无过程语义，决策 0023 的前置对照，N3 已可构造）/ Process V2。GPU 部分：≥3 个预声明 seed 的三臂有界实验（若 N2a/N3 有候选臂则纳入其最佳配置，否则用现有 RW-A/RW-B 已冻结配置），先冻结额外更新数与时限，报全 17 变量 × 6/12/24/48/72h 与三态计数、同 seed 配对；三 seed 只作一致性信息、不称统计显著性。adaptive（E5）默认不启动：除非出现「某些样本值得多计算、且固定 K 形成有用准确率—成本取舍」的有效前沿，才复用现有 gain controller 单独另立预算，并报告相对强 K1 与 Kmax 的实际 latency/吞吐与 K 分布；否则如实写「不启动」并保留该负结果。判据见 docs/goals/n4-m5-confirmation.md §3，冻结来源 docs/plans/0004-r7-main-model-v2.md 评估协议段与 docs/decisions/0023-main-model-first-baseline-freeze.md。禁止：改已冻结判据/阈值/端点/案例集、新增「read_count=1」重新封存同一 M2 test、重跑或覆盖 outputs/ 归档、下载数据、租 GPU、动 main、force push、合并、关闭 #70–#75（关闭是收尾轮 v2-issue-closeout 的事）、自动推进节点。预算 ≤2.5 GPU-h（主计划 §5 第二批之内，开工重读账本）；停止条件为主计划 §5 任一条、需要新判据或新数据、或任一评估失败。不要自行宣布目标完成。

## §1 现状（2026-10-02 只读核对）

- 起点：主计划 §8 `current_node=N1`、`status=paused`；账本已用 4.2045 / 余 19.7955 GPU-h。本轮由用户
  授权一次性进入 N4。
- **五臂对照冻结（决策 0023）**：日常开发只保留旧 Process 模型、新 Process 模型、必要的 Generic 结构
  对照，另加零训练 persistence / train-only climatology；任何「过程结构有用」的声明**必须**带一个
  **同结构、同已知输入、无过程语义**的 Generic 臂（`docs/decisions/0023-main-model-first-baseline-freeze.md:47-50`）。
- **matched-Generic 前置**：`GenericRecursiveWeatherForecaster`（`model/recursive_weather_r7.py:123-203`）
  目前挂不上 `local_solver_state`（classic 无 `solver_init`/`solver_cell`/`solver_gate`/`proposal_head`），
  该前置在 **N3** 接线；本轮依赖它已可构造。
- **adaptive 默认不启动的依据**：#66/S4 已证前置 gate 不成立——最优 K 只在 6h 呈现 seed-一致，且那里
  最优 K 恰是最贵的 K=4、增益 0.00%（`docs/R7_ISSUE_COMMENTS.md` #75 段引）。
- **评估约束（预冻结）**：同 M2 source、train 统计、显式 val init/valid-time、全部 17 通道；M2 test
  **已被看过**，仅作开发材料，**不能新增 `read_count=1` 重新封存同一数据**。
- **现有 runner/比较器可复用**：#60 paired comparison、四张成本表、`training/r7_scheduled_runner.py`
  的 schedule/early-stopping/shared_initial_state；adaptation 侧 `model/r7_halting.py` 的
  `AdaptiveProcessForecaster` / `ForecastGainController` 代码完整但无标定控制器。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | E4 三臂 runner 接线（0 GPU-h） | 旧 Ours / matched Generic V2 / Process V2 的薄 ablation plan（复用现有 runner/checkpoint/evaluation，**不另造平台**）；primary 端点与配对规则运行前冻结 |
| D2 | 评估协议冻结（0 GPU-h） | `protocol.json`（含 primary、17 变量 × 5 时效、单位、配对、成本表口径），digest 在训练前冻结 |
| D3 | ≥3 seed 三臂确认实验（GPU ≤2.5 GPU-h） | `outputs/r7_75_confirmation/`：逐 seed 结果、合并结果、各变量/时效三态计数、同 seed 配对、四张成本表；`scientific_claim:false` + `limitations` |
| D4 | adaptive 决定（0 GPU-h，除非有效前沿出现） | 默认写「不启动」并保留 #66/S4 负结果；若出现有效前沿，另立预算并复用 gain controller，报告相对强 K1/Kmax 的 latency/吞吐与 K 分布 |
| D5 | 证据页与登记 | `docs/R7_75_CONFIRMATION.md` + E 条目 + `docs/R7_EVIDENCE_INDEX.jsonl` 记录 + 账本行 + 主计划 §7/§8 回写 + CI 绑定 |

## §3 判据与预声明读法

判据全部引自冻结文档，**不新增阈值、不放宽任何既有判据**：

- **评估协议预冻结**（`docs/plans/0004-r7-main-model-v2.md:257-266`）：primary 端点运行前写明（建议 t2m
  6/12h 并标注容许退化）；全 17 变量 × 6/12/24/48/72h 公布、长 lead 失败不隐藏；单位正确 RMSE /
  climatology MSE skill / ACC；不同物理单位不直接平均；正 ACC ≠ 正 MSE skill；#60 比较器显式配对；
  ≥3 预声明 seed 只给一致性、无显著阈值；K=1 探针须标「非独立训练 K=1」；四张成本表；`scientific_claim:false`
  + limitations 在四处出现。
- **三级判定**（同文件 `:268-281`）：ENGINEERING PASS / EXPERIMENTAL SUPPORT / SCIENTIFIC SUPPORT
  分开写；**基线冻结不等于宽松**——「过程结构有用」必须带 matched Generic（决策 0023）。
- **完成判据**（issue #75 正文，匿名 API 已核）：主模型 V2 实现 + 有界实验 + 最少必要控制 + 明确
  positive/mixed/negative 结论；**科学增益、工程完成和某个试验否定分别标记**，不能通过「所有 issue
  已关闭」冒充模型研究已完成。
- **adaptive 门槛**（同 issue）：只有「某些样本值得多计算、固定 K 形成有用准确率—成本取舍」才启动；
  先 train-only 拟合、val-only 阈值、固定策略；报告相对强 K1 与 Kmax 两端；控制器若只能回退 full-depth，
  **保留负结果并停止加大复杂度**；不阻塞已成立的固定过程主模型贡献。
- **不要过度外推**：局部两个月开发集 ≠ 一区/二区 SOTA；最终独立年份/季节的正式测试是**后置**事项，
  需另立预算（本计划不含）；不据已曝光 M2 test 选 λ 或扩更新。
- **三态读法**（预声明）：三臂若在 primary 端点上无显著分离 → 如实写 cannot-distinguish，不靠加
  seed/加预算「练到赢」；若 process 臂胜 matched-Generic 且 seed-一致 → 记为 EXPERIMENTAL SUPPORT，
  仍不称 SOTA 或 SCIENTIFIC SUPPORT。

## §4 实施顺序（不跳步）

1. **对表**：跑 `tools/check_campaign_state.py`（退出 0）、重读账本、核 matched-Generic 前置（N3 产出）
   与比较器/四表指针未变。
2. **D1/D2（0 GPU-h）**：三臂 runner 接线 + 评估协议冻结（digest 训练前锁定）；工程 CI 绿。
3. **D3（GPU）**：执行那一刻按决策 0021 记录具名范围（范围 = 3 臂 × ≥3 seed × 预声明更新数；预算
   ≤2.5 GPU-h；共驻余量门槛；失败即停全额计费不重试）；只读 val；两卡可分开跑 seed（记录环境/随机流/
   样本顺序）。
4. **D4（0 GPU-h，除非有效前沿）**：按 §3 写 adaptive 决定；默认「不启动」并保留负结果。
5. **D5 登记**：证据页、E 条目、索引记录、账本行、主计划 §7/§8 回写、CI 绑定；**只提议收尾轮**。

## §5 预算与停止条件

- 预算 **≤2.5 GPU-h**（主计划 §5 第二批之内；开工重读账本，当时余 19.7955）；单次实验 ≤30 min（共驻
  余量门槛）。
- 配额记录：本轮 + 前序 A/B 与应急的合计目标 ≤8.0 GPU-h（见 plan 0009）；若将超出自设额度，**先通知用户**。
- **停止条件**（满足任一即停并报告）：预算用尽或账本不足；需要新判据/新数据/租 GPU/合并 main；结论为
  「不可分辨」；任一评估失败（即停、全额计费、不重试）。
- 目标状态 `active / paused / budget_limited / complete`；**执行者只可提议，不得自宣完成**。

## §6 明确不做

- 不改已冻结判据、阈值、端点或案例集；**不新增 `read_count=1` 重新封存同一 M2 test**。
- 不重跑、不覆盖、不改写任何归档产物与历史证据页（`outputs/` 一律只读）；旧 GPU/CI counts 不复制成新值。
- 不读封存 test 作评估/选择；不下载数据；不租 GPU；不动 main；不 force push；不合并；**本轮不关闭 issue**。
- 不新增 baseline 家族、不重跑外部 baseline 全矩阵；不把三 seed 写成显著性；不建立定时任务或后台续跑。

## §7 进度块

- **状态**：`prepared`（2026-10-02）：本文件为执行前长文；未实现、未跑实验、未登记、未开 GPU。
- **前置**：用户 2026-10-02 授权一次性放行 N2a/N3/N4；N4「默认不启动」的门禁中，adaptive 部分仍默认
  不启动，仅 E4 三臂确认按用户指示执行；matched-Generic 前置在 N3 构造（记录见 plan 0009）。
- **下一动作**：进入执行窗口，按 §4 顺序接线并冻结协议，再按 §5 取具名授权跑 D3。

## §8 下一动作（仅提议）

M5 完成后**只提议**进入节点 **N5**（收尾轮 `docs/goals/v2-issue-closeout.md`：补 #71/#72 证据 +
六 issue 判定登记 + 关闭）；不自行宣告主模型达到 SOTA 或 SCIENTIFIC SUPPORT，也不自动进入最终独立年份/
季节的正式测试（那是后置事项，需另立预算）。

- **后续适用指针（2026-10-04）**：本页中「不关闭 #70–#75／本轮不关闭 issue／关闭是收尾轮的事」为当轮禁止项，如实保留不改写；自决策 0037（`docs/decisions/0037-issue-closeout-window-scope.md`）起，#70–#75 的有条件关闭按 `docs/goals/v2-issue-closeout.md` 执行（单一 `Closes` 提交、精确 CI、非 force ff 推 main、匿名复核；关闭≠科学成功）。
