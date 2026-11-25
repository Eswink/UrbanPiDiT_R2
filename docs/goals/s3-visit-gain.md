<!-- round-node: S3 -->
<!-- round-spec: {"arm": "s3-visit-gain", "code_archive": "outputs/r7_s3_visit_gain_code_0120.zip", "commit_message": "feat(r7): implement the visit-gain damping arm pair (A visit_damped vs B uniform_damped) [model-digest-change]", "commit_paths": ["model/recursive_weather_r7.py", "model/process_forecast_r7.py", "model/process_step_r7.py", "scripts/r7_s3_visit_gain_protocol.py", "scripts/r7_s3_visit_gain_parent.py", "scripts/r7_s3_visit_gain_verdict.py", "scripts/study_r7_s3_visit_gain.py", "tests/test_r7_s3_visit_gain.py", "tests/test_r7_s3_visit_gain_round.py", "docs/goals/s3-visit-gain.md", "docs/goals/loop-depth-headroom-before-architecture.md", "docs/decisions/0119-visit-gain-execution-design.md", "docs/decisions/README.md", "tests/conftest.py", "tests/test_r7_s3_typed_budget.py", "tests/test_r7_s3_typed_budget_host.py", "tests/test_reconcile_r7_static_parent.py", "docs/goals/main-model-climatology-campaign.md", "scripts/r7_s3_visit_gain_constants.py", "scripts/r7_s3_visit_gain_identity.py", "scripts/r7_s3_visit_gain_rate.py", "tests/test_r7_s3_visit_gain_rate.py", "docs/decisions/0120-probe-rate-estimator-and-quiet-machine.md"], "control_record": "s3-loop-depth-headroom-g0", "device_mode": "dual", "driver": "scripts/study_r7_s3_visit_gain.py", "hard_cap_seconds": 216000.0, "output_stem": "r7_s3_visit_gain_20261126", "planned_seconds": 118800.0, "reference_record": "s3-typed-evidence-in-protocol", "register_message": "docs(r7): register the G1-B visit-gain damping readings", "register_paths": ["docs/R7_S3_VISIT_GAIN_RESULT.md", "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md", "docs/goals/main-model-climatology-campaign.md", "docs/goals/s3-visit-gain.md", "docs/goals/loop-depth-headroom-before-architecture.md", "docs/R7_S3_VISIT_GAIN_PROBE_REFUSED.md", "docs/decisions/0119-visit-gain-execution-design.md"], "seeds": [41, 42, 43], "test_path": ["tests/test_r7_s3_visit_gain.py", "tests/test_r7_s3_visit_gain_rate.py"]} -->

# S3 / G1-B — visit-gain 阻尼的两臂对照（A `visit_damped` vs B `uniform_damped`）

本文件是 G1-B 的**唯一执行源**；机器可读的 `round-spec` 在上方。它从
`docs/goals/loop-depth-headroom-before-architecture.md` 派生：那份长文不带 `round-spec`，
所以不能做 `campaign-state` 指针；本文件带，因此可以。

## Objective（objective）

> 以 goal 模式执行 `docs/goals/s3-visit-gain.md` 的 G1-B 分支（上游长文 `docs/goals/loop-depth-headroom-before-architecture.md`）：在 r7/weather-reasoning 分支上，给 `model/recursive_weather_r7.py` 的 `GenericRecursiveCell` 加 `visit_gain` 开关（默认 `None` = 现实现逐操作相同、不增参数、键集不变），从已注册的 p32,132 incumbent 父（三 seed）各训两臂 2,400 updates：A `visit_damped`（`g_k=1/sqrt(1+k)`，k 从 0 起）与 B `uniform_damped`（`g=1/sqrt(1+(K−1)/2)`，K=4）；C 用已注册 incumbent 不重训。主对比 A−B 隔离 visit 对齐与整体缩小步长。判据冻结在 `docs/decisions/0119-visit-gain-execution-design.md`：①72 h `full(65×65)` t2m 的 A−B 三 seed 同号为负；②6 h 与 12 h 的 A−B seed 均值 ≤0；③身份见证（`visit_gain=None` 逐位惰性 + 键集/张量与归档父完全相等；A 在 k=0 与 `None` 逐位相等）；④每 seed 参数键集与父 `state_digest` 逐位相同。任一失败即关闭该支，不加 seed、不扫增益函数族、不调 K、不改判据。先写 ADR 与 brief、冻 `protocol.json`、父迁移并逐 seed 证明 `state_digest` 与归档父逐位相同，再训练。**禁止**：替换架构、动 `MIXING_MODES` 或感受野（0048 容量轴与 0077 H-C 保持关闭）、读 2023 test、`r=0`、自宣技能或目标完成。成本估 11.1 GPU-h **未实测**；0 GPU-h 轮不得加账本行。跑完即停，不做无人值守续跑。

（字符：本 objective 单段，实测 925 code points，≤4000。）

## 上下文与开轮依据（执行前必读）

1. `docs/decisions/0119-visit-gain-execution-design.md` — **本轮的判据来源**。它与决策 0118
   的三处关系必须读懂，否则会误读读数：
   - **B 不是 A 的等平均阻尼臂**：K=4 时 A 的增益均值 0.696114、B 的常数 0.632456，
     B **多阻尼 9.14%**（Jensen）。故 A−B **不可**归因为纯 visit 对齐。
   - **`full(65×65) ≡ interior_32(129×129)`**：65×65 窄实例的全网格就是宽实例的内域，
     分母在 ~1e-10 内一致（`docs/R7_S3_WIDE_REGION_FULL.md:178`）。所以 0118 出口①的
     「`interior_32`」在本实例上**就是** `full`，不是放宽。
   - **出口③换成了同轮身份见证**：0118 字面的「首 update loss 与归档父逐位相同」在 B 上
     构造不可能（g=0.632 在 k=0 就生效），且没有「从 p32,132 起跑」的归档值。
2. `docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md` — G0 读数（`admit-g1b`）。
3. `docs/plans/0036-loop-depth-headroom-before-architecture.md` — 计划（不可变）。

## 判据（训练前冻结进 protocol.json，此后不得改动）

判据与理由见决策 **`docs/decisions/0119-visit-gain-execution-design.md`**（Decision 5）。
本节的规则与那份文档冲突时，以此文件为准；此文件一旦冻结，其取值不得在看过读数后改动。
G0 的准入判据在 `docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md`，
读数在 `docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md`。

### 出口（全部 freeze 进 `protocol.json`）

| # | 判据 | 判定方式 |
| --- | --- | --- |
| **①** | 72 h `full(65×65)` t2m 的 **A−B** 三 seed **同号为负** | 逐 seed 比 RMSE 差；三 seed 符号一致才为 `true`；任一反号即 `false` |
| **②** | 6 h 与 12 h 的 **A−B** 的 **seed 均值 ≤ 0**（两个 lead 各自都要满足） | 先算每 lead 的三 seed 均值，再判 `≤0` |
| **③** | 身份见证 **③a** + **③b** | ③a：`visit_gain=None` 的 cell 输出与未改动表达式逐位相等，且 155 张量与归档父逐张量相等；③b：A 在 `visit_index=0` 与 `None` 逐位相等。**固定 `torch.set_num_threads(1)`** 后比较 |
| **④** | 每 seed 参数键集与父 `state_digest` 逐位相同 | 迁移收据 `state_equality_evidence()` 逐 seed 比对；off 臂 checkpoint 的 `state_digest` 必须与归档父相等 |

### 符号约定与读法纪律

- `A−B < 0` 表示 **A 的 RMSE 更低**（A 更好）。**未测一律 `null`**，不得记 `false`
  也不得记 `true`。
- **B−C 只作描述性**，且必须同时写「B 与 C 参数差 = 0」。
- **A−B 只能读作「在 B 的阻尼水平上，visit 依赖是否更好」**，不得读作「visit 对齐优于常数」。
- 三 seed 是**一致性证据**，不是显著性检验；不构造区间、不做多重比较校正（`r=0`，开发筛选）。

### 探针与包络（沿用 0116 的无截断形态）

- 每 seed 先跑 A/B 各 **400 updates** 探针，用实测速率外推；若剩余包络 60% 装不下，
  立刻中止并记 `refused-insufficient-envelope`。**不允许**烧数小时再被截断。
- 包络按**实测 treated 速率**重定（本轮 treated 是阻尼臂，可能比 incumbent 慢）。
- 任一非正常结束写 `seed{N}_{arm}_partial_receipt.json`，含 `updates_reached` /
  `seconds_per_update` / `gpu_hours_in_failed_attempt` / `progress_file`。
- 真实截断点是 `tools/r7_round_stages.py:292` 的 `min(deadline, now() + per_seed)`。

## 交付物

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | `visit_gain` 开关（默认 `None` 逐操作相同、零新参数、键集不变） | `tests/test_r7_s3_visit_gain.py` 19 条反证（off 逐位等价、两臂共享初始化、键集/参数量相同） |
| D2 | 父迁移 + 每 seed `state_digest` 与归档父逐位比对 | `scripts/r7_s3_visit_gain_parent.py` 的 `migration_receipt.json` + `state_equality_evidence()` |
| D3 | 两臂 3 seed 训练 + 5 lead 评分 | `seed{N}_{arm}_receipt.json` + 逐 lead CSV |
| D4 | 出口 ①–④ 逐条判定 | `readings.json`（独占写），每条 `true/false/null` |
| D5 | 证据页 + 一条 index record | `docs/R7_S3_VISIT_GAIN_RESULT.md` + `docs/R7_EVIDENCE_INDEX.jsonl` |
| D6 | 四件套登记 | 账本行、index、§8 进度块、`campaign-state` 前移；`tools/r7_round.py verify` 全绿 |

## 预算

- 软预算 **118,800 s**（33 h 墙钟）；超了继续并记 `overrun`。
- 硬包络 **216,000 s**；只有它截断，且**截断不算通过**。
- 成本**估算 11.1 GPU-h 未实测**（2 臂 × 3 seed × 2,400 updates × ~1.45 s/upd + 3,200 s/seed 评分，
  双卡三波）。执行时按探针实测重定 `PER_SEED_SECONDS` / `PLANNED_SECONDS` / `HARD_CAP_SECONDS`。
- **无总 GPU-h 上限**，账本只记账。**0 GPU-h 轮不新增账本行**。

## 停止条件

- 任一出口 ①–④ 为 `false` → **关闭该支**，写证据页并停；不加 seed、不扫增益族、不调 K、不改判据。
- 探针判 `refused-insufficient-envelope` → 立即停，写 partial receipt，不改包络重试。
- 任一出口为 `null`（臂缺失/非有限）→ `readings.json` 如实记未测，不得写 `true`。
- 三 seed × 两臂全部完成且出口判定写出 → 立即停，不做续跑。
- **禁止** cron / 守护 / 会话外续跑。
- 触保留授权边界（付费/租卡/独占/方向改变/main 合并/破坏性）→ 停，并在 §8 写恢复第一条命令。

## 进度

- **状态**：**`attempt02` 已按决策 0120 重定探针口径，待重冻后训练**。attempt01 判
  `probe-refused`（0 臂训练、0 评分，出口 ①② 为 `null`、③④ 为 `true`），其读数已登记为
  `record:s3-visit-gain-probe-refused`，**不在其上续跑**。
- **已完成**：ADR 0119（执行设计 + 三处开轮前更正）、ADR **0120**（取代探针口径）、本 brief、
  `visit_gain` 实现 + 53 条反证、父迁移（三 seed `state_digest` 与归档父逐位相同）、
  attempt01 的 `prepare`/`archive` 与身份见证。
- **探针口径已更正（实测）**：独占整机 400 步的四个区间边际速率是
  **1.075 / 1.098 / 1.110 / 1.109 s/upd**，从第一区间起就平；两卡各一 worker 并发
  **1.0796 / 1.0921**。attempt01 实测的 2.48–2.65 是**同期 6,040 个测试**占满 CPU 导致
  loader（`num_workers=4`）供不上数。⇒ 是**测量环境**缺陷，不是估计量或实现缺陷。
  （0119 执行记录里「启动摊销 / 边际 1.58」那条诊断**是错的**，0120 已更正。）
- **0120 后的口径**：`probe()` 用**后段 1/4 边际速率**（与总速率并记，差 >25% 以边际为准）；
  探针期间禁止并发 CPU 负载并记 `loadavg`；包络按 **1.30 s/upd** 重定为
  `PER_SEED_SECONDS 25,280`（2.0×）；`REFUSAL_FRACTION` **保持 0.60**。
- **成本**：attempt01 **1.1330 GPU-h**（全部探针）。账本行已加。
- **未做**：两臂训练、五 lead 评分、出口 ①② 的判定。
- **下一动作**：`commit-guard` → 干净树重建 code zip → `freeze`（`attempt02`）→
  `stage prepare/archive` → **确认机器安静** → `stage-seeds` → `stage reading`。
  恢复第一条命令：`.venv/bin/python tools/r7_round.py state`。

## 纪律

- 不改 `data/**`；不读 2023 test；`r = 0`；`scientific_claim: false`。
- 每轮只跑一个臂；GPU 由统一父调度按共驻余量串行；spawn 前只读核对；**禁止**对非本实验进程发送信号。
- 提交前自跑 `tools/r7_round.py commit-guard`；登记后自跑 `tools/r7_round.py verify`；只推工作分支。
- 四件套 + 账本行 GPU-h 与 record 的 `metrics.gpu_hours` 对齐到 1e-4。
- `report()` 与 CI 绿都不是科学结论；不自宣技能或目标完成。
