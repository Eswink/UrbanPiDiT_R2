# main-model-v2-rw-b-subtraction：RW-B 的减法归因（(a) 门控+锚定提案 / (b) Z 的递推 / (c) role 标记）

**THIS FILE IS A PROMPT TEMPLATE. GOAL MODE HAS NOT BEEN STARTED.**

本文件是**下一轮 goal 模式**的长文目标源，由上一轮长文 `docs/goals/main-model-v2-rw-b-round.md`
的 §8「下一动作」直接派生：那里写定了「给 RW-B 做减法而不是加法」，并禁止在完成这个拆解之前
再给 solver 加任何新部件。启动后第一件事是按 §4 做 fresh-read，**不得直接采信本文件里的数字**。

配套文档：上一轮长文 `docs/goals/main-model-v2-rw-b-round.md`（§5 停止条件 3、§8 下一动作）；
上一轮证据 `docs/R7_72_RW_B_PILOT.md`（E-198/E-199/E-200）；设计契约
`docs/R7_MAIN_MODEL_V2_DESIGN.md` §3.2；决策 0021（实验授权通道）、0022（规模与命名硬上限）、
0023（基线冻结）、0024（goal 自动校验按 best-effort）。

---

## §0 Objective（可粘贴；实测 1134 字符）

> 本轮目标：在不再新增 solver 部件的前提下，把上一轮落地并判为负的 RW-B（local_solver_state）拆成三个可单独开关的部件——(a) 逐位置门控 + 锚定 X_t 的提案、(b) 求解状态 Z 的跨步递推、(c) 来源 role 标记——并用一条预注册的单一对比回答「上一轮 48/72h 的恶化主要由哪一件带来」。判据与边界见 docs/goals/main-model-v2-rw-b-subtraction.md。交付物：D1 0 GPU-h 机制探针（只读上一轮已归档的 32 个 checkpoint、只用 val：逐部件关掉后测第 1 步修正幅度、逐步误差—修正余弦的符号、第 3 步恶化比例，给出哪一件复现 ×1.8 幅度与余弦转正）；D2 真实模型代码——从 local_solver_state 拆出两个子开关（门控+锚定提案 / Z 的递推；(c) 已是独立开关 source_role_markers），全关时与前实现逐位等价、全开时与上一轮 local_solver_state=True 逐位等价，两条都要测试证据；D3 定向测试（开关组合矩阵、门控梯度非零、forward/streamed/adaptive 三路等价、resume 与 BF16）；D4 一轮有界留一对照（协议在第一次优化器更新前冻结：臂 = RW-A / RW-B / RW-B−(a) / RW-B−(b)，2 seed × 400 updates、只读 val、test 封存、≤1.0 GPU-h；逐变量 ×6/12/24/48/72h 全表 + 三态计数 + 参数量/FLOPs/显存/墙钟四表）；D5 预声明单一对比的读数与归因（对每个留一臂按既有比较器算「臂 − RW-A」在 t2m 五时效上的逐 seed 同号三态：(a) 臂是负控制、(b) 臂是主问句，分支规则见长文 §3）；D6 证据文档与未做/limitations 明写、negative 原样保留、CI run id 与 SHA 绑定。禁止：新增 solver 部件、改已冻结判据或阈值、新增端点、重跑或改写上一轮产物、读 test、下载数据、租 GPU、改 main、force push、关闭 #70–#75、自动进入下一轮。预算 ≤1.0 GPU-h（第二批次账本结余 20.701 之内），确认轮另立 ≤2.5 GPU-h；停止条件为预算用尽、D1 判定必须先停、或留一对照仍无法归因（此时按停止条件 3 停止发明新模块，转去重新检查 forecast state / training objective / data regime）。不要自行宣布目标完成。

启动方式（复制下面这一块，整段粘进 `/goal` 后面；**必须带交付物清单**——verifier 不能调工具、
读不到本文件，判据只有在对话里能被核对才算存在）：

```
本轮目标：在不再新增 solver 部件的前提下，把上一轮落地并判为负的 RW-B（local_solver_state）拆成三个可单独开关的部件——(a) 逐位置门控 + 锚定 X_t 的提案、(b) 求解状态 Z 的跨步递推、(c) 来源 role 标记——并用一条预注册的单一对比回答「上一轮 48/72h 的恶化主要由哪一件带来」。判据与边界见 docs/goals/main-model-v2-rw-b-subtraction.md。交付物：D1 0 GPU-h 机制探针（只读上一轮已归档的 32 个 checkpoint、只用 val：逐部件关掉后测第 1 步修正幅度、逐步误差—修正余弦的符号、第 3 步恶化比例，给出哪一件复现 ×1.8 幅度与余弦转正）；D2 真实模型代码——从 local_solver_state 拆出两个子开关（门控+锚定提案 / Z 的递推；(c) 已是独立开关 source_role_markers），全关时与前实现逐位等价、全开时与上一轮 local_solver_state=True 逐位等价，两条都要测试证据；D3 定向测试（开关组合矩阵、门控梯度非零、forward/streamed/adaptive 三路等价、resume 与 BF16）；D4 一轮有界留一对照（协议在第一次优化器更新前冻结：臂 = RW-A / RW-B / RW-B−(a) / RW-B−(b)，2 seed × 400 updates、只读 val、test 封存、≤1.0 GPU-h；逐变量 ×6/12/24/48/72h 全表 + 三态计数 + 参数量/FLOPs/显存/墙钟四表）；D5 预声明单一对比的读数与归因（对每个留一臂按既有比较器算「臂 − RW-A」在 t2m 五时效上的逐 seed 同号三态：(a) 臂是负控制、(b) 臂是主问句，分支规则见长文 §3）；D6 证据文档与未做/limitations 明写、negative 原样保留、CI run id 与 SHA 绑定。禁止：新增 solver 部件、改已冻结判据或阈值、新增端点、重跑或改写上一轮产物、读 test、下载数据、租 GPU、改 main、force push、关闭 #70–#75、自动进入下一轮。预算 ≤1.0 GPU-h（第二批次账本结余 20.701 之内），确认轮另立 ≤2.5 GPU-h；停止条件为预算用尽、D1 判定必须先停、或留一对照仍无法归因（此时按停止条件 3 停止发明新模块，转去重新检查 forecast state / training objective / data regime）。不要自行宣布目标完成。
```

## §1 现状（带 file:line，2026-09-30 核对）

- 分支 `r7/weather-reasoning`，写本文件时 HEAD `0a6b501`；`main` 是它的严格祖先；open issue 恰为 #70–#75。
  **启动时必须自己 `git rev-parse HEAD` 复核**。
- **上一轮 RW-B 已执行并判负**：登记主端点上 `RW-B − RW-A`（t2m，2 seed 逐 seed 同号）
  = 6h **+0.118** / 12h **−0.079** / 24h **−0.075** / 48h **+1.065** / 72h **+1.580** K，
  模态读数 **worsened**；两条预注册证伪条件**均未触发**，故严格读法是「模态不支持且长时效受损」。
  见 `docs/R7_72_RW_B_PILOT.md:84-94`、证据索引记录 `rw-b-bounded-round-negative`、`docs/rules/EVIDENCE.md` 的 E-198/E-199/E-200。
- **机制线索（本轮 D1 要归因的对象）**：RW-B 的第 1 步修正幅度约为 RW-A 的 **1.8 倍**（能量约 3.4 倍），
  误差—修正余弦在第 2–3 步**转正**（+0.070/+0.057，RW-A 全程为负），第 3 步恶化比例 0.61/0.57 超过一半；
  K=1/2/4 深度探针两个 seed 都不单调改善。见 `docs/R7_72_RW_B_PILOT.md:157` 与 E-199。
  上一轮**不能**据此归因到具体部件（臂未做容量匹配，且没有消融臂）——这正是本轮存在的理由。
- **三个部件在实现里的位置（拆分对象）**：`model/local_solver_state_r7.py` 的
  `LocalSolverState`（Z 的 ConvGRU 式递推，:83 / forward :112-135）、`PositionGate`（逐位置门控，:138/:155）、
  `anchored_proposal`（`Y_proposal = X_t + Decoder(Z)`，:184）、`blend_forecast`（:201）；
  唯一的 step 体在 `model/process_step_r7.py:99-119`，RW-B 分支的四步顺序是
  :112 递推 → :114 提案 → :116 门控 → :118 混合。开关接线：
  `model/process_forecast_r7.py:81/198`、`model/r7_halting.py:157`、`model/recursive_weather_r7.py:131/166`。
- **(c) 已可分离、且上一轮已测**：`source_role_markers` 本来就是独立开关，上一轮的第四臂
  `process_local_solver_roles` 就是它；对比是 **0.006–0.028 K 的小幅且方向不一致**（48h 聚合反向）
  ⇒ 本轮**不再花 GPU 重测它**，只引用该对比作为「已排除为主因」的证据，且**不得**写成机制声明
  （`docs/R7_72_RW_B_PILOT.md:215`）。
- **上一轮产物（只读，不覆盖）**：`outputs/r7_72_rw_b_pilot/` 的 `merged_result.json`、
  `paired_comparison.json`、`arm_table.csv`/`rmse_table.csv`/`case_table.csv`/`training_table.csv`/
  `memory_table.csv`/`depth_probe_table.csv`、`seed{41,42}/protocol.json`，以及
  **32 个 checkpoint**（`seed{41,42}/training/<臂>/update_{0100,0200,0300,0400}.pt`，实测 2026-09-30 仍在）。
- **可复用件**：驱动 `scripts/study_r7_72_rw_b.py`（臂与协议族定义都写在文档字符串里）、
  多 seed 入口 `scripts/run_r7_72_rw_b_multiseed.sh`、成本表 `tools/measure_r7_rw_b_cost.py`、
  探针 `training/r7_solver_probes.py`、0 GPU-h 诊断范本 `scripts/study_r7_e0_correction_replay.py`。
- **账本**：第二批次 ≤24 GPU-h 自主分配（`docs/goals/full-auto-campaign.md:39`），
  已用 2.463 + 0.8356 = 3.299，**结余 20.701 GPU-h**；启动时按上一轮 §5 重读，不得沿用本行数字。
- **已知陷阱（承接上一轮）**：`total_updates` 不在 runner 契约里而学习率是它的函数；协议 digest 不在
  checkpoint 里；`model/**` 任一字节变化都会改 `model_code_sha256`（提交信息带 `[model-digest-change]`
  并记 `docs/rules/CHANGELOG.md`）；干净检出验证要用 `git clone` 而不是 `git archive`（上一轮 CI 教训）。
- 工程态：37 条阻断规则 0 违规；门禁是 `R7 CPU CI` 的八步。

## §2 交付物清单

| 编号 | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | 0 GPU-h 机制探针 | 只读 32 个 checkpoint + 只读 val 的逐部件几何读数（第 1 步修正幅度、逐步误差—修正余弦、第 3 步恶化比例）+ 输入 checkpoint 的 sha256 + 命令；**0 GPU-h、不新增阈值** |
| D2 | `local_solver_state` 的两个子开关 | 代码 diff + 两条逐位等价证据（**全关 ≡ 前一版实现**；**`local_solver_state=True` 全开 ≡ 上一轮**） |
| D3 | 定向测试 | 开关组合矩阵、门控梯度非零、三路（forward/streamed/adaptive）等价、resume、BF16；`pytest -q` 的 pass/skip 计数（skip 逐条给理由） |
| D4 | 一轮有界留一对照 | `protocol.json`（第一步更新前冻结，含 digest）+ 逐 seed 结果 + 逐变量 ×6/12/24/48/72h 全表 + 三态计数 + 参数量/FLOPs/显存/墙钟四表 + 实测 GPU-h |
| D5 | 归因结论 | §3 单一对比的读数与分支判定（含「不能归因」这种合法结果）；若触发停止条件 3，附转向清单 |
| D6 | 证据文档与登记 | `docs/R7_*.md` + `docs/R7_EVIDENCE_INDEX.jsonl` 记录 + 未做/limitations 明写、negative 原样保留、CI run id 与 SHA 绑定 |

## §3 判据与证据来源

- **工程判据**：`python tools/check_conventions.py` 37 条阻断 0 违规；`pytest -q` 无失败；
  `git show --check` 干净、新文件提交前跑 `git diff --cached --check`；
  干净检出用 `git clone` 验证（`docs/rules/ci-and-verification.md`）。
- **接口判据**：两个新子开关默认**开**，只在 `local_solver_state=True` 时起作用，因此
  **上一轮的配置逐位不变**；组合矩阵的两端必须逐位相同：
  (i) `local_solver_state=False` ≡ 前一版实现（沿用 `tests/test_r7_switched_path_equivalence.py` 的做法）；
  (ii) `local_solver_state=True` 且两子开关全开 ≡ 上一轮的 RW-B（用同一 seed/config 的逐张量相等）。
  语义由实现定义并写进 docstring：(a) 关 ⇒ 该步走 pre-RW-B 的 correction 路径（Z 若仍递推则是闲置计算，
  不得悄悄把 proposal 换成别的东西）；(b) 关 ⇒ Z 不跨步携带（提案与门控仍施加，Z 停在 `solver_init`）。
- **预声明的单一对比（本轮的核心判据，必须在任何训练之前写进 `protocol.json`）**：
  - 统一比较：对每个留一臂 X，用**既有**比较器（#60 的逐 seed 同号三态 supported / worsened / unresolved）
    算 `X − RW-A` 在 **t2m × 6/12/24/48/72h × 2 seed** 上的读数。**不新增阈值、不新增端点、不改案例集**。
  - 规则（一条预声明，用在两臂上，穷尽且互斥）：
    1. **`RW-B−(a)` 是负控制**：它表示「Z 递推在但提案/门控不参与输出」。若它自己出现同号恶化
       ⇒ 协议或权重有混淆，**停下报告**，本轮不做归因；
    2. **`RW-B−(b)` 是主问句**：
       - 48h 与 72h 两格**不再是 worsened**（逐 seed 同号）⇒ 归因 **(b) Z 的跨步递推**；
       - 48h/72h **仍是 worsened** ⇒ 在「只剩门控+锚定提案被施加」的前提下归因 **(a) 门控+锚定提案**；
       - 两臂都落进 unresolved ⇒ 报「**不能归因**」，按 §5 停止条件 3 停止发明新模块。
  - **(c) role 标记**：只引用上一轮已登记的对比（0.006–0.028 K、方向不一致），不重跑、不写成机制声明。
  - 主端点的**改善**（12/24h −0.075…−0.079 K）不得当作加码理由，也不得当作支持证据。
- **运行资格**：`queued` / `cancelled` / `skipped` / `partial` 不算通过；commit 标签只是触发意图，
  **不是授权**（决策 0021）。
- **科学判据（本轮不要求达到）**：≥3 固定 seed、精确配对、公平信息预算与算力报告、冻结 test、
  不确定度、负结果保留——见 `docs/plans/0004-r7-main-model-v2.md` 的 Scientific gates 一节。

## §4 实施顺序（按机制依赖，不跳步）

1. **fresh-read**：重读 GitHub（#70–#75 全文与评论）、`docs/goals/main-model-v2-rw-b-round.md` §5/§8、
   `docs/R7_72_RW_B_PILOT.md`、`docs/R7_MAIN_MODEL_V2_DESIGN.md` §3.2、第二批次账本。
   已冻结路线不重新规划；与本文件冲突时以仓库/GitHub 现状为准并如实记录。
2. **D1（先做，0 GPU-h）**：只读 checkpoint 与 val 的机制探针。若 checkpoint 缺失、或探针需要
   **新判据**，先停下报告用户，不得自行新增阈值。
3. **D2/D3（实现与测试同轮）**：只拆开关、**不加部件**；`model/**` 变化后提交信息带
   `[model-digest-change]` 并在 `docs/rules/CHANGELOG.md` 记录 digest 变化；新文件受 R-051/R-052/R-053 约束。
4. **D4（有界实验）**：执行那一刻按决策 0021 取授权（写明范围/预算/产物与证据/失败与 skip 处理），
   并**重读账本**；协议在第一次 `optimizer.step()` 前以排他方式冻结；单次实验 ≤30 min，超出拆分。
5. **D5/D6（归因与收尾）**：按 §3 读出分支判定，写证据文档、登记索引、绑定 CI run id、明写未做与 limitations。
   若判定为「不能归因」或 `RW-B−(b)` 仍 worsened 且负控制通过，**停止发明新模块**（§5 停止条件 3）。

## §5 预算与停止条件

- **Pilot**：4 臂 × 2 seed × 每臂 400 updates，单轮自设上限 **≤1.0 GPU-h**（与上一轮同规格；
  上一轮同形状的干净尝试实测 0.4154 GPU-h）。
- **确认轮**：只有留一臂给出可解释信号时才补 ≥3 seed，上限 **≤2.5 GPU-h**。
- 单次实验 ≤30 min；不新下载数据（0 GiB）。
- **停止条件**（满足任一即停并向用户报告，不自行扩大范围）：
  1. 预算用尽或账本显示不足；
  2. D1 判定必须先停（需要新判据、checkpoint 不可用、或必须先解决 seed42 异号）；
  3. **留一对照仍无法归因**：负控制失败，或两臂都 unresolved —— 此时停止发明新模块，
     记录可反驳假设并转去重新检查 forecast state / training objective / data regime；
  4. 需要新数据、新 GPU 租用、合并 main 或任何 destructive 操作。
- 目标状态 `active / paused / budget_limited / complete`，**执行者只可建议，不得自宣完成**。

## §6 与 planner 草案的差异

本轮不委派 planner：任务边界由上一轮长文 §8 的「下一动作」与 §5 停止条件 3 直接写定，
没有可设计空间需要外部方案。

## §7 明确不做

- **不新增 solver 部件**（门控/锚定提案/Z 递推/role 之外的任何新模块），不把 12/24h 的改善当作加码理由。
- 不改已冻结的判据、阈值、端点或案例集；不新增端点；不重跑第一/二/三/四轮的已判决消融。
- **不重跑、不覆盖、不改写** `outputs/r7_72_rw_b_pilot/` 的产物与 `docs/R7_72_RW_B_PILOT.md`（它们是证据）。
- 不读封存 test 做方法选择；不把旧 test 重新包装成 pristine；不下载新数据；不租 GPU。
- 不动 `data/raw|interim|processed`；不修改、移动或重命名归档快照。
- 不改 main、不 force push、不合并、不 release、不创建定时任务、不关闭 #70–#75、不自动进入下一轮。
- 不把 role 标记的小幅一致改善写成机制声明；不把三个 seed 写成显著性；不把「训练 loss 下降」写成任务收益。
- 不为让报告好看而放宽判据、删除测试或弱化断言。

## §8 进度块

- **状态**：`active`（D1–D6 已执行完毕；**执行者不自行宣布目标完成**，等独立校验）
- **起点 SHA**：`e6085bc8a8ab173f6208ed210bf8484f323a56a4`（开工 `git rev-parse HEAD` 复核过）
- **账本**：开工时按上一轮 §5 重读为 **20.701**；本轮 D4 实测 **0.4128 GPU-h** ⇒ **20.288**（D1 为 0 GPU-h）
- **已完成**：
  - **D1（0 GPU-h）**：`scripts/study_r7_rw_b_subtraction_probe.py` + `outputs/r7_rw_b_subtraction_probe/probe.json`
    （CPU 6.2 s，只读上一轮 32 个 checkpoint 与 val，hash 全部记录，`--code-root` 指向训练它们的修订并核对 digest）。
    读数：移除门控+锚定提案后第 1 步幅度塌到 RW-A 的 **0.20/0.24 倍**（两 seed 同向）；
    移除 Z 的递推后**余弦从第 1 步就转正**（+0.010/+0.021）而幅度在两 seed 间从 1.48 跳到 3.04。
    ⇒ 幅度需要 (a) 通路存在，余弦转正出现在任何让 Z 停止跨步递推的配置里，**两者不是同一件带来的**。
    同时测出焦点臂的 `correction_head` **训练时从未收到梯度**（所以「移除 (a)」那一行不能当训练期结论）。
  - **D2/D3**：`model/process_forecast_r7.py` + `model/process_step_r7.py` 拆出两个**默认 True** 的子开关
    `solver_state_recurrence`（Z 停在 `solver_init`，提案与门控仍施加、状态不再传递）与
    `solver_gate_proposal`（走 pre-RW-B 的 correction 通路，Z 若仍递推则只是闲置计算）；
    两开关只在 `local_solver_state=True` 时存在、新增**零参数**。两条逐位等价都有实跑证据
    （全关 ≡ 起点修订；`local_solver_state=True` 全开 ≡ 起点修订的 RW-B），另有「扰动冻结树必须打破等价」的反证。
    `tests/test_r7_rw_b_subtraction.py` 15 项；本机 `pytest -q` **1499 passed / 3 skipped**。
  - **D4**：`scripts/study_r7_72_rw_b_subtraction.py` + `training/r7_rw_b_subtraction_protocol.py`；
    产物 `outputs/r7_72_rw_b_subtraction/`。4 臂 × 2 seed × 400 updates，协议在第一次 `optimizer.step()`
    前以 `'x'` 冻结并回读重算 digest（`58fc74b7…`，8 个 run 全同），只读 val、test 封存、四臂均未早停。
    实测 **0.4128 GPU-h**（训练 0.376 + 评估 0.0368），未超 1.0 的自设上限。
  - **D5**：按预声明规则读出 **`stop-confounded-control`** ⇒ **本轮不做归因**。但诊断出规则前提不成立：
    负控制 `RW-B−(a)` **在构造上退化为 RW-A 本身**——同一份权重下前向/逐步 drafts/adaptive **逐位相同**、
    训练损失**逐位相同**、131 个共享张量梯度**全部逐位相同**且 solver 侧 0 个非零梯度；
    因此它那 3e-05…8e-05 K 的「worsened」是浮点噪声（作为量级参照：RW-B 对 RW-A 的权重相对差是 1.74，
    负控制是 1.6e-4）。主问句 `RW-B−(b)` 自身读数是干净的（48h +0.785 / 72h +1.174，两 seed 同号 worsened），
    **若**负控制可用则规则会指向 (a)，但本轮**不把它写成归因**。
  - **D6**：证据文档 `docs/R7_72_RW_B_SUBTRACTION.md`（negative 原样保留、limitations 与未做事明写）；
    台账 **E-202 – E-206**；证据索引新增两条记录（`rw-b-subtraction-probe`、`rw-b-subtraction-round-cannot-attribute`）
    并重生成 `R7_CANDIDATE_BRIEF.md`；`model_code_sha256` 变化记入 `docs/rules/CHANGELOG.md`（带 `[model-digest-change]`）。
  - **D6 收尾（CI 与干净检出绑定，2026-09-30）**：`git reflog show origin/r7/weather-reasoning` 证明
    本轮只有一次 push（`e6085bc → 13bbcd6`），唯一 run **`36621458964` = success**（八步全绿；
    18 条 workflow 中 17 条实验 workflow 按标签门控 skipped=设计行为，不算失败也不当通过）；
    被同一次 push 带上来的五个中间 SHA 记为「无独立 run」（不是通过）。起点 `e6085bc` 的
    开工前 run `36606529478` = success。干净检出实测：`git clone` 检出 `13bbcd6` →
    **1494 passed / 8 skipped / 0 failed**；本机全量（有产物）**1499 passed / 3 skipped**。
    两处钉住值缺陷在推送前本地全量暴露并修复（`5bf0b73`、`926d2fe`、`13bbcd6`），
    **本轮没有一次红是推上去才发现的**。证据页 §9/§10 已按此定稿。
  - **收尾提交序列（只动文档 / 索引，无模型与实验改动）**：提交 **A** = 证据页 §9/§10 定稿 +
    本进度块 = `5dc479e`；随后以 `--commit 5dc479e` 重跑 `tools/append_rw_b_subtraction_evidence.py`
    重钉索引（digest 取工作树、`evidence_commit` 指向 A——A 就是持有该页版本的提交），
    提交 **B** = 索引 + brief = `590d1f1`，与 A **同一次 push**（`13bbcd6..590d1f1`）。
    按 GitHub 只对推送 head 触发的契约：**B 的 run `36623978223` = success**（八步全绿，逐项已核），
    **A 没有独立 run**（「无证据」，不是通过）。最后一个收尾提交自身的 run 号无法自引用——
    头部提交的固有边界，照写而不是回避；本块的运行记录以 `590d1f1` 的绿为准。
  - **轮末 Mimosa 深度扫描（2026-09-30 04:04，commit/push hook 报 `scanner_enobufs` 后补跑）**：
    跑完，`Run status: inconclusive`、`verdictEffect: none`、封条
    `sha256:3581fc680777f8d0ff9f057ca2a362351b1627d7a45ac0147181e73a91cbdb8d`；
    27 条发现（22 high / 4 medium / 1 low），**逐条核对全部位于只读归档快照 `legacy_v531_full/` 内，
    本轮 focus 文件命中 0 条**，活跃代码里没有一条。**没有对任何一条做分类或修复**，
    因此正确说法是「扫描跑了、发现集中在归档快照、结论是 inconclusive」，
    **不得**声称项目通过了完整安全审计。
- **未做**（详见证据文档 §7）：
  - **未做归因**（负控制触发即停，不为补救加臂或放宽判据）；确认轮（≤2.5 GPU-h）**未动用**；未跑第三 seed。
  - 未读 test、未下载数据、未租 GPU、未改 main、未 force push、未关 #70–#75。
  - **未重跑或改写上一轮任何产物**（`outputs/r7_72_rw_b_pilot/` 只被只读打开）。
  - (c) role 标记**未重测**，只引用上一轮已登记的对照，未写成机制声明。
- **下一动作**（按停止条件 3）：停止发明新模块；记下**可反驳假设**——「换一个真正可区分的负控制
  （例如把 Z 换成同形状冻结随机张量、其余不变），48/72h 的恶化应当仍然出现」；并转去重新检查
  forecast state / training objective / data regime（证据文档 §8 给出了具体第一动作）。
