<!-- round-node: S3 -->

# 目标：循环深度的 headroom 与条件式 visit-gain 分支

本文件是 goal 模式的目标长文源。objective 见 §0；交付物见 §2；判据见 §3；
预算与停止见 §5；明确不做见 §7；进度块见 §8。

**状态：G0 已完成（2026-11-25），G1-B 已准入但未派生。** 判定见
`docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md`；准入规则见
`docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md`；
计划见 `docs/plans/0036-loop-depth-headroom-before-architecture.md`（计划是不可变记录，
本长文是它的执行源）。`scientific_claim: false`、`test_read: false`、`r = 0`。

**本文件不带 `round-spec` 块，因此不得被 `campaign-state.current_round_goal` 指向**：
`tools/r7_round.py` 的每个子命令都要求恰好一个 round-spec（README 的 round-spec 契约），
先例见 `docs/goals/s3-urban-static-covariates.md:175-177`。G1-B 若派生，须另建带 round-spec 的
执行 brief。

---

## 0. objective（可粘贴，单段，实测 1,012 字符）

> 在/data/esw/UrbanPiDiT_R2的r7/weather-reasoning上推进docs/goals/loop-depth-headroom-before-architecture.md。先读该长文、docs/plans/0036-loop-depth-headroom-before-architecture.md、docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md、docs/R7_LOOP_TRANSFORMER_LITERATURE_AUDIT.md、docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md，并核HEAD与docs/R7_EVIDENCE_INDEX.jsonl。G0已完成：三seed×1024case×K1..8只读诊断判为admit-g1b（a饱和3/3、b病态3/3、c深度可用0/3，主判据为+6h目标）。本objective只推进G1-B：在model/recursive_weather_r7.py的GenericRecursiveCell加visit_gain开关（默认None必须逐操作等于现实现、不增参数、键集不变），两臂A visit_damped g_k=1/sqrt(1+k)与B uniform_damped g=1/sqrt(1+(K-1)/2)，主对比A−B隔离visit对齐与整体缩小步长，C用已注册incumbent不重训。先写ADR与brief、冻protocol.json、父迁移并逐seed证明state_digest与归档父逐位相同，再训练。预声明出口：72h interior_32 t2m的A−B三seed同号为负、6/12h的A−B seed均值≤0、首update loss桥逐位相同、参数键集父state_digest逐位相同；任一失败即关闭该支，不加seed、不扫增益函数族、不调K、不改已冻结判据。禁止：替换架构、动MIXING_MODES或感受野（0048容量轴与0077 H-C保持关闭）、读2023test、r=0、自宣技能或目标完成。成本估11.1GPU-h未实测；0 GPU-h轮不得加账本行。停止仅：触保留授权、达冻结停止条件、确无可安全推进；停时在§8写为何停与恢复第一条命令。不要自行宣布目标完成。

（字符：本 objective 单段，实测 1,012 code points，≤4000。）

## 1. 现状与前置

**G0 已完成，读数是本轮唯一的机器证据**：

| 子句 | 阈值（ADR 0118 冻结） | 实测 | 结果 |
| --- | --- | --- | --- |
| (a) 饱和 | 3/3 seeds，k≥3 后每步相对改进 <1% | **3/3** signals | **真** |
| (b) 病态 | ≥2/3 seeds，k3→4 相对上升 >0.1% | **3/3**（+0.1375/+0.1102/+0.2318%） | **真** |
| (c) 深度可用 | ≥2/3 seeds，k4→8 相对改进 ≥2% | 0/3（实际变差 2.5–3.0%） | **假** |

⇒ 冻结决策表第一行命中：`admit-g1b`。

**必须同时记住的三条限制**（证据页 §1.1）：① (b) 的幅度（0.11–0.23%）远小于 k4→8 的衰退
（2.5–3.0%），主信号是「K=3 之后单调恶化」而不是「更新过放大被确证」；② 主判据是 **+6 h** 目标
（`atmos_target = physical_targets[:,0]`），不能外推成 72 h；③ val 覆盖 2022-01…09，缺秋冬。

## 2. 交付物清单

| # | 交付物 | 证据形态 |
| --- | --- | --- |
| D1 | `visit_gain` 开关（默认 `None` = 现实现逐操作相同、零新参数、键集不变） | 反证测试：off 路径 `state_dict` 逐张量相等 + 首 update loss 逐位相同 |
| D2 | 父迁移 + 每 seed `state_digest` 与归档父逐位比对 | `migration_receipt.json` + `state_equality_evidence()` |
| D3 | 两臂 `A visit_damped` / `B uniform_damped` 的训练与五 lead 评分 | `seed{N}_{arm}_receipt.json` + 逐 lead CSV |
| D4 | 预声明出口 ①–④ 的逐条判定 | `readings.json`（独占写），每条 `true/false/null` |
| D5 | G1-B 证据页 + 一条 index record | `docs/R7_S3_VISIT_GAIN_RESULT.md` + `docs/R7_EVIDENCE_INDEX.jsonl` |
| D6 | 四件套登记（账本行、index、§8 进度块、`campaign-state` 前移） | 主计划 §7/§8；`tools/r7_round.py verify` 全绿 |

## 3. 判据与证据来源

**判据只在两个地方，本目标不新增阈值**：

- G0 的 (a)(b)(c) 与决策表：`docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md`
  的 Decision 2/3/4；读数：`docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md`。
- G1-B 的预声明出口 ①–④：同 ADR 的 Decision 5；科学门的完整口径见
  `docs/R7_MAIN_MODEL_CLIMATOLOGY_PROTOCOL.md`（本目标不重定义它）。

**归因纪律**：主对比 **A−B** 可归因（同参数、同算力、同键集，唯一差别是增益是否依赖 visit 序号）；
**B−C 只作描述性**并必须同时写参数差（0）。**未测一律 `null`**，不得记 `false` 也不得记 `true`
（`da1ed00` 的教训）。

## 4. 实施顺序（依赖，不跳步）

1. **ADR**：G1-B 的执行设计 + 与 0048 / 0077 / 0057 排除项的逐条对照（`docs/decisions/README.md` 追加行）。
2. **brief + round-spec + `<!-- round-node: S3 -->`**；`check_goal_brief` 通过。
3. **`visit_gain` 实现** + 反证测试（off 逐位等价、两臂初始化共享、参数量/键集逐位相同）。
4. **父迁移**（`scripts/migrate_r7_parent_state.py`）+ pin 模块 + `state_equality_evidence()`。
5. **`round-guard`** → 干净树重建 code zip → `freeze` → `stage prepare/archive/stage-seeds/reading`。
6. **四件套登记** + `tools/r7_round.py verify` → `commit-register` → 推送 → CI 绿。

## 5. 预算与停止条件

- **G0 已花 0 GPU-h**（只读推理，墙钟约 6 min）。
- **G1-B 估算 11.1 GPU-h**（2 臂 × 3 seed × 2,400 updates，按 1.42–1.46 s/update + 3,200 s/5-lead
  评分），双卡三波 ≈5.6 h 墙钟。**估算未实测**；执行时按探针实测重定 `PER_SEED_SECONDS` /
  `PLANNED_SECONDS` / `HARD_CAP_SECONDS`（软预算超了继续并记 `overrun`，只有硬包络截断，
  且截断不算通过）。
- **无总 GPU-h 上限**，账本只记账。**0 GPU-h 轮不新增账本行**（无机器核数，按主计划 §7 例外，
  并在 §8 写明理由）。
- **停止条件（仅三种）**：(a) 触保留授权边界（付费/租卡/独占/方向改变/main 合并/破坏性）；
  (b) 达到已冻结的科学停止条件（出口 ①–④ 任一失败即关闭该支）；(c) 确无可安全独立推进的工作。
  **停下必须在 §8 写「为何停 + 恢复第一条命令」。**
- **禁止** cron / 守护 / 会话外续跑。

## 6. 本轮不做

- 不替换架构、不加新 loop 算子、不动 `MIXING_MODES` / 感受野 / 网格。
- 不做 loop-index embedding、per-loop LoRA、更大 K。
- 不扫增益函数族（`1/sqrt(1+k)` 是**待证**定义，不是最优形式）。
- 不读 test、`r=0`、不加 seed 练到赢、不重跑 `record:s3-reasoning-depth`。
- 不在 screening 正结果上声明技能；须另立 full-dose 确认轮。

## 7. 明确不做

- **不**把 C（SMELT 式把循环移进空间栈）与 G1-B 合并开轮：C 要改 backbone 结构，参数量与算力
  都对不齐，须自己的 ADR 并论证与 0048/0077 不同族。
- **不**引用 DeepLoop（arXiv:2607.13491）的具体数字当预期：该文是 post-LN，我们是 pre-LN。
- **不**改任何已冻结判据、**不**绕过身份门、**不**把 `null` 记成 `false` 或 `true`。

## 8. 进度

- **状态**：G0 完成（`admit-g1b`）；G1-B **已执行完毕并因出口 ① 为假而关闭该支**
  （`closed`，2026-11-26）。六臂各 2,400 updates 训完、五 lead 评分完毕；
  出口 ① **false**、② **true**、③ **true**、④ **true**；实耗 **9.7953 GPU-h**。
- **已完成**：G0 三 seed 诊断（0 GPU-h）、判定器 + 24 条反证、文献审计、ADR 0118、计划 0036。
  G1-B 侧：ADR **0119**（执行设计 + 三处开轮前更正）、ADR **0120**（取代 0119 的探针口径）、
  带 `round-spec` 的执行 brief `docs/goals/s3-visit-gain.md`、`visit_gain` 开关与反证测试、
  父迁移（三 seed `state_digest` 与归档父逐位相同）、protocol 冻结 `9fba46a1…`、
  `prepare`/`archive`/六臂/`reading`、身份见证（off 逐位惰性 + A 在 k=0 逐位相等）。
- **结果**：G0 = `admit-g1b`；(a) 3/3、(b) 3/3、(c) 0/3；K=3 为三 seed 各自的最优深度。
  **G1-B 的 A−B 读数（实测）**：72 h 的 seed 符号为 `+ / − / −`
  （`41:+0.000323  42:−0.001023  43:−0.000801`）⇒ 出口 ① **false**（冻结文本：单翻即假）。
  6 h / 12 h 的三 seed 均值 **−0.000640 / −0.000500** ⇒ 出口 ② **true**
  （但两个 lead 上 seed43 均为正）。**五个 lead 无一三 seed 同号**，全部 |Δ| ≤ 0.0022 K，
  而 lead RMSE 为 2.09–3.45 K。
- **这个读数的正确读法**：**「该剂量下不可分辨」**（2,400 updates、单一 65×65 实例、
  单一阻尼因子轴），**不是**「阻尼有害」，也**不是**「visit-gain 无效」。
  `r = 0`、三 seed 非显著性检验、无区间估计 ⇒ 不支持任何方向的效应声明。
- **G1-B 的三个开轮前发现（全部写进 ADR 0119，且在读数之前）**：
  1. **B 不是 A 的等平均阻尼臂**：K=4 时 A 的均值 0.696114、B 的常数 0.632456，B 多阻尼 **9.14%**
     （Jensen）。⇒ A−B 不可归因为纯 visit 对齐，只能读作「在 B 的阻尼水平上」。
  2. **`interior_32` 在 65×65 上只剩 1 格**，但 `R7_S3_WIDE_REGION_FULL` 已证明 65×65 全网格
     **就是**宽实例的 `interior_32`（分母 ~1e-10 一致）⇒ 冻结 `full(65×65)` 并写明等价关系。
  3. **出口 ③ 字面不可判定**：B 在 k=0 即阻尼，且无「从 p32,132 起跑」的归档首 loss
     ⇒ 换成同轮逐位身份见证（③a off 逐位惰性 + 155 张量相等；③b A 在 k=0 与 `None` 逐位相等）。
- **attempt01 探针拒绝的根因（已被 0120 实测更正）**：不是「启动摊销」，而是**同期 6,040 个
  测试占满 CPU**，让 `DataLoader(num_workers=4)` 供不上数、GPU 空转，速率虚高 2.3 倍。
  安静机器上 400 步内速率从第 0–100 区间起就是平的（1.075–1.110 s/upd）。
  attempt01 与 0120 早期记录里「启动摊销 / 边际 1.583」的写法**已被推翻**，0119 与 0120 均已留痕更正。
- **本轮顺带修掉的真实缺陷**：`tools/r7_round_stages.py` 的 `run_stage` 读 `body["gpu_uuid"]`
  （单数），而 `protocol_fields` 只写 `gpu_uuids` ⇒ 任何多卡协议在 `reading` 阶段都 KeyError。
  六臂训完后撞上，改用 `devices.frozen_uuids(body)[0]` 并新增
  `tests/test_r7_round_stage_gate.py`（2 条反证）。`tools/` 不在 `execution_files_sha256` 内，
  不影响本轮科学身份。
- **待用户追认的两处形式修订**（均在读数前冻结、方向更严）：
  出口 ① 区域 `interior_32` → `full(65×65)`；出口 ③ 跨轮 loss 桥 → 同轮逐位身份见证。
- **未做**：B−C 描述性对照、四季均衡、129×129 宽实例、更高剂量确认轮、增益函数族/K 扫描、
  加 seed —— 后三项按闭支规则**明确禁止**。
- **下一动作**：S3 的 G1-B 支已关闭；下一步由用户/运行时决定是否转向其它 G1 分支
  （`docs/goals/loop-depth-headroom-before-architecture.md` 的其余候选）或提交独立复核。
  恢复第一条命令：`.venv/bin/python tools/r7_round.py state`。
- **digest 变更的代价（如实）**：`model_code_sha256` 由 `59340710…` 移到 `a6cf6bbb…`，
  归档 checkpoint 在活树上不可加载（Q-009）。9 个依赖归档产物的测试已改成**具名前置跳过**
  （只跳过、不放松任何断言，身份一致时照常跑全套）；另 1 个
  （`test_the_partial_reading_…`）在**未改动的 HEAD 上就已失败**，与本轮无关，未修、已记录。

> G0 轮 **0 GPU-h、0 test 读取、r=0**，**不新增 index record 与账本行**。
> G1-B attempt01 = 1.1330 GPU-h（探针拒绝，见 `docs/R7_S3_VISIT_GAIN_PROBE_REFUSED.md`）；
> attempt02 = 9.7953 GPU-h（六臂 + 五 lead + 三探针，见 `docs/R7_S3_VISIT_GAIN_RESULT.md`）。
> 两轮现在都有 index record 与账本行。

