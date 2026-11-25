# 0036 — 动循环架构之前先量 headroom：0 GPU-h 诊断 + 条件式 visit-gain 分支

- **日期**：2026-11-25
- **状态**：**定稿归档**（archived）
- **分支**：`r7/weather-reasoning`
- **`scientific_claim: false`**、`test_read: false`、`r = 0`
- **准入裁决**：决策 **0118**（G0 判据 + G1-B 条件式准入）
- **执行源**：`docs/goals/loop-depth-headroom-before-architecture.md`
- **上游**：`docs/R7_LOOP_TRANSFORMER_LITERATURE_AUDIT.md`、
  `docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md`

## 实际结果

- **完成情况**：G0 诊断**已执行**（三 seed × 1024 case × K=1..8，0 GPU-h）；判定器与 24 条反证测试
  落地；四份归档文档（文献审计 / 本计划 / ADR 0118 / goal brief）写完。**G1-B 未派生、未训练。**
- **与计划的差异**：计划里 G0 用 `--max-samples 256`，实际跑 **1024**（实现上限），并因 val manifest
  按时间排序而只覆盖 2022-01…09——这条限制写进了证据页 §1.1。判定结果 `admit-g1b` 与「先量再动」的
  预期一致，但**主信号比 (b) 门槛强得多**（k4→8 变差 2.5–3.0% vs k3→4 上升 0.11–0.23%），
  证据页 §5 把「阻尼」与「改写空间表征」两种解读并列摆出，未选边。
- **验证**：`tests/test_r7_s3_loop_depth_headroom.py` 24 passed；`check_conventions --quiet --paths`
  三文件 37 条阻断 0 失败；`check_goal_brief` / `check_campaign_state` / `verify_r7_evidence_index`
  见本轮登记提交记录。
- **遗留**：G1-B 的成本是估算未实测；C（SMELT 式）仍缺「不是 0048/0077 同族」的论证；
  72 h 轨迹与四季均衡都未测。

---

## Context

用户问「用最新的 loop transformer 架构替换我们当前的，或者把 loop transformer 和我们的结合」。
只读测绘（`docs/R7_LOOP_TRANSFORMER_LITERATURE_AUDIT.md`）的结论是：**我们已经是** Huginn 式
prelude–tied-core–coda + deep supervision；「替换」会重开一批已关闭的轴。所以真正的问题是
「先补哪条差异」，而这**没有读数可依据**。

### 开轮前必须面对的事实

1. **循环已经吃掉一半前向时间**。65×65、单 batch、RTX 3090：K=0 为 7.0 ms，K=3 为 13.6 ms
   （loop 占 **49.7%**），K=8 为 24.2 ms（71.7%）。训练步 K=1→3 的 loop 部分约占总步时 25.6%。
2. **循环只更新 16 个 token**（8 anchored + 8 free，`model/process_forecast_r7.py:517`），
   而 1089-token 的 `context` 在整个循环内**逐位不变**（`model/process_step_r7.py:112`）。
3. **既有的 K 轨迹证据指向饱和/欠射**：`docs/R7_65_PREDIAGNOSTIC.md` §3 在 4 个 case 上测到
   K=3/K=4 转负、相邻修正 cosine→0.982、修正范数单调衰减。这条证据只有 4 个 case、800 步
   checkpoint，**不足以决策**——但它**足以否决「残差过放大」这个默认假设**，所以必须先量。
4. **`record:s3-reasoning-depth` 是 `unresolved`，不是 negative**，且它测的是「同一 cell 多跑几遍」，
   不是「循环能否改写空间表征」。
5. **`campaign-state.current_round_goal` 为 `null`**，且本轮 0 GPU-h 轮**不带 round-spec**，
   故**不得**把指针指到本轮的 goal brief（`r7_round.py` 要求恰好一个 round-spec，
   先例 `docs/goals/s3-urban-static-covariates.md:175-177`）。

### 为什么不是「直接开一轮残差缩放」

按先验，残差缩放假定「更新过放大」。若真相是「更新欠射/已饱和」，那一轮训练的负结果
**无法区分**「阻尼无效」与「本来就不需要更多步」——花 11 GPU-h 买到一个不可归因的 null。
G0 是 0 GPU-h 的只读推理，能把这个歧义消掉，**并且按 0118 的决策表可以否决分支**。

---

## 方案

### G0 诊断（本计划已执行）

| 项 | 值 |
| --- | --- |
| 入口 | `diagnose_r7_gain.py` → `training/r7_gain_oracle.py:collect_process_errors` |
| checkpoint | attempt02 的 incumbent 三 seed，`update_0032132.pt`（355 个张量，2.8M 参数） |
| 数据 | `store/manifests_treatment/val.jsonl`（val only，`split` 断言） |
| 预算 | `max_steps=8`、`max_samples=1024`、`step_cost=0` |
| 产物 | `outputs/r7_s3_loop_depth_headroom_20261125/seed{41,42,43}`，各为单个诊断 JSON |
| 判定 | `scripts/r7_s3_loop_depth_headroom.py` → `readings.json`（独占写） |

诊断入口自带三重保护并被测试覆盖：val-only 断言、`K≤16`/`samples≤1024` 硬上限、
`--out` 已存在即拒；前后比对 `state_digest` 证明模型未被改动。

### 判据（读数前已冻结进 ADR 0118，训练后不得改动）

- **(a) 饱和**：3/3 seeds 在 k∈{3..7} 上每步相对改进 <1%。
- **(b) 病态**：≥2/3 seeds 在 k3→4 相对上升 >0.1%。
- **(c) 深度可用**：≥2/3 seeds 在 k4→8 相对改进 ≥2%。
- **决策表按行序优先**：`(b)` 真 ⇒ `admit-g1b`；`(b)` 伪且 `(a)` 真 ⇒ `saturated-negative`；
  `(b)` 伪且 `(a)` 伪且 `(c)` 真 ⇒ `depth-usable-unresolved`；其余 ⇒ `unresolved`。
- **fail-closed**：任一 seed 缺失 / 非有限 / `max_steps<8` / 身份门失败 / `forecaster_unchanged≠true`
  ⇒ 整体 `unresolved`；**子集不外推**；`null` 不当作允许开分支的证据。

### G1-B（条件式，本计划**不实现**）

| 臂 | 因子 | 参数 | 角色 |
| --- | --- | --- | --- |
| **A** | `visit_damped`：`g_k = 1/sqrt(1+k)` | 与 B/C 逐位相同 | 处理 |
| **B** | `uniform_damped`：`g = 1/sqrt(1+(K−1)/2)` | 同上 | **主对照**（同平均阻尼、无 k 依赖） |
| **C** | 已注册 incumbent | 同上 | 描述性参照（pinned，不重训） |

- **主对比 A−B**：隔离「visit 对齐」与「整体缩小步长」。
- **身份**：父 checkpoint 张量逐个**逐位相同**，只需重签 `model_code_sha256`；
  每 seed 的 `state_digest` 与归档父逐位比对。
- **预声明出口**：①72 h `interior_32` t2m 的 A−B 三 seed 同号为负；②6/12 h 的 A−B seed 均值 ≤0；
  ③桥：首 update loss 与归档父同 update 逐位相同；④参数/键集/父 `state_digest` 逐位相同。
  任一失败即关闭该支，不加 seed、不扫增益函数族、不调 K。

---

## 要创建的文件

| 文件 | 性质 |
| --- | --- |
| `docs/R7_LOOP_TRANSFORMER_LITERATURE_AUDIT.md` | 文献审计（0 GPU-h，只读） |
| `docs/plans/0036-loop-depth-headroom-before-architecture.md` | 本文件 |
| `docs/decisions/0118-loop-depth-headroom-gate-and-branch-admission.md` | ADR，冻结 G0 判据与 G1-B 准入 |
| `docs/goals/loop-depth-headroom-before-architecture.md` | goal 长文（无 round-spec） |
| `scripts/r7_s3_loop_depth_headroom.py` | 判定器，189 行 |
| `tests/test_r7_s3_loop_depth_headroom.py` | 24 条反证，214 行 |
| `docs/R7_S3_LOOP_DEPTH_HEADROOM_RESULT.md` | 读数证据页 |

## 复用而非新建

- `diagnose_r7_gain.py:main()` — **不改**，本轮直接调用
- `training/r7_gain_oracle.py:collect_process_errors` — **不改**，已实现 val-only + `state_digest` 前后比对
- `training/r7_gain_oracle.py:retrospective_gain_analysis` — **不改**，提供 `mse_by_step` 与 oracle 深度
- `training/r7_experiment.py:load_checkpoint` — **不改**，身份门按设计拒绝跨 revision checkpoint

## 实测验证（本轮已跑，非计划）

| 检查 | 结果 |
| --- | --- |
| `model_code_digest()` vs attempt02 协议 | `59340710…` **逐位相同** |
| 三 checkpoint sha256 vs 各自 receipt | 三个**全等** |
| `pytest tests/test_r7_s3_loop_depth_headroom.py -q` | **24 passed** |
| `check_conventions --quiet --paths`（三文件） | **37 条阻断 0 失败** |
| 单文件行数 | 189 / 214（R-051 上限 600） |
| GPU 共驻门（启动前） | 两卡各 ~24 GiB free、0 计算进程 |

## 成本

| 段 | 内容 | GPU-h |
| --- | --- | ---: |
| G0 诊断 | 3 seed × 1024 case × K=1..8 只读推理 | **0** |
| 判定器 + 测试 | CPU | **0** |
| **本轮合计** | 墙钟约 6 min（两卡并发），网络 0，磁盘 ~2.1 MB | **0** |

**G1-B（若派生）的成本是估算**：2 臂 × 3 seed × 2,400 updates × 1.42–1.46 s/update + 3,200 s/5-lead
评分 ≈ **11.1 GPU-h**，双卡三波 ≈5.6 h 墙钟。**未实测。**

## 明确不做

- **不替换架构**、不加新 loop 算子、不动 `MIXING_MODES` / 感受野 / 网格（0048、0077 保持关闭）。
- **不做** loop-index embedding、per-loop LoRA、更大 K（Tier 3 或已登记勿重跑）。
- **不**在本轮派生或实现 G1-B；**不**预设增益函数族取值。
- **不读 test**、`r=0`、**不改 `campaign-state` 指针**、**不加账本行**（0 GPU-h 无机器核数）。
- **不**在 screening 正结果上声明技能；**不**改任何已冻结判据。
