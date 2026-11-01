# 0074 — R2（气候态锚异常反馈）的执行设计：batch 2 / 16,071 updates、本轮内重训的配对控制臂、当前修订下的父迁移与桥 (e) 的重设计

- **日期**：2026-11-01
- **状态**：accepted
- **范围**：`docs/plans/0026-matched-exposure-reeval-and-s4.md` §3 Track 1 的**第 2 轮（R2）**的执行设计；
  R2 的**开轮准入**不在这里，见决策 0058；预算常数见决策 0060
- **依据**：`docs/decisions/0058-anchor-anomaly-feedback-reeval-admission.md`、
  `docs/decisions/0060-matched-exposure-budget-correction.md`、
  `docs/goals/s3-matched-exposure-reeval.md` §4.2 的「每轮程序」与 2026-10-28「续 2」的 R2 只读前置核实、
  `docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md`（R1 的实测与桥 (e) 的失败）、
  `outputs/r7_s3_fy_anchor_table_20261025_attempt01/`（已发布的 48 桶表）、
  `outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/`（本轮迁移）

## Context

**① 队列推进到 R2。** R1（训练年份覆盖在匹配算力下重估）已于 2026-10-29 跑完并登记
（`record:s3-fy-year-coverage-matched-result`）：出口 **(a)(b)(c) 真、(d) 假**
（6/12/24 h 的 seed 均值 Δ 为 +0.004932 / +0.006434 / −0.015016 K）⇒ 「训练年份覆盖」支**第二次关闭**，
且**此后不得再以「算力不够」为由重开**。计划 0026 §3 的队列下一条就是 R2。

**② R2 的准入与制度早已在案。** 决策 0058 给出了「不是同一族」的三条论证与新制度的逐字定义；
brief `docs/goals/s3-anchor-anomaly-feedback-reeval.md` 与七 gate 已在；锚表已发布并经只读核实
（`table_sha256 18448ae3df2b260e…`、`[48,17,65,65]`、48 桶全部有帧、`_validate_spec` PASS、
四条恒等反证全过）。**0058 唯一留白的是 batch**，而计划 0026 §4 记录了用户的决定：**batch = 2**，
`updates = 16,071`（`16,071 × 2 = 32,142 ≥` 候选臂 32,142 窗 ⇒ 每臂 ≥1 epoch）。

**③ R1 的实测给本轮的成本与峰值下界。** R1 在**同一个 store、同一套 treatment 清单、同一 spec、
同一批 seed、batch 2 / 16,071 updates** 下实测：每臂训练 **37,996–42,254 s**、每臂五 lead 评分
**2,554–2,665 s**、每 seed（两臂）**88,362–89,520 s**、每 worker `owned_cuda_reserved_peak_bytes`
**5,582,618,624 B = 5,324.0 MiB**、六臂合计 **73.624654 GPU-h**，全部在 0060 的 per-seed 129,600 s 内。

**④ 一个新的前置：模型修订在 R1 冻结之后移动了。** 提交 `896eabf` 把城市静态协变量接进主干
（**默认关**，off 路径为「无模块、无 buffer、无张量」，提交自带的对照实测「无差异」），
`6f21f71` 只改 `data/preprocess/urban_static_aggregate.py`。⇒ `model_code_sha256` 由 R1 的
`60de3b2d…` 变为 **`a38be16c…`**，FY 轮的父钉（`r7_s3_fy_annual_parent`）按设计**无法再被加载**。
本轮**不改任何 `model/**`**（`anomaly_feedback` / `anchor_output_prior` / `climatology_anchor_spec`
三个开关均已存在），所以按既有先例走**模型只迁移**：`scripts/migrate_r7_parent_state.py` 在当前修订下
重跑，产物 `outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/`——
**155 个张量逐 seed 与旧迁移的 `state_digest` 逐位相同**（41 `7c416ccb…`、42 `857d8804…`、
43 `61fdac58…`），只有容器内嵌的代码身份不同；`semantics_equal: false` 与 tendency 迁移
（`outputs/r7_s3_tendency_input_parent_migrated_20261021_attempt02`，同样 `false`）是同一类已接受条件。

**⑤ R1 的桥 (e) 失败了，原因是设计而非工程。** R1 的桥比较的是**本轮 16,071-update 控制臂**与
**已登记 2,400-update FY 候选臂**的 5-lead RMSE seed 均值，5 个 lead 里 **4 个超带宽**
（12/24/48/72 h 分别 −0.144 / −0.185 / −0.254 / −0.233 K）。这不是身份错误，而是**算力不同**：
补了 6.7× 的 updates 之后控制臂本来就该更好。R1 的结果页已如实记录该失败**不能**用来解释 (d)。

## Decision

1. **R2 按计划 0026 §3 的队列开轮**，冻结常数：`updates = 16,071`、`batch_size = 2`
   （epochs floor：`16,071 × 2 = 32,142 ≥` 32,142 窗）；预算取**决策 0060** 的
   per-seed 129,600 s / planned 259,200 s / hard 518,400 s（R1 实测 88,362–89,520 s，留有余量）。
   `estimated_peak_bytes` 取 **R1 的实测 5,582,618,624 B + 48 桶表 13,790,400 B = 5,596,409,024 B**，
   不再用 FY 轮的 4 GiB 估计（它比同 store 同 compute 的实测**低估 1.2 GiB**）。
   若本轮首次实测与 0060 的常数冲突，按 0055/0060 先例**另立预算决策**，不在运行中改硬限。
2. **两臂都训练在 treatment 清单上**（32,142 窗）——本轮的因子是**反馈布尔**，不是清单。
   协议级断言：两臂记录的差集**恰为** `("anomaly_feedback", "model")`，且两个 spec 的差集
   **恰为** `["anomaly_feedback"]`、两臂 `anchor_output_prior` 都为 False、`climatology_anchor_spec`
   逐字相同。控制臂 N = `anomaly_feedback=False`，候选臂 F = `True`（`model/process_step_r7.py:108`）。
3. **父迁移**：用 `outputs/r7_s3_anchor_feedback_65x65_parent_migrated_20261101_attempt01/`
   （当前修订 `a38be16c…`），新钉模块 `scripts/r7_s3_anchor_feedback_parent.py`；
   `state_equality_evidence()` 逐 seed 断言源检查点与 `state_digest` 同 FY 钉一致。
   锚表由 `r7_s3_anchor_isolation_parent.expand_parent_state` 安装（该函数 fork RNG，
   因此模型构造不消耗外层 RNG 流）；`anchor_payload(run=…)` 增加**可选**路径参数，
   缺省仍是宽实例表（既有调用不变）。
4. **桥 (e) 重设计——在同算力下比身份，不在不同算力下比 RMSE。** R2 控制臂 N 的
   **第 1 个 update 的 `loss`** 必须与 **R1 已登记 treatment 臂**的第 1 个 update `loss`
   **逐位相同**（seed 41 `1.1786479949951172`、42 `1.4581815004348755`、43 `0.8535277843475342`）。
   依据：同 store、同清单、同 seed、同 batch、同父权重、同 LR 表；数据顺序由
   `torch.randperm(len(dataset), generator=torch.Generator().manual_seed(contract["seed"] + epoch))`
   决定（`training/r7_long_rollout_runner.py:278`），**不依赖**被 fork 的模型构造 RNG；
   N 的前向被锚轮实测为与无锚模型逐位相同。`gradient_norm` 一并记录但**不作逐位断言**
   （FY 轮两次 attempt 的同协议梯度范数在末位有差）。失败只记 `identity/engineering unresolved`，
   **不进入 (a)–(d)**，且须先定位再继续训练。
5. **恒等反证逐 seed 断言**（`expand_parent_state` 内建，失败即 `raise`）：N 的状态与无锚模型
   除两个 anchor buffer 外逐位相同；N 的前向 ≡ 无锚前向；反馈确被读取；表清零 ⇒ F ≡ N。
   另逐 seed 断言两臂 `anchor_table_sha256` 与 `expanded_state_sha256` 相同。
6. **预声明出口 (a)–(d)**（与 0058/短文一致，冻结后不放宽）：(a) F 的 72 h `full` 区域 t2m skill
   三 seed 全 >0；(b) F−N 的 72 h ΔRMSE 三 seed 全 <0；(c) F 的 48 h skill 三 seed 全 >0；
   (d) F−N 在 6/12/24 h 的 seed 均值 ΔRMSE ≤0。全真 → 只提请独立复核；任一假 →
   **永久关闭「气候态锚异常反馈」支**（不扫表粒度、不减锚、不加 seed、不调 updates、
   不与已关轴组合、不换名重开）。
7. **驱动拆分（R-051）**：`scripts/study_r7_s3_anchor_feedback_65x65.py`（584 行）+
   `scripts/r7_s3_anchor_feedback_support.py`（冻结文本、桥钉值、`arm_spec`/`anchor_block`/
   `require_single_factor`/`arm_name`/`check_model_revision`）+
   `scripts/r7_s3_anchor_feedback_parent.py`（父钉）。
   测试 `tests/test_r7_s3_anchor_feedback_65x65.py`（27 项）**不读 `outputs/`**：
   `frozen_constants(payload)` 把锚 payload 作为参数，合成表在测试内构造。

## Consequences

**变容易的：**
- R2 从「准入已定但 batch 留白」推进到「常数已冻、驱动与测试已落地」，队列不再空转。
- 桥 (e) 从「比不同算力的 RMSE」（R1 实测 4/5 超带宽、只能记 unresolved）变成「比同算力的
  第 1 个 update」，这是一个**能真的抓到身份错误**的检查。
- 峰值估计改用 R1 的实测，共驻门槛从 10,752 MiB 提到 11,993 MiB，不再低估 1.2 GiB。

**变难 / 代价（如实列出）：**
- **成本与 R1 同量级**：R1 实测 73.624654 GPU-h、三 seed 88,362–89,520 s；本轮加锚表
  （13.8 MB buffer）与一次扩状态，量级相同。这是**一条轴**的代价，队列还有 R3/R4/R5。
- **本轮跑在比 R1 新的模型修订上**（`a38be16c…` vs `60de3b2d…`）：R2 的读数与 R1 的读数
  **不能**在协议层逐位比较；桥 (e) 是唯一的跨轮身份见证，且只在第 1 个 update 上成立。
  `896eabf` 的「off 路径无差异」是在**一个合成 loss**上验证的，不是整条训练轨迹；
  若桥失败，必须停下来诊断，而不是继续跑完。
- **桥不通过**的后果被如实降级为 `identity/engineering unresolved`：它**不能**用来解释 (a)–(d)
  的任何一条，也不能反过来「因为桥过了就说结论可信」。
- **输出先验与 proposal 先验仍无法分离**（仓库规则把二者与「锚存在」耦死），本轮延续该限制。
- `per_seed_seconds` 在 `REUSE_PROTOCOL_FIELDS` 白名单内 ⇒ 预算常数一变，旧 attempt 的 seed
  **不能复用**，必须重跑（正确：曝光变了）。
- 与已关闭的宽实例锚隔离臂**不可直接比较**：实例、网格、监督域、数据跨度、锚表粒度、算力全变；
  旧臂的 F−N −0.321199 K 对本轮没有约束力，反之亦然。
- 一次三 seed 的一致性证据，**无显著性**；GPU 共驻，墙钟不是科学判据。
- **本 ADR 只覆盖到「驱动与测试已落地」**：协议未冻、GPU 未跑、四件套未登记。

**备选方案与否决理由：**
- **备选 A：继续用 FY 轮的父钉。** 否决：`load_checkpoint` 按设计拒绝外部实现的检查点；
  绕过它就是 weakening the guard。
- **备选 B：沿用 R1 的桥设计（与 2,400-update 登记臂比 5-lead RMSE）。** 否决：R1 自己的实测
  已证明它量的是**算力**而不是身份（4/5 超带宽）；重复一个已知会失败的设计不是诚实。
- **备选 C：两臂分别用 treatment / control 清单（像 R1 那样）。** 否决：那会把「反馈表征」与
  「训练数据量」两个因子混在一起，正是计划 0026 §1 反对的混淆形态。
- **备选 D：等一个不再变化的模型修订再跑 R2。** 否决：修订移动是常态，每次都等等于永不开轮；
  正确的处置是按先例重跑模型只迁移并留下 state 等价证据。
