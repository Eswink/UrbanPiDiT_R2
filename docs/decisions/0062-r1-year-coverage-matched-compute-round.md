# 0062 — R1 的执行设计：匹配算力重估轮的冻结常数、本轮内重跑的配对控制臂与驱动拆分

- **日期**：2026-10-26
- **状态**：accepted
- **范围**：`docs/goals/s3-matched-exposure-reeval.md` §4.2 队列第 1 轮（R1）的**执行设计**；
  R1 的**开轮授权**不在这里，见决策 0061；预算常数见决策 0060。
- **依据**：`docs/decisions/0060-matched-exposure-budget-correction.md`、
  `docs/decisions/0061-r1-year-coverage-override.md`、
  `docs/decisions/0057-supersede-0056-queue-is-executable.md`、
  `docs/goals/s3-year-coverage.md:68-79`（逐字沿用的预声明出口）、
  `docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md`（关闭读数）、
  `docs/decisions/0045-pinned-seed-reuse-across-attempts.md`（复用白名单）

## Context

**① 本 ADR 不重新授权。** R1 为什么可以开、override 覆盖了哪一条冻结条款、负面后果是什么、
先例被限定在哪一条轴上 —— 全部由 **决策 0061** 决定，本 ADR **不重复也不扩大**它。
本 ADR 只记录 0061 授权之后**这一轮具体怎么跑**，以及其中两处 0060/0061 没写到的执行决定。

**② 本轮的因子与算力（0060/0061 已定，这里只固化）。** 唯一因子 = 训练清单覆盖的年份
（2017–2021 五年 → 2000–2021 二十二年），月份固定 12 个月。相对已关闭那一轮**只改算力**：
`updates` 2,400 → **16,071**、`batch` 1 → **2**（`16,071 × 2 = 32,142 ≥` 候选臂 32,142 窗 ⇒ 每臂 ≥1 epoch）。
store、网格（`full_grid` 4,225 点）、监督域、数据跨度（2000–2021 train / 2022 val）、
2000–2021 train-only 归一化、48 桶 train-only 气候态分母、迁移父、spec、seeds、LR/warmup/clip/
物理权重、2022 全年 val 全部**逐字不变**。

**③ 第一处 0060/0061 没写到的执行决定：配对控制臂必须在本轮内重跑。**
已登记的同因子 5 年臂是**年份覆盖轮的控制臂**，它用的是那一轮自己的归一化与气候态分母
（该轮的 limitation 明写「两臂共用 2000–2021 归一化，因此 2017–2021 控制臂与月份轮的候选臂
不逐字节可比」）。本轮的配对控制臂必须与候选臂共享**本轮**的归一化、气候态分母与算力，
否则 F−N 会同时携带归一化/分母/算力三个差异 —— 那就不是单因素轮了。
另外，决策 0045 的复用白名单含 `per_seed_seconds` / `planned_seconds` / `hard_cap_seconds`，
而 0060 已把这三个值全部改掉 ⇒ 旧 attempt 的 seed **按规则不可复用**，必须重跑。
⇒ `round-spec` 的 `control_record` 指向 `s3-fy-annual-coverage-result`，
但它**只作桥 (e) 的身份/工程参照**，**不作配对控制**。

**④ 第二处：驱动不得复用已关闭那一轮的模块常量。** 上一轮的驱动
（`scripts/study_r7_s3_fy_year_coverage.py`）的 `UPDATES` / 预算常数是模块级常量，被它自己的
`_readings` 等函数直接引用。若新驱动 import 它来「复用评分管路」，读数会静默写成 2,400 updates
的旧值。⇒ 新驱动自包含，逐字复制上一轮除算力与预算外的全部常量。

**⑤ 规模约束（R-051 / R-052）。** 上一轮的驱动恰好 600 行（硬上限）；本轮在它之上只加
batch 参数化与匹配算力的守卫，仍会越线 ⇒ 按仓内既有做法把**纯数据**（预声明判定文本、
limitations、桥表）拆到 `scripts/r7_s3_fy_year_coverage_matched_support.py`。

## Decision

1. **本轮的机器可读 spec 写在 `docs/goals/s3-matched-exposure-reeval.md` 顶部的
   `round-spec` 块**，`campaign-state.current_round_goal` 指向该文件；
   R1 的人读 brief 是 `docs/goals/s3-year-coverage-matched-reeval.md`。
   `planned_seconds = 259,200`、`hard_cap_seconds = 518,400`（取决策 0060 的值）。
   两个提交路径列表分开：`commit_paths` = 驱动 / support / 测试 / 本轮 brief / 本长文 /
   master 长文 / 本 ADR / 决策索引 / `size-thresholds.md`；
   `register_paths` = 证据页 / 证据索引 / candidate brief / master 长文 / 本长文。
   **证据页在实现提交那一刻还不存在**，因此它只出现在 `register_paths`。
2. **两臂的协议差异必须恰好等于**
   `{train_manifest, train_data_identity, train_windows}`；`require_single_factor` 另外断言
   **两臂的 `batch_size` 都等于本轮的 2**（两臂一起退回 batch 1 会互相一致却仍是错的轮次）。
3. **配对控制臂在本轮内重跑**（同 store / 归一化 / 气候态 / 算力 / seed / val）；
   已登记的 FY 候选臂只进桥 (e)，桥的失败只记 `identity/engineering unresolved`。
4. **预声明出口 (a)–(d) 逐字沿用 `docs/goals/s3-year-coverage.md:68-79`**，不放宽、不新增阈值；
   桥 (e) 的带宽仍是「seed 极差或均值的 5%，取较大者」。
   `readings.json` 必须同时记录 `updates`、`batch_size` 与 `closed_round_updates`，
   否则本轮读数无法与已关闭那一轮并列阅读。
5. **驱动拆分**：`scripts/study_r7_s3_fy_year_coverage_matched.py` 自包含（≤600 行，R-051），
   纯数据拆到 `scripts/r7_s3_fy_year_coverage_matched_support.py`；两者都进执行文件清单
   （`execution_files`），因此都进协议的身份哈希。
6. **不动科学门**：评分域、气候态分母、updates（16,071）、batch（2）、LR、warmup、K/depth/dim、
   物理权重、seeds 41/42/43、val 划分全部按 0060/0061 不变。

## Consequences

**变容易的：**

- R1 有了可执行的、逐字对齐 0060/0061 的 spec：工作流只从 `campaign-state.current_round_goal`
   读一次，就能拿到 driver / output_stem / seeds / 两个提交路径列表，argv 保持编译期字面量。
- 「配对控制臂必须本轮重跑」被写成一条可机械检查的断言（两臂 batch 相同、差异恰好三个键），
   而不是一条注释：以后任何一轮若想拿旧臂当配对控制，测试会先失败。
- 驱动自包含这一条把「import 旧模块 ⇒ 静默继承 2,400 updates」这个**会污染读数**的坑封住了。

**变难 / 代价（如实列出）：**

- **本轮的钱全部花在「把算力补上」**：两臂 × 3 seed 约 68 GPU-h（batch 2 外推，下界），
  而它**不保证翻盘**；若 (d) 仍假，买到的只是一个**更强的负结果**（算力已匹配）。这是 0061
  已经写明的负面后果，本 ADR 只是把它落到预算上。
- **代码重复是刻意的**：新驱动与已关闭那一轮的驱动在「不该变」的部分逐字重复，两者会各自漂移。
  缓解：本 ADR 第 4/5 条把「唯一允许的差异」写死（算力 + 预算），并由测试断言
  `changed_from_closed_round == ["updates", "batch_size"]`。代价是同一份评分管路现在有两份拷贝，
  以后修一处缺陷要记得修两处 —— **本轮不改已关闭那一轮的驱动，因为它的协议必须保持可读**。
- **`round-spec` 与「每轮 brief」分居两个文件**：本轮的 spec 在目标长文顶部，人读 brief 另存一份。
  代价是读者要知道「spec 不在 brief 里」；缓解是本文件与 brief 各自逐字写明这一分工。
- **执行文件清单新增一个 support 模块** ⇒ 协议身份哈希变长、任何对 support 的改动都会让
  `validate_protocol` 报「execution source changed」；这是刻意的（纯数据也是身份的一部分）。
- **本轮仍未解决 0061 的纪律代价**：预声明出口一旦被具名授权重开，其「不可翻案」的可预期性
  就已经下降；本 ADR 不能修复这一点，只能把先例范围继续限定在年份覆盖轴。
- 一次三 seed 的一致性证据，**无显著性**；GPU 共驻，墙钟不是科学判据。
- **本 ADR 只写到「实现提交之前」**：`freeze`、GPU 阶段、读数与四件套登记都还没做。

**备选方案与否决理由：**

- **备选 A：复用已登记的同因子 5 年臂当配对控制（不重跑）。** 否决：它的归一化与气候态分母
  是另一套，F−N 会同时携带三个差异；且 0060 改了 `per_seed_seconds`，按 0045 的复用白名单
  该臂的 seed **本来就不可复用**。这会把单因素轮变成三因素轮。
- **备选 B：让新驱动 import 已关闭那一轮的驱动来复用评分管路。** 否决：旧驱动的 `UPDATES` /
  预算是模块级常量，被其 `_readings` 直接引用 ⇒ 新读数会静默写成 2,400 updates，
  正是本轮要消除的混杂；而且那样 `readings.json` 会**自证错误**。
- **备选 C：改已关闭那一轮的驱动，把它参数化后两轮共用。** 否决：已关闭轮的协议与代码是
  冻结证据（`code_zip_sha256` 已登记），改它等于改历史；仓规禁止改写历史证据。
- **备选 D：把新驱动压进 600 行、不拆 support 模块。** 否决：纯数据（limitations + 判定文本 + 桥表）
  占约 36 行，压进去只能靠删掉限制声明或把常量挤成超长行；前者削弱如实登记，后者把 R-019b 推高。
- **备选 E：把 `round-spec` 放进人读 brief、目标长文不带 spec。** 否决：本轮的任务书明确要求
  spec 写在目标长文顶部并由 `campaign-state.current_round_goal` 指向它；且同一文件里只能有一个
  `round-spec` 块（`round_spec()` 会拒绝多块），不能两处都放。
- **备选 F：本轮顺手把 batch 也做成一个待定项（像 0058 那样把 batch 留给用户）。** 否决：
  用户已定 `batch = 2`（决策 0060 第 3 条），且 `updates × batch ≥ 32,142` 已把两者绑死。
