# 0045 跨 attempt 的 pinned seed 复用（有限放宽「no in-place resume」）

- **日期**：2026-10-11
- **状态**：accepted

## Context

用户把「尽可能降低打断风险」列为本轮第一优先级（远程连接可能中断）。调研发现现有驱动**结构上无法
断点续跑**：

- 每个产物都用 `open("x")` 独占创建（`write_exclusive`），attempt 目录必须全新
  （`fresh_output` 拒绝已存在路径 → `FileExistsError`），父路径 `output.mkdir(exist_ok=False)`；
- `failure.json` 的 limitations 明文写着「Failed attempt preserved and charged in full;
  no automatic retry or in-place resume」；
- 一个 `seed` worker 进程内完成「训练 + 评分」，且 `--seed` 一次一个（`wide.SEEDS = (41, 42, 43)`）。

⇒ 一轮跑到第 3 个 seed 时被打断，前两个 seed 的成果**无法被下一次 attempt 复用**，三个 seed 全部重训，
GPU 时间成倍浪费。这正是打断风险里最贵的一环。

**已有的同类先例**：本项目已在**对照臂**上使用「pinned 复用」——例如某轮的单因素臂把「已登记的 2400
剂量臂」当作 pinned 对照复用（见 `docs/goals/s3-wide-long-lead-weight.md` 与对应证据页），复用前绑定
`protocol_sha256` / `data_identity` / 逐 seed 逐 lead 的表哈希。本决策把这个已被接受的机制**限定地**
推广到「同一轮自己的、已完成且身份完全一致的 seed」。

## Decision

1. **允许跨 attempt 的 pinned seed 复用**，且**只允许在 helper 侧实现**，不修改任何驱动的行为：
   驱动仍然只有 `--worker prepare|archive|seed|reading` 与「全新 attempt 目录」语义；复用由
   `tools/r7_round.py` 在调用 `stage seed<N>` 之前判断并落盘。
2. **复用的准入是逐字段全等**，任一不等即拒绝复用并照常重跑：
   `protocol_sha256`、归档 `code.zip` 的 SHA256、`data_identity`、`source_sha256`、seed 值、
   推理深度 K / `reasoning_steps`、`UPDATES`、物理权重向量、监督掩码（含 `mask_sha256`）、
   预算常量（`PER_SEED_SECONDS`/`PLANNED_SECONDS`/`HARD_CAP_SECONDS`）、arm 标识。
   判定输入**只取自上一 attempt 的 `protocol.json` 与该 seed 的 `seed<N>_receipt.json`**，
   不接受任何人工转述。
3. **复用的单位是一个 seed（原子）**，不是「一个 seed 的中途」：只复用**已完成**（receipt 存在且自洽）的
   seed。**不允许**复用半训的 checkpoint，也不允许在旧 attempt 目录里继续写。
4. **落盘方式**：把被复用 seed 的输出子树（`seed<N>/…` 与 `seed<N>_receipt.json`）复制进新 attempt 目录，
   并在新 attempt 的 `protocol.json` 旁记 `reused_seeds`（列出 seed 值、来源 attempt 目录、来源 receipt 的
   SHA256、逐字段比对结论）。随后 `reading` 阶段照常读取，无需改动。
5. **如实记账与如实报告**：
   - 被打断那段 GPU 时间**全额记账**（不因复用而减免）——复用省的是新 attempt 的 GPU 小时，不是历史成本；
   - 该轮证据页必须**逐 seed** 写明「重跑 / 复用」，复用条目要给出源 attempt 目录与 receipt 哈希；
   - index record 的 `limitations` 必须写明本轮存在复用及其准入依据。
6. **不放松的部分**：test 仍封存、`r`/alpha 纪律不变、判据不改、`scientific_claim: false` 不变；
   复用只减少重复计算，不改变任何科学门。

## Consequences

- **收益**：一轮被远程中断后，重跑只补未完成的 seed；最坏情况从「三个 seed 全重训」降到「只补一个」。
- **代价与风险**：① 复用判据本身是新的正确性表面——若某个身份字段漏比，会把不同配置的 seed 混进同一轮；
  故准入必须是**白名单全等**而非「只要 protocol 相等」，并要写反证测试（故意改一个字段必须拒绝复用）。
  ② 复用让「这一轮的三个 seed 是不是同一配置」不再能靠「同一目录」直观判断，必须靠 receipt 与证据页
  逐条核对；③ 若复用被滥用成「跨轮复用」，会破坏单因素归因——故本决策**仅限同一轮的 attempt 之间**，
  跨轮复用仍走「对照臂 pinned 复用」的既有路径与登记。
- **不改**：驱动的 CLI、`fresh_output` 的排他语义、`failure.json` 的失败保留与全额记账、既有已登记证据。
- **引用**：决策 0044（workflow 作执行者与打断抗性）、0030/0038（预算与记账）、0042（七 gate）；
  `docs/rules/reproducibility.md`（R-006）、`docs/rules/gpu-resources.md`（R-054）、
  `docs/plans/0020-campaign-driver-workflow.md`。
