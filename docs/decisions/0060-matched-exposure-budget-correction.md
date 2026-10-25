# 0060 — 匹配算力重估队列的预算更正：每 seed 43,200 s → 129,600 s（planned 259,200 / hard 518,400）

- **状态**：accepted
- **日期**：2026-10-26
- **范围**：`docs/goals/s3-matched-exposure-reeval.md` §4.2 的 Track 1 五轮（R1–R5）。
  沿用决策 0055 的先例：**只改时间，不动任何科学门**。

## Context

现行刻度来自决策 0055：`per_seed_seconds = 43,200 s`、`planned_seconds = 86,400 s`、
`hard_cap_seconds = 172,800 s`。那个刻度是按**年份覆盖轮**的规模标的——控制臂 7,291 窗、
候选臂 32,142 窗，但候选臂当时只跑 **2,400 updates**（`docs/R7_S3_FY_YEAR_COVERAGE_RESULT.md:117-119`
自己写明候选臂只看到约 **7%** 的 epoch）。

匹配算力重估的**新制度**（决策 0057/0058、目标长文 §1）要求**每臂 ≥1 epoch**：
`updates × batch ≥ 32,142`；`batch = 2`（用户已定）⇒ **updates = 16,071**。

成本估算（**外推，须以本轮首次实测替换**）：

- 训练：`plan 0025 §5` 的实测 `seconds_per_update` 在 batch 2 下为 **2.351 s** ⇒
  16,071 updates ≈ **37,783 s ≈ 10.50 h/臂**；两臂 ≈ **21.0 h/seed**。
- 评分：沿用 FY 轮 reading 阶段的量级，约 **0.9 h/seed**。
- ⇒ 每 seed 约 **22 h ≈ 79,000 s**，已经**远超** 43,200 s。

**这不是「放宽判据」**：(a)–(d) 出口、单因子约束、评分域、气候态分母、updates/LR/K/depth/dim、
seed、val 划分全部不动；改的只有**时间预算**。也**不**在运行中改硬限。

**机制点**：`per_seed_seconds` 与 `planned_seconds` / `hard_cap_seconds` 都在
`tools/r7_round_support.py:58-77` 的 `REUSE_PROTOCOL_FIELDS` 白名单内 ⇒ **改预算后旧 attempt 的
seed 不能复用、必须重跑**（决策 0045）。本队列是**新制度**、无旧 attempt 可复用，故不受影响。

## Decision

1. **新预算（Track 1 R1–R5 生效）**：`PER_SEED_SECONDS = 129,600 s`（36 h）、
   `PLANNED_SECONDS = 259,200 s`（72 h）、`HARD_CAP_SECONDS = 518,400 s`（144 h）。
   比例结构 1 : 2 : 4，与 0050/0053/0055 一致。
   129,600 s 是上面估计上界（79,000 s）的 **1.64×**，与 0055 的 1.8× 同量级。
2. **执行时按实测重定**：第一轮（R1）的 `prepare`/`seed` 阶段给出真实 `seconds_per_update` 后，
   若与 2.351 s 差得多，按同一纪律**另立一条更正决策**改刻度，**不在运行中改硬限**。
3. **不动科学门**：单因子约束、评分域、气候态分母、updates（16,071）、batch（2）、LR 2e-5、
   warmup 10、K=4/depth=4/dim=192、物理权重、seeds 41/42/43 全部不变。
4. **`estimated_peak_bytes` 沿用 4,096 MiB** + margin 512 MiB + neighbour reserve 6 GiB
   （batch 2 实测 `peak_reserved` 4,528 MiB，见 `plan 0025 §5`；R-054 的只读余量门不变）。

## Consequences

**变容易的：** 每臂跑满 ≥1 epoch 在墙钟上可行；预算不再成为「用不上多出的年份」的第二个混杂因子。

**变难 / 代价（如实列出）：**
- **单轮墙钟从 24 h 抬到 72 h（hard 144 h）**，队列五轮的墙钟显著变长；软预算超了继续并记
  `soft_overrun_seconds`，只有 hard 截止会截断 attempt（截断记 `budget_limited`/`failed`，
  **全额记账、不算科学通过**）。
- **`per_seed_seconds` 在白名单内** ⇒ 任何以旧预算跑出的 attempt，其 seed 在改预算后**不可复用**，
  必须重跑；这是一条容易被忽略的、会静默浪费 GPU 的机制点。
- **预算抬高本身不能让读数变好**，因此不构成对结果的偏置；但它会**掩盖**「本可以更早发现驱动错误」
  的问题——一个错的驱动现在能烧掉更久。缓解：R1 仍走 `round-guard` + 独立复核 + 最终复核三道门，
  且 stage 逐段跑、前一 stage 失败即停。
- 五轮共用同一刻度，但**各轮规模可能不同**（R3 吃算力不吃显存、R5 模型零改动）；若某轮实测明显偏离，
  按第 2 条另立更正决策，**不**顺手把别的轮一起改。

**备选方案与否决理由：**
- **维持 43,200 s**：新制度下每臂连 1 epoch 都跑不完 ⇒ 直接违反「每臂 ≥1 epoch」的使能条件，否决。
- **减 updates 到 2,400 以适配旧预算**：这正是决策 0057 判定「队列可执行」时要修的那个混杂
  （数据一半已满足、算力一半没满足），否决。
- **在运行中改硬限**：被 0055 的纪律与「不在运行中改硬限」明确禁止，否决。
