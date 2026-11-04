# 0087 — checkpoint 策略：每 5,000 步 + 端点 + val 最优，预注册保留清单，只对新 run 生效

- **日期**：2026-11-04
- **状态**：accepted
- **范围**：本 campaign 新驱动（Stage B/C/E）的 `checkpoint_every` 与保留策略；
  **不改**已冻结轮次的协议，**不删**任何已存在的 checkpoint
- **依据**：决策 0084、`docs/plans/0028-throughput-physics-campaign.md` §2 A4 与 §4 风险 3、
  决策 0068（「剪 checkpoint 致 92 处审计引用悬空」的先例）

## Context

**① 原策略的成本是实测的。** 现行 `checkpoint_every = 20`（`study_r7_s3_wide_single_factor.py:40`），
16,071 updates 的臂因此写出 **804 个 checkpoint**，每臂 32.2/43.4 GiB；R1 的六臂一轮合计
**227 GiB**。每个 checkpoint 还带一次 `save_exclusive` + `load_checkpoint` 回读校验
（计划 0028 §0.4 记 `torch.load` 0.70–0.80 s）。

**② 磁盘预算是硬约束。** `/data` 当前余量见本轮证据页；按计划 0028 的四条硬约束第 4 条，
100 epoch 不稀疏会写出 **~1.3 TB**。按 5,000 步稀疏后每臂 **<8 GiB**。

**③ 0068 的教训是「删文件」而不是「少写文件」。** 0068 记录的是把**已存在**的中间 checkpoint
剪掉后，4,178 份 JSON 里 **92 处引用指向已删文件**（`r7_74_autoregressive_20261003_attempt01` 74 处、
`r7_73_process_supervision` 18 处）。因此本 ADR 的处置是**只对未来的 run 生效**，
对**任何已存在的 checkpoint 一个字节都不动**。

**④ 选点规则必须预注册，否则「val 最优」是一个事后自由度。** 现行 runner 的
`selection_metric` 是 `"frozen endpoint; no validation selection"`（`training/r7_long_rollout_runner.py`
的 report），也就是**不做 val 选点**。本 ADR 不擅自把它改成 val 选点——那会改变读数的含义。

## Decision

**1. `checkpoint_every` 从 20 提高到 5,000，并额外保留端点与 val 最优点。**
本 campaign 的新驱动按 `(5000 步的整数倍) ∪ {total_updates} ∪ {val 最优}` 保留。
16,071 updates 的臂因此保留 **4 个**（5,000 / 10,000 / 15,000 / 16,071）而不是 804 个。

**2. 选点规则预注册为：`selected_checkpoint = 端点`（与现行 runner 一致），
`selection_split = None`、`selection_metric = "frozen endpoint; no validation selection"`。**
**不启用 val 选点**——启用它会让「selected」变成一个由 val 决定的量，而 Stage B 的曲线
（skill 随 updates）正是要观察每个端点，端点被 val 选掉就不再是那条曲线。
若某个 Stage 确实要 val 选点，必须在该 Stage 的 brief 里另行预声明，不得沿用本 ADR。

**3. 保留清单进 protocol（预注册），不做事后清理。**
`protocol.json` 的 `retained_checkpoints` 明写将产生的文件路径集合；run 结束时由驱动断言
实际产生的集合**等于**预注册集合。**任何删除动作都不在本 ADR 授权范围内**。

**4. 与 A3 的交互：checkpoint 步必须做全模型有限性检查（见 0086 §Decision 1）。**
`checkpoint_every = 5000 > FULL_STATE_CHECK_EVERY = 100`，因此 checkpoint 步的
`update % 100 == 0` 可能为假（5,000 是 100 的倍数；但 16,071 不是），
驱动侧必须**显式**在 checkpoint 步强制全检，而不是依赖周期检查恰好命中。

## Consequences

**负面后果（代价，必须如实列出）**

1. **run 中途的细粒度轨迹消失。** 每 20 步一个 checkpoint 曾允许「从任意点重放并比对
   参数」；5,000 步后只能在 4 个点做。**`progress.jsonl`（每 20 update 一行）不受影响**，
   所以 loss 曲线仍在，消失的是**权重**轨迹。
2. **A3 的 NaN 窗口与 A4 的稀疏 checkpoint 叠加**：最坏情况下坏参数可能持续 99 步且
   距离下一个 checkpoint 很远。缓解是逐步的 loss / grad-norm 检查（0086 §Decision 2），
   但它查的是标量不是权重。
3. **本 ADR 对既有轮次零效力**。若将来要清理历史 checkpoint，仍需一次独立授权
   （0068 的口径：按存在性全量扫描 4,178 份 JSON 证明零悬空才放行）。
4. **`training_code_sha256` 会因本改动而变化**，见 0086 负面后果 3。
5. **不适用于已冻结协议**：`study_r7_s3_wide_single_factor.py:40` 的 `CHECKPOINT_EVERY = 20`
   **保持原值**，因为 R1–R4 的协议已冻结且其读数已登记。本 ADR 只约束新驱动。

**正面后果**：每臂 checkpoint 从 804 个降到 4 个，按 40.97/55.11 MiB 计每臂从
32.2/43.4 GiB 降到约 **0.16–0.22 GiB**；100 epoch 的「~1.3 TB」风险随之消失。
同时省掉每次 checkpoint 的 `torch.load` 回读（0.70–0.80 s × 减少的次数）。

**否决**：直接删历史 checkpoint（0068 已证会断审计链）；用 `save_last` 只留一个
（端点 + 中途点同时丢失，无法核对「跑到哪一步坏了」）；启用 val 选点
（会改变 Stage B 曲线的含义，见 §Decision 2）。
