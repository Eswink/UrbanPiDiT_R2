# 0091 — 短批次跨更新边界：`_BatchLoader` 的一个真实缺陷，作废第三点 attempt01

- **日期**：2026-11-05
- **状态**：accepted
- **范围**：`training/r7_long_rollout_runner.py`（`_index_stream` / `_draw_sizes` / `_BatchLoader`）、
  `tests/test_r7_stage_a_read_path.py`
- **依据**：决策 0085（A1/A2 读法与 worker prefetch，本缺陷由它引入）、
  0090（第三点的包络与并列实例）、`docs/goals/s3-throughput-and-physics-campaign.md` §3、
  `docs/plans/0028-throughput-physics-campaign.md`

## Context

**① 事实：第三点 attempt01 在 32,120/32,132 处死于 `StopIteration`，10.063 GPU-h 全额记账、无读数。**

`failed_attempt.json` 记 `failure_reason: "StopIteration: "`、`progress_updates: 1606`、
`resume_permitted: false`。产物（7 个 checkpoint、progress 旁路）保留不删，**不参与任何读数**。

**② 根因不是"跑到最后没数据了"这么简单，而是一个静默的训练完整性缺陷。**

`_BatchLoader` 把**扁平**的索引流交给 `DataLoader(batch_size=contract["batch_size"])`，
并在 docstring 里断言「Batch boundaries therefore coincide with update boundaries by
construction」。**那句话是假的**，而它从未被测试。

`_index_stream` 按 `randperm(seed + epoch)` 逐 epoch 走游标，每个 epoch 的**最后一次抽样是短的**：
32,131 个可用窗、batch 4 ⇒ 每个 epoch 末次只抽 **3** 个样本。于是：

- 扁平流的长度**不是** batch 的整数倍；
- `DataLoader` 按 4 重新切块，**跨过更新边界**；
- 第 8,033 次更新请求 `[9623, 6990, 20742]`（3 个），**实际被喂** `[9623, 6990, 20742, 31158]`
  （4 个，多吃了下一次更新的第一个样本）；
- 此后**每一次**更新的批次都与登记抽样错位，直到流被耗尽。

**实测规模（复算，非估计）**：32,132 updates 下**第 8,033 次起错位**，
**24,099 次更新**被喂了与登记抽样不同的样本组合，**最后一次更新从未执行**。

**③ 旧守卫为什么没拦住。** `batch()` 用 `expected = self.stream[self.consumed:self.consumed + len(indices)]`
比对，然后 `self.consumed += len(indices)`。当 DataLoader 多吃一个样本时，`consumed` 只按调用方
声明的长度前进，于是**下一次比对仍然从"正确"的位置开始**——错位被这个偏移量本身掩盖了。
守卫检查了"索引对不对"，没检查"我实际拿到几个"。

**④ 第一、二点为什么是干净的（可核查的算术，不是运气说法）。**

| 点 | updates | 短抽样出现位置 | 边界是否对齐 |
| --- | ---: | --- | --- |
| p0025 | 2,008 | 无（8,032 = 2,008×4） | **是** |
| p1000 | 8,033 | **仅第 8,033 次**（最后一次） | **是** |
| p4000 | 32,132 | 8,033 / 16,066 / 24,099 / 32,132 | **否** |
| p1600 | 128,528 | 8,033 起每 8,033 次一次 | **否** |

第一点没有任何短抽样；第二点的唯一短抽样落在**最后一次更新**上，此后没有下一次更新去错位。
因此**旧代码与修复后代码在第一、二点上产生逐次相同的批次序列**，它们的读数**不受影响**、
不需要重跑。这一点由上面的位置表可独立复算，不依赖我的判断。

**⑤ 这个缺陷是决策 0085 引入的，不是历史遗留。** Stage A 为了让 worker 池不拥有训练顺序，
把索引流物化并交给 DataLoader；当时的逐位验收（前 200 步 loss 逐位相同）**跑不到第 8,033 步**，
所以它通过了验收却没有暴露这个缺陷。这是验收窗口与被测性质不匹配的一个实例，如实记下。

## Decision

**1. 一个抽样一次批次：`_draw_sizes` 显式给出每次抽样的长度，`_BatchLoader` 用 `batch_sampler`。**
`DataLoader(batch_sampler=self.draws)` 把每个 draw 当作**一个批次**交给数据集，
短抽样**保持短**，不会吸收下一次更新的样本。这从结构上消除了"扁平流被重新切块"的可能，
而不是把 batch 数凑成整数。

**2. `batch()` 改为按**登记的 draw** 比对，并且同时比长度。**
`self.consumed` 现在按 **draw** 前进（每次 +1），期望值取自 `self.draws[self.consumed]`。
超出登记计划再取批次直接 `raise`。旧守卫的偏移量掩盖机制因此不再存在。

**3. 第三点 attempt01 作废，从头重跑（`attempt02`）。**
不续跑（训练器 `resume_permitted: false`），不把 32,120 步的 checkpoint 当端点，
不因"只差 12 步"而放宽 (a)。**10.063 GPU-h 全额记账**。

**4. 第一、二点不重跑。** 依据 Context ④ 的算术：它们的批次序列与修复后代码逐次相同。
**但它们的 `training_code_sha256 cdb89bca…` 是含此缺陷的旧代码**，引用时必须连这条一起读。
新代码的 `training_code_sha256` 是 `af2a5adc…`，第三、四点将在其下运行。

**5. 饱和判定的两支不受影响。** 冻结规则比较 **4 epoch 与 16 epoch**，两者都将在
`af2a5adc…` 下重跑；第一、二点只是曲线的额外纵坐标，不进入该一阶差分。

## Consequences

**负面后果（代价，必须如实列出）**

1. **10.063 GPU-h 被一个静默的批次错位烧掉，且没有产生任何读数。** 这是本轮最贵的一次失败。
2. **第一、二点与第三、四点的 `training_code_sha256` 不同。** 曲线因此**不是**单一代码身份下的
   四条臂。依据 Context ④，行为上等价，但身份上不等价——引用整条曲线时必须写明这一点，
   不得声称「四点同代码」。
3. **验收窗口与被测性质不匹配这件事没有被自动修复。** Stage A 的"前 200 步逐位相同"
   仍然跑不到 8,033 步；新加的测试用**7 窗 / batch 3 / 5 updates** 的微型 fixture 让短抽样
   落在**运行中间**，这是结构性的，不依赖跑得够久。
4. **修复改变了 `training_code_sha256`**（`cdb89bca…` → `af2a5adc…`），因此第三点的
   `protocol.json` 必须重新冻结；旧 attempt01 的协议不可复用。
5. **`batch_sampler` 改变了 worker 侧的批次构造路径。** 逐位等价由 Stage A 那套
   「worker 批次 == 串行批次」测试继续钉住（该测试在修复后仍通过）。

**正面后果**：缺陷从"静默错位 24,099 次更新"变成"结构上不可能发生"；
`batch()` 的守卫现在能真正发现长度不符；回归测试不依赖长跑；
第一、二点的有效性由可复算的算术支撑，而不是由「大概没事」支撑。

**否决**：把 batch 数凑成 32,132 的整数倍（会改数据清单与 `window_sha256`，等于换实验）；
丢弃最后那个 3 样本的短抽样（会静默改变每个 epoch 的样本数）；
把第三点 attempt01 的 32,120 步 checkpoint 当端点（违反预声明 (a)，且模型见过错位批次）；
为「只差 12 步」放宽 (a)（事后放宽冻结判据，明令禁止）；
重跑第一、二点以统一代码身份（行为等价，纯属浪费 4.7 GPU-h）。
