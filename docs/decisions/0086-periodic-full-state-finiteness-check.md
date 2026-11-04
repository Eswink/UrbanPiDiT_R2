# 0086 — 同步守卫：每步全模型有限性检查降为周期检查 + 每 checkpoint 全检

- **日期**：2026-11-04
- **状态**：accepted
- **范围**：`training/r7_long_rollout_runner.py` 的 `_update`、`_run_updates`，新增
  `FULL_STATE_CHECK_EVERY = 100`
- **依据**：决策 0084、`docs/plans/0028-throughput-physics-campaign.md` §2 A3、
  `training/r7_autoregressive_runner.py:66`（`_finite_state` 的原始定义）

## Context

**① 原实现每 update 扫一次全模型。** `_update` 在 `optimizer.step()` 之后无条件调用
`_finite_state(model)`，后者对 `model.state_dict()` 的**每一个浮点张量**做
`torch.isfinite(value).all()` 再 `any(...)` 短路。R2 的臂有 **155 个张量**，
`any()` 遇到第一个 False 就停，但**全为有限时它必须把 155 个

`all()` 全部算完**——每个 `all()` 都是一次 GPU→CPU 的标量归约，因此这是
**每 update 一次全设备同步**，而不只是一次读。

**② 实测这一项不是当前的主要成本，但仍然是无谓的。** 本轮分段计时（batch 4、40 updates）：
`data 0.0137 / h2d 0.0035 / fwd+bwd 1.0977 / opt 0.0079 / sync 0.0`（总计 1.1228 s）。
把 A3 打开后整轮从 `1.1193 → 1.0947 s/update`（同 worker 数、同 batch），
即约 **2.2%**。另测：`_forward_final` 里的两处 `isfinite` **不是**瓶颈
（裸前向 `0.0475 s`，加两次全量 isfinite `0.0471 s`，差值在噪声内，**不动它**）。

**③ 真正的危险性换成了「延迟发现 NaN」。** 每步全检的价值是：一旦某步参数变成 NaN，
下一步立刻 raise，不会带着坏参数继续走。降为周期检查后，最坏情况是**晚 99 步**发现。
本轮把这个交易明写出来，而不是让它悄悄发生。

**④ 每 checkpoint 全检必须保留。** `_publish_checkpoint` 的契约是「不得把非有限状态当作
selected checkpoint 发布」，而它自己在发布前后不再做全扫；checkpoint 路径的正确性依赖
`_update` 里的全检。若只把检查降频而不在 checkpoint 步强制全检，坏权重就可以落盘。

## Decision

**1. `_finite_state` 从「每 update」改为「每 `FULL_STATE_CHECK_EVERY = 100` 步，以及
`update == total_updates`」，并在每个 checkpoint 步强制开启。**
实现：`_update(..., full_state_check=True)` 参数化；`_run_updates` 传入
`full_state_check=(update % FULL_STATE_CHECK_EVERY == 0 or update == total_updates)`。
checkpoint 步必然落在 `update % checkpoint_every == 0` 上；`checkpoint_every ≤ 100` 的
常见配置下它与周期检查重合，**更长 interval 时 checkpoint 步仍需自己的全检**——这一条
写进驱动侧断言（Stage B/C 的驱动在 `checkpoint_every` 上按 5,000 步预注册，见 0087）。

**2. `loss` 与 `gradient_norm` 的逐步检查保留，不减频。** 它们本来就被算成标量
（`float(result.loss.detach())`、`clip_grad_norm_(..., error_if_nonfinite=True)`），
是同一批同步的一部分，减它们不省时间却会丢掉「第几步开始坏」的定位能力。
**`clip_grad_norm_` 的 `error_if_nonfinite=True` 不变**——它才是 NaN 梯度的主守卫。

**3. 线程数与周期数都进 receipt。** 见 0084 负面后果 3：逐位复现必须固定线程数。

## Consequences

**负面后果（代价，必须如实列出）**

1. **最坏晚 `FULL_STATE_CHECK_EVERY − 1 = 99` 步才发现参数里的非有限值。** 这期间的
   参数已经参与过前向；如果 NaN 出现在某个参数上，后续 loss 通常**立刻**变成 NaN 并被
   逐步的 loss/grad-norm 检查抓到，所以真正漏网的窗口是「参数含非有限但仍未污染 loss」的
   退化情形。**这是明知的交易，不是等价替换。**
2. **`full_state_check=False` 的每一步都少一次全设备同步，这改变了 host 与 device 的
   重叠方式**；本轮实测它对总时间的贡献约 2.2%，与「有 155 个张量要扫」的直觉相符，
   但**没有**在 batch 8/14 上单独测过。大 batch 下它的相对占比只会更小。
3. **新增了一个可配置的周期常数**。它不进 `protocol.json` 的语义层（不改模型、不改损失、
   不改数据），但**会改 `training_code_sha256`**，因此每个新轮的 `training_code_sha256`
   都与历史轮不同——这是代码身份的自然结果，不是缺陷，但引用旧轮读数时必须注意。
4. **本 ADR 不声称 A3 带来吞吐突破**：它是 2.2%，而 A 的出口差 8.6%（1.0947 vs 1.0）。
   把 A3 说成「补上那 8.6%」是错的。

**正面后果**：去掉了一次每 update 的全设备同步；checkpoint 发布路径的全检**更严**了
（周期检查 + 端点检查是两条独立的触发，而不是一条）。

**否决**：完全删掉 `_finite_state`（checkpoint 契约会被破坏）；
只在 checkpoint 步检查（run 中途的 NaN 会长跑很久才被发现）；
把检查改成 `torch.isnan(...).any()` 只查 NaN 不查 Inf（会漏掉溢出）。
