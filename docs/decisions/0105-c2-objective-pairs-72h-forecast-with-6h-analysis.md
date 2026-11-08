# 0105 — C2 的 on 臂目标函数**惩罚正确答案**：`matching_term` 跨 lead 配对，前置守卫自查自、恒为零

- **日期**：2026-11-08
- **状态**：accepted
- **范围**：`training/r7_derivative_matching.py::matching_term`（**冻结在跑**）、
  `scripts/r7_s3_derivative_matching_c2_protocol.py::_preflight_cases` 的 `zero_at_target` 守卫、
  `docs/decisions/0095` 的前置检验第 (1) 条。**本条 0 读数；attempt07 作废并全额记账 4.968110 GPU-h。**
- **依据**：决策 0095（C2 设计与前置）、决策 0104（同一类跨 lead 缺陷在 C1 的发现）、
  决策 0084 §6（族判别不成立**不得开轮**）、决策 0098（在跑轮次中缺陷只能作废不能补丁的先例）

## Context

**① 发现路径：为 attempt07 的读数阶段做前置核查时，用真实代码跑通了"约束项是否会进 loss 行"这条链，
顺手问了"这一项在真值处是否为零"。答案是否。**

`matching_term` 的 docstring 写：目标「是 batch 自己的 `atmos_target`，即**最后一个** physical step
的分析场」。实测不是：

```
forecasts shape           (1, 12, 17, 65, 65) -> N = 12   => forecasts[:,-1] 是第 12 步 = +72 h
physical_targets shape    (1, 12, 17, 65, 65)
atmos_target shape        (1, 17, 65, 65)
physical_targets[:,0] == atmos_target : True             => atmos_target 是 +6 h（第一个）
```

`training/r7_long_rollout_dataset.py:168` 写的是 `physical_targets = torch.stack(fields)`，
`fields[0] = sample["atmos_target"]`。所以 `atmos_target` 是 **+6 h**，不是最后一个。
`matching_term` 把 **+72 h 的预测**与 **+6 h 的分析场**配成一对。

**② 后果不是"项不为零"，而是"项在正确答案处为正、且梯度不为零"。** 用**冻结的**尺度
（`divergence 2.5665e-05`、`vorticity 3.5089e-05`）在**冻结的**四个 preflight case 上实测：

| case | `terms(target,target)`（守卫算的量） | **完美 +72h 预测 vs `atmos_target`(+6h)** | 同一场 vs 它自己的 +72h 真值 |
| ---: | ---: | ---: | ---: |
| 1000 | 0.0000000000 | **0.495737** | 0.0000000000 |
| 9000 | 0.0000000000 | **0.591632** | 0.0000000000 |
| 17000 | 0.0000000000 | **0.607382** | 0.0000000000 |
| 25000 | 0.0000000000 | **0.492830** | 0.0000000000 |

并且在 case 1000 上对预测求梯度：

```
C2 term at a PERFECT +72 h forecast : 0.495737   (if the optimum contained truth this is 0)
MSE at the same point               : 0.0000000000
|d(C2)/d(forecast)|                 : 0.011982   (nonzero)
```

⇒ **on 臂的优化器被推离正确的 +72 h 场。** 这不是"剂量太小"（C1 的失效模式）也不是"项恒零"，
而是**目标函数本身就错**。

**③ 为什么前置检验没拦住：那条守卫把 target 和自己比。**
`scripts/r7_s3_derivative_matching_c2_protocol.py:192`：

```python
zero = derivative_matching_terms(target, target, constants["channels"], ...)
```

两个参数都是 `target` ⇒ **恒等于零，与真实的配对无关**。它的 docstring 与 ADR 0095 都把这条
记成「`pred = target` 时精确零」，于是第 (1) 条前置**按构造通过**。这是本仓反复出现的同一类错误：
**「通过了检查」被当成「检查了该检查的东西」**（0103 是形状≠对象，本条是自查≠检验）。

同时注意 `:188-189` 两行**已经**把 `rollout.forecasts[:, -1]` 与 `batch["atmos_target"]` 配在一起
算 `live`，也就是说**跨 lead 配对在 preflight 里就是显式的**，只是没人把它和 `zero_at_target`
的那一行对着读。第 (2) 条「模型处为正」（4.27/3.57/3.60/3.52）**部分来自这个错配**，不全是真实失配。

**④ 为什么必须停而不是跑完再看。** attempt07 的预声明出口是：

- (a) 72 h skill 相对同代码 off 臂不退化（配对、三 seed 均值 ≥ 0）；
- **任一为假 ⇒ 永久关闭 derivative-matching 这一支。**

on 臂现在被一个**实现缺陷**推离正确答案，因此 (a) 若为假，关闭的理由会是 bug 而不是物理。
按 0084 §6 的取向（族判别不成立就不得开轮），**用一个已知有缺陷的目标函数去产生一个
"永久关闭"的结论，是把缺陷洗成科学结论**——这正是目标长文与 AGENTS.md 明令禁止的方向。
反之停下来的代价是已花的 GPU-h，而账本无总上限，只记账。

## Decision

**1. attempt07 作废（void），全额记账，零读数。** 按 0098 的口径：两臂各自 `elapsed_seconds`
之和 = **4.968110 GPU-h**（seed41/off 8,933.4 s + seed42/off 8,951.8 s = 17,885.2 s）。
不是"部分成功"，不得作为任何结论的证据。

**2. 终止动作只对本实验自己的进程，且逐个核对 `/proc/<pid>/cmdline` 含 `attempt07`。**
先 `SIGTERM` 编排器 `stage-seeds`（pid 2290400，先杀它是为了**阻止 seed43 被拉起**），
再对 10 个 worker 逐个发 `SIGTERM`；**未对任何非本实验进程发信号**（R-054）。
实测终止后两卡 **0% / 38 MiB / 17 MiB**，`stage-seeds` 与全部 worker 均已退出。

**3. 记录缺陷，但不在本条里改在跑的代码。** `matching_term` 的正确配对是
`physical_targets[:, -1]`（+72 h），与 `result.forecasts[:, -1]` 同为第 12 步；
修复需重新冻结身份（改 `model_code_sha256` 之外还要改冻结的 `execution_files_sha256`），
属于新 attempt 的 freeze，不是本条的补丁。

**4. 前置守卫必须改成"真值 vs 它自己的那个 lead"，并加反证测试。**
新守卫：`terms(physical_targets[:,-1], physical_targets[:,-1])` 应为 0（这仍然恒零，
所以**真正有判别力的是** `terms(truth_at_lead_L, other_lead_truth) > 0` 这一条反例，
以及 `terms(forecast, paired_truth)` 在 `forecast == paired_truth` 时必须为 0）。
**只加恒零守卫是没用的**，本条明确要求加一条会**失败**的反证。

## Consequences

**变容易的：** 在花掉约 35 GPU-h 之前就发现 on 臂目标函数会惩罚正确答案，并且现在有一条
可复算的、四项独立测量的证据（lead 结构、冻结尺度下的四项值、梯度范数、守卫的自查形状）。
C2 若要重跑，正确配对与有判别力的守卫都有明确规格，不必再靠读 docstring。

**变难 / 代价（如实列出）：**

1. **我烧掉 4.968110 GPU-h 才做这个检查，而不是在 freeze 之前做。** 这条缺陷完全可以在
   0095 的前置里查到——preflight 的 `:188-189` 那两行当时就摆在那里。**我的 freeze 流程有缺口：
   前置只复核了它自己报告的三条结论，没有独立地问"这一项在真值处是否为零（用部署的配对）"。**
2. **attempt07 的 4.968 GPU-h 完全无产出**，且这是本目标的**第三次**作废轮
   （attempt04 6.5555、attempt06 23.684603、attempt07 4.968110），累计作废 **35.208213 GPU-h**。
   前两次是读数阶段缺陷与我自己 `mv` 造成的，这次是目标函数缺陷——三种不同成因，但都指向
   **前置核查不够硬**这一条。
3. **`atmos_target` 语义的混淆已经咬了两次。** 0104 在 C1 发现同一件事（`atmos_target` 是 +6 h，
   而代码按"最后一个 6 h 步"理解）。两处独立代码、同一个假设、同一种错误。
   **根因是这个名字本身**：它叫 `atmos_target` 而不叫 `atmos_target_6h`。本条**没有**重命名它，
   因为那会改数据契约与大量冻结哈希；代价是**第三次同类错误仍然可能发生**。
4. **修好后重跑的价格已经变了。** 现在知道 on 臂在旧配对下是有害的，因此**不能**把 attempt07
   的任何部分当作"接近正确"的基线；新 attempt 从零开始，约 2×32,132×1.13 s ≈ 21.8 h/seed，
   两卡并行三 seed 约 44 h。
5. **本条零读数、零科学结论。** 不得据此声称导数域匹配有效或无效——**它是没有被测过**，
   不是"测了失败"。

**否决：**

- **让它跑完再判 (a)。** 否决：任何 false 都会永久关闭该支，而我们有四条实测证明 on 臂被
  推离正确答案；用缺陷产生的 false 关闭分支，等于把 bug 写成物理结论。
- **在跑的字节上打补丁。** 否决：`matching_term` 已冻结进 `execution_files_sha256`（95 项），
  改它会让在跑身份失效；0098 已立下"作废而不是补丁"的先例。
- **只把 `zero_at_target` 改成 `terms(truth, truth)`。** 否决：那**仍然是恒零**，
  等于把一个无判别力的检查换成另一个无判别力的检查。必须加跨 lead 反例。
- **改名为 `atmos_target_6h` 一劳永逸。** 否决：会改数据契约与大量冻结哈希；记进「未决」，
  由后续决策单独处理，而不是夹在一个作废轮的 ADR 里顺手做。
- **用 attempt07 的 off 臂当新 attempt 的基线。** 否决：off 臂虽然没启用该轴（`_off` 路径
  返回 `None`，逐位不变），但 identity 绑定整份 protocol，且没有完整 seed receipt；
  按 0098 的口径作废轮的东西不复用。

## 未决

- `atmos_target` 是否应重命名为带 lead 的名字（如 `atmos_target_6h`），以及那会牵连多少冻结哈希。
- C2 修正后是否值得重开：本条只证明**旧实现无意义**，不证明修好后 C2 会赢。
  重开须新 ADR 论证族判别（这次要用**部署配对**的 `zero_at_true` 作证据）并重新 freeze。
- 前置核查流程本身需要一个机械化的"用部署配对验证目标函数在真值处为零"的步骤，
  目前它依赖执行者记得问——三次作废轮说明这不够。
