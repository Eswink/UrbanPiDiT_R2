# 0109 — 导数域目标匹配在 4 epoch 父模型上三 seed 一致退化，按预声明永久关闭该支

- **日期**：2026-11-10
- **状态**：accepted
- **范围**：`scripts/study_r7_s3_derivative_matching_c2.py`、`scripts/r7_s3_derivative_matching_c2_protocol.py`、
  `training/r7_derivative_matching.py`、`training/r7_objective_axes.py`、`training/r7_long_rollout_runner.py`；
  attempt08 的读数与登记。**本条 66.993549 GPU-h、三 seed 两臂（六个臂）、有读数。**
- **依据**：决策 0095（C2 的族判别）、0084 §6（新方向的族判别要求）、0105/0106/0107（目标函数缺陷与修正、
  重开决定）、`docs/R7_S3_DERIVATIVE_MATCHING_C2_RESULT.md`（结果页，本 ADR 的 README 行）

## Context

**① 这是 C2 的第四次开轮，前三次都作废，共 35.208213 GPU-h、零读数。**
attempt04 因读数阶段三处缺陷（0098）、attempt06 因我在跑动中 `mv` 了产物目录、
attempt07 因 on 臂目标函数**惩罚正确答案**（0105：`matching_term` 把 `forecasts[:,-1]`（+72 h）
配给 `moved["atmos_target"]`（+6 h），完美 +72 h 预测被罚 0.4957–0.6074，梯度范数 0.011982）。
0106 改为 `physical_targets[:,-1]` 并让缺陷在字段缺失时 raise 而非兜底；0107 据此重开。

**② 前置守卫现在有判别力。** 0105 的教训是守卫"按构造通过"——原 `zero_at_target` 用
`derivative_matching_terms(target, target, ...)`，把 target 与**自己**比，恒零、与部署配对无关。
attempt08 冻结的 protocol 同时记录两条，实测：

| 项 | 四例实测 |
| --- | --- |
| `zero_at_target`（预测 = 真值） | `0.0 / 0.0 / 0.0 / 0.0` |
| `nonzero_for_cross_lead_field`（+6 h 分析场对 +72 h 目标） | `0.495737 / 0.591632 / 0.607382 / 0.492830`，全部 > 0 |
| dose | `0.365767 / 0.410860 / 0.392699 / 0.288887`（带 [0.1, 10] 内） |

**只有两条一起**才说明守卫有判别力：前一条单独仍可能是恒零。

**③ 单因子设计与身份。** `protocol_sha256 = a3ca9fe7b36bb9e6…`、`code_commit = 2e0aac88…`、
`model_code_sha256 = 59340710…`。三 seed（41/42/43）、每臂 **32,132 updates（4 epoch）**。
两臂**只有 `derivative_matching` 一个键不同**（`off: None` / `on: 'c2'`），
同 seed 两臂共享同一父 checkpoint（父 1600 端点，`state_digest` 逐 seed 在
`r7_s3_position_retention_parent.MIGRATED_PARENT_STATE_DIGEST` 里由 worker 启动时断言）。
尺度 `divergence 2.5665e-05`、`vorticity 3.5089e-05 s^-1`，权重 `0.1 / 0.1`。

**④ 预声明出口在开轮前冻结，本轮一字未改。** (a) 72 h t2m skill 相对同代码 off 臂的配对均值 ≥ 0；
(b) on 臂逐步记录两项匹配且严格为正；(c) off 路径逐位等价；(d) 无判据被改。
**任一假即永久关闭该支**，不调权重、不换项、不加 seed。

## Decision

**按预声明关闭「导数域目标匹配」这一支。** 读数由驱动自算（`readings.json`，未手改）：

| 判据 | 结果 |
| --- | --- |
| (a) 72 h 配对均值 ≥ 0 | **假** |
| (b) 两项匹配逐步记录且为正 | **真** |
| (c) off 路径逐位等价 | 真（**恒真常量，不携带信息**） |
| (d) 无判据被改 | 真（**只读两个本页自写字段，不携带信息**） |
| `all` | **假** |
| verdict | `permanently-close-derivative-matching-branch` |

**三 seed 的 72 h t2m skill 与配对差**：

| seed | off | on | Δ |
| --- | --- | --- | --- |
| 41 | +0.230542 | +0.218909 | **−0.011634** |
| 42 | +0.243262 | +0.226920 | **−0.016342** |
| 43 | +0.231662 | +0.218282 | **−0.013380** |
| 均值 | +0.235155 | +0.221370 | **−0.013785** |

**关闭的依据是方向一致，不是幅度大。** 已测 **15 个 lead×seed 配对差全部为负**
（3 seed × 5 lead），最"不坏"的是 −0.002249，**一个正值都没有**。幅度上两个 seed 的 72 h Δ
分别是控制臂极差（0.012720）的 **0.91×** 与 **1.28×**，seed43 为 **1.05×**——同量级。
噪声不会让五档在三个 seed 上全部同号，但也不能说这是"大幅退化"。

**(b) 为真说明约束确实在起作用，这与 (a) 为假并不矛盾**：on 臂三 seed 的全部 32,132 行 loss
都带 `constraint_divergence` / `constraint_vorticity` / `constraint_total`，
`constraint_divergence` 最小值 0.072098–0.072213（严格为正），`constraint_vorticity` 最小值
0.071308（也严格为正，实测；注意判定本身只对散度查正性、对涡度只查存在性）。**生效 ≠ 有益**，
(a) 正是用来分开这两件事的。

**关闭的边界（必须与结论一起引用）**：本条关闭的是**「在 4 epoch 父模型上加这一项导数域目标匹配」**
这一支。B 的饱和判定是 `unsaturated`（Δ = +0.030124 > 0.01），即该父模型**尚未训练到位**，
所以上述关闭**不得**外推为"导数域匹配在任何训练预算下都无用"，也**不得**外推为
"物理信息方向无用"——C1（无目标约束束）与 E1（训练信号族）是不同的支。

## Consequences

**代价与不变困难，如实列出**：

1. **本轮 66.993549 GPU-h 换来一个否定结论**（账本六行现算：prepare 601.5 s + archive 5.5 s +
   seed41 78,333.6 s + seed42 80,860.5 s + seed43 81,367.3 s + reading 8.4 s = 241,176.8 s）。
   加上前三次作废的 35.208213，**C2 一支累计花掉约 102 GPU-h**，全部无正面结果。
2. **这个否定结论的价值受限于它的适用域**：只在 4 epoch 父模型上测过；B 判未饱和意味着
   更大的训练预算可能改变结果，但**本轮不授权**去试（预声明关闭该支）。
3. **(c)(d) 不携带信息这一结构缺陷本轮未修**。`evaluate_exit` 里 `c = True` 是字面量，
   `d` 只读 `readings()` 自己写的两个常量；实测把三 seed 的 skill 全部伪造成 0.5，四项仍全真、
   verdict 仍是 `advance-to-independent-review`。修它需要改
   `scripts/study_r7_s3_derivative_matching_c2.py`，而该文件在 attempt08 的 **95 条执行 pin 之内**
   ——改它会让 seed43 在启动时 `RuntimeError: execution source changed`、作废已花的 ~44 GPU-h。
   故**只记录、留给下一次独立变更**。后果是：未来若只看"四项出口全真"会高估证据强度；
   结果页 §4 已写明，但代码缺陷仍在。
4. **一个方向被关闭，而它的近邻未被测**。C2（导数域**目标匹配**）关掉后，仓库里
   "物理信息"这一半仍无任何实验证据：(b) 成立只说明这一项在动；(a) 说明它不划算。
   守恒、平衡、连续方程、谱损失、CRPS 仍然零命中。
5. **父模型不饱和这件事本身没有被利用**。B 判定 `unsaturated` 授权"最终 run 取 16–64 epoch"，
   但**选定具体配置属 Stage F 菜单与用户手选**，本条**不代为选定**、也**不**开补偿性实验。
6. **两个 seed 的 Δ 与噪声同量级**（0.91× / 1.28× 控制臂极差）。若只看单个 seed，
   seed41 的 −0.011634 完全可能被读成"噪声"。本条据以判决的是**三 seed × 五档的一致性**，
   这一点必须在任何引用中被保留，否则结论会被误读为"单 seed 就够"。

**正面但有限的一侧**：这是 C2 第一次**产出真读数**（前三次零读数），且六个臂的训练/评分/身份
全部通过校验——`constraint_terms_recorded` 三 seed 全 True、checkpoint 与父 state_digest 逐 seed 相符、
峰值 reserved 9,569,304,576 B 未超协议估计、`scientific_claim: false`/`test_read: false`。
