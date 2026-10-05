# R7 S2 / #78 R-B：变化尺度加权损失（dev store，val-only）

<!-- round-node: S2 -->

- 登记日期：2026-10-05；`scientific_claim: false`、`test_read: false`、`actual_pass: false`。
- 本文是 `docs/goals/s2-climatology-mechanism-screening.md` §2 的 #78 R-B 交付物，也是
  `docs/R7_S2_78_CHANGE_SCALE.md` §7 指定的「另轮」。**性质**：单 dev store 上的有界机制筛选
  （三 seed、400 更新、val-only），不构成科学声明。

## 1. 与 R-A 的机制差异（运行前写定，非事后解释）

| | R-A（已登记 screening-negative） | R-B（本轮） |
| --- | --- | --- |
| 改什么 | **解码参数化**：`Y = X_t + (d_c/s_c)*r_c` | **训练目标权重**：`mean_c w_c * err_norm_c²` |
| 不动什么 | loss（逐通道等权纬度加权 MSE） | 解码（保持 `identity`，R-A 结论不回撤） |
| 假设 | 有效步长量纲错配是主约束 | 等权目标把某些通道的相对误差定价错了 |
| 逻辑关系 | R-A 负结果**不否证** R-B（不同因素），但降低先验 | R-B 单独检验「目标权值是约束」这一可证伪命题 |

权重规则（运行前冻结）：`w_c = (1 / runtime_ratio_c)^2` 再归一化到均值 1——这是 issue #78 R-B
公式 `sum_c w_c * mean(((forecast-target)/d_c)²)` 在 store 归一化空间的**同一目标**：状态已除以
`s_c`，而 `runtime_ratio_c = d_c/s_c`。归一化到均值 1 保持目标整体尺度（等价学习率）与
incumbent 一致；退化通道 `ratio=1.0`，其 `(1/1)²` 即等权单位权重。全部只来自 **train-only**
已发布 sidecar，不按 val/test 调参。权重范围 0.0633–6.7788（`z250` 最大、`u10` 最小）。

## 2. 规程与身份（冻结于第一步之前）

| 项 | 值 |
| --- | --- |
| 对照 | 单开关：训练损失 `loss_channel_weights=None`（旧行为逐位不变）vs 17 通道冻结权重向量 |
| 共享控制 | K4、dim192/depth4/heads4/window4/patch2、8+8 processes、LR 2e-4、warmup80、cosine→0.1、batch2、clip1.0、400 更新、验证每 100、patience4；**解码 identity**；两臂共用 checkpoint 选择度量（等权归一化 MSE） |
| seeds | 41/42/43（预声明）；同 seed 初始化逐位共享；参数量 2,948,771、forward 1.5466e10、fwd+bwd 4.6283e10 两臂**逐位相等** |
| 数据 | `outputs/r7_s1_seasons_2017/store`，train 340/val 34/test 22；**test 全程未读** |
| protocol SHA256 | `4bee95613783c2950b9800cbcb70b9b0712afb78fb4f31ed6b8d97f3a20fad73` |
| model_code SHA256 | `3ddab46b1e4c2c7e66449e39ab8247c9c7e642f45023c1c9cc14bca8f74fd217` |
| data_identity | `894b8d1b…`（与 R-A 同一 dev store，未变） |
| change_scale_identity | `60e6ac55…`（同一已发布 train-only sidecar） |
| 预算 | planned 7200 s / hard cap 18000 s；每 seed deadline 5400 s；实测无 overrun |
| paired_comparison SHA256 | `c5eed3993f5126e811457bd291fa27ec638fd8073551c421cb32f114fc5ff1e9` |

## 3. 接线探针（训练前实测，非声明）

每 seed 第一步之前实测：`backward_streamed_truncated(reasoning_steps=0)` 的权重臂总损失 ÷
独立解析重算的等权值，必须等于权重向量对误差谱的效应；恒等臂必须正好为 1。

| seed | max relative error | analytic ratio | trainer(weighted)/analytic | 判定 |
| --- | ---: | ---: | ---: | --- |
| 41 | 0.0 | 0.176336 | 0.176336 | within 1e-3 |
| 42 | 0.0 | 0.177120 | 0.177120 | within 1e-3 |
| 43 | 0.0 | 0.176673 | 0.176673 | within 1e-3 |

**接线为真**：权重确实只改目标的逐通道定价，且训练合同记录了该向量（identity 臂记录为无）。
三 seed 的 `reasoning_steps=0`、`process_weight=0` 路径与解析式**逐位一致**（相对误差 0.0）。

## 4. 注册读数（冻结判据逐字判定）

预注册主格：`process_loss_change_scale − process_loss_identity` 的 t2m val RMSE，6h 与 12h，
按每 seed 同号规则；两 lead 同负才算 supported。

| 主格 | seed41 Δ | seed42 Δ | seed43 Δ | seed-mean Δ | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| t2m 6h | +0.503098 | +0.196521 | +0.230666 | **+0.310095** | **worsened（三 seed 同号为正）** |
| t2m 12h | +0.793534 | +0.312514 | +0.335661 | **+0.480570** | **worsened（三 seed 同号为正）** |

**注册结论：worsened —— 按冻结决定文本（"Falsified if either lead is worsened"）该加权目标
在本预算/本实例上被证伪。** 注意权重把 `z250`（6.78）等高变化通道放大，而 t2m 权重反而
被压到 0.2676；主格恶化与「按变化尺度重新定价会改善 t2m」的预注册方向相反。

## 5. 全变量报告（17 变量 × 5 lead = 85 cell，不并入判定）

| lead | improved（3 seed 同号且负） | worsened | unresolved |
| ---: | ---: | ---: | ---: |
| 6h | 4 | 7 | 6 |
| 12h | 4 | 3 | 10 |
| 24h | 3 | 1 | 13 |
| 48h | 7 | 2 | 8 |
| 72h | 3 | 1 | 13 |
| **合计** | **21** | **14** | **50** |

`case_identity=exact`。值得一提但不并入判定的描述性模式：长 lead（48h）improved 略多——
与 R-A 的长 lead 改善多数格方向相近；这只是描述，未被预注册，本轮不支持任何方向性结论。

## 6. 成本

| 项 | 值 |
| --- | --- |
| 训练合计（三 seed、两臂） | 1107.3 s |
| 评估合计（三 seed × 两臂 × 5 lead） | 164.5 s |
| 每 seed 墙钟 | 532.5 / 520.7 / 542.5 s（含探针与协议） |
| **整轮 GPU-h** | **0.3533** |
| overrun | 0（软预算 7200 s 未超） |
| 网络 | 无（离线；`deny_network`） |

FLOPs：两臂参数量 2,948,771、forward 1.5466e10、fwd+bwd 4.6283e10 逐位相等——单开关只改
损失权重、不改计算图，same-updates 与 same-compute 在此一致。

## 7. 已确认 / 推测 / 未做

**已确认（实测）**：None 路径与旧实现逐位相同（梯度逐位相同，反证测试）；权重路径与解析
`sum_c w_c * mse_c` 值/梯度一致；非法权重 fail-closed；权重推导 = `(1/ratio)²` 归一化、来自
train-only sidecar；训练合同分别记录两臂目标；接线探针三 seed 相对误差 0.0；主格两 lead
三 seed 同号为正 → worsened；全 85 cell 如实报告；test 未读。

**推测（未验证）**：为何放大高空大变化通道会恶化 t2m 短 lead——可能是优化预算重分配、
也可能与归一化空间内误差谱的尺度耦合有关。**不在本 doc 采信。**

**未做**：未读 test；未做 R-C（训练配方延长）；未试其他权重口径（单一预注册口径）；未把
48h 的描述性多数格追认成任何主张；未改任何冻结判据。

## 8. 回主线动作

- #78 R-B 关闭为 **screening-negative（worsened on both registered leads）**：加权目标不进入
  正式确认实例候选；incumbent 保持等权损失的 identity 解码语义。
- **#78 两个子轮（R-A 解码、R-B 目标）均已按判据关闭为 negative**；只剩 R-C（训练配方/延长）
  未试，且它是「已选解码/目标下的学习曲线」实验，与前两者不同因素。
- 证据指针：`outputs/r7_78_rb_loss_pilot/paired_comparison.json`（`primary`、`pairs`、
  `loss_weight_probe`、`mechanism_difference_from_ra`、`table`）；`arm_table.csv`、
  `training_table.csv`、`rmse_table.csv`、`case_table.csv`、`memory_table.csv`；
  sidecar `outputs/r7_s1_seasons_2017/change_scale/`（身份 `60e6ac55…`）。
