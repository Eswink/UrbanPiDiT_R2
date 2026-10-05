# R7 S2 / #78 R-A：归一化变化尺度重参数化解码（dev store，val-only）

<!-- round-node: S2 -->

- 登记日期：2026-10-05；`scientific_claim: false`、`test_read: false`、`actual_pass: false`。
- 本文是 `docs/goals/s2-climatology-mechanism-screening.md` §2 D1/D2 的 #78 R-A 交付物。
- **性质**：单 dev store 上的有界机制筛选（三 seed、400 更新、val-only）。不构成科学声明，
  不构成「超过气候态」的任何一步；它只决定该重参数化是否值得进入正式确认实例。

## 1. 规程与身份（冻结于第一步之前）

| 项 | 值 |
| --- | --- |
| 对照 | 单开关：解码参数化 `identity`（旧行为逐位不变）vs `normalized_change_scale`（`Y = X_t + (d_c/s_c)*r_c`） |
| 变化尺度 | train-only 精确连续 6h 窗口拟合的 `d_c`；sidecar `outputs/r7_s1_seasons_2017/change_scale/`，独立身份 `60e6ac55…`，17 通道 ratio 0.0722–0.7473 |
| 共享控制 | K4、dim192/depth4/heads4/window4/patch2、8+8 processes、LR 2e-4、warmup80、cosine→0.1、batch2、clip1.0、400 更新、验证每 100、patience4；**loss 不变**（纬度加权 MSE） |
| seeds | 41/42/43（预声明）；同 seed 初始化逐位共享；参数量 2,948,771 与 forward/fwd+bwd FLOPs（1.5466e10/4.6283e10）两臂**逐位相等** |
| 数据 | `outputs/r7_s1_seasons_2017/store`，train 340/val 34/test 22；**test 全程未读** |
| protocol SHA256（v2 注册轮） | `84dafc498c69c4ef0cd0e16b4dea5cbe0e2e9fd6915ee782e6977ca62997e074` |
| model_code SHA256 | `701f182383f77b4cb30d833f516174f4c6638973bea260f45b87f18016252646`（#78 接线后；`84e77e8b…` 是 #77 记录，不回改） |
| 预算 | planned 7200 s / hard cap 14400 s；每 seed deadline 4500 s；实测无 overrun |

## 2. 接线探针（训练前实测，非声明）

每 seed 在第一步之前实测：identity 与 scaled 两探针模型同 seed 权重**逐位相同**，
shipped sidecar ratio 必须逐通道乘上初始解码增量。

| seed | max relative error | 判定 |
| --- | ---: | --- |
| 41 | 1.15e-05 | within 1e-3 |
| 42 | 2.64e-05 | within 1e-3 |
| 43 | 1.17e-05 | within 1e-3 |

**机制接线为真**（v1 缺陷轮曾因探针前置不成立而被拒跑，见 §5）。

## 3. 注册读数（冻结判据逐字判定）

预注册主格：`process_decode_change_scale − process_decode_identity` 的 t2m val RMSE，6h 与 12h，
按每 seed 同号规则；两真合取才算 supported。

| 主格 | seed41 Δ | seed42 Δ | seed43 Δ | seed-mean Δ | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| t2m 6h | +0.128008 | +0.089530 | +0.114042 | **+0.110527** | **worsened（三 seed 同号为正）** |
| t2m 12h | +0.171894 | +0.185480 | +0.143072 | **+0.166816** | **worsened（三 seed 同号为正）** |

**注册结论：worsened —— 按冻结决定文本（"Falsified if either lead is worsened"）该重参数化
在本预算/本实例上被证伪。** 正 delta 表示 change-scale 解码的 RMSE 更高；三 seed 全部同号，
不是噪声型不一致。

## 4. 全变量报告（17 变量 × 5 lead = 85 cell，不并入判定）

| lead | improved（3 seed 同号且负） | worsened | unresolved |
| ---: | ---: | ---: | ---: |
| 6h | 0 | 6 | 11 |
| 12h | 1 | 3 | 13 |
| 24h | 9 | 2 | 6 |
| 48h | 11 | 0 | 6 |
| 72h | 9 | 0 | 8 |
| **合计** | **30** | **11** | **44** |

`case_identity=exact`。一个值得如实记下但**不并入判定**的描述性模式：短 lead（6–12h）
worsened 集中、长 lead（24–72h）improved 多数。合理解释是重参数化把解码的有效步长按物理
6h 变化尺度放大/缩小后，短程拟合需要重新学习、而长程表示更接近物理量纲——**这只是假说，
未被预注册，本轮不支持任何方向性结论**。

## 5. 成本与两轮记录（v1 探针缺陷如实计费）

| 轮 | 状态 | 说明 |
| --- | --- | --- |
| v1 (`outputs/r7_78_change_scale_pilot/`) | **探针拒绝轮，非注册轮，零训练** | 三 seed 各 7.1/7.9/6.3 s 在训练前被探针拒绝：harness 只 seed 一次、两探针模型权重不同，前置（逐位同权）不成立；缺陷写入 `FINALIZE_DEFECT.md`，receipt 保留、不手改 |
| v2 (`outputs/r7_78_change_scale_pilot_v2/`) | **注册轮** | 三 seed 各 549.0/553.7/554.9 s；训练合计 1187.6 s（identity 586.1 + scaled 578.2 为三 seed 合计口径），评估 165.8 s；overrun 0 |
| **合计** | **≈0.376 GPU-h** | v1 ≈0.006 + v2 ≈0.370；软预算 7200 s/轮未超；无网络 |

FLOPs：两臂参数量 2,948,771、forward 1.5466e10、fwd+bwd 4.6283e10 逐位相等——单开关不改计算图，
same-updates 与 same-compute 在此一致。

## 6. 已确认 / 推测 / 未做

**已确认（实测）**：接线探针（ratio 确实乘进初始解码，误差 ≤2.6e-5）；两臂同参数量/FLOPs/
同 seed 初始化逐位相同、state-dict key 集合相同（非持久 buffer）；主格两 lead 三 seed 同号为正
→ worsened；全 85 cell 如实报告；test 未读；三轮（v1 拒绝 + v2 三 seed）全部有 receipt。

**推测（未验证）**：短 lead worsens、长 lead improves 的模式是否源于物理量纲或有效步长；
R-B（差值尺度 loss）是否会改变该图景。**不在本 doc 采信。**

**未做**：未读 test；未做 R-B 的差值尺度 loss（另轮）；未试其他 d_c 估计口径（单一预注册口径）；
未把 24–72h 的描述性多数格追认为任何主张；未改任何冻结判据。

## 7. 回主线动作

- #78 R-A 关闭为 **screening-negative（worsened at this budget/instance）**：identity 解码保持
  incumbent 语义；change-scale 解码不进入正式确认实例候选。R-B（差值尺度 loss）是否仍值得单独
  开轮由主线决定——R-A 的负结果不直接否证 R-B，但降低了预期，且必须先写清与 R-A 的机制差异。
- 证据指针：`outputs/r7_78_change_scale_pilot_v2/paired_comparison.json`（`primary`、`pairs`、
  `decode_scale_probe`、`table`）；`arm_table.csv`、`training_table.csv`、`rmse_table.csv`、
  `case_table.csv`、`memory_table.csv`；sidecar `outputs/r7_s1_seasons_2017/change_scale/`；
  v1 探针拒绝记录 `outputs/r7_78_change_scale_pilot/FINALIZE_DEFECT.md`。
