# R7 S2 / #77：位置编码频带单因素筛选（dev store，val-only）

<!-- round-node: S2 -->

- 登记日期：2026-10-05；`scientific_claim: false`、`test_read: false`、`actual_pass: false`。
- 本文是 `docs/goals/main-model-climatology-campaign.md` D2 的 #77 节点交付物。
- **性质**：单 dev store 上的有界机制筛选（三 seed、400 更新、val-only）。不构成科学声明，
  不构成「超过气候态」的任何一步；它只决定该开关是否值得进入正式确认实例。

## 1. 规程与身份（冻结于第一步之前）

| 项 | 值 |
| --- | --- |
| 对照 | 单开关：位置读出的 `position_encoding_mode`，`legacy`（2**-j，逐位旧基）vs `nyquist_band`（同通道预算、[0.5,16] cycles 对数分布） |
| 共享控制 | K4、dim192/depth4/heads4/window4/patch2、8+8 processes、`positional_process_readout=True`、LR 2e-4、warmup80、cosine→0.1、batch2、clip1.0、400 更新、验证每 100、patience4、早停 0.1% 相对 |
| seeds | 41/42/43（预声明）；每 seed 两臂；臂间同 seed 初始化逐位共享（合并门逐 seed 复核） |
| 数据 | `outputs/r7_s1_seasons_2017/store`，data identity `894b8d1b…`，train 340/val 34/test 22 窗口；**test 全程未读** |
| protocol SHA256（v2） | `503df8e43dc2cf56e472b196c7d1f157d77a8b46c6a4d577201487f5485e1c80` |
| model_code SHA256 | `84e77e8b475a1ab5c1f3f031d509e50b91507a6d72c1cf30d48a37d6ed8dbe59`（#77 接线后；旧值 `551261c4…` 不回改） |
| 预算 | planned 7200 s / hard cap 14400 s（整轮）；每 seed deadline 4500 s；实测无 overrun |

## 2. 机制探针（训练前实测，非声明）

| seed | legacy live/dead 通道 | nyquist live/dead | legacy spread/max | nyquist spread/max |
| --- | --- | --- | --- | --- |
| 41 | 16 / 68 | 192 / 0 | 0.4838 | 0.5382 |
| 42 | 16 / 68 | 192 / 0 | 0.4379 | 0.4943 |
| 43 | 16 / 68 | 192 / 0 | 0.4676 | 0.5335 |

开关确实生效：33×33 token 网格上 legacy 基有 68/192 个 fp32 恒定通道、154/192 相位跨度 <1e-3；
`nyquist_band` 全通道活跃且逐位置响应差增大。**机制为真，不等于收益为真。**

## 3. 注册读数（冻结判据逐字判定）

预注册主格：`process_rwa_nyquist − process_rwa_legacy` 的 t2m val RMSE，6h 与 12h，
只在每 seed 增量同号时判定；两真合取才算 supported。

| 主格 | seed41 Δ | seed42 Δ | seed43 Δ | 判定 |
| --- | ---: | ---: | ---: | --- |
| t2m 6h | +0.000557 | −0.000294 | +0.000368 | **unresolved（符号不一致）** |
| t2m 12h | +0.000910 | −0.000674 | +0.000707 | **unresolved（符号不一致）** |

**注册结论：no sign-consistent primary cell —— 按冻结决定文本（"Falsified if either lead is
worsened or both are unresolved"）该机制在本预算/本实例上未成为可用的杠杆。** 负 delta 表示
band 臂更低 RMSE；两格均无 seed 均值（规则禁止对符号不一致求均值）。

## 4. 全变量报告（17 变量 × 5 lead = 85 cell，不并入判定）

| lead | improved（3 seed 同号且负） | worsened | unresolved |
| ---: | ---: | ---: | ---: |
| 6h | 3 | 2 | 12 |
| 12h | 3 | 5 | 9 |
| 24h | 3 | 4 | 10 |
| 48h | 2 | 2 | 13 |
| 72h | 7 | 1 | 9 |
| **合计** | **18** | **14** | **53** |

`case_identity=exact`；两臂同病例、同单位、同区域；无平均跨单位、无事后主格改选。
72h 处 7/17 improved 是全表最偏的一档，但 6–48h 均无方向性——**不追认任何"band 在长 lead
更有用"的说法**，它同样未被预注册且多数格 unresolved。

## 5. 成本与两轮记录（v1 缺陷如实计费）

| 轮 | 状态 | 训练 GPU-s | 评估 GPU-s | 说明 |
| --- | --- | ---: | ---: | --- |
| v1 (`outputs/r7_77_pe_band_pilot/`) | **writer 缺陷，非注册轮** | 1152.4（三 seed 合计 549.0/550.4/575.5 整轮含评估） | 含在内 | finalize 被共享门拒绝：per-arm 训练记录缺 `protocol_sha256`/`switches`；缺陷写入 `FINALIZE_DEFECT.md`，产物保留不删、不手改 |
| v2 (`outputs/r7_77_pe_band_pilot_v2/`) | **注册轮** | 1186.9（三 seed 训练 593.5/591.2 两臂合计；整轮 579.7/555.6/552.4） | 170.5 | 修复 writer（74c2279）后全新目录重跑；v1/v2 均计费 |
| **合计** | | **≈0.65 GPU-h**（1675.3 + 1687.7 ≈ 3363 s） | | 软预算 7200 s/轮未超；无硬截断；无网络 |

FLOPs（同参数量、同 FLOPs，非容量混杂）：2,948,771 参数；forward 1.5466e10 / fwd+bwd 4.6283e10
（两臂逐位相等）。same-updates 与 same-compute 在此完全一致（单因素开关不改计算图）。

## 6. 已确认 / 推测 / 未做

**已确认（实测）**：开关语义生效（探针）；主格在两 seed 滞后符号不一致；两臂参数量/FLOPs/
初始化逐位相同；全流程 val-only、test 未读；三轮执行全部成功、无 overrun；v1 writer 缺陷与
v2 修复、双轮计费、产物双份保留。

**推测（未验证）**：为什么符号不一致（seed 间初始化差异放大了一个极小效应？），以及
legacy 的 154 个低跨度通道是否根本不构成该规模的瓶颈。**两条都留作假说，不在本 doc 采信。**

**未做**：未读 test；未扩种子（第三 seed 已是预声明全集）；未在同一轮换别的开关（一次一因子）；
未把 package-vs-old_ours 改善或本轮的 72h 多数格追认为「超过气候态」；未改任何冻结判据。

## 7. 回主线动作

- #77 就此关闭为 **screening-negative（falsified at this budget/instance）**：频带开关不进入
  正式确认实例的候选配置；`legacy` 保持 incumbent 语义。
- 证据指针：`outputs/r7_77_pe_band_pilot_v2/paired_comparison.json`（含 `primary`、`pairs`、
  `table` 与 `readout_spread`）；`.../arm_table.csv`、`training_table.csv`、`rmse_table.csv`、
  `case_table.csv`、`memory_table.csv`；`outputs/r7_77_pe_band_pilot/FINALIZE_DEFECT.md`（v1）。
- 下一步：#78 R-A 变化尺度重参数化解码（本 doc 之后的主线机制筛选）。
