# R7 #64 B2 — 受控多种子确认性比较（36 天工程段，2016 年 1 月）

Status: **DONE（工程 + 实验段；科学 verdict 是负结果）**。
`scientific_claim: false` 全程。判定绑定提交见 `docs/R7_TASK_QUEUE.md`。
本段的正向结论**只有**「训练协议稳定、可复现、比较器可用」；
「神经基线是否强于零参数控制」的答案在本数据范围内是 **否定的**。

## 0. 起点与判定

| 项 | 值 |
| --- | --- |
| 起点 SHA | `d70dfd4` |
| 收口 SHA | `94828da` |
| CI（CPU，全量 pytest + 34 条阻断规则） | run [36276587108](https://github.com/Eswink/UrbanPiDiT_R2/actions/runs/36276587108) — **success** |
| 17 条实验 workflow | 全部 **skipped**（commit message 未带实验标签，设计行为） |
| 全量测试 | **1046 passed / 3 skipped**（本轮新增 45 个） |
| 34 条阻断规则 | **0 违规**；`--report` 报告型 0 命中 |
| 数据 | `outputs/r7_b2_segment/source.nc` SHA256 `2ba504fe01f0070fa1bda10b546e12c0f23b715bbeb7f1a2ae2a29d54d81af31` |
| 训练总耗 | **1.157 GPU-h**（对照 4.0 上限，累计 **1.723 / 4.0 = 43.1%**） |

## 1. 为什么必须扩数据（不是选择，是几何约束）

#64 B2 明文要求 6/12/24/48/72h **自由 rollout**。冻结的 D1 段（120 时次）
**做不到 72h**：一个 rollout 窗口需要 `history_steps=2` 帧历史（1 历史步 + init）
**加** 12 个 lead 步 = **14 个连续同 split 时次**，而 D1 的 val 块只有
`[2016-01-25, 2016-01-28)` = 3 天 = **12 个时次**。

实测（不是外推）：

```
val lead=48h windows=3
val lead=54h windows=2
val lead=72h REFUSED: no complete held-out rollout windows at requested horizons
```

即 D1 上 72h 自由 rollout 的窗口数是 **0**。B1 能报 48h 是因为它按单时效
请求（每 run 一个 lead，例数 10/9/7/3），但 72h 连一个窗口都没有。

**处置**（决策 0008）：取 36 天段（144 时次），train 区间与 D1 **逐位相同**。
实测核对：

| 项 | D1 | B2 | 判定 |
| --- | --- | --- | --- |
| train 时次数 | 96 | 96 | 同 |
| train state 帧 SHA256 | `95b2deb38bfc9060` | `95b2deb38bfc9060` | **逐位相同** |
| `normalization_mean/std` | — | — | 逐位相同（4 个数组全等） |
| train 窗口数 / 身份 | 94 | 94 | 逐样本 id+index+时间戳相同 |
| val 时次 | 12 | 28 | 扩 |
| val 72h 窗口 | **0** | **15** | 这就是本决定的目的 |
| test 时次 | 12 | 20 | 独立未读块 |

9 个源变量在前 120 时次上**逐位相同**（实测 `np.array_equal` 全 True），
所以 B1 与 B2 共用同一训练分布、同一归一化、同一 target 变量。

成本（对照冻结预算）：decoded **10.6 / 64 GiB**、网络 3.99 GiB、
墙钟 **22.6 min**（单次 ≤30 min）、新产物 21.5 MB / 16 GiB。四项全过。

## 2. 预注册判据原文（冻结于任何一步之前）

协议 digest **`244af00b0834948091506f20f6d3d26a093cff8b142a5f10b084e4b95c6643e2`**，
三个 seed 各自独立算出的 digest **完全相同**（`seed41/42/43/protocol.json`）。

**seed 集合（预先写定）**：探索 `(41, 42)`；确认 `(41, 42, 43)`。确认集包含探索集，
因此已付出的探索运行按同一 digest 复用，**不是**事后挑 seed。每个声明的 seed 都报告。

**训练前固定的协议**（`shared_controls` 原文摘要）：

| 项 | 固定值 |
| --- | --- |
| 优化器 | AdamW lr 2e-4, weight_decay 1e-4 |
| lr schedule | 线性 warmup **80** 步到峰值，然后 cosine 衰减到峰值 0.1，在 update 800 结束 |
| 最大更新 | **800** |
| 验证频率 | 每 **100** 步 |
| 验证集 | **val only**（test 全程未读） |
| checkpoint 选择 | 验证目标最低者；并列取更早的 checkpoint |
| 早停 | 连续 **4** 次验证检查相对改善 < **0.1%** 即停；**只读 validation** |

**技巧检查判据原文**（`SKILL_CRITERIA`，写在跑之前）：

| 时效 | 变量 | 要求 |
| --- | --- | --- |
| 6h | t2m, mslp, v850, u10 | 4/4 |
| 12h | t2m, mslp, v850 | 2/3 |
| 24h | mslp, v850 | 1/2 |

- 适用臂：`native_window`、`generic`；
- 「胜」= 对 persistence **和** climatology **分别**都要赢，且 seed-paired delta 在
  三个 seed 上**同号**（#60 比较器规则），因此 seed 噪声买不到通过；
- 允许 1 个必需格 unresolved；**任一必需格输给任一控制即失败**，不消耗允许额；
- 判据自述：「6h 是训练时效所以最严；t2m 放进 6/12h 是因为它是 B1 里 climatology
  最难赢的变量——一个漏掉 t2m 的 gate 就是为通过而选的 gate」；
- 判据自述状态分离：「『强基线可用』与『ours 胜出』是两个不同状态」——本 gate
  **只**判神经网络臂能否越过零参数控制，**不**判递归臂能否赢非递归臂。

## 3. 四张表（#64 B2 强制；B1 已证「同更新≠同算力」）

### 3.1 参数量

| model | family | trainable params | 相对 2.8M | 带内 | forward FLOPs | fwd+bwd FLOPs |
| --- | --- | --- | --- | --- | --- | --- |
| unet | native | 2,789,903 | −0.361% | ✓ | 8,893,199,172 | 26,338,284,984 |
| native_window | native | 2,803,601 | +0.129% | ✓ | 12,985,192,320 | 38,841,832,704 |
| afno_small | native | 2,831,433 | +1.123% | ✓ | 11,005,050,096 | 32,868,230,624 |
| generic | generic | 2,799,202 | −0.029% | ✓ | 12,843,758,976 | 38,417,532,672 |
| process | process | 2,799,779 | −0.008% | ✓ | 12,843,777,408 | 38,417,551,104 |
| persistence | parameter-free | **0** | n/a | 不进带 | 0 | 0 |
| climatology | parameter-free | **0** | n/a | 不进带 | 0 | 0 |

五臂极差 **1.483%**（`2,789,903–2,831,433`），与 B1 实测一致。
**forward FLOPs 相差 1.46×**（unet 8.89e9 vs native_window 1.30e10）——
这是「同更新≠同算力」在同一张表里的直接体现，不是脚注。

### 3.2 wall time（每 seed 5 臂）

| seed | 训练秒 | GPU-h | 每臂选中 update | 早停 |
| --- | --- | --- | --- | --- |
| 41 | 1372.1 | 0.3811 | unet 800 / native_window 500 / afno 600 / generic 800 / process 700 | 无 |
| 42 | 1369.8 | 0.3805 | unet 800 / native_window 600 / afno 500 / generic 700 / process 700 | 无 |
| 43 | 1421.5 | 0.3948 | unet 700 / native_window 800 / afno 600 / generic 800 / process 600 | 无 |

合计 **4163.4 s = 1.157 GPU-h**。单卡实测 **0.292–0.377 s/update**（17 通道、
batch 2、800 步；unet 最快、recursive 两支最慢）。峰值 reserved 最高
328 MiB/卡（24 GiB 卡），余量充足。

### 3.3 例数（每 lead 的 case 表，三个 seed 相同）

| lead | 可用窗口 | 实际评估 |
| --- | --- | --- |
| 6h | 26 | **26** |
| 12h | 25 | **25** |
| 24h | 23 | **23** |
| 48h | 19 | **19** |
| 72h | 15 | **15** |

105 次评估（3 seed × 7 model × 5 lead）**全部** `split=val`，`test_read: false`；
同 lead 内 7 个 model 的 `n_evaluated` 完全一致（脚本内断言，不一致即 raise）。

### 3.4 训练曲线（完整 epoch 均值，非首末单点）

每臂 18 个 epoch（800 步 / 94 窗口 ≈ 8.5 epoch，末 epoch 不满，故只取完整 epoch）：

| seed | arm | epoch0 均值 | 末完整 epoch 均值 | 比值 |
| --- | --- | --- | --- | --- |
| 41 | unet | 0.18631 | 0.11946 | 0.641 |
| 41 | native_window | 0.18412 | 0.10183 | 0.553 |
| 41 | afno_small | 0.18041 | 0.07108 | 0.394 |
| 41 | generic | 0.18336 | 0.09092 | 0.496 |
| 41 | process | 0.18323 | 0.09077 | 0.495 |
| 42 | unet | 0.18635 | 0.11793 | 0.633 |
| 42 | native_window | 0.18416 | 0.09992 | 0.543 |
| 42 | afno_small | 0.18079 | 0.07262 | 0.402 |
| 42 | generic | 0.18324 | 0.08888 | 0.485 |
| 42 | process | 0.18321 | 0.08904 | 0.486 |
| 43 | unet | 0.18615 | 0.11286 | 0.606 |
| 43 | native_window | 0.18351 | 0.10151 | 0.553 |
| 43 | afno_small | 0.17980 | 0.07348 | 0.409 |
| 43 | generic | 0.18248 | 0.08917 | 0.489 |
| 43 | process | 0.18242 | 0.08845 | 0.485 |

15/15 臂单调下降，降幅 **36–61%**；同 seed 内各臂 epoch0 均值差 ≤3.6%（同一初始化
尺度的证据）；**无早停触发**（说明 800 步上限没有把训练掐在下降中途）。

## 4. 每变量每时效 RMSE 与胜负计数

### 4.1 三个关键变量的 3-seed 均值（每 lead 最优者标 WIN）

**t2m（K）**

| lead | unet | native_window | afno_small | generic | process | persistence | climatology | WIN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | 4.061 | 3.400 | 3.234 | 2.831 | 2.958 | 4.312 | **2.803** | climatology |
| 12h | 4.929 | 4.002 | 3.752 | 3.327 | 3.496 | 5.312 | **2.736** | climatology |
| 24h | 3.913 | 3.976 | 4.010 | 4.177 | 4.006 | 2.815 | **2.551** | climatology |
| 48h | 4.785 | 5.487 | 4.919 | 6.839 | 6.337 | 4.073 | **2.621** | climatology |
| 72h | 4.872 | 6.708 | 4.958 | 7.534 | 6.339 | 4.580 | **2.647** | climatology |

**mslp（Pa）**

| lead | unet | native_window | generic | process | persistence | climatology | WIN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | 213.7 | 187.9 | **158.4** | 166.1 | 223.0 | 502.8 | generic |
| 12h | 298.7 | 268.4 | **243.4** | 257.0 | 319.2 | 500.7 | generic |
| 24h | **415.8** | 425.6 | 503.2 | 480.2 | 456.0 | 505.2 | unet |
| 48h | **569.2** | 620.0 | 722.3 | 729.8 | 704.5 | 535.8 | climatology |
| 72h | **740.5** | 883.4 | 890.0 | 889.4 | 849.1 | 568.0 | climatology |

**v850（m s⁻¹）**

| lead | unet | native_window | generic | process | persistence | climatology | WIN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | **2.353** | 2.377 | 2.453 | 2.403 | 2.465 | 4.614 | unet |
| 12h | **3.450** | 3.609 | 3.681 | 3.581 | 3.633 | 4.683 | unet |
| 24h | 4.683 | 4.879 | 4.856 | **4.557** | 5.105 | 4.814 | process |
| 48h | 5.334 | 6.014 | 5.808 | 5.433 | 6.277 | **4.690** | climatology |
| 72h | **5.004** | 6.587 | 6.205 | 6.391 | 6.021 | 3.649 | climatology |

### 4.2 胜负计数（17 变量，3 seed 均值，逐 lead）

| arm | lead | 胜 persistence | 胜 climatology | **同时胜两者** |
| --- | --- | --- | --- | --- |
| unet | 6h | 13/17 | 16/17 | 12/17 |
| unet | 12h | 13/17 | 16/17 | 12/17 |
| unet | 24h | 11/17 | 9/17 | 9/17 |
| unet | 48h | 9/17 | 2/17 | 2/17 |
| unet | 72h | 10/17 | 1/17 | 1/17 |
| native_window | 6h | 14/17 | 16/17 | 13/17 |
| native_window | 12h | 14/17 | 16/17 | 13/17 |
| native_window | 24h | 9/17 | 8/17 | 6/17 |
| native_window | 48h | 8/17 | 1/17 | 1/17 |
| native_window | 72h | 5/17 | **0/17** | **0/17** |
| afno_small | 6h | 13/17 | 16/17 | 12/17 |
| afno_small | 12h | 12/17 | 16/17 | 11/17 |
| afno_small | 24h | 10/17 | 10/17 | 8/17 |
| afno_small | 48h | 10/17 | 1/17 | 1/17 |
| afno_small | 72h | 11/17 | **0/17** | **0/17** |
| generic | 6h | 14/17 | 16/17 | 13/17 |
| generic | 12h | 13/17 | 16/17 | 12/17 |
| generic | 24h | 8/17 | 8/17 | 5/17 |
| generic | 48h | 6/17 | **0/17** | **0/17** |
| generic | 72h | 5/17 | **0/17** | **0/17** |
| process | 6h | 14/17 | 16/17 | 13/17 |
| process | 12h | 14/17 | 16/17 | 13/17 |
| process | 24h | 10/17 | 9/17 | 7/17 |
| process | 48h | 8/17 | 1/17 | 1/17 |
| process | 72h | 5/17 | **0/17** | **0/17** |

## 5. 预注册判据的判定结果：**未通过**（负结果，保留）

`skill_gate_met: false`；`native_window` 与 `generic` 均为 **false**。

| arm | 对 persistence | 对 climatology | 失败原因 |
| --- | --- | --- | --- |
| native_window | 胜 7/7，unresolved 2，**lost 0** | 胜 6/7，unresolved 1，**lost 2** | **climatology 在 t2m 6h 与 12h 上取胜** |
| generic | 胜 6/7，unresolved 3，**lost 0** | 胜 5/7，unresolved 3，**lost 1** | **climatology 在 t2m 12h 上取胜** |

逐格结果（`OK`=胜、`~~`=unresolved、`XX`=负）：

```
native_window
  OK persi 6h t2m   OK persi 6h mslp   OK persi 6h v850  OK persi 6h u10
  OK persi 12h t2m  OK persi 12h mslp  ~~ persi 12h v850
  OK persi 24h mslp ~~ persi 24h v850
  XX clima 6h t2m   OK clima 6h mslp   OK clima 6h v850  OK clima 6h u10
  XX clima 12h t2m  OK clima 12h mslp  OK clima 12h v850
  OK clima 24h mslp ~~ clima 24h v850
generic
  OK persi 6h t2m   OK persi 6h mslp   ~~ persi 6h v850  OK persi 6h u10
  OK persi 12h t2m  OK persi 12h mslp  ~~ persi 12h v850
  ~~ persi 24h mslp OK persi 24h v850
  ~~ clima 6h t2m   OK clima 6h mslp   OK clima 6h v850  OK clima 6h u10
  XX clima 12h t2m  OK clima 12h mslp  OK clima 12h v850
  ~~ clima 24h mslp ~~ clima 24h v850
```

**判定依据**：判据要求必需格对**两个**控制分别取胜；失合格 `clima 12h t2m` 对两个臂
都是 `lost`，`clima 6h t2m` 对 native_window 是 `lost`。判据明文「任一必需格输给任一
控制即失败，不消耗允许额」，因此不得用允许额消化。**不删坏变量、不放宽判据**。

**与 B1 的一致性**：B1（D1，单 seed，200 步）已在四个神经臂上观察到 t2m 全时效输给
零参数 climatology；B2 用 800 步 × 3 seed 在更宽的数据上**复现**了同一结论，并且
把对手收窄到只有 climatology（对 persistence 两个臂都是 **0 负**）。

## 6. 共享初始权重（#64 B2 第 3 条）

`generic` 与 `process` 共享 backbone / draft_encoder / correction_head。实现：
anchor（`generic`）按声明 seed 建一次，其初始化 `state_dict` 在该臂第一步之前
拷入 `process`，**应用与忽略的参数名逐条记录**（不是假设）：

| seed | arm | provided | applied | ignored |
| --- | --- | --- | --- | --- |
| 41 | generic | False（anchor） | 0 | 0 |
| 41 | **process** | **True** | **72** | **39** |
| 42 | process | True | 72 | 39 |
| 43 | process | True | 72 | 39 |

39 个 ignored 是 anchor 的 generic-only 部分（`latent` / `cell.*` /
`latent_to_context.*`），在 process 里无对应参数，**无法对齐**。

`unet` / `native_window` / `afno_small` **不声明**权重对齐：实测 `generic` 把
backbone 嵌在 `backbone.` 前缀下，与 native 家族的参数字典**交集为 0**，
因此只匹配数据与预算。测试
`test_arms_with_no_shared_structure_cannot_be_force_aligned` 锁住这个事实
（强配会 fail closed，而不是静默半配）。

## 7. seed 噪声（本段最重要的方法论发现）

同一格（model, variable, lead）三 seed 的 `(max−min)/mean`：

| arm | 6h 中位 | 12h | 24h | 48h | 72h 中位 | 72h 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| unet | 2.71% | 2.95% | 9.54% | 14.75% | 20.30% | 39.09% |
| native_window | 3.20% | 5.59% | 7.63% | 17.39% | 23.57% | **51.26%** |
| afno_small | 3.66% | 5.70% | 8.32% | 11.96% | 20.65% | **56.08%** |
| generic | 4.63% | 6.27% | 13.54% | 22.48% | 22.71% | **65.13%** |
| process | 2.75% | 8.14% | 11.12% | 18.41% | 21.74% | **55.08%** |

**读法**：6h（训练时效）seed 噪声 ~3%，单一 seed 尚可代表；**自由 rollout 的
24–72h 噪声达 20–23%（个别格 65%）**，因此 48/72h 的**单 seed 排序不可信**——
这也是 #60 比较器要求「每 seed delta 同号才算 established」的实证理由。
零参数控制在三 seed 上**逐位相同**（`rel_diff=+0.00%`），因为它们是确定性统计量，
与 seed 无关——这同时是一个内部一致性检查。

## 8. 显式局限（逐条，与前段一致）

1. **单一年 1 月**：B2 全部是 2016 年 1 月；(month,hour) 气候态桶只有 **4** 个。
   因此 **climatology 在 t2m 上优于神经基线是本数据范围的产物，不是「基线很强」的
   证据**——一个只有 4 个桶的月内均值几乎就是「当前月份该小时的平均天气」。
2. **val/test 是 2016 段内工程再划分**，**不是** v2 的 2019 验证年或 2021 封存 test。
   报告中不得表述为封存测试集。B2 全程 `test_read: false`（105/105 评估在 val）。
3. **跨季节/跨年结论一律 UNVERIFIED**（本段无法测试）。
4. **训练未收敛**：800 步上限、18 epoch、无早停；这是有界检查点，不是收敛基准。
5. **5 条 B1 局限继续适用**（工程段、单 seed 无显著性 → B2 已部分缓解为 3 seed、
   4 桶气候态、非收敛、评分的是全部 17 通道）。
6. **3 个 seed 不构成显著性检验**；同号一致性不是 p 值。
7. **「强基线可用」与「ours 胜出」是两个状态**：本段可以说的上限是
   「零参数控制仍然很强，神经臂在 t2m 上未越过它」；**没有**任何递归臂胜过
   非递归臂的结论（B1 的 `process_weight=0` 臂与 B2 的五个臂都不构成该检验）。

## 9. 未做的事

- 未做 1→2→4 预测时刻的 training curriculum（#64 明文说它是**另设有界实验**，
  且**不得与内部 reasoning K 混淆**）——本段只做 +6h 训练与自由 rollout。
- 未做 K 的消融（#65 的题目）；B2 所有递归臂用固定 `reasoning_steps=3`。
- 未做 DDP 对照：本机两卡无 NVLink，两个独立 seed 并行更快，未复测 DDP 吞吐
  （`docs/R7_GPU_BRINGUP.md` 已记录 +7% 单卡劣势）。
- 未读 test（设计如此）；未用 test 调任何东西。
- 未删除任何变量或权重；未选 seed；未放宽判据。

## 10. 复现命令

```bash
# 1) 取 36 天段（只读 preflight → 显式 --write）
.venv/bin/python -m data.download.earthmover_spatial_b2 --preflight \
    --report outputs/r7_b2_segment/preflight_report.json
.venv/bin/python -m data.download.earthmover_spatial_b2 --write \
    --out outputs/r7_b2_segment/source.nc \
    --receipt outputs/r7_b2_segment/source_receipt.json --max-decoded-gib 64
.venv/bin/python prepare_r7_local.py --source outputs/r7_b2_segment/source.nc \
    --config configs/r7_era5_b2_segment.yaml --write \
    --store outputs/r7_b2_segment/store/cache.zarr \
    --manifests outputs/r7_b2_segment/store/manifests --max-raw-gib 1

# 2) 每个 seed 一个进程（两卡并行）
for s in 41 42; do
  .venv/bin/python scripts/study_r7_b2_multiseed.py --mode seed --phase explore \
    --seed $s --device-index $((s-41)) \
    --manifests outputs/r7_b2_segment/store/manifests --out outputs/r7_b2_multiseed &
done; wait
.venv/bin/python scripts/study_r7_b2_multiseed.py --mode seed --phase confirm \
    --seed 43 --device-index 0 \
    --manifests outputs/r7_b2_segment/store/manifests --out outputs/r7_b2_multiseed

# 3) 合并 + 四张表 + 预注册 gate
.venv/bin/python scripts/study_r7_b2_multiseed.py --mode finalize --phase confirm \
    --manifests outputs/r7_b2_segment/store/manifests --out outputs/r7_b2_multiseed
```

产物：`outputs/r7_b2_multiseed/{b2_result.json, b2_rmse_table.csv,
b2_parameter_table.csv, b2_flops_table.csv, b2_wall_time_table.csv,
b2_case_count_table.csv, b2_skill_gate.json, b2_comparator_summary.json,
seed{41,42,43}/}`（`outputs/` 不进版本控制，按 R-035）。
