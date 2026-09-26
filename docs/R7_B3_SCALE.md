# R7 #64 B3 — 容量扩模到 18M（同一 B2 段、同一协议）

Status: **DONE（有界容量消融）**。`scientific_claim: false`。
结论是**混合的**：扩容在一个格上首次越过了零参数控制，但**该格不跨 seed 稳定**，
其余全部否决。本文不含任何「ours 胜出」的表述。

## 0. 判定摘要

| 项 | 值 |
| --- | --- |
| 起点 SHA | `eba90e0`（B3 脚本）/ 收口见 `docs/R7_TASK_QUEUE.md` |
| 协议 digest（两 seed 独立算出相同） | `d3f2c4266ce11a0235f7700e6df5df4030244f13` |
| seed | `(41, 42)`，**预先写定**，两卡各一个独立进程（**不用 DDP**） |
| 参数量 | 18,197,777 / 18,115,234 / 18,116,387（**6.47–6.49×** B2） |
| 15–20M 硬带 | 实测 spread **0.459%**，脚本内**越界即拒绝运行** |
| 训练 | 1968.0 s = **0.547 GPU-h**；6/6 臂 18 epoch，loss 比 **0.519–0.599**，无早停 |
| 显存 | 峰值 968 MiB / 24 GiB（余量 96%） |
| 例数 | 与 B2 **完全相同** 26/25/23/19/15 |
| test | `test_read: false`，全部评估在 val |
| 累计预算 | **2.269 / 4.0 GPU-h = 56.7%** |

## 1. 为什么做 B3（B2 留下的可判问题）

B2 的负结果是：五个 ~2.8M 臂在 **t2m 每个时效**都输给零参数 train-only climatology，
**包括训练时效 6h**。这有两种互斥解释：

- **A**：臂容量不足（扩容可修）；
- **B**：这个只有 4 个 (month,hour) 桶的 1 月气候态在本数据范围内就是硬靶
  （扩容不可修）。

B3 的设计就是**区分 A 与 B**：其余一切与 B2 相同，只改容量。

**控制变量**（协议 `held_fixed_from_b2`）：store、抽样顺序、优化器、
warmup+cosine schedule、800 更新预算、验证频率、checkpoint 选择规则、早停规则、
评估时效、**case 集合**全部保持 B2 值；`declared_change` 只有容量一项。
因此 B3 的 18M RMSE 与 B2 的 2.8M RMSE 在同一批 case 上**可直接相减**。

## 2. 预注册：15–20M 是闸门不是意向

```python
MINIMUM_PARAMETERS = 15_000_000
MAXIMUM_PARAMETERS = 20_000_000
if not (MINIMUM_PARAMETERS <= p <= MAXIMUM_PARAMETERS): raise ValueError
if spread > PARAMETER_BAND: raise ValueError
```

实测 18.20M / 18.12M / 18.12M，spread **0.459%**，六项全部在带内。
**不冲 30M**：上限 20M，脚本硬拒。三臂均从 B2 的 2.8M 放大 **6.47–6.49×**。

`generic`/`process` 继续承接 B2 第 3 条的**共享公共初始权重**：实测
`applied=120, ignored=39`（18M 下公共参数从 72 增到 120，ignored 仍 39）。

## 3. 四张表（B1 已证「同更新≠同算力」）

### 3.1 参数量 / FLOPs

| model | params | 相对 18M | 在 15–20M | forward FLOPs | B2 params | 倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| native_window | 18,197,777 | +1.10% | ✓ | 83,986,210,560 | 2,803,601 | 6.49× |
| generic | 18,115,234 | +0.64% | ✓ | 81,543,703,296 | 2,799,202 | 6.47× |
| process | 18,116,387 | +0.65% | ✓ | 81,543,740,160 | 2,799,779 | 6.47× |
| persistence | **0** | n/a | 不适用 | 0 | 0 | — |
| climatology | **0** | n/a | 不适用 | 0 | 0 | — |

**forward FLOPs 从 B2 的 8.9–13.0 GFLOPs 涨到 81.5–84.0 GFLOPs（约 6.3×）**——
容量改变的是算力，本表让它可见而不是假设。

### 3.2 wall time

| seed | model | updates | selected | train s | s/update | 早停 | peak reserved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 41 | native_window | 800 | 700 | 295.112 | 0.368890 | 无 | 944 MiB |
| 41 | generic | 800 | 800 | 341.014 | 0.426268 | 无 | 968 MiB |
| 41 | process | 800 | 800 | 350.432 | 0.438040 | 无 | 968 MiB |
| 42 | native_window | 800 | 800 | 303.981 | 0.379977 | 无 | 944 MiB |
| 42 | generic | 800 | 700 | 332.397 | 0.415496 | 无 | 968 MiB |
| 42 | process | 800 | 700 | 345.015 | 0.431268 | 无 | 968 MiB |

合计 **1968.0 s = 0.547 GPU-h**（两卡并行，墙钟约 16 分钟/seed）。
单步 **0.369–0.438 s**，与 2.8M 的 0.292–0.377 s 相比只涨约 **1.16–1.26×**，
而参数量涨了 6.47×——瓶颈在 65×65×17 的读取/归一化与优化器开销，不在参数量。
这条对「扩容是否值得」是有用的事实：扩到 18M 的**墙钟代价远低于参数代价**。

### 3.3 例数

与 B2 逐 lead 相同（同一 val 段、同一 `max_samples`）：
**6h 26 / 12h 25 / 24h 23 / 48h 19 / 72h 15**；6 臂 × 5 lead × 2 seed 全部 `split=val`，
同 lead 内 case 数一致性由脚本断言。

### 3.4 训练曲线（完整 epoch 均值）

| seed | arm | epoch0 均值 | 末完整 epoch 均值 | 比值 |
| --- | --- | --- | --- | --- |
| 41 | native_window | 0.18125 | 0.10785 | 0.5950 |
| 41 | generic | 0.18035 | 0.09681 | 0.5368 |
| 41 | process | 0.18036 | 0.09533 | 0.5286 |
| 42 | native_window | 0.18229 | 0.10918 | 0.5990 |
| 42 | generic | 0.18247 | 0.09460 | 0.5185 |
| 42 | process | 0.18236 | 0.09624 | 0.5277 |

6/6 单调下降，降幅 **40–48%**（2.8M 是 36–61%）；**无早停**，说明 800 步上限
没有把 18M 掐在下降中途——即「18M 训练不稳定」不是一个可用借口。

## 4. 逐变量逐时效 RMSE 与胜负计数

### 4.1 关键变量（2 seed 均值，18M；每 lead 最优标 WIN）

**t2m（K）**

| lead | native_window | generic | process | persistence | climatology | WIN |
| --- | --- | --- | --- | --- | --- | --- |
| 6h | 3.3121 | **2.7787** | 2.8142 | 4.3121 | 2.8029 | **generic** |
| 12h | 3.9937 | 3.3154 | 3.3662 | 5.3116 | **2.7357** | climatology |
| 24h | 4.3711 | 4.6317 | 4.8800 | 2.8153 | **2.5511** | climatology |
| 48h | 6.5794 | 7.4087 | 8.3760 | 4.0730 | **2.6206** | climatology |
| 72h | 6.7389 | 8.1534 | 9.4508 | 4.5802 | **2.6472** | climatology |

**mslp（Pa）**

| lead | native_window | generic | process | persistence | climatology | WIN |
| --- | --- | --- | --- | --- | --- | --- |
| 6h | 188.4 | **162.5** | 166.3 | 223.0 | 502.8 | generic |
| 12h | 272.7 | **256.1** | 262.8 | 319.2 | 500.7 | generic |
| 24h | **425.9** | 499.5 | 507.4 | 456.0 | 505.2 | native_window |
| 48h | 643.4 | 648.2 | 692.0 | 704.5 | **535.8** | climatology |
| 72h | 989.2 | 878.1 | 963.1 | 849.1 | **568.0** | climatology |

**v850（m s⁻¹）**

| lead | native_window | generic | process | persistence | climatology | WIN |
| --- | --- | --- | --- | --- | --- | --- |
| 6h | **2.3822** | 2.5134 | 2.5153 | 2.4654 | 4.6135 | native_window |
| 12h | **3.6295** | 3.8263 | 3.8094 | 3.6325 | 4.6833 | native_window |
| 24h | 4.8626 | 4.6655 | **4.6432** | 5.1045 | 4.8139 | process |
| 48h | 5.6034 | **5.1252** | 5.2583 | 6.2767 | 4.6896 | climatology |
| 72h | 4.9367 | **5.1584** | 5.0854 | 6.0210 | 3.6488 | climatology |

### 4.2 胜负计数（17 变量，逐 lead，2 seed 均值）

| arm | lead | 胜 persistence | 胜 climatology | 同时胜两者 |
| --- | --- | --- | --- | --- |
| native_window | 6h | 14/17 | 16/17 | 13/17 |
| native_window | 12h | 12/17 | 16/17 | 11/17 |
| native_window | 24h | 9/17 | 8/17 | 6/17 |
| native_window | 48h | 8/17 | **0/17** | **0/17** |
| native_window | 72h | 8/17 | **0/17** | **0/17** |
| generic | 6h | 13/17 | **17/17** | 13/17 |
| generic | 12h | 13/17 | 16/17 | 12/17 |
| generic | 24h | 8/17 | 10/17 | 7/17 |
| generic | 48h | 9/17 | **0/17** | **0/17** |
| generic | 72h | 8/17 | **0/17** | **0/17** |
| process | 6h | 14/17 | 16/17 | 13/17 |
| process | 12h | 13/17 | 16/17 | 12/17 |
| process | 24h | 8/17 | 9/17 | 7/17 |
| process | 48h | 9/17 | **0/17** | **0/17** |
| process | 72h | 7/17 | **0/17** | **0/17** |

## 5. 核心判定：容量只在 6h/t2m 上「看起来」有效，且**不跨 seed 稳定**

### 5.1 扩容相对 B2 2.8M（同 case 直接相减）

| lead | B2 generic | B3 generic | 改善? | B2 native | B3 native | 改善? | climatology |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6h | 2.8313 | **2.7787** | **是** | 3.3998 | 3.3121 | **是** | 2.8029 |
| 12h | 3.3265 | 3.3154 | 是（微） | 4.0021 | 3.9937 | 是（微） | 2.7357 |
| 24h | 4.1765 | 4.6317 | **否** | 3.9756 | 4.3711 | **否** | 2.5511 |
| 48h | 6.8387 | 7.4087 | **否** | 5.4872 | 6.5794 | **否** | 2.6206 |
| 72h | 7.5339 | 8.1534 | **否** | 6.7077 | 6.7389 | **否** | 2.6472 |

85 格中 `improved_vs_b2_2p8M`：native_window **36**、generic **41**、process **41**。
即**约半数格改善**；但长时效（24–72h）在 t2m 上**系统性变差**。

### 5.2 「generic 18M 在 t2m 6h 赢了 climatology」——**逐 seed 看就塌**

这是本次唯一看似支持解释 A 的格，必须按 seed 拆开：

| seed | generic 18M t2m 6h | climatology | delta | 判定 |
| --- | --- | --- | --- | --- |
| 41 | 2.7431 | 2.8029 | **−0.0599** | WIN |
| 42 | 2.8144 | 2.8029 | **+0.0115** | **LOSS** |

均值赢是**两个 seed 方向相反**的产物。按 #60 比较器的判据
（每个 seed delta 必须同号才算 established），**该格是 unresolved，不是胜**。

同一效应在 `process` 上更明显：seed 41 = 2.567（大胜 −0.236），
seed 42 = 3.0615（大败 +0.259）——**单 seed 结论会完全相反**。

作为对照，B2 的 2.8M `generic` 在 t2m 6h 也是同一形态：
seed 41 WIN（−0.0144），seed 42 LOSS（+0.0954），seed 43 LOSS（+0.0041）。

### 5.3 判定

- **容量不足以解释 B2 的负结果**：18M 的 t2m 6h 改善幅度（约 0.05 K）
  **小于 seed 噪声**（同格 ±0.06 K），12h 的改善（0.011 K）同样在噪声内；
  24–72h 反而**系统性变差**。
- **解释 B（气候态在此数据范围内是硬靶）得到支持**：t2m 上
  **全部 5 个时效 × 全部 5 个神经臂**中，唯一不输的格是均值意义上的
  `generic 6h`，而它跨 seed 不稳定。
- **解释 A 未被证否的余地**：48/72h 上所有臂都输给 persistence 的一半格仍存在，
  且长时效 t2m 随容量变差——这更像**自由 rollout 的误差累积**问题
  （B2 已测：72h seed 噪声中位 22.7%），而不是容量问题。
- **状态分离**：本段**没有**任何「ours 胜出」的结论。能说的上限是
  「18M 在 6h 的均值接近 climatology，但差异在 seed 噪声内」。
  递归臂 vs 非递归臂的胜负**不在本段检验范围**（B3 无 K 消融）。

## 6. 显式局限（与 B1/B2 逐条一致）

1. **单一年 1 月**（2016-01），(month,hour) 桶只有 **4** 个；
   **climatology 在 t2m 上的强势是本数据范围的产物，不是「基线很强」的证据**。
2. val/test 是 2016 段内**工程再划分**，不是 v2 的 2019 验证年或 2021 test 候选；
   B3 全程 `test_read: false`。
3. 跨季节/跨年结论一律 **UNVERIFIED**。
4. **2 个 seed 不构成显著性检验**；本文所有「同号/异号」都是描述性判断。
5. **更新预算固定为 800、未按容量调整**：18M 可能只是还没训够，这是本段
   无法区分「容量不足」与「容量足够但步数不足」的地方——如实记录为未决。
6. 容量是唯一声明改变；FLOPs/墙钟已实测报告，未假设相等。
7. 5 条 B1 局限、7 条 B2 局限继续适用。

## 7. 预算与「是否还能继续扩」

| 项 | 值 |
| --- | --- |
| B3 训练 | 0.547 GPU-h |
| 累计（B0/B1 0.566 + B2 1.157 + B3 0.547） | **2.269 / 4.0 GPU-h = 56.7%** |
| 剩余 | 1.731 GPU-h |
| B3 新产物 | 约 1.1 GiB（6 个 18M checkpoint + 评估表）；累计新产物仍远低于 16 GiB |
| decoded | 未触发（B3 零下载） |

若要继续扩模：每加一臂每 seed 约 0.09 GPU-h（800 步 × 0.41 s），
剩余预算足以再跑约 19 个臂-seed。但**本段的证据不支持为追 t2m 而继续扩模**：
6h 的改善已在 seed 噪声内，长时效随容量变差，且解释 B 有 5×5 格的支持。

## 8. 未做的事

- 未做 K 消融（#65 的题目），因此**没有**递归 vs 非递归的结论。
- 未做 1→2→4 curriculum（#64 明文另设有界实验，且不得与 reasoning K 混淆）。
- 未用 test 调任何东西；未选 seed；未为通过而放宽任何判据。
- 未做 DDP 对照（两卡无 NVLink，独立 seed 并行更快；本段两卡各一 seed）。
- 未把 18M 的负结果归因到单一机制（容量/步数/自由 rollout 累积）——
  这是本段**未决**的部分，不写成结论。

## 9. 复现命令

```bash
# 成本先行
.venv/bin/python scripts/study_r7_b3_scale.py --mode probe \
    --manifests outputs/r7_b2_segment/store/manifests \
    --out outputs/r7_b3_scale/probe.json --device-index 0
# 两卡各一 seed（无 DDP）
for s in 41 42; do
  .venv/bin/python scripts/study_r7_b3_scale.py --mode seed --seed $s \
    --device-index $((s-41)) --manifests outputs/r7_b2_segment/store/manifests \
    --out outputs/r7_b3_scale/seed$s &
done; wait
# 四张表 + 跨容量对照（读 B2 自己的表，不重跑）
.venv/bin/python scripts/study_r7_b3_scale.py --mode compare \
    --manifests outputs/r7_b2_segment/store/manifests --out outputs/r7_b3_scale
```

产物：`outputs/r7_b3_scale/{probe.json, b3_parameter_table.csv, b3_flops_table.csv,
b3_wall_time_table.csv, b3_case_count_table.csv, b3_rmse_table.csv,
b3_cross_scale_table.csv, b3_cross_scale_summary.json, seed{41,42}/}`。
