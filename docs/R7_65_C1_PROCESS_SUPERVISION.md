# #65 C1：过程监督（辅助权重 0 / 0.01 / 0.1）的最小消融

**状态：C1 完成；结论 mixed，且**没有**任何「process 胜出」的可建立结论。**

本文件记录一次**预注册**的有界实验。C2/C3 见各自章节（`docs/R7_65_ABLATION.md`）。

| 项 | 值 |
| --- | --- |
| 代码 SHA | `f292b1a`（harness 与诊断），运行时 model digest `d9fb07f2…f665` |
| 协议 digest | **`d0a59c9b21ffe1c336fd02aec93e39d1c40e5426679be3a94974f74b205a8e16`**（第一步之前写定；分析时重新推导并核对，未漂移） |
| 产物 | `outputs/r7_65_c1/`（protocol.json / ablation_result.json / 四张表 / rmse_table.csv）、`outputs/r7_65_c1_analysis/` |
| 训练 | 4 臂 × 3 seed × 800 update；**12/12 完成，无早停**；训练总时长 **3886.0 s ≈ 1.08 GPU-h**（单卡 RTX 3090） |
| 评估 | val only，`test_read: false`，5 个时效（6/12/24/48/72 h） |
| 数据 | B2 36 天段（2016-01），`data_identity = 50517d94…54cd6961` |
| 判据 | 见协议 `decision_rule`：**只有在每个声明 seed 上同号**才称「更好」 |

冻结的四臂（同一协议、同数据、同预算、同调度）：

| 臂 | kind | 结构 | 辅助权重 |
| --- | --- | --- | --- |
| `generic16_aux0` | generic | 16 个 free latent token | 0.0（无辅助损失） |
| `process8_aux0` | process | 8 anchored + 8 free | **0.0** |
| `process8_aux001` | process | 同上 | **0.01** |
| `process8_aux010` | process | 同上 | **0.1** |

Ktrain = **4**（#65 C1 原文指定），`use_forecast_feedback=True`，`spatial_solver_feedback=False`。

---

## 1. 四张表（「同更新 ≠ 同算力」）

| 臂 | 参数量 | forward FLOPs | forward+bwd FLOPs | wall time（3 seed 均值） | 每时效例数 |
| --- | --- | --- | --- | --- | --- |
| generic16_aux0 | 2,799,202 | 14,162,646,912 | 42,374,196,480 | 321.6 s | 15/19/23/25/26 |
| process8_aux0 | 2,799,779 | 14,162,671,488 | 42,374,270,208 | 328.7 s | 15/19/23/25/26 |
| process8_aux001 | 2,799,779 | 14,162,671,488 | 42,374,270,208 | 327.1 s | 15/19/23/25/26 |
| process8_aux010 | 2,799,779 | 14,162,671,488 | 42,374,270,208 | 317.2 s | 15/19/23/25/26 |

**算力对齐是可证的，不是假设的**：process 相对 generic **+0.0206% 参数**、
forward FLOPs 比 **1.000002**、forward+bwd FLOPs 比 **1.000002**。
三条 process 臂的 forward FLOPs **逐位相同**（三档权重只改损失标量，不改图）。
wall time 的 317–329 s 离散（±2%）是 GPU 抖动，不能读成「权重越大越好/越慢」。
**每个时效的例数在所有臂上完全一致**（15/19/23/25/26），所以配对比较不是在不同
case 集上做的。

## 2. 配对结果（#60 比较器，reference = `generic16_aux0`）

`training/r7_coreasoning_compare` 是唯一口径：全键（arm/variable/unit/lead）配对、
按 seed id 配对、缺项 fail closed。每时效 **51 个变量×unit 组合**。

| 臂 | 时效 | improved | worsened | unresolved |
| --- | --- | --- | --- | --- |
| process8_aux0 | 6h | 1 | 2 | 14 |
| | 12h | 1 | 2 | 14 |
| | 24h | 0 | 3 | 14 |
| | 48h | 2 | 7 | 8 |
| | 72h | 1 | 6 | 10 |
| process8_aux001 | 6h | 3 | 4 | 10 |
| | 12h | 2 | 4 | 11 |
| | 24h | 2 | 3 | 12 |
| | 48h | 7 | 3 | 7 |
| | 72h | 6 | 1 | 10 |
| process8_aux010 | 6h | 4 | 4 | 9 |
| | 12h | 3 | 4 | 10 |
| | 24h | 2 | 2 | 13 |
| | 48h | 7 | 1 | 9 |
| | 72h | 7 | 2 | 8 |

**读法（按协议 `decision_rule`，主变量 t2m，训练时效 6h）**：

| 臂 | t2m@6h 三 seed delta | 判定 |
| --- | --- | --- |
| process8_aux0 | −0.168 / +0.062 / −0.185 | **unresolved**（跨 seed 翻号） |
| process8_aux001 | −0.029 / +0.007 / −0.090 | **unresolved**（跨 seed 翻号） |
| process8_aux010 | **+0.128 / +0.348 / +0.050** | **worsened**（3 seed 同号，一致变差） |

**没有任何一臂在主变量上同号改善。** 唯一「同号」的主变量结果方向是 **变差**
（`process8_aux010` 在 t2m@6h 上三 seed 一致 +0.05~+0.35 K）。
把辅助权重从 0 提到 0.1 并没有让 process 追上 generic，**在训练时效上反而更差**。

长时效（48h/72h）上 `aux001`/`aux010` 出现 improved 略多于 worsened（7:3、7:1、7:2），
**但**：①这些格子的 seed 内离散本就达 20–23%（B2 已测），②同一行的 `aux0`（即
**完全没有辅助损失**的 process 臂）在 48h/72h 是 2:7、1:6 —— 也就是说这三条 process 臂
之间的差异方向**在权重轴上不单调**（aux0 与 aux010 长时效相反），
③协议要求的「每 seed 同号」多数格子并不满足（counting 表里的 improved 只是
`direction == improved`，即三 seed 同号改善；未同号者归 unresolved）。
因此**不建立**「辅助监督在长时效有用」的结论。

## 3. 训练曲线（完整 epoch 均值）

`epoch_mean_loss` 为**完整 epoch 的均值**（每 epoch 14 个 update，最后一轮取整）、
`first/last full epoch` 为 0→9，`ratio` = 末/首完整 epoch 均值：

| seed | 臂 | 权重 | 选中 update | ratio |
| --- | --- | --- | --- | --- |
| 41 | generic16_aux0 | 0.0 | 500 | 0.6302 |
| 41 | process8_aux0 | 0.0 | 800 | 0.6323 |
| 41 | process8_aux001 | 0.01 | 500 | 0.6071 |
| 41 | process8_aux010 | 0.1 | 500 | **0.5099** |
| 42 | generic16_aux0 | 0.0 | 700 | 0.6197 |
| 42 | process8_aux0 | 0.0 | 700 | 0.6224 |
| 42 | process8_aux001 | 0.01 | 500 | 0.5958 |
| 42 | process8_aux010 | 0.1 | 500 | **0.4832** |
| 43 | generic16_aux0 | 0.0 | 600 | 0.6189 |
| 43 | process8_aux0 | 0.0 | 800 | 0.6236 |
| 43 | process8_aux001 | 0.01 | 600 | 0.5908 |
| 43 | process8_aux010 | 0.1 | 800 | **0.4784** |

**关键观察**：`aux010` 的**训练损失**下降显著快于其他臂（ratio 0.478–0.510 vs 0.619–0.632），
但它的**验证集 t2m 反而一致变差**。这不是矛盾，而是与预诊断第 1 节一致：
辅助损失里含两个被 floor 压成近常量的水汽通道，且这 8 个 proxy 标签是**输入时刻**的
确定性摘要、**不是**未来真值（`docs/R7_PROCESS_DIAGNOSTICS.md` 已界定）。
优化它等于让 readout 去拟合与预报目标无关的量，训练损失因此更好看，**不构成泛化收益**。
这是「训练损失 ≠ 任务收益」的又一实例，本文件不把它当作正面证据。
（另注：`epoch_mean_loss` 是含辅助项的**总损失**，`aux010` 的这一项本身量级更大，
所以三条臂的 ratio 并不在完全相同的被优化对象上比较——这也是 `aux0` 与 `aux0x`
之间不可直接以 ratio 排序的原因。）

## 4. 结论（#65 C1 判据）

按 #65 明文允许的四种终态（positive / mixed / negative / inconclusive），
**C1 判定为 mixed，偏向 negative**，逐条：

1. **打开辅助监督没有让过程结构胜出**：主变量 t2m@6h 上，`aux0`/`aux001` 跨 seed 翻号
   （unresolved），`aux010` 三 seed **一致变差**。无「同号改善」的格子。
2. **「process 结构本身」与「generic」的差异依然不可建立**：`process8_aux0` 与
   `generic16_aux0` 只差一个投影 + 一个死 readout，本次 800 步 3 seed 下 6h 仍只有
   1:2:14（improved:worsened:unresolved），与 B2 的 73/85 unresolved 同向。
   这与预诊断 §4.1 的结构性解释一致。
3. **算力已对齐**（+0.02% 参数、FLOPs 比 1.000002、例数相同），所以本次的 mixed
   **不能**再用「算力没对齐」解释——这一点与 B3 不同，是本轮新增的可确定性。
4. **不能声称辅助监督有效，也不能声称它有害到可以排除**：`aux010` 在训练时效一致变差
   是本次唯一同号结论；但长时效方向不单调、且水汽标签不可用，故不做因果归因。

## 5. 局限

| 局限 | 说明 |
| --- | --- |
| 单一年 1 月、4 个 (month,hour) 桶 | climatology 占优仍不可外推；跨季节 UNVERIFIED |
| val/test 是 2016 段内工程再划分 | `test_read: false`；不是封存测试集 |
| 800 步非收敛 | 12/12 无早停，是有界检查点 |
| 3 seed 不是显著性检验 | 同号一致性是描述性判据 |
| 辅助损失含 2/8 个不可用通道 | 见 `docs/R7_65_PREDIAGNOSTIC.md` §1；**未修归一化**（决策 D-1） |
| 总损失含辅助项 | 三条臂的 epoch ratio 不在同一被优化对象上 |

## 6. 复现

```bash
.venv/bin/python scripts/study_r7_65_ablation.py --phase c1 --device cuda \
    --out outputs/r7_65_c1
.venv/bin/python scripts/analyze_r7_65_ablation.py --run outputs/r7_65_c1 \
    --reference generic16_aux0 --out outputs/r7_65_c1_analysis
```

分析器会重新推导 `protocol.json` 的 digest 并与 `ablation_result.json` 比对；
两者不一致即拒绝分析（防止事后编辑协议）。

## 7. 完整 epoch 均值曲线（逐 seed，17/17 完整 epoch）

`epoch_mean_loss` 是**完整 epoch（14 update）的均值**；每臂每 seed 都有 17 个完整
epoch（首 = epoch 0，末 = epoch 16；最后一个不完整 epoch 不计入）。

| 臂 | 首 epoch 均值（3 seed 平均） | 末 full epoch 均值 | ratio | 逐 seed（首→末，选中 update） |
| --- | --- | --- | --- | --- |
| generic16_aux0 | 0.1828 | 0.0893 | 0.4886 | s41 0.1832→0.0905 (500)；s42 0.1831→0.0884 (700)；s43 0.1823→0.0891 (600) |
| process8_aux0 | 0.1828 | 0.0898 | 0.4911 | s41 0.1830→0.0908 (800)；s42 0.1832→0.0888 (700)；s43 0.1822→0.0897 (800) |
| process8_aux001 | 0.1914 | 0.0899 | 0.4697 | s41 0.1908→0.0910 (500)；s42 0.1919→0.0895 (500)；s43 0.1915→0.0891 (600) |
| process8_aux010 | **0.2612** | 0.1016 | **0.3890** | s41 0.2538→0.1024 (500)；s42 0.2660→0.1010 (500)；s43 0.2639→0.1015 (800) |

**注意** `aux010` 的首 epoch 均值明显更高（0.2612 vs 0.1828），因为它的总损失**含有
被加权的辅助项**（权重 0.1），而辅助项本身量级更大。因此**跨权重比较 ratio 是不成立的**
（不是同一个被优化对象）；同一权重下逐 seed 的比较才是有效的。全部 12/12 无早停。

原始曲线在 `outputs/r7_65_c1/seed*/training/seed*/<arm>/training_report.json`
的 `losses` 数组内（逐 update，含 lr），上表是其完整 epoch 聚合。
