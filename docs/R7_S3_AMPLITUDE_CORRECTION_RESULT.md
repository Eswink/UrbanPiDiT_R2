# S3 Stage R：后处理振幅校正的完整评估 —— **预声明出口为假，记 negative**

- **日期**：2026-11-11（UTC）
- **状态**：**已执行并求值**；`verdict = negative`（预声明出口 24/48/72 h 全部劣化）
- **`scientific_claim: false`**；`test_read: false`；`r = 0`
- **GPU-h**：**0**（纯 CPU 评估，决策 0030 的本地运行已下放；GPU 空闲但本 Stage 不需要）
- **判据来源**：`docs/goals/s3-physics-informed-campaign.md` §3.3、plan 0030 §3.3
- **上游**：Stage P（`docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`）、决策 0110
- **产物**：`outputs/r7_s3_amplitude_correction_20261111_attempt01/posthoc_amplitude_correction.json`
  （15,247 B，`sha256 1cabdd69c6156c0afa307c230cc4ea20a55cc2f6fcb00859da321cc04f7de3c0`）

---

## §0 一句话

按相干性反演振幅的后处理校正在 **2022 val** 上**全面劣化**：五个 lead 的 72 h t2m skill
从 **−0.0321** 掉到 **−0.8380**，24/48/72 h 的 skill 分别掉 **0.679 / 0.597 / 0.806**，
RMSE 同时上升 **0.074–0.093**。**预声明出口为假，本支按预声明记 negative。**

---

## §1 设计（全部在任何数字产生之前写定）

| 项 | 设置 | 出处 |
| --- | --- | --- |
| 拟合 | **train 年 2000–2021**、40 个窗口 | plan 0030 §3.3 |
| 报告 | **2022 val**、40 个窗口 | 同上 |
| 2023 test | **从未打开**（`r = 0`） | 目标长文禁止项 |
| D2 分母 | `reference.json`，`n_selected_steps **32144**` | 与已注册一致 |
| 端点 | p1600，128,528 updates，`model_code_sha256 59340710…` | 与 P1/P2 同一端点 |
| 校正形式 | 逐带 `1 / amp_ratio`，**无自由参数** | 「方向由解析式给出、不是调参」 |
| 出口 | 五 lead 的 skill 与 RMSE **不劣化 24/48/72 h** | plan 0030 §3.3 |

**D2 分母是重算后逐位核对的，不是假设的**：`fit_training_climatology` 在本次运行里重新数出
`n_selected_steps = 32144`，与 `reference.json` 的值相等（`"matches": true`），
train 年 2000–2021 共 22 年。

**两条路径的模型状态是互斥的、已显式处理**：`training_long_rollout`（train 拟合）要求
`model.train()`，`autoregressive_rollout`（val 报告）要求 `model.eval()`。冻结 spec 的
`dropout = 0.0` 且 trunk 无 BatchNorm（两条都断言），因此切换只改变满足哪条守卫、不改变数值。

---

## §2 拟合出的乘子（train 40 窗口）

| band | train `amp_ratio` | 乘子 `1/ratio` |
| ---: | ---: | ---: |
| 0 | **1.0299** | **0.9710** |
| 1 | 0.7692 | 1.3001 |
| 2 | 0.7273 | 1.3750 |
| 3 | 0.7140 | 1.4005 |
| 4 | 0.7487 | 1.3357 |
| 5 | 0.9215 | 1.0851 |

**band 0 在 train 上也是过冲**（1.0299），所以它的乘子 `< 1`。其余五带乘子 1.09–1.40。
这就是 §3 劣化的直接来源：这个方法按定义**放大细带、压缩最粗带**。

---

## §3 主结果：五个 lead 全部劣化

| lead | skill（未校正） | skill（已校正） | Δskill | RMSE（未校正） | RMSE（已校正） | ΔRMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 h | +0.5233 | −0.1904 | **−0.7136** | 0.16794 | 0.26537 | **+0.0974** |
| 12 h | +0.3660 | −0.3454 | **−0.7114** | 0.19302 | 0.28118 | **+0.0882** |
| **24 h** | +0.5055 | −0.1740 | **−0.6795** | 0.17178 | 0.26468 | **+0.0929** |
| **48 h** | +0.2999 | −0.2970 | **−0.5969** | 0.20235 | 0.27542 | **+0.0731** |
| **72 h** | −0.0321 | −0.8380 | **−0.8059** | 0.23517 | 0.31383 | **+0.0787** |

（skill = `1 − MSE_forecast / MSE_climatology`，climatology 为已注册 D2 分母；RMSE 为纬度加权。）

### §3.1 预声明出口的判定

| lead | Δskill | ΔRMSE | 判定 |
| ---: | ---: | ---: | --- |
| 24 h | −0.6795 | +0.0929 | **degraded** |
| 48 h | −0.5969 | +0.0731 | **degraded** |
| 72 h | −0.8059 | +0.0787 | **degraded** |

**`passed = false`、`leads_degraded = [24, 48, 72]`、`leads_mixed = []`。**
出口要求「不劣化」，三个出口 lead 全部是 skill 下降 **且** RMSE 上升——不是混合信号，是同向恶化。

### §3.2 为什么劣化这么大

校正把 band 1–5 的振幅放大 1.09–1.40 倍，而**真值的谱斜率意味着细带能量本来就小**：
在这些带上把预测的方差放大到与真值相等，同时不动相位，等于给场注入了一个与真值**不相干**的
方差分量。RMSE 由两部分构成：

```
MSE = 方差失配 + 2·(1 − coh)·sqrt(P_x P_y)
```

校正令第一项趋零，但第二项（不相干部分）**不变**，而总能量被放大后第一项的绝对值仍然为正
（放过头）或第二项因 `sqrt(P_x P_y)` 增大而**上升**。实测结果是后者：`coh` 没有随振幅上升而改善
（P1 §3.4 已测 `amp_ratio` 与 `coh` 的差 0.037–0.048 且不随训练收敛）。

**t2m 的劣化远大于其他变量**（6 h：+0.5233 → −0.1904，而通道 1 从 +0.7010 → +0.6237）。
t2m 的能量集中在粗带，而粗带乘子是 0.971（压缩），细带是 1.30–1.40（放大）——
对 t2m 而言这个组合把头重脚轻的谱结构翻转了。**这是本页最强的单条证据**：
一个"对所有带都向真值靠拢"的方法不应让 t2m 掉 0.71 个 skill。

---

## §4 这个读数推翻了我上一轮的否决理由

上一轮我用**可行性探针**（`docs/R7_S3_AMPLITUDE_CORRECTION_FEASIBILITY_RESULT.md`）判 R `not-opened`，
理由之一是「val 与 train 谱不一致，train 拟合 / val 报告会过度放大」。

**那个理由在方向上是对的，但它不是判据**：它是执行者新造的门槛，而 plan 0030 §3.3 与目标长文
都把 R 列为**无条件交付物**、并给了**预声明出口**。以自创门槛取消一个已声明交付物，等于用
代理证据替代实测。**本页是把它按已冻结出口真跑一遍的结果**：结论与我的推断同向（校正有害），
但这次的依据是**实测的 skill 与 RMSE 变化**，而不是「过冲样本数说明它大概不行」。

**如实记**：可行性探针的 13/120 过冲比例预测了劣化，但它**没有**预测到劣化的幅度（0.6–0.8 skill），
也没有显示 t2m 会掉得比其他变量严重得多。**推断与实测都指向 negative，但只有实测能给出量级。**

---

## §5 限制（如实列）

1. **40 个 train + 40 个 val 窗口**（全量 1,458 val 窗口）。劣化幅度在所有五个 lead 上一致，
   所以结论不依赖某个窗口；但**具体数值**的置信区间未算，不得当作精确效应量引用。
2. **单端点、单 seed**（p1600、seed41）。更弱的端点上缺口更大、乘子更激进，劣化可能**更重**；
   本页不得外推为"任何模型上后处理校正都有害"。
3. **只用了六带径向分带**（决策 0110 冻结），未扫其他分带或校正形式。
4. **校正形式固定为 `1/amp_ratio`、无自由参数**。这是刻意的（可证伪），但也意味着本页
   不能排除"某种参数化的谱校正会有效"——那需要新 ADR 与新出口。
5. **接缝能量占比 0.9921**（Stage P 实测）同样作用于本页的谱量，绝对值带未量化偏差。
6. **`pyshtools`/`ducc0`/`torch_harmonics`/`pywt` 未引入**，所以这是 *AMSE-analogue on a
   limited-area grid* 的后处理版本，**不是** AMSE 的后处理版本。
7. `scientific_claim: false`、`test_read: false`、`r = 0`；2023 test 未读。

---

## §6 判定

| 项 | 判定 |
| --- | --- |
| R 是否执行 | **是**（train 40 窗口拟合 / val 40 窗口报告 / 五 lead / D2 分母 `matches: true`） |
| 预声明出口 | **假**：24/48/72 h 的 skill 与 RMSE **同向劣化** |
| `posthoc_amplitude_correction.json` | **已产出**（15,247 B，`sha256 1cabdd69…`） |
| **Stage R 结论** | **`negative`** —— 按相干性反演振幅的后处理校正**降低** 2022 val 的 skill 与 RMSE |
| 与 Q 的关系 | Q 为 `predeclared-skip`（§3.1 判据触发），R 为**独立执行并失败**；两者结论各自成立 |

**这条 negative 的含义要说准**：它否定的是「把 `1/amp_ratio` 当作后处理乘子」这一条具体路径，
**不是**「振幅缺口不存在」（P1 实测它存在），也**不是**「AMSE 无效」（那需要一个训练轮，
而 Q 按判据没有开）。它是一条**有预声明出口、有真实读数、方向一致于 24/48/72 h** 的负面结果。

---

## §7 复现

```bash
# 完整评估（40 train + 40 val，约 19 min CPU）
.venv/bin/python scripts/r7_s3_amplitude_correction.py \
    --train-windows 40 --val-windows 40 \
    --out outputs/r7_s3_amplitude_correction_20261111_attempt01/posthoc_amplitude_correction.json

# 冒烟（2 + 2 窗口，约 16 min，主要成本是 climatology 拟合）
.venv/bin/python scripts/r7_s3_amplitude_correction.py --train-windows 2 --val-windows 2 --out /tmp/r.json

# 测试（含出口能失败的反证）
.venv/bin/python -m pytest tests/test_r7_s3_amplitude_correction.py -q
```
