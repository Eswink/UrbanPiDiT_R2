# S3 Stage R 可行性探针：**按相干性反演振幅在 2022 val 上不成立**

- **日期**：2026-11-11（UTC）
- **状态**：`not-opened` —— R 的**前置探测**（不是 R 本身）已跑完，结论是否决
- **`scientific_claim: false`；`test_read: false`；`r = 0`；GPU-h = 0**（纯 CPU）
- **判据来源**：`docs/goals/s3-physics-informed-campaign.md` §3.3（R 的出口）、plan 0030 §3.3
- **本轮 Stage P**：`docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md`

---

## §0 一句话

Stage R 的前提是"按相干性反演振幅"能在不劣化 skill 的前提下补回缺口。在 **2022 val** 的 20 个窗口上实测，
六个频带的 `amp_ratio` 虽然稳定（std 0.061–0.111）且乘子有界（1.007–1.318），但
**band 5 几乎没有缺口（0.9933）且 8/20 窗口过冲、band 0/1 各有 2–3 例过冲**——
合计 **13/120 带×窗口样本 `amp_ratio > 1`**，在这些样本上乘子 `1/ratio < 1` 会**放大**误差。
**R 不开轮。**

---

## §1 为什么先做可行性探测而不直接开 R

plan 0030 §3.3 给 R 写的开轮前置是"P1 证实 `amp_ratio ≈ coh`"。P1 给的是**相容而非等式**：
冻结 5 lead 口径下 `mean|amp − coh|` 为 0.037–0.048、`max` 0.149–0.179
（见 `docs/R7_S3_SPECTRAL_GEOMETRY_RESULT.md` §3.4）。

在"前置只到相容"的情况下直接花 5–15 GPU-h 做完整评估，是把验收压力全部放在出口上。
所以先花 **0 GPU-h** 问三个决定成败的问题：缺口稳不稳、符号对不对、乘子有没有界。
**它是 R 的前置探测，不是 R 本身**——不产生 `posthoc_amplitude_correction.json`。

---

## §2 方法（可复算）

```bash
.venv/bin/python scripts/r7_s3_amplitude_correction_feasibility.py --windows 20 --out <out>/r_feas.json
```

- **数据**：`outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/val.jsonl`
  （**2022 val**；2023 test 从未打开）
- **入口**：`training.r7_evaluate.EvaluationPreparation` + `ZarrRolloutDataset` —— 与**评分路径同一个**
  窗口定义，所以谱读数与 skill 读数不会来自两套"验证窗口"定义。
  `ZarrLongRolloutDataset` 拒绝非 `train.jsonl` 清单，这个守卫是刻意的，不绕。
- **前向**：`model.r7_rollout.autoregressive_rollout` + `rollout_model_input`，`model.eval()`。
  这条很关键：`training_long_rollout` 要求 `train()` 而评估 rollout 要求 `eval()`，
  R 要校正的是**部署后的**预报，所以谱必须是 eval 模式的。冻结 spec 的 `dropout` 为 0.0
  且 trunk 无 BatchNorm（两者都断言），因此两模式数值一致——但读数仍按评估方式取。
- **端点**：p1600（128,528 updates，`model_code_sha256 59340710…`），与 P1 同一端点。
- **分带**：决策 0110 的径向 rfft2 分带，`band_edges(65,65)` = `[0,1,2,4,8,16,45.2548]`。
- **量**：`amp_ratio = sqrt(PSD_pred/PSD_target)`，逐窗口逐带；乘子 = `1/amp_ratio` 的窗口均值倒数。
- **D2 分母**：`outputs/r7_s3_fy_year_coverage_d2_20261023_attempt01/reference.json`，
  `n_selected_steps = 32144`（与 plan 0030 一致，本探针只读不改）。

---

## §3 主结果

| 带 | `amp_ratio` 均值 | std | 范围 | 过冲窗口 | 乘子 `1/ratio` | 乘子有界 |
| ---: | ---: | ---: | --- | ---: | ---: | :---: |
| 0 | 0.9274 | 0.0616 | 0.826–1.030 | **2/20** | 1.078 | 是 |
| 1 | 0.8662 | 0.1111 | 0.697–1.105 | **3/20** | 1.155 | 是 |
| 2 | 0.7588 | 0.0743 | 0.633–0.898 | 0/20 | 1.318 | 是 |
| 3 | 0.7770 | 0.0638 | 0.682–0.903 | 0/20 | 1.287 | 是 |
| 4 | 0.8095 | 0.0608 | 0.722–0.941 | 0/20 | 1.235 | 是 |
| 5 | **0.9933** | 0.0776 | 0.847–1.119 | **8/20** | 1.007 | 是 |

- 6/6 带 `std < 0.15` ⇒ 缺口**确实是模型的性质**，不是窗口噪声
- 6/6 带乘子落在 `[0.2, 5.0]` ⇒ 没有外推
- 但 **13/120 带×窗口样本 `amp_ratio > 1`**（10.8%）

---

## §4 三条否决理由（都是实测）

### §4.1 band 5 没有缺口，且 8/20 窗口过冲

band 5 的均值 0.9933 意味着该带的振幅几乎已经匹配真值。对它施加 `1/0.9933 = 1.007` 的乘子
既无收益，而在那 8 个过冲窗口上乘子 `< 1`，会把已经偏大的振幅**继续压低**——
在"不劣化 24/48/72 h"的出口下这是直接的失败风险。band 0 与 band 1 各有 2–3 例同类问题。

### §4.2 val 与 train 端点的谱不一致

| 带 | P1 的 p1600（train） | 本探针的 val |
| ---: | ---: | ---: |
| 1 | 0.7784 | 0.8662 |
| 2 | 0.6845 | 0.7588 |
| 3 | 0.6684 | 0.7770 |
| 4 | 0.6989 | 0.8095 |
| 5 | 0.8635 | **0.9933** |

R 的原设计是**train 年拟合、2022 val 报告**。若在 train 上拟合乘子（如 band 5 得 1.158）再套到 val
（真实需要 1.007），该校正在 val 上会**过度放大**。这不是实现细节，是"拟合分布 ≠ 报告分布"的
经典失效——而 R 的出口恰好只测报告分布。

### §4.3 缺口是欠训练成分占主导，缩放不补相位

P1 已测出缺口随训练量单调缩小（0.27871 → 0.22690），即它主要是欠训练而非损失几何造成的。
后处理缩放能把**方差**调到正确，但 `coh` 在这些带上并未随之提高（P1 §3.4：`coh` 与 `amp_ratio`
的差在 0.037–0.048 之间且不随训练量收敛到 0）。**缩放不改变相关**，所以它买到的是一个
"振幅对、相位错"的场——在 RMSE 上可能略有改善，在 skill（以 climatology 为分母）上收益不确定。

---

## §5 限制（如实列）

1. **只读了 20 个 val 窗口**（全量 1,458）。可行性的三条判断在 20 个窗口上已经稳定（std ≤ 0.111），
   但过冲比例 13/120 的**置信区间很宽**，不得当作精确比率引用。
2. **只看 band 0–5 这六个径向带**（决策 0110 的冻结分带），未扫描其他分带。
3. **只测 p1600 一个端点、一个 seed**。P1 的四个端点显示缺口随训练缩小，所以更弱的端点上缺口更大、
   过冲更少——本页的否决**不能**自动推广到其他端点。
4. **未测校正后的 skill 与 RMSE**。本页是否决理由，不是"校正一定更差"的实测；
   它说明的是**前置条件不成立**（乘子在不同带上前提不同、且约 11% 样本反向），
   因此 R 的出口无法在预声明形式下被有意义地检验。
5. **接缝能量占比 0.9921**（Stage P 实测）同样作用于本页的谱量：绝对数值带未量化偏差，只能作相对比较。
6. `scientific_claim: false`、`test_read: false`、`r = 0`；2023 test 未读。

---

## §6 判定

| 项 | 判定 |
| --- | --- |
| R 的前置（`amp_ratio ≈ coh`） | **相容而非证实**（`mean|amp − coh|` 0.037–0.048） |
| 校正乘子的稳定性 | 稳定（6/6 带 std < 0.15） |
| 校正乘子的方向正确性 | **不成立**（13/120 样本 `amp_ratio > 1`，其中 band 5 占 8 例） |
| **Stage R** | **`not-opened`、0 GPU-h、不产生 `posthoc_amplitude_correction.json`** |

**怎么做才可能成立**（留给下一轮，需新 ADR）：只在 `amp_ratio < 1` 稳定的 mid-band（2–4）上校正，
并把出口改成"这三个带的 skill 与 RMSE 不劣化"而不是"五 lead 全部不劣化"。
那是对 R 的**重新定义**，不是本轮可以顺手做的事。

---

## §7 复现

```bash
# 20 个 val 窗口（本页数字）
.venv/bin/python scripts/r7_s3_amplitude_correction_feasibility.py --windows 20 --out <out>/r_feas.json

# 4 个窗口（快速形状检查）
.venv/bin/python scripts/r7_s3_amplitude_correction_feasibility.py --windows 4 --out <out>/r_feas4.json

# 测试
.venv/bin/python -m pytest tests/test_r7_s3_amplitude_correction_feasibility.py -q
```
