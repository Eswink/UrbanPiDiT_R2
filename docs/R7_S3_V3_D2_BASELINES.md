# S3 v3-D2：v3 五年实例上的 train-only 气候态与 persistence 重建（CPU，零 GPU）

**状态：已完成（2026-10-06）；`scientific_claim: false`；test（2023）未读。**
本页对应 `docs/goals/s3-confirmation-baselines-and-candidate.md` 的 v3 控制重建与
`docs/R7_S3_CONFIRMATION_INSTANCE_V3.md` 的 §10 下一步 1。它记录**参数自由的参考基线
读数**，不是模型结果、不是科学声明。

## 1. 结论摘要

- 在 v3 确认实例的 **val（2022）** 上，用与后续模型比较完全相同的评分路径
  （`training/r7_evaluate.evaluate_local`）重建了两个参考：
  **train-only (month,hour) 逐格均值气候态**（只 fit 2017–2021 五个 train 年）与
  **persistence**（最后合法历史帧）。
- 逐 lead 完整覆盖，无收窄：6h 472/472、12h 468/468、24h 460/460、48h 444/444、72h 428/428
  （每 lead 独立 cohort，非全 lead 交集）。
- 气候态桶覆盖完整：16 个 (month,hour) 桶（Jan/Apr/Jul/Oct × 00/06/12/18 UTC）**各 150 个
  train 步**（= 5 年 × 30 天，对比 v2 单年的各 30）；`n_selected_steps = 2400`；
  fail-closed 无缺桶回退。多次运行拟合的 climatology 身份逐字节一致（驱动断言）。
- t2m 参考 RMSE（K，full）：气候态 6/12/24/48/72h = 3.5171 / 3.4889 / 3.4375 / 3.3534 / 3.3184；
  persistence = 4.5470 / 5.9235 / 3.1633 / 4.1795 / 4.7269。persistence 相对 v3 气候态的 skill
  仅在 24h 为正（+0.1531），其余 lead 为负（6h −0.6714、12h −1.8825、48h −0.5533、72h −1.0291）。
- **与 v2-D2 的对比（描述性）**：persistence 的 RMSE 与 v2 完全相同（val 数据未变）；
  五年气候态比单年气候态整体略低 RMSE（例如 6h 3.5404 → 3.5171，−0.0233 K），
  即基线略强——这是预期方向，不构成模型结论。
- 成本：整轮 **992.4 s**（planned 1800 / hard 3600，soft overrun **0.0**）；GPU-h **0.0000**；
  网络 **0**（离线，`deny_network` 进程内生效）；test 从未打开。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 协议 | `outputs/r7_s3_v3_d2_baselines_20261006_attempt01/protocol.json`，`protocol_sha256 a14dd1b2900ba964337a21dd06debffcd47ab8435a0a8412ee31a96ccffb1ed8`（evaluated 前写入，`'x'` 排他） |
| store | `outputs/r7_s3_confirmation_train2017_2021_v3/store/cache.zarr` |
| train manifest | `.../store/manifests/train.jsonl`（身份 `2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac`） |
| val manifest | `.../store/manifests/val.jsonl` |
| 源 | `outputs/r7_s3_confirmation_train2017_2021_v3/source.nc`，`bc2ff9cf…`（3360 stamps） |
| 设备 | CPU（`device_name="cpu"`，无 CUDA 设备打开） |
| 结果 | `result.json`（SHA256 `6ae00858651e9bb1640da14247933e503a32128c10baf6421bf8b7084c8a1d9d`；含每 lead 的 rmse/skill CSV 的 SHA256、桶计数、成本） |

## 3. 评分路径与判据

- 两个参考都经 `evaluate_local` 的 `baseline='climatology' | 'persistence'` 分支：
  气候态不是 rollout，而是按 valid-time 取桶的固定场；persistence 复制最后合法历史帧。
- 评分变量/单位/空间权重与模型评估完全同路径（17 通道、物理单位、cos(latitude) 归一化权重），
  因此后续 v3 incumbent/candidate 的 `mse_skill` 与本页读数逐 case 可比。
- 一致性自证：climatology 臂对自身 skill 恒为 0.0000（rmse_forecast == rmse_climatology），
  证明参考构造无泄漏；两臂每个 lead 的 `n_evaluated == n_available_windows`（无静默丢 case）。
- 驱动新增断言：`training_years == [2017, 2018, 2019, 2020, 2021]`（不是 v2 的 `[2017]`）；
  五处运行的 climatology 身份逐字节一致才放行。

## 4. t2m 读数（K、region full；解释见 §6）

| lead | rmse_climatology | rmse_persistence | persistence skill vs clim | n_eval |
| ---: | ---: | ---: | ---: | ---: |
| 6h | 3.5171 | 4.5470 | −0.6714 | 472 |
| 12h | 3.4889 | 5.9235 | −1.8825 | 468 |
| 24h | 3.4375 | 3.1633 | **+0.1531** | 460 |
| 48h | 3.3534 | 4.1795 | −0.5533 | 444 |
| 72h | 3.3184 | 4.7269 | −1.0291 | 428 |

气候态 RMSE 随 lead 下降的形状与 v2-D2 相同（见 v2 页 §4 的解释），非"越远越容易报"的
物理结论；persistence 在 24h 附近偶然接近该均值场。

## 5. 气候态拟合内容（train-only 自证）

- `kind = train-only-month-hour-grid-mean-v1`，`selection = declared_train_years`，
  `training_years = [2017, 2018, 2019, 2020, 2021]`，`n_selected_steps = 2400`
  （= v3 train 的全部 2400 帧）。
- 16 桶 × 150 = 2400，与 train 时间轴（5 年 × 4 个 30 天块 × 4 UTC）精确对应；无缺桶、无 NaN。
- val/test 的每个 (month,hour) 桶都被 train 覆盖，fail-closed 评分完整运行。

## 6. 已确认与推测

**已确认**：val 五个 lead 的完整 case 覆盖；气候态只 fit 五个 train 年且桶覆盖完整；
多臂共用同一 climatology 身份；persistence 与气候态在相同病例上逐 case 配对；
成本 992.4 s、GPU-h 0、网络 0；test 未读；v2 与 v3 的 persistence 读数逐位相同（val 未变）。

**推测（不写成事实）**：五年气候态略强（6h −0.0233 K）的原因是更多年份的均值更接近
2022 样本的总体均值，未在本轮论证；这对后续判据的影响是双向的（基线更强 ⇒ 模型的
同数据 skill 更难为正），后续轮次的自报 skill 会体现这一点。

## 7. limitations（如实）

- 参考基线只覆盖 val（2022）单一年份、单 ROI、17 通道；没有跨区域/跨年泛化声明。
- 气候态是 (month,hour) 逐格均值，不是任何标准气候态产品的复现。
- 本页不产生任何模型结果；`scientific_claim: false`。
- 与 v2 数字的对比是描述性的（不同实例、不同 climatology），不是受控对比。

## 8. 证据指针

- 协议与结果：`outputs/r7_s3_v3_d2_baselines_20261006_attempt01/protocol.json`、`result.json`
- 逐 lead 产物：`.../climatology/lead_XXXh/`、`.../persistence/lead_XXXh/`（rmse.csv 与
  climatology_skill.csv 的 SHA256 在 result.json）
- 驱动：`scripts/study_r7_s3_v3_d2_baselines.py`（db0b463 起在版本控制内）
- 上游：`docs/R7_S3_CONFIRMATION_INSTANCE_V3.md`（实例身份链）

## 9. 下一步

1. v3-D3-analog：在 v3 train 上重训 400 更新 incumbent 控制（三 seed 逐位初始化保真），评 val 全 cohort。
2. v3 预算剂量筛选（1600 更新 vs v3 400 控制），检验「数据扩年清除长 lead 守门退化」假设；
   通过 primary+gate 合取则进 D5 S4 冻结包。
3. test（2023）保持封印，直到预注册的未见年份确认轮。
