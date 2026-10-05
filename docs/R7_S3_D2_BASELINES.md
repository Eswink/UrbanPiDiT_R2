# S3-D2：v2 确认实例上的 train-only 气候态与 persistence 重建（CPU，零 GPU）

**状态：已完成并登记（2026-10-05）；`scientific_claim: false`；test（2023）未读。**
本页对应 `docs/goals/s3-confirmation-baselines-and-candidate.md` 的 D2 交付物。
它记录**参数自由的参考基线读数**，不是模型结果、不是科学声明。

## 1. 结论摘要

- 在 v2 确认实例的 **val（2022）** 上，用与后续模型比较完全相同的评分路径
  （`training/r7_evaluate.evaluate_local`）重建了两个参考：
  **train-only (month,hour) 逐格均值气候态**（只 fit 2017 train）与 **persistence**（最后合法历史帧）。
- 逐 lead 完整覆盖，无收窄：6h 472/472、12h 468/468、24h 460/460、48h 444/444、72h 428/428
  （每 lead 独立 cohort，非全 lead 交集）。
- 气候态桶覆盖完整：16 个 (month,hour) 桶（Jan/Apr/Jul/Oct × 00/06/12/18 UTC）各 30 个 train 步；
  fail-closed 无缺桶回退。两次运行（climatology/persistence）拟合的 climatology 身份逐字节一致。
- t2m 参考 RMSE（K，full）：气候态 6/12/24/48/72h = 3.5404 / 3.5083 / 3.4526 / 3.3522 / 3.3117；
  persistence = 4.5470 / 5.9235 / 3.1633 / 4.1795 / 4.7269。persistence 相对气候态的 skill
  仅在 24h 为正（+0.1605），其余 lead 为负（6h −0.6495、12h −1.8507、48h −0.5545、72h −1.0373）。
- 成本：整轮 **553.1 s**（planned 1800 / hard 3600，soft overrun **0.0**）；GPU-h **0.0000**；
  网络 **0**（离线，`deny_network` 进程内生效）；test 从未打开。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 协议 | `outputs/r7_s3_d2_baselines_20261005_attempt01/protocol.json`，`protocol_sha256 dba134239c254f1cd909e7b4f86631ef10ee79746b2acb3552b84a6b94d38ca3`（evaluated 前写入，`'x'` 排他） |
| store | `outputs/r7_s3_confirmation_2017_2022_2023_v2/store/cache.zarr` |
| train manifest | `.../store/manifests/train.jsonl`（身份 `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09`） |
| val manifest | `.../store/manifests/val.jsonl` |
| 源 | `outputs/r7_s3_confirmation_2017_2022_2023_v2/source.nc`，`e0b51616…` |
| 设备 | CPU（`device_name="cpu"`，无 CUDA 设备打开） |
| 结果 | `result.json`（含每 lead 的 rmse/skill CSV 的 SHA256、桶计数、成本） |

## 3. 评分路径与判据

- 两个参考都经 `evaluate_local` 的 `baseline='climatology' | 'persistence'` 分支：
  气候态不是 rollout，而是按 valid-time 取桶的固定场；persistence 复制最后合法历史帧。
- 评分变量/单位/空间权重与模型评估完全同路径（17 通道、物理单位、cos(latitude) 归一化权重），
  因此后续 incumbent/candidate 的 `mse_skill` 与本页读数逐 case 可比。
- 一致性自证：climatology 臂对自身 skill 恒为 0.0000（同行 rmse_forecast == rmse_climatology），
  证明参考构造无泄漏；两臂每个 lead 的 `n_evaluated == n_available_windows`（无静默丢 case）。
- 每 lead 同时写出 `climatology_skill.csv`（rmse_forecast / rmse_climatology / mse_skill / unit /
  n_initializations）与 `acc.csv`；`acc_skill_identity` 一致性检查在 `provenance.json`。

## 4. t2m 读数（K、region full；解释见 §6）

| lead | rmse_climatology | rmse_persistence | persistence skill vs clim | n_eval |
| ---: | ---: | ---: | ---: | ---: |
| 6h | 3.5404 | 4.5470 | −0.6495 | 472 |
| 12h | 3.5083 | 5.9235 | −1.8507 | 468 |
| 24h | 3.4526 | 3.1633 | **+0.1605** | 460 |
| 48h | 3.3522 | 4.1795 | −0.5545 | 444 |
| 72h | 3.3117 | 4.7269 | −1.0373 | 428 |

**只看 t2m 数字的形状**（不构成任何判据）：气候态 RMSE 随 lead **下降**（3.54→3.31），
这是 30 天季节块内样本的统计性质（valid-time 越靠后的 2022 块平均越接近该年四个月的均值），
不是"越远越容易报"的物理结论；persistence 在 24h 附近偶然接近该均值场。

## 5. 气候态拟合内容（train-only 自证）

- `kind = train-only-month-hour-grid-mean-v1`，`selection = declared_train_years`，
  `training_years = [2017]`，`n_selected_steps = 480`（= v2 train 的 480 帧。
  train manifest 472 个 6h 窗口覆盖 480 帧中的 472 个 target；气候态按 store 的 train **年份**
  选择帧，等价于只读 2017）。
- 16 桶 × 30 = 480，与 train 时间轴（2017 四个 30 天块 × 4 UTC）精确对应；无缺桶、无 NaN。
- val/test 的每个 (month,hour) 桶都被 train 覆盖（Jan/Apr/Jul/Oct × 00/06/12/18），
  因此 fail-closed 评分可完整运行——这是 v2 年模式切分的前提检查通过。

## 6. 已确认与推测

**已确认**：val 五个 lead 的完整 case 覆盖；气候态只 fit 2017 train 且桶覆盖完整；
两臂共用同一 climatology 身份；persistence 与气候态在相同病例上逐 case 配对；
成本 553.1 s、GPU-h 0、网络 0；test 未读。

**推测（不写成事实）**：气候态 RMSE 随 lead 下降的成因（季节块内平均 vs 2022 样本分布）
未在本轮证明；persistence 在 24h 的优势是否稳定未检验。两者都不影响后续判据——判据只比较
模型与气候态的逐 case MSE。

## 7. limitations（如实）

- 单 ROI、三年度、17 通道；val 仅 2022。气候态是 (month,hour) 网格均值，不是 WeatherBench2 复现。
- 本页是参考读数，**不**证明任何模型优于或劣于参考；也不改变 S4 门的任何阈值。
- 每 lead 的 cohort 各自完整但不相同（472/468/460/444/428）；跨 lead 的绝对值不可直接相加比较。
- `max_samples` 传入 `10**9` 仅为"显式无截断"声明；实际受 `len(ds)` 限制（逐 lead 打印覆盖）。

## 8. 证据指针

- 协议与结果：`outputs/r7_s3_d2_baselines_20261005_attempt01/protocol.json`、`result.json`
- 逐臂逐 lead 表：`outputs/r7_s3_d2_baselines_20261005_attempt01/{climatology,persistence}/lead_*/`
  （`rmse.csv`、`climatology_skill.csv`、`acc.csv`、`provenance.json`）
- 驱动器：`scripts/study_r7_s3_d2_baselines.py`
- 实例：`docs/R7_S3_CONFIRMATION_INSTANCE.md`
