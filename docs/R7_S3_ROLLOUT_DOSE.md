# S3 rollout-dose screen：200 → 800 更新仍支持，但长 lead 绝对气候态门未过

**状态：已完成（2026-10-09）；`scientific_claim: false`；2023 test 未读。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 的 S3 接续。它是**开发筛选读数**，
不是科学确认，也不是 S4 就绪声明。冻结的 reading 阶段失败（工程缺陷）已保留并全额计费；
本页的判读来自一个只读复算工具，见 §6。

## 1. 结论摘要

- 单因素：注册的 S3 long-rollout screen 在 200 更新时余弦日程已退到 0.1× 下限，因此只界定了
  一个剂量点。本轮把更新数改成 **800**，父 checkpoint、12 步物理权重、LR 2e-5、warmup 10、
  FP32、K4、batch 1、clip 1 与判定形式全部不变。
- 冻结判定（`primary_verdict` / `gate_pre_screen` / `advance_decision` 三个**同一个冻结函数**）：
  **primary = supported**（t2m/full 6h 与 12h 三 seed 相对 v3-D3 400 更新控制的 RMSE 增量全为负）；
  **守门 = passed**（u10/v10/mslp 在五个 lead、三 seed 的 45 格里 **0 个**正 cell，最大
  −0.104401）；**decision = advance-to-S4-freeze**。该决定只表示"可另准备冻结包"，
  与注册的 200 更新 screen 同形，不表示 S4 已就绪。
- **绝对气候态门仍未过**：t2m/full seed 均值 skill 为 6h **+0.5816**、12h **+0.3151**、
  24h **+0.3668**、48h **−0.1191**、72h **−0.4309**。
- **剂量响应仍为正但递减**：相对注册的 200 更新 screen，24/48/72h 的 RMSE 分别低
  0.154/0.325/0.465 K，6h/12h 持平（高 0.025/0.008 K）。48h 的 anomaly correlation 从
  0.329 升到 **0.367**、72h 从 0.153 升到 **0.189**——即增量更新确实抬高了 pattern 相关，
  不是只靠幅度校准。
- 全 17 变量在 24/48/72h 相对控制的相对 MSE 变化**全为负**（t2m 48h −0.492、72h −0.516）。
- 幅度比 σ_forecast/σ_observed 在 48/72h 已落到 0.87/0.87（200 更新时为 0.99/1.05，
  l6×400 时 1.42/1.55）：多剂量把残余过幅也修平了。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 执行代码 | `7e384dbeb6a7061a6cfa1a3e1898d44863f442eb`（工作分支 `r7/weather-reasoning`） |
| code.zip SHA256 | `bec1edd5767bbd2c19a92b11f29085ca3f9599d3659f9578147ad1cebe2b5d50` |
| canonical screen protocol | `655fec891491dd42c0d2bc840160bf6f72ceacf13b091d5bc105cd94cc7a74c9` |
| 冻结预算 | planned 9000 / hard 18000 / per-seed 4200 s |
| GPU | `GPU-408ad137-…`（GPU0，默认共驻；每 spawn 前只读余量门，全程未向邻居发任何信号） |
| 数据 | v3 实例 `2564eeaf…`、源 `bc2ff9cf…`；train 2017–2021 / val 2022；cohort 472/468/460/444/428 |
| 控制 | 注册 v3-D3 400 更新读数（按 SHA 复用，未重训） |
| 父 | 注册 v3-BD l6×1600；经 model-only 迁移复用，见 §3 |
| test | 未读（`test_read: false`；15 个 provenance 均为 `split: val`，无 test manifest 路径） |

## 3. 模型身份变更与 model-only 迁移

父 checkpoint 由修订 `66836d2`（`model_code_sha256 3ddab46b…`）训练；climatology-anchor 修订
把该 digest 改成 `d3fb58db…`，因此受审计的 `load_checkpoint` 会**按设计**拒绝它。本轮不改这个
守卫，而是做一次可审计的 model-only 迁移：

1. `scripts/export_r7_parent_state.py` 在归档修订 `66836d2` 下、用该修订自己的 `load_checkpoint`
   导出三个父 state（`export_sha256 d1d9886a…`）；
2. `scripts/migrate_r7_parent_state.py` 复核源 SHA pin、导出 state digest、张量 key/shape/dtype
   与模块语义 digest，然后写新的 `r7-local-v1` checkpoint（当前 digest + 归档 provenance），
   **不带 optimizer/cursor/RNG**；receipt `23daf794…`。

该改动对本 spec 是惰性的，三重证据：模块树只多两个默认属性（`anomaly_feedback=False`、
`backbone.climatology_anchor=None`），源码 diff 里所有新分支都以这两个属性为门，且**固定合成
batch 在两个修订下的 FP32 forward 逐位相同**（同一 sha256）。独立复核 C12 已确认源 pin 与迁移
pin 全部相符。

## 4. t2m/full 读数（skill = MSE skill vs 同数据 2017–2021 train-only 气候态）

| lead | seed41 | seed42 | seed43 | seed 均值 | ACC 均值 | ACC²（线性重标定上限） | σ_f/σ_o |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6h | +0.5763 | +0.5736 | +0.5950 | **+0.5816** | 0.7953 | +0.6326 | 1.02 |
| 12h | +0.3057 | +0.3051 | +0.3346 | **+0.3151** | 0.6651 | +0.4425 | 1.02 |
| 24h | +0.3577 | +0.3454 | +0.3973 | **+0.3668** | 0.6525 | +0.4259 | 0.90 |
| 48h | −0.1510 | −0.1576 | −0.0488 | **−0.1191** | 0.3671 | +0.1355 | 0.87 |
| 72h | −0.4615 | −0.4676 | −0.3636 | **−0.4309** | 0.1894 | +0.0366 | 0.87 |

三 seed 在 6/12/24h 全部为正、48/72h 全部为负；不平均反号。ACC² 是"任何纯幅度/线性重标定所能
达到"的构造性上限：48h **+0.136**、72h **+0.037**——即便完全校准，绝对门仍不可过，
所以 48/72h 的墙仍然是 pattern 相关本身。

## 5. 剂量对比与守门

t2m/full seed 均值 RMSE（candidate800 / registered200 / 差）：

| lead | 800 更新 | 200 更新 | 800 − 200 |
| ---: | ---: | ---: | ---: |
| 6h | 2.274794 | 2.249353 | **+0.025441** |
| 12h | 2.887180 | 2.879550 | **+0.007629** |
| 24h | 2.734883 | 2.888437 | **−0.153554** |
| 48h | 3.546694 | 3.871532 | **−0.324838** |
| 72h | 3.968864 | 4.434195 | **−0.465331** |

守门（对 v3-D3 控制）：u10/v10/mslp × 5 lead × 3 seed = 45 格全部 `relative_mse_change <= 0`，
最大 −0.104401（seed42/72h/v10）。全 17 变量在 24/48/72h 对控制的相对 MSE 变化全负。

## 6. 冻结 reading 阶段失败与只读复算（工程缺陷，非科学失败）

- attempt02 的 `prepare/archive/seed41/seed42/seed43` 五阶段全部 `success`、worker 全部
  `reaped`、`signals=[]`；只有最后的 `reading` 阶段 rc=1，`failure.json` 保留、全额计费。
  根因：共享的 `training/r7_rollout_evidence.py::_references` 用**注册 v3-BD receipt 记录的原始
  checkpoint 路径/SHA** 去比对协议里的 parent pin，同时又要求 `load_checkpoint` 能加载该
  parent——模型身份变更后，这两条对 model-only 迁移后的父不可能同时成立。**是身份记账管线缺陷，
  不是训练或评分失败。**
- 为不弱化任何比对逻辑，本轮**不改共享收集器**，而是另写只读工具
  `scripts/report_r7_rollout_dose.py`：它复核同一份产物与同一批 pin，并把全部判定委托给
  **同一个冻结函数**（`study_r7_s3_v3_rollout_ft.primary_verdict/gate_verdict/advance_decision`
  与 `r7_s3_v3_screen.paired_cells`）。产物
  `outputs/r7_s3_rollout_dose_readings_20261009_attempt01/readings.json`。
- 该工具唯一的偏离写在产物 `deviation` 字段：父的**身份**走迁移 receipt，父的**注册结果**
  仍按 SHA pin 作 provenance。它 CPU 只读，wall 7.95 s，0 GPU-h。
- 该缺陷的修复（让冻结收集器显式表达迁移父）登记为下一动作，不本轮擅自改共享判定代码。

## 7. 独立复核

独立 `general-purpose` 子代理用**自己的脚本**从原始产物复算 14 项（阶段/失败根因/协议 digest/
loss 行与加权恒等/checkpoint pin/cohort 与 provenance/配对 RMSE 增量/守门 45 格/primary 符号/
skill 与 ACC/剂量差/迁移 pin/成本算术/test 封存），**14/14 CONFIRMED，无一处 REFUTED 或
UNCERTAIN，无未核实项**。它另外指出 `readings.json` 里的 `parent_relative_gate`
（对 l6×1600 父，`passed=false`、12 个正格）与对控制的守门是两个不同比较，二者各自自洽。

## 8. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| probe attempt01（父 digest 未迁移，失败） | 1730.858695 | 0.4808 | 保留，全额计费 |
| probe attempt02（成功） | 1694.289826 | 0.4706 | FLOPs 393,859,201,536 与 reserved peak 2,409,627,648 与注册 probe 逐位相同 |
| screen attempt02（reading 阶段失败） | 11970.334049 | 3.3251 | soft overrun 2970.334049 s、hard overrun 0；六 worker 全 reaped、signals=[] |
| 只读复算工具（CPU） | 7.95（wall） | 0.0000 | 0 GPU、0 网络 |
| **本轮合计** | — | **4.2765** | 累计 23.3140 → **27.5905**（cap20/remaining −7.5905 仅会计） |

网络 0（离线）；无下载、无数据发布、无 clone；未改 model/、科学合同、冻结证据、用户配置；
未 signal 任何邻居进程。

## 9. 已确认与推测

**已确认**：五阶段执行与失败根因；协议/代码/数据/控制 pin；800 行损失与冻结加权目标恒等；
三 seed 端点与 15 组评分 cohort；primary supported 与守门 0/45；上表全部 skill/ACC/RMSE 数字；
剂量差；迁移 pin；成本；test 未读；独立复核 14/14。

**推测/未做**：不宣称收敛或 S4 就绪；三 seed 是一致性证据而非显著性；本轮未做同时区间、
未做 interior/edge、未读 test；48/72h 的 pattern 墙为何低仍只是"区域域内可用信息不足"的
**假说**，未由本轮区分。

## 10. limitations 与下一动作

- 单实例、单 ROI、17 通道、开发筛选；不是泛化或 SOTA 声明；`scientific_claim: false`。
- 剂量只测了 200/800 两点；24h 仍有 ACC² +0.426 与 skill +0.367 的 0.06 缺口，
  48/72h 缺口更大，继续加剂量的边际收益递减且**无论如何不能越过 ACC² 上限**。
- **下一动作**：修 `_references` 的迁移父表达（或把迁移注册为可 pin 的父记录）后，把
  "提高 pattern 相关"作为下一杠杆；最可证伪的一支是扩大区域/引入域外大尺度上下文
  （记忆与旧测量显示 ARCO ERA5 的 ROI 放大几乎不增成本），并重建同数据气候态与 incumbent
  后再比较。test 未读、r=0、S4 未启动不变。
