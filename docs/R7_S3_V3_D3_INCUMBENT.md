# S3 v3-D3：v3 五年实例上的 same-data incumbent 重训（GPU 3 seed）

**状态：已完成（2026-10-06）；`scientific_claim: false`；test（2023）未读。**
本页对应 `docs/goals/s3-confirmation-baselines-and-candidate.md` 的 v3 控制重建与
`docs/R7_S3_CONFIRMATION_INSTANCE_V3.md` 的 §10 下一步 2。它是**筛选用控制读数**，
不是科学声明。

## 1. 结论摘要

- 三个预声明 seed（41/42/43）在 CPU 上逐位复现归档 actual C 的初始化
  （`full_initial_state_sha256` 与初始化报告 digest 与归档 pairing 完全一致），
  随后在 v3 的五年 train 切分上完成 **400 次 L6 更新**（冻结端点、无验证选择），
  在 2022 val 的全部 per-lead cohort 上评分（472/468/460/444/428）。
- **同数据 t2m/full skill vs v3 气候态**（三 seed）：
  6h **+0.4148 / +0.2610 / +0.4248**（全正）；12h +0.0285 / −0.0805 / +0.0756（两正一负）；
  24h +0.0218 / −0.1359 / −0.0418（一正两负）；48h −1.1453 / −1.4017 / −1.0692；
  72h −2.1142 / −2.0213 / −1.7365（48–72h 全负）。
- 全 17 变量正 seed-cell 计数（每 lead 51 个）：**6h 51、12h 50、24h 45、48h 2、72h 0**。
- **与 v2-D3（同配方、单年 train）的描述性对比**：6h skill 三个 seed 都更高
  （v2：+0.2143/+0.2754/+0.3262）；12h 从"一正"变为"两正"；24h 均为一正；
  48–72h 仍然全负但 6h/12h 的形状改善——与「数据扩年提升近 lead 能力」的假设方向一致，
  但该对比跨实例（不同 climatology），不是受控比较，不作为判据。
- 成本：整轮 **5404.5 s**（planned 5400 / hard 10800，soft overrun **4.5 s**，按决策 0030
  记账继续）；保守 GPU-h **1.5012**；网络 **0**（离线）；邻居未 signal（GPU0 共驻，门禁只读）；
  test 从未打开。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 协议 | `outputs/r7_s3_v3_d3_incumbent_20261006_attempt01/protocol.json`，canonical `protocol_sha256 9912a67ff3f3c1ae35f743cb956aab80fea43c39ecc61eecf32760d9e0621bbe`（文件 SHA256 `9a5a4def…`；训练前写入） |
| instance | `outputs/r7_s3_confirmation_train2017_2021_v3`（source `bc2ff9cf…`、3360 stamps） |
| store | `.../store/cache.zarr`（data identity `2564eeaf…`） |
| train 数据 | `.../store/manifests/train.jsonl`；机械窗口 2360 input / **2340 usable**（20 个 `missing_exact_t12` 排除，`window_sha256 77a9945d…`） |
| val 数据 | `.../store/manifests/val.jsonl`；per-lead cohort 472/468/460/444/428 |
| 结果 | `result.json`（SHA256 `7a3f2eac…`；含每 seed checkpoint/training_report/每 lead CSV 的 SHA256、GPU 门禁、成本） |
| 保真锚 | 归档 actual-C pairing（`outputs/r7_v2_comparison_20261004_attempt01`）；三 seed 初始化 digest 全部一致（§3） |

## 3. 保真检查（训练前、CPU、逐位）

| seed | initial_state_sha256（前 16） | init_report_sha256（前 16） | 与归档一致 |
| ---: | --- | --- | --- |
| 41 | `a79ea47fa098bfc4` | `b33c756723575681` | 是（逐位） |
| 42 | `ec5bb5ef581e1cfe` | `8070b6d437d06dc9` | 是（逐位） |
| 43 | `844bd23402cabb3f` | `8e2de147b25671b8` | 是（逐位） |

模型 spec 为归档 process spec 逐字（canonical digest 记录在协议）；契约不继承
`process_supervision`（actual-C process 臂无进程监督）；instance 的 process-scale
sidecar 只作实例身份上下文记录，**未**用于训练或评估。

## 4. t2m 读数（skill = `mse_skill` vs v3 五年 train-only 气候态；region full；正值 = 模型更好）

| lead | seed41 | seed42 | seed43 |
| ---: | ---: | ---: | ---: |
| 6h | **+0.4148** | **+0.2610** | **+0.4248** |
| 12h | +0.0285 | −0.0805 | +0.0756 |
| 24h | +0.0218 | −0.1359 | −0.0418 |
| 48h | −1.1453 | −1.4017 | −1.0692 |
| 72h | −2.1142 | −2.0213 | −1.7365 |

（物理 RMSE 与逐变量 skill 在各自 `climatology_skill.csv`，SHA256 记录在 `result.json`。）

每 seed 训练细节：final loss 0.06114 / 0.08334 / 0.13406；训练墙钟 151 / 198 / 158 s；
checkpoint SHA256 完整值记录在 `result.json`（前缀 `6646905e…` / `9929ef11…` / `a01bf0f2…`）。

## 5. 成本与门禁

| 项 | 值 |
| --- | --- |
| 整轮 | 5404.5 s = planned 5400 的 100.1%（soft overrun **4.5 s**，记录继续） / hard 10800 未触 |
| 保守 GPU-h | **1.5012**（连续首 spawn→末 reap 5331.8 s / 3600） |
| GPU | `GPU-408ad137…`（GPU0）；启动前每 seed 只读 nvidia-smi 余量门（free ≥ 2048+2048 MiB）全过；未向邻居发任何信号 |
| 网络 | 0（`deny_network` 进程内生效） |
| test | 未读（`test_read: false`；只打开 train/val manifest） |

## 6. 已确认与推测

**已确认**：三 seed 初始化逐位复现 actual-C；训练到 400 更新冻结端点；val per-lead
cohort 完整；气候态为五年 train-only；成本与门禁如 §5；test 未读。

**推测（不写成事实）**：v3-D3 相对 v2-D3 的近 lead 改善（6h 全 seed 更高、12h 两正）
与数据扩年相关，但跨实例对比不受控（气候态与样本集都变），只能作为 v3 预算剂量筛选的
背景；48–72h 的全负形状提示长 lead 仍是短板，是否由数据量单独解决由 v3-BD 轮回答。

## 7. limitations（如实）

- 筛选用控制读数：单实例、单 ROI、17 通道；三 seed 是一致性证据，不是显著性检验。
- K4 是同一 K4 训练 checkpoint 上的推理深度探针，不是独立模型。
- 跨实例（v2↔v3）数字不是受控比较。
- GPU 共驻运行，latency/内存观察含邻居负载，不作基准。
- 本页不产生任何科学结论；`scientific_claim: false`。

## 8. 证据指针

- 协议与结果：`outputs/r7_s3_v3_d3_incumbent_20261006_attempt01/`（protocol.json、
  result.json、attempt.json；三 seed 目录含 checkpoint 与逐 lead CSV）
- 驱动：`scripts/study_r7_s3_v3_d3_incumbent.py`（db0b463 起在版本控制内）
- 上游：`docs/R7_S3_CONFIRMATION_INSTANCE_V3.md`、`docs/R7_S3_V3_D2_BASELINES.md`

## 9. 下一步

1. v3 预算剂量筛选（l6×1600 vs 本页 400 更新控制；主格 + u10/v10/mslp 守门合取）。
2. 通过合取则 D5 冻结 S4 包；未通过则登记该轮读数并分析长 lead 的下一步。
3. test（2023）保持封印，直到预注册的未见年份确认轮。
