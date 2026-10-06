# S3-V3-BD：扩年实例上的训练预算剂量（l6 1600 更新 vs v3 400 更新控制）— 主格 supported / 守门 17 cell 未过，数据扩年未在 1600 剂量修复长 lead

**状态：已完成并登记（2026-10-06）；`scientific_claim: false`；test（2023）未读。**
本页回答预先声明在 protocol `factor.data_response_text` 的问题：把 train 从单年 2017 扩到
2017–2021 五年（v3 实例）后，在**同一 1600 更新剂量**下，v2 上观察到的长 lead 守门失败
（17 个正 cell）是否被数据量清除或明显缓解。这是 S3 在 v3 实例上的候选轮，
也是本节点「预算响应判归数据」预判的直接检验。

## 1. 结论摘要

- 预注册主格（t2m/full、6h/12h、每 seed 同号、候选更低才 supported）**判定 supported**：
  6h −0.4328/−0.7607/−0.4444 K，12h −0.5351/−0.7199/−0.5158 K；24h 也全负
  （−0.3237/−0.4513/−0.6301 K）；48h 两负一正（seed41 +0.1379）；72h 两正一负
  （seed41 +1.1889、seed42 +0.6898、seed43 −0.5936 K）。
- 守门预审 **不通过、且数量与 v2-BC 完全持平**：17 个正 cell（24h 1、48h 7、72h 9；
  u10 5、v10 6、mslp 6）——v2-BC 为 17 个（48h 8、72h 9）。**扩年到五年没有清除任何
  长 lead 守门 cell 的净数量**（分布上有一个 cell 从 48h 移到 24h）。
- **数据响应读数（预声明、非判据）**：按预声明分支「守门 cell 持续存在或恶化」——
  **长 lead 退化不是单靠训练体量在这个剂量上能修复的**；下一问题转向长 lead 本身需要什么
  （损失权重、更长 lead 监督、或大于 1600 的剂量）。补充的横向事实：同一 1600 剂量下
  seed 均值 skill 对各自（各自实例的）气候态几乎不变——6h +0.595→**+0.5915**、
  12h +0.333→**+0.3171**、24h +0.226→**+0.2082**（v2-BC → v3-BD）；而 v3 实例上的
  400 更新控制（v3-D3）近端已强于 v2-D3（6h +0.41/+0.26/+0.42 vs +0.21/+0.28/+0.33），
  即数据扩年的近端收益主要落到了**控制端**，到 1600 剂量两端趋同。
- 候选 t2m 相对 v3 五年气候态 skill：6h +0.588/+0.586/+0.601、12h +0.307/+0.306/+0.338、
  24h +0.199/+0.127/+0.299、48h −1.268/−1.206/−0.470、72h −3.507/−2.787/−1.177；
  正 seed-cell 计数 51/51/50/22/14（v3-D3 控制为 51/50/45/2/0）。
- 训练 loss 从 ~0.177 降到 ~0.12 后进入平台（最后四段 0.120–0.129）；剂量内停止线仍是
  守门合取规则，不是 loss。
- 按冻结规则 `registered-negative`（primary supported 与 gate 合取未过）→ 不进 D5/S4 冻结包。
- 整轮 6883.9 s（planned 6300 / hard 12600，**soft overrun 583.9 s 记录**），首个 spawn→末次
  reap 6673.9 s，保守 GPU-h **1.9122**；累计本方向 **9.9238 GPU-h**（硬上限 12.0 内，
  余 2.0762）。

## 2. 登记内容（运行前冻结）

| 项 | 值 |
| --- | --- |
| 协议文件 | `outputs/r7_s3_v3_budget_dose_20261006_attempt01/protocol.json`，文件 SHA256 `a82b1464344cd3589026f08a213ba1f40119667ddf1fd010acd4b6db178c1f01`，内部 `protocol_sha256 83002900073b35a74691c50660cd1547d8556787adbd044649c1255a891f7e1e`（训练前 `'x'` 写入并盘上重读校验） |
| 结果 / attempt | `result.json` SHA256 `61d3f1d3eedb5208da616457d4bff1a26a57a4c24a6470f595ebb5bb6cb060e4`；`attempt.json` SHA256 `f3eac60dec672ca73418cf02b605989807ba06a0ac7a4352b59dfced99babb69` |
| 代码 | commit `828ca548934224a64478661e026d1e034ba74640`（protocol `code` 记录 dirty 仅 `.zcode/config.json` 与未跟踪 `.zcodeignore`，均不入本登记提交）；驱动 commit `db0b463` |
| 实例 | `outputs/r7_s3_confirmation_train2017_2021_v3/`（train 2017–2021 / val 2022 / test 2023）；train data identity `2564eeaf…`、source `bc2ff9cf…`、val identity `0c34a887…` |
| 机械差异（单因素） | 训练更新预算：control `l6` 400 updates vs 候选 `l6` **1600 updates**；同 spec、同初始化、同数据、同评估路径、同 warmup 比例（5% 端点：80/1600，控制为 20/400） |
| 剂量声明 | `factor.dose=2`；v2 先验列 UB(800)/BC(1600)；`known_confound` 明写 4× 更新与 FLOPs 不对称；`schedule` 固定 warmup/cosine 约定在实例扩年中不变 |
| 对照 | v3-D3 注册 run 的三 seed incumbent（result `7a3f2eac…`，逐 lead RMSE CSV 以 SHA256 pin 进本协议，读时复核） |
| GPU | GPU0 `GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b`；共驻、只读余量门（三 gate 全过：free 20.15/19.60/22.93 GiB）、不 signal 邻居 |
| 判据文本 | 协议 `decision.primary.text` / `decision.gate_pre_screen.text` / `factor.data_response_text`（本页同义复述，以协议为准） |

## 3. 主格与全 lead 读数（t2m/full，RMSE 增量 = 候选 − 控制，K）

| seed | 6h | 12h | 24h | 48h | 72h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | **−0.4328** | **−0.5351** | −0.3237 | +0.1379 | +1.1889 |
| 42 | **−0.7607** | **−0.7199** | −0.4513 | −0.2162 | +0.6898 |
| 43 | **−0.4444** | **−0.5158** | −0.6301 | −0.7578 | −0.5936 |

候选 t2m 绝对 RMSE（seed41/42/43）：6h 2.2577/2.2628/2.2229、12h 2.9037/2.9067/2.8385、
24h 3.0762/3.2123/2.8785、48h 5.0497/4.9808/4.0660、72h 7.0448/6.4577/4.8957；
控制（v3-D3）6h 2.6905/3.0235/2.6674、12h 3.4388/3.6266/3.3543。

**训练 loss segment 均值（每 100 更新，seed41/42/43 首段→末段）**：0.1768→0.1201 /
0.1795→0.1258 / 0.1706→0.1136；末四段分别在 ~0.12–0.13 / ~0.12–0.13 / ~0.11–0.12，
进入平台。注意本剂量与 400 控制的 LR 日程不同（warmup 80 vs 20、cosine 到 1600 vs 400），
端点 loss 数值不可与控制直接比较。

## 4. 守门预审（u10/v10/mslp，相对 MSE 变化 ≤0.0，5 lead × 3 seed = 45 cell）

- **不通过**：17 个正 cell（24h 1 个、48h 7 个、72h 9 个）；变量分布 u10 5、v10 6、mslp 6。
- 与 v2-BC 的对照（同剂量、单年 train）：17 个（48h 8、72h 9）→ 17 个（24h 1、48h 7、72h 9）。
  **净数量持平，长 lead 集中性不变**（72h 仍 9 个，全部 3 seed × 3 变量除 seed41 的 24h 例外）。
- 预声明数据响应分支据此取值：「守门 cell 持续存在」→ 长 lead 退化在该剂量上不由训练体量
  单独决定。

## 5. 训练与评估执行

| seed | 训练 s | final loss | checkpoint SHA256（前 16） | init state（前 16） |
| ---: | ---: | ---: | --- | --- |
| 41 | 781.9 | 0.11161 | `637857c5d5cc8b3c` | `a79ea47fa098bfc4` |
| 42 | 666.9 | 0.12491 | `8f7369fbbb3983b9` | `ec5bb5ef581e1cfe` |
| 43 | 601.9 | 0.07129 | `08cc7db9629fca86` | `844bd23402cabb3f` |

- 三 seed 初始化均先经 CPU 逐位断言（与归档 actual C pairing 一致，同 D3 家族）。
- l6 目标监督断言（`losses[*].l12` 全 None）、`updates_this_run=selected_update=1600` 断言、
  train data identity `2564eeaf…` 断言通过；每 lead 评估断言 val split/cohort 472/468/460/444/428/
  climatology 只 fit 2017–2021。
- **共驻观测**：seed41 的评估段（5 lead 合计 1737.1 s）明显慢于 seed42/43（751.8 / 758.9 s），
  期间 GPU0 有邻居负载（nvidia-smi 见 3.4 GiB 邻居进程）；数值不受影响，仅计入 latency 口径。
- 每 seed 3600 s deadline 未触发（最大 seed 段 3177.3 s）；硬上限 12600 s 未触。

## 6. 成本

| 项 | 值 |
| --- | --- |
| 整轮墙钟 | 6883.9 s（planned 6300 / hard 12600）；**soft overrun 583.9 s**（决策 0030：软超继续并记录） |
| 首个 spawn→末次 reap | 6673.9 s；保守 GPU-h **1.9122** |
| 训练 FLOPs | 候选 3 seed × 1600 × 31,981,732,992 = 153.51 TFLOP（控制同口径 38.38 TFLOP，比 4.0） |
| 网络 / test | 0 请求；test 未读（`test_read: false`） |
| 失败 | 无；三 seed 全部完成。FLOP 探针（CPU）在协议冻结前执行，属整轮计时 |

## 7. confirmed / 推测

**已确认（事实）**

1. 五年 train 实例上，1600 更新剂量相对同实例 400 更新控制的主格（t2m 6h/12h）三 seed
   同号 supported，且 24h 三 seed 也全部为负。
2. 长 lead 守门失败 cell 数量与 v2-BC 相同（17），分布近似（72h 9 持平）；**数据从 1 年
   扩到 5 年（同剂量 0.7 epoch vs 3.4 epoch）没有清除长 lead 守门代价的净数量**。
3. 同剂量 seed 均值 skill 对各自气候态几乎不变（6h +0.595→+0.5915、12h +0.333→+0.3171、
   24h +0.226→+0.2082）；扩年的近端提升集中体现在 400 更新控制（v3-D3 6h skill
   0.41/0.26/0.42 vs v2-D3 0.21/0.28/0.33），到 1600 剂量被追平。
4. 训练 loss 到 1600 已进入平台（不再有 400→800→1600 的单调下降势能），停止线是守门合取。

**推测（待验证）**

- 长 lead（≥48h）退化更像目标/结构问题（缺少长 lead 的直接监督或缺少适应长 lead 的
  网络机制），而不是训练数据量问题；下一候选应从「更长 lead 监督 / 损失权重 / 效率更高的
  剂量-数据配比」中选一，做单因素筛选。
- 0.7 epoch 意味着五年数据在 1600 剂量下未被充分遍历；「更大的剂量（epoch 匹配或更大）」
  仍是一条未关闭的通道，但它不在本节点软预算的合理剩余内（余 2.0762 GPU-h）。

## 8. limitations

- 单实例（2017–2021 train / 2022 val）、3 seed；描述性一致，不是显著性。
- 候选刻意使用 4× 训练更新与 FLOPs；supported 是对「该剂量配方」而言，不是算力对等比较。
- 剂量不是 epoch 匹配（v3 约 0.7 epoch vs v2 约 3.4 epoch），本页价格的是声明的剂量。
- 控制读数来自 v3-D3 注册 run（pin 而非重训）；跨实例（v2 vs v3）比较是描述性的。
- GPU 共驻；latency 含邻居负载，不作基准。
- validation split only；test 封存至 S4 预注册读取。

## 9. 证据指针

- 运行根：`outputs/r7_s3_v3_budget_dose_20261006_attempt01/`
- 驱动与测试：`scripts/study_r7_s3_v3_budget_dose.py`、`tests/test_r7_s3_v3_studies.py`
- 共享判读：`training/r7_verdict_readings.py`；v3 屏幕支持模块 `training/r7_s3_v3_screen.py`
- 实例与前序：`docs/R7_S3_CONFIRMATION_INSTANCE_V3.md`、`docs/R7_S3_V3_D2_BASELINES.md`、
  `docs/R7_S3_V3_D3_INCUMBENT.md`、`docs/R7_S3_BUDGET_CURVE.md`（v2 先验）
