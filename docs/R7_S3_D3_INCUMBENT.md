# S3-D3：v2 确认实例上的同数据 actual-C incumbent 重训（GPU，3 seed）

**状态：已完成（2026-10-05/06）；`scientific_claim: false`；test（2023）未读。**
本页对应 `docs/goals/s3-confirmation-baselines-and-candidate.md` 的 D3 交付物：
在 v2 确认实例（train 2017 / val 2022）上，按 actual C 的 `process` 臂配置重训四类身份一致的
incumbent，给后续候选筛选提供**同数据、同预算**的对照读数。它不是科学胜出证据。

## 1. 结论摘要

- 三个预声明 seed（41/42/43）全部完成 400 次 L6 更新 + 5 lead 全 cohort 评估；
  整轮 **3955.3 s**（planned 5400 / hard 10800，soft overrun **0.0**），首个 spawn→末个 reap
  3936.4 s，保守 GPU-h **≈1.0987**；GPU 共驻，邻居始终未被 signal。
- **初始化身份逐位复现 actual C**：三 seed 的 `full_initial_state_sha256` 与归档 pairing
  （`a79ea47f…/ec5bb5ef…/844bd234…`）完全一致；冻结前另以当前树重放旧 dev store 的 CPU 目标函数
  （loss 0.192146/0.192453/0.192640，gradient norm 0.215594/0.191849/0.175951）逐位一致。
- **t2m/full 相对同数据 train-only 气候态（D2）的 skill**：6h 三 seed 全正
  （+0.2143/+0.2754/+0.3262，均值 +0.2720）；12h 两负一微正（−0.1655/−0.0697/+0.0037）；
  24h 两负一微正（−0.2907/−0.1088/+0.0526）；48h/72h 三 seed 全负（约 −1.3..−1.7 / −2.1..−2.6）。
  **与 S0（旧 M2 段）形状一致：只在 6h 稳定越过气候态；seed43 把边界推到 12h/24h 的微小正值。**
- 全 17 变量 seed-cell 正数计数：6h 51/51、12h 49/51、24h 47/51、48h 6/51、72h 0/51。
- 训练曲线三 seed 均在 400 更新末段继续下降/波动（block 均值 0.18→0.13..0.17 区间），
  seed42 末批 loss 0.2932 明显高于另两 seed（0.0928/0.0983）——单批 batch=1 的随机尾部，
  如实保留；这不是收敛证据。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 协议文件 | `outputs/r7_s3_d3_incumbent_20261005_attempt01/protocol.json`，文件 SHA256 `d9ead933e3e3ac73a8812790b66967816a2ba7658c4f015e68bc04c5cfcbd65d`，内部 `protocol_sha256 e4d521bb2e57c89a697d5ee37a6d2f505c3b7e25e06c2b6e8db149373d6467ec`（训练前 `'x'` 写入） |
| 结果文件 | `result.json` SHA256 `2c5210e724b0b73d698a2ab3f3413986f43df9bfcd319af6dfab78dc66867b90`；`attempt.json` SHA256 `38010c0003555b4fa1a20b64dedc0210a242d91007a956be2a5b4955c4c05d9e` |
| store | `outputs/r7_s3_confirmation_2017_2022_2023_v2/store/cache.zarr`（train 2017 / val 2022 / test 2023） |
| train manifest | `.../store/manifests/train.jsonl`，data identity `e01828e951e4c41182869b088da2b72131c9fc6c1c2d4abf42456b4e4cd6ed09` |
| val manifest | `.../store/manifests/val.jsonl`，val identity `5d8706c72d583cffb92f3de7ea850058ab64bfb7880cc0e9e036f244df6dad35` |
| 源 | `outputs/r7_s3_confirmation_2017_2022_2023_v2/source.nc`，`e0b51616…` |
| 训练窗口 | input 472 / usable **468**（4 个 `missing_exact_t12` 排除预声明），window_sha256 `f7f0c74c…` |
| 代码 | `scripts/study_r7_s3_d3_incumbent.py`，commit `55ab0b685182061531eb19754546435c2dd31d55`（protocol 内 `code.commit` 同值） |
| 归档对照 | `outputs/r7_v2_comparison_20261004_attempt01/protocol.json` 文件 SHA256 `57e659de…`；spec/initialization 逐字取用 |

### 2.1 与 actual C 的差异（如实）

- **数据实例不同**：actual C 用 M2 双月段（train 2016-01/02），D3 用 v2 三年实例的 2017 train；
  因此旧 dev store 的绝对读数（如 6h t2m RMSE 2.49/2.18/2.19）**不搬来当本实例控制值**。
- **代码路径**：当前树含 #77/#78/#79 新增的默认关闭开关，`model_semantics_sha256` 与归档不同
  （`a78da3db…` vs `00b7b7cc…`）；但 spec 取归档原值（全部新开关 False/identity），
  初始化逐位一致、旧 probe 目标函数逐位重放，行为等价由这两条证据支撑，不是假设。
- **训练窗口多**：468 vs 185；训练 update 数、batch、lr、schedule、clip、λ、K 相同。

## 3. 训练与评估执行

- 训练：`fine_tune`，mode=`l6`、400 updates、steps=4、lr 1e-4、warmup 20、weight_decay 1e-4、
  batch_size=1、clip 1.0、bf16=False、checkpoint_every=20、**frozen endpoint 无验证选择**；
  与 actual C worker 的调用一致（本实例上 process 臂 `process_supervision` 为 None，无 sidecar 参与）。
- 评估：`evaluate_local` 单 lead、`reasoning_steps=4`、max_samples→∞（全 cohort）、
  每 lead 独立 cohort（6/12/24/48/72h → 472/468/460/444/428，与 D2 完全相同），
  climatology 只 fit 2017 train（provenance 核 `training_years==[2017]`）。
- 每 seed 训练前 CPU 断言初始化 digest；每 lead 断言 cohort 数、split、climatology 年份。

| seed | 训练 s | final loss | checkpoint SHA256（末 400） | 6h | 12h | 24h | 48h | 72h 评估 s |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 154.4 | 0.09277 | `a6b610f0…64e36a28a7fc6b` | 77.1 | 98.2 | 130.5 | 220.1 | 303.1 |
| 42 | 154.8 | 0.29320 | `b2d0b6f4…5e2ad44fbe22ba17` | 76.9 | 97.8 | 131.0 | 242.2 | 459.5 |
| 43 | 209.3 | 0.09830 | `7dbdbc06…e3b6f3a7dc09d7f4` | 100.9 | 128.5 | 198.9 | 333.2 | 439.0 |

seed42 起评估变慢是邻居进程（10.4 GiB 常驻 GPU1/GPU0 交替）共驻所致；按策略只读记录、不 signal。

## 4. 读数（描述性，非显著性）

### 4.1 t2m / full / K4，skill = 1 − MSE_model / MSE_climatology（同数据 D2 分母）

| seed | 6h | 12h | 24h | 48h | 72h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | **+0.2143** | −0.1655 | −0.2907 | −1.7115 | −2.6343 |
| 42 | **+0.2754** | −0.0697 | −0.1088 | −1.3401 | −2.0742 |
| 43 | **+0.3262** | +0.0037 | +0.0526 | −1.2948 | −2.2304 |
| seed 均值 | +0.2720 | −0.0772 | −0.1156 | −1.4488 | −2.3130 |

6h RMSE（K）：3.1381 / 3.0137 / 2.9062（对气候态 3.5404）。

### 4.2 守门变量 skill（同数据气候态分母；D4 相对 incumbent 的守门另算）

| 变量 | 6h | 12h | 24h | 48h | 72h |
| --- | ---: | ---: | ---: | ---: | ---: |
| u10 | +0.394/+0.400/+0.407 | +0.125/+0.129/+0.125 | +0.008/+0.004/−0.042 | −0.162/−0.178/−0.233 | −0.259/−0.280/−0.278 |
| v10 | +0.606/+0.584/+0.594 | +0.308/+0.258/+0.280 | +0.080/+0.008/+0.049 | −0.190/−0.295/−0.210 | −0.294/−0.397/−0.271 |
| mslp | +0.894/+0.889/+0.900 | +0.772/+0.751/+0.766 | +0.441/+0.358/+0.378 | −0.282/−0.464/−0.487 | −0.786/−1.008/−1.050 |

（每格按 seed41/42/43 顺序。）

### 4.3 全 17 变量 × 5 lead 计数（正 / 非正，seed-cell 共 51）

| lead | 正 | 非正 |
| ---: | ---: | ---: |
| 6h | 51 | 0 |
| 12h | 49 | 2（t2m 41/42） |
| 24h | 47 | 4（t2m 41/42、v850 41、u10 43） |
| 48h | 6 | 45 |
| 72h | 0 | 51 |

48h 剩余正项集中在高空 u250/其他少数通道（如实保留，不并入判定）。

## 5. 成本与失败处理

| 项 | 值 |
| --- | --- |
| 整轮墙钟 | 3955.3 s（planned 5400 / hard 10800）；soft overrun 0.0 |
| 首个 spawn→末个 reap | 3936.4 s；保守 GPU-h 1.0987（含 CPU 前置/间隔/评估/收尾） |
| 网络 | 0（`deny_network` 进程内生效）；CPU 侧无外部请求 |
| test | 从未打开（`test_read: false`，两次 refusal 断言在驱动内） |
| 失败 | 无失败、无 skip、无 partial；三 seed 全部完成。若失败会写 `failure.json` 并保留 partial |
| 磁盘 | 3 seed × (20 检查点 + 5 评估目录 + report)，无历史产物覆盖 |

## 6. confirmed / 推测

**已确认（事实）**

1. 三 seed 从与归档 actual C 逐位相同的初始化出发，在 v2 2017 train 上完成 400 更新，val 全 cohort 读数齐。
2. 同数据气候态口径下，incumbent 只在 6h 三 seed 全正；12h/24h 出现 seed 间分歧（2 负 1 正）。
3. 48h/72h 全面落后；与 S0 的形状结论（更长 lead 无技巧）在同数据上重现。
4. 成本、身份、cohort 全覆盖断言均通过；test 未读。

**推测（待验证）**

- 训练目标只有 L6 深监督、评估到 72h 的目标-评估错配仍是 48–72h 落后的主因候选；
  D4 的 R-C 对照（two_step、FLOP-matched）直接检验这条假设。
- seed43 的 12h/24h 微正可能是训练随机性（batch=1 单批尾部），不是结构信号。

## 7. limitations

- 单实例（2017 train / 2022 val、单 ROI、17 通道）、3 seed；描述性一致，不是显著性。
- 控制读数与 actual C 的旧 dev store 不可直接比较（不同实例），本实例内可比。
- K4 是同一 K4 训练权重的推理深度探针，非独立训练。
- GPU 共驻，邻居负载出现在部分 lead 的墙钟中；不 signal、不剔除。
- test（2023）留待 S4 预注册读数；本页不产生任何越过气候态的最终声明。

## 8. 对 D4/D5 的含义

1. D4 候选（two_step、200 更新、FLOP 匹配 400 L6）与**本页三 seed 控制**成对读；控制 RMSE CSV
   以 SHA256 pin 进 D4 协议，读时逐 lead 复核。
2. 若 D4 主格（t2m 6h/12h）全 seed 同号且候选更低、且守门预审过，则该候选进 D5 冻结包；
   否则如实登记负结果。
3. 12h 是本实例上最值得盯的边界（seed43 已到 +0.0037）；任何 D5 候选至少要在此不再恶化。

## 9. 证据指针

- 运行根：`outputs/r7_s3_d3_incumbent_20261005_attempt01/`（protocol/result/attempt + 3 seed 全链）
- 驱动与测试：`scripts/study_r7_s3_d3_incumbent.py`（commit `55ab0b6`）、`tests/test_r7_s3_d3_study.py`
- 基线读数：`docs/R7_S3_D2_BASELINES.md`（同 cohort 气候态/persistence）
- 规格来源：`outputs/r7_v2_comparison_20261004_attempt01/`（actual C 归档，只读）
