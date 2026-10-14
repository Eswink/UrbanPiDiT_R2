# S3 H-A（P_0 逐样本输入条件化）单因素臂：72 h 三 seed 全负，按预声明出口关闭 H-A 支

**状态：已完成（2026-10-14，三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-process-init-conditioning.md`。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；`primary`/`gate` 形式在这里只是**报告**，
因为两个对照都是已登记臂的 pinned 复用。

**同一因子的登记**：H-A 因子的**第一次尝试**（attempt01，父身份门在训练前拒绝、**0 GPU-h**）已单独登记为
`s3-process-init-run`（`outcome_class: unresolved`、`candidate_state: blocked`，证据页
`docs/R7_S3_PROCESS_INIT_RUN.md` **逐字节保留**）。本页是同一因子**跑完**的 attempt02，登记为
`s3-process-init-result`。本页**不**向任何既有 record 追加 `correction_record_id` 指针——`s3-process-init-run`
是一条**独立的失败轮登记**（工程性身份门失败，训练从未开始），不是被本页取代的旧版本。

## 1. 结论摘要

- **单因素**：在**已登记的最佳臂**上，只把 `process_init_conditioning` 从 **False 提到 True**——过程状态初值
  `P_0` 由「全体样本共享的 `process_queries`」改为逐样本条件化
  `P_0 = process_queries.expand(B,·,·) + process_init_proj(context.mean(dim=1)).unsqueeze(1)`。
  其余（宽 129×129 输入、`interior_32` 监督、2400 更新、K=4、`depth`=4、`dim`=192、注册物理权重
  `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、迁移 v3-BD 1600 父、seed 41/42/43、评分域与气候态分母）**逐字不变**。
  新增参数 **+37,440**（`process_init_proj = Sequential(LayerNorm(192), Linear(192,192))`，恰 4 个新键）。
- **72 h 三 seed 全负**：`interior_32` t2m skill 为 **−0.2437 / −0.3064 / −0.2171**（均值 **−0.2557**）。
  **预声明出口的正向条件未满足**——它要求 72 h **三 seed 全正**。
- **H-A 不是正向杠杆，且相对已登记的 2400 剂量臂在长 lead 上更差**：相对该臂（**同 seed 配对**、
  pinned 复用、未重训练），24/48/72 h **三 seed 全部更差**（Δrmse 均值 **+0.0103 / +0.0605 / +0.0798 K**），
  6/12 h 仅轻微更好（−0.0027 / −0.0026 K，2/3 seed）。即**逐样本 P_0 条件化把长 lead 推得更差**。
- **量级**：把 72 h skill 从 −0.2557 推回 0，rmse 需再降约 **0.41 K**；本臂相对剂量臂**反向**贡献了
  **+0.0798 K**。方向与幅度都与"P_0 无输入信息是 72 h 瓶颈"的假设**相反**。
- **判定**：按预声明出口，本轮正向条件未满足 ⇒ **关闭 H-A 这一支**；不调 `proj` 宽度、不加 seed。
  本臂**不进 S4 冻结包**。

| lead | Δrmse vs 剂量臂（K，seed 均值） | 范围 | 更优 seed | skill 均值 | skill(41) | skill(42) | skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | −0.0027 | −0.0081 … +0.0047 | 2/3 | +0.5837 | +0.5808 | +0.5737 | +0.5965 |
| 12 h | −0.0026 | −0.0093 … +0.0093 | 2/3 | +0.3298 | +0.3285 | +0.3132 | +0.3475 |
| 24 h | +0.0103 | +0.0057 … +0.0178 | **0/3** | +0.4149 | +0.4079 | +0.4043 | +0.4325 |
| 48 h | +0.0605 | +0.0431 … +0.0694 | **0/3** | +0.0322 | +0.0211 | +0.0022 | +0.0733 |
| 72 h | +0.0798 | +0.0425 … +0.1042 | **0/3** | **−0.2557** | −0.2437 | −0.3064 | −0.2171 |

（Δ 为「本臂 − 已登记 2400 剂量臂」，**正值 = 本臂更差**；两者评分域、气候态分母、cohort pin 逐字相同，
可直接相减。逐 lead 的绝对值与 Δ 见 `outputs/r7_s3_process_init_20261014_attempt02/readings.json`。）

相对**已登记的窄 65×65 800 臂**（`record:s3-rollout-dose`，pinned）本臂五个 lead **三 seed 全部更优**
（Δrmse 均值 **−0.0056 / −0.0310 / −0.1056 / −0.2481 / −0.2508 K**）——但窄臂本身更弱，这只说明
H-A 仍优于窄臂，**不**改变"不如 2400 剂量臂"的结论。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_process_init_20261014_attempt02/` |
| 协议 digest（`protocol.json`，最终；驱动 `validate_protocol` 自洽校验通过） | `ad7566f3bf3d358fc8837e9580eda5b7a60338b848221feba821502c77a61739` |
| 预备/冻结 digest（`preparation_protocol.json`，`freeze` 阶段报出） | `d6cfde3e6e52567b910595986b63222de3c4fa19913905a28b3682b5f52cedb0` |
| 归档 `code.zip` SHA256 | `897b9bc01aae5d2fa0bf97a027ce2d6b3af1810a77091db14b24156cd3327844`（1401 条目） |
| ZIP comment / 代码 commit | `81233b373c7da1f907eb3511b4987fa87e329b23`（comment 与 `code_commit` 逐字一致） |
| `model_code_sha256` | `223b2e5b82093bb5c1b75ffa25ff63983f44e6cdf6a96da772b17e88a748750e`（本轮改了 `model/*.py`，故与容量两轮的 `d3fb58db…` 不同） |
| `training_code_sha256` | `8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974` |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| `train_data_identity` | `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12` |
| 执行文件身份 | 188 个（`validate_protocol` 逐文件核过） |
| 监督块（冻结） | `spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576b…`（与对照臂相同） |
| **H-A 扩张（冻结规则）** | 新增 `process_init_proj`（`Sequential(LayerNorm(192), Linear(192,192))`）**最后、在 `isolated_stream()` 下构造**；注入点＝`P_0 = process_queries.expand(B,·,·) + process_init_proj(context.mean(dim=1)).unsqueeze(1)`；新增键恰 4 个、新增参数 37440；因模块最后构造且不消耗 RNG，**开关关闭时逐位等于现实现**；**逐 seed** 断言 `bit_identical_off_path=true`、`proj_zeroing_is_identity=true` |
| **正控制（逐 seed 实测）** | 同 seed(1234) 下 off/on 主干在**每个共享张量**上 `torch.equal`、差异恰为 4 个 `process_init_proj` 键；把 `process_init_proj.1`（Linear）权重/偏置**清零** ⇒ 注入项恒为 0（`zeroed_proj_output_abs_max=0.0`） |
| 剂量（冻结） | `updates=2400`、`lr=2e-5`、`warmup=10`、K=4、FP32、batch 1、clip 1、12 步物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` |
| 对照 1 | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| 对照 2 | 窄 65×65 800 臂（`record:s3-rollout-dose`，pinned，**未重训练**） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=11.91 GiB`；`headroom_margin_bytes=0.5 GiB` |
| 预算 | planned 86400 s / hard 115200 s / per-seed 28800 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，三 seed 一致） |
| test | `test.jsonl` 从不读取；`test_read=false` |

## 3. 训练与评分回执

| seed | checkpoint SHA256 | 训练 s | 五 lead 评分 s | 训练损失 1→2400 | 峰值 reserved |
| ---: | --- | ---: | ---: | --- | ---: |
| 41 | `4eb3e8e14af342f53c8e67554309a3e2eb9e809c28f6d14662321943b500a4ae` | 5867.6 | 2050.2 | 1.3837→0.7832 | 8.361 GiB |
| 42 | `09d8dee5f6c5e5c20e43f67f4a64875618669b9e5ec474f28dd4ed38e8675201` | 4880.4 | 894.6 | 1.3055→0.5748 | 8.361 GiB |
| 43 | `a0f452bfc15f76d94f57b6a0a65f466fa8975b3662f01ae874bba75a611ab35f` | 4736.5 | 853.3 | 1.1384→0.7269 | 8.361 GiB |

- 三 seed 都到 `total_updates=2400`；每 seed 每 lead `n_evaluated == n_available_windows` 且等于冻结
  cohort（472/468/460/444/428）；每 seed 的 `training_report.json` 的 `supervision` 与冻结
  `mask_sha256` 逐字相同（驱动断言）。
- 三 seed 的 H-A 扩张断言全部通过；逐 seed 的 `parent_state_sha256` / `expanded_state_sha256` 随各自父
  checkpoint 不同而不同，这正是"逐 seed 独立断言"的意义：

| seed | 父 checkpoint SHA256 | `parent_state_sha256` | `expanded_state_sha256` | new_params |
| ---: | --- | --- | --- | ---: |
| 41 | `3bb09c19…` | `7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707` | `08aeaffb5ff097ad03c8269242f5d81e3892eba9db514a5cde68675557b6150d` | 37440 |
| 42 | `91667a68…` | `857d880466f76500ed74c534e757732ae9eff4c2b3f0bbc0fa74b895fcae3623` | `9e509912f0b321a8391d816fb6d3718502fea03a1c9005cab15a6e2901ea3f2c` | 37440 |
| 43 | `b1a716bc…` | `61fdac58b7bfacf828e1eae4968817b1cbd46616442c6d16ac0fbbb63e294e1b` | `b6313c340894c7ec29c4b7f9227bb3a0bcf8517d1a4f274484d6808bdada6858` | 37440 |

- 三 seed 的 `bit_identical_off_path=true` 且正控制 `proj_zeroing_is_identity=true`
  （`zeroed_proj_output_abs_max=0.0`）——**开关关闭时逐位等于现实现**这一条由逐 seed 正控实测，
  不是推断。

## 4. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 473.668 |
| archive | `archive_process.json` | 0 | success | 475.169 |
| seed 41 | `seed41_process.json` | 0 | success | 9292.575 |
| seed 42 | `seed42_process.json` | 0 | success | 7119.604 |
| seed 43 | `seed43_process.json` | 0 | success | 6870.466 |
| reading | `reading_process.json` | 0 | success | 460.998 |

六阶段全 `success`、`returncode=0`、无信号；每阶段 spawn 前只读 `gpu_gate`；
**未对任何非本实验进程发送任何信号**（邻居全程只共享显存）。

## 5. 公平性守卫

- 三 seed 的逐 lead cohort 计数与已登记对照臂**逐字相同**（472/468/460/444/428）。
- reading 阶段按 `boundary_rmse_csv_sha256` 复核 2400 剂量臂、按 `rmse_csv_sha256` 复核窄臂，
  **30/30 全部匹配**（3 seed × 5 lead × 2 对照；两个对照都**未重训练**）。
- 气候态分母取自 `wide_climatology_reference`（train-only、与对照臂同一分母），`readings.json`
  里的 `protocol_sha256` 与 `protocol.json` 的最终 digest **逐字一致**（`ad7566f3…`）。
- H-A 的"off 起点等于注册函数"由**逐 seed 的逐位一致断言 + 清零正控制**保证（§2），因此本臂与对照臂的
  差别被限制在 `process_init_conditioning` 这一个字段上。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 473.668 | 0.1316 | |
| archive（CPU） | 475.169 | 0.1320 | |
| seed 41（2400 更新 + 五 lead） | 9292.575 | 2.5813 | |
| seed 42（2400 更新 + 五 lead） | 7119.604 | 1.9777 | 与 41 并发 |
| seed 43（2400 更新 + 五 lead） | 6870.466 | 1.9085 | 与 42 共用第二卡 |
| reading（CPU） | 460.998 | 0.1281 | |
| **本轮合计** | 24,692.481 | **6.8590** | 累计 71.0296 → **77.888623** |

- **口径**：GPU-h = attempt 目录下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600
  （24,692.481 s ÷ 3600 = 6.8590）；**该 attempt 目录内不存在 attempt 级 wall clock 文件**
  （`device_mode=dual` 由 `r7_round.py stage-seeds` 驱动，不生成 `result.json`/`attempt.json`），
  故无从取"更保守者"——6 个进程之和即最保守口径。逐阶段 GPU-h 为四舍五入值，合计取精确和。
- **soft overrun 0、hard overrun 0**（per-seed 28800 s 内，planned 86400 s 内）。
- 网络 **0**（离线，R-028）。本 attempt 的 GPU 时间与 attempt01（身份门失败、0 GPU-h）**分开记账**；
  attempt01 已在 `s3-process-init-run` 登记为 0 GPU-h，故 H-A 因子两 attempt 合计仍是 **6.8590 GPU-h**。
- **门槛口径的实测修正（方向与容量两轮相反）**：协议里 `estimated_peak_bytes` 写 11.91 GiB，
  三 seed 实测 `owned_cuda_reserved_peak_bytes` 都是 **8.361 GiB**（8,977,907,712 B）——估计**高了 3.55 GiB**。
  本轮余量门因而是**保守**的；真正让三 seed 同时跑下来的是 0047 的 6 GiB 邻居 reserve。
  （容量两轮是**低估**，本轮是**高估**，同一口径连续三轮都没标中，如实记录。）

## 7. 已确认与推测

**已确认（读到的产物）**：上表全部 exit_code / elapsed / receipt / 五 lead 评分 CSV / cohort 计数；
三 seed 的 `bit_identical_off_path=true` 与 `proj_zeroing_is_identity=true`；`readings.json` 的逐 lead
rmse / Δ / skill（由驱动在同一 frozen verdict 路径上从原始 CSV 重算）；两个 pinned 对照的 30 张表
全部匹配其 pin；`readings.json` 的 `protocol_sha256` 等于 `protocol.json` 的最终 digest。

**推测（未证）**：① "P_0 无输入信息不是 72 h 的瓶颈"这一命题由本臂**支持但不构成强证**——本臂价格的是
"**从注册父出发的一条逐样本 `P_0` 路由**"，不是该通路在**隔离**下的效应（通路与父权重联合训练）。
② H-A 在长 lead 上比剂量臂**更差**（+0.06~0.08 K）**是否**为真实负效应，本轮**未做显著性检验**，
量级与 seed 间波动同阶（72h 范围 +0.0425…+0.1042），只能读作"三 seed 同号、方向为负"。
③ 72 h 残余负 skill 的**主因仍未定位**：候选有自回归误差累积、目标/训练信号组织、数据覆盖，
三者本轮**未分离**；H-A 这一支现在被排除。

**未做**：不重训练任何已登记臂、不读 test、不做显著性检验、不报泛化/SOTA/收敛因果、不加 seed、
不调 `proj` 宽度、不做 matched_generic 负控制的实训练（只声明，见 limitations）。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；三 seed 只是一致性证据，**无显著性**。
- 两个对照都是 **pinned 复用**，不是同协议内同批重训练；三臂宿主负载不同（与容量轮同样的共驻条件）。
- **通路与父权重联合训练**：本臂价格的是"从注册父出发、每条样本自己 context 的 `P_0` 路由"，
  **不是**该通路在隔离下的效应；`matched_generic` 负控制（同参数量/FLOPs、只改路由）本轮**只声明、未训练**，
  因此"信息增益"与"单纯加参数/优化轨迹变化"**未分离**。
- 72 h 绝对气候态门**未过**；只监督 `interior_32` 使外圈在训练中无约束。
- **显存估计连续第三次未标中**：本轮**高估**（协议 11.91 GiB vs 实测 8.361 GiB）；下一轮标定应按实测下调。
- 可复现等级：`protocol_sha256` + `code.zip` SHA + 逐 seed receipt + `readings.json` 齐备，
  可按同一协议复现；但 **GPU 训练非逐位可复现**（同机同 seed 也不保证逐位一致），
  故只能复现"同一协议"，不能复现"同一结果"。
- **off 路径逐位一致已实测**（逐 seed 正控），但**开关打开后**主干前向确已改变；本页不对"打开后的前向
  与注册臂逐位不同"作任何弱化。

## 9. 下一动作

按**预声明出口**（`docs/goals/s3-process-init-conditioning.md` §3）与本轮读数：
**关闭 H-A 这一支**——不调 `proj` 宽度、不加 seed、不重复同一 test。

H-A 是**容量轴（决策 0048）永久关闭后的第一根架构轴**；它也未让 72 h 三 seed 全正，且相对已登记
2400 剂量臂在长 lead 上更差。按 `docs/plans/0022-s3-architecture-next-round.md`，架构方向还剩
**H-B / H-C**；此外训练信号 / 目标组织方向仍开放（数据按本轮 D3 **暂缓、放到最最后**）。
下一轮必须先有自己的 round brief 与预声明出口，再冻结新协议。

test 未读、r=0、S4 未启动不变。
