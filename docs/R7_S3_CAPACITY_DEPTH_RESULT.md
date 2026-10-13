# S3 主干容量（depth 4→8）单因素臂：72 h 三 seed 全负，按决策 0048 关闭深度轴

**状态：已完成（2026-10-13，三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-capacity-or-architecture.md`。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；`primary`/`gate` 形式在这里只是**报告**，
因为两个对照都是已登记臂的 pinned 复用。

**同一因子的登记序号**：这是 `depth 4→8` 这一因子的**第三次**登记——
① `s3-capacity-depth`（「未运行」，`docs/R7_S3_CAPACITY_DEPTH.md`）、
② `s3-capacity-depth-run`（seed 41 CUDA OOM，`blocked`，`docs/R7_S3_CAPACITY_DEPTH_RUN.md`）、
③ **本页**（`s3-capacity-depth-result`，三 seed 跑完）。前两条**逐字节保留**，本页只在 ② 上追加
`correction_record_id` 指针（① 上的指针已存在，不重复追加）。

## 1. 结论摘要

- **单因素**：在**已登记的最佳臂**上，只把主干 `depth` 从 **4 提到 8**（`backbone.encoder` 里
  `WindowAttentionBlock` 的层数）。`dim` 仍 **192**、K 仍 **4**、宽 129×129 输入、`interior_32` 监督、
  2400 更新、注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、迁移 v3-BD 1600 父、seed 41/42/43、
  评分域与气候态分母**逐字不变**。参数量 3,286,037 → **5,065,493（+54.2%）**。
- **72 h 三 seed 全负**：`interior_32` t2m skill 为 **−0.1512 / −0.2545 / −0.1432**（均值 **−0.1830**）。
  **预声明出口的正向条件未满足**——它要求 72 h **三 seed 全正**。
- **深度是弱正向杠杆**：相对已登记的 2400 剂量臂（**同 seed 配对**、pinned 复用、未重训练），
  24/48/72 h **三 seed 全部更优**（Δrmse 均值 −0.0078 / −0.0259 / −0.0298 K），6/12 h 基本持平或略差
  （+0.0042 / +0.0046 K，只有 1/3 seed 更优）。即**收益集中在长 lead**。
- **量级不够**：72 h 要把 skill 从 −0.1830 推到 0，rmse 需再降约 **0.29 K**；本臂只贡献了 **0.0298 K**，
  约 **1/10**。方向对，幅度远不足——这与"72 h 是模式墙（ACC 0.1702、ACC² 上限 +0.0290）"的判断一致。
- **判定**：按决策 **0048**，本轮的预声明出口关闭的是**深度轴**；宽度是**独立轴**、被封顶为
  **最后一个**容量臂。本臂**不进 S4 冻结包**。

| lead | Δrmse vs 剂量臂（K，seed 均值） | 范围 | 更优 seed | skill 均值 | skill(41) | skill(42) | skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | +0.0042 | −0.0009 … +0.0082 | 1/3 | +0.5812 | +0.5760 | +0.5758 | +0.5917 |
| 12 h | +0.0046 | −0.0030 … +0.0093 | 1/3 | +0.3264 | +0.3198 | +0.3190 | +0.3404 |
| 24 h | −0.0078 | −0.0104 … −0.0062 | **3/3** | +0.4229 | +0.4186 | +0.4122 | +0.4380 |
| 48 h | −0.0259 | −0.0270 … −0.0253 | **3/3** | +0.0821 | +0.0759 | +0.0435 | +0.1270 |
| 72 h | −0.0298 | −0.0360 … −0.0200 | **3/3** | **−0.1830** | −0.1512 | −0.2545 | −0.1432 |

（Δ 为「本臂 − 已登记 2400 剂量臂」，负值 = 本臂更优；两者评分域、气候态分母、cohort pin 逐字相同，
可直接相减。逐 lead 的绝对值与 Δ 见 `outputs/r7_s3_capacity_depth_20261012_attempt05/readings.json`。）

相对**已登记的窄 65×65 800 臂**（`record:s3-rollout-dose`，pinned）本臂 12/24/48/72 h **三 seed 全部更优**
（Δrmse 均值 **−0.0238 / −0.1237 / −0.3345 / −0.3604 K**），6 h 持平（+0.0013 K）。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_capacity_depth_20261012_attempt05/` |
| 协议 digest（`protocol.json`，最终；驱动 `validate_protocol` 自洽校验通过） | `a8e472f7d888d94bf2b3858f804c805abde7ea2dffd64c8ef4514e723e5d57a2` |
| 归档 `code.zip` SHA256 | `e8802b7f580ead86a2a10e62fb1793ca6635b7db0354b6fc614c53b0a3987502`（1390 条目） |
| ZIP comment / 代码 commit | `4e9fc96dde8134704786d4de0539a3d70b04c169`（comment 与 `code_commit` 逐字一致） |
| `model_code_sha256` | `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab` |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| 执行文件身份 | 187 个（逐文件核过归档） |
| 监督块（冻结） | `spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576b…`（与对照臂相同） |
| **深度扩张（冻结规则）** | parent `depth 4` → target `8`；`added_blocks [4,5,6,7]`；新增块 `i` 在**每个参数**上镜像块 `i-4`，并把 `proj` 与 `ff.net.3` 的输出投影**清零** ⇒ 每个新增块恰为恒等；**逐 seed** 在固定合成批 `(1,2,17,129,129)`、seed 1234 上断言 `encoder_bit_identical=true`、`non_block_keys_unchanged=true`（三 seed 全部通过） |
| 剂量（冻结） | `updates=2400`、`lr=2e-5`、`warmup=10`、K=4、FP32、batch 1、clip 1、12 步物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` |
| 对照 1 | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| 对照 2 | 窄 65×65 800 臂（`record:s3-rollout-dose`，pinned，**未重训练**） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=8.5 GiB`；`headroom_margin_bytes=0.5 GiB` |
| 预算 | planned 86400 s / hard 115200 s / per-seed 28800 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，三 seed 一致） |
| test | `test.jsonl` 从不读取；`test_read=false` |

## 3. 训练与评分回执

| seed | checkpoint SHA256 | 训练 s | 五 lead 评分 s | 训练损失 1→2400 | 峰值 reserved |
| ---: | --- | ---: | ---: | --- | ---: |
| 41 | `51df06081298b3e8c1009e3c5911518b5ad221f270880194881fd3b5e70da1f3` | 6154.9 | 1494.2 | 1.6504→0.8264 | 11.03 GiB |
| 42 | `a928fd4519b7ffa9cc5363251b2b5e673286dbade8ceedb0ba76c59bf6225b05` | 8198.9 | 1517.8 | 1.2072→0.5746 | 11.03 GiB |
| 43 | `804113b6edd9f2c447f616e98823fd0e59590bc24e762c9d99294eaac0177db4` | 8106.4 | 1497.0 | 1.4735→0.7384 | 11.03 GiB |

- 三 seed 都到 `total_updates=2400`；每 seed 每 lead `n_evaluated == n_available_windows` 且等于冻结
  cohort（472/468/460/444/428）；每 seed 的 `training_report.json` 的 `supervision` 与冻结
  `mask_sha256` 逐字相同（驱动断言）。
- 三 seed 的 `depth_expansion` 断言全部通过；逐 seed 的 `parent_state_sha256` /
  `expanded_state_sha256` / `encoder_output_sha256` 随各自父 checkpoint 不同而不同（父分别
  `fba27e26…` / `5d4b05ed…` / `becc3d78…`），这正是"逐 seed 独立断言"的意义。
- seed 42/43 的训练墙钟（8198.9 / 8106.4 s）明显长于 seed 41（6154.9 s），因为 42 与 41 并发、
  43 与邻居共驻占卡——**墙钟不是科学判据**，GPU-h 记账见 §6。

## 4. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 370.960 |
| archive | `archive_process.json` | 0 | success | 374.620 |
| seed 41 | `seed41_process.json` | 0 | success | 8949.900 |
| seed 42 | `seed42_process.json` | 0 | success | 11438.040 |
| seed 43 | `seed43_process.json` | 0 | success | 11453.960 |
| reading | `reading_process.json` | 0 | success | 583.244 |

六阶段全 `success`、`returncode=0`、无信号；每阶段 spawn 前只读 `gpu_gate`；
**未对任何非本实验进程发送任何信号**（邻居全程只共享显存）。

## 5. 公平性守卫

- 三 seed 的逐 lead cohort 计数与已登记对照臂**逐字相同**（472/468/460/444/428）。
- reading 阶段按 `boundary_rmse_csv_sha256` 复核 2400 剂量臂、按 `rmse_csv_sha256` 复核窄臂，
  **30/30 全部匹配**（两个对照都**未重训练**）。
- 气候态分母取自 `wide_climatology_reference`（train-only、与对照臂同一分母），`readings.json`
  里的 `protocol_sha256` 与 `protocol.json` 的最终 digest 逐字一致。
- 深度扩张的"起点等于注册函数"由**逐 seed 的逐位一致断言**保证（§2），因此本臂与对照臂的差别
  被限制在 `depth` 这一个字段上。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 370.960 | 0.1030 | |
| archive（CPU） | 374.620 | 0.1041 | |
| seed 41（2400 更新 + 五 lead） | 8949.900 | 2.4861 | |
| seed 42（2400 更新 + 五 lead） | 11438.040 | 3.1772 | 与 41 并发 |
| seed 43（2400 更新 + 五 lead） | 11453.960 | 3.1817 | 邻居共驻占卡 |
| reading（CPU） | 583.244 | 0.1620 | |
| **本轮合计** | 33,170.720 | **9.2141** | 累计 53.8478 → **63.0619** |

- **soft overrun 0、hard overrun 0**（per-seed 28800 s 内，planned 86400 s 内）。
- 网络 **0**（离线，R-028）。本 attempt 的 GPU 时间与 attempt04 那次 OOM 的 0.5414 GPU-h
  **分开记账、都全额计费**（决策 0045）。
- **门槛口径的实测修正**：协议里 `estimated_peak_bytes` 写 8.5 GiB，三 seed 实测
  `owned_cuda_reserved_peak_bytes` 都是 **11.03 GiB**——估计**低了 2.5 GiB**。真正兜住这一轮的是
  0047 新增的 6 GiB 邻居 reserve（邻居当时只用 5.2 / 4.7 GiB，reserve 之外还留了余量）。
  这是一次**资源判断被实测证伪一半**的记录，下一轮标定必须用 11.03 GiB 而不是 8.5 GiB。

## 7. 已确认与推测

**已确认（读到的产物）**：上表全部 exit_code / elapsed / receipt / 五 lead 评分 CSV / cohort 计数；
三 seed 的 `encoder_bit_identical=true`；`readings.json` 的逐 lead rmse / Δ / skill（与主链独立
复算逐位一致）；两个 pinned 对照的 30 张表全部匹配其 pin。

**推测（未证）**：① "加容量对 72 h 无用"这一更强命题**不能**由本臂单独推出——它只关闭**深度**轴；
宽度轴在当前主线上尚未单因素检验（决策 0048）。② 深度在长 lead 上的微弱正向（三 seed 同号、
Δ 约 −0.03 K）**是否**为真实效应，本轮**未做显著性检验**，量级与 seed 间波动同阶，只能读作"方向一致"。
③ 72 h 的残余负 skill 的**主因仍未定位**：候选有自回归误差累积、目标/训练信号组织、数据覆盖，
三者本轮**未分离**。

**未做**：不重训练任何已登记臂、不读 test、不做显著性检验、不报泛化/SOTA/收敛因果、不加 seed。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；三 seed 只是一致性证据，**无显著性**。
- 两个对照都是 **pinned 复用**，不是同协议内同批重训练；三臂宿主负载不同（邻居共驻时有时无），
  seed 42/43 的训练与评分因此明显更慢。
- 72 h 绝对气候态门**未过**；只监督 `interior_32` 使外圈在训练中无约束。
- **深度扩张把两个东西绑在了一起**：`depth` 本身，以及"新增块由镜像 + 清零初始化"这个起点。
  本臂价格的是"**从注册的 depth-4 函数出发、再加四层**"，不是"任意 depth-8 主干"。
  逐位一致断言把起点固定住了，但没有分离"更多层"与"更强的恒等起点"这两个成分。
- **显存估计被证伪**：协议写 8.5 GiB、实测 11.03 GiB（§6）；下一轮标定必须按实测。
- 可复现等级：`protocol_sha256` + `code.zip` SHA + 逐 seed receipt + `readings.json` 齐备，
  可按同一协议复现；但 **GPU 训练非逐位可复现**（同机同 seed 也不保证逐位一致），
  故只能复现"同一协议"，不能复现"同一结果"。

## 9. 下一动作

按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）与本轮预声明出口：
**关闭深度轴**——不再重试 `depth`、不再加 seed、不重复同一 test、不做宽＋深组合。

容量轴上还剩**唯一一个**名额：**宽度单因素**（`dim` 192 → 256/320，其余逐字不变）。它**必须**：
① 先有自己的 round brief 与预声明出口；② 显式处理"宽度**没有**逐位恒等扩张"——
实测 `dim` 192→256 时 state_dict 的 155 个张量里 **150 个形状变化**，已登记的 dim-192 父状态
**装不进去**，因此"宽度"必然与"初始化/优化轨迹"混杂，**不得**照搬 depth 的 identity-gated 措辞；
③ 按实测把 `estimated_peak_bytes` 标到 **11.03 GiB** 以上（§6）。
若宽度也未能让 72 h 三 seed 全正，则按 0048 的封顶条款，**整条容量轴永久关闭**，转训练信号 /
目标组织 / 数据方向（数据按 D3 暂缓、放到最最后）。

同样合法的替代路径（与出口原文完全一致、零规则风险）：直接走
`docs/plans/0022-s3-architecture-next-round.md` 的 **H-A**（P_0 输入条件化），不碰容量轴。

test 未读、r=0、S4 未启动不变。
