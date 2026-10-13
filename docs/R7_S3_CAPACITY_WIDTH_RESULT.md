# S3 主干容量（width dim 192→256）臂：72 h 三 seed 全负，按决策 0048 永久关闭整条容量轴（深度＋宽度）

**状态：已完成（2026-10-14，三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-capacity-width.md`。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；`primary`/`gate` 形式在这里只是**报告**，
因为两个对照都是已登记臂的 pinned 复用。

**同一因子的登记序号**：这是 `width dim 192→256` 这一因子的**第一条**登记
（`s3-capacity-width-result`）。本因子此前**没有任何**已登记 record，故**不追加** `correction_record_id`
指针（与 depth 因子的三条登记链不同）。

## 1. 结论摘要

- **名义单因素**：在**已登记的最佳臂**上，只把主干 `dim` 从 **192 提到 256**（`backbone.encoder` 里
  `WindowAttentionBlock` 的宽度）。`depth` 仍 **4**、K 仍 **4**、宽 129×129 输入、`interior_32` 监督、
  2400 更新、注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、迁移 v3-BD 1600 父、seed 41/42/43、
  评分域与气候态分母**逐字不变**。参数量 3,286,037 → **5,790,389（+76%）**。
- **⚠ 本臂不是"宽度单因素"（必须照此读）**：宽度**没有逐位恒等扩张**——`dim` 192→256 时 state_dict 的
  **155 个张量里 150 个形状变化**，注册的 dim-192 父状态（v3-BD 1600，dim-192/depth-4）**装不进去**。
  depth 轮靠 identity-gated 镜像 + 输出投影清零做到"起点逐位等于注册函数"（`encoder_bit_identical`）；
  **宽度没有这种构造**。本臂用**冻结零填充扩张**：每个父张量按重叠切片拷入目标张量、其余位置置零
  （1-D 参数拷前 192、余 64 置零）。LayerNorm 在 256 维上归一化、被置零的高 64 维并非严格惰性，
  故本臂测的是「**宽度 + 该零填充初始化**」的**联合效应**，**不是**宽度单因素。
  此处**不照搬** depth 轮的「单因素保真」措辞。
- **72 h 三 seed 全负**：`interior_32` t2m skill 为 **−0.1658 / −0.2833 / −0.1418**（均值 **−0.1970**）。
  **预声明出口的正向条件未满足**——它要求 72 h **三 seed 全正**。
- **宽度不是正向杠杆**：相对已登记的 2400 剂量臂（**同 seed 配对**、pinned 复用、未重训练），
  6/12/24/48 h **三 seed 全部更差**（Δrmse 均值 **+0.2880 / +0.2751 / +0.0705 / +0.0419 K**，0/3 seed 更优），
  只有 72 h 略好（**−0.0090 K**，2/3 seed 更优）。即宽度在短/中 lead 明显更差、在 72 h 基本持平。
- **判定**：按决策 **0048**，宽度是**最后一个**容量臂；它未能让 72 h 三 seed 全正 ⇒ **整条容量轴
  （深度＋宽度）永久关闭**。此后只剩训练信号 / 目标组织 / 数据（数据按 D3 暂缓、放最最后）。
  本臂**不进 S4 冻结包**。

| lead | Δrmse vs 剂量臂（K，seed 均值） | 范围 | 更优 seed | skill 均值 | skill(41) | skill(42) | skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | +0.2880 | +0.2304 … +0.3394 | 0/3 | +0.4701 | +0.4445 | +0.4858 | +0.4800 |
| 12 h | +0.2751 | +0.1791 … +0.3232 | 0/3 | +0.1930 | +0.1633 | +0.2301 | +0.1854 |
| 24 h | +0.0705 | +0.0426 … +0.1053 | 0/3 | +0.3879 | +0.3873 | +0.3883 | +0.3880 |
| 48 h | +0.0419 | +0.0267 … +0.0602 | 0/3 | +0.0430 | +0.0459 | +0.0047 | +0.0786 |
| 72 h | −0.0090 | −0.0222 … +0.0088 | 2/3 | **−0.1970** | −0.1658 | −0.2833 | −0.1418 |

（Δ 为「本臂 − 已登记 2400 剂量臂」，**负值 = 本臂更优**；两者评分域、气候态分母、cohort pin 逐字相同，
可直接相减。逐 lead 的绝对值与 Δ 见 `outputs/r7_s3_capacity_width_20261013_attempt01/readings.json`；
seed 均值为对 `readings.json` 的逐 seed Δ 取算术平均，取值口径与命令见 §6。）

相对**已登记的窄 65×65 800 臂**（`record:s3-rollout-dose`，pinned）本臂 24/48/72 h 更优
（Δrmse 均值 **−0.0454 / −0.2666 / −0.3396 K**，48/72 h **3/3** seed 更优），6/12 h 更差
（**+0.2851 / +0.2466 K**，0/3 seed 更优）。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_capacity_width_20261013_attempt01/` |
| 协议 digest（`protocol.json`，最终；驱动 `validate_protocol` 自洽校验通过） | `277cf27e75043815d0dad7729b1e78e17ac8f0d8dde05acd6f35c05a12df8039` |
| 归档 `code.zip` SHA256 | `14079fe608d3c2c113bc59cd8746a4bd3410b758906b8a5184bf37057aecb15d`（1395 条目） |
| ZIP comment / 代码 commit | `35b1785904716804a3a749bd47bdaf629e580f3f`（comment 与 `code_commit` 逐字一致） |
| `model_code_sha256` | `d3fb58dbd0ed9efbd249fc258cb09c488d543c0a8ad77dac5689ac8f9ba77bab`（未变） |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| train `data_identity` | `2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12` |
| val `data_identity` | `f590883e7e1413e68cf566b5ba002b737ac35a49878250d093eda3b4c65de933` |
| 执行文件身份 | 187 个（逐文件核过归档） |
| 监督块（冻结） | `spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576b…`（与对照臂相同） |
| **宽度扩张（冻结规则，非恒等）** | parent `dim 192` → target `256`；每父张量按重叠切片 `slice(0,min(src,tgt))` 拷入目标、其余位置**置零**；1-D 参数拷前 192、余 64 置零；`bit_identical=**false**`；混杂见 §1 |
| 剂量（冻结） | `updates=2400`、`lr=2e-5`、`warmup=10`、K=4、FP32、batch 1、clip 1、12 步物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)` |
| 对照 1 | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| 对照 2 | 窄 65×65 800 臂（`record:s3-rollout-dose`，pinned，**未重训练**） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=11.0352 GiB`；`headroom_margin_bytes=0.5 GiB` |
| 预算 | planned 86400 s / hard 115200 s / per-seed 28800 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，三 seed 一致） |
| test | `test.jsonl` 从不读取；`test_read=false` |
| **实验 commit 的 CI（如实）** | `R7 CPU CI` run **`38391291040`**（commit `35b1785`）**completed / failure**：第 5–8 步（conventions / campaign state / evidence index / compile）**全绿**，红的是第 9 步全量测试——根因是**本臂自己的测试文件** `tests/test_r7_s3_capacity_width.py:392` 的 `_protocol_body` 对 `capacity.execution_files()` 逐个 `sha256_file(ROOT/name)`，其中含 `outputs/r7_v2_comparison_20261004_attempt01/protocol.json`，该路径在**无 `outputs/` 的干净 CI checkout 上不存在** ⇒ 4 个用例 `FileNotFoundError`（本机因 `outputs/` 存在故通过）。这是**测试侧工程缺陷**，不是读数/科学问题；本轮**未改**该测试（登记员只追加本轮材料）。 |

## 3. 训练与评分回执

| seed | checkpoint SHA256 | 训练 s | 五 lead 评分 s | 训练损失 1→2400 | 峰值 reserved |
| ---: | --- | ---: | ---: | --- | ---: |
| 41 | `c84a2614f1117e215ff3978378af2867564a414f538e0b6d26742f464bc149fe` | 5106.3 | 951.8 | 1.3929→0.7944 | 11.90 GiB |
| 42 | `222dc5e18f0056ce4d1fa86d7e11da0c6d9bb942cacc734536864cfeb7c09625` | 5165.7 | 903.3 | 1.6195→0.5754 | 11.90 GiB |
| 43 | `56c348b48b5df757594cb18422b9dafd3019cc171e7871f429334979adbf5af8` | 10661.7 | 910.2 | 1.7184→0.7336 | 11.90 GiB |

- 三 seed 都到 `total_updates=2400`、`selected_update=2400`；每 seed 每 lead `n_evaluated == n_available_windows`
  且等于冻结 cohort（472/468/460/444/428）；每 seed 的 `training_report.json` 的 `supervision` 与冻结
  `mask_sha256` 逐字相同（驱动断言）。
- 三 seed 的 `width_expansion` 断言全部通过：`non_block_keys_unchanged=true`、`left_at_random=[]`、
  `encoder_output_finite=true`（固定合成批 `(1,2,17,129,129)`、seed 1234）。逐 seed 的
  `parent_state_sha256` / `expanded_state_sha256` 随各自父 checkpoint 不同而不同——这正是"逐 seed 独立断言"的意义：

| seed | 迁移父 checkpoint SHA256 | `parent_state_sha256` | `expanded_state_sha256` |
| ---: | --- | --- | --- |
| 41 | `fba27e26…` | `7c416ccb…` | `69ba6fd9…` |
| 42 | `5d4b05ed…` | `857d8804…` | `b018d689…` |
| 43 | `becc3d78…` | `61fdac58…` | `13979de4…` |

  **注意：本臂没有 `encoder_bit_identical` 断言**（宽度无恒等扩张）；只断言非块键未变与输出有限。
- seed 43 的训练墙钟（10661.7 s）明显长于 seed 41/42（5106.3/5165.7 s），因为 41 与 42 分占两张卡并发、
  43 与邻居共驻占卡——**墙钟不是科学判据**，GPU-h 记账见 §6。

## 4. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 405.806 |
| archive | `archive_process.json` | 0 | success | 399.749 |
| seed 41 | `seed41_process.json` | 0 | success | 7298.404 |
| seed 42 | `seed42_process.json` | 0 | success | 7282.880 |
| seed 43 | `seed43_process.json` | 0 | success | 12831.100 |
| reading | `reading_process.json` | 0 | success | 465.644 |

六阶段全 `success`、`returncode=0`、无信号；每阶段 spawn 前只读 `gpu_gate`；
**未对任何非本实验进程发送任何信号**（邻居全程只共享显存）。

## 5. 公平性守卫

- 三 seed 的逐 lead cohort 计数与已登记对照臂**逐字相同**（472/468/460/444/428）。
- reading 阶段按 `boundary_rmse_csv_sha256` 复核 2400 剂量臂、按 `rmse_csv_sha256` 复核窄臂，
  两个对照都**未重训练**。
- 气候态分母取自 `wide_climatology_reference`（train-only、与对照臂同一分母），`readings.json`
  里的 `protocol_sha256` 与 `protocol.json` 的最终 digest 逐字一致（`277cf27e…`）。
- **宽度扩张的"起点"不是注册函数**：见 §1/§8——本臂价格的是"从注册父出发、经**冻结零填充**再加宽"，
  不是"任意 dim-256 主干"。逐 seed 只断言非块键未变与输出有限，**没有**逐位一致断言。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 405.806 | 0.1127 | |
| archive（CPU） | 399.749 | 0.1110 | |
| seed 41（2400 更新 + 五 lead） | 7298.404 | 2.0273 | |
| seed 42（2400 更新 + 五 lead） | 7282.880 | 2.0230 | 与 41 并发 |
| seed 43（2400 更新 + 五 lead） | 12831.100 | 3.5642 | 邻居共驻占卡 |
| reading（CPU） | 465.644 | 0.1293 | |
| **本轮合计** | 28,683.583 | **7.9677** | 累计 63.0619 → **71.0296** |

- **GPU-h 口径（保守）**：attempt 目录下**全部 6 个** `*_process.json` 的 `elapsed_seconds` 之和
  = 28683.583343 s，÷ 3600 = **7.9677**（表内分项四舍五入到 4 位、合计取未四舍五入总和）。
  attempt 目录下**不存在** attempt 级 wall clock 文件，故无更保守者可比（口径由本轮任务规定）。
- **soft overrun 0、hard overrun 0**（per-seed 28800 s 内，planned 86400 s 内）。
- 网络 **0**（离线，R-028）；磁盘 ≈14 GiB。
- **门槛口径的实测修正（连续第二次）**：协议 `estimated_peak_bytes` 写 **11.0352 GiB**（按 depth 轮实测
  11.03 GiB 标定），三 seed 实测 `owned_cuda_reserved_peak_bytes` 都是 **11.9023 GiB**（12,780,044,288 字节）
  ——**仍低估约 0.87 GiB**。真正兜住这一轮的是 0047 的 **6 GiB 邻居 reserve**。下一轮标定必须用 ≥11.90 GiB。
- **seed 均值口径**：`readings.json` 只写逐 seed 的 Δ 与 skill；seed 均值 = 对 `readings.json` 的逐 seed 值取
  算术平均，用 `.venv/bin/python` 读取该 JSON 后计算（命令见本轮结果登记提交）。

## 7. 已确认与推测

**已确认（读到的产物）**：上表全部 exit_code / elapsed / receipt / 五 lead 评分 CSV / cohort 计数；
三 seed 的 `non_block_keys_unchanged=true` 与 `encoder_output_finite=true`；`readings.json` 的逐 lead
rmse / Δ / skill；两个 pinned 对照按哈希复核匹配其 pin；实验 commit `35b1785` 的 CI `38391291040`
在第 9 步（全量测试）红、根因是**本臂测试文件**对缺失 `outputs/` 路径取哈希（§2，从 GitHub
check-runs annotations 读到，非猜测）。

**推测（未证）**：① 宽度的**净效应**（剥离零填充初始化）本轮**未分离**——本臂只给出"宽度 + 冻结零填充"
的联合效应；② 短/中 lead 明显变差（+0.07…+0.29 K）**是否**主要由该零填充初始化/优化轨迹而非宽度本身造成，
本轮**未做**任何分离实验；③ 72 h 的残余负 skill 的**主因仍未定位**：候选有自回归误差累积、目标/训练信号组织、
数据覆盖，三者本轮**未分离**。

**未做**：不重训练任何已登记臂、不读 test、不做显著性检验、不报泛化/SOTA/收敛因果、不加 seed、
不做宽＋深组合、**未修**本臂测试文件的 CI 缺陷（登记员只追加材料）。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；三 seed 只是一致性证据，**无显著性**。
- 两个对照都是 **pinned 复用**，不是同协议内同批重训练；三臂宿主负载不同，seed 43 明显更慢。
- **本臂把两个东西绑在了一起**：宽度本身，以及"注册父经**冻结零填充**扩张"这个起点。
  实测 `dim` 192→256 时 155 个张量里 **150 个形状变化**，**无逐位恒等扩张**（决策 0048 第 4 条）。
  本臂价格的是"**从注册父出发、用冻结零填充再加宽**"，**不是**"任意 dim-256 主干"、**也不是**宽度单因素。
  此处**不得**照搬 depth 轮的 identity-gated「单因素保真」措辞。
- 只监督 `interior_32` 使外圈在训练中无约束；72 h 绝对气候态门**未过**。
- **显存估计被证伪（第二次）**：协议写 11.0352 GiB、实测 11.9023 GiB（§6）；下一轮标定必须按实测。
- **实验 commit 的 CI 为红**（§2）：第 9 步全量测试因本臂测试文件对缺失 `outputs/` 路径取哈希而失败；
  这是工程缺陷、非读数问题，但意味着该 commit 的 CI **不是绿的**。
- 可复现等级：`protocol_sha256` + `code.zip` SHA + 逐 seed receipt + `readings.json` 齐备，
  可按同一协议复现；但 **GPU 训练非逐位可复现**，且**宽度扩张本身不是恒等**，故只能复现"同一协议 + 同一零填充起点"。

## 9. 下一动作

按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）与本轮预声明出口：
宽度臂 72 h 三 seed 全负 ⇒ **整条容量轴（深度＋宽度）永久关闭**——此后**不得**再开任何容量臂：
不重试 `depth`、不再扫 `dim`、不做宽＋深组合、不以"再调一点容量"为名复活该支。

此后只剩**训练信号 / 目标组织 / 数据**三根轴，且**数据按本轮 Objective 的 D3 暂缓、放到最最后**。
零规则风险的下一步是 `docs/plans/0022-s3-architecture-next-round.md` 的 **H-A**（P_0 输入条件化）
等架构/训练信号方向；开跑前先写自己的 round brief 与预声明出口。

test 未读、r=0、S4 未启动不变。
