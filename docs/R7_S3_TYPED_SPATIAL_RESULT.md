# S3 H-B 空间分辨的类型化证据（pooled → 保留 patch 位置）单因素臂：B→C 无差异（~1e-6 K），72 h 三 seed 全负，按预声明出口关闭 H-B 支

**状态：已完成（2026-10-14，两臂 × 三 seed 全部跑完并评分）；`scientific_claim: false`；2023 test 未读；r 不消耗。**
本页对应 `docs/goals/main-model-climatology-campaign.md` §8 与 `docs/goals/s3-typed-spatial.md`。
它是**开发筛选读数**，不是科学确认，也不是 S4 就绪声明；`primary`/`gate` 形式在这里只是**报告**，
因为外部对照（A）是已登记臂的 pinned 复用，而协议内对照（B）是本轮同协议内重训练的。

**同一因子的登记序号**：这是「类型化证据 token 的空间路由」这一因子的**第一次**登记。
此前 #79 的 `record:s2-79-typed-evidence-screening` 是**不同实例**（不同协议、不同对照），
**不是**同协议对照，故本页**不**向任何旧 record 追加 `correction_record_id`。

## 1. 结论摘要

- **单因素**：在**已登记的最佳臂**上，只把 typed-evidence 通路对 patch 位置的**聚合方式**从
  `pooled`（对 P 个位置取均匀均值，即 #79 现状）改为 `spatial`（用一个共享可学习 query 对 P 做
  softmax 注意力）。其余（宽 129×129 输入、`interior_32` 监督、2400 更新、K=4、depth=4、`dim`=192、
  注册物理权重 `(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`、迁移 v3-BD 1600 父、seed 41/42/43、评分域与气候态
  分母）逐字不变。**唯一因子是证据 token 在 patch 位置维上的聚合方式**（pooled 均匀均值 vs spatial
  softmax 注意力）。
- **B/C 同协议内同跑**：pooled（B，`mode=typed_routing`）与 spatial（C，`mode=typed_spatial`）在**同一协议、
  同一 seed、同一预算、同一通路初始化**下各跑一遍。两臂共享张量**逐位一致**（逐 seed 的
  `pathway_init_sha256` 两臂相等，见 §3），C 恰多一个 `typed_evidence.position_query`（**+192 参数**，
  `spatial_extra_params=192`）。query 为 0 时 C 的聚合**等于** B 的均值（`zero_query_is_pooled=true`、
  `zero_query_max_abs=0.0`）；清零四个 per-type 投影则两模式证据**恒为 0**（`zeroed_projection_is_identity=true`）。
- **B→C 无差异**：C−B 的 `interior_32` t2m Δrmse 在五个 lead 上 seed 均值为
  **+1.2e-7 / −1.3e-6 / −2.8e-6 / −6.5e-8 / +3.2e-6 K**（|Δ| ≤ 3.2e-6 K，即 float32 数值噪声量级），
  且 72 h **三 seed 不同号**（−5.2e-6 / +1.3e-5 / +1.9e-6）。即位置聚合**改变不了**读数。
- **72 h 三 seed 全负**：C 臂 `interior_32` t2m skill 为 **−0.1827 / −0.2791 / −0.1596**（均值 −0.2071），
  B 臂几乎同值（−0.1827 / −0.2791 / −0.1596，均值 −0.2071）。**预声明出口**（B→C 在 72 h 三 seed 同号
  **且** C 的 72 h 三 seed 全正）**未满足**——两条都未满足。
- **结论（必须照此读）**：B→C 无差异 ⇒ 结论是**类型路由本身无效**（同信息、不同路由不改变读数），
  **不是**「空间保留无用」。72 h 的负 skill 在两臂里都出现，故它与位置聚合无关。
- **相对已登记 2400 剂量臂 A**（`record:s3-wide-interior-dose`，pinned 复用、未重训练）：spatial 臂
  6/12 h 略优（Δ −0.0026 / −0.0022 K，3/3 seed），24/48/72 h 略差（+0.0011 / +0.0052 / +0.0068 K，
  1/3、0/3、0/3 seed 更优）。
- **相对已登记窄 65×65 800 臂**（`record:s3-rollout-dose`，pinned）：spatial 臂 12/24/48/72 h 三 seed 全优
  （Δ −0.0306 / −0.1148 / −0.3033 / −0.3238 K），6 h 基本持平（−0.0055 K，2/3）。
- **判定**：按**决策 0048**，H-B 是 H-A 关闭后的第二根**架构**轴；本轮预声明出口未满足 ⇒ **关闭 H-B
  这一支**——不调注入宽度、不加 seed、不重跑。本臂**不进 S4 冻结包**。

**表 A：B→C 配对 delta（C − B，同协议同 seed 同通路初始化）**

| lead | Δrmse B→C（K，seed 均值） | 范围 | 三 seed 同号 | C skill 均值 | C skill(41) | C skill(42) | C skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | +1.2e-7 | −1.1e-6 … +1.0e-6 | 否 | +0.5837 | +0.580152 | +0.575910 | +0.594942 |
| 12 h | −1.3e-6 | −2.8e-6 … −0.5e-6 | 是（全负） | +0.3296 | +0.325227 | +0.318130 | +0.345453 |
| 24 h | −2.8e-6 | −4.0e-6 … −1.2e-6 | 是（全负） | +0.4190 | +0.415332 | +0.407812 | +0.433864 |
| 48 h | −6.5e-8 | −3.1e-6 … +4.4e-6 | 否 | +0.0643 | +0.057363 | +0.025748 | +0.109649 |
| 72 h | +3.2e-6 | −5.2e-6 … +1.3e-5 | **否** | **−0.2071** | −0.182708 | −0.279100 | −0.159561 |

**表 B：spatial（C）相对已登记 2400 剂量臂 A 的配对 delta**

| lead | Δrmse vs 剂量臂（K，seed 均值） | 范围 | 更优 seed | skill 均值 | skill(41) | skill(42) | skill(43) |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 6 h | −0.0026 | −0.0037 … −0.0012 | **3/3** | +0.5837 | +0.580152 | +0.575910 | +0.594942 |
| 12 h | −0.0022 | −0.0033 … −0.0011 | **3/3** | +0.3296 | +0.325227 | +0.318130 | +0.345453 |
| 24 h | +0.0011 | −0.0006 … +0.0025 | 1/3 | +0.4190 | +0.415332 | +0.407812 | +0.433864 |
| 48 h | +0.0052 | +0.0034 … +0.0069 | 0/3 | +0.0643 | +0.057363 | +0.025748 | +0.109649 |
| 72 h | +0.0068 | +0.0027 … +0.0123 | 0/3 | **−0.2071** | −0.182708 | −0.279100 | −0.159561 |

（Δ 为「本臂 − 对照」，负值 = 本臂更优；B 与 C 的评分域、气候态分母、cohort pin 逐字相同，A 为
同 seed 配对的 pinned 复用，故可直接相减。逐 lead 的绝对值与 Δ 见
`outputs/r7_s3_typed_spatial_20261014_attempt01/readings.json`。C 与 B 的逐 lead Δ 见**表 A**。）

相对**已登记的窄 65×65 800 臂**（`record:s3-rollout-dose`，pinned；其逐 seed 逐 lead t2m RMSE 取自该臂
自己的 `readings.json` 的 `candidate_rmse`，本轮的驱动**不**复核该臂 pin）spatial 臂 12/24/48/72 h
**三 seed 全部更优**（Δrmse 均值 **−0.0306 / −0.1148 / −0.3033 / −0.3238 K**），6 h 基本持平
（−0.0055 K，2/3 seed）。pooled 臂相对窄臂的均值与 spatial 几乎相同（−0.0055 / −0.0306 / −0.1148 /
−0.3033 / −0.3238 K）——再次印证 B≈C。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| attempt 目录 | `outputs/r7_s3_typed_spatial_20261014_attempt01/` |
| 协议 digest（`protocol.json`，最终；嵌入字段；与 `readings.json` 逐字一致） | `7a529c54c88197d4fa94e75b64baf41bbe1cd9b86ebdc8ebf34314015b0a27b1` |
| 准备/预备 digest（`freeze` 阶段报告，`preparation_protocol.json`） | `1ef9dc6c1ea15f74a41e87a5af49f8f9134d8070b099884c63e8f569fc0542b0` |
| 归档 `code.zip` SHA256 | `c32c73f3283a3ccda079f1b085c6e92ddd4e13f01959061f6f39fa746deac991`（1406 条目） |
| ZIP comment / 代码 commit | `3c1d2b050bc9f2f7c09913020531ae7013d001ee`（comment 与 `code_commit.txt` 逐字一致） |
| `model_code_sha256` | `15523927b4584a6a5d2a351c017a1549bc31b816e0142097bee645d493466237`（本轮改了 `model/*.py`，故与 H-A 的 `223b2e5b…` 不同） |
| `source_sha256` | `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701` |
| `training_code_sha256` | `8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974` |
| 执行文件身份 | 188 个（`execution_files_sha256` 的键数） |
| 监督块（冻结） | `spatial_mask`、`[129,129]`、4225 点、`mask_sha256=4a2a576bf948b92924720fa28a44c128a61445dd29b4906805930430e0a0919a`（与对照臂相同） |
| typed evidence | sidecar store `outputs/r7_s1_seasons_2017/store/cache.zarr`；17 通道；`field_active` 全 `true`；两臂 `mode` 分别为 `typed_routing`（B）/ `typed_spatial`（C） |
| **声明差异（冻结）** | C 恰多 `typed_evidence.position_query`（`dim`=192 参数）；pooled 通路 148800 参数、spatial 通路 148992 参数（**+0.13%**）；query 为 0 时两臂聚合相等 |
| 父（迁移 v3-BD 1600，逐 seed pin） | 41 `a376fdac…` / 42 `5f395bde…` / 43 `4fa69a8a…`（`outputs/r7_s3_rollout_dose_parent_migrated_20261014_attempt02/`） |
| 设备（决策 0047 双卡） | `device_mode=dual`；`gpu_uuids=[GPU-408ad137…, GPU-9d1624af…]`；`neighbour_reserve_bytes=6 GiB`；`estimated_peak_bytes=9.5 GiB`；`headroom_margin_bytes=0.5 GiB` |
| 预算 | planned 86400 s / hard 115200 s / per-seed 43200 s（**soft/hard overrun 均 0**） |
| cohort pin | 472 / 468 / 460 / 444 / 428（逐 lead，两臂三 seed 一致） |
| 外部参考 A | 2400 剂量臂（`record:s3-wide-interior-dose`，pinned，**未重训练**） |
| 次级参考 | 窄 65×65 800 臂（`record:s3-rollout-dose`，pinned，**未重训练**） |
| test | `test.jsonl` 从不读取；`test_read=false` |

## 3. 训练与评分回执

| seed | 臂 | checkpoint SHA256 | 训练 s | 五 lead 评分 s | 通路初始化 `pathway_init_sha256` | 峰值 reserved |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 41 | pooled (B) | `ac9c10ab3b8cd0795a55e612d396a2dee5ca75577930b9c2c661a59dcdc3efc8` | 5633.7 | 2547.6 | `18dfa0a2f042362f47578f42716b7b9bdfd730425c6d387a114bd14c0a245843` | 9.04 GiB |
| 41 | spatial (C) | `94c82de9eac9a95dbe8796e33954afaa939b65d82d6d73b3f160dfd2a749e074` | 6532.9 | 1059.3 | `18dfa0a2f042362f47578f42716b7b9bdfd730425c6d387a114bd14c0a245843` | 9.28 GiB |
| 42 | pooled (B) | `4ea96b4c8f03b898e563e6e72a44b5c4b32fa5a1e5ad03fd26f97d04eaf54d4c` | 6399.2 | 2989.0 | `e5d78ca9d33fee76a110d63ef99c1f936ff37e595f122c708f86abcd1afdc5ed` | 9.04 GiB |
| 42 | spatial (C) | `d5a2334e1de5d5901ec03002ef40eb4e29068c37c24199456fb22441ca6ec737` | 8038.4 | 1108.3 | `e5d78ca9d33fee76a110d63ef99c1f936ff37e595f122c708f86abcd1afdc5ed` | 9.28 GiB |
| 43 | pooled (B) | `c66d23cccf054c423432a6ca04c2122eb1ddfe038a1a0f8be63a862102aa7e1d` | 5599.1 | 1057.0 | `2a7f855e06fbc2012e713a478c6bc5b31243c2a10aaff51f0478b39e44498eb8` | 9.04 GiB |
| 43 | spatial (C) | `0a52ddc1e1255e3360251232199f673c11051ef1083c234966b6fa4e107c19ea` | 5691.5 | 1014.5 | `2a7f855e06fbc2012e713a478c6bc5b31243c2a10aaff51f0478b39e44498eb8` | 9.28 GiB |

- 六个训练都到 `total_updates=2400`；每 seed 每臂每 lead `n_evaluated == n_available_windows` 且等于冻结
  cohort（472/468/460/444/428）；每 seed 每臂的 `training_report` 的 `supervision` 与冻结 `mask_sha256`
  逐字相同（驱动断言）。
- **同 seed 两臂的通路初始化逐位相同**：`pathway_init_sha256` 在两臂相等（41 `18dfa0a2…` / 42 `e5d78ca9…`
  / 43 `2a7f855e…`）；`parent_state_sha256` 随 seed 不同（41 `7c416ccb…` / 42 `857d8804…` / 43 `61fdac58…`），
  `expanded_state_sha256` 因 C 多一个 query 而两臂不同（pooled `e9286dc6…/f6e4f39e…/940a1f2f…`、
  spatial `d97adc0f…/c9c90f61…/2fb364f6…`）——这正是「共享张量逐位一致、只多一个 query」的意思。
- **正控制逐 seed 通过**：`shared_bit_identical=true`、`n_shared_compared=163`；`zero_query_is_pooled=true`
  且 `zero_query_max_abs=0.0`（query=0 时 spatial==pooled）；`zeroed_projection_is_identity=true` 且
  `zeroed_projection_nonzero_spatial=0`、`zeroed_projection_nonzero_pooled=0`（清零投影 ⇒ 证据恒 0）；
  `moved_max_abs_after_random_query`>0（0.43–0.61，随机 query 确实移动输出）；`spatial_extra_params=192`。
- seed 42/43 的部分墙钟较长（训练 8038.4 / 5691.5 s）源于**两臂并发与邻居共驻**——**墙钟不是科学判据**，
  GPU-h 记账见 §6。

## 4. 逐阶段回执

| 阶段 | 文件 | exit_code | status | elapsed_seconds |
| --- | --- | ---: | --- | ---: |
| prepare | `prepare_process.json` | 0 | success | 465.450 |
| archive | `archive_process.json` | 0 | success | 474.422 |
| seed 41（两臂） | `seed41_process.json` | 0 | success | 17908.350 |
| seed 42（两臂） | `seed42_process.json` | 0 | success | 20760.210 |
| seed 43（两臂） | `seed43_process.json` | 0 | success | 15571.386 |
| reading | `reading_process.json` | 0 | success | 415.117 |

六阶段全 `success`、`returncode=0`、无信号；每阶段 spawn 前只读 `gpu_gate`（`nvidia-smi read-only`）；
**未对任何非本实验进程发送任何信号**（邻居全程只共享显存，R-054 / 决策 0026）。

## 5. 公平性守卫

- 两臂三 seed 的逐 lead cohort 计数与已登记对照臂**逐字相同**（472/468/460/444/428）。
- **共享初始化（#79 先例的阻断式防护）**：同 seed 两臂的 `pathway_init_sha256` 相等（§3），
  共享张量 `shared_bit_identical=true`、比较 163 个键 ⇒ B→C 的差异只能归因到位置聚合（在已声明
  的 +192 参数之内），**不是**归因到协议/容量/随机初始化差异。
- **正/负控制**：正控制三条（query=0 == pooled；清零投影 ⇒ 恒零；随机 query 移动输出）逐 seed 通过（§3）；
  负控制＝ B 臂本身（同信息、pooled）。
- reading 阶段按 `boundary_rmse_csv_sha256` 复核已登记 2400 剂量臂的逐 seed 逐 lead 表（
  `scripts/study_r7_s3_typed_spatial.py:426-427`，drift 即 `raise`）；reading 阶段 exit 0 ⇒ 15 张表匹配其 pin，
  A **未重训练**。
- 气候态分母取自 `wide_climatology_reference`（train-only、与对照臂同一分母），`readings.json` 里的
  `protocol_sha256` 与 `protocol.json` 的最终 digest 逐字一致。

## 6. 成本

| 阶段 | whole 秒 | GPU-h | 说明 |
| --- | ---: | ---: | --- |
| prepare（CPU） | 465.450 | 0.1293 | |
| archive（CPU） | 474.422 | 0.1318 | |
| seed 41（两臂 2400 更新 + 五 lead） | 17908.350 | 4.9745 | |
| seed 42（两臂 2400 更新 + 五 lead） | 20760.210 | 5.7667 | 与 41 并发（双卡） |
| seed 43（两臂 2400 更新 + 五 lead） | 15571.386 | 4.3254 | |
| reading（CPU） | 415.117 | 0.1153 | |
| **本轮合计** | 55,594.936 | **15.4430** | 累计 77.8886 → **93.3316** |

- **GPU-h 口径（保守）**：attempt 目录下全部 6 个 `*_process.json` 的 `elapsed_seconds` 之和 ÷ 3600
  = 55,594.935626 s ÷ 3600 = **15.443038 GPU-h**；attempt 目录下**无** attempt 级 wall clock 文件，故直接取
  该和。因 seed 41 与 seed 42 在两张卡上**并发**，逐阶段求和大于真实墙钟——这是**保守（偏大）**口径。
- **soft overrun 0、hard overrun 0**（per-seed 43200 s 内，planned 86400 s 内）。
- 网络 **0**（离线，R-028）。磁盘 ≈14 GiB。
- **峰值口径**：协议 `estimated_peak_bytes` 写 9.5 GiB，实测两臂峰值 9.04 GiB（pooled）与 9.28 GiB（spatial）
  ——本轮估计**略高**（高估约 0.22 GiB），故余量门是保守的；两臂共享同一 `per_seed_seconds`/peak 估计，
  不破坏 B→C 可比性。

## 7. 已确认与推测

**已确认（读到的产物）**：上表全部 exit_code / elapsed / receipt / 逐 lead 评分 CSV / cohort 计数；
六个 checkpoint SHA256；逐 seed 两臂 `pathway_init_sha256` 相等；正控制三条逐 seed 通过；
`readings.json` 的逐 lead rmse / Δ / skill（B→C、vs 剂量臂）；reading 阶段 exit 0 表明 2400 剂量臂的
15 张 `boundary_rmse.csv` 匹配其 pin。

**推测（未证）**：① B→C 无差异**是否**等价于「类型化证据通路在本设置下对 t2m 完全无贡献」——本轮只
证明**位置聚合方式**不改变读数，未分离「通路贡献≈0」与「路由细节不重要」。② 72 h 残余负 skill 的
**主因仍未定位**：候选有自回归误差累积、目标/训练信号组织、数据覆盖，三者本轮**未分离**。③ 若换更大
容量/不同信息，spatial 路由**是否**会有差异——本轮**未测**，不能由本 null 外推。

**未做**：不重训练任何已登记臂（A、窄臂均 pinned 复用）、不读 test、不做显著性检验、不报泛化/SOTA/收敛
因果、不加 seed。

### 7b. 关键诊断：训练后的 query 有多大（决定这个 null 怎么读）

**已确认（直接从三 seed 的 spatial checkpoint 读出）**：

| seed | spatial `\|position_query\|` | `max\|q\|` | pooled `\|projections.0.weight\|` |
| --- | --- | --- | --- |
| 41 | 0.007393 | 0.001812 | 8.037 |
| 42 | 0.012458 | 0.002324 | 8.021 |
| 43 | 0.008398 | 0.002182 | 8.016 |

命令：`torch.load(<spatial checkpoint>)['model']['typed_evidence.position_query'].norm()`（pooled 臂无此键）。

**含义（这是本页最重要的一句）**：query **确实**从 0 训练出了非零值（说明梯度通了、参数在更新），
但它的模长比同层的投影小**三个数量级**。注意力打分是 `(v·q)/sqrt(dim)`，故打分幅度约
`0.01 × O(1) / 13.9 ≈ 7e-4`，softmax 权重相对均匀分布只偏移约 **1e-3 相对量**——也就是说，
spatial 臂只在**池化聚合附近一个很小的邻域**里探索。

**因此这个 null 必须读作「弱操纵下的 null」**：它证明「在本轮学到的这点位置偏移之内，读数不变」，
**不能**证明「保留空间结构对长 lead 无用」。把 B→C 的 null 当成对 H-B 假设（空间结构被均值抹掉才是
#79 负结果的原因）的**强反驳**是**超出证据**的。要真正检验该假设，需要让 query 有足够大的学习幅度
（例如改初始化尺度或缩放打分），而本轮**没有**做这件事。

## 8. limitations

- 单实例、单 ROI、17 通道、2017–2021 train / 2022 val 的开发筛选；两臂三 seed 只是一致性证据，**无显著性**。
- **已声明的差异（必须记住）**：C 的聚合用**一个共享的** `[dim]` query，故两臂除 `typed_evidence.position_query`
  外逐位相同；C 恰多 **192 个参数（+0.13% of the 148800-parameter pathway）**。因此 B→C 的差值只能归因到
  位置加权**在已声明的这 192 个参数之内**——**不**声称参数量逐位相等。
- 外部参考 A（2400 剂量臂）与窄臂都是 **pinned 复用**，不是同协议内同批重训练；协议内对照 B 是本轮
  重训练的，故 **B→C 是 in-protocol**，而 C/B vs A 是跨协议比较。
- 两臂都**与注册父的权重联合训练**，故每臂价格的是「从注册父出发的一条 typed-evidence 路由」，不是
  通路本身的孤立效应。
- `interior_32` 监督使外圈在训练中无约束；只评分 `interior_32`。
- **B→C 的 null 在 float32 数值噪声量级（~1e-6 K）**：本设计能分辨约 1e-6 K 的差异，而它没有出现；
  但同一尺度也意味着本实验**无法**区分「两臂路由完全相同」与「两臂路由差异 < 1e-6 K」。
- **这个 null 是弱操纵下的 null（见 §7b）**：训练后的 `position_query` 模长（0.0074–0.0125）比同层投影
  （≈8.02）小三个数量级，注意力权重相对均匀只偏移约 1e-3，故本轮只探索了池化聚合附近的小邻域。
  **不能**据此说「保留空间结构对 48/72 h 无用」；只能说「在本轮学到的偏移幅度内读数不变」。
  本页与 §8 的出口结论「类型路由本身无效」限定在**本设置、本学习幅度**内。
- 三 seed 是一致性证据，不是显著性检验；GPU 运行共驻，seed 42/43 的部分墙钟因此较长。
- 可复现等级：`protocol_sha256` + `code.zip` SHA + 逐 seed×两臂 receipt + `readings.json` 齐备，
  可按同一协议复现；但 **GPU 训练非逐位可复现**（同机同 seed 也不保证逐位一致），
  故只能复现「同一协议」，不能复现「同一结果」。

## 9. 下一动作

按**决策 0048**（`docs/decisions/0048-capacity-axis-scope-and-cap.md`）与本轮预声明出口：
**关闭 H-B 这一支**——不调注入宽度、不加 seed、不重跑。**结论如实写清**：B→C 无差异 ⇒ 类型路由本身
无效，而非「空间保留无用」。

H-B 是 H-A 关闭后的第二根架构轴。此后架构方向还剩 `docs/plans/0022` 的 **H-C**；训练信号 / 目标组织
方向亦开放（数据按 D3 暂缓、放到最最后）。下一轮应先写新的 round brief，再 `round-guard` → `freeze`
→ 逐 stage → 四件套登记。

test 未读、r=0、S4 未启动不变。
