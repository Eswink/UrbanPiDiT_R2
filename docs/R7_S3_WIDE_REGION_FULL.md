# S3 宽区域全量获取、v4 store 与宽区 train-only 气候态

**状态：进行中（2026-10-09）；`scientific_claim: false`。**
本页对应 `docs/goals/s3-wide-region-full.md`（本轮轮次目标）与主计划
`docs/goals/main-model-climatology-campaign.md` §8。它记录一次**数据获取与发布**以及一份
**参数自由基线**，不产生任何模型结论、不评分、不读 test。

## 1 目的与单因素

S3 rollout-dose 轮把 48/72h 的墙标定为 **pattern 相关**：t2m/full ACC 0.367/0.189，线性重标定
上限 ACC² 仅 +0.1355/+0.0366，任何剂量都过不了 train-only 气候态门。10 m/s 的系统 72h 走
~2600 km（23°），大于冻结的 16° 目标框，因此"域内可用信息不足"是可反驳假设。

本轮只改一个因素：读取区域 65×65（27–43N/107–123E）→ 129×129（19–51N/99–131E，各方向 +8°）。
宽网格的中心 65×65 块按断言**就是**冻结目标框（cell for cell），因此冻结气候态与 val cohort
不变，评分可用 `boundary_masks(129,129,(32,))` 的 `interior_32` 掩码精确取得。源/快照/17 通道/
0.25°/6h 节奏/split/模型 spec/归档初始化全部不变。

## 2 冻结协议与代码身份

| 项 | 值 |
| --- | --- |
| 获取协议 | `outputs/r7_s3_wide_region_full_20261009_attempt02/acquisition_protocol.json`，canonical `protocol_sha256 be50afbc100091d55aee7117baaf623927bfded8a27fed4246da29fb1a1cf8c5` |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0，DOI `10.24381/cds.adbb2d47`） |
| 区域 | 19–51N/99–131E、0.25°、129×129；目标框 27–43N/107–123E、65×65 |
| 计划 | 28 part × 120 stamps = 3360 stamps（2017–2023 × 四季，每年一个 30 天窗口） |
| 预算（冻结） | 每 part deadline 1800 s、watchdog `timeout -k 60 -s TERM 1860`、`--max-decoded-gib 16`、planned 网络 100,000,000,000 B、网络硬上限 150 GiB、planned 14400 s、hard 64800 s、磁盘峰值 12 GiB、余量门 50 GiB、cleanup reserve 180 s、**pool 4** |
| 恢复规则（冻结） | **不实现续传**；失败 part 只以**新 artifact 名**重试，绝不删除部分文件 |
| 只读 preflight | `preflight_train.json`（SHA256 `2ec28e50…`）、`preflight_valtest.json`（SHA256 `aa1d98d3…`）；执行者自审 13/13，`preflight_review.json` |
| 复用 | pilot 的 winter_2018 part（`b05f2163…`，read-plan digest `d5618cbf…`）核 hash 后复用，不重取 |

## 3 下载器缺陷：R-021 拆分留下的四处不可运行路径

committed 的宽区 `--preflight` 与 `--write` **都不可运行**——pilot 是在 R-021 拆分**之前**跑的，
拆分把 helper 在模块间搬动时留下了断引用，而当时只有"名字同一性"测试，没有作用域检查。
四处缺陷（全部在 attempt02 之前发现并修复）：

1. `earthmover_wide_io.validate_wide_namespace` 引用未定义的 `surface_chunk_bytes`（`--preflight` 立即 `NameError`）。
2. `earthmover_spatial_w1` 未 import 五个已搬到 `earthmover_spatial_d1` 的 helper
   （`_open_pinned_session`/`_net_recv_bytes`/`_collect_frames`/`_channel_payloads`/`_write_netcdf`）。
3. `_wide_part_receipt` 引用其调用方的局部变量 `started`——**下载完成后写回执时崩溃**，
   于是留下一个完整 nc 却没有回执（attempt01 的 `part_winter_2017_r2.nc` 即此）。
4. `_wide_part_receipt` 对已是字符串的版本号调用 `.__version__`。

修复与守卫：

- 修 1：新增模块级 `per_stamp_field_bytes(read_count)`（= 19 × 4,152,960 B）。
- 修 2：补全 import。
- 修 3/4：`_wide_part_receipt` 改用 `measured` 里的 `elapsed` 与版本字符串。
- 守卫 A `test_wide_modules_have_no_undefined_global_names`：对三个模块做作用域走查，
  任何"既非 import、也非定义、也非局部"的全局名即失败。**反证**：对拆分前的 HEAD 版本，
  它报出全部七处未定义名（`extract_wide_part` 五处、`merge_wide_parts` 两处、`_wide_part_receipt` 一处）。
- 守卫 B `test_wide_part_receipt_composes_from_its_own_arguments`：只用 `artifact`/`measured`
  合成回执；它当场抓出缺陷 4（名字走查看不见的类型错误）。
- 修 1 另有 `test_wide_stamp_field_bytes_matches_the_frozen_cost_model`（19 ≠ 17 通道数）。

attempt01（`outputs/r7_s3_wide_region_full_20261009_attempt01/`）作为失败记录保留：2 个
`failed-no-fallback` 回执、1 个无回执的孤儿 nc、被取代的协议与批日志；不删、不追认。

## 4 并发实测与 pool 决定

单 part 实测 1218.5 s（POOL=1）。实测不同并发下的聚合速率：

| 并发 | 聚合 recv | 每 part 隐含 | 结果 |
| ---: | ---: | ---: | --- |
| 1 | ~2.9 MB/s | 2.9 MB/s | 正常（validation part 1218.5 s） |
| 2 | ~4.5 MB/s | ~2.2 MB/s | 正常 |
| 4 | ~8.2 MB/s | ~2.0 MB/s | 正常（每 part ~1660–1752 s，均在 1800 s 内） |
| 6 | **~0** | — | **死锁**：CPU 时间冻结、socket 空闲、无字节流动 |

因此 pool 固定为 **4**（冻结值），并把该实测写进协议的 `concurrency_note`。

## 5 逐 part 实测

28 个 (year, season) part 全部 `downloaded-real-source`（27 新取 + 1 复用 pilot）。
每 part 恰 120 stamps、四 UTC init 小时各 30、shape `[120,3,129,129]`、level 轴 `[250,500,850]`、
17 通道全有限、`read_plan_protocol_sha256 d5618cbf…` 与 pilot 相同。

| # | part | status | 墙钟(s) | net(B,宿主级) | charged(B) | reads | sha256 | note |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | winter_2017 | `downloaded-real-source` | 1218.5 | 3,570,436,631 | 9,482,837,880 | 2287 | 98b228c1c7d1 | |
| 2 | spring_2017 | `downloaded-real-source` | 1657.9 | 13,971,494,673 | 9,482,837,880 | 2287 | 4ab68d3d9ace | |
| 3 | summer_2017 | `downloaded-real-source` | 1752.2 | 14,327,509,807 | 9,482,837,880 | 2287 | 2e7ea056773a | |
| 4 | autumn_2017 | `downloaded-real-source` | 1680.5 | 14,110,641,890 | 9,482,837,880 | 2287 | 8e76c42170ad | |
| 5 | winter_2018 | `downloaded-real-source` | 1223.1 | 3,565,096,475 | 9,482,837,880 | 2287 | b05f216353c2 | 复用 pilot |
| 6 | spring_2018 | `downloaded-real-source` | 1709.7 | 14,226,352,678 | 9,482,837,880 | 2287 | 24360ff53717 | |
| 7 | summer_2018 | `downloaded-real-source` | 1723.4 | 14,092,301,750 | 9,482,837,880 | 2287 | c0079ba83a8f | |
| 8 | autumn_2018 | `downloaded-real-source` | 1754.6 | 14,225,362,033 | 9,482,837,880 | 2287 | 236fe32e1070 | |
| 9 | winter_2019 | `downloaded-real-source` | 1684.1 | 13,817,993,245 | 9,482,837,880 | 2287 | a3336ed756ea | |
| 10 | spring_2019 | `downloaded-real-source` | 1744.1 | 14,196,212,942 | 9,482,837,880 | 2287 | 00c609dbdbc9 | |
| 11 | summer_2019 | `downloaded-real-source` | 1643.3 | 12,774,296,493 | 9,482,837,880 | 2287 | 240c13300af5 | |
| 12 | autumn_2019 | `downloaded-real-source` | 1596.7 | 12,564,535,810 | 9,482,837,880 | 2287 | ca4df031735e | |
| 13 | winter_2020 | `downloaded-real-source` | 1414.1 | 6,741,862,093 | 9,482,837,880 | 2287 | 2aae32c091bf | 首跑失败，`_r2` 重取 |
| 14 | spring_2020 | `downloaded-real-source` | 1764.1 | 13,134,195,078 | 9,482,837,880 | 2287 | 73c6fe4d1939 | |
| 15 | summer_2020 | `downloaded-real-source` | 1782.9 | 14,203,965,524 | 9,482,837,880 | 2287 | f0fa316f23cf | |
| 16 | autumn_2020 | `downloaded-real-source` | 1695.2 | 13,865,106,471 | 9,482,837,880 | 2287 | 60fb1df94b2f | |
| 17 | winter_2021 | `downloaded-real-source` | 1710.4 | 13,958,689,245 | 9,482,837,880 | 2287 | b5af11870b0d | |
| 18 | spring_2021 | `downloaded-real-source` | 1748.3 | 14,118,147,965 | 9,482,837,880 | 2287 | 5cef75fbd014 | |
| 19 | summer_2021 | `downloaded-real-source` | 1743.0 | 13,999,113,997 | 9,482,837,880 | 2287 | 75a235bbe32b | |
| 20 | autumn_2021 | `downloaded-real-source` | 1650.4 | 13,456,725,944 | 9,482,837,880 | 2287 | 68f2582de8dd | |
| 21 | winter_2022 | `downloaded-real-source` | 1521.9 | 7,090,291,804 | 9,482,837,880 | 2287 | 7e5f63245367 | 首跑失败，`_r2` 重取 |
| 22 | spring_2022 | `downloaded-real-source` | 1727.4 | 13,929,799,417 | 9,482,837,880 | 2287 | 0a134e986bd0 | |
| 23 | summer_2022 | `downloaded-real-source` | 1708.7 | 14,023,116,891 | 9,482,837,880 | 2287 | 8e31ee92480a | |
| 24 | autumn_2022 | `downloaded-real-source` | 1658.2 | 13,757,908,212 | 9,482,837,880 | 2287 | d4f218f55c6e | |
| 25 | winter_2023 | `downloaded-real-source` | 1704.1 | 13,994,506,621 | 9,482,837,880 | 2287 | 5b1511beb315 | |
| 26 | spring_2023 | `downloaded-real-source` | 1789.5 | 14,266,248,072 | 9,482,837,880 | 2287 | 5c0fa6e2bc8f | |
| 27 | summer_2023 | `downloaded-real-source` | 1431.6 | 6,957,113,861 | 9,482,837,880 | 2287 | 2621efc69f95 | |
| 28 | autumn_2023 | `downloaded-real-source` | 1498.3 | 7,117,589,681 | 9,482,837,880 | 2287 | 5bd50b76b1b5 | |

**两次真实失败（保留，不追认）**：`winter_2020`（14:25:35→14:55:41，1806 s）与
`winter_2022`（15:25:30→15:55:36，1806 s）都在**冻结的 1800 s 每 part 期限**上被
`RuntimeError: extraction deadline exceeded` 拒绝，**均未写出任何产物**（无部分文件）。
按冻结恢复规则以**新 artifact 名** `_r2` 重取，两次均在 1414.1/1521.9 s 成功。首跑失败回执
`part_winter_2020_receipt.json`、`part_winter_2022_receipt.json` 原样保留。

**墙钟与并发的相关性**（可解释观测）：POOL=4 的 part 落在 1596–1789 s；最终 2 part 的一波
（summer_2023/autumn_2023）与两次 `_r2` 重取（POOL=2）落在 1414–1522 s；POOL=1 的
validation part 1218.5 s。即每 part 墙钟随并发升高而上升，冻结的 1800 s 期限在 POOL=4 下已接近
——这正是两次失败的直接原因，也是「下一批若要提高并发须同时上调 per-part 期限」的依据。

## 6 合并与中心块身份核验

- 合并 28 part → `outputs/r7_s3_wide_region_full_20261009_attempt02/source.nc`
  （`2,003,552,683` 字节，SHA256 `6bc9a1a2d93d345ce0dcb39d4f3a26f175ea3bcc15082ac4931a179ca13b7701`），
  `merge_receipt.json` 记录 28 part、3360 stamps、`2017-01-01T00:00` … `2023-10-30T18:00`。
- 合并断言：时间轴 = 28 part 记录 stamp 集合的精确并集，唯一、递增、覆盖四个 UTC init 小时；
  网格必须是 129×129。
- **与 v3 的时间轴逐位相同**（3360/3360，`np.array_equal` 为 True），变量集合相同。
- **中心块身份**（`scripts/verify_r7_wide_target_block.py`，报告 `target_block_check.json`）：
  `verdict: pass`，中心块 rows/cols `[32,97]`、points `[65,65]`，`stacked_channels` 与 v3 `source.nc`
  **逐位相同**（`max_abs_diff 0.0`），`units_match: True`，参考 SHA `bc2ff9cf…`（v3 源）。
- **store 级身份**（更强）：宽 store `state` 的中心 `[:,:,32:97,32:97]` 与 v3 store `state`
  **逐位相同**（`max_abs_diff 0.0`），中心 lat/lon 亦逐位相同；两者归一化不同（宽实例在更宽场上
  重新拟合，符合预期）。

## 7 v4 wide store

- 冻结协议 `outputs/r7_s3_wide_instance_v4_20261009_attempt01/store_build_protocol.json`
  （canonical `03f2c647766ab848b855e83724b2aaca830ecc502ac838f37c174838fc4fcd78`），
  只读 preflight（`preflight_store_report.json`，SHA `51dd90cc…`）与执行者自审
  （`store_build_review.json`，13 项）通过后才 `--write`。
- 发布：`store/cache.zarr`（zarr v3）+ `store/manifests/`；`BUILD_COMPLETE.json` =
  `{"schema_version":1,"build_complete":true}`。
- `zarr_metadata.json`：shape `[3360,17,129,129]`、chunks `[16,17,64,64]`、17 通道、
  `split_years {train 2017-2021, val 2022, test 2023}`、`samples_by_split {train 2360, val 472, test 472}`
  ——**与 v3 完全相同**；`native_grid_spacing_deg 0.25`；`normalization_years` = 五个 train 年
  （train-only）；`process_diagnostics_enabled: true`。
- `write_report.json`：`mode written-local-cache`、shape `[3360,17,129,129]`、`raw_state_GiB 3.541`、
  源指纹 `full-local-file 6bc9a1a2…`。
- 配置：复用**区域无关**的 `configs/r7_s3_confirmation_v3.yaml`（只含 split_years、history/lead 节奏、
  time_chunk、compute_process_targets；源由 CLI 给出），因此 split 合同与 v3 完全一致。
- **宽 store data identity**：`2360d42b7393cd6f53cd6f48961abc7bcd62948d3c2fa69b2d810d798d137c12`
  （与 v3 的 `2564eeaf…` 不同——区域不同，符合预期）。
- 构建耗时 168.8 s（CPU、离线、0 GPU）。test manifest 已生成但**本轮从不打开**。

## 8 宽区 train-only 气候态与 persistence

- 驱动 `scripts/study_r7_s3_wide_d2_baselines.py`，冻结协议 canonical
  `591ede1d65962399f5dd1ffbd40de564848d4061cee7925d86f475c513939f32`；输出
  `outputs/r7_s3_wide_d2_baselines_20261009_attempt01/`。CPU、离线（`deny_network`）、0 GPU。
- 10 次 `evaluate_local`（五 lead × {climatology, persistence}），val 2022 全 cohort
  （n_evaluated == n_available_windows），`boundary_margins=(32,)`。气候态由**宽 store 的 train
  split（2017–2021）**拟合、fail-closed；每次调用都断言 `training_years == [2017..2021]` 且各次
  拟合到同一气候态。
- 分区：`full` 16641 点、`interior_32` **4225 点 = 65×65**（面积占比 0.2564）、`edge_32` 12416 点。
- 整轮 **2461.6 s**（planned 5400、soft overrun 0、hard 14400 未触）。

**t2m 物理 RMSE（K）**：

| lead | clim full(129) | clim interior_32 | pers full(129) | pers interior_32 | pers 的 interior_32 skill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6h | 3.2142 | 3.5171 | 4.5282 | 4.5470 | −0.6714 |
| 12h | 3.1964 | 3.4889 | 5.8832 | 5.9235 | −1.8825 |
| 24h | 3.1697 | 3.4375 | 2.9410 | 3.1633 | +0.1531 |
| 48h | 3.1241 | 3.3534 | 3.8191 | 4.1795 | −0.5533 |
| 72h | 3.1115 | 3.3184 | 4.2297 | 4.7269 | −1.0291 |

**关键交叉核对（本轮最强的身份结论）**：宽实例 `interior_32` 的 t2m 气候态 RMSE 与**已登记的
v3-D2**（65×65 实例）逐 lead 在 **~1e-10** 内相同（3.517116 / 3.488909 / 3.437473 / 3.353443 /
3.318361），persistence 同样（4.547019 / 5.923480 / 3.163328 / 4.179511 / 4.726945）。即宽实例
**可评分内域的分母与已登记窄实例完全相同**——下一轮单因素比较的 skill 分母在宽/窄两臂间**逐位一致**，
不需要为宽臂重建或重新拟合一个不同的分母。

`full(129)` 列与 v3 不同（域更宽），这是预期的：它只在宽臂内部可比，不用于跨实例比较。
test manifest 已生成但从未打开；本页不产生任何模型 skill 声明。

## 9 成本、复现等级与明确未做

- **GPU-h：0.0000**（全轮 CPU/网络：下载、合并、store、气候态）。
- **墙钟**：validation part 1218.5 s；26-part 批次 12223 s（13:26:54→16:50:37，POOL=4）；
  两次 `_r2` 重取各 1414.1/1521.9 s（POOL=2，并行）；合并与核验为秒级；store 构建 168.8 s；
  宽 D2 基线 2461.6 s。整轮 ≈ 4.5 h 墙钟。
- **网络**：`network_body_bytes` 是宿主级计数器（见 §10 的偏差说明），逐 part 在并发下被放大。
  可直接测量的量：waves 2–7（22 part）宿主级增量 **76.66 GB**；两次重取 **7.10 GB**；
  validation part（POOL=1，无兄弟流量）**3.570 GB**；wave 1（4 part）由最早/最晚完成 part 的窗口
  界在 **13.97–14.33 GB**。据此**全量新增流量 ≈ 94–95 GB**（27 个新 part），另加两次失败尝试各约
  3.5 GB（≈7 GB），合计 **≈101 GB**。冻结计划 100,000,000,000 B、硬上限 150 GiB：**软超约 1%，
  硬上限未触**。`merge_receipt.json` 里的 `network_body_bytes_total 340,056,615,303 B` 是各 part
  宿主级读数的**朴素求和**，在 POOL=4 下约 4× 高估，**不作为成本**。
- **磁盘**：source.nc 1.9 GB、parts 1.9 GB、store 2.8 GB、D2 输出 3.5 MB；attempt01 失败记录 68 MB。
  峰值远低于冻结的 12 GiB 与 50 GiB 余量门（实测余量 504 GB）。
- **可复现等级：config-reproducible**——协议/源 SHA/代码身份/合并回执/中心块与 store 身份齐备；
  下载与 store 构建为确定性 CPU 路径（store 构建 168.8 s 可重放）；不声称 bit-reproducible
  （网络侧速率与并发相关，且 GPU 不涉及）。
- **明确未做**：未训练、未评分任何模型、未读 test、未改 `data/raw|interim|processed` 或任何归档、
  未改 `read_plan_frozen`/D1/S1 下载器字节、未删 attempt01 失败件、未付费/租卡/独占/main 合并/发版/
  关 issue、未自建 cron/守护。

## 10 明确未做与 limitations

- 未训练、未评分、未读 test（2023 采集但从不打开评分）、未改 `data/raw|interim|processed`、
  未改 `read_plan_frozen`/D1/S1 下载器字节、未付费/租卡/独占/main 合并/发版/关 issue。
- **已知测量偏差**：`network_body_bytes` 是**宿主级** `/proc/net/dev` 计数器在**该 part 读取窗口**
  内的增量，不是该 part 的自身流量。并发 4 时每个进程的窗口覆盖整波，故每 part 读数约被放大 4×
  （3.57 GB → ~14.0–14.3 GB）。协议判据"每 part 网络在 pilot 的 25% 内"因此**在并发下不可按字面
  评估**，按 `stop_rules` 记为该判据的**偏差**（仪器限制，非数据缺陷），并另报批量总量。
- 解码 charged bytes（9,482,837,880）与 chunk reads（2287）是**进程内**计数器，逐 part 与 pilot
  **完全相同**，这两条判据正常通过。
