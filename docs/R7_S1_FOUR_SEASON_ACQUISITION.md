# S1：2017 四季节区域 ERA5 获取与开发 store 发布

**状态：获取/合并/发布已完成并登记（2026-10-05）；dev 实例，`scientific_claim: false`。**
本页对应 `docs/goals/main-model-climatology-campaign.md` 的 D3 首步与决策 0038 的具名扩围；
它记录一次**数据准备**，不产生任何模型超越气候态的结论。

## 1. 结论摘要

- 一个 2017 年四季节区域段（ROI 27–43N/107–123E，0.25°，65×65，17 通道，480 个 6 小时时间点）
  已从钉住的匿名 Earthmover ERA5 icechunk 快照下载、按精确时间戳合并、并发布为带
  `BUILD_COMPLETE` 的本地 store。
- 实测网络 **14,390,323,242 字节（13.40 GiB）**，为冻结计划 `14,504,924,706` 的 99.2%，
  远低于 24 GiB 硬上限；四次 part 共 4,604.4 秒（76.7 分钟），每次 18.6–19.6 分钟。
- store 的切分是**时间范围模式**：train 覆盖四个季节月的各前 24 天，val/test 落在十月后段；
  被评分的十月桶全部被 train 覆盖，所以 fail-closed 的 train-only 气候态可以工作。
- 氪金/GPU 零消耗：本节点 0 GPU-h、0 付费资源。

## 2. 为什么切分与最初设想不同（在漂移前就拒绝）

`outputs/r7_s1_seasons_2017/acquisition_protocol.json` 冻结的意图是
`train=1月 / val=4月 / test=7月`。构建前机械核对发现这是**不可评分的**：

1. train-only 气候态按 `(month, hour)` 分桶且**失败即关闭**（`data/r7_evaluation.py` 的
   `normalized_climatology`：缺桶抛 `missing training climatology bucket`，禁止回退），
   所以 1 月以外的桶不在 train 时 val/test 根本无法评分；
2. 共享的 `parse_split_time_ranges` 要求每个 split 内区间不重叠、且全局
   `train < val < test`，因此无法把"四个季节各取前段作 train、后段作 val/test"交错表达为
   一个合规声明。

采用的合规切分把四个季节整块放进 train 与 val/test 共用的月份：train 为四个月的第 1–24 天，
val 为十月 16–24 日，test 为十月 25 日–11 月 1 日。该决定写在
`store_build_protocol.json`（`split.rationale` 与 `not_the_final_instance`），**在任何训练或
评分之前**冻结；最终跨年公平实例（完整未见年 2022/2023）是另一批独立获取，见 §6。

## 3. 身份链

| 项 | 值 |
| --- | --- |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`（匿名，CC-BY-4.0） |
| 获取协议 | `outputs/r7_s1_seasons_2017/acquisition_protocol.json`，SHA256 `5abdea4b00c05a643e1f5706250acf4d9bedafde530663516ef1a79c7328ddd1` |
| 只读 D1 预检 | 快照/网格/单位/层级核齐；报告 SHA256 `ce1d04cd5eba968de72e663e02cc37c5109770aeeda361505c52d0f7efa19249` |
| 合并源 | `source.nc`，SHA256 `3b2c2dad4c8054ff735930a4539ec554ea420d4f98a1bd91bd43ff0f6729d8b5`（77,177,840 字节） |
| store 构建协议 | `outputs/r7_s1_seasons_2017/store_build_protocol.json`，SHA256 `5f0c4e415e178702ff4d77b7a55afed725edb3a9432478fb3025c6c205ed5dab` |
| 预检报告 | `preflight_store_report.json`：480×17×65×65、0.25°、train/val/test 窗口 340/34/22 |
| 审阅回执 | `store_build_review.json`：执行者自审，`decision=accept to --write` |
| store | `outputs/r7_s1_seasons_2017/store/cache.zarr` + `manifests/`（`BUILD_COMPLETE.json` 在） |
| data identity | `894b8d1b6c08d49f93255558fdacb1290ace690de43757d84369a91b3b28e02c`（train manifest 派生） |
| test 读取 | **否**：本节点从未打开 test JSONL 或对 test 评分 |

## 4. 四次 part 实测

| part | 时间点 | 网络（字节） | MiB/时间点 | 墙钟（秒） | 产物 SHA256（前 16） |
| --- | --- | --- | --- | --- | --- |
| winter | 120 | 3,569,705,460 | 28.37 | 1167.7 | `bd7954af76a7e5ec` |
| spring | 120 | 3,637,453,408 | 28.91 | 1176.2 | `0ddb48c8d509dc03` |
| summer | 120 | 3,592,505,043 | 28.55 | 1112.2 | `175f331a168e9d01` |
| autumn | 120 | 3,343,332,331 | 26.57 | 1148.3 | `0d50413b7e74a7a3` |
| 合计 | 480 | 14,390,323,242 | 28.10 | 4604.4 | — |

速率与 M2 与 S1 winter pilot 一致（28.4–28.9 MiB/时间点），说明冻结的按时间点预算模型可复用。

## 5. 已确认与推测

**已确认**：源身份与快照 pin；17 通道单位（K/Pa/m s⁻¹/kg kg⁻¹/m² s⁻²）；0.25° 网格与
65×65 ROI；480 个 6 小时时间点覆盖四个 UTC 初始化小时；合并时间轴等于各 part 记录时间戳的
精确并集（唯一、递增、无跨月缺口）；t2m 与 geopotential 全域有限；store 的
`normalization_mean` 与按 train 标签重算的均值的最大相对偏差 5.3e-08；
`BUILD_COMPLETE.json` 与 zarr `build_complete=true` 齐备。

**推测**：正式跨年实例上，四季节覆盖会削弱"同月气候态"的强度，从而给出更有意义的
skill 基线 —— 这是本研究下一步要检验的假设，不是本页已证明的事实。

## 6. 局限与下一步

- 单年、单区域；val/test 只是十月内 6–15 天窗口，不是完整季节；train-only 气候态仅覆盖
  四个月份的采样月。
- 该实例不产生任何科学声明；正式确认要求**一个完整真正未见年份**与四季，计划按
  决策 0038 另批获取（train 2017 / val 2022 / test 2023，各含四个季节月），
  新批同样先冻结再 pilot；本页不预写其结果。
- 下一步：S2 单因素机制试验（#77 频带、随后 #78 变化尺度）在**本 dev store**上做有界
  筛选；无论结果如何，最终门只在正式未见实例上判定。

## 7. 证据指针

- 获取协议：`outputs/r7_s1_seasons_2017/acquisition_protocol.json`
- D1 只读预检：`outputs/r7_s1_seasons_2017/preflight_d1_namespace.json`、`preflight_review.json`
- part 回执：`outputs/r7_s1_seasons_2017/parts/part_*_receipt.json`（各含 payload SHA256 与预算读数）
- 合并回执：`outputs/r7_s1_seasons_2017/merge_receipt.json`
- store 构建：`store_build_protocol.json`、`preflight_store_report.json`、`store_build_review.json`
- 工具：`data/download/earthmover_spatial_s1.py`、`data/download/season_plan_s1.py`
