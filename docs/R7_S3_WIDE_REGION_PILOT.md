# S3 宽区（域外大尺度上下文）采集：pilot 实测成本与中心块身份

**状态：已完成（2026-10-09）；`scientific_claim: false`；2023 test 未读、r=0。**
本页对应 `docs/goals/s3-wide-region.md` 与 `docs/goals/main-model-climatology-campaign.md` §8 的
S3 接续。它是**数据制度可行性**证据，不产生任何 skill/ACC/气候态读数，也不表示区域假设已被检验。

## 1. 结论摘要

- 48/72h 的墙是 pattern 相关本身：t2m/full ACC 0.367/0.189，线性重标定上限 ACC² 仅
  +0.1355/+0.0366，即便完全校准也过不了绝对气候态门（`docs/R7_S3_ROLLOUT_DOSE.md` §4/§10）。
  10 m/s 的系统 72h 走 ~2600 km（23°），远大于冻结 16° 目标框——"域内可用信息不足"因此是
  **可反驳**假设，本轮的 pilot 就是为它的实验准备数据制度。
- 只改一个因素：读取区域 27–43N/107–123E 65×65 → **19–51N/99–131E 129×129**（各方向 +8°，
  面积 3.94×）。源/快照/17 通道/0.25°/6h 节奏/2017–2021+2022 split/模型 spec/归档初始化全部不变。
- **中心块 = 冻结目标框**：129 网格的中心 65×65 块按断言就是冻结框，因此注册气候态与 val cohort
  仍然适用；评分可用 `boundary_masks(129,129,(32,))` 的 `interior_32` 掩码精确取得，与旧 65×65
  读数逐格可比。这一点由测试（掩码与中心块逐元素相等）与实跑（与 v3 `source.nc` 逐位相同）双向确认。
- **ROI 放大在成本上几乎免费**（pilot 实测，对照注册的同季目标框 part）：

  | 量 | 宽区 129×129 | 目标框 65×65（注册） | 比 |
  | --- | ---: | ---: | ---: |
  | 网络字节 | 3,565,096,475 | 3,554,716,077 | **1.0029** |
  | 解码 charged 字节 | 9,482,837,880 | 9,482,837,880 | **1.0000** |
  | chunk reads | 2287 | 2287 | **1.0000** |
  | 墙钟（秒） | 1223.141 | 1267.312 | 0.965 |
  | 存储 NetCDF 字节 | 70,207,133 | 18,977,715 | 3.699（面积比 3.939） |

  原因是源布局为"每时刻一张全球场、一个 chunk"，读任意 ROI 都要解码同一个全局 chunk；
  放大的代价只落在**存储**上（~70 MB/part），而存储不是瓶颈（`/data` 余 516 GB）。
- 冻结判据 **18/18 通过**；中心块身份核验 **pass**（9 个源变量、17 个堆叠通道、经纬坐标、
  全部单位，120 stamp 逐位相同）。
- 本轮 GPU-h **0.0000**（纯 CPU/网络）；未训练、未评分、未读 test。

## 2. 身份与协议

| 项 | 值 |
| --- | --- |
| 执行代码 | `8ace116a9187c8c88ed15f1110de5d6ab4848a3e`（工作分支 `r7/weather-reasoning`） |
| 读计划 | `data/download/read_plan_wide.py`，format `r7-era5-wide-read-plan-v1`，digest `d5618cbf9709a43883666d65fd6729f9e048f1103bee4415e33a777b5fe035a5` |
| 下载器 | `data/download/earthmover_wide_io.py`、`data/download/earthmover_spatial_w1.py` |
| pilot 协议 | `outputs/r7_s3_wide_region_pilot_20261009_attempt01/acquisition_protocol.json`，`protocol_sha256 bfdcfa892a551f33c4d8530cbc02d58a43dd1aa7f7a9c25ff186bdfce1419526`（下载前写入，`'x'` 排他） |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`，快照 `ZFKDHBCTBVHVXM3BQFV0`，匿名，CC-BY-4.0 |
| 范围 | 2018 冬 120 stamp，`2018-01-01T00:00Z`–`2018-01-30T18:00Z`，四个 UTC 起报时次各 30 |
| 形状 | `[120, 3, 129, 129]`，存储层轴 `[250, 500, 850]` hPa |
| 控制 | 注册目标框同季 part `outputs/r7_s3_batch3_20182021/parts/part_winter_2018_receipt.json`，SHA `070c356b46951430d8801fcb2e3426cd1d7fc33402aa98d041e8ccb19d3eb0b1` |
| 产物 | `part_winter_2018.nc` 70,207,133 B，SHA `b05f216353c2d71abfd649d46b603522fba33851565aa06bafd01775881b12fc` |
| 预算 | per-part soft 1800 s、watchdog TERM 1860 s、`--max-decoded-gib 16`、planned network 4.0e9 B、disk peak 6.0e8 B、margin 2.0e10 B |
| test | 未读（未出现 test manifest；2023 仍封存） |

## 3. 区域设计与"中心块 = 冻结目标框"

- 目标框从 `read_plan_frozen` 读取并**断言**，不重写：27/43/107/123，65 点，0.25°。
- 宽区 = 目标框四边各外扩 32 格（8°）：`19 = 27 − 8`、`51 = 43 + 8`、`99 = 107 − 8`、`131 = 123 + 8`，
  129 点。`containment_evidence()` 逐边断言这些恒等式，`validate_wide_namespace` 再对**实际选中的
  坐标数组**断言其中间块等于冻结框，所以"纸上相等而选错格"无法通过。
- 与评估侧的一致性：`boundary_masks(129,129,(32,))` 的 `interior_32` 掩码 = 行/列 32..96，
  正好 65×65；测试断言该掩码与 `central_block()` 逐元素相等，且该掩码下的纬度序列与 65 网格
  的纬度序列逐值相等。**这就是"同一目标框、同一气候态、可直接比较"的实现依据。**
- 冻结路径不动：`read_plan_frozen` 的 `frozen_protocol()` digest、`earthmover_spatial_d1` 的
  65×65 断言与 ROI 字面量、`earthmover_spatial_s1` 全部保持；测试直接读源文件断言这些字面量仍在。

## 4. 冻结判据与中心块身份核验

- `scripts/check_r7_wide_pilot_receipt.py`（只读）逐条对照 `success_criteria`：
  **18/18 通过，0 偏离**。含状态、无合成回退、读计划 digest、快照、stamp 数与首末时刻、
  四时次覆盖、形状、层轴、通道数、通道清单、每通道 payload 存在且 finite、payload digest 存在、
  网络字节 ≤ 1.2× 控制、解码字节 ≤ 1.01× 控制、墙钟 ≤ 1800 s、控制回执文件 digest 相符。
  缺字段算失败（有反证测试），不算跳过。
- `scripts/verify_r7_wide_target_block.py`（只读）：把 pilot NetCDF 的中心 65×65 块与注册 v3
  `source.nc` 在同 120 stamp 上逐通道比对，**verdict = pass**：
  `latitude`/`longitude` 逐位相同、9 个源变量全部 `bit_identical` 且 `max_abs_diff = 0.0`、
  17 个堆叠通道 `bit_identical = True`、单位全部相符。
- 两个工具都带"故意违规必须报错"的反证：错状态、合成回退、读计划漂移、快照漂移、stamp 数少、
  时次覆盖错、形状退回 65×65、层轴漂移、通道数错、缺通道、通道非 finite、digest 缺失、
  网络/解码超预算、墙钟超时、控制回执被替换、中心块单格被扰动、中心块外扰动不得误报、单位被改、
  stamp 对齐错位。三个新测试文件合计 **45** 个用例（`test_r7_wide_region_acquisition.py` 17、`test_r7_wide_target_block_check.py` 8、`test_r7_wide_pilot_receipt_check.py` 20）。独立复核另用变异反证：把 `TARGET_MARGIN_CELLS` 改成 30 后 5 个几何/协议用例失败、协议 digest 由 `d5618cbf…` 变为 `7c4137a6…`，证明断言确实承重。

## 5. 工程缺陷与修正（诚实记录）

1. **读计数估计错误（下载前发现并修正）**：`read_plan_wide` 最初按"17 通道"估每 stamp 17 次字段读；
   只读 preflight 实测 `per_stamp_field_reads = 19`（4 个地面变量 + 5 个气压变量 × 3 层），
   与注册目标框 part 的 2287 reads / 120 stamp 相符。协议在**任何下载发生之前**把该数字改为 19、
   重算读计划与协议 digest，并把前值与原因写进 `revisions`（前 digest `207eb544…` → `d5618cbf…`，
   前解码估计 8,472,038,400 → 9,468,748,800）。没有字节被计费。
2. **回执判据检查器的两处缺陷**（运行中暴露，已修并留反证）：控制回执的 `decoded` 字段实际嵌在
   `decoded_chunk_budget.charged_bytes`；"控制回执就是协议点名的那个"最初错比对回执内部字段，
   改为比对文件 digest。首次带缺陷的输出保留为
   `criteria_check_attempt01_checker_defect.json`（1 项误报 deviation），不删。
3. **中心块核验工具的索引缺陷**：`wide_state[positions]` 用参考的时间下标去索引宽区数组，
   在参考比宽区长时越界；改为按时间对齐取参考侧。已补两个回归用例（参考更长、错位一格）。
4. 三处规模计数（R-020 函数体、R-023 参数）一度各涨，通过拆分 `validate_wide_namespace`、
   `extract_wide_part`、`compare_central_block`、`check` 与参数打包回到基线，**未放宽任何阈值、
   未新增例外**；`docs/rules/size-thresholds.md` 的 measured marker 不需要改动。

## 6. 成本

| 阶段 | 量 | 说明 |
| --- | ---: | --- |
| 只读 preflight（含 1 stamp） | ~1 分钟，网络少量 | 读元数据 + 1 个 stamp，0 GPU |
| pilot 实跑（120 stamp） | 1223.141 s | soft 1800 / watchdog 1860 未触；exit 0 |
| 判据核验 + 中心块核验（CPU 只读） | 数分钟 | 0 网络、0 GPU |
| **GPU-h** | **0.0000** | 采集与核验全为 CPU/网络；下载路径自身不引用 torch/CUDA，但 `read_plan_frozen` 经 `training.r7_experiment.canonical_digest` 传递 import torch，因此"0 GPU-h"的依据是**未打开 CUDA 设备**（`torch.cuda.is_initialized()` 仍为 False），不是"未 import torch" |
| 网络 | 3,565,096,475 B | 与目标框 part 比 1.0029 |
| 磁盘（保留） | 70,207,133 B | 已计入新排他 `outputs/` 路径；无删除、无复活 |

累计 GPU-h 仍为 **27.5905**（cap20/remaining −7.5905 仅会计字段）；本轮不加 GPU 账。

## 7. 已确认与推测

**已确认**：宽区 129×129 选点与"中心块 = 冻结目标框"的恒等式与实际选点断言；单位/层轴/形状/
通道清单与冻结 17 通道一致；冻结判据 18/18；中心块与注册 v3 `source.nc` 逐位相同（120 stamp、
9 变量、17 通道、坐标、单位）；实测网络/解码/读次/墙钟/存储数字与对照比；下载前修正与三处工具缺陷
如实保留；0 GPU-h、0 test 读取。

**推测/未做**：不宣称"区域放大一定提高 skill"——本轮只证明**能便宜地取到**更宽的区域且目标框身份不变；
不宣称每 stamp 网络成本恒等是一条成本模型（只有 1 个 part 的 1 次观测）；未做全量批次、未建 v4 store、
未重建气候态/incumbent、未训练、未评分；48/72h 的墙是否由域内信息不足造成仍**未被检验**。

## 8. limitations 与下一动作

- 单 part、单年、单季：是成本与身份 pilot，不是四季或气候样本；`scientific_claim: false`。
- 8° 半宽只是对平流问题的**部分**回答（72h 需 ~23°），协议 §limitations 已写明；若要更强的检验，
  需另议更大的半宽与其算力代价。
- 本轮 0 GPU-h，因此不改变任何模型/机制结论；账本只加网络与磁盘成本。
- 下一动作（S3）：按实测成本冻结全量批次协议（train 2017–2021 + val 2022 = 24 part，
  预计网络 ~85.6 GB、墙钟 ~8.2 h 串行（24 × 1223.141 s）或按网络/磁盘预算并行分片），随后用 `prepare_r7_local.py`
  建 v4 store 与 sidecar，重建同数据气候态与 incumbent，再以 `interior_32` 掩码与旧 65×65
  读数逐格比较。全量协议必须另立并冻结，不沿用本 pilot 的数字。test 未读、r=0、S4 未启动不变。
