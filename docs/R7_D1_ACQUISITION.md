# D1 数据获取（#63 收口）：Earthmover spatial 路径，2026-09-26

Status: **DONE（工程段）**。决策 0004 冻结的 Earthmover icechunk `spatial` namespace
读取路径已实现、实测执行并通过发布契约与离线重放验证。
`scientific_claim: false`——D1 是窗口/单位/IO 验证段，#63 明文「不能拿这一步报告
期刊技巧」；本文所有数字是工程事实，不是科学结论。

起点 `cb58890`；本轮新文件：`data/download/earthmover_spatial_d1.py`、
`scripts/verify_r7_d1_store.py`、`configs/r7_era5_d1_local.yaml`、决策 0005。

## 1. 源身份与产物

| 项 | 值 |
| --- | --- |
| 源 | `s3://earthmover-icechunk-era5/icechunkV2`（匿名只读，CC-BY-4.0） |
| 快照 | `ZFKDHBCTBVHVXM3BQFV0`（读取时 pin 校验） |
| 布局 | `spatial` namespace：一时次一张全球场，chunk `(1,721,1440)` / `(1,1,721,1440)` |
| 范围 | 2016-01-01T00:00 .. 2016-01-30T18:00，6 小时节奏，**120/120 时次精确存在** |
| ROI | 65×65 @ 0.25°，27–43N / 107–123E（来自 `read_plan_frozen`，断言不重写） |
| 通道 | 17 通道，顺序 = `DEFAULT_R7_ERA5_CHANNELS`（t2m u10 v10 mslp z850 t850 q850 u850 v850 z500 t500 q500 u500 v500 z250 u250 v250） |
| 单位 | K、m s**-1、m s**-1、Pa、m**2 s**-2（z 保持位势，非位势高度）、K、kg kg**-1、…（逐通道 verbatim 校验，未做任何换算） |
| 层轴 | 源 13 层值集与 audited 集合一致（**降序存储**，`attest_levels` 的升序断言不适用，本路径改为值集相等 + 顺序原样记录）；存储保留 250/500/850 hPa |
| 产物 | `outputs/r7_d1_earthmover/source.nc`，18,825,653 字节 |
| 源 SHA256 | `2ec0d4447b994db144e39a9247654ca9ddfe2e53f6c39169d598a423d3a3020e` |

## 2. 实测成本（回执 `outputs/r7_d1_earthmover/source_receipt.json`）

| 项 | 值 | 预算 | 判定 |
| --- | --- | --- | --- |
| 实测网络字节 | **3,549,905,898 B = 3.31 GiB** | ≤3.5 GiB（goal） | ✓ |
| 墙钟 | **1201.1 s = 20.0 min** | ≤30 min（goal） | ✓ |
| decoded 收费 | 9.04 GiB（120 时次 × 19 chunk + 坐标） | ≤64 GiB（冻结） | ✓ |
| 新产物 | 18.8 MB | ≤16 GiB（冻结） | ✓ |

网络量是 `/proc/net/dev` 非环回接口**实测**差值（与 `R7_D1_READ_SPEED.md` 同法），
不是 decoded 估算。实测 3.31 GiB 略高于测速估计 2.95 GiB（测速只计 17 场/时次，
获取实际读 19 场/时次：多读的 2 场是 q@250/u@250 之外的轴一致性成本——详见
回执 `limitations`），量级与测速一致。decoded 9.04 GiB 与 #63 冻结成本表的
9.77 GiB 估计吻合。

## 3. 发布契约（复用审计路径）

- 获取：`earthmover_pilot` 的 `DecodedBudget`/`bounded_selection` 原样复用
  （预算先检查后扣费、逐 chunk 有限性 + CF 缺失值守卫）；发布守卫
  （`failed-no-fallback` receipt、排他创建、原子落盘 `os.link`、payload/artifact
  SHA256）与 `arco_regional_bounded` 同构。`earthmover_pilot` 的 12×12 temporal
  语义一字未动。
- 转换：`prepare_r7_local.py --write` → `outputs/r7_d1_earthmover/store/`
  （`BUILD_COMPLETE.json` ✓、train-only 归一化、窗口 manifest 94/10/10）。
  D1 是单年连续段，三个非空**年份** split 不存在——发布契约新增
  `split_time_ranges` 时间段切分模式（决策 0005）：train=24 天 / val=3 天 /
  test=3 天，全部在声明的 train 年份 2016 内；store 元数据同时记录年份声明与
  精确时间段。该 val/test 是 2016 段内的工程再划分，**不是** v2 的 2019 验证年
  或 2021 test 候选。
- preflight 先行：`outputs/r7_d1_earthmover/preflight_report.json`（只读，校验
  布局/坐标/单位/层轴 + 一个采样帧，写入零数据文件）。

## 4. 离线重放验证（`scripts/verify_r7_d1_store.py`，全过）

`outputs/r7_d1_earthmover/replay_verification.json`：
source SHA256 重导一致；`BUILD_COMPLETE` ✓；时间轴 = 冻结 D1 请求；
通道顺序 = 公布计划；**17 通道 payload SHA256 逐一与 receipt 一致**；
窗口全部落在声明的区间内、不跨界、逐样本 index/时间戳核对；归一化统计与
「train 区间重算」（fp64）一致（相对差 ≤4.7e-8）。

首轮验证曾报 normalization 失败：是**验证脚本自身**用 fp32 累加均值/方差的
噪声（400k 行 fp32 求和），改 fp64 重算后与 RunningMoments 的存储值一致。
数据与发布路径无问题；修正过程如实记录于此。

## 5. 验证计数

- `tests/test_r7_earthmover_spatial_d1.py`：16 passed（离线 fixture，无网络）；
  冻结 chunk 几何 `(1,721,1440)`/`(1,1,721,1440)` 独立 pin 测试防上游静默重切。
- `tests/test_r7_time_range_splits.py`：7 passed（含「year 模式行为不变」反证）。
- 全量见本轮收尾记录。

## 6. 局限（如实）

- 单一 30 天工程段；不做季节、多年、test 声明。
- 存储 3 层（250/500/850 hPa）而非全部 13 层：17 通道计划只需这些；层选择
  在 receipt 显式声明（decoded 省 24 GiB）。
- 层轴顺序与 ARCO 相反是源的事实，已按「值集相等」核验并原样记录，未重排。
- 测速估计（17.6 min / 2.95 GiB）与实测（20.0 min / 3.31 GiB）的差异来自
  多读 2 场/时次 + 单次运行的负载波动；绝对时间仍应视为量级。
