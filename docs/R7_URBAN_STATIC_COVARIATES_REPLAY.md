# R7 城市静态协变量：决策 0070 编排重跑的复现读数（attempt03）

> **本页是 `docs/R7_URBAN_STATIC_COVARIATES.md`（attempt01，已登记、已被账本行
> `s3-urban-static-covariates-acquisition` 的 `evidence_sha256` 钉住）之外的新页面。**
> attempt01 与 `attempt02`（超时失败）的字节未动。本页不声称与 attempt01 逐位相同。
>
> **`scientific_claim: false`**；**`test_read: false`**；**0 GPU-h**；r 仍为 0。
> 本页是数据准备读数与工程判定，不是天气真值或机制结论。

- **日期**：2026-10-31（UTC；逐阶段时刻见 §0 时间线）
- **产物**：`outputs/r7_urban_static_covariates_attempt03/`
- **`static_identity`**：`6f65f88a780a62153bdd7d24211e0367ed2385dc102f1471167c665ceade74d9`
- **`protocol_sha256`**：`f42305d0cf5ac01d2e96dbe03d419bcb93715ab6dda6395d8d519bd74a05350c`
- **`fields_sha256` / `output_sha256`**：`ff27e5d45d86a57445c97f80986da7c39921f745968c76c46b649c3c13f24725`
- **`source_sha256`**：`86f813d04dce0e52a4d853304ff9c419b8176df4ea81402f89a5a07454030768`
- **不做**：不改 `model/**`、不重建 store、不读 2023 test、不改 attempt01/02 的任何字节。

## 0. 一句话结论（逐条对应预声明出口，不合并成"通过"）

六个阶段都**实跑并产出**（`verify-endpoints`/`freeze`/`download`/`aggregate`/`recheck`/`publish`
六份产物都在盘上，`BUILD_COMPLETE.json` = `{"schema_version": 1, "build_complete": true}`），
但**两条预声明出口为假**：

| 出口（决策 0070 第 4 条） | 实测 | 判定 |
| --- | --- | --- |
| **E1** 六相位全绿 + `build_complete: true` | 六阶段产物齐全、退出码全 0、`build_complete: true` | **真** |
| **E2** 可达层 ≥13 且 LandScan 仍 `failed-no-fallback` | **可达层只有 10**（attempt01 为 13）；LandScan 仍 `failed-no-fallback`（未顶替） | **假** |
| **E3** `static_identity` 与 attempt01 的关系如实记录 | `6f65f88a…` ≠ attempt01 的 `8671ed7c…`；差异**不是**时间戳，见 §5 | 已记录（**不等价**） |
| **E4** recheck 最大相对偏差 ≤ `1e-05` | **`913872.595324707`**，12 项比对中 **6 项**超容差 | **假** |
| **E5** 网络 ≤40 GiB、磁盘余量 ≥60 GiB、墙钟 ≤ 86,400 s | 网络 12.78 GiB、`/data` 余量 523 GiB、墙钟 9,013 s（≤ planned 21,600 ⇒ 无 overrun） | **真** |
| **E6** 既有 94 条 `record_id` 不重复 | 本页登记为**新** `record:s3-urban-static-covariates-replay`；既有 94 条未改 | 见 `verify` 读数 |

**E2 与 E4 为假是这一轮的实际结果，不重试到偶然过、不放宽判据。**
本页**不**宣布本轮通过，也**不**宣布发布场可用于科学用途。

### 时间线（全部取自产物字段与 mtime，实测）

| 时刻（UTC） | 事件 | 证据 |
| --- | --- | --- |
| `06:36:23` | 端点核对完成 | `endpoint_report.json` 的 `accessed_utc` |
| `06:36:28` | 协议冻结 | `protocol.json` 的 `frozen_utc` |
| `06:36:49` | 首个原生瓦片落地 | `native/**` 最早 mtime（`native/**` 共 **234** 个文件，mtime **全部**落在 `06:36:49–07:14:03`） |
| `07:14:03` | 下载完成 | `download_receipt.json` 的 `downloaded_utc`（下载 2,260 s） |
| `09:06:18` | 聚合完成 | `aggregate_receipt.json` 的 `aggregated_utc`（聚合 6,735 s ≈ 1.87 h） |
| `09:06:31` | 再派生核对完成 | `recheck_receipt.json` 的 `checked_utc` |
| `09:06:36` | 发布完成 | `published/manifest/manifest.json` 的 `published_utc` |
| — | **总墙钟 9,013 s** | `09:06:36 − 06:36:23`（planned 21,600 / hard 86,400；soft overrun 0） |

**冻结先于下载**这一条在本轮**有独立测量证据**：`protocol.json`（`06:36:28`）早于
第一个原生瓦片 mtime（`06:36:49`）**21 s**。驱动里的 `frozen_before_any_download`
是**硬编码 `True`**（`scripts/r7_urban_static_round.py:211`，与 attempt01 同一处），
本页不把它当作测量；这里的时序结论来自上面的文件时刻。

## 1. 相位 1：逐层只读端点核对（访问日期 2026-10-31 UTC）

`endpoint_report.json`（`accessed_utc = 2026-10-31T06:36:23+00:00`，`schema` = `r7-urban-static-endpoints-v1`），
逐 tile `HEAD` 经 `http_public.open_public`。

| 层 | tile 数 | 可达 | HTTP 码分布 | 逐层 content-length 合计 | 下载文件 | 下载字节 |
| --- | --- | --- | --- | --- | --- | --- |
| `worldcover_v100_2020` | 49 | 48 | 200×48、**404×1** | 4,110,718,311 B | 48 | 4,110,718,311 |
| `worldcover_v200_2021` | 49 | 48 | 200×48、**404×1** | 4,056,204,735 B | 48 | 4,056,204,735 |
| `ghsl_built_s_2015` | 9 | **7** | 200×7、**502×1**、**超时×1** | 230,366,078 B | 7 | 230,366,078 |
| `ghsl_built_s_2020` | 9 | 9 | 200×9 | 276,030,196 B | 9 | 276,030,196 |
| `ghsl_built_h_2018` | 9 | **8** | 200×8、**超时×1** | 413,399,459 B | 8 | 413,399,459 |
| `ghsl_pop_2000` | 9 | 9 | 200×9 | 800,234,591 B | 9 | 800,234,591 |
| `ghsl_pop_2005` | 9 | 9 | 200×9 | 859,580,811 B | 9 | 859,580,811 |
| `ghsl_pop_2010` | 9 | 9 | 200×9 | 914,246,529 B | 9 | 914,246,529 |
| `ghsl_pop_2015` | 9 | 9 | 200×9 | 1,005,852,561 B | 9 | 1,005,852,561 |
| `ghsl_pop_2020` | 9 | 9 | 200×9 | 1,060,537,776 B | 9 | 1,060,537,776 |
| **合计** | — | **10 层** | — | **13,727,171,047 B** | **165** | **13,727,171,047 B** |

（上表每个数字都是我从 `endpoint_report.json` 的 `probes` / `total_content_length_bytes`
与 `download_receipt.json` 的 `records` 逐条统计出来的；两边的逐层字节**逐层相等**，
合计也相等。注意：`expected_bytes` 只统计**可达** tile 的 content-length，所以
「expected == downloaded」证明的是**可达集合**被完整取回，**不**证明产品的全部 tile 都拿到了。）

### 许可页（同一访问时刻）

| URL | HTTP | content-length | 说明 |
| --- | --- | --- | --- |
| `https://registry.opendata.aws/esa-worldcover/` | **无状态码** | — | `Connection reset by peer`（本轮失败） |
| GHSL `…/GHS_BUILT_S_…_4326_3ss/copyright.txt` | **200** | 540 B | `text/plain`；attempt01 抄录其正文含 CC BY 4.0 |
| `https://creativecommons.org/licenses/by/4.0/` | **403** | 17 B | 与 attempt01 的 04:02 读数同现象（那次 06:35 复测为 200）；本轮**未**复测 |

**如实记**：本轮 WorldCover 许可页没取到状态码（连接被重置），CC BY 页仍 403。
「许可」一栏沿用的是 attempt01 已登记的来源与页面 URL，**不是**本轮重新确认的许可意见
（`endpoint_report.json` 的 `limitations[0]` 自己也写明"reachability is what this report measures;
it is not a licence opinion"）。

### 丢层（5 层，`failed-no-fallback`，未合成顶替，R-008）

| 层 | 原因 | 证据（本轮实测） |
| --- | --- | --- |
| `ghsl_built_s_2000` | 9/9 tile 全部超时 | 9 个 probe 均 `RuntimeError: … The read operation timed out` |
| `ghsl_built_s_2005` | 9/9 tile 全部超时 | 同上 |
| `ghsl_built_s_2010` | 9/9 tile 全部超时 | 同上 |
| `landscan_global` | **需注册账号**，非匿名 | 目录里的静态条目（`failed_no_fallback: true`，`reason` 写明 ORNL 要申请人账号）；**本轮没有重新拨号**，沿用目录声明 |
| `copernicus_dem_glo30` | 不在决策 0063 具名清单 | 静态条目（`failed_no_fallback: false`）；AWS 匿名镜像在 attempt01 实测可达，本轮未取 |

**这三个 `ghsl_built_s` 历元在 attempt01 里是 9/9 HTTP 200**（我读了
`outputs/r7_urban_static_covariates_attempt01/endpoint_report.json` 逐层核对）。
⇒ **同一批 URL 在 2026-10-27 可达、在 2026-10-31 全部超时**。这是一次**可达性退化**，
不是"这些层不存在"。按 R-008 如实丢层，**没有**用任何其他层或合成值顶替。

### 逐 tile 缺瓦（不是丢层，如实记）

| 层 | 缺的 tile | 本轮状态 |
| --- | --- | --- |
| `worldcover_v100_2020` | `ESA_WorldCover_10m_2020_v100_N27E123_Map.tif` | **404**（产品按设计不发布的全海 3° tile，与 attempt01 同） |
| `worldcover_v200_2021` | `ESA_WorldCover_10m_2021_v200_N27E123_Map.tif` | **404**（同上） |
| `ghsl_built_s_2015` | `…_E2015_…_R5_C29.zip` | **超时**（`The read operation timed out`） |
| `ghsl_built_s_2015` | `…_E2015_…_R7_C29.zip` | **HTTP 502** |
| `ghsl_built_h_2018` | `…_ANBH_E2018_…_R7_C29.zip` | **超时** |

**缺瓦的下游后果（本轮实测，不是估计）**：把这些层与 attempt01 逐格比对，
`built_surface_frac_2015` 有 **278 个格**在 attempt03 为 0 而 attempt01 非 0
（行 0–15 与 57–64、列 0–11），`building_height_m` 有 **96 个格**（行 57–64、列 0–11）。
这些格**没有**被填充、插值或顶替 —— 它们就是缺 tile 的结果。
协议里记的是**逐 tile** 的 `absent_tiles` + `absent_failed_no_fallback: true`，
层本身仍保留（因为还有可达 tile），与 attempt01 对 WorldCover 的处理口径一致。

## 2. 相位 2：协议冻结（R-006）

- **`frozen_utc` = `2026-10-31T06:36:28+00:00`**；`protocol_sha256 = f42305d0cf5ac01d2e96dbe03d419bcb93715ab6dda6395d8d519bd74a05350c`。
- **`protocol_sha256_at_download` 与 `protocol_sha256` 相同**（`aggregate_receipt.json` 里两个字段逐字相等）
  —— 本轮**没有**发生 attempt01 那种"下载后补 `epoch`/`year` 记账键"的改动，
  因为这两个键在 `freeze` 的 `_layer_plan` 里已经写出（`scripts/r7_urban_static_round.py:190-200`）。
- 预算（`protocol.json` 的 `budget`）：`gpu_hours 0.0`、`planned_seconds 21600`、
  `hard_cap_seconds 86400`、`network_bytes_cap 42949672960`（40 GiB）、
  `disk_headroom_bytes 64424509440`（60 GiB）、`single_tile_bytes_cap 536870912`（512 MiB）。
- 与 attempt01 的协议差异（我用脚本逐键 diff 过两份 `protocol.json`）：
  **层数 13 → 10**（少了 `ghsl_built_s` 的 2000/2005/2010）、**丢层 2 → 5**、
  `ghsl_built_s_2015` 的 `urls` 9 → 7、`ghsl_built_h_2018` 9 → 8、
  `frozen_utc` 不同、`protocol_sha256` 不同。
  **获取规则本身（ROI、网格、面积基准、`spatial_resampling`、预算）逐字未变。**

## 3. 相位 3：下载

- **165 个文件、13,727,171,047 字节（12.78 GiB）**；与协议里 `expected_bytes` 之和**逐字节相等**（见 §1 表）。
- **`reused: false` ×165**：attempt02 的 `native/`（195 tile、14,484,496,452 B）**没有**被复用，
  同一批 URL 是**重新下载**的。这是成本事实，如实记。
- 网络客户端仍只在 `data/download/urban_static_sources.py`（R-016），全部经 `http_public.open_public`。
- 逐文件 sha256 在 `download_receipt.json`（165 条全带 `sha256`）。
  下面的 `source_sha256` 是我用 receipt 的 `{url: sha256}` 规范化后**重算**的，
  与 manifest 逐字相等（§5）。

## 4. 相位 4：聚合口径与产物

**逐字声明**：下列字段是对真实观测求的**派生统计量**，按**目标格界**做**面积加权**粗化；
`spatial_resampling = area_weighted_coarsen`（`docs/DATA_SPEC.md` §L2）。
**不是** `interpolation`，**更不是**"把粗场插成细真值"，**不得**当观测真值使用。

- **面积基准**（`aggregate_receipt.json` 的 `area_basis`，与协议同）：WGS84 **等积球**
  `R = 6371007.181 m`；原生格权重 `R² · dlon_rad · (sin(lat_top) − sin(lat_bottom))`。
- **归属规则**：原生**列**按与目标列的精确宽度重叠拆成两份（跨格列的份额计入两侧，
  `data/preprocess/urban_static_aggregate.py:123-147`）；原生**行**按 sin 带重叠比例
  （`:149-166`）；归约按"目标行 × 原生行块"向量化（`:172-203`）。
- **对齐实测**（我用 `rasterio` 读原生 tile 的 transform 核对）：
  - WorldCover `1/12000°`：`(x − EDGE_WEST)/res = −22500.0`、`(EDGE_NORTH − y)/res = 193500.0`
    —— **整数**，即原生格边与 0.25° 目标格边**重合**，没有跨界的半格；
  - GHSL `3 arcsec`：`(x − EDGE_WEST)/res = −8259.500481`、`(EDGE_NORTH − y)/res = −7169.500035`
    —— **半格错位**，每个目标格界都切开一个原生格。这正是决策 0071 认定格心/格界两种规则
    系统性不同的原因；本轮发布的是**格界**口径。
- **逐角色口径**：土地覆盖 = 类别面积占比（分母是整格）；建成面 = Σ(建成 m²) ÷ 格面积；
  建筑高度 = 对**有值**原生格的面积加权均值；人口 = Σ(人数) ÷ 格面积 km²。
- **产物**：`published/fields/static_fields.npz`，**33 个数组**，
  `fields_sha256 = ff27e5d4…`（我实测 `sha256sum` 该文件，与 `aggregate_receipt.json`
  和 `published/fields/` 的副本三方一致）。
  `built_surface_frac` 形状 **(2, 65, 65)**、`built_surface_frac_epochs = [2015, 2020]`；
  `pop_density_km2` 形状 (5, 65, 65)、`pop_density_km2_epochs = [2000, 2005, 2010, 2015, 2020]`。
- **覆盖率**：`land_cover_coverage_frac_*` 最小 0.5、13 个格 < 1；
  `land_cover_mapped_frac_2020` 平均 **0.7667**、`_2021` 平均 **0.7395**（与 attempt01 相同到 float32）。
- **数值概览**（我从 npz 直接读）：
  `built_surface_frac_2020` 均值 0.020775、最大 0.298241；
  `building_height_m` 均值 2.8999、最大 16.2070；
  `pop_density_km2_2020` 均值 318.249、最大 22,513.10。

## 5. 身份：`static_identity` 及其组成（**与 attempt01 不同，且不逐位相同**）

`scripts/r7_urban_static_round.py:457-497` 的 `_publish` 把身份写成
`sha256(规范化 JSON of 下列五键)`，我**独立重算**了这两步（用 receipt 与 npz 重算，不看 manifest 的中间值）：

| 组成键 | attempt03 实测 | attempt01（对照） |
| --- | --- | --- |
| `protocol_sha256` | `f42305d0cf5ac01d2e96dbe03d419bcb93715ab6dda6395d8d519bd74a05350c` | `8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904` |
| `source_sha256` | `86f813d04dce0e52a4d853304ff9c419b8176df4ea81402f89a5a07454030768`（**我重算并匹配**） | `0115dc3c809e0e6019096a3933cb3cc730c84f7a65326f81598b846212b30d42` |
| `aggregation_spec` | ROI/网格/面积基准/`area_weighted_coarsen` —— **与 attempt01 逐字相同** | 同左 |
| `output_sha256` | `ff27e5d45d86a57445c97f80986da7c39921f745968c76c46b649c3c13f24725` | `1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5` |
| `recheck_max_relative_delta` | `913872.595324707` | `5.1535351518462713e-08` |
| **`static_identity`** | **`6f65f88a780a62153bdd7d24211e0367ed2385dc102f1471167c665ceade74d9`**（**我重算并匹配**） | `8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03` |

**结论（逐项，不合并）**：

1. **两个身份不同。** 不存在"同身份两值"的问题 —— 身份确实变了。
2. **差异来源不是时间戳。** `published_utc` **不在**身份键里（`_publish` 只喂上面五个键），
   所以决策 0070 预判的"时间戳导致不同"在本轮**没有发生**。
3. **差异来自四个实质键**：协议（层集合变了）、源指纹（165 vs 195 个文件、少 3 个历元）、
   输出字节（33 vs 36 数组）、以及 recheck 读数（差 13 个数量级）。
4. `aggregation_spec`（规则声明）是**唯一逐字相同**的键 —— 即"规则没改，输入和结果改了"。
5. **不得**把本轮产物当作 attempt01 的替代品：数组集合不同（缺 `built_surface_frac_2000/2005/2010`），
   且若干格因缺 tile 变成 0。

## 6. 相位 5：再派生核对（**未通过**，如实记根因）

`recheck_receipt.json`（`checked_utc 2026-10-31T09:06:31+00:00`）：
`max_relative_delta = 913872.595324707`、`tolerance = 1e-05`、**`passed: false`**。

| 格 | 字段 | 独立（recheck） | 已发布 | 相对偏差 | 判 |
| --- | --- | --- | --- | --- | --- |
| (7,64) | `lc_built_frac` | 259,670.622 | 0.284143 | **9.139e+05** | FAIL |
| (7,64) | `building_height_m` | 6.446391 | 6.449189 | **4.339e-04** | FAIL |
| (7,64) | `built_surface_frac` | 0.115102748 | 0.115102746 | 2.426e-08 | ok |
| (7,64) | `pop_density_km2` | 1372.8413 | 1372.8413 | 6.679e-09 | ok |
| (32,32) | `lc_built_frac` | 94,154.190 | 0.112194 | **8.392e+05** | FAIL |
| (32,32) | `building_height_m` | 3.072675 | 3.072767 | **2.992e-05** | FAIL |
| (32,32) | `built_surface_frac` | 0.049561427 | 0.049561426 | 2.308e-08 | ok |
| (32,32) | `pop_density_km2` | 623.604129 | 623.604126 | 4.517e-09 | ok |
| (55,5) | `lc_built_frac` | 8,338.195 | 0.0105770 | **7.883e+05** | FAIL |
| (55,5) | `building_height_m` | 3.897768 | 3.897664 | **2.652e-05** | FAIL |
| (55,5) | `built_surface_frac` | 0.0026366297 | 0.0026366296 | 1.812e-08 | ok |
| (55,5) | `pop_density_km2` | 244.711106 | 244.711105 | 8.895e-10 | ok |

→ **6/12 超容差**（`lc_built_frac` 3 项全超、`building_height_m` 3 项全超；
两个 extensive 字段 `built_surface_frac` / `pop_density_km2` 6 项全过）。

### 6.1 我做的定位（说明差异出在哪一侧，**不**判定发布值正确）

`data/preprocess/urban_static_recheck.py`（决策 0071 的"第三套实现"）里：

- **土地覆盖分支**：分子是 `(block == code) * share`（`:90-93`），而分母是
  `(sin 带 of 目标格) * native_cols`（`:108-109`）。分子**没有**乘原生行的 sin 带，
  分母却按"sin 带 × 原生列数"缩放 ⇒ 商被放大 `Σshare / (band × native_cols)` 倍
  （`Σshare` ≈ 格内原生格数 9,000,000，`band × native_cols` ≈ 10.7）。
  实测三个格的 `独立/发布` 比值：**913,873.6 / 839,207.5 / 788,333.1**，
  与 `9000000/(band×3000)` 的预测（`band` = 0.0032805 / 0.0035742 / 0.0038070
  ⇒ 914,489 / 839,343 / 788,027）同量级 —— 量级对得上，具体到个位不是本页要断言的。
- **建筑高度分支**：权重是 `share`（面积占比）**未乘** sin 带（`:95-97`），于是算的是
  "按格内像素等权"的均值而不是"按真实面积"的均值 ⇒ 与发布值差 3e-5–4e-4 相对，
  正是"flat mean vs area-weighted mean"的量级（我单独算过同一格两种均值：
  `(32,32)` flat 3.072675069 vs area-weighted 3.072766925，发布值 3.072767019）。
- **两个 extensive 分支**（`built_surface`、`population`，`:98-99`、`:113-117`）不涉及带权，
  所以 6 项全过 —— 这条对照本身说明聚合侧与核对侧在**归属几何**上是一致的。

**第四份演算（我本轮新写，用来把"缺带权"这一个因素单独隔离）**：
`outputs/r7_urban_static_covariates_replay_probe_20261031_attempt01/probe.py`
（`sha256 = cd713088cde30d8a330d85c599db17e587d27871f747e132c0d95d768469cfa4`），
沿用**同一套**面积重叠几何，但把 sin 带放回分子/权重后重算 12 项：

- 最大相对偏差 **5.1535351518462713e-08**（＝attempt01 那个 float32 存储精度量级）；
  12 项**全部**回到 ≤1e-07。

**这条证据能说什么、不能说什么**：
- 能说：本轮的发布值**与"格界面积重叠 + sin 带加权"这一口径自洽到 float32 精度**，
  而 recheck 的分歧可由它**自己**漏掉带权解释掉。
- **不能**说发布值就是对的：这个 probe 与 recheck 共用同一套重叠公式（只在带权上不同），
  它**不是**一份独立实现的归属规则；管它是"第四份实现"不成立，只能算"同一几何下的带权校正"。
  真正独立的语义核对仍是**未完成项**。
- 因此 **E4 判为假**，发布场在补齐之前**不得**用于科学或下游用途。
  这与 `docs/R7_URBAN_STATIC_COVARIATES.md` 顶部（决策 0071）对该页数值的处理口径一致。

**交叉读数（只读比对，未重算文档里的数）**：attempt03 发布的
`built_surface_frac_2020` 在 `(7,64)` = 0.115102746、`(32,32)` = 0.049561427，
与 `docs/goals/s3-urban-static-covariates.md` §「格界归属修复」里已记的独立格界值
（0.11510275 / 0.04956143）在 8–9 位有效数字上一致 —— 说明本轮发布值**是格界口径**，
而不是 attempt01 的格心口径（attempt01 同两格为 0.11549331 / 0.04954307）。

## 7. 与 attempt01 的逐项对比

| 项 | attempt01（已登记，未动） | attempt03（本轮） | 差异说明 |
| --- | --- | --- | --- |
| 可达层数 | 13 | **10** | `ghsl_built_s` 2000/2005/2010 本轮 9/9 超时（attempt01 均 200） |
| 丢层 | 2（LandScan、DEM） | **5** | 上述 3 层 + 同两个静态条目 |
| 逐 tile 缺瓦 | 2（WorldCover 全海 tile ×2） | **5**（+`built_s_2015` ×2、`built_h_2018` ×1） | GHSL 侧新出现的 502/超时 |
| 下载 | 195 文件 / 14,484,496,452 B | 165 文件 / 13,727,171,047 B | 少 3 个 `ghsl_built_s` 历元与 3 个缺瓦 |
| 协议 digest | `8cc3c673…`（下载前 `c4028670…`） | `f42305d0…`（下载前后**相同**） | 本轮无"下载后补键"改动 |
| 数组数 | 36 | **33** | 缺 `built_surface_frac_{2000,2005,2010}` |
| `built_surface_frac` 形状 | (5,65,65) | **(2,65,65)** | 历元轴 `[2015,2020]` |
| `fields_sha256` | `1e9473f0…` | `ff27e5d4…` | 见下逐字段 |
| recheck | 5.15e-08，`passed: true` | **913872.6，`passed: false`** | §6 |
| `source_sha256` | `0115dc3c…` | `86f813d0…` | 文件集合不同 |
| `static_identity` | `8671ed7c…` | **`6f65f88a…`** | §5：四个实质键全变 |
| 网络 | 14,484,496,452 B（13.49 GiB） | 13,727,171,047 B（12.78 GiB） | — |
| 磁盘（本轮目录） | ≈20 GiB | **19,440,320,534 B（18.1 GiB）**；`native/` 19,438,858,786 B、`published/` 612,627 B | 三者都在 `/data` 上留存 |
| 墙钟 | ≈9,550 s | **9,013 s** | 均 < planned 21,600 |

### 7.1 逐字段差异（我从两份 npz 直接比对，4225 个格）

| 字段 | 不同的格 | 共有的非零格上的相对差 |
| --- | --- | --- |
| `lc_*`（五类，含 2020/2021） | **1–2 格** | 最大绝对差 4.66e-10 |
| `land_cover_coverage_frac_*` | 0 | 完全相同 |
| `land_cover_mapped_frac_*` | 1 格 | 5.96e-08（float32 末位） |
| `built_surface_frac_2020` | 3,704 | 中位 **0.204%**、p95 1.05%、最大 5.98%（≥0.001 的 3,161 格） |
| `building_height_m` | 3,703 | 中位 **0.036%**、p95 0.341%、最大 30.09%（≥0.5 的 3,726 格） |
| `pop_density_km2_2020` | 3,709 | 中位 **0.209%**、p95 1.41%、最大 19.48%（≥1 的 3,607 格） |
| `built_surface_frac_2015` | 3,744 | 另有 278 格在 attempt03 为 0（缺 tile，见 §1） |

**土地覆盖几乎不变的原因是可实测的**：WorldCover 原生格边与 0.25° 目标格边**重合**
（偏移 `−22500.0` / `193500.0`，整数），所以格心规则与格界规则选中**同一批**原生格；
两者只在边界 float 舍入上差 1–2 个格（≈1e-10）。GHSL 半格错位，因此同一批 GHSL 层
在两种规则下系统性不同 —— 中位 0.2% 与决策 0071 抽样得到的 0.139%（建面）同量级。

## 8. 成本（实测，不是估计）

| 项 | 实测 |
| --- | --- |
| GPU-h | **0.0** |
| 网络 | 13,727,171,047 B（12.78 GiB），165 文件，`reused=false ×165` |
| 磁盘（本轮） | 19,440,320,534 B（18.1 GiB）；`published/` 仅 612,627 B |
| `/data` 余量 | 523 GiB（`df -h /data`，本轮登记时实测；门 60 GiB） |
| 墙钟 | 9,013 s（端点 06:36:23Z → 发布 09:06:36Z）；下载 2,260 s、聚合 6,735 s |
| 预算 | planned 21,600 / hard 86,400 s；**soft overrun 0、hard 未触** |
| 失败/重跑了多少 | 本轮 = `attempt03`；`attempt02`（6 h 阶段超时，195 tile 在盘、无发布）**保留未删**；`attempt01` 未动 |

## 9. 缺口与限制（如实列出，`scientific_claim: false`）

1. **E2 假：层可达性退化。** 可达层 10 < 13；`ghsl_built_s` 的 2000/2005/2010 **整层丢失**
   （9/9 tile 超时），因此建面**历元轴只有 2 个**（2015/2020），而 attempt01 有 5 个。
   下游若需要 2000–2010 的建面，本轮产物**不能**替代 attempt01。
2. **缺 tile 造成 0 值格**：`built_surface_frac_2015` 278 格、`building_height_m` 96 格
   在 attempt03 为 0（行 0–15 与 57–64、列 0–11）。**没有**合成、插值或顶替（R-008）。
3. **E4 假：recheck 未过。** 6/12 项超容差（§6）。定位结论是"recheck 侧漏了 sin 带"，
   但**没有**一份独立归属实现来确认发布值语义正确；在补齐之前**不得**用于科学或下游用途。
4. **`static_identity` 不同且不逐位相同**（§5）。四个实质键都变了；`aggregation_spec` 是唯一
   逐字相同的键。**不得**声称"输入与规范相同、仅时间戳不同" —— 本轮 `published_utc`
   根本不在身份键内。
5. **许可页本轮未取到**：WorldCover registry 页连接被重置（无状态码）、CC BY 页 403。
   许可信息沿用 attempt01 已登记的来源，**不是**本轮重新确认的许可意见。
6. **`frozen_before_any_download` 是硬编码 `True`**（`scripts/r7_urban_static_round.py:211`），
   属声明不是测量；本轮的时序结论改由产物时刻独立支撑（§0）。
7. **`expected_bytes` 只覆盖可达 tile**，所以"逐字节相等"不等于"产品完整"。
8. `attempt02` 的 195 个已下载 tile **未被复用**（`reused: false`），本轮把同一批字节**又下了一遍**；
   磁盘上 attempt01/02/03 合计约 56 GiB 仍保留。
9. **土地覆盖只发布 5 个类别**（`docs/DATA_SPEC.md` §L2），五类之和不等于 1
   （mapped 平均 0.7667/0.7395）。
10. **多历元不是逐年观测**：GHS-BUILT-S / GHS-POP 是五年间隔、跨代传感器合成。
11. **人口交叉核对仍缺**（LandScan 注册门），**地形（DEM）仍缺**（不在具名清单）。
12. **0.25° 偏粗**：建成区在格内只是占比；原生瓦片保留在 `native/` 供延后的 1 km 轮。
13. **2023 test 未读、r 仍为 0**；本轮 0 GPU-h，不涉及模型训练或评分；
    **不宣布**本轮通过、不宣布发布场可科学使用、不自宣 goal 完成。
14. 账本 §7 的「合计已用」行显示 `247.356641`，而 `used_gpu_h` 为 `323.940495`
    —— 这是既有行的历史显示值，**本轮未改**，仅在此记录该差异。

## 10. 与编排路径的关系

本轮由 `CreateWorkflow`（`dryRun: false`，run `dwfrun-47a053b8-6d53-4686-a66c-bfcac1240a6f`，
`baselineCommit: cb69f2a`，`maxRounds: 1`）驱动；六个 stage 的 argv 名与驱动
`scripts/r7_urban_static_round.py` 的 `COMMANDS`（`:513-514`）一致。
F1 的字节基线是 `cb69f2a`，我实测 `git diff --name-only cb69f2a..HEAD` 只列出
`docs/goals/s3-urban-static-covariates.md`（一个文档文件），**五个实现文件逐字节未变**。
⇒ 「本轮运行期间实现没被悄悄改动」在本轮**有可机械复核的证据**；
「recheck 失败」因此是**实现既有的缺陷**（写于 `bb7ce2c`），不是运行中被改坏的。

**下一步（不在本页授权范围内）**：修 `data/preprocess/urban_static_recheck.py` 的带权口径
（自己的提交 + 自己的测试），然后再跑一次 `recheck`；若发布值不变、recheck 通过，
才谈得上 E4 成立。**本轮不放宽容差、不重试到偶然过、不改已冻判据。**

## 11. 登记绑定（如实说明指针对的是哪个提交）

索引记录 `s3-urban-static-covariates-replay` 的 `evidence_commit` / `evidence_first_commit`
指向 **`9144096ed705efec473c2afe8e1be5ec2d264207`**（登记前的 `HEAD`，即记下聚合超时根因与修复的
那次提交），**不是**包含本页的那次提交 —— 因为本页与那条记录是在**同一个登记提交**里落盘的，
提交 sha 在写记录时还不存在。这与既有 `s3-urban-static-covariates-wiring` 记录的做法一致
（其 `evidence_commit` 指代码提交 `67ebf14`，而不是创建证据页的 `3e14eec`）。
`evidence_sha256` 是本页**定稿后**的实际字节 hash（下方 `verify` 会再核一次）。

