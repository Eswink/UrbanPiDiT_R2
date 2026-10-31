# R7 城市静态协变量：决策 0073 重派生（attempt04）的读数

> **本页是 `docs/R7_URBAN_STATIC_COVARIATES.md`（attempt01）与
> `docs/R7_URBAN_STATIC_COVARIATES_REPLAY.md`（attempt03）之外的新页面。**
> 两页均未改动一个字节；attempt01 的证据摘要仍被账本行
> `s3-urban-static-covariates-acquisition` 的 `evidence_sha256` 钉住。
> **本页不声称与任何前序 attempt 逐位相同**，差异逐项列在 §6。
>
> **`scientific_claim`: false**；**`test_read`: false**；**0 GPU-h**；r 仍为 0。
> 本页是数据准备读数与工程判定，不是天气真值或机制结论。

- **日期**：2026-10-31（UTC；逐阶段时刻见 §5）
- **产物**：`outputs/r7_urban_static_covariates_attempt04/`
- **`static_identity`**：`063fb8059c4fd4d704f58dc4ce03aa1db0addf3ec34e1fe30e8efa8491df22d1`
- **`protocol_sha256`**：`a893e54bcf1e18e592937a5392c824723540c27cb2d347a24ffa8292ad157096`
  （== 下载时记录的 `protocol_sha256_at_download`）
- **`fields_sha256` / `output_sha256`**：`dfbab775a654230c4400f76e589331292934c9bdd2730dfe8dacfdfd0dceef05`
- **`source_sha256`**：`0115dc3c809e0e6019096a3933cb3cc730c84f7a65326f81598b846212b30d42`
  （与 attempt01 **逐字相同**）
- **`recheck`**：`max_relative_delta = 5.1535351323099157e-08`、`passed: true`（容差 `1e-05`）
- **不做**：不改 `model/**`、不重建 store、不读 2023 test、不改 attempt01/02/03 的任何字节。

## 0. 一句话结论

13 个具名层**全部匿名可达并已取回**（195 文件 / `14,484,496,452 B`，与冻结的
`expected_bytes` **逐字节相等**），按**格界 + 面积加权**聚合到 R7 冻结的 65×65 / 0.25° 网格，
GHSL 四个 role 的 **coverage 掩码全部为 1.0**（4225/4225 格），
独立再派生的 12 项比对**最大相对偏差 `5.15e-08`**、`passed: true`，发布走
`contracts.fresh_outputs`（`BUILD_COMPLETE.json` + `static_identity`）。
LandScan 与 Copernicus DEM 仍按 R-008 如实留缺口并写明 `reason`（未用任何层顶替）。

**`static_identity` 与 attempt01 不同**（`063fb805…` vs `8671ed7c…`），差异来源逐项见 §6；
**不声称逐位相同**。本页不宣布本轮"通过"，发布场是否可用于科学由复核者裁定。

## 1. 相位 1：只读端点核对

访问日期 **2026-10-31**（`endpoint_report.json` 的 `accessed_utc = 2026-10-31T14:42:22+00:00`）。
逐层逐 tile 做 `HEAD`（经 `http_public.open_public`，SSRF + 重定向校验 + timeout），
共 **197 次探测**：**195 个 `HTTP 200` + 2 个 `HTTP 404`**。
探测带有限重试（`data/download/urban_static_sources.py:265-300`：超时 60 → 90 → 120 s、
间隔 2/4 s），本轮 **3 个探测用了 2 次才拿到 200**（`ghsl_built_s_2000/R7_C30`、
`ghsl_built_s_2020/R7_C31`、`ghsl_pop_2020/R7_C31`），
最终 `failure_class` 只有 `none` 与 `absent` 两类、**`unreliable` 为空**。

| 层 | 源 / 版本 | 分辨率 | 许可 | tile 数 | 可达 | 逐层 content-length 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| `worldcover_v100_2020` | ESA WorldCover `v100`(2020) | 10 m | CC-BY-4.0 | 49 | 48 | 4,110,718,311 B |
| `worldcover_v200_2021` | ESA WorldCover `v200`(2021) | 10 m | CC-BY-4.0 | 49 | 48 | 4,056,204,735 B |
| `ghsl_built_s_2000` | GHSL GHS-BUILT-S R2023A `4326_3ss` | 3 arcsec ≈100 m | CC-BY-4.0 | 9 | 9 | 210,920,564 B |
| `ghsl_built_s_2005` | 同上 | 同上 | 同上 | 9 | 9 | 228,509,011 B |
| `ghsl_built_s_2010` | 同上 | 同上 | 同上 | 9 | 9 | 241,852,050 B |
| `ghsl_built_s_2015` | 同上 | 同上 | 同上 | 9 | 9 | 261,671,692 B |
| `ghsl_built_s_2020` | 同上 | 同上 | 同上 | 9 | 9 | 276,030,196 B |
| `ghsl_built_h_2018` | GHSL GHS-BUILT-H R2023A **ANBH** `4326_3ss` | 3 arcsec ≈100 m | CC-BY-4.0 | 9 | 9 | 458,137,625 B |
| `ghsl_pop_2000` | GHSL GHS-POP R2023A `4326_3ss`（**counts**） | 3 arcsec ≈100 m | CC-BY-4.0 | 9 | 9 | 800,234,591 B |
| `ghsl_pop_2005` | 同上 | 同上 | 同上 | 9 | 9 | 859,580,811 B |
| `ghsl_pop_2010` | 同上 | 同上 | 同上 | 9 | 9 | 914,246,529 B |
| `ghsl_pop_2015` | 同上 | 同上 | 同上 | 9 | 9 | 1,005,852,561 B |
| `ghsl_pop_2020` | 同上 | 同上 | 同上 | 9 | 9 | 1,060,537,776 B |

- 逐层 content-length 之和 = **14,484,496,452 B**，与 `download_receipt.json` 的
  `total_bytes` **相等**（我按 `endpoint_report.json` 重算过）。
- 所有层 `crs = EPSG:4326`、`licence = CC-BY-4.0`、
  `licence_url = https://creativecommons.org/licenses/by/4.0/`。

### 许可页（同一 `accessed_utc`，`HEAD`）

| URL | HTTP | content-length | content-type |
| --- | --- | --- | --- |
| `https://registry.opendata.aws/esa-worldcover/` | **200** | 2,234 B | `text/html` |
| GHSL `.../GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss/copyright.txt` | **200** | 540 B | `text/plain; charset=utf-8` |
| `https://creativecommons.org/licenses/by/4.0/` | **403** | 17 B | `text/plain; charset=UTF-8` |

第三个 403 与 attempt01（`2026-10-27T04:02:21Z`）**同一现象**：CC 许可正文页对 HEAD 返回 403。
`endpoint_report.json` 的 `limitations` 明写"reachability is what this report measures; it is
not a licence opinion"，所以**本轮不构成任何新的许可意见**。

### 丢层（`failed-no-fallback`，R-008，未合成顶替）

| 层 | role | 原因 | 证据 |
| --- | --- | --- | --- |
| `landscan_global` | `population_cross_check` | 需 ORNL 申请账号，非匿名 | `failed_no_fallback: true`；站点可达但数据目录在注册墙后 |
| `copernicus_dem_glo30` | `terrain` | **不在具名清单** | `failed_no_fallback: false`；AWS 匿名镜像实测可达，但决策 0063 未授权该层，本轮不取 |

两条都带**非空 `reason`**（`scripts/r7_urban_static_round.py` 的 `require_dropped_reasons`
在 `freeze`/`publish` 入口断言，决策 0072 第 ④ 条）。

### 一个缺 tile（如实记，不是丢层）

`N27E123`（3°×3°，27–30°N / 123–126°E，**东海全海**）在 ESA WorldCover 桶里两年都 **404**
（`content_type = application/xml`，即 S3 的错误体）。这是产品**按设计不发布**的全海 tile，
记为**逐 tile** 的 `absent_tiles`（`absent_failed_no_fallback: true`），
**不**据此丢整个 WorldCover 层。受影响目标格由 `land_cover_coverage_frac_*` 暴露：
**13 个格、覆盖率 0.5**，实测坐标 `(row, col) = (52..64, 64)`（col 64 的东半格，rows 52–64）。
**没有**为缺口合成任何值。

## 2. 相位 2：冻结协议（R-006）

- `protocol.json` 的 `frozen_utc = 2026-10-31T14:42:29+00:00`，即端点核对完成 7 s 之后。
- **`protocol_sha256 = a893e54bcf1e18e592937a5392c824723540c27cb2d347a24ffa8292ad157096`**
  与我按 `scripts/r7_urban_static_round.py:77-79` 的规范化 `json.dumps(sort_keys, 无空白)`
  独立重算的 digest **逐字一致**。
- 冻结内容：ROI 格界 `26.875–43.125°N / 106.875–123.125°E`、目标网格 65×65 / 0.25° /
  行 0 最北 / lon 升序 / `float32`、逐层 source/version/crs/许可/`tile_count`/`expected_bytes`/
  聚合规则/角色、`dropped_layers`、输出布局、预算、`limitations`。
- 预算（冻结值，未改）：`gpu_hours: 0.0`、`planned_seconds: 21600`、
  `hard_cap_seconds: 86400`、`network_bytes_cap: 42949672960`（40 GiB）、
  `disk_headroom_bytes: 64424509440`（60 GiB）、`single_tile_bytes_cap: 536870912`（512 MiB）。
- **`frozen_before_any_download` 仍是硬编码 `True`**（`scripts/r7_urban_static_round.py:150`），
  是声明而非测量（决策 0071 §③ 已记）。本轮的时序**由产物独立支撑**：
  协议 `14:42:29Z` 早于首个原生瓦片 mtime `14:43:48Z`，**79.8 s**。
- 本轮的协议 digest **下载前后相同**（`aggregate_receipt.json` 的
  `protocol_sha256` == `protocol_sha256_at_download` == `a893e54b…`），
  与 attempt01 那次"下载后补两个记账键导致 digest 变更"不同。

## 3. 相位 3：下载

- **195 个文件、14,484,496,452 字节（13.4897 GiB）**，与冻结的 `expected_bytes` 合计
  **逐字节相等**；`download_receipt.json` 的 `total_bytes` 同值。
- 195 条 record **全部 `reused: false`** ⇒ 没有复用任何在盘文件，同一批 URL 被重新取了一遍。
- 网络客户端只在 `data/download/urban_static_sources.py`（R-016 唯一网络面），
  全部经 `http_public.open_public`。
- 逐文件 `sha256` 记在 `download_receipt.json`。GHSL 的 `.zip` 保留（可复核的下载物），
  单个 GeoTIFF 成员解出为原生分辨率瓦片，两个 sha256 都记：
  盘上 `native/` 共 **294 个文件 = 195 `.tif` + 99 `.zip`**。
- 实际网络字节 13.49 GiB < 40 GiB 上限（33.7%）；磁盘余量门 60 GiB 未被触。
- **本轮逐文件 sha256 与 attempt01 完全相同**（195/195 的 `(url, bytes, sha256)` 无一条不同，
  我逐条比过两份 `download_receipt.json`）⇒ `source_sha256` 两轮**逐字相同**。

## 4. 相位 4：聚合到 0.25°（派生统计量，**不是**观测真值）

**逐字声明**（沿用 `protocol.json` 的 `derived_statistic_not_truth`）：下列字段是**对真实观测
求的派生统计量**，按**目标格界**做**面积加权**粗化；`spatial_resampling = area_weighted_coarsen`
（`docs/DATA_SPEC.md` §L2）。**不是** `interpolation`，**更不是**"把粗场插成细真值"。
**不得**把聚合结果当观测真值使用。

- **面积基准**：WGS84 **等积球**（authalic，`R = 6371007.181 m`）；原生格权重
  `R² · dlon_rad · (sin(lat_top) − sin(lat_bottom))`，即把原生网格放到**圆柱等积投影**后
  按投影面积加权。
- **格界**：目标格 `26.875–43.125°N / 106.875–123.125°E`，0.25°；row 0 最北（与天气实例的
  `latitude` 降序对齐），lon 升序。
- **对齐实测**（我在本 attempt 的原生瓦片上用 `rasterio` 量的）：
  - WorldCover 原生格 `8.3333e-05°`（1/12000°），原点 `105.0 / 27.0`；目标西界落在
    第 **`22500.0`** 列、北界在第 **`−193500.0`** 行 —— **整数偏移，格边重合**；
  - GHSL `4326_3ss` 原生格 `8.3333e-04°`，目标西界在第 **`8259.500481`** 列、
    北界在第 **`7169.500035`** 行 —— **半格错位**，必须按**格界面积重叠**归属
    （决策 0071），不能按格心。
- **逐角色口径**（`protocol.json` 的 `aggregation`，逐字）：
  - 土地覆盖 `class area fraction per cell edge box`（5 类；分母是整格面积）；
  - 建成面 `sum(built m^2) / cell area`；
  - 建筑高度 `area-weighted mean over native cells with a value`（ANBH，只对有值原生格加权）；
  - 人口 `sum(counts) / cell area`（GHS-POP 单位是 counts）。
- **coverage 掩码（决策 0073 第 2 条）**：新增 11 个数组
  `built_surface_coverage_frac_{2000,2005,2010,2015,2020}`、`building_height_coverage_frac`、
  `pop_density_coverage_frac_{2000,2005,2010,2015,2020}`。
  `aggregate` 相位对任何 `< 1.0 − 1e-9` 的格 `SystemExit`（`scripts/r7_urban_static_round.py:273`
  的 `COVERAGE_TOLERANCE`、`:276-299` 的 `require_full_coverage`、`:347-348` 的调用点）。
- 输出：`published/fields/static_fields.npz`，**47 个数组**（attempt01 是 36 个，
  多出的 11 个即上面的 coverage 掩码）。
  `fields_sha256 = dfbab775a654230c4400f76e589331292934c9bdd2730dfe8dacfdfd0dceef05`，
  与 `sha256sum published/fields/static_fields.npz` **逐字一致**。

### coverage 读数（逐层，来自 `aggregate_receipt.json` 的 `layer_receipts`）

| 层 | role | 已发布 `*_coverage_frac_*` 的最小值（float32） | receipt 记录的 float64 最小值 | 严格 `<1.0` 的格数 |
| --- | --- | --- | --- | --- |
| `worldcover_v100_2020` | land_cover | 0.5 | 0.5 | 13 |
| `worldcover_v200_2021` | land_cover | 0.5 | 0.5 | 13 |
| `ghsl_built_s_2000/2005/2010/2015/2020` | built_surface | **1.0** | `0.9999999999999672` | 65 |
| `ghsl_built_h_2018` | building_height | **1.0** | `1.0` | 0 |
| `ghsl_pop_2000/2005/2010` | population | **1.0** | `1.0` | 0 |
| `ghsl_pop_2015/2020` | population | **1.0** | `0.9999999999999672` | 65 |

- 四个 GHSL role 的 **4225 个格全部覆盖**：float64 缺口最大 `3.28e-14`（相邻瓦片边界只到
  浮点精度），换算成发布的 `float32` 后**逐格为 `1.0`**；因此 `aggregate` 的容差闸门
  （`1e-9`）通过，**没有任何格被当成 `0` 写入**。
- **如实记一处记账不一致**：`layer_receipts` 的 `cells_below_full_coverage` 用的是**严格
  `< 1.0`**（`scripts/r7_urban_static_round.py:351`），所以上表那六层报 **65**，
  而闸门用的是 `1 − 1e-9`。两者都是真读数、不互相矛盾，但同一份 receipt 里两个口径并存，
  读者容易误读为"65 格缺数据"。本轮**不改**这段代码（属于代码变更，需自己的测试与决策），
  只在此登记。
- `worldcover_*` 的覆盖缺口**不触发闸门**：土地覆盖分支没有调用 `require_full_coverage`
  （`:306-322` 只写数组、不 gate），因为 48/49 是产品按设计的全海缺 tile，
  若 gate 会让整个层无法发布。缺口由 `land_cover_coverage_frac_*` **显式暴露**，未合成任何值。

## 5. 相位 5：独立再派生核对

抽样格 `(7,64)`、`(32,32)`、`(55,5)`，每格对 4 个字段**从原始瓦片重算**：
`data/preprocess/urban_static_recheck.py` **不 import 聚合器**（模块 docstring 明写
"deliberately imports nothing from `urban_static_aggregate`"，`:34-40` 的 import 只有
`numpy` 与 `urban_static_sources`），归属按**精确球面面积重叠**独立推出，
且**逐 role 用不同权重**（土地覆盖/高度用绝对 `m²`，建成面/人口用无量纲重叠比例，
见该模块 docstring 的表与决策 0072 §③）。

| 格 | 字段 | 独立重算 | 已发布 | 相对偏差 |
| --- | --- | --- | --- | --- |
| (7,64) | `lc_built_frac` | 0.2841428366706066 | 0.2841428220272064 | 5.1535351323099157e-08 |
| (7,64) | `building_height_m` | 6.449189373716697 | 6.449189186096191 | 2.9092107609856937e-08 |
| (7,64) | `built_surface_frac` | 0.11510274838480024 | 0.11510274559259415 | 2.425837956309839e-08 |
| (7,64) | `pop_density_km2` | 1372.8412994238968 | 1372.84130859375 | 6.679470650642049e-09 |
| (32,32) | `lc_built_frac` | 0.11219417621494986 | 0.11219417303800583 | 2.8316479735700823e-08 |
| (32,32) | `building_height_m` | 3.072766924857883 | 3.0727670192718506 | 3.072604176068507e-08 |
| (32,32) | `built_surface_frac` | 0.049561427187453364 | 0.04956142604351044 | 2.308131580983184e-08 |
| (32,32) | `pop_density_km2` | 623.6041287933278 | 623.6041259765625 | 4.51691261824752e-09 |
| (55,5) | `lc_built_frac` | 0.010576994558487955 | 0.01057699415832758 | 3.783309032147837e-08 |
| (55,5) | `building_height_m` | 3.8976643279562695 | 3.8976643085479736 | 4.979468297660013e-09 |
| (55,5) | `built_surface_frac` | 0.002636629670212134 | 0.0026366296224296093 | 1.8122577586214874e-08 |
| (55,5) | `pop_density_km2` | 244.71110556433842 | 244.7111053466797 | 8.894518168582217e-10 |

**最大相对偏差 `5.1535351323099157e-08`**，判据 `<= 1e-05`（该容差就是 float32 **存储**
精度，不是给算法差异留的余地）。`recheck_receipt.json` 的 `passed = true`，
`checked_utc = 2026-10-31T18:27:34+00:00`。

**这条核对的能力边界（如实记）**：抽样的 3 个格 × 4 个字段仍然只覆盖
`worldcover_v200_2021`、`ghsl_built_s_2020`、`ghsl_built_h_2018`、`ghsl_pop_2020` 四个层，
`built_surface_frac` 取的是 `[-1]`（2020）。**它对"层内不完整"没有覆盖能力**——
这正是 `attempt03` 的 288 格静默零值落在抽样之外的实证（`docs/R7_URBAN_STATIC_COVARIATES.md` §12.2）。
本轮该盲区由**coverage 掩码 + `aggregate` 的 fail-closed 闸门**兜住，**不是**由 recheck 兜住；
两者不可互相替代。

### 发布契约（相位 6）

`data/preprocess/contracts.py` 的 `fresh_outputs` 独占创建 `published/fields/` 与
`published/manifest/`；`BUILD_COMPLETE.json` = `{"schema_version": 1, "build_complete": true}`。
`static_identity` 覆盖**源文件指纹 + 聚合规范 + 输出字节 + 协议 + recheck 读数**五个键
（`scripts/r7_urban_static_round.py:450-460`）；`published_utc` **不在**其中。
`publish` 对 `recheck_receipt["passed"] is not True` 直接 `SystemExit`（`:495`，决策 0072 第 ③ 条），
所以本 attempt 的发布**隐含 recheck 已通过**——这是闸门，不是事后追认。

我从五个键**独立重算** `static_identity` 与 `source_sha256`（用 `:77-79` 的同一个规范化
JSON digest 规则），逐字匹配 manifest；
独立重算 `source_sha256` 也从 `download_receipt.json` 的 `url→sha256` 映射得到同一值。

## 6. 与 attempt01 的逐项对比（**不声称逐位相同**）

| 项 | attempt01 | attempt04 | 判读 |
| --- | --- | --- | --- |
| `static_identity` | `8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03` | `063fb8059c4fd4d704f58dc4ce03aa1db0addf3ec34e1fe30e8efa8491df22d1` | **不同** |
| `protocol_sha256` | `8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904` | `a893e54bcf1e18e592937a5392c824723540c27cb2d347a24ffa8292ad157096` | 不同，**差异只有 `frozen_utc`**（见下） |
| `source_sha256` | `0115dc3c…b30d42` | `0115dc3c…b30d42` | **逐字相同** |
| `aggregation_spec` | `area_weighted_coarsen` + 同一 `area_basis` + 同一 `grid` | 同左 | **逐字相同** |
| `output_sha256` | `1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5` | `dfbab775a654230c4400f76e589331292934c9bdd2730dfe8dacfdfd0dceef05` | 不同（数组 36 → 47，且数值不同） |
| `recheck_max_relative_delta` | `5.1535351518462713e-08` | `5.1535351323099157e-08` | 不同（末 8 位） |
| 下载文件 / 字节 | 195 / `14,484,496,452` | 195 / `14,484,496,452` | 相同；**逐文件 sha256 195/195 相同** |
| `published_utc` | `2026-10-27T06:31:18Z` | `2026-10-31T18:27:39Z` | 不同，但**不在身份五键内** |
| 数组数 | 36 | 47 | +11 coverage 掩码 |
| 聚合归属规则 | 格心（`7065f19` 版 `urban_static_aggregate.py`） | 格界面积重叠（`6f21f71` 起） | 规则不同，见 §6.2 |
| recheck 实现 | 与聚合共用格心规则（`docs/R7_URBAN_STATIC_COVARIATES.md` §5 抬头警告） | 第三实现 + 逐 role 权重（`bb7ce2c`/`4a3f5dd`） | 核对器不同 |

### 6.1 身份为什么不同（逐键归因，**不是**时间戳）

五个键里**三个不同**、**一个相同**（`aggregation_spec`）、一个末位不同（recheck 读数）：

1. **`protocol_sha256` 不同，但语义没有变。** 我把 attempt01 的 `protocol.json` 去掉
   `protocol_sha256` 后**只把 `frozen_utc` 换成 attempt04 的值**重算 digest，得到
   `a893e54b…`，与 attempt04 的 `protocol_sha256` **逐字相等**；反向替换也精确回到
   `8cc3c673…`。⇒ **协议 digest 的差异完全且仅由 `frozen_utc` 造成**，
   逐层 URL/`tile_count`/`expected_bytes`/聚合规则/预算/ROI/网格/`dropped_layers`/`limitations`
   在两轮之间**逐字相同**（我逐字段比过 13 个 layer plan 与全部顶层键）。
2. **`source_sha256` 相同**：195 个 `url→sha256` 逐条相同。
3. **`output_sha256` 不同**：数组数 36 → 47（coverage 掩码），且 27 个共有数组的数值不同
   （9 个逐位相同：`latitude`、`longitude`、`built_surface_frac_epochs`、
   `pop_density_km2_epochs`、两个 `land_cover_coverage_frac_*`、`lc_water_frac`、
   `lc_water_frac_2021`、`lc_tree_frac_2020`）。数值差异的**主要来源是归属规则变了**
   （格心 → 格界，决策 0071），不是数据变了：源字节逐条相同。
4. **`recheck_max_relative_delta` 末位不同**：`5.1535351518…e-08` → `5.1535351323…e-08`。
   两轮的**最大值都来自 `lc_built_frac@(7,64)`**，即 WorldCover 路径——而 WorldCover 原生格边
   与目标格边重合（§4 的 `22500.0` / `−193500.0`），故两轮该值都停在 float32 存储精度量级；
   末位差异来自求和顺序与覆盖数组布局变化。

⇒ 结论：**身份不同是"同一批源字节 + 同一套聚合规范 + 变化的 `frozen_utc` 与已改的归属规则"
的必然结果**，不是因为数据漂移。**仍不声称与 attempt01 逐位相同**。

### 6.2 数值差异的分布（attempt04 vs attempt01，4225 格）

只统计 `|attempt01| > 1e-4` 的格（避免 0 附近比值失真）：

| 字段 | 计入格数 | 中位相对差 | p95 | 最大 | 比值范围 | 比值在 [0.5, 2] 之外 |
| --- | --- | --- | --- | --- | --- | --- |
| `built_surface_frac_2000` | 3489 | 0.002122 | 0.012836 | 0.2449 | [0.9331, 1.2449] | **0** |
| `built_surface_frac_2005` | 3519 | 0.002140 | 0.012832 | 0.1938 | [0.9239, 1.1938] | **0** |
| `built_surface_frac_2010` | 3537 | 0.002140 | 0.012776 | 0.1601 | [0.9293, 1.1601] | **0** |
| `built_surface_frac_2015` | 3552 | 0.002132 | 0.012914 | 0.1407 | [0.9257, 1.1407] | **0** |
| `built_surface_frac_2020` | 3579 | 0.002087 | 0.012074 | 0.1248 | [0.9279, 1.1248] | **0** |
| `pop_density_km2_2020` | 3810 | 0.002034 | 0.015597 | 0.8056 | [0.5971, 1.8056] | **0** |
| `building_height_m` | 3822 | 0.000350 | 0.003280 | 0.0503 | [0.9693, 1.0503] | **0** |
| `lc_built_frac` | 3687 | 0 | 0 | 8.26e-08 | [1.0000, 1.0000] | **0** |

- **土地覆盖只差 1–2 个格**：`lc_built_frac` 全网格只有 `(5,0)`（相对 `1.023e-07`）与
  `(13,20)`（相对 `8.258e-08`）两格不同，绝对差分别是 `9.09e-13` 与 `4.66e-10`；
  `land_cover_mapped_frac_2020` 最大相对差 `5.98e-08`。WorldCover 原生格边与 0.25° 目标格边
  **重合**，两种归属规则给出同一答案（§4 实测）。
- **GHSL 三层系统性不同**，中位 `0.204%`（建面 2020）/ `0.203%`（人口）/ `0.035%`（高度），
  与决策 0071 抽样得到的 `0.139%` 同量级、方向一致：GHSL 是 **3 arcsec 半格错位**
  （`8259.500481` / `7169.500035`），格心与格界归属本就不同。
- **没有任何整块归零，也没有任何格的数量级差异**：全部字段的 `attempt04/attempt01` 比值
  落在 `[0.597, 1.806]`，超 2 倍的格数为 **0**；
  `attempt01` 里 `> 1e-4` 而在 attempt04 变 0 的格数为 **0**。
- 反方向有 **1 个格**从 0 变正：`(23,63)` 在 `built_surface_frac_2020`（0 → `6.96e-07`）、
  `built_surface_frac_2015`（0 → `6.90e-07`）、`pop_density_km2_2020`（0 → `0.00316`）、
  `building_height_m`（0 → `2.4325`）四个字段上同时出现。这与格心规则把某个边界原生格
  归到相邻目标格一致（格界规则把它归回本格），**不是**新数据。
- `attempt03` 的 288 个静默零值格（192 NW + 96 SW）在本 attempt **全部有值**：
  `built_surface_frac_2015` 的 NW 块均值 `0 → 0.00302352`、SW 块 `0 → 0.00376627`；
  `building_height_m` 的 SW 块 `0 → 3.10041`（NW 块两轮都是 `2.50191`，本来不空）。

## 7. 缺口与限制（如实列出，未做即写未做）

- **LandScan 未取**（需注册，`failed-no-fallback`，未顶替）⇒ 人口交叉核对缺失。
- **地形（DEM）未取**（不在具名清单）⇒ 2 m 温度的静态地形协变量缺失。
- **WorldCover 一个全海 tile 不存在**（`N27E123`，两年都 404）⇒ 13 个格（rows 52–64 × col 64）
  的东半格无数据，由 `land_cover_coverage_frac_*`（0.5）暴露；未合成。
- **土地覆盖只发布 5 个类别**（`docs/DATA_SPEC.md` §L2：built/tree/crop/water/bare），
  其余类别（草地/灌丛/雪/湿地/红树/苔藓）未发布，故 5 个占比之和不等于 1：
  `land_cover_mapped_frac_*` 均值 **0.7667（2020）/ 0.7395（2021）**。
- **多历元不是逐年观测**：GHS-BUILT-S / GHS-POP 是 **1975–2030 五年间隔、跨代传感器合成**；
  本轮取 2000/2005/2010/2015/2020，`built_surface_frac_epochs = [2000,2005,2010,2015,2020]`。
- **0.25° 偏粗**：建成区在格内只是"占比"，不含城市内结构；原生瓦片已保留供延后的 1 km 轮。
- **许可页 HEAD 403**（`creativecommons.org/licenses/by/4.0/`，两轮同一现象）；
  许可信息沿用既有来源，**不是**本轮重新确认的许可意见。
- **`frozen_before_any_download` 是硬编码 `True`**（`scripts/r7_urban_static_round.py:150`）；
  时序结论由产物时刻支撑（79.8 s），该字段本身不是测量。
- **`layer_receipts.cells_below_full_coverage` 与闸门容差不一致**（§4 末），
  六层报 65 而闸门按 `1−1e-9` 通过；未改代码，仅登记。
- **recheck 的抽样设计对"层内不完整"没有覆盖能力**（§5 末）。
- **attempt04 的 receipt 里没有 workflow run id**：`endpoint_report.json`/`protocol.json`/
  `download_receipt.json`/`aggregate_receipt.json`/`recheck_receipt.json` 都没有
  `run`/`workflow` 字段（attempt03 的证据页记录了 `dwfrun-47a053b8`，本 attempt 没有对应记录）。
  ⇒ 我只能证明**六个阶段的产物本身是实跑的**（各 receipt 的 UTC 时刻与 303 个在盘文件的
  mtime 交错），**不能**据本 attempt 的产物证明它由哪个编排器驱动。这一点**未做**，如实记。
- **CI 不装 rasterio**：`tests/test_r7_urban_static_round.py` 里需要栅格的用例在 CI 上 `skip`，
  **skip 不算通过**。本页不引用任何 CI 读数。
- **`scientific_claim: false`**：这是数据准备读数，不是科学结论；本轮 0 GPU-h，
  2023 test 未读，r 仍为 0。

## 8. 口径与成本（实测，不是估计）

| 项 | 实测 |
| --- | --- |
| GPU-h | **0.0** |
| 网络 | `14,484,496,452 B`（13.4897 GiB）、195 文件、`reused: false` ×195、占 40 GiB 上限 **33.7%** |
| 磁盘（本 attempt 目录） | `20,975,449,463 B`（19.5349 GiB）；其中 `native/` `20,973,407,596 B`、`published/` `884,013 B`；共 303 个文件 |
| `/data` 余量 | 登记时 `531,325,603,840 B`（**494.84 GiB**）≥ 60 GiB 门 |
| 墙钟 | **13,517 s**：端点核对 `14:42:22Z` → 发布 `18:27:39Z` |
| 分段 | 端点→冻结 7 s；冻结→下载完成 **8,689 s**；下载→聚合 **4,808 s**；聚合→recheck 8 s；recheck→publish 5 s |
| 时长口径 | 名义 `planned 21,600 s` / `hard 86,400 s`；实测 13,517 s 在 planned 内，**soft overrun 0**、hard 未触 |

网络与磁盘只记会计，**不加 GPU 账**（本轮 GPU-h 记 0）。

## 9. 下一动作

发布场是否可用于科学，由独立复核裁定，本页不宣布。就本 attempt 自身而言，
**已满足**：13/13 层可达、GHSL coverage 4225/4225、recheck `passed: true`、
`BUILD_COMPLETE` 为真、身份可独立重算。
**仍未解决**的三项（按性质分开记，不合并）：

1. **`attempt03` 保持双重不可用**（recheck 当时未过 + 288 格静默零值），不得被本 attempt 顶替叙述；
2. **recheck 抽样覆盖不足**（4 层 × 3 格）——只由 coverage 闸门兜底，需要一份独立的
   全格校验设计（自己的 ADR）；
3. **`cells_below_full_coverage` 口径不一致**（§4 末），属代码变更，需自己的测试。

第一条命令：`.venv/bin/python tools/r7_round.py state`

## 7. E5 的两处异常值：已查清，均非缺陷（执行者独立复核）

`attempt04` 与 `attempt01` 的字段差异（格心→格界）整体符合决策 0071 的预期，但有两个数
**超出** 0071 记录的「中位 `0.139%`、最大 `1.225%`」，必须解释而不是略过：

| 字段 | 中位差 | 最大差 | 解释 |
| --- | --- | --- | --- |
| `lc_built_frac` | `0.0000%` | `0.0000%` | 4225 格里只有 2 格有差，最大**绝对**差 `4.66e-10`（float32 存储精度） |
| `building_height_m` | `0.0350%` | `5.0250%` | 面积加权均值的边界取舍 |
| `built_surface_frac`（2020） | `0.2042%` | `122.77%` | 见下 |
| `pop_density_km2`（2020） | `0.2034%` | `80.56%` | 见下 |

**`lc_built_frac` 几乎为零是正确的，不是"没生效"**：WorldCover 原生分辨率 0.008333°
（0.25° 的 **1/3000**），一个目标格内有 3000×3000 个原生像素。格心与格界归属在这个尺度上
只在最外一圈像素上不同，差异远小于 float32 精度。决策 0071 实测的 `0.139%`–`1.225%`
是针对 **GHSL 3 arcsec**（1/300，粗 10 倍），两者不能互相套用。

**`built_surface` 的 `122.77%` 出现在极小值格上**，绝对差只有 `3.5e-07`：

```
(16,48) a01=2.814e-07  a04=6.269e-07  rel=122.77%   ← a01 排名 418/4225
(63,55) a01=6.696e-08  a04=1.150e-07  rel= 71.72%
( 2, 3) a01=2.545e-07  a04=4.290e-07  rel= 68.55%
```

这些格的原生累计建成面积近乎为零（排名在最低 10%），两种归属规则在边界像素上的取舍
不同，相对误差自然被放大；**绝对量级仍是 1e-07 量级**，对任何统计都无影响。
判据应看「有没有整块归零或数量级差异」——E5 实测**没有**：`zeros a01=403 / a04=402`，
两侧零值格数几乎相同（差 1），与 `attempt03` 那种「192/192 精确全零」形态完全不同。

## 8. 本轮遗留的两处记账缺陷（已确证，未修）

由登记员报出、执行者独立复核确认。两者都是**读数口径**问题，不影响已发布数值：

1. **同一份 `layer_receipts` 里两个 coverage 口径并存。**
   `scripts/r7_urban_static_round.py:351` 的 `cells_below_full_coverage` 用**严格 `< 1.0`**，
   而闸门 `require_full_coverage` 用 `1 − 1e-9`（`:283`）。于是六个 GHSL 层在同一份
   receipt 里报 `cells_below_full_coverage = 65`，同时闸门判定"全部覆盖"。
   两个都是真读数（float64 缺口 `3.28e-14` 确实 `< 1.0`），但读者会误读成"65 格缺数据"。
   修法应让 receipt 复用 `COVERAGE_TOLERANCE`；**本轮不改**（属代码变更，需自己的测试与决策）。
2. **`land_cover` 分支没有 coverage 闸门。**
   `:306-322` 只写 `land_cover_coverage_frac_*`、不调用 `require_full_coverage`。
   这是**刻意的**：WorldCover 的 48/49 是产品按设计缺的全海 tile，若 gate 会让整个层无法发布。
   缺口由数组显式暴露、未合成任何值。但它意味着决策 0073 第 2 条「所有 role 都必须带
   coverage 掩码，不完备即 fail-closed」在 `land_cover` 上**只做到了"带掩码"，没做到
   "fail-closed"**。这一点必须在下次改这块代码时明确写进决策，不能当成已完成。

E1–E6 六项出口的逐项读数见 §3–§6；本节的缺陷不影响其中任何一项的判定。
