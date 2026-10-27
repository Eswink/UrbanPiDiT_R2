# R7 城市静态协变量获取与发布（S3 / 决策 0063 / 计划 0027）

- **日期**：2026-10-27（UTC 读数见各 receipt）
- **`scientific_claim`: false**；**`test_read`: false**；**0 GPU-h**
- **产物**：`outputs/r7_urban_static_covariates_attempt01/`
- **`static_identity`**：`8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03`
- **不是** issue #9（那份 BLOCKED 针对 km 级**动态**真值；静态协变量不在其结论范围内）。
  本轮**不改** `model/**`、**不重建** store、**不加** state 通道。

## 0. 一句话结论

13 个具名层（ESA WorldCover v100/v200、GHSL GHS-BUILT-S 五历元、GHS-BUILT-H 2018、
GHS-POP 五历元）**全部匿名可达并已取回**，聚合到 R7 冻结的 65×65 / 0.25° 网格后
用 `contracts.fresh_outputs` 独占发布（含 `BUILD_COMPLETE.json` 与 `static_identity`）；
LandScan 因**需注册**如实丢层（R-008，未用任何层顶替）。抽样格的**独立再派生**与聚合结果
在 float32 存储精度内一致（最大相对偏差 5.15e-08）。

## 1. 相位 1：只读端点核对（第一道硬闸门）

访问日期 **2026-10-27**（`endpoint_report.json` 的 `accessed_utc` = `2026-10-27T04:02:21+00:00`）。
逐层逐 tile 做 `HEAD`（经 `http_public.open_public`，SSRF + 重定向校验 + timeout）。

| 层 | 源 / 版本 | 分辨率 | 许可 | tile 数 | 可达 | 逐层 content-length 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| `worldcover_v100_2020` | ESA WorldCover v100(2020) | 10 m | CC-BY-4.0 | 49 | 48 | 4,110,718,311 B |
| `worldcover_v200_2021` | ESA WorldCover v200(2021) | 10 m | CC-BY-4.0 | 49 | 48 | 4,056,204,735 B |
| `ghsl_built_s_2000/2005/2010/2015/2020` | GHSL GHS-BUILT-S R2023A `4326_3ss` | 3 arcsec ≈ 100 m | CC BY 4.0 | 9 每历元 | 9 | 210.9 / 228.5 / 241.9 / 261.7 / 276.0 MB |
| `ghsl_built_h_2018` | GHSL GHS-BUILT-H R2023A **ANBH** `4326_3ss` | 3 arcsec ≈ 100 m | CC BY 4.0 | 9 | 9 | 458.1 MB |
| `ghsl_pop_2000/2005/2010/2015/2020` | GHSL GHS-POP R2023A `4326_3ss`（**counts**） | 3 arcsec ≈ 100 m | CC BY 4.0 | 9 每历元 | 9 | 800.2 / 859.6 / 914.2 / 1005.9 / 1060.5 MB |

- 许可页：`https://registry.opendata.aws/esa-worldcover/`（HTTP 200）、
  GHSL `copyright.txt`（HTTP 200，正文写明 "licensed under the Creative Commons
  Attribution 4.0 International (CC BY 4.0) licence"）、
  `https://creativecommons.org/licenses/by/4.0/`（04:02 的 HEAD 记 **403**——瞬时；
  06:35 直接 GET 复测 **200**，两次都如实记下）。
- **闸门项 ①（GHSL 分辨率冲突）已解**：目录列表实测 GHSL R2023A 对同一产品**同时发布**
  3 arcsec（≈100 m）与 30 arcsec（≈1 km）两种；本轮取 `4326_3ss`（≈100 m）。检索里
  "100 m vs 1 km" 的冲突是"两个都发布"，不是二选一。
- **闸门项 ②（Copernicus DEM 的 AWS 镜像是否匿名）已实测**：
  `https://copernicus-dem-30m.s3.amazonaws.com/.../Copernicus_DSM_COG_10_N39_00_E114_00_DEM.tif`
  → HTTP 200、`image/tiff`、43,009,038 B，**匿名可达**。但 DEM **不是**决策 0063 的具名层，
  本轮**不获取**；只作为 finding 记录，若将来要用需另行提出。

### 丢层（`failed-no-fallback`，R-008，未合成顶替）

| 层 | 原因 | 证据 |
| --- | --- | --- |
| LandScan Global | **需申请账号**，非匿名 | `https://landscan.ornl.gov` 站点可达（HTTP 200，HTML），但数据目录在注册墙后；决策 0063 把 LandScan 列为**可选**并禁止账号门源 ⇒ 丢层 |
| Copernicus DEM GLO-30 | 不在具名清单 | 见上；AWS 镜像匿名可达，但本轮不授权该层 |

### 一个缺 tile 的处理（如实记，不是丢层）

`N27E123`（3°×3°，27–30°N / 123–126°E，**东海全海**）在 ESA WorldCover 桶里
**不存在**（v100/v200 两年都 404；桶列表实测 `N27E120`、`N27E126` 存在，`N27E123` 缺）。
它是产品**按设计不发布**的全海 tile，不是网络不可达。处理方式：

- 记为**逐 tile 的 `failed-no-fallback`**（`absent_tiles`），**不**据此丢掉整个 WorldCover 层
  （48/49 tile 可达，覆盖 ROI 的全部陆地）；
- 受影响的目标格用 `land_cover_coverage_frac_2020/2021` 数组**显式暴露**（13 个格，
  覆盖率 0.5——即 col 64、rows 52–64 的东半格）；**没有**为缺口合成任何值。

## 2. 相位 2：冻结协议（R-006）

`protocol.json` 冻结于**任何下载之前**：ROI 精确格界、目标网格、逐层
source/version/licence/预期字节/聚合规则/输出布局、磁盘峰值与余量门、网络字节上限、
软/硬墙钟。

- **冻结时 digest**（下载 receipt 记录的）：`protocol_sha256 = c402867022c5dee3876fb597dfb0af3de26ecf167ec367d617aec6bb83708cef`
- **当前 digest**：`8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904`
- **更正（如实记，非放宽）**：下载之后给逐层计划补了两个**记账字段** `epoch`/`year`
  （聚合器需要），因此 digest 变了。**与获取有关的字段（URL 列表、tile 数、预期字节、
  聚合规则、预算、ROI、网格）逐字未变**；两个 digest 都在 `aggregate_receipt.json` 里
  （`protocol_sha256_at_download` / `protocol_sha256`）。
- 预算：`gpu_hours: 0.0`、`planned_seconds: 21600`、`hard_cap_seconds: 86400`、
  `network_bytes_cap: 40 GiB`、`disk_headroom_bytes: 60 GiB`、`single_tile_bytes_cap: 512 MiB`。

## 3. 相位 3：下载

- **195 个文件、14,484,496,452 字节（13.49 GiB）**，与冻结的 `expected_bytes` **逐字节相等**
  （`download_receipt.json`）。
- 网络客户端**只**在 `data/download/urban_static_sources.py`（R-016 唯一网络面），
  全部经 `http_public.open_public`；**没有**复用 `worldcover_cog.py` 的 `rasterio.open(url)` 直连路径。
- 逐文件 sha256 记在 `download_receipt.json`；GHSL 的 `.zip` 保留（可复核的下载物），
  其中单个 GeoTIFF 成员解出为原生分辨率瓦片（供延后的 1 km 轮用），两个 sha256 都记。
- 实际网络字节 = 14,484,496,452 B < 40 GiB 上限；磁盘最低余量 242 GiB > 60 GiB 门。

## 4. 相位 4：聚合到 0.25°（派生统计量，**不是**观测真值）

**逐字声明**：下列字段是**对真实观测求的派生统计量**，按**目标格界**做**面积加权**粗化；
`spatial_resampling = area_weighted_coarsen`（`docs/DATA_SPEC.md` §L2）。
**不是** `interpolation`，**更不是**"把粗场插成细真值"。**不得**把聚合结果当观测真值使用。

- **面积基准**：WGS84 **等积球**（authalic，`R = 6371007.181 m`）；原生格权重
  `R² · dlon_rad · (sin(lat_top) − sin(lat_bottom))`，即把原生网格放到**圆柱等积投影**后
  按投影面积加权。权重**逐原生行**施加（不是按读块近似）。
- **对齐**：GHSL `4326_3ss` 是 3 arcsec（0.25° = 300 原生格）、WorldCover 是 1/12000°
  （0.25° = 3000 原生格），每个原生格**完整落在**唯一目标格内。
- **格界**：`26.875–43.125°N / 106.875–123.125°E`；row 0 是最北行（与天气实例的
  `latitude` 降序对齐）；lon 升序。
- **逐角色口径**：
  - 土地覆盖（WorldCover class 50/10/40/80/60）：**该格面积的类别占比**（分母是整格面积，
    不是"5 个类别之和"——这样未发布的类别与 nodata 不会抬高已发布类别的份额）；
  - 建成面（GHS-BUILT-S，单位 m²/格）：**Σ(建成 m²) ÷ 格面积 m²**；
  - 建筑高度（GHS-BUILT-H **ANBH**，单位 m）：**面积加权平均**，只对**有值**的原生格加权；
  - 人口（GHS-POP `4326_3ss`，单位 **counts**）：**Σ(人口) ÷ 格面积 km²**。
- **多历元声明**：GHS-BUILT-S / GHS-POP 是 **1975–2030 五年间隔、跨代传感器合成**，
  **不是逐年观测**；本轮取 2000/2005/2010/2015/2020，堆成 `(5, 65, 65)`，
  历元轴单独存 `built_surface_frac_epochs` / `pop_density_km2_epochs`。
- 输出：`published/fields/static_fields.npz`（36 个数组，`fields_sha256 =
  1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5`）。
- 覆盖率：`land_cover_coverage_frac_2020/2021` 最小 **0.5**，13 个格低于 1；
  `land_cover_mapped_frac_*` 平均 **0.7395**（其余是 WorldCover 未发布的类别，如草地/灌丛）。

### 发布契约（相位 6）

`data/preprocess/contracts.py` 的 `fresh_outputs`：`published/fields/`（独占）+
`published/manifest/`（`manifest.json` + `BUILD_COMPLETE.json`）。
`static_identity` 覆盖 **源文件指纹 + 聚合规范 + 输出字节**：

- `source_sha256 = 0115dc3c809e0e6019096a3933cb3cc730c84f7a65326f81598b846212b30d42`
  （195 个下载物的 url→sha256 映射的规范化 digest）
- `output_sha256 = 1e9473f04bf1fa47ac279680a4ac4b54de933fee6afde934478e73f6e2dc37e5`
- `protocol_sha256 = 8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904`
- `static_identity = 8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03`
- **将来接线时绑进 `protocol_sha256`**（**不是** `data_identity`），使"两份静态场不同 ⇒
  协议身份不同"可机械检查。

## 5. 相位 5：独立再派生核对（不复用聚合中间量）

抽样格 `(7,64)`、`(32,32)`、`(55,5)`，每格对 4 个字段**从原始瓦片重算**——
原生格归属在复算里由 tile transform 独立推出，reduction 是一条直白的加权循环，
**不调用**聚合器的任何 helper。

| 格 | 字段 | 独立重算 | 已发布 | 相对偏差 |
| --- | --- | --- | --- | --- |
| (7,64) | `lc_built_frac` | 0.284143 | 0.284143 | 5.15e-08 |
| (7,64) | `building_height_m` | 6.466724 | 6.466723 | 1.19e-08 |
| (7,64) | `built_surface_frac` | 0.115493 | 0.115493 | 2.62e-08 |
| (7,64) | `pop_density_km2` | 1378.270620 | 1378.270630 | 7.22e-09 |
| (32,32) | `lc_built_frac` | 0.112194 | 0.112194 | 2.83e-08 |
| (32,32) | `building_height_m` | 3.071560 | 3.071560 | 7.31e-10 |
| (32,32) | `built_surface_frac` | 0.049543 | 0.049543 | 3.11e-08 |
| (32,32) | `pop_density_km2` | 623.149491 | 623.149475 | 2.50e-08 |
| (55,5) | `lc_built_frac` | 0.010577 | 0.010577 | 3.78e-08 |
| (55,5) | `building_height_m` | 3.901780 | 3.901780 | 2.83e-08 |
| (55,5) | `built_surface_frac` | 0.002632 | 0.002632 | 4.02e-08 |
| (55,5) | `pop_density_km2` | 244.683597 | 244.683594 | 1.45e-08 |

**最大相对偏差 5.15e-08**，判据 `<= 1e-5`（该容差就是 float32 **存储**精度，
不是给算法差异留的余地）。`recheck_receipt.json` 的 `passed = true`。

## 6. 口径与成本（实测，不是估计）

| 项 | 实测 |
| --- | --- |
| GPU-h | **0.0** |
| 网络 | 14,484,496,452 B（13.49 GiB），195 个文件 |
| 磁盘（本轮产物） | `native/` 20 GiB + `published/` 0.7 MB；/data 余量从 269 GiB 降到 242 GiB |
| 墙钟 | 端点核对完成 04:02:21Z → 发布 06:31:18Z；其中下载 ≈ 69 min、聚合 ≈ 55 min、再派生+发布 ≈ 2 min |
| 时长口径 | 名义 planned 21,600 s / hard 86,400 s；**实测总墙钟 ≈ 9,500 s**（在 planned 内，无 overrun） |

## 7. 缺口与限制（如实列出）

- **LandScan 未取**：需注册，`failed-no-fallback`，**未**用别的层顶替（人口交叉核对因此缺失）。
- **地形（DEM）未取**：不在具名清单；AWS 匿名镜像已实测可达，但本轮不授权该层 ⇒
  2 m 温度的静态地形协变量**缺失**。
- **WorldCover 一个全海 tile 不存在**：受影响 13 个格的东半格无数据，由
  `land_cover_coverage_frac_*` 暴露；未合成。
- **土地覆盖只发布 5 个类别**（built/tree/crop/water/bare，`docs/DATA_SPEC.md` §L2 的契约），
  其余类别（草地/灌丛/雪/湿地/红树/苔藓）未发布，因此 5 个占比之和不等于 1
  （`land_cover_mapped_frac_*` 平均 0.7395）。
- **多历元不是逐年观测**（见 §4）。
- **0.25° 偏粗**：建成区在格内是"占比"，不含城市内结构；原生分辨率瓦片已保留供延后的 1 km 轮。
- **协议 digest 在下载后因记账字段变更**（见 §2），两个 digest 都已登记。
- **许可页一次瞬时 403**（见 §1），已复测 200。
- **`scientific_claim: false`**：这是数据准备读数，不是科学结论；本轮 0 GPU-h，
  2023 test 未读，r 仍为 0。

## 8. 下一动作

**接线是下一轮**：把 `static_identity` 绑进 `protocol_sha256`（**不是** `data_identity`），
并设计时不变条件通路与城市感知/物理机制（需自己的 ADR 与预声明出口）。
**本轮不改 `model/**`、不重建 store、不加 state 通道。**

第一条命令：`.venv/bin/python tools/r7_round.py state`

## 9. 待办：`campaign-state` 指针（未做，原因如实记）

`docs/goals/main-model-climatology-campaign.md` 的 `<!-- campaign-state: ... -->` 里
`current_round_goal` **本轮未改**：R1（`outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/`）
的 workflow **仍在跑**，而 `tools/r7_round.py` 的每个子命令都从
`campaign-state.current_round_goal` 解析当前轮；此刻改指针会让 R1 的 seed 42/43
`stage` 与登记 `round-guard`/`ledger-plan` 解析到本长文（无 round-spec 块）而**失败**。
因此指针更新**推迟到 R1 的 workflow 退出之后**，命令：
把该行的 `current_round_goal` 改成 `docs/goals/s3-urban-static-covariates.md`。
