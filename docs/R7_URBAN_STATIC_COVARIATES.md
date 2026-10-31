# R7 城市静态协变量获取与发布（S3 / 决策 0063 / 计划 0027）

> ## ⚠ 口径警告（2026-10-30，决策 0071）
>
> **本页记录的数值是「格心口径」，与 `docs/DATA_SPEC.md:37` 声明的「按格界」不符。**
> `data/preprocess/urban_static_aggregate.py` 当时按原生格心归属
> （`:114-115`、`:123-124`），而 GHSL 3 arcsec 网格与 0.25° 目标网格**半格错位**
> （目标北界在第 `-7169.500035` 行、西界在第 `8259.500481` 列），故两种规则系统性不同。
> 精确面积重叠重算的抽样差：中位 `0.139%`、最大 `1.225%`。
>
> 本页 §5 的「独立再派生核对」**不构成语义正确性证据**：`r7_urban_static_round.py:433-436`
> 与聚合共用格心规则，`5.15e-08` 只证明两条路径互相一致。
>
> **本页数值不得用于科学或下游用途**，须以决策 0071 的格界重派生结果取代。
> 本页作为历史记录保留，不删除。
>
> 另：`frozen_before_any_download` 是**硬编码 `True`**（`scripts/r7_urban_static_round.py:210`），
> 属声明而非测量。该轮的真实时序已由决策 0071 §A 的摘要重建法独立证明
> （`c4028670…` 可精确重建，协议确实先于下载冻结）。

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
source/version/licence/预期字节/聚合规则/输出布局、磁盘**余量门**、网络字节上限、
软/硬墙钟。

- **冻结时 digest**（下载 receipt 记录的）：`protocol_sha256 = c402867022c5dee3876fb597dfb0af3de26ecf167ec367d617aec6bb83708cef`
- **当前 digest**：`8cc3c673aa98d69fc9baaa8dd10d539233c738118ceeaeef63851e40a8e54904`
- **更正（如实记，非放宽）**：下载之后给逐层计划补了两个**记账字段** `epoch`/`year`
  （聚合器需要），因此 digest 变了。**与获取有关的字段（URL 列表、tile 数、预期字节、
  聚合规则、预算、ROI、网格）逐字未变**；两个 digest 都在 `aggregate_receipt.json` 里
  （`protocol_sha256_at_download` / `protocol_sha256`）。
- 预算：`gpu_hours: 0.0`、`planned_seconds: 21600`、`hard_cap_seconds: 86400`、
  `network_bytes_cap: 40 GiB`、`disk_headroom_bytes: 60 GiB`、`single_tile_bytes_cap: 512 MiB`。
- **磁盘峰值未进冻结协议（如实记）**：phase ② 要求的「磁盘峰值与余量门」里，`protocol.json`
  冻结的只有**余量门** `disk_headroom_bytes`；**峰值只有实测值**（§8：`native/` 20 GiB +
  `published/` 0.7 MB，`/data` 余量 269 → 242 GiB），不是冻结字段。本轮**不**追改已发布的
  `protocol.json`（改它会改 `protocol_sha256` 与 `static_identity`、断身份链）；
  `estimated_peak_bytes` 记在下一轮协议里，见决策 0064。
- **预算更正**：按首次实测（整轮 9550 s、网络 13.49 GiB、磁盘消耗 27 GiB，三者均未触门）
  核对名义刻度 ⇒ planned / hard / 网络 / 磁盘预算**维持不变**，见决策 0064。

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
- **CI 不装 rasterio**：CI 安装的依赖集是 `requirements-r7-data.txt`，其中**没有** rasterio
  （只有 `requirements-data.txt` 列它，而按仓库既有约定 CI 不装该文件）。因此
  `tests/test_r7_urban_static_round.py` 里 3 个**需要栅格**的用例在 CI 上 **`skip`**——
  **skip 不算通过**：CI 实跑 **8 passed / 3 skipped**（无失败），聚合器与驱动改为
  **惰性** `import rasterio`，所以 CI 仍能导入并跑纯几何/失败语义用例。
  聚合代码本身的覆盖来自**本机** 11 passed，以及**真实轮**的独立再派生核对
  （§5，最大相对偏差 5.15e-08）。
- **`scientific_claim: false`**：这是数据准备读数，不是科学结论；本轮 0 GPU-h，
  2023 test 未读，r 仍为 0。

## 8. 下一动作

**接线是下一轮**：把 `static_identity` 绑进 `protocol_sha256`（**不是** `data_identity`），
并设计时不变条件通路与城市感知/物理机制（需自己的 ADR 与预声明出口）。
**本轮不改 `model/**`、不重建 store、不加 state 通道。**

第一条命令：`.venv/bin/python tools/r7_round.py state`

## 9. `campaign-state` 指针：已按序完成（含一次已被后续轮次超越的推迟）

**本轮执行期间未改指针（当时的正确做法）**：R1
（`outputs/r7_s3_fy_year_coverage_matched_20261026_attempt01/`）的 workflow 仍在跑，而
`tools/r7_round.py` 的每个子命令都从 `campaign-state.current_round_goal` 解析当前轮；
此刻改指针会让 R1 的 seed 42/43 `stage` 与登记 `round-guard`/`ledger-plan` 解析到本长文
（无 round-spec 块）而失败。因此指针更新**推迟到 R1 退出之后**。

**推迟条件已满足，且指针已按序推进两次**（实测，`git log -S` 核对）：
- R1 的 workflow `dwfrun-442a17eb-ba11-4af9-9b56-6460f32a1e0b` 于 `2026-10-28T04:09:42Z`
  settled `completed`；PID 777767/777788 已不存在。其末个 seed 失败（`seed43` exit 1）
  是 R1 自己的科学读数（已登记为 mixed/not-candidate），与本轮无关。
- 提交 `5e4adf9` 把 `current_round_goal` 指向本长文
  `docs/goals/s3-urban-static-covariates.md`；提交 `896eabf` 起再推进到
  `docs/goals/s3-urban-static-wiring.md`（接线轮，决策 0069）。
- 当前 `30dc58a` 的指针：`current_round_goal = docs/goals/s3-urban-static-wiring.md`、
  `previous_round_goal = docs/goals/s3-urban-static-covariates.md`、
  `previous_round_evidence = docs/R7_URBAN_STATIC_COVARIATES.md`。
  **此刻不得把它改回本长文**——那会回退一个已登记的轮次。

登记后校验器读数（实测）：`tools/r7_round.py verify` → `passed: true`，三条
（`check_conventions.py` 37 条阻断规则 `failing=0`、`check_campaign_state.py`
`failures=0 notes=8`、`verify_r7_evidence_index.py` `PASS ... 94 records`）退出码均为 0。

## 10. 编排路线：`dryRun: false` **未执行**，六相位由驱动直跑完成

目标长文要求「用 workflow 编排」，而本条要求**只完成到编译侧**：

- **已做**：加载 `dynamic-workflows` 技能后，用 `CreateWorkflow` 对草稿
  `.zcode/workflow-drafts/r7-urban-static-round.dwf.ts` 的**当前字节**编译两次通过
  （`dwfrun-1f67679b-dd53-43a2-9fa8-14b468dbfb1c`、`dwfrun-87bbd520-01a9-4ada-894f-122db5230455`），
  两次都是 `dryRun` 默认值 `true`。
- **未做（如实记）**：`CreateWorkflow` 以 `dryRun: false` 的**运行从未发生**。本轮六个相位
  （`verify-endpoints`/`freeze`/`download`/`aggregate`/`recheck`/`publish`）是**直接调用**
  `scripts/r7_urban_static_round.py` 逐个子命令跑完的，产物与登记全部为真（见 §1–§6），
  但**不是**经由 workflow 的 phase 编排路径。
- **现在不能按字面补跑的理由**：`campaign-state.current_round_goal` 已推进到
  `docs/goals/s3-urban-static-wiring.md`，`previous_round_goal` 才是本长文。非 dryRun 的
  草稿会重跑整条采集轮——那是对一个**已登记轮次**的重复推进（与 campaign 的
  「不重复已登记轮次」相冲突），且草稿的 `commit-guard`/`verify` 依赖的当前指针不是本长文。
  因此该缺口**保留为未做**，不以回退指针来补。
- **这条缺口对产物无影响**：产物、身份、digest、重算比对、独立发布与四件套登记都不经过
  workflow 编排；受影响的是「本轮是否由一个可恢复的编排器驱动」这一工程属性。
  证据页与账本行都已标 `scientific_claim: false`。
- **若要补**：需要一个自己的 ADR 与预声明出口，在**指针重新指向本长文**的独立 attempt
  里跑 `dryRun: false`，并接受它会重跑六相位（产物须落在新的排他 `outputs/` 路径）。

## 11. 格界重派生（`attempt03`）与 recheck 的量纲缺陷（决策 0072）

### 11.1 `attempt03` 是什么

按决策 0071 第 1 条以**格界面积重叠**重派生，用 workflow 非 dryRun 驱动
（`dwfrun-47a053b8`，六相位全跑、忠实性闸门 F1–F4 首次全绿）。产物落全新排他路径
`outputs/r7_urban_static_covariates_attempt03/`，`static_identity = 6f65f88a…`。
下载 165 文件 / `13,727,171,047 B`；协议 10 层（**不是** attempt01 的 13 层，见 11.3）。

### 11.2 recheck 报了 `passed: false`，那是**真缺陷**

`attempt03/recheck_receipt.json`：`passed: false`、`max_relative_delta = 913872.595`、
容差 `1e-05`。12 项比较里两个字段通过、两个失败：

| 字段 | `relative_delta`（修复前） |
| --- | --- |
| `built_surface_frac` | `1.8e-08`–`2.4e-08` ✅ |
| `pop_density_km2` | `8.9e-10`–`6.7e-09` ✅ |
| `lc_built_frac` | `7.9e+05`–`9.1e+05` ❌ |
| `building_height_m` | `2.7e-05`–`4.3e-04` ❌ |

根因是 `data/preprocess/urban_static_recheck.py` 的**分子与分母不同量纲**：
`share = lat_fraction * lon_fraction` 是"每像素落入目标框的比例"（无量纲，整格求和
实测 `9,000,000` = 3000×3000 个像素），分母却写成 `sin_band × native_cols = 9.8416`。
`9.0e6 / 9.8416 = 914,634`，与实测 `913,872` 同量级。`building_height` 同源——
分子分母都用无量纲 `share`，得到未按面积加权的均值。

### 11.3 先证明**聚合器是对的**，再修 recheck

在改任何代码之前，用**完全独立的暴力实现**（逐瓦片、逐原生像素、绝对球面面积 `m²`）
复算同一格：

- 已发布 `lc_built_frac[7,64] = 0.28414282`
- 独立绝对面积复算 `0.284142837`
- 相对差 `5.15e-08`

⇒ 已发布数值**正确**。缺陷只在 recheck 一侧。修好后在**真实 `attempt03` 瓦片**上重跑
12 项比较：

```
max_relative_delta = 5.154e-08   (容差 1e-05)   ← 修复前 913872.595
```

四个 role 的正确权重**不同**，修法是分开处理（对 `built_surface`/`population` 乘绝对面积
会双计，实测偏差跳到 `6.5e+03`–`7.5e+03`）：`land_cover`/`building_height` 用绝对球面重叠面积
`m²`，`built_surface`/`population` 用重叠比例；`land_cover` 的分母统一为 `cell_area_m2`。

### 11.4 三条 fail-closed 守卫（防复发）

1. **`publish()` 对 `recheck.passed is not True` 直接 `SystemExit`**。此前只把结果写进
   manifest，于是 `attempt03` 在 `9.1e+05` 的状态下照常发布了 `static_identity`。
2. **被丢弃的具名层必须有非空 `reason`**（`freeze`/`_publish` 入口断言）。此前
   `verify_endpoints()` 构造 dropped 记录时**不含 `reason` 键**，`attempt03` 实测三条
   `reason: null`：`ghsl_built_s_2000/2005/2010`（13 层 → 10 层，无任何解释）。
   **这是未解决的数据问题**：这三层在 attempt01/02 可达，在 attempt03 不可达，
   原因尚未查明（本轮只保证它必须被写下来，没有修可达性）。
3. **新增 `tests/test_r7_urban_static_recheck.py`（14 用例）**。该模块此前**没有任何测试**，
   缺陷是靠一次真实运行的 `passed: false` 暴露的。反证已实测：把修复前的模块放回去跑
   新测试，**4 个用例失败**；换回修复后 14 个全过。

### 11.5 `attempt03` 能否用于科学：**不能**

字段经独立复算是正确的，但**当时的闸门没拦住**。这两件事不能互相抵消——一个当时没通过
自身校验、事后因"校验器有病"被追认的产物，可信度来自事后复算，不来自当时的闸门。
按新守卫，这样的 attempt 不会再被发布。**重新派生一个 recheck 通过的 attempt 是下一动作**，
在它落地前 `attempt03` 记 `scientific_claim: false`。

`attempt01`/`attempt02` 的格心口径产物与 `attempt03` 的格界口径产物**都在**，
两者数值不同（中位 `0.139%`、最大 `1.225%`），不得混用。
