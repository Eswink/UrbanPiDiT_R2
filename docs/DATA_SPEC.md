# Data Acquisition / Contract Spec

## 第一阶段 ROI

- Coarse 京津冀：113–120°E, 38–42°N；
- 北京 1 km 广域缓存：115.2–117.8°E, 39.2–41.2°N；
- 标准 Urban tile：训练时从广域缓存裁 128 km（1 km）或等物理范围的 tile。

## 时间

优先共同窗口：2018-01-01 至 2021-08-31。成功获得 SMBFD/HRCLDAS 后再扩展。

## L0 Coarse

ERA5：surface + z/t/q(or r)/u/v at 850/700/500/250 hPa；建议约 25–35 channels。CMFD V2.0 可作为中国地表 forcing / upstream robustness source。

## L1 动态 target

优先 SMBFD / HRCLDAS ~1 km hourly。若暂不可得，不得把 ERA5 bicubic 到 1 km 当真实 target；仅允许做 pipeline smoke。

## L2 城市证据

WorldCover 10m、Copernicus DEM 30m、LCZ 100m、CNBH 10m、GlobalBuildingAtlas、人口、ECOSTRESS LST。所有 static layer 统一投影到 metric CRS 后再聚合至 urban tile。

> **上面这段是 V6 时代的口径**（ROI 见本页 §第一阶段 ROI 的京津冀盒 + "urban tile"）。
> **2026-10-27 起，本轮（决策 0063）的 L2 契约以下面的表格为准**：ROI 锚到 **R7 冻结 ROI**
> （27.0–43.0°N / 107.0–123.0°E），目标网格 **65×65 / 0.25°**，且**只做匿名可下载的层**。
> 上面那一行的层名保留为历史口径；未在下表出现的层（Copernicus DEM、LCZ、CNBH、
> GlobalBuildingAtlas、ECOSTRESS LST）本轮**不获取**，原因逐条见下表脚注。

### L2 契约（决策 0063，2026-10-27）

**ROI 与网格（锚到 R7 冻结实例，不是本页 §第一阶段 ROI）**

- ROI：`27.0–43.0°N / 107.0–123.0°E`，来源 `data/download/read_plan_frozen.py:34-36`。
- 格心：lat `43.0 → 27.0` 步长 `−0.25`（降序，65 个）；lon `107.0 → 123.0` 步长 `+0.25`（升序，65 个）。
- **格界**：`26.875–43.125°N / 106.875–123.125°E`（每格 ±0.125°）。聚合按**格界**而非格心。
- 形状：`[65, 65]`，`float32`，与天气实例的 `latitude`/`longitude` 逐点对齐。

**层清单（可缩小不可扩大）**

| 变量名 | 源 / 版本 | 原生分辨率 | 许可 | 单位 | 定义 |
| --- | --- | --- | --- | --- | --- |
| `lc_built_frac` | ESA WorldCover v200(2021) / v100(2020)，class 50 | 10 m | CC-BY-4.0 | 1（0–1） | 该格内 built-up 类别**面积占比** |
| `lc_tree_frac` | 同上，class 10 | 10 m | CC-BY-4.0 | 1 | 林地面积占比 |
| `lc_crop_frac` | 同上，class 40 | 10 m | CC-BY-4.0 | 1 | 耕地面积占比 |
| `lc_water_frac` | 同上，class 80 | 10 m | CC-BY-4.0 | 1 | 水体面积占比 |
| `lc_bare_frac` | 同上，class 60 | 10 m | CC-BY-4.0 | 1 | 裸地面积占比 |
| `built_surface_frac` | GHSL GHS-BUILT-S（多历元，取 2000/2005/2010/2015/2020） | ~100 m | 免费 | 1（0–1） | 该格内建成面**面积占比** |
| `building_height_m` | GHSL GHS-BUILT-H（2018） | 100 m | 免费 | m | 该格内**面积加权平均**建筑净高 |
| `pop_density_km2` | GHSL GHS-POP（多历元，同 GHS-BUILT-S） | 100 m | 免费 | persons km⁻² | 该格内人口 / 格面积 |

**聚合规则（必须逐字声明，不许含糊）**

- 原生栅格**先投影到等面积 metric CRS**，再对每格界做**面积加权**统计（类别取占比、连续量取
  面积加权均值）。
- 这是对**真实观测**求的**派生统计量**；`spatial_resampling` 记 `area_weighted_coarsen`
  （附源 CRS 与目标格界），**不是** `interpolation`，**更不是**"把粗场插成细真值"。
  证据页必须写明：聚合结果**不得**被当作观测真值使用。
- 多历元层（GHS-BUILT-S / GHS-POP）是**五年间隔、跨代传感器合成**，**不是逐年观测**；
  若用作时间变化量，必须逐字声明这一点。

**输出布局**

- 独立发布产物（决策 0063 第 3 条）：自己的排他 `outputs/` 目录，含静态场文件、
  `manifest`（记 `source_sha256`、聚合规范、许可、`limitations`）与 `BUILD_COMPLETE.json`。
- `static_identity`：sha256，覆盖**源文件指纹 + 聚合规范 + 输出字节**。将来模型接线时
  **绑进 `protocol_sha256`**（不是 `data_identity`），使"两份静态场不同 ⇒ 协议身份不同"可机械检查。
- **原生分辨率瓦片一并保留**（供延后的 1 km 轮用），不删。

**本轮不获取的层及原因**

- Copernicus DEM 30m / ALOS AW3D30 / SRTM：检索显示官方门户**需注册**；且地形不是 urban 量。
  若执行轮实测 Copernicus DEM 的 **AWS Open Data 镜像匿名可达**，可作为可选层另行提出。
- LCZ 100m / CNBH 10m / GlobalBuildingAtlas / 建筑 footprint / ECOSTRESS LST：需单独访问核对，
  且多为注册门或体量巨大；本轮不做。
- MODIS MCD12Q1（唯一带年际信号的土地覆盖）：需 NASA Earthdata 账号 ⇒ 与"不建账号"冲突。
- 任何 km 级**动态**真值：issue #9 继续 BLOCKED，见 `docs/R7_URBAN_EXTENSION_BLOCKED.md`。

## L3 Micro teacher

UMC4/12；后续争取 U3DWind 北京 subset。模拟数据用于过程预训练/微尺度专家，不直接冒充真实观测。
