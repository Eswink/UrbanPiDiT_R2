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

## L3 Micro teacher

UMC4/12；后续争取 U3DWind 北京 subset。模拟数据用于过程预训练/微尺度专家，不直接冒充真实观测。
