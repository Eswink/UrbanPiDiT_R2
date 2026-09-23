# 数据预处理约定

第一阶段建议：

- L0 京津冀 coarse ROI：113–120°E, 38–42°N；
- 北京 1 km 广域缓存：115.2–117.8°E, 39.2–41.2°N；
- 共同时间窗：2018-01-01 ~ 2021-08-31；
- 动态天气主 target：优先 SMBFD / HRCLDAS ~1 km hourly；
- Coarse context：ERA5 surface + 850/700/500/250 hPa；
- 静态证据：WorldCover / DEM / LCZ / CNBH / GlobalBuildingAtlas / ECOSTRESS；
- micro teacher：UMC4/12，后续争取 U3DWind 北京 subset。

生产数据推荐保存为 Zarr；本仓库的 `ManifestNPZDataset` 只作为可测试的最小接口。
