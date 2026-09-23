# 交付说明

本交付是 UrbanPiDiT-R² V6 的**工程/研究 MVP**，已完成架构重构、端到端合成训练链路和三轮自审。它不是已经在真实 1 km 数据上完成训练的最终天气模型。

## 推荐实际执行顺序

1. 在本地 4090D 运行 `scripts/benchmark_gpu.py`，确认真实 VRAM；
2. 获取/整理 ERA5 + SMBFD/HRCLDAS + 北京静态形态数据；
3. 按 `data/schema.py` 生成 train/val/test samples；
4. 先 `force_zoom` 训练 Urban Expert；
5. 基于固定 Zoom 模型生成 Router utility / `zoom_target`；
6. 训练 Process/Router/Verifier；
7. 做 adaptive K 与 STOP/ZOOM 消融；
8. reasoning 主线稳定后再开启 residual diffusion 与 LLM explanation layer。


## Real-data smoke 阶段已完成

本阶段新增并实际执行：

1. UCI Beijing 真实逐小时动态 fixture + SHA256/provenance；
2. 北京东城区真实 GeoJSON 静态栅格化 smoke；
3. raw-hour-first 无泄漏 train/val/test 切分；
4. UCI / Google ARCO ERA5 / ESA WorldCover COG/WMS 下载器；
5. 下载失败机器可读审计，禁止 synthetic silent fallback；
6. 真实 batch forward/backward + hard STOP + hard ZOOM + adaptive reasoning；
7. Lightning `fit -> checkpoint -> restore -> test`。

当前沙箱外部 DNS 被隔离，因此正式 ERA5/WorldCover 二进制下载器已实现但未在此环境成功取回文件。该限制已写入 `audit/real_data_download_status.json`。本次实际进入模型的真实数据及其科学边界见 `docs/REAL_DATA_SMOKE.md`。
