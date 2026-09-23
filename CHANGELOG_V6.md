# V6 Upgrade Changelog

## 结构

- 顶层 `data/`：数据契约、synthetic smoke、manifest NPZ loader、preprocess、raw/interim/processed/manifests；
- 顶层 `model/`：V6 全部模型代码；
- `legacy_v531_full/`：原上传工程完整快照；
- `training/`：V6 Lightning 与多目标 loss；
- `audit/`：三轮实际自审。

## V5.3.1 → V6 核心变化

- 固定 grid-token dense attention → patch + local window SDPA；
- dense morphology pair matrix → O(Nk) local sparse process graph；
- proxy-only conditioning → explicit anchored/free Process State Bank；
- fixed-depth transformer → shared recurrent Process Reasoner；
- uniform high-res compute → per-sample STOP/ZOOM Router；
- full-domain refinement → selected-only Urban Expert；
- diagnostics-only → trained forecast-confidence Verifier；
- 单 H×W data contract → coarse/urban heterogeneous multiscale contract；
- diffusion 主体 → optional residual diffusion scaffold。

## Real-data smoke stage

- 增加 UCI Beijing 小型真实逐小时气象 fixture 与 checksum/provenance；
- 增加真实东城区 GeoJSON 静态栅格化 smoke；
- 修复滑窗跨 split 的时间泄漏：先切 raw hours，再在 split 内构窗；
- 增加 UCI / Google ARCO ERA5 / WorldCover COG / WorldCover WMS 下载器；
- 增加网络下载状态审计，禁止 synthetic silent fallback；
- 增加真实数据 STOP/ZOOM/adaptive reasoning forward/backward smoke；
- Lightning real-data `fit -> checkpoint -> restore -> test` 通过；
- 增加三轮真实数据自审文档与最终 scorecard。
