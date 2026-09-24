---
name: real-data-acquisition
description: 当要引入新的真实数据源或重建派生数据集时使用；覆盖只读 preflight、人审后显式授权写入、发布契约与审计记录。
---

# 获取与发布真实数据

## 何时使用

- 你拿到了一份新的 ERA5/世界覆盖/站点数据，想引入项目；
- 你要重建一个派生数据集（新的 zarr 缓存 + manifest）；
- 你要新增一个下载器或数据源探测脚本。

**不适用**：
- 只是想读一下数据看看（那就用 preflight 的只读模式，不要走写流程）；
- 用合成数据做工程 smoke（合成数据不经过本流程，但必须标注不是天气真值）。

## 前置条件

1. 源文件已在本地（区域子集，不是全球原始档）。云端下载需要单独授权，且要走
   `data/download/` 里已有的带预算的下载器。
2. 已知源的来源、时间跨度、区域、变量清单、单位。
3. 已明确输出目标路径（必须是**新的**、非嵌套的路径）。

## 步骤

1. **只读 preflight。**
   ```bash
   python prepare_r7_local.py --source /path/to/regional-era5.nc --config configs/r7_era5_local.yaml
   ```
   它会读取坐标/schema 加一个采样帧并打印报告，**不创建缓存、不下载、不写任何文件**。
   - 期望：输出 `source_preflight.json` 内容，含 `scientific_training_certified: False`
     与文件的 fingerprint。
2. **人审报告。** 检查：变量与单位是否符合 17 通道计划（K、Pa、m s-1、kg kg-1、m2 s-2）；
   坐标是否为 hPa/millibar；split 年份是否与实际可用年份一致。
   - 单位不对 ⇒ **停止**。项目**不会**静默把摄氏转开尔文或把米转位势高度
     （`docs/R7_REAL_DATA_PREFLIGHT.md:16-18`）。
3. **显式授权写入。** 只有在看过报告之后才执行：
   ```bash
   python prepare_r7_local.py --source /path/to/regional-era5.nc --config configs/r7_era5_local.yaml \
     --write --store /new/path/era5.zarr --manifests /new/path/manifests --max-raw-gib 2
   ```
   `--write` 与 `--max-raw-gib` 都是必需的；没有它们不会写。
4. **确认发布契约。** 写入成功后目标 manifest 目录应含 `BUILD_COMPLETE.json`；
   zarr 应含 `schema_version=1`、`time_unit=ns`、`build_complete=true`。
   - 样例：`data/preprocess/contracts.py:10-50`、`data/preprocess/r7_era5_zarr.py:147`。
5. **记录审计。** 若这是通过下载器获取的，确认 receipt 含：源身份、快照 id、
   `decoded_budget_bytes`、`decoded_charged_bytes`、每变量 `payload_sha256`、
   `network_body_bytes`、许可与 DOI。
   - 样例：`data/download/earthmover_pilot.py:236-256`。

## 检查点

- **步骤 1 后**：报告里若出现 `scope='file-stat-only'`（文件 >64MiB 时只做 stat，
  不给内容 hash），**要意识到这个 fingerprint 弱于完整 SHA256**。
  对科学运行，应设法取得完整 hash 或信任来源侧提供的 checksum。
- **步骤 3 前**：确认目标路径不存在。已存在的路径会被 `fresh_outputs` 拒绝——
  这是设计行为，**不要**通过删除旧目录来"让重试成功"（`docs/R7_DATA_PUBLICATION.md:6`）。
  正确做法是选一个新目标。
- **步骤 3 后**：写入过程会做完整有限性检查；若失败，部分产物会保留供检视，
  但**不会**被标记完成，读取方也打不开它。失败后选新目标重试。
- **步骤 4 后**：缺 `BUILD_COMPLETE.json` ⇒ 这不是一个可用的数据集，不要拿去训练。
- **任何时候**：输入文件**永不被修改**。若你的流程改写了输入，那是 bug 不是配置问题。

## 常见失败

- **把全球数据拉下来**：本流程刻意只授权区域子集。全球下载需要单独的边界协议
  （源/区域/变量/时间跨度/存储目标/最大下载量/时间与金钱上限）。
- **字节预算搞错语义**：`decoded_budget_bytes` 是**解码后**的读取量，
  不是 HTTP 流量、也不是峰值内存。`docs/R7_TASK_QUEUE.md:42-43` 明确记录过这个区分。
- **把 stat-only fingerprint 当成完整校验**：见上。
- **合成回退**：下载失败必须留下 `status='failed-no-fallback'`、`synthetic_fallback=False`
  的记录并重新抛出（R-008）。**禁止**静默改用合成数据继续。
- **忽略单位**：项目不会猜测单位；缺失单位会被写成 `'unknown'` 而不是猜一个。

## 完成判据

- preflight 报告已生成并被人审过；
- 新目标含 `BUILD_COMPLETE.json`（与 zarr 的 `build_complete=true`）；
- 若是下载获得：receipt 含完整审计字段且 `synthetic_fallback=False`；
- 输入文件未被修改；
- 训练用的数据集记录里有 `source_sha256`、`split_policy`、`normalization`、`limitations`。

## 明确不覆盖

- 不认证源本身的科学性：成功的 preflight **不**证明数据适合科学训练
  （`docs/R7_REAL_DATA_PREFLIGHT.md:27-30`）；
- 不覆盖 ERA5 的"原生 0.25°"命名争议——文档已明确该措辞指"模型直接在提供的
  0.25° 分布网格上学习"，不指 ERA5 原始分析是原生 0.25° 数值模拟；
- 不覆盖城市形态（WorldCover/建筑/DEM）数据的获取；
- 不覆盖精细分辨率动态真值（那是被阻塞的研究关卡 #9，需要真实共址数据）。
