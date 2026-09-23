# Real-Data E2E Smoke Test

## 1. 目标

这一阶段只回答一个工程问题：**真实外部数据能否经过下载/捕获 → provenance → 预处理 → manifest → `R2DataModule` → UrbanPiDiT-R² → forward/backward → Lightning checkpoint/test 的完整链路。**

它不用于证明 1 km 城市天气精度，也不用于证明 reasoning 科学有效性。

## 2. 本次真正进入模型的数据

### 2.1 动态气象：UCI Beijing PM2.5 meteorological observations

- Canonical DOI: `10.24432/C5JS49`
- Canonical URL: `https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data`
- Local fixture: `data/raw/real_smoke/beijing_uci_pm25_smoke.csv`
- SHA256: `277a2b00094398a9959aa8d4060b9035d581c89fbc009309097fd3bc38241129`
- 当前 fixture 仅保留 73 条连续小时记录。
- 使用字段：`TEMP`, `DEWP`, `PRES`, `Iws`。

注意：`Iws` 是原数据里的 **cumulated wind speed**，不是标准瞬时 10 m 风速。

### 2.2 静态地理：北京东城区真实行政边界

- Upstream mirror: `https://github.com/VIP233333/Coordinate-data-of-the-six-urban-districts-of-Beijing`
- Local fixture: `data/raw/real_smoke/beijing_dongcheng_boundary_smoke.geojson`
- SHA256: `ac90e8136273f7b41c4c03b673feb274bf51438b30b0f59976d0e93d43db90d6`

栅格化为 4 个通道：

1. `x_norm`：位置编码；
2. `y_norm`：位置编码；
3. `real_district_mask`：真实 GeoJSON 行政边界栅格；
4. `derived_district_edge`：由真实 mask 派生的边缘。

**行政边界不是城市形态。** 该通道只验证真实 geospatial vector → raster → model 的处理链。正式训练必须替换为 WorldCover、CNBH/建筑高度、建筑 footprint、DEM、LCZ 等。

## 3. 科学边界

当前 fixture 明确标记：

```text
scientific_training_ready = false
spatiotemporal_colocation_valid = false
```

原因：

- UCI 动态数据是北京机场附近单站小时记录，不是 ERA5/HRCLDAS/SMBFD 网格；
- 单站观测被广播为网格只用于验证 tensor/data contract；
- 东城区行政边界与机场站点并非严格共址；
- 不允许从本 smoke 的 forecast error 得出天气预测能力结论。

## 4. 时间切分

先切 **原始小时记录**，再分别构造历史窗口，避免滑窗跨 split 泄漏。

当前 fixture：

```text
train source rows: 0–50
val   source rows: 51–61
test  source rows: 62–72
```

对应构造：

```text
train: 24 samples
val:    4 samples
test:   4 samples
```

## 5. 固定归一化

为避免在 smoke fixture 上拟合统计量：

```text
TEMP -> x / 40
DEWP -> x / 40
PRES -> (x - 1000) / 50
Iws  -> x / 100
```

生产阶段应只用训练集估计标准化统计量，并持久化版本化 stats manifest。

## 6. 正式下载器

### UCI

```bash
python scripts/prepare_real_smoke.py --network
```

顺序：UCI 官方下载 → GitHub 文本镜像。两者都失败时显式退出，绝不生成 synthetic fallback。

### Google ARCO ERA5

```bash
pip install -r requirements-data.txt
python -m data.download.arco_era5 \
  --time 2021-07-01T00:00:00 \
  --hours 6 \
  --lat-min 39 --lat-max 41 \
  --lon-min 115 --lon-max 118
```

公共 Zarr：

```text
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
```

### ESA WorldCover 10 m 分类 COG

```bash
python -m data.download.worldcover_cog \
  --bbox 116.37 39.85 116.46 39.98
```

北京该 smoke ROI 位于 tile：

```text
N39E114
```

下载器通过 rasterio/GDAL HTTP range 只读取 ROI，而非整张 3°×3° COG。

### WorldCover WMS preview

```bash
python -m data.download.worldcover_smoke --size 64
```

该路径仅用于验证图片服务接入。**WMS RGB 不得作为 land-cover 分类训练真值。**

### 一次性网络状态审计

```bash
python scripts/try_real_downloads.py
cat audit/real_data_download_status.json
```

它会记录每个正式源的成功/失败及原始异常，不做静默 fallback。

## 7. 当前沙箱的真实结果

当前执行环境无法从容器解析外部 DNS；同时 ARCO 下载所需 `zarr/gcsfs` 未预装。因此 UCI 网络下载、WorldCover COG/WMS、ARCO Zarr 均被明确记录为 blocked/failed。

为了不伪造成功，本仓库保留了通过受控外部检索实际捕获的小型 UCI 真实 fixture，并以 checksum 固定；这份 fixture 完成了本阶段 E2E smoke。

## 8. 重现命令

```bash
python scripts/prepare_real_smoke.py
python scripts/smoke_real_data.py
pytest -q
python train.py --config configs/r2_v6_real_smoke.yaml
```

期望链路：

```text
real fixture
  -> checksum/provenance
  -> raw-hour split
  -> window/package NPZ
  -> JSONL manifest
  -> ManifestNPZDataset
  -> R2 forward/backward
  -> STOP/ZOOM routing smoke
  -> Lightning fit
  -> best checkpoint restore
  -> test
```
