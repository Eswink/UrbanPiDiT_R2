# UrbanPiDiT-R² V6

**Reasoning-Driven Adaptive-Resolution Urban Weather Forecasting**

本仓库是 `UrbanPiDiT-V5.3.1-MorphoProcessDiT` 的 V6 技术路线升级版。V6 不再把 0.25° 网格上的城市形态代理量与天气动力学强行放在同一固定分辨率中，而是将模型拆为：

1. **Coarse Atmospheric Context**：大尺度/中尺度背景天气；
2. **Process-Space Reasoning**：可重复迭代、参数共享的气象过程状态；
3. **Compute-Aware Router**：决定 STOP/ZOOM，并输出区域重要性；
4. **Urban Expert**：只对需要精细化的城市 tile 进行高分辨率残差预测；
5. **Verifier**：过程语义、跨尺度一致性与置信度验证；
6. **Residual Diffusion（可选）**：只对 unresolved residual 做概率细化。

## 设计硬约束

- **4090D 24GB first**：核心模型必须能在单张 24GB 消费级 GPU 上训练；
- 不使用逐像素全局 attention；
- 不构造 `N×N` morphology distance/adjacency matrix；
- 训练时默认 BF16（V100 使用 FP16）；
- 大尺度 teacher/context 可离线缓存，不在主训练图反向传播；
- 第一阶段以 1 km 左右动态天气 target + 10–100 m 城市形态证据为主，不伪造 250 m 动态天气真值。

## 目录

```text
UrbanPiDiT_R2_V6/
├── data/                  # 数据、数据契约、预处理
├── model/                 # V6 模型代码
├── training/              # 训练/损失/Lightning
├── configs/               # 4090D 与 smoke 配置
├── tests/                 # 单元与端到端 smoke 测试
├── scripts/               # smoke / 参数与显存估算
├── docs/                  # 架构、数据与升级说明
└── audit/                 # 三轮自审记录
```

## 快速 smoke test

```bash
python scripts/smoke_forward.py --config configs/r2_v6_smoke.yaml
pytest -q
```

## 合成数据训练 smoke

```bash
python train.py --config configs/r2_v6_smoke.yaml
```

## 正式数据接口

正式数据采用 manifest + NPZ/Zarr 适配思路。最小 batch contract：

```python
{
  "coarse_history":  [B, Tc, Cc, Hc, Wc],
  "urban_history":   [B, Tu, Cu, Hu, Wu],
  "urban_static":    [B, Cs, Hu, Wu],
  "urban_baseline":  [B, Cout, Hu, Wu],
  "urban_target":    [B, Cout, Hu, Wu],
  "process_targets": [B, P_anchored],       # 可选
  "zoom_target":     [B],                   # 可选
}
```

`urban_baseline` 是 coarse/regional field 投影到 urban tile 的低成本基线；V6 学习其 residual。Router 在推理时可直接 STOP 并返回 baseline，或调用 Urban Expert 做 ZOOM。

## 版本定位

- V5.3.1：Morphology-conditioned Process Diffusion Transformer；
- V6-R²：**forecast-native Process-Space Reasoning + Adaptive Resolution + Verifier-guided test-time computation**。

详见 `docs/ARCHITECTURE.md` 与 `docs/UPGRADE_FROM_V531.md`。

## 真实数据 E2E smoke

V6 现在包含一个严格标注为 **非科学训练用途** 的真实数据 smoke 链：

```bash
python scripts/prepare_real_smoke.py
python scripts/smoke_real_data.py
python train.py --config configs/r2_v6_real_smoke.yaml
```

同时提供正式数据源下载器：

```bash
pip install -r requirements-data.txt
python scripts/try_real_downloads.py
python -m data.download.arco_era5
python -m data.download.worldcover_cog
```

详见 `docs/REAL_DATA_SMOKE.md` 与 `audit/real_data_final_scorecard.md`。
