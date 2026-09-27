# UrbanPiDiT-R² V6

**Reasoning-Driven Adaptive-Resolution Urban Weather Forecasting**

> **当前活跃主线不是本节描述的 V6，而是 R7（`r7/weather-reasoning` 分支）。**
> 接手开发前请先读 `AGENTS.md`（工程契约与任务路由）与
> `docs/R7_TASK_QUEUE.md`（研究关卡与待办状态）。
> R7 的入口点、数据与产物约定见 `docs/rules/`，多步骤流程见 `.agents/skills/`
> （索引在 `docs/skills/README.md`）。
> 下面各节记录的是已交付的 V6 MVP，仍然有效但不再是开发重点。

## R7 分栏导航

按你**想要的结论强度**选择入口；三栏的证据等级不同，不要混用它们的数字（#67）。

| 栏 | 是什么 | 入口 | 能说什么 |
| --- | --- | --- | --- |
| **① Smoke / 工程** | 小规模 CPU 或 GPU 检查：可学习性、内存、DDP 契约、数据契约、下载通道 | `docs/R7_B0_LEARNABILITY.md`、`docs/R7_GPU_BRINGUP.md`、`docs/R7_DDP_K_CONTRACT.md`、`docs/R7_STREAMED_TRAINING.md`、`docs/R7_SECURITY_SCAN_TRIAGE.md` | 「代码按设计工作」。**不**是科学结论 |
| **② Negative study / 负结果集** | 已完成的对照实验，多数结论为负或 mixed，全部按预注册判据报告 | `docs/R7_B1_D1_COMPARISON.md`、`docs/R7_B2_MULTISEED.md`、`docs/R7_B3_SCALE.md`、`docs/R7_65_PREDIAGNOSTIC.md`、`docs/R7_65_C1_PROCESS_SUPERVISION.md`、`docs/R7_65_C2_C3_FEEDBACK_AND_DEPTH.md`、`docs/R7_66_GATE_AUDIT.md`、`docs/R7_ROADMAP_GATE_STATUS.md` | 「在这些数据与预算下，哪些方法**没有**被证明有效」 |
| **③ Publication benchmark / 封存基准** | 期刊评估的冻结协议与封存 test 报告 | 协议：`docs/R7_67_PUBLICATION_PROTOCOL.md`（**已冻结**）；报告：`docs/R7_67_SEALED_TEST_REPORT.md`（**BLOCKED**：冻结 test 划分在 2 月，train-only 气候态桶只有 1 月，读者正确拒绝回退） | 「协议已定；test 数字**尚未产出**，且原因已查明」 |

**现状一句话**：R7 目前是**负结果为主的科学状态**——过程递归、辅助监督、草稿反馈与
自适应推理都没有建立可重复的收益；工程链（数据契约、身份校验、CI、hooks）已完整。
不要把 ① 的通过当作 ② 或 ③ 的结论。

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
