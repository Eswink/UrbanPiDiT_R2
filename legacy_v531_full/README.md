# UrbanPiDiT-V5.3.1-MorphoProcessDiT-ReviewerEvidence

本仓库当前主版本为 **UrbanPiDiT-V5.3.1-MorphoProcessDiT**，即
**Morphology-conditioned Process Diffusion Transformer** 的审稿证据强化版本。

V5.3.1 的目标不是把城市形态代理量声明为真实物理参数，也不把反事实诊断声明为因果证明；
它把城市静态形态字段转化为 **morphology-derived process proxies**，用于条件调制、过程图传播、
微气象弱残差分支和可审计诊断输出。

V5.3.1 主要改进点：
1. **统一版本权威层**
   - `urbanpidit_version.py` 作为论文、脚本、发布包和诊断输出的单一命名来源。
2. **统一 forward diagnostics**
   - 默认 `model(...)` 仍返回预测张量；显式 `return_diagnostics=True` 时返回 `(prediction, diagnostics)`。
3. **形态过程调制证据链**
   - 暴露 process proxy、process AdaLN、城市形态控制分支、各向异性过程图、微气象分支和残差预测头的诊断键。
4. **审稿人可运行证据脚本**
   - 提供 reviewer evidence 与 morphology counterfactual 诊断入口，用于生成 JSON 证据产物。
5. **历史训练流程兼容**
   - V4 的两阶段训练、multi-lead、rollout 微调与基线评估入口继续保留。

---

## 快速开始

### 1) V5.3.1 reviewer evidence smoke check

```bash
python experiments/run_v531_reviewer_evidence_checks.py \
  --config configs/urbanpidit_v53_morpho_process.yaml \
  --output outputs/v531_reviewer_evidence.json
```

### 2) Morphology counterfactual diagnostics

```bash
python experiments/run_morphology_counterfactual_suite.py \
  --config configs/urbanpidit_v53_morpho_process.yaml \
  --output outputs/v531_morphology_counterfactual.json
```

### 3) 单阶段训练（兼容旧版）

```bash
python train.py --config configs/beijing.yaml
```

### 4) 两阶段训练（历史 V4 流程）

```bash
python train.py --config configs/beijing_v4_two_stage.yaml
```

---

## 两阶段训练要点

`train.two_stage.enabled: true` 时，`train.py` 会自动依次执行：

- **Stage-1**（示例：6h）
  - 建议只训练/评估 `lead_times=[1]`，并监控 `val/RMSE`。
- **Stage-2**（示例：6/12/18/24h）
  - 用 Stage-1 的 **best checkpoint** 初始化（默认只加载 `net.*` 权重，更稳健）。
  - 启用 `forecast.lead_weight_anneal`：从偏短 lead 的加权策略逐步过渡到更均衡的策略。

---

## 可选：Rollout 微调（适配 autoregressive 推理）

如果你们的线上/评测采用 `multi_horizon_inference: autoregressive`（1-step 逐步滚动），
可以在 Stage-2 覆写中开启：

```yaml
forecast:
  rollout_finetune:
    enabled: true
    max_steps: 4
    curriculum_epochs: 20
    loss_on: all
    teacher_forcing:
      start_ratio: 1.0
      end_ratio: 0.0
      decay_epochs: 20
```

注意：rollout 微调只要求 dataloader 提供 `batch['y']`（即配置 `forecast.lead_times`），
并不强制启用 lead-time conditioning。

---

## 评估指标（ACC）

本项目新增 **ACC（Anomaly Correlation Coefficient，异常相关系数）**，用于衡量“预报距平场”和“验证距平场”的空间型式相似度。

- 距平定义：anomaly = field − climatology（默认 climatology 为训练集 train-only 的长期均值场）
- 权重：默认使用 cos(纬度) 做面积权重（规则经纬网）
- 记录方式：在 val/test 阶段按 lead 记录 `val/ACC@6h`、`val/ACC_t2m@6h` 等，并提供 `ACC_mean_leads` 汇总

Climatology 文件会在 `MetroWeatherDataModule.prepare_data()` 阶段自动生成：`{data_root}/normalize_clim_train.npz`。

配置示例：

```yaml
metrics:
  acc:
    enabled: true
    climatology: train_mean_field
    weighting: coslat
    center: true
    log_per_channel: true
```

---

## 目录结构

```
UrbanPiDiT/
  data/               # 数据集与 DataModule
  models/             # UrbanPiDiT 网络结构
  pidit_lit.py        # LightningModule（训练/验证/测试/推理）
  train.py            # v4 训练入口（支持两阶段）
  evaluate.py         # 评估脚本
  configs/            # 配置文件
  baselines/          # 三个基线：Linear / XGBoost&RF / 3-Layer MLP
```

---

## 基线模型（按你们要求新增）

所有基线都 **直接复用 `data/loader.py`** 加载数据（含 train-only 归一化统计与 climatology 缓存），
并在 **反归一化（物理量）空间** 计算 `RMSE/MAE/ACC`，与 `pidit_lit.py` 的口径保持一致。

### 1) 极简主义基线 (The Floor): Linear Regression (Ridge/Lasso)

输入处理：把 `x_ctx` 的所有历史时刻/变量/像素展平成一个长向量；输出同理展平。

```bash
python -m baselines.linear --config configs/beijing.yaml --kind ridge --alpha 1.0
```

### 2) 强表格基线 (The Kaggle Killer): Random Forest / XGBoost

实现方式：按“像素为样本”做表格学习（每个格点一行），特征为该像素处的历史/静态通道；
可选加入 (y,x) 归一化坐标特征（默认开启）。

```bash
python -m baselines.tree --config configs/beijing.yaml --model xgb
python -m baselines.tree --config configs/beijing.yaml --model rf
```

### 3) 基础深度基线 (The DL Baseline): Simple MLP (3-Layer)

不带卷积、不带注意力、不带扩散；输入输出同 linear baseline 的展平口径。

```bash
python -m baselines.mlp --config configs/beijing.yaml --epochs 50 --hidden 1024
```

### 一键跑完三个基线

```bash
python -m baselines.run_all --config configs/beijing.yaml --out_dir outputs/baselines
```

---

## 版本说明

- V5.3.1：当前主版本，统一版本元数据、forward diagnostics、morphology counterfactual 与 reviewer evidence JSON 输出。
- V5.3：引入 morphology-derived process proxies、process-aware graph、process AdaLN 和形态控制分支。
- V4：历史兼容训练流程，包含“两阶段训练 + lead 权重退火 + 可选 rollout 微调”。
