# UrbanPiDiT-V5.3 Morpho-Process DiT 方法说明

V5.3 的核心目标是把城市形态从“静态上下文 token”升级为“可诊断的过程代理表示”，再让这些过程代理直接参与扩散去噪轨迹。整体设计不声称构建完整物理求解器，而是提供一组 micro-met-inspired、可消融、可反事实验证的形态过程调制路径。

## 1. 输入和中间表示

V5.3 使用现有 5 类城市静态字段：

- `landcover`
- `building_surface`
- `buildings`
- `building_volume`
- `population`

这些字段先进入 `MorphologyProcessProxyEncoder`，输出两类中间产物：

| 产物 | 形状 | 用途 |
|---|---|---|
| `proxy_fields` | 7 个 `[B,1,H,W]` 场 | 提供可解释过程代理，用于调制、诊断和软一致性约束 |
| `level_features` | 3 层 FPN-style 特征 | 提供空间控制分支的多尺度形态上下文 |

7 个 process proxy fields 分别是：

| proxy | 含义边界 | 主要下游路径 |
|---|---|---|
| `roughness_proxy` | 粗糙度/阻力倾向代理 | process graph、wind consistency、control residual |
| `drag_proxy` | 近地风速衰减倾向代理 | MicroMet residual、soft consistency |
| `heat_storage_proxy` | 城市热储存倾向代理 | process AdaLN、diurnal consistency |
| `impervious_proxy` | 不透水面倾向代理 | moisture consistency、residual head |
| `evap_proxy` | 蒸散/湿度调节倾向代理 | moisture consistency |
| `ventilation_block_proxy` | 通风阻塞倾向代理 | anisotropic graph、wind/tcc diagnostics |
| `anthropogenic_heat_proxy` | 人为热释放倾向代理 | diurnal modulation、residual head |

所有 proxy 只作为无量纲、可学习的过程代理，不等同于真实物理参数。

## 2. 三条主调制路径

### 2.1 Process-conditioned AdaLN

`ProcessConditionedAdaLN` 将 7 个 proxy fields 压缩成 token-level process context，并对每个 Transformer block 的表征调制做弱残差补充：

```text
static morphology
  -> process proxy fields
  -> token-level process context
  -> block-wise AdaLN shift/scale supplement
  -> diffusion denoising trajectory
```

关键约束：

- 每层有独立 `alpha_p`。
- `alpha_p` 初始为 0。
- 初始状态等价于不启用该路径。
- 诊断重点是 `alpha_p` 演化、block-wise modulation norm、关闭该路径后的误差变化。

### 2.2 Urban morphology control branch

`UrbanMorphologyControlBranch` 使用 proxy encoder 的多尺度特征生成逐层 token residual：

```text
FPN-style morphology features
  -> lightweight decoder
  -> per-block token residuals
  -> Transformer hidden state
```

关键约束：

- 每层 residual projection zero-init。
- 初始 residual 为 0。
- 输出可记录每层 residual norm，用于说明形态信息是否在深层去噪中起作用。

### 2.3 Anisotropic urban process graph

`AnisotropicUrbanProcessGraph` 将城市图从普通邻近关系升级为风向、形态、过程代理和扩散时间共同调制的过程图：

```text
spatial proximity
+ continuous morphology
+ categorical morphology
+ wind direction
+ roughness / ventilation proxy
+ diffusion timestep
  -> anisotropic adjacency
  -> process-aware token propagation
```

关键诊断：

- `graph_entropy` 随 diffusion timestep 的变化。
- wind rotation counterfactual 后邻接和预测的变化。
- high/low roughness intervention 后风场残差差异。

## 3. MicroMet token branch 和残差头

V5.3 保留 V5.2 的 physical-space MicroMet 路径，同时新增 token-space 模式：

| 模式 | 作用域 | 默认行为 |
|---|---|---|
| `pre/post` | 物理场空间 | 兼容 V5.2 |
| `interleaved/inside/mid` | 旧 interleaved 安全适配路径 | 兼容旧测试和旧配置 |
| `token/token_interleaved` | V5.3 token residual branch | 仅 `use_micromet_token_branches: true` 时启用 |

`MorphologyResidualPredictionHead` 位于输出侧，用 proxy fields 生成空间 gate，再对主预测做 zero-init 残差修正。该路径用于捕捉形态相关的局部偏差，而不是替代主干预测。

## 4. 训练损失和日志链路

`ProcessProxyConsistencyLoss` 被接入 `PhysicalConsistencyLoss` facade。它只提供低权重 soft consistency signals：

| term | 约束方向 | 梯度边界 |
|---|---|---|
| `roughness_wind_decay` | 高 roughness 区域风速不应系统性高于低 roughness 区域 | proxy detach |
| `diurnal_range_by_building` | 建筑体量代理与温度日变化幅度存在可诊断关系 | proxy detach |
| `impervious_d2m_corr` | 不透水面代理与露点/湿度响应存在可诊断关系 | proxy detach |
| `wind_tcc_spread` | 高风区域云量空间离散度应更受约束 | proxy detach |

`proxy_fields` 由模型 forward 缓存，训练、验证、测试统一通过物理一致性入口读取。若 proxy 缺失、辅助损失关闭或非主 lead 不施加辅助项，则返回 zero terms，保证旧行为兼容和日志 key 稳定。

## 5. 消融和回退

V5.3 所有新增路径都由 `ablation.use_*` 开关控制：

| 开关 | 控制范围 |
|---|---|
| `use_process_proxy_encoder` | process proxy 编码器 |
| `use_process_adaln` | block 内部 process AdaLN 调制 |
| `use_urban_control_branch` | 逐层形态控制残差 |
| `use_anisotropic_process_graph` | 风向/过程感知图传播 |
| `use_micromet_token_branches` | V5.3 token-space MicroMet branch |
| `use_morphology_residual_head` | 输出侧形态残差头 |
| `use_process_consistency_loss` | proxy consistency soft loss |
| `degrade_to_v52` | 一键关闭 V5.3 新增结构路径 |

所有新增结构路径采用 zero-init 或弱残差，初始状态近似 V5.2，便于稳定训练和审稿消融。

## 6. 审稿回应口径

推荐表述：

- morphology-derived process proxy
- micro-met-inspired residual modulation
- process-aware graph inductive bias
- soft process consistency diagnostics
- counterfactual morphology sensitivity

避免表述：

- full physics-informed model
- exact urban canopy parameter recovery
- causal proof
- complete physical solver

## 7. 最小证据包

用于 rebuttal 的最小证据建议包含：

1. `degrade_to_v52` 与 V5.2 输出/指标一致性。
2. 六个模块级消融：proxy encoder、process AdaLN、control branch、process graph、token MicroMet、residual head。
3. 形态反事实：static shuffle、landcover permutation、building volume removal、wind rotation。
4. 诊断曲线：proxy statistics、`alpha_p`、control residual norm、graph entropy、spatial gate mean。
5. proxy consistency 日志：证明该损失是低权重软诊断，不是主要性能来源。