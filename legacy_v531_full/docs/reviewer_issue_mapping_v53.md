# V5.3 Reviewer Issue Mapping

本文档把 V5.3 Morpho-Process DiT 的结构改动、审稿质疑、消融项、诊断证据和 rebuttal 表述边界放在同一张执行表里。目标不是扩大物理声称，而是把“城市形态如何进入模型并调制微气象预测过程”讲清楚、测清楚、消融清楚。

## 1. Reviewer attack surface -> V5.3 response

| Reviewer attack surface | V5.2 风险 | V5.3 mechanism | 需要展示的证据 |
|---|---|---|---|
| 只是把 DiT 用到城市天气预测，领域创新不足 | 形态字段主要作为 static tokens/context，容易被认为是通用 backbone 拼接 | `MorphologyProcessProxyEncoder` 把静态形态变成 7 个 morphology-derived process proxy fields | proxy map、proxy statistics、no_proxy_encoder 消融 |
| Cross-attention 只是弱条件注入，不能说明形态调制去噪过程 | 静态信息更多影响 context，不一定进入每层 denoising dynamics | `ProcessConditionedAdaLN` 在每个 block 内补充 process shift/scale | `alpha_p` 演化、block-wise modulation norm、no_process_adaln 消融 |
| 城市图是普通 kNN/message passing，不够城市过程感知 | 邻接主要来自空间或形态相似度 | `AnisotropicUrbanProcessGraph` 使用风向、形态、roughness proxy、ventilation proxy 和 diffusion timestep | graph entropy vs t、wind rotation counterfactual、no_process_graph 消融 |
| 缺少可解释机制，难以证明形态真的起作用 | 输出改进可能来自容量增加 | `UrbanMorphologyControlBranch` 和 `MorphologyResidualPredictionHead` 提供可记录 residual/gate | control residual norm、spatial gate map、residual norm vs proxy |
| 物理论述过强，可能被质疑不严谨 | 容易被要求证明完整 urban canopy physics | `ProcessProxyConsistencyLoss` 只作为 low-weight soft diagnostics，且 proxy detach | loss/process_proxy_total、缺失 proxy zero terms、no_proxy_loss 消融 |
| 缺少反事实验证 | 只给指标不能说明模型是否依赖形态 | 形态置换、扰动、风向旋转、roughness intervention | impact map、变量级误差变化、proxy/residual 变化 |
| 新增模块可能破坏旧模型或只是工程堆叠 | 无法证明稳定回退 | zero-init / weak residual / `degrade_to_v52` | 初始等价、V5.2 回归测试、模块关闭后行为稳定 |

## 2. Mechanism -> ablation -> diagnostic matrix

| Mechanism | Ablation key | 主要诊断 | 预期现象 |
|---|---|---|---|
| Process proxy encoder | `use_process_proxy_encoder: false` | proxy mean/std、proxy-output residual relation | 关闭后所有 process-conditioned 路径失去统一形态代理，反事实敏感性下降 |
| Process AdaLN | `use_process_adaln: false` | `process_adaln/alpha_p_*`、modulation norm | 长 lead 和形态复杂区域收益下降 |
| Urban control branch | `use_urban_control_branch: false` | `urban_control/residual_norm_*` | 局部空间修正能力下降，impact map 变弱 |
| Anisotropic process graph | `use_anisotropic_process_graph: false` | graph entropy、wind scale、block scale | 风向旋转反事实响应减弱 |
| MicroMet token branches | `use_micromet_token_branches: false` | token residual norm、branch strength | token-space micro-met residual 消失，V5.2 physical-space mode 不受影响 |
| Morphology residual head | `use_morphology_residual_head: false` | spatial gate mean/std、output residual norm | 输出侧形态局部偏差修正减弱 |
| Proxy consistency loss | `use_process_consistency_loss: false` 或 `losses.process.enabled: false` | `loss/process_proxy_total` | 主性能不应完全依赖该损失；它应主要稳定诊断方向 |
| Full rollback | `degrade_to_v52: true` | output diff、旧测试通过率 | 新增路径关闭，行为回到 V5.2-compatible |

## 3. Counterfactual suite

`experiments/run_morphology_counterfactual_suite.py` 用于构造最小形态反事实证据包。

| Counterfactual | 输入变化 | 回应的问题 | 建议输出 |
|---|---|---|---|
| `zero_static` | 静态形态置零 | 模型是否依赖城市形态 | 变量级 RMSE/MAE delta、impact map |
| `shuffle_static` | batch 内形态交换 | 性能是否来自真实空间对应关系 | delta map、proxy distribution shift |
| `permute_landcover` | landcover 类别置换 | 类别形态是否参与过程代理 | landcover-sensitive variables delta |
| `remove_building_volume` | 建筑体量置零/均值化 | 建筑形态是否影响风/温度响应 | wind/t2m residual delta |
| `remove_population` | population 置零/均值化 | 人为热代理是否有贡献 | nighttime t2m/d2m delta |
| `rotate_wind_90` | 历史风向旋转 | 图是否风向感知 | graph entropy、u10/v10 delta |
| `high_roughness_intervention` | roughness 代理增强 | 模型是否响应粗糙度增强 | wind speed decay、control residual norm |
| `low_roughness_intervention` | roughness 代理减弱 | 模型是否响应粗糙度减弱 | wind speed increase、impact map |

## 4. Reviewer hardening checks

`experiments/run_v53_reviewer_hardening_checks.py` 应输出一个诊断 JSON/CSV，至少包含：

| Category | Keys |
|---|---|
| proxy statistics | proxy mean/std/min/max/correlation |
| process AdaLN | `alpha_p`、modulation norm per block |
| control branch | residual norm per block、active block ratio |
| process graph | graph entropy、wind scale、block scale、edge sparsity |
| MicroMet token branch | branch residual norm、token residual norm |
| residual head | spatial gate mean/std、residual norm |
| loss diagnostics | process proxy total、四个 proxy term、zero fallback count |

## 5. Rebuttal wording templates

### 对“只是换 backbone”的回应

V5.3 does not only append static morphology tokens to a DiT backbone. It converts urban morphology into morphology-derived process proxy fields and injects them into the denoising trajectory through block-wise process conditioning, spatial control residuals, and an anisotropic process-aware graph.

可配中文稿：V5.3 并非只把静态城市形态作为 cross-attention token 拼到 DiT 上，而是先生成可诊断的形态过程代理，再通过 block 内调制、空间控制残差和各向异性过程图直接影响去噪轨迹。

### 对“图不够城市过程感知”的回应

The graph is no longer a purely spatial kNN graph. Its adjacency is modulated by wind direction, continuous and categorical morphology, process proxy fields, and diffusion timestep, enabling process-aware anisotropic propagation.

可配中文稿：V5.3 的城市图不再只是空间邻近图，而是由风向、连续/类别形态、过程代理和扩散时间共同调制，因此能表达各向异性的城市过程传播。

### 对“物理声称过强”的回应

We intentionally keep the formulation as micro-met-inspired soft process constraints rather than a full physical solver. Proxy fields are detached in the consistency loss and are used for diagnostics and weak regularization, not for claiming exact physical parameter recovery.

可配中文稿：我们有意把该部分限定为 micro-met-inspired 的软过程一致性，而不是完整物理求解器。proxy fields 在一致性损失中 detach，只用于弱正则和诊断，不声称恢复真实物理参数。

### 对“缺少机制验证”的回应

We provide module-level ablations and morphology counterfactuals, including static shuffling, landcover permutation, building-volume removal, wind rotation, and roughness interventions, together with proxy, graph, and residual diagnostics.

可配中文稿：V5.3 提供模块级消融与形态反事实实验，包括静态形态交换、landcover 置换、建筑体量移除、风向旋转和 roughness 干预，并同步记录 proxy、graph 和 residual 诊断项。

## 6. 最小提交材料清单

| 材料 | 文件/来源 | 用途 |
|---|---|---|
| V5.3 方法说明 | `docs/method_v53_morpho_process_dit.md` | 方法章节和 rebuttal 主叙事 |
| 审稿映射 | `docs/reviewer_issue_mapping_v53.md` | rebuttal response matrix |
| V5.3 配置 | `configs/urbanpidit_v53_morpho_process.yaml` | 复现实验设置 |
| 模块测试 | `tests/test_process_*.py`, `tests/test_urban_control_branch.py`, `tests/test_anisotropic_process_graph.py` | 证明 shape/zero-init/梯度契约 |
| 集成测试 | `tests/test_morpho_process_forward.py`, `tests/test_model_registry.py` | 证明 forward 和 config registry 可用 |
| 反事实脚本 | `experiments/run_morphology_counterfactual_suite.py` | 生成 counterfactual evidence |
| hardening 脚本 | `experiments/run_v53_reviewer_hardening_checks.py` | 生成诊断 evidence |

## 7. 表述边界

建议使用：

- morphology-derived process proxy
- micro-met-inspired residual modulation
- process-aware graph inductive bias
- soft process consistency diagnostics
- morphology counterfactual evidence

避免使用：

- full physical solver
- causal urban canopy proof
- exact physical parameter recovery
- complete physics-informed model
- deterministic physical causality