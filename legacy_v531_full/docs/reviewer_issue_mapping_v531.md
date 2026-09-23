# V5.3.1 Reviewer Issue Mapping

本文档用于把 TGRS 审稿攻击面、V5.3.1 证据链、可复现脚本和表述边界放在同一个执行表里。V5.3.1 的目标不是扩大物理声称，而是证明模型不是“普通 DiT + cross-attention + static morphology tokens”。

## 1. 攻击面到证据链

| Reviewer attack surface | V5.3.1 response | 证据入口 |
|---|---|---|
| 只是把静态形态 token 拼到 DiT | 静态形态先变成 morphology-derived process proxy，再进入过程调制、过程图和残差控制 | `run_v531_reviewer_evidence_checks.py` diagnostics、proxy availability tests |
| Cross-attention 不等于过程调制 | `ProcessConditionedAdaLN` 和 control branch 在去噪过程中提供 block-level / spatial residual 调制 | `process_adaln/`、`urban_control/` diagnostics |
| 城市图可能只是装饰性 kNN | `AnisotropicUrbanProcessGraph` 输出 relation/source 级别统计，并暴露 reviewer-facing graph keys | `tests/test_process_graph_relation_diagnostics.py` |
| 过程代理可能只是静态编码别名 | diagnostics 显式记录 proxy source/availability，并在反事实干预中报告 proxy delta | `tests/test_v531_proxy_availability.py`、counterfactual JSON |
| 反事实实验可能被误读为因果证明 | 明确定位为 counterfactual intervention diagnostic，不作为 causal proof | `run_morphology_counterfactual_suite.py` 输出说明与 README/docs 表述边界 |
| 验证/测试可能泄漏目标场 | contract 测试拦截模型输入，确认 target 只参与指标计算，不进入生成输入 | `tests/test_no_target_leakage_contract.py` |
| 新版本命名与旧工程目录混乱 | `urbanpidit_version.py` 作为 canonical version authority，历史目录名只作工程兼容 | version metadata、registry/config/package tests |

## 2. Evidence artifacts

| Artifact | 内容 | Reviewer value |
|---|---|---|
| `configs/urbanpidit_v531_reviewer_evidence.yaml` | V5.3.1 reviewer-evidence 默认配置 | 固定复现实验设置 |
| `experiments/run_v531_reviewer_evidence_checks.py` | 小批量 forward + diagnostics JSON | 证明 forward diagnostics API 可用 |
| `experiments/run_v53_reviewer_hardening_checks.py` | V5.3 兼容 wrapper | 保留旧脚本入口，不改变主张口径 |
| `experiments/run_morphology_counterfactual_suite.py` | baseline + intervention batch forward | 证明形态干预会反映到 proxy/graph/prediction delta |
| `utils/diagnostics.py` | Lightning 标量日志过滤 | 避免把文本、shape list、高维对象写进训练日志 |
| `tests/test_no_target_leakage_contract.py` | validation/test contract | 锁定无 target leakage |

## 3. Rebuttal wording templates

### 3.1 Not static-token conditioning only

V5.3.1 does not merely append static morphology tokens to a DiT backbone. It converts urban morphology into morphology-derived process proxies and injects them into the denoising process through process-conditioned modulation, morphology control residuals, and an anisotropic process graph.

中文口径：V5.3.1 并非只把静态城市形态作为 cross-attention token 拼到 DiT 上，而是先生成可诊断的形态过程代理，再通过过程调制、空间控制残差和各向异性过程图影响去噪过程。

### 3.2 Process graph is diagnostic, not decorative

The graph diagnostics are relation-wise and source-aware. They expose edge sparsity, entropy, morphology/category relation ratios, wind/process modulation strength, and reviewer-facing aliases such as `process_graph/same_landcover_edge_ratio`.

中文口径：过程图诊断按 relation/source 拆分，输出稀疏度、熵、形态/类别关系比例、风向/过程调制强度，并提供 reviewer-facing JSON key，避免图模块只停留在装饰性描述。

### 3.3 Counterfactual intervention boundary

The counterfactual suite is used as an intervention diagnostic. It measures whether morphology perturbations propagate to proxy fields, process graph statistics, and predictions. It is not presented as causal proof.

中文口径：反事实套件用于干预诊断，观察形态扰动是否传递到 proxy、过程图统计和预测输出；它不是因果证明。

### 3.4 Safe physical-claim boundary

We use micro-met-inspired process proxies as diagnostic and conditioning signals. They are not claimed as measured physical parameters or a complete physical solver.

中文口径：我们使用 micro-met-inspired process proxies 作为诊断和条件信号，不声称它们是真实物理观测参数，也不声称模型是完整物理求解器。

## 4. 最小复现命令

```bash
conda run -p /3240608030/weather-q/weather python experiments/run_v531_reviewer_evidence_checks.py \
  --config configs/urbanpidit_v531_reviewer_evidence.yaml \
  --output /tmp/v531_reviewer_evidence_checks.json \
  --batch-size 1
```

```bash
conda run -p /3240608030/weather-q/weather python experiments/run_morphology_counterfactual_suite.py \
  --config configs/urbanpidit_v531_reviewer_evidence.yaml \
  --output /tmp/v531_morphology_counterfactual_single.json \
  --batch-size 1 \
  --cases zero_static
```

```bash
conda run -p /3240608030/weather-q/weather python -m pytest -q tests/
```

## 5. 表述边界清单

推荐使用：

- Morphology-conditioned Process Diffusion Transformer
- morphology-derived process proxy
- process-aware graph diagnostics
- process-guided diagnostics
- counterfactual intervention diagnostic

避免使用：

- Physics-Informed Diffusion Transformer
- complete physics-informed model
- exact physical parameter recovery
- causal proof
- deterministic physical causality