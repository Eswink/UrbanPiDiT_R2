# UrbanPiDiT-V5.3.1 MorphoProcessDiT Reviewer Evidence

V5.3.1 是 V5.3 Morpho-Process DiT 的审稿证据强化版本。主张边界是：模型是 morphology-conditioned process diffusion transformer；城市形态先转成 morphology-derived process proxy，再进入去噪过程、过程图和诊断链路。

V5.3.1 不把 proxy 声称为真实物理参数，也不把反事实干预诊断声称为因果证明。

## 1. 与 V5.3 的关系

| 层级 | 作用 |
|---|---|
| V5.3 | 方法基础：process proxy、process AdaLN、urban control branch、anisotropic process graph、MicroMet token branches、morphology residual head |
| V5.3.1 | 审稿证据强化：统一版本权威、显式 diagnostics API、reviewer evidence harness、counterfactual intervention suite、relation-wise graph diagnostics、safe claim wording |

历史目录名仍为 `UrbanPiDiT_V4_with_baselines`，这是工程兼容名，不代表论文方法版本。

## 2. Reviewer-facing evidence chain

```text
static morphology
  -> morphology-derived process proxy
  -> process-conditioned modulation / process graph / residual control
  -> forward diagnostics JSON
  -> ablation and counterfactual intervention diagnostics
```

关键证据产物：

| 证据 | 入口 | 输出重点 |
|---|---|---|
| Forward diagnostics | `experiments/run_v531_reviewer_evidence_checks.py` | version、feature flags、proxy availability、process graph、gate/residual norm、leakage flags |
| Counterfactual suite | `experiments/run_morphology_counterfactual_suite.py` | baseline/intervention delta、proxy delta、process graph delta、prediction delta |
| Relation-wise graph diagnostics | `tests/test_process_graph_relation_diagnostics.py` | relation/source 级别 graph stats 与 reviewer-facing key |
| Proxy availability flags | `tests/test_v531_proxy_availability.py` | proxy/source availability diagnostic，避免静默退回普通 static token |
| No target leakage contract | `tests/test_no_target_leakage_contract.py` | validation/test 只用 target 计算指标，不把 target 喂入生成输入 |

## 3. 推荐复现实验入口

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

## 4. Diagnostics contract

V5.3.1 的 forward 返回协议：

| 调用方式 | 返回 |
|---|---|
| `model(...)` | prediction tensor |
| `model(..., return_diagnostics=True)` | `(prediction, diagnostics)` |

`diagnostics` 面向 JSON 和日志消费，包含但不限于：

- `version/`：版本与方法元数据。
- `model/`：模块开关和兼容状态。
- `proxy/`：morphology-derived process proxy 统计和 availability flags。
- `process_adaln/`：过程调制强度。
- `urban_control/`：控制残差强度。
- `process_graph/`：过程图统计、relation-wise diagnostics。
- `micromet/`：MicroMet token/residual branch gate 和 norm。
- `residual_prediction/`：输出侧形态残差头诊断。
- `leakage/`：target 是否被提供给 forward 的契约标记。

训练日志只聚合稳定标量项；文本、shape list 和高维对象不进入 Lightning 标量日志。

## 5. 表述边界

推荐表述：

- Morphology-conditioned Process Diffusion Transformer
- morphology-derived process proxy
- process-aware / process-guided diagnostics
- counterfactual intervention diagnostic

避免表述：

- Physics-Informed Diffusion Transformer
- proxy 是真实物理参数
- 反事实诊断证明因果
- complete physics-informed model

## 6. 最小测试证据

当前关键回归组合：

```bash
conda run -p /3240608030/weather-q/weather python -m pytest -q \
  tests/test_v531_reviewer_evidence_checks.py \
  tests/test_morphology_counterfactual_suite.py \
  tests/test_morpho_process_forward.py \
  tests/test_model_registry.py \
  tests/test_config_loading.py \
  tests/test_anisotropic_process_graph.py \
  tests/test_process_proxy_encoder.py \
  tests/test_process_graph_relation_diagnostics.py \
  tests/test_v531_proxy_availability.py \
  tests/test_no_target_leakage_contract.py
```

全量测试入口：

```bash
conda run -p /3240608030/weather-q/weather python -m pytest -q tests/
```