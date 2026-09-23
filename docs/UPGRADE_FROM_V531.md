# 从 V5.3.1 到 V6 的迁移

## 保留的思想

- morphology/process conditioning；
- process diagnostics；
- counterfactual / perturbation mentality；
- physical consistency loss；
- probabilistic residual generation（作为后期模块）。

## 明确废弃的主路径

1. `pixel = token` 的高分辨率用法；
2. 每层 full self-attention + full cross-attention；
3. `torch.cdist -> dense N×N adjacency`；
4. 所有静态/动态变量强制共享同一 `H×W`；
5. 把 morphology proxy 直接等同于真实物理过程。

## 新增

- `ProcessStateInitializer`；
- `RecurrentProcessReasoner`；
- `ReasoningRouter(STOP/ZOOM)`；
- `UrbanExpert` patch/window backbone；
- `SparseGridProcessGraph`；
- `ForecastVerifier`；
- 多尺度 data contract；
- residual diffusion scaffold。

旧版核心源码快照保存在 `model/legacy_v531/` 与 `data/legacy_v531/`，用于回归比较，不作为 V6 主路径。
