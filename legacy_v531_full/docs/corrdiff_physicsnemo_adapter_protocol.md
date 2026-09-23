# CorrDiff PhysicsNeMo Loader Adapter Protocol

## Source provenance

- Upstream repository: NVIDIA PhysicsNeMo, `https://github.com/NVIDIA/physicsnemo.git`.
- Uploaded full archive used for this patch: `physicsnemo-main.zip`.
- Extracted upstream example: `examples/weather/corrdiff`.
- Patch source snapshot: `baselines/external_sources/physicsnemo_weather_corrdiff`.
- Source manifest: `baselines/external_sources/PHYSICSNEMO_CORRDIFF_SOURCE_MANIFEST.json`.

The previous incomplete `conf.zip` source snapshot should no longer be used.

## Adapter-vs-official distinction

The included adapter is designed for fair comparison under the UrbanPiDiT loader. It is not an official PhysicsNeMo execution path.  The official source snapshot is included for traceability, while the adapter provides a native PyTorch module matching the shared baseline interface.

## Fairness contract

The CorrDiff adapter must be trained and evaluated with:

1. the same UrbanPiDiT train/validation/test splits;
2. the same input history length `k`;
3. the same forecast lead times;
4. the same dynamic variables;
5. explicitly stated static-information policy;
6. the same normalization statistics;
7. the same metrics: RMSE, MAE, Bias, ACC, CRPS, plus per-variable versions;
8. the same early stopping and checkpoint protocol as other external baselines;
9. TensorBoard logging with resume-compatible global steps.

## Static-information variants

Use at least:

| Alias | Purpose |
|---|---|
| `corrdiff_same_static` | Fair morphology-aware CorrDiff comparison |
| `corrdiff_dynamic_only` | Dynamic-only CorrDiff lower bound |
| `corrdiff_static_zero` | Static-ablation sanity check |
| `corrdiff_static_shuffle` | Static spatial/batch consistency test |

## Probabilistic evaluation

When `eval_ensemble_size > 1`, the adapter returns `[B, E, L, C, H, W]`.  Deterministic metrics are computed from the ensemble mean, while CRPS uses the ensemble samples.  This is preferable to using deterministic MAE as a CRPS placeholder.