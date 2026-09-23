# Physical Consistency Loss Protocol

This document describes the upgraded physical-consistency objective and its logging protocol.

## Goal

The physical-consistency objective should improve scientific plausibility without mixing unrelated penalties into an opaque scalar.

The upgraded loss groups are:

1. Feasibility.
2. Structure.
3. Process consistency.

Legacy loss keys and weights remain available for backward compatibility.

## Inputs and leakage boundary

The physical loss is computed from:

- model prediction
- supervised target for the same lead
- train-only normalization statistics
- configured variable names and physical bounds

It does not provide future targets as model input. The target is only used inside the supervised training loss calculation.

## Feasibility group

Purpose:

- Penalize meteorologically invalid or implausible values.

Typical constraints:

| Constraint | Meaning |
|---|---|
| `tcc_01` | total cloud cover should remain in [0, 1] after denormalization |
| `tp_nonneg` | precipitation should be non-negative |
| `sp_nonneg` | surface pressure should be non-negative |
| `dew_leq_t` | dew point should not exceed air temperature |

Primary logs:

- `loss_feasibility`
- `phys_tcc_01`
- `phys_tp_nonneg`
- `phys_sp_nonneg`
- `phys_dew_leq_t`

## Structure group

Purpose:

- Preserve spatial and spectral structure, not only pixel-wise magnitude.

Subgroups:

| Subgroup | Meaning | Legacy key |
|---|---|---|
| FFT spectrum | frequency-domain shape consistency | `loss_fft` |
| Gradient/Sobel | local spatial-gradient consistency | `loss_grad` |

Primary logs:

- `loss_structure`
- `loss_fft`
- `loss_grad`

Expected effect:

- Less over-smoothing.
- Better spatial edge and texture preservation.
- More informative diagnosis of whether improvements come from local or spectral structure.

## Process-consistency group

Purpose:

- Add process-level meteorological proxies while keeping them separately visible.

Current supported scope includes wind-field process terms such as divergence-style penalties. Additional process terms can be added without changing the external loss contract.

Primary logs:

- `loss_process_consistency`
- process-specific `phys_*` keys

## Weighting protocol

Configuration can use both grouped and legacy fields.

Example:

```yaml
losses:
  feasibility:
    enabled: true
    lambda: 0.2
  structure:
    enabled: true
    fft_enabled: true
    grad_enabled: true
    lambda_fft: 0.05
    lambda_grad: 0.5
  process:
    enabled: true
    lambda_rh: 0.0
    lambda_drag: 0.0
    lambda_diurnal: 0.0

physics:
  lambda: 0.2
  lambda_fft: 0.05
  lambda_grad: 0.5
```

Compatibility rule:

- `physics.lambda` maps to feasibility-style physical loss when grouped config is absent.
- `physics.lambda_fft` and `physics.lambda_grad` keep their previous meanings.
- Old dashboards depending on `loss_fft`, `loss_grad`, or `loss_phys_spatial` remain usable.

## Reporting protocol

Training/validation reports should distinguish:

1. Forecast error: RMSE, MAE, ACC.
2. Feasibility violations.
3. Structure mismatch.
4. Process-consistency mismatch.
5. Event metrics for threshold-sensitive variables.

Recommended table:

| Variant | RMSE mean | ACC mean | Feasibility | Structure | Process | Event F1 |
|---|---:|---:|---:|---:|---:|---:|
| base | TBD | TBD | TBD | TBD | TBD | TBD |
| + feasibility | TBD | TBD | TBD | TBD | TBD | TBD |
| + structure | TBD | TBD | TBD | TBD | TBD | TBD |
| + process | TBD | TBD | TBD | TBD | TBD | TBD |

## Interpretation cautions

- A lower physical penalty is not automatically a better forecast if RMSE/ACC degrade substantially.
- A process proxy is not a full numerical weather model constraint.
- The grouped logs are diagnostic aids; final claims should be based on forecast metrics plus physically meaningful error analysis.