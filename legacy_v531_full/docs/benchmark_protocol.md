# Fair Static-Information Benchmark Protocol

This protocol defines how to compare UrbanPiDiT and baselines without giving one model a hidden static-information advantage.

## Benchmark question

The benchmark separates three effects:

1. Dynamic-history forecasting skill.
2. Added value of static urban morphology.
3. Robustness to static-field perturbation.

A valid comparison must state which static information each model receives.

## Shared data scope

All model families should use the same:

- dynamic variables: `d2m`, `sp`, `t2m`, `tcc`, `tp`, `u10`, `v10`
- historical context length: `k`
- forecast leads: 6h, 12h, 18h, 24h when `time_step_hours=6` and `lead_times=[1,2,3,4]`
- static variables: `landcover`, `building_surface`, `buildings`, `building_volume`, `population`
- train-only normalization and climatology statistics
- validation/test splits

## Static schema

Use one schema for all models:

```yaml
static_schema:
  categorical: [landcover]
  continuous: [building_surface, buildings, building_volume, population]
  categorical_cardinality:
    landcover: 20
```

Rules:

- Continuous static fields can be normalized with train-only statistics.
- Categorical static fields must not be z-scored as continuous values.
- If a baseline cannot consume static fields, it must be marked `dynamic_only`.

## Static policies

| Policy | Meaning | Intended comparison |
|---|---|---|
| `dynamic_only` | historical meteorology only | dynamic skill floor |
| `same_static` | same static fields as UrbanPiDiT | fair static-aware comparison |
| `static_zero` | static channels set to zero | static reliance check |
| `static_shuffle` | static spatial structure perturbed | morphology-structure reliance check |

## Baseline families

Shape-safe baselines:

- `persistence`
- `climatology`
- `linear_trend`
- `moving_average`
- `exp_smoothing`
- `residual_climatology`
- `spatial_mean`
- `local_mean`
- `static_analog`
- `static_persistence_blend`

Legacy trainable baselines:

- ridge/lasso linear regression
- random forest / XGBoost-style tree baseline
- simple MLP

The shape-safe baselines are primarily for controlled protocol checks and reproducible low-cost comparisons. Legacy trainable baselines remain useful for capacity comparisons.

## Required result tables

### Table A: dynamic-only baselines

| Model | Static policy | RMSE 6h | RMSE 12h | RMSE 18h | RMSE 24h | ACC mean |
|---|---|---:|---:|---:|---:|---:|
| persistence | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| climatology | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| linear_trend | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| moving_average | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| exp_smoothing | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| residual_climatology | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| spatial_mean | dynamic_only | TBD | TBD | TBD | TBD | TBD |
| local_mean | dynamic_only | TBD | TBD | TBD | TBD | TBD |

### Table B: static-aware controls

| Model | Static policy | RMSE mean | ACC mean | Interpretation |
|---|---|---:|---:|---|
| static_analog | same_static | TBD | TBD | static morphology available |
| static_analog | static_zero | TBD | TBD | static removed |
| static_analog | static_shuffle | TBD | TBD | static structure perturbed |
| static_persistence_blend | same_static | TBD | TBD | static morphology available |
| static_persistence_blend | static_zero | TBD | TBD | static removed |
| static_persistence_blend | static_shuffle | TBD | TBD | static structure perturbed |

### Table C: proposed-model static ablation

| Variant | Static encoder | Graph | Canopy | RMSE mean | ACC mean | Event F1 |
|---|---:|---:|---:|---:|---:|---:|
| base | off | off | off | TBD | TBD | TBD |
| static_encoder_only | on | off | off | TBD | TBD | TBD |
| morphology_graph_only | off | on | off | TBD | TBD | TBD |
| wind_aware_graph_only | off | on | off | TBD | TBD | TBD |
| urban_canopy_only | off | off | on | TBD | TBD | TBD |
| combined | on | on | on | TBD | TBD | TBD |

## Leakage controls

Allowed:

- historical context `x_ctx`
- static fields from the configured static schema
- train-only normalization/climatology
- lead-time metadata

Forbidden:

- future target fields as model input
- target-period-derived static fields
- validation/test statistics used in training preprocessing
- tuning baselines on test metrics

## Commands

Dry-run manifest for fair benchmark:

```bash
python -m experiments.fair_benchmark --config configs/fair_benchmark.yaml --out_dir outputs/experiments/fair_benchmark
```

Run only when compute is intended:

```bash
python -m experiments.fair_benchmark --config configs/fair_benchmark.yaml --out_dir outputs/experiments/fair_benchmark --execute
```

Single-baseline configs are under `configs/baselines/` for isolated runs and audits.