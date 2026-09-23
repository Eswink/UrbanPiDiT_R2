# Rebuttal Response Map

This note maps the code upgrade to likely TGRS reviewer concerns. It is a response map, not a claim of final empirical superiority.

## Reviewer concern: static urban information is not fairly controlled

Response scope:

- Added a single static schema shared by UrbanPiDiT and baselines.
- Split static fields into categorical and continuous groups.
- Kept categorical land-cover out of continuous z-score statistics.
- Added controlled static-information policies:
  - `dynamic_only`: historical meteorology only.
  - `same_static`: same static fields available to the proposed model.
  - `static_zero`: static channels removed by zeroing.
  - `static_shuffle`: spatial static structure perturbed.

Primary artifacts:

- `configs/fair_benchmark.yaml`
- `configs/fair_baselines.yaml`
- `baselines/forecast_base.py`
- `baselines/forecast_models.py`
- `baselines/forecast_runner.py`
- `tests/test_baseline_shapes.py`
- `tests/test_config_loading.py`

What this strengthens:

- Baseline comparison is no longer ambiguous about who sees static information.
- Static-feature gains can be separated from dynamic-history gains.
- Static perturbation controls expose whether a model relies on meaningful morphology or only extra channels.

## Reviewer concern: the urban module is architectural decoration

Response scope:

- Added separate, disable-by-default modules for static morphology encoding, morphology graph, wind-aware morphology graph, dynamic/static variable graph split, and canopy coupling.
- Added ablation manifest generation for each module family.
- Kept all new modules optional to preserve V4 compatibility.

Primary artifacts:

- `models/static_encoder.py`
- `models/morphology_graph.py`
- `models/urban_canopy.py`
- `models/urban_pidit.py`
- `experiments/ablation.py`
- `configs/urbanpidit_v5_canopy.yaml`

Expected table structure:

| Variant | Static encoder | Morphology graph | Wind-aware graph | Dynamic VG | Static VG | Canopy | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| base | off | off | off | off | off | off | V4-compatible reference |
| static_encoder_only | on | off | off | off | off | off | static representation value |
| morphology_graph_only | off | on | off | off | off | off | spatial-morphology relation value |
| wind_aware_graph_only | off | on | on | off | off | off | wind-direction conditioning value |
| dynamic_vg_only | off | off | off | on | off | off | dynamic variable relation value |
| static_vg_only | off | off | off | off | on | off | static-variable relation value |
| urban_canopy_only | off | off | off | off | off | on | canopy coupling value |
| combined | on | on | on | on | on | on | full V5 candidate |

## Reviewer concern: physical consistency is mixed into a single opaque loss

Response scope:

- Split physical consistency into feasibility, structure, and process consistency groups.
- Preserved old logging keys while adding grouped logs.
- Kept training objective compatibility: legacy `lambda_phys`, `lambda_fft`, and `lambda_grad` semantics remain valid.

Primary artifacts:

- `losses/physical_consistency.py`
- `pidit_lit.py`
- `docs/physical_consistency.md`

Reviewer-facing interpretation:

- Feasibility: invalid value suppression, such as cloud fraction range and dew point relation.
- Structure: spectral and spatial-gradient shape preservation.
- Process consistency: meteorological process proxies such as wind divergence.

## Reviewer concern: single-city evidence is not enough

Response scope:

- Added multi-region dataset wrappers and leave-one-city-out split generation.
- Added dry-run experiment manifests so cross-region protocol can be audited before compute-heavy training.

Primary artifacts:

- `data/splits.py`
- `data/static_preprocess.py`
- `data/multi_region_loader.py`
- `run_leave_one_city_out.py`
- `experiments/leave_one_city_out.py`
- `configs/multi_city.yaml`
- `tests/test_multi_region.py`

Expected table structure:

| Held-out city | Train cities | Val city | Test city | RMSE mean | ACC mean | Event F1 |
|---|---|---|---|---:|---:|---:|
| Beijing | Shanghai, Guangzhou | Beijing | Beijing | TBD | TBD | TBD |
| Shanghai | Beijing, Guangzhou | Shanghai | Shanghai | TBD | TBD | TBD |
| Guangzhou | Beijing, Shanghai | Guangzhou | Guangzhou | TBD | TBD | TBD |

## Reviewer concern: new components might leak future labels

Response scope:

- New static modules consume static fields and historical/current context only.
- Wind-aware graph reads wind direction from historical context, not future target fields.
- Experiment scripts default to manifest or dry-run behavior and do not launch long training unless explicitly requested.

Leakage boundary:

- Allowed: `x_ctx`, `static_raw`, `static_cont`, `static_cat`, current denoising state, train-only normalization/climatology statistics.
- Not allowed: future `y`, future `x0`, validation/test target statistics, lead-specific future observations beyond the supervised loss target.

## Current status

Implemented infrastructure and tests establish comparability and reproducibility. Final paper claims still require running the generated benchmark, ablation, perturbation, and leave-one-city-out manifests on the intended data splits.