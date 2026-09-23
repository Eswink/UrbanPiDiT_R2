# UrbanPiDiT V5.1: MicroMet-Coupled Morphology Graph

This document states the V5.1 method scope for reviewer-facing writing. It is a method description and audit note, not a claim that final numerical results have already been obtained.

## Method objective

V5.1 strengthens UrbanPiDiT along two axes:

1. Urban morphology is no longer treated only as extra static channels.
2. Physical feasibility is evaluated through explicit, auditable proxy groups instead of a single opaque penalty.

The target task remains unchanged:

- Input: historical 8×8 Beijing meteorological fields plus static urban fields.
- Forecast targets: 6/12/18/24h ahead for seven variables.
- Variables: `d2m`, `sp`, `t2m`, `tcc`, `tp`, `u10`, `v10`.
- Static fields: `landcover`, `building_surface`, `buildings`, `building_volume`, `population`.

## Data boundary

Allowed information for V5.1 modules:

| Source | Usage |
|---|---|
| historical dynamic context | wind direction, current meteorological state, context encoding |
| current denoising state | residual correction and transformer processing |
| static morphology fields | graph relation, roughness/drag/storage proxies |
| `hour_of_day` from historical/context anchor | optional diurnal gate |
| train-only statistics | normalization, climatology, controlled diagnostics |

Disallowed information:

| Source | Rule |
|---|---|
| future target `y` | supervised loss only; never module input |
| future clean state `x0` | not used as side information |
| validation/test target statistics | not used for normalization or climatology |
| target-period filename timestamp | not used to derive `hour_of_day` |

## V5.1 module map

| Module | Role | Default | Leakage boundary |
|---|---|---:|---|
| `StaticMorphologyEncoder` | separates categorical and continuous static representation | off | static fields only |
| `HeterogeneousMorphologyGraph` | links grid cells by morphology and spatial proximity | off | static fields only |
| `WindAwareMorphologyGraph` | reweights morphology graph by historical wind direction | off | historical wind only |
| `MicroMetCouplingOperator` | gated residual coupling for drag, storage, moisture, ventilation | off | current/context/static/hour only |
| `PhysicalConsistencyLoss` | feasibility, structure, process proxy diagnostics | optional | prediction/target only inside loss |

All innovation modules remain controlled by `ablation.use_*` switches.

## MicroMet coupling structure

The coupling operator is a residual orchestrator over current model features. It maps static morphology into interpretable proxy fields:

| Proxy | Intended meaning | Static basis |
|---|---|---|
| `roughness_proxy` | surface roughness tendency | buildings, building volume, land cover |
| `drag_proxy` | momentum drag tendency | roughness and building density |
| `heat_storage_proxy` | thermal inertia tendency | built surface and building volume |
| `impervious_proxy` | reduced infiltration/evaporation tendency | built surface and land cover |
| `evap_proxy` | moisture availability tendency | land cover and low built fraction |
| `ventilation_block_proxy` | airflow obstruction tendency | building density and roughness |
| `anthropogenic_heat_proxy` | human activity heat tendency | population and built density |

Branches:

| Branch | Controlled variables | Behavior |
|---|---|---|
| momentum drag | `u10`, `v10` feature channels | roughness/drag-conditioned wind residual |
| thermal storage | `t2m` feature channel | hour-gated heat storage response |
| moisture evaporation | `d2m`, `t2m` feature channels | moisture/imperviousness-conditioned residual |
| ventilation mixing | spatial feature map | wind-aware anisotropic mixing |

Training stability guardrail:

- Residual gates are initialized near zero.
- Disabled branches become identity paths.
- `hour_of_day=None` skips the diurnal branch rather than inventing a hidden day/night assumption.

## Morphology graph design

The base graph combines three relation types:

```text
relation_score = geo proximity + continuous morphology similarity + categorical land-cover agreement
```

Wind-aware graph then adjusts the base relation using historical wind:

```text
base morphology graph
  -> historical wind direction alignment
  -> roughness/ventilation blocking attenuation
  -> normalized wind-aware graph
```

The graph is not a learned shortcut to the target. It is built from static morphology and history-only wind context.

## Physical-feasibility evaluation

V5.1 separates diagnostics into three groups:

| Group | Meaning | Examples |
|---|---|---|
| feasibility | valid value ranges and basic relations | `tp >= 0`, `0 <= tcc <= 1`, `d2m <= t2m` |
| structure | spatial/spectral consistency | FFT spectrum, Sobel gradient, wind divergence |
| process | simplified process proxies | RH consistency, morphology-conditioned drag, diurnal thermal response |

This supports more defensible claims:

- A model can improve RMSE while harming feasibility; report both.
- A low process proxy loss is not proof of full physical closure.
- Physical diagnostics are evaluation aids and optional training regularizers.

## Suggested ablation table

| Variant | Static encoder | Morphology graph | Wind graph | MicroMet | Purpose |
|---|---:|---:|---:|---:|---|
| V4 base | off | off | off | off | compatibility baseline |
| static encoder | on | off | off | off | static representation value |
| morphology graph | on | on | off | off | morphology relation value |
| wind-aware graph | on | on | on | off | history-wind propagation value |
| micromet only | on | off | off | on | coupling value without graph |
| full V5.1 | on | on | on | on | complete proposed path |

## Suggested reporting table

| Variant | RMSE mean | ACC mean | Feasibility score | Structure score | Process score | Static perturbation drop |
|---|---:|---:|---:|---:|---:|---:|
| V4 base | TBD | TBD | TBD | TBD | TBD | TBD |
| morphology graph | TBD | TBD | TBD | TBD | TBD | TBD |
| wind-aware graph | TBD | TBD | TBD | TBD | TBD | TBD |
| micromet only | TBD | TBD | TBD | TBD | TBD | TBD |
| full V5.1 | TBD | TBD | TBD | TBD | TBD | TBD |

## Reviewer-facing phrasing

Use:

- "micro-meteorology-inspired residual coupling"
- "urban morphology conditioned graph relation"
- "history-wind-aware spatial propagation"
- "physical-feasibility proxy diagnostics"

Avoid:

- "fully resolved urban canopy physics"
- "causal proof of urban process"
- "future-aware correction"
- "physically exact constraint"