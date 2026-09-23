# Graph Design and Ablation Protocol

This document defines the V5.1 graph scope, ablation variants, and expected diagnostics.

## Graph objective

The graph path should answer one narrow methodological question:

> Does explicit urban morphology relation modeling improve forecast behavior beyond using static fields as ordinary channels?

The graph path is therefore separated from the rest of the model and controlled by explicit switches.

## Input boundary

Allowed:

- static categorical field: `landcover`
- static continuous fields: `building_surface`, `buildings`, `building_volume`, `population`
- historical wind channels from context: `u10`, `v10`
- grid coordinates

Not allowed:

- future target wind
- future precipitation/cloud/temperature labels
- validation/test-derived statistics
- target-period timestamp as hidden side information

## Base morphology graph

The base graph has three relation components:

| Component | Purpose | Example |
|---|---|---|
| geographic proximity | prevents distant unrelated cells from dominating | nearby 8×8 cells are easier to connect |
| continuous morphology similarity | captures built-density and population similarity | similar building volume/population |
| categorical agreement | preserves land-cover type relation | same landcover category |

Conceptual form:

```text
edge_weight(i, j)
  = normalize(
      spatial_kernel(i, j)
      * continuous_similarity(i, j)
      * categorical_compatibility(i, j)
    )
```

## Wind-aware graph

The wind-aware graph starts from the base graph and applies history-only wind modulation.

Expected behavior:

| Condition | Expected graph behavior |
|---|---|
| stronger wind | stronger anisotropy |
| calm wind | fallback close to base morphology graph |
| high roughness path | attenuated propagation |
| aligned wind direction | enhanced directional relation |

This is a propagation prior, not a forecast target shortcut.

## Required ablation variants

| Variant | `use_static_morphology_encoder` | `use_morphology_graph` | `use_wind_aware_graph` | Purpose |
|---|---:|---:|---:|---|
| V4 base | false | false | false | no morphology graph path |
| static encoder | true | false | false | representation only |
| morphology graph | true | true | false | static relation value |
| wind-aware graph | true | true | true | historical-wind modulation value |
| full V5.1 | true | true | true | combined with coupling modules |

## Diagnostic outputs

`experiments/run_graph_diagnostics.py` reports lightweight graph diagnostics:

| Diagnostic | Meaning |
|---|---|
| `mean_degree` | effective sparse neighborhood size |
| `edge_weight_mean` | normalization sanity check |
| `same_category_edge_ratio` | land-cover relation strength |
| `graph_entropy` | concentration/diversity of graph weights |
| `wind_anisotropy_mean` | directional modulation strength |
| `roughness_blocking_mean` | average roughness attenuation |
| `calm_wind` | whether fallback path is active |

## Static perturbation controls

Graph evidence should be paired with static perturbation:

| Perturbation | Interpretation |
|---|---|
| `fill_mean` | removes local morphology contrast |
| `shuffle_spatial` | keeps marginal distribution but breaks spatial layout |
| `landcover_only` | checks categorical contribution |
| `continuous_only` | checks continuous morphology contribution |
| `remove_*` | isolates individual static variable contribution |

If graph performance remains unchanged under spatial static shuffling, the graph claim should be weakened.

## Minimal reviewer-facing evidence package

1. Main forecast metrics: RMSE, MAE, ACC by lead time.
2. Graph ablation table.
3. Graph diagnostics table.
4. Static perturbation degradation table.
5. Leakage boundary statement.

## Claim guardrail

Acceptable claim:

> The graph introduces an ablatable morphology- and history-wind-conditioned spatial prior, and diagnostics show whether it changes relation structure as intended.

Overclaim to avoid:

> The graph proves causal urban transport or fully captures airflow dynamics.