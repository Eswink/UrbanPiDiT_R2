# Method Changes

This document records the method-level changes introduced after the V4 baseline. The goal is to support stronger scientific evidence, not to hide additional capacity inside an unfair comparison.

## Scope of the upgraded system

The upgraded system keeps the original UrbanPiDiT training and evaluation flow compatible, then adds optional components:

1. Shared configuration construction.
2. Static schema with categorical/continuous separation.
3. Static morphology encoder.
4. Heterogeneous morphology graph.
5. Wind-aware morphology graph.
6. Dynamic/static variable graph split.
7. Urban canopy coupling.
8. Physical consistency loss groups.
9. Fair static-information baselines.
10. Multi-region and leave-one-city-out infrastructure.
11. Dry-run-first experiment manifests.

## Data boundary

Allowed inputs for the new method components:

- Historical dynamic context: `x_ctx`.
- Static city fields: `static_raw`, `static_cont`, `static_cat`.
- Current denoising state inside the diffusion step.
- Train-only normalization and climatology statistics.
- Lead-time metadata.

Disallowed inputs:

- Future labels outside supervised loss computation.
- Future `x0` as side information.
- Validation/test-derived normalization or climatology statistics.
- Static fields whose values are computed from target-period labels.

## Static schema

Static variables are now represented by an explicit schema:

```yaml
static_schema:
  categorical: [landcover]
  continuous: [building_surface, buildings, building_volume, population]
  categorical_cardinality:
    landcover: 20
```

Effect:

- Continuous morphology variables use train-only continuous statistics.
- Categorical variables remain category indices before embedding.
- Old configurations without `static_schema` still use the previous `x_ctx` path.

## Urban morphology path

The morphology path has three independent levels:

| Level | Module | Information used | Default |
|---|---|---|---|
| Static encoding | `StaticMorphologyEncoder` | static fields only | off |
| Spatial graph | `HeterogeneousMorphologyGraph` | static similarity and spatial proximity | off |
| Wind-aware graph | `WindAwareMorphologyGraph` | static graph plus historical wind direction | off |

Design principle:

- The graph should represent urban spatial relation, not target leakage.
- Wind direction comes from history/current context only.
- The graph path is inserted before Transformer processing and remains separately ablatable.

## Variable graph split

The old `use_variable_graph` switch is preserved. Two finer switches were added:

- `use_dynamic_vg`: relation modeling among dynamic meteorological variables.
- `use_static_vg`: relation modeling involving static morphology variables.

Compatibility rule:

- Old `use_variable_graph: true` still enables both dynamic and static VG by default.
- Explicit `use_dynamic_vg` and `use_static_vg` override the inherited behavior.

## Urban canopy coupling

`UrbanCanopyCoupling` is a residual, gated module over the current dynamic state and static morphology context.

Intended role:

- Represent local urban canopy effects as a controlled inductive bias.
- Keep the module independent and ablatable.
- Start close to identity mapping for training stability.

Default:

- Disabled in old configs.
- Enabled only in explicit V5-style configs such as `configs/urbanpidit_v5_canopy.yaml`.

## Physical consistency groups

The previous physical losses are organized into:

| Group | Meaning | Example logs |
|---|---|---|
| feasibility | valid value and basic relation constraints | `loss_feasibility`, `phys_tcc_01`, `phys_dew_leq_t` |
| structure | spectral and spatial structure | `loss_structure`, `loss_fft`, `loss_grad` |
| process consistency | process proxies | `loss_process_consistency`, `phys_wind_div` |

Compatibility:

- Legacy loss keys are still emitted.
- Legacy weight semantics are preserved.

## Experiment protocol changes

New scripts are manifest-first:

- `experiments/fair_benchmark.py`
- `experiments/ablation.py`
- `experiments/perturbation.py`
- `experiments/leave_one_city_out.py`
- `experiments/event_eval.py`
- `experiments/collect.py`
- `experiments/export.py`

They generate auditable configs and commands first. Long-running training or evaluation requires explicit execution flags.

## Backward compatibility

The upgrade should not change old V4 behavior unless new switches are enabled.

Compatibility targets:

- `configs/beijing.yaml`
- `configs/beijing_v4_two_stage.yaml`
- `configs/complete_config.yaml`
- old checkpoints using the original `x_ctx` static tail
- original train/evaluate/benchmark entrypoints