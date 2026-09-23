# V5.1 to V5.2 Change Log

V5.2 MicroMetRefine is a targeted correction over V5.1. It does not change the forecasting task or the V4/V5.1 training interface.

## Key changes

| Area | V5.1 | V5.2 |
|---|---|---|
| Graph static path | could fall back to generic `static_feat` | prioritizes `static_cont` and `static_cat`; raw static is fallback only |
| Graph diagnostics | relation statistics existed but schema source was less explicit | reports schema source, fallback flags, channel counts, edge-source contribution, wind-aware statistics |
| MicroMet mapper | raw static could be preferred | schema-separated static fields are preferred; categorical input is handled separately |
| Coupling location | `pre` coupling could dominate default interpretation | refined config uses `interleaved` coupling with block-level diagnostics |
| Process lambda | process terms could be zeroed in configs | V5.2 config enables RH/drag/diurnal process weights explicitly |
| Drag proxy | absolute wind-speed comparison | wind-speed residual compared against latest historical state |
| Diurnal proxy | coarse thermal response | day/night response diagnostics are separated |
| Static perturbation | V5.1 protocol naming | V5.2 protocol naming and interleaved refined config, still dry-run by default |
| Reviewer mapping | V5.1 artifact map | V5.2 map targets schema-faithful graph, MicroMetRefine, and evaluation protocol |

## Compatibility

| Compatibility item | Status |
|---|---|
| Existing checkpoints | preserved through optional modules and safe construction |
| V4/V5.1 config style | preserved |
| old model forward API | preserved |
| old physical loss log keys | preserved |
| dry-run experiment workflow | preserved |

## Leakage boundary refinements

V5.2 keeps the model-input and physical-evaluation boundaries separate:

```text
model forward:
  x_t + x_ctx + lead_time + static_raw/static_cont/static_cat + hour_of_day

physical evaluation:
  prediction + target + train stats + static fields + latest historical state + hour_of_day
```

`latest_state` is not passed into the model. It is derived from the historical context or the current autoregressive rollout context and is used only by process-consistency diagnostics.

## New/updated artifacts

| Type | Artifact |
|---|---|
| full config | `configs/urbanpidit_v52_micromet_refine.yaml` |
| mechanism ablation config | `configs/mechanism_ablation_v52.yaml` |
| static perturbation config | `configs/static_perturbation.yaml` |
| smoke test | `experiments/smoke_test_v52.py` |
| method doc | `docs/method_v52_micromet_refine.md` |
| reviewer map | `docs/reviewer_2_3_response_mapping_v52.md` |
| guardrail | `docs/physical_claims_guardrail.md` |

## Claim boundary

V5.2 supports the following level of claim:

> Urban morphology is used as a schema-faithful, ablatable source of graph relations and micro-meteorology-inspired process proxies, and physical feasibility is evaluated through auditable proxy diagnostics.

V5.2 does not support the following claims:

- complete physical realism
- resolved urban canopy simulation
- causal proof of urban morphology effects
- physically exact conservation
- future-aware correction