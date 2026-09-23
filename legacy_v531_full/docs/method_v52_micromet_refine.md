# UrbanPiDiT V5.2: MicroMetRefine

This document states the V5.2 method refinement scope for reviewer-facing writing. V5.2 is a small corrective iteration over V5.1, focused on making the urban morphology path and physical-feasibility protocol more defensible.

## Objective

V5.2 addresses six weak points in V5.1:

| V5.1 weak point | V5.2 refinement |
|---|---|
| Static fields could collapse into generic continuous tokens | schema-faithful `static_cont` + `static_cat` graph path |
| MicroMet mapper preferred raw static tensors | mapper now prioritizes schema-separated static fields |
| Coupling could be interpreted as a pre/post adapter | main V5.2 config uses interleaved coupling inside the denoising backbone |
| Drag consistency used absolute wind speed | drag proxy uses wind-speed residual relative to latest history state |
| Diurnal proxy was too coarse | day/night thermal response diagnostics are separated |
| Physical feasibility sounded like a single loss | feasibility, structure, and process diagnostics are reported separately |

The forecasting task is unchanged: historical 8×8 Beijing weather context plus static urban fields predicts 6/12/18/24h future fields for `d2m`, `sp`, `t2m`, `tcc`, `tp`, `u10`, and `v10`.

## Data boundary

Allowed inputs for V5.2 mechanism modules:

| Source | Used by |
|---|---|
| historical dynamic context | wind extraction, latest-state process context |
| current denoising state | MicroMet residual branches |
| static continuous fields | graph relation and morphology proxy mapping |
| static categorical fields | land-cover-aware graph and mapper inputs |
| current/context `hour_of_day` | optional diurnal response gate |

Disallowed as mechanism inputs:

| Source | Rule |
|---|---|
| future target `y` | supervised loss/evaluation only |
| future clean state `x0` | not used as module side information |
| target-derived future timestamp | not used for process context |
| validation/test target statistics | not used as training normalization or context |

## Refined module relation

```text
static_cont + static_cat
  -> schema-faithful morphology graph
  -> wind-aware graph diagnostics
  -> interleaved MicroMet residual modulation
  -> DiT denoising backbone
  -> forecast

forecast + latest historical state + static fields + hour
  -> physical-feasibility evaluation protocol
```

The graph and MicroMet branches remain optional and ablatable. The V5.2 configuration enables them for the refined experiment, but manifest scripts remain dry-run unless explicitly executed.

## Schema-faithful morphology graph

V5.2 makes the static path explicit:

- `static_cont` carries continuous morphology fields such as building surface, building count, volume, and population.
- `static_cat` carries categorical land-cover fields without treating them as normalized continuous measurements.
- `static_raw/static_feat` is only a fallback when schema-separated tensors are unavailable.

Reviewer-facing claim:

> Urban morphology is not merely appended as static tokens; it controls the graph relation through separate continuous, categorical, geographic, and wind-aware components.

## MicroMetRefine coupling

The V5.2 coupling remains a weak residual mechanism, not a full canopy solver.

| Branch | Process proxy | Boundary |
|---|---|---|
| momentum drag | roughness/drag/ventilation proxy modulates `u10/v10` residuals | current state + static fields |
| thermal storage | heat storage/impervious/population proxy with hour gate | current state + static fields + hour |
| moisture evaporation | evaporation and imperviousness proxy | current state + static fields |
| ventilation mixing | wind-aware morphology graph mixing | current/context wind + static fields |

Diagnostics report proxy means, mapper fallback flags, branch gates, graph statistics, and interleaved block-level residual records.

## Physical-feasibility protocol

V5.2 treats physical feasibility as an evaluation protocol with optional regularization, not as proof of physical closure.

| Group | V5.2 behavior |
|---|---|
| feasibility | selected value bounds and basic meteorological relations |
| structure | FFT, gradient, and wind-divergence proxies |
| process RH | Magnus-formula relative humidity consistency |
| process drag | high-drag vs low-drag wind-speed residual gap |
| process diurnal | daytime warming and nighttime heat-release gaps |

Important scope statement:

> These are auditable proxy diagnostics for physical feasibility. They do not solve Navier-Stokes, do not implement a complete urban canopy model, and do not prove causal urban-process realism.

## Evidence interfaces

| Evidence need | Artifact |
|---|---|
| refined full config | `configs/urbanpidit_v52_micromet_refine.yaml` |
| mechanism ablation manifest | `configs/mechanism_ablation_v52.yaml`, `experiments/run_mechanism_ablation.py` |
| static perturbation controls | `configs/static_perturbation.yaml`, `experiments/run_static_perturbation.py` |
| graph diagnostics | `experiments/run_graph_diagnostics.py` |
| process evaluation | `experiments/run_process_consistency_eval.py` |
| smoke check | `experiments/smoke_test_v52.py` |

## Recommended reviewer wording

Use:

- "schema-faithful morphology graph"
- "morphology-derived learnable process proxies"
- "interleaved micro-meteorology-inspired residual modulation"
- "physical-feasibility evaluation protocol"
- "residual-based drag consistency"
- "day/night diurnal thermal response proxy"

Avoid:

- "physically exact urban simulation"
- "resolved urban canopy solver"
- "causal proof of urban morphology effects"
- "future-aware correction"
- "guaranteed physical consistency"