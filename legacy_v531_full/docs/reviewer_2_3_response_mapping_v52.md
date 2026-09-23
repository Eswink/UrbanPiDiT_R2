# Reviewer 2/3 Response Mapping for V5.2

This document maps V5.2 MicroMetRefine artifacts to the main reviewer-facing claims. It is a construction aid for rebuttal and revision text.

## Concern: UrbanPiDiT is only DiT plus static tokens

Response:

V5.2 routes urban morphology through a schema-faithful graph and interleaved micro-meteorology modulation path. Static fields affect relation structure and process proxies, not only cross-attention context.

Artifacts:

| Evidence | Location |
|---|---|
| schema-first graph call path | `models/urban_pidit.py` |
| heterogeneous and wind-aware graph | `models/morphology_graph.py` |
| interleaved MicroMet coupling | `models/micromet_coupling.py` |
| V5.2 full config | `configs/urbanpidit_v52_micromet_refine.yaml` |
| smoke evidence | `experiments/smoke_test_v52.py` |

Reviewer-facing wording:

> V5.2 uses static urban morphology to build categorical-continuous-geographic graph relations and to produce morphology-derived micro-meteorology proxies that are injected into the denoising backbone in an interleaved manner.

## Concern: urban morphology must modulate micro-meteorological processes

Response:

V5.2 maps static morphology into named learnable proxies: roughness, drag, heat storage, imperviousness, evaporation potential, ventilation blocking, and anthropogenic heat. These proxies drive weak residual branches for wind, temperature, moisture, and ventilation mixing.

Artifacts:

| Evidence | Location |
|---|---|
| proxy mapper | `models/micromet_coupling.py` |
| proxy diagnostics | `tests/test_micromet_coupling.py` |
| coupling config | `configs/urbanpidit_v52_micromet_refine.yaml` |
| mechanism variants | `configs/mechanism_ablation_v52.yaml` |

Reviewer-facing wording:

> We do not claim to estimate physical canopy parameters directly. Instead, urban morphology is converted into learnable process proxies that modulate wind drag, thermal response, moisture coupling, and ventilation mixing as ablatable residual mechanisms.

## Concern: Urban Graph does not really use categorical/continuous/geographic/wind-aware information

Response:

V5.2 separates schema inputs and reports diagnostics for relation composition.

Artifacts:

| Evidence | Location |
|---|---|
| schema diagnostics | `models/morphology_graph.py` |
| graph tests | `tests/test_morphology_graph.py`, `tests/test_wind_aware_graph.py` |
| graph diagnostics entry | `experiments/run_graph_diagnostics.py` |

Expected diagnostics:

| Diagnostic | Interpretation |
|---|---|
| `static_cont_channels` | continuous morphology path is active |
| `static_cat_channels` | categorical land-cover path is active |
| `same_category_edge_ratio` | land-cover category contributes to relation structure |
| `geo_distance_mean` | geographic relation component is measurable |
| `cont_distance_mean` | continuous morphology similarity is measurable |
| `anisotropy_strength` | wind-aware reweighting changes graph directionality |
| `roughness_blocking_mean` | roughness/ventilation blocking contributes to propagation |

Reviewer-facing wording:

> The graph diagnostics verify whether the expected static schema and wind-aware components are active before interpreting any performance differences.

## Concern: physical feasibility cannot be only ReLU bounds

Response:

V5.2 keeps value-bound feasibility but adds process-level diagnostics with explicit availability flags and current-state context.

Artifacts:

| Evidence | Location |
|---|---|
| grouped physical protocol | `losses/physical_consistency.py` |
| process tests | `tests/test_process_consistency.py` |
| process eval entry | `experiments/run_process_consistency_eval.py` |
| claim guardrail | `docs/physical_claims_guardrail.md` |

V5.2 process terms:

| Term | What it checks | Required context |
|---|---|---|
| RH consistency | dew point / temperature relation via Magnus formula | prediction only |
| residual drag | high-drag cells should not show larger wind-speed increase than low-drag cells | latest historical state + static fields |
| diurnal response | day/night thermal response gap by heat-storage proxy | latest historical state + static fields + hour |

Reviewer-facing wording:

> Physical feasibility is reported as a protocol: bounds, structure, and process-proxy groups are logged separately, and residual-based process terms are skipped when the required history context is unavailable.

## Concern: label leakage through process context or hour information

Response:

V5.2 separates model static inputs from physical-evaluation context. `latest_state` is derived from historical context or autoregressive rollout state. It is not passed into the model forward path. `target` is used only in supervised/evaluation loss terms.

Artifacts:

| Evidence | Location |
|---|---|
| model static keyword filter | `pidit_lit.py` |
| latest-state attachment for physics only | `pidit_lit.py` |
| no-future interface test | `tests/test_process_consistency.py` |
| hour exposure gate | `configs/urbanpidit_v52_micromet_refine.yaml` |

Reviewer-facing wording:

> Future targets are never used as model side information or process context; they only enter supervised error and evaluation calculations.

## Concern: ablation evidence is insufficient

Response:

V5.2 defines a mechanism ablation ladder that isolates graph, wind-aware graph, interleaved coupling, and process terms.

Artifacts:

| Variant | Purpose |
|---|---|
| `base_dit_cross_attention` | DiT + cross-attention baseline |
| `plus_hetero_graph` | categorical/continuous/geographic graph contribution |
| `plus_wind_aware_graph` | history-wind graph contribution |
| `plus_interleaved_micromet` | MicroMet process modulation contribution |
| `plus_process_rh` | RH proxy contribution |
| `plus_process_drag` | residual drag proxy contribution |
| `plus_process_diurnal` | day/night thermal proxy contribution |
| `full_v52_micromet_refine` | complete refined path |

Reviewer-facing wording:

> The ablation ladder separates representation, relation, process modulation, and process-evaluation effects, avoiding a single all-or-nothing comparison.

## Suggested response table

| Reviewer issue | V5.2 artifact | Evidence type | Claim strength |
|---|---|---|---|
| domain novelty | graph + MicroMetRefine | mechanism ablation | methodological contribution |
| static fairness | schema path + perturbation | degradation controls | controlled comparison |
| physical feasibility | grouped protocol | proxy diagnostic table | plausibility support |
| graph interpretability | graph diagnostics | relation statistics | mechanism transparency |
| leakage boundary | filtered model kwargs + tests | interface verification | reproducibility guardrail |

## Guardrail statement

> V5.2 should be described as a domain-specific inductive-bias and evaluation-protocol refinement. It should not be described as a full urban canopy model, a PDE solver, or causal proof of urban-process realism.