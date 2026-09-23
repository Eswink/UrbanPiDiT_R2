# Reviewer 2/3 Response Mapping for V5.1

This document maps V5.1 artifacts to likely TGRS reviewer concerns. It is intended as a rebuttal construction aid.

## Concern: methodological novelty is not domain-specific enough

Response:

V5.1 adds domain-specific structure around urban morphology and micro-meteorology instead of only increasing model capacity.

Artifacts:

| Evidence | Location |
|---|---|
| micro-meteorology coupling | `models/micromet_coupling.py` |
| morphology and wind-aware graph | `models/morphology_graph.py` |
| V5.1 full configuration | `configs/urbanpidit_v51_micromet.yaml` |
| mechanism ablation configuration | `configs/mechanism_ablation.yaml` |
| method description | `docs/method_v51_micromet_coupled_graph.md` |

Reviewer-facing wording:

> We introduced an ablatable micro-meteorology-inspired coupling operator and a morphology/wind-aware graph that uses static urban form and historical wind context, while preserving the original V4 training pipeline for controlled comparison.

## Concern: improvements may come from unfair static information

Response:

Static information is made explicit, schema-controlled, and perturbable.

Artifacts:

| Evidence | Location |
|---|---|
| categorical/continuous static schema | `configs/*.yaml` |
| static perturbation protocol | `experiments/run_static_perturbation.py` |
| perturbation config | `configs/static_perturbation.yaml` |
| fair baseline protocol | `configs/fair_benchmark.yaml` |

Reviewer-facing wording:

> We report whether each model receives static fields and include perturbation controls that preserve or destroy different aspects of static morphology.

## Concern: physical consistency is opaque

Response:

Physical evaluation is split into feasibility, structure, and process proxy groups.

Artifacts:

| Evidence | Location |
|---|---|
| grouped physical consistency | `losses/physical_consistency.py` |
| process eval entry | `experiments/run_process_consistency_eval.py` |
| process eval config | `configs/process_consistency_eval.yaml` |
| protocol document | `docs/physical_consistency.md` |
| claim guardrail | `docs/physical_claims_guardrail.md` |

Reviewer-facing wording:

> We separately report physical-feasibility violations, spatial/spectral structure mismatch, and process-proxy consistency rather than collapsing them into a single uninterpretable score.

## Concern: graph module is not interpretable

Response:

The graph has observable diagnostics and separate ablation switches.

Artifacts:

| Evidence | Location |
|---|---|
| graph implementation | `models/morphology_graph.py` |
| graph diagnostics entry | `experiments/run_graph_diagnostics.py` |
| graph diagnostics config | `configs/graph_diagnostics.yaml` |
| graph protocol | `docs/graph_design_and_ablation.md` |

Reviewer-facing wording:

> We inspect graph degree, same-landcover edge ratio, entropy, wind anisotropy, and roughness blocking to verify that the graph changes relation structure in the intended domain-specific direction.

## Concern: risk of label leakage

Response:

V5.1 modules are constrained to history, current diffusion state, and static fields. `hour_of_day` is config-gated and derived only from historical/context anchors.

Artifacts:

| Evidence | Location |
|---|---|
| hour gate in data layer | `data/loader.py` |
| config gate | `data.expose_hour` in `configs/*.yaml` |
| safe model construction | `models/model_registry.py` |
| tests | `tests/test_climatology.py`, `tests/test_process_consistency.py` |

Reviewer-facing wording:

> No V5.1 module consumes future labels or future clean states as side information; future targets are used only in supervised loss computation.

## Concern: experiments may be hard to reproduce

Response:

V5.1 experiment entries are manifest-first and dry-run by default.

Artifacts:

| Experiment | Config | Entry |
|---|---|---|
| mechanism ablation | `configs/mechanism_ablation.yaml` | `experiments/run_mechanism_ablation.py` |
| graph diagnostics | `configs/graph_diagnostics.yaml` | `experiments/run_graph_diagnostics.py` |
| process consistency | `configs/process_consistency_eval.yaml` | `experiments/run_process_consistency_eval.py` |
| static perturbation | `configs/static_perturbation.yaml` | `experiments/run_static_perturbation.py` |
| smoke forward | `configs/urbanpidit_v51_micromet.yaml` | `experiments/smoke_test_v51.py` |

Reviewer-facing wording:

> We provide manifest-first experiment scripts so that configurations, commands, and output paths can be audited before long-running training or evaluation.

## Suggested response table

| Reviewer issue | Code artifact | Planned evidence | Claim strength |
|---|---|---|---|
| domain novelty | micromet + graph modules | mechanism ablation | methodological contribution |
| static fairness | fair baseline + perturbation | static degradation table | controlled comparison |
| physical plausibility | feasibility/process metrics | proxy diagnostic table | physical-feasibility support |
| interpretability | graph diagnostics | graph statistics | mechanism transparency |
| leakage | gated hour/static/history boundary | tests + protocol | reproducibility guardrail |