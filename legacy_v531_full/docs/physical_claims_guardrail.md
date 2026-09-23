# Physical Claims Guardrail

This document defines acceptable physical-language boundaries for UrbanPiDiT V5.1/V5.2.

## Core rule

V5.1/V5.2 provides physical-feasibility proxies and urban-morphology inductive biases. It does not provide a closed-form numerical weather model, a resolved urban canopy solver, or causal proof of urban processes. V5.2 refines the proxy protocol through schema-faithful graph inputs, residual-based drag diagnostics, and day/night diurnal diagnostics; these remain soft proxies rather than physical closure.

## Acceptable claims

| Claim | Why acceptable |
|---|---|
| "physical-feasibility proxy diagnostics" | diagnostics check selected value ranges and relations |
| "process-consistency proxies" | RH, drag, and diurnal terms are simplified proxies |
| "micro-meteorology-inspired coupling" | coupling uses morphology-conditioned residual branches |
| "urban morphology conditioned spatial prior" | graph uses static morphology and grid proximity |
| "history-wind-aware propagation" | wind-aware graph uses historical/context wind only |
| "morphology-derived learnable proxies" | V5.2 proxy fields are diagnostic/regularization aids, not measured physical parameters |
| "schema-faithful morphology graph" | graph uses separated continuous/categorical fields and reports fallback state |
| "residual-based drag consistency" | drag is evaluated through wind-speed changes from historical context |
| "day/night diurnal response proxy" | thermal response is separated by current/context hour when available |

## Claims to avoid

| Avoid | Safer replacement |
|---|---|
| "physically exact" | "physically motivated" |
| "solves urban canopy physics" | "canopy-inspired residual coupling" |
| "proves causal urban effect" | "supports an ablation-based association" |
| "fully conservative atmosphere" | "improves selected feasibility diagnostics" |
| "future-aware correction" | "history/context-conditioned correction" |
| "guaranteed physical consistency" | "reduced violations under reported proxies" |

## Metric interpretation

Physical-feasibility scores should be reported beside forecast metrics.

Correct interpretation:

- Feasibility score improves: fewer selected invalid values or relation violations.
- Structure score improves: closer selected spectral/spatial proxy behavior.
- Process score improves: closer selected simplified process proxy behavior.

Incorrect interpretation:

- A better feasibility score alone proves better forecast skill.
- A lower process loss proves full process realism.
- A graph diagnostic proves causal transport.

## Leakage guardrail

Allowed module inputs:

- historical dynamic context
- current denoising state
- static urban fields
- current/context-derived `hour_of_day` when `data.expose_hour: true`
- train-only climatology/statistics

Disallowed module inputs:

- future target `y`
- future clean state `x0`
- validation/test target statistics
- future timestamp parsed from target filename

## Reporting requirements

Reviewer-facing tables should include:

1. Forecast metrics by lead time.
2. Physical-feasibility diagnostics.
3. Process proxy diagnostics when enabled.
4. Static perturbation results.
5. Module ablations.
6. A statement that new modules are disabled unless explicitly enabled.

## Recommended paragraph

> We use physical-feasibility and process-consistency proxies as auditable diagnostics and optional regularizers. These terms evaluate selected meteorological plausibility conditions, including value bounds, dew-point/temperature relation, spatial/spectral structure, RH consistency, morphology-conditioned drag, and diurnal thermal response. They should be interpreted as proxy evidence for improved physical plausibility, not as a replacement for a full atmospheric or urban canopy model.