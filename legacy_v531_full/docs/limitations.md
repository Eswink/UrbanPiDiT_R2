# Limitations and Audit Notes

This document states the current limits of the upgraded UrbanPiDiT codebase. It is intended to keep the rebuttal and future experiments defensible.

## Empirical claims are not completed by infrastructure alone

The current upgrade provides:

- controlled static-information paths
- new urban morphology and canopy modules
- grouped physical-consistency logging
- multi-region split infrastructure
- dry-run-first experiment scripts
- light regression tests

It does not by itself prove final numerical superiority. Paper-level claims still require completed benchmark, ablation, perturbation, and leave-one-city-out runs.

## Static information remains a confound if not reported explicitly

Static fields can improve forecasts because they encode city morphology. They can also create unfair comparisons if only the proposed model receives them.

Required reporting:

- whether each model uses static fields
- which static policy is used
- whether land-cover is categorical or treated as continuous
- whether static perturbation results are included

Do not compare `same_static` proposed models only against `dynamic_only` baselines without labeling the comparison as static-augmented.

## Multi-city config is a template

`configs/multi_city.yaml` contains placeholder roots for non-Beijing cities. Before running full experiments:

- replace placeholder paths with actual processed data roots
- verify all regions expose the same dynamic/static variables or provide explicit schema reconciliation
- confirm train-only normalization and climatology are generated per region
- inspect leave-one-city-out manifests before launching training

## Physical losses are proxies

The physical-consistency loss groups are useful diagnostics, but they are not a complete atmospheric model.

Known limits:

- feasibility constraints are variable-wise and local
- FFT/gradient structure terms do not guarantee correct dynamics
- process terms are simplified proxies
- weights require sensitivity analysis

Reviewer-facing claims should avoid saying the model is physically closed or physically exact.

## Canopy coupling is an inductive bias, not a resolved canopy model

`UrbanCanopyCoupling` uses static morphology and current/historical dynamic context to provide a learnable local correction.

Limits:

- no explicit building-resolved radiation solver
- no full urban energy balance closure
- no explicit anthropogenic heat inventory unless provided as static input
- no guarantee of extrapolation to unseen morphology without cross-region validation

Correct phrasing:

- "canopy-inspired coupling"
- "urban morphology conditioned residual correction"
- "ablatable physical inductive bias"

Avoid phrasing:

- "fully resolves urban canopy physics"
- "proves physical causality"

## Graph modules need ablation evidence

The morphology and wind-aware graphs are designed to encode plausible relations:

- static similarity
- spatial proximity
- historical wind direction

But they need ablation evidence to support usefulness.

Required comparisons:

- base
- morphology graph only
- wind-aware graph only
- combined
- static perturbation controls

## Event metrics are threshold-sensitive

Event metrics depend on threshold choice. For precipitation and cloud-cover events:

- report thresholds clearly
- avoid selecting thresholds after seeing test results
- include at least one non-event aggregate metric such as RMSE/ACC

## Dry-run scripts do not replace experiment execution

Experiment scripts generate auditable manifests by default. This avoids accidental long runs, but means the output is a plan until executed explicitly.

Use `--execute` only when the data paths, compute budget, and output directories are confirmed.

## Backward compatibility risk

The upgrade is designed so old configs remain valid. Still, changes should be checked with:

- old config loading
- old dummy forward path
- static encoder path
- graph path
- canopy path
- combined path

The intended validation suite is listed in the project workflow and should be rerun after substantive edits.