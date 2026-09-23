# Variable-wise refinement diagnosis (#53)

The #52 validation-only oracle found worsening mean one-step errors with depth,
and no missed-delayed-benefit examples in the examined cases. Diagnose the actual
updates before changing the solver or halting algorithm.

For each normalized variable and validation case, let e=Y_k-Y_true and
 d=Y_(k+1)-Y_k. With normalized cosine-latitude weights:

    MSE_after - MSE_before = 2*mean(e*d) + mean(d*d)

A positive cross term indicates the update is not locally reducing squared
error. A negative cross term with positive total change indicates overshoot.
These are algebraic descriptions, not causal explanations. The per-case oracle
alpha=clip(-mean(e*d)/mean(d*d),0,1) uses true future labels and is strictly a
RETROSPECTIVE diagnostic. It never changes the forecast, model, threshold, or
inference policy. Zero-vector cosine values are encoded as zero with masks.

The workflow reuses the original #50 checkpoint and exact reconstructed data
identity; three seeds,24 balanced validation cases,K0..3, no training or test
selection. Results stay separated by variable; no mixed-physical-unit total.
A ten-minute CPU bound is enforced; source-cloud networking is forbidden after
artifact retrieval. The code snapshot and raw case/round measurements are kept.

    python scripts/diagnose_r7_corrections.py --artifact-root <study50> --out outputs/new_audit

A stronger optimization control is independent work. Neither a useful oracle
nor lower training loss establishes deployable improvements or publication SOTA.
