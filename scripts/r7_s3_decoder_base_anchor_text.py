"""Frozen prose for the S3 decoder-base-anchor round (kept out of the driver for R-051)."""
from __future__ import annotations

DESIGN_DECISION_TEXT = (
    "Single factor: the decoder's base semantics. The registered decoder is persistence-anchored "
    "(forecast = X_t + head(...), X_t = history[:, -1], and every local-solver proposal is "
    "Y = X_t + Decoder(Z)). The candidate keeps the same parent, weights, layers, output scale, "
    "last layer, near-zero-initialized output head, 2400 updates, LR 2e-5, warmup 10, K=4, "
    "depth 4, dim 192, data, loss, physical weights, seeds and scoring domain, and changes only "
    "decoder_base_mode from persistence to zero. The same tensor sets X_t is replaced by zeros, so "
    "the direct head output becomes its tendency and the local-solver proposal becomes its own "
    "decoder output. The climatology anchor stays installed but lazy in both arms "
    "(anchor_output_prior=False, anomaly_feedback=False); zeroing its table changes neither arm.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Predeclared exit, no new thresholds: (a) the "
    "candidate's 72 h interior_32 t2m skill is > 0 on seeds 41/42/43; (b) same-seed candidate "
    "minus control 72 h RMSE is < 0 on all three seeds (negative = candidate better); (c) the "
    "candidate's 48 h skill is > 0 on all three seeds; (d) the seed-mean candidate minus control "
    "RMSE at 6/12/24 h is <= 0; (e) the bridge: the pinned control's 15 seed x lead RMSE "
    "differences against the pinned s3-wide-interior-dose arm stay inside the anchor round's "
    "-4.8e-6 ... +1.8e-5 K band. All of (a)-(e) -> independent review only, never automatic S4 "
    "promotion. Any of (a)-(d) false -> permanently close the zero-base decoder branch. An (e) "
    "failure only sets identity_unresolved=true: it is an engineering identity failure, not a "
    "scientific judgement.")
HYPOTHESIS_TEXT = (
    "If the persistence base is load-bearing in this fine-tune instance, then removing it while "
    "holding the residual parent, the near-zero output head and the 2400-update budget fixed will "
    "make the candidate worse than the pinned persistence control at 6/12 h and will not leave "
    "72 h interior_32 t2m skill positive on all three seeds. If instead the candidate is better on "
    "all three seeds at 72 h (paired delta negative everywhere) with positive 72 h skill, then the "
    "persistence base is not load-bearing in this instance, which is evidence about this residual "
    "fine-tune, not about absolute forecasting in general.")
DIFFERENCE_TEXT = (
    "One factor only: decoder_base_mode persistence -> zero. Control is the pinned "
    "s3-anchor-isolation-result control arm (persistence base, decoder_base_mode absent = "
    "persistence), not retrained here. Candidate = the same spec plus decoder_base_mode='zero'. "
    "Both arms carry the same published 16-bucket train-only climatology table with "
    "anchor_output_prior=False and anomaly_feedback=False, so the anchor value reaches no output "
    "(the forward still evaluates it) and zeroing it "
    "changes neither arm. The forecast head's direct anchor and R7ForecastOutput.base_state are the "
    "same tensor and are zeroed together, so the local solver's proposal_anchor is zero as well. "
    "The output head's last layer is not reset, change_scale_mode is untouched, and there are no "
    "new parameters: both modes share one state_dict key/shape set. Same wide 129x129 input, same "
    "interior_32 mask, same 2400 updates, K=4, depth 4, dim 192, same registered physical weights "
    "(1,.5,0,.5,0,0,0,.5,0,0,0,.5), same migrated v3-BD 1600 parent, same LR 2e-5, warmup 10, FP32, "
    "batch 1 and clip 1, same seeds and same scoring domain/climatology denominator.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "the control is a pinned previous-code reading (s3-anchor-isolation-result); the new-code "
    "persistence path is only asserted bitwise-equivalent, so the comparison mixes base semantics "
    "with run-to-run training variance",
    "zero-base is not a from-scratch absolute model: it inherits the residual parent's near-zero "
    "output head and trains for 2400 updates, so a negative result cannot be extrapolated to "
    "'absolute prediction is infeasible'",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident",
    "the climatology table remains installed but is read by nothing in either arm; zeroing it is "
    "an identity control, not a mechanism",
    "validation split only; the test split stays sealed until a later preregistered read",
]
