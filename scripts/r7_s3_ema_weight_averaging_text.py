"""Frozen narrative for the S3 EMA weight-averaging round (implementation only)."""
from __future__ import annotations

DESIGN_DECISION_TEXT = (
    "Single factor: the weight source of the scored checkpoint. The candidate trains the registered "
    "wide/interior_32 recipe for exactly 2400 updates and then scores the plain EMA of its trainable "
    "parameters, decay=0.999, initialized from the migrated v3-BD 1600 parent, updated after every "
    "optimizer step, with no bias correction and no warm-up. The final update-2400 optimizer weights "
    "are still saved but are not scored. Everything else is held word for word: the forward pass, the "
    "deep-supervised latitude/area MSE, the data, AdamW, LR 2e-5, warmup 10, 2400 updates, K=4, "
    "depth=4, dim=192, the registered sparse physical weights, the K-axis deep weights, seeds "
    "41/42/43, the interior_32 scoring region and the climatology denominator. The paired control is "
    "the pinned s3-anchor-isolation-result control arm (same recipe, final-weight scoring) and the "
    "external reference is the pinned s3-wide-interior-dose arm; neither is retrained here.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the candidate's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2 at 6/12/24/48/72 h, (b) the same-seed paired delta of the candidate "
    "minus the pinned control's interior_32 t2m RMSE, (c) the candidate minus the pinned "
    "s3-wide-interior-dose arm, and (d) the control minus that dose arm (the identity bridge), at all "
    "five leads; all 17 variables are also reported, but the exit conditions (a)-(d) are on "
    "interior_32 t2m only. The predeclared exit is (a) 72 h skill > 0 on "
    "all three seeds, (b) 72 h candidate-minus-control delta < 0 on all three seeds, (c) 48 h skill > "
    "0 on all three seeds, (d) the 6/12/24 h seed-mean delta <= 0, and the bridge (e) inside the "
    "anchor round's -4.8e-6..+1.8e-5 K band. All of (a)-(e) asks for independent review; any failure "
    "of (a)-(d) permanently closes plain EMA decay=0.999 without a sweep, extra seed, update change, "
    "combination with a closed axis, rename, or a BEMA/SWA/checkpoint-averaging substitute; (e) alone "
    "is identity_unresolved, an engineering identity failure rather than a scientific verdict.")
HYPOTHESIS_TEXT = (
    "If the registered 2400-update endpoint's residual error is partly optimization noise rather than "
    "a stable weight deficit, then replacing the update-2400 optimizer weights with a decay=0.999 "
    "parameter EMA taken over the same trajectory should lower the same-seed 72 h interior_32 t2m "
    "RMSE relative to the pinned final-weight control on all three seeds while keeping 6/12/24 h no "
    "worse and 48 h skill positive, and should push 72 h skill past zero. If the 72 h paired delta is "
    "nonnegative on any seed or the short leads get worse, plain EMA is not a useful scoring checkpoint "
    "and the branch is closed; BEMA, SWA and checkpoint averaging are not a remedy for the same branch.")
DIFFERENCE_TEXT = (
    "One factor only: the scored checkpoint's weights. Candidate = plain EMA (decay=0.999, initialized "
    "from the migrated v3-BD 1600 parent, every trainable parameter, updated after every optimizer "
    "step, no bias correction, no warm-up) of the same 2400-update trajectory; control = that "
    "trajectory's update-2400 optimizer weights (pinned s3-anchor-isolation-result control, not "
    "retrained). Both arms use the same wide 129x129 input, the same interior_32 mask (mask_sha256 "
    "4a2a576b...), 2400 updates, K=4, depth 4, dim 192, the registered physical weights "
    "(1,.5,0,.5,0,0,0,.5,0,0,0,.5), the registered K-axis deep weights linspace(1,2,5)/sum with "
    "final_weight=2, the migrated parent attempt04, seeds 41/42/43, LR 2e-5, warmup 10, FP32, batch 1 "
    "and clip 1. There is no model-code change, so model_code_sha256 is unchanged and no new parent "
    "migration is needed; only training code and the new driver change.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "the candidate and the pinned control are separate GPU trajectories that share the parent, seeds, "
    "data and the 2400-update/per-arm plan budget but are not bit-identical training runs; the "
    "wall-clock envelope differs (candidate 21600/43200/86400 s, two-arm control "
    "43200/86400/115200 s), and the paired delta is not a same-trajectory switch",
    "the EMA is the plain un-debiased average; decay=0.999 over 2400 updates has an effective window "
    "of about 1000 updates, and the round does not sweep decay",
    "the control is a pinned registered arm (s3-anchor-isolation-result), not retrained inside this "
    "protocol; the external dose reference (s3-wide-interior-dose) is pinned as well",
    "interior_32 supervision leaves the outer ring unconstrained during training; only interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident; validation "
    "split only and the test split stays sealed until a later preregistered read",
]
