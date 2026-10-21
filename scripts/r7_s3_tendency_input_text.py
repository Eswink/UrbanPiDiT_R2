"""Frozen prose for the S3 explicit-tendency-input round (kept out of the driver for R-051)."""
from __future__ import annotations

DESIGN_DECISION_TEXT = (
    "Single factor: the model input representation. The registered 2400-update "
    "wide/interior_32 long-rollout recipe keeps its model spec, data, parent, seeds, loss, "
    "12-step physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), deep K-axis supervision, "
    "2400 updates, LR 2e-5, warmup 10, K=4, depth=4, dim=192, batch 1, clip 1, scoring "
    "domain, climatology denominator and test policy word for word; only "
    "tendency_input goes from False (control) to True (candidate). The encoder input "
    "becomes [X_{t-1}, X_t, D_t] instead of [X_{t-1}, X_t], with D_t = X_t - X_{t-1} in "
    "the same normalized input space. The candidate adds one separate 17->dim "
    "convolutional patch (kernel_size=stride=patch_size=2, bias=False) whose weight is "
    "zero-initialized, so the parameter delta is 17*192*2*2 = 13,056; this is the "
    "declared capacity confound, reported rather than assumed away. Because D_t is a "
    "linear function of the two frames already in the input, the branch adds no "
    "information: a null (exactly zero final weights or exactly identical readings) is a "
    "real possible outcome and is not tuned away. No tendency definition, scale or sign "
    "sweep, no extra seed, no update/LR/warmup/K/depth/dim change and no combination "
    "with a closed axis.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Predeclared exit, no new thresholds: (a) the "
    "candidate's 72 h interior_32 t2m skill is > 0 on seeds 41/42/43; (b) same-seed candidate "
    "minus pinned s3-anchor-isolation-result control 72 h RMSE is < 0 on all three seeds "
    "(negative = candidate better); (c) the candidate's 48 h skill is > 0 on all three seeds; "
    "(d) the seed-mean candidate minus control RMSE at 6/12/24 h is <= 0; (e) the bridge: the "
    "pinned control's 15 seed x lead RMSE differences against the pinned s3-wide-interior-dose "
    "arm stay inside the anchor round's -4.8e-6 ... +1.8e-5 K band. All of (a)-(e) -> "
    "candidate_state needs-review and independent review only, never automatic S4 promotion. "
    "Any of (a)-(d) false -> permanently close the explicit-tendency-input branch (no tendency "
    "definition/scale/sign sweep, no extra seed, no update/LR/warmup/K/depth/dim change, no "
    "combination with a closed axis, no rename). An (e) failure only sets "
    "identity_unresolved=true: an engineering identity failure, not a scientific judgement. "
    "Null/inert is recorded only if the final tendency-patch weight is exactly all-zero or "
    "every paired delta is exactly 0.0 at recorded precision; a small nonzero effect follows "
    "(a)-(d) and is never relabelled null.")
HYPOTHESIS_TEXT = (
    "On the frozen 2400-update wide/interior_32 long-rollout recipe, a zero-initialized "
    "explicit 6 h tendency branch D_t = X_t - X_{t-1} (same normalized input space, +13,056 "
    "parameters) added to the encoder input projection will improve 72 h interior_32 t2m skill "
    "over the pinned s3-anchor-isolation-result control on all three seeds and satisfy the "
    "predeclared (a)-(e) exit; because D_t is a linear function of the existing two frames the "
    "branch adds no information, so a null (exactly zero weights or exactly identical "
    "readings) is a real possible outcome.")
DIFFERENCE_TEXT = (
    "One factor only: tendency_input False -> True. The control spec is exactly the pinned "
    "s3-anchor-isolation-result control spec (no tendency_input and no decoder_base_mode key), "
    "and the candidate spec is that spec plus tendency_input=True; the encoder receives "
    "[X_{t-1}, X_t, D_t] and the extra 17->dim patch is zero-initialized and bias-free. Every "
    "other model layer, the persistence decoder base, the data/source, the loss, the 2400 "
    "updates, LR 2e-5, warmup 10, K=4, depth=4, dim=192, batch 1, clip 1, deep K-axis "
    "supervision, physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), seeds 41/42/43, scoring "
    "domain and climatology denominator are unchanged. Because model/*.py changes, "
    "model_code_sha256 moves to 60de3b2d..., and the committed migrate_r7_parent_state.py was "
    "rerun under the new revision into a fresh exclusive directory (receipt 73c019dd...); the "
    "per-seed migrated state digests still match 7c416ccb.../857d8804.../61fdac58..., and the "
    "expanded states add exactly the published anchor buffers and the zero tendency patch. The "
    "declared capacity confound is +13,056 parameters (3,286,037 -> 3,299,093 at dim=192, "
    "patch_size=2).")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one "
    "ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or "
    "generalization claim",
    "the paired control is a pinned previous-code reading (s3-anchor-isolation-result) and is "
    "not retrained, so the comparison mixes the input branch with a model-code version "
    "difference and run-to-run training variance",
    "D_t = X_t - X_{t-1} is a linear function of the two frames already supplied to the "
    "encoder, so the branch adds no information; only optimization/parameterization can change, "
    "and an exactly-zero final patch or exactly identical readings is a real possible null",
    "the branch adds 13,056 parameters (0.4% of 3,286,037) and extra compute; the declared "
    "capacity confound is not controlled by a parameter-matched arm",
    "the tendency patch is constructed under a rewound RNG stream, so every pre-existing "
    "parameter is initialized exactly as in the control; the tendency columns themselves start "
    "at zero, so update-0 behaviour is the control's, and any effect is a training-trajectory "
    "effect rather than an input-information effect",
    "only one tendency definition (backward difference over the registered 6 h frame spacing) "
    "and one scale (same normalized input space) are tested; no sweep is licensed by this round",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident "
    "and not bitwise reproducible; validation split only and the test split stays sealed",
]
