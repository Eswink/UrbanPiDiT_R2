"""Frozen prose constants for the spatial-gradient-loss round.

Kept out of the driver so the driver stays inside the R-051 600-line hard cap; the
values are imported by ``scripts/study_r7_s3_spatial_gradient_loss.py`` and are part
of its execution-file identity.
"""
from __future__ import annotations

DESIGN_DECISION_TEXT = (
    "Single factor: add a frozen first-difference spatial-gradient penalty to the registered deep-supervised MSE. "
    "Per physical 6 h step and per draft, L = L_mse + 0.1 * L_grad over interior_32 adjacent pairs (4160 H + 4160 W; "
    "H weight = mean cos(lat), W weight = row cos(lat), normalized over selected pairs; no sweep). "
    "The K-axis deep weights, final_weight=2, 12-step physical weights, wide 129x129 input, interior_32 supervision, "
    "2400 updates, K=4, depth 4, dim 192, the migrated parent, seeds, scoring region and climatology denominator are unchanged.")
DECISION_RULE_TEXT = (
    "Predeclared exit, no new numeric threshold: (a) candidate 72 h interior_32 t2m skill > 0 on all three seeds; "
    "(b) paired 72 h candidate-minus-control RMSE < 0 on all three seeds (control = pinned record "
    "s3-anchor-isolation-result control arm); (c) candidate 48 h skill > 0 on all three seeds; "
    "(d) seed-mean candidate-minus-control delta at 6/12/24 h <= 0; (e) identity bridge: control-minus-registered-dose "
    "RMSE inside the anchor round's -4.8e-6..+1.8e-5 K band. "
    "All of (a)-(e) -> request independent review. Any of (a)-(d) false -> permanently close the "
    "spatial-gradient-loss branch (no lambda sweep, no seed addition, no update change, no combination with closed "
    "axes, no rename). (e) false is identity/engineering unresolved, not a scientific verdict. Report all five leads "
    "and all 17 variables; no scientific pass.")
HYPOTHESIS_TEXT = (
    "If the registered MSE's smoothing bias is binding at 2400 updates, adding the frozen lambda_grad=0.1 "
    "first-difference penalty on the same interior_32 supervision should lower the paired 72 h interior_32 t2m RMSE "
    "vs the pinned control on all three seeds without raising the 6/12/24 h seed-mean delta. "
    "If the paired 72 h delta is not negative on all three seeds, or any short lead degrades, the hypothesis is false and the branch closes.")
DIFFERENCE_TEXT = (
    "One factor only: the training objective gains a frozen 0.1 * L_grad first-difference penalty on every draft of every physical step. "
    "The masked latitude-weighted MSE, normalized linspace(1,2,5)/sum K-axis deep weights, final_weight=2, "
    "registered physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), wide 129x129 input, interior_32 mask, 2400 updates, "
    "K=4, depth 4, dim 192, migrated v3-BD 1600 parent, LR 2e-5, warmup 10, FP32, batch 1 and clip 1 are held fixed; "
    "the model forward is untouched.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32, one ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "control and external reference are pinned registered arms (s3-anchor-isolation-result control, s3-wide-interior-dose), not retrained inside this protocol",
    "lambda_grad=0.1 is a single frozen design coefficient; no sweep or tuning is performed, so the response curve in lambda is unknown",
    "the candidate continues migration attempt04; the parent was trained without the gradient penalty, so this round prices a route from that parent, not a from-scratch retrain",
    "the penalty acts on normalized forward differences of all 17 channels inside interior_32; cross-channel or physical-unit gradients are not separately weighted",
    "interior_32 leaves the outer ring unconstrained; only interior_32 is scored, and all 17 variables are reported without cross-unit averaging",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident; validation split only and the test split stays sealed until a later preregistered read",
]
