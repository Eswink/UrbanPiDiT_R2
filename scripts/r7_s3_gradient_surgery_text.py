"""Frozen prose for the S3 short/long gradient-surgery round (kept out of the driver for R-051)."""
from __future__ import annotations

DESIGN_DECISION_TEXT = (
    "Single factor: training-time gradient surgery. The registered 12-step interior_32 "
    "long-rollout loss is the weighted sum of the physical-step losses; this round keeps that "
    "loss definition, the model, data, parent, seeds, 2400 updates, LR 2e-5, warmup 10, K=4, "
    "depth 4, dim 192, batch 1, clip 1, deep K-axis supervision, physical weights "
    "(1,.5,0,.5,0,0,0,.5,0,0,0,.5), scoring domain, climatology denominator and test policy "
    "unchanged, and changes only the backward: gradient_surgery 'off' (registered) -> 'pcgrad' "
    "(candidate). Group A is weighted steps 0/1/3 (6/12/24 h, weights 1.0/0.5/0.5) and group B "
    "is weighted steps 7/11 (48/72 h, weights 0.5/0.5), both from the same per_step_losses "
    "tensor; zero-weight steps belong to neither group but still advance the physical history "
    "exactly as registered. Two autograd.grad calls per update produce g_A and g_B over all "
    "trainable parameters; reductions run in float64; if dot(g_A,g_B) < 0 then "
    "g_A <- g_A - (dot/||g_B||^2) g_B, otherwise unchanged; update = g_A + g_B; then the "
    "existing clip_grad_norm_ (clip=1) and optimizer.step. The order is fixed A -> B (a "
    "predeclared one-sided specialization: only g_A is projected onto g_B's normal plane; "
    "g_B is unchanged, and this is not the symmetric two-way PCGrad Algorithm 1; no random "
    "order is added).")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Predeclared exit, no new thresholds: (a) the "
    "candidate's 72 h interior_32 t2m skill is > 0 on seeds 41/42/43; (b) same-seed candidate "
    "minus paired control 72 h RMSE is < 0 on all three seeds (negative = candidate better); "
    "(c) the candidate's 48 h skill is > 0 on all three seeds; (d) the seed-mean candidate "
    "minus control RMSE at 6/12/24 h is <= 0; (e) the bridge: the pinned control's 15 seed x "
    "lead RMSE differences against the pinned s3-wide-interior-dose arm stay inside the anchor "
    "round's -4.8e-6 ... +1.8e-5 K band. All of (a)-(e) -> independent review only, never "
    "automatic S4 promotion. Any of (a)-(d) false -> permanently close the short/long "
    "gradient-surgery branch (no group/coefficient/order sweep, no extra seed, no update/LR/"
    "clip change, no combination with a closed axis). An (e) failure only sets "
    "identity_unresolved=true: an engineering identity failure, not a scientific judgement. "
    "If every update of every seed reports zero conflict and zero projection, the candidate "
    "update is the registered gradient up to FP association; label the outcome unresolved (or "
    "negative if worse), not-candidate, and close the branch without tuning to force conflicts.")
HYPOTHESIS_TEXT = (
    "On the registered 12-step interior_32 long-rollout recipe, conditioning the update on "
    "short/long gradient conflict (project the short-horizon group g_A onto the long-horizon "
    "group g_B's normal plane only when dot(g_A,g_B)<0, then add) will lower 72 h t2m "
    "interior_32 RMSE relative to the pinned registered control on all three seeds without "
    "degrading 6/12/24 h; if the groups never conflict, the candidate is the registered "
    "gradient up to FP association and the hypothesis is null.")
DIFFERENCE_TEXT = (
    "One factor only: training-time gradient surgery off -> pcgrad. The model spec is exactly "
    "the pinned s3-anchor-isolation-result control spec (no decoder_base_mode key; default "
    "persistence), the model code is unchanged (model_code_sha256 bec3859f...), and the "
    "decoder-round parent migration is reused (receipt 002f0843...; state digests "
    "7c416ccb.../857d8804.../61fdac58..., expanded 00ada5.../baed2f.../4dda64..., anchor init "
    "de0e76...). Only training/r7_long_rollout_runner.py and the new "
    "training/r7_gradient_surgery.py change in the training path, so training_code_sha256 "
    "moves. Group A = weighted steps 0/1/3, group B = weighted steps 7/11, from the same "
    "per_step_losses tensor; zero-weight steps are in neither group but still advance the "
    "history. The two-group sum is algebraically the registered total but not bitwise equal "
    "(different FP association); the report records the per-update scalar residual. Same wide "
    "129x129 input, same interior_32 mask (mask_sha256 4a2a576b...), same 2400 updates, K=4, "
    "depth 4, dim 192, same registered physical weights, same migrated v3-BD 1600 parent, same "
    "LR 2e-5, warmup 10, FP32, batch 1 and clip 1, same seeds and same scoring "
    "domain/climatology denominator.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one "
    "ROI, 17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or "
    "generalization claim",
    "the paired control is a pinned previous-code reading (s3-anchor-isolation-result) and is "
    "not retrained, so the comparison mixes gradient surgery with run-to-run training variance "
    "and a training-code version difference",
    "the order is fixed A -> B; the PCGrad paper's Appendix H reports random task order can be "
    "better when conflicts are common, and no random-order RNG is added here",
    "the two groups are not independent tasks: the late-horizon gradient flows through the "
    "earlier predicted states of the same rollout, so the group partition is not a "
    "multi-task decomposition with independent task gradients",
    "the two-group sum is algebraically the registered total but not bitwise equal to the "
    "registered scalar's backward; the per-update residual is a descriptive diagnostic",
    "the prior in-repo four-case diagnostic (docs/R7_S3_TRAIN_GRADIENT_COMPONENTS.md, "
    "weighted_gram) read positive group cosines +0.202574/+0.138378/+0.598252/+0.471881, so "
    "global conflicts may be rare and a zero-conflict null is a real possible outcome; the "
    "long-horizon group also dominated in norm (A 1.28-3.70 vs B 5.23-14.61), so the "
    "projection's leverage may be small",
    "conflict is not harm: arXiv:2609.08554v1 measures 30.6% negative pairwise variable "
    "cosines but warns that conflict frequency does not reliably predict the harmed fraction",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident "
    "and not bitwise reproducible; validation split only and the test split stays sealed",
]
