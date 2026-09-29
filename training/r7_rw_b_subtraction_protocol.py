"""The frozen protocol and the pre-declared comparison for the RW-B subtraction round.

This module exists so that *what was registered before the run* is one readable object:
the four arms, the switch keys, the single pre-declared comparison with its branch rule,
and the protocol body whose digest is the pin. ``scripts/study_r7_72_rw_b_subtraction.py``
imports it and writes it with ``'x'`` before the first optimizer step; keeping the
registration here rather than inline in the driver means the registration can be read
(and its digest recomputed) without importing a training script.

Nothing here measures anything. The decision text below is the round's own, copied from
``docs/goals/main-model-v2-rw-b-subtraction.md`` section 3, and it is frozen: the branch
rule is applied to the numbers after the run, never re-picked from them.
"""
from __future__ import annotations

from pathlib import Path

SEEDS = (41, 42)
UPDATES = 400
BATCH_SIZE = 2
LR = 2e-4
WEIGHT_DECAY = 1e-4
CLIP = 1.0
WARMUP_UPDATES = 80
MINIMUM_LR_RATIO = 0.1
VALIDATION_EVERY = 100
EARLY_STOPPING_PATIENCE = 4
MINIMUM_IMPROVEMENT = 0.001
VALIDATION_LEADS = (6,)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 3
PROCESS_WEIGHT = 0.0
EVALUATION_MAX_SAMPLES = 64
COMPARATOR_DEPTH = 0
CORRECTION_PROBE_SAMPLES = 8
DEADLINE_SECONDS = 1800.0

BASE_ARM_CONFIG = {"architecture": "window", "dim": 192, "depth": 4, "heads": 4,
                   "window_size": 4, "patch_size": 2, "dropout": 0.0,
                   "anchored_processes": 8, "free_processes": 8,
                   "use_forecast_feedback": True, "spacetime_inputs": True,
                   "default_reasoning_steps": REASONING_STEPS}
RW_A_CONFIG = dict(BASE_ARM_CONFIG, positional_process_readout=True)
RW_B_CONFIG = dict(RW_A_CONFIG, local_solver_state=True)
ARMS = (
    ("process_spacetime_rwa", "process", RW_A_CONFIG),
    ("process_local_solver", "process", RW_B_CONFIG),
    ("process_local_solver_no_gate_proposal", "process",
     dict(RW_B_CONFIG, solver_gate_proposal=False)),
    ("process_local_solver_no_recurrence", "process",
     dict(RW_B_CONFIG, solver_state_recurrence=False)),
)
ARM_NAMES = tuple(entry[0] for entry in ARMS)
SWITCH_KEYS = (("spacetime_inputs", False), ("spacetime_field_mode", "fields"),
               ("positional_process_readout", False),
               ("local_solver_state", False), ("source_role_markers", False),
               ("solver_state_recurrence", True), ("solver_gate_proposal", True))
RW_A_ARM = "process_spacetime_rwa"
RW_B_ARM = "process_local_solver"
NO_GATE_ARM = "process_local_solver_no_gate_proposal"
NO_RECURRENCE_ARM = "process_local_solver_no_recurrence"
PAIRS = ((NO_GATE_ARM, RW_A_ARM), (NO_RECURRENCE_ARM, RW_A_ARM), (RW_B_ARM, RW_A_ARM))
PRIMARY_VARIABLE = "t2m"
PRIMARY_LEADS = (6, 12, 24, 48, 72)
SEGMENT_NOTE = ("M2 two-month segment, 2016-01-01..2016-02-29, 240 consecutive 6-hourly "
                "steps of real ERA5, 17 channels on a 65x65 grid; train [01-01, 02-17), "
                "val [02-17, 02-23); the test block is never read")
FLOP_CONVENTION = ("FlopCounterMode over one forward pass under torch.enable_grad(): "
                   "linear/conv/matmul and aten-dispatched attention matmuls are counted, "
                   "elementwise and normalization ops are not, and no parameter hooks are "
                   "used (SDPA is parameterless and a hook would undercount attention). "
                   "Backward is measured, never assumed to be 2x forward.")
PRIMARY_DECISION_TEXT = (
    "Read the seed-paired mean delta of t2m RMSE per lead for each leave-one-out arm "
    "against 'process_spacetime_rwa', only where the comparator's per-seed sign rule "
    "makes the cell sign-consistent (disagreement is unresolved and is never averaged "
    "into a verdict). A negative delta means the leave-one-out arm has the lower RMSE. "
    "This round registers one single comparison in two roles: "
    "'process_local_solver_no_gate_proposal - process_spacetime_rwa' is the negative "
    "control (Z's recurrence is on but the proposal and gate do not reach the output), "
    "and 'process_local_solver_no_recurrence - process_spacetime_rwa' is the primary "
    "question (the gate and the anchored proposal are applied to a Z that does not "
    "advance). The branch rule is frozen before the run: if the negative control itself "
    "shows a same-sign degradation the protocol or the weights are confounded and no "
    "attribution is reported; otherwise, if the primary's 48 h and 72 h cells are no "
    "longer worsened, the 48/72 h degradation is attributed to (b) the cross-step "
    "recurrence of Z, and if they are still worsened it is attributed to (a) the "
    "per-position gate and the anchored proposal, there being nothing else left "
    "applied. If either arm lands unresolved the result is 'cannot attribute'. 'RW-B - "
    "RW-A' is reported beside both as the round's own reference and is never re-picked "
    "after the numbers exist. No threshold beyond the comparator's own sign rule is "
    "introduced and no existing threshold is relaxed."
)
LIMITATIONS = [
    "one bounded four-arm run at 400 updates, not a convergence or SOTA comparison",
    "two seeds: sign agreement across two seeds is consistency, not significance, and no "
    "significance threshold is introduced or relaxed; two seeds are weaker than the "
    "three the earlier rounds used",
    "the arms are NOT capacity-matched: RW-A and RW-B each add parameters and FLOPs, so "
    "every pair here is capacity-confounded and only the measured counts say by how much",
    "the leave-one-out arms remove a piece at training time instead of reusing the "
    "archived checkpoints, because the gate+proposal-removed row would otherwise route "
    "an arm through a correction head that arm never trained",
    "the reference arm RW-A is retrained in this round rather than reused from the "
    "previous one: the model code digest and the protocol differ, so the two rounds' "
    "numbers are never pooled",
    "the correction probe reports error/update geometry on validation windows only, and "
    "its retrospective damping uses future truth: a diagnostic, never a deployable rule",
    "one winter segment of one year in one region: no seasonal, cross-year or cross-region "
    "conclusion is testable",
    "the space-time conditioning pathway is on in every arm, so this round says nothing "
    "about the pathway itself",
    "validation-split only; the test split stays sealed and was never opened for selection",
    "the val climatology is the store's 8-bucket (month, hour) train-only mean, not a "
    "strong seasonal climatology",
]


def refused_test_manifest(manifest: Path) -> None:
    """This round is validation-only; a test manifest must never reach the reader."""
    if Path(manifest).name == "test.jsonl":
        raise ValueError("this round is validation-only: the test split stays sealed")


def protocol_payload(manifests_dir, identity, channels, measured):
    """The frozen protocol. Identical for every seed; its digest is the pin."""
    from training.r7_arm_harness import arm_config
    from training.r7_experiment import canonical_digest

    body = {
        "format": "r7-72-rw-b-subtraction-protocol-v1",
        "frozen_before_any_step": True,
        "issue": "#72 M2-B follow-up (RW-B subtraction: gate+anchored proposal / Z "
                 "recurrence)",
        "objective": ("attribute the registered 48/72 h degradation of RW-B to the "
                      "per-position gate with the anchored proposal, or to the cross-step "
                      "recurrence of the solver state, by removing one piece at a time "
                      "under the same protocol family"),
        "data": {
            "store": str(Path(manifests_dir).parent / "cache.zarr"),
            "train_manifest": str(Path(manifests_dir) / "train.jsonl"),
            "val_manifest": str(Path(manifests_dir) / "val.jsonl"),
            "test_manifest": str(Path(manifests_dir) / "test.jsonl"),
            "test_read": False, "test_policy": "sealed; only the val split is scored",
            "data_identity": str(identity),
            "split_mode": "time_ranges (docs/decisions/0005-*.md, 0008-*.md)",
            "normalization": "store train-only centered mean/std, applied at read time",
            "segment": SEGMENT_NOTE},
        "seeds": list(SEEDS),
        "seed_policy": ("declared before running; every declared seed is reported "
                        "whether or not it favours any arm"),
        "arms": [{"name": name, "kind": kind,
                  "model_config": arm_config(kind, channels, config),
                  "switches": {key: config.get(key, default) for key, default in SWITCH_KEYS},
                  "parameters": measured[name]["parameters"],
                  "forward_flops": measured[name]["forward_flops"],
                  "forward_backward_flops": measured[name]["forward_backward_flops"]}
                 for name, kind, config in ARMS],
        "primary_registration": {
            "registered_before_first_optimizer_step": True,
            "frozen_before_any_step": True,
            "variable": PRIMARY_VARIABLE,
            "leads_hours": list(PRIMARY_LEADS),
            "pairs": [{"focus": focus, "baseline": RW_A_ARM,
                       "reading": f"{focus} - {RW_A_ARM}", "role": role}
                      for focus, role in ((NO_GATE_ARM, "negative control"),
                                          (NO_RECURRENCE_ARM, "primary question"),
                                          (RW_B_ARM, "round reference"))],
            "decision_text": PRIMARY_DECISION_TEXT,
            "branch_rule": {
                "stop_if_negative_control_worsened": (
                    "the negative control shows a same-sign degradation: the protocol or "
                    "the weights are confounded, so no attribution is reported"),
                "attribute_to_recurrence_if": (
                    "the primary question's 48 h and 72 h cells are no longer worsened"),
                "attribute_to_gate_proposal_if": (
                    "the primary question's 48 h and 72 h cells are still worsened, since "
                    "nothing else is left applied"),
                "cannot_attribute_if": "either leave-one-out arm lands unresolved",
            },
            "secondary_reporting": ("all 17 variables x 5 leads are reported for every "
                                    "pair, and the aggregate improved/worsened/unresolved "
                                    "counts are reported separately from the primary "
                                    "verdict; the primary is never re-picked after the "
                                    "run"),
            "aggregate_rule": ("the comparator's per-seed sign rule at depth "
                               f"{COMPARATOR_DEPTH}; a cell is improved/worsened only "
                               "when every declared seed agrees in sign, otherwise it "
                               "is unresolved"),
        },
        "arm_pairing": {
            "shared_initialization_anchor": RW_A_ARM,
            "rule": ("every arm is constructed from the same seed with the added modules "
                     "built last under a rewound stream, so every tensor two arms share "
                     "starts bitwise identical; the reference arm's state_dict is loaded "
                     "into each other arm before its first step through the trainer's "
                     "transfer rule, and the applied/ignored names are recorded"),
            "explicitly_not_matched": ["parameter count", "forward/backward FLOPs",
                                       "wall time", "peak memory"]},
        "comparisons": [{"focus": focus, "baseline": baseline, "depth": COMPARATOR_DEPTH}
                        for focus, baseline in PAIRS],
        "shared_controls": {
            "optimizer": f"AdamW lr {LR} weight_decay {WEIGHT_DECAY} (fixed in the runner)",
            "lr_schedule": (f"linear warmup over {WARMUP_UPDATES} updates to the peak, then "
                            f"cosine decay to {MINIMUM_LR_RATIO} of peak at {UPDATES}"),
            "max_updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "clip": CLIP,
            "validation_every": VALIDATION_EVERY,
            "validation_lead_hours": list(VALIDATION_LEADS),
            "early_stopping": (f"stop when {EARLY_STOPPING_PATIENCE} consecutive validation "
                               f"checks fail to improve by {MINIMUM_IMPROVEMENT:.1%} "
                               "relative; reads validation only"),
            "checkpoint_selection_rule": ("lowest mean latitude-weighted normalized "
                                          "validation MSE; ties keep the earlier one"),
            "loss": "latitude-weighted MSE, streamed truncated BPTT",
            "reasoning_steps": REASONING_STEPS,
            "process_weight": PROCESS_WEIGHT,
            "sample_order": "torch.randperm(len(dataset), generator=manual_seed(seed+epoch))",
            "evaluation_leads_hours": list(EVALUATION_LEADS),
            "evaluation_policy": ("free autoregressive rollout on val, one run per (arm, "
                                  "seed, lead) with the other leads held fixed"),
            "evaluation_max_samples": EVALUATION_MAX_SAMPLES,
            "selection_split": "val only; test is sealed and never read",
            "comparison": ("training/r7_coreasoning_compare.summarize/compare (the "
                           f"#60-fixed comparator), paired per seed at depth {COMPARATOR_DEPTH}"),
            "comparison_rule": ("a cell counts as improved/worsened only when every "
                               "declared seed's delta agrees in sign; disagreement is "
                               "unresolved and never averaged away"),
            "correction_probe": (f"per-step error/update geometry over "
                                 f"{CORRECTION_PROBE_SAMPLES} validation windows, "
                                 "training/r7_correction_diagnostic.collect_correction_terms"),
            "wall_clock_limit_seconds": DEADLINE_SECONDS,
        },
        "cost_reporting": {
            "tables": ["arm_table.csv: parameters, forward and forward+backward FLOPs",
                       "training_table.csv: wall clock and seconds per update",
                       "memory_table.csv: peak allocated and reserved bytes",
                       "rmse_table.csv: physical RMSE and climatology skill, all 17 "
                       "variables x 5 leads x 4 arms x every seed",
                       "case_table.csv: cases scored per arm and lead"],
            "note": ("forward-only FLOPs never stand in for training cost; the four cost "
                     "views are reported together"),
        },
        "flop_convention": FLOP_CONVENTION,
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }
    return dict(body, protocol_sha256=canonical_digest(body))


def verified_registration(protocol_path):
    """Re-read the protocol from disk and check the frozen registration matches.

    Runs *before the first optimizer step*, so "the verdict was written down before any
    step" is a check rather than a claim about the run afterwards.
    """
    import json

    from training.r7_experiment import canonical_digest

    frozen = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != canonical_digest(
            {key: value for key, value in frozen.items() if key != "protocol_sha256"}):
        raise RuntimeError("the protocol on disk is not the one this run measured")
    registration = frozen["primary_registration"]
    declared = tuple((entry["focus"], entry["baseline"])
                     for entry in registration["pairs"])
    if (registration["variable"] != PRIMARY_VARIABLE
            or tuple(registration["leads_hours"]) != PRIMARY_LEADS
            or declared != ((NO_GATE_ARM, RW_A_ARM), (NO_RECURRENCE_ARM, RW_A_ARM),
                            (RW_B_ARM, RW_A_ARM))):
        raise RuntimeError("the primary registration on disk is not the declared one")
    if not registration["decision_text"].strip():
        raise RuntimeError("the primary registration carries no decision text")
    if not registration.get("branch_rule"):
        raise RuntimeError("the primary registration carries no branch rule")
    return frozen
