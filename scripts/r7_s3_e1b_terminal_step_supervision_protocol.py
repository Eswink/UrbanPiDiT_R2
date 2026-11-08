"""The Stage E1b protocol layer: final-only draft supervision vs the registered deep form.

One sentence: at **128,528 updates = 16 epoch** on the store Stage B calibrated, does supervising
only the final draft (``drafts[:, -1]``, weight 1.0) instead of the registered deep form
(``linspace(1,2,5)/sum``, 73.333% of the mass on non-final drafts) change 72 h t2m skill?

Provenance: decision **0096** (E1b is the second of the two members the objective names, and the
ADR records in advance that a *mixed* result here closes this branch permanently -- the closed
arm's reading was already mixed, not null, so "it needs more compute" is not an available excuse);
decision **0057** (the reopening queue's "not the same family" requirement plus the new regime).

Why this is a separate module from E1a. Both rounds share the regime, the seeds and the store, and
differ in exactly one thing -- E1a moves a weight vector, E1b moves the draft-supervision
organisation. Merging them into one protocol would make a win unattributable, which is the same
reason 0092 and 0095 refused mixed axes. The shared arithmetic is therefore duplicated
deliberately, and the duplication is recorded as a cost in the ADR.

The factor travels as an argument: ``draft_supervision`` is a keyword of
``fine_tune_long_rollout`` (default ``"deep"``) and is bound into the contract at
``training/r7_long_rollout_runner.py:197``. The registered value is a constant of
``training/r7_recursive_losses.py``, which this round **does not modify**; only which of the two
declared modes is requested changes. So ``training_code_digest`` is unchanged by the round itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_curve_point_large_protocol as curve
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

FORMAT = "r7-s3-e1b-terminal-step-supervision-protocol-v1"
CANDIDATE, CONTROL = "final_only", "deep"
SEEDS = (41, 42, 43)

#: 16 epoch on 32,131 usable windows: the new regime that legitimises the reopening (0096).
UPDATES = 128528
STEPS = 4
BATCH_SIZE = 4
REASONING_STEPS = 4

#: The two organisation modes, taken from the module that defines them rather than retyped.
from training.r7_recursive_losses import (  # noqa: E402  (after sys.path is set)
    DRAFT_SUPERVISION_DEEP, DRAFT_SUPERVISION_FINAL_ONLY, DRAFT_SUPERVISION_MODES)

if (CONTROL, CANDIDATE) != (DRAFT_SUPERVISION_DEEP, DRAFT_SUPERVISION_FINAL_ONLY):
    raise RuntimeError("the arm names must be the two declared supervision modes")
if set(DRAFT_SUPERVISION_MODES) != {CONTROL, CANDIDATE}:
    raise RuntimeError(f"this round assumes exactly two modes; the module declares "
                       f"{DRAFT_SUPERVISION_MODES}")

#: The physical-weight vector is shared by both arms: E1b moves organisation only.
SPARSE_WEIGHTS = tuple(wide.PHYSICAL_WEIGHTS)

#: Slowest measured batch-4 long-run rate on this store (decision 0090's lesson: derive the
#: envelope from a pessimistic measured rate, never from a best case).
MEASURED_SECONDS_PER_UPDATE = 1.1740
MEASURED_ON = "p0025 batch-4 long run, the slowest of the four Stage B point measurements"
SCORING_ALLOWANCE_SECONDS = 2588.1

ENTRY_FILES = ("scripts/study_r7_s3_e1b_terminal_step_supervision.py",)
CODE_ARCHIVE = ROOT / "outputs/r7_s3_e1b_terminal_step_supervision_code.zip"


def budgets():
    """The duration envelope: three times the two-arm estimate, hard cap twice that."""
    per_arm = UPDATES * MEASURED_SECONDS_PER_UPDATE + SCORING_ALLOWANCE_SECONDS
    two_arms = 2 * per_arm
    per_seed = float(math.ceil(3 * two_arms / 3600) * 3600)
    return {"measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
            "measured_on": MEASURED_ON,
            "updates": UPDATES,
            "train_seconds_per_arm": UPDATES * MEASURED_SECONDS_PER_UPDATE,
            "scoring_allowance_seconds": SCORING_ALLOWANCE_SECONDS,
            "per_arm_seconds_estimate": per_arm,
            "two_arms_seconds_estimate": two_arms,
            "estimate_basis": (f"{UPDATES} updates x {MEASURED_SECONDS_PER_UPDATE} s/update at "
                               "batch 4 plus one five-lead scoring pass, per arm; two arms per seed; "
                               "an estimate from the slowest measured point, not a measurement of "
                               "this round"),
            "per_seed_seconds": per_seed,
            "planned_seconds": per_seed,
            "hard_cap_seconds": 2 * per_seed,
            "six_arm_gpu_hours_estimate": round(6 * per_arm / 3600, 3),
            "soft_rule": "soft overrun continues and is recorded; only the hard cap truncates",
            "decision": "0060"}


def arm_spec():
    """The archived process spec, unchanged: this round touches no model switch."""
    spec = dict(curve.model_spec())
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    for switch in ("climatology_anchor_spec", "anomaly_feedback", "anchor_output_prior",
                   "process_init_conditioning", "typed_evidence_mode", "global_mixing_mode",
                   "static_conditioning"):
        if spec.get(switch):
            raise RuntimeError(f"the archived process spec already carries {switch}")
    return spec


def arm_supervision(arm):
    """The one factor: the draft-supervision organisation. The weight vector is shared."""
    if arm not in (CONTROL, CANDIDATE):
        raise ValueError(f"unknown arm {arm!r}")
    return arm


def frozen_constants():
    """Everything frozen before any data or model is touched."""
    return {
        "format": FORMAT,
        "stage": "S3-THROUGHPUT-PHYSICS-E1B",
        "decision": "0096",
        "instance": str(curve.INSTANCE),
        "store": str(curve.STORE),
        "source_sha256": curve.SOURCE_SHA256,
        "store_shape": list(curve.STORE_SHAPE),
        "split_years": {"train": curve.CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(curve.TRAIN_MANIFEST),
        "val_manifest": str(curve.VAL_MANIFEST),
        "test_manifest_never_read": str(curve.TEST_MANIFEST),
        "usable_windows": curve.USABLE_WINDOWS,
        "val_cohorts": dict(curve.VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": [1], "scored_region": "full",
                   "n_grid_points": 4225,
                   "note": "the whole native 65x65 grid; interior_32 is 1x1 here"},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
                   "lr": curve.LR, "warmup": curve.WARMUP, "weight_decay": curve.WEIGHT_DECAY,
                   "batch_size": BATCH_SIZE, "clip": curve.CLIP, "bf16": False,
                   "checkpoint_every": curve.CHECKPOINT_EVERY,
                   "supervision": "full_grid", "epochs": round(UPDATES * BATCH_SIZE
                                                               / curve.USABLE_WINDOWS, 6),
                   "physical_weights": list(SPARSE_WEIGHTS),
                   "physical_weights_note": "shared by both arms; E1b moves organisation only"},
        "arms": {
            CANDIDATE: {"draft_supervision": DRAFT_SUPERVISION_FINAL_ONLY,
                        "internal_deep_supervision": False,
                        "description": "final draft only, weight 1.0; no intermediate target term"},
            CONTROL: {"draft_supervision": DRAFT_SUPERVISION_DEEP,
                      "internal_deep_supervision": True,
                      "description": "registered deep form, linspace(1,2,5) normalized"},
        },
        "factor": {"name": "terminal-step supervision organisation",
                   "control": "deep (registered, 73.333% non-final mass)",
                   "treatment": "final_only",
                   "arms_differ_only_in": ["draft_supervision"]},
        "budgets": budgets(),
        "test_read": False,
        "scientific_claim": False,
        "r": 0,
        "limitations": [
            "Treatment and control differ only in draft organisation; a win cannot be attributed "
            "to any specific intermediate draft.",
            "Three seeds give a paired comparison, not significance across hypothesis spaces.",
            "The closed final-only arm's reading was mixed (72h paired deltas "
            "+0.004698/+0.007867/-0.008708), not null; a mixed result here closes this branch "
            "permanently rather than inviting more compute (decision 0096 clause 5).",
            "Normalized training MSE is not held-out physical skill.",
        ],
    }


def execution_files():
    """Decision 0067: this round's import closure, from a named entry file."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0])


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """The preparation protocol: constants plus the code identity, before any data is read."""
    from training.r7_experiment import canonical_digest
    body = dict(frozen_constants())
    body.update({"code_archive_input": str(code_archive), "code_commit": code_commit,
                 "code_zip_sha256": code_sha256,
                 "started_perf_counter": started, "deadline_perf_counter": deadline,
                 "per_seed_seconds": budgets()["per_seed_seconds"],
                 "planned_seconds": budgets()["planned_seconds"],
                 "hard_cap_seconds": budgets()["hard_cap_seconds"]})
    body["protocol_sha256"] = canonical_digest(body)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    wide.write_exclusive(output / "preparation_protocol.json", body)
    return body


def prepared_protocol(output):
    """Add the live identity and the measured windows, which must pass before a protocol exists."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import (canonical_digest, dataset_identity, model_code_digest)
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    windows = preflight_long_rollout_windows(curve.TRAIN_MANIFEST, physical_steps=12)
    if windows["usable_windows"] != curve.USABLE_WINDOWS:
        raise RuntimeError(f"the manifest moved: {windows['usable_windows']} usable windows, "
                           f"not the frozen {curve.USABLE_WINDOWS}")
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "train_data_identity": dataset_identity(curve.TRAIN_MANIFEST)[0],
            "val_data_identity": dataset_identity(curve.VAL_MANIFEST)[0],
            "train_windows": windows,
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()},
            "arms_parent": {"pins": {}, "checkpoints": parent_pins.pins()}}
    parent_pins.install_parent_pins({"arms": {"parent": body["arms_parent"]}})
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output):
    """Re-check every frozen identity, and that the arms differ only in draft organisation."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest mismatch")
    if model_code_digest() != body["model_code_sha256"]:
        raise RuntimeError("the model revision moved after freeze")
    if training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("the training revision moved after freeze")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution source changed: {name}")
    if body.get("test_read") is not False:
        raise RuntimeError("this round is validation-only")
    on, off = body["arms"][CANDIDATE], body["arms"][CONTROL]
    if on["draft_supervision"] != DRAFT_SUPERVISION_FINAL_ONLY:
        raise RuntimeError("the candidate arm is not final_only")
    if on["internal_deep_supervision"] is not False:
        raise RuntimeError("final_only must not claim internal deep supervision")
    if off["draft_supervision"] != DRAFT_SUPERVISION_DEEP:
        raise RuntimeError("the control arm is not the registered deep form")
    if body["recipe"]["physical_weights"] != list(SPARSE_WEIGHTS):
        raise RuntimeError("the shared weight vector is not the registered one")
    if body["factor"]["arms_differ_only_in"] != ["draft_supervision"]:
        raise RuntimeError("the declared factor is not draft organisation alone")
    return body
