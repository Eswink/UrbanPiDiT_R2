"""The Stage E1a protocol layer: 16-epoch dense-12 supervision on/off, three seeds paired.

What this round asks, in one sentence: at **128,528 updates = 16 epoch** on the same 65x65
multi-year store Stage B calibrated, does replacing the registered sparse physical-weight vector
``(1,.5,0,.5,0,0,0,.5,0,0,0,.5)`` with the dense ``(0.25,)*12`` change 72 h t2m skill?

Design provenance, in order: decision **0096** (why this is a reopening rather than a renamed
retry, why it splits into E1a/E1b, and the 3.9x budget overrun it declares), decision **0057**
(the reopening queue requires an explicit "not the same family" argument plus the new regime),
and `docs/goals/s3-throughput-and-physics-campaign.md` Stage E.

Why the factors travel as *arguments*. ``physical_weights`` is a per-call keyword of
``fine_tune_long_rollout`` with no default, and the registered vector lives in
``scripts/study_r7_s3_wide_single_factor.py``, outside the 13 ``SOURCE_FILES`` that
``training_code_digest()`` hashes. So this round needs **no change under ``training/``** and its
``training_code_sha256`` equals C2's. Decision 0096's original claim that E1 must edit
``training/`` was measured to be wrong and is corrected there.

Why this cannot reuse C2's protocol. That module asserts the candidate axis is exactly ``"c2"``
and hard-fails otherwise, and its driver writes the derivative-matching key into the contract
directly. E1a varies a *weight vector*, not an auxiliary objective, so the axis assertion would
have to be deleted -- which would silently change a frozen round's validator. A separate module
is the honest seam; the duplication is the price and is recorded in the ADR.
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

FORMAT = "r7-s3-e1a-dense-supervision-protocol-v1"
CANDIDATE, CONTROL = "on", "off"
SEEDS = (41, 42, 43)

#: 16 epoch on the 32,131 usable windows: 128,528 updates at batch 4. This is the *new regime*
#: that makes the reopening legitimate (decision 0096), 53.6x the 2,400 updates of the closed arm.
UPDATES = 128528
STEPS = 4
BATCH_SIZE = 4
REASONING_STEPS = 4

#: The registered sparse vector, verbatim from ``study_r7_s3_wide_single_factor``: the control.
SPARSE_WEIGHTS = tuple(wide.PHYSICAL_WEIGHTS)
#: The treatment: uniform over the same twelve physical steps. Both sum to 3.0, so "dense vs
#: sparse" is an allocation change at fixed total weight, not a doze change.
DENSE_WEIGHTS = (0.25,) * 12
if len(SPARSE_WEIGHTS) != 12 or len(DENSE_WEIGHTS) != 12:
    raise RuntimeError("both vectors must cover the twelve 6-hour physical steps")
if abs(sum(SPARSE_WEIGHTS) - sum(DENSE_WEIGHTS)) > 1e-12:
    raise RuntimeError("the two vectors must carry the same total weight; otherwise the round "
                       "changes two things at once")

#: Measured on this hardware: the 16-epoch Stage B point ran at 1.1096 s/update; C2's two
#: concurrent arms measured 1.1349/1.1497 and 1.1612/1.1121 across attempts. Use the *slowest*
#: observed long-run value so the envelope cannot truncate the round (decision 0090's lesson).
MEASURED_SECONDS_PER_UPDATE = 1.1740
MEASURED_ON = "p0025 batch-4 long run, the slowest of the four Stage B point measurements"
SCORING_ALLOWANCE_SECONDS = 2588.1

ENTRY_FILES = ("scripts/study_r7_s3_e1a_dense_supervision.py",)
CODE_ARCHIVE = ROOT / "outputs/r7_s3_e1a_dense_supervision_code.zip"


def budgets():
    """The duration envelope, derived from the slowest measured batch-4 rate on this store.

    A seed runs **two** arms, so the envelope is three times the two-arm estimate rounded up to a
    whole hour, and the hard cap is twice that -- the 1:2 ratio every earlier round used. The
    three-times factor is deliberate slack for the second arm plus scoring, not a target.
    """
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


def arm_weights(arm):
    """The one factor: the physical-weight vector. Everything else is shared."""
    if arm == CANDIDATE:
        return list(DENSE_WEIGHTS)
    if arm == CONTROL:
        return list(SPARSE_WEIGHTS)
    raise ValueError(f"unknown arm {arm!r}")


def frozen_constants():
    """Everything frozen before any data or model is touched."""
    return {
        "format": FORMAT,
        "stage": "S3-THROUGHPUT-PHYSICS-E1A",
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
                                                               / curve.USABLE_WINDOWS, 6)},
        "arms": {
            CANDIDATE: {"physical_weights": list(DENSE_WEIGHTS), "sum": sum(DENSE_WEIGHTS),
                        "description": "dense: uniform 0.25 over all twelve 6-hour steps"},
            CONTROL: {"physical_weights": list(SPARSE_WEIGHTS), "sum": sum(SPARSE_WEIGHTS),
                      "description": "registered sparse vector, verbatim"},
        },
        "factor": {"name": "dense physical-step supervision", "control": "sparse (registered)",
                   "treatment": "dense uniform", "arms_differ_only_in": ["physical_weights"]},
        "budgets": budgets(),
        "test_read": False,
        "scientific_claim": False,
        "r": 0,
        "limitations": [
            "Treatment and control differ only in the weight vector; a win cannot be attributed "
            "to any single physical step.",
            "Three seeds give a paired comparison, not significance across hypothesis spaces.",
            "The closed dense-supervision arm ran 2,400 updates; this round's 128,528 updates is a "
            "different regime, so results are not comparable across the two (decision 0057).",
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
    """Re-check every frozen identity, including that the two arms differ only where declared."""
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
    # The single-factor property, asserted rather than assumed: the two arms must carry the same
    # total weight and differ only in the vector, or the round would vary allocation *and* dose.
    on, off = body["arms"][CANDIDATE], body["arms"][CONTROL]
    if on["physical_weights"] != list(DENSE_WEIGHTS):
        raise RuntimeError("the candidate arm is not the dense vector")
    if off["physical_weights"] != list(SPARSE_WEIGHTS):
        raise RuntimeError("the control arm is not the registered sparse vector")
    if abs(on["sum"] - off["sum"]) > 1e-12:
        raise RuntimeError("the arms do not carry the same total weight")
    if body["factor"]["arms_differ_only_in"] != ["physical_weights"]:
        raise RuntimeError("the declared factor is not the weight vector alone")
    return body
