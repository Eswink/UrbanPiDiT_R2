"""Stage B point 1's frozen protocol layer: constants, identity, freeze and validate.

Split from the driver so both files stay under R-051's 600-line cap. Everything here is
about *what the point is frozen to*; the training, scoring and reading machinery lives in
``scripts/study_r7_s3_throughput_physics_b.py``.

The point is one arm at one endpoint: no factor comparison, no seeds beyond one. Stage B is
a *calibration* curve -- "does 72 h skill still rise with training volume" -- so it has no
control arm to be paired against. Its only predeclared judgement is the saturation rule in
``docs/goals/s3-throughput-and-physics-campaign.md`` §3, computed later from all four points.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts.r7_s3_fy_year_coverage_climatology import (
    CLIMATOLOGY_BUCKETS, CLIMATOLOGY_YEARS, D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS,
    climatology_reference)
from training import r7_device_selection as devices

FORMAT = "r7-s3-throughput-physics-b-protocol-v1"
INSTANCE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
STORE = INSTANCE / "store/cache.zarr"
SOURCE = INSTANCE / "source.nc"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"
SOURCE_BYTES, STORE_SHAPE = 5492365397, [34084, 17, 65, 65]
TRAIN_MANIFEST = INSTANCE / "store/manifests_treatment/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests_treatment/val.jsonl"
TEST_MANIFEST = INSTANCE / "store/manifests_treatment/test.jsonl"
TRAIN_WINDOWS, VAL_COUNT, TEST_COUNT = 32142, 1458, 472

#: The four points of the curve, frozen as a ladder: (label, updates).
#: 0.25 / 1 / 4 / 16 epoch at batch 4 over 32,131 usable windows.
POINTS = (
    ("p0025", 2008),
    ("p1000", 8033),
    ("p4000", 32132),
    ("p1600", 128528),
)
#: This round runs point 1 only. The other three are separate attempts, never resumed into.
POINT = "p0025"
UPDATES = dict(POINTS)[POINT]
BATCH_SIZE = 4
STEPS, LR, WARMUP = wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, CLIP = wide.WEIGHT_DECAY, wide.CLIP
PHYSICAL_WEIGHTS, REASONING_STEPS = wide.PHYSICAL_WEIGHTS, wide.REASONING_STEPS
#: One seed: the curve is a calibration axis, not a significance claim (goal §3).
SEEDS = (41,)
CHECKPOINT_EVERY = 5000
BOUNDARY_MARGINS, REGION, FULL_GRID_POINTS, DEVICE_MODE = (1,), "full", 4225, "dual"
#: The measured per-update wall clock at batch 4 (Stage A, four runs: 1.0848/1.1349/1.1107/1.1812).
#: The budget below is DERIVED from it rather than rounded to a convenient figure, so a future
#: reader can recompute the estimate instead of trusting it.
MEASURED_SECONDS_PER_UPDATE = 1.1279
MEASURED_SECONDS_PER_UPDATE_RUNS = (1.0848, 1.1349, 1.1107, 1.1812)
#: Training time for this point, plus a scoring allowance for five leads over the val cohort.
SCORING_ALLOWANCE_SECONDS = 3600.0
TRAIN_SECONDS = UPDATES * MEASURED_SECONDS_PER_UPDATE
POINT_SECONDS = TRAIN_SECONDS + SCORING_ALLOWANCE_SECONDS
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 39600., 43200., 86400.
ESTIMATED_PEAK_BYTES = 6270 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20

#: The frozen saturation rule, verbatim from the goal document §3. Recorded, never recomputed
#: here: a single point cannot evaluate a two-point first difference.
SATURATION_RULE = {
    "delta": "skill_72h(16 epoch) - skill_72h(4 epoch)",
    "unsaturated_if": "delta > 0.01 -> final run at 16-64 epoch",
    "saturated_if": ("delta < 0.005 AND skill_72h(16 epoch) < skill_72h(4 epoch) "
                     "-> final run at the saturation point"),
    "else": "record only; no extrapolation, no extra points",
    "note": ("all three outcomes are registered as they fall, including 'saturated', which "
             "contradicts the original premise and is therefore a valid result"),
}

LIMITATIONS = [
    "development calibration on the 65x65 multi-year instance only (2000-2021 train / "
    "2022 val, 17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed; r stays 0",
    "ONE seed: this point estimates a learning-curve ordinate, not a significance claim, and "
    "no confidence interval is computed or implied",
    "the four curve points are separate attempts with no checkpoint resume; a point that fails "
    "is re-run from scratch, not continued",
    "the completion-time budget is an estimate from the measured 1.1279 s/update at batch 4 "
    "on this store, not a measurement of this point",
    "the model revision moved under decision 0080/0083 after the R1/R2 runs; the parent is the "
    "documented model-only migration under the current revision, whose per-seed tensor state "
    "is the witness that the same weights were installed",
    "the training-code identity differs from every earlier round because Stage A changed the "
    "read path and the finite-state cadence, so this point's training_code_sha256 is new by "
    "construction and is not comparable as an identity to earlier arms",
    "a curve ordinate at 0.25 epoch cannot say anything about 16 epoch; the saturation "
    "judgement belongs to the whole four-point ladder, not to this point",
]


def _model_spec():
    spec, _, _ = dose.recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    return spec


def _point_blocks():
    """The curve's ladder, recorded so the registered points cannot drift silently."""
    return {"pool": list(POINTS), "point": POINT, "updates": UPDATES,
            "batch_size": BATCH_SIZE, "seeds": list(SEEDS),
            # Episode fraction over the 32,131 USABLE windows (not the 32,142 manifest rows);
            # the label in POINTS is the round number the goal document uses.
            "epochs": {label: round(updates * BATCH_SIZE / 32131, 6) for label, updates in POINTS},
            "nominal_epochs": {label: nominal for label, nominal in
                               (("p0025", 0.25), ("p1000", 1.0),
                                ("p4000", 4.0), ("p1600", 16.0))},
            "axis": ("updates, not epochs: R1 measured that updates rather than samples move "
                     "skill, and batch is held at 4 so the curve is not confounded by it")}


def frozen_constants():
    """Everything frozen before any data or model is touched."""
    return {
        "format": FORMAT, "stage": "S3-THROUGHPUT-PHYSICS-B", "instance": str(INSTANCE),
        "store": str(STORE), "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
        "store_shape": list(STORE_SHAPE),
        "split_years": {"train": CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(TRAIN_MANIFEST), "val_manifest": str(VAL_MANIFEST),
        "test_manifest_never_read": str(TEST_MANIFEST),
        "train_windows": TRAIN_WINDOWS, "val_count": VAL_COUNT, "test_count": TEST_COUNT,
        "val_cohorts": dict(VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS),
                   "scored_region": REGION, "n_grid_points": FULL_GRID_POINTS,
                   "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"},
        "curve": _point_blocks(),
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": wide.BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS),
                   "reasoning_steps": REASONING_STEPS,
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)",
                   "epochs": round(UPDATES * BATCH_SIZE / 32131, 6),
                   "batch_size_choice": ("fixed at 4 for every point of the curve: the A6 "
                                         "ceiling is a Stage F cost option, and changing batch "
                                         "during calibration would confound the very curve "
                                         "being measured")},
        "checkpoint_policy": {
            "checkpoint_every": CHECKPOINT_EVERY,
            "retained": "integer multiples of checkpoint_every plus the frozen endpoint",
            "selection": "frozen endpoint; no validation selection",
            "decision": "0087",
            "note": "applies to this new run only; no existing checkpoint is deleted"},
        "finite_state_policy": {
            "full_check_every": 100,
            "also_at": "every checkpoint step and the frozen endpoint",
            "per_step_unchanged": "loss and gradient norm (clip_grad_norm_ error_if_nonfinite)",
            "decision": "0086"},
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1",
                        "buckets": CLIMATOLOGY_BUCKETS, "training_years": CLIMATOLOGY_YEARS,
                        "reference_format": D2_REFERENCE.name,
                        "reference_path": str(D2_REFERENCE)},
        "saturation_rule": dict(SATURATION_RULE),
        "gpu": {"policy": "shared", "uuid": wide.GPU_UUID,
                "estimated_peak_mib": ESTIMATED_PEAK_BYTES // 2**20,
                "headroom_margin_mib": HEADROOM_MARGIN_BYTES // 2**20,
                "neighbour_reserve_gib": 6,
                "gate": ("read-only free check before every seed; required = max(estimate, "
                         "owned peak) + margin + neighbour reserve; no neighbour signals"),
                "billing": "continuous first spawn through last owned reap; failures charged"},
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS,
                     "measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
                    "measured_runs": list(MEASURED_SECONDS_PER_UPDATE_RUNS),
                    "train_seconds_estimate": TRAIN_SECONDS,
                    "scoring_allowance_seconds": SCORING_ALLOWANCE_SECONDS,
                    "point_seconds_estimate": POINT_SECONDS,
                    "estimate_basis": (f"{UPDATES} updates x the measured "
                                       f"{MEASURED_SECONDS_PER_UPDATE} s/update at batch 4, plus "
                                       "a scoring allowance: an estimate, not a measurement"),
                    "soft_rule": ("soft overrun continues and is recorded; only the hard cap "
                                  "truncates"),
                    "decision": "0060"},
        "test_read": False,
        "test_read_policy": ("the 2023 four-month test split is never opened; only the train "
                             "manifest (fit) and the val manifest (scoring) are read"),
        "scientific_claim": False, "limitations": list(LIMITATIONS),
    }


def arm_protocol(spec, *, train_identity, windows, canonical_digest):
    """The single arm's frozen record."""
    return {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
            "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "clip": CLIP, "bf16": wide.BF16, "checkpoint_every": CHECKPOINT_EVERY,
            "physical_weights": list(PHYSICAL_WEIGHTS),
            "train_manifest": str(TRAIN_MANIFEST), "train_data_identity": train_identity,
            "train_windows": windows,
            "model": {"kind": "process", "spec": spec,
                      "spec_canonical_digest": canonical_digest(spec)},
            "selection": "frozen endpoint; no validation selection"}


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    if model_code_digest() != parent_pins.MODEL_CODE_SHA256:
        raise RuntimeError("the working tree's model revision is not the one the parent was "
                           "migrated under; commit or clear model/ changes first")
    source_stat, source_digest = SOURCE.stat(), sha256_file(SOURCE)
    if source_digest != SOURCE_SHA256:
        raise RuntimeError("the merged multi-year source drifted from its frozen SHA256")
    spec = _model_spec()
    windows = preflight_long_rollout_windows(TRAIN_MANIFEST, physical_steps=12)
    if windows.get("input_windows") != TRAIN_WINDOWS:
        raise RuntimeError("the treatment manifest's window count changed")
    if windows.get("usable_windows") != 32131:
        raise RuntimeError("the treatment manifest's usable window count changed")
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    body = {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "source_file": {"path": str(SOURCE), "sha256": source_digest,
                            "bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()},
            "arms": {"curve": arm_protocol(spec, train_identity=train_identity,
                                           windows=windows, canonical_digest=canonical_digest),
                     "parent": {"pins": {}, "checkpoints": parent_pins.pins()}},
            "manifest_sha256": {"train": sha256_file(TRAIN_MANIFEST),
                                "val": sha256_file(VAL_MANIFEST)},
            "train_data_identity": train_identity,
            "val_data_identity": dataset_identity(VAL_MANIFEST)[0],
            "climatology_reference": climatology_reference(),
            "code": dose.recipe._shared().code_state()}
    parent_pins.install_parent_pins(body)
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**frozen_constants(),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(DEVICE_MODE, single_uuid=wide.GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS, "started_perf_counter": started,
            "deadline_perf_counter": deadline,
            "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "preparation_protocol.json", body)
    return body


def _prepared_protocol(output):
    from training.r7_experiment import canonical_digest
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
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
    if body["curve"]["point"] != POINT or body["curve"]["updates"] != UPDATES:
        raise RuntimeError("the frozen curve point differs from this driver's constant")
    return body


ENTRY_FILES = ("scripts/study_r7_s3_throughput_physics_b.py",)


def execution_files():
    """Decision 0067: this round's import closure, never a glob over a package directory."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0])
