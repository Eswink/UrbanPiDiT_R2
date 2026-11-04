"""The Stage B curve's point-2/3/4 protocol layer, parameterised by point label.

Stage B is a *calibration* curve: one arm, one seed, batch fixed at 4, and the only thing
that varies across the four points is how many updates a run gets. Point 1 (0.25 epoch,
2,008 updates) has its own frozen module, ``r7_s3_throughput_physics_b_protocol``, which is
deliberately **not** retrofitted: its registered evidence pins the SHA256 of every file in
its execution closure, so editing that source after the fact would silently invalidate the
identity the point was registered under. Points 2-4 therefore share this parameterised
module instead, and each point is a thin binding that names its label.

Everything here is *what a point is frozen to*; training, scoring and reading live in
``study_r7_s3_curve_point.py``. The only predeclared judgement is the saturation rule in
``docs/goals/s3-throughput-and-physics-campaign.md`` §3, which no single point can evaluate.
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
    CLIMATOLOGY_BUCKETS, CLIMATOLOGY_YEARS, D2_REFERENCE, VAL_COHORTS,
    climatology_reference)
from training import r7_device_selection as devices

FORMAT = "r7-s3-curve-point-protocol-v1"
INSTANCE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
STORE = INSTANCE / "store/cache.zarr"
SOURCE = INSTANCE / "source.nc"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"
SOURCE_BYTES, STORE_SHAPE = 5492365397, [34084, 17, 65, 65]
TRAIN_MANIFEST = INSTANCE / "store/manifests_treatment/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests_treatment/val.jsonl"
TEST_MANIFEST = INSTANCE / "store/manifests_treatment/test.jsonl"
TRAIN_WINDOWS, VAL_COUNT, TEST_COUNT = 32142, 1458, 472
USABLE_WINDOWS = 32131

#: The four points of the curve, frozen as a ladder: (label, updates).
#: 0.25 / 1 / 4 / 16 epoch at batch 4 over 32,131 usable windows.
POINTS = (
    ("p0025", 2008),
    ("p1000", 8033),
    ("p4000", 32132),
    ("p1600", 128528),
)
NOMINAL_EPOCHS = {"p0025": 0.25, "p1000": 1.0, "p4000": 4.0, "p1600": 16.0}
BATCH_SIZE = 4
STEPS, LR, WARMUP = wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, CLIP = wide.WEIGHT_DECAY, wide.CLIP
PHYSICAL_WEIGHTS, REASONING_STEPS = wide.PHYSICAL_WEIGHTS, wide.REASONING_STEPS
#: One seed: the curve is a calibration axis, not a significance claim (goal §3).
SEEDS = (41,)
CHECKPOINT_EVERY = 5000
BOUNDARY_MARGINS, REGION, FULL_GRID_POINTS, DEVICE_MODE = (1,), "full", 4225, "dual"

#: Point 1's *measured* throughput and scoring cost, which is what supersedes the earlier
#: Stage A guess. Recorded as measurements with their source point, so a reader can tell an
#: estimate from a measurement and recompute the budget below rather than trusting it.
MEASURED_SECONDS_PER_UPDATE = 1.174
MEASURED_POINT_UPDATES = 2008
MEASURED_TRAIN_SECONDS = 2358.3
MEASURED_SCORING_SECONDS = 2633.8
MEASURED_PEAK_RESERVED_BYTES = 9372172288
#: The scoring allowance is point 1's measured five-lead cost, not a round number.
SCORING_ALLOWANCE_SECONDS = MEASURED_SCORING_SECONDS
#: Measured on point 1 for the identical batch/recipe/loader, so this is not a guess.
ESTIMATED_PEAK_BYTES = MEASURED_PEAK_RESERVED_BYTES
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
    "the completion-time budget is derived from point 1's measured 1.174 s/update at batch 4 "
    "on this store, which is an estimate for this point, not a measurement of it",
    "the model revision moved under decision 0080/0083 after the R1/R2 runs; the parent is the "
    "documented model-only migration under the current revision, whose per-seed tensor state "
    "is the witness that the same weights were installed",
    "the training-code identity differs from every earlier round because Stage A changed the "
    "read path and the finite-state cadence, so this point's training_code_sha256 is new by "
    "construction and is not comparable as an identity to earlier arms",
    "point 1's frozen module is not retrofitted under this parameterised one: its registered "
    "evidence pins its execution closure, so points 2-4 share this source instead",
]


def updates_for(point):
    """The frozen update count for a point label; an unknown label is refused, not defaulted."""
    ladder = dict(POINTS)
    if point not in ladder:
        raise RuntimeError(f"unknown curve point {point!r}; the ladder is {sorted(ladder)}")
    return ladder[point]


def model_spec():
    spec, _, _ = dose.recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    return spec


def point_blocks(point):
    """The curve's ladder plus this point's place on it, recorded so it cannot drift."""
    updates = updates_for(point)
    return {"pool": [list(entry) for entry in POINTS], "point": point, "updates": updates,
            "batch_size": BATCH_SIZE, "seeds": list(SEEDS),
            # Episode fraction over the 32,131 USABLE windows (not the 32,142 manifest rows).
            "epochs": {label: round(count * BATCH_SIZE / USABLE_WINDOWS, 6)
                       for label, count in POINTS},
            "nominal_epochs": dict(NOMINAL_EPOCHS),
            "axis": ("updates, not epochs: R1 measured that updates rather than samples move "
                     "skill, and batch is held at 4 so the curve is not confounded by it")}


def budgets(point):
    """This point's duration envelope, derived from point 1's measurements.

    ``per_seed`` is ~3x the derived estimate: point 1 showed a +4.1% training-time variance,
    so 3x covers that with room for a slower card while still bounding a hung worker far
    below the round's hard cap. ``planned`` and ``hard`` keep point 1's 1:2 ratio.
    """
    updates = updates_for(point)
    train = updates * MEASURED_SECONDS_PER_UPDATE
    return {"measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
            "measured_runs": [MEASURED_SECONDS_PER_UPDATE],
            "measured_on_point": "p0025",
            "measured_point_seconds": round(MEASURED_TRAIN_SECONDS + MEASURED_SCORING_SECONDS, 1),
            "train_seconds_estimate": train,
            "scoring_allowance_seconds": SCORING_ALLOWANCE_SECONDS,
            "point_seconds_estimate": train + SCORING_ALLOWANCE_SECONDS,
            "estimate_basis": (f"{updates} updates x point 1's measured "
                               f"{MEASURED_SECONDS_PER_UPDATE} s/update at batch 4, plus point "
                               "1's measured five-lead scoring cost: an estimate, not a "
                               "measurement of this point"),
            "per_seed_seconds": 36000.0, "planned_seconds": 39600.0,
            "hard_cap_seconds": 79200.0,
            "soft_rule": ("soft overrun continues and is recorded; only the hard cap truncates"),
            "decision": "0060"}


def frozen_constants(point):
    """Everything frozen before any data or model is touched."""
    updates = updates_for(point)
    return {
        "format": FORMAT, "stage": "S3-THROUGHPUT-PHYSICS-B", "instance": str(INSTANCE),
        "store": str(STORE), "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
        "store_shape": list(STORE_SHAPE),
        "split_years": {"train": CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(TRAIN_MANIFEST), "val_manifest": str(VAL_MANIFEST),
        "test_manifest_never_read": str(TEST_MANIFEST),
        "train_windows": TRAIN_WINDOWS, "val_count": VAL_COUNT, "test_count": TEST_COUNT,
        "usable_windows": USABLE_WINDOWS,
        "val_cohorts": dict(VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS),
                   "scored_region": REGION, "n_grid_points": FULL_GRID_POINTS,
                   "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"},
        "curve": point_blocks(point),
        "recipe": {"mode": "long_rollout", "updates": updates, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": wide.BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS),
                   "reasoning_steps": REASONING_STEPS,
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)",
                   "epochs": round(updates * BATCH_SIZE / USABLE_WINDOWS, 6),
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
                "estimate_provenance": "point 1's measured reserved peak, same batch/recipe",
                "headroom_margin_mib": HEADROOM_MARGIN_BYTES // 2**20,
                "neighbour_reserve_gib": 6,
                "gate": ("read-only free check before every seed; required = max(estimate, "
                         "owned peak) + margin + neighbour reserve; no neighbour signals"),
                "billing": "continuous first spawn through last owned reap; failures charged"},
        "budgets": budgets(point),
        "test_read": False,
        "test_read_policy": ("the 2023 four-month test split is never opened; only the train "
                             "manifest (fit) and the val manifest (scoring) are read"),
        "scientific_claim": False, "limitations": list(LIMITATIONS)}


def arm_protocol(point, spec, *, train_identity, windows, canonical_digest):
    """The single arm's frozen record."""
    updates = updates_for(point)
    return {"mode": "long_rollout", "updates": updates, "steps": STEPS, "lr": LR,
            "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "clip": CLIP, "bf16": wide.BF16, "checkpoint_every": CHECKPOINT_EVERY,
            "physical_weights": list(PHYSICAL_WEIGHTS),
            "train_manifest": str(TRAIN_MANIFEST), "train_data_identity": train_identity,
            "train_windows": windows,
            "model": {"kind": "process", "spec": spec,
                      "spec_canonical_digest": canonical_digest(spec)},
            "selection": "frozen endpoint; no validation selection"}


def identity(point):
    """The live identity block: source, code revisions, parent pins, manifests, climatology."""
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
    spec = model_spec()
    windows = preflight_long_rollout_windows(TRAIN_MANIFEST, physical_steps=12)
    if windows.get("input_windows") != TRAIN_WINDOWS:
        raise RuntimeError("the treatment manifest's window count changed")
    if windows.get("usable_windows") != USABLE_WINDOWS:
        raise RuntimeError("the treatment manifest's usable window count changed")
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    body = {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "source_file": {"path": str(SOURCE), "sha256": source_digest,
                            "bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files(point)},
            "arms": {"curve": arm_protocol(point, spec, train_identity=train_identity,
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


def freeze(point, output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    tuned = budgets(point)
    body = {**frozen_constants(point),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(DEVICE_MODE, single_uuid=wide.GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": tuned["planned_seconds"],
            "hard_cap_seconds": tuned["hard_cap_seconds"],
            "per_seed_seconds": tuned["per_seed_seconds"],
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "preparation_protocol.json", body)
    return body


def prepared_protocol(point, output):
    from training.r7_experiment import canonical_digest
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **identity(point)}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(point, output):
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
    if body["curve"]["point"] != point or body["curve"]["updates"] != updates_for(point):
        raise RuntimeError("the frozen curve point differs from this binding's constant")
    return body


def entry_file(point):
    """The named binding driver for this point (decision 0067: never a glob)."""
    return f"scripts/study_r7_s3_throughput_physics_b_{point}.py"


def execution_files(point):
    """Decision 0067: this point's import closure, from a named entry file."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(entry_file(point))
