"""R4's frozen protocol layer: constants, payload loading, freeze and validate.

Split out of ``scripts/study_r7_s3_position_retention_65x65.py`` so both files stay under the
repository's 600-line cap (R-051). Everything here is about *what the round is frozen to*;
the training and reading machinery stays in the driver.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.r7_s3_position_retention_support import (
    ARMS, CANDIDATE, CLOSED_ROUND_BATCH_SIZE, CLOSED_ROUND_RECORD, CLOSED_ROUND_UPDATES,
    CONTROL, DECISION_RULE, ENTROPY_BAND, ENTROPY_ONE_HOT_FLOOR, LIMITATIONS, MODES,
    POSITION_SCALE, QUERY_NORM_RANGE, SIDECAR_DATA_IDENTITY, SIDECAR_RUN,
    SIDECAR_TYPED_EVIDENCE_IDENTITY, arm_name, arm_spec, require_single_factor)
from scripts.r7_s3_fy_year_coverage_climatology import (
    CLIMATOLOGY_BUCKETS, CLIMATOLOGY_STEPS, CLIMATOLOGY_T2M_RMSE, CLIMATOLOGY_YEARS, D2_FORMAT,
    D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS, climatology_reference)
from training import r7_device_selection as devices

FORMAT = "r7-s3-position-retention-65x65-protocol-v1"
INSTANCE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
STORE = INSTANCE / "store/cache.zarr"
SOURCE = INSTANCE / "source.nc"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"
SOURCE_BYTES, STORE_SHAPE = 5492365397, [34084, 17, 65, 65]
TRAIN_MANIFEST = INSTANCE / "store/manifests_treatment/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests_treatment/val.jsonl"
TEST_MANIFEST = INSTANCE / "store/manifests_treatment/test.jsonl"
TRAIN_WINDOWS, VAL_COUNT, TEST_COUNT = 32142, 1458, 472
# Matched compute with R2 so the two rounds are comparable in shape, not in factor.
UPDATES, BATCH_SIZE = 16071, 2
STEPS, LR, WARMUP = wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, CLIP = wide.WEIGHT_DECAY, wide.CLIP
CHECKPOINT_EVERY, BF16, PHYSICAL_WEIGHTS = wide.CHECKPOINT_EVERY, wide.BF16, wide.PHYSICAL_WEIGHTS
REASONING_STEPS, SEEDS = wide.REASONING_STEPS, wide.SEEDS
BOUNDARY_MARGINS, REGION, FULL_GRID_POINTS, DEVICE_MODE = (1,), "full", 4225, "dual"
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 129600., 259200., 518400.
# The spatial arm adds exactly one [dim] query; the rest of the model is unchanged.
SIDECAR_DIR = ROOT / SIDECAR_RUN
ESTIMATED_PEAK_BYTES = 5596409024
HEADROOM_MARGIN_BYTES = 512 * 2**20
ENTRY_FILES = ("scripts/study_r7_s3_position_retention_65x65.py",)


def execution_files():
    """Decision 0067: this round's import closure, never a glob over a package directory."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0])


_CACHED_PAYLOAD = None


def typed_payload():
    """The published 65x65 sidecar's field statistics plus the store's own de-normalisation.

    The ``denorm_*`` vectors must be this store's: the closed arm's payload carries the
    129x129 store's, and applying it here mis-decodes winds by 14-31% (decision 0080).
    """
    global _CACHED_PAYLOAD
    if _CACHED_PAYLOAD is None:
        import numpy as np
        import zarr

        from data.preprocess.r7_typed_evidence_scale import load_typed_evidence_scale
        meta = load_typed_evidence_scale(SIDECAR_DIR)
        if meta["typed_evidence_identity"] != SIDECAR_TYPED_EVIDENCE_IDENTITY:
            raise RuntimeError("the 65x65 typed-evidence sidecar identity moved")
        if meta["data_identity"] != SIDECAR_DATA_IDENTITY:
            raise RuntimeError("the sidecar was not fitted on the treatment train manifest")
        root = zarr.open_group(str(STORE), mode="r")
        channels = list(root.attrs["channels"])
        if channels != list(meta["channels"]):
            raise RuntimeError("the sidecar channel order does not match the store")
        _CACHED_PAYLOAD = {
            "channels": channels,
            "denorm_mean": [float(v) for v in np.asarray(root["normalization_mean"][:])],
            "denorm_std": [float(v) for v in np.asarray(root["normalization_std"][:])],
            "field_scale": [float(v) for v in meta["runtime_scale"]],
            "field_mean": [float(v) for v in meta["runtime_mean"]],
            "field_std": [float(v) for v in meta["runtime_std"]],
            "field_active": [bool(v) for v in meta["active_mask"]],
        }
    return _CACHED_PAYLOAD


def frozen_constants(payload):
    """Every protocol field that does not depend on the built store (unit-testable)."""
    return {
        "format": FORMAT, "stage": "S3-POSITION-RETENTION-65X65", "instance": str(INSTANCE),
        "store": str(STORE),
        "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
        "store_shape": list(STORE_SHAPE),
        "split_years": {"train": CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(TRAIN_MANIFEST), "val_manifest": str(VAL_MANIFEST),
        "test_manifest_never_read": str(TEST_MANIFEST),
        "train_windows": TRAIN_WINDOWS, "val_count": VAL_COUNT, "test_count": TEST_COUNT,
        "val_cohorts": dict(VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS),
                   "scored_region": REGION, "n_grid_points": FULL_GRID_POINTS,
                   "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"},
        "factor": {"name": "typed-evidence patch-position aggregation at matched compute",
                   "control": "typed_routing: uniform mean over the P patch positions",
                   "candidate": "typed_spatial: softmax over P with one shared learned query",
                   "controlled": ["store", "train manifest", "normalization", "model trunk",
                                  "migrated parent", "typed-evidence payload",
                                  "position_scale", "K", "depth", "dim", "physical weights",
                                  "seeds", "val split", "climatology denominator", "lr",
                                  "warmup", "clip", "supervision", "updates", "batch_size"],
                   "closed_round": {"record": CLOSED_ROUND_RECORD,
                                    "updates": CLOSED_ROUND_UPDATES,
                                    "batch_size": CLOSED_ROUND_BATCH_SIZE,
                                    "instance": "129x129 wide, interior_32, scale 1"},
                   "changed_from_closed_round": ["instance", "grid", "supervision region",
                                                 "data span", "typed-evidence payload",
                                                 "position_scale", "query starting point",
                                                 "updates", "batch_size"],
                   "declared_parameter_difference": (
                       "the spatial arm carries one extra [dim] query (+192 parameters, "
                       "+0.13%); position_scale is a constant and adds none"),
                   "not_controlled": ["parameter count (+192 in the spatial arm)"]},
        "position_scale": {"value": POSITION_SCALE, "learnable": False,
                           "entropy_band": list(ENTROPY_BAND),
                           "one_hot_floor": ENTROPY_ONE_HOT_FLOOR,
                           "query_norm_range": list(QUERY_NORM_RANGE),
                           "rule": ("frozen before the run by decision 0081 on real val frames; "
                                    "never swept, never adjusted after reading results; the query "
                                    "starts at exactly zero so this constant sets forward "
                                    "sensitivity and not learning amplitude (decision 0079)")},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS),
                   "reasoning_steps": REASONING_STEPS,
                   "epochs_floor": ("updates * batch_size = 32142 >= the manifest, so each arm "
                                    "sees >= 1 epoch"),
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)"},
        "typed_evidence": {"sidecar_run": SIDECAR_RUN,
                           "typed_evidence_identity": SIDECAR_TYPED_EVIDENCE_IDENTITY,
                           "data_identity": SIDECAR_DATA_IDENTITY,
                           "fit_split": "train",
                           "mode_control": MODES[CONTROL], "mode_candidate": MODES[CANDIDATE],
                           "starting_point": ("both arms are constructed under isolated_stream() "
                                              "with a fresh initialisation, so the query starts at "
                                              "exactly zero and the spatial arm equals the pooled "
                                              "arm before any update (decision 0080 route 2)"),
                           "closed_arm_payload_rule": (
                               "the closed arm's payload is NOT reused: it carries the 129x129 "
                               "store's de-normalisation vectors and field statistics")},
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1",
                        "buckets": CLIMATOLOGY_BUCKETS, "training_years": CLIMATOLOGY_YEARS,
                        "n_selected_steps": CLIMATOLOGY_STEPS,
                        "t2m_rmse": dict(CLIMATOLOGY_T2M_RMSE),
                        "reference_format": D2_FORMAT, "reference_path": str(D2_REFERENCE)},
        "bridge_e": {"scientific": False,
                     "rule": ("four identities, all asserted per seed before training is scored: "
                              "both arms start with |position_query| == 0; the same seed's two "
                              "arms share one pathway_init_sha256; both arms carry the same "
                              "position_scale; and zeroing the query makes the spatial arm's "
                              "aggregation equal the pooled mean bit for bit. A failure is "
                              "identity/engineering unresolved, never a verdict")},
        "decision": {"exits": {"a": "candidate 72h t2m skill > 0, all seeds",
                               "b": "candidate-control 72h delta RMSE < 0, all seeds",
                               "c": "candidate 48h t2m skill > 0, all seeds",
                               "d": "candidate-control delta RMSE at 6/12/24h, seed mean <= 0"},
                     "advance": "all of (a)-(d) -> independent review",
                     "close": ("any of (a)-(d) false -> close the position-retention branch at "
                               "matched compute"),
                     "text": DECISION_RULE},
        "gpu": {"policy": "shared", "uuid": wide.GPU_UUID,
                "estimated_peak_mib": ESTIMATED_PEAK_BYTES // 2**20,
                "headroom_margin_mib": HEADROOM_MARGIN_BYTES // 2**20,
                "neighbour_reserve_gib": 6,
                "gate": ("read-only free check before every seed; required = max(estimate, "
                         "owned peak) + margin + neighbour reserve; no neighbour signals"),
                "billing": "continuous first spawn through last owned reap; failures charged"},
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS,
                    "soft_rule": ("soft overrun continues and is recorded; only the hard cap "
                                  "truncates"),
                    "decision": "0060"},
        "test_read": False,
        "test_read_policy": ("the 2023 four-month test split is never opened; only the train "
                             "manifest (fit) and the val manifest (scoring) are read"),
        "scientific_claim": False, "limitations": list(LIMITATIONS),
    }


def arm_protocol(spec, mode, *, train_identity, windows, canonical_digest):
    """One arm's frozen record. The only fields the two arms may differ in are named here."""
    return {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
            "lr": LR, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "clip": CLIP, "bf16": BF16,
            "checkpoint_every": CHECKPOINT_EVERY, "physical_weights": list(PHYSICAL_WEIGHTS),
            "train_manifest": str(TRAIN_MANIFEST), "train_data_identity": train_identity,
            "train_windows": windows, "typed_evidence_mode": mode,
            "position_scale": POSITION_SCALE,
            "model": {"kind": "process", "spec": spec,
                      "spec_canonical_digest": canonical_digest(spec)},
            "selection": "frozen endpoint; no validation selection"}


def _model_spec():
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    return spec


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
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    payload = typed_payload()
    arms = {name: arm_protocol(arm_spec(spec, payload, mode), mode,
                               train_identity=train_identity, windows=windows,
                               canonical_digest=canonical_digest)
            for name, mode in ARMS}
    require_single_factor(arms[CONTROL], arms[CANDIDATE])
    body = {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "source_file": {"path": str(SOURCE), "sha256": source_digest,
                            "bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()},
            "arms": {**arms, "parent": {"pins": {}, "checkpoints": parent_pins.pins()}},
            "manifest_sha256": {"train": sha256_file(TRAIN_MANIFEST),
                                "val": sha256_file(VAL_MANIFEST)},
            "train_data_identity": train_identity,
            "val_data_identity": dataset_identity(VAL_MANIFEST)[0],
            "climatology_reference": climatology_reference(),
            "code": recipe._shared().code_state()}
    parent_pins.install_parent_pins(body)
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**frozen_constants(typed_payload()),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(DEVICE_MODE, single_uuid=wide.GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
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
    return body


ENTRY_FILES = ("scripts/study_r7_s3_position_retention_65x65.py",)
