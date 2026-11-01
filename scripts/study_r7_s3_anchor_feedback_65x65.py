"""S3 R2: the climatology-anchor anomaly feedback, re-asked at matched compute.

Decision 0058 admitted this round on the 65x65 multi-year institution and plan 0026 schedules
it as Track 1's R2 at the user's batch decision: the K-step recurrence's feedback is either the
absolute draft or ``draft - C_valid``. The closed arm died on the *absolute* 72 h gate (2/3
seeds), not on the paired gate it passed 3/3, and on this institution the unanchored arms
already clear that gate -- so the question is a different one, and the closed arm's readings
transfer in neither direction.

Both arms train on the 2000-2021 treatment manifest at 16,071 updates and batch 2 (so
``updates * batch = 32,142`` is at least one epoch), install the same published 48-bucket
table and hold the output and proposal priors off; the only factor is the feedback boolean.
A separate file, not a parameterisation of the closed driver: the closed protocol must stay
readable as it was frozen, and an importing module would inherit its constants silently.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_anchor_feedback_parent as parent_pins
from scripts import r7_s3_anchor_isolation_parent as anchor_parent
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.r7_s3_anchor_feedback_support import (
    ARMS, BRIDGE_BATCH_SIZE, BRIDGE_RECORD, BRIDGE_UPDATE1_GRADIENT_NORM, BRIDGE_UPDATE1_LOSS,
    BRIDGE_UPDATES, CANDIDATE, CLOSED_ROUND_BATCH_SIZE, CLOSED_ROUND_RECORD, CLOSED_ROUND_UPDATES,
    CONTROL, DECISION_RULE, LIMITATIONS, anchor_block, arm_name, arm_spec, check_model_revision,
    require_single_factor)
from scripts.r7_s3_fy_year_coverage_climatology import (
    CLIMATOLOGY_BUCKETS, CLIMATOLOGY_STEPS, CLIMATOLOGY_T2M_RMSE, CLIMATOLOGY_YEARS, D2_FORMAT,
    D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS, climatology_reference)
from training import r7_device_selection as devices

FORMAT = "r7-s3-anchor-feedback-65x65-protocol-v1"
INSTANCE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
STORE = INSTANCE / "store/cache.zarr"
SOURCE = INSTANCE / "source.nc"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"
SOURCE_BYTES, STORE_SHAPE = 5492365397, [34084, 17, 65, 65]
ANCHOR_TABLE_RUN = ROOT / "outputs/r7_s3_fy_anchor_table_20261025_attempt01"
TRAIN_MANIFEST = INSTANCE / "store/manifests_treatment/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests_treatment/val.jsonl"
TEST_MANIFEST = INSTANCE / "store/manifests_treatment/test.jsonl"
TRAIN_WINDOWS, VAL_COUNT, TEST_COUNT = 32142, 1458, 472
# The matched-compute factor (plan 0026 section 4): 16,071 * 2 = 32,142 >= the manifest.
UPDATES, BATCH_SIZE = 16071, 2
STEPS, LR, WARMUP = wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, CLIP = wide.WEIGHT_DECAY, wide.CLIP
CHECKPOINT_EVERY, BF16, PHYSICAL_WEIGHTS = wide.CHECKPOINT_EVERY, wide.BF16, wide.PHYSICAL_WEIGHTS
REASONING_STEPS, SEEDS = wide.REASONING_STEPS, wide.SEEDS
BOUNDARY_MARGINS, REGION, FULL_GRID_POINTS, DEVICE_MODE = (1,), "full", 4225, "dual"
# Decision 0060's Track 1 constants; R1 measured 88,362-89,520 s per seed against the cap.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 129600., 259200., 518400.
# R1 measured 5,582,618,624 B of owned reserved peak on this store at batch 2; the 48-bucket
# table adds 13,790,400 B. The FY round's 4 GiB estimate undershot the same measurement.
ESTIMATED_PEAK_BYTES = 5596409024
HEADROOM_MARGIN_BYTES = 512 * 2**20
ENTRY_FILES = ("scripts/study_r7_s3_anchor_feedback_65x65.py",)


def execution_files():
    """Decision 0067: this round's import closure, never a glob over a package directory.

    The closed rounds inherited the wide round's 194-entry list, 93% of which was "whatever
    .py files were in the tree"; that is the defect that cost R1 two seeds. The identity
    surface narrows from 202 entries to this closure, so a module an unrelated line adds no
    longer fails this round's last check. The archived v2 protocol that ``recipe._shared()``
    reads is a local ``outputs/`` artifact and is deliberately not here: the spec it provides
    is pinned value-level by ``spec_canonical_digest``.
    """
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0])


_CACHED_PAYLOAD = None


def anchor_payload():
    """Load and verify the published 48-bucket table once per process."""
    global _CACHED_PAYLOAD
    if _CACHED_PAYLOAD is None:
        _CACHED_PAYLOAD = anchor_parent.anchor_payload(ANCHOR_TABLE_RUN)
    return _CACHED_PAYLOAD


def frozen_constants(payload):
    """Every protocol field that does not depend on the built store (unit-testable).

    The anchor payload is a parameter, not a global read, so the tests can exercise this with a
    synthetic table and CI never needs ``outputs/``; the freeze path passes the published one.
    """
    pins = payload["pins"]
    return {
        "format": FORMAT, "stage": "S3-ANCHOR-FB-65X65", "instance": str(INSTANCE),
        "store": str(STORE),
        "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES, "store_shape": list(STORE_SHAPE),
        "split_years": {"train": CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(TRAIN_MANIFEST), "val_manifest": str(VAL_MANIFEST),
        "test_manifest_never_read": str(TEST_MANIFEST),
        "train_windows": TRAIN_WINDOWS, "val_count": VAL_COUNT, "test_count": TEST_COUNT,
        "val_cohorts": dict(VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS), "scored_region": REGION,
                   "n_grid_points": FULL_GRID_POINTS, "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"},
        "factor": {"name": "climatology-anchor anomaly feedback at matched compute",
                   "control": "anomaly_feedback=False (the recurrence sees the absolute draft)",
                   "candidate": "anomaly_feedback=True (the recurrence sees draft - C_valid)",
                   "controlled": ["store", "train manifest", "normalization", "model trunk",
                                  "migrated parent", "anchor table", "K", "depth", "dim",
                                  "physical weights", "seeds", "val split",
                                  "climatology denominator", "lr", "warmup", "clip",
                                  "supervision", "updates", "batch_size"],
                   "closed_round": {"record": CLOSED_ROUND_RECORD, "updates": CLOSED_ROUND_UPDATES,
                                    "batch_size": CLOSED_ROUND_BATCH_SIZE,
                                    "instance": "129x129 wide, interior_32, 16-bucket table"},
                   "changed_from_closed_round": ["instance", "grid", "supervision region",
                                                 "data span", "anchor table", "updates",
                                                 "batch_size"],
                   "not_controlled": []},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS), "reasoning_steps": REASONING_STEPS,
                   "epochs_floor": "updates * batch_size = 32142 >= the manifest, so each arm sees >= 1 epoch",
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)"},
        "climatology_anchor": {"arms": {name: anchor_block(name, feedback, payload)
                                        for name, feedback in ARMS}},
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1",
                        "buckets": CLIMATOLOGY_BUCKETS, "training_years": CLIMATOLOGY_YEARS,
                        "n_selected_steps": CLIMATOLOGY_STEPS, "t2m_rmse": dict(CLIMATOLOGY_T2M_RMSE),
                        "reference_format": D2_FORMAT, "reference_path": str(D2_REFERENCE)},
        "bridge_e": {"record_id": BRIDGE_RECORD, "updates": BRIDGE_UPDATES,
                     "batch_size": BRIDGE_BATCH_SIZE, "scientific": False,
                     "update1_loss": {str(seed): value for seed, value in BRIDGE_UPDATE1_LOSS.items()},
                     "update1_gradient_norm": {str(seed): value
                                               for seed, value in BRIDGE_UPDATE1_GRADIENT_NORM.items()},
                     "rule": ("the control arm's first-update loss must equal the registered R1 "
                              "treatment arm's bit for bit -- same store, manifest, seed, batch, "
                              "parent and schedule, and a forward the anchor round measured as "
                              "bit-identical to the unanchored model; a mismatch is identity/"
                              "engineering unresolved, never a verdict")},
        "decision": {"exits": {"a": "candidate 72h t2m skill > 0, all seeds",
                               "b": "candidate-control 72h delta RMSE < 0, all seeds",
                               "c": "candidate 48h t2m skill > 0, all seeds",
                               "d": "candidate-control delta RMSE at 6/12/24h, seed mean <= 0"},
                     "advance": "all of (a)-(d) -> independent review",
                     "close": "any of (a)-(d) false -> close the climatology-anchor anomaly-feedback "
                              "branch at matched compute",
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
                    "soft_rule": "soft overrun continues and is recorded; only the hard cap truncates",
                    "decision": "0060"},
        "test_read": False,
        "test_read_policy": ("the 2023 four-month test split is never opened; only the train "
                             "manifest (fit) and the val manifest (scoring) are read"),
        "scientific_claim": False, "limitations": list(LIMITATIONS),
    }


def arm_protocol(spec, feedback, *, train_identity, windows, canonical_digest):
    """One arm's frozen record. The only fields the two arms may differ in are named here."""
    return {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
            "lr": LR, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "clip": CLIP, "bf16": BF16,
            "checkpoint_every": CHECKPOINT_EVERY, "physical_weights": list(PHYSICAL_WEIGHTS),
            "train_manifest": str(TRAIN_MANIFEST), "train_data_identity": train_identity,
            "train_windows": windows, "anomaly_feedback": bool(feedback),
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
    check_model_revision(model_code_digest())
    # Hash the 5.49 GB source once, here; later stages compare this digest plus the file's
    # size and mtime instead of re-reading it in each of the five stages.
    source_stat, source_digest = SOURCE.stat(), sha256_file(SOURCE)
    if source_digest != SOURCE_SHA256:
        raise RuntimeError("the merged multi-year source drifted from its frozen SHA256")
    spec = _model_spec()
    windows = preflight_long_rollout_windows(TRAIN_MANIFEST, physical_steps=12)
    if windows.get("input_windows") != TRAIN_WINDOWS:
        raise RuntimeError("the treatment manifest's window count changed")
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    arms = {name: arm_protocol(arm_spec(spec, anchor_payload(), feedback), feedback,
                               train_identity=train_identity, windows=windows,
                               canonical_digest=canonical_digest)
            for name, feedback in ARMS}
    require_single_factor(arms[CONTROL], arms[CANDIDATE])
    body = {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "source_file": {"path": str(SOURCE), "sha256": source_digest,
                            "bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
            "arms": {**arms, "parent": {"pins": {}, "checkpoints": parent_pins.pins()}},
            "manifest_sha256": {"train": sha256_file(TRAIN_MANIFEST), "val": sha256_file(VAL_MANIFEST)},
            "train_data_identity": train_identity,
            "val_data_identity": dataset_identity(VAL_MANIFEST)[0],
            "climatology_reference": climatology_reference(),
            "code": recipe._shared().code_state()}
    parent_pins.install_parent_pins(body)
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**frozen_constants(anchor_payload()),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(DEVICE_MODE, single_uuid=wide.GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS, "started_perf_counter": started,
            "deadline_perf_counter": deadline, "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
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
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError("execution source changed: " + name)
    check_model_revision(model_code_digest())
    if body["model_code_sha256"] != model_code_digest() \
            or body["training_code_sha256"] != training_code_digest():
        raise RuntimeError("model/training source identity changed")
    # The source is hashed once in prepare and that digest is bound into this protocol;
    # re-hashing 5.49 GB per stage buys nothing the frozen digest plus the file's own
    # size/mtime does not. A same-size, same-mtime replacement is the one mutation this
    # no longer catches, and it would have to be deliberate.
    recorded_source = body.get("source_file") or {}
    observed_source = SOURCE.stat()
    if (recorded_source.get("path") != str(SOURCE) or recorded_source.get("sha256") != SOURCE_SHA256
            or recorded_source.get("sha256") != body["source_sha256"]
            or observed_source.st_size != recorded_source.get("bytes")
            or observed_source.st_mtime_ns != recorded_source.get("mtime_ns")):
        raise RuntimeError("the merged multi-year source identity changed")
    # The dataset identity is computed once in prepare; the protocol digest binds it to this
    # body, and the manifest bytes are checked cheaply here.
    if body["manifest_sha256"] != {"train": sha256_file(TRAIN_MANIFEST),
                                   "val": sha256_file(VAL_MANIFEST)}:
        raise RuntimeError("a train/val manifest changed after freeze")
    require_single_factor(body["arms"][CONTROL], body["arms"][CANDIDATE])
    for name, feedback in ARMS:
        arm = body["arms"][name]
        spec = arm["model"]["spec"]
        if (arm["train_manifest"] != str(TRAIN_MANIFEST)
                or arm["train_windows"].get("input_windows") != TRAIN_WINDOWS
                or len(arm["train_windows"].get("window_sha256", "")) != 64
                or int(arm["updates"]) != UPDATES or int(arm["batch_size"]) != BATCH_SIZE
                or arm["anomaly_feedback"] is not bool(feedback)
                or spec.get("anomaly_feedback") is not bool(feedback)
                or spec.get("anchor_output_prior") is not False
                or spec.get("climatology_anchor_spec") != anchor_payload()["spec"]
                or int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4
                or arm["model"]["spec_canonical_digest"] != canonical_digest(spec)):
            raise RuntimeError(f"the {name} arm changed from the frozen record")
    if body["climatology_anchor"] != {"arms": {name: anchor_block(name, feedback, anchor_payload())
                                               for name, feedback in ARMS}}:
        raise RuntimeError("the frozen climatology-anchor rule changed")
    if body["climatology_reference"] != climatology_reference():
        raise RuntimeError("the climatology reference changed")
    if body["arms"]["parent"]["checkpoints"] != parent_pins.pins() \
            or body["arms"]["parent"]["pins"]["checkpoints"] != parent_pins.pins() \
            or body["arms"]["parent"]["pins"]["model_only_migration"] \
            != parent_pins.migration_provenance():
        raise RuntimeError("the migrated parent pins changed")
    if not devices.assignment_is_valid(body, wide.GPU_UUID):
        raise RuntimeError("declared GPU differs from the frozen assignment")
    if (body["planned_seconds"] != PLANNED_SECONDS
            or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS
            or int(body["estimated_peak_bytes"]) != ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("execution budget or gate differs from freeze")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["format"] != FORMAT or body["decision"]["text"] != DECISION_RULE):
        raise RuntimeError("frozen claims/decision rule changed")
    if int(body["recipe"]["updates"]) != UPDATES or int(body["recipe"]["batch_size"]) != BATCH_SIZE:
        raise RuntimeError("the matched-compute factor changed")
    return body


def _parent_model(body, seed, feedback):
    from training.r7_experiment import load_checkpoint
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    path, expected = parent_pins.pinned_parent(seed)
    if (sha256_file(path) != expected
            or body["arms"]["parent"]["checkpoints"][str(seed)]["sha256"] != expected):
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != parent_pins.ENDPOINT or contract["seed"] != seed
            or contract["mode"] != "l6" or contract["data_identity"] != shared.V3_DATA_IDENTITY
            or contract["source_sha256"] != shared.V3_SOURCE_SHA256):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"][arm_name(feedback)]["model"]["spec"]
    model, state, expansion = anchor_parent.expand_parent_state(
        saved["model"], spec, anchor_payload(), seed=seed)
    return model, state, path, expected, expansion


def _arm_contract(body, name, feedback, parent, parent_sha, expansion):
    arm = body["arms"][name]
    return {"kind": "process", "model": arm["model"]["spec"],
            "data_identity": arm["train_data_identity"], "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": parent_pins.ENDPOINT,
                               "anchor_table_run": str(ANCHOR_TABLE_RUN),
                               "anchor_table_sha256": expansion["anchor_table_sha256"],
                               "description": ("registered v3-BD l6 endpoint, model-only migration "
                                               "under the current revision, then the published "
                                               "48-bucket train-only table; "
                                               f"anomaly_feedback={bool(feedback)}")},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(PHYSICAL_WEIGHTS),
                               "window_sha256": arm["train_windows"]["window_sha256"],
                               "excluded_sample_ids": arm["train_windows"]["excluded_sample_ids"]}}


def _check_provenance(provenance, seed, name, lead):
    if provenance["split"] != "val" or "test_read" in provenance:
        raise RuntimeError(f"seed {seed} {name} lead {lead} did not stay on val")
    if (provenance["n_evaluated"] != VAL_COHORTS[str(lead)]
            or provenance["n_available_windows"] != VAL_COHORTS[str(lead)]
            or provenance["lead_hours"] != [lead]):
        raise RuntimeError(f"seed {seed} {name} lead {lead} cohort differs from the frozen one")
    regions = {region["region"]: region for region in
               (provenance["boundary_scoring"] or {}).get("regions", [])}
    if regions.get(REGION, {}).get("n_grid_points") != FULL_GRID_POINTS:
        raise RuntimeError(f"seed {seed} {name} lead {lead} full region is not 65x65")
    climatology = provenance["climatology"]
    if (climatology["training_years"] != CLIMATOLOGY_YEARS
            or len(climatology["bucket_counts"]) != CLIMATOLOGY_BUCKETS):
        raise RuntimeError(f"seed {seed} {name} lead {lead} climatology is not the 48-bucket fit")


def _train_arm(output, body, seed, name, feedback, deadline):
    """Fine-tune one arm from the migrated parent and score its five leads."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    spec = body["arms"][name]["model"]["spec"]
    model, state, parent, parent_sha, expansion = _parent_model(body, seed, feedback)
    contract = _arm_contract(body, name, feedback, parent, parent_sha, expansion)
    folder = output / f"seed{seed}/training/{name}"
    print(f"[seed {seed}] arm {name}: training {UPDATES} updates at batch {BATCH_SIZE} "
          "from the migrated parent", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        TRAIN_MANIFEST, folder, model=model, contract=contract, parent_weights=state,
        physical_weights=PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline, batch_size=BATCH_SIZE)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["arms"][name]["train_data_identity"]
            or report["supervision"]["kind"] != "full_grid"):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    update1 = dict(report["losses"][0])
    receipt = {"seed": seed, "arm": name, "anomaly_feedback": bool(feedback),
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES,
               "batch_size": BATCH_SIZE, "reasoning_steps": STEPS,
               "train_data_identity": report["data_identity"],
               "train_windows_sha256": body["arms"][name]["train_windows"]["window_sha256"],
               "expansion": expansion,
               "update1": {"loss": update1["loss"], "gradient_norm": update1["gradient_norm"]},
               "evaluations": {}}
    print(f"[seed {seed}] arm {name}: trained in {report['elapsed_seconds']:.0f}s; "
          "building the shared evaluation inputs", flush=True)
    # One manifest reader, one store scan and one training climatology for all five leads:
    # none of them depends on the lead, and rebuilding them per lead dominated the arm.
    prepared = EvaluationPreparation(VAL_MANIFEST, deadline=deadline)
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{name}/lead_{lead:03d}h"
        print(f"[seed {seed}] arm {name}: evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(VAL_MANIFEST), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=BOUNDARY_MARGINS, deadline=deadline,
            prepared=prepared)
        _check_provenance(provenance, seed, name, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
        print(f"[seed {seed}] arm {name}: lead {lead}h done in "
              f"{provenance['elapsed_seconds']:.0f}s over {provenance['n_evaluated']} cases", flush=True)
    return receipt


def _seed(output, body, seed, deadline):
    import torch
    receipts = {}
    for name, feedback in ARMS:
        print(f"[seed {seed}] arm {name}: start", flush=True)
        receipt = _train_arm(output, body, seed, name, feedback, deadline)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
        print(f"[seed {seed}] arm {name}: done", flush=True)
    # Both arms must carry the same published climatology bytes; a mismatch means the factor is
    # not the only difference between them.
    tables = {name: receipts[name]["expansion"]["anchor_table_sha256"] for name, _ in ARMS}
    if len(set(tables.values())) != 1:
        raise RuntimeError(f"seed {seed}: the arms do not share one climatology table: {tables}")
    if receipts[CONTROL]["expansion"]["expanded_state_sha256"] \
            != receipts[CANDIDATE]["expansion"]["expanded_state_sha256"]:
        raise RuntimeError(f"seed {seed}: the arms do not start from one expanded state")


def evaluate_exit(seeds, *, seeds_order=SEEDS):
    """The predeclared (a)-(d) conjunction; disagreement is never averaged away."""
    seeds = {str(key): value for key, value in seeds.items()}
    def skill(seed, lead):
        return seeds[str(seed)]["arms"][CANDIDATE]["full_t2m_skill"][str(lead)]
    def delta(seed, lead):
        return seeds[str(seed)]["delta_candidate_minus_control"][str(lead)]
    a = all(skill(seed, 72) > 0.0 for seed in seeds_order)
    b = all(delta(seed, 72) < 0.0 for seed in seeds_order)
    c = all(skill(seed, 48) > 0.0 for seed in seeds_order)
    means = {lead: sum(delta(seed, lead) for seed in seeds_order) / len(seeds_order)
             for lead in (6, 12, 24)}
    d = all(value <= 0.0 for value in means.values())
    passed = bool(a and b and c and d)
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d),
            "seed_mean_delta_6_12_24": {str(lead): value for lead, value in means.items()},
            "all": passed, "compute": {"updates": UPDATES, "batch_size": BATCH_SIZE},
            "verdict": "advance-to-independent-review" if passed
            else "close-anchor-anomaly-feedback-branch-at-matched-compute"}


def bridge_verdict(receipts):
    """Rule (e): an identity check at identical compute, explicitly not a scientific verdict."""
    seeds = {}
    for seed in SEEDS:
        observed = receipts[str(seed)][CONTROL]["update1"]
        expected = BRIDGE_UPDATE1_LOSS[int(seed)]
        seeds[str(seed)] = {"expected_loss": expected, "observed_loss": observed["loss"],
                            "expected_gradient_norm": BRIDGE_UPDATE1_GRADIENT_NORM[int(seed)],
                            "observed_gradient_norm": observed["gradient_norm"],
                            "bit_identical_loss": observed["loss"] == expected}
    return {"scientific": False, "record_id": BRIDGE_RECORD,
            "note": ("R1's own bridge failed four of five leads because it compared different "
                     "compute; this one compares update 1 at the same store, manifest, seed, "
                     "batch, parent and schedule"),
            "seeds": seeds, "passed": all(item["bit_identical_loss"] for item in seeds.values())}


def _readings(output, body):
    from training.r7_arm_harness import sha256_file
    receipts = {str(seed): {name: json.loads((output / f"seed{seed}_{name}_receipt.json")
                                             .read_text(encoding="utf-8"))
                            for name, _ in ARMS} for seed in SEEDS}
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": DECISION_RULE,
                "factor": body["factor"]["name"], "updates": UPDATES, "batch_size": BATCH_SIZE,
                "closed_round": {"record": CLOSED_ROUND_RECORD, "updates": CLOSED_ROUND_UPDATES,
                                 "batch_size": CLOSED_ROUND_BATCH_SIZE},
                "declared_difference": ("both arms train on the same manifest at the same compute "
                                        "and install the same 48-bucket table with "
                                        "anchor_output_prior=False; they differ only in "
                                        "anomaly_feedback"),
                "anchor_table": {key: anchor_payload()["pins"][key]
                                 for key in ("run", "table_sha256", "spec_sha256", "n_buckets",
                                             "n_selected_steps", "source_sha256",
                                             "train_data_identity")},
                "climatology_reference_sha256": sha256_file(D2_REFERENCE), "seeds": {}}
    for seed in SEEDS:
        entry = {"seed": seed, "arms": {}}
        for name, _ in ARMS:
            arm = {"anomaly_feedback": bool(dict(ARMS)[name]), "full_t2m_rmse": {}, "full_t2m_skill": {}}
            for lead in EVALUATION_LEADS:
                table = Path(receipts[str(seed)][name]["evaluations"][str(lead)]["dir"]) / "boundary_rmse.csv"
                value = wide._region_variable_rmse(table, REGION, "t2m")
                denominator = body["climatology_reference"]["leads"][str(lead)]["t2m_rmse"]
                arm["full_t2m_rmse"][str(lead)] = value
                arm["full_t2m_skill"][str(lead)] = 1.0 - (value / denominator) ** 2
            entry["arms"][name] = arm
        entry["delta_candidate_minus_control"] = {str(lead): (
            entry["arms"][CANDIDATE]["full_t2m_rmse"][str(lead)]
            - entry["arms"][CONTROL]["full_t2m_rmse"][str(lead)]) for lead in EVALUATION_LEADS}
        readings["seeds"][str(seed)] = entry
    readings["exit_conditions"] = evaluate_exit(readings["seeds"])
    readings["bridge_e"] = bridge_verdict(receipts)
    return readings


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        body = _prepared_protocol(output)
        print(f"[worker] prepare done: protocol {body['protocol_sha256'][:12]}", flush=True)
        return body
    if phase == "archive":
        body = dose.interior._archive(output, validate_protocol(output))
        print("[worker] archive done", flush=True)
        return body
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", _readings(output, body))
        print("[worker] reading done", flush=True)
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)
    print(f"[seed {seed}] done", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise RuntimeError("dual-card mode is driven by `r7_round.py stage-seeds`, not by main")
    return worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    main()
