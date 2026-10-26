"""S3 FY year coverage, re-run with matched compute: 5 years vs 22 years, months fixed.

Decision 0054's year-coverage round closed the training-year coverage branch: exits
(a)(b)(c) were true, (d) was false, and the round's own reading said the treatment saw about
7% of an epoch. Decision 0061 authorises exactly one re-open, under a *named user override*
rather than a new regime: store, grid, supervision region and data span are byte-for-byte
the closed round's, and the only change is compute (``updates`` 2,400 -> 16,071, ``batch``
1 -> 2, so ``updates * batch >= 32,142`` and each arm sees at least one epoch). Decision
0060 supplies the budget constants.

Everything else is copied verbatim from ``scripts/study_r7_s3_fy_year_coverage.py``: same
65x65 multi-year store, same 2000-2021 train-only normalization, same 48-bucket train-only
climatology denominator, same migrated parent, same spec (K=4/depth=4/dim=192), same seeds
41/42/43, same 2022 full-year val split, same LR/warmup/clip/physical weights, same
predeclared exits (a)-(d), same non-scientific bridge (e). The bridge compares this round's
control against the registered FY annual-coverage candidate arm; their normalizations and
climatology denominators differ, so it can only catch an identity or engineering error.
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

from scripts import r7_s3_fy_annual_parent as parent_pins
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.r7_s3_fy_year_coverage_climatology import (
    CLIMATOLOGY_BUCKETS, CLIMATOLOGY_STEPS, CLIMATOLOGY_T2M_RMSE, CLIMATOLOGY_YEARS, D2_FORMAT,
    D2_REFERENCE, EVALUATION_LEADS, VAL_COHORTS, climatology_reference)
from scripts.r7_s3_fy_year_coverage_matched_support import (
    BRIDGE_RECORD, BRIDGE_UPDATES, CLOSED_ROUND_UPDATES, DECISION_RULE, LIMITATIONS, bridge_reference)
from training import r7_device_selection as devices

FORMAT = "r7-s3-fy-year-coverage-matched-protocol-v1"
INSTANCE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01"
STORE = INSTANCE / "store/cache.zarr"
SOURCE = INSTANCE / "source.nc"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"
SOURCE_BYTES, STORE_SHAPE = 5492365397, [34084, 17, 65, 65]
CONTROL, TREATMENT = "control", "treatment"
ARMS = ((CONTROL, INSTANCE / "store/manifests_control"),
        (TREATMENT, INSTANCE / "store/manifests_treatment"))
TRAIN_COUNTS = {CONTROL: 7291, TREATMENT: 32142}
VAL_COUNT, TEST_COUNT = 1458, 472
# The one changed factor's arithmetic: 16,071 * 2 = 32,142 >= the treatment manifest.
UPDATES, BATCH_SIZE = 16071, 2
STEPS, LR, WARMUP = wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, CLIP = wide.WEIGHT_DECAY, wide.CLIP
CHECKPOINT_EVERY, BF16, PHYSICAL_WEIGHTS = wide.CHECKPOINT_EVERY, wide.BF16, wide.PHYSICAL_WEIGHTS
REASONING_STEPS, SEEDS = wide.REASONING_STEPS, wide.SEEDS
BOUNDARY_MARGINS, REGION, FULL_GRID_POINTS, DEVICE_MODE = (1,), "full", 4225, "dual"
# Decision 0060: the closed round's 43,200 s was calibrated on 2,400 updates per arm.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 129600., 259200., 518400.
ESTIMATED_PEAK_BYTES, HEADROOM_MARGIN_BYTES = 4096 * 2**20, 512 * 2**20
ENTRY_FILES = ("scripts/r7_s3_fy_annual_parent.py", "scripts/study_r7_s3_fy_year_coverage_matched.py",
               "scripts/r7_s3_fy_year_coverage_climatology.py", "scripts/build_r7_s3_fy_year_manifests.py",
               "scripts/r7_s3_fy_year_coverage_matched_support.py",
               "scripts/build_r7_s3_fy_multiyear_source.py", *wide.execution_files())


def execution_files():
    return sorted(set(ENTRY_FILES))


def manifest_path(name, split):
    """The arm's manifest file, read from ``ARMS`` so the two cannot drift."""
    folders = dict(ARMS)
    if name not in folders:
        raise ValueError(f"unknown arm {name!r}; declared arms are {[arm for arm, _ in ARMS]}")
    return folders[name] / f"{split}.jsonl"


def frozen_constants():
    """Every protocol field that does not depend on the built store (unit-testable)."""
    return {
        "format": FORMAT, "stage": "S3-FY-YC-MATCHED", "instance": str(INSTANCE),
        "store": str(STORE),
        "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES, "store_shape": list(STORE_SHAPE),
        "split_years": {"train": CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifests": {name: str(manifest_path(name, "train")) for name, _ in ARMS},
        "val_manifests": {name: str(manifest_path(name, "val")) for name, _ in ARMS},
        "test_manifest_never_read": {name: str(manifest_path(name, "test")) for name, _ in ARMS},
        "train_file_counts": dict(TRAIN_COUNTS), "val_count": VAL_COUNT, "test_count": TEST_COUNT,
        "val_cohorts": dict(VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS), "scored_region": REGION,
                   "n_grid_points": FULL_GRID_POINTS, "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"},
        "factor": {"name": "training year coverage at matched compute",
                   "control": "2017-2021 (5 years, 12 months)",
                   "treatment": "2000-2021 (22 years, 12 months)",
                   "months_held_fixed": list(range(1, 13)),
                   "controlled": ["store", "normalization", "model spec", "migrated parent",
                                  "K", "depth", "dim", "physical weights", "seeds", "val split",
                                  "climatology denominator", "months", "lr", "warmup", "clip",
                                  "supervision", "updates", "batch_size"],
                   "closed_round_updates": CLOSED_ROUND_UPDATES,
                   "changed_from_closed_round": ["updates", "batch_size"],
                   "not_controlled": []},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS), "reasoning_steps": REASONING_STEPS,
                   "epochs_floor": "updates * batch_size = 32142 >= the treatment manifest, so each arm sees >= 1 epoch",
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)"},
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1",
                        "buckets": CLIMATOLOGY_BUCKETS, "training_years": CLIMATOLOGY_YEARS,
                        "n_selected_steps": CLIMATOLOGY_STEPS, "t2m_rmse": dict(CLIMATOLOGY_T2M_RMSE),
                        "reference_format": D2_FORMAT, "reference_path": str(D2_REFERENCE)},
        "bridge_e": {"record_id": BRIDGE_RECORD, "updates": BRIDGE_UPDATES, "scientific": False,
                     "reference": bridge_reference(),
                     "rule": ("control seed mean must stay within the registered arm's seed spread "
                              "or 5% of its mean, whichever is larger, at every lead; a failure is "
                              "identity/engineering unresolved, not a verdict")},
        "decision": {"exits": {"a": "treatment 72h t2m skill > 0, all seeds", "b": "treatment-control 72h delta RMSE < 0, all seeds",
                               "c": "treatment 48h t2m skill > 0, all seeds", "d": "treatment-control delta RMSE at 6/12/24h, seed mean <= 0"},
                     "advance": "all of (a)-(d) -> independent review",
                     "close": "any of (a)-(d) false -> close the training-year coverage branch at matched compute",
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
                             "manifests (fit) and the val manifests (scoring) are read"),
        "scientific_claim": False, "limitations": list(LIMITATIONS),
    }


def arm_protocol(spec, *, train_manifest, train_identity, windows, canonical_digest):
    """One arm's frozen record. The only fields the two arms may differ in are named here."""
    return {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
            "lr": LR, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "clip": CLIP, "bf16": BF16,
            "checkpoint_every": CHECKPOINT_EVERY,
            "physical_weights": list(PHYSICAL_WEIGHTS),
            "train_manifest": str(train_manifest),
            "train_data_identity": train_identity, "train_windows": windows,
            "model": {"kind": "process", "spec": spec,
                      "spec_canonical_digest": canonical_digest(spec)},
            "selection": "frozen endpoint; no validation selection"}


ARM_FACTOR_KEYS = ("train_data_identity", "train_manifest", "train_windows")


def difference_keys(control, treatment):
    """Fields that differ between the two arms; must equal ARM_FACTOR_KEYS exactly."""
    return sorted(key for key in set(control) | set(treatment)
                  if control.get(key) != treatment.get(key))


def require_single_factor(control, treatment):
    """Raise unless the only arm differences are the training manifest and its identity."""
    observed = tuple(difference_keys(control, treatment))
    if observed != ARM_FACTOR_KEYS:
        raise ValueError(f"the arms differ in {observed}, expected exactly {ARM_FACTOR_KEYS}")
    if control["model"] != treatment["model"]:
        raise ValueError("the arms use different model specs")
    if control["batch_size"] != BATCH_SIZE or treatment["batch_size"] != BATCH_SIZE:
        raise ValueError("an arm is not running at the round's frozen batch size")


def check_model_revision(observed, *, expected=parent_pins.MODEL_CODE_SHA256):
    """The data round must run on the revision that produced the migrated parent."""
    if observed != expected:
        raise RuntimeError(f"model revision {observed} is not the pinned {expected}; "
                           "commit or clear the working-tree model changes first")


def _model_spec():
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    if any(spec.get(key) for key in ("climatology_anchor_spec", "anomaly_feedback",
                                     "anchor_output_prior", "process_init_conditioning",
                                     "typed_evidence_mode", "global_mixing_mode",
                                     "tendency_input")):
        raise RuntimeError("the archived process spec already carries a closed-architecture switch")
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
    arms, manifest_sha, val_identity = {}, {}, None
    for name, _ in ARMS:
        train, val = manifest_path(name, "train"), manifest_path(name, "val")
        train_identity, _ = dataset_identity(train)
        manifest_sha[name] = {"train": sha256_file(train), "val": sha256_file(val)}
        if val_identity is None:
            val_identity = dataset_identity(val)[0]
        elif manifest_sha[name]["val"] != manifest_sha[CONTROL]["val"]:
            raise RuntimeError("the two val manifests are not byte-identical")
        arms[name] = arm_protocol(spec, train_manifest=train, train_identity=train_identity,
            windows=preflight_long_rollout_windows(train, physical_steps=12),
            canonical_digest=canonical_digest)
    require_single_factor(arms[CONTROL], arms[TREATMENT])
    body = {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "source_file": {"path": str(SOURCE), "sha256": source_digest,
                            "bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
            "arms": {**arms, "parent": {"pins": {}, "checkpoints": parent_pins.pins()}},
            "manifest_sha256": manifest_sha, "val_data_identity": val_identity,
            "climatology_reference": climatology_reference(),
            "code": recipe._shared().code_state()}
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
    # The full dataset identity is computed once in prepare; repeating it here would validate
    # every record against the store on every stage. The manifest bytes are checked cheaply,
    # and the protocol digest binds the recorded dataset identity to this body.
    observed = {name: {"train": sha256_file(manifest_path(name, "train")),
                       "val": sha256_file(manifest_path(name, "val"))} for name, _ in ARMS}
    if body["manifest_sha256"] != observed:
        raise RuntimeError("a train/val manifest changed after freeze")
    if observed[CONTROL]["val"] != observed[TREATMENT]["val"]:
        raise RuntimeError("the two val manifests are no longer byte-identical")
    require_single_factor(body["arms"][CONTROL], body["arms"][TREATMENT])
    for name, _ in ARMS:
        arm = body["arms"][name]
        train = manifest_path(name, "train")
        if (arm["train_manifest"] != str(train)
                or arm["train_windows"].get("input_windows") != TRAIN_COUNTS[name]
                or len(arm["train_windows"].get("window_sha256", "")) != 64
                or arm["model"]["spec_canonical_digest"]
                != canonical_digest(arm["model"]["spec"])):
            raise RuntimeError(f"the {name} arm changed from the frozen record")
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


def _parent_model(body, seed):
    from training.r7_experiment import load_checkpoint, make_model
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
    spec = body["arms"][CONTROL]["model"]["spec"]
    model = make_model("process", spec)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved["model"], path, expected


def _arm_contract(body, name, parent, parent_sha):
    arm = body["arms"][name]
    return {"kind": "process", "model": arm["model"]["spec"],
            "data_identity": arm["train_data_identity"], "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent_pins.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": parent_pins.ENDPOINT,
                               "description": ("registered v3-BD l6 endpoint, model-only "
                                               "migration; region-agnostic spec reused on the "
                                               "65x65 multi-year store")},
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


def _train_arm(output, body, seed, name, deadline):
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha = _parent_model(body, seed)
    folder = output / f"seed{seed}/training/{name}"
    print(f"[seed {seed}] arm {name}: training {UPDATES} updates at batch {BATCH_SIZE} "
          "from the migrated parent", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        body["arms"][name]["train_manifest"], folder, model=model,
        contract=_arm_contract(body, name, parent, parent_sha), parent_weights=state,
        physical_weights=PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline, batch_size=BATCH_SIZE)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or int(report["contract"]["batch_size"]) != BATCH_SIZE
            or report["data_identity"] != body["arms"][name]["train_data_identity"]
            or report["supervision"]["kind"] != "full_grid"):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    receipt = {"seed": seed, "arm": name, "training_dir": str(folder),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "batch_size": BATCH_SIZE,
               "reasoning_steps": STEPS, "train_data_identity": report["data_identity"],
               "train_windows_sha256": body["arms"][name]["train_windows"]["window_sha256"],
               "evaluations": {}}
    print(f"[seed {seed}] arm {name}: trained in {report['elapsed_seconds']:.0f}s; "
          "building the shared evaluation inputs", flush=True)
    # One manifest reader, one store scan and one training climatology for all five leads:
    # none of them depends on the lead, and rebuilding them per lead dominated the arm.
    prepared = EvaluationPreparation(manifest_path(name, "val"), deadline=deadline)
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{name}/lead_{lead:03d}h"
        print(f"[seed {seed}] arm {name}: evaluating lead {lead}h", flush=True)
        provenance = evaluate_local(
            str(manifest_path(name, "val")), output_dir=destination, checkpoint=checkpoint,
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
    for name, _ in ARMS:
        print(f"[seed {seed}] arm {name}: start", flush=True)
        receipt = _train_arm(output, body, seed, name, deadline)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
        print(f"[seed {seed}] arm {name}: done", flush=True)
    if receipts[CONTROL]["train_data_identity"] == receipts[TREATMENT]["train_data_identity"]:
        raise RuntimeError(f"seed {seed}: the arms trained on the same manifest identity")


def evaluate_exit(seeds, *, seeds_order=SEEDS):
    """The predeclared (a)-(d) conjunction; disagreement is never averaged away."""
    seeds = {str(key): value for key, value in seeds.items()}
    def skill(seed, lead):
        return seeds[str(seed)]["arms"][TREATMENT]["full_t2m_skill"][str(lead)]
    def delta(seed, lead):
        return seeds[str(seed)]["delta_treatment_minus_control"][str(lead)]
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
            else "close-training-year-coverage-branch-at-matched-compute"}


def bridge_verdict(control_rmse, reference, *, seeds_order=SEEDS):
    """Rule (e): an engineering bridge, explicitly not a scientific verdict."""
    control_rmse = {str(key): value for key, value in control_rmse.items()}
    leads = {}
    for lead in EVALUATION_LEADS:
        values = [control_rmse[str(seed)][str(lead)] for seed in seeds_order]
        mean = sum(values) / len(values)
        expected = reference[str(lead)]
        difference = mean - expected["mean"]
        leads[str(lead)] = {"control_mean": mean, "reference_mean": expected["mean"],
                            "difference": difference, "band": expected["band"],
                            "passed": abs(difference) <= expected["band"]}
    return {"scientific": False, "record_id": BRIDGE_RECORD,
            "note": ("normalization and climatology denominator differ; a failure is "
                     "identity/engineering unresolved, not a scientific verdict"),
            "leads": leads, "passed": all(item["passed"] for item in leads.values())}


def _readings(output, body):
    from training.r7_arm_harness import sha256_file
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": DECISION_RULE,
                "factor": body["factor"]["name"], "updates": UPDATES, "batch_size": BATCH_SIZE,
                "closed_round_updates": CLOSED_ROUND_UPDATES,
                "declared_difference": ("the arms share the store, normalization, model spec, "
                                        "parent, seeds, val split and climatology; only the train "
                                        "manifest and its data identity differ"),
                "climatology_reference_sha256": sha256_file(D2_REFERENCE), "seeds": {}}
    for seed in SEEDS:
        entry = {"seed": seed, "arms": {}}
        for name, _ in ARMS:
            receipt = json.loads(
                (output / f"seed{seed}_{name}_receipt.json").read_text(encoding="utf-8"))
            arm = {"train_data_identity": receipt["train_data_identity"],
                   "full_t2m_rmse": {}, "full_t2m_skill": {}}
            for lead in EVALUATION_LEADS:
                table = Path(receipt["evaluations"][str(lead)]["dir"]) / "boundary_rmse.csv"
                value = wide._region_variable_rmse(table, REGION, "t2m")
                denominator = body["climatology_reference"]["leads"][str(lead)]["t2m_rmse"]
                arm["full_t2m_rmse"][str(lead)] = value
                arm["full_t2m_skill"][str(lead)] = 1.0 - (value / denominator) ** 2
            entry["arms"][name] = arm
        entry["delta_treatment_minus_control"] = {str(lead): (
            entry["arms"][TREATMENT]["full_t2m_rmse"][str(lead)]
            - entry["arms"][CONTROL]["full_t2m_rmse"][str(lead)]) for lead in EVALUATION_LEADS}
        readings["seeds"][str(seed)] = entry
    readings["exit_conditions"] = evaluate_exit(readings["seeds"])
    readings["bridge_e"] = bridge_verdict(
        {str(seed): readings["seeds"][str(seed)]["arms"][CONTROL]["full_t2m_rmse"]
         for seed in SEEDS}, body["bridge_e"]["reference"])
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
    print(f"[worker] seed {seed} done", flush=True)


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
