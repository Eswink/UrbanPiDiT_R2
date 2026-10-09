"""Single-factor wide-region (129x129) arm: the input region is the only change.

The registered 800-update S3 rollout-dose screen read "supported" on the narrow
65x65 store, but its absolute t2m skill stays negative at 48/72 h. This driver
applies that recipe unchanged to the wide 129x129 store - the one changed factor -
and scores the frozen central 65x65 box (``boundary_masks(129,129,(32,))``
``['interior_32']``). The migrated v3-BD 1600 parent is reused (region-agnostic
model spec). A positive result supports "the frozen box lacks long-lead
information"; a null/negative result does not refute it, since supervision-area
dilution is a possible cause. Structure mirrors ``study_r7_s3_rollout_dose.py``:
a three-stage protocol freeze, ``--worker {prepare,archive,seed,reading}``, a
``code.zip`` archive plus receipt, an ``execution_files_sha256`` identity, the
read-only GPU co-residency gate before every spawn, and bounded workers. The 2023
test split is never read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import study_r7_s3_rollout_dose as dose
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts import study_r7_s3_v3_rollout_ft_worker as short_worker
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training.r7_study_process import bounded_process, worker_environment

FORMAT = "r7-s3-wide-single-factor-protocol-v1"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"  # co-resident GPU0 at freeze time
PHYSICAL_WEIGHTS = (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
UPDATES, LR, WARMUP = 800, 2e-5, 10
STEPS, WEIGHT_DECAY, BATCH_SIZE, CLIP, CHECKPOINT_EVERY, BF16 = 4, 1e-4, 1, 1., 20, False
SEEDS = (41, 42, 43)
EVALUATION_LEADS = (6, 12, 24, 48, 72)
REASONING_STEPS = 4
BOUNDARY_MARGINS = (32,)  # 129x129 interior_32 == the frozen central 65x65 box
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 4200., 9000., 18000.
INSTANCE = ROOT / "outputs/r7_s3_wide_instance_v4_20261009_attempt01"
STORE = INSTANCE / "store/cache.zarr"
TRAIN_MANIFEST = INSTANCE / "store/manifests/train.jsonl"
VAL_MANIFEST = INSTANCE / "store/manifests/val.jsonl"
TEST_MANIFEST = INSTANCE / "store/manifests/test.jsonl"
SOURCE_PREFLIGHT = INSTANCE / "store/manifests/source_preflight.json"
DEFAULT_OUT = ROOT / "outputs/r7_s3_wide_single_factor_20261009_attempt01"
WIDE_CLIMATOLOGY = ROOT / "outputs/r7_s3_wide_d2_baselines_20261009_attempt01/climatology"
EVIDENCE_INDEX = ROOT / "docs/R7_EVIDENCE_INDEX.jsonl"
NARROW_CONTROL_RECORD = "s3-rollout-dose"
# Measured on the wide val manifest with the frozen evaluate_local cohort rule (per-lead
# complete rollout windows, manifest init-times only). They equal the registered v3
# cohorts training/r7_s3_v3_screen.V3_VAL_COHORTS exactly, so they are pinned as such.
WIDE_VAL_COHORTS = {"6": 472, "12": 468, "24": 460, "48": 444, "72": 428}
ENTRY_FILES = (*short_worker.EXECUTION_ENTRY_FILES, "scripts/study_r7_s3_rollout_dose.py",
               "scripts/study_r7_s3_wide_single_factor.py")

TEST_READ_POLICY = ("the 2023 test split is never opened by this study; only the wide train "
                    "manifest (fit) and the wide val manifest (scoring) are read, and "
                    "test.jsonl is never read")
DESIGN_DECISION_TEXT = (
    "Single factor: the input region. The registered 800-update long-rollout recipe is applied "
    "unchanged to the wide 129x129 store, with full 129x129 supervision, and scored on the frozen "
    "central 65x65 box (boundary_masks(129,129,(32,))['interior_32']). A positive result supports "
    "'the frozen box lacks long-lead information'; a null/negative result does NOT refute it, because "
    "supervision-area dilution is a possible cause, and a same-supervision-domain, input-only-wider "
    "control arm would be needed to separate dilution from information gain.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of the wide "
    "arm's interior_32 t2m RMSE minus the registered narrow arm's (index record s3-rollout-dose, 800 "
    "updates, seeds 41/42/43) full-region t2m RMSE at the five leads {6,12,24,48,72} h, and (b) the "
    "wide arm's interior_32 t2m skill 1-(rmse_wide/rmse_climatology)**2 against the wide train-only "
    "(2017-2021) climatology. The narrow arm is a pinned reference, not retrained here.")
HYPOTHESIS_TEXT = (
    "At 48-72 h a synoptic system travels farther than the 16-degree frozen box, so widening the "
    "input region to 129x129 - the only changed factor - should let the same registered rollout recipe "
    "carry more long-lead information into the frozen central 65x65 box.")
DIFFERENCE_TEXT = (
    "One factor only: the input region. Same registered 800-update long-rollout recipe, same migrated "
    "v3-BD 1600 parent state (region-agnostic model spec), same LR 2e-5, warmup 10, 12-step physical "
    "weights, FP32, K4, batch 1 and clip 1 as the registered narrow arm.")
LIMITATIONS = [
    "development screening on one wide instance (129x129, one ROI, 17 channels, 2017-2021 train / "
    "2022 val); no significance, convergence, SOTA or generalization claim",
    "the narrow control is a pinned registered reference (docs/R7_S3_ROLLOUT_DOSE.md / index "
    "record s3-rollout-dose), not retrained inside this protocol",
    "the parent state was trained on the narrow 65x65 instance and is reused on the wide store "
    "because the model spec is region-agnostic; recorded here as a protocol note",
    "full 129x129 supervision confounds information gain with supervision-area dilution: a "
    "same-supervision-domain, input-only-wider arm would be needed to separate them",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident",
    "validation split only; the test split stays sealed until a later preregistered read",
]


def write_exclusive(path, body):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def execution_files():
    return sorted(set(short_worker.execution_files()) | set(ENTRY_FILES))


def source_sha256():
    """The source the wide store was built from, read from the build's own record."""
    report = json.loads(SOURCE_PREFLIGHT.read_text(encoding="utf-8"))
    fingerprint = report["fingerprint"]
    if fingerprint.get("scope") != "full-local-file":
        raise ValueError("the store source fingerprint is not the full local file")
    return fingerprint["sha256"]


def protocol_constants():
    """The frozen fields that do not depend on the built store (unit-testable offline)."""
    return {
        "format": FORMAT,
        "kind": "screen",
        "region": {"points": [129, 129], "frozen_box_points": [65, 65], "interior_selector": "boundary_masks(129,129,(32,))['interior_32']"},
        "store": str(STORE),
        "train_manifest": str(TRAIN_MANIFEST),
        "val_manifest": str(VAL_MANIFEST),
        "test_manifest_never_read": str(TEST_MANIFEST),
        "boundary_margins": list(BOUNDARY_MARGINS),
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "lr": LR, "warmup": WARMUP,
                   "physical_weights": list(PHYSICAL_WEIGHTS), "steps": STEPS,
                   "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "clip": CLIP,
                   "checkpoint_every": CHECKPOINT_EVERY, "bf16": BF16},
        "seeds": list(SEEDS),
        "evaluation_leads_hours": list(EVALUATION_LEADS),
        "evaluation": {"device": "cuda:0", "reasoning_steps": REASONING_STEPS, "max_samples": 10**9,
                       "boundary_margins": list(BOUNDARY_MARGINS)},
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS, "per_seed_seconds": PER_SEED_SECONDS},
        "design_decision": DESIGN_DECISION_TEXT,
        "decision_rule": DECISION_RULE_TEXT,
        "hypothesis": HYPOTHESIS_TEXT,
        "difference": DIFFERENCE_TEXT,
        "test_read": False,
        "test_read_policy": TEST_READ_POLICY,
        "scientific_claim": False,
        "limitations": LIMITATIONS,
    }


def _index_record(record_id):
    for line in EVIDENCE_INDEX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("record_id") == record_id:
                return record
    raise FileNotFoundError(f"evidence index record missing: {record_id}")


def narrow_control_pins():
    """Pin the registered narrow (65x65) rollout-dose arm this round is compared against."""
    from training.r7_arm_harness import sha256_file
    record = _index_record(NARROW_CONTROL_RECORD)
    doc = ROOT / record["evidence_path"]
    if sha256_file(doc) != record["evidence_sha256"]:
        raise RuntimeError("the registered narrow control doc drifted from its index record")
    run = dose.DEFAULT_SCREEN
    if not run.is_dir():
        raise FileNotFoundError(f"the registered narrow arm output is missing: {run}")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "updates": UPDATES, "seeds": list(SEEDS), "run_dir": str(run), "seeds_evaluations": {}}
    for seed in SEEDS:
        receipt = json.loads((run / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {"checkpoint_sha256": receipt["checkpoint_sha256"], "leads": {}}
        for lead in EVALUATION_LEADS:
            record_lead = receipt["evaluations"][str(lead)]
            if record_lead["n_evaluated"] != WIDE_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"registered narrow seed {seed} lead {lead} cohort differs")
            entry["leads"][str(lead)] = {"rmse_csv_sha256": record_lead["rmse_csv_sha256"],
                                         "dir": record_lead["dir"]}
        pins["seeds_evaluations"][str(seed)] = entry
    return pins


def _region_variable_rmse(path, region, variable):
    import csv
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] == region and row["variable"] == variable:
                return float(row["rmse"])
    raise RuntimeError(f"{region}/{variable} missing from {path}")


def wide_climatology_reference():
    """Pin the wide train-only climatology interior_32 t2m RMSE used for the skill read."""
    from training.r7_arm_harness import sha256_file
    if not WIDE_CLIMATOLOGY.is_dir():
        raise FileNotFoundError(f"wide climatology reference missing: {WIDE_CLIMATOLOGY}")
    reference = {"root": str(WIDE_CLIMATOLOGY), "leads": {}}
    for lead in EVALUATION_LEADS:
        path = WIDE_CLIMATOLOGY / f"lead_{lead:03d}h" / "boundary_rmse.csv"
        if not path.is_file():
            raise FileNotFoundError(f"wide climatology boundary table missing: {path}")
        reference["leads"][str(lead)] = {
            "boundary_rmse_csv": str(path), "sha256": sha256_file(path),
            "interior_32_t2m_rmse": _region_variable_rmse(path, "interior_32", "t2m")}
    return reference


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    shared = recipe._shared()
    parent = recipe.parent_pins()
    parent_result = json.loads((recipe.PARENT_RUN / "result.json").read_text(encoding="utf-8"))
    parent["evaluations"] = {seed: data["evaluations"] for seed, data in parent_result["seeds"].items()}
    parent["parent_result_path"] = str(recipe.PARENT_RUN / "result.json")
    migration = json.loads(dose.MIGRATION_RECEIPT.read_text(encoding="utf-8"))
    if migration["archived_model_code_sha256"] == migration["current_model_code_sha256"]:
        raise RuntimeError("migration receipt does not record a model identity change")
    for seed in SEEDS:
        record = migration["seeds"][str(seed)]
        if (record["migrated_path"] != str(dose.pinned_parent(seed)[0])
                or record["migrated_sha256"] != dose.MIGRATED_PARENT_SHA256[seed]
                or record["source_checkpoint_sha256"] != recipe.PARENT_CHECKPOINT_SHA256[seed]
                or record["state_digest"] == ""):
            raise RuntimeError(f"migration receipt disagrees with the pinned parent for seed {seed}")
    parent["checkpoints"] = {str(seed): {"path": str(dose.pinned_parent(seed)[0]),
                                        "sha256": dose.MIGRATED_PARENT_SHA256[seed]} for seed in SEEDS}
    parent["model_only_migration"] = {
        "receipt_path": str(dose.MIGRATION_RECEIPT), "receipt_sha256": sha256_file(dose.MIGRATION_RECEIPT),
        "export_sha256": migration["export_sha256"], "archived_code_commit": migration["archived_code_commit"],
        "archived_model_code_sha256": migration["archived_model_code_sha256"],
        "current_model_code_sha256": migration["current_model_code_sha256"],
        "operation": "archived-revision export, then strict current-code install; no optimizer/cursor/RNG",
        "forward_equivalence": "bit-identical FP32 forward on a fixed synthetic batch under both revisions",
        "region_note": ("the parent was trained on the narrow 65x65 instance; the model spec is "
                        "region-agnostic, so the same state is valid on the wide 129x129 store")}
    windows = preflight_long_rollout_windows(TRAIN_MANIFEST, physical_steps=len(PHYSICAL_WEIGHTS))
    spec, _, _ = shared.archived_process_spec()
    return {"instance": str(INSTANCE), "train_manifest": str(TRAIN_MANIFEST),
            "val_manifest": str(VAL_MANIFEST), "test_manifest_never_read": str(TEST_MANIFEST),
            "source_sha256": source_sha256(),
            "train_data_identity": dataset_identity(TRAIN_MANIFEST)[0],
            "val_data_identity": dataset_identity(VAL_MANIFEST)[0], "train_windows": windows,
            "model_code_sha256": model_code_digest(), "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
            "arms": {"candidate": {"mode": "long_rollout", "updates": UPDATES, "lr": LR,
                      "warmup": WARMUP, "physical_weights": list(PHYSICAL_WEIGHTS), "steps": STEPS,
                      "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "clip": CLIP,
                      "checkpoint_every": CHECKPOINT_EVERY, "bf16": BF16,
                      "selection": "frozen endpoint; no validation selection",
                      "model": {"kind": "process", "spec": spec,
                                "spec_canonical_digest": canonical_digest(spec)}},
                     "parent": {"updates": 1600, "pins": parent},
                     "narrow_control": narrow_control_pins()},
            "wide_climatology_reference": wide_climatology_reference()}


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**protocol_constants(),
            "gpu_uuid": GPU_UUID, "estimated_peak_bytes": 2**31, "headroom_margin_bytes": 2**31,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body["protocol_sha256"] = canonical_digest(body)
    write_exclusive(output / "preparation_protocol.json", body)
    return body


def _prepared_protocol(output):
    from training.r7_experiment import canonical_digest
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest({k: v for k, v in preliminary.items()
                                                           if k != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    write_exclusive(output / "prepared_protocol.json", body)
    return body


def _check_candidate(body):
    from training.r7_experiment import canonical_digest
    spec, _, _ = recipe._shared().archived_process_spec()
    expected = {"mode": "long_rollout", "updates": UPDATES, "lr": LR, "warmup": WARMUP,
                "physical_weights": list(PHYSICAL_WEIGHTS), "steps": STEPS, "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE, "clip": CLIP, "checkpoint_every": CHECKPOINT_EVERY,
                "bf16": BF16, "selection": "frozen endpoint; no validation selection",
                "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}
    if body["arms"]["candidate"] != expected or body["gpu_uuid"] != GPU_UUID:
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")


def _check_budgets(body):
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS
            or abs(body["deadline_perf_counter"] - body["started_perf_counter"] - HARD_CAP_SECONDS) > 1e-6):
        raise RuntimeError("execution budget differs from frozen deadline")


def _check_reading_fields(body):
    narrow = body["arms"]["narrow_control"]
    if (narrow["record_id"] != NARROW_CONTROL_RECORD or narrow["updates"] != UPDATES
            or narrow["seeds"] != list(SEEDS)):
        raise RuntimeError("the pinned narrow control differs from the registered record")
    if (body["format"] != FORMAT or body["kind"] != "screen" or body["boundary_margins"] != [32]
            or body["region"]["points"] != [129, 129] or body["region"]["frozen_box_points"] != [65, 65]
            or body["design_decision"] != DESIGN_DECISION_TEXT or body["decision_rule"] != DECISION_RULE_TEXT
            or body["test_read_policy"] != TEST_READ_POLICY):
        raise RuntimeError("frozen region/reading fields changed")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")


def validate_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items() if k != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError("execution source changed: " + name)
    for split, manifest in (("train", TRAIN_MANIFEST), ("val", VAL_MANIFEST)):
        if (body[split + "_manifest"] != str(manifest)
                or dataset_identity(manifest)[0] != body[split + "_data_identity"]):
            raise RuntimeError("dataset identity changed")
    if (model_code_digest() != body["model_code_sha256"]
            or training_code_digest() != body["training_code_sha256"]):
        raise RuntimeError("model/training source identity changed")
    if body["source_sha256"] != source_sha256():
        raise RuntimeError("the wide store source identity changed")
    if body["train_windows"] != preflight_long_rollout_windows(TRAIN_MANIFEST, physical_steps=12):
        raise RuntimeError("train-window metadata changed")
    if body["seeds"] != list(SEEDS) or body["evaluation_leads_hours"] != list(EVALUATION_LEADS):
        raise RuntimeError("seed/lead inventory differs")
    _check_candidate(body)
    _check_budgets(body)
    _check_reading_fields(body)
    return body


def _parent_model(body, seed):
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    path, expected = dose.pinned_parent(seed)
    if (sha256_file(path) != expected
            or body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"] != expected):
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != 1600 or contract["seed"] != seed or contract["mode"] != "l6"
            or contract["data_identity"] != shared.V3_DATA_IDENTITY
            or contract["source_sha256"] != shared.V3_SOURCE_SHA256):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"]["candidate"]["model"]["spec"]
    model = make_model("process", spec)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved["model"], path, expected


def _seed(output, body, seed, deadline):
    import torch
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha = _parent_model(body, seed)
    spec = body["arms"]["candidate"]["model"]["spec"]
    contract = {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
                "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
                "training_code_sha256": body["training_code_sha256"],
                "initialization": {"parent_run": str(dose.MIGRATED_PARENT_RUN),
                                   "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                                   "parent_endpoint_updates": 1600,
                                   "description": ("registered v3-BD l6 endpoint, model-only migration; "
                                                   "region-agnostic spec reused on the wide store")},
                "autoregression": {"physical_steps": 12, "physical_weights": list(PHYSICAL_WEIGHTS),
                    "window_sha256": body["train_windows"]["window_sha256"],
                    "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}}
    folder = output / f"seed{seed}/training/candidate"
    checkpoint, report = fine_tune_long_rollout(TRAIN_MANIFEST, folder, model=model, contract=contract,
        parent_weights=state, physical_weights=PHYSICAL_WEIGHTS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline)
    if report["updates_this_run"] != UPDATES or report["data_identity"] != body["train_data_identity"]:
        raise RuntimeError(f"seed {seed} did not reach the frozen wide endpoint on the wide data")
    receipt = {"seed": seed, "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "evaluations": {}}
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h"
        provenance = evaluate_local(VAL_MANIFEST, output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0", reasoning_steps=REASONING_STEPS,
            boundary_margins=BOUNDARY_MARGINS, deadline=deadline)
        _check_provenance(provenance, seed, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    write_exclusive(output / f"seed{seed}_receipt.json", receipt)


def _check_provenance(provenance, seed, lead):
    if provenance["split"] != "val":
        raise RuntimeError(f"seed {seed} lead {lead} was scored on {provenance['split']}, not val")
    if provenance["n_evaluated"] != provenance["n_available_windows"]:
        raise RuntimeError(f"seed {seed} lead {lead} covered {provenance['n_evaluated']} of "
                           f"{provenance['n_available_windows']} val windows")
    if provenance["boundary_scoring"] is None:
        raise RuntimeError(f"seed {seed} lead {lead} has no boundary scoring; interior_32 would be missing")
    if provenance["n_evaluated"] != WIDE_VAL_COHORTS[str(lead)]:
        raise RuntimeError(f"seed {seed} lead {lead} cohort differs from the frozen wide cohort")


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    if phase == "prepare":
        return _prepared_protocol(output)
    if phase == "archive":
        return _archive(output, validate_protocol(output))
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(GPU_UUID)
    if phase == "reading":
        write_exclusive(output / "readings.json", _readings(output, body))
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)


def _archive(output, body):
    import hashlib
    from io import BytesIO
    import zipfile
    from training.r7_arm_harness import sha256_file
    archive = Path(body["code_archive_input"])
    if any(p.is_symlink() for p in (archive, *archive.parents)):
        raise ValueError("linked archive input forbidden")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != body["code_zip_sha256"]:
        raise RuntimeError("prepared code archive SHA mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as zipped:
        if zipped.comment.decode() != body["code_commit"]:
            raise RuntimeError("archive commit differs from freeze")
        for name, expected in body["execution_files_sha256"].items():
            if name.endswith(".py") and hashlib.sha256(zipped.read(name)).hexdigest() != expected:
                raise RuntimeError("archive omits or changes execution source")
    with (output / "code_commit.txt").open("x", encoding="utf-8") as handle:
        handle.write(body["code_commit"] + "\n")
    with (output / "code.zip").open("xb") as handle:
        handle.write(payload)
    write_exclusive(output / "archive_receipt.json", {"scientific_claim": False,
                    "code_commit": body["code_commit"], "code_zip_sha256": sha256_file(output / "code.zip")})


def _readings(output, body):
    """Paired wide-vs-narrow delta and wide interior_32 t2m skill; report only, no pass."""
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    narrow = body["arms"]["narrow_control"]
    climatology = body["wide_climatology_reference"]
    readings = {"scientific_claim": False, "test_read": False, "protocol_sha256": body["protocol_sha256"],
                "decision_rule": body["decision_rule"], "seeds": {}}
    for seed in SEEDS:
        receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {"wide_interior_t2m_rmse": {}, "narrow_full_t2m_rmse": {},
                 "delta_wide_minus_narrow": {}, "wide_interior_t2m_skill": {}}
        for lead in EVALUATION_LEADS:
            wide_dir = Path(receipt["evaluations"][str(lead)]["dir"])
            wide_rmse = _region_variable_rmse(wide_dir / "boundary_rmse.csv", "interior_32", "t2m")
            narrow_lead = narrow["seeds_evaluations"][str(seed)]["leads"][str(lead)]
            narrow_dir = Path(narrow_lead["dir"])
            if sha256_file(narrow_dir / "rmse.csv") != narrow_lead["rmse_csv_sha256"]:
                raise RuntimeError("the registered narrow arm RMSE drifted from its freeze pin")
            narrow_rmse = shared.read_rmse_table(narrow_dir / "rmse.csv")["t2m"]
            climatology_rmse = climatology["leads"][str(lead)]["interior_32_t2m_rmse"]
            entry["wide_interior_t2m_rmse"][str(lead)] = wide_rmse
            entry["narrow_full_t2m_rmse"][str(lead)] = narrow_rmse
            entry["delta_wide_minus_narrow"][str(lead)] = wide_rmse - narrow_rmse
            entry["wide_interior_t2m_skill"][str(lead)] = 1.0 - (wide_rmse / climatology_rmse) ** 2
        readings["seeds"][str(seed)] = entry
    return readings


def run_attempt(output, *, code_archive, code_commit, code_sha256):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = fresh_output(output)
    if not Path(STORE).is_dir():
        raise FileNotFoundError(f"the wide store is not built yet: {STORE}")
    if (not code_archive or not isinstance(code_commit, str) or len(code_commit) != 40
            or not isinstance(code_sha256, str) or len(code_sha256) != 64):
        raise ValueError("prepared exact code archive/commit/SHA required before execution")
    started = time.perf_counter()
    deadline = started + HARD_CAP_SECONDS
    output.mkdir(parents=True, exist_ok=False)
    processes, receipts = [], {}
    try:
        freeze(output, started=started, deadline=deadline, code_archive=Path(code_archive).absolute(),
               code_commit=code_commit, code_sha256=code_sha256)
        phases = [("prepare", None), ("archive", None)] + [("seed", seed) for seed in SEEDS] \
            + [("reading", None)]
        peak, body = 0, None
        for phase, seed in phases:
            gate = recipe._shared().gpu_gate(GPU_UUID, estimated_peak_bytes=2**31,
                                            owned_reserved_peak_bytes=peak, margin_bytes=2**31)
            tag = phase if seed is None else f"seed{seed}"
            write_exclusive(output / f"{tag}_spawn_gate.json", gate)
            limit = min(deadline, time.perf_counter() + PER_SEED_SECONDS) if phase == "seed" else deadline
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                       "--out", str(output), "--deadline", str(limit)]
            if seed is not None:
                command += ["--seed", str(seed)]
            process = bounded_process(command, cwd=ROOT, log_path=output / f"{tag}.log", deadline=limit,
                                      env=worker_environment(GPU_UUID))
            processes.append({"phase": phase, "seed": seed, **process})
            write_exclusive(output / f"{tag}_process.json", processes[-1])
            if process["status"] != "success":
                raise RuntimeError(f"{tag} failed: {process['status']}")
            if phase == "prepare":
                body = json.loads((output / "prepared_protocol.json").read_text(encoding="utf-8"))
                write_exclusive(output / "protocol.json", body)
            elif phase == "seed":
                receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
                receipts[str(seed)] = receipt
                peak = max(peak, receipt["owned_cuda_reserved_peak_bytes"])
            print(json.dumps({"phase_done": phase, "seed": seed,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
        elapsed = time.perf_counter() - started
        if elapsed >= deadline:
            raise RuntimeError("round hard deadline passed before publication")
        result = {"scientific_claim": False, "test_read": False, "protocol_sha256": body["protocol_sha256"],
                  "train_data_identity": body["train_data_identity"],
                  "val_data_identity": body["val_data_identity"], "elapsed_seconds_total": elapsed,
                  "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS),
                  "hard_overrun_seconds": max(0., elapsed - HARD_CAP_SECONDS), "processes": processes,
                  "seeds": receipts, "limitations": body["limitations"]}
        result.update(json.loads((output / "readings.json").read_text(encoding="utf-8")))
        write_exclusive(output / "result.json", result)
        write_exclusive(output / "attempt.json", {"status": "complete", "scientific_claim": False,
            "test_read": False, "protocol_sha256": body["protocol_sha256"], "elapsed_seconds_total": elapsed,
            "decision_rule": body["decision_rule"]})
        return result
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        write_exclusive(output / "failure.json", {"status": "failed", "scientific_claim": False,
            "test_read": False, "elapsed_seconds_total": elapsed,
            "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS),
            "hard_overrun_seconds": max(0., elapsed - HARD_CAP_SECONDS), "processes": processes,
            "seeds_completed": list(receipts), "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "limitations": ["Failed attempt preserved and charged in full; no automatic retry or in-place resume"]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--code-archive", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--code-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.out, args.worker, args.seed, args.deadline)
    return run_attempt(args.out or DEFAULT_OUT, code_archive=args.code_archive,
                       code_commit=args.code_commit, code_sha256=args.code_sha256)


if __name__ == "__main__":
    main()
