"""S3 EMA weight averaging: score plain EMA weights instead of the update-2400 weights.

The registered wide/interior_32 recipe trains for 2400 updates and scores the final
optimizer weights. This driver keeps every input to that trajectory word for word and
changes only the weight source of the scored checkpoint: the candidate scores a plain
parameter EMA (``decay=0.999``, initialized from the migrated v3-BD 1600 parent,
every trainable parameter, updated after every optimizer step, no bias correction, no
warm-up). The final update-2400 optimizer checkpoint is still written but is not
scored. The paired control is the pinned ``record:s3-anchor-isolation-result`` control
arm (same recipe, final-weight scoring); the external reference is the pinned
``record:s3-wide-interior-dose``. Neither is retrained, the 2023 test split is never
read, and ``scientific_claim`` stays false. Only training code changes, so
``model_code_sha256`` is unchanged and no new parent migration is needed.
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
from scripts import r7_s3_anchor_isolation_parent as anchor_parent
from scripts import study_r7_s3_reasoning_depth as depth_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from training import r7_device_selection as devices
from training.r7_ema import require_ema_decay
interior = dose.interior
FORMAT = "r7-s3-ema-weight-averaging-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
CANDIDATE, SEEDS = "candidate", wide.SEEDS
DEVICE_MODE = "dual"
# One arm per seed: the same single-arm envelope as the spatial-gradient round.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 21600., 43200., 86400.
ESTIMATED_PEAK_BYTES = 9728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
PAIRED_CONTROL_RECORD = "s3-anchor-isolation-result"
PAIRED_CONTROL_RUN = ROOT / "outputs/r7_s3_anchor_isolation_20261015_attempt01"
#: The registered sparse contour, verbatim from study_r7_s3_wide_single_factor.py:38.
REGISTERED_PHYSICAL_WEIGHTS = tuple(wide.PHYSICAL_WEIGHTS)
N_PHYSICAL_STEPS = len(REGISTERED_PHYSICAL_WEIGHTS)
DRAFT_SUPERVISION = "deep"
EMA_DECAY = 0.999
EMA_INITIALIZATION = "parent_checkpoint"
EMA_SCOPE = "all trainable parameters"
EMA_BIAS_CORRECTION = False
EMA_WARMUP_UPDATES = 0
#: Anchor-round reported N - A spread; this is an identity band, not a new threshold.
BRIDGE_MIN, BRIDGE_MAX = -4.8e-6, 1.8e-5
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "scripts/r7_s3_anchor_isolation_parent.py",
               "scripts/study_r7_s3_ema_weight_averaging.py",
               "scripts/r7_s3_ema_weight_averaging_text.py", "training/r7_ema.py")
from scripts.r7_s3_ema_weight_averaging_text import (
    DECISION_RULE_TEXT, DESIGN_DECISION_TEXT, DIFFERENCE_TEXT, HYPOTHESIS_TEXT, LIMITATIONS,
)
def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))
def _deep_weights(steps=STEPS):
    """The registered linspace(1,2,K+1) normalized weights, in float64 arithmetic."""
    values = [1.0 + (2.0 - 1.0) * index / steps for index in range(steps + 1)]
    return tuple(value / sum(values) for value in values)
REGISTERED_DEEP_WEIGHTS = _deep_weights()
def physical_weight_block():
    """The frozen physical axis, asserted unchanged at exactly the registered contour."""
    weights = tuple(REGISTERED_PHYSICAL_WEIGHTS)
    if len(weights) != 12 or sum(weights) != 3.0:
        raise RuntimeError("the physical-weight axis must stay at the registered 12-step contour summing to 3.0")
    direct = [index for index, weight in enumerate(weights) if weight > 0]
    if direct != [0, 1, 3, 7, 11]:
        raise RuntimeError("the registered direct-supervision steps changed")
    return {"weights": list(weights), "sum": sum(weights), "direct_steps": direct}
def draft_organization_block():
    """The K-axis organization, asserted unchanged on both sides."""
    deep = tuple(REGISTERED_DEEP_WEIGHTS)
    if len(deep) != STEPS + 1 or abs(sum(deep) - 1.0) > 1e-12:
        raise RuntimeError("the registered deep weights must have K+1 slots summing to one")
    if abs(sum(deep[:-1]) - 11.0 / 15.0) > 1e-12:
        raise RuntimeError("the registered non-final draft mass must be 11/15 = 73.333%")
    return {"slots": STEPS + 1, "draft_supervision": DRAFT_SUPERVISION,
            "weights_linspace_1_2_normalized": list(deep), "nonfinal_mass": sum(deep[:-1]),
            "final_weight": 2.0, "final_weight_normalized": deep[-1],
            "change": "unchanged; K-axis organization is not this round's factor"}
def ema_block():
    """The frozen EMA rule, asserted at exactly the predeclared constants."""
    decay = require_ema_decay(EMA_DECAY)
    if (decay != 0.999 or EMA_BIAS_CORRECTION is not False or EMA_WARMUP_UPDATES != 0
            or EMA_INITIALIZATION != "parent_checkpoint" or EMA_SCOPE != "all trainable parameters"):
        raise RuntimeError("the frozen EMA rule changed")
    return {"decay": decay, "initialization": EMA_INITIALIZATION, "scope": EMA_SCOPE,
            "update": "after every optimizer step", "bias_correction": EMA_BIAS_CORRECTION,
            "warmup_updates": EMA_WARMUP_UPDATES,
            "scored_checkpoint": "EMA weights replace the update-2400 optimizer weights for scoring",
            "final_optimizer_weights": "written to the ordinary checkpoint but not scored",
            "sweep": "none; decay is a frozen design coefficient, not a pass/close threshold"}
def arm_spec():
    """The archived process spec, unchanged: this round touches no model switch."""
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError(f"the archived process spec is not dim {DIM} depth {DEPTH}")
    if (spec.get("climatology_anchor_spec") or spec.get("anomaly_feedback")
            or spec.get("anchor_output_prior") or spec.get("process_init_conditioning")
            or spec.get("typed_evidence_mode") or spec.get("global_mixing_mode")):
        raise RuntimeError("the archived process spec already carries an architecture switch")
    return spec
def protocol_constants():
    """The dose round's frozen fields with the EMA weight source added."""
    body = depth_arm.protocol_constants()
    body["format"] = FORMAT
    body["recipe"].update(updates=UPDATES, steps=STEPS,
                          physical_weights=list(REGISTERED_PHYSICAL_WEIGHTS))
    body["evaluation"]["reasoning_steps"] = REASONING_STEPS
    body["candidate_draft_supervision"] = DRAFT_SUPERVISION
    body["control_draft_supervision"] = DRAFT_SUPERVISION
    body["ema_weight_averaging"] = ema_block()
    body["draft_supervision_organization"] = draft_organization_block()
    body["physical_weight_vector"] = physical_weight_block()
    body["budgets"] = {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                       "per_seed_seconds": PER_SEED_SECONDS}
    body.update(design_decision=DESIGN_DECISION_TEXT, decision_rule=DECISION_RULE_TEXT,
                hypothesis=HYPOTHESIS_TEXT, difference=DIFFERENCE_TEXT, limitations=list(LIMITATIONS))
    return body
def _index_record(record_id):
    return dose._index_record(record_id)
def previous_arm_pins():
    """Pin the registered 2400-update interior-dose arm used as the external reference."""
    return depth_arm.previous_arm_pins()
def paired_control_pins(expected_identity=None, expected_source=None):
    """Pin the H-D control arm N and verify it was scored on final optimizer weights."""
    from training.r7_arm_harness import sha256_file
    record = _index_record(PAIRED_CONTROL_RECORD)
    if sha256_file(ROOT / record["evidence_path"]) != record["evidence_sha256"]:
        raise RuntimeError("the H-D result doc drifted from its index record")
    if (expected_identity is not None and record["data_identity"] != expected_identity) or (expected_source is not None and record["source_sha256"] != expected_source):
        raise RuntimeError("the H-D control record data/source identity differs from this run")
    readings_path = PAIRED_CONTROL_RUN / "readings.json"
    readings = json.loads(readings_path.read_text(encoding="utf-8"))
    if (readings["scientific_claim"] is not False or readings["test_read"] is not False
            or readings["protocol_sha256"] != record["protocol_sha256"]):
        raise RuntimeError("the H-D control readings differ from the registered protocol")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "updates": UPDATES, "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
            "draft_supervision": DRAFT_SUPERVISION, "ema_decay": None,
            "scored_checkpoint_kind": "final_optimizer_weights", "seeds": list(SEEDS),
            "run_dir": str(PAIRED_CONTROL_RUN), "readings_sha256": sha256_file(readings_path),
            "seeds_readings": {}}
    for seed in SEEDS:
        receipt = json.loads((PAIRED_CONTROL_RUN / f"seed{seed}_control_receipt.json").read_text(encoding="utf-8"))
        report = json.loads((PAIRED_CONTROL_RUN / f"seed{seed}/training/control/training_report.json").read_text(encoding="utf-8"))
        prior = readings["seeds"][str(seed)]["arms"]["control"]
        if (receipt["arm"] != "control" or receipt["anomaly_feedback"] is not False
                or int(receipt["updates"]) != UPDATES or int(receipt["reasoning_steps"]) != REASONING_STEPS
                or receipt["parent_checkpoint_sha256"] != anchor_parent.MIGRATED_PARENT_SHA256[seed]
                or "ema_decay" in receipt or "ema_checkpoint" in receipt
                or Path(receipt["checkpoint"]).name != f"update_{UPDATES:07d}.pt"):
            raise RuntimeError(f"the H-D seed {seed} control receipt is not the final-weight arm")
        recorded_lambda = report["contract"]["autoregression"].get("spatial_gradient_lambda")
        if (report["objective"] != "deep_supervised_latitude_area_mse"
                or report["internal_deep_supervision"] is not True
                or int(report["contract"]["steps"]) != REASONING_STEPS
                or report["contract"]["autoregression"]["physical_weights"]
                != list(REGISTERED_PHYSICAL_WEIGHTS)
                or recorded_lambda is not None
                or "ema_decay" in report or "ema_checkpoint" in report):
            raise RuntimeError(f"the H-D seed {seed} control report is not the registered deep objective")
        entry = {"checkpoint_sha256": receipt["checkpoint_sha256"], "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"], "leads": {}}
        for lead in wide.EVALUATION_LEADS:
            lead_record = receipt["evaluations"][str(lead)]
            folder = Path(lead_record["dir"])
            if sha256_file(folder / "boundary_rmse.csv") != lead_record["boundary_rmse_csv_sha256"]:
                raise RuntimeError(f"the H-D control table drifted for seed {seed} lead {lead}")
            if int(lead_record["n_evaluated"]) != wide.WIDE_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"the H-D control cohort differs for seed {seed} lead {lead}")
            entry["leads"][str(lead)] = {
                "dir": str(folder), "boundary_rmse_csv_sha256": lead_record["boundary_rmse_csv_sha256"],
                "n_evaluated": lead_record["n_evaluated"],
                "interior_t2m_rmse": prior["interior_t2m_rmse"][str(lead)],
                "interior_t2m_skill": prior["interior_t2m_skill"][str(lead)],
                "delta_minus_previous_dose": prior["delta_minus_previous_dose"][str(lead)]}
        pins["seeds_readings"][str(seed)] = entry
    return pins
def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = interior._identity()
    spec = arm_spec()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    body["arms"] = {"candidate": {
        "mode": "long_rollout", "updates": UPDATES, "lr": wide.LR, "warmup": wide.WARMUP,
        "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS), "physical_steps": N_PHYSICAL_STEPS,
        "draft_supervision": DRAFT_SUPERVISION, "spatial_gradient_lambda": 0.0,
        "ema_decay": EMA_DECAY, "ema_initialization": EMA_INITIALIZATION,
        "ema_scope": EMA_SCOPE, "ema_bias_correction": EMA_BIAS_CORRECTION,
        "ema_warmup_updates": EMA_WARMUP_UPDATES, "scored_checkpoint_kind": "ema",
        "steps": STEPS, "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE,
        "clip": wide.CLIP, "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
        "selection": "frozen endpoint; the scored checkpoint is the EMA state",
        "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec), "anchor_keys_absent": ["climatology_anchor_spec", "anchor_output_prior", "anomaly_feedback"]}},
        "parent": {"updates": 1600, "pins": {}}}
    body["arms"]["previous_dose_arm"] = previous_arm_pins()
    body["arms"]["paired_control_arm"] = paired_control_pins(body["train_data_identity"], body["source_sha256"])
    anchor_parent.install_parent_pins(body)
    body["ema_weight_averaging"] = ema_block()
    body["draft_supervision_organization"] = draft_organization_block()
    return body
def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**protocol_constants(),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(DEVICE_MODE, single_uuid=wide.GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "preparation_protocol.json", body)
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
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body
def validate_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest({k: v for k, v in body.items()
                                                    if k != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError("execution source changed: " + name)
    for split, manifest in (("train", wide.TRAIN_MANIFEST), ("val", wide.VAL_MANIFEST)):
        if (body[split + "_manifest"] != str(manifest)
                or dataset_identity(manifest)[0] != body[split + "_data_identity"]):
            raise RuntimeError("dataset identity changed")
    if (model_code_digest() != body["model_code_sha256"]
            or training_code_digest() != body["training_code_sha256"]):
        raise RuntimeError("model/training source identity changed")
    if body["source_sha256"] != wide.source_sha256():
        raise RuntimeError("the wide store source identity changed")
    if body["train_windows"] != preflight_long_rollout_windows(
            wide.TRAIN_MANIFEST, physical_steps=N_PHYSICAL_STEPS):
        raise RuntimeError("train-window metadata changed")
    if (body["supervision"] != interior.supervision_block() or body["format"] != FORMAT
            or body["ema_weight_averaging"] != ema_block()
            or body["draft_supervision_organization"] != draft_organization_block()
            or body["candidate_draft_supervision"] != DRAFT_SUPERVISION
            or body["control_draft_supervision"] != DRAFT_SUPERVISION
            or body["physical_weight_vector"] != physical_weight_block()):
        raise RuntimeError("the frozen supervision/EMA factor changed")
    if (body["recipe"]["updates"] != UPDATES or body["recipe"]["steps"] != STEPS
            or body["evaluation"]["reasoning_steps"] != REASONING_STEPS
            or body["recipe"]["physical_weights"] != list(REGISTERED_PHYSICAL_WEIGHTS)):
        raise RuntimeError("the frozen recipe/physical-weight vector changed")
    if not devices.assignment_is_valid(body, wide.GPU_UUID):
        raise RuntimeError("declared GPU differs from the frozen assignment")
    arm, spec = body["arms"][CANDIDATE], body["arms"][CANDIDATE]["model"]["spec"]
    if (arm["physical_weights"] != list(REGISTERED_PHYSICAL_WEIGHTS)
            or arm["physical_steps"] != N_PHYSICAL_STEPS or int(arm["updates"]) != UPDATES
            or int(arm["steps"]) != STEPS or arm["draft_supervision"] != DRAFT_SUPERVISION
            or arm["ema_decay"] != EMA_DECAY or arm["ema_bias_correction"] is not False
            or arm["ema_warmup_updates"] != EMA_WARMUP_UPDATES
            or arm["ema_initialization"] != EMA_INITIALIZATION
            or arm["scored_checkpoint_kind"] != "ema"
            or int(spec.get("dim", 0)) != DIM
            or int(spec.get("depth", 0)) != DEPTH or spec != arm_spec()
            or arm["model"]["spec_canonical_digest"] != canonical_digest(spec)):
        raise RuntimeError("the candidate arm differs from the frozen recipe")
    if (body["arms"]["previous_dose_arm"]["record_id"] != PREVIOUS_DOSE_ARM_RECORD
            or body["arms"]["previous_dose_arm"]["updates"] != UPDATES
            or body["arms"]["previous_dose_arm"]["seeds"] != list(SEEDS)):
        raise RuntimeError("the pinned dose arm differs from the registered record")
    paired = body["arms"]["paired_control_arm"]
    if (paired["record_id"] != PAIRED_CONTROL_RECORD or paired["updates"] != UPDATES
            or paired["physical_weights"] != list(REGISTERED_PHYSICAL_WEIGHTS)
            or paired["draft_supervision"] != DRAFT_SUPERVISION
            or paired["ema_decay"] is not None
            or paired["scored_checkpoint_kind"] != "final_optimizer_weights"
            or paired["seeds"] != list(SEEDS)
            or paired["protocol_sha256"] != _index_record(PAIRED_CONTROL_RECORD)["protocol_sha256"]
            or paired["data_identity"] != body["train_data_identity"]
            or paired["source_sha256"] != body["source_sha256"]):
        raise RuntimeError("the pinned paired control differs from the registered record")
    for seed in SEEDS:
        if (body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"]
                != anchor_parent.MIGRATED_PARENT_SHA256[seed]):
            raise RuntimeError(f"the migrated parent pin changed for seed {seed}")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    if (int(body["estimated_peak_bytes"]) < MIN_ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("the peak estimate was not recalibrated")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(wide.TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")
    return body
def _parent_model(body, seed):
    import torch
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    path, expected = anchor_parent.pinned_parent(seed)
    if (sha256_file(path) != expected
            or body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"] != expected):
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != 1600 or contract["seed"] != seed or contract["mode"] != "l6"
            or contract["data_identity"] != recipe._shared().V3_DATA_IDENTITY
            or contract["source_sha256"] != recipe._shared().V3_SOURCE_SHA256):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"][CANDIDATE]["model"]["spec"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = make_model("process", spec)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved["model"], path, expected
def _arm_contract(body, spec, parent, parent_sha):
    return {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(anchor_parent.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": 1600,
                               "description": ("registered v3-BD l6 endpoint, model-only migration, "
                                               "then 2400 registered wide/interior_32 updates")},
            "autoregression": {"physical_steps": N_PHYSICAL_STEPS,
                "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
                "draft_supervision": DRAFT_SUPERVISION,
                "ema_decay": EMA_DECAY, "ema_initialization": EMA_INITIALIZATION,
                "ema_scope": EMA_SCOPE, "ema_bias_correction": EMA_BIAS_CORRECTION,
                "ema_warmup_updates": EMA_WARMUP_UPDATES,
                "window_sha256": body["train_windows"]["window_sha256"],
                "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                "supervision": body["supervision"]["runner_block"]}}
def _train_arm(output, body, seed, deadline, mask):
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    spec = body["arms"][CANDIDATE]["model"]["spec"]
    model, state, parent, parent_sha = _parent_model(body, seed)
    folder = output / f"seed{seed}/training/{CANDIDATE}"
    final_checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model,
        contract=_arm_contract(body, spec, parent, parent_sha), parent_weights=state,
        physical_weights=REGISTERED_PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=wide.LR,
        warmup=wide.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        supervision_mask=mask, draft_supervision=DRAFT_SUPERVISION,
        gradient_lambda=0.0, ema_decay=EMA_DECAY)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]
            or report["objective"] != "deep_supervised_latitude_area_mse"
            or report["internal_deep_supervision"] is not True
            or report["ema_decay"] != EMA_DECAY
            or report["ema_initialization"] != EMA_INITIALIZATION
            or report["ema_bias_correction"] is not False
            or report["ema_warmup_updates"] != EMA_WARMUP_UPDATES
            or report["ema_updates_applied"] != UPDATES
            or report["selected_checkpoint_kind"] != "ema"
            or report["selected_checkpoint"] != report["ema_checkpoint"]
            or report["contract"]["autoregression"]["physical_weights"]
            != list(REGISTERED_PHYSICAL_WEIGHTS)
            or report["contract"]["autoregression"]["draft_supervision"] != DRAFT_SUPERVISION
            or report["contract"]["autoregression"].get("spatial_gradient_lambda") is not None):
        raise RuntimeError(f"seed {seed} did not reach the frozen EMA endpoint")
    ema_checkpoint = report["ema_checkpoint"]
    if Path(ema_checkpoint).name != f"ema_update_{UPDATES:07d}.pt" or not Path(ema_checkpoint).is_file():
        raise RuntimeError("the scored checkpoint is not the frozen EMA checkpoint")
    receipt = {"seed": seed, "arm": CANDIDATE, "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
               "scientific_claim": False, "limitations": LIMITATIONS,
               "physical_steps": N_PHYSICAL_STEPS, "draft_supervision": DRAFT_SUPERVISION,
               "ema_decay": EMA_DECAY, "ema_initialization": EMA_INITIALIZATION,
               "ema_scope": EMA_SCOPE, "ema_bias_correction": EMA_BIAS_CORRECTION,
               "ema_warmup_updates": EMA_WARMUP_UPDATES, "ema_updates_applied": UPDATES,
               "selected_checkpoint_kind": "ema", "checkpoint": ema_checkpoint,
               "checkpoint_sha256": sha256_file(ema_checkpoint),
               "ema_checkpoint": ema_checkpoint,
               "final_optimizer_checkpoint": report["final_optimizer_checkpoint"],
               "final_optimizer_checkpoint_sha256": sha256_file(report["final_optimizer_checkpoint"]),
               "objective": report["objective"], "internal_deep_supervision": report["internal_deep_supervision"],
               "training_dir": str(folder), "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "reasoning_steps": STEPS,
               "dim": DIM, "evaluations": {}}
    for lead in wide.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{CANDIDATE}/lead_{lead:03d}h"
        provenance = evaluate_local(wide.VAL_MANIFEST, output_dir=destination, checkpoint=ema_checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=wide.BOUNDARY_MARGINS, deadline=deadline)
        wide._check_provenance(provenance, seed, lead)
        if provenance.get("checkpoint_sha256") != receipt["checkpoint_sha256"]:
            raise RuntimeError("the evaluation did not score the frozen EMA checkpoint")
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "rmse_csv_sha256": sha256_file(destination / "rmse.csv"),
            "boundary_rmse_csv_sha256": sha256_file(destination / "boundary_rmse.csv"),
            "skill_csv_sha256": sha256_file(destination / "climatology_skill.csv"),
            "provenance_sha256": sha256_file(destination / "provenance.json")}
    return receipt
def _seed(output, body, seed, deadline):
    import torch
    receipt = _train_arm(output, body, seed, deadline, interior.supervision_mask().to("cuda:0"))
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    wide.write_exclusive(output / f"seed{seed}_receipt.json", receipt)
def _all_variable_rmse(path):
    """Every variable's interior_32 RMSE, for the required 17-variable report."""
    import csv
    values = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["region"] != "interior_32":
                continue
            name = row["variable"]
            if name in values:
                raise RuntimeError(f"duplicate interior_32 row for {name}")
            value = float(row["rmse"])
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite interior_32 RMSE for {name}")
            values[name] = value
    if len(values) != 17:
        raise RuntimeError(f"expected 17 interior_32 variables, found {len(values)}")
    return values
def _table_readings(folder, climatology):
    """RMSE, skill and the full 17-variable table for one pinned/new evaluation folder."""
    folder = Path(folder)
    rmse = wide._region_variable_rmse(folder / "boundary_rmse.csv", "interior_32", "t2m")
    return {"interior_t2m_rmse": rmse,
            "interior_t2m_skill": 1.0 - (rmse / climatology) ** 2,
            "all_variable_interior_rmse": _all_variable_rmse(folder / "boundary_rmse.csv")}
def _dose_readings(body, seed):
    from training.r7_arm_harness import sha256_file
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"interior_t2m_rmse": {}, "interior_t2m_skill": {}}
    for lead in wide.EVALUATION_LEADS:
        previous = body["arms"]["previous_dose_arm"]["seeds_evaluations"][str(seed)]["leads"][str(lead)]
        folder = Path(previous["dir"])
        if sha256_file(folder / "boundary_rmse.csv") != previous["boundary_rmse_csv_sha256"]:
            raise RuntimeError("the registered 2400-update dose arm table drifted from its pin")
        row = _table_readings(folder, climatology[str(lead)]["interior_32_t2m_rmse"])
        arm["interior_t2m_rmse"][str(lead)] = row["interior_t2m_rmse"]
        arm["interior_t2m_skill"][str(lead)] = row["interior_t2m_skill"]
    return arm
def _control_readings(body, seed):
    from training.r7_arm_harness import sha256_file
    pinned = body["arms"]["paired_control_arm"]["seeds_readings"][str(seed)]
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"interior_t2m_rmse": {}, "interior_t2m_skill": {},
           "delta_minus_previous_dose": {}, "all_variable_interior_rmse": {},
           "scored_checkpoint_kind": body["arms"]["paired_control_arm"]["scored_checkpoint_kind"]}
    for lead in wide.EVALUATION_LEADS:
        entry, folder = pinned["leads"][str(lead)], Path(pinned["leads"][str(lead)]["dir"])
        if sha256_file(folder / "boundary_rmse.csv") != entry["boundary_rmse_csv_sha256"]:
            raise RuntimeError("the pinned H-D control table drifted from its freeze pin")
        row = _table_readings(folder, climatology[str(lead)]["interior_32_t2m_rmse"])
        if row["interior_t2m_rmse"] != entry["interior_t2m_rmse"]:
            raise RuntimeError("the recomputed H-D control RMSE differs from its readings.json")
        arm["interior_t2m_rmse"][str(lead)] = row["interior_t2m_rmse"]
        arm["interior_t2m_skill"][str(lead)] = row["interior_t2m_skill"]
        arm["delta_minus_previous_dose"][str(lead)] = entry["delta_minus_previous_dose"]
        arm["all_variable_interior_rmse"][str(lead)] = row["all_variable_interior_rmse"]
    return arm
def _candidate_readings(output, body, seed):
    from training.r7_arm_harness import sha256_file
    receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    if (receipt["seed"] != seed or receipt["physical_weights"] != list(REGISTERED_PHYSICAL_WEIGHTS)
            or receipt["draft_supervision"] != DRAFT_SUPERVISION
            or receipt.get("ema_decay") != EMA_DECAY
            or receipt.get("ema_initialization") != EMA_INITIALIZATION
            or receipt.get("ema_bias_correction") is not False
            or receipt.get("ema_warmup_updates") != EMA_WARMUP_UPDATES
            or receipt.get("ema_updates_applied") != UPDATES
            or receipt.get("selected_checkpoint_kind") != "ema"
            or receipt.get("checkpoint") != receipt.get("ema_checkpoint")
            or Path(receipt.get("checkpoint", "")).name != f"ema_update_{UPDATES:07d}.pt"):
        raise RuntimeError("the candidate receipt is not the frozen EMA arm")
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"physical_weights": receipt["physical_weights"],
           "draft_supervision": receipt["draft_supervision"],
           "ema_decay": receipt["ema_decay"], "ema_initialization": receipt["ema_initialization"],
           "scored_checkpoint_kind": receipt["selected_checkpoint_kind"],
           "interior_t2m_rmse": {}, "interior_t2m_skill": {}, "all_variable_interior_rmse": {}}
    for lead in wide.EVALUATION_LEADS:
        entry = receipt["evaluations"][str(lead)]; folder = Path(entry["dir"])
        if sha256_file(folder / "boundary_rmse.csv") != entry["boundary_rmse_csv_sha256"]:
            raise RuntimeError("the candidate table drifted from its receipt")
        if entry["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
            raise RuntimeError("the candidate evaluation did not score the EMA checkpoint")
        row = _table_readings(folder, climatology[str(lead)]["interior_32_t2m_rmse"])
        arm["interior_t2m_rmse"][str(lead)] = row["interior_t2m_rmse"]
        arm["interior_t2m_skill"][str(lead)] = row["interior_t2m_skill"]
        arm["all_variable_interior_rmse"][str(lead)] = row["all_variable_interior_rmse"]
    return arm
def decision_flags(seeds):
    """The predeclared exit, computed from the reported readings only."""
    def values(key, lead):
        return [seeds[str(seed)][key][str(lead)] for seed in SEEDS]
    def deltas(lead):
        return values("delta_candidate_minus_control", lead)
    a = all(value > 0 for value in values("candidate_skill", 72))
    b = all(value < 0 for value in deltas(72))
    c = all(value > 0 for value in values("candidate_skill", 48))
    d = all(sum(deltas(lead)) / len(SEEDS) <= 0 for lead in (6, 12, 24))
    bridge = [value for seed in SEEDS
              for value in seeds[str(seed)]["control_minus_previous_dose"].values()]
    if len(bridge) != len(SEEDS) * len(wide.EVALUATION_LEADS):
        raise RuntimeError("the bridge must cover every seed x lead")
    e = bool(bridge) and all(BRIDGE_MIN <= value <= BRIDGE_MAX for value in bridge)
    scientific = a and b and c and d
    return {"a_72h_skill_all_positive": a, "b_72h_delta_negative_all_seeds": b,
            "c_48h_skill_all_positive": c, "d_short_lead_seed_mean_not_worse": d,
            "e_bridge_consistent": e, "scientific_exit_met": scientific,
            "predeclared_exit_met": scientific and e, "identity_unresolved": not e}
def _readings(output, body):
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
                "updates": UPDATES, "dim": DIM, "reasoning_steps": REASONING_STEPS,
                "factor": "scored checkpoint weights: final optimizer weights -> plain EMA decay=0.999",
                "ema_weight_averaging": body["ema_weight_averaging"],
                "draft_supervision_organization": body["draft_supervision_organization"],
                "declared_difference": body["difference"], "seeds": {}}
    for seed in SEEDS:
        candidate, control = _candidate_readings(output, body, seed), _control_readings(body, seed)
        registered = _dose_readings(body, seed)
        readings["seeds"][str(seed)] = {
            "seed": seed, "candidate": candidate, "paired_control": control,
            "registered_dose": registered, "candidate_skill": candidate["interior_t2m_skill"],
            "delta_candidate_minus_control": {
                str(lead): candidate["interior_t2m_rmse"][str(lead)]
                - control["interior_t2m_rmse"][str(lead)] for lead in wide.EVALUATION_LEADS},
            "delta_candidate_minus_registered_dose": {
                str(lead): candidate["interior_t2m_rmse"][str(lead)]
                - registered["interior_t2m_rmse"][str(lead)] for lead in wide.EVALUATION_LEADS},
            "control_minus_previous_dose": {
                str(lead): control["interior_t2m_rmse"][str(lead)]
                - registered["interior_t2m_rmse"][str(lead)] for lead in wide.EVALUATION_LEADS}}
    readings["decision"] = decision_flags(readings["seeds"])
    return readings
def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    if phase == "prepare":
        return _prepared_protocol(output)
    if phase == "archive":
        return interior._archive(output, validate_protocol(output))
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside frozen round")
    recipe._shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", _readings(output, body))
        return
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    _seed(output, body, seed, deadline)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "archive", "seed", "reading"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise SystemExit("dual-card rounds are driven by `r7_round.py stage-seeds`; "
                         "invoke this driver with --worker <prepare|archive|seed|reading>")
    return worker(args.out, args.worker, args.seed, args.deadline)
if __name__ == "__main__":
    main()
