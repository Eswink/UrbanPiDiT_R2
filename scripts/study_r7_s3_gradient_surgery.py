"""S3 gradient surgery: condition the update on short/long gradient conflict.

The registered wide/interior_32 recipe trains for 2400 updates on a single scalar: the
weighted sum of the twelve physical-step losses.  This driver keeps the model, spec,
data, parent, seeds, loss definition, physical weights, 2400 updates, LR 2e-5, warmup
10, K=4, depth 4, dim 192, batch 1, clip 1, scoring domain, climatology denominator and
test policy word for word, and changes only the training-time backward:
``gradient_surgery`` goes from ``off`` (registered ``loss.backward()``) to ``pcgrad``.

Group A is weighted steps 0/1/3 (6/12/24 h, weights 1.0/0.5/0.5); group B is weighted
steps 7/11 (48/72 h, weights 0.5/0.5); both come from the same ``per_step_losses``
tensor.  Two ``autograd.grad`` calls over all trainable parameters produce g_A and g_B;
reductions run in float64; if ``dot(g_A,g_B) < 0`` then
``g_A <- g_A - (dot/||g_B||^2) g_B``; the update is ``g_A + g_B``, followed by the
existing clip and optimizer step.  The order is fixed A -> B.  No sweep, no third group,
no loss-weight change and no combination with a closed axis.

The paired control is the pinned ``record:s3-anchor-isolation-result`` control arm
(registered gradient, final optimizer weights, not retrained).  The external reference
is the pinned ``record:s3-wide-interior-dose``.  The decoder-round parent migration is
reused because the model code is unchanged.  The 2023 test split is never read and
``scientific_claim`` stays false.
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

from scripts import r7_s3_decoder_base_anchor_parent as parent
from scripts import r7_s3_gradient_surgery_support as report
from scripts import study_r7_s3_anchor_isolation as anchor_arm
from scripts import study_r7_s3_reasoning_depth as depth_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from training import r7_device_selection as devices
from training.r7_gradient_surgery import (
    GROUP_A_STEPS, GROUP_A_WEIGHTS, GROUP_B_STEPS, GROUP_B_WEIGHTS, summarize_updates,
)
from scripts.r7_s3_gradient_surgery_text import (
    DECISION_RULE_TEXT, DESIGN_DECISION_TEXT, DIFFERENCE_TEXT, HYPOTHESIS_TEXT, LIMITATIONS,
)
interior = dose.interior
FORMAT = "r7-s3-gradient-surgery-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
CANDIDATE, SEEDS = "candidate", wide.SEEDS
DEVICE_MODE = "dual"
# One arm per seed: the same single-arm envelope as the decoder/EMA rounds.
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
CANDIDATE_GRADIENT_SURGERY = "pcgrad"
CONTROL_GRADIENT_SURGERY = "off"
#: The pinned control protocol's recorded training revision (not retrained this round).
CONTROL_TRAINING_CODE_SHA256 = "8ad77d9349bdc4930e6bbba05f0c417814fe3093fe39aa921e816af197494974"
#: Anchor-round reported N - A spread; this is an identity band, not a new threshold.
BRIDGE_MIN, BRIDGE_MAX = -4.8e-6, 1.8e-5
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "scripts/r7_s3_anchor_isolation_parent.py",
               "scripts/study_r7_s3_anchor_isolation.py",
               "scripts/r7_s3_decoder_base_anchor_parent.py",
               "scripts/r7_s3_decoder_base_anchor_support.py",
               "scripts/r7_s3_gradient_surgery_support.py",
               "scripts/r7_s3_gradient_surgery_text.py",
               "scripts/study_r7_s3_gradient_surgery.py",
               "training/r7_gradient_surgery.py")


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


def gradient_surgery_block():
    """The frozen two-group surgery rule, asserted at exactly the predeclared constants."""
    if (CANDIDATE_GRADIENT_SURGERY != "pcgrad" or CONTROL_GRADIENT_SURGERY != "off"):
        raise RuntimeError("the frozen candidate/control gradient-surgery modes changed")
    weights = tuple(REGISTERED_PHYSICAL_WEIGHTS)
    if tuple(weights[index] for index in GROUP_A_STEPS) != GROUP_A_WEIGHTS:
        raise RuntimeError("group A does not carry the registered weights")
    if tuple(weights[index] for index in GROUP_B_STEPS) != GROUP_B_WEIGHTS:
        raise RuntimeError("group B does not carry the registered weights")
    if [index for index, weight in enumerate(weights) if weight > 0] != [0, 1, 3, 7, 11]:
        raise RuntimeError("the registered direct-supervision steps changed")
    return {
        "candidate_mode": CANDIDATE_GRADIENT_SURGERY, "control_mode": CONTROL_GRADIENT_SURGERY,
        "groups": {"A": {"steps": list(GROUP_A_STEPS), "leads_hours": [6, 12, 24],
                         "weights": list(GROUP_A_WEIGHTS)},
                   "B": {"steps": list(GROUP_B_STEPS), "leads_hours": [48, 72],
                         "weights": list(GROUP_B_WEIGHTS)}},
        "zero_weight_steps": [2, 4, 5, 6, 8, 9, 10],
        "loss_definition": ("the registered training_long_rollout per_step_losses tensor and "
                            "registered weights define both group losses; zero-weight steps are "
                            "in neither group but still advance the history exactly as registered"),
        "conflict_test": ("dot(g_A,g_B) < 0 over all trainable parameters; flatten/dot/norm "
                          "reductions in float64"),
        "projection": ("if conflict: g_A <- g_A - (dot/||g_B||^2) g_B; otherwise unchanged"),
        "update": ("g_A + g_B assigned to .grad, then the existing clip_grad_norm_ (clip=1) and "
                   "optimizer.step"),
        "order": ("fixed A -> B, a predeclared 2-task specialization of PCGrad Algorithm 1; "
                  "Appendix H's random-order advantage is a stated limitation, no RNG added"),
        "backward": "two autograd.grad calls per update (retain_graph on the first)",
        "sweep": ("none; no projection-coefficient/group/order sweep, no third group and no "
                  "loss-weight change"),
    }


def arm_spec():
    """The pinned H-D control spec, verbatim: this round touches no model switch."""
    spec = dict(anchor_arm.arm_spec(False))
    if "decoder_base_mode" in spec:
        raise RuntimeError("the pinned control spec must not carry a decoder base mode")
    if (spec.get("anchor_output_prior") is not False or spec.get("anomaly_feedback") is not False
            or spec.get("climatology_anchor_spec") is None
            or int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH):
        raise RuntimeError("the anchor control spec is not the frozen lazy-anchor instance")
    return spec


#: The control pinning and reading helpers live in their own module to keep R-051.
paired_control_pins = report.paired_control_pins
decision_flags = report.decision_flags


def protocol_constants():
    """The dose round's frozen fields with the gradient-surgery factor added."""
    body = depth_arm.protocol_constants()
    body["format"] = FORMAT
    body["recipe"].update(updates=UPDATES, steps=STEPS,
                          physical_weights=list(REGISTERED_PHYSICAL_WEIGHTS))
    body["evaluation"]["reasoning_steps"] = REASONING_STEPS
    body["candidate_draft_supervision"] = DRAFT_SUPERVISION
    body["control_draft_supervision"] = DRAFT_SUPERVISION
    body["gradient_surgery"] = gradient_surgery_block()
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


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = interior._identity()
    spec = arm_spec()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    arms = body.setdefault("arms", {})
    arms["candidate"] = {
        "mode": "long_rollout", "updates": UPDATES, "lr": wide.LR, "warmup": wide.WARMUP,
        "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS), "physical_steps": N_PHYSICAL_STEPS,
        "draft_supervision": DRAFT_SUPERVISION, "spatial_gradient_lambda": 0.0,
        "ema_decay": None, "gradient_surgery": CANDIDATE_GRADIENT_SURGERY,
        "steps": STEPS, "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE,
        "clip": wide.CLIP, "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
        "selection": "frozen endpoint; no validation selection",
        "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec),
                  "anchor_keys": list(parent.ANCHOR_KEYS)}}
    arms.setdefault("parent", {"updates": 1600, "pins": {}})
    arms["previous_dose_arm"] = previous_arm_pins()
    arms["paired_control_arm"] = paired_control_pins(body["train_data_identity"],
                                                     body["source_sha256"])
    parent.install_parent_pins(body)
    control_training = arms["paired_control_arm"]["training_code_sha256"]
    body["code_version_comparison"] = {
        "candidate_training_code_sha256": body["training_code_sha256"],
        "control_training_code_sha256": control_training,
        "equal": body["training_code_sha256"] == control_training,
        "declared_difference": (
            "the candidate runs the current training code with gradient_surgery='pcgrad', "
            "draft_supervision='deep', gradient_lambda=0.0 and ema_decay=None; the pinned "
            "control ran the earlier training revision with gradient_surgery='off'. The "
            "difference is recorded rather than assumed away; the only arithmetic change is "
            "the backward path."),
        "active_path": {"draft_supervision": DRAFT_SUPERVISION, "gradient_lambda": 0.0,
                        "ema_decay": None, "gradient_surgery": CANDIDATE_GRADIENT_SURGERY}}
    body["gradient_surgery"] = gradient_surgery_block()
    body["draft_supervision_organization"] = draft_organization_block()
    body["physical_weight_vector"] = physical_weight_block()
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
    if model_code_digest() != body["model_code_sha256"]:
        raise RuntimeError("model source identity changed")
    if body["model_code_sha256"] != parent.CURRENT_MODEL_CODE_SHA256:
        raise RuntimeError("model code differs from the predeclared unchanged revision")
    if training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("training source identity changed")
    if body["source_sha256"] != wide.source_sha256():
        raise RuntimeError("the wide store source identity changed")
    if body["train_windows"] != preflight_long_rollout_windows(
            wide.TRAIN_MANIFEST, physical_steps=N_PHYSICAL_STEPS):
        raise RuntimeError("train-window metadata changed")
    if (body["supervision"] != interior.supervision_block() or body["format"] != FORMAT
            or body["gradient_surgery"] != gradient_surgery_block()
            or body["draft_supervision_organization"] != draft_organization_block()
            or body["candidate_draft_supervision"] != DRAFT_SUPERVISION
            or body["control_draft_supervision"] != DRAFT_SUPERVISION
            or body["physical_weight_vector"] != physical_weight_block()):
        raise RuntimeError("the frozen gradient-surgery/supervision factor changed")
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
            or arm["gradient_surgery"] != CANDIDATE_GRADIENT_SURGERY
            or arm["spatial_gradient_lambda"] != 0.0 or arm["ema_decay"] is not None
            or int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH
            or "decoder_base_mode" in spec or spec != arm_spec()
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
            or paired["gradient_surgery"] != CONTROL_GRADIENT_SURGERY
            or paired["training_code_sha256"] != CONTROL_TRAINING_CODE_SHA256
            or paired["seeds"] != list(SEEDS)
            or paired["protocol_sha256"] != _index_record(PAIRED_CONTROL_RECORD)["protocol_sha256"]
            or paired["data_identity"] != body["train_data_identity"]
            or paired["source_sha256"] != body["source_sha256"]):
        raise RuntimeError("the pinned paired control differs from the registered record")
    code = body.get("code_version_comparison", {})
    if (code.get("candidate_training_code_sha256") != body["training_code_sha256"]
            or code.get("control_training_code_sha256") != paired["training_code_sha256"]
            or code.get("equal") != (code.get("candidate_training_code_sha256")
                                    == code.get("control_training_code_sha256"))
            or code.get("active_path", {}).get("gradient_surgery")
            != CANDIDATE_GRADIENT_SURGERY):
        raise RuntimeError("the recorded training-code comparison changed")
    for seed in SEEDS:
        if (body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"]
                != parent.MIGRATED_PARENT_SHA256[seed]):
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
    from training.r7_experiment import load_checkpoint
    from training.r7_arm_harness import sha256_file
    path, expected = parent.pinned_parent(seed)
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
    model, state, expansion = parent.expand_parent_state(
        saved["model"], spec, parent.anchor_payload(), seed=seed)
    return model, state, path, expected, expansion


def _arm_contract(body, spec, parent_path, parent_sha, expansion):
    return {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(parent.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent_path),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": 1600,
                               "parent_migration_receipt_sha256": parent.MIGRATION_RECEIPT_SHA256,
                               "anchor_table_run": str(parent.ANCHOR_TABLE_RUN),
                               "anchor_table_sha256": expansion["anchor_table_sha256"],
                               "description": ("registered v3-BD l6 endpoint, model-only migration "
                                               "under the unchanged current model revision, then "
                                               "the published train-only climatology table; "
                                               "gradient_surgery=pcgrad")},
            "autoregression": {"physical_steps": N_PHYSICAL_STEPS,
                "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
                "draft_supervision": DRAFT_SUPERVISION,
                "gradient_surgery": CANDIDATE_GRADIENT_SURGERY,
                "window_sha256": body["train_windows"]["window_sha256"],
                "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                "supervision": body["supervision"]["runner_block"]}}


def _train_arm(output, body, seed, deadline, mask):
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    spec = body["arms"][CANDIDATE]["model"]["spec"]
    model, state, parent_path, parent_sha, expansion = _parent_model(body, seed)
    folder = output / f"seed{seed}/training/{CANDIDATE}"
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model,
        contract=_arm_contract(body, spec, parent_path, parent_sha, expansion),
        parent_weights=state, physical_weights=REGISTERED_PHYSICAL_WEIGHTS, steps=STEPS,
        updates=UPDATES, lr=wide.LR, warmup=wide.WARMUP, seed=seed, device_name="cuda:0",
        deadline=deadline, supervision_mask=mask, draft_supervision=DRAFT_SUPERVISION,
        gradient_lambda=0.0, ema_decay=None, gradient_surgery=CANDIDATE_GRADIENT_SURGERY)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]
            or report["objective"] != "deep_supervised_latitude_area_mse"
            or report["internal_deep_supervision"] is not True
            or report.get("gradient_surgery") != CANDIDATE_GRADIENT_SURGERY
            or report["contract"]["autoregression"].get("gradient_surgery")
            != CANDIDATE_GRADIENT_SURGERY
            or "decoder_base_mode" in report["contract"]["model"]
            or "decoder_base_mode" in report["contract"]["autoregression"]
            or report["contract"]["autoregression"]["physical_weights"]
            != list(REGISTERED_PHYSICAL_WEIGHTS)
            or report["contract"]["autoregression"]["draft_supervision"] != DRAFT_SUPERVISION
            or report["contract"]["autoregression"].get("spatial_gradient_lambda") is not None
            or "ema_decay" in report or "ema_checkpoint" in report):
        raise RuntimeError(f"seed {seed} did not reach the frozen gradient-surgery endpoint")
    summary = summarize_updates(report["losses"])
    if summary["updates"] != UPDATES:
        raise RuntimeError("the surgery diagnostics do not cover the frozen update budget")
    if Path(checkpoint).name != f"update_{UPDATES:07d}.pt" or not Path(checkpoint).is_file():
        raise RuntimeError("the scored checkpoint is not the frozen update-2400 checkpoint")
    receipt = {"seed": seed, "arm": CANDIDATE, "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
               "scientific_claim": False, "limitations": LIMITATIONS,
               "physical_steps": N_PHYSICAL_STEPS, "draft_supervision": DRAFT_SUPERVISION,
               "gradient_surgery": CANDIDATE_GRADIENT_SURGERY,
               "gradient_surgery_summary": summary,
               "scored_checkpoint_kind": "final_optimizer_weights",
               "first_update_loss": float(report["losses"][0]["loss"]),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "objective": report["objective"],
               "internal_deep_supervision": report["internal_deep_supervision"],
               "training_dir": str(folder),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "reasoning_steps": STEPS,
               "dim": DIM, "expansion": expansion, "evaluations": {}}
    for lead in wide.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{CANDIDATE}/lead_{lead:03d}h"
        provenance = evaluate_local(wide.VAL_MANIFEST, output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=wide.BOUNDARY_MARGINS, deadline=deadline)
        wide._check_provenance(provenance, seed, lead)
        if provenance.get("checkpoint_sha256") != receipt["checkpoint_sha256"]:
            raise RuntimeError("the evaluation did not score the frozen gradient-surgery checkpoint")
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
        wide.write_exclusive(output / "readings.json", report.readings(output, body))
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
