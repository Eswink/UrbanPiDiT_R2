"""S3 decoder-base-anchor: score a zero decoder base instead of the persistence base.

The registered wide/interior_32 recipe is persistence-anchored: the decoder returns
``X_t + tendency`` with ``X_t = history[:, -1]`` and every local-solver proposal is
``Y = X_t + Decoder(Z)``.  This driver keeps the same parent, weights, layer shapes,
near-zero output head, 2400 updates, K=4, dim/depth, data, loss, physical weights,
seeds, scoring domain and climatology denominator, and changes only the base semantics:
the candidate sets ``decoder_base_mode='zero'``, which zeroes the decoder's direct
anchor and ``R7ForecastOutput.base_state`` together (so the local solver's
``proposal_anchor`` is zero as well).

The paired control is the pinned ``record:s3-anchor-isolation-result`` control arm
(persistence base, same lazy climatology anchor, not retrained).  The external
reference is the pinned ``record:s3-wide-interior-dose``.  Only model code changed,
so a new parent migration was required; it is pinned by
``scripts/r7_s3_decoder_base_anchor_parent.py``.  The 2023 test split is never read and
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
from scripts import r7_s3_decoder_base_anchor_support as report
from scripts import study_r7_s3_anchor_isolation as anchor_arm
from scripts import study_r7_s3_reasoning_depth as depth_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from training import r7_device_selection as devices
from scripts.r7_s3_decoder_base_anchor_text import (
    DECISION_RULE_TEXT, DESIGN_DECISION_TEXT, DIFFERENCE_TEXT, HYPOTHESIS_TEXT, LIMITATIONS,
)
interior = dose.interior
FORMAT = "r7-s3-decoder-base-anchor-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
CANDIDATE, SEEDS = "candidate", wide.SEEDS
DEVICE_MODE = "dual"
# One arm per seed: the same single-arm envelope as the EMA round.
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
CANDIDATE_DECODER_BASE_MODE = "zero"
CONTROL_DECODER_BASE_MODE = "persistence"
#: Anchor-round reported N - A spread; this is an identity band, not a new threshold.
BRIDGE_MIN, BRIDGE_MAX = -4.8e-6, 1.8e-5
BUDGET_PROFILE_NOTE = ("single-arm candidate envelope 21600/43200/86400 s; the pinned two-arm "
                       "H-D control used 43200/86400/115200 s")
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "model/weather_forecaster_r7.py",
               "model/process_forecast_r7.py", "scripts/r7_s3_decoder_base_anchor_parent.py",
               "scripts/r7_s3_decoder_base_anchor_support.py",
               "scripts/study_r7_s3_anchor_isolation.py",
               "scripts/r7_s3_anchor_isolation_parent.py",
               "scripts/study_r7_s3_decoder_base_anchor.py",
               "scripts/r7_s3_decoder_base_anchor_text.py")


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


def decoder_base_block(mode=CANDIDATE_DECODER_BASE_MODE):
    """The frozen base semantics, asserted at exactly the predeclared modes."""
    if mode not in ("zero", "persistence"):
        raise RuntimeError("the decoder base mode must be zero or persistence")
    if CANDIDATE_DECODER_BASE_MODE != "zero" or CONTROL_DECODER_BASE_MODE != "persistence":
        raise RuntimeError("the frozen candidate/control base semantics changed")
    return {"candidate_mode": CANDIDATE_DECODER_BASE_MODE,
            "control_mode": CONTROL_DECODER_BASE_MODE,
            "control_declaration": "absent in the pinned spec means persistence",
            "candidate_forecast": "0 + tendency = tendency",
            "control_forecast": "history[:, -1] + tendency",
            "base_state": ("the decoder's direct anchor and R7ForecastOutput.base_state are the "
                           "same tensor and are zeroed together; the combination zero with "
                           "anchor_output_prior=True and an installed climatology anchor is "
                           "rejected, so the two cannot desynchronize"),
            "local_solver": ("Y_proposal = proposal_anchor + Decoder(Z); with a zero base_state the "
                             "proposal_anchor is zero and the proposal is the decoder tendency"),
            "output_head": ("last layer untouched; no reset, no change_scale, no output-scale change"),
            "parameters": ("one state_dict key/shape set for both modes; no new tensors or parameters"),
            "lazy_anchor": ("the published train-only climatology table stays installed in both arms "
                            "with anchor_output_prior=False and anomaly_feedback=False; the forward "
                            "still evaluates it but its value reaches no output, and zeroing it "
                            "changes neither arm"),
            "sweep": "none; the base mode is a frozen factor, not a pass/close threshold"}


def arm_spec(mode=CANDIDATE_DECODER_BASE_MODE):
    """The H-D control spec plus the one new model switch; every other field is verbatim."""
    spec = dict(anchor_arm.arm_spec(False))
    if (spec.get("anchor_output_prior") is not False or spec.get("anomaly_feedback") is not False
            or spec.get("climatology_anchor_spec") is None
            or int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH):
        raise RuntimeError("the anchor control spec is not the frozen lazy-anchor instance")
    if "decoder_base_mode" in spec:
        raise RuntimeError("the archived spec already carries a decoder base mode")
    spec["decoder_base_mode"] = mode
    return spec


def _without_mode(spec):
    body = dict(spec)
    body.pop("decoder_base_mode", None)
    return body


#: The control pinning and reading helpers live in their own module to keep R-051.
paired_control_pins = report.paired_control_pins
decision_flags = report.decision_flags


def protocol_constants():
    """The dose round's frozen fields with the decoder-base factor added."""
    body = depth_arm.protocol_constants()
    body["format"] = FORMAT
    body["recipe"].update(updates=UPDATES, steps=STEPS,
                          physical_weights=list(REGISTERED_PHYSICAL_WEIGHTS))
    body["evaluation"]["reasoning_steps"] = REASONING_STEPS
    body["candidate_draft_supervision"] = DRAFT_SUPERVISION
    body["control_draft_supervision"] = DRAFT_SUPERVISION
    body["decoder_base"] = decoder_base_block()
    body["draft_supervision_organization"] = draft_organization_block()
    body["physical_weight_vector"] = physical_weight_block()
    body["budgets"] = {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                       "per_seed_seconds": PER_SEED_SECONDS}
    body["budget_profile_note"] = BUDGET_PROFILE_NOTE
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
        "decoder_base_mode": CANDIDATE_DECODER_BASE_MODE,
        "steps": STEPS, "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE,
        "clip": wide.CLIP, "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
        "selection": "frozen endpoint; no validation selection",
        "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec),
                  "decoder_base_mode": CANDIDATE_DECODER_BASE_MODE,
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
            "the candidate runs the current training code with draft_supervision='deep', "
            "gradient_lambda=0.0 and ema_decay=None; the pinned control ran the earlier "
            "training revision. The difference is recorded rather than assumed away; the "
            "active arithmetic path is the registered deep/zero-gradient/no-EMA path."),
        "active_path": {"draft_supervision": DRAFT_SUPERVISION, "gradient_lambda": 0.0,
                        "ema_decay": None}}
    body["decoder_base"] = decoder_base_block()
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
    if (model_code_digest() != body["model_code_sha256"]
            or training_code_digest() != body["training_code_sha256"]):
        raise RuntimeError("model/training source identity changed")
    if body["source_sha256"] != wide.source_sha256():
        raise RuntimeError("the wide store source identity changed")
    if body["train_windows"] != preflight_long_rollout_windows(
            wide.TRAIN_MANIFEST, physical_steps=N_PHYSICAL_STEPS):
        raise RuntimeError("train-window metadata changed")
    if (body["supervision"] != interior.supervision_block() or body["format"] != FORMAT
            or body["decoder_base"] != decoder_base_block()
            or body["draft_supervision_organization"] != draft_organization_block()
            or body["candidate_draft_supervision"] != DRAFT_SUPERVISION
            or body["control_draft_supervision"] != DRAFT_SUPERVISION
            or body["physical_weight_vector"] != physical_weight_block()):
        raise RuntimeError("the frozen decoder-base/supervision factor changed")
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
            or arm["decoder_base_mode"] != CANDIDATE_DECODER_BASE_MODE
            or arm["spatial_gradient_lambda"] != 0.0
            or int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH
            or spec != arm_spec() or _without_mode(spec) != _without_mode(arm_spec())
            or arm["model"]["spec_canonical_digest"] != canonical_digest(spec)
            or arm["model"]["decoder_base_mode"] != CANDIDATE_DECODER_BASE_MODE):
        raise RuntimeError("the candidate arm differs from the frozen recipe")
    if (body["arms"]["previous_dose_arm"]["record_id"] != PREVIOUS_DOSE_ARM_RECORD
            or body["arms"]["previous_dose_arm"]["updates"] != UPDATES
            or body["arms"]["previous_dose_arm"]["seeds"] != list(SEEDS)):
        raise RuntimeError("the pinned dose arm differs from the registered record")
    paired = body["arms"]["paired_control_arm"]
    if (paired["record_id"] != PAIRED_CONTROL_RECORD or paired["updates"] != UPDATES
            or paired["physical_weights"] != list(REGISTERED_PHYSICAL_WEIGHTS)
            or paired["draft_supervision"] != DRAFT_SUPERVISION
            or paired["decoder_base_mode"] != CONTROL_DECODER_BASE_MODE
            or paired["seeds"] != list(SEEDS)
            or paired["protocol_sha256"] != _index_record(PAIRED_CONTROL_RECORD)["protocol_sha256"]
            or paired["data_identity"] != body["train_data_identity"]
            or paired["source_sha256"] != body["source_sha256"]
            or not isinstance(paired.get("training_code_sha256"), str)
            or len(paired["training_code_sha256"]) != 64):
        raise RuntimeError("the pinned paired control differs from the registered record")
    code = body.get("code_version_comparison", {})
    if (code.get("candidate_training_code_sha256") != body["training_code_sha256"]
            or code.get("control_training_code_sha256") != paired["training_code_sha256"]
            or code.get("equal") != (code.get("candidate_training_code_sha256")
                                    == code.get("control_training_code_sha256"))):
        raise RuntimeError("the recorded training-code comparison changed")
    if body.get("budget_profile_note") != BUDGET_PROFILE_NOTE:
        raise RuntimeError("the budget-profile declaration changed")
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
                                               "under the decoder-base revision, then the published "
                                               "train-only climatology table; decoder_base_mode=zero")},
            "autoregression": {"physical_steps": N_PHYSICAL_STEPS,
                "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
                "draft_supervision": DRAFT_SUPERVISION,
                "decoder_base_mode": CANDIDATE_DECODER_BASE_MODE,
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
        gradient_lambda=0.0)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]
            or report["objective"] != "deep_supervised_latitude_area_mse"
            or report["internal_deep_supervision"] is not True
            or report["contract"]["model"].get("decoder_base_mode") != CANDIDATE_DECODER_BASE_MODE
            or report["contract"]["autoregression"].get("decoder_base_mode")
            != CANDIDATE_DECODER_BASE_MODE
            or report["contract"]["autoregression"]["physical_weights"]
            != list(REGISTERED_PHYSICAL_WEIGHTS)
            or report["contract"]["autoregression"]["draft_supervision"] != DRAFT_SUPERVISION
            or report["contract"]["autoregression"].get("spatial_gradient_lambda") is not None
            or "ema_decay" in report or "ema_checkpoint" in report):
        raise RuntimeError(f"seed {seed} did not reach the frozen zero-base endpoint")
    if Path(checkpoint).name != f"update_{UPDATES:07d}.pt" or not Path(checkpoint).is_file():
        raise RuntimeError("the scored checkpoint is not the frozen update-2400 checkpoint")
    receipt = {"seed": seed, "arm": CANDIDATE, "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
               "scientific_claim": False, "limitations": LIMITATIONS,
               "physical_steps": N_PHYSICAL_STEPS, "draft_supervision": DRAFT_SUPERVISION,
               "decoder_base_mode": CANDIDATE_DECODER_BASE_MODE,
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
            raise RuntimeError("the evaluation did not score the frozen zero-base checkpoint")
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
