"""S3 dense physical-step supervision: 5 sparse direct targets -> all 12 6 h steps.
The registered recipe supervises twelve 6 h physical steps with (1,.5,0,.5,0,0,0,.5,0,0,0,.5):
five steps carry a direct target loss and seven transition steps (18/30/36/42/54/60/66 h)
carry exactly zero, so their only signal is full-BPTT backpropagated from later leads.
This driver changes that vector and nothing else: the candidate uses (0.25,)*12 -- the
same total weight 3.0, but every step is directly supervised. The model, training code,
the migrated v3-BD 1600 parent (attempt04), the wide 129x129 input, the interior_32 mask,
2400 updates, K=4, depth 4, dim 192, the seeds, the scoring region and the climatology
denominator are held word-for-word, so model_code_sha256 / training_code_sha256 stay fixed
and no new migration is needed. The control arm is the pinned control of
record:s3-anchor-isolation-result (same parent, seeds and budget, sparse weights; its
anchor is inert, so its forward is the registered unanchored model bit for bit). The
external reference is the pinned record:s3-wide-interior-dose. Neither is retrained here; the 2023 test split is never read.
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
from scripts import r7_s3_anchor_isolation_parent as anchor_parent
from scripts import study_r7_s3_reasoning_depth as depth_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training import r7_device_selection as devices
from training.r7_study_process import bounded_process, worker_environment
interior = dose.interior
FORMAT = "r7-s3-dense-step-supervision-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
CANDIDATE, SEEDS = "candidate", wide.SEEDS
DEVICE_MODE = "dual"
# One arm per seed instead of the H-D round's two: 2400 updates plus five-lead scoring.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 21600., 43200., 86400.
ESTIMATED_PEAK_BYTES = 9728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
DEFAULT_OUT = ROOT / "outputs/r7_s3_dense_step_supervision_20261018_attempt01"
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
PAIRED_CONTROL_RECORD = "s3-anchor-isolation-result"
PAIRED_CONTROL_RUN = ROOT / "outputs/r7_s3_anchor_isolation_20261015_attempt01"
#: The registered sparse contour, verbatim from study_r7_s3_wide_single_factor.py:38.
REGISTERED_PHYSICAL_WEIGHTS = tuple(wide.PHYSICAL_WEIGHTS)
#: The same total weight 3.0 spread uniformly over all 12 direct targets.
DENSE_PHYSICAL_WEIGHTS = (0.25,) * len(REGISTERED_PHYSICAL_WEIGHTS)
N_PHYSICAL_STEPS = len(DENSE_PHYSICAL_WEIGHTS)
#: Anchor-round reported N - A spread; this is an identity band, not a new threshold.
BRIDGE_MIN, BRIDGE_MAX = -4.8e-6, 1.8e-5
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "scripts/r7_s3_anchor_isolation_parent.py",
               "scripts/study_r7_s3_dense_step_supervision.py")
DESIGN_DECISION_TEXT = (
    "Single factor: the direct-supervision weight vector over the 12 6 h physical steps. The registered contour supervises 5 steps "
    "(6/12/24/48/72 h) and gives 7 transition steps (18/30/36/42/54/60/66 h) zero direct weight; they are still reached by full-BPTT. "
    "The candidate keeps the total weight at 3.0 and supervises all 12 uniformly at 0.25. Everything else -- wide 129x129 input, "
    "interior_32 supervision, 2400 updates, K=4, depth 4, dim 192, the migrated v3-BD 1600 parent, seeds, scoring region and "
    "climatology denominator -- is unchanged, and no model or training source is touched.")
DECISION_RULE_TEXT = (
    "Predeclared exit, no new numeric threshold: (a) candidate 72 h interior_32 t2m skill > 0 on all three seeds; (b) paired 72 h "
    "candidate-minus-control RMSE < 0 on all three seeds, control = pinned record s3-anchor-isolation-result control arm; (c) candidate "
    "48 h skill > 0 on all three seeds; (d) seed-mean candidate-minus-control delta at 6/12/24 h <= 0; (e) identity bridge: "
    "control-minus-registered-dose RMSE inside the anchor round's reported -4.8e-6..+1.8e-5 K band. All of (a)-(e) -> request "
    "independent review. Any of (a)-(d) false -> permanently close the dense-coverage branch. (e) false is identity/engineering "
    "unresolved, not a scientific verdict. Report all five leads and all 17 variables; no scientific pass is claimed.")
HYPOTHESIS_TEXT = (
    "If the registered recipe's remaining long-lead error is partly caused by the seven transition steps receiving no direct target "
    "signal, then supervising all twelve steps at the same total weight 3.0 should improve the paired 72 h RMSE against the sparse "
    "control on all three seeds and lift absolute 72 h skill above the climatology denominator without degrading 6/12/24 h. If the "
    "paired deltas or the absolute gate fail, coverage is not the binding constraint at this dose.")
DIFFERENCE_TEXT = (
    "One factor only: physical_weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5) -> (0.25,)*12. Both sum to exactly 3.0 and are accepted by "
    "validate_physical_weights. The candidate adds direct objectives on 18/30/36/42/54/60/66 h and lowers the 48/72 h weights from "
    "0.5 to 0.25; this is a coverage change, not the already-closed long-lead emphasis change. Same wide 129x129 input, interior_32 "
    "mask (mask_sha256 4a2a576b...), 2400 updates, K=4, depth 4, dim 192, migrated v3-BD 1600 parent, LR 2e-5, warmup 10, FP32, "
    "batch 1 and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32, one ROI, 17 channels, 2017-2021 train / 2022 val); "
    "no significance, convergence, SOTA or generalization claim",
    "control and external reference are pinned registered arms (s3-anchor-isolation-result control, s3-wide-interior-dose), not "
    "retrained inside this protocol",
    "the candidate continues migration attempt04; the parent was trained with the sparse contour, so this round prices a route "
    "from that parent, not a from-scratch retrain",
    "the runner does not normalize physical weights by their sum; equal total weight 3.0 is a design identity, not a "
    "learning-rate equivalence proof",
    "interior_32 leaves the outer ring unconstrained; only interior_32 is scored, and all 17 variables are reported without "
    "cross-unit averaging",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident; validation split only and the test "
    "split stays sealed until a later preregistered read",
]
def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))
def physical_weight_block():
    """The frozen coverage factor and the arithmetic that makes it a single factor."""
    control, dense = tuple(REGISTERED_PHYSICAL_WEIGHTS), tuple(DENSE_PHYSICAL_WEIGHTS)
    if len(control) != 12 or len(dense) != 12:
        raise RuntimeError("the physical-step axis must have exactly 12 steps")
    if abs(sum(control) - 3.0) > 0.0 or abs(sum(dense) - 3.0) > 0.0:
        raise RuntimeError("both weight vectors must sum to exactly 3.0")
    control_steps = [index for index, weight in enumerate(control) if weight > 0]
    dense_steps = [index for index, weight in enumerate(dense) if weight > 0]
    newly = [index for index, weight in enumerate(control) if weight == 0]
    hours = [6 * index + 6 for index in newly]
    if control_steps != [0, 1, 3, 7, 11] or dense_steps != list(range(12)):
        raise RuntimeError("the sparse/dense support differs from the frozen design")
    if hours != [18, 30, 36, 42, 54, 60, 66]:
        raise RuntimeError("the newly supervised lead hours differ from the frozen design")
    return {"control_weights": list(control), "candidate_weights": list(dense),
            "control_sum": sum(control), "candidate_sum": sum(dense),
            "control_direct_steps": control_steps, "candidate_direct_steps": dense_steps,
            "newly_supervised_steps": newly, "newly_supervised_lead_hours": hours,
            "change": "direct-supervision coverage only; total physical weight fixed at 3.0"}
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
    """The dose round's frozen fields with the physical-weight vector swapped in."""
    body = depth_arm.protocol_constants()
    body["format"] = FORMAT
    body["recipe"].update(updates=UPDATES, steps=STEPS,
                          physical_weights=list(DENSE_PHYSICAL_WEIGHTS))
    body["evaluation"]["reasoning_steps"] = REASONING_STEPS
    body["control_physical_weights"] = list(REGISTERED_PHYSICAL_WEIGHTS)
    body["physical_step_supervision"] = physical_weight_block()
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
def paired_control_pins():
    """Pin the H-D control arm N: same current-code parent and budget, sparse weights."""
    from training.r7_arm_harness import sha256_file
    record = _index_record(PAIRED_CONTROL_RECORD)
    if sha256_file(ROOT / record["evidence_path"]) != record["evidence_sha256"]:
        raise RuntimeError("the H-D result doc drifted from its index record")
    readings_path = PAIRED_CONTROL_RUN / "readings.json"
    readings = json.loads(readings_path.read_text(encoding="utf-8"))
    if (readings["scientific_claim"] is not False or readings["test_read"] is not False
            or readings["protocol_sha256"] != record["protocol_sha256"]):
        raise RuntimeError("the H-D control readings differ from the registered protocol")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "updates": UPDATES, "physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
            "seeds": list(SEEDS), "run_dir": str(PAIRED_CONTROL_RUN),
            "readings_sha256": sha256_file(readings_path), "seeds_readings": {}}
    for seed in SEEDS:
        receipt = json.loads((PAIRED_CONTROL_RUN / f"seed{seed}_control_receipt.json").read_text(
            encoding="utf-8"))
        prior = readings["seeds"][str(seed)]["arms"]["control"]
        if (receipt["arm"] != "control" or receipt["anomaly_feedback"] is not False
                or int(receipt["updates"]) != UPDATES or int(receipt["reasoning_steps"]) != REASONING_STEPS
                or receipt["parent_checkpoint_sha256"] != anchor_parent.MIGRATED_PARENT_SHA256[seed]):
            raise RuntimeError(f"the H-D seed {seed} control receipt differs from the frozen arm")
        entry = {"checkpoint_sha256": receipt["checkpoint_sha256"], "parent_checkpoint_sha256": receipt["parent_checkpoint_sha256"], "leads": {}}
        for lead in wide.EVALUATION_LEADS:
            lead_record = receipt["evaluations"][str(lead)]
            folder = Path(lead_record["dir"])
            if sha256_file(folder / "boundary_rmse.csv") != lead_record["boundary_rmse_csv_sha256"]:
                raise RuntimeError(f"the H-D control table drifted for seed {seed} lead {lead}")
            entry["leads"][str(lead)] = {
                "dir": str(folder), "boundary_rmse_csv_sha256": lead_record["boundary_rmse_csv_sha256"],
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
        "physical_weights": list(DENSE_PHYSICAL_WEIGHTS), "physical_steps": N_PHYSICAL_STEPS,
        "steps": STEPS, "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE,
        "clip": wide.CLIP, "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
        "selection": "frozen endpoint; no validation selection",
        "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}},
        "parent": {"updates": 1600, "pins": {}}}
    body["arms"]["previous_dose_arm"] = previous_arm_pins()
    body["arms"]["paired_control_arm"] = paired_control_pins()
    anchor_parent.install_parent_pins(body)
    body["physical_step_supervision"] = physical_weight_block()
    body["control_physical_weights"] = list(REGISTERED_PHYSICAL_WEIGHTS)
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
            or body["physical_step_supervision"] != physical_weight_block()):
        raise RuntimeError("the frozen supervision/coverage factor changed")
    if (body["recipe"]["updates"] != UPDATES or body["recipe"]["steps"] != STEPS
            or body["evaluation"]["reasoning_steps"] != REASONING_STEPS
            or body["recipe"]["physical_weights"] != list(DENSE_PHYSICAL_WEIGHTS)
            or body["control_physical_weights"] != list(REGISTERED_PHYSICAL_WEIGHTS)):
        raise RuntimeError("the frozen recipe/physical-weight factor changed")
    if not devices.assignment_is_valid(body, wide.GPU_UUID):
        raise RuntimeError("declared GPU differs from the frozen assignment")
    arm, spec = body["arms"][CANDIDATE], body["arms"][CANDIDATE]["model"]["spec"]
    if (arm["physical_weights"] != list(DENSE_PHYSICAL_WEIGHTS)
            or arm["physical_steps"] != N_PHYSICAL_STEPS or int(arm["updates"]) != UPDATES
            or int(arm["steps"]) != STEPS or int(spec.get("dim", 0)) != DIM
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
            or paired["seeds"] != list(SEEDS)
            or paired["protocol_sha256"] != _index_record(PAIRED_CONTROL_RECORD)["protocol_sha256"]):
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
                                               "then the dense 12-step direct-supervision vector")},
            "autoregression": {"physical_steps": N_PHYSICAL_STEPS,
                "physical_weights": list(DENSE_PHYSICAL_WEIGHTS),
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
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model,
        contract=_arm_contract(body, spec, parent, parent_sha), parent_weights=state,
        physical_weights=DENSE_PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=wide.LR,
        warmup=wide.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline, supervision_mask=mask)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]
            or report["contract"]["autoregression"]["physical_weights"] != list(DENSE_PHYSICAL_WEIGHTS)):
        raise RuntimeError(f"seed {seed} did not reach the frozen dense endpoint")
    receipt = {"seed": seed, "arm": CANDIDATE, "physical_weights": list(DENSE_PHYSICAL_WEIGHTS),
               "physical_steps": N_PHYSICAL_STEPS,
               "control_physical_weights": list(REGISTERED_PHYSICAL_WEIGHTS),
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "reasoning_steps": STEPS,
               "dim": DIM, "evaluations": {}}
    for lead in wide.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{CANDIDATE}/lead_{lead:03d}h"
        provenance = evaluate_local(wide.VAL_MANIFEST, output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=wide.BOUNDARY_MARGINS, deadline=deadline)
        wide._check_provenance(provenance, seed, lead)
        receipt["evaluations"][str(lead)] = {
            "dir": str(destination), "n_evaluated": provenance["n_evaluated"],
            "n_available_windows": provenance["n_available_windows"],
            "elapsed_seconds": provenance["elapsed_seconds"],
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
           "delta_minus_previous_dose": {}, "all_variable_interior_rmse": {}}
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
    receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
    if receipt["seed"] != seed or receipt["physical_weights"] != list(DENSE_PHYSICAL_WEIGHTS):
        raise RuntimeError("the candidate receipt differs from the frozen dense arm")
    climatology = body["wide_climatology_reference"]["leads"]
    arm = {"physical_weights": receipt["physical_weights"], "interior_t2m_rmse": {},
           "interior_t2m_skill": {}, "all_variable_interior_rmse": {}}
    for lead in wide.EVALUATION_LEADS:
        row = _table_readings(receipt["evaluations"][str(lead)]["dir"], climatology[str(lead)]["interior_32_t2m_rmse"])
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
                "factor": "direct supervision over the 12 6 h physical steps",
                "physical_step_supervision": body["physical_step_supervision"],
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
def run_attempt(output, *, code_archive, code_commit, code_sha256):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = fresh_output(output)
    if not Path(wide.STORE).is_dir():
        raise FileNotFoundError(f"the wide store is not built yet: {wide.STORE}")
    if (not code_archive or not isinstance(code_commit, str) or len(code_commit) != 40
            or not isinstance(code_sha256, str) or len(code_sha256) != 64):
        raise ValueError("prepared exact code archive/commit/SHA required before execution")
    if DEVICE_MODE != "single":  # dual-card rounds are driven by `r7_round.py stage-seeds`
        raise RuntimeError("dual-card mode is driven by `r7_round.py stage-seeds`, not run_attempt")
    started = time.perf_counter()
    deadline = started + HARD_CAP_SECONDS
    output.mkdir(parents=True, exist_ok=False)
    processes, receipts = [], {}
    try:
        freeze(output, started=started, deadline=deadline, code_archive=Path(code_archive).absolute(),
               code_commit=code_commit, code_sha256=code_sha256)
        phases = [("prepare", None), ("archive", None)] + [("seed", s) for s in SEEDS] \
            + [("reading", None)]
        peak, body = 0, None
        for phase, seed in phases:
            gate = recipe._shared().gpu_gate(wide.GPU_UUID, estimated_peak_bytes=ESTIMATED_PEAK_BYTES,
                                            owned_reserved_peak_bytes=peak, margin_bytes=HEADROOM_MARGIN_BYTES)
            tag = phase if seed is None else f"seed{seed}"
            wide.write_exclusive(output / f"{tag}_spawn_gate.json", gate)
            limit = min(deadline, time.perf_counter() + PER_SEED_SECONDS) if phase == "seed" else deadline
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                       "--out", str(output), "--deadline", str(limit)]
            if seed is not None:
                command += ["--seed", str(seed)]
            process = bounded_process(command, cwd=ROOT, log_path=output / f"{tag}.log", deadline=limit,
                                      env=worker_environment(wide.GPU_UUID))
            processes.append({"phase": phase, "seed": seed, **process})
            wide.write_exclusive(output / f"{tag}_process.json", processes[-1])
            if process["status"] != "success":
                raise RuntimeError(f"{tag} failed: {process['status']}")
            if phase == "prepare":
                body = json.loads((output / "prepared_protocol.json").read_text(encoding="utf-8"))
                wide.write_exclusive(output / "protocol.json", body)
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
                  "supervision": body["supervision"], "updates": UPDATES, "dim": DIM,
                  "reasoning_steps": STEPS, "physical_step_supervision": body["physical_step_supervision"],
                  "train_data_identity": body["train_data_identity"],
                  "val_data_identity": body["val_data_identity"], "elapsed_seconds_total": elapsed,
                  "soft_overrun_seconds": max(0., elapsed - PLANNED_SECONDS),
                  "hard_overrun_seconds": max(0., elapsed - HARD_CAP_SECONDS), "processes": processes,
                  "seeds": receipts, "limitations": body["limitations"]}
        result.update(json.loads((output / "readings.json").read_text(encoding="utf-8")))
        wide.write_exclusive(output / "result.json", result)
        wide.write_exclusive(output / "attempt.json", {"status": "complete", "scientific_claim": False,
            "test_read": False, "protocol_sha256": body["protocol_sha256"], "elapsed_seconds_total": elapsed,
            "decision_rule": body["decision_rule"]})
        return result
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        wide.write_exclusive(output / "failure.json", {"status": "failed", "scientific_claim": False,
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
