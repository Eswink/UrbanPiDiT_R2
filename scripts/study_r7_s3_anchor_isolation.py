"""H-D: isolate the climatology-anchor package's anomaly-feedback mechanism on the registered arm.

H-C was closed by its predeclared exit (G->L null at every lead, 72 h negative on both arms), the
architecture axes are closed, and decision 0048 closed the capacity axis, so this driver takes the
one direction that has a positive long-lead signal in this campaign: the train-only climatology
anchor package. That package changed three mechanisms together -- the output prior, the proposal
prior and the recurrence's feedback representation -- and its own limitations say it therefore
carries "no isolated mechanism attribution".

This round separates the one mechanism the repository lets us separate. ``anomaly_feedback`` changes
only the draft feedback fed to the K-step recurrence, from the absolute draft to ``draft - C_valid``.
The output prior and the proposal prior are coupled to the anchor's mere presence by
``ProcessForecastCoReasoner``'s own validation rule, so they are held OFF in both arms through the
new ``anchor_output_prior=False`` switch rather than pretended apart. Both arms install the same
published train-only climatology table, so they differ by exactly one boolean.

Every other factor is held fixed: the wide 129x129 input, the interior_32 mask, 2400 updates, K=4,
depth 4, dim 192, the physical weights, the migrated v3-BD 1600 parent, the seeds, the scoring region
and the climatology denominator. Report-only.
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

FORMAT = "r7-s3-anchor-isolation-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
CONTROL, CANDIDATE = "control", "candidate"
ARMS = ((CONTROL, False), (CANDIDATE, True))
SEEDS = wide.SEEDS
DEVICE_MODE = "dual"
# Two arms per seed, each 2400 updates on the same trunk shape as H-C, which MEASURED 9170-9174 MiB
# peak and 4936-5391 s per arm. The envelope is the H-C one, recalibrated against that measurement.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 43200., 86400., 115200.
ESTIMATED_PEAK_BYTES = 9728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
DEFAULT_OUT = ROOT / "outputs/r7_s3_anchor_isolation_20261015_attempt01"
PREVIOUS_DOSE_ARM_RUN = ROOT / "outputs/r7_s3_wide_interior_dose_20261009_attempt02"
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "model/weather_forecaster_r7.py", "model/process_step_r7.py",
               "model/process_forecast_r7.py", "scripts/r7_s3_anchor_table_build.py",
               "scripts/r7_s3_anchor_isolation_parent.py", "scripts/study_r7_s3_anchor_isolation.py")

DESIGN_DECISION_TEXT = (
    "Single factor: the draft feedback the K-step reasoning recurrence sees, absolute versus the "
    "anomaly relative to a train-only valid-time climatology. H-C was closed by its predeclared exit, "
    "the architecture axes and the capacity axis are closed, and the climatology-anchor package is the "
    "only direction with a positive long-lead signal (t2m 48/72 h better than both controls). That "
    "package changed three mechanisms at once, so this round isolates the one the repository allows: "
    "anomaly feedback. Both arms install the same published climatology table and keep the output and "
    "proposal priors off, so they differ by exactly one boolean.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of the candidate "
    "arm's interior_32 t2m RMSE minus this protocol's own control arm's (the feedback attribution), (b) "
    "each arm's delta against the registered 2400-update dose arm's interior_32 t2m RMSE (index record "
    "s3-wide-interior-dose, pinned, not retrained here), and (c) each arm's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2, at the five leads {6,12,24,48,72} h. The external reference is pinned.")
HYPOTHESIS_TEXT = (
    "If the climatology-anchor package's 48/72 h gain came from the recurrence seeing a "
    "climatology-relative feedback rather than from the output prior, then giving the K-step "
    "recurrence the anomaly (draft minus the train-only valid-time climatology) while leaving the "
    "decoder on its registered persistence anchor should move 48/72 h interior_32 t2m skill relative "
    "to a control arm that installs the same table and differs only in that boolean. If candidate minus "
    "control is null, the feedback representation is inert and the package's gain could only have come "
    "from the output prior, which the package screen measured as destroying 6/12/24 h.")
DIFFERENCE_TEXT = (
    "One factor only: anomaly_feedback False -> True. Both arms carry climatology_anchor_spec (the "
    "published 16-bucket train-only table) and anchor_output_prior=False, so the decoder and the "
    "solver proposal keep the registered persistence anchor and the anchor is read only by the "
    "recurrence's feedback. With anomaly_feedback off the anchor is read by nothing and the control "
    "arm is bit-identical to the unanchored registered model outside the two anchor buffers; with it "
    "on the feedback becomes draft - C_valid. The two arms add the same two tensors with identical "
    "verified bytes, so there is no parameter-count difference at all. Same wide 129x129 input, same "
    "interior_32 mask (mask_sha256 4a2a576b...), same 2400 updates, same K=4, same depth 4, same "
    "dim 192, same registered physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), same migrated v3-BD 1600 "
    "parent (dim 192), same LR 2e-5, warmup 10, FP32, batch 1 and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "the output prior and the proposal prior are NOT separated from each other: the anchor's presence "
    "switches both, so this round holds both off and isolates only the feedback representation",
    "the derived climatology is a 4-month seasonal mean because the store's train split covers only "
    "Jan/Apr/Jul/Oct; it is not a 12-month annual climatology, and the current val cannot test a "
    "full-year climatology either",
    "the external reference is a pinned registered arm (2400-update interior s3-wide-interior-dose), "
    "not retrained inside this protocol; the control arm IS retrained here",
    "both arms are trained jointly with the registered parent's weights, so each prices a route from "
    "the parent rather than the mechanism in isolation",
    "interior_32 supervision leaves the outer ring unconstrained during training; only interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident; validation "
    "split only and the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))


_cached_payload = None


def anchor_payload():
    """Load and verify the published climatology table once per process."""
    global _cached_payload
    if _cached_payload is None:
        _cached_payload = anchor_parent.anchor_payload()
    return _cached_payload


def arm_spec(feedback):
    """The archived process spec with the anchor and the one-factor boolean bound in."""
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError(f"the archived process spec is not dim {DIM} depth {DEPTH}")
    if (spec.get("climatology_anchor_spec") or spec.get("anomaly_feedback")
            or spec.get("anchor_output_prior") or spec.get("process_init_conditioning")
            or spec.get("typed_evidence_mode") or spec.get("global_mixing_mode")):
        raise RuntimeError("the archived process spec already carries an architecture switch")
    spec["climatology_anchor_spec"] = anchor_payload()["spec"]
    spec["anchor_output_prior"] = False
    spec["anomaly_feedback"] = bool(feedback)
    return spec


def anchor_block(feedback):
    """The frozen H-D mechanism rule and its declared controls, for one arm."""
    pins = anchor_payload()["pins"]
    return {"arm": CANDIDATE if feedback else CONTROL, "anomaly_feedback": bool(feedback),
            "anchor_table": {key: pins[key] for key in (
                "table_sha256", "spec_sha256", "table_file_sha256", "n_buckets",
                "n_selected_steps", "source_sha256", "train_data_identity")},
            "anchor_output_prior": False,
            "decoder_and_proposal_anchor": "registered persistence anchor (last history frame)",
            "feedback": ("draft - C_valid" if feedback else "draft"),
            "new_keys": list(anchor_parent.ANCHOR_KEYS),
            "new_params": 0,
            "parameter_matched_by_construction": True,
            "shared_initialization": ("both arms install the same published table bytes; the anchor "
                                      "content digest is identical across arms and seeds"),
            "identity_control": ("with the table zeroed, draft - 0 = draft, so the candidate collapses "
                                 "onto the control bit for bit; asserted per seed by the expansion"),
            "control_identity": ("with anomaly_feedback off the anchor is read by nothing, so the "
                                 "control arm is bit-identical to the unanchored registered model "
                                 "outside the two anchor buffers")}


def protocol_constants():
    """The dose round's frozen fields with the feedback factor swapped in."""
    body = depth_arm.protocol_constants()
    body["format"] = FORMAT
    body["recipe"]["updates"] = UPDATES
    body["recipe"]["steps"] = STEPS
    body["evaluation"]["reasoning_steps"] = REASONING_STEPS
    body["budgets"] = {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                       "per_seed_seconds": PER_SEED_SECONDS}
    body["design_decision"] = DESIGN_DECISION_TEXT
    body["decision_rule"] = DECISION_RULE_TEXT
    body["hypothesis"] = HYPOTHESIS_TEXT
    body["difference"] = DIFFERENCE_TEXT
    body["limitations"] = list(LIMITATIONS)
    return body


def _index_record(record_id):
    return dose._index_record(record_id)


def previous_arm_pins():
    """Pin the registered 2400-update interior-dose arm this round compares against.

    ``study_r7_s3_reasoning_depth`` pins the same record and run at the same 2400 updates, so the
    helper is delegated rather than copied: one definition of what "the registered dose arm" means.
    """
    return depth_arm.previous_arm_pins()


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = interior._identity()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    template = dict(body["arms"].pop("candidate"))
    for name, feedback in ARMS:
        spec = arm_spec(feedback)
        body["arms"][name] = dict(template, updates=UPDATES, anomaly_feedback=bool(feedback),
                                  model={"kind": "process", "spec": spec,
                                         "spec_canonical_digest": canonical_digest(spec)})
    body["arms"]["previous_dose_arm"] = previous_arm_pins()
    anchor_parent.install_parent_pins(body)
    body["climatology_anchor"] = {"arms": {name: anchor_block(feedback) for name, feedback in ARMS}}
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
    if body["train_windows"] != preflight_long_rollout_windows(wide.TRAIN_MANIFEST, physical_steps=12):
        raise RuntimeError("train-window metadata changed")
    if body["supervision"] != interior.supervision_block() or body["format"] != FORMAT:
        raise RuntimeError("the frozen supervision factor changed")
    if (body["recipe"]["updates"] != UPDATES or body["recipe"]["steps"] != STEPS
            or body["evaluation"]["reasoning_steps"] != REASONING_STEPS):
        raise RuntimeError("the frozen recipe changed")
    if body["climatology_anchor"] != {"arms": {name: anchor_block(feedback)
                                               for name, feedback in ARMS}}:
        raise RuntimeError("the frozen climatology-anchor rule changed")
    if not devices.assignment_is_valid(body, wide.GPU_UUID):
        raise RuntimeError("declared GPU differs from the frozen assignment")
    for name, feedback in ARMS:
        arm = body["arms"][name]
        spec = arm["model"]["spec"]
        if (arm["anomaly_feedback"] is not bool(feedback) or int(arm["updates"]) != UPDATES
                or int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH
                or spec.get("anomaly_feedback") is not bool(feedback)
                or spec.get("anchor_output_prior") is not False
                or spec.get("climatology_anchor_spec") != anchor_payload()["spec"]
                or arm["model"]["spec_canonical_digest"] != canonical_digest(spec)):
            raise RuntimeError(f"the {name} arm differs from the frozen recipe")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    if (int(body["estimated_peak_bytes"]) < MIN_ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("the H-D peak estimate was not recalibrated")
    previous = body["arms"]["previous_dose_arm"]
    if (previous["record_id"] != PREVIOUS_DOSE_ARM_RECORD or previous["updates"] != UPDATES
            or previous["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned 2400-update dose arm differs from the registered record")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(wide.TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")
    return body


def arm_name(feedback):
    if type(feedback) is not bool:
        raise ValueError(f"anomaly_feedback must be boolean, got {type(feedback).__name__}")
    for name, value in ARMS:
        if value == bool(feedback):
            return name
    raise ValueError(f"unknown anomaly_feedback value {feedback}")


def _parent_model(body, seed, feedback):
    from training.r7_experiment import load_checkpoint
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    path, expected = anchor_parent.pinned_parent(seed)
    if (sha256_file(path) != expected
            or body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"] != expected):
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != 1600 or contract["seed"] != seed or contract["mode"] != "l6"
            or contract["data_identity"] != shared.V3_DATA_IDENTITY
            or contract["source_sha256"] != shared.V3_SOURCE_SHA256):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"][arm_name(feedback)]["model"]["spec"]
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError("the arm spec is not the frozen dim-192 depth-4 trunk")
    model, state, expansion = anchor_parent.expand_parent_state(
        saved["model"], spec, anchor_payload(), seed=seed)
    return model, state, path, expected, expansion


def _arm_contract(body, spec, feedback, parent, parent_sha, expansion):
    return {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(anchor_parent.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": 1600,
                               "anchor_table_run": str(anchor_parent.ANCHOR_TABLE_RUN),
                               "anchor_table_sha256": expansion["anchor_table_sha256"],
                               "description": ("registered v3-BD l6 endpoint, model-only migration, "
                                               "then the published train-only climatology table; "
                                               f"anomaly_feedback={bool(feedback)}")},
            "autoregression": {"physical_steps": 12, "physical_weights": list(wide.PHYSICAL_WEIGHTS),
                "window_sha256": body["train_windows"]["window_sha256"],
                "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                "supervision": body["supervision"]["runner_block"]}}


def _train_arm(output, body, seed, name, feedback, deadline, mask):
    """Fine-tune one arm from the migrated parent and score its five leads."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    spec = body["arms"][name]["model"]["spec"]
    model, state, parent, parent_sha, expansion = _parent_model(body, seed, feedback)
    contract = _arm_contract(body, spec, feedback, parent, parent_sha, expansion)
    folder = output / f"seed{seed}/training/{name}"
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model,
        contract=contract, parent_weights=state, physical_weights=wide.PHYSICAL_WEIGHTS, steps=STEPS,
        updates=UPDATES, lr=wide.LR, warmup=wide.WARMUP, seed=seed, device_name="cuda:0",
        deadline=deadline, supervision_mask=mask)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    receipt = {"seed": seed, "arm": name, "anomaly_feedback": bool(feedback),
               "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "reasoning_steps": STEPS,
               "dim": DIM, "expansion": expansion, "evaluations": {}}
    for lead in wide.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{name}/lead_{lead:03d}h"
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
    mask = interior.supervision_mask().to("cuda:0")
    receipts = {}
    for name, feedback in ARMS:
        receipt = _train_arm(output, body, seed, name, feedback, deadline, mask)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
    # Both arms must carry the same published climatology bytes; a mismatch means the factor is not
    # the only difference between them.
    tables = {name: receipts[name]["expansion"]["anchor_table_sha256"] for name in receipts}
    if len(set(tables.values())) != 1:
        raise RuntimeError(f"seed {seed}: the arms do not share one climatology table: {tables}")


def _reference_delta(body, seed, lead, arm_rmse):
    """The pinned registered 2400-update dose arm's RMSE, verified against its freeze pin."""
    from training.r7_arm_harness import sha256_file
    previous = body["arms"]["previous_dose_arm"]
    previous_lead = previous["seeds_evaluations"][str(seed)]["leads"][str(lead)]
    previous_dir = Path(previous_lead["dir"])
    if sha256_file(previous_dir / "boundary_rmse.csv") != previous_lead["boundary_rmse_csv_sha256"]:
        raise RuntimeError("the registered 2400-update dose arm table drifted from its freeze pin")
    previous_rmse = wide._region_variable_rmse(previous_dir / "boundary_rmse.csv", "interior_32", "t2m")
    return previous_rmse, arm_rmse - previous_rmse


def _arm_readings(output, body, seed, name, feedback, climatology):
    receipt = json.loads((output / f"seed{seed}_{name}_receipt.json").read_text(encoding="utf-8"))
    arm = {"anomaly_feedback": bool(feedback), "interior_t2m_rmse": {},
           "previous_dose_t2m_rmse": {}, "delta_minus_previous_dose": {},
           "interior_t2m_skill": {}}
    for lead in wide.EVALUATION_LEADS:
        arm_dir = Path(receipt["evaluations"][str(lead)]["dir"])
        arm_rmse = wide._region_variable_rmse(arm_dir / "boundary_rmse.csv", "interior_32", "t2m")
        dose_rmse, delta_dose = _reference_delta(body, seed, lead, arm_rmse)
        climatology_rmse = climatology["leads"][str(lead)]["interior_32_t2m_rmse"]
        arm["interior_t2m_rmse"][str(lead)] = arm_rmse
        arm["previous_dose_t2m_rmse"][str(lead)] = dose_rmse
        arm["delta_minus_previous_dose"][str(lead)] = delta_dose
        arm["interior_t2m_skill"][str(lead)] = 1.0 - (arm_rmse / climatology_rmse) ** 2
    return arm


def _readings(output, body):
    climatology = body["wide_climatology_reference"]
    pins = anchor_payload()["pins"]
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
                "updates": UPDATES, "dim": DIM, "reasoning_steps": REASONING_STEPS,
                "factor": "reasoning-recurrence feedback: absolute draft vs draft minus C_valid",
                "declared_difference": ("both arms install the same 16-bucket train-only climatology "
                                        "table with anchor_output_prior=False and add zero parameters; "
                                        "they differ only in anomaly_feedback"),
                "anchor_table": {key: pins[key] for key in ("table_sha256", "spec_sha256",
                                                            "n_buckets", "n_selected_steps",
                                                            "source_sha256", "train_data_identity")},
                "confound": ("the output prior and the proposal prior are coupled by the anchor's "
                             "presence and are therefore both off; the package's 48/72 h gain cannot "
                             "be attributed to them from this round"),
                "seeds": {}}
    for seed in wide.SEEDS:
        entry = {"seed": seed, "arms": {}}
        for name, feedback in ARMS:
            entry["arms"][name] = _arm_readings(output, body, seed, name, feedback, climatology)
        entry["delta_candidate_minus_control"] = {
            str(lead): (entry["arms"][CANDIDATE]["interior_t2m_rmse"][str(lead)]
                        - entry["arms"][CONTROL]["interior_t2m_rmse"][str(lead)])
            for lead in wide.EVALUATION_LEADS}
        readings["seeds"][str(seed)] = entry
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
    assigned = devices.assigned_uuid(body)
    recipe._shared().pin_declared_gpu(assigned)
    if phase == "reading":
        wide.write_exclusive(output / "readings.json", _readings(output, body))
        return
    if seed not in wide.SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
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
        phases = [("prepare", None), ("archive", None)] + [("seed", s) for s in wide.SEEDS] \
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
                for name, _ in ARMS:
                    receipt = json.loads(
                        (output / f"seed{seed}_{name}_receipt.json").read_text(encoding="utf-8"))
                    receipts[f"{seed}_{name}"] = receipt
                    peak = max(peak, receipt["owned_cuda_reserved_peak_bytes"])
            print(json.dumps({"phase_done": phase, "seed": seed,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
        elapsed = time.perf_counter() - started
        if elapsed >= deadline:
            raise RuntimeError("round hard deadline passed before publication")
        result = {"scientific_claim": False, "test_read": False, "protocol_sha256": body["protocol_sha256"],
                  "supervision": body["supervision"], "updates": UPDATES, "dim": DIM,
                  "reasoning_steps": STEPS, "train_data_identity": body["train_data_identity"],
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
