"""Capacity-width arm: raise the trunk ``dim`` 192 -> 256 on the registered 2400-update arm.

The capacity-depth arm (``scripts/study_r7_s3_capacity_depth.py``) raised the trunk
``depth`` 4 -> 8 on the registered wide interior-dose arm and left 72 h t2m skill negative
on all three seeds; decision 0048 closed the depth axis and capped the width axis as the
last capacity arm. This driver returns to the main line with the one remaining capacity
factor: on the registered wide interior-dose arm it changes exactly one model-spec field,
``dim`` 192 -> 256 (the trunk width of every ``WindowAttentionBlock`` in ``backbone.encoder``),
and holds the wide 129x129 input, the ``interior_32`` supervision mask, the 2400-update
budget, K=4, the registered physical weights, the migrated v3-BD 1600 parent, the seeds, the
evaluation region and the climatology denominator fixed.

Unlike the depth axis there is **no bit-identical width expansion**: at dim 192 -> 256, 150
of the 155 state_dict tensors change shape and the dim-192 parent cannot load into a dim-256
trunk. The expansion rule is frozen here as a deterministic zero-pad: each parent tensor is
copied into the target tensor's overlapping slice and every other position is zeroed, so the
dim-256 trunk starts from the registered parent under a frozen initialization. LayerNorm
normalizes over all 256 channels and the zeroed high 64 are not strictly inert, so the arm
prices the joint effect of the extra width and this initialization, not width alone. A fixed
synthetic batch is checked for finiteness per seed (no bit-identical claim); its digests are
recorded.

It reports against the pinned 2400-update dose arm (primary, index ``s3-wide-interior-dose``)
and the pinned registered narrow arm (secondary); neither is retrained. The 2023 test split
is never read.
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

from scripts import study_r7_s3_reasoning_depth as depth_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from training import r7_device_selection as devices
from training.r7_study_process import bounded_process, worker_environment

interior = dose.interior

FORMAT = "r7-s3-capacity-width-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
PARENT_DIM = 192
DIM = 256
DEPTH = 4                       # trunk depth unchanged (decision 0048: only dim moves)
SEEDS = wide.SEEDS
# Decision 0047: one seed per card, both cards used by default. ``single`` keeps the old
# one-card behaviour and is what every other (registered) arm still freezes.
DEVICE_MODE = "dual"
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 28800., 86400., 115200.
# Footprint proxy: the dose arm runs the same K=4 and 2400 updates; the dim-256 trunk is wider
# (5,790,389 params, +76%) but no deeper. The depth round estimated 8.5 GiB and measured 11.03 GiB
# per seed, so start from that MEASURED peak (decision 0048/§8 mandate >= 11.03 GiB); the round
# re-calibrates after the first seed's owned reserved peak. A judgement, not a guarantee.
ESTIMATED_PEAK_BYTES = 11300 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
EQUIVALENCE_SHAPE = (1, 2, 17, 129, 129)
EQUIVALENCE_SEED = 1234
DEFAULT_OUT = ROOT / "outputs/r7_s3_capacity_width_20261013_attempt01"
PREVIOUS_DOSE_ARM_RUN = ROOT / "outputs/r7_s3_wide_interior_dose_20261009_attempt02"
EVIDENCE_INDEX = ROOT / "docs/R7_EVIDENCE_INDEX.jsonl"
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "scripts/study_r7_s3_capacity_width.py")

DESIGN_DECISION_TEXT = (
    "Single factor: the trunk width. The registered wide interior-dose arm (wide 129x129 input, "
    "interior_32 supervision, 2400 updates, K=4, registered physical weights, migrated v3-BD 1600 "
    "parent) read 48 h positive on all three seeds but 72 h negative on all three. Every training-recipe "
    "lever, the reasoning depth K and the backbone depth were tried and none turned 72 h positive, so "
    "this arm changes one model-spec field - the trunk dim 192 -> 256 - and holds every other factor "
    "fixed. Unlike the depth arm, width has no bit-identical expansion: the dim-192 parent cannot load "
    "into a dim-256 trunk, so the parent is migrated by a frozen zero-pad rule and the width and that "
    "initialization are not separated.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of this arm's "
    "interior_32 t2m RMSE minus the registered 2400-update dose arm's interior_32 t2m RMSE (index record "
    "s3-wide-interior-dose, pinned, not retrained here), (b) the same delta against the pinned registered "
    "narrow arm (index record s3-rollout-dose, 800 updates), and (c) this arm's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2, at the five leads {6,12,24,48,72} h. Both references are pinned, not "
    "retrained here.")
HYPOTHESIS_TEXT = (
    "If the residual 72 h error of the registered 2400-update arm is dominated by the trunk's per-step "
    "representational capacity rather than by the training recipe or the inference depth, then widening "
    "the backbone dim 192 -> 256 at fixed depth 4, fixed input region, fixed interior_32 supervision, "
    "fixed 2400 updates, fixed K=4, fixed physical weights, fixed parent and fixed seeds should push 72 h "
    "skill toward or past the absolute climatology gate. If 72 h stays negative on all three seeds, the "
    "whole capacity axis (depth and width) is closed per decision 0048.")
DIFFERENCE_TEXT = (
    "One factor only: the trunk dim 192 -> 256 (backbone.encoder WindowAttentionBlock width), applied to "
    "both training and scoring through the bound model spec. Same depth 4, same wide 129x129 input, same "
    "interior_32 supervision mask (mask_sha256 4a2a576b...), same 2400 updates, same K=4 "
    "(steps/reasoning_steps), same registered physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), same "
    "migrated v3-BD 1600 parent (dim-192, zero-pad migrated), same LR 2e-5, warmup 10, FP32, batch 1 and "
    "clip 1. The width and the frozen zero-pad initialization are confounded (no bit-identical width "
    "expansion exists).")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "both references are pinned registered arms (2400-update interior s3-wide-interior-dose, narrow "
    "s3-rollout-dose), not retrained inside this protocol",
    "no bit-identical width expansion exists (150/155 tensors change shape), so width and the zero-pad "
    "initialization are not separated - the arm prices 'more width from the registered parent by a frozen "
    "zero-pad', not 'more width from scratch'",
    "restricting supervision to interior_32 leaves the outer ring unconstrained during training; only "
    "interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident",
    "validation split only; the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))


def capacity_spec():
    """The archived process spec with only the trunk width raised to 256."""
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != PARENT_DIM:
        raise RuntimeError(f"the archived process spec is not dim {PARENT_DIM}")
    spec["dim"] = DIM
    return spec


def width_expansion_block():
    """The frozen zero-pad width-expansion rule and the confound it carries."""
    return {"parent_dim": PARENT_DIM, "target_dim": DIM,
            "rule": "each parent tensor is copied into the target tensor's overlapping slice "
                    "(per-dimension slice(0, min(src, tgt))) and every other position is zeroed",
            "pad_rule": "1-D parameters keep the first 192 entries and the remaining 64 are zeroed",
            "bit_identical": False,
            "confound": ("width and the frozen zero-pad initialization are not separated: LayerNorm "
                         "normalizes over all 256 channels and the zeroed high 64 are not strictly inert, "
                         "so the arm measures the joint effect of the extra width and this initialization"),
            "shape_selfcheck": {"module": "backbone.encoder", "dtype": "float32",
                                "batch_shape": list(EQUIVALENCE_SHAPE), "seed": EQUIVALENCE_SEED,
                                "claim": "encoder output is finite (no bit-identical claim)"}}


def protocol_constants():
    """The dose round's frozen fields with the single factor dim swapped in."""
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
    """Pin the registered 2400-update interior-dose arm this round widens against."""
    from training.r7_arm_harness import sha256_file
    record = _index_record(PREVIOUS_DOSE_ARM_RECORD)
    doc = ROOT / record["evidence_path"]
    if sha256_file(doc) != record["evidence_sha256"]:
        raise RuntimeError("the registered 2400-update dose arm doc drifted from its index record")
    run = PREVIOUS_DOSE_ARM_RUN
    if not run.is_dir():
        raise FileNotFoundError(f"the registered 2400-update dose arm output is missing: {run}")
    pins = {"record_id": record["record_id"], "evidence_path": record["evidence_path"],
            "evidence_sha256": record["evidence_sha256"], "protocol_sha256": record["protocol_sha256"],
            "data_identity": record["data_identity"], "source_sha256": record["source_sha256"],
            "updates": UPDATES, "seeds": list(wide.SEEDS), "run_dir": str(run), "seeds_evaluations": {}}
    for seed in wide.SEEDS:
        receipt = json.loads((run / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {"checkpoint_sha256": receipt["checkpoint_sha256"], "leads": {}}
        for lead in wide.EVALUATION_LEADS:
            record_lead = receipt["evaluations"][str(lead)]
            if record_lead["n_evaluated"] != wide.WIDE_VAL_COHORTS[str(lead)]:
                raise RuntimeError(f"registered 2400-dose arm seed {seed} lead {lead} cohort differs")
            # This arm is scored on interior_32, so its pinned table is the boundary
            # table; storing rmse.csv's hash here would never match at reading time.
            entry["leads"][str(lead)] = {
                "boundary_rmse_csv_sha256": record_lead["boundary_rmse_csv_sha256"],
                "dir": record_lead["dir"]}
        pins["seeds_evaluations"][str(seed)] = entry
    return pins


def expand_parent_state_width(parent_state, spec):
    """Build the dim-256 trunk from a dim-192 parent by the frozen zero-pad expansion.

    Returns ``(model, expanded_state, receipt)``. Each parent tensor is copied into the
    target tensor's overlapping slice (per-dimension ``slice(0, min(src, tgt))``) and every
    other position is zeroed; keys the parent lacks keep the target's random init. There is
    **no** bit-identical width expansion, so no equivalence claim is made: the expansion is
    deterministic given the parent, but LayerNorm normalizes over all 256 channels and the
    zeroed high 64 are not strictly inert, so the arm prices the joint effect of the extra
    width and this frozen initialization. A fixed synthetic batch only checks finiteness.
    """
    import torch
    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    target = make_model("process", spec)
    shapes = target.state_dict()
    state, left_at_random = {}, []
    for key, tgt in shapes.items():
        if key not in parent_state:
            state[key] = tgt.detach().clone()
            left_at_random.append(key)
            continue
        src = parent_state[key]
        new = torch.zeros_like(tgt)
        window = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt.shape))
        new[window] = src[window]
        state[key] = new
    target.load_state_dict(state, strict=True)
    torch.manual_seed(EQUIVALENCE_SEED)
    probe = torch.randn(*EQUIVALENCE_SHAPE)
    with torch.no_grad():
        tokens, _ = target.backbone.encoder(probe)
    if not bool(torch.isfinite(tokens).all()):
        raise RuntimeError("the width-expanded trunk produced a non-finite encoder output")
    parent_dim = int(parent_state["backbone.encoder.norm.weight"].shape[0])
    target_dim = int(shapes["backbone.encoder.norm.weight"].shape[0])
    non_block = [key for key in left_at_random if ".blocks." not in key]
    receipt = {"parent_dim": parent_dim, "target_dim": target_dim,
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": _state_digest(state),
               "non_block_keys_unchanged": not non_block, "left_at_random": sorted(left_at_random),
               "batch_shape": list(EQUIVALENCE_SHAPE), "seed": EQUIVALENCE_SEED,
               "encoder_output_finite": True}
    return target, state, receipt


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = interior._identity()
    spec = capacity_spec()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    body["arms"]["candidate"]["updates"] = UPDATES
    body["arms"]["candidate"]["model"] = {"kind": "process", "spec": spec,
                                          "spec_canonical_digest": canonical_digest(spec)}
    body["arms"]["previous_dose_arm"] = previous_arm_pins()
    body["width_expansion"] = width_expansion_block()
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


def _expected_candidate():
    from training.r7_experiment import canonical_digest
    spec = capacity_spec()
    return {"mode": "long_rollout", "updates": UPDATES, "lr": wide.LR, "warmup": wide.WARMUP,
            "physical_weights": list(wide.PHYSICAL_WEIGHTS), "steps": STEPS,
            "weight_decay": wide.WEIGHT_DECAY, "batch_size": wide.BATCH_SIZE, "clip": wide.CLIP,
            "checkpoint_every": wide.CHECKPOINT_EVERY, "bf16": wide.BF16,
            "selection": "frozen endpoint; no validation selection",
            "model": {"kind": "process", "spec": spec, "spec_canonical_digest": canonical_digest(spec)}}


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
        raise RuntimeError("the frozen capacity factor changed")
    if body["width_expansion"] != width_expansion_block():
        raise RuntimeError("the frozen width-expansion rule changed")
    if body["arms"]["candidate"] != _expected_candidate() or not devices.assignment_is_valid(
            body, wide.GPU_UUID):
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")
    spec = body["arms"]["candidate"]["model"]["spec"]
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError("the candidate trunk is not dim 256 depth 4")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    if (int(body["estimated_peak_bytes"]) < MIN_ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("the dim-256 peak estimate was not recalibrated")
    narrow = body["arms"]["narrow_control"]
    previous = body["arms"]["previous_dose_arm"]
    if (narrow["record_id"] != wide.NARROW_CONTROL_RECORD or narrow["updates"] != 800
            or narrow["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned narrow control differs from the registered record")
    if (previous["record_id"] != PREVIOUS_DOSE_ARM_RECORD or previous["updates"] != UPDATES
            or previous["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned 2400-update dose arm differs from the registered record")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(wide.TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")
    return body


def _parent_model(body, seed):
    from training.r7_experiment import load_checkpoint
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    path, expected = wide.dose.pinned_parent(seed)
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
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError("the candidate spec is not the frozen dim-256 depth-4 trunk")
    model, state, expansion = expand_parent_state_width(saved["model"], spec)
    return model, state, path, expected, expansion


def _seed(output, body, seed, deadline):
    import torch
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    mask = interior.supervision_mask().to("cuda:0")
    model, state, parent, parent_sha, expansion = _parent_model(body, seed)
    spec = body["arms"]["candidate"]["model"]["spec"]
    contract = {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
                "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
                "training_code_sha256": body["training_code_sha256"],
                "initialization": {"parent_run": str(wide.dose.MIGRATED_PARENT_RUN),
                                   "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                                   "parent_endpoint_updates": 1600,
                                   "description": ("registered v3-BD l6 endpoint, model-only migration, "
                                                   "then frozen zero-pad dim 192 -> 256 expansion")},
                "autoregression": {"physical_steps": 12, "physical_weights": list(wide.PHYSICAL_WEIGHTS),
                    "window_sha256": body["train_windows"]["window_sha256"],
                    "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                    "supervision": body["supervision"]["runner_block"]}}
    folder = output / f"seed{seed}/training/candidate"
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model, contract=contract,
        parent_weights=state, physical_weights=wide.PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES,
        lr=wide.LR, warmup=wide.WARMUP, seed=seed, device_name="cuda:0", deadline=deadline,
        supervision_mask=mask)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]):
        raise RuntimeError(f"seed {seed} did not reach the frozen dim-{DIM} K={STEPS} endpoint")
    receipt = {"seed": seed, "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "reasoning_steps": STEPS,
               "dim": DIM, "width_expansion": expansion, "evaluations": {}}
    for lead in wide.EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/candidate/lead_{lead:03d}h"
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
    receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
    wide.write_exclusive(output / f"seed{seed}_receipt.json", receipt)


def _reference_deltas(body, entry, lead, arm_rmse):
    """Paired deltas against both pinned references, verifying their pinned RMSE tables."""
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    narrow = body["arms"]["narrow_control"]
    previous = body["arms"]["previous_dose_arm"]
    narrow_lead = narrow["seeds_evaluations"][str(entry["seed"])]["leads"][str(lead)]
    narrow_dir = Path(narrow_lead["dir"])
    if sha256_file(narrow_dir / "rmse.csv") != narrow_lead["rmse_csv_sha256"]:
        raise RuntimeError("the registered narrow arm RMSE drifted from its freeze pin")
    previous_lead = previous["seeds_evaluations"][str(entry["seed"])]["leads"][str(lead)]
    previous_dir = Path(previous_lead["dir"])
    if sha256_file(previous_dir / "boundary_rmse.csv") != previous_lead["boundary_rmse_csv_sha256"]:
        raise RuntimeError("the registered 2400-update dose arm table drifted from its freeze pin")
    narrow_rmse = shared.read_rmse_table(narrow_dir / "rmse.csv")["t2m"]
    previous_rmse = wide._region_variable_rmse(previous_dir / "boundary_rmse.csv", "interior_32", "t2m")
    return narrow_rmse, arm_rmse - narrow_rmse, previous_rmse, arm_rmse - previous_rmse


def _readings(output, body):
    climatology = body["wide_climatology_reference"]
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
                "updates": UPDATES, "dim": DIM,
                "confound": ("width and the frozen zero-pad initialization are not separated "
                             "(no bit-identical width expansion); the arm measures their joint effect"),
                "reasoning_steps": REASONING_STEPS, "seeds": {}}
    for seed in wide.SEEDS:
        receipt = json.loads((output / f"seed{seed}_receipt.json").read_text(encoding="utf-8"))
        entry = {"seed": seed, "interior_t2m_rmse": {}, "narrow_full_t2m_rmse": {},
                 "delta_minus_narrow": {}, "previous_dose_t2m_rmse": {},
                 "delta_minus_previous_dose": {}, "interior_t2m_skill": {}}
        for lead in wide.EVALUATION_LEADS:
            arm_dir = Path(receipt["evaluations"][str(lead)]["dir"])
            arm_rmse = wide._region_variable_rmse(arm_dir / "boundary_rmse.csv", "interior_32", "t2m")
            narrow_rmse, delta_narrow, dose_rmse, delta_dose = _reference_deltas(body, entry, lead, arm_rmse)
            climatology_rmse = climatology["leads"][str(lead)]["interior_32_t2m_rmse"]
            entry["interior_t2m_rmse"][str(lead)] = arm_rmse
            entry["narrow_full_t2m_rmse"][str(lead)] = narrow_rmse
            entry["delta_minus_narrow"][str(lead)] = delta_narrow
            entry["previous_dose_t2m_rmse"][str(lead)] = dose_rmse
            entry["delta_minus_previous_dose"][str(lead)] = delta_dose
            entry["interior_t2m_skill"][str(lead)] = 1.0 - (arm_rmse / climatology_rmse) ** 2
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
