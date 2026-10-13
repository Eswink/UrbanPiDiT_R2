"""H-A process-state input conditioning: per-sample ``P_0`` on the registered 2400-update arm.

Decision 0048 closed the capacity axis permanently. This driver takes the first architecture factor
from ``docs/plans/0022`` §3, **H-A**: condition the process-state initial value ``P_0`` on each
sample's own context instead of the batch-shared ``process_queries``. It flips exactly one model-spec
field, ``process_init_conditioning`` False -> True, holding the wide 129x129 input, the
``interior_32`` mask, 2400 updates, K=4, depth 4, ``dim`` 192, the physical weights, the migrated
v3-BD 1600 parent, the seeds, the scoring region and the climatology denominator fixed.

The pathway is built last under ``isolated_stream()``, so the "off" path is bit-identical to the
previous implementation and switching it on moves no pre-existing weight (asserted per seed). The
parent loads unchanged except for the four new ``process_init_proj`` keys, and zeroing
``process_init_proj.1`` makes the injection identically zero (the forward pass falls back to the
registered arm bit for bit). Report-only against the pinned dose and narrow arms; test never read.
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

FORMAT = "r7-s3-process-init-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
NEW_KEYS = ("process_init_proj.0.weight", "process_init_proj.0.bias",
            "process_init_proj.1.weight", "process_init_proj.1.bias")
NEW_PARAMS = 37440              # LayerNorm(192) 384 + Linear(192,192) 37056
SEEDS = wide.SEEDS
DEVICE_MODE = "dual"
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 28800., 86400., 115200.
# Footprint proxy: the trunk is unchanged (dim 192, depth 4) and H-A only adds one LayerNorm+Linear on
# the pooled context. The width round MEASURED 11.9023 GiB per seed, so the estimate is >= 11.90 GiB.
ESTIMATED_PEAK_BYTES = 12198 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
EQUIVALENCE_SEED = 1234
DEFAULT_OUT = ROOT / "outputs/r7_s3_process_init_20261014_attempt01"
PREVIOUS_DOSE_ARM_RUN = ROOT / "outputs/r7_s3_wide_interior_dose_20261009_attempt02"
EVIDENCE_INDEX = ROOT / "docs/R7_EVIDENCE_INDEX.jsonl"
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "model/process_forecast_r7.py", "scripts/study_r7_s3_process_init.py")

DESIGN_DECISION_TEXT = (
    "Single factor: the process-state input conditioning. The registered wide interior-dose arm read 48 h "
    "positive but 72 h negative on all three seeds and no lever tried since turned it positive, so this arm "
    "flips one model-spec field - process_init_conditioning False -> True - holding all else fixed; built last "
    "under isolated_stream(), the off path is identical.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of this arm's "
    "interior_32 t2m RMSE minus the registered 2400-update dose arm's interior_32 t2m RMSE (index record "
    "s3-wide-interior-dose, pinned, not retrained here), (b) the same delta against the pinned registered "
    "narrow arm (index record s3-rollout-dose, 800 updates), and (c) this arm's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2, at the five leads {6,12,24,48,72} h. Both references are pinned.")
HYPOTHESIS_TEXT = (
    "If the residual 72 h error of the registered 2400-update arm is dominated by the reasoning start "
    "carrying no input information - P_0 identical for every sample - rather than by the training recipe, "
    "depth or capacity, then conditioning P_0 on each sample's own context at every other fixed factor "
    "should push long-lead (48/72 h) t2m skill toward or past the absolute climatology gate; if 72 h stays "
    "negative on all three seeds, the H-A branch is closed.")
DIFFERENCE_TEXT = (
    "One factor only: the process-state input conditioning, process_init_conditioning False -> True, applied "
    "to training and scoring through the bound model spec. P_0 becomes process_queries.expand(B,-1,-1) + "
    "process_init_proj(context.mean(dim=1)).unsqueeze(1), with process_init_proj = Sequential(LayerNorm(dim), "
    "Linear(dim,dim)) built last under isolated_stream(). Same wide 129x129 input, same interior_32 mask "
    "(mask_sha256 4a2a576b...), same 2400 updates, same K=4, same depth 4, same dim 192, same registered "
    "physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), same migrated v3-BD 1600 parent (dim 192), same LR 2e-5, "
    "warmup 10, FP32, batch 1 and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "both references are pinned registered arms (2400-update interior s3-wide-interior-dose, narrow "
    "s3-rollout-dose), not retrained inside this protocol",
    "the pathway is trained jointly with the registered parent's weights, so the arm prices 'a per-sample "
    "P_0 route from the parent', not the pathway in isolation; matched_generic is declared, not trained",
    "interior_32 supervision leaves the outer ring unconstrained during training; only interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident; validation "
    "split only and the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))


def capacity_spec():
    """The archived process spec with only ``process_init_conditioning`` switched on."""
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != DIM:
        raise RuntimeError(f"the archived process spec is not dim {DIM}")
    if spec.get("process_init_conditioning", False):
        raise RuntimeError("the archived process spec already conditions the process state")
    spec["process_init_conditioning"] = True
    return spec


def process_init_block():
    """The frozen H-A process-state input-conditioning rule and its declared controls."""
    return {"conditioning": ("P_0 = process_queries.expand(B,-1,-1) + "
                             "process_init_proj(context.mean(dim=1)).unsqueeze(1)"),
            "new_keys": list(NEW_KEYS), "new_params": NEW_PARAMS,
            "built_last_under_isolated_stream": True,
            "bit_identical_off_path": True,
            "positive_control": ("zeroing process_init_proj.1 (the Linear) makes the injected term "
                                 "identically zero, so the forward pass is bit-identical to the registered "
                                 "2400-update dose arm"),
            "negative_control": ("matched_generic: a same-parameter/same-FLOPs route that broadcasts the "
                                 "pooled context to every process slot instead of a per-sample term, "
                                 "separating information gain from added parameters; declared, not trained")}


def protocol_constants():
    """The dose round's frozen fields with the single factor process_init_conditioning swapped in."""
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
    """Pin the registered 2400-update interior-dose arm this round conditions P_0 against."""
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


def _positive_control(spec):
    """Same-seed off/on bit-identity plus the zeroing identity, measured on the bound spec.

    The pathway is built last under ``isolated_stream()`` and consumes no RNG, so the off and on
    trunks built under the same seed are ``torch.equal`` on every shared tensor and differ only by
    the four ``process_init_proj`` keys; zeroing ``process_init_proj.1`` must then zero the
    projection output. Any failure aborts the run.
    """
    import torch
    from training.r7_experiment import make_model
    off_spec = dict(spec, process_init_conditioning=False)
    on_spec = dict(spec, process_init_conditioning=True)
    torch.manual_seed(EQUIVALENCE_SEED)
    off_state = make_model("process", off_spec).state_dict()
    torch.manual_seed(EQUIVALENCE_SEED)
    on_model = make_model("process", on_spec)
    on_state = on_model.state_dict()
    new_keys = sorted(set(on_state) - set(off_state))
    shared = [key for key in off_state if key in on_state]
    bit_identical = (set(new_keys) == set(NEW_KEYS) and not (set(off_state) - set(on_state))
                     and all(torch.equal(off_state[key], on_state[key]) for key in shared))
    with torch.no_grad():
        on_model.process_init_proj[1].weight.zero_()
        on_model.process_init_proj[1].bias.zero_()
        context = torch.randn(3, 5, int(spec["dim"]))
        pooled = on_model.process_init_proj(context.mean(dim=1))
    proj_zeroing_is_identity = bool(torch.count_nonzero(pooled) == 0)
    if not bit_identical or not proj_zeroing_is_identity:
        raise RuntimeError("H-A positive control failed: the off path is not bit-identical or the "
                           "zeroed projection is not identically zero")
    return {"seed": EQUIVALENCE_SEED, "bit_identical_off_path": bit_identical, "new_keys": new_keys,
            "proj_zeroing_is_identity": proj_zeroing_is_identity,
            "zeroed_proj_output_abs_max": float(pooled.abs().max())}


def expand_parent_state_process_init(parent_state, spec):
    """Load a dim-192 parent into a dim-192 trunk that adds the H-A conditioning pathway.

    Returns ``(model, state, receipt)``. The target state differs from the parent by exactly the four
    ``process_init_proj`` keys, which keep the target's random init; those are the only new keys and
    the "off" path is bit-identical to the registered arm. The receipt records the raw
    positive-control output.
    """
    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    target = make_model("process", spec)
    target_state = target.state_dict()
    missing = [key for key in target_state if key not in parent_state]
    if set(missing) != set(NEW_KEYS):
        raise RuntimeError(f"the H-A pathway must add exactly {sorted(NEW_KEYS)}, "
                           f"got {sorted(missing)}")
    state = dict(parent_state)
    for key in missing:
        state[key] = target_state[key].detach().clone()
    target.load_state_dict(state, strict=True)
    control = _positive_control(spec)
    receipt = {"parent_keys": len(parent_state), "new_keys": sorted(missing),
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": _state_digest(state),
               "new_params": sum(int(target_state[key].numel()) for key in missing),
               "proj_zeroing_is_identity": control["proj_zeroing_is_identity"],
               "bit_identical_off_path": control["bit_identical_off_path"],
               "positive_control": control}
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
    body["process_init"] = process_init_block()
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
    if body["process_init"] != process_init_block():
        raise RuntimeError("the frozen process-init conditioning rule changed")
    if body["arms"]["candidate"] != _expected_candidate() or not devices.assignment_is_valid(
            body, wide.GPU_UUID):
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")
    spec = body["arms"]["candidate"]["model"]["spec"]
    if (int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH
            or spec.get("process_init_conditioning") is not True):
        raise RuntimeError("the candidate is not dim 192 depth 4 with process_init_conditioning on")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    if (int(body["estimated_peak_bytes"]) < MIN_ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("the H-A peak estimate was not recalibrated")
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
        raise RuntimeError("the candidate spec is not the frozen dim-192 depth-4 trunk")
    model, state, expansion = expand_parent_state_process_init(saved["model"], spec)
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
                                                   "then H-A process_init_conditioning=True (4 new keys)")},
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
               "dim": DIM, "process_init": expansion, "evaluations": {}}
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
                "updates": UPDATES, "dim": DIM, "process_init_conditioning": True,
                "confound": ("the conditioning pathway is trained jointly with the registered parent's "
                             "weights; the arm prices 'a per-sample P_0 route from the registered parent', "
                             "not the pathway in isolation"),
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
