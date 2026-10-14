"""H-C global context mixing: one post-backbone mixing layer on the registered 2400-update arm.

H-A was closed by its predeclared exit (72 h negative on all three seeds), H-B by its own exit
(B->C null), and decision 0048 closed the capacity axis, so this driver takes the third architecture
factor in ``docs/plans/0022`` §4: the trunk mixes context tokens only inside ``window_size`` windows
and carries no N x N attention anywhere, so 72 h error -- which is large-scale -- may be unreachable
for a purely local mixing range. This round adds exactly one context-mixing layer after the backbone
and trains two arms inside the one protocol: ``global`` attends over every token, ``local`` keeps the
trunk's windowed attention. Both arms declare the same tensors with the same shapes, so they are
parameter-matched by construction (444,864 each at dim 192); their compute is *not* matched and the
gap is declared and reported rather than claimed away.

Every other factor is held fixed: the wide 129x129 input, the interior_32 mask, 2400 updates, K=4,
depth 4, dim 192, the physical weights, the migrated v3-BD 1600 parent, the seeds, the scoring region
and the climatology denominator. Because this driver edits ``model/*.py`` the migrated parent is
pinned by ``r7_s3_global_mixing_parent`` under the current digest. Report-only.
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

from scripts import r7_s3_global_mixing_parent as mixing_parent
from scripts import study_r7_s3_reasoning_depth as depth_arm
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from scripts.study_r7_s3_v3_rollout_isolated import fresh_output
from model.global_mixing_r7 import GLOBAL, LOCAL
from training import r7_device_selection as devices
from training.r7_study_process import bounded_process, worker_environment

interior = dose.interior

FORMAT = "r7-s3-global-mixing-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
DIM = 192                       # trunk width unchanged (decision 0048: no capacity axis)
DEPTH = 4                       # trunk depth unchanged
ARMS = ((GLOBAL, GLOBAL), (LOCAL, LOCAL))
SEEDS = wide.SEEDS
DEVICE_MODE = "dual"
# Two arms per seed, each 2400 updates on the same trunk shape as H-B, which MEASURED 4.33-5.77
# GPU-h per seed (15.443038 GPU-h for its six stages). The envelope is the H-B one, recalibrated
# against that measurement rather than carried over unexamined.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 43200., 86400., 115200.
# H-B measured 9.276 GiB (spatial) / 9.041 GiB (pooled) peak. One extra layer over 1089 tokens at
# dim 192 adds an N x N score matrix of 4 heads x 1089^2 x 4 bytes ~= 19 MB, so the envelope holds.
ESTIMATED_PEAK_BYTES = 9728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
DEFAULT_OUT = ROOT / "outputs/r7_s3_global_mixing_20261015_attempt01"
PREVIOUS_DOSE_ARM_RUN = ROOT / "outputs/r7_s3_wide_interior_dose_20261009_attempt02"
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "model/global_mixing_r7.py",
               "scripts/study_r7_s3_global_mixing.py", "scripts/r7_s3_global_mixing_parent.py")

DESIGN_DECISION_TEXT = (
    "Single factor: the spatial range over which the context tokens are mixed, at fixed parameters. H-A "
    "was closed by its predeclared exit and H-B by its own (B->C null); decision 0048 closed the capacity "
    "axis. This arm therefore takes docs/plans/0022's third factor: one post-backbone context-mixing layer, "
    "global (full attention) versus local (the trunk's windowed attention), both parameter-matched at "
    "444,864 so the contrast is the mixing range and not the added capacity.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of the global arm's "
    "interior_32 t2m RMSE minus this protocol's own local arm's (the range attribution), (b) each arm's "
    "delta against the registered 2400-update dose arm's interior_32 t2m RMSE (index record "
    "s3-wide-interior-dose, pinned, not retrained here), and (c) each arm's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2, at the five leads {6,12,24,48,72} h. The external reference is pinned.")
HYPOTHESIS_TEXT = (
    "If the long-lead (48/72 h) floor comes from the trunk's local mixing range -- window_size 4 at "
    "patch_size 2 covers about 2.0 degrees of a 65x65 domain and there is no N x N attention anywhere -- "
    "then one full-attention mixing layer after the backbone should move 48/72 h interior_32 t2m skill "
    "relative to a parameter-matched extra windowed layer trained in this same protocol. If L->G is null, "
    "the conclusion is that the bottleneck is not the receptive field (and is not separable from the added "
    "capacity); if both are null, it is neither.")
DIFFERENCE_TEXT = (
    "One factor only: the attention range of one added post-backbone context-mixing layer, "
    "global_mixing_mode None -> 'global' (full N x N attention) with the in-protocol control 'local' "
    "(WindowAttentionBlock at the trunk's window_size), applied to training and scoring through the bound "
    "model spec. Both modes declare the same twelve tensors with the same shapes -- LayerNorm, qkv, proj, "
    "LayerNorm, FeedForward(mlp_ratio 4) -- so both add exactly 444864 parameters at dim 192; parameter "
    "counts are aligned by construction. Compute is NOT aligned and is declared: at N=1089, d=192, heads=4, "
    "window=4 the attention length per query is 1089 vs 16 (68.1x), the SDPA sub-cost 9.1079e8 vs 1.5925e7 "
    "(57.2x), and the whole layer 1.8751e9 vs 9.8024e8 (1.91x, the FF dominates). A win for the global arm "
    "may therefore come from compute rather than from globalness; that confound is a limitation of this "
    "round. Same wide 129x129 input, same interior_32 mask (mask_sha256 4a2a576b...), same 2400 updates, "
    "same K=4, same depth 4, same dim 192, same registered physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), "
    "same migrated v3-BD 1600 parent (dim 192), same LR 2e-5, warmup 10, FP32, batch 1 and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "the two arms are parameter-matched but NOT compute-matched: the global arm's attention costs 57.2x "
    "the local arm's SDPA sub-cost (68.1x attention length, 1.91x whole layer), so a global-arm win is "
    "not separable from the extra compute",
    "the external reference is a pinned registered arm (2400-update interior s3-wide-interior-dose), not "
    "retrained inside this protocol; the local control IS retrained here, so L->G is in-protocol",
    "both arms are trained jointly with the registered parent's weights, so each prices a route from the "
    "parent rather than the layer in isolation",
    "interior_32 supervision leaves the outer ring unconstrained during training; only interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident; validation "
    "split only and the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))


def arm_spec(mode):
    """The archived process spec with only the context-mixing mode switched on."""
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError(f"the archived process spec is not dim {DIM} depth {DEPTH}")
    if spec.get("global_mixing_mode") or spec.get("process_init_conditioning") \
            or spec.get("typed_evidence_mode"):
        raise RuntimeError("the archived process spec already carries an architecture switch")
    spec["global_mixing_mode"] = mode
    return spec


def mixing_block(mode):
    """The frozen H-C mixing rule and its declared controls, for one arm."""
    return {"mode": mode,
            "range": ("global: full N x N attention over all context tokens; local: the trunk's "
                      "windowed attention at window_size 4"),
            "new_keys": list(mixing_parent.mixing_keys(mode)),
            "new_params": 444864,
            "parameter_matched_by_construction": True,
            "compute_matched": False,
            "declared_compute_gap": {"attention_length_ratio": 68.0625, "attention_ratio": 57.19140625,
                                     "layer_ratio": 1.9129018382845497,
                                     "reference": dict(mixing_parent.FLOPS_REFERENCE)},
            "built_last_under_isolated_stream": True,
            "identity_control": ("zeroing the output projection AND the FF output layer makes the layer "
                                 "the exact identity, so the whole forward returns to the off path bit "
                                 "for bit; zeroing the projection alone does not"),
            "shared_initialization": ("both arms are expanded under torch.manual_seed(seed), so their "
                                      "twelve added tensors are bit-identical within a round seed and "
                                      "their pathway_init_sha256 must match"),
            "positive_control": ("a failed control raises instead of being recorded and ignored")}


def protocol_constants():
    """The dose round's frozen fields with the context-mixing factor swapped in."""
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
    """Pin the registered 2400-update interior-dose arm this round compares against."""
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


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest
    body = interior._identity()
    body["execution_files_sha256"] = {name: sha256_file(ROOT / name) for name in execution_files()}
    template = dict(body["arms"].pop("candidate"))
    for name, mode in ARMS:
        spec = arm_spec(mode)
        body["arms"][name] = dict(template, updates=UPDATES, mode=mode,
                                  model={"kind": "process", "spec": spec,
                                         "spec_canonical_digest": canonical_digest(spec)})
    body["arms"]["previous_dose_arm"] = previous_arm_pins()
    mixing_parent.install_parent_pins(body)
    body["global_mixing"] = {"arms": {name: mixing_block(mode) for name, mode in ARMS}}
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
    if body["global_mixing"] != {"arms": {name: mixing_block(mode) for name, mode in ARMS}}:
        raise RuntimeError("the frozen context-mixing rule changed")
    if not devices.assignment_is_valid(body, wide.GPU_UUID):
        raise RuntimeError("declared GPU differs from the frozen assignment")
    for name, mode in ARMS:
        arm = body["arms"][name]
        spec = arm["model"]["spec"]
        if (arm["mode"] != mode or int(arm["updates"]) != UPDATES
                or int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH
                or spec.get("global_mixing_mode") != mode
                or arm["model"]["spec_canonical_digest"] != canonical_digest(spec)):
            raise RuntimeError(f"the {name} arm differs from the frozen recipe")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    if (int(body["estimated_peak_bytes"]) < MIN_ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("the H-C peak estimate was not recalibrated")
    previous = body["arms"]["previous_dose_arm"]
    if (previous["record_id"] != PREVIOUS_DOSE_ARM_RECORD or previous["updates"] != UPDATES
            or previous["seeds"] != list(wide.SEEDS)):
        raise RuntimeError("the pinned 2400-update dose arm differs from the registered record")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["test_manifest_never_read"] != str(wide.TEST_MANIFEST)):
        raise RuntimeError("frozen claims/test-read policy changed")
    return body


def arm_name(mode):
    """The protocol arm name for a mixing mode."""
    for name, value in ARMS:
        if value == mode:
            return name
    raise ValueError(f"unknown context-mixing mode {mode}")


def _parent_model(body, seed, mode):
    from training.r7_experiment import load_checkpoint
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    path, expected = mixing_parent.pinned_parent(seed)
    if (sha256_file(path) != expected
            or body["arms"]["parent"]["pins"]["checkpoints"][str(seed)]["sha256"] != expected):
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != 1600 or contract["seed"] != seed or contract["mode"] != "l6"
            or contract["data_identity"] != shared.V3_DATA_IDENTITY
            or contract["source_sha256"] != shared.V3_SOURCE_SHA256):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"][arm_name(mode)]["model"]["spec"]
    if int(spec.get("dim", 0)) != DIM or int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError("the arm spec is not the frozen dim-192 depth-4 trunk")
    model, state, expansion = mixing_parent.expand_parent_state(saved["model"], spec, mode, seed=seed)
    return model, state, path, expected, expansion


def _arm_contract(body, spec, mode, parent, parent_sha, expansion):
    return {"kind": "process", "model": spec, "data_identity": body["train_data_identity"],
            "source_sha256": body["source_sha256"], "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(mixing_parent.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": 1600,
                               "description": ("registered v3-BD l6 endpoint, model-only migration, "
                                               f"then H-C context mixing mode={mode} "
                                               f"({len(expansion['new_keys'])} new keys)")},
            "autoregression": {"physical_steps": 12, "physical_weights": list(wide.PHYSICAL_WEIGHTS),
                "window_sha256": body["train_windows"]["window_sha256"],
                "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"],
                "supervision": body["supervision"]["runner_block"]}}


def _train_arm(output, body, seed, name, mode, deadline, mask):
    """Fine-tune one arm from the migrated parent and score its five leads."""
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    spec = body["arms"][name]["model"]["spec"]
    model, state, parent, parent_sha, expansion = _parent_model(body, seed, mode)
    contract = _arm_contract(body, spec, mode, parent, parent_sha, expansion)
    folder = output / f"seed{seed}/training/{name}"
    checkpoint, report = fine_tune_long_rollout(wide.TRAIN_MANIFEST, folder, model=model,
        contract=contract, parent_weights=state, physical_weights=wide.PHYSICAL_WEIGHTS, steps=STEPS,
        updates=UPDATES, lr=wide.LR, warmup=wide.WARMUP, seed=seed, device_name="cuda:0",
        deadline=deadline, supervision_mask=mask)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["train_data_identity"]
            or report["supervision"] != body["supervision"]["runner_block"]):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    receipt = {"seed": seed, "arm": name, "mode": mode, "training_dir": str(folder),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
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
    for name, mode in ARMS:
        receipt = _train_arm(output, body, seed, name, mode, deadline, mask)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
    # The L->G contrast is attributable to the mixing range only if the two arms started from one
    # initialization; a mismatch here means the delta carries a fresh random draw too.
    digests = {name: receipts[name]["expansion"]["pathway_init_sha256"] for name in receipts}
    if len(set(digests.values())) != 1:
        raise RuntimeError(f"seed {seed}: the arms do not share one mixing initialization: {digests}")


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


def _arm_readings(output, body, seed, name, mode, climatology):
    receipt = json.loads((output / f"seed{seed}_{name}_receipt.json").read_text(encoding="utf-8"))
    arm = {"mode": mode, "interior_t2m_rmse": {}, "previous_dose_t2m_rmse": {},
           "delta_minus_previous_dose": {}, "interior_t2m_skill": {}}
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
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": body["decision_rule"],
                "updates": UPDATES, "dim": DIM, "reasoning_steps": REASONING_STEPS,
                "factor": "context-mixing range: full attention vs the trunk's windowed attention",
                "declared_difference": ("both arms add 444864 parameters (parameter-matched by "
                                        "construction); compute is NOT matched -- 68.1x attention length, "
                                        "57.2x SDPA sub-cost, 1.91x whole layer"),
                "confound": ("the global arm's extra compute is not separable from its globalness, and "
                             "both arms are trained jointly with the registered parent's weights"),
                "seeds": {}}
    for seed in wide.SEEDS:
        entry = {"seed": seed, "arms": {}}
        for name, mode in ARMS:
            entry["arms"][name] = _arm_readings(output, body, seed, name, mode, climatology)
        entry["delta_global_minus_local"] = {
            str(lead): (entry["arms"][GLOBAL]["interior_t2m_rmse"][str(lead)]
                        - entry["arms"][LOCAL]["interior_t2m_rmse"][str(lead)])
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
