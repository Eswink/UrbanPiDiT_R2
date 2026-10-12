"""Capacity-depth arm: raise the trunk ``depth`` 4 -> 8 on the registered 2400-update arm.

The training-recipe branch for 72 h is closed: input region (index67), supervision
domain (index68), dose (index69), long-lead weights (index70) and the reasoning depth
K (index71, interrupted single seed) all failed to turn 72 h positive on three seeds.
This driver returns to the main line with a factor that is **not** part of the training
recipe: the trunk capacity. On the registered wide interior-dose arm it changes exactly
one model-spec field, ``depth`` 4 -> 8 (the number of ``WindowAttentionBlock`` layers in
``backbone.encoder``), and holds the wide 129x129 input, the ``interior_32`` supervision
mask, the 2400-update budget, K=4, the registered physical weights, the migrated v3-BD
1600 parent, the seeds, the evaluation region and the climatology denominator fixed.

The parent was trained at depth 4, so a depth-8 trunk cannot load it strictly. The
expansion rule is frozen here: each added block mirrors block ``(i - 4)`` in every
parameter and its two output projections (``proj`` and ``ff.net.3``) are zeroed, so each
added block is an exact identity and the depth-8 trunk starts at the registered depth-4
function. A fixed synthetic batch must produce a bit-identical FP32 encoder output before
and after expansion; that check runs per seed and its digests are recorded.

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
from training.r7_study_process import bounded_process, worker_environment

interior = dose.interior

FORMAT = "r7-s3-capacity-depth-protocol-v1"
UPDATES = 2400
STEPS = wide.STEPS              # K stays at the registered 4
REASONING_STEPS = wide.REASONING_STEPS
PARENT_DEPTH = 4
DEPTH = 8
SEEDS = wide.SEEDS
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 28800., 86400., 115200.
# Footprint proxy for THIS arm is the dose arm, not the K=8 arm. Both run K=4 and the same
# 2400 updates, and K dominates the reserved peak: the dose arm (K=4) measured 8562 MiB while
# the K=8 arm measured 13878 MiB. Depth 4->8 only doubles the encoder, whose one-pass
# activations over 4225 tokens are ~0.1 GiB, so the footprint is ~8.5 GiB -- not 14 GiB.
# 2026-10-12 (user-authorized): the host is co-resident and a neighbour job now holds ~14 GiB
# on the declared GPU, so the earlier 15+4 GiB requirement blocked the round indefinitely. The
# user asked for a rough gate rather than an exact peak: 8.5 GiB estimate (>= the 8562 MiB
# measured proxy) + 0.5 GiB margin = 9 GiB required, which the shared card clears.
ESTIMATED_PEAK_BYTES = 8704 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
MIN_ESTIMATED_PEAK_BYTES = 8192 * 2**20
EQUIVALENCE_SHAPE = (1, 2, 17, 129, 129)
EQUIVALENCE_SEED = 1234
GATE_KEYS = ("proj.weight", "proj.bias", "ff.net.3.weight", "ff.net.3.bias")
DEFAULT_OUT = ROOT / "outputs/r7_s3_capacity_depth_20261012_attempt01"
PREVIOUS_DOSE_ARM_RUN = ROOT / "outputs/r7_s3_wide_interior_dose_20261009_attempt02"
EVIDENCE_INDEX = ROOT / "docs/R7_EVIDENCE_INDEX.jsonl"
PREVIOUS_DOSE_ARM_RECORD = "s3-wide-interior-dose"
ENTRY_FILES = (*depth_arm.ENTRY_FILES, "scripts/study_r7_s3_capacity_depth.py")

DESIGN_DECISION_TEXT = (
    "Single factor: the trunk capacity. The registered wide interior-dose arm (wide 129x129 input, "
    "interior_32 supervision, 2400 updates, K=4, registered physical weights, migrated v3-BD 1600 "
    "parent) read 48 h positive on all three seeds but 72 h negative on all three. Every training-recipe "
    "lever and the reasoning depth K were tried and none turned 72 h positive, so this arm changes one "
    "model-spec field - the backbone depth 4 -> 8 - and holds every other factor fixed. The four added "
    "blocks mirror blocks 0-3 and their output projections are zeroed, so the depth-8 trunk starts at the "
    "registered depth-4 function exactly.")
DECISION_RULE_TEXT = (
    "Report only; no scientific pass is declared. Report (a) the per-seed paired delta of this arm's "
    "interior_32 t2m RMSE minus the registered 2400-update dose arm's interior_32 t2m RMSE (index record "
    "s3-wide-interior-dose, pinned, not retrained here), (b) the same delta against the pinned registered "
    "narrow arm (index record s3-rollout-dose, 800 updates), and (c) this arm's interior_32 t2m skill "
    "1-(rmse/rmse_climatology)**2, at the five leads {6,12,24,48,72} h. Both references are pinned, not "
    "retrained here.")
HYPOTHESIS_TEXT = (
    "If the residual 72 h error of the registered 2400-update arm is dominated by the trunk's per-step "
    "representational capacity rather than by the training recipe or the inference depth, then doubling "
    "the backbone depth 4 -> 8 at fixed input region, fixed interior_32 supervision, fixed 2400 updates, "
    "fixed K=4, fixed physical weights, fixed parent and fixed seeds should push 72 h skill toward or past "
    "the absolute climatology gate. If 72 h stays negative on all three seeds, the capacity branch for "
    "72 h is closed and the next hypothesis is architectural or data-side.")
DIFFERENCE_TEXT = (
    "One factor only: the trunk depth 4 -> 8 (backbone.encoder WindowAttentionBlock layers), applied to "
    "both training and scoring through the bound model spec. Same wide 129x129 input, same interior_32 "
    "supervision mask (mask_sha256 4a2a576b...), same 2400 updates, same K=4 (steps/reasoning_steps), "
    "same registered physical weights (1,.5,0,.5,0,0,0,.5,0,0,0,.5), same migrated v3-BD 1600 parent, "
    "same LR 2e-5, warmup 10, FP32, batch 1 and clip 1.")
LIMITATIONS = [
    "development screening on one wide instance (129x129 input, interior_32 supervision, one ROI, "
    "17 channels, 2017-2021 train / 2022 val); no significance, convergence, SOTA or generalization claim",
    "both references are pinned registered arms (2400-update interior s3-wide-interior-dose, narrow "
    "s3-rollout-dose), not retrained inside this protocol",
    "the parent state was trained at depth 4; the added blocks are identity-gated (mirrored parameters "
    "with zeroed output projections), so the depth change and the expansion initialization are not "
    "separated - the arm prices 'more depth from the registered function', not 'more depth from scratch'",
    "restricting supervision to interior_32 leaves the outer ring unconstrained during training; only "
    "interior_32 is scored",
    "three seeds are consistency evidence, not a significance test; GPU runs are co-resident",
    "validation split only; the test split stays sealed until a later preregistered read",
]


def execution_files():
    return sorted(set(depth_arm.execution_files()) | set(ENTRY_FILES))


def capacity_spec():
    """The archived process spec with only the trunk depth raised to 8."""
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("depth", 0)) != PARENT_DEPTH:
        raise RuntimeError(f"the archived process spec is not depth {PARENT_DEPTH}")
    spec["depth"] = DEPTH
    return spec


def depth_expansion_block():
    """The frozen depth-expansion rule and its equivalence claim."""
    return {"parent_depth": PARENT_DEPTH, "target_depth": DEPTH,
            "added_blocks": list(range(PARENT_DEPTH, DEPTH)),
            "mirror_rule": "added block i mirrors block (i - parent_depth) in every parameter",
            "gate_keys": list(GATE_KEYS),
            "gate_rule": "the gated output projections are zeroed, so each added block is an exact identity",
            "equivalence": {"module": "backbone.encoder", "dtype": "float32",
                            "batch_shape": list(EQUIVALENCE_SHAPE), "seed": EQUIVALENCE_SEED,
                            "claim": "bit-identical FP32 encoder output before and after expansion"}}


def protocol_constants():
    """The dose round's frozen fields with the single factor depth swapped in."""
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
    """Pin the registered 2400-update interior-dose arm this round deepens against."""
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


def expand_parent_state(parent_state, spec):
    """Build the depth-8 trunk from a depth-4 parent by identity-gated block expansion.

    Returns ``(model, expanded_state, receipt)``. Every added block mirrors block
    ``(i - parent_depth)`` and its gated output projections are zeroed, so the expansion
    is deterministic given the parent state and starts at the registered function. The
    equivalence claim is verified here, per seed, on a fixed synthetic batch.
    """
    import torch
    from training.r7_experiment import make_model
    from training.r7_autoregressive_runner import _state_digest
    target = make_model("process", spec)
    keys = target.state_dict()
    missing = [key for key in keys if key not in parent_state]
    if any(key not in keys for key in parent_state):
        raise RuntimeError("the depth-4 parent has keys the depth-8 trunk does not")
    added = sorted({int(key.split(".")[3]) for key in missing})
    if added != list(range(PARENT_DEPTH, DEPTH)):
        raise RuntimeError(f"the expansion added blocks {added}, not {list(range(PARENT_DEPTH, DEPTH))}")
    state = dict(parent_state)
    for block in added:
        source = block - PARENT_DEPTH
        prefix = f"backbone.encoder.blocks.{block}."
        source_prefix = f"backbone.encoder.blocks.{source}."
        for key in keys:
            if key.startswith(prefix):
                source_key = source_prefix + key[len(prefix):]
                if source_key not in parent_state:
                    raise RuntimeError(f"the mirror source {source_key} is missing from the parent")
                state[key] = parent_state[source_key].clone()
        for gate in GATE_KEYS:
            state[prefix + gate] = torch.zeros_like(keys[prefix + gate])
    target.load_state_dict(state, strict=True)
    reference = make_model("process", {**spec, "depth": PARENT_DEPTH})
    reference.load_state_dict(parent_state, strict=True)
    torch.manual_seed(EQUIVALENCE_SEED)
    probe = torch.randn(*EQUIVALENCE_SHAPE)
    with torch.no_grad():
        expanded_tokens, expanded_hw = target.backbone.encoder(probe)
        reference_tokens, reference_hw = reference.backbone.encoder(probe)
    identical = bool(torch.equal(expanded_tokens, reference_tokens)) and expanded_hw == reference_hw
    if not identical:
        raise RuntimeError("the depth-8 expansion is not bit-identical to the depth-4 parent")
    outside = {key: value for key, value in state.items() if ".blocks." not in key}
    if _state_digest(outside) != _state_digest(
            {key: value for key, value in parent_state.items() if ".blocks." not in key}):
        raise RuntimeError("the expansion changed a parameter outside the encoder blocks")
    receipt = {"parent_depth": PARENT_DEPTH, "target_depth": DEPTH, "added_blocks": added,
               "parent_state_sha256": _state_digest(parent_state),
               "expanded_state_sha256": _state_digest(state),
               "encoder_output_sha256": _state_digest({"tokens": expanded_tokens}),
               "encoder_bit_identical": identical, "non_block_keys_unchanged": True,
               "batch_shape": list(EQUIVALENCE_SHAPE), "seed": EQUIVALENCE_SEED}
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
    body["depth_expansion"] = depth_expansion_block()
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**protocol_constants(),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
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
    if body["depth_expansion"] != depth_expansion_block():
        raise RuntimeError("the frozen depth-expansion rule changed")
    if body["arms"]["candidate"] != _expected_candidate() or body["gpu_uuid"] != wide.GPU_UUID:
        raise RuntimeError("declared GPU/candidate differs from frozen recipe")
    if int(body["arms"]["candidate"]["model"]["spec"]["depth"]) != DEPTH:
        raise RuntimeError("the candidate trunk is not depth 8")
    if (body["planned_seconds"] != PLANNED_SECONDS or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS):
        raise RuntimeError("execution budget differs from freeze")
    if (int(body["estimated_peak_bytes"]) < MIN_ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("the depth-8 peak estimate was not recalibrated")
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
    if int(spec.get("depth", 0)) != DEPTH:
        raise RuntimeError("the candidate spec is not the frozen depth-8 trunk")
    model, state, expansion = expand_parent_state(saved["model"], spec)
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
                                                   "then identity-gated depth 4 -> 8 expansion; region-agnostic "
                                                   "spec reused on the wide store")},
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
        raise RuntimeError(f"seed {seed} did not reach the frozen depth-{DEPTH} K={STEPS} endpoint")
    receipt = {"seed": seed, "training_dir": str(folder), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES, "reasoning_steps": STEPS,
               "depth": DEPTH, "depth_expansion": expansion, "evaluations": {}}
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
                "updates": UPDATES, "depth": DEPTH, "reasoning_steps": REASONING_STEPS, "seeds": {}}
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
    recipe._shared().pin_declared_gpu(wide.GPU_UUID)
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
                  "supervision": body["supervision"], "updates": UPDATES, "depth": DEPTH,
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
