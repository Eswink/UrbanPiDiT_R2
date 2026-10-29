"""S3 urban static-covariate wiring round: pathway and identity, no scientific claim.

Decision 0063 froze how the static covariates are produced and published; decision 0069
authorises wiring them into the trunk. Both are explicit that this is an *engineering* round:

- the field's ``static_identity`` is bound into ``protocol_sha256`` and **not** into
  ``data_identity``, so two different static fields cannot share one protocol identity while
  the store and its manifests keep describing the data;
- the trunk gains one off-by-default switch whose off path is the pre-existing computation
  operation for operation;
- the four predeclared exits (W1-W4) are engineering assertions. None of them says anything
  about whether urban static covariates help, and this round makes no such claim.

The arms differ in exactly the switch. They share the store, the train and val manifests, the
seed, the update count, the recipe and the initialization - which is why the arm diff is a
pathway difference and not a data or capacity one. ``parameter_delta`` is 0 by construction:
the static term is a fixed projection over a fixed polynomial expansion of the published
field, so nothing about it is learned.

Two consequences worth stating here rather than in prose only:

- **This round moves ``model_code_sha256``.** Every registered migrated parent was produced
  under ``60de3b2d...`` and ``load_checkpoint`` refuses a foreign implementation by design, so
  this code cannot load them. The arms therefore start from a seeded initialization, and their
  numbers are *not* comparable with the R1-R5 parent-anchored arms.
- **Evaluation is capped** (`MAX_SAMPLES`) to keep five leads x two arms inside the round's
  wall clock. The t2m numbers verify the pathway end to end; they are not a skill reading.
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

from model import static_conditioning_r7 as static
from scripts import r7_static_wiring_support as support
from scripts import study_r7_s3_fy_year_coverage_matched as matched
from scripts import study_r7_s3_wide_single_factor as wide

FORMAT = "r7-s3-static-covariates-wiring-protocol-v1"
STAGE = "S3-STATIC-WIRING"
INSTANCE = matched.INSTANCE
STORE = matched.STORE
SOURCE_SHA256 = matched.SOURCE_SHA256
SEED = 41
SEEDS = (SEED,)
UPDATES, BATCH_SIZE = 120, 2
STEPS, LR, WARMUP = wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, CLIP = wide.WEIGHT_DECAY, wide.CLIP
CHECKPOINT_EVERY, BF16 = wide.CHECKPOINT_EVERY, wide.BF16
PHYSICAL_WEIGHTS, REASONING_STEPS = wide.PHYSICAL_WEIGHTS, wide.REASONING_STEPS
EVALUATION_LEADS, REGION = matched.EVALUATION_LEADS, matched.REGION
BOUNDARY_MARGINS, DEVICE_MODE = matched.BOUNDARY_MARGINS, matched.DEVICE_MODE
#: Pathway check, not a skill reading: ten cases per lead is enough to prove five tables come
#: out of an ``on`` arm, and it keeps five leads x two arms inside the wall clock.
MAX_SAMPLES = 10
#: Measured before freezing, not estimated: 2.44 s/update on the real store at batch 2
#: (20 updates in 48.7 s), so 120 updates is about 293 s per arm; one evaluation lead costs
#: 12.6 s warm but the first costs 736 s cold (23 s of it is the shared preparation). Two arms
#: x (293 s train + ~800 s first-lead + 4 x 13 s) is about 3,700 s, so per-seed 3,600 s is a
#: soft line rather than a ceiling and planned 7,200 s covers the whole round with margin.
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 3600., 7200., 14400.
ESTIMATED_PEAK_BYTES, HEADROOM_MARGIN_BYTES = 4096 * 2**20, 512 * 2**20
#: The pre-change fingerprint (decision 0069 W2). Recorded here so the round cannot restate it.
REFERENCE = ROOT / "outputs/r7_static_wiring_20261029/pre_change_reference.json"
REFERENCE_SHA256 = "4d0dc228699ae374b277f57cf3693cde1233ccdd554557dc68ad88a2a6ca3230"
ENTRY_FILES = ("scripts/study_r7_static_covariates_wiring.py",
               "scripts/r7_static_wiring_support.py", "scripts/r7_static_wiring_reference.py",
               "model/static_conditioning_r7.py")


def execution_files():
    """Decision 0067: the import closure plus the files it is named with, never a glob."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0], extra=ENTRY_FILES[1:])


def arm_spec(name):
    """The frozen R1 spec, plus the one switch on the treatment arm."""
    spec = matched._model_spec()
    if "static_conditioning" in spec:
        raise RuntimeError("the archived spec already carries a static switch")
    if name == "treatment":
        spec["static_conditioning"] = True
    elif name != "control":
        raise ValueError(f"unknown arm {name!r}")
    return spec


def static_section():
    """The frozen ``static`` block: the field's identity and how the trunk consumes it."""
    from training.r7_arm_harness import sha256_file
    fields, variables = static.load_static_fields()
    manifest = static.require_static_identity()
    return support.static_block(
        manifest, fields, variables, parameter_delta=0,
        expansion=static.EXPANSION_PER_VARIABLE, scales=static.STATIC_SCALES,
        source_paths={"manifest": str(static.STATIC_MANIFEST),
                      "manifest_sha256": sha256_file(static.STATIC_MANIFEST),
                      "fields": str(static.STATIC_FIELDS),
                      "fields_sha256": sha256_file(static.STATIC_FIELDS)})


#: One manifest for both arms. R1's ``ARMS`` map two *different* manifests (5 vs 22 years), and
#: reusing that map here would have made the arms differ in data as well as in the switch -
#: which is R1's factor, not this round's. The 22-year arm is used because it is the superset
#: and it is the one whose windows cover every month.
SHARED_MANIFESTS = matched.ARMS[1][1]


def manifest_path(split):
    """The one manifest both arms read, ``train`` or ``val``."""
    if split not in ("train", "val"):
        raise ValueError(f"this round reads train and val only, not {split!r}")
    return SHARED_MANIFESTS / f"{split}.jsonl"


def frozen_constants():
    """Every protocol field that does not depend on the running tree."""
    return {
        "format": FORMAT, "stage": STAGE, "instance": str(INSTANCE), "store": str(STORE),
        "source_sha256": SOURCE_SHA256, "store_shape": list(matched.STORE_SHAPE),
        "split_years": {"train": matched.CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(manifest_path("train")),
        "val_manifest": str(manifest_path("val")),
        "test_manifest_never_read": str(SHARED_MANIFESTS / "test.jsonl"),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS),
                   "scored_region": REGION, "n_grid_points": matched.FULL_GRID_POINTS},
        "factor": {"name": support.FACTOR_NAME, "control": "static_conditioning=False",
                   "treatment": "static_conditioning=True",
                   "controlled": list(support.CONTROLLED), "not_controlled": []},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS),
                   "reasoning_steps": REASONING_STEPS,
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)",
                   "initialization": ("seeded initialization; no parent checkpoint, because this "
                                      "round changes model_code_sha256 and load_checkpoint "
                                      "refuses the registered migrated parents")},
        "evaluation": {"leads": list(EVALUATION_LEADS), "max_samples": MAX_SAMPLES,
                       "note": ("a pathway check, not a skill reading: ten cases per lead prove "
                                "the on path scores end to end inside the wall clock")},
        "static": static_section(),
        "exits": {"w1": "static_identity reaches protocol_sha256 and data_identity is unmoved",
                  "w2": "off path reproduces the pre-change fingerprint exactly",
                  "w3": "both arms score five leads with equal boundary_rmse row counts",
                  "w4": "the model's field equals an independent re-derivation of the npz"},
        "decision": {"advance": "all of (W1)-(W4) -> engineering-positive",
                     "close": "any of (W1)-(W4) false -> negative, no on/off comparison read",
                     "text": support.DECISION_RULE},
        "gpu": {"policy": "shared", "uuid": static_gpu_uuid(),
                "estimated_peak_mib": ESTIMATED_PEAK_BYTES // 2**20,
                "headroom_margin_mib": HEADROOM_MARGIN_BYTES // 2**20,
                "gate": ("read-only free check before every seed; no neighbour signals")},
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS, "decision": "0069",
                    "soft_rule": "soft overrun continues and is recorded; only the hard cap truncates"},
        "test_read": False,
        "test_read_policy": ("the 2023 test split is never opened; only train (fit) and val "
                             "(scoring) manifests are read"),
        "scientific_claim": False, "limitations": list(support.LIMITATIONS),
    }


def static_gpu_uuid():
    return wide.GPU_UUID


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    from training import r7_device_selection as devices
    body = {**frozen_constants(),
            "gpu_uuid": static_gpu_uuid(), "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(DEVICE_MODE, single_uuid=static_gpu_uuid()),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS, "started_perf_counter": started,
            "deadline_perf_counter": deadline,
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "preparation_protocol.json", body)
    return body


def _identity():
    """Store, manifests, code and the static product: everything the arms share or bind."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import (canonical_digest, dataset_identity, make_model,
                                        model_code_digest, seed_everything)
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    import torch
    source_stat = matched.SOURCE.stat()
    train, val = manifest_path("train"), manifest_path("val")
    identity, _ = dataset_identity(train)
    windows = preflight_long_rollout_windows(train, physical_steps=12)
    arms = {}
    for name in support.ARMS:
        spec = arm_spec(name)
        with torch.random.fork_rng(devices=[]):
            seed_everything(SEED)
            template = make_model("process", spec)
        arms[name] = {
            "train_manifest": str(train), "train_data_identity": identity,
            "manifest_sha256": {"train": sha256_file(train), "val": sha256_file(val)},
            "train_windows_sha256": windows["window_sha256"],
            "model": {"kind": "process", "spec": spec,
                      "spec_canonical_digest": canonical_digest(spec)},
            "initial_parameters": sum(p.numel() for p in template.parameters()),
            "n_tensors": len(template.state_dict())}
    if arms["control"]["train_data_identity"] != arms["treatment"]["train_data_identity"]:
        raise RuntimeError("the arms resolved different train data identities; this round's "
                           "factor is the switch, not the data")
    if arms["control"]["initial_parameters"] != arms["treatment"]["initial_parameters"]:
        raise RuntimeError("the arms build different parameter counts; parameter_delta must be 0")
    if arms["control"]["manifest_sha256"] != arms["treatment"]["manifest_sha256"]:
        raise RuntimeError("the arms resolved different manifest bytes")
    return {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "source_file": {"path": str(matched.SOURCE), "bytes": source_stat.st_size,
                            "mtime_ns": source_stat.st_mtime_ns},
            "model_code_sha256": model_code_digest(), "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()},
            "arms": arms, "train_windows": windows,
            "val_data_identity": dataset_identity(val)[0]}


def _prepared_protocol(output):
    """Freeze the identity, then record the three probes that need no GPU.

    The probes run here rather than in a separate stage because none of them trains or
    scores: W1 is digest arithmetic, W2 re-runs a CPU fingerprint, W4 re-derives the field.
    Keeping them at prepare time also means a pathway that is wrong fails before any GPU
    second is spent.
    """
    from training.r7_experiment import canonical_digest
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    body.pop("protocol_sha256")
    # W1's mechanism: the static block already carries ``static_identity``, so it enters this
    # digest. ``data_identity`` lives under ``arms[*].train_data_identity`` and is untouched.
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    observations = {"w1": identity_probe(body), "w2": off_path_probe(body),
                    "w4": field_probe(body)}
    wide.write_exclusive(output / "exits_partial.json", observations)
    return body


def validate_protocol(output):
    """Re-derive the digest and re-check every identity the freeze promises."""
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_arm_harness import sha256_file
    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if body["model_code_sha256"] != model_code_digest():
        raise RuntimeError("model code changed after the protocol froze")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution source drifted: {name}")
    current = {name: sha256_file(ROOT / name) for name in execution_files()}
    if set(current) != set(body["execution_files_sha256"]):
        raise RuntimeError("incomplete execution source inventory")
    if body["static"]["static_identity"] != static.require_static_identity()["static_identity"]:
        raise RuntimeError("the static product no longer carries the frozen identity")
    if body["static"]["parameter_delta"] != 0:
        raise RuntimeError("the frozen parameter delta changed")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["format"] != FORMAT or body["decision"]["text"] != support.DECISION_RULE):
        raise RuntimeError("frozen claims/decision rule changed")
    return body


def identity_probe(body):
    """W1: mutating ``static_identity`` must move ``protocol_sha256`` and nothing else."""
    from training.r7_experiment import canonical_digest
    original = body.get("static", {}).get("static_identity")
    if not isinstance(original, str) or not original:
        # A body with no static identity has nothing to bind; that is the wiring being absent,
        # which is a W1 failure rather than an exception.
        return {"passed": False, "reason": "the protocol carries no static_identity to bind",
                "original_static_identity": original}
    mutated = json.loads(json.dumps(body))
    mutated["static"]["static_identity"] = ("0" if original[0] != "0" else "1") + original[1:]
    stripped = lambda payload: {key: value for key, value in payload.items()
                                if key != "protocol_sha256"}
    before = canonical_digest(stripped(body))
    after = canonical_digest(stripped(mutated))
    data_before = {name: arm["train_data_identity"] for name, arm in body["arms"].items()}
    data_after = {name: arm["train_data_identity"] for name, arm in mutated["arms"].items()}
    leaked = any(identity == original for identity in data_before.values())
    return {"passed": before != after and data_before == data_after and not leaked,
            "original_static_identity": original,
            "mutated_static_identity": mutated["static"]["static_identity"],
            "protocol_sha256_before": before, "protocol_sha256_after": after,
            "data_identity_unchanged": data_before == data_after,
            # The two halves are not the same statement. ``data_before == data_after`` is what
            # the mutation could break; ``leaked`` is what a driver that folded the field into
            # the store identity would look like. Both are checked, and neither is implied by
            # the protocol digest moving.
            "data_identity_carries_the_static_identity": bool(leaked),
            "data_identity": data_before,
            "note": ("the mutation is in memory only; the frozen protocol file is not touched")}


def off_path_probe(body):
    """W2: re-run the pre-change fingerprint on the live tree and compare field by field."""
    from scripts.r7_static_wiring_reference import compare, record
    from training.r7_arm_harness import sha256_file
    if sha256_file(REFERENCE) != REFERENCE_SHA256:
        raise RuntimeError("the pre-change reference drifted from its pinned SHA256")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if reference["values"]["model_code_sha256"] == body["model_code_sha256"]:
        raise RuntimeError("the reference and the live tree carry the same model digest; the "
                           "reference was not captured before the edit")
    differences = compare(reference, record("HEAD"))
    return {"passed": not differences, "differences": differences,
            "reference_model_code_sha256": reference["values"]["model_code_sha256"],
            "live_model_code_sha256": body["model_code_sha256"],
            "reference_sha256": REFERENCE_SHA256,
            "note": ("the reference must predate the model edit, which is why its model digest "
                     "is required to differ from the live one")}


def field_probe(body):
    """W4: the model's field equals the published npz, and an independent re-derivation agrees."""
    import numpy as np
    import torch
    variables = body["static"]["variables"]
    fields, loaded = static.load_static_fields(variables)
    with np.load(static.STATIC_FIELDS) as archive:
        buffer_equal = all(np.array_equal(fields[index].numpy(),
                                          np.asarray(archive[name], dtype=np.float32))
                           for index, name in enumerate(loaded))
        raw = [[[float(value) for value in row] for row in
                np.asarray(archive[name], dtype=np.float64)] for name in variables]
    token_hw = (33, 33)
    loop = support.independent_patch_mean(raw, token_hw=token_hw, patch_size=2)
    vector = static.static_features(fields, token_hw=token_hw, patch_size=2,
                                    variables=variables)
    loop_tensor = torch.tensor(loop, dtype=torch.float64)
    # Compare the *patch average* the loop produces with the vectorised patch average re-derived
    # here, then separately confirm the module's own features are a pure function of it.
    from model.layers.patch_grid import pad_patch_grid
    scales = torch.tensor([static.STATIC_SCALES.get(name, 1.) for name in variables],
                          dtype=torch.float64).reshape(-1, 1, 1)
    padded = pad_patch_grid((fields.double() / scales).unsqueeze(0), 2)
    cells = padded.reshape(1, len(variables), 33, 2, 33, 2).mean(dim=(3, 5))
    vector_cells = cells.reshape(len(variables), -1).transpose(0, 1)
    raw_cells = loop_tensor
    scaled_loop = loop_tensor / scales.reshape(-1).double()
    discrepancy = float((scaled_loop - vector_cells).abs().max())
    centre = vector_cells.mean(dim=0, keepdim=True)
    half = (vector_cells.amax(dim=0, keepdim=True)
            - vector_cells.amin(dim=0, keepdim=True)).clamp(min=1e-6)
    rebuilt = ((vector_cells - centre) / (0.5 * half)).clamp(-3., 3.)
    expected = torch.cat([rebuilt, rebuilt.square()], dim=1)
    module_equal = torch.equal(vector.double(), expected)
    within_bound = discrepancy <= support.REDUCTION_ORDER_TOLERANCE
    return {"passed": bool(buffer_equal and module_equal and within_bound),
            "buffer_bitwise_equal": bool(buffer_equal),
            "independent_patch_mean_max_abs_difference": discrepancy,
            "independent_patch_mean_tolerance": support.REDUCTION_ORDER_TOLERANCE,
            "independent_patch_mean_within_bound": bool(within_bound),
            "module_features_equal_expansion": bool(module_equal),
            "variables": loaded, "token_grid": list(token_hw),
            "note": ("the buffer is bitwise the published npz; the loop re-derivation shares no "
                     "code with the vectorised path and is judged at the float64 reduction-order "
                     "bound, because a sequential sum and a batched mean accumulate differently")}


def _arm_contract(body, name):
    """The frozen contract for one arm; ``data_identity`` stays the store's, not the field's."""
    arm = body["arms"][name]
    return {"kind": "process", "model": arm["model"]["spec"],
            "data_identity": arm["train_data_identity"], "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {
                "description": ("seeded initialization under the round's frozen seed; no parent "
                                "checkpoint is loadable under this round's model_code_sha256"),
                "seed": SEED, "parent_checkpoint": None},
            "static_conditioning": {"enabled": name == "treatment",
                                    "static_identity": body["static"]["static_identity"],
                                    "binding": "protocol_sha256 only; data_identity unmoved"},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(PHYSICAL_WEIGHTS),
                               "window_sha256": body["train_windows"]["window_sha256"],
                               "excluded_sample_ids": body["train_windows"]["excluded_sample_ids"]}}


def _train_arm(output, body, name, deadline):
    from training.r7_experiment import make_model, seed_everything
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_arm_harness import sha256_file
    import torch
    spec = body["arms"][name]["model"]["spec"]
    with torch.random.fork_rng(devices=[]):
        seed_everything(SEED)
        model = make_model("process", spec)
    folder = output / f"seed{SEED}/training/{name}"
    print(f"[seed {SEED}] arm {name}: {UPDATES} updates at batch {BATCH_SIZE} from a seeded "
          "initialization", flush=True)
    checkpoint, report = fine_tune_long_rollout(
        body["arms"][name]["train_manifest"], folder, model=model,
        contract=_arm_contract(body, name), physical_weights=PHYSICAL_WEIGHTS, parent_weights=None,
        steps=STEPS, updates=UPDATES, lr=LR, warmup=WARMUP, seed=SEED, device_name="cuda:0",
        deadline=deadline, batch_size=BATCH_SIZE)
    if (report["updates_this_run"] != UPDATES
            or report["data_identity"] != body["arms"][name]["train_data_identity"]):
        raise RuntimeError(f"arm {name} did not reach the frozen endpoint")
    return {"seed": SEED, "arm": name, "training_dir": str(folder), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_report_sha256": sha256_file(folder / "training_report.json"),
            "elapsed_seconds": report["elapsed_seconds"], "updates": UPDATES,
            "batch_size": BATCH_SIZE, "reasoning_steps": STEPS,
            "train_data_identity": report["data_identity"],
            "static_conditioning": name == "treatment",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "first_loss": float(report["losses"][0]["loss"]) if report.get("losses") else None,
            "last_loss": float(report["losses"][-1]["loss"]) if report.get("losses") else None}


def _score(output, body, name, receipt, deadline):
    """Five leads on val, with the shared preparation built once."""
    from training.r7_evaluate import EvaluationPreparation, evaluate_local
    from training.r7_arm_harness import sha256_file
    manifest = manifest_path("val")
    prepared = EvaluationPreparation(manifest, deadline=deadline)
    tables = {}
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{SEED}/evaluation/{name}/lead_{lead:03d}h"
        print(f"[seed {SEED}] arm {name}: scoring lead {lead}h", flush=True)
        provenance = evaluate_local(
            manifest, output_dir=destination, checkpoint=receipt["checkpoint"],
            lead_hours=(lead,), max_samples=MAX_SAMPLES, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=BOUNDARY_MARGINS,
            deadline=deadline, prepared=prepared)
        if provenance["split"] != "val" or "test_read" in provenance:
            raise RuntimeError(f"arm {name} lead {lead} did not stay on val")
        table = destination / "boundary_rmse.csv"
        tables[str(lead)] = {"dir": str(destination), "rows": support.rows_of(table),
                             "boundary_rmse_csv_sha256": sha256_file(table),
                             "n_evaluated": provenance["n_evaluated"],
                             "elapsed_seconds": provenance["elapsed_seconds"]}
    return tables


def _seed(output, body, deadline):
    import torch
    receipts = {}
    for name in support.ARMS:
        receipt = _train_arm(output, body, name, deadline)
        receipt["evaluations"] = _score(output, body, name, receipt, deadline)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{SEED}_{name}_receipt.json", receipt)
        receipts[name] = receipt
        print(f"[seed {SEED}] arm {name}: done", flush=True)
    control, treatment = receipts["control"], receipts["treatment"]
    if control["train_data_identity"] != treatment["train_data_identity"]:
        raise RuntimeError("the arms trained on different data identities; the factor is broken")
    return receipts


def path_probe(output, receipts):
    """W3: both arms scored all five leads, with the same number of rows per lead."""
    per_arm = {name: receipts[name]["evaluations"] for name in support.ARMS}
    leads = [str(lead) for lead in EVALUATION_LEADS]
    missing = [(name, lead) for name in support.ARMS for lead in leads
               if lead not in per_arm[name]]
    control_rows = {lead: per_arm["control"][lead]["rows"] for lead in leads if lead in per_arm["control"]}
    treatment_rows = {lead: per_arm["treatment"][lead]["rows"]
                      for lead in leads if lead in per_arm["treatment"]}
    counts_equal = control_rows == treatment_rows and set(control_rows) == set(leads)
    scored = all(per_arm[name][lead]["n_evaluated"] > 0
                 for name in support.ARMS for lead in leads if lead in per_arm[name])
    return {"passed": bool(not missing and counts_equal and scored),
            "missing": [list(item) for item in missing],
            "control_rows": control_rows, "treatment_rows": treatment_rows,
            "row_counts_equal": bool(counts_equal), "every_lead_scored": bool(scored),
            "evaluations": per_arm,
            "note": ("an on/off t2m difference is recorded but not read: this round verifies "
                     "that the pathway runs, not that the covariates help")}


def _exits(output, body, receipts):
    from training.r7_arm_harness import sha256_file
    observations = json.loads((output / "exits_partial.json").read_text(encoding="utf-8"))
    if sha256_file(output / "exits_partial.json") is None:
        raise RuntimeError("the partial exit record vanished")
    observations["w3"] = path_probe(output, receipts)
    verdict = support.exits(observations)
    return {"scientific_claim": False, "test_read": False,
            "protocol_sha256": body["protocol_sha256"], "decision_rule": support.DECISION_RULE,
            "factor": support.FACTOR_NAME, "updates": UPDATES, "batch_size": BATCH_SIZE,
            "static": {"static_identity": body["static"]["static_identity"],
                       "output_sha256": body["static"]["output_sha256"],
                       "variables": body["static"]["variables"],
                       "parameter_delta": body["static"]["parameter_delta"]},
            "observations": observations, "exits": verdict}


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    print(f"[worker] phase={phase} seed={seed} out={output}", flush=True)
    if phase == "prepare":
        body = _prepared_protocol(output)
        print(f"[worker] prepare done: protocol {body['protocol_sha256'][:12]} "
              f"static {body['static']['static_identity'][:12]}", flush=True)
        return 0
    body = validate_protocol(output)
    if (deadline is None or not math.isfinite(deadline)
            or deadline > body["deadline_perf_counter"] or deadline <= time.perf_counter()):
        raise RuntimeError("worker deadline outside the frozen round")
    from training import r7_device_selection as devices
    from scripts.study_r7_s3_v3_rollout_ft import _shared
    _shared().pin_declared_gpu(devices.assigned_uuid(body))
    if phase == "exits":
        # Re-derived on the promoted protocol, which is what the seed stage will read.
        body = validate_protocol(output)
        observations = {"w1": identity_probe(body), "w2": off_path_probe(body),
                        "w4": field_probe(body)}
        wide.write_exclusive(output / "exits_partial.json", observations)
        print("[worker] exits written (w3 needs the two arms)", flush=True)
        return 0
    if phase != "seed":
        raise RuntimeError(f"unknown phase {phase!r}")
    if seed not in SEEDS or deadline - time.perf_counter() > PER_SEED_SECONDS:
        raise RuntimeError("seed/deadline differs from freeze")
    receipts = _seed(output, body, deadline)
    wide.write_exclusive(output / "readings.json", _exits(output, body, receipts))
    print("[worker] seed done and exits written", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("prepare", "exits", "seed"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.worker:
        raise RuntimeError("this round is driven by `r7_round.py stage`, not by a bare main")
    return worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
