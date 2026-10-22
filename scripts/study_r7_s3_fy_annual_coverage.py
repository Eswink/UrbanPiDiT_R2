"""S3 FY annual coverage: 4 months vs 12 months of training data, years fixed.

The architecture, capacity and training-signal families are closed, and the user chose the
data direction on 2026-10-21. The FY source already on disk covers 2017-2022 on the native
65x65 grid; this round derives one full-year store from it and asks a single question: at
the same 2,400-update budget, does training on all twelve months beat training on the
registered four-month selection (Jan/Apr/Jul/Oct, days 1-30, UTC 00/06/12/18), with the
years fixed at 2017-2021?

Both arms share the store, the full-year normalization, the model spec, the migrated v3-BD
parent, the seeds, the 2022 full-year val split and the 48-bucket train-only climatology;
they differ only in their train manifest. The round is 65x65 only: the repository has no
65->129 upsampling contract, so the result is never presented as a wide-instance result.
Validation-only, offline, co-resident on two declared GPU UUIDs with a read-only headroom
gate before every spawn. The 2023 test split is never opened.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_fy_annual_parent as fy_parent
from scripts import study_r7_s3_wide_interior_dose as dose
from scripts import study_r7_s3_wide_single_factor as wide
from scripts import study_r7_s3_v3_rollout_ft as recipe
from training import r7_device_selection as devices

FORMAT = "r7-s3-fy-annual-coverage-protocol-v1"
INSTANCE = ROOT / "outputs/r7_s3_fy_annual_20261021_attempt01"
STORE = INSTANCE / "store/cache.zarr"
SOURCE = INSTANCE / "source.nc"
SOURCE_SHA256 = "449470415aa8b845a6d8c3870140bd47e43bb5e3dc7b6a87c302cd26976afbc8"
CONTROL, TREATMENT = "control", "treatment"
ARMS = ((CONTROL, INSTANCE / "store/manifests_control_h72"),
        (TREATMENT, INSTANCE / "store/manifests_treatment"))
TRAIN_COUNTS = {CONTROL: 2140, TREATMENT: 7302}
VAL_COUNT, TEST_COUNT = 1458, 472
VAL_COHORTS = {"6": 1458, "12": 1457, "24": 1455, "48": 1451, "72": 1447}
CLIMATOLOGY_T2M_RMSE = {
    "6": 3.8878542160913172, "12": 3.8887477466606346, "24": 3.8906948018885616,
    "48": 3.8946928171358484, "72": 3.8970789101854089,
}
CLIMATOLOGY_YEARS, CLIMATOLOGY_BUCKETS = [2017, 2018, 2019, 2020, 2021], 48
D2_REFERENCE = ROOT / "outputs/r7_s3_fy_annual_d2_20261021_attempt01/reference.json"
D2_FORMAT, BRIDGE_RECORD, BRIDGE_UPDATES = "r7-s3-fy-annual-d2-reference-v1", "s3-v3-budget-dose", 1600
BRIDGE_VALUES = {
    "6": [2.2576727382956983, 2.2627956234561597, 2.2229286358562574],
    "12": [2.9037256040660173, 2.9066938336085633, 2.8385020742760827],
    "24": [3.076150935611291, 3.2122558019026357, 2.878542562213169],
    "48": [5.0496735991471065, 4.980754833624437, 4.066026449728292],
    "72": [7.044753781949937, 6.457676654363861, 4.895728801808265],
}
UPDATES, STEPS, LR, WARMUP = 2400, wide.STEPS, wide.LR, wide.WARMUP
WEIGHT_DECAY, BATCH_SIZE, CLIP = wide.WEIGHT_DECAY, wide.BATCH_SIZE, wide.CLIP
CHECKPOINT_EVERY, BF16, PHYSICAL_WEIGHTS = wide.CHECKPOINT_EVERY, wide.BF16, wide.PHYSICAL_WEIGHTS
EVALUATION_LEADS, REASONING_STEPS, SEEDS = wide.EVALUATION_LEADS, wide.REASONING_STEPS, wide.SEEDS
BOUNDARY_MARGINS, REGION, FULL_GRID_POINTS, DEVICE_MODE = (1,), "full", 4225, "dual"
# Budget corrected after first real use (0053): one arm measured 7,170-7,201 s (2 arms = 14,400 s).
PER_SEED_SECONDS, PLANNED_SECONDS, HARD_CAP_SECONDS = 21600., 43200., 86400.
ESTIMATED_PEAK_BYTES, HEADROOM_MARGIN_BYTES = 4096 * 2**20, 512 * 2**20
ENTRY_FILES = ("scripts/r7_s3_fy_annual_parent.py", "scripts/study_r7_s3_fy_annual_coverage.py", "scripts/build_r7_s3_fy_control_manifest.py", *wide.execution_files())

LIMITATIONS = [
    "development screening on the 65x65 instance only (2017-2021 train / 2022 val, 17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed",
    "both arms share the full-year 2017-2021 normalization, so the four-month control is not byte-comparable with the registered four-month narrow arm",
    "the budget is fixed at 2400 updates: the treatment sees about one third of an epoch while the control sees about one, so a positive result prices data diversity at this budget, not data volume at every budget",
    "the external bridge to s3-v3-budget-dose differs in normalization, val cases and update budget; it is an identity/engineering check, never a scientific verdict",
    "three seeds are consistency evidence, not a significance test",
    "GPU runs are co-resident; latency/memory observations include neighbor load",
    "validation split only; the test split stays sealed until a later preregistered read"]
DECISION_RULE = (
    "Exit (a): treatment 72 h t2m skill > 0 on seeds 41/42/43. (b): treatment minus control 72 h t2m delta RMSE < 0 on all three seeds. "
    "(c): treatment 48 h t2m skill > 0 on all three seeds. (d): treatment minus control delta RMSE at 6/12/24 h, averaged over seeds, <= 0. "
    "All four true -> ask for independent review; any false -> permanently close the full-year coverage branch. The bridge (e) is reported but is not a scientific verdict.")


def execution_files():
    return sorted(set(ENTRY_FILES))


def manifest_path(name, split):
    """The arm's manifest file, read from ``ARMS`` so the two cannot drift.

    The control arm reads ``manifests_control_h72`` (the complete-72h subset) rather than the
    unfiltered ``manifests_control``, whose 6 h records admit windows that roll into another
    month and would read frames the four-month control arm must not see.
    """
    folders = dict(ARMS)
    if name not in folders:
        raise ValueError(f"unknown arm {name!r}")
    return folders[name] / f"{split}.jsonl"


def bridge_reference():
    """The registered narrow arm's seed readings and the rule-(e) noise band."""
    body = {}
    for lead, values in BRIDGE_VALUES.items():
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        body[lead] = {"values": list(values), "mean": mean, "spread": spread,
                      "band": max(spread, 0.05 * abs(mean))}
    return body


def frozen_constants():
    """Every protocol field that does not depend on the built store (unit-testable)."""
    return {
        "format": FORMAT, "stage": "S3-FY-AC", "instance": str(INSTANCE), "store": str(STORE),
        "source_sha256": SOURCE_SHA256, "source_bytes": 2036671757,
        "store_shape": [9244, 17, 65, 65],
        "split_years": {"train": CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifests": {name: str(manifest_path(name, "train")) for name, _ in ARMS},
        "val_manifests": {name: str(manifest_path(name, "val")) for name, _ in ARMS},
        "test_manifest_never_read": {name: str(manifest_path(name, "test")) for name, _ in ARMS},
        "train_file_counts": dict(TRAIN_COUNTS), "val_count": VAL_COUNT,
        "test_count": TEST_COUNT, "val_cohorts": dict(VAL_COHORTS),
        "region": {"points": [65, 65], "boundary_margins": list(BOUNDARY_MARGINS),
                   "scored_region": REGION, "n_grid_points": FULL_GRID_POINTS,
                   "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"},
        "factor": {"name": "training month coverage", "control": "4 months (Jan/Apr/Jul/Oct)",
                   "treatment": "12 months", "years": CLIMATOLOGY_YEARS,
                   "controlled": ["store", "normalization", "model spec", "migrated parent",
                                  "updates", "K", "depth", "dim", "physical weights", "seeds",
                                  "val split", "climatology denominator"],
                   "not_controlled": ["2000-2016 years (a later separate factor)"]},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS, "lr": LR,
                   "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": CLIP, "bf16": BF16, "checkpoint_every": CHECKPOINT_EVERY,
                   "physical_weights": list(PHYSICAL_WEIGHTS),
                   "reasoning_steps": REASONING_STEPS,
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)"},
        "climatology": {"kind": "train-only-month-hour-grid-mean-v1",
                        "buckets": CLIMATOLOGY_BUCKETS, "training_years": CLIMATOLOGY_YEARS,
                        "n_selected_steps": 7304, "t2m_rmse": dict(CLIMATOLOGY_T2M_RMSE),
                        "reference_format": D2_FORMAT, "reference_path": str(D2_REFERENCE)},
        "bridge_e": {"record_id": BRIDGE_RECORD, "updates": BRIDGE_UPDATES,
                     "scientific": False, "reference": bridge_reference(),
                     "rule": ("control seed mean must stay within the registered arm's seed "
                              "spread or 5% of its mean, whichever is larger, at every lead; "
                              "a failure is identity/engineering unresolved, not a verdict")},
        "decision": {"exits": {"a": "treatment 72h t2m skill > 0, all seeds",
                               "b": "treatment-control 72h delta RMSE < 0, all seeds",
                               "c": "treatment 48h t2m skill > 0, all seeds",
                               "d": "treatment-control delta RMSE at 6/12/24h, seed mean <= 0"},
                     "advance": "all of (a)-(d) -> independent review",
                     "close": "any of (a)-(d) false -> close the full-year coverage branch",
                     "text": DECISION_RULE},
        "gpu": {"policy": "shared", "uuid": wide.GPU_UUID,
                "estimated_peak_mib": ESTIMATED_PEAK_BYTES // 2**20,
                "headroom_margin_mib": HEADROOM_MARGIN_BYTES // 2**20,
                "neighbour_reserve_gib": 6,
                "gate": ("read-only free check before every seed; required = max(estimate, "
                         "owned peak) + margin + neighbour reserve; no neighbour signals"),
                "billing": "continuous first spawn through last owned reap; failures charged"},
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS,
                    "soft_rule": "soft overrun continues and is recorded; only the hard cap truncates"},
        "test_read": False,
        "test_read_policy": ("the 2023 four-month test split is never opened; only the train "
                             "manifests (fit) and the val manifests (scoring) are read"),
        "scientific_claim": False, "limitations": list(LIMITATIONS),
    }


def arm_protocol(spec, *, train_manifest, train_identity, windows, canonical_digest):
    """One arm's frozen record. The only fields the two arms may differ in are named here."""
    return {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
            "lr": LR, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "clip": CLIP, "bf16": BF16,
            "checkpoint_every": CHECKPOINT_EVERY,
            "physical_weights": list(PHYSICAL_WEIGHTS),
            "train_manifest": str(train_manifest),
            "train_data_identity": train_identity, "train_windows": windows,
            "model": {"kind": "process", "spec": spec,
                      "spec_canonical_digest": canonical_digest(spec)},
            "selection": "frozen endpoint; no validation selection"}


ARM_FACTOR_KEYS = ("train_data_identity", "train_manifest", "train_windows")


def difference_keys(control, treatment):
    """Fields that differ between the two arms; must equal ARM_FACTOR_KEYS exactly."""
    return sorted(key for key in set(control) | set(treatment)
                  if control.get(key) != treatment.get(key))


def require_single_factor(control, treatment):
    """Raise unless the only arm differences are the training manifest and its identity."""
    observed = tuple(difference_keys(control, treatment))
    if observed != ARM_FACTOR_KEYS:
        raise ValueError(f"the arms differ in {observed}, expected exactly {ARM_FACTOR_KEYS}")
    if control["model"] != treatment["model"]:
        raise ValueError("the arms use different model specs")


def check_model_revision(observed, *, expected=fy_parent.MODEL_CODE_SHA256):
    """The data round must run on the revision that produced the migrated parent."""
    if observed != expected:
        raise RuntimeError(f"model revision {observed} is not the pinned {expected}; "
                           "commit or clear the working-tree model changes first")


def _model_spec():
    spec, _, _ = recipe._shared().archived_process_spec()
    spec = dict(spec)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    if any(spec.get(key) for key in ("climatology_anchor_spec", "anomaly_feedback",
                                     "anchor_output_prior", "process_init_conditioning",
                                     "typed_evidence_mode", "global_mixing_mode",
                                     "tendency_input")):
        raise RuntimeError("the archived process spec already carries a closed-architecture switch")
    return spec


def climatology_reference():
    """Load and byte-verify the D2 climatology reference this round divides by."""
    from training.r7_arm_harness import sha256_file
    if not D2_REFERENCE.is_file():
        raise FileNotFoundError(f"the FY climatology reference is missing: {D2_REFERENCE}")
    reference = json.loads(D2_REFERENCE.read_text(encoding="utf-8"))
    if reference.get("format") != D2_FORMAT or reference.get("scientific_claim") is not False:
        raise RuntimeError("the FY climatology reference is not the frozen D2 artifact")
    leads = {}
    for lead in EVALUATION_LEADS:
        record = reference["leads"][str(lead)]
        if record["n_evaluated"] != record["n_available_windows"] \
                or record["n_evaluated"] != VAL_COHORTS[str(lead)]:
            raise RuntimeError(f"D2 lead {lead} does not cover the frozen val cohort")
        if len(record["bucket_counts"]) != CLIMATOLOGY_BUCKETS:
            raise RuntimeError(f"D2 lead {lead} does not carry the 48 buckets")
        csv_path = Path(record["boundary_rmse_csv"])
        if sha256_file(csv_path) != record["boundary_rmse_sha256"]:
            raise RuntimeError(f"D2 lead {lead} boundary table drifted from its digest")
        with (csv_path.parent / "rmse.csv").open(encoding="utf-8") as handle:
            t2m = next(row for row in csv.DictReader(handle) if row["variable"] == "t2m")
        value, unit = float(t2m["rmse"]), t2m["unit"]
        if unit != "K" or abs(value - CLIMATOLOGY_T2M_RMSE[str(lead)]) > 1e-12:
            raise RuntimeError(f"D2 lead {lead} t2m RMSE differs from the frozen constant")
        leads[str(lead)] = {"t2m_rmse": value, "unit": unit,
                            "n_evaluated": record["n_evaluated"],
                            "boundary_rmse_sha256": record["boundary_rmse_sha256"]}
    return {"path": str(D2_REFERENCE), "sha256": sha256_file(D2_REFERENCE),
            "format": D2_FORMAT, "bucket_count": CLIMATOLOGY_BUCKETS,
            "training_years": CLIMATOLOGY_YEARS, "n_selected_steps": 7304, "leads": leads}


def _identity():
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    check_model_revision(model_code_digest())
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("the merged FY source drifted from its frozen SHA256")
    spec = _model_spec()
    arms, manifest_sha, val_identity = {}, {}, None
    for name, _ in ARMS:
        train, val = manifest_path(name, "train"), manifest_path(name, "val")
        train_identity, _ = dataset_identity(train)
        manifest_sha[name] = {"train": sha256_file(train), "val": sha256_file(val)}
        if val_identity is None:
            val_identity = dataset_identity(val)[0]
        elif manifest_sha[name]["val"] != manifest_sha[CONTROL]["val"]:
            raise RuntimeError("the two val manifests are not byte-identical")
        arms[name] = arm_protocol(
            spec, train_manifest=train, train_identity=train_identity,
            windows=preflight_long_rollout_windows(train, physical_steps=12),
            canonical_digest=canonical_digest)
    require_single_factor(arms[CONTROL], arms[TREATMENT])
    body = {"instance": str(INSTANCE), "store": str(STORE), "source_sha256": SOURCE_SHA256,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "execution_files_sha256": {name: sha256_file(ROOT / name) for name in execution_files()},
            "arms": {**arms, "parent": {"pins": {}, "checkpoints": fy_parent.pins()}},
            "manifest_sha256": manifest_sha, "val_data_identity": val_identity,
            "climatology_reference": climatology_reference(),
            "code": recipe._shared().code_state()}
    fy_parent.install_parent_pins(body)
    return body


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    from training.r7_experiment import canonical_digest
    body = {**frozen_constants(),
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
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    body = {**preliminary, **_identity()}
    body.pop("protocol_sha256")
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output):
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    body = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("protocol digest mismatch")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError("execution source changed: " + name)
    check_model_revision(model_code_digest())
    if body["model_code_sha256"] != model_code_digest() \
            or body["training_code_sha256"] != training_code_digest():
        raise RuntimeError("model/training source identity changed")
    if sha256_file(SOURCE) != body["source_sha256"]:
        raise RuntimeError("the merged FY source identity changed")
    # The full dataset identity is computed once in prepare; repeating it here would validate
    # every record against the store on every stage. The manifest bytes and source hash are
    # checked cheaply, and the protocol digest binds the recorded dataset identity to this body.
    observed = {name: {"train": sha256_file(manifest_path(name, "train")),
                       "val": sha256_file(manifest_path(name, "val"))} for name, _ in ARMS}
    if body["manifest_sha256"] != observed:
        raise RuntimeError("a train/val manifest changed after freeze")
    if observed[CONTROL]["val"] != observed[TREATMENT]["val"]:
        raise RuntimeError("the two val manifests are no longer byte-identical")
    require_single_factor(body["arms"][CONTROL], body["arms"][TREATMENT])
    for name, _ in ARMS:
        arm = body["arms"][name]
        train = manifest_path(name, "train")
        if (arm["train_manifest"] != str(train)
                or arm["train_windows"].get("input_windows") != TRAIN_COUNTS[name]
                or len(arm["train_windows"].get("window_sha256", "")) != 64
                or arm["model"]["spec_canonical_digest"]
                != canonical_digest(arm["model"]["spec"])):
            raise RuntimeError(f"the {name} arm changed from the frozen record")
    if body["climatology_reference"] != climatology_reference():
        raise RuntimeError("the climatology reference changed")
    if body["arms"]["parent"]["checkpoints"] != fy_parent.pins() \
            or body["arms"]["parent"]["pins"]["checkpoints"] != fy_parent.pins() \
            or body["arms"]["parent"]["pins"]["model_only_migration"] != fy_parent.migration_provenance():
        raise RuntimeError("the migrated parent pins changed")
    if not devices.assignment_is_valid(body, wide.GPU_UUID):
        raise RuntimeError("declared GPU differs from the frozen assignment")
    if (body["planned_seconds"] != PLANNED_SECONDS
            or body["hard_cap_seconds"] != HARD_CAP_SECONDS
            or body["per_seed_seconds"] != PER_SEED_SECONDS
            or int(body["estimated_peak_bytes"]) != ESTIMATED_PEAK_BYTES
            or int(body["headroom_margin_bytes"]) != HEADROOM_MARGIN_BYTES):
        raise RuntimeError("execution budget or gate differs from freeze")
    if (body["scientific_claim"] is not False or body["test_read"] is not False
            or body["format"] != FORMAT or body["decision"]["text"] != DECISION_RULE):
        raise RuntimeError("frozen claims/decision rule changed")
    return body


def _parent_model(body, seed):
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file
    shared = recipe._shared()
    path, expected = fy_parent.pinned_parent(seed)
    if (sha256_file(path) != expected
            or body["arms"]["parent"]["checkpoints"][str(seed)]["sha256"] != expected):
        raise RuntimeError("parent checkpoint pin changed")
    saved = load_checkpoint(path)
    contract = saved["contract"]
    if (saved["updates"] != fy_parent.ENDPOINT or contract["seed"] != seed
            or contract["mode"] != "l6" or contract["data_identity"] != shared.V3_DATA_IDENTITY
            or contract["source_sha256"] != shared.V3_SOURCE_SHA256):
        raise RuntimeError("parent seed/data/source/endpoint changed")
    spec = body["arms"][CONTROL]["model"]["spec"]
    model = make_model("process", spec)
    model.load_state_dict(saved["model"], strict=True)
    return model, saved["model"], path, expected


def _arm_contract(body, name, parent, parent_sha):
    arm = body["arms"][name]
    return {"kind": "process", "model": arm["model"]["spec"],
            "data_identity": arm["train_data_identity"], "source_sha256": body["source_sha256"],
            "protocol_sha256": body["protocol_sha256"],
            "training_code_sha256": body["training_code_sha256"],
            "initialization": {"parent_run": str(fy_parent.MIGRATED_PARENT_RUN),
                               "parent_checkpoint": str(parent),
                               "parent_checkpoint_sha256": parent_sha,
                               "parent_endpoint_updates": fy_parent.ENDPOINT,
                               "description": ("registered v3-BD l6 endpoint, model-only "
                                               "migration; region-agnostic spec reused on the "
                                               "65x65 FY store")},
            "autoregression": {"physical_steps": 12,
                               "physical_weights": list(PHYSICAL_WEIGHTS),
                               "window_sha256": arm["train_windows"]["window_sha256"],
                               "excluded_sample_ids": arm["train_windows"]["excluded_sample_ids"]}}


def _check_provenance(provenance, seed, name, lead):
    if provenance["split"] != "val" or "test_read" in provenance:
        raise RuntimeError(f"seed {seed} {name} lead {lead} did not stay on val")
    if (provenance["n_evaluated"] != VAL_COHORTS[str(lead)]
            or provenance["n_available_windows"] != VAL_COHORTS[str(lead)]
            or provenance["lead_hours"] != [lead]):
        raise RuntimeError(f"seed {seed} {name} lead {lead} cohort differs from the frozen one")
    regions = {region["region"]: region for region in
               (provenance["boundary_scoring"] or {}).get("regions", [])}
    if regions.get(REGION, {}).get("n_grid_points") != FULL_GRID_POINTS:
        raise RuntimeError(f"seed {seed} {name} lead {lead} full region is not 65x65")
    climatology = provenance["climatology"]
    if (climatology["training_years"] != CLIMATOLOGY_YEARS
            or len(climatology["bucket_counts"]) != CLIMATOLOGY_BUCKETS):
        raise RuntimeError(f"seed {seed} {name} lead {lead} climatology is not the 48-bucket fit")


def _train_arm(output, body, seed, name, deadline):
    from training.r7_long_rollout_runner import fine_tune_long_rollout
    from training.r7_evaluate import evaluate_local
    from training.r7_arm_harness import sha256_file
    model, state, parent, parent_sha = _parent_model(body, seed)
    folder = output / f"seed{seed}/training/{name}"
    checkpoint, report = fine_tune_long_rollout(
        body["arms"][name]["train_manifest"], folder, model=model,
        contract=_arm_contract(body, name, parent, parent_sha), parent_weights=state,
        physical_weights=PHYSICAL_WEIGHTS, steps=STEPS, updates=UPDATES, lr=LR, warmup=WARMUP,
        seed=seed, device_name="cuda:0", deadline=deadline)
    if (report["updates_this_run"] != UPDATES or report["contract"]["steps"] != STEPS
            or report["data_identity"] != body["arms"][name]["train_data_identity"]
            or report["supervision"]["kind"] != "full_grid"):
        raise RuntimeError(f"seed {seed} arm {name} did not reach the frozen endpoint")
    receipt = {"seed": seed, "arm": name, "training_dir": str(folder),
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "training_report_sha256": sha256_file(folder / "training_report.json"),
               "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": report["elapsed_seconds"],
               "supervision": report["supervision"], "updates": UPDATES,
               "reasoning_steps": STEPS, "train_data_identity": report["data_identity"],
               "train_windows_sha256": body["arms"][name]["train_windows"]["window_sha256"],
               "evaluations": {}}
    for lead in EVALUATION_LEADS:
        destination = output / f"seed{seed}/evaluation/{name}/lead_{lead:03d}h"
        provenance = evaluate_local(
            str(manifest_path(name, "val")), output_dir=destination, checkpoint=checkpoint,
            lead_hours=(lead,), max_samples=10**9, device_name="cuda:0",
            reasoning_steps=REASONING_STEPS, boundary_margins=BOUNDARY_MARGINS, deadline=deadline)
        _check_provenance(provenance, seed, name, lead)
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
    receipts = {}
    for name, _ in ARMS:
        receipt = _train_arm(output, body, seed, name, deadline)
        receipt["owned_cuda_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved())
        wide.write_exclusive(output / f"seed{seed}_{name}_receipt.json", receipt)
        receipts[name] = receipt
    if receipts[CONTROL]["train_data_identity"] == receipts[TREATMENT]["train_data_identity"]:
        raise RuntimeError(f"seed {seed}: the arms trained on the same manifest identity")


def evaluate_exit(seeds, *, seeds_order=SEEDS):
    """The predeclared (a)-(d) conjunction; disagreement is never averaged away."""
    seeds = {str(key): value for key, value in seeds.items()}
    def skill(seed, lead):
        return seeds[str(seed)]["arms"][TREATMENT]["full_t2m_skill"][str(lead)]
    def delta(seed, lead):
        return seeds[str(seed)]["delta_treatment_minus_control"][str(lead)]
    a = all(skill(seed, 72) > 0.0 for seed in seeds_order)
    b = all(delta(seed, 72) < 0.0 for seed in seeds_order)
    c = all(skill(seed, 48) > 0.0 for seed in seeds_order)
    means = {lead: sum(delta(seed, lead) for seed in seeds_order) / len(seeds_order)
             for lead in (6, 12, 24)}
    d = all(value <= 0.0 for value in means.values())
    passed = bool(a and b and c and d)
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d),
            "seed_mean_delta_6_12_24": {str(lead): value for lead, value in means.items()},
            "all": passed,
            "verdict": "advance-to-independent-review" if passed
            else "close-fy-annual-coverage-branch"}


def bridge_verdict(control_rmse, reference, *, seeds_order=SEEDS):
    """Rule (e): an engineering bridge, explicitly not a scientific verdict."""
    control_rmse = {str(key): value for key, value in control_rmse.items()}
    leads = {}
    for lead in EVALUATION_LEADS:
        values = [control_rmse[str(seed)][str(lead)] for seed in seeds_order]
        mean = sum(values) / len(values)
        expected = reference[str(lead)]
        difference = mean - expected["mean"]
        leads[str(lead)] = {"control_mean": mean, "reference_mean": expected["mean"],
                            "difference": difference, "band": expected["band"],
                            "passed": abs(difference) <= expected["band"]}
    return {"scientific": False, "record_id": BRIDGE_RECORD,
            "note": ("normalization, val cases and update budget differ; a failure is "
                     "identity/engineering unresolved, not a scientific verdict"),
            "leads": leads, "passed": all(item["passed"] for item in leads.values())}


def _readings(output, body):
    from training.r7_arm_harness import sha256_file
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": body["protocol_sha256"], "decision_rule": DECISION_RULE,
                "factor": body["factor"]["name"], "updates": UPDATES,
                "declared_difference": ("the arms share the store, normalization, model spec, "
                                        "parent, seeds, val split and climatology; only the train "
                                        "manifest and its data identity differ"),
                "climatology_reference_sha256": sha256_file(D2_REFERENCE), "seeds": {}}
    for seed in SEEDS:
        entry = {"seed": seed, "arms": {}}
        for name, _ in ARMS:
            receipt = json.loads(
                (output / f"seed{seed}_{name}_receipt.json").read_text(encoding="utf-8"))
            arm = {"train_data_identity": receipt["train_data_identity"],
                   "full_t2m_rmse": {}, "full_t2m_skill": {}}
            for lead in EVALUATION_LEADS:
                table = Path(receipt["evaluations"][str(lead)]["dir"]) / "boundary_rmse.csv"
                value = wide._region_variable_rmse(table, REGION, "t2m")
                denominator = body["climatology_reference"]["leads"][str(lead)]["t2m_rmse"]
                arm["full_t2m_rmse"][str(lead)] = value
                arm["full_t2m_skill"][str(lead)] = 1.0 - (value / denominator) ** 2
            entry["arms"][name] = arm
        entry["delta_treatment_minus_control"] = {
            str(lead): (entry["arms"][TREATMENT]["full_t2m_rmse"][str(lead)]
                        - entry["arms"][CONTROL]["full_t2m_rmse"][str(lead)])
            for lead in EVALUATION_LEADS}
        readings["seeds"][str(seed)] = entry
    readings["exit_conditions"] = evaluate_exit(readings["seeds"])
    readings["bridge_e"] = bridge_verdict(
        {str(seed): readings["seeds"][str(seed)]["arms"][CONTROL]["full_t2m_rmse"]
         for seed in SEEDS}, body["bridge_e"]["reference"])
    return readings


def worker(output, phase, seed, deadline):
    from scripts.r7_m3_offline import deny_network
    deny_network()
    output = Path(output)
    if phase == "prepare":
        return _prepared_protocol(output)
    if phase == "archive":
        return dose.interior._archive(output, validate_protocol(output))
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
        raise RuntimeError("dual-card mode is driven by `r7_round.py stage-seeds`, not by main")
    return worker(args.out, args.worker, args.seed, args.deadline)


if __name__ == "__main__":
    main()
