"""The typed-evidence budget round's frozen registry, capacity assertions and (f) preflight.

This round trains three arms that differ only in whether the #79 typed-evidence pathway is on and,
if on, whether the four physical fields are routed per type or fused. The thing that must be frozen
is *which parent*, at *which digests*, under *which code identity*, with *which capacity
difference*, scored on *which case set*.

Why the capacity assertion is the load-bearing clause: ``C - B`` is the only pair whose parameter
count and forward FLOPs are bit-identical, so it is the only contrast that can attribute a delta to
the physical *type structure*. ``B - A`` adds the pathway's own 148,800 parameters, and that number
is reported rather than explained away. If a harness ever made A and B equal in capacity it would
have to remove the pathway, so the assertion is "A is strictly smaller", not "all three are equal".

``--preflight-only`` is the cheap gate the workflow runs before freezing anything. It prints one
JSON document and exits non-zero when (f) is false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_typed_budget_preflight as preflight_module  # noqa: E402

FORMAT = "r7-s3-typed-evidence-budget-protocol-v1"
STAGE = "S3-TYPED-EVIDENCE-BUDGET"

#: The three arms. ``A`` is the production configuration: ``typed_evidence_mode`` is absent, which
#: is what every registered run carries. ``B`` and ``C`` share the four fields, the shared patch
#: encoder and the parameter count; they differ in whether one fused projection consumes all four
#: types at every patch position or each type keeps its own projection.
INCUMBENT, GENERIC, TYPED = "incumbent", "generic", "typed"
ARMS = ((INCUMBENT, None), (GENERIC, "generic_fusion"), (TYPED, "typed_routing"))
ARM_NAMES = tuple(name for name, _ in ARMS)

SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 48, 72)
STEP_HOURS = 6

#: 32,132 updates at batch 4 is exactly 4.000 epochs over the 32,131 usable windows -- the same
#: rung the registered throughput ladder calls p4000, whose 72 h t2m skill is +0.230544. Every
#: previously closed physics arm ran at 400-4,000 updates (0.01-0.5 epoch), so this is the first
#: budget at which "the pathway has no increment" would be a statement about the mechanism rather
#: than about the dose.
UPDATES = 32_132
BATCH_SIZE = 4
STEPS = 4
LR = 2e-5
WARMUP = 10
WEIGHT_DECAY = 1e-4
CLIP = 1.0
#: The registered autoregression weights, copied verbatim from the ladder's protocol. Not a knob.
PHYSICAL_WEIGHTS = (1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
CHECKPOINT_EVERY = 5_000

#: The clause tables are read from ``boundary_rmse.csv``'s ``region == "full"`` rows, and
#: ``evaluate_local`` writes that file **only** when ``boundary_margins`` is given. Omitting it does
#: not error at evaluation time: it silently omits the file and every reader fails later, after the
#: GPU has been paid for (the recorded probe failure was exactly this ``FileNotFoundError``).
#: ``full`` is the undivided grid, so this adds no region stratification to any clause -- margin 1 is
#: the repo-wide convention (``r7_s3_curve_point_protocol.BOUNDARY_MARGINS``), and its ``full`` rows
#: equal ``rmse.csv`` to 1 ULP (measured: 2.665020267467307 vs ...67 for t2m at lead 24 h).
BOUNDARY_MARGINS = (1,)
REGION = "full"

TRAIN_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/train.jsonl"
VAL_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/val.jsonl"
STORE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr"
#: The 65x65 typed-evidence sidecar. Its data identity must equal the FY store's own train identity,
#: otherwise the field standardisation was fitted on a different dataset.
SIDECAR_RUN = ROOT / "outputs/r7_s3_typed_evidence_fy65_20261103_attempt01"
SIDECAR_DATA_IDENTITY = "81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0"

#: The registered 1600-update parent, migrated under the current model revision. Every seeded
#: ladder rung starts here, so the three arms share one initialization by construction.
PARENT_RUN = ROOT / "outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01"
PARENT_MIGRATION_RECEIPT = PARENT_RUN / "migration_receipt.json"
PARENT_UPDATES = 1_600
PARENT_MODEL_CODE_SHA = "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324"

#: The round the two typed arms were already priced at in the wrong dose. This round reuses the same
#: arm pair, so the similarity must be declared rather than hidden.
CLOSED_ROUND_RECORD = "s3-typed-spatial-result"
CLOSED_ROUND_UPDATES = 2_400

#: Measured on this store: 1.1536 s/update at batch 4 on the p4000 rung (docs/.../P4000_RESULT.md:107).
MEASURED_SECONDS_PER_UPDATE = 1.1536
#: The measured five-lead scoring cost of one arm: sum over leads 6/12/24/48/72 of
#: ``provenance.json``'s elapsed on the spectral round (231.7 + 308.0 + 426.6 + 711.1 + 1,004.9 s),
#: at 1,458/1,457/1,455/1,451/1,447 cases. Not a per-lead figure, despite the constant's history.
MEASURED_SECONDS_PER_EVALUATION = 2_682.3
#: The duration envelope, derived from **two** bounds that must both hold.
#:
#: *Lower*: a seed worker runs **three** arms and scores five leads each before exiting, so it needs
#: 3 x (32,132 x 1.1536 + 2,682.3) = 119,254 s = 33.13 h measured. A shorter envelope does not
#: degrade gracefully -- ``_check_deadline`` raises inside the update loop and truncates the arm.
#: *Upper*: three seeds on the two frozen cards need two waves, and the whole round must fit the
#: brief's §3 hard cap of 144 GPU-h, i.e. 72 h of wall clock over two cards. So one seed's envelope
#: cannot exceed 72/2 = 36 h once the ~0.4 h of prepare/archive/reading is reserved.
#: 35 h sits inside both bounds with 5.6 % margin over the measured need.
#:
#: ``PLANNED``/``HARD_CAP`` are that same §3 budget (soft 96 GPU-h, hard 144 GPU-h) as wall clock.
#: The soft figure is expected to be overdrawn -- 9 arms x 11.04 h = 99.3 GPU-h exceeds it -- and a
#: soft overrun continues and is recorded; only the hard cap truncates, and a truncated worker is
#: not a pass.
PLANNED_SECONDS = 172_800.0
HARD_CAP_SECONDS = 259_200.0
PER_SEED_SECONDS = 126_000.0

ESTIMATED_PEAK_BYTES = 9_728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
DEVICE_MODE = "dual"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"

#: The (e) guard: p4000 -> p1600 already moved z500@72h from 427.35 to 445.78 (+4.3 percent), so a
#: budget increase by itself is known to push this metric the wrong way. The arm is allowed another
#: 5 percent, not an unbounded drift.
Z500_72H_REFERENCE = 427.35
Z500_72H_TOLERANCE = 0.05
#: A lead counts toward (a)/(b) only when the paired delta is at least this much, so a statistically
#: resolvable but operationally meaningless sign does not carry the claim alone.
MINIMUM_LEAD_GAIN_K = 0.005
MINIMUM_CONCORDANT_LEADS = 2

INDEPENDENT_ARM_ARGUMENT = (
    "The three arms differ only in the #79 typed-evidence pathway. A carries the production spec: "
    "the typed_evidence_mode key is absent, exactly as every registered R7 run. B and C both carry "
    "the same four physical fields (850 hPa divergence, 850 hPa vorticity, 850 hPa temperature "
    "advection, 850-500 hPa static stability), the same shared patch encoder and the same payload, "
    "and their parameter count and forward FLOPs are bit-identical, so B->C isolates whether the "
    "fields must be routed by type. A has no pathway and is therefore strictly smaller by the "
    "pathway's own 148,800 parameters; A->B prices the pathway including that capacity, so a B-A "
    "win may not be described as capacity-free.")

LIMITATIONS = [
    "B and C are capacity-matched but their computation graphs are not identical: generic mode "
    "fuses at every patch position and then pools, typed mode pools per type and then projects. "
    "The contrast is therefore 'where the fusion happens', not 'whether the physics is present'",
    "A -> B adds the pathway's 148,800 parameters (+4.5 percent of the trunk). That difference is "
    "inherent to switching the pathway on and cannot be removed; a (b) win is reported with it",
    "the same arm pair was already priced at 400 updates (#79) and 2,400 updates "
    "(s3-typed-spatial-result, B->C |delta rmse| <= 3.2e-6 K). This round changes the dose, not "
    "the mechanism, and a positive result here does not retroactively invalidate those readings",
    "three seeds are consistency evidence, not a significance test; the block bootstrap interval "
    "is over initialisation times, not over seeds",
    "z500 at 72 h already worsened from 427.35 to 445.78 between p4000 and p1600, so guard (e) can "
    "fail for reasons unrelated to the pathway; a failure limits the claim to t2m and must say so",
    "development screening on one 65x65 instance (2000-2021 train / 2022 val, 17 channels); the "
    "129x129 wide instance is a different store and is not covered",
    "the 2023 test split stays sealed; r = 0",
    "GPU runs are co-resident; latency and memory observations include neighbour load",
]


def _arm_path(relative: Path) -> str:
    return str(relative.relative_to(ROOT))


def registered_spec() -> dict:
    """The production process spec, read from a registered checkpoint's own contract.

    Hand-writing the spec here would let this module describe a model the registered weights are
    not: the trunk has 28 keys and only some of them are obvious. The Stage A ``off`` arm is the
    closest registered run (same parent, same 4,000-update fine-tune, pathway off), so its contract
    is the reference and every deviation would be a difference the arm contrast did not declare.
    """
    from training.r7_experiment import load_checkpoint

    reference = (ROOT / "outputs/r7_s3_spectral_geometry_a_20261112_attempt02"
                 / "seed41/training/off/update_0004000.pt")
    if not reference.is_file():
        raise FileNotFoundError(f"the production reference checkpoint is missing: {reference}")
    spec = dict(load_checkpoint(reference)["contract"]["model"])
    for banned in ("typed_evidence_mode", "typed_evidence"):
        spec.pop(banned, None)
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the reference spec is not the registered dim-192 depth-4 trunk")
    if int(spec.get("in_channels", 0)) != 17 or int(spec.get("out_channels", 0)) != 17:
        raise RuntimeError("the reference spec is not the 17-channel instance")
    return spec


def arm_spec(mode):
    """The production spec for one arm, with the pathway bound in only when a mode is given.

    The spec is read from a registered checkpoint (``registered_spec``) rather than reconstructed,
    so the incumbent arm is the model that every registered run trained. A mode of ``None`` leaves
    the key absent, which is what all 193 registered runs carry.
    """
    spec = registered_spec()
    if mode is not None:
        if mode not in ("generic_fusion", "typed_routing"):
            raise ValueError(f"unknown typed-evidence mode {mode!r}")
        spec["typed_evidence_mode"] = mode
    return spec


def payload() -> tuple[dict, dict]:
    """The published 65x65 typed-evidence sidecar payload and its own metadata."""
    from data.preprocess.r7_typed_evidence_scale import load_typed_evidence_scale
    import numpy as np
    import zarr

    meta = load_typed_evidence_scale(SIDECAR_RUN)
    root = zarr.open_group(str(meta["store"]), mode="r")
    channels = list(root.attrs["channels"])
    if channels != list(meta["channels"]):
        raise ValueError("the sidecar channel order does not match the store")
    import json as _json
    declared = _json.loads((SIDECAR_RUN / "typed_evidence_scale.json").read_text(encoding="utf-8"))
    if declared.get("data_identity") != SIDECAR_DATA_IDENTITY:
        raise RuntimeError("the typed-evidence sidecar is not the pinned 65x65 build")
    body = {"channels": channels,
            "denorm_mean": [float(v) for v in np.asarray(root["normalization_mean"][:])],
            "denorm_std": [float(v) for v in np.asarray(root["normalization_std"][:])],
            "field_scale": [float(v) for v in meta["runtime_scale"]],
            "field_mean": [float(v) for v in meta["runtime_mean"]],
            "field_std": [float(v) for v in meta["runtime_std"]],
            "field_active": [bool(v) for v in meta["active_mask"]]}
    return meta, body


def parent_pin(seed):
    """The migrated 1600-update parent for one seed, re-checked against its migration receipt."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import model_code_digest

    if seed not in SEEDS:
        raise ValueError(f"declared seeds are {SEEDS}; {seed} is not among them")
    receipt = json.loads(PARENT_MIGRATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("current_model_code_sha256") != model_code_digest():
        raise RuntimeError("the pinned parent migration was produced under a different model "
                           "revision; every checkpoint would be unloadable")
    record = (receipt.get("seeds") or {}).get(str(seed)) or {}
    path = PARENT_RUN / f"seed{seed}" / f"parent_update_{PARENT_UPDATES:07d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"the migrated parent is missing: {path}")
    observed = sha256_file(path)
    if record and record.get("migrated_sha256") not in (None, observed):
        raise RuntimeError(f"parent seed {seed} drifted from its migration receipt")
    return path, observed


def capacity() -> dict:
    """The three arms' parameter counts, and the two capacity claims the contrast depends on.

    (1) B and C must be equal: same shared encoder, same payload, only the projection differs, and
    the two projection shapes carry the same number of weights. (2) A must be strictly smaller by
    exactly the pathway's own tensors, so "switching the physics on costs this much" is a measured
    number rather than a claim.
    """
    from training.r7_experiment import make_model

    meta, body = payload()
    counts = {}
    for name, mode in ARMS:
        spec = arm_spec(mode)
        if mode is not None:
            spec = dict(spec, typed_evidence=dict(body))
        model = make_model("process", spec)
        counts[name] = int(sum(parameter.numel() for parameter in model.parameters()))
    pathway = counts[GENERIC] - counts[INCUMBENT]
    generic_equals_typed = counts[GENERIC] == counts[TYPED]
    incumbent_is_smaller = counts[INCUMBENT] < counts[GENERIC]
    return {
        "passed": bool(generic_equals_typed and incumbent_is_smaller and pathway > 0),
        "parameters": counts,
        "pathway_parameters": pathway,
        "pathway_fraction_of_incumbent": pathway / counts[INCUMBENT],
        "generic_equals_typed": bool(generic_equals_typed),
        "incumbent_strictly_smaller": bool(incumbent_is_smaller),
        "rule": ("B and C must be parameter-identical so B->C is attributable to the type "
                 "structure; A must be strictly smaller so the pathway's own capacity cost is a "
                 "reported number, never an unstated confound"),
        "sidecar_identity": SIDECAR_DATA_IDENTITY,
        "sidecar_fields": list(meta["fields"]),
    }


def preflight() -> dict:
    """Exit (f): the six clauses live in ``r7_s3_typed_budget_preflight`` and run by name."""
    clauses = preflight_module.evaluate()
    parents = {}
    for seed in SEEDS:
        path, digest = parent_pin(seed)
        parents[str(seed)] = {"path": _arm_path(path), "sha256": digest}
    return {
        "format": FORMAT,
        "stage": STAGE,
        "store": _arm_path(STORE),
        "val_manifest": _arm_path(VAL_MANIFEST),
        "parents": parents,
        "clauses": clauses["clauses"],
        "f_preflight_passed": clauses["f_preflight_passed"],
        "scientific_claim": False,
        "test_read": False,
        "r": 0,
        "gpu_hours": 0.0,
        "limitations": ["this is a preflight, not a reading: it synthesises inputs and touches no "
                        "forecast, no validation case and no test case"],
    }


def arm_registry() -> list[dict]:
    """The nine arm records: three modes x three seeds, each with its spec and parent pin."""
    _, body = payload()
    records = []
    for seed in SEEDS:
        path, digest = parent_pin(seed)
        for name, mode in ARMS:
            spec = arm_spec(mode)
            records.append({
                "arm": name, "seed": seed, "typed_evidence_mode": mode,
                "model": {"spec": spec},
                "typed_evidence": None if mode is None else {
                    "fields": list(body["field_active"]),
                    "sidecar": _arm_path(SIDECAR_RUN),
                    "data_identity": SIDECAR_DATA_IDENTITY},
                "parent": {"checkpoint": _arm_path(path), "sha256": digest,
                           "endpoint_updates": PARENT_UPDATES},
            })
    if len(records) != len(ARMS) * len(SEEDS):
        raise RuntimeError("the registry does not hold the predeclared nine arms")
    return records


def _frozen_constants() -> dict:
    return {
        "format": FORMAT, "stage": STAGE,
        "arms": [{"arm": name, "typed_evidence_mode": mode} for name, mode in ARMS],
        "seeds": list(SEEDS),
        "lead_hours": list(LEADS),
        "step_hours": STEP_HOURS,
        "updates": UPDATES, "batch_size": BATCH_SIZE, "steps": STEPS,
        "lr": LR, "warmup_updates": WARMUP, "weight_decay": WEIGHT_DECAY, "clip": CLIP,
        "physical_weights": list(PHYSICAL_WEIGHTS),
        "checkpoint_every": CHECKPOINT_EVERY,
        "train_manifest": _arm_path(TRAIN_MANIFEST),
        "val_manifest": _arm_path(VAL_MANIFEST),
        "store": _arm_path(STORE),
        "scientific_claim": False, "test_read": False, "r": 0,
        "independent_arm_argument": INDEPENDENT_ARM_ARGUMENT,
        "closed_dose": {"record_id": CLOSED_ROUND_RECORD, "updates": CLOSED_ROUND_UPDATES,
                        "rule": "the same arm pair was already priced at this dose; this round "
                                "changes the dose only and does not invalidate that reading"},
        "estimation": {
            "measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
            "measured_seconds_per_evaluation": MEASURED_SECONDS_PER_EVALUATION,
            "measured_on": "batch 4 on the p4000 rung of the throughput ladder",
            "updates_total": UPDATES * len(SEEDS) * len(ARMS),
            "estimate_seconds": round(UPDATES * len(SEEDS) * len(ARMS)
                                      * MEASURED_SECONDS_PER_UPDATE
                                      + len(SEEDS) * len(ARMS) * MEASURED_SECONDS_PER_EVALUATION, 1),
            "per_seed_seconds_estimate": round(len(ARMS) * (UPDATES * MEASURED_SECONDS_PER_UPDATE
                                                           + MEASURED_SECONDS_PER_EVALUATION), 1),
            "per_seed_note": ("one seed worker runs all three arms and scores five leads each, so "
                              "this -- not one arm -- is the quantity PER_SEED_SECONDS must exceed"),
            "note": "an estimate from a measured per-update rate, not a measurement of this round",
        },
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS,
                    "soft_rule": ("soft overrun continues and is recorded; only the hard cap "
                                  "truncates a worker, and a truncated worker is not a pass"),
                    "decision": "0030"},
        "exits": {
            "a": ("C - B is negative (lower t2m RMSE) on at least two leads, agrees in sign on all "
                  "three seeds, and the block bootstrap interval excludes zero"),
            "b": "B - A is negative on at least two leads",
            "c": ("if (a) holds and (b) does not, only the type structure may be claimed, never "
                  "that the physics adds skill"),
            "d": "the same arm's increment keeps its direction between 4 and 16 epochs",
            "e": "z500 at 72 h must not worsen by more than 5 percent against the A arm",
            "f": "this preflight, every clause true",
            "reading_rule": ("A clause with no arm contrast to compute is null (untested), never "
                             "false and never true. A single-seed sign flip is reported but never "
                             "counted. The (a) pair is the only one whose capacity is matched."),
        },
        "thresholds": {
            "minimum_lead_gain_k": MINIMUM_LEAD_GAIN_K,
            "minimum_concordant_leads": MINIMUM_CONCORDANT_LEADS,
            "z500_72h_reference": Z500_72H_REFERENCE,
            "z500_72h_tolerance": Z500_72H_TOLERANCE,
        },
        "limitations": LIMITATIONS,
    }


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol, including the nine-arm registry and the capacity numbers."""
    from training.r7_device_selection import protocol_fields
    from training.r7_experiment import canonical_digest

    body = {**_frozen_constants(),
            "registry": arm_registry(),
            "gpu_uuid": GPU_UUID,
            "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **protocol_fields(DEVICE_MODE, single_uuid=GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata and the arm registry prepared in a bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    _write_exclusive(Path(output) / "preparation_protocol.json", body)
    return body


def train_windows() -> dict:
    """The train windows the training contract must declare, discovered from the manifest.

    ``fine_tune_long_rollout`` refuses a contract whose ``autoregression`` block does not carry the
    window digest, and its dataset re-checks the exclusions against **its own** discovery. That
    discovery is the 12-step long-rollout one, not the plain two-frame one: the two yield different
    ``window_sha256`` values and different exclusion lists (the long-rollout list names eleven
    boundary samples, one per physical step, where the plain one names a single ``t+12`` sample).
    Using the plain discovery here would build a contract the trainer rejects. Metadata only: this
    reads the manifest, never the store's fields.
    """
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows

    found = preflight_long_rollout_windows(str(TRAIN_MANIFEST), physical_steps=12)
    return {"window_sha256": found["window_sha256"],
            "excluded_sample_ids": list(found["excluded_sample_ids"])}


def prepared_protocol(output):
    """Add the live identities and the measured (f); a failing preflight writes no protocol."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, dataset_identity, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest

    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    measurements = preflight()
    if not measurements["f_preflight_passed"]:
        raise RuntimeError(f"exit (f) failed before any GPU second was spent: "
                           f"{ {name: clause['passed'] for name, clause in measurements['clauses'].items()} }")
    train_identity, _ = dataset_identity(TRAIN_MANIFEST)
    val_identity, _ = dataset_identity(VAL_MANIFEST)
    if train_identity != SIDECAR_DATA_IDENTITY:
        raise RuntimeError(f"the typed-evidence sidecar was fitted on data identity "
                           f"{SIDECAR_DATA_IDENTITY}, the train manifest is {train_identity}")
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "train_data_identity": train_identity,
            "val_data_identity": val_identity,
            "train_windows": train_windows(),
            "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
            "val_manifest_sha256": sha256_file(VAL_MANIFEST),
            "preflight": measurements,
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()}}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    _write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output):
    """Re-check every frozen identity, the parent pins and the preflight."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest

    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest mismatch")
    if model_code_digest() != body["model_code_sha256"]:
        raise RuntimeError("the model revision moved after freeze")
    if model_code_digest() != PARENT_MODEL_CODE_SHA:
        raise RuntimeError("model_code_sha256 is not the parent pin any more")
    if training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("the training revision moved after freeze")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution source changed: {name}")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    if body.get("test_read") is not False or body.get("scientific_claim") is not False:
        raise RuntimeError("this round claims nothing and never reads the test split")
    measured = body["preflight"]
    if measured.get("f_preflight_passed") is not True:
        raise RuntimeError("the frozen preflight did not pass")
    registry = body["registry"]
    if len(registry) != len(ARMS) * len(SEEDS):
        raise RuntimeError(f"the frozen registry holds {len(registry)} arms, not "
                           f"{len(ARMS) * len(SEEDS)}")
    for record in registry:
        if sha256_file(ROOT / record["parent"]["checkpoint"]) != record["parent"]["sha256"]:
            raise RuntimeError(f"arm {record['arm']}/seed{record['seed']} parent moved")
    if sha256_file(VAL_MANIFEST) != body["val_manifest_sha256"]:
        raise RuntimeError("the validation manifest moved after freeze")
    return body


def execution_files():
    """Decision 0067: this round's import closure, from a named entry file."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files("scripts/study_r7_s3_typed_budget.py",
                                    extra=("scripts/r7_s3_typed_budget_protocol.py",
                                           "scripts/r7_s3_typed_budget_preflight.py",
                                           "scripts/r7_typed_budget_verdict.py"))


def _write_exclusive(path: Path, body: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=False, allow_nan=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)
    if not arguments.preflight_only:
        raise SystemExit("this module is the protocol layer; the round runs through "
                         "`r7_round.py stage`. Use --preflight-only for the (f) gate.")
    from scripts.r7_m3_offline import deny_network
    deny_network()
    measurements = preflight()
    payload_body = {"format": FORMAT, "stage": STAGE, "scientific_claim": False,
                    "test_read": False, "gpu_hours": 0.0, **measurements}
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(payload_body, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"f_preflight_passed": measurements["f_preflight_passed"],
                      "clauses": {name: clause["passed"]
                                  for name, clause in measurements["clauses"].items()},
                      "parameters": measurements["clauses"]
                      ["capacity_is_matched_where_it_must_be"]["parameters"],
                      "out": str(arguments.out) if arguments.out else None}, indent=2))
    return 0 if measurements["f_preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
