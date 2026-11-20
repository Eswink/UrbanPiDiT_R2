"""The in-protocol round's frozen registry, capacity assertions and duration envelope.

This round answers one question that no earlier round answered: does the #79 typed-evidence
pathway change a score when ``typed`` and ``generic`` are compared *inside one protocol* at a dose
where the model is still improving? The previous round priced the same three arms at 32,132 updates
but truncated every ``typed`` arm on a per-seed envelope derived from the **incumbent** rate, so
``C - B`` was never scored at any dose above 2,400 updates.

Two things are frozen here that the earlier protocol got wrong.

**The envelope uses the treated rate.** ``PER_SEED_SECONDS`` is derived from the measured treated
rate (1.3886/1.4266 s/update) rather than the incumbent rate (1.1536). The earlier round derived it
from the incumbent rate and its guard test read the same constant, so the test was green by
construction while the run was structurally doomed. :func:`envelope` recomputes the requirement from
the treated constant and :data:`ENVELOPE_MARGIN` states the multiple, so a test can compare two
*independent* numbers.

**The criterion is a family, not one variable.** The previous round graded t2m against a 0.005 K
floor; the measured effect there was a third of the floor while z500 moved 13 of 15 seed-by-lead
cells with an interval excluding zero. Grading one variable chosen before the data existed is
honest; grading the variable that happens to look best is not. So the family is partitioned by
**mechanism**: ``t500,t850,u850,v850`` are the four channels :func:`typed_local_fields` consumes,
and that set is fixed by the model's own declaration, not by a reading.

``--preflight-only`` is the cheap gate the executor runs before freezing anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_in_protocol_preflight as preflight_module  # noqa: E402

FORMAT = "r7-s3-typed-evidence-in-protocol-v1"
STAGE = "S3-TYPED-EVIDENCE-IN-PROTOCOL"

INCUMBENT, GENERIC, TYPED = "incumbent", "generic", "typed"
ARMS = ((INCUMBENT, None), (GENERIC, "generic_fusion"), (TYPED, "typed_routing"))
ARM_NAMES = tuple(name for name, _ in ARMS)
TREATED_ARMS = (GENERIC, TYPED)

SEEDS = (41, 42, 43)
LEADS = (6, 12, 24, 48, 72)
STEP_HOURS = 6

#: 32,132 updates at batch 4 is exactly 4.000 epochs over the 32,131 usable windows -- the rung the
#: throughput ladder calls p4000. Every closed physics arm ran at 0.075-0.5 epoch, and the 2,400
#: typed comparison measured |delta rmse| <= 3.2e-6 K, so this is the first dose at which a null
#: would be about the mechanism rather than about the dose.
UPDATES = 32_132
BATCH_SIZE = 4
STEPS = 4
LR = 2e-5
WARMUP = 10
WEIGHT_DECAY = 1e-4
CLIP = 1.0
PHYSICAL_WEIGHTS = (1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
CHECKPOINT_EVERY = 5_000

#: ``evaluate_local`` writes ``boundary_rmse.csv`` **only** when ``boundary_margins`` is given, and
#: omitting it fails later rather than at evaluation time. ``full`` is the undivided grid.
BOUNDARY_MARGINS = (1,)
REGION = "full"

TRAIN_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/train.jsonl"
VAL_MANIFEST = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/manifests_treatment/val.jsonl"
STORE = ROOT / "outputs/r7_s3_fy_multiyear_20261023_attempt01/store/cache.zarr"
SIDECAR_RUN = ROOT / "outputs/r7_s3_typed_evidence_fy65_20261103_attempt01"
SIDECAR_DATA_IDENTITY = "81307008b572e3ff032d5251e788c582f339b0e3dbbc3fea4226d44c6a335af0"
SOURCE_SHA256 = "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"

PARENT_RUN = ROOT / "outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01"
PARENT_MIGRATION_RECEIPT = PARENT_RUN / "migration_receipt.json"
PARENT_UPDATES = 1_600
PARENT_MODEL_CODE_SHA = "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324"

#: The two doses this same arm pair was already priced at. Declaring both keeps a citation from
#: presenting this round as a re-run: it changes the dose, not the mechanism.
CLOSED_DOSES = ({"record_id": "s2-79-typed-evidence", "updates": 400},
                {"record_id": "s3-typed-spatial-result", "updates": 2_400})

#: The treated rate is the **measured** one, and it is an upper bound on the three measurements
#: available: generic seed41 1.4043, seed42 1.4312, seed43 1.4271 s/update over a full 32,132-update
#: run, and the truncated typed arms' last-4,000-update marginal rates 1.4565/1.4618/1.4618. 1.47
#: sits above all six so the envelope is not sized on the friendliest arm. This constant is what
#: :func:`envelope` reads; the incumbent rate is kept only to price A, never to size the envelope.
TREATED_SECONDS_PER_UPDATE = 1.47
INCUMBENT_SECONDS_PER_UPDATE = 1.12
#: Five-lead scoring of one arm, measured: 3,176.62 / 3,013.97 / 2,987.30 s. Rounded up.
MEASURED_SECONDS_PER_EVALUATION = 3_200.0
#: The requirement is recomputed, never asserted against itself: 3 arms x (updates x treated rate +
#: scoring) is what one seed worker must complete before it exits.
ENVELOPE_MARGIN = 2.0

#: Soft budget continues and records an overrun; only the envelope truncates. Nine arms cost about
#: 116.7 GPU-h of compute, which is 58.4 h of wall clock spread over two co-resident cards. The soft
#: figure is stated in wall clock because that is what the wrapper measures, and an overrun is
#: expected to be small and recorded rather than fatal.
#:
#: **On the duration authorization.** The user authorized no hard cap on training time. A literal
#: infinity is not reachable without editing ``training/``: ``_validate_options`` rejects a
#: non-finite deadline, the driver refuses ``None``, and ``subprocess.run`` is handed a finite
#: timeout. Editing ``training/`` would change ``training_code_sha256`` and invalidate the parent
#: pin, so this round instead makes the envelope large enough that a truncation cannot happen for a
#: reason this round could have measured -- and, if the rate is still misjudged, refuses early at
#: update 400 rather than burning ten hours first. :func:`envelope` recomputes the requirement and
#: the driver checks it before the first update.
PLANNED_SECONDS = 210_240.0
#: Two waves (three seeds on two cards) each inside its own envelope, plus the CPU stages. The
#: requirement is 151,302 s per seed at the treated rate; 310,000 s is 2.05x it, and the cap must
#: cover two such waves.
HARD_CAP_SECONDS = 660_000.0
PER_SEED_SECONDS = 310_000.0

ESTIMATED_PEAK_BYTES = 9_728 * 2**20
HEADROOM_MARGIN_BYTES = 512 * 2**20
DEVICE_MODE = "dual"
GPU_UUID = "GPU-408ad137-a60e-6a04-e2c8-22f5f64e5e3b"

#: The two families, partitioned by mechanism rather than by observation. FAMILY_PRIMARY is the set
#: :func:`typed_local_fields` reads; it is fixed in ``model/r7_process_tensor_diagnostics`` and does
#: not depend on any reading. FAMILY_SECONDARY is every other channel.
FAMILY_PRIMARY_CHANNELS = ("t500", "t850", "u850", "v850")
FAMILY_PRIMARY_LABEL = "physical-pathway-inputs"
FAMILY_SECONDARY_LABEL = "other-channels"

#: The family-wise error rate and the bootstrap shape. ``block_length`` is in cases: the validation
#: split is 6-hourly, so 28 cases is seven days and 1,447 cases give 51 overlapping blocks, thirteen
#: times the ``MINIMUM_BLOCKS`` the instrument requires.
ALPHA_FAMILY = 0.05
BLOCK_LENGTH_CASES = 28
RESAMPLES = 4_000
RESAMPLE_SEED = 41

#: The gain floor, now dimensionless. ``delta skill = (mse_control - mse_treatment) / mse_climatology``
#: is comparable across 17 variables whose units span K, m/s, Pa, kg/kg and m**2 s**-2; the previous
#: round's 0.005 K could only ever be meaningful for temperature. 0.001 is a tenth of a percent of
#: climatology error: small enough that a real mechanism clears it, large enough that a resolvable
#: sign alone does not.
MINIMUM_CELL_GAIN_SKILL = 0.001
MINIMUM_CELLS_PER_FAMILY = 1
MINIMUM_LEADS_PER_FAMILY = 2
#: A family whose intersection of case keys falls below this is not evaluable as declared.
MINIMUM_FAMILY_CASES = 400

Z500_72H_TOLERANCE = 0.05
MAX_WORSE_CELL_FRACTION = 0.5

INDEPENDENT_ARM_ARGUMENT = (
    "The three arms differ only in the #79 typed-evidence pathway. A carries the production spec: "
    "the typed_evidence_mode key is absent, exactly as every registered R7 run. B and C both carry "
    "the same four physical fields (850 hPa divergence, 850 hPa vorticity, 850 hPa temperature "
    "advection, 850-500 hPa static stability), the same shared patch encoder and the same payload, "
    "and their parameter count and forward FLOPs are bit-identical, so B->C isolates whether the "
    "fields must be routed by type. A is strictly smaller by the pathway's own 148,800 parameters, "
    "so A->B prices the pathway *including* that capacity and may never be described as "
    "capacity-free. Every arm is trained inside this protocol from the same pinned parent: no arm "
    "is reused from an earlier round, which is what makes C->B in-protocol rather than a "
    "cross-protocol comparison.")

LIMITATIONS = [
    "B and C are capacity-matched but their computation graphs are not identical: generic mode fuses "
    "at every patch position and then pools, typed mode pools per type and then projects. The "
    "contrast is therefore 'where the fusion happens', not 'whether the physics is present'",
    "A -> B adds the pathway's 148,800 parameters (+4.5 percent of the trunk). That difference is "
    "inherent to switching the pathway on and cannot be removed inside this design; a B-A win is "
    "reported with it and is descriptive, not attributable",
    "the same arm pair was already priced at 400 updates (#79, t2m -0.0533 K on 6 h and 12 h) and "
    "2,400 updates (s3-typed-spatial-result, B->C |delta rmse| <= 3.2e-6 K). This round changes the "
    "dose, not the mechanism, and a positive result here does not retroactively invalidate those",
    "each lead evaluates a slightly different case set (1,458/1,457/1,455/1,451/1,447 for 6/12/24/"
    "48/72 h). The families are therefore built on the 1,447-case intersection shared by all five "
    "leads; the 11 cases only visible at short leads are excluded from every interval",
    "the climatology denominator is a train-only month-hour grid mean held fixed across arms and "
    "resamples; uncertainty in that denominator is not propagated into the intervals",
    "the simultaneous interval is over initialisation times and covers the family-wise error rate; "
    "it does not include seed-to-seed variability, which the three-seed sign rule covers separately",
    "three seeds are consistency evidence, not a significance test",
    "z500 at 72 h already worsened from 427.35 to 445.78 between p4000 and p1600 in the throughput "
    "ladder, so the dynamics guard can fail for reasons unrelated to the pathway; a failure limits "
    "the claim and must say so",
    "development screening on one 65x65 instance (2000-2021 train / 2022 val, 17 channels); the "
    "129x129 wide instance is a different store and is not covered",
    "the 2023 test split stays sealed; r = 0",
    "GPU runs are co-resident; latency and memory observations include neighbour load",
]


def _arm_path(relative: Path) -> str:
    return str(relative.relative_to(ROOT))


def registered_spec() -> dict:
    """The production process spec, read from a registered checkpoint's own contract.

    Hand-writing the spec would let this module describe a model the registered weights are not: the
    trunk has 28 keys and only some are obvious. Reading it from a registered checkpoint keeps the
    ``incumbent`` arm the model every registered run trained.
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
    """The production spec for one arm, with the pathway bound in only when a mode is given."""
    spec = registered_spec()
    if mode is not None:
        if mode not in ("generic_fusion", "typed_routing"):
            raise ValueError(f"unknown typed-evidence mode {mode!r}")
        spec["typed_evidence_mode"] = mode
    return spec


def payload() -> tuple[dict, dict]:
    """The sidecar payload; the implementation lives with the preflight that needs it by name."""
    return preflight_module.payload()


def store_channels() -> list[str]:
    """The store's pinned channel order."""
    return preflight_module.store_channels()


def families() -> dict:
    """The two families, partitioned by mechanism; see the preflight module."""
    return preflight_module.families()


def capacity() -> dict:
    """The three arms' parameter counts and the two claims the contrast depends on."""
    return preflight_module.capacity()


def envelope() -> dict:
    """One seed's requirement recomputed from the treated rate, versus the frozen envelope."""
    return preflight_module.envelope()


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


def preflight() -> dict:
    """Exit (f): the six clauses live in the preflight module and run by name."""
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
        "families": families(),
        "envelope": envelope(),
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
        "closed_doses": list(CLOSED_DOSES),
        "families": families(),
        "envelope": envelope(),
        "budgets": {"planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
                    "per_seed_seconds": PER_SEED_SECONDS,
                    "soft_rule": ("a soft overrun continues and is recorded; only the envelope "
                                  "truncates a worker, and a truncated worker is not a pass"),
                    "decision": "0116"},
        "exits": {
            "a": ("C - B clears the primary family: at least one cell's simultaneous interval "
                  "excludes zero in the improving direction, those cells span at least two leads, "
                  "every clearing cell agrees in sign on all three seeds, and the largest clearing "
                  "cell reaches the declared skill floor"),
            "b": "B - A under the same rule, reported descriptively because A is 148,800 parameters smaller",
            "g": ("the dynamics guard: no treated arm may worsen z500 at 72 h by more than 5 percent "
                  "against its own seed's incumbent, and no arm may be worse than A on more than "
                  "half of all family cells"),
            "f": "this preflight, every clause true",
            "reading_rule": ("A family with no arm contrast to compute is null (untested), never "
                             "false and never true. A single-seed sign flip is reported but never "
                             "counted. The family's own case-key intersection is reported because "
                             "each lead evaluates a slightly different set."),
        },
        "thresholds": {
            "alpha_family": ALPHA_FAMILY, "block_length_cases": BLOCK_LENGTH_CASES,
            "resamples": RESAMPLES, "resample_seed": RESAMPLE_SEED,
            "minimum_cell_gain_skill": MINIMUM_CELL_GAIN_SKILL,
            "minimum_cells_per_family": MINIMUM_CELLS_PER_FAMILY,
            "minimum_leads_per_family": MINIMUM_LEADS_PER_FAMILY,
            "minimum_family_cases": MINIMUM_FAMILY_CASES,
            "z500_72h_tolerance": Z500_72H_TOLERANCE,
            "max_worse_cell_fraction": MAX_WORSE_CELL_FRACTION,
        },
        "limitations": LIMITATIONS,
    }


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol, including the nine-arm registry."""
    from training.r7_device_selection import protocol_fields
    from training.r7_experiment import canonical_digest

    measured = envelope()
    if not measured["sufficient"]:
        raise RuntimeError(f"the per-seed envelope {PER_SEED_SECONDS} s is only "
                           f"{measured['margin']}x the {measured['need_seconds']} s the treated "
                           f"rate requires; declaring {ENVELOPE_MARGIN}x is the frozen contract")
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

    ``fine_tune_long_rollout`` refuses a contract whose ``autoregression`` block omits the window
    digest, and its discovery is the 12-step long-rollout one, not the plain two-frame one. Using the
    plain discovery would build a contract the trainer rejects.
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
    # A null identity is the shape defect 19 took: the validator compared None to None and passed.
    for label, value in (("train", train_identity), ("validation", val_identity)):
        if not isinstance(value, str) or len(value) < 32:
            raise RuntimeError(f"the {label} data identity is not a digest: {value!r}")
    if train_identity != SIDECAR_DATA_IDENTITY:
        raise RuntimeError(f"the typed-evidence sidecar was fitted on data identity "
                           f"{SIDECAR_DATA_IDENTITY}, the train manifest is {train_identity}")
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "train_data_identity": train_identity,
            "val_data_identity": val_identity,
            "train_windows": train_windows(),
            "source_sha256": SOURCE_SHA256,
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
    """Re-check every frozen identity, the parent pins, the envelope and the preflight."""
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
    for label in ("train_data_identity", "source_sha256", "val_manifest_sha256"):
        value = body.get(label)
        if not isinstance(value, str) or len(value) < 32:
            raise RuntimeError(f"the frozen {label} is not a digest: {value!r}")
    measured = body["preflight"]
    if measured.get("f_preflight_passed") is not True:
        raise RuntimeError("the frozen preflight did not pass")
    if measured["envelope"]["sufficient"] is not True:
        raise RuntimeError("the frozen envelope does not cover the treated requirement")
    if float(body["per_seed_seconds"]) < float(measured["envelope"]["need_seconds"]):
        raise RuntimeError("the per-seed envelope is below the recomputed requirement")
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
    return explicit_execution_files("scripts/study_r7_s3_in_protocol.py",
                                    extra=("scripts/r7_s3_in_protocol_protocol.py",
                                           "scripts/r7_s3_in_protocol_preflight.py",
                                           "scripts/r7_s3_in_protocol_verdict.py",
                                           "scripts/r7_s3_joint_criterion.py"))


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
                      "envelope": {key: measurements["envelope"][key]
                                   for key in ("need_hours", "per_seed_seconds", "margin",
                                               "sufficient", "hard_cap_covers_waves")},
                      "family_grids": {name: family["grids"]
                                       for name, family in measurements["families"].items()},
                      "out": str(arguments.out) if arguments.out else None}, indent=2))
    return 0 if measurements["f_preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
