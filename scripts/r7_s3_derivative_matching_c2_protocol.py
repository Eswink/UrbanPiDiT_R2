"""The Stage C2 protocol layer: constants, preflight, freeze/prepare/validate.

Split out of ``study_r7_s3_derivative_matching_c2.py`` (598 lines, over R-021's 400-line target)
along the same seam ``r7_s3_curve_point_large_protocol`` / ``_driver`` use: this module owns
*what a round is frozen to*, the driver owns training and reading. The preflight lives here
because decision 0095 makes it a freeze precondition: no protocol is promoted unless the
measurement passes. It calls ``training.r7_derivative_matching`` itself rather than
reimplementing it, so the frozen numbers are the ones the training run will produce.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_s3_curve_point_large_protocol as curve
from scripts import r7_s3_position_retention_parent as parent_pins
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

FORMAT = "r7-s3-derivative-matching-c2-protocol-v1"
CANDIDATE, CONTROL = "on", "off"
SEEDS = (41, 42, 43)
UPDATES = 32132                    # 4 epoch at batch 4 over 32,131 usable windows
STEPS = curve.STEPS                # K stays at the registered 4
BATCH_SIZE = curve.BATCH_SIZE
W_DIVERGENCE, W_VORTICITY = 0.1, 0.1
#: Derived from the analysis field's own distribution (decision 0095): the mean ``|div|`` and
#: ``|vort|`` over the preflight cases. Recomputed at prepare time and asserted equal to these,
#: so a store whose fields moved fails instead of rescaling.
DIVERGENCE_SCALE, VORTICITY_SCALE = 2.5665e-05, 3.5089e-05
#: Train manifest only; the same indices the ADR's measurement used.
PREFLIGHT_CASES = (1000, 9000, 17000, 25000)
#: The preflight must see the truth score exactly zero and the dose within this band of L_mse.
#: A band rather than a threshold because the question is "can this term act at all", and the
#: measured answer is 0.43-0.57 with the declared weights. Frozen before the run.
DOSE_MIN, DOSE_MAX = 0.1, 10.0
#: Budget from the fourth Stage B point's measured 1.1096 s/update at batch 4 plus that point's
#: measured five-lead scoring cost.
MEASURED_SECONDS_PER_UPDATE = 1.1096
MEASURED_ON_POINT = "p1600"
SCORING_ALLOWANCE_SECONDS = 2588.1
ESTIMATED_PEAK_BYTES = 9569304576
HEADROOM_MARGIN_BYTES = 512 * 2**20
ENTRY_FILES = ("scripts/study_r7_s3_derivative_matching_c2.py",)

LIMITATIONS = [
    "development calibration on the 65x65 multi-year instance only (2000-2021 train / 2022 val, "
    "17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed; r stays 0",
    "w_div = w_vort = 0.1 are this round's choice: legacy has no derivative-domain target "
    "matching term to quote, so they rest on equal weighting plus one dose measurement, not on a "
    "sweep. If the conclusion is weight-sensitive this round cannot tell",
    "the two terms are varied together: even a full pass says 'this combination did not regress "
    "72 h', never which term did the work",
    "the scales come from four train cases' analysis fields; another season or region would move "
    "them, and they change each term's size though not the truth-is-zero property",
    "C2 reads the target, so it is not untargeted physical information: what it can teach is "
    "'produce the target's derivative structure', not 'satisfy a physical law'. A pass must not "
    "be written up as the model having learned physics",
    "three seeds are a paired comparison against the same-code off arm, not a significance claim",
    "the training-code identity differs from every earlier round, so this arm is not comparable "
    "as a code identity to Stage B's arms",
    "the duration envelope is quantized to whole hours at three times the estimate, so it is "
    "wider than strictly needed; that slack is deliberate, not a statement about the runtime",
]

def budgets():
    """The duration envelope, quantized from the latest measured point.

    ``per_seed`` is three times the derived estimate rounded up to a whole hour; ``planned`` is
    that envelope and ``hard`` twice it, preserving the 1:2 ratio every earlier round used.
    """
    estimate = UPDATES * MEASURED_SECONDS_PER_UPDATE + SCORING_ALLOWANCE_SECONDS
    per_seed = float(math.ceil(3 * estimate / 3600) * 3600)
    return {"measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
            "measured_on_point": MEASURED_ON_POINT,
            "train_seconds_estimate": UPDATES * MEASURED_SECONDS_PER_UPDATE,
            "scoring_allowance_seconds": SCORING_ALLOWANCE_SECONDS,
            "point_seconds_estimate": estimate,
            "estimate_basis": (f"{UPDATES} updates x {MEASURED_ON_POINT}'s measured "
                               f"{MEASURED_SECONDS_PER_UPDATE} s/update at batch 4, plus that "
                               "point's measured five-lead scoring cost: an estimate, not a "
                               "measurement of this round"),
            "per_seed_seconds": per_seed, "planned_seconds": per_seed,
            "hard_cap_seconds": 2 * per_seed,
            "soft_rule": "soft overrun continues and is recorded; only the hard cap truncates",
            "decision": "0060"}


PER_SEED_SECONDS = budgets()["per_seed_seconds"]
PLANNED_SECONDS = budgets()["planned_seconds"]
HARD_CAP_SECONDS = budgets()["hard_cap_seconds"]


def arm_spec():
    """The archived process spec, unchanged: this round touches no model switch."""
    spec = dict(curve.model_spec())
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    if (spec.get("climatology_anchor_spec") or spec.get("anomaly_feedback")
            or spec.get("anchor_output_prior") or spec.get("process_init_conditioning")
            or spec.get("typed_evidence_mode") or spec.get("global_mixing_mode")):
        raise RuntimeError("the archived process spec already carries an architecture switch")
    return spec


def _store_constants():
    """The store's channel order and normalization, read once and pinned by digest.

    The training batch carries no normalization (the dataset normalises inside ``__getitem__``),
    so the constraint's denormalisation has to be handed these constants explicitly. Pinning
    their digest makes "which mean/std were used" an auditable field rather than a hidden one.
    """
    import hashlib
    import numpy as np
    import zarr
    root = zarr.open_group(str(curve.STORE), mode="r")
    channels = list(root.attrs["channels"])
    mean = np.asarray(root["normalization_mean"][:], dtype=np.float64).tolist()
    std = np.asarray(root["normalization_std"][:], dtype=np.float64).tolist()
    if len(channels) != len(mean) or len(mean) != len(std):
        raise RuntimeError("the store's channels and normalization lengths disagree")
    payload = json.dumps({"channels": channels, "mean": mean, "std": std},
                         sort_keys=True).encode("utf-8")
    return {"channels": channels, "mean": mean, "std": std,
            "sha256": hashlib.sha256(payload).hexdigest()}


def _preflight_inputs(seed):
    """The parent, the store constants and the coordinates the preflight needs, once."""
    import numpy as np
    import torch
    import zarr
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_arm_harness import sha256_file

    path, expected = parent_pins.pinned_parent(seed)
    if sha256_file(path) != expected:
        raise RuntimeError(f"seed {seed} parent checkpoint pin changed")
    saved = load_checkpoint(path)
    model = make_model("process", arm_spec())
    model.load_state_dict(saved["model"], strict=True)
    model.train()
    constants = _store_constants()
    root = zarr.open_group(str(curve.STORE), mode="r")
    latitude = torch.tensor(np.asarray(root["latitude"][:]), dtype=torch.float32)
    longitude = torch.tensor(np.asarray(root["longitude"][:]), dtype=torch.float32)
    return {"model": model, "constants": constants, "latitude": latitude,
            "longitude": longitude, "parent_checkpoint_sha256": expected}


def _preflight_cases(context):
    """One row per case: the live term, the zero-at-target check, and the dose against the MSE."""
    import torch
    from torch.utils.data import default_collate
    from data.r7_long_rollout_dataset import (ZarrLongRolloutDataset,
                                              preflight_long_rollout_windows)
    from training.r7_derivative_matching import derivative_matching_terms
    from training.r7_physical_constraints import denormalize
    from training.r7_long_rollout_runner import _rollout

    constants = context["constants"]
    mean = torch.tensor(constants["mean"], dtype=torch.float32)
    std = torch.tensor(constants["std"], dtype=torch.float32)
    windows = preflight_long_rollout_windows(curve.TRAIN_MANIFEST, physical_steps=12)
    dataset = ZarrLongRolloutDataset(curve.TRAIN_MANIFEST, physical_steps=12,
                                     expected_exclusions=windows["excluded_sample_ids"],
                                     expected_window_sha256=windows["window_sha256"])
    # A minimal rollout contract: the preflight needs the same forward the training loop uses
    # (K steps, the registered physical weights, deep draft supervision, no gradient penalty),
    # and nothing about the protocol identity, which does not exist yet at prepare time.
    contract = {"kind": "process", "model": arm_spec(), "steps": STEPS,
                "autoregression": {"physical_steps": 12,
                                   "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                                   "draft_supervision": "deep",
                                   "spatial_gradient_lambda": 0.0}}
    scales = {"divergence": DIVERGENCE_SCALE, "vorticity": VORTICITY_SCALE}
    rows, magnitudes = [], []
    for case in PREFLIGHT_CASES:
        batch = default_collate([dataset[case]])
        rollout = _rollout(context["model"], batch, contract, None)
        prediction = denormalize(rollout.forecasts[:, -1], (mean, std), constants["channels"])
        # The target MUST be the step this forecast predicts. ``atmos_target`` is
        # ``physical_targets[:, 0]`` -- the +6 h field -- and pairing it with the +72 h forecast is
        # exactly the defect that voided attempt07 (decisions 0105/0106): the term then charges a
        # perfect forecast, so its optimum cannot contain the truth.
        targets = batch["physical_targets"]
        if targets.shape[1] != rollout.forecasts.shape[1]:
            raise RuntimeError(f"physical_targets has {targets.shape[1]} steps but the rollout "
                               f"produced {rollout.forecasts.shape[1]}")
        target = denormalize(targets[:, -1], (mean, std), constants["channels"])
        # A cross-lead field, kept only to prove the guard below can actually FAIL. If this one is
        # ever zero, the guard is vacuous again.
        other_lead = denormalize(batch["atmos_target"], (mean, std), constants["channels"])
        live = derivative_matching_terms(prediction, target, constants["channels"],
                                         context["latitude"], context["longitude"], scales=scales)
        zero = derivative_matching_terms(target, target, constants["channels"],
                                         context["latitude"], context["longitude"], scales=scales)
        mismatched = derivative_matching_terms(other_lead, target, constants["channels"],
                                               context["latitude"], context["longitude"],
                                               scales=scales)
        mse = float(rollout.loss.detach())
        rows.append({"case": case, "divergence": float(live["divergence"]),
                     "vorticity": float(live["vorticity"]), "total": float(live["total"]),
                     "mse": mse, "dose": float(live["total"]) / mse,
                     "zero_at_target": float(zero["total"]),
                     # The counterexample: a DIFFERENT lead's analysis field must score > 0. A
                     # target-vs-itself comparison is zero by construction and proves nothing.
                     "nonzero_for_cross_lead_field": float(mismatched["total"])})
        # The scales are declared as coming from the ANALYSIS field's distribution, so the
        # provenance is measured on the target. Measuring the prediction would make the
        # protocol's stated basis unverifiable.
        magnitudes.append({"case": case,
                           "analysis_mean_abs_divergence": _mean_abs_derivative(
                               target, constants["channels"], context["latitude"],
                               context["longitude"], kind="divergence"),
                           "analysis_mean_abs_vorticity": _mean_abs_derivative(
                               target, constants["channels"], context["latitude"],
                               context["longitude"], kind="vorticity")})
    return rows, magnitudes


def preflight(seed):
    """The decision-0095 preflight, run through the module the training run will call.

    Three questions, each answered by measurement rather than argument: does ``pred = target``
    score exactly zero (the family property C1 lacked); does the trained parent's forecast score
    above zero (the term is not inert); is the dose the same order as the MSE (C1's failure was
    a dose 5 orders too small)?

    Returns the measurements and raises if any answer is wrong, so a failing preflight leaves no
    protocol behind. Uses the train manifest only.
    """
    context = _preflight_inputs(seed)
    rows, magnitudes = _preflight_cases(context)
    if any(row["zero_at_target"] != 0.0 for row in rows):
        raise RuntimeError("the preflight found a nonzero value at pred = target, so this term is "
                           "not a target match and the family argument fails")
    if any(row["nonzero_for_cross_lead_field"] <= 0.0 for row in rows):
        # The discriminator. ``zero_at_target`` alone cannot fail on a target-vs-itself
        # comparison, which is precisely how attempt07's cross-lead pairing slipped through
        # (decision 0105). A different lead's field MUST score above zero; if it does not, the
        # guard has stopped discriminating and a real mismatch could pass again.
        raise RuntimeError(
            "the preflight's cross-lead counterexample scored zero, so the zero-at-target guard "
            "cannot distinguish the deployed pairing from a mismatched one; refusing to freeze")
    if any(row["total"] <= 0.0 for row in rows):
        raise RuntimeError("the preflight found the term inert on the trained parent")
    if any(not (DOSE_MIN <= row["dose"] <= DOSE_MAX) for row in rows):
        raise RuntimeError(f"the preflight dose left the declared band [{DOSE_MIN}, {DOSE_MAX}]: "
                           f"{[round(row['dose'], 4) for row in rows]}")
    # Re-derive the scales from the analysis and assert they equal the declared constants: a
    # store whose fields moved must fail here rather than silently rescale every term.
    derived = {"divergence": sum(item["analysis_mean_abs_divergence"] for item in magnitudes)
               / len(magnitudes),
               "vorticity": sum(item["analysis_mean_abs_vorticity"] for item in magnitudes)
               / len(magnitudes)}
    for name, declared in (("divergence", DIVERGENCE_SCALE), ("vorticity", VORTICITY_SCALE)):
        if abs(derived[name] - declared) / declared > 0.5:
            raise RuntimeError(
                f"the {name} scale declared in the protocol ({declared:.4e}) is not the analysis "
                f"field's measured magnitude ({derived[name]:.4e}); the store moved")
    return {"cases": list(PREFLIGHT_CASES), "rows": rows, "derived_scales": derived,
            "scales": {"divergence": DIVERGENCE_SCALE, "vorticity": VORTICITY_SCALE},
            "weights": {"divergence": W_DIVERGENCE, "vorticity": W_VORTICITY},
            "derivative_magnitudes": magnitudes, "dose_band": [DOSE_MIN, DOSE_MAX],
            "store_constants_sha256": context["constants"]["sha256"],
            "parent_checkpoint_sha256": context["parent_checkpoint_sha256"],
            "statement": ("run through training.r7_derivative_matching itself, so these are the "
                          "numbers the training run will produce, not a reimplementation's")}


def _mean_abs_derivative(field, channels, latitude, longitude, *, kind):
    """The raw ``mean |div|`` or ``mean |vort|`` in physical units, for the scale provenance."""
    from model.r7_process_tensor_diagnostics import (
        spherical_divergence, spherical_vorticity)
    index = {name: position for position, name in enumerate(channels)}
    u = field[..., index["u850"], :, :]
    v = field[..., index["v850"], :, :]
    operator = spherical_divergence if kind == "divergence" else spherical_vorticity
    return float(operator(u, v, latitude, longitude).abs().mean())


def frozen_constants():
    """Everything frozen before any data or model is touched."""
    return {"format": FORMAT, "stage": "S3-THROUGHPUT-PHYSICS-C2",
            "instance": str(curve.INSTANCE), "store": str(curve.STORE),
            "source_sha256": curve.SOURCE_SHA256, "store_shape": list(curve.STORE_SHAPE),
            "split_years": {"train": curve.CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
            "train_manifest": str(curve.TRAIN_MANIFEST),
            "val_manifest": str(curve.VAL_MANIFEST),
            "test_manifest_never_read": str(curve.TEST_MANIFEST),
            "usable_windows": curve.USABLE_WINDOWS, "val_cohorts": dict(curve.VAL_COHORTS),
            "region": {"points": [65, 65], "boundary_margins": [1], "scored_region": "full",
                       "n_grid_points": 4225,
                       "note": "the whole native 65x65 grid; interior_32 is 1x1 here"},
            "arms": {CANDIDATE: {"derivative_matching": "c2", "updates": UPDATES,
                                 "steps": STEPS, "batch_size": BATCH_SIZE,
                                 "seeds": list(SEEDS)},
                     CONTROL: {"derivative_matching": None, "updates": UPDATES,
                               "steps": STEPS, "batch_size": BATCH_SIZE,
                               "seeds": list(SEEDS)}},
            "derivative_matching": {"weights": {"divergence": W_DIVERGENCE,
                                                "vorticity": W_VORTICITY},
                                    "scales": {"divergence": DIVERGENCE_SCALE,
                                               "vorticity": VORTICITY_SCALE},
                                    "scale_basis": ("the analysis field's own mean |div| and "
                                                    "|vort| over the preflight cases, re-derived "
                                                    "at prepare time and asserted equal"),
                                    "application": "last 6 h forecast, once per update",
                                    "reads_target": True},
            "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
                       "lr": curve.LR, "warmup": curve.WARMUP,
                       "weight_decay": curve.WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                       "clip": curve.CLIP, "bf16": wide.BF16,
                       "checkpoint_every": curve.CHECKPOINT_EVERY,
                       "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                       "reasoning_steps": curve.REASONING_STEPS,
                       "epochs": round(UPDATES * BATCH_SIZE / curve.USABLE_WINDOWS, 6)},
            "checkpoint_policy": {"checkpoint_every": curve.CHECKPOINT_EVERY,
                                  "retained": "multiples of checkpoint_every plus the endpoint",
                                  "selection": "frozen endpoint; no validation selection",
                                  "decision": "0087"},
            "budgets": budgets(),
            "test_read": False, "scientific_claim": False,
            "limitations": list(LIMITATIONS)}


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol. The preflight runs at prepare, before this is promoted."""
    from training.r7_experiment import canonical_digest
    body = {**frozen_constants(),
            "gpu_uuid": wide.GPU_UUID, "estimated_peak_bytes": ESTIMATED_PEAK_BYTES,
            **devices.protocol_fields(curve.DEVICE_MODE, single_uuid=wide.GPU_UUID),
            "headroom_margin_bytes": HEADROOM_MARGIN_BYTES,
            "execution_mode": "direct owned bounded workers, no descendants; failure stops this attempt",
            "planned_seconds": PLANNED_SECONDS, "hard_cap_seconds": HARD_CAP_SECONDS,
            "per_seed_seconds": PER_SEED_SECONDS,
            "started_perf_counter": started, "deadline_perf_counter": deadline,
            "identity": "metadata prepared in bounded CPU worker below",
            "code_archive_input": str(code_archive), "code_commit": code_commit,
            "code_zip_sha256": code_sha256}
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(Path(output) / "preparation_protocol.json", body)
    return body


def prepared_protocol(output):
    """Add the live identity AND the preflight, which must pass before a protocol exists."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import (canonical_digest, dataset_identity, model_code_digest)
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    windows = preflight_long_rollout_windows(curve.TRAIN_MANIFEST, physical_steps=12)
    train_identity, _ = dataset_identity(curve.TRAIN_MANIFEST)
    constants = _store_constants()
    # The preflight runs here, on the pinned parent, and its numbers go into the protocol.
    # A failing preflight therefore leaves no protocol.json for archive or seed to read.
    measurements = preflight(SEEDS[0])
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "train_data_identity": train_identity,
            "val_data_identity": dataset_identity(curve.VAL_MANIFEST)[0],
            "train_windows": windows,
            "store_constants": constants,
            "preflight": measurements,
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()},
            "arms_parent": {"pins": {}, "checkpoints": parent_pins.pins()}}
    parent_pins.install_parent_pins({"arms": {"parent": body["arms_parent"]}})
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def execution_files():
    """Decision 0067: this round's import closure, from a named entry file."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0])


def validate_protocol(output):
    """Re-check every frozen identity, including the preflight's own result."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest mismatch")
    if model_code_digest() != body["model_code_sha256"]:
        raise RuntimeError("the model revision moved after freeze")
    if training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("the training revision moved after freeze")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution source changed: {name}")
    if body.get("test_read") is not False:
        raise RuntimeError("this round is validation-only")
    measured = body["preflight"]
    if any(row["zero_at_target"] != 0.0 for row in measured["rows"]):
        raise RuntimeError("the frozen preflight recorded a nonzero value at pred = target")
    if any(not (DOSE_MIN <= row["dose"] <= DOSE_MAX) for row in measured["rows"]):
        raise RuntimeError("the frozen preflight recorded a dose outside the declared band")
    if measured["scales"] != body["derivative_matching"]["scales"]:
        raise RuntimeError("the frozen preflight scales differ from the protocol's")
    if body["arms"][CANDIDATE]["derivative_matching"] != "c2":
        raise RuntimeError("the candidate arm is not the c2 axis")
    if body["arms"][CONTROL]["derivative_matching"] is not None:
        raise RuntimeError("the control arm must not carry the c2 axis")
    return body
