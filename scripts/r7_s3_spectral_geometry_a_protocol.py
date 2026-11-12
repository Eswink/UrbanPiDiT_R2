"""Stage A's protocol layer: constants, the beta_j freeze, the (e) preflight, freeze/prepare.

Split out of the driver along the same seam C2 used: this module owns *what a round is frozen
to*, the driver owns training and reading. Everything except ``beta_weights`` is held word for
word from the Stage B / C2 recipe (65x65 multi-year instance, K=4, depth 4, dim 192, LR 2e-5,
warmup 10, weight_decay 1e-4, clip 1.0, batch 4, physical weights, scoring region, climatology
denominator, seeds 41/42/43), so the on/off difference is one factor.

**The one factor** (decision 0112 D2): ``beta_j = [mean AMSE_j]^-1`` computed in the
normalised space the loss sees, from consecutive 6 h ground-truth pairs of one train year, and
frozen here. The loss shape is FastNet equation (10) as read from arXiv:2509.17601 §4.1.3 on
2026-11-12: a **pure sum** ``sum_j beta_j AMSE_j`` with no ``1/J`` and no parallel per-gridpoint
MSE term inside it (see ``training.r7_spectral_geometry.beta_weighted_amse``).

**What this round does NOT claim.** ``beta_j`` here is an *AMSE-analogue* normalisation on a
limited-area ``rfft2`` band decomposition (decision 0110), not the paper's spherical-harmonic
quantity. The name in every artifact is "beta_j AMSE-analogue normalisation".
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

FORMAT = "r7-s3-spectral-geometry-a-protocol-v1"
CANDIDATE, CONTROL = "on", "off"
SEEDS = (41, 42, 43)
#: 4,000 updates from the p1600 endpoint. AMSE's own paper fine-tunes rather than pre-trains
#: (arXiv:2509.17601 §4.1.2), and decision 0112 fixes this count so it is not scanned.
UPDATES = 4000
STEPS = curve.STEPS                       # K stays at the registered 4
BATCH_SIZE = 4                            # the plan's frozen batch, not ``wide``'s 1
#: The train manifest only; the same eight cases every physics probe in this campaign used.
PREFLIGHT_CASES = (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)
#: The (a) amplitude read uses the **same eight train cases Stage P1 measured**, so the two
#: rounds' per-band ``amp_ratio`` numbers are directly comparable. It is in-sample, which is
#: recorded as a limitation: the exit compares the two arms on identical cases, and a
#: memorised forecast would raise both arms at once, which biases the reading toward *not*
#: finding a difference rather than toward a false positive.
AMPLITUDE_CASES = PREFLIGHT_CASES
LEADS = wide.EVALUATION_LEADS             # (6, 12, 24, 48, 72)
#: The five leads the band aggregate reads, matching Stage P1's frozen gate lead set.
FROZEN_LEADS = LEADS

#: Basis for the duration envelope: point 3's measured 1.1096 s/update at batch 4.
MEASURED_SECONDS_PER_UPDATE = 1.1096
MEASURED_ON_POINT = "p4000"
#: Two arms x five leads of scoring per seed, on the measured five-lead cost of one arm.
SCORING_ALLOWANCE_SECONDS = 2 * 2588.1
ESTIMATED_PEAK_BYTES = 9569304576
HEADROOM_MARGIN_BYTES = 512 * 2**20
ENTRY_FILES = ("scripts/study_r7_s3_spectral_geometry_a.py",)

#: ``beta_j`` from Step C (``outputs/r7_s3_spectral_geometry_20261111_attempt01``'s probe run,
#: year 2001, normalised space). The prepare stage re-derives these through the same code path
#: and refuses to freeze if the store's values moved, so the frozen list is auditable rather
#: than copied by hand. Order is the store's ``channels`` attribute.
BETA_YEAR = 2001
BETA_PINNED = {
    "t2m": 0.00710339, "u10": 0.00238928, "v10": 0.00279479, "mslp": 0.0247631,
    "z850": 0.0188734, "t850": 0.0313664, "q850": 0.0181982, "u850": 0.0037179,
    "v850": 0.00322454, "z500": 0.105721, "t500": 0.0603037, "q500": 0.00597987,
    "u500": 0.00698083, "v500": 0.00285463, "z250": 0.207282, "u250": 0.0144935,
    "v250": 0.00419768,
}

LIMITATIONS = [
    "development calibration on the 65x65 multi-year instance only (2000-2021 train / 2022 val, "
    "17 channels); the 129x129 wide instance is a different store and is not covered",
    "the FY source has no 2023: the test split is four months and stays sealed; r stays 0",
    "beta_j is an AMSE-analogue normalisation on a limited-area rfft2 band decomposition, not "
    "FastNet's spherical-harmonic quantity; no number from that paper is a result of this round",
    "beta_j is estimated from a single year (2001); the inter-year variance is unmeasured here, "
    "and the paper's claim that it is minimal is not verified on this store",
    "4,000 updates is short against FastNet's fine-tuning schedule (36,525 + 4x7,305); a "
    "negative reading may be a dose artefact rather than a property of the geometry",
    "the on arm changes the loss shape as well as the weights: the pooled band sum becomes a "
    "per-variable sum, because a per-variable weight cannot apply to a pooled quantity",
    "three seeds are a paired comparison against the same-code off arm, not a significance claim",
    "the training-code identity differs from every earlier round, so this arm is not comparable "
    "as a code identity to Stage B's or Stage P's arms",
    "the duration envelope is quantized to whole hours at three times the estimate, so it is "
    "wider than strictly needed; that slack is deliberate, not a statement about the runtime",
]


def budgets():
    """The duration envelope, quantized from the measured point the plan quotes."""
    estimate = UPDATES * MEASURED_SECONDS_PER_UPDATE + SCORING_ALLOWANCE_SECONDS
    per_seed = float(math.ceil(3 * estimate / 3600) * 3600)
    return {"measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
            "measured_on_point": MEASURED_ON_POINT,
            "train_seconds_estimate": UPDATES * MEASURED_SECONDS_PER_UPDATE,
            "scoring_allowance_seconds": SCORING_ALLOWANCE_SECONDS,
            "point_seconds_estimate": estimate,
            "point_seconds_note": ("two arms (training and five-lead scoring) per seed: this is "
                                   "the per-seed estimate, not a per-arm one"),
            "estimate_basis": (f"2 arms x {UPDATES} updates x {MEASURED_ON_POINT}'s measured "
                               f"{MEASURED_SECONDS_PER_UPDATE} s/update at batch 4, plus 2 x the "
                               "measured five-lead scoring cost of one arm: an estimate, not a "
                               "measurement of this round"),
            "multiplier": 3.0,
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
    """The store's channel order and normalisation, read once and pinned by digest."""
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


def _derived_beta(store):
    """Re-derive beta_j through the Step C code path, so the frozen list is not hand-typed."""
    from scripts.r7_s3_gap_decay_and_beta import beta_weights
    return beta_weights(store, year=BETA_YEAR, normalized=True)


def _assert_beta_matches_step_c(derived, channels):
    """The frozen weight must be the Step C weight, checked per channel."""
    rows = derived.get("channels") or {}
    missing = [name for name in channels if name not in rows]
    if missing:
        raise RuntimeError(f"the store is missing channels {missing}; beta_j cannot be frozen")
    values, worst = [], 0.0
    for name in channels:
        pinned, measured = BETA_PINNED[name], rows[name]["beta"]
        if measured is None or not math.isfinite(float(measured)):
            raise RuntimeError(f"beta_j for {name} came back {measured!r}; refusing to freeze")
        worst = max(worst, abs(float(measured) - float(pinned)) / float(pinned))
        values.append(float(measured))
    if worst > 1e-5:
        raise RuntimeError(f"beta_j moved away from the Step C reading by {worst:.3e} relative; "
                           "the store or the decomposition changed, so this is a new round")
    if max(values) == min(values):
        raise RuntimeError("beta_j came back constant across variables; the weighting is inert")
    return values, worst


def _preflight_inputs(seed):
    """The parent and the coordinates the preflight needs, once."""
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
    root = zarr.open_group(str(curve.STORE), mode="r")
    latitude = torch.tensor(np.asarray(root["latitude"][:]), dtype=torch.float32)
    longitude = torch.tensor(np.asarray(root["longitude"][:]), dtype=torch.float32)
    return {"model": model, "latitude": latitude, "longitude": longitude,
            "parent_checkpoint_sha256": expected}


def _preflight_cases(context, beta):
    """Per case and per step: every variable's AMSE, the pooled MSE, and the dose.

    Decision 0112's exit (e) asks two things of this measurement: ``beta_j`` is not constant
    across variables, and the AMSE is finite at **all eight cases x twelve steps**. A third
    number is recorded because the plan's §3.2 requires it: ``amse_total / L_mse``, the dose the
    on arm will actually see.
    """
    import torch
    from torch.utils.data import default_collate
    from data.r7_long_rollout_dataset import (ZarrLongRolloutDataset,
                                              preflight_long_rollout_windows)
    from training.r7_long_rollout_runner import _rollout
    from training.r7_spectral_geometry import (amse_analogue, beta_weighted_amse,
                                              per_channel_amse)

    windows = preflight_long_rollout_windows(curve.TRAIN_MANIFEST, physical_steps=12)
    dataset = ZarrLongRolloutDataset(curve.TRAIN_MANIFEST, physical_steps=12,
                                     expected_exclusions=windows["excluded_sample_ids"],
                                     expected_window_sha256=windows["window_sha256"])
    contract = {"kind": "process", "model": arm_spec(), "steps": STEPS,
                "autoregression": {"physical_steps": 12,
                                   "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                                   "draft_supervision": "deep",
                                   "spatial_gradient_lambda": 0.0}}
    weight = torch.as_tensor(beta, dtype=torch.float32)
    rows = []
    for case in PREFLIGHT_CASES:
        batch = default_collate([dataset[case]])
        rollout = _rollout(context["model"], batch, contract, None)
        forecasts = rollout.forecasts.detach()
        targets = batch["physical_targets"]
        if forecasts.shape[1] != targets.shape[1]:
            raise RuntimeError(f"physical_targets has {targets.shape[1]} steps but the rollout "
                               f"produced {forecasts.shape[1]}")
        per_step = []
        for step in range(forecasts.shape[1]):
            channel = per_channel_amse(forecasts[:, step], targets[:, step])
            per_step.append({"step": step + 1, "lead_hours": 6 * (step + 1),
                             "amse_per_channel": [float(value) for value in channel],
                             "all_finite": bool(torch.isfinite(channel).all())})
        weighted = float(beta_weighted_amse(forecasts, targets, weight))
        pooled = float(torch.stack([amse_analogue(forecasts[:, step], targets[:, step])
                                    for step in range(forecasts.shape[1])]).mean())
        mse = float(rollout.loss.detach())
        rows.append({"case": case, "steps": per_step, "amse_total": weighted,
                     "amse_pooled_unweighted": pooled, "mse": mse,
                     "dose": weighted / mse if mse > 0 else float("inf"),
                     "dose_pooled": pooled / mse if mse > 0 else float("inf")})
    return rows


def preflight(seed, beta):
    """Decision 0112's exit (e), run through the module the training run will call.

    Raises if any of the three answers is wrong, so a failing preflight leaves no protocol.
    Also records the dose (``amse_total / L_mse``), which plan 0031 §3.2 requires and whose
    interpretation the round freezes in :func:`dose_decision` *before* any training runs.
    """
    context = _preflight_inputs(seed)
    rows = _preflight_cases(context, beta)
    if max(beta) == min(beta):
        raise RuntimeError("beta_j is constant across variables, so the weighting is inert")
    for row in rows:
        for step in row["steps"]:
            if not step["all_finite"]:
                raise RuntimeError(f"case {row['case']} step {step['step']} produced a "
                                   "non-finite AMSE; the loss would poison the gradient")
    doses = [row["dose"] for row in rows]
    return {"cases": list(PREFLIGHT_CASES), "steps_per_case": 12, "rows": rows,
            "beta": list(beta), "beta_max_over_min": max(beta) / min(beta),
            "dose_band_observed": [min(doses), max(doses)],
            "dose_pooled_band_observed": [min(row["dose_pooled"] for row in rows),
                                          max(row["dose_pooled"] for row in rows)],
            "dose_decision": dose_decision(doses),
            "parent_checkpoint_sha256": context["parent_checkpoint_sha256"],
            "statement": ("run through training.r7_spectral_geometry's own per-channel AMSE and "
                          "beta-weighted sum, so these are the numbers the training run will "
                          "produce, not a reimplementation's")}


def dose_decision(doses) -> dict:
    """What the measured dose means for what the on arm is actually training on.

    Plan 0031 §6 risk 3 reserves this to the A round and requires it be written down **before**
    freezing: "AMSE 是全损失替换还是附加项、权重多少，是 A 轮 ADR 必须冻结前写定的。本轮不预设。"

    **The decision: no divisor.** The term enters at its natural scale -- FastNet equation (10)
    is a pure ``sum_j beta_j AMSE_j`` with no ``1/J`` and no free weight (see
    ``training.r7_spectral_geometry.beta_weighted_amse``). The measured dose is quoted below
    and interpreted rather than tuned, because every candidate divisor would be a fitted
    constant and the round's constraints forbid scanning one.

    **Why no divisor is the faithful choice, not a shortcut.** ``beta_j AMSE_j`` is dimensionless
    and its magnitude is ``sum_j (current mismatch / that variable's year-mean mismatch)``, i.e.
    ~``J`` times a few -- a number that is *not* commensurable with a per-grid-point normalised
    MSE. FastNet never adds the two: its ``L_MSH`` **is** the objective (arXiv:2509.17601
    §4.1.3, equation 10), with no per-gridpoint MSE term beside it. The repository's runner adds
    any auxiliary axis to ``result.loss`` and cannot be changed (597/600 lines, R-051), so the
    MSE can only be made negligible, not removed. At the measured dose it *is* negligible.

    **Consequence, recorded before the run so it cannot be read back into the result.** The on
    arm trains on the beta-weighted AMSE-analogue with the MSE as a residual of order 1/dose;
    the off arm trains on the MSE alone. The on/off difference therefore tests *"the beta-weighted
    AMSE-analogue as the objective"* -- which is the paper's own §4.1 MSH-vs-MSE comparison --
    and **not** "a small beta_j correction added to an unchanged MSE". A negative outcome is
    evidence about the former and must not be read as a verdict on the latter. The three exits
    are unchanged; only this interpretation is fixed in advance.

    The ratio that would be needed to make the term a perturbation rather than the objective is
    a divisor of order the measured dose, which varies case to case (see the band below), so it
    is not a constant of the problem -- which is the second reason not to invent one.
    """
    low, high = min(doses), max(doses)
    if high < 1.0:
        regime = ("inert: the axis would be a small perturbation, i.e. the C1 failure mode "
                  "(a dose too small to act)")
    elif low > 20.0:
        regime = ("objective-dominating: the MSE is a residual of order 1/dose, so the on arm "
                  "trains on the beta-weighted AMSE-analogue, which is FastNet's L_MSH regime")
    else:
        regime = "mixed: both terms are the same order, so the on/off difference is attributable"
    return {
        "divisor": 1,
        "divisor_rationale": ("equation (10) is a pure sum; every alternative divisor would be a "
                              "fitted constant and scanning one is forbidden by the round's "
                              "constraints"),
        "measured_band": [low, high],
        "regime": regime,
        "on_arm_is_the_paper_regime": bool(low > 20.0),
        "interpretation": ("the on/off difference tests the beta-weighted AMSE-analogue as the "
                           "training objective (the paper's MSH-vs-MSE comparison), not a small "
                           "perturbation added to an unchanged MSE"),
        "residual_mse_share": [1.0 / (1.0 + high), 1.0 / (1.0 + low)],
        "decision": "frozen-before-training",
    }


def frozen_constants():
    """Everything frozen before any data or model is touched."""
    return {"format": FORMAT, "stage": "S3-PHYSICS-INFORMED-SEQUENCE-A",
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
            "arms": {CANDIDATE: {"spectral_geometry": "amse-analogue", "updates": UPDATES,
                                 "steps": STEPS, "batch_size": BATCH_SIZE,
                                 "seeds": list(SEEDS), "beta_free": False},
                     CONTROL: {"spectral_geometry": None, "updates": UPDATES,
                               "steps": STEPS, "batch_size": BATCH_SIZE,
                               "seeds": list(SEEDS), "beta_free": True}},
            "beta_normalisation": {
                "method_name": "beta_j AMSE-analogue normalisation",
                "form": ("sum_j beta_j AMSE_j over the 17 variables, per forecast step, averaged "
                         "over steps; no 1/J divisor and no parallel per-gridpoint MSE inside the "
                         "term -- FastNet equation (10) as read from arXiv:2509.17601 section "
                         "4.1.3 on 2026-11-12"),
                "space": "normalised (the space the loss sees)",
                "year": BETA_YEAR, "pairs": "consecutive 6 h ground-truth pairs",
                "parameter_free": True,
                "not_the_paper_quantity": ("the paper's beta_j is an inverse spherical-harmonic "
                                           "AMSE; this is a limited-area rfft2 band analogue "
                                           "(decision 0110)"),
                "frozen": True,
                "recomputed_during_training": False},
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
            "amplitude_cases": {"manifest": str(curve.TRAIN_MANIFEST),
                                "cases": list(AMPLITUDE_CASES),
                                "leads": list(FROZEN_LEADS),
                                "why": ("the same eight train cases Stage P1 measured, so the "
                                        "per-band amp_ratio numbers are comparable across the two "
                                        "rounds; in-sample, which biases toward finding no "
                                        "difference between the arms rather than toward a false "
                                        "positive")},
            "test_read": False, "scientific_claim": False,
            "limitations": list(LIMITATIONS)}


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol. The preflight runs at prepare, before promotion."""
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
    """Add the live identity AND the beta_j freeze plus the (e) preflight."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import (canonical_digest, dataset_identity, model_code_digest)
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    import zarr
    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    windows = preflight_long_rollout_windows(curve.TRAIN_MANIFEST, physical_steps=12)
    train_identity, _ = dataset_identity(curve.TRAIN_MANIFEST)
    constants = _store_constants()
    store = zarr.open_group(str(curve.STORE), mode="r")
    derived = _derived_beta(store)
    beta, worst = _assert_beta_matches_step_c(derived, constants["channels"])
    # The preflight runs here, on the pinned parent, and its numbers go into the protocol. A
    # failing preflight therefore leaves no protocol.json for archive or seed to read.
    measurements = preflight(SEEDS[0], beta)
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "train_data_identity": train_identity,
            "val_data_identity": dataset_identity(curve.VAL_MANIFEST)[0],
            "train_windows": windows,
            "store_constants": constants,
            "beta_weights": {"channels": constants["channels"], "values": beta,
                             "year": BETA_YEAR, "space": "normalised",
                             "step_c_max_relative_difference": worst,
                             "matched_step_c": True},
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
    """Re-check every frozen identity, including the beta_j freeze and the preflight."""
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
    if len(measured["rows"]) != len(PREFLIGHT_CASES):
        raise RuntimeError("the frozen preflight does not cover every case")
    for row in measured["rows"]:
        if len(row["steps"]) != 12:
            raise RuntimeError(f"case {row['case']} does not carry twelve steps")
        if not all(step["all_finite"] for step in row["steps"]):
            raise RuntimeError("the frozen preflight recorded a non-finite AMSE")
    beta = body["beta_weights"]["values"]
    if len(beta) != len(body["beta_weights"]["channels"]):
        raise RuntimeError("the frozen beta_j does not cover every channel")
    if max(beta) == min(beta):
        raise RuntimeError("the frozen beta_j is constant across variables")
    if body["arms"][CANDIDATE]["spectral_geometry"] != "amse-analogue":
        raise RuntimeError("the candidate arm is not the amse-analogue axis")
    if body["arms"][CONTROL]["spectral_geometry"] is not None:
        raise RuntimeError("the control arm must not carry the spectral geometry axis")
    return body
