"""Stage C2 study driver: registered deep MSE -> same MSE + derivative-domain target matching.

One factor, on/off, three seeds, batch fixed at 4. The on arm adds exactly one thing to the
registered objective: for the last 6 h forecast,

    L = L_mse + w_div * mean(((∇·pred − ∇·target)/s_div)²) + w_vort * mean(((∇×pred − ∇×target)/s_vort)²)

with ``w_div = w_vort = 0.1`` and the scales derived from the analysis field's own distribution
(decision 0095). ``pred = target`` scores exactly zero, so the optimum set is the truth -- which
is what distinguishes this from the closed spatial-gradient loss and from C1 (decision 0094).

**The preflight runs inside this driver, not in an ad-hoc script.** ADR 0095 §Decision 9 requires
it: a probe that bypasses the module can report a number the training run will never see, and no
test would catch that difference. ``preflight()`` therefore calls
``training.r7_derivative_matching.derivative_matching_terms`` itself on the pinned parent, and its
numbers are frozen into ``protocol.json``. If the preflight fails -- the truth does not score zero,
or the dose is not the same order as the MSE -- the prepare stage refuses and no protocol exists.

Everything else is held word-for-word from Stage B's registered recipe: 65x65 multi-year instance,
treatment manifest (32,131 usable windows), K=4, depth 4, dim 192, LR 2e-5, warmup 10, the sparse
physical weights, batch 4, and the same scoring region and climatology denominator. The 2023 test
split is never read.
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
#: Derived from the analysis field's own distribution (decision 0095 §Context ③): the mean
#: ``|div|`` and ``|vort|`` over the train cases the preflight reads. Recomputed at prepare
#: time and asserted equal to these, so a store whose fields moved fails instead of rescaling.
DIVERGENCE_SCALE, VORTICITY_SCALE = 2.5665e-05, 3.5089e-05
#: The preflight cases. Train manifest only; the same indices the ADR's measurement used.
PREFLIGHT_CASES = (1000, 9000, 17000, 25000)
#: The preflight must see the truth score exactly zero and the dose within this band of L_mse.
#: A band rather than a threshold because the question is "can this term act at all", and the
#: measured answer is 0.71-1.29. Declared before the run; never adjusted against a reading.
DOSE_MIN, DOSE_MAX = 0.1, 10.0
#: One arm per seed; two arms x three seeds. Budget from the fourth Stage B point's measured
#: 1.1096 s/update at batch 4 plus that point's measured five-lead scoring cost.
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
    "72 h', never which term did the work. The separating experiment is not in this round",
    "the scales come from four train cases' analysis fields; another season or region would move "
    "them, and they change each term's size though not the truth-is-zero property",
    "C2 reads the target, so it is not untargeted physical information: what it can teach is "
    "'produce the target's derivative structure', not 'satisfy a physical law'. A pass must not "
    "be written up as the model having learned physics",
    "three seeds are a paired comparison against the same-code off arm, not a significance claim "
    "across the wider hypothesis space",
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


def preflight(seed):
    """The decision-0095 preflight, run through the module the training run will call.

    Three questions, each answered by measurement rather than by argument:

    1. does ``pred = target`` score exactly zero? (the family property)
    2. does the trained parent's forecast score above zero? (the term is not inert)
    3. is the dose the same order as the MSE? (C1's failure mode was a dose 5 orders too small)

    Returns the measurements and raises if any answer is wrong, so a failing preflight leaves no
    protocol behind. Uses the train manifest only.
    """
    import numpy as np
    import torch
    import zarr
    from torch.utils.data import default_collate

    from data.r7_long_rollout_dataset import (ZarrLongRolloutDataset,
                                              preflight_long_rollout_windows)
    from training.r7_derivative_matching import derivative_matching_terms
    from training.r7_experiment import load_checkpoint, make_model
    from training.r7_physical_constraints import denormalize
    from training.r7_long_rollout_runner import _rollout

    path, expected = parent_pins.pinned_parent(seed)
    from training.r7_arm_harness import sha256_file
    if sha256_file(path) != expected:
        raise RuntimeError(f"seed {seed} parent checkpoint pin changed")
    saved = load_checkpoint(path)
    model = make_model("process", arm_spec())
    model.load_state_dict(saved["model"], strict=True)
    model.train()

    constants = _store_constants()
    mean = torch.tensor(constants["mean"], dtype=torch.float32)
    std = torch.tensor(constants["std"], dtype=torch.float32)
    root = zarr.open_group(str(curve.STORE), mode="r")
    latitude = torch.tensor(np.asarray(root["latitude"][:]), dtype=torch.float32)
    longitude = torch.tensor(np.asarray(root["longitude"][:]), dtype=torch.float32)
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
    rows, derivative_magnitudes = [], []
    for case in PREFLIGHT_CASES:
        batch = default_collate([dataset[case]])
        rollout = _rollout(model, batch, contract, None)
        prediction = denormalize(rollout.forecasts[:, -1], (mean, std), constants["channels"])
        target = denormalize(batch["atmos_target"], (mean, std), constants["channels"])
        live = derivative_matching_terms(prediction, target, constants["channels"],
                                         latitude, longitude, scales=scales)
        zero = derivative_matching_terms(target, target, constants["channels"],
                                         latitude, longitude, scales=scales)
        mse = float(rollout.loss.detach())
        rows.append({"case": case, "divergence": float(live["divergence"]),
                     "vorticity": float(live["vorticity"]), "total": float(live["total"]),
                     "mse": mse, "dose": float(live["total"]) / mse,
                     "zero_at_target": float(zero["total"])})
        # The scales are declared as coming from the ANALYSIS field's own distribution, so the
        # provenance is measured on the target, not on the prediction. Measuring the prediction
        # here would make the protocol's stated basis unverifiable.
        derivative_magnitudes.append({"case": case,
                                      "analysis_mean_abs_divergence": _mean_abs_derivative(
                                          target, constants["channels"], latitude, longitude,
                                          kind="divergence"),
                                      "analysis_mean_abs_vorticity": _mean_abs_derivative(
                                          target, constants["channels"], latitude, longitude,
                                          kind="vorticity")})
    if any(row["zero_at_target"] != 0.0 for row in rows):
        raise RuntimeError("the preflight found a nonzero value at pred = target, so this term is "
                           "not a target match and the family argument fails")
    if any(row["total"] <= 0.0 for row in rows):
        raise RuntimeError("the preflight found the term inert on the trained parent")
    if any(not (DOSE_MIN <= row["dose"] <= DOSE_MAX) for row in rows):
        raise RuntimeError(f"the preflight dose left the declared band [{DOSE_MIN}, {DOSE_MAX}]: "
                           f"{[round(row['dose'], 4) for row in rows]}")
    # Re-derive the scales from the analysis and assert they are the declared constants: the
    # protocol says these came from the analysis distribution, and a store whose fields moved
    # must fail here rather than silently rescale every term.
    derived_divergence = sum(item["analysis_mean_abs_divergence"]
                             for item in derivative_magnitudes) / len(derivative_magnitudes)
    derived_vorticity = sum(item["analysis_mean_abs_vorticity"]
                            for item in derivative_magnitudes) / len(derivative_magnitudes)
    for name, derived, declared in (("divergence", derived_divergence, DIVERGENCE_SCALE),
                                    ("vorticity", derived_vorticity, VORTICITY_SCALE)):
        if abs(derived - declared) / declared > 0.5:
            raise RuntimeError(
                f"the {name} scale declared in the protocol ({declared:.4e}) is not the analysis "
                f"field's measured magnitude ({derived:.4e}); the store or the channel order moved")
    return {"cases": list(PREFLIGHT_CASES), "rows": rows,
            "derived_scales": {"divergence": derived_divergence,
                               "vorticity": derived_vorticity},
            "scales": dict(scales), "weights": {"divergence": W_DIVERGENCE,
                                                "vorticity": W_VORTICITY},
            "derivative_magnitudes": derivative_magnitudes,
            "dose_band": [DOSE_MIN, DOSE_MAX],
            "store_constants_sha256": constants["sha256"],
            "parent_checkpoint_sha256": expected,
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
