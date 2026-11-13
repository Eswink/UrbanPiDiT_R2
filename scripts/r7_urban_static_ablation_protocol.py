"""S3 urban static covariates: the three-arm protocol layer (control / treatment / placebo).

Decision 0113 freezes *what this round is*: one factor with three levels, all sharing one code
revision, one parent, one seed and one update count, differing **only** in the values of two
persistent buffers (``backbone.static_conditioning_pathway.field`` and ``.projection``).
``model/**`` is never touched: ``model_code_digest()`` covers every ``.py`` under ``model/`` and
``load_checkpoint`` requires checkpoint and tree to agree, so any model edit would make all three
registered parents unloadable - the exact failure the wiring round already paid for.

Why a placebo arm rather than on/off. ``StaticConditioning.projection`` is a fixed
``randn(16, 192, seed=0)/sqrt(16)`` (``model/static_conditioning_r7.py:178-184``, a buffer, not a
parameter), so *any* spatially varying 8-field yields distinct per-token codes; and the trunk
already carries ``position_features()``'s ``sin/cos(lat)``/``sin/cos(lon)`` harmonics. A two-arm
positive therefore cannot exclude "the pathway is a finer position lookup". The placebo arm -
the same published 8-field with each covariate's cells spatially permuted, same published
projection - makes that alternative measurable instead of arguable.

**A measured confound this module records rather than hides.** A per-cell permutation leaves each
covariate's raw multiset bit-identical after sorting (the criterion the round freezes), but the
pathway does not see cells - it sees 2x2 patch means (``static_features``). Permuting cells
therefore *attenuates* the token-level signal: measured on this product, the placebo's per-token
feature spread is 0.83x and its projected-code spread 0.69x the treatment's, while its spatial
autocorrelation collapses from 0.74 to 0.00. So the placebo differs from the treatment in **two**
ways (no spatial structure, lower code amplitude), and exit (f) must be read with that in mind:
a true (f) shows that the coherent full-amplitude static term beats a structureless attenuated
one - it does not by itself separate "structure" from "amplitude". ``preflight`` measures that
ratio, and ``family_argument`` says what (f) does and does not license. The alternative
constructions (block permutations) were measured too and are recorded in the preflight as
rejected candidates, so the trade-off is visible rather than unconsidered.

**What this round does not claim.** The static field is a *derived statistic* of real native
observations aggregated to 0.25 deg cell edges, not an observational truth field (decision 0063),
and ``pop_density_km2`` has a single source because LandScan is account-gated. The 2023 test split
is never read: ``r = 0``.
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
from scripts import r7_urban_static_placebo as placebo_mod
from scripts import study_r7_s3_wide_single_factor as wide
from training import r7_device_selection as devices

FORMAT = "r7-s3-urban-static-ablation-protocol-v1"
STAGE = "S3-URBAN-STATIC-ABLATION"
#: The three levels of the single factor. ``control`` constructs no pathway at all.
ARMS = ("control", "treatment", "placebo")
CONTROL, TREATMENT, PLACEBO = ARMS
SEEDS = (41, 42, 43)
UPDATES = 4000
STEPS = curve.STEPS                       # K stays at the registered 4
BATCH_SIZE = 4                            # the registered batch, not ``wide``'s 1
LEADS = wide.EVALUATION_LEADS             # (6, 12, 24, 48, 72)
#: Fixed seed for the spatial permutation. Frozen so the placebo is one reproducible field, not a
#: family of them; scanning it would be a weight scan in disguise and is forbidden.
PLACEBO_SEED = placebo_mod.PLACEBO_SEED
#: The two buffers the on arms add. Named so a rename in ``model/`` fails loudly instead of
#: letting this round reconcile nothing.
BUFFER_KEYS = ("backbone.static_conditioning_pathway.field",
               "backbone.static_conditioning_pathway.projection")
ENTRY_FILES = ("scripts/study_r7_urban_static_ablation.py",
               "scripts/r7_urban_static_ablation_protocol.py",
               "tools/reconcile_r7_static_parent.py")

#: Measured basis for the duration envelope: point 3's 1.1096 s/update at batch 4, and the
#: measured five-lead scoring cost of one arm (Stage A, ``SCORING_ALLOWANCE_SECONDS``).
MEASURED_SECONDS_PER_UPDATE = 1.1096
MEASURED_ON_POINT = "p4000"
SCORING_SECONDS_PER_ARM = 2588.1
ESTIMATED_PEAK_BYTES = 9569304576
HEADROOM_MARGIN_BYTES = 512 * 2**20

LIMITATIONS = [
    "scientific_claim: false - this round reports paired arm differences on one model, one "
    "instance, one resolution (65x65) and one region (27-43N / 107-123E); it is not a claim "
    "about global weather forecasting and not a claim about the real atmospheric effect of "
    "urbanisation",
    "the placebo's per-cell permutation preserves each covariate's raw multiset exactly but the "
    "pathway reads 2x2 patch means, so the placebo's token-level amplitude is measured lower than "
    "the treatment's (see preflight.placebo_contrast); exit (f) cannot separate 'spatial "
    "structure' from 'code amplitude' on its own",
    "the static field is a derived statistic of real native observations aggregated to 0.25 deg "
    "cell edges, NOT interpolation and not an observational truth field (decision 0063); the "
    "result's domain of applicability is this published covariate set",
    "pop_density_km2 has a single source: LandScan is account-gated and was recorded "
    "failed-no-fallback, so no cross-source consistency check exists for the population layer",
    "GHSL built_surface and population are five-yearly cross-sensor composites, not annual "
    "observations; they are 'that epoch's product', not 2020 truth",
    "the five lc_* shares do not sum to 1 (unmapped share about 0.23-0.26), so lc_bare_frac is "
    "bare land's share among mapped classes, not its absolute share",
    "the projection is a fixed random matrix, so no per-covariate attribution is available: even "
    "(a) and (f) both true would not license 'built fraction is the driver'",
    "4,000 updates is borrowed from Stage A's dose argument (it moved its own direct target three "
    "seeds consistently at this budget); a negative here therefore cannot be attributed to a dose "
    "too small to act, but a longer adaptation period is not excluded",
    "three seeds are a paired comparison against same-code arms, not a significance claim; no "
    "multiplicity correction is applied because the five exits are predeclared and conjunctive",
    "the 2023 test split is never read: r = 0, and every number here is 2022 validation",
]


#: The two numbers the goal document freezes: a soft 115,200 s and a hard 230,400 s. The
#: *per-seed* allowance is derived from the three-arm measurement below instead of being copied
#: from the brief's figure, because three arms cost about 1.5x Stage A's two and a per-seed cap of
#: 115,200 s would have truncated the round's own estimate. The brief's two numbers are unchanged
#: and are what the round's total is accounted against.
PLANNED_SECONDS = 115200.0
HARD_CAP_SECONDS = 230400.0


def _per_seed_envelope() -> float:
    """Per-seed hard allowance: three times the three-arm estimate, quantized to whole hours.

    Sized the way Stage A sized its own per-seed cap (3x its estimate, quantized up), not by
    copying a number that was derived for a two-arm seed.
    """
    return float(math.ceil(3 * _per_seed_estimate() / 3600) * 3600)


def _per_seed_estimate() -> float:
    """Training plus scoring for all three arms of one seed: an estimate, from measurements."""
    return len(ARMS) * UPDATES * MEASURED_SECONDS_PER_UPDATE + len(ARMS) * SCORING_SECONDS_PER_ARM


PER_SEED_SECONDS = _per_seed_envelope()


def budgets() -> dict:
    """The frozen duration envelope, with its measured basis stated rather than estimated."""
    train = UPDATES * MEASURED_SECONDS_PER_UPDATE
    scoring = len(ARMS) * SCORING_SECONDS_PER_ARM
    per_seed_estimate = _per_seed_estimate()
    return {
        "measured_seconds_per_update": MEASURED_SECONDS_PER_UPDATE,
        "measured_on_point": MEASURED_ON_POINT,
        "train_seconds_per_arm": round(train, 1),
        "scoring_seconds_per_seed": round(scoring, 1),
        "per_seed_estimate_seconds": round(per_seed_estimate, 1),
        "per_seed_estimate_basis": (
            f"{len(ARMS)} arms x {UPDATES} updates x {MEASURED_ON_POINT}'s measured "
            f"{MEASURED_SECONDS_PER_UPDATE} s/update at batch 4, plus {len(ARMS)} x the measured "
            f"five-lead scoring cost of one arm ({SCORING_SECONDS_PER_ARM} s): an estimate from "
            "another round's measurement, not a measurement of this round"),
        "per_seed_seconds": PER_SEED_SECONDS,
        "per_seed_cap_over_estimate": round(PER_SEED_SECONDS / per_seed_estimate, 3),
        "planned_seconds": PLANNED_SECONDS,
        "hard_cap_seconds": HARD_CAP_SECONDS,
        "soft_rule": ("soft overrun continues and is recorded; only the hard cap truncates a "
                      "worker, and a truncated worker is not a pass"),
        "decision": "0030",
    }


def arm_spec(arm: str):
    """The archived process spec, plus this arm's static switch.

    ``control`` must reproduce the registered spec **word for word** - no ``static_conditioning``
    key at all - because exit (d) is the claim that the control arm is the pre-existing
    computation. ``treatment`` and ``placebo`` are the *same* spec as each other: they differ only
    in buffer values, which is the whole point of freezing the difference at the state-dict level.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; this round has exactly {ARMS}")
    spec = dict(curve.model_spec())
    if int(spec.get("dim", 0)) != 192 or int(spec.get("depth", 0)) != 4:
        raise RuntimeError("the archived process spec is not the frozen dim-192 depth-4 trunk")
    if "static_conditioning" in spec:
        raise RuntimeError("the archived spec already carries a static switch; this round's "
                           "factor would then not be the switch it declares")
    if arm != CONTROL:
        spec["static_conditioning"] = True
    return spec


def _static_product():
    """The published field, its identity and the module's own view of it."""
    from model import static_conditioning_r7 as static
    manifest = static.require_static_identity()
    fields, variables = static.load_static_fields()
    if tuple(fields.shape[-2:]) != tuple(manifest["grid"]["points"]):
        raise RuntimeError("the published field's grid does not match its manifest")
    return static, manifest, fields, variables


def _load_parent_state(seed):
    """The pinned p1600 parent state, with its pin, its digest and its code identity checked."""
    from training.r7_experiment import load_checkpoint, model_code_digest
    from training.r7_arm_harness import sha256_file
    from training.r7_autoregressive_runner import _state_digest
    if model_code_digest() != parent_pins.MODEL_CODE_SHA256:
        raise RuntimeError(
            f"model_code_sha256 is {model_code_digest()}, not the parent pin "
            f"{parent_pins.MODEL_CODE_SHA256}: model/** moved, so no registered parent can be "
            "loaded and the three arms cannot share one code revision. Do not migrate around "
            "this - find the edit.")
    path, expected = parent_pins.pinned_parent(seed)
    if sha256_file(path) != expected:
        raise RuntimeError(f"seed {seed} parent checkpoint pin changed")
    saved = load_checkpoint(path)
    if saved["updates"] != parent_pins.ENDPOINT:
        raise RuntimeError(f"seed {seed} parent is not the frozen endpoint {parent_pins.ENDPOINT}")
    state = saved["model"]
    digest = _state_digest(state)
    if digest != parent_pins.MIGRATED_PARENT_STATE_DIGEST[seed]:
        raise RuntimeError(f"seed {seed}: the imported state is not the pinned weights")
    if any("static_conditioning_pathway" in name for name in state):
        raise RuntimeError(f"seed {seed}: the parent already carries static buffers; it is not the "
                           "off-arm parent this round expects")
    return {"path": path, "sha256": expected, "updates": int(saved["updates"]),
            "state": state, "state_digest": digest, "model_code_sha256": parent_pins.MODEL_CODE_SHA256}


def _arm_state(parent, treatment_pair, placebo, *, arm):
    """The one state dict this arm loads; the mechanism lives in the placebo/arm support module."""
    return placebo_mod.arm_state(parent, treatment_pair, placebo, arm=arm, arms=ARMS,
                                 control=CONTROL, buffer_keys=BUFFER_KEYS)


def _build(arm):
    return placebo_mod.build_arm(arm, arm_spec)


def _load_strict(state, arm):
    return placebo_mod.load_strict(state, arm, arm_spec)


def _parameter_measurements(model, arm):
    return placebo_mod.parameter_measurements(model, arm, BUFFER_KEYS)



def preflight(fields, variables, manifest) -> dict:
    """Exit (e), computed end to end; every clause measured, none written as a literal.

    Five required clauses - all finite, identity verified, per-arm parameter delta measured zero,
    placebo multiset bit-identical per covariate, and 3 seeds x 3 arms loading ``strict=True`` -
    plus the placebo contrast that decides how exit (f) may be read.
    """
    import torch
    from training.r7_arm_harness import sha256_file
    finite = bool(torch.isfinite(fields).all())
    shape_ok = tuple(int(value) for value in fields.shape) == (len(variables), 65, 65)
    treatment_pair = (fields, variables)
    placebo, permutations = placebo_mod.placebo_field(fields)
    contrast = placebo_mod.placebo_contrast(fields, placebo, variables)

    seed_rows, parameters = {}, {}
    for arm in ARMS:
        built = _build(arm)
        parameters[arm] = _parameter_measurements(built, arm)
    baseline = parameters[CONTROL]["parameters"]
    for arm in ARMS:
        parameters[arm]["parameter_delta"] = parameters[arm]["parameters"] - baseline
    parameter_delta_zero = all(parameters[arm]["parameter_delta"] == 0 for arm in ARMS)
    buffers_are_not_parameters = all(not parameters[arm]["static_buffers_are_parameters"]
                                     for arm in ARMS)
    pathway_presence = {"control": parameters[CONTROL]["has_static_pathway"],
                        "treatment": parameters[TREATMENT]["has_static_pathway"],
                        "placebo": parameters[PLACEBO]["has_static_pathway"]}

    for seed in SEEDS:
        parent = _load_parent_state(seed)
        arms = {}
        for arm in ARMS:
            state = _arm_state(parent, treatment_pair, placebo, arm=arm)
            model = _load_strict(state, arm)
            from training.r7_autoregressive_runner import _state_digest as state_digest
            digest = state_digest(model.state_dict())
            arms[arm] = {
                "strict_load_passed": True,
                "state_tensors": len(state),
                "initial_state_digest": digest,
                "parameter_delta": parameters[arm]["parameter_delta"],
                "added_keys_vs_parent": sorted(set(state) - set(parent["state"])),
            }
        seed_rows[str(seed)] = {
            "parent_checkpoint": str(parent["path"]),
            "parent_checkpoint_sha256": parent["sha256"],
            "parent_endpoint_updates": parent["updates"],
            "parent_state_digest": parent["state_digest"],
            "parent_model_code_sha256": parent["model_code_sha256"],
            "arms": arms,
        }
    strict_all = all(row["strict_load_passed"]
                     for seed_row in seed_rows.values() for row in seed_row["arms"].values())
    # The two on arms must differ from each other *only* in the field buffer, or the contrast is
    # not the one the round claims. Measured on the actual tensors, not inferred from the code.
    differing = _differing_keys(treatment_pair, placebo)
    only_field_differs = differing == [BUFFER_KEYS[0]]

    e_passed = bool(finite and shape_ok and strict_all and parameter_delta_zero
                    and buffers_are_not_parameters and contrast["raw_multiset_all_identical"]
                    and only_field_differs and pathway_presence == {
                        "control": False, "treatment": True, "placebo": True})
    return {
        "e_preflight_passed": e_passed,
        "static_identity": manifest["static_identity"],
        "static_identity_verified": True,
        "fields_all_finite": finite,
        "fields_shape": list(fields.shape),
        "fields_shape_is_8x65x65": shape_ok,
        "variables": list(variables),
        "fields_sha256": sha256_file(_static_product()[0].STATIC_FIELDS),
        "manifest_sha256": sha256_file(_static_product()[0].STATIC_MANIFEST),
        "static_pathway_present": pathway_presence,
        "parameters": parameters,
        "parameter_delta_zero_per_arm": parameter_delta_zero,
        "static_buffers_are_not_parameters": buffers_are_not_parameters,
        "placebo_seed": PLACEBO_SEED,
        "placebo_permutation_identity_cells": [
            int((order == torch.arange(order.numel())).sum()) for order in permutations],
        "placebo_contrast": contrast,
        "seeds": seed_rows,
        "strict_load_all_seeds_all_arms": strict_all,
        "treatment_vs_placebo_differing_keys": differing,
        "treatment_vs_placebo_differ_only_in_field": only_field_differs,
        "clauses": {
            "eight_layers_finite_on_65x65": bool(finite and shape_ok),
            "static_identity_verified": True,
            "parameter_delta_measured_zero_all_arms": parameter_delta_zero,
            "placebo_multiset_bitwise_identical": contrast["raw_multiset_all_identical"],
            "strict_load_all_seeds_all_arms": strict_all,
            "treatment_placebo_differ_only_in_field": only_field_differs,
        },
    }


def _differing_keys(treatment_pair, placebo) -> list:
    """Which state keys differ between the treatment and placebo arms, measured on the tensors."""
    import torch
    parent = _load_parent_state(SEEDS[0])
    treatment = _arm_state(parent, treatment_pair, placebo, arm=TREATMENT)
    placebo_state = _arm_state(parent, treatment_pair, placebo, arm=PLACEBO)
    if set(treatment) != set(placebo_state):
        raise RuntimeError("the two on arms do not even carry the same key set")
    return sorted(key for key in treatment
                  if not torch.equal(treatment[key].to("cpu"), placebo_state[key].to("cpu")))


def frozen_constants() -> dict:
    """Everything frozen before any training starts."""
    static, manifest, fields, variables = _static_product()
    variability = None
    try:
        variability = placebo_mod.temporal_variability(curve.STORE)
    except Exception as error:      # recorded, never silently skipped
        variability = {"measured": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "format": FORMAT, "stage": STAGE,
        "instance": str(curve.INSTANCE), "store": str(curve.STORE),
        "source_sha256": curve.SOURCE_SHA256, "store_shape": list(curve.STORE_SHAPE),
        "split_years": {"train": curve.CLIMATOLOGY_YEARS, "val": [2022], "test": [2023]},
        "train_manifest": str(curve.TRAIN_MANIFEST),
        "val_manifest": str(curve.VAL_MANIFEST),
        "test_manifest_never_read": str(curve.TEST_MANIFEST),
        "usable_windows": curve.USABLE_WINDOWS,
        "region": {"points": [65, 65], "boundary_margins": [1], "scored_region": "full",
                   "n_grid_points": 4225,
                   "note": "the whole native 65x65 grid; interior_32 is 1x1 here"},
        "arms": {arm: {"name": arm,
                       "static_conditioning": arm != CONTROL,
                       "field": ("absent" if arm == CONTROL else
                                 "published 8-layer product" if arm == TREATMENT else
                                 "published 8-layer product, per-cell permuted"),
                       "projection": ("absent" if arm == CONTROL else "published fixed projection"),
                       "updates": UPDATES, "steps": STEPS, "batch_size": BATCH_SIZE,
                       "seeds": list(SEEDS)} for arm in ARMS},
        "factor": {
            "name": "urban static covariate pathway, three levels",
            "levels": list(ARMS),
            "differing_quantity": "the two persistent buffers "
                                  f"{list(BUFFER_KEYS)} and nothing else",
            "shared": ["code revision (model_code_sha256)", "parent checkpoint", "seed",
                       "update count", "batch size", "lr", "warmup", "weight decay", "clip",
                       "physical weights", "train manifest", "val manifest", "scoring region",
                       "climatology denominator"],
            "not_shared_by_design": "buffer values only",
            "parameter_delta": 0,
            "parameter_delta_measured": True},
        "recipe": {"mode": "long_rollout", "updates": UPDATES, "steps": STEPS,
                   "lr": curve.LR, "warmup": curve.WARMUP,
                   "weight_decay": curve.WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "clip": curve.CLIP, "bf16": wide.BF16,
                   "checkpoint_every": curve.CHECKPOINT_EVERY,
                   "physical_weights": list(curve.PHYSICAL_WEIGHTS),
                   "reasoning_steps": curve.REASONING_STEPS,
                   "supervision": "full_grid (no spatial mask on the native 65x65 grid)",
                   "initialization": ("pinned p1600 registered parent; state dict plus the two "
                                      "published buffers, loaded strict=True"),
                   "epochs": round(UPDATES * BATCH_SIZE / curve.USABLE_WINDOWS, 6)},
        "checkpoint_policy": {"checkpoint_every": curve.CHECKPOINT_EVERY,
                              "retained": "multiples of checkpoint_every plus the endpoint",
                              "selection": "frozen endpoint; no validation selection",
                              "decision": "0087"},
        "budgets": budgets(),
        "static": {
            "static_identity": manifest["static_identity"],
            "output_sha256": manifest["identity_components"]["output_sha256"],
            "derived_statistic_not_truth": manifest["derived_statistic_not_truth"],
            "variables": list(variables), "variable_count": len(variables),
            "grid": {"points": manifest["grid"]["points"],
                     "spacing_deg": manifest["grid"]["spacing_deg"],
                     "lat_order": manifest["grid"]["lat_order"],
                     "lon_order": manifest["grid"]["lon_order"]},
            "expansion_per_variable": static.EXPANSION_PER_VARIABLE,
            "scales": dict(static.STATIC_SCALES),
            "placebo_seed": PLACEBO_SEED,
            "placebo_rule": ("each covariate's 65x65 cells permuted independently under one fixed "
                             "seed; channels are never shuffled between variables; each "
                             "covariate's mask multiset stays bit-identical after sorting"),
            "parameter_delta": 0},
        "family_argument": placebo_mod.family_argument(fields, variables, variability),
        "exits": {
            "a": "72 h t2m skill: paired delta (treatment - control) >= 0 for all three seeds",
            "b": "24 h and 48 h t2m skill: paired delta (treatment - control) >= 0 for all seeds",
            "c": "6 h and 12 h t2m skill: paired delta (treatment - control) >= -0.005",
            "d": ("control arm bitwise-equivalent to the pre-change computation, carried by the "
                  "static_conditioning=False -> pathway=None code path plus a test that fails if "
                  "it is violated; never by a reading (decision 0109)"),
            "e": ("preflight: 8 layers finite on 65x65, static_identity verified, per-arm "
                  "parameter delta measured zero, placebo multiset bit-identical per covariate, "
                  "3 seeds x 3 arms strict load"),
            "f": ("attribution: 72 h t2m RMSE improvement of treatment over placebo "
                  "(placebo - treatment) >= 0.001 K")},
        "decision": {
            "advance": "all of (a)(b)(c)(e)(f) true -> advance-to-independent-review",
            "attribution_false": ("(a) true but (f) false -> the round is recorded NEGATIVE: the "
                                  "static effect is independent of spatial structure and "
                                  "equivalent to a position code"),
            "close": "any of (a)(b)(c) false -> permanently close the urban static input branch",
            "f_is_attribution_not_success": (
                "(f) asks whether the effect depends on the covariates' spatial arrangement; it is "
                "not a success criterion and never replaces (a)"),
            "text": ("(a)(b)(c) are treatment-vs-control paired skill deltas; (f) is "
                     "treatment-vs-placebo in kelvin. (a) true with (f) false is a negative result, "
                     "not a partial success. No arm may be reported as SOTA, no arm outperforms "
                     "a published baseline by this round's evidence, and no exit licenses a claim "
                     "about the real atmospheric effect of urbanisation.")},
        "test_read": False,
        "test_read_policy": ("the 2023 test split is never opened; only train (fit) and val "
                            "(scoring) manifests are read, so r stays 0"),
        "scientific_claim": False, "limitations": list(LIMITATIONS),
    }


def execution_files():
    """Decision 0067: this round's import closure, from a named entry file."""
    from tools.r7_explicit_inventory import explicit_execution_files
    return explicit_execution_files(ENTRY_FILES[0], extra=ENTRY_FILES[1:])


def freeze(output, *, started, deadline, code_archive, code_commit, code_sha256):
    """Write the preparation protocol; the preflight runs at prepare, before promotion."""
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
    """Add the live identity and the measured exit (e); a failing preflight writes no protocol."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import (canonical_digest, dataset_identity, model_code_digest)
    from training.r7_long_rollout_runner import training_code_digest
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    output = Path(output)
    preliminary = json.loads((output / "preparation_protocol.json").read_text(encoding="utf-8"))
    if preliminary["protocol_sha256"] != canonical_digest(
            {key: value for key, value in preliminary.items() if key != "protocol_sha256"}):
        raise RuntimeError("preparation protocol digest mismatch")
    _static_module, manifest, fields, variables = _static_product()
    windows = preflight_long_rollout_windows(curve.TRAIN_MANIFEST, physical_steps=12)
    train_identity, _ = dataset_identity(curve.TRAIN_MANIFEST)
    measurements = preflight(fields, variables, manifest)
    if not measurements["e_preflight_passed"]:
        raise RuntimeError(f"exit (e) failed before any GPU second was spent: "
                           f"{measurements['clauses']}")
    body = {**preliminary,
            "model_code_sha256": model_code_digest(),
            "training_code_sha256": training_code_digest(),
            "train_data_identity": train_identity,
            "val_data_identity": dataset_identity(curve.VAL_MANIFEST)[0],
            "train_windows": windows,
            "preflight": measurements,
            "execution_files_sha256": {name: sha256_file(ROOT / name)
                                       for name in execution_files()},
            "arms_parent": {"pins": {}, "checkpoints": parent_pins.pins()}}
    parent_pins.install_parent_pins({"arms": {"parent": body["arms_parent"]}})
    body.pop("protocol_sha256", None)
    body["protocol_sha256"] = canonical_digest(body)
    wide.write_exclusive(output / "prepared_protocol.json", body)
    return body


def validate_protocol(output):
    """Re-check every frozen identity, including the preflight and the arm definitions."""
    from training.r7_arm_harness import sha256_file
    from training.r7_experiment import canonical_digest, model_code_digest
    from training.r7_long_rollout_runner import training_code_digest
    body = json.loads((Path(output) / "protocol.json").read_text(encoding="utf-8"))
    if body["protocol_sha256"] != canonical_digest(
            {key: value for key, value in body.items() if key != "protocol_sha256"}):
        raise RuntimeError("frozen protocol digest mismatch")
    if model_code_digest() != body["model_code_sha256"]:
        raise RuntimeError("the model revision moved after freeze; the three arms no longer share "
                           "one code identity")
    if model_code_digest() != parent_pins.MODEL_CODE_SHA256:
        raise RuntimeError("model_code_sha256 is not the parent pin any more")
    if training_code_digest() != body["training_code_sha256"]:
        raise RuntimeError("the training revision moved after freeze")
    for name, expected in body["execution_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"execution source changed: {name}")
    if set(body["execution_files_sha256"]) != set(execution_files()):
        raise RuntimeError("incomplete execution source inventory")
    if body.get("test_read") is not False or body.get("scientific_claim") is not False:
        raise RuntimeError("this round is validation-only and claims nothing")
    measured = body["preflight"]
    if measured.get("e_preflight_passed") is not True:
        raise RuntimeError("the frozen preflight did not pass")
    if not measured["parameter_delta_zero_per_arm"]:
        raise RuntimeError("the frozen preflight recorded a non-zero parameter delta")
    if not measured["strict_load_all_seeds_all_arms"]:
        raise RuntimeError("the frozen preflight did not load every seed and arm strictly")
    if sorted(measured["parameters"]) != sorted(ARMS):
        raise RuntimeError("the frozen preflight does not cover every arm")
    if tuple(body["static"]["variables"]) != tuple(_static_product()[3]):
        raise RuntimeError("the frozen static variable list moved")
    if body["static"]["static_identity"] != _static_product()[1]["static_identity"]:
        raise RuntimeError("the static product no longer carries the frozen identity")
    for arm in ARMS:
        if arm not in body["arms"]:
            raise RuntimeError(f"the frozen protocol is missing arm {arm!r}")
    return body


def main(argv=None) -> int:
    """``--preflight-only`` is the cheap gate the workflow runs before freezing anything.

    It prints one JSON document and exits non-zero when exit (e) is false, so the caller reads a
    status rather than parsing prose. It never trains, never scores and never writes into the
    repository: the output path is the caller's.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)
    if not arguments.preflight_only:
        raise SystemExit("this module is the protocol layer; the round runs through "
                         "`r7_round.py stage`. Use --preflight-only for the (e) gate.")
    from scripts.r7_m3_offline import deny_network
    deny_network()
    _static_module, manifest, fields, variables = _static_product()
    measurements = preflight(fields, variables, manifest)
    payload = {"format": FORMAT, "stage": STAGE, "scientific_claim": False, "test_read": False,
               "gpu_hours": 0.0, **measurements}
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"e_preflight_passed": measurements["e_preflight_passed"],
                      "clauses": measurements["clauses"],
                      "parameter_delta_zero_per_arm": measurements["parameter_delta_zero_per_arm"],
                      "strict_load_all_seeds_all_arms":
                          measurements["strict_load_all_seeds_all_arms"],
                      "placebo_multiset_bitwise_identical":
                          measurements["placebo_contrast"]["raw_multiset_all_identical"],
                      "code_std_ratio_placebo_over_treatment":
                          measurements["placebo_contrast"]["code_std_ratio_placebo_over_treatment"],
                      "out": str(arguments.out) if arguments.out else None}, indent=2))
    return 0 if measurements["e_preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
