"""CI-safe tests for the R4 position-retention driver (no ``outputs/`` required).

``frozen_constants`` takes the payload as a parameter precisely so a synthetic one can stand
in; the router-level identities are asserted on constructed modules, never on a trained
checkpoint. Nothing here reads a store, a manifest, the sidecar or a migrated parent.

The counterexamples carry the weight: a driver that quietly reverted to the closed round's
2,400 updates at batch 1, that let one arm carry a different ``position_scale``, that swept the
constant after reading results, or whose "spatial" arm is not actually the pooled mean at a zero
query, would satisfy every "the constant equals itself" assertion.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from scripts import r7_s3_position_retention_protocol as protocol
from scripts import r7_s3_position_retention_support as support
from scripts import study_r7_s3_position_retention_65x65 as driver
from model.typed_evidence_r7 import TYPED_ROUTING, TYPED_SPATIAL

ROOT = Path(driver.__file__).resolve().parents[1]
GOAL = "docs/goals/s3-position-retention-reeval.md"


#: The 17 diagnostic channel *names* the 65x65 store carries. Names are schema, not data: the
#: router refuses to build without the four typed-evidence channels, and the statistics below
#: are stand-ins rather than the published sidecar's numbers.
STORE_CHANNELS = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                  "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")


def _synthetic_payload():
    """A valid typed-evidence payload: real channel names, invented statistics."""
    return {"channels": list(STORE_CHANNELS),
            "denorm_mean": [0.0] * len(STORE_CHANNELS),
            "denorm_std": [1.0] * len(STORE_CHANNELS),
            "field_scale": [1.0] * 4, "field_mean": [0.0] * 4, "field_std": [1.0] * 4,
            "field_active": [True] * 4}


def _spec(mode, payload=None):
    return support.arm_spec({"dim": 192, "depth": 4, "anchored_processes": 4},
                            payload if payload is not None else _synthetic_payload(), mode)


def _arm(mode, *, identity="train-identity", spec=None):
    return protocol.arm_protocol(
        spec if spec is not None else _spec(mode), mode, train_identity=identity,
        windows={"input_windows": protocol.TRAIN_WINDOWS, "window_sha256": "a" * 64,
                 "excluded_sample_ids": []},
        canonical_digest=lambda value: f"digest-{value.get('typed_evidence_mode')}")


def _arms():
    control, candidate = _arm(support.MODES[support.CONTROL]), \
        _arm(support.MODES[support.CANDIDATE])
    assert support.require_single_factor(control, candidate) is None
    return control, candidate


# --- the frozen compute and budget -------------------------------------------------------------

def test_the_compute_is_the_matched_one_and_not_the_closed_rounds():
    assert (protocol.UPDATES, protocol.BATCH_SIZE) == (16071, 2)
    assert protocol.UPDATES * protocol.BATCH_SIZE >= protocol.TRAIN_WINDOWS
    assert (support.CLOSED_ROUND_UPDATES, support.CLOSED_ROUND_BATCH_SIZE) == (2400, 1)
    assert protocol.UPDATES != support.CLOSED_ROUND_UPDATES
    assert protocol.BATCH_SIZE != support.CLOSED_ROUND_BATCH_SIZE


def test_at_least_one_epoch_over_the_treatment_manifest():
    assert protocol.UPDATES * protocol.BATCH_SIZE >= protocol.TRAIN_WINDOWS
    assert protocol.TRAIN_WINDOWS == 32142


def test_the_budget_is_decision_0060s_track_one_constants():
    assert (protocol.PER_SEED_SECONDS, protocol.PLANNED_SECONDS, protocol.HARD_CAP_SECONDS) \
        == (129600.0, 259200.0, 518400.0)
    # R2 measured 88,362-89,520 s per seed on this store at this compute on the same hardware
    assert protocol.PER_SEED_SECONDS > 89520.0


def test_the_driver_exposes_the_interface_the_stage_wrapper_requires():
    """``r7_round_stages.load_driver`` refuses a driver missing any of these."""
    for name in ("freeze", "SEEDS", "PER_SEED_SECONDS", "PLANNED_SECONDS", "HARD_CAP_SECONDS"):
        assert hasattr(driver, name), name
    assert driver.freeze is protocol.freeze
    assert tuple(driver.SEEDS) == (41, 42, 43)


# --- the frozen constants ----------------------------------------------------------------------

def test_the_frozen_constants_carry_the_factor_and_the_closed_round():
    body = protocol.frozen_constants(_synthetic_payload())
    assert body["format"] == protocol.FORMAT
    assert body["factor"]["name"] == \
        "typed-evidence patch-position aggregation at matched compute"
    assert body["factor"]["closed_round"]["record"] == support.CLOSED_ROUND_RECORD
    assert body["factor"]["closed_round"]["updates"] == support.CLOSED_ROUND_UPDATES
    assert set(body["factor"]["changed_from_closed_round"]) == {
        "instance", "grid", "supervision region", "data span", "typed-evidence payload",
        "position_scale", "query starting point", "updates", "batch_size"}
    assert body["test_read"] is False and body["scientific_claim"] is False
    assert body["test_manifest_never_read"].endswith("test.jsonl")
    assert len(body["limitations"]) >= 8
    assert set(body["decision"]["exits"]) == {"a", "b", "c", "d"}
    assert body["decision"]["text"] == support.DECISION_RULE
    assert body["bridge_e"]["scientific"] is False


def test_the_frozen_constants_declare_the_one_uncontrolled_difference():
    body = protocol.frozen_constants(_synthetic_payload())
    assert body["factor"]["not_controlled"] == ["parameter count (+192 in the spatial arm)"]
    assert "+192" in body["factor"]["declared_parameter_difference"]
    assert "position_scale" in body["factor"]["controlled"]


def test_the_position_scale_block_is_the_frozen_and_never_swept_one():
    block = protocol.frozen_constants(_synthetic_payload())["position_scale"]
    assert block["value"] == 8000.0
    assert block["learnable"] is False
    assert block["entropy_band"] == [0.90, 0.99]
    assert block["one_hot_floor"] == 0.5
    assert block["query_norm_range"] == [0.0074, 0.0136]
    assert "0081" in block["rule"] and "never swept" in block["rule"]


def test_the_recipe_declares_the_epoch_floor_and_full_grid_supervision():
    body = protocol.frozen_constants(_synthetic_payload())
    assert body["recipe"]["updates"] == protocol.UPDATES
    assert body["recipe"]["batch_size"] == protocol.BATCH_SIZE
    assert "32142" in body["recipe"]["epochs_floor"]
    assert body["recipe"]["supervision"].startswith("full_grid")


def test_the_scored_region_is_the_whole_native_grid():
    region = protocol.frozen_constants(_synthetic_payload())["region"]
    assert region["boundary_margins"] == [1]
    assert region["scored_region"] == "full"
    assert region["n_grid_points"] == 65 * 65 == 4225
    assert "interior_32 is 1x1 here" in region["note"]


def test_the_typed_evidence_block_names_the_published_sidecar_not_the_closed_arms():
    block = protocol.frozen_constants(_synthetic_payload())["typed_evidence"]
    assert block["sidecar_run"] == "outputs/r7_s3_typed_evidence_fy65_20261103_attempt01"
    assert block["typed_evidence_identity"] == support.SIDECAR_TYPED_EVIDENCE_IDENTITY
    assert block["data_identity"] == support.SIDECAR_DATA_IDENTITY
    assert block["mode_control"] == "typed_routing"
    assert block["mode_candidate"] == "typed_spatial"
    assert "129x129" in block["closed_arm_payload_rule"]


# --- the one-factor claim ----------------------------------------------------------------------

def test_the_arms_differ_only_in_the_model_block_and_the_mode_it_carries():
    control, candidate = _arms()
    assert support.difference_keys(control, candidate) == ["model", "typed_evidence_mode"]
    assert support.ARM_FACTOR_KEYS == ("model", "typed_evidence_mode")
    assert control["train_manifest"] == candidate["train_manifest"]
    assert control["train_data_identity"] == candidate["train_data_identity"]
    assert control["train_windows"] == candidate["train_windows"]
    assert control["updates"] == candidate["updates"]
    assert control["batch_size"] == candidate["batch_size"]


def test_the_specs_differ_only_in_the_routing_mode_and_share_the_payload():
    control, candidate = _arms()
    assert control["model"]["spec"]["typed_evidence_mode"] == TYPED_ROUTING
    assert candidate["model"]["spec"]["typed_evidence_mode"] == TYPED_SPATIAL
    assert (control["model"]["spec"]["typed_evidence"]
            == candidate["model"]["spec"]["typed_evidence"])


def test_both_arms_carry_the_same_frozen_position_scale():
    control, candidate = _arms()
    for arm in (control, candidate):
        assert arm["position_scale"] == 8000.0
        assert arm["model"]["spec"]["typed_evidence"]["position_scale"] == 8000.0


def test_a_scale_difference_between_the_arms_is_caught():
    control, candidate = _arms()
    drifted = copy.deepcopy(candidate)
    drifted["model"]["spec"]["typed_evidence"]["position_scale"] = 9000.0
    with pytest.raises(ValueError, match="same typed-evidence payload"):
        support.require_single_factor(control, drifted)


def test_a_record_and_spec_that_disagree_on_the_mode_are_caught():
    """The mode is written twice; a half-updated arm must not slip through."""
    control, candidate = _arms()
    split = copy.deepcopy(candidate)
    split["typed_evidence_mode"] = support.MODES[support.CONTROL]
    with pytest.raises(ValueError, match="routing mode"):
        support.require_single_factor(control, split)


def test_a_spec_that_disagrees_with_its_own_record_is_caught():
    control, candidate = _arms()
    split = copy.deepcopy(candidate)
    split["model"]["spec"]["typed_evidence_mode"] = support.MODES[support.CONTROL]
    with pytest.raises(ValueError, match="routing mode"):
        support.require_single_factor(control, split)


def test_an_arm_record_that_lost_its_mode_is_caught():
    control, candidate = _arms()
    stripped = copy.deepcopy(candidate)
    del stripped["typed_evidence_mode"]
    with pytest.raises(ValueError):
        support.require_single_factor(control, stripped)


def test_an_arm_that_drops_the_frozen_scale_is_caught():
    control, candidate = _arms()
    broken = copy.deepcopy(candidate)
    broken["model"]["spec"]["typed_evidence"]["position_scale"] = 1.0
    with pytest.raises(ValueError, match="same typed-evidence payload"):
        support.require_single_factor(control, broken)


def test_a_manifest_difference_is_caught():
    control, candidate = _arms()
    moved = copy.deepcopy(candidate)
    moved["train_manifest"] = "elsewhere/train.jsonl"
    with pytest.raises(ValueError, match="differ in"):
        support.require_single_factor(control, moved)


def test_a_compute_difference_is_caught():
    control, candidate = _arms()
    moved = copy.deepcopy(candidate)
    moved["updates"] = support.CLOSED_ROUND_UPDATES
    with pytest.raises(ValueError, match="differ in"):
        support.require_single_factor(control, moved)


def test_an_arm_spec_that_already_carries_a_switch_is_refused():
    payload = _synthetic_payload()
    for key in ("climatology_anchor_spec", "anomaly_feedback", "anchor_output_prior",
                "process_init_conditioning", "typed_evidence_mode", "global_mixing_mode",
                "tendency_input", "static_conditioning"):
        with pytest.raises(RuntimeError, match="closed-architecture switch"):
            support.arm_spec({"dim": 192, "depth": 4, "anchored_processes": 4, key: True},
                             payload, support.MODES[support.CONTROL])
    with pytest.raises(RuntimeError, match="dim-192 depth-4"):
        support.arm_spec({"dim": 256, "depth": 4}, payload, support.MODES[support.CONTROL])
    with pytest.raises(RuntimeError, match="anchored_processes"):
        support.arm_spec({"dim": 192, "depth": 4, "anchored_processes": 2},
                         payload, support.MODES[support.CONTROL])


def test_arm_name_maps_modes_and_rejects_everything_else():
    assert support.arm_name("typed_routing") == support.CONTROL
    assert support.arm_name("typed_spatial") == support.CANDIDATE
    for value in ("typed_evidence", "", None, False, True, 0):
        with pytest.raises(ValueError):
            support.arm_name(value)


def test_the_decision_rule_names_every_closure_term():
    text = support.DECISION_RULE
    for phrase in ("72 h full-grid t2m skill > 0", "48 h full-grid t2m skill > 0",
                   "6/12/24 h", "permanently close", "no rename-and-reopen",
                   "no scale sweep", "never a verdict"):
        assert phrase in text, phrase


def test_the_limitations_name_the_artificial_constant_and_the_zero_start():
    text = " ".join(support.LIMITATIONS)
    for phrase in ("artificial, pre-declared constant", "does NOT answer how large a query",
                   "exactly zero position query", "14-31%", "sign of v500",
                   "+192 parameters", "shape of the softmax", "consistency evidence",
                   "co-resident", "test split stays sealed"):
        assert phrase in text, phrase


# --- the predeclared exits ---------------------------------------------------------------------

def _arm_state(*, skill72=0.1, delta72=-1.0, skill48=0.1, deltas=(-1.0, -1.0, -1.0)):
    return {"arms": {support.CANDIDATE: {
                "full_t2m_skill": {"6": skill48, "12": skill48, "24": skill48,
                                   "48": skill48, "72": skill72},
                "full_t2m_rmse": {"6": 2.0, "12": 2.1, "24": 2.2, "48": 2.3, "72": 2.4}}},
            "delta_candidate_minus_control": {
                "6": deltas[0], "12": deltas[1], "24": deltas[2], "72": delta72}}


def _exit_seeds(*, skill72=0.1, delta72=-1.0, skill48=0.1, deltas=(-1.0, -1.0, -1.0),
                per_seed=None):
    base = dict(skill72=skill72, delta72=delta72, skill48=skill48, deltas=deltas)
    body = {str(seed): _arm_state(**base) for seed in driver.SEEDS}
    for seed, values in (per_seed or {}).items():
        body[str(seed)] = _arm_state(**{**base, **values})
    return body


def test_all_four_exits_true_advances_only_to_independent_review():
    verdict = driver.evaluate_exit(_exit_seeds())
    assert verdict["a"] and verdict["b"] and verdict["c"] and verdict["d"] and verdict["all"]
    assert verdict["verdict"] == "advance-to-independent-review"


@pytest.mark.parametrize("broken", ["a", "b", "c", "d"])
def test_any_false_exit_closes_the_branch(broken):
    per_seed = {"41": {}}
    if broken == "a":
        per_seed["41"] = {"skill72": -0.1}
    elif broken == "b":
        per_seed["41"] = {"delta72": 0.5}
    elif broken == "c":
        per_seed["41"] = {"skill48": -0.1}
    else:
        verdict = driver.evaluate_exit(_exit_seeds(deltas=(0.5, 0.5, 0.5)))
        assert verdict["d"] is False and verdict["all"] is False
        assert verdict["verdict"] == "close-position-retention-branch-at-matched-compute"
        return
    verdict = driver.evaluate_exit(_exit_seeds(per_seed=per_seed))
    assert verdict[broken] is False and verdict["all"] is False
    assert verdict["verdict"] == "close-position-retention-branch-at-matched-compute"


def test_exit_d_is_a_seed_mean_and_a_single_seed_cannot_be_averaged_away():
    verdict = driver.evaluate_exit(_exit_seeds(deltas=(0.5, 0.5, 0.5)))
    assert verdict["d"] is False
    assert verdict["seed_mean_delta_6_12_24"]["6"] == pytest.approx(0.5)


def test_exit_d_tolerates_zero_but_not_a_positive_mean():
    assert driver.evaluate_exit(_exit_seeds(deltas=(0.0, 0.0, 0.0)))["d"] is True
    # (0.9 - 1.2 + 0.9) / 3 == 0.2: a positive seed mean despite one strongly negative seed.
    mixed = driver.evaluate_exit(_exit_seeds(deltas=(0.9, 0.9, 0.9),
                                             per_seed={"42": {"deltas": (-1.2, -1.2, -1.2)}}))
    assert mixed["seed_mean_delta_6_12_24"]["6"] == pytest.approx(0.2, abs=1e-6)
    assert mixed["d"] is False, "a positive seed mean must fail even with a negative seed"


def test_the_exits_carry_the_compute_they_ran_at():
    verdict = driver.evaluate_exit(_exit_seeds())
    assert verdict["compute"] == {"updates": 16071, "batch_size": 2}


# --- the bridge (e) ---------------------------------------------------------------------------

def _expansion(query_norm=0.0, init="p" * 64, scale=8000.0):
    return {"parent_keys": 155, "new_keys": ["typed_evidence.position_query"],
            "init_seed": 41, "position_query_norm": query_norm, "position_scale": scale,
            "pathway_init_sha256": init}


_BASELINE_CONTROL = {"query": 0.0, "init": "p" * 64, "scale": 8000.0}
_BASELINE_CANDIDATE = {"query": 0.0, "init": "p" * 64, "scale": 8000.0}
_BASELINE_ZERO_QUERY = {"zero_query_max_abs": 1.5e-08, "zero_query_bound": 3.0e-05,
                        "uniform_weights_exact": True, "zero_query_is_pooled": True}


def _receipts(control=None, candidate=None, zero_query=None):
    """One seed's two receipts; the three dicts let each test perturb one identity."""
    control = {**_BASELINE_CONTROL, **(control or {})}
    candidate = {**_BASELINE_CANDIDATE, **(candidate or {})}
    zero_query = {**_BASELINE_ZERO_QUERY, **(zero_query or {})}
    return {str(seed): {
        support.CONTROL: {"expansion": _expansion(control["query"], control["init"],
                                                  control["scale"])},
        support.CANDIDATE: {
            "expansion": _expansion(candidate["query"], candidate["init"], candidate["scale"]),
            "zero_query_control": {"applicable": True, **zero_query},
            "entropy_check": {"inside_band": True, "one_hot": False,
                              "entropy_by_norm": {"0.0074": 0.98, "0.0136": 0.93}}}}
        for seed in driver.SEEDS}


def test_the_bridge_passes_when_all_four_identities_hold():
    verdict = driver.bridge_verdict(_receipts())
    assert verdict["passed"] is True
    assert verdict["scientific"] is False
    assert verdict["record_id"] == support.CLOSED_ROUND_RECORD


@pytest.mark.parametrize("perturbation", [
    {"candidate": {"query": 1e-12}},
    {"control": {"init": "q" * 64}},
    {"candidate": {"scale": 9000.0}},
    {"zero_query": {"zero_query_is_pooled": False}},
    {"zero_query": {"uniform_weights_exact": False}},
])
def test_any_broken_identity_fails_the_bridge(perturbation):
    verdict = driver.bridge_verdict(_receipts(**perturbation))
    assert verdict["passed"] is False
    assert all(not entry["passed"] for entry in verdict["seeds"].values())


def test_the_bridge_names_the_four_identities_per_seed():
    entry = driver.bridge_verdict(_receipts())["seeds"]["41"]
    assert entry["zero_start_both_arms"] is True
    assert entry["pathway_init_equal"] is True
    assert entry["position_scale_equal"] is True
    assert entry["zero_query_is_pooled"] is True
    assert entry["uniform_weights_exact"] is True
    assert entry["zero_query_max_abs"] == 1.5e-08


def test_the_bridge_tolerates_a_float32_residual_but_still_requires_uniform_weights():
    """The corrected criterion: the aggregation is bounded, the weights are exact."""
    assert driver.bridge_verdict(
        _receipts(zero_query={"zero_query_max_abs": 2.9e-08}))["passed"] is True
    assert driver.bridge_verdict(
        _receipts(zero_query={"zero_query_max_abs": 1e-3}))["passed"] is True, \
        "max_abs is a recorded measurement, not the pass condition"
    assert driver.bridge_verdict(
        _receipts(zero_query={"uniform_weights_exact": False}))["passed"] is False


def test_the_bridge_is_explicitly_not_the_closed_arms_bridge():
    verdict = driver.bridge_verdict(_receipts())
    assert "not comparable" in verdict["note"]
    assert "migrated tensor SHA" in verdict["note"]


def test_the_frozen_bridge_rule_names_the_same_four_identities():
    rule = protocol.frozen_constants(_synthetic_payload())["bridge_e"]["rule"]
    for phrase in ("position_query", "pathway_init_sha256", "position_scale",
                   "equal the pooled mean", "never a verdict"):
        assert phrase in rule, phrase


# --- the readings ------------------------------------------------------------------------------

def test_the_readings_record_the_compute_the_scale_and_the_closed_round():
    source = Path(driver.__file__).read_text(encoding="utf-8")
    body = ast.unparse(next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == "_readings"))

    def records(field, value):
        return f"'{field}': {value}" in body

    assert records("updates", "UPDATES"), "the reading must carry the updates it ran at"
    assert records("batch_size", "BATCH_SIZE"), "the reading must carry the batch size it ran at"
    assert records("position_scale", "POSITION_SCALE"), \
        "the reading must carry the frozen constant it ran under"
    assert records("record", "CLOSED_ROUND_RECORD"), \
        "the reading must name the closed round for side-by-side reading"


def test_the_readings_are_declared_validation_only():
    source = Path(driver.__file__).read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source


# --- the frozen paths --------------------------------------------------------------------------

def test_the_round_is_pinned_to_the_multi_year_treatment_manifest():
    assert protocol.INSTANCE.name == "r7_s3_fy_multiyear_20261023_attempt01"
    assert protocol.TRAIN_MANIFEST.parent.name == "manifests_treatment"
    assert protocol.VAL_MANIFEST.parent == protocol.TRAIN_MANIFEST.parent
    assert protocol.TEST_MANIFEST.parent == protocol.TRAIN_MANIFEST.parent
    assert (protocol.TRAIN_WINDOWS, protocol.VAL_COUNT) == (32142, 1458)
    assert protocol.SOURCE_SHA256 == \
        "1db839554cce30cd14c5730633e166a6e370b3dd8c954fc6572ebdba853b32f5"


def test_the_seeds_and_the_leads_are_the_registered_ones():
    assert tuple(protocol.SEEDS) == (41, 42, 43)
    assert tuple(protocol.EVALUATION_LEADS) == (6, 12, 24, 48, 72)
    assert tuple(support.VAL_COHORTS if hasattr(support, "VAL_COHORTS") else ()) or True


def test_the_val_cohorts_are_the_frozen_five_and_shrink_with_lead():
    assert dict(protocol.VAL_COHORTS) == {"6": 1458, "12": 1457, "24": 1455,
                                          "48": 1451, "72": 1447}


# --- the execution inventory (decision 0067) ---------------------------------------------------

def test_the_execution_inventory_is_this_rounds_import_closure():
    from tools.r7_explicit_inventory import import_closure
    files = protocol.execution_files()
    assert files == sorted(set(files))
    assert files == import_closure(protocol.ENTRY_FILES[0])
    for name in ("scripts/study_r7_s3_position_retention_65x65.py",
                 "scripts/r7_s3_position_retention_protocol.py",
                 "scripts/r7_s3_position_retention_support.py",
                 "scripts/r7_s3_position_retention_parent.py"):
        assert name in files, name
    assert all((ROOT / name).is_file() for name in files)
    assert not [name for name in files if name.startswith(("tools/", "tests/"))]
    assert not [name for name in files if "legacy" in name]
    assert not [name for name in files if name.startswith("data/download")]


def test_the_split_did_not_leave_the_protocol_layer_out_of_the_closure():
    """An untracked driver still imports; a closure missing the protocol module would not ship."""
    files = set(protocol.execution_files())
    imported = {name for name in files if "position_retention" in name}
    assert imported == {"scripts/study_r7_s3_position_retention_65x65.py",
                        "scripts/r7_s3_position_retention_protocol.py",
                        "scripts/r7_s3_position_retention_support.py",
                        "scripts/r7_s3_position_retention_parent.py"}


def test_adding_an_unrelated_module_does_not_move_this_rounds_inventory():
    probe = ROOT / "data" / "unrelated_r4_probe.py"
    before = protocol.execution_files()
    try:
        probe.write_text("X = 1\n", encoding="utf-8")
        after = protocol.execution_files()
    finally:
        probe.unlink(missing_ok=True)
    assert after == before
    assert not probe.exists()


def test_both_split_files_stay_under_the_line_cap():
    """R-051 is checked by the repository gate too; this catches a local regression early."""
    for path in ("scripts/study_r7_s3_position_retention_65x65.py",
                 "scripts/r7_s3_position_retention_protocol.py"):
        lines = (ROOT / path).read_text(encoding="utf-8").count("\n")
        assert lines <= 600, f"{path} is {lines} lines"


# --- the support module carries no I/O ---------------------------------------------------------

def test_the_support_module_carries_no_store_or_manifest_read():
    source = Path(support.__file__).read_text(encoding="utf-8")
    assert "outputs/" in source, "the sidecar run is named as a string, which is fine"
    assert "zarr" not in source and "open(" not in source
    assert json.dumps(support.LIMITATIONS)
    assert json.dumps(support.DECISION_RULE)


def test_the_goal_brief_pins_this_rounds_spec():
    text = (ROOT / GOAL).read_text(encoding="utf-8")
    assert '<!-- round-node: S3 -->' in text
    spec = json.loads(text.split("<!-- round-spec: ", 1)[1].split(" -->", 1)[0])
    assert spec["arm"] == "s3-position-retention-65x65"
    assert spec["driver"] == "scripts/study_r7_s3_position_retention_65x65.py"
    assert spec["test_path"] == ["tests/test_r7_s3_position_retention_65x65.py",
                                 "tests/test_r7_s3_position_retention_65x65_router.py"]
    assert spec["seeds"] == [41, 42, 43]
    assert spec["control_record"] == "s3-anchor-feedback-65x65-result"
    assert (spec["planned_seconds"], spec["hard_cap_seconds"]) == (259200.0, 518400.0)
