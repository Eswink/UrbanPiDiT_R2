"""CI-safe tests for the FY annual-coverage driver (no ``outputs/`` required).

The round's data artifacts are large and live outside the repository, so these tests cover
the frozen protocol shape, the one-factor claim, the predeclared verdict logic and the
parent/model-revision guards. Nothing here reads a store or a manifest.
"""
from __future__ import annotations

import copy

import pytest

from scripts import study_r7_s3_fy_annual_coverage as driver


def _arm(name, *, spec=None, manifest=None, identity=None, windows=None):
    return driver.arm_protocol(
        spec if spec is not None else {"dim": 192, "depth": 4},
        train_manifest=manifest or f"{name}/train.jsonl",
        train_identity=identity or f"{name}-identity",
        windows=windows or {"window_sha256": f"{name}-windows"},
        canonical_digest=lambda value: f"digest-{value.get('dim')}-{value.get('depth')}")


def _seed(seed, *, skill72=0.1, delta72=-1.0, skill48=0.1, deltas=( -1.0, -1.0, -1.0)):
    return {"arms": {driver.TREATMENT: {
                "full_t2m_skill": {"6": skill48, "12": skill48, "24": skill48,
                                   "48": skill48, "72": skill72},
                "full_t2m_rmse": {"6": 2.0, "12": 2.1, "24": 2.2, "48": 2.3, "72": 2.4}}},
            "delta_treatment_minus_control": {
                "6": deltas[0], "12": deltas[1], "24": deltas[2], "72": delta72}}


def _seeds(**overrides):
    body = {str(seed): _seed(seed) for seed in driver.SEEDS}
    for seed, values in overrides.items():
        body[str(seed)] = _seed(int(seed), **values)
    return body


def _provenance():
    buckets = {f"{month:02d}-{hour:02d}": 1 for month in range(1, 13)
               for hour in (0, 6, 12, 18)}
    return {"split": "val", "n_evaluated": driver.VAL_COHORTS["6"],
            "n_available_windows": driver.VAL_COHORTS["6"], "lead_hours": [6],
            "boundary_scoring": {"regions": [
                {"region": "full", "n_grid_points": driver.FULL_GRID_POINTS},
                {"region": "interior_1", "n_grid_points": 3969}]},
            "climatology": {"training_years": driver.CLIMATOLOGY_YEARS,
                            "bucket_counts": buckets}}


def test_frozen_constants_pin_the_two_arm_factor():
    body = driver.frozen_constants()
    assert body["train_file_counts"] == {"control": 2140, "treatment": 7302}
    assert body["val_count"] == 1458 and body["test_count"] == 472
    assert body["val_cohorts"] == {"6": 1458, "12": 1457, "24": 1455,
                                   "48": 1451, "72": 1447}
    assert body["region"] == {"points": [65, 65], "boundary_margins": [1],
                              "scored_region": "full", "n_grid_points": 4225,
                              "note": "the whole native 65x65 grid; interior_32 is 1x1 here and unused"}
    assert body["recipe"]["updates"] == 2400 and body["recipe"]["steps"] == 4
    assert body["recipe"]["physical_weights"] == [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5]
    assert body["climatology"]["buckets"] == 48
    assert body["climatology"]["training_years"] == [2017, 2018, 2019, 2020, 2021]
    assert body["test_read"] is False and body["scientific_claim"] is False
    assert body["factor"]["years"] == [2017, 2018, 2019, 2020, 2021]


def test_arm_difference_is_exactly_the_training_manifest():
    control, treatment = _arm("control"), _arm("treatment")
    assert driver.difference_keys(control, treatment) == list(driver.ARM_FACTOR_KEYS)
    driver.require_single_factor(control, treatment)
    for field in ("updates", "steps", "lr", "warmup", "weight_decay", "batch_size",
                  "clip", "bf16", "physical_weights", "model"):
        assert control[field] == treatment[field]


def test_any_extra_arm_difference_is_refused():
    control, treatment = _arm("control"), _arm("treatment")
    changed = copy.deepcopy(treatment)
    changed["lr"] = control["lr"] / 2
    with pytest.raises(ValueError):
        driver.require_single_factor(control, changed)
    changed = copy.deepcopy(treatment)
    changed["model"] = {"kind": "process", "spec": {"dim": 128, "depth": 4}}
    with pytest.raises(ValueError):
        driver.require_single_factor(control, changed)
    changed = copy.deepcopy(treatment)
    changed["train_data_identity"] = control["train_data_identity"]
    with pytest.raises(ValueError):
        driver.require_single_factor(control, changed)


def test_exit_requires_all_four_conditions():
    verdict = driver.evaluate_exit(_seeds())
    assert verdict["a"] and verdict["b"] and verdict["c"] and verdict["d"]
    assert verdict["all"] is True
    assert verdict["verdict"] == "advance-to-independent-review"


def test_exit_a_is_per_seed_72h_skill():
    verdict = driver.evaluate_exit(_seeds(**{"43": {"skill72": -0.01}}))
    assert verdict["a"] is False and verdict["all"] is False
    assert verdict["verdict"] == "close-fy-annual-coverage-branch"


def test_exit_b_is_a_per_seed_sign_not_an_average():
    # Two large negative seeds and one positive seed: the mean is negative, but (b) must fail.
    verdict = driver.evaluate_exit(_seeds(**{"43": {"delta72": 0.1}}))
    assert verdict["b"] is False
    assert verdict["seed_mean_delta_6_12_24"]["6"] < 0  # (d)'s mean is still negative


def test_exit_c_is_each_seed_48h_skill():
    verdict = driver.evaluate_exit(_seeds(**{"42": {"skill48": 0.0}}))
    assert verdict["c"] is False


def test_exit_d_uses_the_seed_mean_over_6_12_24():
    verdict = driver.evaluate_exit(_seeds(**{"43": {"deltas": (-1.0, -1.0, 4.0)}}))
    assert verdict["d"] is False
    assert verdict["seed_mean_delta_6_12_24"]["24"] > 0


def test_exit_refuses_missing_seed_data():
    with pytest.raises(KeyError):
        driver.evaluate_exit({"41": _seed(41)})


def test_bridge_band_is_the_larger_of_seed_spread_and_five_percent():
    reference = driver.bridge_reference()
    for lead, values in driver.BRIDGE_VALUES.items():
        mean = sum(values) / len(values)
        expected = max(max(values) - min(values), 0.05 * abs(mean))
        assert reference[lead]["band"] == pytest.approx(expected)


def test_bridge_is_reported_but_never_scientific():
    reference = driver.bridge_reference()
    control = {str(seed): {lead: reference[lead]["mean"] + 0.5 * reference[lead]["band"]
                           for lead in map(str, driver.EVALUATION_LEADS)}
               for seed in driver.SEEDS}
    passed = driver.bridge_verdict(control, reference)
    assert passed["scientific"] is False and passed["passed"] is True
    # The bridge statistic is the seed mean, so a single-seed perturbation of 2 bands
    # still moves the mean by only 2/3 band. Perturb every seed to move the mean past
    # the band, which is the counterexample the rule actually claims.
    for seed in driver.SEEDS:
        control[str(seed)]["72"] = reference["72"]["mean"] + 2.0 * reference["72"]["band"]
    failed = driver.bridge_verdict(control, reference)
    assert failed["scientific"] is False and failed["passed"] is False


def test_parent_pins_are_three_and_bound_to_the_current_revision():
    pins = driver.fy_parent.pins()
    assert sorted(pins) == ["41", "42", "43"]
    assert driver.fy_parent.MODEL_CODE_SHA256 == \
        "3fe747b28e7d0ef53febc07739f210f103514a73fd19bfba6643670db65ffeb1"
    assert all(item["sha256"] for item in pins.values())
    assert all(f"seed{seed}" in item["path"] for seed, item in pins.items())


def test_clean_model_revision_is_a_precondition():
    driver.check_model_revision(driver.fy_parent.MODEL_CODE_SHA256)
    with pytest.raises(RuntimeError):
        driver.check_model_revision("0" * 64)


def test_manifests_are_not_interchangeable_but_val_is_shared():
    body = driver.frozen_constants()
    assert body["train_manifests"]["control"] != body["train_manifests"]["treatment"]
    assert body["train_file_counts"]["control"] != body["train_file_counts"]["treatment"]
    # Different directories, same dataset identity is checked at freeze time.
    assert body["val_manifests"]["control"] != body["val_manifests"]["treatment"]


def test_provenance_must_be_val_full_grid_and_48_buckets():
    driver._check_provenance(_provenance(), 41, "control", 6)
    wrong_region = copy.deepcopy(_provenance())
    wrong_region["boundary_scoring"]["regions"][0]["n_grid_points"] = 3969
    with pytest.raises(RuntimeError):
        driver._check_provenance(wrong_region, 41, "control", 6)
    missing = copy.deepcopy(_provenance())
    missing["boundary_scoring"] = None
    with pytest.raises(RuntimeError):
        driver._check_provenance(missing, 41, "control", 6)
    wrong_years = copy.deepcopy(_provenance())
    wrong_years["climatology"]["training_years"] = [2017]
    with pytest.raises(RuntimeError):
        driver._check_provenance(wrong_years, 41, "control", 6)
    wrong_cohort = copy.deepcopy(_provenance())
    wrong_cohort["n_evaluated"] = 472
    with pytest.raises(RuntimeError):
        driver._check_provenance(wrong_cohort, 41, "control", 6)


def test_decision_rule_is_frozen_into_the_constants():
    rule = driver.frozen_constants()["decision"]
    assert set(rule["exits"]) == {"a", "b", "c", "d"}
    assert rule["text"] == driver.DECISION_RULE
    bridge = driver.frozen_constants()["bridge_e"]
    assert bridge["scientific"] is False
    assert bridge["record_id"] == "s3-v3-budget-dose" and bridge["updates"] == 1600
