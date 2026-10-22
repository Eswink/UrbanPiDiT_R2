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
    """Decision 0052 rebinds the round to ``60de3b2d...``; the state pins do not move."""
    pins = driver.fy_parent.pins()
    assert sorted(pins) == ["41", "42", "43"]
    assert driver.fy_parent.MODEL_CODE_SHA256 == \
        "60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e"
    assert driver.fy_parent.MODEL_CODE_SHA256 != \
        "3fe747b28e7d0ef53febc07739f210f103514a73fd19bfba6643670db65ffeb1"
    assert all(item["sha256"] for item in pins.values())
    assert all(f"seed{seed}" in item["path"] for seed, item in pins.items())
    assert "r7_s3_tendency_input_parent_migrated_20261021_attempt02" in \
        driver.fy_parent.MIGRATED_PARENT_RUN.name


def test_the_rebinding_keeps_decision_0050s_parent_state():
    """The static half of the equality claim; ``state_equality_evidence`` is its runtime form.

    The two migrations read the same v3-BD 1600 source checkpoints and carry the same tensor
    ``state_digest`` per seed, so the round's factor stays the training-manifest month
    coverage. Only the embedded code identity, and therefore the files' own SHA256 values,
    differ.
    """
    parent = driver.fy_parent
    assert set(parent.MIGRATED_PARENT_STATE_DIGEST) == {41, 42, 43}
    assert set(parent.ATTEMPT04_PARENT_SHA256) == {41, 42, 43}
    assert all(parent.MIGRATED_PARENT_SHA256[seed] != parent.ATTEMPT04_PARENT_SHA256[seed]
               for seed in (41, 42, 43))
    # The state digests are the same values the attempt04 receipt records; the parent module
    # re-checks that at freeze time against both receipts.
    assert parent.MIGRATED_PARENT_STATE_DIGEST[41] == \
        "7c416ccb4054cf7e077788a43b02917e6a2dcf2503058152a7f0ab4b0ab70707"


def test_the_two_post_hd_switches_are_inert_at_their_registered_defaults():
    """The rebinding rests on this, so it is asserted rather than assumed.

    ``tendency_input`` and ``decoder_base_mode`` were added to ``model/`` after the revision
    decision 0050 froze. If either changed the registered forward, running the FY round at
    HEAD would change the model as well as the data, and the round would no longer be a
    single-factor data test. A model built from a spec that names both at their registered
    defaults must therefore reproduce the registered training forward bit for bit.
    """
    import torch
    from test_r7_m4_autoregressive_rollout import tiny_spec
    from training.r7_experiment import make_model
    from training.r7_long_rollout import training_long_rollout
    spec = tiny_spec("process")
    assert "tendency_input" not in spec and "decoder_base_mode" not in spec
    torch.manual_seed(31)
    registered = make_model("process", spec)
    torch.manual_seed(31)
    explicit = make_model("process", {**spec, "tendency_input": False,
                                      "decoder_base_mode": "persistence"})
    shared = registered.state_dict()
    other = explicit.state_dict()
    assert set(shared) == set(other)
    assert all(torch.equal(value, other[name]) for name, value in shared.items())
    batch = {"coarse_history": torch.randn(1, 2, 17, 4, 5,
                                          generator=torch.Generator().manual_seed(7)),
             "latitude": torch.linspace(50., 44., 4),
             "longitude": torch.linspace(90., 98., 5),
             "lead_time_hours": torch.tensor([6.]),
             "init_utc_hour": torch.tensor([18.]),
             "init_day_of_year": torch.tensor([366.]),
             "init_calendar_year": torch.tensor([2016.]),
             "history_offsets_hours": torch.tensor([[-6., 0.]]),
             "physical_targets": torch.randn(1, 4, 17, 4, 5,
                                             generator=torch.Generator().manual_seed(8))}
    weights = (1., .5, 0., .5)
    first = training_long_rollout(registered, batch, 2, physical_weights=weights)
    second = training_long_rollout(explicit, batch, 2, physical_weights=weights)
    assert torch.equal(first.loss, second.loss)
    assert torch.equal(first.per_step_losses, second.per_step_losses)


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


def test_the_control_arm_reads_the_complete_horizon_subset():
    """The control arm must not read the unfiltered 6 h manifest.

    ``manifests_control`` still admits windows whose ``t+6..t+72`` targets roll past day 30
    into another month; the long-rollout dataset would read those frames from the full-year
    store and the control arm would no longer be a four-month arm. The filtered subset is
    ``manifests_control_h72``, and the driver must name it.
    """
    control = driver.manifest_path(driver.CONTROL, "train")
    assert control.parent.name == "manifests_control_h72"
    assert control.parent.name != "manifests_control"
    assert driver.manifest_path(driver.CONTROL, "val").parent == control.parent
    assert driver.manifest_path(driver.TREATMENT, "train").parent.name == "manifests_treatment"
    assert {name for name, _ in driver.ARMS} == {driver.CONTROL, driver.TREATMENT}
    assert all(driver.manifest_path(name, "train").parent == folder
               for name, folder in driver.ARMS)
    with pytest.raises(ValueError):
        driver.manifest_path("candidate", "train")


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


def test_the_budget_matches_the_measured_two_arm_cost():
    """Decision 0053: the first protocol's per-seed budget was 1.6x too small.

    One arm's 2,400 updates plus its five-lead evaluation measured 7,169.6-7,201.4 s, so a
    seed's two arms need about 14,400 s. The frozen 9,000 s killed the first attempt's
    workers at their limit, which is a failed attempt, not something to retry smaller.
    """
    assert (driver.PER_SEED_SECONDS, driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) == \
        (21600.0, 43200.0, 86400.0)
    assert driver.PER_SEED_SECONDS > 2 * 7201.4
    assert driver.PLANNED_SECONDS == 2 * driver.PER_SEED_SECONDS
    assert driver.HARD_CAP_SECONDS == 4 * driver.PER_SEED_SECONDS


def test_decision_rule_is_frozen_into_the_constants():
    rule = driver.frozen_constants()["decision"]
    assert set(rule["exits"]) == {"a", "b", "c", "d"}
    assert rule["text"] == driver.DECISION_RULE
    bridge = driver.frozen_constants()["bridge_e"]
    assert bridge["scientific"] is False
    assert bridge["record_id"] == "s3-v3-budget-dose" and bridge["updates"] == 1600
