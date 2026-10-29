"""CI-safe tests for the matched-compute year-coverage driver (no ``outputs/`` required).

The round's data artifacts are large and live outside the repository, so these tests cover
the frozen protocol shape, the one-factor claim, the matched-compute constants, the
predeclared verdict logic, the bridge table and the parent/model-revision guards. Nothing
here reads a store or a manifest, and the one training-path test runs on a synthetic
in-tree fixture.

The counterexamples matter as much as the positive cases: a driver that silently drifted
back to the closed round's 2,400 updates, or that trained the two arms at different batch
sizes, would still satisfy every "the constant equals itself" assertion.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch

from scripts import r7_s3_fy_year_coverage_matched_support as support
from scripts import study_r7_s3_fy_year_coverage_matched as driver

ROOT = Path(driver.__file__).resolve().parents[1]
GOAL = "docs/goals/s3-matched-exposure-reeval.md"


def _arm(name, *, spec=None, manifest=None, identity=None, windows=None, batch_size=None):
    return driver.arm_protocol(
        spec if spec is not None else {"dim": 192, "depth": 4},
        train_manifest=manifest or f"{name}/train.jsonl",
        train_identity=identity or f"{name}-identity",
        windows=windows or {"window_sha256": f"{name}-windows"},
        canonical_digest=lambda value: f"digest-{value.get('dim')}-{value.get('depth')}")


def _seed(*, skill72=0.1, delta72=-1.0, skill48=0.1, deltas=(-1.0, -1.0, -1.0)):
    return {"arms": {driver.TREATMENT: {
                "full_t2m_skill": {"6": skill48, "12": skill48, "24": skill48,
                                   "48": skill48, "72": skill72},
                "full_t2m_rmse": {"6": 2.0, "12": 2.1, "24": 2.2, "48": 2.3, "72": 2.4}}},
            "delta_treatment_minus_control": {
                "6": deltas[0], "12": deltas[1], "24": deltas[2], "72": delta72}}


def _seeds(**overrides):
    body = {str(seed): _seed() for seed in driver.SEEDS}
    for seed, values in overrides.items():
        body[str(seed)] = _seed(**values)
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


# --- the frozen protocol ------------------------------------------------------------------

def test_frozen_constants_pin_the_matched_compute_factor():
    body = driver.frozen_constants()
    assert body["train_file_counts"] == {"control": 7291, "treatment": 32142}
    assert body["val_count"] == 1458 and body["test_count"] == 472
    assert body["val_cohorts"] == {"6": 1458, "12": 1457, "24": 1455,
                                   "48": 1451, "72": 1447}
    assert body["region"]["scored_region"] == "full"
    assert body["region"]["n_grid_points"] == 4225
    assert body["recipe"]["updates"] == 16071 and body["recipe"]["batch_size"] == 2
    assert body["recipe"]["steps"] == 4
    assert body["recipe"]["physical_weights"] == [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5]
    assert body["climatology"]["buckets"] == 48
    assert body["climatology"]["training_years"] == list(range(2000, 2022))
    assert body["test_read"] is False and body["scientific_claim"] is False


def test_every_arm_sees_at_least_one_epoch():
    """Decision 0060/0061: updates * batch >= the treatment manifest, or the round is not matched."""
    assert driver.UPDATES * driver.BATCH_SIZE >= driver.TRAIN_COUNTS[driver.TREATMENT]
    assert driver.UPDATES == 16071 and driver.BATCH_SIZE == 2
    body = driver.frozen_constants()
    assert body["recipe"]["epochs_floor"].startswith("updates * batch_size = 32142")
    # The closed round's exposure is the thing this round exists to fix.
    assert driver.UPDATES > 6 * support.CLOSED_ROUND_UPDATES


def test_the_only_change_from_the_closed_round_is_compute():
    """Decision 0061: same store, grid, supervision region and data span; only compute moves."""
    factor = driver.frozen_constants()["factor"]
    assert factor["changed_from_closed_round"] == ["updates", "batch_size"]
    assert factor["closed_round_updates"] == 2400 == support.CLOSED_ROUND_UPDATES
    for held in ("store", "normalization", "model spec", "migrated parent", "K", "depth",
                 "dim", "physical weights", "seeds", "val split", "climatology denominator",
                 "months", "lr", "warmup", "clip", "supervision"):
        assert held in factor["controlled"]
    assert factor["not_controlled"] == []


def test_the_factor_is_year_coverage_with_the_months_held_fixed():
    factor = driver.frozen_constants()["factor"]
    assert factor["months_held_fixed"] == list(range(1, 13))
    assert factor["control"] == "2017-2021 (5 years, 12 months)"
    assert factor["treatment"] == "2000-2021 (22 years, 12 months)"
    assert driver.CLIMATOLOGY_YEARS == list(range(2000, 2022))
    assert driver.TRAIN_COUNTS[driver.TREATMENT] > 4 * driver.TRAIN_COUNTS[driver.CONTROL]


def test_arm_difference_is_exactly_the_training_manifest():
    control, treatment = _arm("control"), _arm("treatment")
    assert driver.difference_keys(control, treatment) == list(driver.ARM_FACTOR_KEYS)
    driver.require_single_factor(control, treatment)
    for field in ("updates", "steps", "lr", "warmup", "weight_decay", "batch_size",
                  "clip", "bf16", "physical_weights", "model"):
        assert control[field] == treatment[field]
    assert control["batch_size"] == driver.BATCH_SIZE == 2


def test_any_extra_arm_difference_is_refused():
    control, treatment = _arm("control"), _arm("treatment")
    for field, value in (("lr", driver.LR / 2),
                         ("batch_size", 1),
                         ("model", {"kind": "process", "spec": {"dim": 128, "depth": 4}}),
                         ("train_data_identity", "control-identity")):
        changed = copy.deepcopy(treatment)
        changed[field] = value
        with pytest.raises(ValueError):
            driver.require_single_factor(control, changed)


def test_an_arm_at_the_wrong_batch_size_is_refused_even_if_the_arms_agree():
    """Both arms at batch 1 would agree with each other and still be the wrong round."""
    control, treatment = _arm("control"), _arm("treatment")
    for arm in (control, treatment):
        arm["batch_size"] = 1
    with pytest.raises(ValueError, match="frozen batch size"):
        driver.require_single_factor(control, treatment)


def test_manifests_are_not_interchangeable_but_val_is_shared():
    body = driver.frozen_constants()
    assert body["train_manifests"]["control"] != body["train_manifests"]["treatment"]
    assert body["train_file_counts"]["control"] != body["train_file_counts"]["treatment"]
    assert body["val_manifests"]["control"] != body["val_manifests"]["treatment"]


def test_the_control_arm_reads_the_year_filtered_subset():
    """A control arm naming the store's own manifest would train on all twenty-two years."""
    control = driver.manifest_path(driver.CONTROL, "train")
    assert control.parent.name == "manifests_control"
    assert control.parent.name != "manifests"
    assert driver.manifest_path(driver.CONTROL, "val").parent == control.parent
    assert driver.manifest_path(driver.TREATMENT, "train").parent.name == "manifests_treatment"
    assert {name for name, _ in driver.ARMS} == {driver.CONTROL, driver.TREATMENT}
    assert all(driver.manifest_path(name, "train").parent == folder
               for name, folder in driver.ARMS)
    with pytest.raises(ValueError):
        driver.manifest_path("candidate", "train")


def test_the_control_filter_refuses_every_out_of_range_read_member():
    """The control's whole read set must lie in 2017-2021, not just its init time."""
    from scripts import build_r7_s3_fy_year_manifests as builder
    base = {"sample_id": "s", "init_time": "2018-06-01T00:00:00",
            "history_times": ["2018-05-31T12:00:00", "2018-05-31T18:00:00"]}
    assert builder.why_dropped(base) is None
    history = {**base, "history_times": ["2016-12-31T12:00:00", "2016-12-31T18:00:00"]}
    assert builder.why_dropped(history)["reason"] == "history_outside_declared_years"
    assert builder.why_dropped({**base, "init_time": "2022-01-01T00:00:00"})["reason"] == \
        "init_outside_declared_years"
    late = {**base, "init_time": "2021-12-30T18:00:00",
            "history_times": ["2021-12-30T06:00:00", "2021-12-30T12:00:00"]}
    reason = builder.why_dropped(late)
    assert reason["reason"] == "target_outside_declared_years" and reason["member"] == "t+30h"


# --- the predeclared exits ----------------------------------------------------------------

def test_exit_requires_all_four_conditions():
    verdict = driver.evaluate_exit(_seeds())
    assert verdict["a"] and verdict["b"] and verdict["c"] and verdict["d"]
    assert verdict["all"] is True
    assert verdict["verdict"] == "advance-to-independent-review"
    assert verdict["compute"] == {"updates": 16071, "batch_size": 2}


def test_exit_a_is_per_seed_72h_skill():
    verdict = driver.evaluate_exit(_seeds(**{"43": {"skill72": -0.01}}))
    assert verdict["a"] is False and verdict["all"] is False
    assert verdict["verdict"] == "close-training-year-coverage-branch-at-matched-compute"


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


def test_exit_d_flips_on_the_same_shape_the_closed_round_measured():
    """The closed round's (d) failure shape: 6 h and 12 h slightly positive, 24 h negative.

    Decision 0061's falsifiable prediction is that matching compute flips exactly this
    shape. The exit must still call that shape false, or the re-run could not test it.
    """
    closed_shape = {"41": {"deltas": (0.009267, 0.003540, -0.018799)},
                    "42": {"deltas": (0.003540, 0.005199, -0.018799)},
                    "43": {"deltas": (0.005199, 0.004365, -0.018799)}}
    verdict = driver.evaluate_exit(_seeds(**closed_shape))
    assert verdict["d"] is False
    assert verdict["seed_mean_delta_6_12_24"]["6"] > 0
    assert verdict["seed_mean_delta_6_12_24"]["12"] > 0
    assert verdict["seed_mean_delta_6_12_24"]["24"] < 0


def test_exit_refuses_missing_seed_data():
    with pytest.raises(KeyError):
        driver.evaluate_exit({"41": _seed()})


def test_decision_rule_is_frozen_into_the_constants():
    rule = driver.frozen_constants()["decision"]
    assert set(rule["exits"]) == {"a", "b", "c", "d"}
    assert rule["text"] == support.DECISION_RULE == driver.DECISION_RULE
    assert "close the training-year coverage branch" in rule["close"]


# --- the non-scientific bridge ------------------------------------------------------------

def test_bridge_band_is_the_larger_of_seed_spread_and_five_percent():
    reference = driver.bridge_reference()
    for lead, values in support.BRIDGE_VALUES.items():
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
    # The bridge statistic is the seed mean, so one perturbed seed moves it by only 2/3 band.
    for seed in driver.SEEDS:
        control[str(seed)]["72"] = reference["72"]["mean"] + 2.0 * reference["72"]["band"]
    failed = driver.bridge_verdict(control, reference)
    assert failed["scientific"] is False and failed["passed"] is False


def test_the_bridge_reference_is_the_registered_fy_annual_coverage_candidate_arm():
    """The bridge is an identity check, not a paired control.

    That arm trained on 2017-2021 with its own normalization and its own climatology
    denominator, so it can only catch a store or normalization mistake before the seeds
    are read. The paired control for this round is re-run inside the round.
    """
    assert driver.BRIDGE_RECORD == "s3-fy-annual-coverage-result"
    assert driver.BRIDGE_UPDATES == 2400
    assert driver.frozen_constants()["bridge_e"]["scientific"] is False
    reference = driver.bridge_reference()
    assert sorted(reference) == ["12", "24", "48", "6", "72"]
    assert reference["72"]["mean"] == pytest.approx(3.825635930809)
    assert reference["6"]["mean"] == pytest.approx(2.2863432848298924)


def test_the_bridge_reference_matches_the_registered_evidence_page():
    """The three per-lead values must be the registered arm's own readings, not invented."""
    page = (ROOT / "docs/R7_S3_FY_ANNUAL_COVERAGE_RESULT.md").read_text(encoding="utf-8")
    for lead, values in support.BRIDGE_VALUES.items():
        for value in values:
            assert f"{value:.6f}" in page, f"bridge {lead} h value {value:.6f} is not in the page"


# --- identity and provenance guards -------------------------------------------------------

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
    on_test = copy.deepcopy(_provenance())
    on_test["split"] = "test"
    with pytest.raises(RuntimeError):
        driver._check_provenance(on_test, 41, "control", 6)


def test_parent_pins_are_three_and_bound_to_the_current_revision():
    """The round does not change ``model/``, so it reuses the closed round's parent pins."""
    pins = driver.parent_pins.pins()
    assert sorted(pins) == ["41", "42", "43"]
    assert driver.parent_pins.MODEL_CODE_SHA256 == \
        "60de3b2dc5de2871f45c7a563153ef8e1cfa688be38367c89d0c718dfc06df3e"
    assert all(item["sha256"] for item in pins.values())
    assert all(f"seed{seed}" in item["path"] for seed, item in pins.items())
    assert "r7_s3_tendency_input_parent_migrated_20261021_attempt02" in \
        driver.parent_pins.MIGRATED_PARENT_RUN.name
    assert driver.parent_pins.ENDPOINT == 1600


def test_clean_model_revision_is_a_precondition():
    driver.check_model_revision(driver.parent_pins.MODEL_CODE_SHA256)
    with pytest.raises(RuntimeError):
        driver.check_model_revision("0" * 64)


def test_the_source_is_hashed_in_prepare_and_only_compared_afterwards():
    """W1c: the 5.49 GB source must not be re-read by each of the five later stages."""
    import ast
    source = Path(driver.__file__).read_text(encoding="utf-8")
    functions = {node.name: node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)}

    def hashes_the_source(node):
        return any(isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                   and inner.func.id == "sha256_file"
                   and any(isinstance(arg, ast.Name) and arg.id == "SOURCE" for arg in inner.args)
                   for inner in ast.walk(node))

    assert hashes_the_source(functions["_identity"]), "prepare must hash the source"
    assert not hashes_the_source(functions["validate_protocol"]), \
        "later stages must not re-read the source"
    assert "source_file" in ast.unparse(functions["validate_protocol"])
    assert driver.SOURCE_BYTES == 5492365397


# --- budget, round-spec and the training path ---------------------------------------------

def test_the_budget_is_decision_0060s_matched_compute_scale():
    """The closed round's 43,200 s cannot pay for one epoch of the 32,142-window arm."""
    assert (driver.PER_SEED_SECONDS, driver.PLANNED_SECONDS, driver.HARD_CAP_SECONDS) == \
        (129600.0, 259200.0, 518400.0)
    assert driver.PLANNED_SECONDS == 2 * driver.PER_SEED_SECONDS
    assert driver.HARD_CAP_SECONDS == 4 * driver.PER_SEED_SECONDS
    # 16,071 updates at the measured 2.351 s/update is about 10.5 h per arm; two arms plus
    # scoring is about 22 h, already five times the closed round's 43,200 s ceiling.
    assert 2 * driver.UPDATES * 2.351 > 43200.0
    assert driver.PER_SEED_SECONDS > 2 * driver.UPDATES * 2.351


def _round_spec():
    """R1's spec, read from its own brief.

    It used to go through ``campaign_state()["current_round_goal"]`` and assert that pointer
    equals ``GOAL``. That is a snapshot, not an invariant: the campaign pointer has to move
    on when R1 is registered, so the assertion goes red on the very commit that closes the
    round. The durable claim is that R1's brief still carries exactly one spec naming this
    driver - true during and after the round, and what a reader replaying R1 needs.
    """
    from tools.r7_round_support import round_spec
    return round_spec(ROOT / GOAL)


def test_the_round_spec_names_this_driver_test_and_the_matched_compute():
    spec = _round_spec()
    assert spec["arm"] == "s3-fy-year-coverage-matched"
    assert spec["driver"] == "scripts/study_r7_s3_fy_year_coverage_matched.py"
    assert spec["test_path"] == ["tests/test_r7_s3_fy_year_coverage_matched.py"]
    assert spec["control_record"] == "s3-fy-annual-coverage-result"
    assert spec["seeds"] == list(driver.SEEDS) == [41, 42, 43]
    assert spec["device_mode"] == driver.DEVICE_MODE == "dual"
    assert spec["planned_seconds"] == driver.PLANNED_SECONDS
    assert spec["hard_cap_seconds"] == driver.HARD_CAP_SECONDS
    assert spec["code_archive"] == "outputs/r7_s3_fy_year_coverage_matched_code.zip"


def test_the_round_spec_splits_the_two_commits():
    """The two commits own disjoint path sets: the implementation commit cannot stage the page.

    This used to assert the evidence page did not exist in the working tree. That is a
    snapshot, not an invariant, and it inverts at exactly the wrong moment: the registration
    commit creates the page, so the round's own suite would go red on the one commit whose
    job is to register it. The durable claim is set disjointness -- the page lives in
    ``register_paths`` alone -- which holds both before and after registration. The
    historical half is checked too, when the implementation commit is reachable: that
    commit must not have contained the page.
    """
    spec = _round_spec()
    commit, register = set(spec["commit_paths"]), set(spec["register_paths"])
    assert commit and register
    # The living campaign and goal docs legitimately appear in both commits, so the
    # invariant is not full disjointness: it is that each registration-only artifact
    # -- the page, the index, the brief -- belongs to the register commit alone.
    register_only = ("docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md",
                     "docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md")
    for path in register_only:
        assert path in register, f"{path} must be registered"
        assert path not in commit, f"{path} must not be staged by the implementation commit"
    assert "docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md" in register
    for path in (spec["driver"], spec["test_path"][0], GOAL,
                 "docs/goals/s3-year-coverage-matched-reeval.md"):
        assert path in commit, f"{path} must be part of the implementation commit"
    assert (ROOT / "docs/goals/s3-year-coverage-matched-reeval.md").is_file()
    for path in ("docs/R7_EVIDENCE_INDEX.jsonl", "docs/R7_CANDIDATE_BRIEF.md",
                 "docs/goals/main-model-climatology-campaign.md"):
        assert path in register
    assert spec["commit_message"] != spec["register_message"]

    found = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--format=%H", "--fixed-strings",
         "--grep", spec["commit_message"]],
        capture_output=True, text=True, check=False,
    )
    if found.returncode == 0 and found.stdout.strip():
        sha = found.stdout.split()[0]
        listed = subprocess.run(
            ["git", "-C", str(ROOT), "show", "--pretty=format:", "--name-only", sha],
            capture_output=True, text=True, check=False,
        )
        if listed.returncode == 0:
            assert "docs/R7_S3_FY_YEAR_COVERAGE_MATCHED_RESULT.md" not in listed.stdout.split()


def test_the_round_spec_paths_all_exist_and_the_driver_matches_its_module():
    spec = _round_spec()
    for path in spec["commit_paths"]:
        assert (ROOT / path).exists(), f"commit path {path} does not exist"
    assert (ROOT / spec["driver"]).resolve() == Path(driver.__file__).resolve()


def test_the_training_path_accepts_batch_two_on_a_cpu_tiny_model(tmp_path):
    """CPU smoke: the real rollout/contract path accepts the round's batch size.

    The only changed factor in this round is compute, and ``batch_size`` is half of it, so
    this is the one configuration the round must not discover is unsupported at the GPU
    stage. ``fine_tune_long_rollout`` is the real runner, the fixture is synthetic and
    in-tree, and the assertion is that each update really consumed two samples.
    """
    import pandas as pd
    from test_r7_m4_autoregressive_rollout import fixture_manifest, tiny_spec
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    from training.r7_experiment import canonical_digest, dataset_identity, make_model
    from training.r7_long_rollout_runner import fine_tune_long_rollout

    torch.set_num_threads(1)
    paths, _ = fixture_manifest(tmp_path / "fixture",
                                times=pd.date_range("2016-01-01", periods=24, freq="6h"))
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=12)
    assert summary["usable_windows"] >= driver.BATCH_SIZE
    spec = tiny_spec("process")
    torch.manual_seed(23)
    model = make_model("process", spec)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    contract = {"kind": "process", "model": spec,
                "data_identity": dataset_identity(paths["train"])[0],
                "source_sha256": canonical_digest("synthetic-source"),
                "protocol_sha256": canonical_digest("synthetic-protocol"),
                "initialization": {"format": "explicit-seeded-fixture", "seed": 23},
                "autoregression": {"physical_steps": 12,
                                   "physical_weights": list(driver.PHYSICAL_WEIGHTS),
                                   "window_sha256": summary["window_sha256"],
                                   "excluded_sample_ids": summary["excluded_sample_ids"]}}
    _, report = fine_tune_long_rollout(
        paths["train"], tmp_path / "run", model=model, contract=contract, parent_weights=weights,
        physical_weights=driver.PHYSICAL_WEIGHTS, steps=driver.STEPS, updates=2, seed=7, warmup=1,
        checkpoint_every=1, batch_size=driver.BATCH_SIZE)
    assert report["contract"]["batch_size"] == 2
    assert report["contract"]["steps"] == 4
    assert report["contract"]["autoregression"]["physical_weights"] == list(driver.PHYSICAL_WEIGHTS)
    assert [row["samples"] for row in report["losses"]] == [2, 2]
    assert all(len(row["per_step_losses"]) == len(driver.PHYSICAL_WEIGHTS)
               for row in report["losses"])
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in report["losses"])


def test_the_training_path_still_refuses_a_foreign_physical_weight_vector(tmp_path):
    """Counterexample for the smoke above: the run is not accepting anything it is handed."""
    import pandas as pd
    from test_r7_m4_autoregressive_rollout import fixture_manifest, tiny_spec
    from data.r7_long_rollout_dataset import preflight_long_rollout_windows
    from training.r7_experiment import canonical_digest, dataset_identity, make_model
    from training.r7_long_rollout_runner import fine_tune_long_rollout

    torch.set_num_threads(1)
    paths, _ = fixture_manifest(tmp_path / "fixture",
                                times=pd.date_range("2016-01-01", periods=24, freq="6h"))
    summary = preflight_long_rollout_windows(paths["train"], physical_steps=12)
    spec = tiny_spec("process")
    torch.manual_seed(23)
    model = make_model("process", spec)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    contract = {"kind": "process", "model": spec,
                "data_identity": dataset_identity(paths["train"])[0],
                "source_sha256": canonical_digest("synthetic-source"),
                "protocol_sha256": canonical_digest("synthetic-protocol"),
                "initialization": {"format": "explicit-seeded-fixture", "seed": 23},
                "autoregression": {"physical_steps": 12,
                                   "physical_weights": [1.] * 12,
                                   "window_sha256": summary["window_sha256"],
                                   "excluded_sample_ids": summary["excluded_sample_ids"]}}
    with pytest.raises(ValueError):
        fine_tune_long_rollout(
            paths["train"], tmp_path / "refused", model=model, contract=contract,
            parent_weights=weights, physical_weights=driver.PHYSICAL_WEIGHTS, steps=driver.STEPS,
            updates=1, seed=7, warmup=1, checkpoint_every=1, batch_size=driver.BATCH_SIZE)


def test_the_readings_record_the_matched_compute_it_ran_at():
    """A reading that does not carry its own compute would be unreadable next to the closed round."""
    import ast
    import re
    source = Path(driver.__file__).read_text(encoding="utf-8")
    body = ast.unparse(next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == "_readings"))

    def records(field, value):
        return re.search(rf"['\"]{field}['\"]\s*:\s*{value}\b", body) is not None

    assert records("updates", "UPDATES"), "the reading must carry the updates it ran at"
    assert records("batch_size", "BATCH_SIZE"), "the reading must carry the batch size it ran at"
    assert records("closed_round_updates", "CLOSED_ROUND_UPDATES"), \
        "the reading must name the closed round's exposure for side-by-side reading"


def test_the_support_module_carries_no_store_or_manifest_read():
    """The round's frozen prose must stay pure data, or the driver's split bought nothing."""
    source = Path(support.__file__).read_text(encoding="utf-8")
    assert "outputs/" not in source
    assert "import json" not in source
    assert json.dumps(support.BRIDGE_VALUES)  # plain JSON-serialisable data
