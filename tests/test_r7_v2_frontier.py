"""CPU-only synthetic engineering fixtures; not weather or latency evidence."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import math

import pytest

from training.r7_v2_frontier import (
    ADAPTIVE_GATE_SCHEMA, ISOLATED_SCOPE, evaluate_adaptive_gate,
)
from training.r7_v2_tables import digest, pooled_statistics

CHANNELS = ["t2m"] + [f"variable_{index}" for index in range(1, 17)]
UNITS = ["K"] + ["m/s"] * 8 + ["Pa"] * 8
GROUPS = (("old_ours", 4), ("process", 1), ("process", 2), ("process", 4))


def _cohort(lead, count):
    start = datetime(2016, 1, 1)
    return [[(start + timedelta(hours=6 * index)).isoformat(),
             [(start + timedelta(hours=6 * index + lead)).isoformat()]] for index in range(count)]


def _fixture():
    protocol = {
        "stage": "C", "scientific_claim": False, "test_read": False,
        "reporting": {"adaptive_gate": deepcopy(ADAPTIVE_GATE_SCHEMA),
                      "primary": {"variable": "t2m", "unit": "K", "lead_hours": [6, 12],
                                  "degradation_tolerance": 0., "rule": "existing same-seed strict sign"},
                      "tolerances": {"K": 0.}},
        "data": {"channels": CHANNELS[:], "units": UNITS[:],
                 "evaluation_cases": {str(lead): {"cases": _cohort(lead, count), "n_available": count}
                                      for lead, count in ((6, 22), (12, 21))}},
    }
    records, rows, latency = [], [], []
    for seed in (41, 42, 43):
        for lead in (6, 12):
            cases = protocol["data"]["evaluation_cases"][str(lead)]["cases"]
            for arm, k in GROUPS:
                pins = {"protocol_sha256": digest("fixture-protocol"), "model_code_sha256": digest("fixture-code"),
                        "data_identity": digest("fixture-data"), "provenance_sha256": digest([seed, lead, arm, k]),
                        "worker_receipt_sha256": digest(["fixture-receipt", seed, lead, arm, k])}
                group = []
                for index, (init, valid) in enumerate(cases):
                    mse = 16. if arm == "old_ours" else {1: (1., 9.), 2: (4., 4.), 4: (4., 1.)}[k][index % 2]
                    stats = {"mse": mse, "climatology_mse": 4., "acc_dot": 4.,
                             "acc_forecast_energy": mse + 4., "acc_target_energy": 4.}
                    calculated = pooled_statistics([stats])
                    calculated.pop("bad_case_counts")
                    row = {"seed": seed, "arm": arm, "K": k, "region": "full", "variable": "t2m", "unit": "K",
                           "lead_hours": lead, "init_time": init, "valid_times": valid,
                           "sample_id": f"fixture-{index}", **calculated, **pins}
                    group.append(row)
                rows.extend(group)
                records.append({"seed": seed, "arm": arm, "K": k, "region": "full", "variable": "t2m", "unit": "K",
                                "lead_hours": lead, **pooled_statistics(group), "cases": sorted(cases),
                                "case_set_sha256": digest(sorted(cases)), **pins})
                if arm == "process":
                    latency.append({"seed": seed, "arm": arm, "K": k, "lead_hours": lead,
                                    "median_seconds_per_batch": k * .01, "scope": ISOLATED_SCOPE,
                                    "gpu_latency_measured": True, "provenance_sha256": pins["provenance_sha256"]})
    return protocol, records, rows, latency


@pytest.fixture
def inputs():
    return _fixture()


def _evaluate(inputs, *, verified=True):
    return evaluate_adaptive_gate(*inputs, inventory_verified=verified)


def _refused(result):
    assert result["status"] == "refused" and result["evaluated"] is False
    assert result["start_training"] is False and result["gate_met"] is None
    assert result["oracle_deployable"] is False and result["scientific_claim"] is False
    assert result["controller_training_executed"] is False
    json.dumps(result, allow_nan=False)


def _rewrite_mse(inputs, seed, lead, arm, k, values):
    _, records, rows, _ = inputs
    group = [row for row in rows if (row["seed"], row["lead_hours"], row["arm"], row["K"]) == (seed, lead, arm, k)]
    for index, row in enumerate(group):
        mse = values[index % len(values)]
        stats = {"mse": mse, "climatology_mse": 4., "acc_dot": 4.,
                 "acc_forecast_energy": mse + 4., "acc_target_energy": 4.}
        calculated = pooled_statistics([stats])
        calculated.pop("bad_case_counts")
        row.update(calculated)
    record = next(row for row in records if (row["seed"], row["lead_hours"], row["arm"], row["K"]) == (seed, lead, arm, k))
    record.update(pooled_statistics(group))


def test_exact_actual_per_seed_schema_eligible_is_json_pure_and_not_deployment(inputs, tmp_path):
    before = deepcopy(inputs)
    result = _evaluate(inputs)
    assert result["status"] == "eligible" and result["evaluated"] is True
    assert result["start_training"] is True and result["gate_met"] is True
    assert all(item == {"evaluated": True, "met": True} for item in result["conditions"].values())
    evidence = result["seed_lead_evidence"]
    assert {(row["seed"], row["lead_hours"]) for row in evidence} == {(s, l) for s in (41, 42, 43) for l in (6, 12)}
    assert all(set(row["fixed_K"]) == {"1", "2", "4"} and set(row["actual_latency"]) == {"1", "2", "4"} for row in evidence)
    assert evidence[0]["oracle"]["rmse"] == 1.
    assert evidence[0]["oracle"]["rmse"] < evidence[0]["best_fixed_rmse"]
    assert evidence[0]["oracle"]["unique_optimal_K_counts"] == {"1": 11, "2": 0, "4": 11}
    assert all(row["oracle"]["uses_future_ground_truth"] is True for row in evidence)
    assert result["oracle_deployable"] is result["oracle_deployed"] is result["controller_training_executed"] is False
    assert result["scientific_claim"] is False and result["test_read"] is False and result["limitations"]
    assert inputs == before
    path = tmp_path / "synthetic_engineering_gate.json"
    path.write_text(json.dumps(result, allow_nan=False))
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize("seed,lead", [(s, l) for s in (41, 42, 43) for l in (6, 12)])
@pytest.mark.parametrize("mode", ["equal", "worse"])
def test_package_condition_fails_for_any_single_seed_lead_even_if_other_five_improve(inputs, seed, lead, mode):
    process_mse = next(row["mse"] for row in inputs[1] if
                       (row["seed"], row["lead_hours"], row["arm"], row["K"]) == (seed, lead, "process", 4))
    _rewrite_mse(inputs, seed, lead, "old_ours", 4, [process_mse if mode == "equal" else .5])
    result = _evaluate(inputs)
    assert result["status"] == "not-started" and result["evaluated"] is True
    assert result["conditions"]["package_improvement"]["met"] is False
    assert result["conditions"]["accuracy_cost_tradeoff"]["met"] is True
    assert result["conditions"]["case_heterogeneity"]["met"] is True
    assert result["start_training"] is False


@pytest.mark.parametrize("seed,lead", [(s, l) for s in (41, 42, 43) for l in (6, 12)])
def test_accuracy_condition_fails_for_any_seed_lead_not_cross_seed_mean(inputs, seed, lead):
    k4 = next(row["mse"] for row in inputs[1] if
              (row["seed"], row["lead_hours"], row["arm"], row["K"]) == (seed, lead, "process", 4))
    _rewrite_mse(inputs, seed, lead, "process", 1, [k4])
    result = _evaluate(inputs)
    assert result["status"] == "not-started" and result["evaluated"] is True
    assert result["conditions"]["accuracy_cost_tradeoff"]["met"] is False
    assert result["conditions"]["package_improvement"]["met"] is True
    assert result["start_training"] is False


@pytest.mark.parametrize("seed,lead", [(s, l) for s in (41, 42, 43) for l in (6, 12)])
@pytest.mark.parametrize("median", [.04, .05])
def test_actual_latency_condition_fails_when_k1_equal_or_slower_for_any_seedlead(inputs, seed, lead, median):
    row = next(row for row in inputs[3] if (row["seed"], row["lead_hours"], row["K"]) == (seed, lead, 1))
    row["median_seconds_per_batch"] = median
    result = _evaluate(inputs)
    assert result["status"] == "not-started" and result["evaluated"] is True
    assert result["conditions"]["accuracy_cost_tradeoff"]["met"] is False
    assert result["conditions"]["package_improvement"]["met"] is True
    assert result["conditions"]["case_heterogeneity"]["met"] is True
    assert result["start_training"] is False


@pytest.mark.parametrize("values", [((9.,), (4.,), (1.,)), ((9.,), (1.,), (1.,)), ((1.,), (1.,), (1.,))])
def test_oracle_all_k4_wins_or_ties_do_not_create_heterogeneity(inputs, values):
    for k, errors in zip((1, 2, 4), values):
        _rewrite_mse(inputs, 43, 12, "process", k, errors)
    result = _evaluate(inputs)
    assert result["status"] == "not-started" and result["evaluated"] is True
    assert result["conditions"]["case_heterogeneity"]["met"] is False
    oracle = result["seed_lead_evidence"][-1]["oracle"]
    assert oracle["unique_optimal_depths"] <= 1
    assert oracle["rmse"] == result["seed_lead_evidence"][-1]["best_fixed_rmse"]
    assert result["start_training"] is False and result["oracle_deployable"] is False


def test_oracle_can_have_strict_gain_but_ties_do_not_count_as_unique_preferences(inputs):
    _rewrite_mse(inputs, 43, 12, "process", 1, [1., 9.])
    _rewrite_mse(inputs, 43, 12, "process", 2, [1., 4.])
    _rewrite_mse(inputs, 43, 12, "process", 4, [4., 1.])
    result = _evaluate(inputs)
    row = result["seed_lead_evidence"][-1]
    assert row["oracle"]["rmse"] < row["best_fixed_rmse"]
    assert row["oracle"]["tie_case_count"] == 11
    assert row["oracle"]["unique_optimal_K_counts"] == {"1": 0, "2": 0, "4": 10}
    assert result["conditions"]["case_heterogeneity"]["met"] is False
    assert result["status"] == "not-started" and result["start_training"] is False


@pytest.mark.parametrize("verified", [False, None, 1, "true", {}, {"verified": True}])
def test_engineering_condition_requires_literal_upstream_inventory_acceptance(inputs, verified):
    result = _evaluate(inputs, verified=verified)
    _refused(result)
    assert result["conditions"]["complete_engineering_evidence"] == {"evaluated": True, "met": False}


@pytest.mark.parametrize("change", ["missing-gate", "empty-gate", "unknown-format", "extra", "candidate", "baseline", "region",
                                    "variable", "unit", "seeds", "leads", "ks", "stage", "primary-variable", "primary-unit",
                                    "primary-leads", "primary-tolerance", "missing-tolerance", "tolerance", "boolean-tolerance"])
def test_missing_unknown_or_changed_frozen_gate_primary_never_defaults_to_pass(inputs, change):
    protocol = inputs[0]
    reporting = protocol["reporting"]
    gate = reporting["adaptive_gate"]
    if change == "missing-gate": reporting.pop("adaptive_gate")
    elif change == "empty-gate": reporting["adaptive_gate"] = {}
    elif change == "unknown-format": gate["format"] = "unrecognized"
    elif change == "extra": gate["rmse_threshold"] = 0.
    elif change in ("candidate", "baseline", "region", "variable", "unit"): gate[change] = "posthoc"
    elif change == "seeds": gate["seeds"] = [41, 42]
    elif change == "leads": gate["lead_hours"] = [6]
    elif change == "ks": gate["kernels"] = [1, 4]
    elif change == "stage": protocol["stage"] = "B"
    elif change == "primary-variable": reporting["primary"]["variable"] = "variable_1"
    elif change == "primary-unit": reporting["primary"]["unit"] = "C"
    elif change == "primary-leads": reporting["primary"]["lead_hours"] = [12, 6]
    elif change == "primary-tolerance": reporting["primary"]["degradation_tolerance"] = .01
    elif change == "missing-tolerance": reporting.pop("tolerances")
    elif change == "tolerance": reporting["tolerances"]["K"] = .01
    elif change == "boolean-tolerance": reporting["tolerances"]["K"] = False
    _refused(_evaluate(inputs))


@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown-extra", "unknown-seed", "unknown-arm", "unknown-K", "unknown-lead"])
def test_missing_duplicate_or_undeclared_primary_inputs_are_refused(inputs, index, change):
    rows = inputs[index]
    if change == "missing": rows.pop()
    elif change == "duplicate": rows.append(deepcopy(rows[0]))
    elif change == "unknown-extra": rows[0]["undeclared"] = True
    elif change == "unknown-seed": rows[0]["seed"] = 44
    elif change == "unknown-arm": rows[0]["arm"] = "posthoc_candidate"
    elif change == "unknown-K": rows[0]["K"] = 3
    elif change == "unknown-lead": rows[0]["lead_hours"] = 18
    _refused(_evaluate(inputs))


@pytest.mark.parametrize("change", ["count", "case-set", "case-digest", "rmse", "mse", "skill", "acc", "physical-stats", "bad-count",
                                    "unit", "per-case-unit", "case-time", "case-valid", "case-duplicate", "sample-id", "missing-sample-id",
                                    "pin", "case-pin", "normalized-raw", "nonfinite", "negative", "case-count", "cross-seed-table"])
def test_cohort_units_actual_aggregate_rebuild_and_case_statistics_guards(inputs, change):
    records, rows = inputs[1:3]
    record = records[0]
    row = rows[0]
    if change == "count": record["n_initializations"] -= 1
    elif change == "case-set": record["cases"][-1] = _cohort(6, 23)[-1]
    elif change == "case-digest": record["case_set_sha256"] = digest("other-cohort")
    elif change == "rmse": record["rmse"] += .01
    elif change == "mse": record["mse"] += .01
    elif change == "skill": record["mse_skill"] += .01
    elif change == "acc": record["pooled_acc"] += .01
    elif change == "physical-stats": row["acc_target_energy"] += 1
    elif change == "bad-count": record["bad_case_counts"]["worse_than_climatology"] -= 1
    elif change == "unit": record["unit"] = "C"
    elif change == "per-case-unit": row["unit"] = "normalized"
    elif change == "case-time": row["init_time"], row["valid_times"] = _cohort(6, 23)[-1]
    elif change == "case-valid": row["valid_times"] = _cohort(12, 1)[0][1]
    elif change == "case-duplicate": row["init_time"], row["valid_times"] = rows[1]["init_time"], rows[1]["valid_times"]
    elif change == "sample-id": row["sample_id"] = "mismatched"
    elif change == "missing-sample-id": row.pop("sample_id")
    elif change == "pin": record["provenance_sha256"] = "unknown"
    elif change == "case-pin": row["data_identity"] = digest("other-data")
    elif change == "normalized-raw": row["acc_statistic_units"] = "normalized_anomaly_squared"
    elif change == "nonfinite": row["mse"] = float("nan")
    elif change == "negative": row["mse"] = -1.
    elif change == "case-count": row["n_initializations"] = 2
    elif change == "cross-seed-table": records[0] = {"seeds": [41, 42, 43], "seed_rmse": {"41": 4., "42": 4., "43": 4.}}
    _refused(_evaluate(inputs))


@pytest.mark.parametrize("change", ["zero", "negative", "nan", "not-measured", "theory", "scope", "hash", "provenance", "missing-k2"])
def test_missing_nonpositive_theoretical_or_inconsistent_cost_is_refused(inputs, change):
    row = inputs[3][0]
    if change == "zero": row["median_seconds_per_batch"] = 0.
    elif change == "negative": row["median_seconds_per_batch"] = -.01
    elif change == "nan": row["median_seconds_per_batch"] = float("nan")
    elif change == "not-measured": row["gpu_latency_measured"] = False
    elif change == "theory": row["median_seconds_per_batch"] = "FLOPs"
    elif change == "scope": row["scope"] = "whole evaluation including IO and metrics"
    elif change == "hash": row["provenance_sha256"] = "not-a-digest"
    elif change == "provenance": row["provenance_sha256"] = digest("different-receipt")
    elif change == "missing-k2": inputs[3][:] = [row for row in inputs[3] if row["K"] != 2]
    _refused(_evaluate(inputs))


@pytest.mark.parametrize("change", ["missing-seed", "duplicate-case", "time-wrong", "count-wrong", "t2m-unit", "missing-variable"])
def test_frozen_cohort_and_17_variable_declaration_are_not_inferred(inputs, change):
    data = inputs[0]["data"]
    cases = data["evaluation_cases"]["6"]["cases"]
    if change == "missing-seed": inputs[1][:] = [row for row in inputs[1] if row["seed"] != 43]
    elif change == "duplicate-case": cases[-1] = cases[0]
    elif change == "time-wrong": cases[0][1] = _cohort(12, 1)[0][1]
    elif change == "count-wrong": cases.pop()
    elif change == "t2m-unit": data["units"][0] = "C"
    elif change == "missing-variable": data["channels"].pop(); data["units"].pop()
    _refused(_evaluate(inputs))


def test_known_nonprimary_17_variable_regions_arms_and_long_leads_are_not_gate_candidates(inputs):
    records, rows, latency = inputs[1:]
    for variable, unit in zip(CHANNELS, UNITS):
        for region in ("full", "interior", "edge_2"):
            if variable == "t2m" and region == "full":
                continue
            records.append({**deepcopy(records[0]), "variable": variable, "unit": unit, "region": region})
            rows.append({**deepcopy(rows[0]), "variable": variable, "unit": unit, "region": region})
    records.append({**deepcopy(records[0]), "arm": "matched_generic"})
    rows.append({**deepcopy(rows[0]), "arm": "matched_generic"})
    records.append({**deepcopy(records[0]), "lead_hours": 72})
    rows.append({**deepcopy(rows[0]), "lead_hours": 72})
    latency.append({**deepcopy(latency[0]), "arm": "matched_generic"})
    latency.append({**deepcopy(latency[0]), "lead_hours": 72})
    result = _evaluate(inputs)
    assert result["status"] == "eligible" and len(result["seed_lead_evidence"]) == 6
    assert {row["variable"] for row in result["seed_lead_evidence"]} == {"t2m"}
    assert {row["lead_hours"] for row in result["seed_lead_evidence"]} == {6, 12}


def test_pool_mse_before_sqrt_not_average_case_rmse_or_acc(inputs):
    result = _evaluate(inputs)
    first = result["seed_lead_evidence"][0]
    assert first["fixed_K"]["1"]["rmse"] == math.sqrt(5.)
    assert first["fixed_K"]["1"]["rmse"] != (math.sqrt(1.) + math.sqrt(9.)) / 2
    record = first["aggregate_evidence"][1]
    mean_acc = (2 / math.sqrt(5.) + 2 / math.sqrt(13.)) / 2
    assert record["pooled_acc"] == pytest.approx(2 / 3)
    assert record["pooled_acc"] != pytest.approx(mean_acc)


def test_reordering_does_not_change_exact_pairing_or_gate(inputs):
    before = _evaluate(inputs)
    shuffled = (inputs[0], *[list(reversed(rows)) for rows in inputs[1:]])
    after = _evaluate(shuffled)
    assert before == after


def test_missing_all_evidence_schema_is_refused_not_empty_all_pass():
    _refused(evaluate_adaptive_gate({}, [], [], [], inventory_verified=True))


def test_no_future_gt_strategy_or_controller_can_enter_exact_schema(inputs):
    inputs[0]["reporting"]["adaptive_gate"]["controller"] = "per-case minimum future ground truth"
    _refused(_evaluate(inputs))
