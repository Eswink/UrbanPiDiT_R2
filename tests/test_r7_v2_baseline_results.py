"""Mandatory zero-trained baseline schema and strict same-case counterexamples."""
from copy import deepcopy
import csv
import json

import pytest

from training.r7_v2_protocol import BASELINE_REPORTING
from training.r7_v2_results import verify_baseline_artifacts


def baseline_fixture(provenance):
    """Add constant zero-training references independent of the model arm/K."""
    regions, initializations, flat = [], [], []
    for case in provenance["initializations"]:
        rows = []
        for kind in ("persistence", "climatology"):
            for original in case["region_metrics"]:
                row = deepcopy(original)
                # fixture std=2, target physical anomaly=2: normalized TE=1.
                row.update(baseline_kind=kind, zero_train_updates=0, parameters=0, trainable_parameters=0,
                           mse=1. if kind == "persistence" else 4., rmse=1. if kind == "persistence" else 2.,
                           climatology_mse=4., mse_climatology=4., rmse_climatology=2.,
                           acc_dot=.5 if kind == "persistence" else 0.,
                           acc_forecast_energy=.25 if kind == "persistence" else 0., acc_target_energy=1.,
                           mse_skill=.75 if kind == "persistence" else 0., acc=1. if kind == "persistence" else None,
                           acc_status="defined" if kind == "persistence" else "undefined_zero_anomaly_energy",
                           skill_status="defined", bad_reasons=[] if kind == "persistence" else ["undefined_acc"],
                           n_initializations=1)
                rows.append(row)
        initializations.append({"sample_id": case["sample_id"], "init_time": case["init_time"],
                                "valid_times": case["valid_times"], "region_metrics": rows})
        flat.extend({"sample_id": case["sample_id"], "init_time": case["init_time"], "valid_time": case["valid_times"][0],
                     "valid_times": case["valid_times"], "reasoning_steps": None, **row} for row in rows)
    count = len(initializations)
    for original in initializations[0]["region_metrics"]:
        row = deepcopy(original)
        row["n_initializations"] = count
        for key in ("acc_dot", "acc_forecast_energy", "acc_target_energy"):
            row[key] *= count
        regions.append(row)
    provenance.update(baseline_region_metrics=regions, baseline_initializations=initializations,
                      baseline_definition={"persistence": {"kind": "known-last-history-frame", "training_updates": 0},
                                           "climatology": {"kind": "train-only-month-hour", "training_updates": 0},
                                           "case_pairing": "same exact requested-lead initializations and full/interior/edge_2"},
                      climatology={"kind": "train-only-month-hour-grid-mean-v1"})
    return regions, flat


def baseline_inputs():
    from test_r7_v2_results import actual_artifacts
    protocol, job, provenance, _, _ = actual_artifacts()
    protocol["baseline_reporting"] = BASELINE_REPORTING
    regions, cases = baseline_fixture(provenance)
    return protocol, job, provenance, deepcopy(regions), deepcopy(cases)


def test_baselines_same_cases_physical_units_zero_training_no_fake_model_k():
    protocol, job, provenance, regions, cases = baseline_inputs()
    verified = verify_baseline_artifacts(provenance, regions, cases, protocol, job)
    assert len(verified) == 102
    assert {row["baseline_kind"] for row in verified} == {"persistence", "climatology"}
    assert all(row["zero_train_updates"] == row["parameters"] == row["trainable_parameters"] == 0 for row in verified)
    climate = [row for row in verified if row["baseline_kind"] == "climatology"]
    assert all(row["mse_skill"] == 0 and row["pooled_acc"] is None for row in climate)
    assert all(row["model_depth_applicable"] is False and row["reference_K"] == 4 for row in verified)
    assert all(row["cases"] == protocol["data"]["evaluation_cases"]["6"]["cases"] for row in verified)


@pytest.mark.parametrize("change", ["missing-kind", "missing-case", "unit", "trained", "parameters", "case",
                                    "embedded", "formula", "count", "fake-k", "climatology-kind", "missing-contract"])
def test_mandatory_baseline_guards_fail_without_partial_or_pseudotraining(change):
    protocol, job, provenance, regions, cases = baseline_inputs()
    if change == "missing-kind": regions = [row for row in regions if row["baseline_kind"] == "persistence"]
    elif change == "missing-case": cases.pop()
    elif change == "unit": cases[0]["unit"] = "C"
    elif change == "trained": cases[0]["zero_train_updates"] = 1
    elif change == "parameters": regions[0]["parameters"] = 1
    elif change == "case": cases[0]["init_time"] = "2017-01-01T00:00:00"
    elif change == "embedded": provenance["baseline_initializations"][0]["region_metrics"].pop()
    elif change == "formula": cases[0]["acc_dot"] += .1
    elif change == "count": regions[0]["n_initializations"] -= 1
    elif change == "fake-k": cases[0]["reasoning_steps"] = 4
    elif change == "climatology-kind": provenance["baseline_definition"]["climatology"]["kind"] = "future-climatology"
    elif change == "missing-contract": protocol.pop("baseline_reporting")
    with pytest.raises((ValueError, KeyError)):
        verify_baseline_artifacts(provenance, regions, cases, protocol, job)


def test_baseline_actual_helper_schema_matches_results_on_tmp_store(tmp_path, monkeypatch):
    from test_r7_v2_evaluation import fixture as make_fixture, _run
    fixture = make_fixture.__wrapped__(tmp_path, monkeypatch)
    report = _run(fixture, lead=12, k=2, name="baseline-real-helper")
    cases = [[item["init_time"], item["valid_times"]] for item in report["initializations"]]
    protocol = {"data": {"channels": report["channels"], "units": report["units"],
                          "normalization_std": fixture["root"]["normalization_std"][:].tolist(),
                          "evaluation_cases": {"12": {"cases": cases}}}, "baseline_reporting": BASELINE_REPORTING}
    job = {"seed": 41, "arm": "process", "lead": 12, "reasoning_steps": 2}
    directory = tmp_path / "baseline-real-helper"
    with (directory / "baseline_region_metrics.csv").open(encoding="utf-8") as stream:
        regional = list(csv.DictReader(stream))
    with (directory / "baseline_per_case_metrics.csv").open(encoding="utf-8") as stream:
        per_case = list(csv.DictReader(stream))
    verified = verify_baseline_artifacts(report, regional, per_case, protocol, job)
    assert len(verified) == 102 and all(row["n_initializations"] == 21 for row in verified)
    assert all(row["model_depth_applicable"] is False for row in verified)
    assert len(per_case) == 21 * 102
    assert all(row["reasoning_steps"] == "" for row in per_case)
