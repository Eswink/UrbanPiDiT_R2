"""Synthetic producer-label identity checks; no actual B/C artifacts or model/data imports."""
from __future__ import annotations

import copy
import json
import math

import pytest

from training.r7_v2_utc_statistics import checked_metric, identity


def synthetic_metric(*, kind="model", mse=1.0, climate=1.0, forecast=1.0,
                     dot=0.5, skill=0.0, acc=0.5, reasons=()):
    data = {"channels": ["synthetic_temperature"], "units": ["K"], "normalization_std": [1.0]}
    job = {"phase": "evaluate", "seed": 41, "arm": "continue_l6", "lead": 6, "reasoning_steps": 4}
    init = "2001-01-01T00:00:00"
    valid = "2001-01-01T06:00:00"
    row = {
        "sample_id": "SYNTHETIC_SCALAR_ONLY", "init_time": init,
        "valid_time": valid, "valid_times": [valid], "region": "full",
        "variable": "synthetic_temperature", "unit": "K", "lead_hours": 6,
        "reasoning_steps": 4 if kind == "model" else None,
        "n_initializations": 1, "mse": mse, "climatology_mse": climate,
        "mse_climatology": climate, "rmse": math.sqrt(mse),
        "rmse_climatology": math.sqrt(climate), "mse_skill": skill,
        "acc": acc, "acc_dot": dot, "acc_forecast_energy": forecast,
        "acc_target_energy": climate, "acc_statistic_units": "normalized_anomaly_squared",
        "acc_status": "defined" if acc is not None else "undefined_zero_anomaly_energy",
        "skill_status": "defined" if skill is not None else "undefined_zero_climatology_energy",
        "bad_reasons": list(reasons), "margin_cells": 0, "n_grid_points": 4,
        "full_area_fraction": 1.0,
    }
    if kind != "model":
        row.update(baseline_kind=kind, zero_train_updates=0, parameters=0, trainable_parameters=0)
    return data, job, row, copy.deepcopy(row), identity(row, 6)


def check(fixture, kind="model"):
    data, job, row, expected, ident = fixture
    return checked_metric(row, data, ident, job, kind, expected)


def test_serialized_climatology_zero_label_survives_binary64_recomputed_sign():
    mse = math.nextafter(1.0, math.inf)
    assert 1.0 - mse == -2.220446049250313e-16
    fixture = synthetic_metric(kind="climatology", mse=mse, forecast=0.0,
                               dot=0.0, skill=0.0, acc=None, reasons=("undefined_acc",))
    result = check(fixture, "climatology")
    assert result["mse"] == mse
    assert result["climatology_mse"] == 1.0
    assert fixture[2]["mse_skill"] == fixture[3]["mse_skill"] == 0.0
    assert fixture[2]["bad_reasons"] == ["undefined_acc"]


def test_real_negative_serialized_metrics_and_flags_are_retained():
    fixture = synthetic_metric(mse=3.0, dot=-0.5, skill=-2.0, acc=-0.5,
                               reasons=("negative_mse_skill", "negative_acc"))
    result = check(fixture)
    assert result["mse"] == 3.0 and result["acc_dot"] == -0.5
    assert fixture[2]["mse_skill"] == -2.0 and fixture[2]["acc"] == -0.5


@pytest.mark.parametrize("flags", [("negative_acc",), ("negative_mse_skill",), ()])
def test_dropped_genuine_negative_flags_are_rejected(flags):
    fixture = synthetic_metric(mse=3.0, dot=-0.5, skill=-2.0, acc=-0.5,
                               reasons=("negative_mse_skill", "negative_acc"))
    fixture[2]["bad_reasons"] = list(flags)
    with pytest.raises(ValueError, match="negative/undefined per-case labels"):
        check(fixture)


def test_tiny_but_serialized_negative_is_not_clamped_or_filtered():
    mse = math.nextafter(1.0, math.inf)
    skill = 1.0 - mse
    fixture = synthetic_metric(kind="climatology", mse=mse, forecast=0.0, dot=0.0,
                               skill=skill, acc=None, reasons=("negative_mse_skill", "undefined_acc"))
    result = check(fixture, "climatology")
    assert result["mse"] == mse and fixture[2]["mse_skill"] < 0.0
    fixture[2]["bad_reasons"] = ["undefined_acc"]
    with pytest.raises(ValueError, match="negative/undefined per-case labels"):
        check(fixture, "climatology")


def test_serialized_zero_rejects_added_negative_label_despite_recomputed_sign():
    fixture = synthetic_metric(kind="climatology", mse=math.nextafter(1.0, math.inf),
                               forecast=0.0, dot=0.0, skill=0.0, acc=None,
                               reasons=("undefined_acc",))
    fixture[2]["bad_reasons"] = ["negative_mse_skill", "undefined_acc"]
    fixture[3]["bad_reasons"] = list(fixture[2]["bad_reasons"])
    with pytest.raises(ValueError, match="negative/undefined per-case labels"):
        check(fixture, "climatology")


@pytest.mark.parametrize("flags", [("undefined_acc",), ("undefined_mse_skill",), ()])
def test_dropped_undefined_flags_are_rejected(flags):
    fixture = synthetic_metric(mse=0.0, climate=0.0, forecast=0.0, dot=0.0,
                               skill=None, acc=None, reasons=("undefined_mse_skill", "undefined_acc"))
    assert check(fixture)["mse"] == 0.0
    fixture[2]["bad_reasons"] = list(flags)
    with pytest.raises(ValueError, match="negative/undefined per-case labels"):
        check(fixture)


@pytest.mark.parametrize("field", ["mse_skill", "acc"])
def test_undefined_metric_imputation_is_still_rejected(field):
    fixture = synthetic_metric(mse=0.0, climate=0.0, forecast=0.0, dot=0.0,
                               skill=None, acc=None, reasons=("undefined_mse_skill", "undefined_acc"))
    fixture[2][field] = 0.0
    with pytest.raises(ValueError, match="undefined metric cannot be imputed"):
        check(fixture)


def test_numeric_comparison_tolerance_is_not_relaxed():
    fixture = synthetic_metric()
    fixture[2]["mse_skill"] = 0.01
    with pytest.raises(ValueError, match="same-case metric mismatch"):
        check(fixture)


def test_embedded_diagnostic_signature_remains_exact():
    fixture = synthetic_metric(mse=3.0, dot=-0.5, skill=-2.0, acc=-0.5,
                               reasons=("negative_mse_skill", "negative_acc"))
    fixture[3]["bad_reasons"] = ["negative_acc"]
    with pytest.raises(ValueError, match="CSV/embedded diagnostic mismatch"):
        check(fixture)


def test_invalid_defined_status_is_still_rejected():
    fixture = synthetic_metric()
    fixture[2]["acc_status"] = "undefined_zero_anomaly_energy"
    with pytest.raises(ValueError, match="metric undefined/defined status mismatch"):
        check(fixture)


def csv_scalars(fixture):
    row = fixture[2]
    for key, value in list(row.items()):
        row[key] = json.dumps(value) if isinstance(value, list) else "" if value is None else str(value)
    return fixture


@pytest.mark.parametrize("case", ["zero_roundoff", "negative", "positive", "undefined"])
def test_raw_csv_string_and_null_diagnostic_identity(case):
    kind = "model"
    if case == "zero_roundoff":
        kind = "climatology"
        fixture = synthetic_metric(kind=kind, mse=math.nextafter(1.0, math.inf),
                                   forecast=0.0, dot=0.0, acc=None, reasons=("undefined_acc",))
    elif case == "negative":
        fixture = synthetic_metric(mse=3.0, dot=-0.5, skill=-2.0, acc=-0.5,
                                   reasons=("negative_mse_skill", "negative_acc"))
    elif case == "positive":
        fixture = synthetic_metric(mse=0.5, dot=0.75, skill=0.5, acc=0.75)
    else:
        fixture = synthetic_metric(mse=0.0, climate=0.0, forecast=0.0, dot=0.0,
                                   skill=None, acc=None, reasons=("undefined_mse_skill", "undefined_acc"))
    expected = fixture[3]
    check(csv_scalars(fixture), kind)
    assert json.loads(fixture[2]["bad_reasons"]) == expected["bad_reasons"]
    if case == "negative":
        assert float(fixture[2]["mse_skill"]) < 0.0 and float(fixture[2]["acc"]) < 0.0


@pytest.mark.parametrize("flags", [[], ["negative_acc"], ["undefined_acc"]])
def test_raw_csv_negative_diagnostic_tampering_still_rejected(flags):
    fixture = csv_scalars(synthetic_metric(mse=3.0, dot=-0.5, skill=-2.0, acc=-0.5,
                                          reasons=("negative_mse_skill", "negative_acc")))
    fixture[2]["bad_reasons"] = json.dumps(flags)
    with pytest.raises(ValueError, match="negative/undefined per-case labels"):
        check(fixture)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "True"])
def test_serialized_metric_nonfinite_or_boolean_text_still_rejected(value):
    fixture = csv_scalars(synthetic_metric())
    fixture[2]["mse_skill"] = value
    with pytest.raises(ValueError, match="finite measurement"):
        check(fixture)
