"""CI-safe checks for the S3 update-budget screen's registered reading.

The reading is the load-bearing part of this round: the verdict functions must
actually implement the frozen decision text, so every rule here carries a
counterproof (mixed signs must be unresolved, never averaged; the gate fails on
exactly one positive cell; zeros pass; perturbation propagates; incomplete
tables are rejected; the go/no-go requires both the primary and the gate).
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_driver():
    path = ROOT / "scripts" / "study_r7_s3_update_budget.py"
    spec = importlib.util.spec_from_file_location("study_r7_s3_update_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ub = _load_driver()


def test_frozen_constants():
    assert ub.CANDIDATE_UPDATES == 800
    assert ub.CONTROL_UPDATES == 400
    assert ub.CANDIDATE_MODE == "l6"
    assert ub.WARMUP == 40  # 5% of the 800-update endpoint
    assert ub.PRIMARY_LEADS == (6, 12)
    assert ub.GATE_TOLERANCE == 0.0
    assert ub.SEEDS == (41, 42, 43)


def test_test_split_is_refused():
    with pytest.raises(ValueError, match="sealed"):
        ub.refused_test_manifest("/somewhere/test.jsonl")
    ub.refused_test_manifest("/somewhere/val.jsonl")  # val is fine


def _cells(deltas):
    out = {}
    for key, value in deltas.items():
        lead, variable = key.split("|")
        out[key] = {"lead_hours": int(lead), "variable": variable,
                    "candidate_rmse": 10.0 + value, "control_rmse": 10.0,
                    "rmse_delta": value,
                    "relative_mse_change": (10.0 + value) ** 2 / 100.0 - 1.0}
    return out


def _uniform(t2m_by_seed, gate_cells=None):
    family = {}
    for seed, deltas in t2m_by_seed.items():
        cells = {}
        for lead, value in deltas.items():
            cells[f"{lead}|t2m"] = value
        for lead in (6, 12, 24, 48, 72):
            for variable in ("u10", "v10", "mslp"):
                cells[f"{lead}|{variable}"] = (gate_cells or {}).get((seed, lead, variable), -0.1)
        family[str(seed)] = _cells(cells)
    return family


def test_primary_supported_requires_both_leads_all_seeds_lower():
    verdict = ub.primary_verdict(_uniform({41: {6: -0.2, 12: -0.1},
                                           42: {6: -0.1, 12: -0.05},
                                           43: {6: -0.3, 12: -0.2}}))
    assert verdict["overall"] == "supported"


def test_primary_sign_disagreement_is_unresolved_and_never_averaged():
    verdict = ub.primary_verdict(_uniform({41: {6: -0.2, 12: -0.1},
                                           42: {6: -0.1, 12: -0.05},
                                           43: {6: +0.4, 12: -0.2}}))
    assert verdict["overall"] == "unsupported"
    assert verdict["per_lead"][6] == "unresolved"


def test_primary_worsened_and_partial_support_are_not_supported():
    worsened = ub.primary_verdict(_uniform({41: {6: 0.2, 12: 0.1},
                                            42: {6: 0.1, 12: 0.05},
                                            43: {6: 0.3, 12: 0.2}}))
    assert worsened["overall"] == "worsened"
    partial = ub.primary_verdict(_uniform({41: {6: -0.2, 12: -0.1},
                                           42: {6: -0.1, 12: +0.01},
                                           43: {6: -0.3, 12: -0.2}}))
    assert partial["overall"] == "unsupported"


def test_gate_fails_on_exactly_one_positive_cell():
    family = _uniform({41: {6: -0.1, 12: -0.1}, 42: {6: -0.1, 12: -0.1},
                       43: {6: -0.1, 12: -0.1}},
                      gate_cells={(43, 72, "mslp"): 0.5})
    verdict = ub.gate_verdict(family)
    assert verdict["passed"] is False
    assert len(verdict["failures"]) == 1
    failure = verdict["failures"][0]
    assert (failure["seed"], failure["lead_hours"], failure["variable"]) == (43, 72, "mslp")
    assert failure["relative_mse_change"] == pytest.approx((10.5 ** 2 - 100) / 100)


def test_gate_zero_cells_pass():
    family = _uniform({41: {6: -0.1, 12: -0.1}, 42: {6: -0.1, 12: -0.1},
                       43: {6: -0.1, 12: -0.1}},
                      gate_cells={(41, 6, "u10"): 0.0})
    assert ub.gate_verdict(family)["passed"] is True


def test_advance_requires_both_primary_and_gate():
    supported = {"overall": "supported"}
    failing_gate = {"passed": False}
    passing_gate = {"passed": True}
    assert ub.advance_decision(supported, passing_gate) == "advance-to-S4-freeze"
    assert ub.advance_decision(supported, failing_gate) == "registered-negative"
    assert ub.advance_decision({"overall": "worsened"}, passing_gate) == "registered-negative"
    assert ub.advance_decision({"overall": "unresolved"}, passing_gate) == "registered-negative"


def test_loss_windows_carry_segment_means():
    report = {"losses": [{"loss": float(value)} for value in range(1, 201)]}
    windows = ub.loss_windows(report, size=100)
    assert windows == {"1-100": 50.5, "101-200": 150.5}
    short = ub.loss_windows({"losses": [{"loss": 1.0}, {"loss": 3.0}]}, size=100)
    assert short == {"1-2": 2.0}


def _write_rmse_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


def test_paired_cells_need_full_variable_sets_and_propagate_perturbations(tmp_path):
    variables = ["t2m", "u10", "v10", "mslp"] + [f"filler{index}" for index in range(13)]
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    for lead in (6, 12, 24, 48, 72):
        _write_rmse_table(control / f"lead_{lead:03d}h" / "rmse.csv",
                          [{"variable": name, "rmse": "10.0"} for name in variables])
        rows = [{"variable": name, "rmse": "10.0"} for name in variables]
        rows[0] = {"variable": "t2m", "rmse": "9.5"}
        _write_rmse_table(candidate / f"lead_{lead:03d}h" / "rmse.csv", rows)
    cells = ub.paired_cells(candidate, control)
    assert cells["6|t2m"]["rmse_delta"] == pytest.approx(-0.5)
    assert cells["6|u10"]["rmse_delta"] == pytest.approx(0.0)
    _write_rmse_table(candidate / "lead_012h" / "rmse.csv",
                      [{"variable": name, "rmse": "8.0" if name == "t2m" else "10.0"}
                       for name in variables])
    cells = ub.paired_cells(candidate, control)
    assert cells["12|t2m"]["rmse_delta"] == pytest.approx(-2.0)


def test_paired_cells_reject_incomplete_tables(tmp_path):
    control = tmp_path / "control"
    for lead in (6, 12, 24, 48, 72):
        _write_rmse_table(control / f"lead_{lead:03d}h" / "rmse.csv",
                          [{"variable": "t2m", "rmse": "10.0"}])
    with pytest.raises(RuntimeError, match="17 variables"):
        ub.paired_cells(control, control)
