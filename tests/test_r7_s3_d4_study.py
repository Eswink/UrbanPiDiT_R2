"""CI-safe checks for the S3-D4 R-C candidate driver's registered reading.

The reading is the load-bearing part of this round: a verdict function that
returned a fixed string would pass any test asserting only that it returns a
string, so every rule here has a counterproof (mixed signs must be unresolved,
never averaged; the gate fails on exactly one positive cell; a perturbed CSV
must move the paired delta by the same amount).
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
    path = ROOT / "scripts" / "study_r7_s3_d4_rc_candidate.py"
    spec = importlib.util.spec_from_file_location("study_r7_s3_d4_rc_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d4 = _load_driver()


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
    verdict = d4.primary_verdict(_uniform({41: {6: -0.2, 12: -0.1},
                                           42: {6: -0.1, 12: -0.05},
                                           43: {6: -0.3, 12: -0.2}}))
    assert verdict["overall"] == "supported"


def test_primary_sign_disagreement_is_unresolved_and_never_averaged():
    verdict = d4.primary_verdict(_uniform({41: {6: -0.2, 12: -0.1},
                                           42: {6: -0.1, 12: -0.05},
                                           43: {6: +0.4, 12: -0.2}}))
    assert verdict["overall"] == "unsupported"
    assert verdict["per_lead"][6] == "unresolved"


def test_primary_worsened_and_partial_support_are_not_supported():
    worsened = d4.primary_verdict(_uniform({41: {6: 0.2, 12: 0.1},
                                            42: {6: 0.1, 12: 0.05},
                                            43: {6: 0.3, 12: 0.2}}))
    assert worsened["overall"] == "worsened"
    partial = d4.primary_verdict(_uniform({41: {6: -0.2, 12: -0.1},
                                           42: {6: -0.1, 12: +0.01},
                                           43: {6: -0.3, 12: -0.2}}))
    assert partial["overall"] == "unsupported"


def test_gate_fails_on_exactly_one_positive_cell():
    family = _uniform({41: {6: -0.1, 12: -0.1}, 42: {6: -0.1, 12: -0.1}, 43: {6: -0.1, 12: -0.1}},
                      gate_cells={(43, 72, "mslp"): 0.5})
    verdict = d4.gate_verdict(family)
    assert verdict["passed"] is False
    assert len(verdict["failures"]) == 1
    failure = verdict["failures"][0]
    assert (failure["seed"], failure["lead_hours"], failure["variable"]) == (43, 72, "mslp")
    assert failure["relative_mse_change"] == pytest.approx((10.5 ** 2 - 100) / 100)


def test_gate_zero_cells_pass():
    family = _uniform({41: {6: -0.1, 12: -0.1}, 42: {6: -0.1, 12: -0.1}, 43: {6: -0.1, 12: -0.1}},
                      gate_cells={(41, 6, "u10"): 0.0})
    assert d4.gate_verdict(family)["passed"] is True


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
    cells = d4.paired_cells(candidate, control)
    assert cells["6|t2m"]["rmse_delta"] == pytest.approx(-0.5)
    assert cells["6|u10"]["rmse_delta"] == pytest.approx(0.0)
    _write_rmse_table(candidate / "lead_012h" / "rmse.csv",
                      [{"variable": name, "rmse": "8.0" if name == "t2m" else "10.0"}
                       for name in variables])
    cells = d4.paired_cells(candidate, control)
    assert cells["12|t2m"]["rmse_delta"] == pytest.approx(-2.0)


def test_paired_cells_reject_incomplete_tables(tmp_path):
    control = tmp_path / "control"
    for lead in (6, 12, 24, 48, 72):
        _write_rmse_table(control / f"lead_{lead:03d}h" / "rmse.csv",
                          [{"variable": "t2m", "rmse": "10.0"}])
    with pytest.raises(RuntimeError, match="17 variables"):
        d4.paired_cells(control, control)
