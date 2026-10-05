"""CI-safe checks for the S3 budget-curve screen's registered reading.

The reading is the load-bearing part of this round: the verdict functions must
actually implement the frozen decision text (the second declared dose of the
update-budget factor), so the counterproofs mirror the UB round's while pinning
this round's own frozen constants (1600 updates, warmup 80, dose statement).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_driver():
    path = ROOT / "scripts" / "study_r7_s3_budget_curve.py"
    spec = importlib.util.spec_from_file_location("study_r7_s3_budget_curve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bc = _load_driver()


def test_frozen_constants():
    assert bc.CANDIDATE_UPDATES == 1600
    assert bc.CONTROL_UPDATES == 400
    assert bc.CANDIDATE_MODE == "l6"
    assert bc.WARMUP == 80  # 5% of the 1600-update endpoint
    assert bc.PRIMARY_LEADS == (6, 12)
    assert bc.GATE_TOLERANCE == 0.0
    assert bc.SEEDS == (41, 42, 43)


def test_test_split_is_refused():
    import pytest
    with pytest.raises(ValueError, match="sealed"):
        bc.refused_test_manifest("/somewhere/test.jsonl")
    bc.refused_test_manifest("/somewhere/val.jsonl")


def test_budget_response_text_declares_the_final_dose():
    assert "final declared budget dose" in bc.BUDGET_RESPONSE_TEXT
    protocol_text = bc.PRIMARY_DECISION_TEXT
    assert "400 -> 800 -> 1600" in protocol_text


def test_loss_windows_segment_means_for_1600():
    report = {"losses": [{"loss": float(i)} for i in range(1, 1601)]}
    windows = bc.loss_windows(report, size=100)
    assert len(windows) == 16
    assert windows["1-100"] == 50.5
    assert windows["1501-1600"] == 1550.5


def test_verdict_bindings_match_the_shared_implementation():
    from training.r7_verdict_readings import (advance_decision as shared_advance,
                                              gate_relative_mse_verdict,
                                              per_seed_sign_verdict)
    cells = {}
    for seed in (41, 42, 43):
        seed_cells = {}
        for lead in (6, 12):
            seed_cells[f"{lead}|t2m"] = {"lead_hours": lead, "variable": "t2m",
                                         "candidate_rmse": 9.0, "control_rmse": 10.0,
                                         "rmse_delta": -1.0, "relative_mse_change": -0.19}
        for lead in (6, 12, 24, 48, 72):
            for variable in ("u10", "v10", "mslp"):
                seed_cells[f"{lead}|{variable}"] = {"lead_hours": lead, "variable": variable,
                                                    "candidate_rmse": 9.5, "control_rmse": 10.0,
                                                    "rmse_delta": -0.5,
                                                    "relative_mse_change": -0.0975}
        cells[str(seed)] = seed_cells
    assert bc.primary_verdict(cells) == per_seed_sign_verdict(
        cells, seeds=bc.SEEDS, primary_leads=bc.PRIMARY_LEADS,
        primary_variable=bc.PRIMARY_VARIABLE)
    assert bc.gate_verdict(cells) == gate_relative_mse_verdict(
        cells, seeds=bc.SEEDS, evaluation_leads=bc.EVALUATION_LEADS,
        gate_variables=bc.GATE_VARIABLES, tolerance=bc.GATE_TOLERANCE)
    assert bc.advance_decision({"overall": "supported"}, {"passed": True}) == "advance-to-S4-freeze"
    assert bc.advance_decision({"overall": "supported"}, {"passed": False}) == "registered-negative"
