"""CI-safe checks for the S3 v3 (five-train-year) study drivers.

The three v3 rounds (D2-analog baselines, D3-analog incumbent control, budget
dose) all hard-code frozen constants and refuse the sealed test split; the
fidelity-critical part is that the D3-analog and dose drivers initialize every
seed through the shared actual-C pairing check rather than trusting labels.
Counterproofs keep the checks honest: a changed mode must change the assembled
contract, a wrong cohort must fail the pin check, and the dose verdicts must
equal the shared registered readings.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d2v3 = _load("study_r7_s3_v3_d2_baselines")
d3v3 = _load("study_r7_s3_v3_d3_incumbent")
bdv3 = _load("study_r7_s3_v3_budget_dose")
rft3 = _load("study_r7_s3_v3_rollout_ft")

from training import r7_s3_v3_screen as screen  # noqa: E402


def test_frozen_constants():
    assert d2v3.EVALUATION_LEADS == (6, 12, 24, 48, 72)
    assert d3v3.SEEDS == (41, 42, 43)
    assert (d3v3.MODE, d3v3.UPDATES, d3v3.STEPS) == ("l6", 400, 4)
    assert d3v3.LAMBDA12 == 0.5
    assert d3v3.PLANNED_SECONDS_ROUND < d3v3.HARD_CAP_SECONDS_ROUND
    assert d3v3.DEADLINE_SECONDS_PER_SEED <= d3v3.HARD_CAP_SECONDS_ROUND
    assert bdv3.CANDIDATE_UPDATES == 1600
    assert bdv3.CONTROL_UPDATES == 400
    assert bdv3.CANDIDATE_MODE == "l6"
    assert bdv3.WARMUP == 80  # 5% of the 1600-update endpoint
    assert bdv3.PRIMARY_LEADS == (6, 12)
    assert bdv3.GATE_TOLERANCE == 0.0


def test_v3_instance_identity_constants_are_pinned():
    assert screen.V3_SOURCE_SHA256 == \
        "bc2ff9cfadcce604fc243bb999b3c430d5164d17fcf1de201273716a5db065f8"
    assert screen.V3_DATA_IDENTITY == \
        "2564eeaf5ac3b9d0bb47670149e6d3e16ecbb55a4c504e0a410d5a840c010cac"
    assert screen.V3_TRAIN_WINDOWS["input_windows"] == 2360
    assert screen.V3_TRAIN_WINDOWS["usable_windows"] == 2340
    assert len(screen.V3_TRAIN_WINDOWS["excluded_sample_ids"]) == 20
    assert screen.V3_CLIMATOLOGY_YEARS == [2017, 2018, 2019, 2020, 2021]
    assert screen.V3_VAL_COHORTS == {"6": 472, "12": 468, "24": 460, "48": 444, "72": 428}


def test_test_split_is_refused():
    with pytest.raises(ValueError, match="sealed"):
        screen.refused_test_manifest(ROOT / "outputs/whatever/test.jsonl")
    screen.refused_test_manifest(ROOT / "outputs/whatever/val.jsonl")
    screen.refused_test_manifest(ROOT / "outputs/whatever/train.jsonl")


def _toy_contract():
    spec = {"architecture": "window", "in_channels": 17, "out_channels": 17,
            "history_steps": 2, "dim": 8, "depth": 1, "heads": 2, "window_size": 4,
            "patch_size": 2, "anchored_processes": 8, "free_processes": 8}
    initialization = {"format": "test-fixture"}
    windows = {"excluded_sample_ids": [], "window_sha256": "0" * 64}
    return screen.contract_for(41, spec, initialization, "ab" * 32, "cd" * 32, windows,
                               source_sha256=screen.V3_SOURCE_SHA256, mode="l6",
                               lambda12=0.5, arm="candidate")


def test_contract_matches_the_archived_actual_c_recipe():
    contract = _toy_contract()
    assert contract["kind"] == "process"
    assert contract["autoregression"]["mode"] == "l6"
    assert contract["autoregression"]["lambda12"] == 0.0
    assert contract["autoregression"]["physical_rollout_bptt"] is True
    assert contract["autoregression"]["internal_reasoning_bptt"] is True
    assert "process_supervision" not in contract
    assert contract["initialization"] == {"format": "test-fixture"}


def test_contract_mode_is_load_bearing():
    spec = {"architecture": "window", "in_channels": 17, "out_channels": 17,
            "history_steps": 2, "dim": 8, "depth": 1, "heads": 2, "window_size": 4,
            "patch_size": 2, "anchored_processes": 8, "free_processes": 8}
    contract = screen.contract_for(41, spec, {"format": "test-fixture"}, "ab" * 32,
                                   "cd" * 32, {"excluded_sample_ids": [], "window_sha256": "0" * 64},
                                   source_sha256=screen.V3_SOURCE_SHA256, mode="two_step",
                                   lambda12=0.5, arm="candidate")
    assert contract["autoregression"]["mode"] == "two_step"
    assert contract["autoregression"]["lambda12"] == 0.5


def test_control_pins_reject_a_missing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(bdv3, "D3_RESULT", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        bdv3.control_pins()


def test_loss_windows_segment_means_for_1600():
    report = {"losses": [{"loss": float(i)} for i in range(1, 1601)]}
    windows = screen.loss_windows(report, size=100)
    assert len(windows) == 16
    assert windows["1-100"] == 50.5
    assert windows["1501-1600"] == 1550.5


def test_paired_cells_reads_and_checks_the_tables(tmp_path):
    import csv
    variables = ["t2m", "u10", "v10", "mslp",
                 "z850", "t850", "q850", "u850", "v850",
                 "z500", "t500", "q500", "u500", "v500",
                 "z250", "u250", "v250"]
    candidate = tmp_path / "candidate" / "lead_006h"
    control = tmp_path / "control" / "lead_006h"
    candidate.mkdir(parents=True)
    control.mkdir(parents=True)
    with (candidate / "rmse.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "rmse"])
        for name in variables:
            writer.writerow([name, 9.0])
    with (control / "rmse.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "rmse"])
        for name in variables:
            writer.writerow([name, 10.0])
    cells = screen.paired_cells(tmp_path / "candidate", tmp_path / "control", (6,))
    assert cells["6|t2m"]["rmse_delta"] == -1.0
    assert abs(cells["6|t2m"]["relative_mse_change"] - (81.0 - 100.0) / 100.0) < 1e-12
    # counterproof: an 8-variable control table must be rejected
    with (control / "rmse.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "rmse"])
        for name in variables[:8]:
            writer.writerow([name, 10.0])
    with pytest.raises(RuntimeError, match="17 variables"):
        screen.paired_cells(tmp_path / "candidate", tmp_path / "control", (6,))


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
    assert bdv3.primary_verdict(cells) == per_seed_sign_verdict(
        cells, seeds=bdv3.SEEDS, primary_leads=bdv3.PRIMARY_LEADS,
        primary_variable=bdv3.PRIMARY_VARIABLE)
    assert bdv3.gate_verdict(cells) == gate_relative_mse_verdict(
        cells, seeds=bdv3.SEEDS, evaluation_leads=bdv3.EVALUATION_LEADS,
        gate_variables=bdv3.GATE_VARIABLES, tolerance=bdv3.GATE_TOLERANCE)
    assert bdv3.advance_decision({"overall": "supported"}, {"passed": True}) == \
        "advance-to-S4-freeze"
    assert bdv3.advance_decision({"overall": "worsened"}, {"passed": True}) == \
        "registered-negative"


def test_rollout_ft_frozen_constants():
    assert rft3.FT_MODE == "two_step"
    assert (rft3.FT_UPDATES, rft3.FT_LAMBDA12, rft3.FT_WARMUP) == (200, 0.5, 10)
    assert rft3.FT_LR == 2e-5
    assert rft3.PARENT_ENDPOINT_UPDATES == 1600
    assert rft3.PARENT_GATE_FAILURES_VS_CONTROL == 17
    assert rft3.PRIMARY_LEADS == (6, 12)
    assert rft3.GATE_TOLERANCE == 0.0
    assert rft3.SEEDS == (41, 42, 43)
    assert rft3.PLANNED_SECONDS_ROUND < rft3.HARD_CAP_SECONDS_ROUND
    # the round must fit the remaining campaign cap at freeze: worst case below 12.0 GPU-h
    assert 9.9238 + rft3.HARD_CAP_SECONDS_ROUND / 3600.0 <= 12.0


def test_rollout_ft_response_text_binds_the_registered_reading():
    assert "17 positive cells" in rft3.ROLLOUT_RESPONSE_TEXT
    assert "never a pass/fail" in rft3.ROLLOUT_RESPONSE_TEXT
    assert "1600" in rft3.PRIMARY_DECISION_TEXT or "rollout_ft" in rft3.PRIMARY_DECISION_TEXT
    assert "relative MSE change" in rft3.GATE_DECISION_TEXT


def test_rollout_ft_parent_pins_reject_missing_and_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(rft3, "PARENT_RUN", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        rft3.pinned_parent(41)
    with pytest.raises(FileNotFoundError):
        rft3.parent_pins()
    fake_run = tmp_path / "parent"
    directory = fake_run / "seed41" / "training" / "candidate"
    directory.mkdir(parents=True)
    (directory / "update_0001600.pt").write_bytes(b"not the registered endpoint")
    (fake_run / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rft3, "PARENT_RUN", fake_run)
    with pytest.raises(RuntimeError, match="drifted"):
        rft3.pinned_parent(41)
    with pytest.raises(RuntimeError, match="drifted"):
        rft3.parent_pins()


def test_rollout_ft_parent_pins_match_the_registered_files():
    result = rft3.PARENT_RUN / "result.json"
    if not result.is_file():
        pytest.skip("registered v3-BD parent is not present on this machine")
    import hashlib
    assert hashlib.sha256(result.read_bytes()).hexdigest() == rft3.PARENT_RESULT_SHA256
    for seed in rft3.SEEDS:
        path, observed = rft3.pinned_parent(seed)
        assert observed == rft3.PARENT_CHECKPOINT_SHA256[seed]


def test_rollout_ft_contract_uses_the_declared_two_step_mode():
    spec = {"architecture": "window", "in_channels": 17, "out_channels": 17,
            "history_steps": 2, "dim": 8, "depth": 1, "heads": 2, "window_size": 4,
            "patch_size": 2, "anchored_processes": 8, "free_processes": 8}
    initialization = {"parent_checkpoint": "registered", "parent_endpoint_updates": 1600}
    contract = screen.contract_for(41, spec, initialization, "ab" * 32, "cd" * 32,
                                   {"excluded_sample_ids": [], "window_sha256": "0" * 64},
                                   source_sha256=screen.V3_SOURCE_SHA256, mode=rft3.FT_MODE,
                                   lambda12=rft3.FT_LAMBDA12, arm="candidate")
    assert contract["autoregression"]["mode"] == "two_step"
    assert contract["autoregression"]["lambda12"] == 0.5
    assert contract["initialization"] == initialization
    # counterproof: the declared constants are load-bearing for the assembled contract
    control = screen.contract_for(41, spec, initialization, "ab" * 32, "cd" * 32,
                                  {"excluded_sample_ids": [], "window_sha256": "0" * 64},
                                  source_sha256=screen.V3_SOURCE_SHA256, mode="l6",
                                  lambda12=rft3.FT_LAMBDA12, arm="candidate")
    assert control["autoregression"]["mode"] == "l6"
    assert control["autoregression"]["lambda12"] == 0.0


def test_rollout_ft_key_windows_require_l12():
    report = {"losses": [{"l12": float(i)} for i in range(1, 101)]}
    windows = rft3._key_windows(report, "l12", size=50)
    assert windows == {"1-50": 25.5, "51-100": 75.5}
    with pytest.raises(RuntimeError, match="l12"):
        rft3._key_windows({"losses": [{"l12": 1.0}, {"l12": None}]}, "l12")
    with pytest.raises(RuntimeError, match="missing"):
        rft3._key_windows({"losses": [{"loss": 1.0}]}, "l12")
