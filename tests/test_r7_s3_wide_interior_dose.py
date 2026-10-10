"""CI-safe checks for the wide interior-supervision dose driver (800 -> 2400 updates).

The dose round changes exactly one factor against the registered 800-update
interior arm: the update budget. The fidelity-critical parts are that the frozen
supervision mask is unchanged, that the two pinned references are the registered
narrow arm and the registered 800-update interior arm (neither retrained), that
the pinned 800 arm's own tables still match their freeze pins, and that the
driver refuses an existing output or a missing store.
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


dose = _load("study_r7_s3_wide_interior_dose")
interior = dose.interior
wide = dose.wide


def test_only_the_update_budget_changed():
    assert dose.FORMAT == "r7-s3-wide-interior-dose-protocol-v1"
    assert dose.UPDATES == 2400
    assert wide.UPDATES == 800  # the registered arm's dose is untouched
    assert (wide.SEEDS, wide.LR, wide.WARMUP) == ((41, 42, 43), 2e-5, 10)
    assert wide.PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert wide.BOUNDARY_MARGINS == (32,)
    assert dose.PER_SEED_SECONDS > wide.PER_SEED_SECONDS
    assert dose.PLANNED_SECONDS < dose.HARD_CAP_SECONDS
    assert dose.PER_SEED_SECONDS <= dose.HARD_CAP_SECONDS


def test_protocol_constants_carry_the_dose_and_the_frozen_supervision():
    body = dose.protocol_constants()
    assert body["format"] == dose.FORMAT
    assert body["recipe"]["updates"] == 2400
    assert body["supervision"] == interior.supervision_block()
    assert body["supervision"]["runner_block"]["selected_cells"] == 4225
    assert body["budgets"]["per_seed_seconds"] == dose.PER_SEED_SECONDS
    # the other recipe factors must be the registered ones, unchanged
    assert body["recipe"]["lr"] == 2e-5 and body["recipe"]["warmup"] == 10
    assert body["recipe"]["physical_weights"] == [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5]
    assert body["recipe"]["steps"] == 4 and body["recipe"]["bf16"] is False
    assert body["seeds"] == [41, 42, 43]
    assert body["scientific_claim"] is False and body["test_read"] is False


def test_expected_candidate_is_the_registered_recipe_at_the_new_dose():
    candidate = dose._expected_candidate()
    assert candidate["updates"] == 2400
    assert candidate["mode"] == "long_rollout" and candidate["steps"] == 4
    assert candidate["model"]["kind"] == "process"


def test_the_pinned_800_update_arm_is_the_registered_record():
    pins = dose.previous_arm_pins()
    assert pins["record_id"] == "s3-wide-interior-supervision"
    assert pins["updates"] == 800
    assert pins["seeds"] == [41, 42, 43]
    for seed in (41, 42, 43):
        assert len(pins["seeds_evaluations"][str(seed)]["leads"]) == 5
        assert pins["seeds_evaluations"][str(seed)]["checkpoint_sha256"]


def test_the_pinned_800_arm_tables_still_match_their_freeze_pins():
    from training.r7_arm_harness import sha256_file

    pins = dose.previous_arm_pins()
    for seed in (41, 42, 43):
        for lead in (6, 12, 24, 48, 72):
            entry = pins["seeds_evaluations"][str(seed)]["leads"][str(lead)]
            assert sha256_file(Path(entry["dir"]) / "rmse.csv") == entry["rmse_csv_sha256"]


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        dose.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        dose.main(["--out", str(out)])
    assert not out.exists()
