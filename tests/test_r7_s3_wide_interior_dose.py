"""CI-safe checks for the wide interior-supervision dose driver (800 -> 2400 updates).

The dose round changes exactly one factor against the registered 800-update
interior arm: the update budget. The fidelity-critical parts are that the frozen
supervision mask is unchanged, that the pinned references are the registered
narrow arm and the registered 800-update interior arm (neither retrained), and
that the 800 arm's pinned table is the *boundary* table this driver actually
reads -- storing its ``rmse.csv`` hash instead made the frozen reading stage
raise after every seed had already trained. The pin logic is exercised against a
synthetic output tree so it is checked on a checkout with no ``outputs/``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIABLES = ("t2m", "u10", "v10", "mslp", "tcc", "tcwv", "sp", "z500", "t500", "u500",
             "v500", "q500", "z850", "t850", "u850", "v850", "r850")


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dose = _load("study_r7_s3_wide_interior_dose")
interior = dose.interior
wide = dose.wide

COHORTS = ((6, 472), (12, 468), (24, 460), (48, 444), (72, 428))


def _rmse_csv():
    lines = ["lead_hours,variable,rmse,unit,n_initializations"]
    for variable in VARIABLES:
        lines.append(f"6,{variable},1.0,K,472")
    return "\n".join(lines) + "\n"


def _boundary_csv():
    lines = ["region,variable,rmse,unit,n_initializations"]
    for region in ("full", "interior_32", "edge_32"):
        for variable in VARIABLES:
            lines.append(f"{region},{variable},1.0,K,472")
    return "\n".join(lines) + "\n"


def _synthetic_run(tmp_path):
    """Minimal stand-in for the registered 800-update interior arm's output tree."""
    from training.r7_arm_harness import sha256_file

    run = tmp_path / "prev"
    for seed in (41, 42, 43):
        evaluations = {}
        for lead, cohort in COHORTS:
            folder = run / f"seed{seed}" / f"lead_{lead:03d}h"
            folder.mkdir(parents=True)
            (folder / "rmse.csv").write_text(_rmse_csv(), encoding="utf-8")
            (folder / "boundary_rmse.csv").write_text(_boundary_csv(), encoding="utf-8")
            evaluations[str(lead)] = {
                "dir": str(folder), "n_evaluated": cohort, "n_available_windows": cohort,
                "rmse_csv_sha256": sha256_file(folder / "rmse.csv"),
                "boundary_rmse_csv_sha256": sha256_file(folder / "boundary_rmse.csv")}
        (run / f"seed{seed}_receipt.json").write_text(
            json.dumps({"checkpoint_sha256": f"{seed:064d}", "evaluations": evaluations}),
            encoding="utf-8")
    return run


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


def test_the_candidate_dose_is_the_new_one_and_the_reference_is_not():
    # The dose must appear in the frozen candidate recipe while the pinned
    # references keep the registered 800-update budget. Asserted from the
    # constants rather than _expected_candidate() so this holds on a checkout
    # with no outputs/ (that helper reads the archived protocol there).
    body = dose.protocol_constants()
    assert dose.UPDATES == 2400 and wide.UPDATES == 800
    assert body["recipe"]["updates"] == dose.UPDATES
    assert body["recipe"]["updates"] != wide.UPDATES
    assert "800 -> 2400" in body["design_decision"]
    assert "tripling the update budget" in body["hypothesis"]


def test_previous_arm_pins_read_a_registered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dose, "PREVIOUS_ARM_RUN", _synthetic_run(tmp_path))
    pins = dose.previous_arm_pins()
    assert pins["record_id"] == "s3-wide-interior-supervision"
    assert pins["updates"] == 800
    assert pins["seeds"] == [41, 42, 43]
    assert len(pins["seeds_evaluations"]) == 3
    for seed in (41, 42, 43):
        leads = pins["seeds_evaluations"][str(seed)]["leads"]
        assert len(leads) == 5
        for lead in ("6", "12", "24", "48", "72"):
            entry = leads[lead]
            # the pinned hash must be the *boundary* table the reading stage reads,
            # not rmse.csv (pinning the wrong one broke a full attempt)
            assert entry["boundary_rmse_csv_sha256"] != entry.get("rmse_csv_sha256")
            from training.r7_arm_harness import sha256_file
            assert sha256_file(Path(entry["dir"]) / "boundary_rmse.csv") \
                == entry["boundary_rmse_csv_sha256"]
            assert sha256_file(Path(entry["dir"]) / "rmse.csv") \
                != entry["boundary_rmse_csv_sha256"]


def test_previous_arm_pins_refuse_a_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dose, "PREVIOUS_ARM_RUN", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        dose.previous_arm_pins()


def test_previous_arm_pins_refuse_a_cohort_that_differs_from_the_pin(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path)
    receipt_path = run / "seed42_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evaluations"]["72"]["n_evaluated"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(dose, "PREVIOUS_ARM_RUN", run)
    with pytest.raises(RuntimeError):
        dose.previous_arm_pins()


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        dose.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        dose.main(["--out", str(out)])
    assert not out.exists()
