"""CI-safe checks for the long-lead physical-weight rebalance driver.

The round changes exactly one factor against the registered 2400-update interior
arm: the physical weights on the 48 h and 72 h steps, 0.5 -> 1.0. The
fidelity-critical parts are that only those two weights move, that the 2400
budget and the interior_32 mask are unchanged, that the pinned 2400 reference is
read from the boundary table it is scored on, and that the driver refuses an
existing output or a missing store. Everything is asserted against constants or
a synthetic output tree, so it holds on a checkout with no ``outputs/``.
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
COHORTS = ((6, 472), (12, 468), (24, 460), (48, 444), (72, 428))


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llw = _load("study_r7_s3_wide_long_lead_weight")
dose = llw.dose
interior = llw.interior
wide = llw.wide


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
    from training.r7_arm_harness import sha256_file

    run = tmp_path / "prev2400"
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


def test_only_the_two_long_lead_weights_changed():
    assert llw.FORMAT == "r7-s3-wide-long-lead-weight-protocol-v1"
    assert llw.UPDATES == 2400 == dose.UPDATES
    assert llw.REGISTERED_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert wide.PHYSICAL_WEIGHTS == llw.REGISTERED_WEIGHTS
    assert llw.PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., 1., 0., 0., 0., 1.)
    # exactly the two long-lead non-zero slots moved, everything else identical
    changed = [i for i, (new, old) in enumerate(zip(llw.PHYSICAL_WEIGHTS, llw.REGISTERED_WEIGHTS))
               if new != old]
    assert changed == [7, 11]
    assert all(new == 1.0 and old == 0.5
               for new, old in zip(llw.PHYSICAL_WEIGHTS, llw.REGISTERED_WEIGHTS) if new != old)


def test_protocol_constants_carry_the_weights_and_the_frozen_factors():
    body = llw.protocol_constants()
    assert body["format"] == llw.FORMAT
    assert body["recipe"]["physical_weights"] == list(llw.PHYSICAL_WEIGHTS)
    assert body["recipe"]["updates"] == 2400
    assert body["supervision"] == interior.supervision_block()
    assert body["supervision"]["runner_block"]["selected_cells"] == 4225
    assert body["recipe"]["lr"] == 2e-5 and body["recipe"]["warmup"] == 10
    assert body["recipe"]["steps"] == 4 and body["recipe"]["bf16"] is False
    assert body["seeds"] == [41, 42, 43]
    assert body["scientific_claim"] is False and body["test_read"] is False


def test_previous_arm_pins_read_a_registered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llw, "PREVIOUS_ARM_RUN", _synthetic_run(tmp_path))
    pins = llw.previous_arm_pins()
    assert pins["record_id"] == "s3-wide-interior-dose"
    assert pins["updates"] == 2400
    assert len(pins["seeds_evaluations"]) == 3
    from training.r7_arm_harness import sha256_file
    for seed in (41, 42, 43):
        leads = pins["seeds_evaluations"][str(seed)]["leads"]
        assert len(leads) == 5
        for lead in ("6", "12", "24", "48", "72"):
            entry = leads[lead]
            assert sha256_file(Path(entry["dir"]) / "boundary_rmse.csv") \
                == entry["boundary_rmse_csv_sha256"]
            assert sha256_file(Path(entry["dir"]) / "rmse.csv") \
                != entry["boundary_rmse_csv_sha256"]


def test_previous_arm_pins_refuse_a_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llw, "PREVIOUS_ARM_RUN", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        llw.previous_arm_pins()


def test_previous_arm_pins_refuse_a_cohort_that_differs_from_the_pin(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path)
    receipt_path = run / "seed43_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evaluations"]["48"]["n_evaluated"] = 999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(llw, "PREVIOUS_ARM_RUN", run)
    with pytest.raises(RuntimeError):
        llw.previous_arm_pins()


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        llw.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        llw.main(["--out", str(out)])
    assert not out.exists()
