"""Offline tests for the M2 bucket comparison driver (#69).

The defect these tests exist for: the earlier per-arm tables wrote a physical
forecast RMSE next to a **normalized** climatology RMSE under the same unit
label, and a "9.09x" ratio was published from it (docs/decisions/0010-*). A
comparison driver that can silently do that again is worse than no driver, so the
unit agreement between the two files it reads is a hard pre-condition here, and
the unit-free skill sign is cross-checked against the physical comparison.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.study_r7_69_bucket_comparison as module
from scripts.study_r7_69_bucket_comparison import (
    ARMS,
    BASELINES,
    LEADS,
    compare_against_climatology,
)


def _write_metrics(run_dir, *, forecast=2.0, climatology=2.5, unit="K",
                   skill_unit=None, mse_skill=None, variable="t2m"):
    """One evaluation directory's two CSVs, as r7_evaluate writes them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "rmse.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse", "unit", "n_initializations"])
        writer.writerow([6, variable, forecast, unit, 10])
    if mse_skill is None:
        mse_skill = 1.0 - (forecast ** 2) / (climatology ** 2)
    with (run_dir / "climatology_skill.csv").open("w", encoding="utf-8",
                                                  newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse_forecast", "rmse_climatology",
                         "mse_skill", "unit", "n_initializations"])
        writer.writerow([6, variable, forecast, climatology, mse_skill,
                         skill_unit or unit, 10])
    return run_dir


def _entry(run_dir, *, arm="generic", seed=41, lead=6):
    return {"arm": arm, "seed": seed, "lead_hours": lead, "n_evaluated": 10,
            "units": ["K"], "rmse_csv": str(run_dir / "rmse.csv"),
            "skill_csv": str(run_dir / "climatology_skill.csv")}


def test_reads_physical_metrics_and_reports_the_ratio(tmp_path):
    run_dir = _write_metrics(tmp_path / "run", forecast=2.0, climatology=2.5)
    metrics = module._physical(_entry(run_dir))
    values = metrics["t2m"]
    assert values["forecast"] == 2.0 and values["climatology"] == 2.5
    assert values["ratio"] == pytest.approx(0.8)
    assert values["mse_skill"] == pytest.approx(1.0 - 4.0 / 6.25)


def test_refuses_a_unit_label_mismatch_between_the_two_files(tmp_path):
    """The exact defect: same directory, two files, two different unit labels."""
    run_dir = _write_metrics(tmp_path / "run", unit="K", skill_unit="normalized")
    with pytest.raises(ValueError, match="refusing to compare across unit labels"):
        module._physical(_entry(run_dir))


def test_refuses_normalized_metrics_for_a_physical_claim(tmp_path):
    run_dir = _write_metrics(tmp_path / "run", unit="normalized")
    with pytest.raises(ValueError, match="cannot be made from them"):
        module._physical(_entry(run_dir))


def test_skill_sign_and_physical_comparison_must_agree(tmp_path):
    """A mixed-unit table shows up as this disagreement, so it must be fatal.

    Constructed by hand: the physical RMSE says the forecast wins (2.0 < 2.5) but
    the recorded skill is negative, which can only happen if the baseline was
    computed on a different scale than the forecast.
    """
    run_dir = _write_metrics(tmp_path / "run", forecast=2.0, climatology=2.5,
                             mse_skill=-0.5)
    results = {"evaluation": {"generic_s41@6h": _entry(run_dir)}}
    with pytest.raises(RuntimeError, match="mixed-unit"):
        compare_against_climatology(results)


def test_counts_better_and_worse_per_lead(tmp_path):
    wins = _write_metrics(tmp_path / "win", forecast=1.0, climatology=2.0)
    losses = _write_metrics(tmp_path / "loss", forecast=3.0, climatology=2.0)
    results = {"evaluation": {
        "generic_s41@6h": _entry(wins, seed=41),
        "process_s41@6h": _entry(losses, arm="process", seed=41),
    }}
    comparison = compare_against_climatology(results)
    assert comparison["totals"] == {"better": 1, "worse": 1, "unresolved": 0}
    assert comparison["per_lead"]["6"]["better"] == 1
    assert comparison["per_lead"]["6"]["worse"] == 1
    assert comparison["per_variable"]["t2m"] == {"better": 1, "worse": 1,
                                                 "unresolved": 0}


def test_parameter_free_controls_are_not_counted_as_arms(tmp_path):
    """Climatology cannot beat itself; counting it would inflate the totals."""
    same = _write_metrics(tmp_path / "clim", forecast=2.0, climatology=2.0)
    results = {"evaluation": {"climatology_s41@6h": _entry(same, arm="climatology")}}
    comparison = compare_against_climatology(results)
    assert comparison["totals"] == {"better": 0, "worse": 0, "unresolved": 0}


def test_declared_arm_and_lead_sets_are_complete():
    assert set(ARMS) == {"unet", "native_window", "afno_small", "generic", "process"}
    assert set(BASELINES) == {"persistence", "climatology"}
    assert LEADS == (6, 12, 24, 48, 72)


def test_discover_arms_refuses_a_partial_arm_set(tmp_path):
    """A comparison missing an arm cannot answer the question and must not run."""
    train_root = tmp_path / "multiseed"
    for arm in ("unet", "generic"):
        run = train_root / "seed41" / arm
        run.mkdir(parents=True)
        checkpoint = run / "update_0000100.pt"
        checkpoint.write_bytes(b"weights")
        (run / "training_report.json").write_text(json.dumps({
            "selected_checkpoint": str(checkpoint), "selected_update": 100,
        }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing declared arms"):
        module.discover_arms(train_root)


def test_discover_arms_reports_a_missing_checkpoint_file(tmp_path):
    train_root = tmp_path / "multiseed"
    run = train_root / "seed41" / "unet"
    run.mkdir(parents=True)
    (run / "training_report.json").write_text(json.dumps({
        "selected_checkpoint": str(run / "absent.pt"), "selected_update": 100,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no existing selected checkpoint"):
        module.discover_arms(train_root)
