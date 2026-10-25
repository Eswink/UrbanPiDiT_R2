"""CI-safe tests for the S4 FY-2016 confirmation driver.

Everything here is synthetic: no ``outputs/`` artifact, no store, no checkpoint and
no GPU. The tests pin the parts that a wrong implementation would silently get
wrong -- region naming, latitude weighting, the season grouping, the skill
definition, the insufficient-block guard and the strict-JSON sanitizer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "study_r7_s4_fy_2016_confirmation",
    ROOT / "scripts" / "study_r7_s4_fy_2016_confirmation.py")
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def test_region_masks_are_renamed_full_interior_edge():
    masks = driver.region_masks(9, 9, (1,))
    assert [name for name, _, _ in masks] == ["full", "interior", "edge"]
    full = dict((name, mask) for name, _, mask in masks)
    assert int(full["full"].sum()) == 81
    assert int(full["interior"].sum()) == 49
    assert int(full["edge"].sum()) == 32
    assert bool((full["interior"] | full["edge"]).all())


def test_case_mse_is_zero_on_a_perfect_forecast_and_weighted_otherwise():
    lat = torch.linspace(-60, 60, 5, dtype=torch.float64)
    std = np.array([2.0, 3.0], dtype=np.float32)
    masks = driver.region_masks(5, 5, (1,))
    target = torch.zeros(1, 2, 2, 5, 5, dtype=torch.float64)
    perfect = driver.case_mse(target.clone(), target, std, lat, masks)
    for region in ("full", "interior", "edge"):
        assert perfect[region].shape == (2, 2)
        assert np.allclose(perfect[region], 0.0)
    biased = target.clone()
    biased[:, :, 0] = 1.0
    got = driver.case_mse(biased, target, std, lat, masks)
    # channel 0 carries std 2.0, so a unit normalized bias is a 2.0 physical bias.
    assert np.allclose(got["full"][:, 0], 4.0)
    assert np.allclose(got["full"][:, 1], 0.0)


def test_case_mse_weights_by_latitude_cosine():
    lat = torch.tensor([0.0, 60.0], dtype=torch.float64)
    std = np.array([1.0], dtype=np.float32)
    masks = [("full", 0, torch.ones(2, 2, dtype=torch.bool))]
    target = torch.zeros(1, 1, 1, 2, 2, dtype=torch.float64)
    prediction = target.clone()
    prediction[0, 0, 0, 0, :] = 1.0  # only the equator row is wrong
    got = driver.case_mse(prediction, target, std, lat, masks)
    weights = np.array([1.0, 0.5])  # cos(0) and cos(60)
    expected = float((weights * np.array([1.0, 0.0])).sum() / weights.sum())
    assert got["full"][0, 0] == pytest.approx(expected)


def test_group_indices_split_the_year_into_seasons():
    times = ["2016-01-01T00:00:00", "2016-03-01T00:00:00", "2016-06-01T00:00:00",
             "2016-09-01T00:00:00", "2016-12-01T00:00:00"]
    groups = driver.group_indices(times)
    assert set(groups) == {"total set", "declared year 2016", "DJF", "MAM", "JJA", "SON"}
    assert groups["DJF"].tolist() == [0, 4]
    assert groups["MAM"].tolist() == [1]
    assert groups["JJA"].tolist() == [2]
    assert groups["SON"].tolist() == [3]
    assert groups["total set"].tolist() == [0, 1, 2, 3, 4]
    assert groups["declared year 2016"].tolist() == groups["total set"].tolist()


def test_aggregate_uses_the_ratio_of_group_means():
    model = np.array([1.0, 3.0, 5.0, 7.0])
    climatology = np.array([2.0, 2.0, 2.0, 2.0])
    row = driver.aggregate(model, climatology, np.array([0, 1, 2, 3]))
    assert row["mse_model"] == pytest.approx(4.0)
    assert row["skill"] == pytest.approx(1.0 - 4.0 / 2.0)
    assert row["rmse_model"] == pytest.approx(2.0)
    assert row["status"] == "defined"


def test_aggregate_flags_a_nonpositive_climatology_instead_of_dividing():
    row = driver.aggregate(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0, 1]))
    assert row["status"] == "undefined_nonpositive_climatology"
    assert row["skill"] is None


def test_aggregate_flags_an_empty_group_instead_of_returning_nan():
    row = driver.aggregate(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([], dtype=int))
    assert row["status"] == "undefined_empty_group"
    assert row["skill"] is None and row["mse_model"] is None


def test_paired_interval_reports_insufficient_evidence_below_one_block():
    cells = {"6h|t2m": (["a", "b", "c"], np.array([0.1, 0.2, 0.3]))}
    out = driver.paired_interval(cells, block_length=14, resamples=10, seed=1, alpha=0.025)
    assert out["status"] == "insufficient-evidence"
    assert "block length" in out["reason"]


def test_paired_interval_with_no_cells_is_not_a_pass():
    out = driver.paired_interval({}, block_length=2, resamples=10, seed=1, alpha=0.025)
    assert out["status"] == "insufficient-evidence"


def test_sanitize_replaces_non_finite_and_records_the_path():
    report = []
    out = driver.sanitize({"a": [1.0, float("nan")], "b": {"c": float("inf")}}, "", report)
    assert out["a"][0] == 1.0 and out["a"][1] is None and out["b"]["c"] is None
    assert ".a[1]" in report and ".b.c" in report


def test_make_scan_rejects_a_normalization_of_the_wrong_width(monkeypatch):
    class _FakeScan:
        names = ("t2m", "u10")
        mean = np.zeros(2, dtype=np.float32)
        std = np.ones(2, dtype=np.float32)

    monkeypatch.setattr("data.r7_evaluation.RolloutStoreScan", lambda *a, **k: _FakeScan())
    with pytest.raises(ValueError):
        driver.make_scan("unused", history_steps=2, step_hours=6,
                         mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32),
                         allowed_year=2016)


def test_make_scan_rejects_a_nonpositive_std(monkeypatch):
    class _FakeScan:
        names = ("t2m",)
        mean = np.zeros(1, dtype=np.float32)
        std = np.ones(1, dtype=np.float32)

    monkeypatch.setattr("data.r7_evaluation.RolloutStoreScan", lambda *a, **k: _FakeScan())
    with pytest.raises(ValueError):
        driver.make_scan("unused", history_steps=2, step_hours=6,
                         mean=np.zeros(1, dtype=np.float32), std=np.zeros(1, dtype=np.float32),
                         allowed_year=2016)


def test_make_scan_replaces_normalization_and_year_membership(monkeypatch):
    class _FakeScan:
        names = ("t2m", "u10")
        mean = np.zeros(2, dtype=np.float32)
        std = np.ones(2, dtype=np.float32)
        allowed_years = frozenset({2023})

    monkeypatch.setattr("data.r7_evaluation.RolloutStoreScan", lambda *a, **k: _FakeScan())
    scan = driver.make_scan("unused", history_steps=2, step_hours=6,
                            mean=np.array([5.0, 6.0]), std=np.array([2.0, 3.0]),
                            allowed_year=2016)
    assert scan.allowed_years == frozenset({2016})
    assert np.allclose(scan.mean, [5.0, 6.0]) and np.allclose(scan.std, [2.0, 3.0])


def test_load_protocol_rejects_a_foreign_format(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        driver.load_protocol(path)
