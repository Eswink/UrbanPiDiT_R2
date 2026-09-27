"""Tests for the #67 day-block paired resampling and region stratification.

The load-bearing property is that the resampling unit is the **day**, not the
case: two arms scored on the same init times must produce an interval that
reflects day-level dependence, and a single-day evaluation must refuse to emit
an interval rather than silently falling back to a case-level one (which would
overstate precision). The paired statistic must also refuse mismatched case sets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_r7_67_paired_blocks import (  # noqa: E402
    day_of, paired_day_bootstrap, read_per_case, region_summary,
)


def _write_provenance(directory, cases, channel="t2m"):
    """Write a minimal provenance.json with one lead and one channel."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "channels": [channel],
        "lead_hours": [6],
        "initializations": [
            {"init_time": init_time, "valid_times": [], "mse": [[mse]],
             "cumulative_reasoning_steps": [1]}
            for init_time, mse in cases.items()
        ],
    }
    (directory / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_read_per_case_keys_by_init_time(tmp_path):
    directory = _write_provenance(tmp_path / "a", {
        "2016-01-25T06:00:00": 1.0, "2016-01-25T12:00:00": 4.0})
    cases = read_per_case(directory)
    assert cases == {"2016-01-25T06:00:00": 1.0, "2016-01-25T12:00:00": 4.0}


def test_read_per_case_refuses_a_missing_channel(tmp_path):
    directory = _write_provenance(tmp_path / "a", {"2016-01-25T06:00:00": 1.0})
    with pytest.raises(ValueError, match="absent"):
        read_per_case(directory, channel="q850")


def test_read_per_case_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_per_case(tmp_path / "absent")


def test_day_of_is_the_utc_calendar_day():
    assert day_of("2016-01-25T06:00:00") == "2016-01-25"
    assert day_of("2016-01-25T18:00:00") == "2016-01-25"
    assert day_of("2016-01-26T00:00:00") == "2016-01-26"


def test_paired_difference_is_the_mean_case_delta():
    reference = {"2016-01-25T06:00:00": 4.0, "2016-01-25T12:00:00": 4.0,
                 "2016-01-26T06:00:00": 4.0, "2016-01-26T12:00:00": 4.0}
    arm = dict(reference, **{"2016-01-25T06:00:00": 3.0})  # one case better by 1
    result = paired_day_bootstrap(reference, arm, resamples=200, seed=1)
    assert result["paired"] is True
    assert result["n_cases"] == 4
    assert result["n_days"] == 2
    assert result["difference"] == pytest.approx(-0.25)


def test_interval_reflects_day_blocks_not_cases():
    """With a perfectly consistent day-level effect the interval should be
    tight; the point of the test is that it is formed at all and brackets the
    observed difference."""
    reference = {}
    arm = {}
    for day in ("2016-01-25", "2016-01-26", "2016-01-27"):
        for hour in ("00", "06", "12", "18"):
            stamp = f"{day}T{hour}:00:00"
            reference[stamp] = 4.0
            arm[stamp] = 3.0
    result = paired_day_bootstrap(reference, arm, resamples=500, seed=7)
    assert result["n_days"] == 3
    assert result["difference"] == pytest.approx(-1.0)
    assert result["interval_low"] <= -1.0 <= result["interval_high"]
    assert "day blocks" in result["unit"]


def test_a_single_day_refuses_to_emit_an_interval():
    """A case-level interval over one day would overstate precision, so the
    function returns the difference with an explicit null interval and reason."""
    reference = {"2016-01-25T06:00:00": 4.0, "2016-01-25T12:00:00": 4.0}
    arm = {"2016-01-25T06:00:00": 3.0, "2016-01-25T12:00:00": 3.0}
    result = paired_day_bootstrap(reference, arm, resamples=100, seed=1)
    assert result["paired"] is True
    assert result["n_days"] == 1
    assert result["interval_low"] is None and result["interval_high"] is None
    assert "fewer than two day blocks" in result["reason"]


def test_mismatched_case_sets_refuse_to_produce_a_paired_statistic():
    reference = {"2016-01-25T06:00:00": 4.0, "2016-01-26T06:00:00": 4.0}
    arm = {"2016-01-25T06:00:00": 3.0}  # arm missing a case
    result = paired_day_bootstrap(reference, arm, resamples=100, seed=1)
    assert result["paired"] is False
    assert "undefined" in result["reason"]
    assert "difference" not in result


def test_resampling_is_deterministic_for_a_fixed_seed_and_varies_across_seeds():
    """A constant per-case delta has zero variance, so the seed could not matter;
    the fixture therefore uses day-dependent deltas."""
    reference = {}
    arm = {}
    for position, day in enumerate(("2016-01-25", "2016-01-26", "2016-01-27",
                                    "2016-01-28")):
        for hour in ("00", "06", "12", "18"):
            stamp = f"{day}T{hour}:00:00"
            reference[stamp] = 4.0
            arm[stamp] = 4.0 - 0.2 * (position + 1)
    first = paired_day_bootstrap(reference, arm, resamples=400, seed=11)
    second = paired_day_bootstrap(reference, arm, resamples=400, seed=11)
    assert first == second
    third = paired_day_bootstrap(reference, arm, resamples=400, seed=12)
    assert (third["interval_low"], third["interval_high"]) != \
        (first["interval_low"], first["interval_high"])


def test_region_summary_reports_none_when_no_boundary_csv(tmp_path):
    directory = _write_provenance(tmp_path / "a", {"2016-01-25T06:00:00": 1.0})
    summary = region_summary({"a": directory})
    assert summary["a"] is None


def test_region_summary_reads_regions_for_the_requested_variable(tmp_path):
    directory = _write_provenance(tmp_path / "a", {"2016-01-25T06:00:00": 1.0})
    (directory / "boundary_rmse.csv").write_text(
        "lead_hours,variable,region,margin,rmse,n_cells,unit\n"
        "6,t2m,full,0,2.0,144,K\n"
        "6,t2m,interior_1,1,1.8,100,K\n"
        "6,t2m,edge_1,1,2.6,44,K\n"
        "6,q850,full,0,9.9,144,kg kg**-1\n",
        encoding="utf-8")
    summary = region_summary({"a": directory}, channel="t2m")
    regions = summary["a"]["regions"]
    assert set(regions) == {"full", "interior_1", "edge_1"}
    assert regions["full"]["rmse"] == 2.0
    assert regions["edge_1"]["n_cells"] == 44
    # the q850 rows must not leak into a t2m stratification
    assert all("q850" not in region for region in regions)


def test_region_summary_reports_a_missing_variable_explicitly(tmp_path):
    directory = _write_provenance(tmp_path / "a", {"2016-01-25T06:00:00": 1.0})
    (directory / "boundary_rmse.csv").write_text(
        "lead_hours,variable,region,margin,rmse,n_cells,unit\n"
        "6,q850,full,0,9.9,144,kg kg**-1\n",
        encoding="utf-8")
    summary = region_summary({"a": directory}, channel="t2m")
    assert summary["a"] == {"status": "no boundary rows for this variable"}
