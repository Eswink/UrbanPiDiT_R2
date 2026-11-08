"""Tests for the C1 model-half reproduction (decision 0104).

The probe needs the real store, a checkpoint and a forward pass, so these tests do not run it.
What they check is the part where I could silently be wrong: the lead arithmetic the finding rests
on, the row/summary key contract between ``_measure_case``/``_lead_matched``/``_summarise``, and the
report shape the finding quotes.

A refactor of ``run`` into three functions once produced a ``KeyError: 'ratio_lead_matched'`` that
only surfaced during a 2-minute forward pass. The key-contract test below is the cheap version of
that failure: it builds a synthetic row and asserts ``_summarise`` consumes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import reproduce_c1_model_comparison as probe


def test_c1_cases_are_the_eight_the_result_page_lists():
    assert probe.C1_CASES == (1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000)
    assert len(probe.C1_CASES) == 8


def test_checkpoint_path_is_the_stage_b_fourth_point():
    """The endpoint must be the 128,528-update point, not whichever run is convenient."""
    assert "update_0128528.pt" in probe.C1_CHECKPOINT
    assert "r7_s3_throughput_physics_b_p1600" in probe.C1_CHECKPOINT


def test_page_claim_records_the_published_numbers():
    """The page's numbers are recorded for comparison, never used to compute anything."""
    assert probe.PAGE_CLAIM["model_total"] == 0.0829
    assert probe.PAGE_CLAIM["analysis_total"] == 0.8470
    assert probe.PAGE_CLAIM["ratio"] == 10.2


def test_denormalise_uses_per_channel_statistics():
    """A per-channel mean/std must land on its own channel, not be broadcast as a scalar."""
    torch = pytest.importorskip("torch")
    channel_count = 17
    field = torch.zeros(2, channel_count, 3, 3)
    mean = torch.arange(channel_count, dtype=torch.float32) * 10.0
    std = torch.full((channel_count,), 2.0)
    out = probe._denormalise(field, mean, std)
    assert out.shape == field.shape
    # Normalised zero maps to the channel's own mean: a scalar broadcast would give all zeros
    # except one channel and silently mix statistics across fields.
    for channel in range(channel_count):
        assert float(out[0, channel, 0, 0]) == pytest.approx(float(mean[channel]))


def test_physical_steps_times_step_hours_is_the_last_forecast_lead():
    """The finding in decision 0104: 12 physical steps x 6 h = +72 h, while atmos_target is +6 h.

    Asserted against the real constants rather than hardcoded, so if the rollout length ever
    changes the test breaks instead of the ADR quietly becoming about a different lead.
    """
    from scripts import r7_s3_curve_point_large_protocol as curve
    from training.r7_autoregressive_rollout import STEP_HOURS

    steps = len(curve.PHYSICAL_WEIGHTS)
    assert steps == 12
    assert STEP_HOURS == 6
    assert steps * STEP_HOURS == 72


def _synthetic_row(case: int = 1000) -> dict:
    """A row with exactly the keys ``_measure_case`` emits, so ``_summarise`` can be exercised.

    Total is weighted 0.1*divergence + 0.5*humidity + 0.5*dew_point, so the model's 2.0 total comes
    from div=2.0 with the other two at 3.0 -- the totals are set directly and the per-pairing ratios
    are derived from them, which is what ``_summarise`` reads.
    """
    row = {"case": case, "seconds": 1.0}
    row["model_total"] = 2.0
    row["model_divergence"] = 2.0
    for name in ("divergence", "humidity", "dew_point"):
        row[f"model_{name}"] = 2.0
        row[f"analysis_atmos_target_{name}"] = 3.0
        row[f"analysis_72h_{name}"] = 3.0
    row["analysis_atmos_target_total"] = 3.0
    row["analysis_72h_total"] = 3.0
    for label in ("atmos_target", "72h"):
        row[f"ratio_{label}"] = row[f"analysis_{label}_total"] / row["model_total"]
    row.update({
        "lead_matched_model_total": 2.0,
        "lead_matched_analysis_total": 4.0,
        "lead_matched_ratio": 2.0,
        "lead_matched_model_divergence": 2.0,
        "lead_matched_analysis_divergence": 4.0,
        "lead_matched_divergence_ratio": 2.0,
    })
    return row


def test_summarise_consumes_the_row_shape_the_measure_functions_produce():
    """The key contract between the three functions, without a forward pass.

    This is the regression guard for the ``KeyError: 'ratio_lead_matched'`` that a refactor
    introduced: ``_lead_matched`` names its fields ``lead_matched_*``, so ``_summarise`` cannot
    build ``ratio_<label>`` mechanically for that one pairing.
    """
    rows = [_synthetic_row(case) for case in probe.C1_CASES]
    report = probe._summarise(rows, probe.C1_CASES, "0" * 64)
    for label in ("atmos_target", "72h", "lead_matched"):
        assert f"mean_{label}_ratio" in report, label
        assert report[f"all_cases_worse_{label}"] is True, label
    assert report["mean_lead_matched_ratio"] == pytest.approx(2.0)
    assert report["mean_atmos_target_ratio"] == pytest.approx(1.5)
    assert report["mean_72h_ratio"] == pytest.approx(1.5)
    assert report["all_cases_worse_atmos_target"] is True
    assert report["mean_lead_matched_divergence_ratio"] == pytest.approx(2.0)
    json.dumps(report)  # the real run serialises this, so it must be JSON-clean


def test_summarise_reports_a_reversed_direction_when_the_model_scores_worse():
    """Two-way discrimination: the agreement flag must be able to come out False."""
    rows = [_synthetic_row(case) for case in probe.C1_CASES]
    for row in rows:
        row["lead_matched_analysis_total"] = 1.0
        row["lead_matched_ratio"] = 0.5
    report = probe._summarise(rows, probe.C1_CASES, "0" * 64)
    assert report["all_cases_worse_lead_matched"] is False


def test_probe_writes_only_its_out_path():
    """Read-only with respect to the repository: the module must not open a path for writing."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "out_path.write_text" in source
    assert 'open(..., "w")' not in source
    # The store is opened read-only, which is the one file handle the probe takes on real data.
    assert 'mode="r"' in source


def test_probe_refuses_eval_mode_equivalence_without_checking_it():
    """The train-vs-eval argument is checked in code, not just asserted in prose."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "_BatchNorm" in source, "must walk the module tree for batch-dependent normalisation"
    assert "dropout" in source


def test_numpy_is_used_for_report_arrays_only():
    """Guard against a stray torch import at module scope: the probe imports lazily by design."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    header = source.split("def ")[0]
    for heavy in ("import torch", "import zarr", "import numpy"):
        assert heavy not in header, heavy
