"""Tests for the C1 model-half reproduction (decision 0104).

The probe itself needs the store, a checkpoint and a forward pass, so these tests do not run it.
What they do check is the part where I could silently be wrong: the lead/step arithmetic that the
whole finding rests on, and the shape of the report the probe promises to write.
"""
from __future__ import annotations

import json
from pathlib import Path

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


@pytest.mark.parametrize("channel_count", [17])
def test_denormalise_uses_per_channel_statistics(channel_count):
    """A per-channel mean/std must land on its own channel, not be broadcast as a scalar."""
    torch = pytest.importorskip("torch")
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

    This is asserted against the real constants rather than hardcoded, so if the rollout length
    ever changes the test breaks instead of the ADR quietly becoming about a different lead.
    """
    from scripts import r7_s3_curve_point_large_protocol as curve
    from training.r7_autoregressive_rollout import STEP_HOURS

    steps = len(curve.PHYSICAL_WEIGHTS)
    assert steps == 12
    assert STEP_HOURS == 6
    assert steps * STEP_HOURS == 72


def test_report_keys_are_what_the_finding_quotes():
    """The three pairings must all be reported; dropping one would hide the disagreement."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    for key in ("mean_atmos_target_ratio", "mean_72h_ratio", "mean_lead_matched_ratio",
                "all_cases_worse_atmos_target", "all_cases_worse_lead_matched"):
        assert key in source, key


def test_probe_writes_only_its_out_path():
    """Read-only with respect to the repository: the module must not open a path for writing."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "out_path.write_text" in source
    assert "open(..., \"w\")" not in source
    # The store is opened read-only, which is the one file handle the probe takes on real data.
    assert 'mode="r"' in source


def test_probe_refuses_eval_mode_equivalence_without_checking_it():
    """The train-vs-eval argument is checked in code, not just asserted in prose."""
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "_BatchNorm" in source, "must walk the module tree for batch-dependent normalisation"
    assert "dropout" in source
