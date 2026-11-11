"""Counterproofs for the Stage R post-hoc amplitude correction; CPU-only, no data, no claim.

The probes here pin the three things that decide the verdict rather than the numbers that
happen to come out:

1. **The multiplier is exactly ``1/amp_ratio``**, fitted in the power domain over train windows.
   A correction with a free coefficient would make the exit unfalsifiable, so the arithmetic is
   pinned.
2. **The exit can fail.** ``degraded`` must require skill to fall *and* RMSE to rise; the smoke
   run on two windows produced exactly that combination at 24/48/72 h, so a test that could only
   report "not degraded" would have let a fabricated pass through (decision 0105).
3. **The metric shapes are the accumulators' own.** An earlier draft squared the RMSE the
   accumulator had already square-rooted and passed a channel count where a name sequence was
   required; both mistakes surface here rather than in a result page.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/r7_s3_amplitude_correction.py")


def _module():
    spec = importlib.util.spec_from_file_location("r_corr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


def _band(ratio):
    return {"amp_ratio": ratio, "psd_pred": ratio ** 2, "psd_target": 1.0}


# --- the fitting arithmetic ------------------------------------------------------------

def test_the_multiplier_inverts_the_power_domain_ratio(probe):
    """Two windows with PSD ratio 4 in band 0 give amp_ratio 2 and multiplier 0.5."""
    windows = [{"bands": [_band(2.0)]}, {"bands": [_band(2.0)]}]
    fitted = probe.band_multipliers(windows)
    assert fitted[0]["amp_ratio"] == pytest.approx(2.0)
    assert fitted[0]["multiplier"] == pytest.approx(0.5)


def test_the_fit_sums_power_rather_than_averaging_ratios(probe):
    """Power-domain aggregation: unequal windows must not be averaged as bare ratios."""
    high = {"bands": [{"amp_ratio": 3.0, "psd_pred": 9.0, "psd_target": 1.0}]}
    low = {"bands": [{"amp_ratio": 1.0, "psd_pred": 0.01, "psd_target": 0.01}]}
    fitted = probe.band_multipliers([high, low])
    # sum(psd_pred)=9.01, sum(psd_target)=1.01 -> ratio sqrt(8.92) ~ 2.987, not mean(3,1)=2
    assert fitted[0]["amp_ratio"] == pytest.approx((9.01 / 1.01) ** 0.5)
    assert fitted[0]["amp_ratio"] > 2.5


def test_an_empty_band_has_no_multiplier(probe):
    windows = [{"bands": [{"amp_ratio": float("nan"), "psd_pred": 0.0, "psd_target": 0.0}]}]
    fitted = probe.band_multipliers(windows)
    assert fitted[0]["multiplier"] is None


# --- the exit must be able to fail ----------------------------------------------------

def _row(lead, skill_before, skill_after, rmse_before, rmse_after):
    return {"lead_hours": lead, "skill_before": skill_before, "skill_after": skill_after,
            "rmse_before": rmse_before, "rmse_after": rmse_after}


def test_the_exit_fails_on_the_smoke_run_shape(probe):
    """Skill down and RMSE up at 24/48/72 is the failure the smoke run actually produced."""
    rows = [_row(24, 0.418, -0.289, 0.171, 0.255),
            _row(48, 0.408, 0.170, 0.241, 0.286),
            _row(72, 0.412, 0.088, 0.208, 0.260)]
    verdict = probe.exit_verdict(rows)
    assert verdict["passed"] is False
    assert verdict["leads_degraded"] == [24, 48, 72]
    assert verdict["leads_mixed"] == []


def test_the_exit_passes_on_an_improvement(probe):
    rows = [_row(24, 0.40, 0.42, 0.20, 0.19),
            _row(48, 0.40, 0.41, 0.21, 0.20),
            _row(72, 0.41, 0.42, 0.22, 0.21)]
    verdict = probe.exit_verdict(rows)
    assert verdict["passed"] is True
    assert verdict["leads_degraded"] == []


def test_a_mixed_lead_is_not_a_pass(probe):
    """Skill up but RMSE up is not "not degraded"; it must be reported separately."""
    rows = [_row(24, 0.40, 0.45, 0.20, 0.21),
            _row(48, 0.40, 0.41, 0.21, 0.20),
            _row(72, 0.41, 0.42, 0.22, 0.21)]
    verdict = probe.exit_verdict(rows)
    assert verdict["passed"] is False
    assert verdict["leads_mixed"] == [24]


def test_only_the_exit_leads_decide(probe):
    """A degradation at 6/12 h must not by itself fail an exit declared on 24/48/72."""
    rows = [_row(6, 0.5, 0.1, 0.1, 0.3),
            _row(12, 0.5, 0.1, 0.1, 0.3),
            _row(24, 0.40, 0.42, 0.20, 0.19),
            _row(48, 0.40, 0.41, 0.21, 0.20),
            _row(72, 0.41, 0.42, 0.22, 0.21)]
    verdict = probe.exit_verdict(rows)
    assert verdict["passed"] is True
    assert verdict["exit_leads"] == [24, 48, 72]


def test_an_empty_table_cannot_pass(probe):
    assert probe.exit_verdict([])["passed"] is False


# --- the correction arithmetic --------------------------------------------------------

def test_the_correction_scales_a_band_by_its_multiplier(probe):
    """Doubling one band's multiplier must double that band's power, not the whole field."""
    from training import r7_spectral_geometry as geometry

    rows = torch.arange(65, dtype=torch.float64)[:, None]
    cols = torch.arange(65, dtype=torch.float64)[None, :]
    field = torch.sin(rows / 3.0) * torch.cos(cols / 5.0)
    edges = geometry.band_edges(65, 65)
    multipliers = {band: {"multiplier": (2.0 if band == 3 else 1.0)}
                   for band in range(len(edges) - 1)}
    corrected = probe.apply_correction(field, multipliers, edges)
    before = geometry.band_statistics(field, field, edges)
    after = geometry.band_statistics(corrected, corrected, edges)
    ratio = after[3]["psd_pred"] / before[3]["psd_pred"]
    assert ratio == pytest.approx(4.0, rel=1e-6)   # amplitude x2 => power x4
    unchanged = after[2]["psd_pred"] / before[2]["psd_pred"]
    assert unchanged == pytest.approx(1.0, rel=1e-6)


def test_a_unit_multiplier_leaves_the_field_unchanged(probe):
    from training import r7_spectral_geometry as geometry

    rows = torch.arange(65, dtype=torch.float64)[:, None]
    cols = torch.arange(65, dtype=torch.float64)[None, :]
    field = torch.sin(rows / 3.0) * torch.cos(cols / 5.0)
    edges = geometry.band_edges(65, 65)
    multipliers = {band: {"multiplier": 1.0} for band in range(len(edges) - 1)}
    corrected = probe.apply_correction(field, multipliers, edges)
    assert torch.allclose(corrected, field, atol=1e-12)


def test_a_missing_multiplier_leaves_the_band_alone(probe):
    from training import r7_spectral_geometry as geometry

    rows = torch.arange(65, dtype=torch.float64)[:, None]
    cols = torch.arange(65, dtype=torch.float64)[None, :]
    field = torch.sin(rows / 3.0) * torch.cos(cols / 5.0)
    edges = geometry.band_edges(65, 65)
    corrected = probe.apply_correction(field, {}, edges)
    assert torch.allclose(corrected, field, atol=1e-12)


# --- the declared constants ------------------------------------------------------------

def test_the_exit_leads_are_the_frozen_three(probe):
    assert probe.EXIT_LEADS == (24, 48, 72)


def test_the_frozen_leads_are_the_five(probe):
    assert probe.FROZEN_LEADS == (6, 12, 24, 48, 72)


def test_the_d2_reference_is_read_not_redefined(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert "D2_REFERENCE" in source
    assert "reference.json" in source


# --- the refusal surface and the standing flags ----------------------------------------

@pytest.mark.parametrize("worker", ["prepare", "archive", "seed"])
def test_training_workers_exit_nonzero(probe, worker):
    with pytest.raises(SystemExit) as caught:
        probe.main(["--worker", worker, "--out", "/tmp/unused.json"])
    assert caught.value.code not in (0, None)


def test_the_probe_never_reads_the_test_split(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TEST_MANIFEST" not in source


def test_the_probe_declares_no_scientific_claim(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source
    assert '"r": 0' in source
