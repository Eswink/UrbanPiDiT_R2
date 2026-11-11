"""Counterproofs for the Stage R feasibility probe; CPU-only, no data read, no claim.

Three properties are pinned here because each one is a way the probe could have produced a
plausible number that means nothing:

1. **The multiplier must be the reciprocal of the measured ratio.** A correction that divides by
   something else is not the correction under discussion, and the sign convention is what decides
   whether an overshooting band is amplified or damped.
2. **An overshooting band must be visible.** ``amp_ratio > 1`` means the multiplier is below 1,
   i.e. the correction reduces amplitude where the model already had too much. If the summary
   failed to count those windows the probe would report "stable" for a band whose correction is
   partly backwards.
3. **The bounded-multiplier test must be able to fail.** A multiplier outside ``[0.2, 5]`` is an
   extrapolation, and a test that cannot report it is not a guard (decision 0105).

The probe itself reads no data in these tests: fixtures stand in for the window set so the
arithmetic and the counting are checked without touching the store.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/r7_s3_amplitude_correction_feasibility.py")


def _module():
    spec = importlib.util.spec_from_file_location("r_feas", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


def _window(band_ratios):
    """One synthetic window whose bands carry the given amplitude ratios."""
    return {"index": 0,
            "bands": [{"band": index, "amp_ratio": ratio, "coherence": 0.9}
                      for index, ratio in enumerate(band_ratios)]}


def test_multiplier_is_the_reciprocal_of_the_mean_ratio(probe):
    report = {"per_window": [_window([0.5, 0.8]), _window([0.5, 0.8])]}
    summary = probe.summarise(report)
    assert summary["per_band"][0]["amp_ratio_mean"] == pytest.approx(0.5)
    assert summary["per_band"][0]["multiplier"] == pytest.approx(2.0)


def test_an_overshooting_band_is_counted(probe):
    """``amp_ratio > 1`` puts the multiplier below 1, and that must be counted."""
    report = {"per_window": [_window([1.2]), _window([0.9]), _window([1.05])]}
    summary = probe.summarise(report)
    band = summary["per_band"][0]
    assert band["overshoot_windows"] == 2
    assert band["multiplier"] == pytest.approx(1.0 / ((1.2 + 0.9 + 1.05) / 3))
    assert summary["overshoot_band_samples"] == 2
    assert summary["windows_total_band_samples"] == 3


def test_no_overshoot_on_a_damped_band(probe):
    report = {"per_window": [_window([0.7]), _window([0.75])]}
    summary = probe.summarise(report)
    assert summary["per_band"][0]["overshoot_windows"] == 0


def test_the_bounded_multiplier_test_can_fail(probe):
    """A ratio so small the multiplier leaves [0.2, 5] must be reported as unbounded."""
    report = {"per_window": [_window([0.05])]}   # multiplier 20
    summary = probe.summarise(report)
    band = summary["per_band"][0]
    assert band["multiplier"] == pytest.approx(20.0)
    assert band["multiplier_bounded"] is False
    assert summary["bands_with_bounded_multiplier"] == 0


def test_a_bounded_multiplier_passes(probe):
    report = {"per_window": [_window([0.8])]}    # multiplier 1.25
    summary = probe.summarise(report)
    assert summary["per_band"][0]["multiplier_bounded"] is True


def test_the_stability_test_can_fail(probe):
    """A band whose ratio swings widely must not be reported as stable."""
    report = {"per_window": [_window([0.3]), _window([0.9])]}
    summary = probe.summarise(report)
    # std of {0.3, 0.9} is 0.3, above the 0.15 threshold
    assert summary["per_band"][0]["amp_ratio_std"] == pytest.approx(0.3)
    assert summary["bands_with_stable_ratio"] == 0


def test_a_stable_band_is_reported_stable(probe):
    report = {"per_window": [_window([0.70]), _window([0.72])]}
    summary = probe.summarise(report)
    assert summary["bands_with_stable_ratio"] == 1


def test_nan_ratios_are_skipped_not_counted_as_zero(probe):
    """An empty target band gives a NaN ratio; it must not enter the mean as 0.0."""
    report = {"per_window": [_window([float("nan")]), _window([0.8])]}
    summary = probe.summarise(report)
    band = summary["per_band"][0]
    assert band["windows"] == 1
    assert band["amp_ratio_mean"] == pytest.approx(0.8)


# --- the refusal surface and the standing flags ----------------------------------------

@pytest.mark.parametrize("worker", ["prepare", "archive", "seed"])
def test_training_workers_exit_nonzero(probe, worker):
    with pytest.raises(SystemExit) as caught:
        probe.main(["--worker", worker, "--windows", "1"])
    assert caught.value.code not in (0, None)


def test_the_probe_never_reads_the_test_split(probe):
    """``r = 0``: the feasibility probe must not name the sealed test manifest."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TEST_MANIFEST" not in source
    assert "TEST_MANIFEST".replace("TEST_MANIFEST", "test.jsonl") not in source


def test_the_probe_declares_no_scientific_claim(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source
    assert '"r": 0' in source


def test_the_d2_reference_is_read_not_redefined(probe):
    """The correction must divide by the registered D2 denominator, not a local constant."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "D2_REFERENCE" in source
    assert "reference.json" in source
