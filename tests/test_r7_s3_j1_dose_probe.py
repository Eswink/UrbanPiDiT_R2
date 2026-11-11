"""Counterproofs for the J3 dose probe; CPU-only, no claim.

The defect these tests exist for is the one that shipped first and produced a number four
orders of magnitude too large: the per-channel normalisation constants are shaped
``(C, 1, 1)``, so applying them to a bare ``[H, W]`` slice broadcasts into ``[17, H, W]`` and
every "channel" becomes a blend dominated by ``mslp`` (~101635) and ``z250`` (~104149). The
divergence penalty then reads 1.5e6 instead of 5.3, which is the difference between "far
outside the dose band" and "absurdly outside it" -- both fail, but only one of them is a
measurement.

``test_a_sliced_channel_is_caught`` reproduces the failure mode directly, so a future refactor
that reintroduces the slice fails here rather than in a result page.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts/r7_s3_j1_dose_probe.py")


def _module():
    spec = importlib.util.spec_from_file_location("dose_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


# --- the shipped defect ---------------------------------------------------------------

def test_a_sliced_channel_is_caught(probe):
    """The ``[H, W]`` slice + ``(C, 1, 1)`` constants broadcast blend must be refused.

    This is the exact shape of the bug: the fields below are what ``physical(fields[index])``
    produces for a 17-channel block, and the guard must reject them rather than let a squared
    constraint be computed on them.
    """
    channel_count = 17
    fields = {}
    for index, name in enumerate(("t2m", "u850", "v850", "mslp", "z250")):
        # A blend: each "channel" is the same 65x65 grid scaled by every channel's offset.
        blend = torch.full((65, 65), 101635.0 + 104149.0 * index + 286.0)
        fields[name] = blend
    with pytest.raises(ValueError, match="t2m averages"):
        probe._assert_physical_units(fields, torch.zeros(1), {})


def test_a_plausible_field_passes_the_guard(probe):
    """The same guard must accept a real-looking physical block."""
    rows = torch.arange(65, dtype=torch.float64)[:, None].expand(65, 65)
    fields = {"t2m": 273.15 + 0.0 * rows, "u850": 1.0 + 0.01 * rows}
    probe._assert_physical_units(fields, torch.zeros(1), {})


def test_a_blended_wind_field_is_caught(probe):
    """``t2m`` can be right while the wind is a blend; the guard checks both."""
    rows = torch.arange(65, dtype=torch.float64)[:, None].expand(65, 65)
    fields = {"t2m": 273.15 + 0.0 * rows, "u850": 5000.0 + 0.0 * rows}
    with pytest.raises(ValueError, match="u850"):
        probe._assert_physical_units(fields, torch.zeros(1), {})


# --- the dose arithmetic ---------------------------------------------------------------

def test_dose_is_the_ratio_of_penalty_to_l_mse(probe):
    verdict = probe.dose_verdict(penalty=5.0, l_mse=0.5)
    assert verdict["dose"] == pytest.approx(10.0)
    assert verdict["in_band"] is False


def test_a_dose_inside_the_frozen_band_passes(probe):
    verdict = probe.dose_verdict(penalty=0.05, l_mse=1.0)
    assert verdict["dose"] == pytest.approx(0.05)
    assert verdict["in_band"] is True


def test_the_band_edges_are_inclusive(probe):
    assert probe.dose_verdict(1e-3, 1.0)["in_band"] is True
    assert probe.dose_verdict(1e-1, 1.0)["in_band"] is True
    assert probe.dose_verdict(9.99e-4, 1.0)["in_band"] is False
    assert probe.dose_verdict(1.01e-1, 1.0)["in_band"] is False


def test_a_non_positive_denominator_yields_no_dose(probe):
    """A zero loss would make the ratio infinite; that is reported, never divided through."""
    for bad in (0.0, -1.0, float("nan")):
        verdict = probe.dose_verdict(penalty=1.0, l_mse=bad)
        assert verdict["dose"] is None
        assert verdict["in_band"] is False


def test_the_band_is_the_frozen_one(probe):
    """J3's band comes from plan 0029 section 2; this script may read it but not redefine it."""
    assert probe.DOSE_BAND == (1e-3, 1e-1)


# --- the undoseable direction is reported, not patched --------------------------------

def test_hydrostatic_has_no_declared_scale(probe):
    """Hydrostatic is in m^2 and nothing declares a scale, so its dose must stay null.

    Dividing a mean square in metres by a normalised MSE would mix units, and inventing a
    height scale after seeing the number is the fitted-parameter move J3 exists to prevent.
    """
    import re

    source = SCRIPT.read_text(encoding="utf-8")
    assert "hydrostatic" in source
    # No hydrostatic scale constant may appear: adding one would be a new, unfrozen parameter.
    assert not re.search(r"HYDROSTATIC_SCALE\s*=", source)
    assert not re.search(r"HEIGHT_SCALE\s*=", source)


# --- the refusal surface --------------------------------------------------------------

@pytest.mark.parametrize("worker", ["prepare", "archive", "seed"])
def test_training_workers_exit_nonzero(probe, worker):
    with pytest.raises(SystemExit) as caught:
        probe.main(["--worker", worker, "--cases", "1000"])
    assert caught.value.code not in (0, None)


def test_the_probe_declares_no_scientific_claim(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source
    assert '"r": 0' in source
