"""Counterproofs for the Stage P2 J1 direction gate; CPU-only, no data, no claim.

J1 is the criterion C1 died on (decision 0094): a constraint that charges the **analysis
field** more than the model has its descent direction pointing away from truth, so it must not
be opened. Three properties are pinned here, and each one caught a real defect in the first
draft of this probe:

1. **Hydrostatic truth must score exactly zero.** The first draft built a "balanced" field
   using only ``t850`` while the residual averaged ``t850`` and ``t500``, so its own truth
   scored 6354 instead of 0 -- a 6.6e-5 ratio that would have read as "the model is far worse
   than truth" for entirely the wrong reason.
2. **An inert term must not pass.** The first draft's moisture row scored ``0.0 <= 0.0`` and
   reported ``passes_j1 = True``. That is precisely C1's humidity defect -- zero violations on
   both sides while carrying weight 0.5 -- so a dead term dressed as a constraint. It is now
   reported as ``inert_on_both_sides`` and refuses to open the direction.
3. **Dose must be a real number to pass.** ``NaN``/``None`` must not satisfy the band test by
   accident of comparison semantics.

The coefficients are checked against the values the plan quotes, so a wrong ``R`` or ``g``
fails here rather than silently rescaling every penalty in the round.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/r7_s3_operator_j1_probe.py")


def _module():
    spec = importlib.util.spec_from_file_location("j1_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


# --- the coefficients the plan quotes ------------------------------------------------

def test_thickness_coefficients_match_the_plan(probe):
    """``plan 0030`` §3 quotes 15.53 and 20.29 m/K; a wrong R or g must not pass silently."""
    assert probe.thickness_coefficient(850.0, 500.0) == pytest.approx(15.53, abs=0.01)
    assert probe.thickness_coefficient(500.0, 250.0) == pytest.approx(20.29, abs=0.01)


def test_thickness_rejects_an_inverted_pair(probe):
    """850 below 500 is not a valid thickness pair."""
    with pytest.raises(ValueError):
        probe.thickness_coefficient(500.0, 850.0)
    with pytest.raises(ValueError):
        probe.thickness_coefficient(500.0, 500.0)


# --- property 1: a balanced field scores exactly zero --------------------------------

def test_a_hydrostatically_balanced_field_scores_exactly_zero(probe):
    """pred == the relation must be exactly 0.0, not approximately.

    The truth is constructed with the *same* mean temperature the residual uses. The first
    draft used a different one and its own truth scored 6354.
    """
    coefficient = probe.thickness_coefficient(850.0, 500.0)
    t850 = torch.full((6, 7), 285.0, dtype=torch.float64)
    t500 = torch.full((6, 7), 275.0, dtype=torch.float64)
    z850 = torch.full((6, 7), 1500.0, dtype=torch.float64)
    z500 = z850 - coefficient * 0.5 * (t850 + t500)
    residual = probe.hydrostatic_residual(z850, z500, t850, t500, 850.0, 500.0)
    assert float(residual.abs().max()) == 0.0


def test_a_thickness_error_is_linear_in_the_height_offset(probe):
    """A 10 m offset must show up as a 10 m residual *relative to* the balanced baseline.

    The baseline itself is not zero here (a 4300 m thickness against zero temperature is a
    large imbalance); the property under test is the *slope*, so the two residuals are
    differenced. Reading the offset as an absolute value was the first draft's mistake.
    """
    t850 = torch.zeros((4, 4), dtype=torch.float64)
    t500 = torch.zeros((4, 4), dtype=torch.float64)
    z850 = torch.zeros((4, 4), dtype=torch.float64)
    z500 = torch.full((4, 4), 4300.0, dtype=torch.float64)
    baseline = float(probe.hydrostatic_residual(
        z850, z500, t850, t500, 850.0, 500.0).mean())
    offset = float(probe.hydrostatic_residual(
        z850, z500 + 10.0, t850, t500, 850.0, 500.0).mean())
    assert offset - baseline == pytest.approx(-10.0)


# --- property 2: an inert term must not pass ----------------------------------------

def test_an_inert_term_does_not_pass_j1(probe):
    """``0.0 <= 0.0`` is arithmetically true and scientifically empty (C1's dead humidity)."""
    verdict = probe.j1_verdict(0.0, 0.0, dose=1e-2)
    assert verdict["inert_on_both_sides"] is True
    assert verdict["passes_j1"] is False
    assert verdict["open_this_direction"] is False


def test_truth_charged_less_passes_j1(probe):
    """The direction the gate is looking for: truth cheap, model expensive."""
    verdict = probe.j1_verdict(1.0, 10.0, dose=1e-2)
    assert verdict["passes_j1"] is True
    assert verdict["ratio_analysis_over_model"] == pytest.approx(0.1)
    assert verdict["open_this_direction"] is True


def test_truth_charged_more_fails_j1(probe):
    """C1's failure mode, at a smaller multiple than the 10.2x it actually measured."""
    verdict = probe.j1_verdict(12.0, 1.0, dose=1e-2)
    assert verdict["passes_j1"] is False
    assert verdict["ratio_analysis_over_model"] == pytest.approx(12.0)
    assert verdict["open_this_direction"] is False


# --- property 3: the dose gate ------------------------------------------------------

def test_dose_outside_the_band_blocks_opening(probe):
    """C2's actual situation: direction right, dose 0.29-0.41, far above the band."""
    verdict = probe.j1_verdict(1.0, 10.0, dose=0.35)
    assert verdict["passes_j1"] is True
    assert verdict["passes_j3"] is False
    assert verdict["open_this_direction"] is False


@pytest.mark.parametrize("dose", [None, float("nan"), float("inf")])
def test_a_non_finite_dose_never_passes(probe, dose):
    """An unmeasured dose must not satisfy the band by comparison accident.

    ``None``/``nan`` are reported back as ``None``; ``inf`` is a finite *number* that simply
    lies outside the band, so it is reported as itself rather than nulled. Either way it must
    not pass.
    """
    verdict = probe.j1_verdict(1.0, 10.0, dose=dose)
    assert verdict["passes_j3"] is False
    if dose is not None and dose == dose:
        assert verdict["dose"] == dose
    else:
        assert verdict["dose"] is None


def test_the_dose_band_bounds_are_inclusive(probe):
    assert probe.j1_verdict(1.0, 10.0, dose=1e-3)["passes_j3"] is True
    assert probe.j1_verdict(1.0, 10.0, dose=1e-1)["passes_j3"] is True
    assert probe.j1_verdict(1.0, 10.0, dose=1e-3 * 0.99)["passes_j3"] is False
    assert probe.j1_verdict(1.0, 10.0, dose=1e-1 * 1.01)["passes_j3"] is False


# --- units come before any penalty ---------------------------------------------------

def test_unit_report_flags_a_geopotential_field_mistaken_for_height(probe):
    """A geopotential (m^2 s^-2) z would be ~9.8x larger; the check must say so."""
    height_like = probe.check_units({"z850": torch.full((4, 4), 1500.0),
                                    "z500": torch.full((4, 4), 5800.0)})
    assert height_like["z_thickness_850_500_m"]["looks_like_geopotential_height"] is True
    assert height_like["z_850_absolute_m"]["looks_like_geopotential_height"] is True

    geopotential_like = probe.check_units({"z850": torch.full((4, 4), 14700.0),
                                          "z500": torch.full((4, 4), 56840.0)})
    assert geopotential_like["z_850_absolute_m"]["looks_like_geopotential_height"] is False
    assert geopotential_like["z_thickness_850_500_m"]["looks_like_geopotential_height"] is False


def test_unit_report_distinguishes_kelvin_from_celsius(probe):
    """285 is Kelvin; 12 is Celsius. The flag must not accept both as Kelvin."""
    kelvin = probe.check_units({"t850": torch.full((4, 4), 285.0)})
    assert kelvin["t_850"]["looks_like_kelvin"] is True
    assert kelvin["t_850"]["looks_like_celsius"] is False
    celsius = probe.check_units({"t850": torch.full((4, 4), 12.0)})
    assert celsius["t_850"]["looks_like_kelvin"] is False


def test_unit_report_rejects_a_normalised_humidity_field(probe):
    """A standardised q has negatives below -0.01; real kg/kg does not."""
    physical = probe.check_units({"q850": torch.full((4, 4), 0.008)})
    assert physical["q_850"]["looks_like_kg_per_kg"] is True
    normalised = probe.check_units({"q850": torch.linspace(-3.0, 3.0, 16).reshape(4, 4)})
    assert normalised["q_850"]["looks_like_kg_per_kg"] is False


# --- the record shape and the refusal surface ----------------------------------------

def test_the_record_is_json_serialisable_without_nan(probe, tmp_path):
    """Strict JSON has no NaN, so every non-finite value must be null before it is written."""
    out = tmp_path / "j1.json"
    assert probe.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    payload = json.loads(text)
    assert payload["scientific_claim"] is False
    assert payload["test_read"] is False
    assert payload["r"] == 0
    for name, row in payload["directions"].items():
        if name == "note":
            continue
        assert row["dose"] is None
        assert row["open_this_direction"] is False


def test_the_synthetic_rows_reproduce_the_expected_directions(probe):
    """A passes, B fails the way C1 failed, D is inert -- on fields built to show each."""
    rows = probe._direction_rows()
    assert rows["A_hydrostatic"]["passes_j1"] is True
    assert rows["A_hydrostatic"]["penalty_analysis"] == 0.0
    assert rows["B_column_divergence"]["passes_j1"] is False
    assert rows["B_column_divergence"]["ratio_analysis_over_model"] > 1.0
    assert rows["D_moisture_negative"]["inert_on_both_sides"] is True
    assert rows["D_moisture_negative"]["passes_j1"] is False


@pytest.mark.parametrize("worker", ["prepare", "archive", "seed"])
def test_training_workers_exit_nonzero(probe, worker):
    """A pre-freeze probe must refuse to create a protocol or an archive."""
    with pytest.raises(SystemExit) as caught:
        probe.main(["--worker", worker])
    assert caught.value.code not in (0, None)


def test_an_unknown_worker_also_exits(probe):
    with pytest.raises(SystemExit):
        probe.main(["--worker", "nonsense"])


def test_the_probe_declares_no_scientific_claim(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source
    assert '"r": 0' in source
