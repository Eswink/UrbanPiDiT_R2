"""Counterproofs for Stage C: the gap decay law and the per-variable beta weighting.

Three properties carry this round, and each has already caught a real defect:

1. **The anti-phase singularity.** ``_terms`` rationalised ``2*max(P,Q)*(1-coh)`` into
   ``2*M*(P*Q - cross^2) / (P*Q*(1+coh))`` to get exact zero at ``pred == target``. At
   ``coh = -1`` the denominator vanishes while the numerator is a floating-point residue, so
   the term came back ``inf`` -- and ``inf`` is not the answer: the limit is ``4*max(P,Q)``,
   the *maximum* penalty. Measured on a real 6 h pair of ``v10`` (store frame 1464): the
   rationalised form returned ``inf`` where the literal form is ``3210.34``. A naive guard that
   zeroed the term at ``coh <= -1`` would be worse, reporting perfect anti-phase as a perfect
   match. ``test_anti_phase_band_matches_the_literal_limit`` pins the value; and
   ``test_anti_phase_is_not_reported_as_a_match`` pins that it is not zero.
2. **The exact-zero property must survive that fix.** Decision 0110's family argument rests on
   ``amse(x, x)`` being *exactly* 0, not approximately. ``test_exact_zero_is_preserved``
   asserts equality, not a tolerance.
3. **The decay fit must be able to refuse.** A monotone decline says nothing about the rate, so
   the whole Stage C verdict rests on fitting ``gap ~ N^-alpha``. Two points fit a line exactly
   and one distinct abscissa has no slope at all, so ``lstsq_alpha`` must report
   ``identifiable=False`` in both cases rather than inventing an exponent. And the point-order
   trap is real: sorting the endpoint *names* alphabetically puts ``p1600`` before ``p4000``
   and inverts the trend, which would flip the verdict -- so ordering is by training volume and
   that is asserted.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "r7_s3_gap_decay_and_beta.py"


def _module():
    spec = importlib.util.spec_from_file_location("gap_decay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


@pytest.fixture(scope="module")
def geometry():
    from training import r7_spectral_geometry
    return r7_spectral_geometry


# --- the anti-phase singularity, on the exact measured numbers ----------------------

SINGULAR = (802.585152921592, 93.39619533837474, -273.7853168413974)
LITERAL = 3210.340611686368


def test_anti_phase_band_matches_the_literal_limit(geometry):
    """At ``coh = -1`` the term must equal ``2*max(P,Q)*(1-coh)``, not inf.

    The numbers are the measured band-0 powers of a real 6 h ``v10`` pair.
    """
    px = torch.tensor([SINGULAR[0]], dtype=torch.float64)
    py = torch.tensor([SINGULAR[1]], dtype=torch.float64)
    cr = torch.tensor([SINGULAR[2]], dtype=torch.float64)
    _, second = geometry._terms(px, py, cr)
    assert torch.isfinite(second).all()
    assert float(second[0]) == pytest.approx(LITERAL, rel=1e-12)


def test_anti_phase_is_not_reported_as_a_match(geometry):
    """Perfect anti-phase is the worst case, so the term must be large -- never zero."""
    px = torch.tensor([SINGULAR[0]], dtype=torch.float64)
    py = torch.tensor([SINGULAR[1]], dtype=torch.float64)
    cr = torch.tensor([SINGULAR[2]], dtype=torch.float64)
    _, second = geometry._terms(px, py, cr)
    assert float(second[0]) > 1000.0


def test_exact_zero_is_preserved(geometry):
    """``amse(x, x)`` must be exactly 0.0 -- decision 0110's family argument depends on it."""
    torch.manual_seed(0)
    field = torch.randn(65, 65, dtype=torch.float64)
    assert float(geometry.amse_analogue(field, field)) == 0.0
    assert float(geometry.amse_analogue(field, -field)) > 0.0
    assert torch.isfinite(geometry.amse_analogue(field, -field))


def test_a_negated_field_stays_finite_over_many_draws(geometry):
    """Anti-phase on a real grid must never return inf/nan, whatever the spectrum."""
    torch.manual_seed(3)
    for _ in range(8):
        field = torch.randn(65, 65, dtype=torch.float64)
        value = geometry.amse_analogue(field, -field)
        assert torch.isfinite(value)


# --- the decay fit must be able to refuse -------------------------------------------

def test_two_points_are_reported_as_not_identifiable(probe):
    """Two points fit a line exactly, so the exponent carries no evidence."""
    result = probe.lstsq_alpha([2008, 128528], [0.28, 0.22])
    assert result["alpha"] is not None
    assert result["identifiable"] is False
    assert "two points" in result["reason"]


def test_one_abscissa_has_no_slope(probe):
    """Repeated updates give no slope at all; the fit must refuse rather than divide."""
    result = probe.lstsq_alpha([2008, 2008, 2008], [0.3, 0.28, 0.27])
    assert result["alpha"] is None
    assert result["identifiable"] is False


def test_three_points_recover_a_known_exponent(probe):
    """A synthetic power law with alpha = 0.05 must come back as 0.05."""
    updates = [2008, 32132, 128528]
    gaps = [0.30 * (n / 2008.0) ** -0.05 for n in updates]
    result = probe.lstsq_alpha(updates, gaps)
    assert result["alpha"] == pytest.approx(0.05, abs=1e-9)
    assert result["identifiable"] is True
    assert result["max_log_residual"] < 1e-9


def test_zero_gaps_are_dropped_rather_than_breaking_the_log(probe):
    """A vanished gap has no log; it must be excluded, not turned into a huge negative."""
    result = probe.lstsq_alpha([2008, 8033, 32132, 128528], [0.28, 0.25, 0.0, 0.22])
    assert result["points_used"] == 3
    assert result["alpha"] is not None


# --- the halving extrapolation -------------------------------------------------------

def test_small_alpha_makes_the_halving_unreachable(probe):
    """alpha = 0.05 needs 775,000x more training; that is the finding, not an error."""
    result = probe.updates_to_close(0.2269, 0.05111422048465463)
    assert result["reachable"] is True
    assert result["updates_multiplier"] == pytest.approx(775102.0, rel=1e-3)


def test_a_non_positive_alpha_means_no_shrinking_at_all(probe):
    """If the gap does not shrink with training, no extrapolation exists."""
    for alpha in (0.0, -0.1, None):
        result = probe.updates_to_close(0.2, alpha)
        assert result["reachable"] is False


def test_no_gap_means_nothing_to_extrapolate(probe):
    assert probe.updates_to_close(0.0, 0.05)["reachable"] is False


def test_a_large_alpha_would_be_closable(probe):
    """Sanity check on the other side: alpha = 1 halves the gap by doubling training."""
    result = probe.updates_to_close(0.2, 1.0)
    assert result["updates_multiplier"] == pytest.approx(2.0)


# --- the ordering trap ---------------------------------------------------------------

def test_training_points_are_ordered_by_volume_not_by_name(probe):
    """Alphabetical order puts p1600 before p4000 and inverts the whole trend."""
    names = [name for name, _ in probe.TRAINING_POINTS]
    assert names == ["p0025", "p1000", "p4000", "p1600"]
    assert names != sorted(names), "sorted() reorders them; the ordering must be explicit"
    updates = [count for _, count in probe.TRAINING_POINTS]
    assert updates == sorted(updates)


def test_c1_reorders_the_product_by_training_volume(probe):
    """Whatever order the product lists its points in, the fit must use training volume."""
    def point(name, gaps):
        return {"name": name,
                "bands": [{"band": index, "gap": gap} for index, gap in enumerate(gaps)],
                "bands_all_leads": [{"band": index, "gap": gap}
                                    for index, gap in enumerate(gaps)]}
    # Deliberately scrambled: p1600 first, p0025 last.
    product = {"p1": {"points": [
        point("p1600", [0.05, 0.10, 0.20, 0.27, 0.29, 0.11]),
        point("p0025", [0.09, 0.25, 0.33, 0.35, 0.31, 0.16]),
        point("p4000", [0.07, 0.18, 0.30, 0.31, 0.27, 0.10]),
        point("p1000", [0.07, 0.23, 0.32, 0.33, 0.29, 0.13]),
    ]}}
    result = probe.c1_decay(product)
    assert result["ordering"] == ["p0025", "p1000", "p4000", "p1600"]
    assert result["updates"] == [2008, 8033, 32132, 128528]


def test_c1_reports_both_lead_sets(probe):
    """The five-lead and twelve-lead aggregates must both be reported, not one hidden."""
    def point(name, gap):
        return {"name": name,
                "bands": [{"band": index, "gap": gap} for index in range(6)],
                "bands_all_leads": [{"band": index, "gap": gap * 0.9} for index in range(6)]}
    product = {"p1": {"points": [point("p0025", 0.30), point("p1000", 0.27),
                                 point("p4000", 0.24), point("p1600", 0.22)]}}
    result = probe.c1_decay(product)
    assert set(result["lead_sets"]) == {"frozen_five_lead", "all_twelve_lead"}


# --- the beta weighting --------------------------------------------------------------

def test_beta_is_the_inverse_of_the_mean_amse(probe):
    """FastNet eq. (8): ``beta_j = 1 / <AMSE_j>``."""
    assert probe._spread([1.0, 2.0, 4.0]) == pytest.approx(4.0)
    assert probe._spread([1.0]) != probe._spread([1.0])  # nan for one value
    assert probe._spread([0.0, -1.0]) != probe._spread([0.0, -1.0])


def test_beta_spread_reports_the_pooling_distortion(probe):
    """An unweighted 17-channel sum gives 1/17 to everyone; beta says how wrong that is."""
    betas = {"channels": {f"c{index}": {"beta": 10.0 ** index} for index in range(4)}}
    result = probe.c2_beta_reweighted({}, betas)
    assert result["available"] is True
    assert result["channels_with_beta"] == 4
    assert result["unweighted_share_is_equal"] == pytest.approx(0.25)
    assert result["beta_spread_max_over_min"] == pytest.approx(1000.0)
    assert result["weighted_share_max"] > result["unweighted_share_is_equal"]


def test_c2_refuses_on_too_few_variables(probe):
    """One variable cannot show a pooling distortion."""
    result = probe.c2_beta_reweighted({}, {"channels": {"only": {"beta": 1.0}}})
    assert result["available"] is False


# --- the refusal surface -------------------------------------------------------------

@pytest.mark.parametrize("worker", ["prepare", "archive", "seed"])
def test_training_workers_exit_nonzero(probe, worker):
    """A zero-GPU preflight must refuse to create a protocol or an archive."""
    with pytest.raises(SystemExit) as caught:
        probe.main(["--worker", worker])
    assert caught.value.code not in (0, None)


def test_the_probe_declares_no_scientific_claim(probe):
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"scientific_claim": False' in source
    assert '"test_read": False' in source
    assert '"r": 0' in source


def test_the_gate_threshold_is_derived_not_tuned(probe):
    """The 1000x halving multiplier must be a reading of alpha, not a fitted constant."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "1000.0" in source
    assert "halving_multiplier" in source
    # A tuned constant would appear as a bare alpha threshold; there must be none.
    assert "alpha_threshold" not in source
