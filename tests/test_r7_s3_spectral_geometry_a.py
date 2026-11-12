"""Stage A's counterproof tests: the (d) early return, the beta_j weighting, and the (e) guards.

Decision 0109's lesson is the reason this file exists. C2's exit ``c`` was the literal ``True``
in the driver and ``d`` read two constants the driver itself wrote, so a fabricated reading
still reported ``all=True``. The (d) clause here is therefore carried by *code paths*: a test
that fails if the dispatcher ever builds the axis when it is unset, and a test that fails if the
returned object stops matching what the runner consumes.

Every test either fails on a real defect or asserts a property the protocol depends on; none
re-states a value the module itself produced.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import r7_spectral_geometry as geometry


class _Result:
    """The two attributes the axis reads off the rollout result."""

    def __init__(self, forecasts):
        self.forecasts = forecasts
        self.loss = torch.zeros(())


def _pair(batch=2, steps=3, channels=17, size=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    forecast = torch.randn(batch, steps, channels, size, size, generator=generator)
    target = torch.randn(batch, steps, channels, size, size, generator=generator)
    return forecast, target


# --------------------------------------------------------------------------- (d) the off path

def test_the_axis_returns_none_before_constructing_anything():
    """The registered arithmetic is kept bit for bit only by returning before anything exists."""
    from training.r7_objective_axes import auxiliary_term

    assert auxiliary_term(None, None, {"autoregression": {}}) is None


def test_off_arm_matches_the_registered_arithmetic():
    """(d)'s carrier: with the axis unset the dispatcher must not construct the loss at all.

    This is the test that replaces a literal ``True``. It plants an object whose attribute
    access *raises*, so if the dispatcher ever touched ``result.forecasts`` on an off round the
    test fails with that exception rather than silently passing. A fabricated reading cannot
    satisfy it, because there is no reading -- only the code path.
    """
    from training.r7_objective_axes import auxiliary_term

    class _Explodes:
        def __getattr__(self, name):  # noqa: D105 - deliberate
            raise AssertionError(f"the off path touched result.{name}; it must return None first")

    assert auxiliary_term(_Explodes(), _Explodes(), {"autoregression": {}}) is None


def test_off_arm_is_none_even_when_the_targets_are_present():
    """A populated batch must not tempt the dispatcher into building the axis."""
    from training.r7_objective_axes import auxiliary_term

    forecast, target = _pair()
    contract = {"autoregression": {"physical_steps": 12}}
    assert auxiliary_term(_Result(forecast), {"physical_targets": target}, contract) is None


def test_the_axis_is_registered_in_the_dispatcher():
    from training.r7_objective_axes import AXES

    assert "spectral_geometry" in AXES


# ------------------------------------------------------------- the runner's contract (the bug)

def test_the_axis_matches_the_runner_contract():
    """``_update`` does ``result.loss + constraint['total']``; a bare tensor crashes there.

    Measured before the fix: ``IndexError: too many indices for tensor of dimension 0`` at the
    first update. Stage Q was skipped, so the axis had never been run end to end and the
    mismatch went unseen -- this test is the only thing standing between that and a wasted round.
    """
    forecast, target = _pair()
    contract = {"autoregression": {"spectral_geometry": "amse-analogue"}}
    term = geometry.spectral_geometry_term(_Result(forecast), {"physical_targets": target}, contract)
    assert isinstance(term, dict), f"the runner indexes ['total']; got {type(term).__name__}"
    assert "total" in term
    # The exact expression the runner evaluates.
    objective = _Result(forecast).loss + term["total"]
    assert torch.isfinite(objective)


def test_the_axis_records_a_beta_weighted_key_for_the_report():
    """The on arm's per-update report must show the weighted term, not just the pooled one."""
    forecast, target = _pair()
    beta = [0.5] * 17
    contract = {"autoregression": {"spectral_geometry": "amse-analogue", "beta_weights": beta}}
    term = geometry.spectral_geometry_term(_Result(forecast), {"physical_targets": target}, contract)
    # ``_update`` copies every tensor value into ``constraint_<name>`` in the report.
    assert isinstance(term["beta_weighted"], torch.Tensor)
    assert float(term["beta_weighted"]) > 0.0


# ------------------------------------------------------------------ (e) preflight properties

def test_per_channel_amse_keeps_the_variable_axis():
    """A per-variable weight cannot apply to a pooled quantity; the axis must survive."""
    forecast, target = _pair()
    values = geometry.per_channel_amse(forecast[:, 0], target[:, 0])
    assert tuple(values.shape) == (17,)
    assert bool((values > 0).all())


def test_beta_weighting_is_a_pure_sum_not_a_mean():
    """FastNet eq. (10) sums over variables; a ``1/J`` here would rescale the whole term.

    The test is constructed so the two conventions differ by exactly ``J``: every channel has
    the same per-channel AMSE, so the sum is ``J`` times one channel's value and a mean would be
    one channel's value. If a future edit divides by ``J`` this fails by a factor of 17.
    """
    channels = 17
    target = torch.zeros(1, 1, channels, 8, 8)
    forecast = torch.ones(1, 1, channels, 8, 8)
    beta = [1.0] * channels
    total = geometry.beta_weighted_amse(forecast, target, torch.tensor(beta))
    one = geometry.per_channel_amse(forecast[:, 0], target[:, 0])[0]
    assert torch.allclose(total, one * channels, rtol=1e-5), (
        f"a pure sum must be {channels}x one channel, got {float(total)} vs {float(one)}")


def test_beta_weighting_scales_linearly_in_beta():
    """Doubling a weight must double that variable's contribution -- the weight acts.

    ``batch=1, steps=1`` so the two functions' different reductions (batch mean vs step mean)
    are both identities and the comparison is exact.
    """
    forecast, target = _pair(batch=1, steps=1, channels=2)
    one = torch.tensor([1.0, 1.0])
    two = torch.tensor([2.0, 1.0])
    base = geometry.beta_weighted_amse(forecast, target, one)
    bumped = geometry.beta_weighted_amse(forecast, target, two)
    per_channel = geometry.per_channel_amse(forecast[:, 0], target[:, 0])
    assert torch.allclose(bumped - base, per_channel[0], rtol=1e-5)


def test_beta_weighting_is_exact_zero_at_truth():
    """J4's family property must survive the weighting, or the optimum is not the truth."""
    forecast, _ = _pair()
    beta = torch.linspace(0.1, 2.0, 17)
    total = geometry.beta_weighted_amse(forecast, forecast, beta)
    assert float(total) == 0.0


def test_the_axis_rejects_a_nonpositive_beta():
    forecast, target = _pair()
    for bad in ([0.0] * 17, [-1.0] + [0.5] * 16):
        contract = {"autoregression": {"spectral_geometry": "amse-analogue", "beta_weights": bad}}
        with pytest.raises(ValueError, match="strictly positive"):
            geometry.spectral_geometry_term(_Result(forecast), {"physical_targets": target}, contract)


def test_the_axis_rejects_a_nonfinite_beta():
    forecast, target = _pair()
    beta = [float("nan")] + [0.5] * 16
    contract = {"autoregression": {"spectral_geometry": "amse-analogue", "beta_weights": beta}}
    with pytest.raises(ValueError, match="finite"):
        geometry.spectral_geometry_term(_Result(forecast), {"physical_targets": target}, contract)


def test_the_axis_rejects_a_beta_of_the_wrong_length():
    forecast, target = _pair()
    contract = {"autoregression": {"spectral_geometry": "amse-analogue",
                                   "beta_weights": [0.5] * 16}}
    with pytest.raises(ValueError, match="17 channels"):
        geometry.spectral_geometry_term(_Result(forecast), {"physical_targets": target}, contract)


def test_a_constant_beta_is_the_pooled_proportional_but_not_equal():
    """``beta_j`` constant is NOT inert: eq. (10) still sums over 17 variables.

    Records the arithmetic a reader will otherwise wonder about -- a constant weight of ``c``
    makes the term ``c * sum_j AMSE_j``, which differs from the pooled ``mean_k`` form Stage P
    measured. So "beta_j not all equal" ((e)) is about the *weighting* being informative, not
    about the term being non-zero.
    """
    forecast, target = _pair(batch=1, steps=1)
    constant = torch.full((17,), 0.25)
    total = geometry.beta_weighted_amse(forecast, target, constant)
    per_channel = geometry.per_channel_amse(forecast[:, 0], target[:, 0])
    assert torch.allclose(total, 0.25 * per_channel.sum(), rtol=1e-5)


# ------------------------------------------------------------------ the (a) reading's math

def test_fine_band_amp_ratio_is_a_ratio_of_power():
    """The (a) reading must be ``sqrt(sum PSD_pred / sum PSD_target)``, not a mean of ratios.

    Stage P1's aggregate sums PSDs before dividing; averaging per-case ratios instead would let
    a near-empty band contribute a wild ratio with full weight. This pins the convention the
    two rounds' numbers are compared under.
    """
    import numpy as np
    from scripts import study_r7_s3_spectral_geometry_a as driver

    class _Row(dict):
        pass

    per_case = [{"case": 0, "leads": [{"lead_hours": 6, "bands": [
        {"band": 4, "psd_pred": 9.0, "psd_target": 100.0, "cross": 20.0},
        {"band": 5, "psd_pred": 16.0, "psd_target": 400.0, "cross": 40.0}]}]}]
    rows = driver._band_aggregate(per_case)
    assert [row["band"] for row in rows] == [4, 5]
    assert math.isclose(rows[0]["amp_ratio"], 0.3, rel_tol=1e-9)
    assert math.isclose(rows[1]["amp_ratio"], 0.2, rel_tol=1e-9)
    assert math.isclose(rows[0]["gap"], 0.7, rel_tol=1e-9)


def test_the_amplitude_aggregate_only_reads_the_frozen_leads():
    """A lead outside the frozen five must not enter the (a) reading."""
    from scripts import study_r7_s3_spectral_geometry_a as driver

    per_case = [{"case": 0, "leads": [
        {"lead_hours": 6,
         "bands": [{"band": 5, "psd_pred": 1.0, "psd_target": 4.0, "cross": 1.0}]},
        {"lead_hours": 18,
         "bands": [{"band": 5, "psd_pred": 1000.0, "psd_target": 1.0, "cross": 1.0}]}]}]
    rows = driver._band_aggregate(per_case)
    assert math.isclose(rows[0]["amp_ratio"], 0.5, rel_tol=1e-9), (
        "the 18 h lead leaked into the frozen five-lead aggregate")


# --------------------------------------------------------------- the exit's own structure

def test_the_exit_does_not_hardcode_the_d_verdict():
    """Decision 0109: (d) must not be a literal in the driver's exit block."""
    source = (ROOT / "scripts" / "study_r7_s3_spectral_geometry_a.py").read_text(encoding="utf-8")
    assert "d_off_path_not_evaluated_here" in source
    # The literal form must not appear as an assigned verdict.
    for forbidden in ("c = True", "d = True", "d_off_path_unchanged\": True"):
        assert forbidden not in source, f"a hardcoded verdict reappeared: {forbidden!r}"


def test_the_exit_can_report_a_negative():
    """A fabricated positive reading must not be reachable: feed it a losing pair and check."""
    from scripts import study_r7_s3_spectral_geometry_a as driver

    def _entry(skill_24, skill_48, skill_72):
        return {"off": {"rmse": {}, "skill": {"6": 0.0, "12": 0.0, "24": 0.5, "48": 0.5, "72": 0.5}},
                "on": {"rmse": {}, "skill": {"6": 0.0, "12": 0.0, "24": skill_24,
                                             "48": skill_48, "72": skill_72},
                       "beta_weighted_terms_recorded": True, "constraint_terms_recorded": True}}

    def _result(on_amp, off_amp, skill_72, skill_24=0.6, skill_48=0.6):
        return {"seeds": {str(seed): _entry(skill_24, skill_48, skill_72) for seed in (41, 42, 43)},
                "amplitude": {str(seed): {"fine_band_amp_ratio": {"on": on_amp, "off": off_amp}}
                              for seed in (41, 42, 43)},
                "beta_weights": {"values": [0.1, 0.2, 0.3]},
                "preflight": {"cases": [1, 2], "rows": [
                    {"case": 1, "steps": [{"all_finite": True}] * 12},
                    {"case": 2, "steps": [{"all_finite": True}] * 12}]}}

    wins = driver.evaluate_exit(_result(0.9, 0.8, 0.6))
    assert wins["a_fine_band_amp_raised"] and wins["b_72h_skill_nonnegative"]
    assert wins["c_mid_skill_nonnegative"] and wins["e_preflight_passed"]
    assert wins["verdict"] == "advance-to-independent-review"

    # Amplitude fell: (a) is false and the branch closes even though the skill rose.
    loses = driver.evaluate_exit(_result(0.7, 0.8, 0.6))
    assert not loses["a_fine_band_amp_raised"]
    assert loses["verdict"] == "permanently-close-the-spectral-geometry-branch"

    # Skill fell at 72 h only.
    assert driver.evaluate_exit(_result(0.9, 0.8, 0.4))["b_72h_skill_nonnegative"] is False
    # Skill fell at 48 h only.
    assert driver.evaluate_exit(_result(0.9, 0.8, 0.6, skill_48=0.4))["c_mid_skill_nonnegative"] is False


def test_the_exit_can_fail_the_preflight():
    """(e) must be able to report false, including on a non-finite step."""
    from scripts import study_r7_s3_spectral_geometry_a as driver

    base = {"seeds": {str(s): {"off": {"skill": {"24": 0.5, "48": 0.5, "72": 0.5}},
                               "on": {"skill": {"24": 0.6, "48": 0.6, "72": 0.6}}}
                      for s in (41, 42, 43)},
            "amplitude": {str(s): {"fine_band_amp_ratio": {"on": 0.9, "off": 0.8}}
                          for s in (41, 42, 43)},
            "beta_weights": {"values": [0.1, 0.2, 0.3]},
            "preflight": {"cases": [1], "rows": [{"case": 1, "steps": [{"all_finite": True}] * 12}]}}
    assert driver.evaluate_exit(base)["e_preflight_passed"] is True

    broken = {**base, "preflight": {"cases": [1], "rows": [
        {"case": 1, "steps": [{"all_finite": True}] * 11 + [{"all_finite": False}]}]}}
    assert driver.evaluate_exit(broken)["e_preflight_passed"] is False

    constant = {**base, "beta_weights": {"values": [0.5, 0.5, 0.5]}}
    assert driver.evaluate_exit(constant)["e_preflight_passed"] is False


# ------------------------------------------------------------------ protocol-level constants

def test_the_round_freezes_the_counts_decision_0112_fixed():
    """4,000 updates, three seeds, batch 4, two arms: the objective names them all."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    assert proto.UPDATES == 4000
    assert tuple(proto.SEEDS) == (41, 42, 43)
    assert proto.BATCH_SIZE == 4
    assert proto.STEPS == 4
    assert tuple(proto.LEADS) == (6, 12, 24, 48, 72)
    assert len(proto.PREFLIGHT_CASES) == 8


def test_the_pinned_beta_covers_every_channel_and_is_not_constant():
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    assert len(proto.BETA_PINNED) == 17
    values = list(proto.BETA_PINNED.values())
    assert max(values) > min(values)


def test_the_beta_pin_matches_the_step_c_artifact():
    """The frozen list must be the Step C reading, not a hand-typed approximation."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    artifact = Path("/tmp/stepc3.json")
    if not artifact.is_file():
        pytest.skip("the Step C artifact is a /tmp file; re-run scripts/r7_s3_gap_decay_and_beta.py")
    import json
    record = json.loads(artifact.read_text(encoding="utf-8"))
    channels = record["beta"]["channels"]
    for name, pinned in proto.BETA_PINNED.items():
        measured = channels[name]["beta"]
        assert abs(measured - pinned) / pinned < 1e-5, f"{name}: {measured} vs {pinned}"


def test_the_protocol_is_validation_only():
    """Every frozen body carries the two flags the campaign requires."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    body = proto.frozen_constants()
    assert body["scientific_claim"] is False
    assert body["test_read"] is False
    assert body["beta_normalisation"]["frozen"] is True
    assert body["beta_normalisation"]["recomputed_during_training"] is False
    assert "AMSE-analogue" in body["beta_normalisation"]["method_name"]


def test_the_off_arm_declares_no_beta():
    """The control arm's contract must not carry a weight it never uses."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    arms = proto.frozen_constants()["arms"]
    assert arms[proto.CONTROL]["spectral_geometry"] is None
    assert arms[proto.CANDIDATE]["spectral_geometry"] == "amse-analogue"


# --------------------------------------------------------- the dose decision, fixed in advance

def test_the_dose_decision_uses_no_divisor():
    """Equation (10) is a pure sum; a divisor would be a fitted constant."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    decision = proto.dose_decision([58.0, 95.0])
    assert decision["divisor"] == 1
    assert decision["decision"] == "frozen-before-training"


def test_the_dose_decision_reports_the_objective_dominating_regime():
    """At the measured 58-95 the MSE is a residual, so the on arm trains on the AMSE term."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    decision = proto.dose_decision([58.016, 95.217])
    assert decision["on_arm_is_the_paper_regime"] is True
    assert "objective-dominating" in decision["regime"]
    low, high = decision["residual_mse_share"]
    assert 0.0 < low < high < 0.05


def test_the_dose_decision_can_name_the_inert_regime():
    """The same function must be able to report the C1 failure mode, or it decides nothing."""
    from scripts import r7_s3_spectral_geometry_a_protocol as proto

    inert = proto.dose_decision([0.01, 0.5])
    assert inert["on_arm_is_the_paper_regime"] is False
    assert "inert" in inert["regime"]
    mixed = proto.dose_decision([0.5, 5.0])
    assert mixed["on_arm_is_the_paper_regime"] is False
    assert "mixed" in mixed["regime"]
