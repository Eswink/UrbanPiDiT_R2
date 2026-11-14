"""Counter-proofs for the balance round's verdict: every clause must be able to fail.

Decision 0109's lesson is that a criterion which cannot fail is not a criterion, and the round's
whole claim rests on (b). So each clause below is exercised against a *constructed* table where it
must come out false, and against one where it must come out true -- a clause that returns the same
answer either way is not measuring anything.

The clauses are pure functions of per-arm tables, so these tests need no GPU, no store and no
dumped fields. They construct the tables directly, which also documents what shape the real
reading has.

**What each test defends.**

1. ``test_a_dead_diagnostic_fails`` -- three arms whose balance medians are identical must fail (a).
   This is the failure mode (a) exists to name: a diagnostic that ignores the arm.
2. ``test_a_rejects_an_unstable_ordering`` -- distinct values that swap places between seeds are
   not "consistently distinct"; a sweep of noise across seeds must not pass.
3. ``test_b_detects_a_reversal`` and ``test_b_is_false_when_balance_follows_rmse`` -- (b) is true
   only when the better-RMSE arm is the worse-balanced one, and false when the two agree. The
   second is the Bonavita-falsifying case and must be reported as a plain false.
4. ``test_b_ignores_a_single_seed_flip`` -- one seed reversing out of three is noise and must not
   decide the main clause, while still appearing in the reported any-seed count.
5. ``test_c_fails_when_seed_spread_exceeds_the_arm_gap`` -- (c) has teeth.
6. ``test_d_fails_when_the_diagnostic_cannot_separate_truth`` -- the positive control must fail if
   the model field is *not* more unbalanced than its own analysis.
7. ``test_untested_clauses_are_null_not_false`` -- the distinction the whole workflow depends on:
   with one arm, (a)(b)(c) are ``None``, never ``False`` and never ``True``.
8. ``test_the_verdict_string_names_the_failure_mode`` -- (a) false and (b) false must produce
   *different* verdicts, because they are different scientific facts.
9. ``test_rmse_reproduces_the_registered_latitude_weighting`` -- the dump's RMSE must equal the
   accumulator's definition on a synthetic case, since (b) is decided on that number.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import r7_balance_exits as exits
from scripts import r7_balance_verdict as verdict


def _table(family, arm, seed, *, balance_850=None, rmse=None, model_geo=10.0,
           truth_geo=1.0, model_noise=10.0, truth_noise=1.0, lead="72"):
    """One arm's table in the shape the reading stage produces.

    The keys are built with the producer's own functions rather than typed out. A fixture that
    spelled them by hand could agree with a broken consumer while disagreeing with
    `r7_balance_verdict.case_metrics` -- which is exactly the defect the end-to-end smoke caught.
    """
    balance_row = {verdict.truth_key("geostrophic:500"): {"median": truth_geo, "n": 1,
                                                          "spread": 0.0},
                   verdict.truth_key("spectral:fine_scale_noise"): {"median": truth_noise,
                                                                    "n": 1, "spread": 0.0},
                   verdict.model_key("geostrophic:500"): {"median": model_geo, "n": 1,
                                                           "spread": 0.0},
                   verdict.model_key("spectral:fine_scale_noise"): {"median": model_noise,
                                                                     "n": 1, "spread": 0.0}}
    if balance_850 is not None:
        balance_row[verdict.model_key("hydrostatic:850-500")] = {"median": balance_850, "n": 1,
                                                                 "spread": 0.0}
    return {"family": family, "arm": arm, "seed": seed, "dir": "/nonexistent",
            "n_cases": 1, "balance": {lead: balance_row}, "rmse": {lead: rmse or {}}}


THREE_ARMS = {"arms": ("control", "treatment", "placebo"), "seeds": (41, 42, 43)}


# --- (a) the diagnostic must not be dead -------------------------------------------------------

def test_a_dead_diagnostic_fails():
    """Three arms with identical medians: the diagnostic did not see the perturbation."""
    tables = {f"urban_static/{arm}/seed{seed}": _table("urban_static", arm, seed, balance_850=0.03)
              for arm in THREE_ARMS["arms"] for seed in (41, 42, 43)}
    clause = exits.clause_a(tables, {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is False
    assert clause["rows"][0]["distinct_on_every_seed"] is False


def test_a_rejects_an_unstable_ordering():
    """Distinct but shuffling medians are unresolved, not ordered."""
    tables = {}
    values = {41: (0.030, 0.040, 0.050), 42: (0.050, 0.030, 0.040), 43: (0.040, 0.050, 0.030)}
    for seed, triple in values.items():
        for arm, value in zip(THREE_ARMS["arms"], triple):
            tables[f"urban_static/{arm}/seed{seed}"] = _table(
                "urban_static", arm, seed, balance_850=value)
    clause = exits.clause_a(tables, {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is False
    assert clause["rows"][0]["same_ordering_on_every_seed"] is False


def test_a_passes_when_the_arms_are_separated_and_stable():
    tables = {f"urban_static/{arm}/seed{seed}": _table("urban_static", arm, seed,
                                                       balance_850=0.020 + 0.005 * index)
              for index, arm in enumerate(THREE_ARMS["arms"]) for seed in (41, 42, 43)}
    clause = exits.clause_a(tables, {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is True


def test_a_tolerance_rejects_float_noise():
    """Gaps below the predeclared tolerance must not count as distinct."""
    tables = {}
    for index, arm in enumerate(THREE_ARMS["arms"]):
        for seed in (41, 42, 43):
            tables[f"urban_static/{arm}/seed{seed}"] = _table(
                "urban_static", arm, seed,
                balance_850=0.0300 + index * exits.ARM_DISTINCT_TOLERANCE / 4)
    assert exits.clause_a(tables, {"urban_static": THREE_ARMS}, (72,))["value"] is False


def test_a_does_not_require_more_than_numerical_resolution():
    """A gap far above float32 resolution but far below a 'scientific' effect must still count.

    (a) asks only whether the diagnostic responded at all; how large the response is belongs to
    (c). A threshold set at an effect size would double-count (c) and could call a live
    diagnostic dead.
    """
    tables = {}
    for index, arm in enumerate(THREE_ARMS["arms"]):
        for seed in (41, 42, 43):
            tables[f"urban_static/{arm}/seed{seed}"] = _table(
                "urban_static", arm, seed, balance_850=0.0280 + index * 1.0e-4)
    clause = exits.clause_a(tables, {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is True, "a 1e-4 gap is 5 orders of magnitude above float32 noise"


# --- (b) the main clause: RMSE and balance must be able to disagree ----------------------------

def _b_tables(reversals, *, lead="72"):
    """Three seeds; ``reversals`` says per seed whether the better-RMSE arm is worse balanced.

    The reversal is encoded **only** in the hydrostatic metric. The geostrophic and spectral
    numbers are made to follow the RMSE ordering exactly, so a seed marked ``False`` contains no
    reversal on any metric -- otherwise the fixture would be testing the wrong thing and the
    "agrees" case would still report a reversal somewhere.
    """
    tables = {}
    for seed_index, reversed_ in enumerate(reversals):
        seed = 41 + seed_index
        # treatment has the better (lower) RMSE on every seed.
        rmse_control, rmse_treatment = 2.0, 1.5
        if reversed_:
            balance_control, balance_treatment = 0.02, 0.05
        else:
            balance_control, balance_treatment = 0.05, 0.02
        # Unbalance follows the error ranking: worst RMSE is the most unbalanced.
        tables[f"urban_static/control/seed{seed}"] = _table(
            "urban_static", "control", seed, balance_850=balance_control,
            rmse={"z500": rmse_control}, model_geo=12.0, model_noise=12.0)
        tables[f"urban_static/treatment/seed{seed}"] = _table(
            "urban_static", "treatment", seed, balance_850=balance_treatment,
            rmse={"z500": rmse_treatment}, model_geo=10.0, model_noise=10.0)
        tables[f"urban_static/placebo/seed{seed}"] = _table(
            "urban_static", "placebo", seed, balance_850=0.03,
            rmse={"z500": 1.8}, model_geo=11.0, model_noise=11.0)
    return tables


def test_b_detects_a_reversal():
    clause = exits.clause_b(_b_tables([True, True, True]), {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is True
    assert clause["n_reversed_all_seeds"] >= 1


def test_b_is_false_when_balance_follows_rmse():
    """The Bonavita-falsifying case: the better-RMSE arm is also the better-balanced one."""
    clause = exits.clause_b(_b_tables([False, False, False]), {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is False
    assert clause["n_reversed_all_seeds"] == 0


def test_b_ignores_a_single_seed_flip():
    """One seed out of three is noise: it must be reported, not used to decide."""
    clause = exits.clause_b(_b_tables([True, False, False]), {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is False
    assert clause["n_reversed_any_seed"] >= 1
    assert clause["n_reversed_all_seeds"] == 0


# --- (c) the seed spread must be smaller than the arm gap -------------------------------------

def test_c_fails_when_seed_spread_exceeds_the_arm_gap():
    tables = {}
    spreads = {41: 0.010, 42: 0.040, 43: 0.070}
    for seed, value in spreads.items():
        tables[f"urban_static/control/seed{seed}"] = _table(
            "urban_static", "control", seed, balance_850=value)
        tables[f"urban_static/treatment/seed{seed}"] = _table(
            "urban_static", "treatment", seed, balance_850=value + 0.005)
        tables[f"urban_static/placebo/seed{seed}"] = _table(
            "urban_static", "placebo", seed, balance_850=value + 0.010)
    clause = exits.clause_c(tables, {"urban_static": THREE_ARMS}, (72,))
    assert clause["value"] is False
    assert clause["rows"][0]["widest_seed_range"] > clause["rows"][0]["smallest_arm_gap"]


def test_c_passes_when_the_arms_are_far_apart_and_stable():
    tables = {}
    for index, arm in enumerate(THREE_ARMS["arms"]):
        for seed, jitter in zip((41, 42, 43), (0.0, 0.001, -0.001)):
            tables[f"urban_static/{arm}/seed{seed}"] = _table(
                "urban_static", arm, seed, balance_850=0.020 + 0.010 * index + jitter)
    assert exits.clause_c(tables, {"urban_static": THREE_ARMS}, (72,))["value"] is True


# --- (d) the positive control must be able to fail --------------------------------------------

def test_d_fails_when_the_diagnostic_cannot_separate_truth():
    tables = {"urban_static/control/seed41": _table(
        "urban_static", "control", 41, balance_850=0.03, model_geo=1.0, truth_geo=1.0,
        model_noise=1.0, truth_noise=1.0)}
    clause = exits.clause_d(tables, (72,))
    assert clause["value"] is False


def test_d_passes_when_the_model_is_worse_than_its_own_analysis():
    tables = {"urban_static/control/seed41": _table("urban_static", "control", 41,
                                                    balance_850=0.03)}
    assert exits.clause_d(tables, (72,))["value"] is True


# --- untested is null, never false -------------------------------------------------------------

def test_untested_clauses_are_null_not_false():
    """A single arm gives no contrast: (a)(b)(c) must be null, and the verdict must say untested."""
    tables = {"urban_static/control/seed41": _table("urban_static", "control", 41,
                                                    balance_850=0.03)}
    judgement = exits.evaluate(tables, {"urban_static": THREE_ARMS}, (72,), preflight_passed=True)
    for name in ("a_arms_differ", "b_rmse_and_balance_disagree", "c_seed_spread_below_arm_gap"):
        assert judgement["values"][name] is None, name
    assert judgement["verdict"] == "untested-no-arm-contrast"


def test_a_partial_seed_set_is_untested_not_a_verdict():
    """One seed of five arms must not decide anything. The end-to-end smoke found this.

    With seed 41 only, (a) came back ``false`` ("the diagnostic is dead") and (b) came back
    ``true`` -- both verdicts the frozen criteria, which are stated on three seeds, do not
    authorise. The completeness gate must make all three arm-contrast clauses ``null``.
    """
    tables = _b_tables([True])          # one seed, three arms, a real reversal
    judgement = exits.evaluate(tables, {"urban_static": THREE_ARMS}, (72,), preflight_passed=True)
    for name in ("a_arms_differ", "b_rmse_and_balance_disagree", "c_seed_spread_below_arm_gap"):
        assert judgement["values"][name] is None, name
    assert judgement["verdict"] == "untested-no-arm-contrast"
    assert "urban_static" in judgement["clauses"]["b_rmse_and_balance_disagree"][
        "skipped_families"]


def test_a_false_and_b_false_are_different_verdicts():
    """(a) false means dead; (b) false means redundant. They must not share a verdict string."""
    dead = {f"urban_static/{arm}/seed{seed}": _table("urban_static", arm, seed, balance_850=0.03,
                                                     rmse={"z500": 1.0 + index})
            for index, arm in enumerate(THREE_ARMS["arms"]) for seed in (41, 42, 43)}
    dead_verdict = exits.evaluate(dead, {"urban_static": THREE_ARMS}, (72,), True)["verdict"]

    live = _b_tables([False, False, False])
    redundant_verdict = exits.evaluate(live, {"urban_static": THREE_ARMS}, (72,), True)["verdict"]

    assert dead_verdict == "negative-diagnostic-is-dead"
    assert redundant_verdict == "negative-diagnostic-is-redundant-with-rmse"
    assert dead_verdict != redundant_verdict


def test_both_a_and_b_true_is_the_only_licensing_verdict():
    tables = _b_tables([True, True, True])
    for index, arm in enumerate(THREE_ARMS["arms"]):
        for seed in (41, 42, 43):
            key = f"urban_static/{arm}/seed{seed}"
            tables[key]["balance"]["72"][verdict.model_key("hydrostatic:850-500")]["median"] = (
                0.020 + 0.010 * index)
    judgement = exits.evaluate(tables, {"urban_static": THREE_ARMS}, (72,), True)
    assert judgement["values"]["a_arms_differ"] is True
    assert judgement["values"]["b_rmse_and_balance_disagree"] is True
    assert judgement["verdict"] == "balance-is-an-independent-axis"


# --- the RMSE definition (b) is decided on ----------------------------------------------------

def test_rmse_reproduces_the_registered_latitude_weighting():
    """A constant error field must give exactly that constant, area weighting or not."""
    height = width = 5
    latitude = np.linspace(43.0, 27.0, height)
    channels = ["t2m", "z500"]
    target = np.zeros((1, len(channels), height, width), dtype=np.float64)
    forecast = np.zeros_like(target)
    forecast[:, 0, :, :] = 2.5
    forecast[:, 1, :, :] = -4.0
    case = {"forecast": forecast, "target": target, "latitude": latitude,
            "longitude": np.linspace(107.0, 123.0, width), "lead_hours": [72],
            "channels": channels}
    row = verdict.case_rmse(case)["72"]
    assert row["t2m"] == pytest.approx(2.5, rel=1e-12)
    assert row["z500"] == pytest.approx(4.0, rel=1e-12)


def test_rmse_weights_by_cosine_latitude():
    """A polar-only error must count for less than an equatorial-only one of the same size."""
    latitude = np.array([0.0, 30.0, 60.0])
    width = 3
    target = np.zeros((1, 1, 3, width), dtype=np.float64)
    polar = np.zeros_like(target)
    polar[:, :, 2, :] = 1.0
    equatorial = np.zeros_like(target)
    equatorial[:, :, 0, :] = 1.0
    base = {"target": target, "latitude": latitude,
            "longitude": np.linspace(110.0, 112.0, width), "lead_hours": [6],
            "channels": ["t2m"]}
    polar_rmse = verdict.case_rmse({**base, "forecast": polar})["6"]["t2m"]
    equatorial_rmse = verdict.case_rmse({**base, "forecast": equatorial})["6"]["t2m"]
    assert equatorial_rmse > polar_rmse
    # The analytic value: sqrt(cos(0) / (cos0 + cos30 + cos60)).
    expected = math.sqrt(1.0 / (1.0 + math.cos(math.radians(30)) + math.cos(math.radians(60))))
    assert equatorial_rmse == pytest.approx(expected, rel=1e-12)


def test_every_declared_variable_has_a_defensible_balance_partner():
    """The (b) map must not list a metric the diagnostics do not produce."""
    produced = {"hydrostatic:850-500", "hydrostatic:500-250",
                "geostrophic:850", "geostrophic:500", "geostrophic:250",
                "column_divergence", "spectral:fine_scale_noise"}
    for variable, metrics in exits.VARIABLE_TO_BALANCE:
        assert metrics, variable
        assert set(metrics) <= produced, variable


def test_the_clauses_read_the_keys_the_producer_actually_writes():
    """End-to-end key agreement, on a synthetic case -- the defect the smoke found.

    `case_metrics` wrote ``forecast:hydrostatic:850-500`` while the clauses looked up
    ``hydrostatic:850-500``, so on a full round every clause would have come back null while
    each unit test passed on hand-built tables. This test builds the table *through the
    producer* and then requires each clause to find its number.
    """
    import numpy as np

    height = width = 9
    latitude = np.linspace(43.0, 27.0, height)
    longitude = np.linspace(107.0, 123.0, width)
    channels = ["t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250"]
    # A near-hydrostatic column with a uniform wind, plus a small imbalance.
    t850, t500 = 280.0, 250.0
    z850 = np.full((height, width), 1_500.0)
    thickness = 287.05 * 0.5 * (t850 + t500) * np.log(850.0 / 500.0)
    z500 = z850 + thickness
    z250 = z500 + thickness
    frames = {name: np.zeros((height, width)) for name in channels}
    frames.update({"t850": np.full((height, width), t850),
                   "t500": np.full((height, width), t500),
                   "t2m": np.full((height, width), 280.0),
                   "q850": np.full((height, width), 0.005),
                   "q500": np.full((height, width), 0.001),
                   "z850": z850, "z500": z500, "z250": z250,
                   "u850": np.full((height, width), 10.0), "u500": np.full((height, width), 20.0),
                   "u250": np.full((height, width), 30.0)})
    stack = np.stack([frames[name] for name in channels])[None]
    case = {"forecast": stack * 1.02, "target": stack, "latitude": latitude,
            "longitude": longitude, "lead_hours": [72], "channels": channels}
    produced = verdict.case_metrics(case)["72"]
    for metric in verdict.HEADLINE_METRICS:
        assert verdict.model_key(metric) in produced, metric
        assert verdict.truth_key(metric) in produced, metric
    # And the clauses must find the forecast number through the same key.
    table = {"family": "urban_static", "arm": "control", "seed": 41, "dir": "/nonexistent",
             "n_cases": 1, "balance": {"72": produced}, "rmse": {"72": {"z500": 1.0}}}
    assert exits._balance(table, "72", "hydrostatic:850-500") is not None
    assert exits._balance(table, "72", "geostrophic:500", side="target") is not None
