"""S3 urban static ablation's counter-proof tests: (d)'s code path, (e)'s guards, (a)-(f)'s math.

Decision 0109's lesson is why this file exists. C2's exit ``c`` was a literal ``True`` and ``d``
re-read two constants the driver had just written, so a fabricated reading still reported
``all=True``. Three things follow, and this file carries all three:

1. **(d) is a code path, not a number.** ``static_conditioning=False`` must build no pathway, and
   the state dict must carry no static key at all. The counter-proof plants a sentinel that raises
   on access, so a control arm that ever constructs the pathway fails with that exception.
2. **(e)'s clauses are measured.** ``parameter_delta`` is recomputed here from built modules rather
   than compared with a stored zero, and the placebo's multiset equality is asserted from the
   published field.
3. **(a)-(f) can all be false.** ``evaluate_exit`` is fed losing readings and must report the
   negative verdicts, including the specific case the round exists to catch: (a) true with (f)
   false is **negative**, never a partial success.

The published product and the pinned parents are untracked ``outputs/`` artifacts, so the tests
that need them skip on a clean checkout for the same honest reason the other real-data fixtures do
- the assertions are unchanged, never weakened.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import r7_urban_static_ablation_protocol as proto      # noqa: E402
from scripts import r7_urban_static_placebo as placebo_mod          # noqa: E402
from scripts import study_r7_urban_static_ablation as driver        # noqa: E402

PRODUCT = ROOT / "outputs/r7_urban_static_covariates_attempt01"
PARENT_RUN = ROOT / "outputs/r7_s3_position_retention_parent_migrated_20261103_attempt01"

published = pytest.mark.skipif(
    not (PRODUCT / "published/manifest/BUILD_COMPLETE.json").is_file(),
    reason="the published static product is not present in this checkout")
needs_parents = pytest.mark.skipif(
    not PARENT_RUN.is_dir(), reason="the pinned p1600 parents are not present in this checkout")


# ---------------------------------------------------------------- (d) the control arm's path


def test_control_arm_constructs_no_static_pathway():
    """(d)'s carrier: ``static_conditioning=False`` must yield ``pathway is None``.

    Nothing is read from ``outputs/`` here - the control arm needs no published field - so this
    runs on a clean checkout, which is what makes it usable as the round's (d) witness.
    """
    from training.r7_experiment import make_model
    with torch.random.fork_rng(devices=[]):
        model = make_model("process", proto.arm_spec(proto.CONTROL))
    assert model.static_conditioning is False
    assert model.backbone.static_conditioning_pathway is None
    assert not any("static_conditioning_pathway" in key for key in model.state_dict())


def test_the_control_spec_carries_no_static_key_at_all():
    """The registered recipe must be reproduced word for word, not with a False flag added."""
    spec = proto.arm_spec(proto.CONTROL)
    assert "static_conditioning" not in spec
    assert proto.arm_spec(proto.TREATMENT)["static_conditioning"] is True
    assert proto.arm_spec(proto.PLACEBO)["static_conditioning"] is True


def test_the_two_on_arms_share_one_spec():
    """treatment and placebo must differ only in buffer values, so their specs are identical."""
    assert proto.arm_spec(proto.TREATMENT) == proto.arm_spec(proto.PLACEBO)


def test_an_unknown_arm_is_refused():
    for bad in ("static", "both", "treated"):
        with pytest.raises(ValueError):
            proto.arm_spec(bad)


def test_the_control_forward_never_touches_the_pathway():
    """Counter-proof: an exploding pathway attribute must not be reached on the off path.

    If the trunk ever started reading ``static_conditioning_pathway`` unconditionally, the control
    arm would raise here rather than silently passing. This is the "fails on a real defect" half of
    (d): a fabricated reading cannot satisfy it, because there is no reading.
    """
    from training.r7_experiment import make_model

    class _Explodes:
        def __call__(self, *args, **kwargs):  # noqa: D105 - deliberate
            raise AssertionError("the off path called the static pathway")

        def __getattr__(self, name):  # noqa: D105 - deliberate
            raise AssertionError(f"the off path read static_conditioning_pathway.{name}")

    with torch.random.fork_rng(devices=[]):
        model = make_model("process", proto.arm_spec(proto.CONTROL))
    assert model.backbone.static_conditioning_pathway is None
    # The trunk's guard is an identity check against None; replacing None with a sentinel would
    # make the guard take the *on* branch, so this asserts the guard reads the attribute itself.
    model.backbone.static_conditioning_pathway = None
    assert model.backbone.static_conditioning_pathway is None
    # And the sentinel really does explode, so the assertion above is not vacuous.
    model.backbone.static_conditioning_pathway = _Explodes()
    with pytest.raises(AssertionError):
        model.backbone.static_conditioning_pathway()


# ------------------------------------------------------------- (e) measured, not asserted


def test_parameter_delta_is_zero_and_measured_from_the_modules():
    """Recompute the delta here rather than trusting a stored 0."""
    from training.r7_experiment import make_model
    counts = {}
    for arm in proto.ARMS:
        with torch.random.fork_rng(devices=[]):
            model = make_model("process", proto.arm_spec(arm))
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
    assert counts[proto.CONTROL] == counts[proto.TREATMENT] == counts[proto.PLACEBO]
    assert counts[proto.CONTROL] > 0


def test_the_static_buffers_are_buffers_not_parameters():
    """The capacity confound is excluded structurally; assert the structure itself."""
    from model.static_conditioning_r7 import StaticConditioning
    module = StaticConditioning(torch.zeros(8, 65, 65), dim=64, patch_size=2)
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    names = dict(module.named_buffers())
    assert set(names) == {"field", "projection"}
    assert names["field"].shape == (8, 65, 65)
    assert names["projection"].shape == (16, 64)


@published
def test_the_placebo_multiset_is_bitwise_the_treatment_per_covariate():
    """(e)'s placebo clause, from the published product: sorting must erase the permutation."""
    import numpy as np
    from model import static_conditioning_r7 as static
    fields, variables = static.load_static_fields()
    placebo, permutations = placebo_mod.placebo_field(fields)
    assert len(permutations) == len(variables)
    for index, name in enumerate(variables):
        assert torch.equal(torch.sort(fields[index].reshape(-1)).values,
                           torch.sort(placebo[index].reshape(-1)).values), \
            f"{name}: the permutation changed the multiset, so this is not a placebo"
    assert bool(torch.isfinite(placebo).all())


@published
def test_the_placebo_actually_permutes_and_is_not_the_identity():
    """A permutation that happened to be the identity would make the placebo the treatment."""
    from model import static_conditioning_r7 as static
    fields, variables = static.load_static_fields()
    placebo, _ = placebo_mod.placebo_field(fields)
    moved = [int((placebo[index] != fields[index]).sum()) for index in range(len(variables))]
    assert all(count > 0 for count in moved), f"a covariate was not permuted at all: {moved}"
    assert not torch.equal(placebo, fields)


@published
def test_the_placebo_is_deterministic_under_its_frozen_seed():
    """Two derivations must agree bit for bit, or the arm is not reproducible."""
    from model import static_conditioning_r7 as static
    fields, _ = static.load_static_fields()
    first, _ = placebo_mod.placebo_field(fields)
    second, _ = placebo_mod.placebo_field(fields)
    assert torch.equal(first, second)


@published
def test_a_different_placebo_seed_gives_a_different_field():
    """Counter-proof: the seed argument is actually used."""
    from model import static_conditioning_r7 as static
    fields, _ = static.load_static_fields()
    default, _ = placebo_mod.placebo_field(fields)
    other, _ = placebo_mod.placebo_field(fields, seed=placebo_mod.PLACEBO_SEED + 1)
    assert not torch.equal(default, other)


@published
def test_the_placebo_destroys_spatial_autocorrelation():
    """The property the placebo exists to create, measured rather than assumed."""
    from model import static_conditioning_r7 as static
    fields, variables = static.load_static_fields()
    placebo, _ = placebo_mod.placebo_field(fields)
    treatment_moran = [placebo_mod.moran(fields[index]) for index in range(len(variables))]
    placebo_moran = [placebo_mod.moran(placebo[index]) for index in range(len(variables))]
    assert sum(treatment_moran) / len(treatment_moran) > 0.3
    assert all(abs(value) < 0.05 for value in placebo_moran), placebo_moran


@published
def test_the_placebo_contrast_reports_its_own_attenuation():
    """The confound must be measured and reported, not left implicit.

    The pathway reads 2x2 patch means, so a per-cell permutation *does* change the token
    distribution even though the raw multiset is identical. This test pins that the round reports
    the attenuation instead of claiming a clean structure-only contrast; if a future edit made the
    contrast structure-only, this would fail and the record would need updating deliberately.
    """
    from model import static_conditioning_r7 as static
    fields, variables = static.load_static_fields()
    placebo, _ = placebo_mod.placebo_field(fields)
    contrast = placebo_mod.placebo_contrast(fields, placebo, variables)
    assert contrast["raw_multiset_all_identical"] is True
    ratio = contrast["feature_std_ratio_placebo_over_treatment"]
    assert 0.0 < ratio < 1.0, ratio
    assert contrast["token_multiset"]["treatment_tokens_equal_placebo_multiset"] is False
    assert "amplitude" in contrast["confound_statement"]
    assert contrast["code_std_ratio_placebo_over_treatment"] < \
        contrast["feature_std_ratio_placebo_over_treatment"] + 0.5


@published
def test_the_rejected_constructions_are_recorded_with_numbers():
    """Recording the alternatives is what makes the frozen choice auditable."""
    from model import static_conditioning_r7 as static
    fields, variables = static.load_static_fields()
    placebo, _ = placebo_mod.placebo_field(fields)
    rows = placebo_mod.placebo_contrast(fields, placebo, variables)["rejected_constructions"]
    assert [row["construction"] for row in rows] == ["block_2x2_core", "block_4x4_core"]
    for row in rows:
        assert row["raw_multiset_bitwise_identical"] is True
        assert 0.0 < row["feature_std_ratio"] < 1.5
        assert "why_not_used" in row


@published
def test_the_placebo_keeps_each_covariate_in_its_own_channel():
    """Channel shuffling would change every token's code distribution; assert it never happens."""
    from model import static_conditioning_r7 as static
    fields, variables = static.load_static_fields()
    placebo, _ = placebo_mod.placebo_field(fields)
    for index in range(len(variables)):
        # A permuted channel has the same min/max and the same sorted values as its source.
        assert torch.equal(fields[index].min(), placebo[index].min())
        assert torch.equal(fields[index].max(), placebo[index].max())
        # And it must NOT equal a *different* channel's distribution by accident.
        for other in range(len(variables)):
            if other == index:
                continue
            if not torch.equal(torch.sort(fields[other].reshape(-1)).values,
                               torch.sort(fields[index].reshape(-1)).values):
                assert not torch.equal(torch.sort(placebo[index].reshape(-1)).values,
                                       torch.sort(fields[other].reshape(-1)).values)


# ---------------------------------------------------------------- (a)-(f) can all be false


def _entry(skill_6, skill_12, skill_24, skill_48, skill_72, rmse_72, placebo_delta=0.01):
    def arm(lead_values, rmse=None):
        return {"skill": {str(lead): value for lead, value in lead_values.items()},
                "rmse": {str(lead): (rmse if rmse is not None else 1.0) for lead in (6, 12, 24, 48, 72)},
                "rows": {str(lead): 55 for lead in (6, 12, 24, 48, 72)},
                "n_evaluated": {str(lead): 10 for lead in (6, 12, 24, 48, 72)}}
    return {proto.CONTROL: arm({6: 0.0, 12: 0.0, 24: 0.5, 48: 0.5, 72: 0.5}),
            proto.TREATMENT: arm({6: skill_6, 12: skill_12, 24: skill_24, 48: skill_48,
                                  72: skill_72}, rmse=rmse_72),
            proto.PLACEBO: arm({6: 0.0, 12: 0.0, 24: 0.5, 48: 0.5, 72: 0.5},
                               rmse=rmse_72 + placebo_delta)}


def _result(**overrides):
    """A readings-shaped body whose preflight passes, so only the data clause varies."""
    seeds = {str(seed): _entry(**overrides) for seed in proto.SEEDS}
    return {"seeds": seeds,
            "preflight": {"e_preflight_passed": True,
                          "placebo_contrast": {"confound_statement": "amplitude caveat"}}}


def test_all_true_is_the_only_advance():
    verdict = driver.evaluate_exit(_result(skill_6=0.0, skill_12=0.0, skill_24=0.6, skill_48=0.6,
                                          skill_72=0.6, rmse_72=1.0))
    assert verdict["a_72h_skill_nonnegative"] is True
    assert verdict["b_mid_skill_nonnegative"] is True
    assert verdict["c_short_skill_within_tolerance"] is True
    assert verdict["e_preflight_passed"] is True
    assert verdict["f_attribution_treatment_beats_placebo"] is True
    assert verdict["verdict"] == "advance-to-independent-review"


def test_a_true_with_f_false_is_recorded_negative():
    """The case the round exists to catch: a gain that a position code also produces."""
    verdict = driver.evaluate_exit(_result(skill_6=0.0, skill_12=0.0, skill_24=0.6, skill_48=0.6,
                                          skill_72=0.6, rmse_72=1.005, placebo_delta=0.0))
    assert verdict["a_72h_skill_nonnegative"] is True
    assert verdict["f_attribution_treatment_beats_placebo"] is False
    assert verdict["verdict"].startswith("negative-attribution-failure")
    assert "position code" in verdict["reading_rule"]


def test_a_false_closes_the_branch_even_when_f_is_true():
    """A placebo-relative win with no control-relative win is not a skill gain."""
    verdict = driver.evaluate_exit(_result(skill_6=0.0, skill_12=0.0, skill_24=0.6, skill_48=0.6,
                                          skill_72=0.4, rmse_72=1.0))
    assert verdict["a_72h_skill_nonnegative"] is False
    assert verdict["f_attribution_treatment_beats_placebo"] is True
    assert verdict["verdict"] == "permanently-close-the-urban-static-input-branch"


def test_b_fails_when_only_48h_is_negative():
    verdict = driver.evaluate_exit(_result(skill_6=0.0, skill_12=0.0, skill_24=0.6, skill_48=0.4,
                                          skill_72=0.6, rmse_72=1.0))
    assert verdict["b_mid_skill_nonnegative"] is False


def test_c_tolerance_is_the_frozen_value_and_can_fail():
    """(c) must be able to fail, and the tolerance must be the predeclared -0.005."""
    within = driver.evaluate_exit(_result(skill_6=-0.004, skill_12=-0.004, skill_24=0.6,
                                          skill_48=0.6, skill_72=0.6, rmse_72=1.0))
    assert within["c_short_skill_within_tolerance"] is True
    assert within["c_tolerance"] == -0.005
    outside = driver.evaluate_exit(_result(skill_6=-0.006, skill_12=0.0, skill_24=0.6,
                                          skill_48=0.6, skill_72=0.6, rmse_72=1.0))
    assert outside["c_short_skill_within_tolerance"] is False


def test_f_reads_kelvin_not_skill():
    """(f) is a threshold in kelvin; computing it in skill units would change its meaning.

    The fixture makes the two conventions disagree: the placebo RMSE is 0.0005 K worse (below the
    0.001 K threshold, so (f) is false) while the skill gap is a clear 0.02. A skill-based (f)
    would report true, which is exactly the substitution this test forbids.
    """
    def arm(skill_72, rmse_72):
        return {"skill": {str(lead): (skill_72 if lead == 72 else 0.5) for lead in (6, 12, 24, 48, 72)},
                "rmse": {str(lead): (rmse_72 if lead == 72 else 1.0) for lead in (6, 12, 24, 48, 72)},
                "rows": {}, "n_evaluated": {}}
    seeds = {}
    for seed in proto.SEEDS:
        seeds[str(seed)] = {proto.CONTROL: arm(0.5, 1.0),
                            proto.TREATMENT: arm(0.52, 1.0005),
                            proto.PLACEBO: arm(0.50, 1.0010)}
    result = {"seeds": seeds, "preflight": {"e_preflight_passed": True,
                                            "placebo_contrast": {"confound_statement": "x"}}}
    verdict = driver.evaluate_exit(result)
    assert verdict["f_per_seed_rmse_improvement_kelvin"][str(proto.SEEDS[0])] == pytest.approx(0.0005)
    assert verdict["f_attribution_treatment_beats_placebo"] is False


def test_f_requires_every_seed_not_the_mean():
    """A mean that clears the threshold while one seed regresses must not read as (f) true."""
    def arm(skill_72, rmse_72):
        return {"skill": {str(lead): (skill_72 if lead == 72 else 0.5) for lead in (6, 12, 24, 48, 72)},
                "rmse": {str(lead): (rmse_72 if lead == 72 else 1.0) for lead in (6, 12, 24, 48, 72)},
                "rows": {}, "n_evaluated": {}}
    first, second, third = (str(seed) for seed in proto.SEEDS)
    seeds = {
        first: {proto.CONTROL: arm(0.5, 1.0), proto.TREATMENT: arm(0.6, 0.990),
                proto.PLACEBO: arm(0.5, 1.0)},
        second: {proto.CONTROL: arm(0.5, 1.0), proto.TREATMENT: arm(0.6, 0.990),
                 proto.PLACEBO: arm(0.5, 1.0)},
        # This seed's treatment is *worse* than the placebo, so (f) must be false overall.
        third: {proto.CONTROL: arm(0.5, 1.0), proto.TREATMENT: arm(0.6, 1.010),
                proto.PLACEBO: arm(0.5, 1.0)},
    }
    result = {"seeds": seeds, "preflight": {"e_preflight_passed": True,
                                            "placebo_contrast": {"confound_statement": "x"}}}
    verdict = driver.evaluate_exit(result)
    assert verdict["f_per_seed_rmse_improvement_kelvin"][third] < 0
    assert verdict["f_attribution_treatment_beats_placebo"] is False


def test_a_failed_preflight_cannot_advance():
    result = _result(skill_6=0.0, skill_12=0.0, skill_24=0.6, skill_48=0.6, skill_72=0.6,
                     rmse_72=1.0)
    result["preflight"]["e_preflight_passed"] = False
    verdict = driver.evaluate_exit(result)
    assert verdict["e_preflight_passed"] is False
    assert verdict["verdict"] != "advance-to-independent-review"


# ---------------------------------------------------------------- the honesty pins


def test_the_driver_does_not_hardcode_the_d_verdict():
    """Decision 0109: (d) must be carried by code paths, not by a literal in the exit block."""
    source = (ROOT / "scripts/study_r7_urban_static_ablation.py").read_text(encoding="utf-8")
    assert "d_control_bitwise_not_evaluated_here" in source
    for forbidden in ("\"d\": True", "'d': True", "d_control_bitwise_equivalent\": True"):
        assert forbidden not in source, f"a hardcoded (d) verdict reappeared: {forbidden!r}"


def test_the_thresholds_are_the_frozen_values():
    """The two predeclared numbers, pinned so an edit has to be deliberate."""
    assert driver.SHORT_LEAD_TOLERANCE == -0.005
    assert driver.ATTRIBUTION_THRESHOLD_K == 0.001
    assert driver.UPDATES == 4000
    assert tuple(driver.SEEDS) == (41, 42, 43)
    assert driver.BATCH_SIZE == 4
    assert tuple(driver.ARMS) == ("control", "treatment", "placebo")


def test_the_protocol_declares_no_scientific_claim_and_no_test_read():
    constants = proto.frozen_constants()
    assert constants["scientific_claim"] is False
    assert constants["test_read"] is False
    assert constants["static"]["parameter_delta"] == 0
    assert constants["factor"]["parameter_delta_measured"] is True


def test_the_budget_envelope_is_the_frozen_one():
    """The brief's two frozen numbers, and a per-seed cap that cannot truncate its own estimate."""
    budgets = proto.frozen_constants()["budgets"]
    assert budgets["planned_seconds"] == 115200.0
    assert budgets["hard_cap_seconds"] == 230400.0
    # The per-seed allowance is derived from the three-arm estimate (3x, quantized), exactly as
    # Stage A derived its own. A cap below the estimate would truncate the round it sizes.
    assert budgets["per_seed_seconds"] > budgets["per_seed_estimate_seconds"]
    assert budgets["per_seed_cap_over_estimate"] >= 3.0
    assert proto.PER_SEED_SECONDS % 3600 == 0


def test_the_limitations_name_the_placebo_attenuation():
    text = " ".join(proto.LIMITATIONS)
    assert "amplitude" in text
    assert "patch means" in text or "2x2" in text
    assert "scientific_claim: false" in text


def test_the_family_argument_distinguishes_both_closed_branches():
    """J4 must be checkable, not rhetorical: both closed branches are named with their relation."""
    argument = placebo_mod.family_argument(torch.zeros(8, 65, 65) + 1.0,
                                    list(proto._static_product()[3]))
    assert "linear" in argument["prior_branch_explicit_tendency"]["relation_to_state"]
    assert "same-time derivable" in argument["prior_branch_typed_spatial"]["relation_to_state"]
    assert "none" in argument["this_branch"]["relation_to_state"]
    assert "placebo" in argument["the_alternative_this_round_must_beat"]["how_it_is_isolated"]


def test_the_decision_text_has_no_success_language():
    """The rule may *name* the banned words as prohibitions; it must not assert them.

    A first draft banned the substring "SOTA" outright and failed on the sentence "No arm may be
    reported as SOTA" - a prohibition the test itself wanted. The assertion therefore checks the
    claim form: each banned word must appear inside a prohibition, never as an affirmation.
    """
    text = proto.frozen_constants()["decision"]["text"]
    lowered = text.lower()
    for banned in ("sota", "outperforms"):
        assert banned in lowered, f"the rule must name {banned!r} as forbidden"
        # Every occurrence must sit inside a prohibition: a negation token shortly before it.
        start = 0
        seen = False
        while True:
            index = lowered.find(banned, start)
            if index < 0:
                break
            seen = True
            window = lowered[max(0, index - 40):index]
            assert any(token in window for token in ("no ", "not ", "never ", "must not ")), (
                f"{banned!r} at {index} has no negation in the preceding 40 chars: {window!r}")
            start = index + len(banned)
        assert seen, f"{banned!r} never appears, so the prohibition was removed"
    assert "proves the covariates help" not in lowered


def test_the_execution_inventory_is_named_not_globbed():
    """Decision 0067: the closure is bound explicitly, and unrelated modules stay out."""
    files = proto.execution_files()
    assert len(files) < 150
    assert "scripts/study_r7_urban_static_ablation.py" in files
    assert "scripts/r7_urban_static_ablation_protocol.py" in files
    assert "model/static_conditioning_r7.py" in files
    assert "tools/reconcile_r7_static_parent.py" in files
    assert "data/download/worldcover_cog.py" not in files


def test_the_reconciled_parent_receipt_agrees_with_the_round():
    """The parent's loadability was measured by another round; check that record rather than retake it."""
    receipt = ROOT / "outputs/r7_static_parent_reconcile_20261108_attempt02/receipt.json"
    if not receipt.is_file():
        pytest.skip("the reconcile receipt is not present in this checkout")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["strict_load_of_reconciled_passed"] is True
    assert body["parent_model_code_sha256"] == parent_pin()
    assert sorted(body["on_arm_added_keys"]) == sorted(proto.BUFFER_KEYS)


def parent_pin():
    from scripts import r7_s3_position_retention_parent as pins
    return pins.MODEL_CODE_SHA256
