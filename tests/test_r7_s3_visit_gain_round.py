"""Counter-proofs for the G1-B round machinery (decision 0119).

Two things are worth testing mechanically, and they are different in kind.

The **verdict** is pure arithmetic over receipts, so a synthetic arm pair can drive every
branch: a sign flip, a missing arm, a tie, a non-finite value. The rule that matters is that
none of those ever becomes a *pass*, and none of them becomes ``false`` when the honest
answer is ``null``.

The **protocol and driver** are checked structurally: the gains are the frozen ones, the
uniform arm is not the visit arm's mean, a switched-on cell cannot load as a switched-off
one, and the frozen envelope covers what its own treated rate requires. Those are the
claims a later edit could quietly break while every number still looked plausible.
"""
from __future__ import annotations

import csv
import json
import math

import pytest

from scripts import r7_s3_visit_gain_protocol as proto
from scripts import r7_s3_visit_gain_verdict as verdict

SEEDS = proto.SEEDS


# ------------------------------------------------------------------ gains and envelope

def test_frozen_gains_are_the_decision_0119_forms():
    assert proto.visit_gains(proto.VISIT_DAMPED) == pytest.approx(
        [1.0, 1 / math.sqrt(2), 1 / math.sqrt(3), 0.5])
    assert proto.visit_gains(proto.UNIFORM_DAMPED) == pytest.approx([1 / math.sqrt(2.5)] * 4)


def test_uniform_arm_is_not_the_visit_arms_mean_gain():
    table = proto.gain_table()
    assert table["equal_mean_damping"] is False
    assert table["uniform_constant"] < table["visit_mean"]
    assert table["uniform_relative_shortfall"] == pytest.approx(0.09145, abs=5e-5)


def test_envelope_covers_what_its_own_rate_requires():
    """Recomputed, not compared to the constant it was derived from (decision 0116's lesson)."""
    measured = proto.envelope()
    assert measured["sufficient"] is True
    assert measured["margin"] >= proto.ENVELOPE_MARGIN
    assert measured["hard_cap_covers_waves"] is True


def test_a_slower_rate_would_make_the_frozen_envelope_insufficient():
    """The guard must be able to fail: if it cannot, it is not a guard."""
    need = proto.UPDATES * proto.TREATED_SECONDS_PER_UPDATE + proto.MEASURED_SECONDS_PER_EVALUATION
    slower = proto.PER_SEED_SECONDS / (len(proto.ARMS) * need) * proto.TREATED_SECONDS_PER_UPDATE
    too_slow_need = len(proto.ARMS) * (proto.UPDATES * slower * 1.5
                                       + proto.MEASURED_SECONDS_PER_EVALUATION)
    assert proto.PER_SEED_SECONDS < proto.ENVELOPE_MARGIN * too_slow_need


def test_region_is_full_with_the_equivalence_note():
    assert proto.REGION == "full"
    assert proto.BOUNDARY_MARGINS == (1,)
    assert "interior_32" in proto.REGION_NOTE


def test_interior_32_is_degenerate_on_this_instance():
    """The recorded reason the region is ``full``: 65 - 2*32 leaves one cell."""
    from training.r7_boundary_metrics import boundary_masks

    masks = dict((name, int(mask.sum())) for name, _, mask in boundary_masks(65, 65, (32,)))
    assert masks["interior_32"] == 1
    wide = dict((name, int(mask.sum())) for name, _, mask in boundary_masks(129, 129, (32,)))
    assert wide["interior_32"] == 4225


# ------------------------------------------------------------------------ arm specs

def test_both_arms_bind_their_gain_into_the_parent_spec():
    for arm, mode in proto.ARMS:
        spec = proto.arm_spec(arm, parent_spec=proto.registered_parent_spec())
        assert spec["visit_gain"] == mode
        assert int(spec["dim"]) == 192 and int(spec["depth"]) == 4
        assert int(spec["default_reasoning_steps"]) == proto.REASONING_STEPS


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        proto.arm_spec("visit_exponential")


def test_switched_on_cell_is_not_loadable_as_the_switched_off_arm():
    """Exit 4's premise: the gain mode is part of the module's identity fields.

    If the mode were not a module field, a damped checkpoint would pass the parent's
    identity check and the two arms would be indistinguishable at load time.
    """
    from model.recursive_weather_r7 import GenericRecursiveCell

    off = GenericRecursiveCell(32, 4, visit_gain=None)
    on = GenericRecursiveCell(32, 4, visit_gain=proto.VISIT_DAMPED)
    assert off.visit_gain is None and on.visit_gain == proto.VISIT_DAMPED
    assert set(off.state_dict()) == set(on.state_dict())


# -------------------------------------------------------------------- verdict: exit 1

def _exits(**overrides):
    deltas = {"72": {"41": -0.001, "42": -0.002, "43": -0.003},
              "12": {"41": -0.001, "42": 0.001, "43": -0.001},
              "6": {"41": -0.001, "42": 0.001, "43": -0.001}}
    deltas.update(overrides.pop("deltas", {}))
    return deltas


def test_exit_one_passes_only_when_every_seed_is_negative():
    result = verdict.exit_one({"41": -0.001, "42": -0.002, "43": -0.003})
    assert result["value"] is True
    assert set(result["signs"].values()) == {-1}


def test_exit_one_fails_on_one_sign_flip():
    result = verdict.exit_one({"41": -0.001, "42": 0.002, "43": -0.003})
    assert result["value"] is False


def test_exit_one_fails_on_a_tie():
    """A zero delta is not negative; a tie must not pass as agreement."""
    result = verdict.exit_one({"41": -0.001, "42": 0.0, "43": -0.003})
    assert result["value"] is False


def test_exit_one_is_null_when_a_seed_is_missing():
    result = verdict.exit_one({"41": -0.001, "42": -0.002})
    assert result["value"] is None


def test_exit_one_is_null_on_a_non_finite_delta():
    result = verdict.exit_one({"41": -0.001, "42": float("nan"), "43": -0.003})
    assert result["value"] is None


# -------------------------------------------------------------------- verdict: exit 2

def test_exit_two_judges_the_seed_mean_not_each_seed():
    """One seed may flip; that is what separates exit 2 from a weaker exit 1."""
    result = verdict.exit_two(_exits())
    assert result["value"] is True
    assert result["leads"]["6"]["deltas"]["42"] > 0


def test_exit_two_fails_when_a_lead_mean_is_positive():
    deltas = _exits(deltas={"6": {"41": 0.01, "42": 0.01, "43": -0.001}})
    result = verdict.exit_two(deltas)
    assert result["value"] is False
    assert result["leads"]["6"]["passed"] is False
    assert result["leads"]["12"]["passed"] is True


def test_exit_two_is_null_when_an_arm_is_missing():
    deltas = _exits()
    deltas["12"] = {"41": float("nan"), "42": 0.001, "43": -0.001}
    assert verdict.exit_two(deltas)["value"] is None


# ------------------------------------------------------------------ verdict: exits 3/4

def _identity(**override):
    body = {
        "off_path": {"bitwise_equal_to_pre_switch_expression": True, "key_set_equal": True,
                     "tensors_equal": True, "parameter_count_equal": True,
                     "no_new_buffers": True},
        "visit_damped_k0": {"bitwise_equal": True, "gain_at_zero": 1.0},
    }
    body.update(override)
    return body


def test_exit_three_passes_on_complete_positive_evidence():
    assert verdict.exit_three(_identity())["value"] is True


def test_exit_three_fails_if_only_the_off_path_is_inert():
    """A's k=0 bridge is a separate claim; passing 3a does not imply it."""
    assert verdict.exit_three(_identity(
        visit_damped_k0={"bitwise_equal": False}))["value"] is False


def test_exit_three_fails_on_a_single_false_off_path_field():
    broken = _identity()
    broken["off_path"] = {**broken["off_path"], "no_new_buffers": False}
    assert verdict.exit_three(broken)["value"] is False


def test_exit_three_is_null_on_incomplete_evidence():
    incomplete = _identity()
    del incomplete["off_path"]["tensors_equal"]
    assert verdict.exit_three(incomplete)["value"] is None


def test_exit_four_requires_every_seed_bitwise():
    migration = {"seeds": {str(seed): {"state_digest_equal": True, "key_set_equal": True,
                                       "bitwise_identical": True, "tensor_count": 155,
                                       "source_matches_frozen_pin": True}
                           for seed in SEEDS}}
    assert verdict.exit_four(migration)["value"] is True
    migration["seeds"]["42"]["state_digest_equal"] = False
    assert verdict.exit_four(migration)["value"] is False
    del migration["seeds"]["43"]
    assert verdict.exit_four(migration)["value"] is None


# ---------------------------------------------------------------------- branch status

def _receipts(a_rmse, b_rmse):
    """Two arms' receipts whose written CSVs carry the given RMSEs."""
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    receipts = {}
    for seed in SEEDS:
        pair = {}
        for arm, rmse in ((proto.VISIT_DAMPED, a_rmse), (proto.UNIFORM_DAMPED, b_rmse)):
            evaluations = {}
            for index, lead in enumerate(proto.LEADS):
                folder = root / f"seed{seed}/{arm}/lead_{lead:03d}h"
                folder.mkdir(parents=True, exist_ok=True)
                table = folder / "boundary_rmse.csv"
                with table.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["region", "margin_cells", "n_grid_points",
                                     "full_area_fraction", "lead_hours", "variable", "unit",
                                     "rmse", "n_initializations"])
                    writer.writerow(["full", 0, 4225, 1.0, lead, "t2m", "K",
                                     rmse[seed][index], 1447])
                evaluations[str(lead)] = {"dir": str(folder)}
            pair[arm] = {"seed": seed, "arm": arm, "evaluations": evaluations}
        receipts[str(seed)] = pair
    return receipts


def _rmse_table(a, b):
    return ({seed: [a + 0.01 * seed + 0.001 * i for i in range(len(proto.LEADS))]
             for seed in SEEDS},
            {seed: [b + 0.01 * seed + 0.001 * i for i in range(len(proto.LEADS))]
             for seed in SEEDS})


def test_branch_is_open_when_every_exit_passes():
    a, b = _rmse_table(3.0, 3.05)
    migration = {"seeds": {str(seed): {"state_digest_equal": True, "key_set_equal": True,
                                       "bitwise_identical": True, "tensor_count": 155,
                                       "source_matches_frozen_pin": True}
                           for seed in SEEDS}}
    body = verdict.readings(_receipts(a, b), _identity(), migration, protocol_sha256="0" * 64)
    assert body["exits"]["one"]["value"] is True
    assert body["exits"]["two"]["value"] is True
    assert body["branch_status"] == "open"
    assert body["scientific_claim"] is False and body["r"] == 0


def test_branch_closes_when_exit_one_flips_on_one_seed():
    a, b = _rmse_table(3.0, 3.05)
    a[42] = [value + 0.5 for value in a[42]]       # seed 42 now worse in A
    migration = {"seeds": {str(seed): {"state_digest_equal": True, "key_set_equal": True,
                                       "bitwise_identical": True, "tensor_count": 155,
                                       "source_matches_frozen_pin": True}
                           for seed in SEEDS}}
    body = verdict.readings(_receipts(a, b), _identity(), migration, protocol_sha256="0" * 64)
    assert body["exits"]["one"]["value"] is False
    assert body["branch_status"] == "closed"


def test_branch_is_null_when_an_arm_never_wrote_a_receipt():
    """A missing arm is untested. Recording false would claim a measured failure."""
    a, b = _rmse_table(3.0, 3.05)
    receipts = _receipts(a, b)
    del receipts["43"][proto.VISIT_DAMPED]
    migration = {"seeds": {str(seed): {"state_digest_equal": True, "key_set_equal": True,
                                       "bitwise_identical": True, "tensor_count": 155,
                                       "source_matches_frozen_pin": True}
                           for seed in SEEDS}}
    body = verdict.readings(receipts, _identity(), migration, protocol_sha256="0" * 64)
    assert body["exits"]["one"]["value"] is None
    assert body["branch_status"] == "null"


def test_readings_refuses_to_be_rewritten(tmp_path):
    body = {"scientific_claim": False}
    verdict.write_readings(tmp_path, body)
    with pytest.raises(FileExistsError):
        verdict.write_readings(tmp_path, body)


def test_delta_is_the_only_sign_convention():
    assert verdict.delta(3.0, 3.5) < 0      # A lower -> negative -> A better
    assert verdict.delta(3.5, 3.0) > 0
