"""Counterproofs for the Stage E1a protocol layer; CPU only, no store, no GPU, no claim.

Three properties decide whether this round is a single-factor test at all, so each is asserted
with exact equality rather than a tolerance, and each has a counterproof that the assertion can
fail:

* the treatment and control **carry the same total weight** (otherwise the round changes
  allocation *and* dose, and a win cannot be attributed to either);
* the vectors **actually differ** (otherwise the round is two identical arms);
* the regime is **16 epoch at 128,528 updates**, which is what makes reopening the closed arm
  legitimate under decision 0057 rather than a renamed retry.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

proto = importlib.import_module("scripts.r7_s3_e1a_dense_supervision_protocol")


def test_the_two_vectors_carry_the_same_total_weight():
    """Fixed dose, varying allocation: the definition of this round's single factor."""
    assert abs(sum(proto.DENSE_WEIGHTS) - sum(proto.SPARSE_WEIGHTS)) < 1e-12
    assert abs(sum(proto.DENSE_WEIGHTS) - 3.0) < 1e-12


def test_the_two_vectors_actually_differ():
    """A counterproof against the embarrassing case: two identical arms compared to each other."""
    assert list(proto.DENSE_WEIGHTS) != list(proto.SPARSE_WEIGHTS)
    assert proto.arm_weights(proto.CANDIDATE) != proto.arm_weights(proto.CONTROL)


def test_both_vectors_cover_the_twelve_physical_steps():
    """The runner indexes twelve 6-hour transitions; a short vector would fail at step 12."""
    assert len(proto.DENSE_WEIGHTS) == 12 and len(proto.SPARSE_WEIGHTS) == 12


def test_the_control_is_the_registered_vector_verbatim():
    """The control must be the repository's registered vector, not a retyped lookalike."""
    from scripts import study_r7_s3_wide_single_factor as wide
    assert list(proto.SPARSE_WEIGHTS) == list(wide.PHYSICAL_WEIGHTS)


def test_the_regime_is_sixteen_epochs_of_the_frozen_store():
    """128,528 updates over 32,131 usable windows at batch 4 is 16.000498 epoch."""
    constants = proto.frozen_constants()
    assert constants["recipe"]["updates"] == 128528
    assert constants["recipe"]["batch_size"] == 4
    assert constants["usable_windows"] == 32131
    assert abs(constants["recipe"]["epochs"] - 16.000498) < 1e-6


def test_an_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        proto.arm_weights("neither")


def test_the_envelope_covers_both_arms_of_a_seed():
    """A seed trains two arms; an envelope below that would truncate the round (decision 0090)."""
    budgets = proto.budgets()
    assert budgets["per_seed_seconds"] > budgets["two_arms_seconds_estimate"]
    assert budgets["hard_cap_seconds"] == 2 * budgets["per_seed_seconds"]
    assert budgets["six_arm_gpu_hours_estimate"] > 0


def test_the_declared_factor_is_the_weight_vector_alone():
    constants = proto.frozen_constants()
    assert constants["factor"]["arms_differ_only_in"] == ["physical_weights"]
    assert constants["scientific_claim"] is False
    assert constants["test_read"] is False
    assert constants["r"] == 0


def test_the_budget_estimate_is_not_passed_off_as_a_measurement():
    """The estimate basis must say so in words; an estimate silently read as measured misleads."""
    basis = proto.budgets()["estimate_basis"]
    assert "estimate" in basis.lower()
    assert "not a measurement" in basis.lower()
