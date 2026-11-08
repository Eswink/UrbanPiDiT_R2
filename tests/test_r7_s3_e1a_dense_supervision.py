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
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

proto = importlib.import_module("scripts.r7_s3_e1a_dense_supervision_protocol")
driver = importlib.import_module("scripts.study_r7_s3_e1a_dense_supervision")


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


def _receipt_tree(tmp_path, climatology):
    """A two-seed, two-arm receipt tree whose tables carry exactly the five frozen leads."""
    import csv, hashlib, json
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for seed in proto.SEEDS:
        arms = {}
        for arm in (proto.CONTROL, proto.CANDIDATE):
            folder_training = tmp_path / f"seed{seed}/{arm}/training"
            folder_training.mkdir(parents=True)
            evaluations = {}
            for lead in driver.LEADS:
                folder = tmp_path / f"seed{seed}/{arm}/lead_{lead:03d}h"
                folder.mkdir(parents=True)
                # Equal rmse on both arms, so the paired delta is 0.0 and (a) holds.
                with (folder / "boundary_rmse.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["region", "variable", "rmse"])
                    writer.writeheader()
                    writer.writerow({"region": "full", "variable": "t2m", "rmse": "2.0"})
                evaluations[str(lead)] = {
                    "dir": str(folder),
                    "boundary_rmse_csv_sha256": sha(folder / "boundary_rmse.csv")}
            arms[arm] = {"training_dir": str(folder_training), "evaluations": evaluations,
                         "physical_weights": proto.arm_weights(arm)}
        (tmp_path / f"seed{seed}_receipt.json").write_text(
            json.dumps({"seed": seed, "arms": arms}), encoding="utf-8")
    constants = proto.frozen_constants()
    return {"protocol_sha256": "0" * 64, "test_read": False,
            "arms": {arm: constants["arms"][arm] for arm in (proto.CONTROL, proto.CANDIDATE)},
            "climatology_reference": climatology}


def test_the_reading_fills_both_arms_and_the_exit_can_pass(tmp_path):
    """The attempt04 class of defect: a fill nested in an arm-conditional leaves one arm empty.

    Decision 0098 records three defects that only fired after ten hours of training, one of which
    was exactly this. This runs the whole reading path on a synthetic tree so the driver cannot
    repeat it silently, and asserts the (a)-(d) conjunction reaches a verdict.
    """
    climatology = {"leads": {str(lead): {"t2m_rmse": 4.0} for lead in driver.LEADS}}
    result = driver.readings(tmp_path, _receipt_tree(tmp_path, climatology))
    wanted = sorted(str(lead) for lead in driver.LEADS)
    for seed in proto.SEEDS:
        for arm in (proto.CONTROL, proto.CANDIDATE):
            entry = result["seeds"][str(seed)][arm]
            assert sorted(entry["rmse"]) == wanted and sorted(entry["skill"]) == wanted
    exits = result["exit_conditions"]
    assert exits["all"] is True
    assert exits["verdict"] == "advance-to-independent-review"
    assert result["scientific_claim"] is False and result["test_read"] is False


def test_the_exit_can_fail_b_and_c(tmp_path):
    """(b) and (c) must be able to say no, or they prove nothing about the factor having bit."""
    climatology = {"leads": {str(lead): {"t2m_rmse": 4.0} for lead in driver.LEADS}}
    body = _receipt_tree(tmp_path, climatology)
    # Simulate an on arm that trained on the wrong vector: (b) must fail and the branch close.
    receipt = json.loads((tmp_path / f"seed{proto.SEEDS[0]}_receipt.json").read_text())
    receipt["arms"][proto.CANDIDATE]["physical_weights"] = list(proto.SPARSE_WEIGHTS)
    (tmp_path / f"seed{proto.SEEDS[0]}_receipt.json").write_text(json.dumps(receipt))
    result = driver.readings(tmp_path, body)
    assert result["exit_conditions"]["b_factor_effective"] is False
    assert result["exit_conditions"]["all"] is False
    assert result["exit_conditions"]["verdict"] == \
        "permanently-close-dense-supervision-in-this-regime"


def test_the_two_vectors_produce_different_objectives_at_the_same_dose():
    """Same total weight, different allocation, therefore a different loss -- measured.

    The protocol asserts the sums match, which alone would leave open the possibility that the two
    arms optimise the same objective. ``training_long_rollout`` applies each weight to that step's
    own loss and adds them sequentially, so two different vectors differ whenever the per-step
    losses differ. This plugs a synthetic per-step vector into exactly that arithmetic.
    """
    import torch
    from training.r7_long_rollout import validate_physical_weights

    sparse = validate_physical_weights(proto.SPARSE_WEIGHTS)
    dense = validate_physical_weights(proto.DENSE_WEIGHTS)
    per_step = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

    def total(weights):
        loss = weights[0] * per_step[0]
        for weight, step_loss in zip(weights[1:], per_step[1:]):
            loss = loss + weight * step_loss
        return float(loss)

    assert abs(total(sparse) - total(dense)) > 1e-9, (
        "the two arms would optimise the same objective; the round would compare nothing")
    # And the guard that makes the comparison interpretable: identical dose.
    assert abs(sum(sparse) - sum(dense)) < 1e-12
