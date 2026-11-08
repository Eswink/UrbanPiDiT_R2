"""Counterproofs for the Stage E1b protocol layer and its reading path; CPU only, no store.

The properties that decide whether this round means anything, each asserted exactly:

* the two arms are the **two declared supervision modes** and nothing else (a third mode appearing
  in `training/` must fail here rather than be silently ignored);
* `final_only` **must not claim internal deep supervision** -- the contract's own description of
  itself is the thing a mislabelled arm would get wrong;
* the **weight vector is shared**, so E1b tests organisation and not allocation (that is E1a);
* the reading path fills both arms and the (a)-(d) conjunction can reach either verdict.
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

proto = importlib.import_module("scripts.r7_s3_e1b_terminal_step_supervision_protocol")
driver = importlib.import_module("scripts.study_r7_s3_e1b_terminal_step_supervision")


def test_the_arms_are_exactly_the_two_declared_modes():
    """A third supervision mode appearing upstream must fail here, not be quietly ignored."""
    from training.r7_recursive_losses import DRAFT_SUPERVISION_MODES
    assert set(DRAFT_SUPERVISION_MODES) == {proto.CONTROL, proto.CANDIDATE}
    assert proto.CONTROL == "deep" and proto.CANDIDATE == "final_only"


def test_final_only_does_not_claim_internal_deep_supervision():
    """The mode's own description of itself: the field a mislabelled arm would falsify."""
    constants = proto.frozen_constants()
    assert constants["arms"][proto.CANDIDATE]["internal_deep_supervision"] is False
    assert constants["arms"][proto.CONTROL]["internal_deep_supervision"] is True


def test_the_weight_vector_is_shared_between_arms():
    """E1b moves organisation only; a differing vector would make it E1a plus E1b."""
    from scripts import study_r7_s3_wide_single_factor as wide
    constants = proto.frozen_constants()
    assert constants["recipe"]["physical_weights"] == list(wide.PHYSICAL_WEIGHTS)
    assert constants["arms"][proto.CANDIDATE] != constants["arms"][proto.CONTROL]


def test_the_declared_factor_is_draft_organisation_alone():
    constants = proto.frozen_constants()
    assert constants["factor"]["arms_differ_only_in"] == ["draft_supervision"]
    assert constants["scientific_claim"] is False
    assert constants["test_read"] is False
    assert constants["r"] == 0


def test_the_regime_matches_stage_b_s_sixteen_epoch_point():
    constants = proto.frozen_constants()
    assert constants["recipe"]["updates"] == 128528
    assert constants["usable_windows"] == 32131
    assert abs(constants["recipe"]["epochs"] - 16.000498) < 1e-6


def test_an_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        proto.arm_supervision("neither")


def test_the_envelope_covers_both_arms_and_is_described_as_an_estimate():
    budgets = proto.budgets()
    assert budgets["per_seed_seconds"] > budgets["two_arms_seconds_estimate"]
    assert budgets["hard_cap_seconds"] == 2 * budgets["per_seed_seconds"]
    assert "not a measurement" in budgets["estimate_basis"].lower()


def _receipt_tree(tmp_path, climatology):
    """A two-seed receipt tree whose tables carry exactly the five frozen leads."""
    import csv, hashlib
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
                with (folder / "boundary_rmse.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["region", "variable", "rmse"])
                    writer.writeheader()
                    writer.writerow({"region": "full", "variable": "t2m", "rmse": "2.0"})
                evaluations[str(lead)] = {
                    "dir": str(folder),
                    "boundary_rmse_csv_sha256": sha(folder / "boundary_rmse.csv")}
            arms[arm] = {"training_dir": str(folder_training), "evaluations": evaluations,
                         "draft_supervision": proto.arm_supervision(arm)}
        (tmp_path / f"seed{seed}_receipt.json").write_text(
            json.dumps({"seed": seed, "arms": arms}), encoding="utf-8")
    constants = proto.frozen_constants()
    return {"protocol_sha256": "0" * 64, "test_read": False,
            "arms": {arm: constants["arms"][arm] for arm in (proto.CONTROL, proto.CANDIDATE)},
            "climatology_reference": climatology}


def test_the_reading_fills_both_arms_and_the_exit_can_pass(tmp_path):
    """The attempt04 class of defect: a fill nested in an arm-conditional leaves one arm empty."""
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


def test_a_mislabelled_arm_closes_the_branch(tmp_path):
    """If the candidate arm trained deep, (b) must fail -- otherwise (b) asserts nothing."""
    climatology = {"leads": {str(lead): {"t2m_rmse": 4.0} for lead in driver.LEADS}}
    body = _receipt_tree(tmp_path, climatology)
    receipt = json.loads((tmp_path / f"seed{proto.SEEDS[0]}_receipt.json").read_text())
    receipt["arms"][proto.CANDIDATE]["draft_supervision"] = proto.DRAFT_SUPERVISION_DEEP
    (tmp_path / f"seed{proto.SEEDS[0]}_receipt.json").write_text(json.dumps(receipt))
    result = driver.readings(tmp_path, body)
    assert result["exit_conditions"]["b_factor_effective"] is False
    assert result["exit_conditions"]["all"] is False
    assert result["exit_conditions"]["verdict"] == \
        "permanently-close-final-only-supervision-in-this-regime"


def test_final_only_cuts_the_intermediate_drafts_gradient_exactly():
    """The factor is real, measured by gradient rather than inferred from the branch name.

    Reading ``recursive_losses`` shows a separate ``final_only`` branch, but a branch that merely
    *looks* different would leave a round comparing two identical objectives. So this asserts the
    property that makes the comparison meaningful: under ``final_only`` every non-final draft
    receives an *exact* zero gradient and only the last one moves, while under the registered
    ``deep`` form all five do. Exact zeros, not a tolerance -- the branch genuinely skips them.
    """
    import torch
    from training.r7_recursive_losses import (DRAFT_SUPERVISION_DEEP,
                                              DRAFT_SUPERVISION_FINAL_ONLY,
                                              deep_supervised_forecast_mse)
    torch.manual_seed(0)
    batch, steps, channels, height, width = 2, 5, 3, 4, 4
    target = torch.randn(batch, channels, height, width)
    drafts = target.unsqueeze(1).repeat(1, steps, 1, 1, 1) + 0.1 * torch.randn(
        batch, steps, channels, height, width)
    latitude = torch.linspace(50.0, 45.0, height)

    def per_draft_gradient(mode):
        leaf = drafts.clone().requires_grad_(True)
        deep_supervised_forecast_mse(leaf, target, latitude, draft_supervision=mode).backward()
        return leaf.grad.abs().sum(dim=(0, 2, 3, 4))

    final = per_draft_gradient(DRAFT_SUPERVISION_FINAL_ONLY)
    assert bool(final[:-1].sum() == 0), f"intermediate drafts moved: {final.tolist()}"
    assert float(final[-1]) > 0
    deep = per_draft_gradient(DRAFT_SUPERVISION_DEEP)
    assert bool((deep > 0).all()), f"the registered form should touch every draft: {deep.tolist()}"
    # And the two objectives are numerically different on the same inputs.
    a = deep_supervised_forecast_mse(drafts, target, latitude,
                                     draft_supervision=DRAFT_SUPERVISION_DEEP)
    b = deep_supervised_forecast_mse(drafts, target, latitude,
                                     draft_supervision=DRAFT_SUPERVISION_FINAL_ONLY)
    assert not bool(torch.isclose(a, b))
