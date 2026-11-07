"""The C2 reading stage's exit logic, tested on synthetic receipts; no GPU, no data, no claim.

Split out of ``test_r7_s3_derivative_matching_c2.py`` because that file sits at R-021's 400-line
target and these three properties are a coherent unit: they are the ones that would have caught
the three defects which stopped ``attempt04`` after six GPU-hours (decision 0098). Every fixture
here is written to ``tmp_path``; nothing reads the store and nothing trains.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

def _c2_receipt_tree(tmp_path, driver, climatology):
    """A two-seed, two-arm receipt tree whose tables carry exactly the five frozen leads."""
    import csv, hashlib, json
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for seed in driver.SEEDS:
        arms = {}
        for arm in (driver.CONTROL, driver.CANDIDATE):
            training_dir = tmp_path / f"seed{seed}/{arm}/training"
            training_dir.mkdir(parents=True)
            losses = ([{"constraint_divergence": 0.5, "constraint_vorticity": 0.25}]
                      if arm == driver.CANDIDATE else [])
            (training_dir / "training_report.json").write_text(json.dumps({"losses": losses}),
                                                              encoding="utf-8")
            evaluations = {}
            for lead in driver.LEADS:
                folder = tmp_path / f"seed{seed}/{arm}/lead_{lead:03d}h"
                folder.mkdir(parents=True)
                with (folder / "boundary_rmse.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["region", "variable", "rmse"])
                    writer.writeheader()
                    # Equal on both arms, so the paired delta is 0.0 and (a) holds.
                    writer.writerow({"region": driver.curve.REGION, "variable": "t2m",
                                     "rmse": "2.0"})
                evaluations[str(lead)] = {"dir": str(folder),
                                          "boundary_rmse_csv_sha256": sha(folder / "boundary_rmse.csv")}
            arms[arm] = {"training_dir": str(training_dir), "evaluations": evaluations}
        (tmp_path / f"seed{seed}_receipt.json").write_text(
json.dumps({"seed": seed, "arms": arms}), encoding="utf-8")
    return {"protocol_sha256": "0" * 64, "derivative_matching": {}, "climatology_reference": climatology,
            "preflight": {"rows": [{"total": 1.0}]}}


def test_reading_attributes_resolve_and_the_exit_consumes_both_arms(tmp_path):
    """The reading stage's defects, each of which only fired after ten hours of training.

    Attempt 04 stopped at 32% of its first arm because the reading stage could not have produced
    a verdict: ``readings`` and ``_train_arm`` both read ``curve.EVALUATION_LEADS``, which the
    large-protocol module never binds (it imports ``wide``), and the rmse/skill fill sat inside
    ``if arm == CANDIDATE``, leaving the control arm empty so ``evaluate_exit`` raised
    ``KeyError('72')``. This runs ``readings`` on a synthetic tree, then exercises (b) both ways.
    """
    import scripts.study_r7_s3_derivative_matching_c2 as driver

    assert driver.LEADS == (6, 12, 24, 48, 72) and driver.LEADS is driver.wide.EVALUATION_LEADS
    assert not hasattr(driver.curve, "EVALUATION_LEADS")  # the module that must NOT carry it
    climatology = {"leads": {str(lead): {"t2m_rmse": 4.0} for lead in driver.LEADS}}
    result = driver.readings(tmp_path, _c2_receipt_tree(tmp_path, driver, climatology))

    wanted = sorted(str(lead) for lead in driver.LEADS)
    for seed in driver.SEEDS:
        for arm in (driver.CONTROL, driver.CANDIDATE):
            entry = result["seeds"][str(seed)][arm]
            assert sorted(entry["rmse"]) == wanted and sorted(entry["skill"]) == wanted
    assert result["seeds"][str(driver.SEEDS[0])][driver.CONTROL]["skill"]["72"] == pytest.approx(
        1.0 - (2.0 / 4.0) ** 2)
    exits = result["exit_conditions"]
    assert exits["a_72h_skill_mean_delta"] and exits["b_matching_bit"] and exits["all"]
    assert exits["verdict"] == "advance-to-independent-review"
    # (b) must also be able to say no, or it proves nothing about the term having bit.
    def synthetic(recorded, preflight_total=1.0):
        skill = {str(lead): 0.2 for lead in driver.LEADS}
        return {"scientific_claim": False, "test_read": False,
                "preflight": {"rows": [{"total": preflight_total}]},
                "seeds": {str(seed): {driver.CONTROL: {"rmse": {}, "skill": dict(skill)},
                                      driver.CANDIDATE: {"rmse": {}, "skill": dict(skill),
                                                         "constraint_terms_recorded": recorded}}
                          for seed in driver.SEEDS}}

    assert driver.evaluate_exit(synthetic(True))["b_matching_bit"] is True
    assert driver.evaluate_exit(synthetic(False))["b_matching_bit"] is False
    assert driver.evaluate_exit(synthetic(True, preflight_total=0.0))["b_matching_bit"] is False
    closed = driver.evaluate_exit(synthetic(False))
    assert not closed["all"]
    assert closed["verdict"] == "permanently-close-derivative-matching-branch"


def test_the_fill_lives_in_the_lead_loop_not_the_candidate_branch():
    """A structural counterproof: the bug was indentation, so assert the shape too.

    A first draft only asserted that nothing under ``if arm == CANDIDATE`` touched ``["rmse"]``,
    which passed on the buggy code because the whole nested block *was* the fill. This asserts
    the positive property: both fills are stores inside the per-lead loop.
    """
    tree = ast.parse(Path("scripts/study_r7_s3_derivative_matching_c2.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "readings")
    seed_loop = next(node for node in function.body
                     if isinstance(node, ast.For) and ast.unparse(node.target) == "seed")
    arm_loop = next(node for node in seed_loop.body
                    if isinstance(node, ast.For) and ast.unparse(node.target) == "arm")
    lead_loop = next(node for node in arm_loop.body
                     if isinstance(node, ast.For) and ast.unparse(node.target) == "lead")

    def stores(node):
        return [ast.unparse(target) for item in ast.walk(node) if isinstance(item, ast.Assign)
                for target in item.targets if isinstance(target, ast.Subscript)]

    fills = ("['rmse'][str(lead)]", "['skill'][str(lead)]")
    filled = [target for target in stores(lead_loop) if target.endswith(fills)]
    assert len(filled) == 2, f"expected both fills in the lead loop, saw {stores(lead_loop)}"
    candidate_if = next(node for node in arm_loop.body
                        if isinstance(node, ast.If) and ast.unparse(node.test) == "arm == CANDIDATE")
    assert not [target for target in stores(candidate_if) if target.endswith(fills)]
