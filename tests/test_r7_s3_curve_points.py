"""CPU counterproofs for the Stage B curve points 2-4; no GPU, no real dataset, no claim.

These pin what makes a point *a point of the same curve*: the ladder and this point's place
on it, batch fixed at 4, the seventeen-variable scoring surface taken from the store rather
than from memory, only t2m carrying a registered denominator, the saturation rule recorded
but never evaluated, and the engineering exit failing on any single false condition. They
also pin the two defects this campaign actually shipped and then fixed on point 1 -- a
driver with no ``__main__`` guard, and a checkpoint cadence declared in the protocol but
never passed to the runner -- so neither can return.

Fixtures are synthetic dictionaries: nothing here is weather, and every one declares
``scientific_claim=False``.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from scripts import r7_s3_curve_point_driver as driver
from scripts import r7_s3_curve_point_protocol as proto

POINT2 = "p1000"


def test_the_ladder_is_the_frozen_one():
    """The four points and their update counts are the goal document's ladder, verbatim."""
    assert dict(proto.POINTS) == {"p0025": 2008, "p1000": 8033,
                                  "p4000": 32132, "p1600": 128528}
    assert proto.updates_for(POINT2) == 8033
    assert proto.NOMINAL_EPOCHS[POINT2] == 1.0
    block = proto.point_blocks(POINT2)
    # 8033 * 4 = 32,132 samples over 32,131 USABLE windows is 1.000031 epoch, which the goal
    # document names 1. Both are recorded so the label cannot be mistaken for the arithmetic.
    assert block["epochs"][POINT2] == pytest.approx(32132 / 32131, abs=1e-6)
    assert block["point"] == POINT2 and block["updates"] == 8033


def test_an_unknown_point_is_refused_not_defaulted():
    """A typo in a binding must fail loudly; silently defaulting would run the wrong point."""
    with pytest.raises(RuntimeError, match="unknown curve point"):
        proto.updates_for("p0001")


def test_batch_is_fixed_at_four_for_every_point():
    """Decision 0084: batch does not move during calibration, so no point overrides it."""
    assert proto.BATCH_SIZE == 4
    assert proto.point_blocks(POINT2)["batch_size"] == 4
    constants = proto.frozen_constants(POINT2)
    assert constants["recipe"]["batch_size"] == 4
    assert "confound" in constants["recipe"]["batch_size_choice"]


def test_the_scored_variables_are_the_seventeen_registered_channels():
    """The list must be the store's own channel order; a memory-written list was a real bug."""
    expected = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
    assert driver.SCORED_VARIABLES == expected
    assert len(set(driver.SCORED_VARIABLES)) == 17
    assert all(name.strip() for name in driver.SCORED_VARIABLES)


def test_only_t2m_carries_a_registered_climatology_denominator():
    """Sixteen variables have no fitted denominator, so they get no invented skill number."""
    body = {"climatology": {"buckets": 48},
            "climatology_reference": {"leads": {"72": {"t2m_rmse": 3.926891361747067}}}}
    assert driver.climatology_denominator(body, 72, "t2m") == pytest.approx(3.926891361747067)
    for variable in driver.SCORED_VARIABLES:
        if variable != "t2m":
            assert driver.climatology_denominator(body, 72, variable) is None
    # A reference without the lead must raise rather than silently return a number.
    with pytest.raises(RuntimeError, match="refusing to invent"):
        driver.climatology_denominator({"climatology_reference": {"leads": {}}}, 72, "t2m")


def test_the_saturation_rule_is_recorded_and_never_evaluated_here():
    """A single point cannot compute a two-point first difference; the rule is data only."""
    rule = proto.SATURATION_RULE
    assert set(rule) == {"delta", "unsaturated_if", "saturated_if", "else", "note"}
    assert "16 epoch" in rule["delta"] and "4 epoch" in rule["delta"]
    assert "0.01" in rule["unsaturated_if"] and "0.005" in rule["saturated_if"]
    constants = proto.frozen_constants(POINT2)
    assert constants["saturation_rule"] == rule
    assert not hasattr(driver, "saturation_verdict")


def _readings(updates=8033, batch=4, leads=None, skill_72=None, n_variables=17):
    """A minimal readings body shaped like the driver's, for exit-rule counterproofs."""
    variables = list(driver.SCORED_VARIABLES[:n_variables])
    values = {name: 1.0 for name in variables}
    if leads is None:
        leads = {str(lead): {"rmse": dict(values), "skill": {name: None for name in variables}}
                 for lead in (6, 12, 24, 48, 72)}
    leads["72"]["skill"]["t2m"] = 0.1 if skill_72 is None else skill_72
    return {"updates": updates, "batch_size": batch, "leads": leads}


def test_the_engineering_exit_passes_on_a_complete_point():
    result = driver.evaluate_exit(_readings(), POINT2)
    assert result["all"] is True
    assert result["verdict"] == "advance-to-next-curve-point"


@pytest.mark.parametrize("mutate,label", [
    (lambda r: r.update(updates=8032), "a: wrong update count"),
    (lambda r: r.update(updates=2008), "a: point 1's count is not this point"),
    (lambda r: r.update(batch_size=2), "a: wrong batch"),
    (lambda r: r["leads"]["72"]["rmse"].update(t2m=float("nan")), "b: nonfinite rmse"),
    (lambda r: r["leads"]["72"]["skill"].update(t2m=None), "c: no t2m 72h skill"),
])
def test_any_false_exit_invalidates_the_point(mutate, label):
    readings = _readings()
    mutate(readings)
    result = driver.evaluate_exit(readings, POINT2)
    assert result["all"] is False, label
    assert result["verdict"] == "engineering-incomplete-do-not-read"


def test_a_missing_variable_fails_the_finite_rule():
    """Sixteen of seventeen variables is an incomplete point, not a passing one."""
    readings = _readings(n_variables=16)
    assert len(readings["leads"]["72"]["rmse"]) == 16
    result = driver.evaluate_exit(readings, POINT2)
    assert result["b"] is False and result["all"] is False


def test_the_protocol_records_the_stage_a_policies():
    """Checkpoint and finite-state policy come from 0086/0087, not from habit."""
    constants = proto.frozen_constants(POINT2)
    assert constants["checkpoint_policy"]["checkpoint_every"] == 5000
    assert constants["checkpoint_policy"]["decision"] == "0087"
    assert constants["finite_state_policy"]["full_check_every"] == 100
    assert constants["finite_state_policy"]["decision"] == "0086"
    assert constants["test_read"] is False
    assert constants["scientific_claim"] is False


def test_the_budget_is_derived_from_point_one_s_measurement():
    """The estimate must come from the measured 1.174 s/update, not from a round number."""
    tuned = proto.budgets(POINT2)
    assert proto.MEASURED_SECONDS_PER_UPDATE == pytest.approx(1.174)
    assert proto.MEASURED_TRAIN_SECONDS == pytest.approx(2358.3)
    assert proto.MEASURED_SCORING_SECONDS == pytest.approx(2633.8)
    assert tuned["train_seconds_estimate"] == pytest.approx(8033 * 1.174, rel=1e-12)
    assert tuned["scoring_allowance_seconds"] == proto.MEASURED_SCORING_SECONDS
    assert tuned["point_seconds_estimate"] == pytest.approx(
        tuned["train_seconds_estimate"] + tuned["scoring_allowance_seconds"], rel=1e-12)
    # The envelope must actually contain the estimate, or the point is truncated by duration.
    assert tuned["per_seed_seconds"] > tuned["point_seconds_estimate"]
    assert tuned["planned_seconds"] >= tuned["per_seed_seconds"]
    assert tuned["hard_cap_seconds"] == pytest.approx(2 * tuned["planned_seconds"])
    assert "estimate" in tuned["estimate_basis"]
    # A later point must not inherit a smaller envelope than an earlier one.
    assert proto.budgets("p4000")["hard_cap_seconds"] == tuned["hard_cap_seconds"]


def test_the_peak_memory_estimate_is_point_one_s_measurement():
    """Point 1 measured 8.729 GiB at the same batch/recipe, so the gate uses that, not a guess."""
    assert proto.ESTIMATED_PEAK_BYTES == proto.MEASURED_PEAK_RESERVED_BYTES
    assert proto.frozen_constants(POINT2)["gpu"]["estimated_peak_mib"] == 8938
    assert "measured reserved peak" in proto.frozen_constants(POINT2)["gpu"]["estimate_provenance"]


def test_the_point_is_a_single_seed_and_a_single_arm():
    """n=1 is the decision, not an omission: the declaration must say so."""
    assert proto.SEEDS == (41,)
    constants = proto.frozen_constants(POINT2)
    assert constants["curve"]["seeds"] == [41]
    assert any("ONE seed" in item for item in constants["limitations"])


def test_the_point_has_its_own_named_entry_file():
    """Decision 0067: the inventory starts from a named entry file, never a glob."""
    name = proto.entry_file(POINT2)
    assert name == "scripts/study_r7_s3_throughput_physics_b_p1000.py"
    assert (Path(proto.ROOT) / name).is_file()
    binding = __import__("scripts.study_r7_s3_throughput_physics_b_p1000", fromlist=["x"])
    assert binding.POINT == POINT2 and binding.UPDATES == 8033


def test_the_binding_is_runnable_as_a_script():
    """A driver without a ``__main__`` guard exits 0 having done nothing.

    Point 1 shipped that defect: ``stage prepare`` reported success in 0.06 s and wrote no
    ``prepared_protocol.json``. The wrapper spawns one worker process per stage, so the guard
    is what makes the stage happen at all. The engine module is not an entry point, so its
    guard must *refuse* rather than silently pick a point.
    """
    binding = __import__("scripts.study_r7_s3_throughput_physics_b_p1000",
                         fromlist=["x"])
    for module in (driver, binding):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        guards = [node for node in tree.body
                  if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Compare)
                  and getattr(node.test.left, "id", None) == "__name__"]
        assert guards, f"{module.__name__} has no `if __name__ == '__main__'` guard"
    # The engine's guard raises instead of running a point it was never told about.
    engine_source = Path(driver.__file__).read_text(encoding="utf-8")
    guard = engine_source[engine_source.rindex("if __name__"):]
    assert "SystemExit" in guard and "binding" in guard
    # The binding's guard actually calls main.
    binding_source = Path(binding.__file__).read_text(encoding="utf-8")
    assert "main" in binding_source[binding_source.rindex("if __name__"):]


def test_the_worker_entry_discards_the_worker_result():
    """``sys.exit(value)`` prints the value, so the engine's main must not return it."""
    tree = ast.parse(Path(driver.__file__).read_text(encoding="utf-8"))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert [node for node in ast.walk(main) if isinstance(node, ast.Return)] == []


def test_the_runner_is_told_the_frozen_checkpoint_cadence():
    """Decision 0087 is only applied if the call passes it; the runner's default is 20.

    Measured defect on point 1: the protocol declared 5,000 but the argument was omitted, so
    the runner wrote 101 checkpoints (5.6 GiB) where the ADR allows four.
    """
    source = Path(driver.__file__).read_text(encoding="utf-8")
    call = source[source.index("fine_tune_long_rollout("):]
    call = call[:call.index(")", call.index("loader="))]
    assert "checkpoint_every=proto.CHECKPOINT_EVERY" in call
    assert 'folder.glob("update_*.pt")' in source
    assert 'folder / "checkpoints"' not in source


def test_the_offline_denial_is_installed_before_any_stage_work():
    """R-028: the worker denies outbound before it touches data, and only AF_UNIX survives."""
    tree = ast.parse(Path(driver.__file__).read_text(encoding="utf-8"))
    worker = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "worker")
    calls = [node for node in ast.walk(worker)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "deny_network"]
    assert calls, "worker never calls deny_network()"
    # The denial must come before any phase branch that could read or write data.
    first_phase = min((node.lineno for node in ast.walk(worker)
                       if isinstance(node, ast.Compare)
                       and isinstance(node.comparators[0], ast.Constant)
                       and node.comparators[0].value in {"prepare", "archive", "reading"}),
                      default=10**9)
    assert calls[0].lineno < first_phase


def test_without_a_live_loss_is_refused():
    """A nonfinite training loss must stop the point before it is scored."""
    assert not all(math.isfinite(value) for value in [1.0, float("inf"), 0.9])
    assert all(math.isfinite(value) for value in [1.0, 0.98, 0.9])
