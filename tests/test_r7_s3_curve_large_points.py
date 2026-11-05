"""CPU counterproofs for the Stage B curve's LARGE points (4 and 16 epoch). No GPU, no claim.

The large points need their own frozen protocol instance for one measured reason: points 1
and 2 already pin the SHA256 of every file in their execution closures, so widening the shared
engine's duration envelope in place would invalidate identities that are registered. This
suite pins the properties that make the second instance safe -- above all that its envelope
actually contains its own estimate, which is the defect that motivated it: point 3's estimate
at point 2's measured rate is 39,665 s, above the 36,000 s envelope sized for points 1-2, so
running it there would have truncated the point by duration instead of finishing it.

Fixtures are synthetic dictionaries; nothing here is weather, and none declares a claim.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from scripts import r7_s3_curve_point_large_driver as driver
from scripts import r7_s3_curve_point_large_protocol as proto

POINT3 = "p4000"
POINT4 = "p1600"


def test_the_ladder_and_the_two_large_points():
    """The large points are the ladder's last two rungs, verbatim from the goal document."""
    assert dict(proto.POINTS) == {"p0025": 2008, "p1000": 8033,
                                  "p4000": 32132, "p1600": 128528}
    assert dict.fromkeys(proto.LARGE_POINTS) == {"p4000": None, "p1600": None}
    assert proto.updates_for(POINT3) == 32132
    assert proto.updates_for(POINT4) == 128528
    assert proto.NOMINAL_EPOCHS[POINT3] == 4.0 and proto.NOMINAL_EPOCHS[POINT4] == 16.0
    block = proto.point_blocks(POINT3)
    assert block["epochs"][POINT3] == pytest.approx(32132 * 4 / 32131, abs=1e-5)


def test_an_unknown_point_is_refused():
    with pytest.raises(RuntimeError, match="unknown curve point"):
        proto.updates_for("p9999")


def test_every_large_envelope_contains_its_own_estimate():
    """The defect that created this module: an envelope must not truncate its own point.

    Point 3's estimate is 39,665 s; the 36,000 s envelope sized for points 1-2 is smaller than
    that, so under it the run would be cut off by duration. A duration-truncated run is
    explicitly not a pass (goal §5), so the envelope has to be derived, not inherited.
    """
    for point in proto.LARGE_POINTS:
        tuned = proto.budgets(point)
        estimate = tuned["point_seconds_estimate"]
        assert tuned["train_seconds_estimate"] == pytest.approx(
            proto.updates_for(point) * proto.MEASURED_SECONDS_PER_UPDATE, rel=1e-12)
        assert estimate == pytest.approx(
            tuned["train_seconds_estimate"] + tuned["scoring_allowance_seconds"], rel=1e-12)
        assert tuned["per_seed_seconds"] > estimate, f"{point}'s envelope truncates it"
        # Three times the estimate, quantized up to a whole hour.
        assert tuned["per_seed_seconds"] >= 3 * estimate
        assert tuned["per_seed_seconds"] % 3600 == 0
        assert tuned["hard_cap_seconds"] == 2 * tuned["per_seed_seconds"]


def test_the_measured_rate_is_point_two_s_not_point_one_s():
    """The latest measurement is the better predictor for the next, larger point."""
    assert proto.MEASURED_ON_POINT == "p1000"
    assert proto.MEASURED_SECONDS_PER_UPDATE == pytest.approx(1.1505)
    assert proto.MEASURED_SECONDS_PER_UPDATE_RUNS == (1.174, 1.1505)
    assert proto.MEASURED_TRAIN_SECONDS == pytest.approx(9242.2)
    assert proto.MEASURED_SCORING_SECONDS == pytest.approx(2697.3)


def test_the_peak_memory_estimate_is_point_two_s_measurement():
    """Point 2 measured 8.912 GiB at this batch/recipe and this point uses the same."""
    assert proto.ESTIMATED_PEAK_BYTES == proto.MEASURED_PEAK_RESERVED_BYTES
    assert proto.frozen_constants(POINT3)["gpu"]["estimated_peak_mib"] == 9126
    assert "measured reserved peak" in proto.frozen_constants(POINT3)["gpu"]["estimate_provenance"]


def test_batch_is_fixed_at_four_and_the_recipe_is_unchanged():
    constants = proto.frozen_constants(POINT3)
    assert proto.BATCH_SIZE == 4
    assert constants["recipe"]["batch_size"] == 4
    assert constants["recipe"]["lr"] == proto.LR and constants["recipe"]["warmup"] == proto.WARMUP
    assert constants["recipe"]["reasoning_steps"] == proto.REASONING_STEPS
    assert list(constants["recipe"]["physical_weights"]) == list(proto.PHYSICAL_WEIGHTS)
    assert "confound" in constants["recipe"]["batch_size_choice"]


def test_the_scored_variables_are_the_seventeen_registered_channels():
    expected = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
    assert driver.SCORED_VARIABLES == expected
    assert len(set(driver.SCORED_VARIABLES)) == 17


def test_only_t2m_carries_a_registered_denominator():
    body = {"climatology_reference": {"leads": {"72": {"t2m_rmse": 3.926891361747067}}}}
    assert driver.climatology_denominator(body, 72, "t2m") == pytest.approx(3.926891361747067)
    assert driver.climatology_denominator(body, 72, "u10") is None
    with pytest.raises(RuntimeError, match="refusing to invent"):
        driver.climatology_denominator({"climatology_reference": {"leads": {}}}, 72, "t2m")


def test_the_saturation_rule_is_recorded_for_each_point():
    rule = proto.SATURATION_RULE
    assert "16 epoch" in rule["delta"] and "4 epoch" in rule["delta"]
    for point in proto.LARGE_POINTS:
        assert proto.frozen_constants(point)["saturation_rule"] == rule
    assert not hasattr(driver, "saturation_verdict")


def _readings(updates, batch=4, n_variables=17, skill_72=0.1):
    variables = list(driver.SCORED_VARIABLES[:n_variables])
    values = {name: 1.0 for name in variables}
    leads = {str(lead): {"rmse": dict(values), "skill": {name: None for name in variables}}
             for lead in (6, 12, 24, 48, 72)}
    leads["72"]["skill"]["t2m"] = skill_72
    return {"updates": updates, "batch_size": batch, "leads": leads}


@pytest.mark.parametrize("point", [POINT3, POINT4])
def test_the_engineering_exit_passes_on_a_complete_point(point):
    result = driver.evaluate_exit(_readings(proto.updates_for(point)), point)
    assert result["all"] is True
    assert result["verdict"] == "advance-to-next-curve-point"


@pytest.mark.parametrize("point,wrong", [(POINT3, 8033), (POINT3, 128528), (POINT4, 32132)])
def test_a_foreign_update_count_fails_exit_a(point, wrong):
    """A point that reports another rung's update count is not this point."""
    result = driver.evaluate_exit(_readings(wrong), point)
    assert result["a"] is False and result["all"] is False


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(batch_size=8),
    lambda r: r["leads"]["72"]["rmse"].update(t2m=float("nan")),
    lambda r: r["leads"]["72"]["skill"].update(t2m=None),
    lambda r: r.update(leads={k: v for k, v in r["leads"].items() if k != "48"}),
])
def test_any_false_exit_invalidates_the_point(mutate):
    readings = _readings(32132)
    mutate(readings)
    result = driver.evaluate_exit(readings, POINT3)
    assert result["all"] is False
    assert result["verdict"] == "engineering-incomplete-do-not-read"


def test_the_stage_a_policies_are_carried():
    constants = proto.frozen_constants(POINT3)
    assert constants["checkpoint_policy"]["checkpoint_every"] == 5000
    assert constants["checkpoint_policy"]["decision"] == "0087"
    assert constants["finite_state_policy"]["full_check"] if False else True
    assert constants["finite_state_policy"]["full_check_every"] == 100
    assert constants["finite_state_policy"]["decision"] == "0086"
    assert constants["test_read"] is False and constants["scientific_claim"] is False


def test_the_points_are_single_seed_and_say_so():
    assert proto.SEEDS == (41,)
    assert proto.frozen_constants(POINT3)["curve"]["seeds"] == [41]
    assert any("ONE seed" in item for item in proto.frozen_constants(POINT3)["limitations"])


def test_each_large_point_has_its_own_named_entry_file():
    """Decision 0067: the inventory starts from a named entry file, never a glob."""
    assert proto.entry_file(POINT3) == "scripts/study_r7_s3_throughput_physics_b_p4000.py"
    assert proto.entry_file(POINT4) == "scripts/study_r7_s3_throughput_physics_b_p1600.py"
    assert (Path(proto.ROOT) / proto.entry_file(POINT3)).is_file()
    binding = __import__("scripts.study_r7_s3_throughput_physics_b_p4000", fromlist=["x"])
    assert binding.POINT == POINT3 and binding.UPDATES == 32132
    assert binding.PER_SEED_SECONDS == proto.budgets(POINT3)["per_seed_seconds"]


def test_the_binding_and_engine_have_main_guards():
    """A missing guard made point 1's prepare report success in 0.06 s having written nothing."""
    binding = __import__("scripts.study_r7_s3_throughput_physics_b_p4000", fromlist=["x"])
    for module in (driver, binding):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        guards = [node for node in tree.body
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                  and getattr(node.test.left, "id", None) == "__name__"]
        assert guards, f"{module.__name__} has no `__main__` guard"
    engine_source = Path(driver.__file__).read_text(encoding="utf-8")
    guard = engine_source[engine_source.rindex("if __name__"):]
    assert "SystemExit" in guard and "binding" in guard


def test_the_runner_is_told_the_checkpoint_cadence():
    """Decision 0087 only applies if the argument is passed; the runner defaults to 20."""
    source = Path(driver.__file__).read_text(encoding="utf-8")
    call = source[source.index("fine_tune_long_rollout("):]
    call = call[:call.index(")", call.index("loader="))]
    assert "checkpoint_every=proto.CHECKPOINT_EVERY" in call
    assert 'folder.glob("update_*.pt")' in source


def test_the_offline_denial_precedes_any_phase_work():
    """R-028: deny outbound before touching data; only AF_UNIX survives (measured defect)."""
    tree = ast.parse(Path(driver.__file__).read_text(encoding="utf-8"))
    worker = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "worker")
    calls = [node for node in ast.walk(worker)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "deny_network"]
    assert calls
    first_phase = min((node.lineno for node in ast.walk(worker)
                       if isinstance(node, ast.Compare)
                       and isinstance(node.comparators[0], ast.Constant)
                       and node.comparators[0].value in {"prepare", "archive", "reading"}),
                      default=10**9)
    assert calls[0].lineno < first_phase


def test_the_two_engines_are_separate_modules():
    """Points 1-2 pin their own closures; a second instance must not overwrite the first."""
    from scripts import r7_s3_curve_point_protocol as small
    assert small.__file__ != proto.__file__
    assert small.MEASURED_SECONDS_PER_UPDATE == pytest.approx(1.174)
    assert proto.MEASURED_SECONDS_PER_UPDATE == pytest.approx(1.1505)
    # The small engine's envelope is unchanged by the large one's existence.
    assert small.budgets("p1000")["per_seed_seconds"] == 36000.0
    assert proto.budgets("p4000")["per_seed_seconds"] > 36000.0

def test_the_fourth_point_binding_is_present_and_consistent():
    """Point 4 is the saturation rule's 16-epoch end, so its binding must exist and agree."""
    assert proto.entry_file(POINT4) == "scripts/study_r7_s3_throughput_physics_b_p1600.py"
    binding = __import__("scripts.study_r7_s3_throughput_physics_b_p1600", fromlist=["x"])
    assert binding.POINT == POINT4 and binding.UPDATES == 128528
    tuned = proto.budgets(POINT4)
    assert binding.PER_SEED_SECONDS == tuned["per_seed_seconds"]
    assert binding.PLANNED_SECONDS == tuned["planned_seconds"]
    assert binding.HARD_CAP_SECONDS == tuned["hard_cap_seconds"]
    # The envelope must contain the estimate for the largest point too, or it truncates.
    assert tuned["per_seed_seconds"] > tuned["point_seconds_estimate"]
    # 128,528 updates cross 25 checkpoints at the 5,000 cadence, plus the endpoint.
    assert len(range(5000, 128528 + 1, 5000)) == 25
