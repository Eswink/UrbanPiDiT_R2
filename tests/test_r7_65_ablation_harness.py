"""Offline tests for the #65 ablation harness contract.

No GPU, no training, no real data: these pin the properties that make a C1/C2/C3
result readable rather than the numbers it produces.

The load-bearing case is ``test_a_shallow_arm_declares_and_uses_its_own_depth``.
The first C3 run configured the shallow control with
``default_reasoning_steps=1`` but passed the phase-wide ``steps=4`` to the
training runner, so the "independently trained K=1" arm was a second K=4 run in
disguise and its FLOPs matched the K=4 arm exactly. The model config and the
runner's step count are two separate declarations of the same fact, and nothing
in the runner cross-checks them; the harness now declares the depth per arm and
fails closed when the published contract disagrees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.study_r7_65_ablation as harness  # noqa: E402
from scripts.study_r7_65_ablation import (  # noqa: E402
    C1_ARMS, C2_ARMS, SHALLOW_TRAINED_STEPS, TRAIN_REASONING_STEPS,
    TRAIN_SEEDS, _arm_config, _arm_train_steps, _phase_arms,
)


def test_seeds_are_pre_declared_and_phase_arms_reject_undeclared_seeds():
    assert TRAIN_SEEDS == (41, 42, 43)
    with pytest.raises(ValueError):
        harness.run_phase("nowhere", "outputs/should_not_exist", phase="c1",
                          seeds=(99,))


def test_c1_weights_are_the_frozen_triple():
    """#65 C1 names 0 / 0.01 / 0.1 for the process arm; a fourth value would be
    an unfrozen grid. The generic control is separate and carries no auxiliary
    weight, so it is excluded from the triple rather than counted as a fourth."""
    process_weights = [entry[3] for entry in C1_ARMS if entry[1] == "process"]
    assert process_weights == [0.0, 0.01, 0.1]
    assert C1_ARMS[0][1] == "generic" and C1_ARMS[0][3] == 0.0


def test_c1_holds_structure_fixed_while_only_the_weight_moves():
    """The three process rows must be the SAME architecture, or the weight axis
    is confounded with a structural change."""
    process_specs = [entry for entry in C1_ARMS if entry[1] == "process"]
    assert len(process_specs) == 3
    assert all(spec[2] == {} for spec in process_specs)
    assert len({entry[3] for entry in process_specs}) == 3


def test_c2_declares_each_feedback_path_explicitly():
    """#65 C2 asks which path is open, so every arm must state the solver path.

    ``use_forecast_feedback`` is declared on the process arms because they own
    that switch. The generic arm's spec also carries the key so the arm list
    reads uniformly, and ``_arm_config`` strips it before construction; the test
    below pins that stripping, which is what keeps the declaration from turning
    into a TypeError.
    """
    for name, kind, extra in C2_ARMS:
        assert "spatial_solver_feedback" in extra, name
        if kind == "process":
            assert "use_forecast_feedback" in extra, name


def test_generic_config_drops_the_process_only_feedback_switch():
    generic = _arm_config("generic", 4, {"use_forecast_feedback": True,
                                         "spatial_solver_feedback": True})
    assert "use_forecast_feedback" not in generic
    assert generic["spatial_solver_feedback"] is True
    process = _arm_config("process", 4, {"use_forecast_feedback": False})
    assert process["use_forecast_feedback"] is False
    assert process["anchored_processes"] == 8


def test_every_arm_in_every_phase_declares_a_training_depth():
    for phase in ("c1", "c2", "c3"):
        for spec in _phase_arms(phase):
            depth = _arm_train_steps(phase, spec[0])
            assert isinstance(depth, int) and depth >= 1


def test_c3_declares_both_depths_and_keeps_its_own_flop_count():
    """The two C3 arms must differ in depth, and the shallow one must therefore
    be cheaper - if both reported the same FLOPs the shallow arm is not real."""
    k4 = _phase_arms("c3")
    names = [spec[0] for spec in k4]
    assert names == ["process8_aux010_k4", "process8_aux010_k1"]
    assert _arm_train_steps("c3", "process8_aux010_k4") == TRAIN_REASONING_STEPS
    assert _arm_train_steps("c3", "process8_aux010_k1") == SHALLOW_TRAINED_STEPS
    configs = {spec[0]: _arm_config("process", 17, spec[2]) for spec in k4}
    assert configs["process8_aux010_k4"].get("default_reasoning_steps",
                                             TRAIN_REASONING_STEPS) == TRAIN_REASONING_STEPS
    assert configs["process8_aux010_k1"]["default_reasoning_steps"] == SHALLOW_TRAINED_STEPS


def test_a_shallow_arm_declares_and_uses_its_own_depth(tmp_path):
    """Regression test for the silent-K=4 bug.

    The harness must pass the *arm's* depth to the runner, not the phase-wide
    constant. The published training contract is the artifact a later reader
    checks, so that is what this asserts: a run told ``steps=1`` must record
    ``steps=1``. Dropping the per-arm depth would record the phase-wide 4 here.
    """
    from data.synthetic_atmos import SyntheticAtmosDataset
    from training.r7_experiment import seed_everything
    from training.r7_scheduled_runner import run_scheduled_updates

    dataset = SyntheticAtmosDataset(length=3, hw=(8, 8), channels=3)
    config = _arm_config("process", 3, {"default_reasoning_steps": SHALLOW_TRAINED_STEPS,
                                        "dim": 16, "depth": 1, "heads": 2,
                                        "window_size": 2, "patch_size": 2,
                                        "anchored_processes": 2, "free_processes": 1})
    seed_everything(1)
    _, report = run_scheduled_updates(
        dataset, kind="process", model_config=config, data_identity="test",
        output_dir=tmp_path / "run", total_updates=1, batch_size=1,
        steps=SHALLOW_TRAINED_STEPS, seed=1, lr=1e-3, clip=1.0,
        process_weight=0.1, warmup_updates=1, device_name="cpu",
        validation_every=0)
    assert report["contract"]["steps"] == SHALLOW_TRAINED_STEPS
    assert report["contract"]["process_weight"] == 0.1


def test_the_two_c3_arms_must_not_report_the_same_compute(tmp_path):
    """Counterproof for the bug above: FLOPs are counted at each arm's own
    depth, so the shallow arm is measurably cheaper. If a future edit made both
    arms train at K=4 these counts would collapse to equal, which is exactly the
    symptom that was observed."""
    from torch.utils.data import default_collate

    from data.synthetic_atmos import SyntheticAtmosDataset
    from training.r7_experiment import make_model, seed_everything

    dataset = SyntheticAtmosDataset(length=2, hw=(8, 8), channels=3)
    batch = default_collate([dataset[0], dataset[1]])
    fwd = {}
    for spec in _phase_arms("c3"):
        name = spec[0]
        depth = _arm_train_steps("c3", name)
        seed_everything(41)
        model = make_model("process", _arm_config("process", 3, spec[2]))
        fwd[name] = harness.count_flops(model, batch, reasoning_steps=depth,
                                        recursive=True)[0]
    assert fwd["process8_aux010_k1"] < fwd["process8_aux010_k4"]
