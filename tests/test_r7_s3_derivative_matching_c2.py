"""CPU counterproofs for Stage C2's derivative-domain target matching; no GPU, no data, no claim.

Two properties carry the whole design, so both are asserted with exact equality rather than a
tolerance:

* **``pred == target`` scores exactly 0.0.** That is the family argument against the closed
  spatial-gradient loss, which charges the truth because a real field has gradients. If this
  ever became a small positive number, C2 would have the same failure mode.
* **The optimum is the truth, not a constant field.** A constant wind field is non-divergent and
  non-rotational, so it scores *positively* against a structured target -- the opposite of what a
  ``‖∇pred‖²`` penalty does.

Decision 0094's two lessons are counterproofed here as well: channels are matched by name (a
positional zip would read the wrong fields on this store) and every term is dimensionless (the
absolute form was measured 5 orders of magnitude too small to act).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from training import r7_derivative_matching as matching

STORE_NAMES = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
               "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
SCALES = {"divergence": 2.5665e-05, "vorticity": 3.5089e-05}


def _grid(h=6, w=7):
    lat = torch.linspace(50.0, 45.0, h, dtype=torch.float64)
    lon = torch.linspace(10.0, 16.0, w, dtype=torch.float64)
    return lat, lon


def _field(h=6, w=7):
    """A structured wind field in the store's real 17-channel order."""
    channels = torch.zeros((len(STORE_NAMES), h, w), dtype=torch.float64)
    index = {name: i for i, name in enumerate(STORE_NAMES)}
    channels[index["u850"]] = torch.linspace(-8.0, 8.0, w, dtype=torch.float64).repeat(h, 1)
    channels[index["v850"]] = torch.linspace(-5.0, 5.0, h, dtype=torch.float64).unsqueeze(1).repeat(1, w)
    return channels


def test_the_weights_are_the_declared_ones():
    """Legacy has no derivative-domain target-matching term, so these are this round's choice."""
    assert matching.W_DIVERGENCE == 0.1
    assert matching.W_VORTICITY == 0.1
    source = Path(matching.__file__).read_text(encoding="utf-8")
    assert "this round's choice" in source.lower()
    assert "legacy has no derivative-domain" in source


def test_matching_a_field_against_itself_is_exactly_zero():
    """The family argument: the optimum contains the truth, so the truth must cost nothing."""
    lat, lon = _grid()
    field = _field()
    body = matching.derivative_matching_terms(field, field, STORE_NAMES, lat, lon, scales=SCALES)
    assert float(body["divergence"]) == 0.0
    assert float(body["vorticity"]) == 0.0
    assert float(body["total"]) == 0.0


def test_a_different_field_is_charged():
    """If the term were inert it would be indistinguishable from the zero-at-truth property."""
    lat, lon = _grid()
    prediction, target = _field(), _field()
    target[STORE_NAMES.index("u850")] *= 0.5
    body = matching.derivative_matching_terms(prediction, target, STORE_NAMES, lat, lon,
                                              scales=SCALES)
    assert float(body["divergence"]) > 0.0
    assert float(body["total"]) > 0.0


def test_a_constant_field_is_not_the_optimum():
    """The opposite of a gradient penalty: a constant wind field is charged against a real target.

    A ``‖∇pred‖²`` penalty prefers this field. C2 must not, or it would be the same family.
    """
    lat, lon = _grid()
    target = _field()
    constant = torch.zeros_like(target)
    constant[STORE_NAMES.index("u850")] = 4.0
    constant[STORE_NAMES.index("v850")] = -2.0
    body = matching.derivative_matching_terms(constant, target, STORE_NAMES, lat, lon,
                                              scales=SCALES)
    assert float(body["total"]) > 0.0


def test_the_terms_are_dimensionless():
    """Decision 0094: an absolute form was measured 1e-11..2e-6 against L_mse ~ 1 and could not act.

    A difference of exactly one declared scale unit must contribute ``weight * 1.0`` per cell, so
    the term is O(1) at the scale rather than O(scale^2).
    """
    lat, lon = _grid()
    # u850 shifted by a constant produces a pure divergence/vorticity difference only through the
    # spherical operator; assert the arithmetic contract instead: a uniform doubling of the field
    # difference scales the term by 4 (quadratic) and never by the raw physical magnitude.
    prediction, target = _field(), _field()
    target[STORE_NAMES.index("u850")] *= 0.5
    one = matching.derivative_matching_terms(prediction, target, STORE_NAMES, lat, lon,
                                             scales=SCALES)
    target[STORE_NAMES.index("u850")] *= 0.0   # now zero u850, i.e. twice the original difference
    two = matching.derivative_matching_terms(prediction, target, STORE_NAMES, lat, lon,
                                             scales=SCALES)
    assert float(two["divergence"]) == pytest.approx(4.0 * float(one["divergence"]), rel=1e-10)


def test_scales_change_the_size_not_the_sign():
    """The scale is a unit, not a threshold: halving it must not change which field is charged."""
    lat, lon = _grid()
    prediction, target = _field(), _field()
    target[STORE_NAMES.index("u850")] *= 0.5
    loose = matching.derivative_matching_terms(prediction, target, STORE_NAMES, lat, lon,
                                               scales=SCALES)
    tight = matching.derivative_matching_terms(
        prediction, target, STORE_NAMES, lat, lon,
        scales={k: v / 2 for k, v in SCALES.items()})
    assert float(tight["total"]) > float(loose["total"]) > 0.0


def test_channels_are_matched_by_name_not_by_position():
    """Decision 0094's first defect: the store order is not the operator order.

    ``t2m,u10,v10,mslp,z850,t850,q850,u850,v850,...`` puts u850 at 7 and v850 at 8, while a
    positional read would take whatever sits at 0 and 1. Here the non-wind channels carry values
    that would dominate any positional mis-read, and the answer must still be exactly zero.
    """
    lat, lon = _grid()
    field = _field()
    field[STORE_NAMES.index("t2m")] = 1.0e6
    field[STORE_NAMES.index("mslp")] = 1.0e5
    field[STORE_NAMES.index("u10")] = torch.linspace(-900.0, 900.0, 7, dtype=torch.float64).repeat(6, 1)
    body = matching.derivative_matching_terms(field, field, STORE_NAMES, lat, lon, scales=SCALES)
    assert float(body["total"]) == 0.0


def test_a_missing_wind_channel_is_refused():
    lat, lon = _grid()
    names = tuple(name for name in STORE_NAMES if name != "v850")
    field = torch.zeros((len(names), 6, 7), dtype=torch.float64)
    with pytest.raises(KeyError, match="missing"):
        matching.derivative_matching_terms(field, field, names, lat, lon, scales=SCALES)


def test_a_channel_count_mismatch_is_refused():
    lat, lon = _grid()
    field = _field()
    with pytest.raises(ValueError, match="names were declared"):
        matching.derivative_matching_terms(field, field, STORE_NAMES[:5], lat, lon, scales=SCALES)


def test_a_missing_or_nonpositive_scale_is_refused():
    lat, lon = _grid()
    field = _field()
    with pytest.raises(KeyError, match="missing scales"):
        matching.derivative_matching_terms(field, field, STORE_NAMES, lat, lon,
                                           scales={"divergence": 1e-5})
    with pytest.raises(ValueError, match="must be positive"):
        matching.derivative_matching_terms(field, field, STORE_NAMES, lat, lon,
                                           scales={"divergence": 0.0, "vorticity": 1e-5})


def test_an_unknown_weight_name_is_refused():
    lat, lon = _grid()
    field = _field()
    with pytest.raises(KeyError, match="unknown derivative-matching weights"):
        matching.derivative_matching_terms(field, field, STORE_NAMES, lat, lon, scales=SCALES,
                                           weights={"smoothness": 1.0})


def test_the_module_reuses_the_shared_operators_and_does_not_import_legacy():
    """Decision 0095: reuse the repository's operators, never the legacy loss module."""
    tree = ast.parse(Path(matching.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("legacy" in name for name in imported), imported
    source = Path(matching.__file__).read_text(encoding="utf-8")
    assert "spherical_divergence" in source and "spherical_vorticity" in source


def test_the_terms_read_the_target_because_that_is_what_c2_is():
    """C1 was untargeted and was refuted; C2 must be a target match, so a target argument is required."""
    import inspect
    for name in ("divergence_matching", "vorticity_matching", "derivative_matching_terms"):
        parameters = list(inspect.signature(getattr(matching, name)).parameters)
        assert "target" in parameters, (name, parameters)


def test_the_dispatcher_returns_none_before_building_anything_when_every_axis_is_off():
    """The off path must not touch a graph: a "weight 0" branch would still change the numbers.

    Asserted structurally, because a numerical equality test cannot see a zero-weight term that
    still allocates and still adds 0.0 into the objective.
    """
    from training.r7_objective_axes import auxiliary_term
    contract = {"autoregression": {"physical_steps": 12}}
    assert auxiliary_term(None, {}, contract) is None
    # An explicit None on either axis is the same as absent.
    contract = {"autoregression": {"physical_constraints": None, "derivative_matching": None}}
    assert auxiliary_term(None, {}, contract) is None


def test_the_dispatcher_refuses_two_auxiliary_axes_at_once():
    """Decision 0095: an on/off difference that mixes two new terms cannot be attributed."""
    from training.r7_objective_axes import auxiliary_term
    contract = {"autoregression": {"physical_constraints": "c1", "derivative_matching": "c2"}}
    with pytest.raises(ValueError, match="cannot be enabled together"):
        auxiliary_term(None, {}, contract)


def test_the_dispatcher_routes_each_axis_to_its_own_module():
    """C1 and C2 are separate decisions; the dispatcher must not conflate them."""
    from training import r7_objective_axes
    assert r7_objective_axes.AXES == ("physical_constraints", "derivative_matching")
    source = Path(r7_objective_axes.__file__).read_text(encoding="utf-8")
    assert "r7_physical_constraints import constraint_term" in source
    assert "r7_derivative_matching import matching_term" in source


def test_the_c2_axis_requires_the_normalization_channels_and_scales():
    """Without all three the denormalisation or the unit would be an unregistered hidden variable."""
    from training.r7_derivative_matching import matching_term
    with pytest.raises(ValueError, match="requires the store's normalization"):
        matching_term(None, {}, {"autoregression": {"derivative_matching": "c2"}})
    with pytest.raises(ValueError, match="must be absent or 'c2'"):
        matching_term(None, {}, {"autoregression": {"derivative_matching": "c1"}})


def test_the_runner_has_one_dispatch_point_not_one_per_axis():
    """R-051 keeps the runner near its cap, so both axes share a single hook."""
    runner = Path(matching.__file__).parent.joinpath("r7_long_rollout_runner.py").read_text(
        encoding="utf-8")
    assert runner.count("r7_objective_axes import auxiliary_term") == 1
    assert "r7_derivative_matching import" not in runner
    assert "r7_physical_constraints import" not in runner


def test_the_runner_records_every_axis_term_without_a_hardcoded_name_list():
    """C1 returns divergence/humidity/dew_point and C2 returns divergence/vorticity.

    The runner used to record a fixed C1 name tuple, so the first C2 update would have raised
    KeyError on ``humidity``. Asserted on the source because the alternative is a GPU run.
    """
    runner = Path(matching.__file__).parent.joinpath("r7_long_rollout_runner.py").read_text(
        encoding="utf-8")
    assert "for name in (\"divergence\", \"humidity\", \"dew_point\", \"total\")" not in runner
    assert "for name, value in constraint.items()" in runner
    # And it must only record tensors: C2's dict also carries weights/scales dicts.
    assert "isinstance(value, torch.Tensor)" in runner


def test_the_c2_terms_dict_carries_tensors_and_metadata_separately():
    """The runner's filter is only safe if the tensors are tensors and the rest is not."""
    lat, lon = _grid()
    field = _field()
    body = matching.derivative_matching_terms(field, field, STORE_NAMES, lat, lon, scales=SCALES)
    assert isinstance(body["divergence"], torch.Tensor)
    assert isinstance(body["vorticity"], torch.Tensor)
    assert isinstance(body["total"], torch.Tensor)
    assert isinstance(body["weights"], dict) and isinstance(body["scales"], dict)


def test_every_driver_global_resolves_before_a_gpu_stage_sees_it():
    """A NameError inside a function body is invisible to import and to any test that skips it.

    Two defects reached a GPU stage that way after the driver was split from its protocol module:
    ``freeze`` referenced PLANNED_SECONDS and the driver called a bare ``arm_spec()``. Both were
    only found by running the stage. This resolves every global each function loads, which is the
    check that would have caught them on a laptop.
    """
    import builtins
    import dis
    import importlib
    import types

    problems = []
    for module_name in ("scripts.study_r7_s3_derivative_matching_c2",
                        "scripts.r7_s3_derivative_matching_c2_protocol"):
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if not isinstance(obj, types.FunctionType) or obj.__module__ != module.__name__:
                continue
            for instruction in dis.get_instructions(obj.__code__):
                if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
                    continue
                key = instruction.argval
                if key not in module.__dict__ and not hasattr(builtins, key):
                    problems.append(f"{module_name}.{name}: {key}")
    assert not problems, problems


# --------------------------------------------------------------------- decision 0105/0106 pairing
# These are the tests that would have caught attempt07's defect. The module-level ``zero_at_target``
# test above (field vs itself) is zero by construction and cannot tell a lead-matched pairing from a
# cross-lead one; the tests below drive ``matching_term`` with a batch that carries BOTH fields set
# to different values, so reading the wrong one is observable rather than merely undesirable.

def _matching_contract():
    """A contract with the three things ``matching_term`` requires."""
    return {"autoregression": {"derivative_matching": "c2",
                              "normalization": {"mean": [0.0] * len(STORE_NAMES),
                                                "std": [1.0] * len(STORE_NAMES)},
                              "channels": list(STORE_NAMES),
                              "derivative_scales": dict(SCALES)}}


def _fake_rollout(forecast_last):
    """The minimal object ``matching_term`` reads: ``forecasts`` with a trailing step."""
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    return SimpleNamespace(forecasts=torch.zeros(1, 12, len(STORE_NAMES), 65, 65,
                                                 requires_grad=True).clone())


def test_matching_term_uses_the_last_physical_target_not_atmos_target():
    """The defect itself: a batch whose +6 h and +72 h targets differ must score against +72 h.

    ``atmos_target`` is deliberately set to a wildly different field, so a module that reads it
    instead of ``physical_targets[:, -1]`` produces a different number rather than the same one.
    """
    torch = pytest.importorskip("torch")
    from training.r7_derivative_matching import matching_term

    lat = torch.linspace(27.0, 43.0, 65)
    lon = torch.linspace(107.0, 123.0, 65)
    torch.manual_seed(0)
    last = torch.randn(1, len(STORE_NAMES), 65, 65)
    batch = {"physical_targets": torch.zeros(1, 12, len(STORE_NAMES), 65, 65),
             "atmos_target": torch.full((1, len(STORE_NAMES), 65, 65), 99.0),
             "latitude": lat, "longitude": lon}
    batch["physical_targets"][:, -1] = last
    result = _fake_rollout(None)
    result.forecasts = torch.zeros(1, 12, len(STORE_NAMES), 65, 65)
    result.forecasts[:, -1] = last          # a PERFECT +72 h forecast
    terms = matching_term(result, batch, _matching_contract())
    # Lead-matched: the forecast equals the target it is scored against, so exactly zero.
    assert float(terms["total"]) == 0.0, (
        "a perfect +72 h forecast must score zero against its own target; a nonzero value means "
        "the term read a different lead (this is decision 0105's defect)")


def test_matching_term_refuses_a_batch_without_physical_targets():
    """``atmos_target`` alone must be refused, not silently used as the +6 h stand-in."""
    from training.r7_derivative_matching import matching_term
    batch = {"atmos_target": pytest.importorskip("torch").zeros(1, len(STORE_NAMES), 65, 65)}
    with pytest.raises(ValueError, match="requires physical_targets"):
        matching_term(_fake_rollout(None), batch, _matching_contract())


def test_matching_term_refuses_a_target_with_the_wrong_number_of_steps():
    """A target array whose N disagrees with the rollout cannot be lead-matched; refuse."""
    torch = pytest.importorskip("torch")
    from training.r7_derivative_matching import matching_term
    batch = {"physical_targets": torch.zeros(1, 6, len(STORE_NAMES), 65, 65),
             "atmos_target": torch.zeros(1, len(STORE_NAMES), 65, 65),
             "latitude": torch.linspace(27.0, 43.0, 65),
             "longitude": torch.linspace(107.0, 123.0, 65)}
    with pytest.raises(ValueError, match="no\\s+lead-matched target"):
        matching_term(_fake_rollout(None), batch, _matching_contract())
