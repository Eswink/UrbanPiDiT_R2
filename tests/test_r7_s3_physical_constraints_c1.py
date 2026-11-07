"""CPU counterproofs for Stage C1's physical constraints; no GPU, no data, no claim.

The property that matters most is the one decision 0092 rests on: **each term returns exactly
zero on a field that already satisfies its constraint**. If that fails, these are not
constraints but smoothing priors in disguise -- the closed spatial-gradient loss's failure
mode -- so it is asserted with exact equality rather than a tolerance.

The second property is the family argument's other half: the optimum is *not* a constant
field. A constant non-divergent wind field, a constant non-negative humidity and a dew point
below T all score zero, which a ``‖∇pred‖²`` penalty would not.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from training import r7_physical_constraints as constraints


def _grid(h=5, w=6):
    lat = torch.linspace(50.0, 46.0, h, dtype=torch.float64)
    lon = torch.linspace(10.0, 15.0, w, dtype=torch.float64)
    return lat, lon


def test_the_weights_are_the_declared_ones():
    """w_div and w_dew are quoted from legacy; w_q is this round's choice and must say so."""
    assert constraints.W_DIVERGENCE == 0.1
    assert constraints.W_DEW_POINT == 0.5
    assert constraints.W_HUMIDITY == 0.5
    source = Path(constraints.__file__).read_text(encoding="utf-8")
    assert "legacy" in source and "physical_consistency.py:68-70" in source
    assert "this round's choice" in source.lower() or "This round's choice" in source


def test_a_nondivergent_field_scores_exactly_zero():
    """A constant wind field has zero divergence, so the term must be exactly 0.0, not ~0."""
    lat, lon = _grid()
    u = torch.full((5, 6), 7.0, dtype=torch.float64)
    v = torch.full((5, 6), -3.0, dtype=torch.float64)
    value = constraints.divergence_penalty(u, v, lat, lon)
    assert float(value) == 0.0


def test_a_divergent_field_scores_positive():
    """The term must actually see divergence, or it is inert."""
    lat, lon = _grid()
    # A u that varies strongly in longitude is divergent by construction.
    u = torch.linspace(-40.0, 40.0, 6, dtype=torch.float64).repeat(5, 1)
    v = torch.zeros((5, 6), dtype=torch.float64)
    value = constraints.divergence_penalty(u, v, lat, lon)
    assert float(value) > 0.0


def test_the_tolerance_suppresses_small_divergence_only():
    """Below the frozen dead-band the term is silent; above it, it charges.

    The dead-band is in units of ``DIVERGENCE_SCALE`` (decision 0094), so a field whose
    divergence is well under one scale unit is free and one that exceeds it is charged.
    """
    lat, lon = _grid()
    u = torch.linspace(-1e-6, 1e-6, 6, dtype=torch.float64).repeat(5, 1)
    v = torch.zeros((5, 6), dtype=torch.float64)
    assert float(constraints.divergence_penalty(u, v, lat, lon)) == 0.0
    assert constraints.DIVERGENCE_TOLERANCE == 1.0
    assert constraints.DIVERGENCE_SCALE == 1e-5
    # A field well above one scale unit must charge.
    strong = torch.linspace(-40.0, 40.0, 6, dtype=torch.float64).repeat(5, 1)
    assert float(constraints.divergence_penalty(strong, v, lat, lon)) > 1.0


def test_every_term_is_dimensionless_so_it_can_be_added_to_a_normalized_loss():
    """Decision 0094: the pre-freeze measurement found the absolute form 1e-11..2e-6 against L~1.

    A violation of exactly one declared scale unit must score exactly 1.0 (outside the
    divergence dead-band), which is what makes the bundle O(1) next to the MSE rather than
    numerically invisible. This is asserted arithmetically, not by a run.
    """
    # Humidity: q = -1 scale unit -> relu(-q/scale) = 1 -> mean of 1 = 1.
    assert float(constraints.humidity_nonnegative(
        torch.tensor([-constraints.HUMIDITY_SCALE], dtype=torch.float64))) == pytest.approx(1.0)
    # Dew point: Td - T = 1 scale unit -> 1.
    assert float(constraints.dew_point_le_temperature(
        torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64)
    )) == pytest.approx(1.0 / constraints.TEMPERATURE_SCALE ** 2)
    # Divergence: the hinge subtracts the dead-band, so one scale unit is still free by design
    # and two scale units charge (2 - 1)^2 = 1.
    assert constraints.DIVERGENCE_TOLERANCE == 1.0
    # The scales themselves are frozen constants, recorded so a run cannot silently retune them.
    assert (constraints.DIVERGENCE_SCALE, constraints.HUMIDITY_SCALE,
            constraints.TEMPERATURE_SCALE) == (1e-5, 1e-3, 1.0)


def test_nonnegative_humidity_scores_exactly_zero():
    q = torch.tensor([0.0, 1e-6, 0.005, 0.02], dtype=torch.float64)
    assert float(constraints.humidity_nonnegative(q)) == 0.0


def test_negative_humidity_is_charged():
    q = torch.tensor([0.01, -0.002], dtype=torch.float64)
    value = float(constraints.humidity_nonnegative(q))
    assert value > 0.0
    # Quadratic in the dimensionless violation: (-0.002 / 1e-3)^2 / 2 = 2.0.
    assert value == pytest.approx((-0.002 / constraints.HUMIDITY_SCALE) ** 2 / 2, rel=1e-12)


def test_dew_point_below_temperature_scores_exactly_zero():
    t = torch.tensor([280.0, 290.0], dtype=torch.float64)
    td = torch.tensor([275.0, 290.0], dtype=torch.float64)
    assert float(constraints.dew_point_le_temperature(td, t)) == 0.0


def test_dew_point_above_temperature_is_charged():
    t = torch.tensor([280.0], dtype=torch.float64)
    td = torch.tensor([282.0], dtype=torch.float64)
    # Dimensionless: ((282 - 280) / 1 K)^2 = 4.0.
    assert float(constraints.dew_point_le_temperature(td, t)) == pytest.approx(
        (2.0 / constraints.TEMPERATURE_SCALE) ** 2, rel=1e-12)


def test_the_three_terms_together_are_zero_on_a_satisfying_field():
    """The family argument's core: a physically admissible field costs nothing.

    A constant wind field, small non-negative humidity and a dew point below T. A gradient
    penalty would still charge this field for its structure; these constraints do not.
    """
    lat, lon = _grid()
    channels = torch.zeros((8, 5, 6), dtype=torch.float64)
    names = ("mslp", "t850", "q850", "u850", "v850", "t500", "u500", "v500")
    index = {name: i for i, name in enumerate(names)}
    channels[index["t850"]] = 285.0
    channels[index["q850"]] = 0.004
    channels[index["u850"]] = 5.0
    channels[index["v850"]] = -2.0
    body = constraints.physical_constraint_terms(channels, names, lat, lon)
    assert float(body["divergence"]) == 0.0
    assert float(body["humidity"]) == 0.0
    assert float(body["dew_point"]) == 0.0
    assert float(body["total"]) == 0.0


def test_the_optimum_is_not_a_constant_field():
    """A structured but admissible field also costs zero, so the optimum set is not degenerate."""
    lat, lon = _grid()
    channels = torch.zeros((8, 5, 6), dtype=torch.float64)
    names = ("mslp", "t850", "q850", "u850", "v850", "t500", "u500", "v500")
    index = {name: i for i, name in enumerate(names)}
    # A spatially varying temperature and humidity: not constant, still admissible.
    channels[index["t850"]] = torch.linspace(280.0, 292.0, 6, dtype=torch.float64).repeat(5, 1)
    channels[index["q850"]] = torch.linspace(0.002, 0.010, 6, dtype=torch.float64).repeat(5, 1)
    channels[index["u850"]] = 4.0
    channels[index["v850"]] = -1.0
    body = constraints.physical_constraint_terms(channels, names, lat, lon)
    assert float(body["total"]) == 0.0


def test_channels_are_matched_by_name_not_by_position():
    """The store's order is t2m,u10,v10,mslp,z850,... -- a positional zip would read the wrong fields.

    This is the counterproof for the defect found before freezing: with a positional zip the
    divergence operator would receive ``t2m`` where it asked for ``mslp`` and ``mslp`` where it
    asked for ``u850``, and every term would return a plausible but meaningless number. Here the
    same physical field is presented in the store's real 17-channel order and must give exactly
    the same answer as the 8-channel contiguous presentation above.
    """
    lat, lon = _grid()
    store_names = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
                   "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
    channels = torch.zeros((len(store_names), 5, 6), dtype=torch.float64)
    index = {name: i for i, name in enumerate(store_names)}
    channels[index["t850"]] = 285.0
    channels[index["q850"]] = 0.004
    channels[index["u850"]] = 5.0
    channels[index["v850"]] = -2.0
    # Values that would make a positional zip loudly wrong: t2m is huge and u10 diverges.
    channels[index["t2m"]] = 1.0e6
    channels[index["u10"]] = torch.linspace(-500.0, 500.0, 6, dtype=torch.float64).repeat(5, 1)
    body = constraints.physical_constraint_terms(channels, store_names, lat, lon)
    assert float(body["total"]) == 0.0


def test_a_reordered_store_is_refused_rather_than_rescaled():
    """A name that is absent must be a KeyError, not a silent positional read."""
    lat, lon = _grid()
    channels = torch.zeros((8, 5, 6), dtype=torch.float64)
    names = ("mslp", "t850", "q850", "u850", "v850", "t500", "u500", "v250")
    with pytest.raises(KeyError, match="missing the named channels"):
        constraints.physical_constraint_terms(channels, names, lat, lon)
    with pytest.raises(ValueError, match="names were declared"):
        constraints.physical_constraint_terms(channels, names[:7], lat, lon)


def test_an_unknown_weight_name_is_refused():
    lat, lon = _grid()
    channels = torch.zeros((8, 5, 6), dtype=torch.float64)
    names = ("mslp", "t850", "q850", "u850", "v850", "t500", "u500", "v500")
    with pytest.raises(KeyError, match="unknown constraint weights"):
        constraints.physical_constraint_terms(channels, names, lat, lon,
                                              weights={"smoothness": 1.0})


def test_a_too_narrow_prediction_is_refused():
    lat, lon = _grid()
    names = ("mslp", "t850", "q850", "u850", "v850", "t500", "u500", "v500")
    with pytest.raises(ValueError, match="channels"):
        constraints.physical_constraint_terms(torch.zeros((3, 5, 6)), names[:3], lat, lon)
    # Enough channels but the wrong count of names is also a ValueError, not a silent zip.
    with pytest.raises(ValueError, match="names were declared"):
        constraints.physical_constraint_terms(torch.zeros((8, 5, 6)), names[:7], lat, lon)


def test_the_module_does_not_import_legacy():
    """Decision 0092: legacy bundles closed and unauthorised terms, so it is quoted, not used."""
    source = Path(constraints.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("legacy" in name for name in imported), imported


def test_the_terms_are_untargeted():
    """No term may read a target: the public surface takes a prediction and coordinates only."""
    import inspect
    for name in ("divergence_penalty", "humidity_nonnegative", "dew_point_le_temperature"):
        signature = inspect.signature(getattr(constraints, name))
        for parameter in signature.parameters:
            assert "target" not in parameter.lower(), (name, parameter)
    signature = inspect.signature(constraints.physical_constraint_terms)
    assert "target" not in " ".join(signature.parameters).lower()


def test_relu_not_softplus_so_satisfying_fields_are_free():
    """A softplus hinge would charge a satisfying field; the code must call clamp_min instead.

    The check is on *calls*, not on the text: the module's docstring names softplus on purpose,
    to record why it is not used, so a substring test would fail on its own explanation.
    """
    tree = ast.parse(Path(constraints.__file__).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {getattr(node.func, "attr", getattr(node.func, "id", "")) for node in calls}
    assert "clamp_min" in names
    assert "softplus" not in names


def test_denormalize_round_trips_the_store_convention():
    """The inverse of the dataset's own ``(x − mean)/std`` must be exact arithmetic."""
    channels = ("t2m", "q850", "u850")
    mean = torch.tensor([280.0, 0.005, 3.0], dtype=torch.float64)
    std = torch.tensor([10.0, 0.004, 2.0], dtype=torch.float64)
    normalized = torch.zeros((1, 3, 2, 2), dtype=torch.float64)
    physical = constraints.denormalize(normalized, (mean, std), channels)
    # Zero in normalized space is the mean in physical space.
    for index, expected in enumerate(mean):
        assert torch.all(physical[..., index, :, :] == expected)
    # And a known normalized value maps back through the same affine form.
    normalized[..., 1, 0, 0] = 2.0
    physical = constraints.denormalize(normalized, (mean, std), channels)
    assert float(physical[..., 1, 0, 0]) == pytest.approx(0.005 + 2.0 * 0.004)


def test_denormalize_refuses_a_channel_count_mismatch():
    """A store whose channel order moved must fail here, not rescale the wrong variable."""
    mean = torch.zeros(3, dtype=torch.float64)
    std = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="channels names"):
        constraints.denormalize(torch.zeros((2, 2, 2)), (mean, std), ("a", "b", "c"))
    with pytest.raises(ValueError, match="entries"):
        constraints.denormalize(torch.zeros((3, 2, 2)), (torch.zeros(2), torch.ones(2)),
                                ("a", "b", "c"))


def test_the_axis_is_absent_by_default_so_the_registered_path_is_untouched():
    """``physical_constraints`` must be an opt-in contract key, never a default."""
    from training.r7_physical_constraints import constraint_term as _constraint_term
    import inspect
    source = inspect.getsource(_constraint_term)
    # The off path must return before constructing anything: an early None is what keeps the
    # registered arithmetic bit-identical, and a "weight 0" path would not.
    assert "if axis is None:\n        return None" in source
    tree = ast.parse(source)
    # ast.walk yields in breadth-first order, not source order, so sort by line: the guard
    # must be the *first* return executed, which is the one with the smallest line number.
    first = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Return)),
                   key=lambda node: node.lineno)[0]
    assert isinstance(first.value, ast.Constant) and first.value.value is None


def test_an_unknown_constraint_axis_is_refused():
    from training.r7_physical_constraints import constraint_term as _constraint_term
    with pytest.raises(ValueError, match="must be absent or 'c1'"):
        _constraint_term(None, {}, {"autoregression": {"physical_constraints": "c2"}})


def test_the_constraint_refuses_to_mix_with_sam_or_gradient_surgery():
    """A single-factor claim is only honest if the combination is refused, not merely unused."""
    source = Path(constraints.__file__).parent.joinpath("r7_long_rollout_runner.py").read_text(
        encoding="utf-8") + Path(constraints.__file__).read_text(encoding="utf-8")
    assert "cannot be combined with gradient surgery or SAM" in source


def test_the_c1_axis_requires_the_normalization_and_channel_order():
    """Without them the denormalisation would be an unregistered hidden variable."""
    from training.r7_physical_constraints import constraint_term as _constraint_term
    with pytest.raises(ValueError, match="requires the store's normalization"):
        _constraint_term(None, {}, {"autoregression": {"physical_constraints": "c1"}})
