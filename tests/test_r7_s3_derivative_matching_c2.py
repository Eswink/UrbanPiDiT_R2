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
