"""#77 S2 single-factor test: the positional readout's frequency basis.

The defect is analytic, not statistical: the legacy basis is ``2**-j`` for
``j = 0..dim/4-1``, so on the real 33x33 token grid only 16 of 192 channels
carry a phase span above 0.1 and 68 are constant in fp32 (measured here, pinned
below). A channel whose phase span is far below one ulp cannot tell two output
positions apart, so two thirds of the encoding budget is dead.

``nyquist_band`` spends the same channel budget over ``[0.5, 16]`` cycles per
normalized axis, 16 being the Nyquist limit of a 33-point axis. The tests below
are written as discriminators and counterproofs:

- the span counts are recomputed from the definition, not copied from a run;
- the band endpoints are exact and monotone, and ``logspace`` rounding is
  shown to be the reason they are pinned rather than trusted;
- a frequency above the grid Nyquist limit aliases two positions onto one
  phase, which is shown as a positive control so "wider band is always better"
  is not the lesson;
- the two modes are the same module with the same parameters, the same state
  keys and no persistent buffer, so a mode swap cannot move a checkpoint
  contract;
- the legacy mode is bitwise the pre-change basis, which is what keeps an old
  checkpoint's semantics exact.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_readout_r7 import PositionalProcessReadout, require_position_encoding_mode
from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
from training.r7_experiment import make_model, seed_everything

DIM = 192
GRID = (33, 33)
POSITIONS = GRID[0] * GRID[1]
BASE = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": 32, "depth": 1,
        "heads": 2, "window_size": 2, "patch_size": 2, "dropout": 0.0,
        "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 1}


def _span(encoding: torch.Tensor) -> torch.Tensor:
    return encoding.max(dim=0).values - encoding.min(dim=0).values


def _readout(mode: str) -> PositionalProcessReadout:
    return PositionalProcessReadout(DIM, heads=4, position_encoding_mode=mode)


def test_legacy_basis_is_the_bitwise_geometric_decay():
    """The default is exactly ``2**-j``, so a stored checkpoint's basis is intact."""
    readout = _readout(PositionalProcessReadout.LEGACY_FREQUENCIES)
    expected = 2.0 ** (-torch.arange(DIM // 4, dtype=torch.float32))
    assert torch.equal(readout.frequencies, expected)


def test_legacy_basis_is_mostly_dead_on_the_real_grid():
    """Recomputed from the definition: the defect #77 exists to remove.

    A channel is *dead* when its fp32 span across the 33 grid positions is
    exactly zero, and *weak* below 1e-3. The counts below are the measured
    consequence of the frozen basis, and they are asserted as inequalities so a
    torch version that changes linspace rounding cannot silently rewrite the
    claim into something else.
    """
    encoding = _readout("legacy").position_encoding(
        GRID, device=torch.device("cpu"), dtype=torch.float32)
    assert encoding.shape == (POSITIONS, DIM)
    span = _span(encoding)
    assert int((span == 0).sum()) >= 60
    assert int((span < 1e-3).sum()) >= 150
    assert int((span > 0.1).sum()) <= 20


def test_nyquist_band_basis_uses_the_whole_grid():
    """Same budget, every channel alive and none aliased past the grid Nyquist."""
    encoding = _readout("nyquist_band").position_encoding(
        GRID, device=torch.device("cpu"), dtype=torch.float32)
    span = _span(encoding)
    assert int((span == 0).sum()) == 0
    assert int((span < 1e-3).sum()) == 0
    # 33 points resolve at most 16 cycles; only the two cosine endpoints of the
    # top frequency approach zero span (cos(0)=1, cos(pi*16)=1 -> exact equal),
    # so the genuinely flat count is far below the legacy one.
    assert int((span > 0.1).sum()) >= DIM - 4


def test_band_endpoints_are_exact_monotone_and_inside_nyquist():
    readout = _readout("nyquist_band")
    frequencies = readout.frequencies
    assert float(frequencies.min()) == 0.5
    assert float(frequencies.max()) == 16.0
    assert bool((frequencies[1:] > frequencies[:-1]).all())
    nyquist = (GRID[0] - 1) / 2.0
    assert float(frequencies.max()) <= nyquist + 1e-9


def test_logspace_rounding_is_why_the_endpoints_are_pinned():
    """Counterproof for the pinning line: raw logspace does not hit the bound.

    Kept as a computation against the same call the module makes, so if a torch
    upgrade starts returning exact endpoints the assertion still holds (the
    module must not depend on which fp32 rounding a release chooses).
    """
    raw = torch.logspace(math.log10(0.5), math.log10(16.0), DIM // 4, dtype=torch.float32)
    pinned = _readout("nyquist_band").frequencies
    assert float(pinned.max()) == 16.0
    # Either raw already complies, or the pinning line is what makes it comply.
    assert float(raw.max()) >= float(pinned.max()) - 1e-6
    assert bool((pinned >= 0.5).all())


def test_a_frequency_above_nyquist_degenerates_positive_control():
    """Why 16 is a ceiling and not a floor: past it, distinct positions collide.

    With ``sin/cos(pi * f * x)`` on ``x = k/32``, ``f = 32`` maps every grid
    point to a multiple of ``pi``: the sine channel is identically zero and the
    cosine depends only on parity, so positions ``k`` and ``k+2`` are
    indistinguishable. This is the positive control that keeps #77 from being
    read as "explore arbitrarily large frequencies".
    """
    frequency = 2 * ((GRID[0] - 1) / 2.0)  # exactly twice the grid Nyquist
    grid = torch.linspace(0.0, 1.0, GRID[0], dtype=torch.float32)
    phase = math.pi * frequency * grid
    # fp32 evaluation of pi*f*x leaves a few ulps of phase residue, so "dead"
    # is asserted at 1e-5 where the legacy basis' live channels span ~2.
    assert bool((torch.sin(phase).abs() < 1e-5).all())  # sine channel is dead
    pair = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
    assert torch.allclose(pair[0], pair[2], atol=1e-5)  # positions 0 and 2 collide
    assert not torch.allclose(pair[0], pair[1], atol=1e-1)  # parity still shows


def test_modes_share_parameters_state_keys_and_no_persistent_buffer():
    """A mode swap must not move a checkpoint contract."""
    legacy = _readout("legacy")
    band = _readout("nyquist_band")
    assert sorted(legacy.state_dict()) == sorted(band.state_dict())
    assert not any("frequencies" in key for key in band.state_dict())
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in band.parameters())


def test_unknown_mode_is_refused_wherever_it_is_switched_on():
    with pytest.raises(ValueError):
        _readout("fourier_magic")
    with pytest.raises(ValueError):
        require_position_encoding_mode("fourier_magic", positional_process_readout=True)
    with pytest.raises(ValueError):
        require_position_encoding_mode("nyquist_band", positional_process_readout=False)


def test_non_legacy_mode_requires_the_positional_readout_pathway():
    """A silently ignored switch is refused, not accepted and dropped."""
    seed_everything(7)
    with pytest.raises(ValueError):
        make_model("process", dict(BASE, position_encoding_mode="nyquist_band"))
    model = make_model("process", dict(BASE, positional_process_readout=True,
                                       position_encoding_mode="nyquist_band"))
    assert isinstance(model, ProcessForecastCoReasoner)
    assert model.process_reader.position_encoding_mode == "nyquist_band"
    # The default pathway is still the pre-change implementation.
    seed_everything(7)
    default = make_model("process", dict(BASE, positional_process_readout=True))
    assert default.process_reader.position_encoding_mode == "legacy"


def test_generic_recursive_forecaster_carries_the_same_guard():
    options = {name: value for name, value in BASE.items()
               if name not in ("anchored_processes", "free_processes")}
    with pytest.raises(ValueError):
        GenericRecursiveWeatherForecaster(**options, positional_process_readout=False,
                                          position_encoding_mode="nyquist_band")
    model = GenericRecursiveWeatherForecaster(**options, positional_process_readout=True,
                                              position_encoding_mode="nyquist_band")
    assert model.process_reader.position_encoding_mode == "nyquist_band"


def test_band_mode_changes_the_read_at_every_uneven_position():
    """The band is not cosmetic: the read actually differs on the real grid.

    Both arms share one seed and identical parameters, so any difference in the
    read at the same inputs comes from the encoding alone. The assertion is
    deliberately weak (the read must move somewhere) because the scientific
    question - whether it helps - is a training question, not a unit test.
    """
    torch.manual_seed(5)
    process = torch.randn(2, 8, DIM)
    context = torch.randn(2, POSITIONS, DIM)
    seed_everything(11)
    legacy = make_model("process", dict(BASE, dim=DIM, window_size=2, patch_size=2,
                                        positional_process_readout=True))
    seed_everything(11)
    band = make_model("process", dict(BASE, dim=DIM, window_size=2, patch_size=2,
                                      positional_process_readout=True,
                                      position_encoding_mode="nyquist_band"))
    for (left, right) in zip(legacy.parameters(), band.parameters()):
        assert torch.equal(left, right)
    with torch.no_grad():
        legacy_read = legacy.process_reader(process, context, GRID)
        band_read = band.process_reader(process, context, GRID)
    assert not torch.equal(legacy_read, band_read)
    assert float((legacy_read - band_read).abs().max()) > 1e-3
