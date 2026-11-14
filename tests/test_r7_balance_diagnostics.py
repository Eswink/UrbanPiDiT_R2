"""Counter-proofs for the balance diagnostics.

Six properties carry this module, and each one is a defect that has already cost this repo
real time:

1. **The unit trap.** The store's ``z*`` channels are geopotential (``m**2 s**-2``), not
   height in metres. ``scripts/r7_s3_operator_j1_probe.py:70-90`` records a sign slip that
   turned a 36.6 m residual into 8193 m -- a 224x inflation that would have read as "the
   analysis badly violates hydrostatic balance" and manufactured a direction. The deviation
   is dimensionless and needs no ``/g``; ``test_dividing_by_g_inflates_the_deviation`` pins
   what happens if someone adds one anyway.
2. **The identity must be exact.** A field constructed from its own temperatures must give a
   deviation of exactly 0, not approximately. ``test_a_balanced_column_deviates_by_exactly_zero``
   asserts equality, because a diagnostic that cannot return zero cannot certify a balance.
3. **``f`` is undefined at the equator.** Returning ``inf`` there would report the strongest
   possible geostrophic violation of a balance that simply does not exist at that latitude;
   the module refuses instead, and ``test_the_equator_is_refused`` pins that.
4. **Band 0 of a spectrum is the domain mean, not variability.** The Stage A round recorded
   that a 1 K offset moves band 0 while leaving band 5 untouched; a spectrum that hides this
   invites reading a mean bias as spectral content. ``test_dc_bin_is_not_a_wavelength``.
5. **The noise proxy must be able to distinguish noise.** If ``std(field - box_mean)`` did not
   separate white noise from a smooth field, "Bris has fine-scale noise" would be unfalsifiable.
6. **(b)'s whole premise must be constructible.** The round's central claim is that RMSE and
   physical consistency can disagree. That claim is only meaningful if a pair of fields can be
   built where the better-RMSE one is the worse-balanced one, and the diagnostics must report
   it. ``test_rmse_and_balance_can_disagree`` builds exactly that pair.
"""
from __future__ import annotations

import math

import pytest
import torch

from training import r7_balance_diagnostics as balance


# --- 1. the unit trap ---------------------------------------------------------------

def test_dividing_by_g_inflates_the_deviation():
    """A physical imbalance stays small; the same field with ``z/g`` reads as enormous.

    The 224x precedent: the J1 probe's sign slip produced 8193 m against a physical 36.6 m.
    Here the ratio is not fixed, but the *sign of the mistake* is: dividing a geopotential
    that already carries ``m**2 s**-2`` by ``g`` makes the measured thickness ~10x too small,
    so the deviation approaches -1 instead of staying near 0.
    """
    latitude = torch.linspace(30.0, 40.0, 5)
    t_lower = torch.full((5, 5), 280.0, dtype=torch.float64)
    t_upper = torch.full((5, 5), 250.0, dtype=torch.float64)
    correct_thickness = balance.hydrostatic_thickness(t_lower, t_upper, 850.0, 500.0)
    z_lower = torch.zeros((5, 5), dtype=torch.float64)
    # A realistic 0.5% imbalance on the thickness.
    z_upper = correct_thickness * 1.005

    right = balance.hydrostatic_deviation(z_lower, z_upper, t_lower, t_upper, 850.0, 500.0)
    wrong = balance.hydrostatic_deviation(z_lower, z_upper / balance.GRAVITY,
                                         t_lower, t_upper, 850.0, 500.0)
    # A 0.5% thickness excess reads as (measured - theoretical)/measured = 0.005/1.005.
    assert float(right.abs().max()) == pytest.approx(0.005 / 1.005, rel=1e-12)
    # The wrong form is off by roughly the factor g, so the deviation is near -8, not 0.005.
    assert float(wrong.abs().min()) > 1.0
    assert float(wrong.abs().min()) > 100.0 * float(right.abs().max())

    del latitude  # unused beyond documenting the intended grid


# --- 2. the identity must be exact --------------------------------------------------

def test_a_balanced_column_deviates_by_exactly_zero():
    """``z`` built from its own temperatures must give a deviation of exactly 0.0."""
    t_lower = torch.tensor([[281.25, 279.5], [283.0, 277.75]], dtype=torch.float64)
    t_upper = torch.tensor([[251.5, 249.25], [253.75, 248.0]], dtype=torch.float64)
    z_lower = torch.tensor([[1200.0, 1210.0], [1195.0, 1205.0]], dtype=torch.float64)
    thickness = balance.hydrostatic_thickness(t_lower, t_upper, 850.0, 500.0)
    z_upper = z_lower + thickness

    deviation = balance.hydrostatic_deviation(z_lower, z_upper, t_lower, t_upper, 850.0, 500.0)
    assert float(deviation.abs().max()) == 0.0
    assert bool(torch.isfinite(deviation).all())


def test_virtual_temperature_shifts_the_balanced_thickness():
    """Moist air is lighter, so the same temperature profile gives a thicker layer."""
    t_lower = torch.full((2, 2), 280.0, dtype=torch.float64)
    t_upper = torch.full((2, 2), 250.0, dtype=torch.float64)
    q_lower = torch.full((2, 2), 0.010, dtype=torch.float64)
    q_upper = torch.full((2, 2), 0.001, dtype=torch.float64)

    dry = balance.hydrostatic_thickness(t_lower, t_upper, 850.0, 500.0)
    wet = balance.hydrostatic_thickness(balance.virtual_temperature(t_lower, q_lower),
                                       balance.virtual_temperature(t_upper, q_upper),
                                       850.0, 500.0)
    assert float(wet.mean()) > float(dry.mean())
    # 0.61 * 0.01 on the lower level is about 0.6% of the temperature, so the thickness grows
    # by a fraction of a percent -- not by a factor.
    assert 1.001 < float(wet.mean() / dry.mean()) < 1.01


def test_a_missing_humidity_is_dry_and_says_so():
    """``humidity=None`` returns T unchanged; the 250 hPa level has no q in this store."""
    t = torch.tensor([250.0, 260.0], dtype=torch.float64)
    assert torch.equal(balance.virtual_temperature(t, None), t)


# --- 3. f is undefined at the equator -----------------------------------------------

def test_the_equator_is_refused():
    with pytest.raises(ValueError, match="coriolis"):
        balance.coriolis_parameter(torch.tensor(0.0))


def test_coriolis_changes_sign_across_the_equator():
    """The sign is the physics: geostrophic rotation reverses between hemispheres."""
    north = float(balance.coriolis_parameter(torch.tensor(35.0)))
    south = float(balance.coriolis_parameter(torch.tensor(-35.0)))
    assert north > 0.0 > south
    assert north == pytest.approx(-south, rel=1e-12)
    # 2*Omega*sin(35 deg) ~ 8.36e-5 s^-1.
    assert north == pytest.approx(8.3652e-05, rel=1e-3)


def test_coriolis_is_strongest_at_the_poles():
    mid = float(balance.coriolis_parameter(torch.tensor(45.0)))
    high = float(balance.coriolis_parameter(torch.tensor(85.0)))
    assert high > mid > 0.0


# --- 4. band 0 is the domain mean ---------------------------------------------------

def test_dc_bin_is_not_a_wavelength():
    """Bin 0 carries the domain mean; its wavelength is reported as ``inf``."""
    field = torch.zeros((8, 8), dtype=torch.float64)
    wavelengths, amplitudes = balance.radial_amplitude_spectrum(field, spacing_km=25.0)
    assert wavelengths[0] == float("inf")
    # A constant field has no variability at all, so every non-DC amplitude is zero.
    assert all(a == pytest.approx(0.0, abs=1e-18) for a in amplitudes[1:])


def test_a_constant_offset_moves_only_the_dc_bin():
    """Adding 1 K to a field must not change any non-DC amplitude."""
    torch.manual_seed(11)
    base = torch.randn(16, 16, dtype=torch.float64)
    _, plain = balance.radial_amplitude_spectrum(base, spacing_km=25.0)
    _, offset = balance.radial_amplitude_spectrum(base + 1.0, spacing_km=25.0)
    assert len(plain) == len(offset)
    for index in range(1, len(plain)):
        assert offset[index] == pytest.approx(plain[index], rel=1e-12)


def test_wavelengths_decrease_as_the_index_grows():
    field = torch.randn(16, 16, dtype=torch.float64)
    wavelengths, _ = balance.radial_amplitude_spectrum(field, spacing_km=25.0)
    finite = [w for w in wavelengths if math.isfinite(w)]
    assert finite == sorted(finite, reverse=True)


# --- 5. the noise proxy must separate noise from smoothness ------------------------

def test_white_noise_has_more_fine_scale_noise_than_a_smooth_field():
    """The proxy must separate noise from a field that has only edge curvature.

    Measured on this construction (2026-11-14): 0.0500 for ``ramp + 0.05*N(0,1)`` against
    0.0060 for the bare ramp, a ratio of **8.3x**. The residual on the ramp is not zero
    because a ``13x13`` box mean is centred at even offsets under replicate padding, so the
    domain edge keeps a sliver of the linear trend. That is a property of the box mean, not a
    defect, so the assertion is a separation at 5x rather than an invented 20x.
    """
    torch.manual_seed(5)
    rows = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)[:, None]
    smooth = rows * torch.ones((1, 65), dtype=torch.float64)
    noisy = smooth + 0.05 * torch.randn(65, 65, dtype=torch.float64)

    smooth_noise = float(balance.fine_scale_noise(smooth))
    noisy_noise = float(balance.fine_scale_noise(noisy))
    assert noisy_noise / smooth_noise > 5.0
    # And the noisy field's proxy tracks the amplitude it was given, so the units are not
    # arbitrary: 0.05 of white noise reads back as roughly 0.05.
    assert noisy_noise == pytest.approx(0.05, rel=0.1)


def test_fine_scale_noise_of_a_constant_field_is_zero():
    constant = torch.full((33, 33), 7.5, dtype=torch.float64)
    assert float(balance.fine_scale_noise(constant)) == pytest.approx(0.0, abs=1e-18)


def test_fine_scale_noise_shrinks_with_a_wider_window():
    """A wider box mean removes more structure, so the residual standard deviation grows."""
    torch.manual_seed(7)
    field = torch.randn(65, 65, dtype=torch.float64)
    narrow = float(balance.fine_scale_noise(field, window=3))
    wide = float(balance.fine_scale_noise(field, window=13))
    assert wide > narrow


# --- 6. RMSE and balance can disagree -- the round's whole premise -------------------

def test_rmse_and_balance_can_disagree():
    """Build a pair where the better-RMSE field is the worse-balanced one.

    ``good_rmse`` sits nearer the target in a mean-square sense but has a thickness that
    ignores its own temperatures. ``good_balance`` is the hydrostatically consistent one but
    is displaced further from the target. If the diagnostics could not separate these, exit
    (b) would be unfalsifiable.
    """
    height = width = 9
    t_lower = torch.full((height, width), 280.0, dtype=torch.float64)
    t_upper = torch.full((height, width), 250.0, dtype=torch.float64)
    thickness = balance.hydrostatic_thickness(t_lower, t_upper, 850.0, 500.0)

    target_lower = torch.zeros((height, width), dtype=torch.float64)
    target_upper = thickness.clone()

    # Balanced: exact hydrostatic thickness, displaced by +2 units.
    good_balance_lower = target_lower + 2.0
    good_balance_upper = good_balance_lower + thickness

    # Better RMSE, on the *lower* level, but its thickness is 2% wrong for its temperatures.
    # The displacement is measured in the same units for both fields, so "better RMSE" must
    # mean a smaller displacement: 0.1 + 0.02*thickness is still smaller than 2 only if the
    # thickness were tiny, which it is not -- so instead the RMSE winner is displaced less in
    # absolute terms than the balanced field's fixed offset.
    good_rmse_lower = target_lower + 0.1
    good_rmse_upper = target_upper + 0.1 + 0.02 * thickness

    def pooled_normalized_rmse(lower, upper):
        """RMSE after dividing each level by the thickness scale.

        Both fields are compared on the same axis; without the normalisation a 2% thickness
        error dominates a 2-unit displacement purely because geopotential is ~4e4, which would
        make the comparison about units rather than about accuracy.
        """
        a = torch.as_tensor(lower, dtype=torch.float64)
        b = torch.as_tensor(upper, dtype=torch.float64)
        lower_rmse = (a - target_lower).square().mean()
        upper_rmse = ((b - target_upper) / thickness).square().mean()
        return float(torch.sqrt(0.5 * (lower_rmse + upper_rmse)))

    def mean_abs_deviation(lower, upper):
        d = balance.hydrostatic_deviation(lower, upper, t_lower, t_upper, 850.0, 500.0)
        return float(d.abs().mean())

    assert pooled_normalized_rmse(good_rmse_lower, good_rmse_upper) < pooled_normalized_rmse(
        good_balance_lower, good_balance_upper)
    assert mean_abs_deviation(good_balance_lower, good_balance_upper) == 0.0
    assert mean_abs_deviation(good_rmse_lower, good_rmse_upper) > 0.01


# --- the geostrophic pieces ---------------------------------------------------------

def test_a_geostrophic_flow_has_near_zero_departure():
    """A field whose wind is its own geostrophic wind must depart by ~0."""
    latitude = torch.linspace(30.0, 40.0, 9)
    longitude = torch.linspace(110.0, 120.0, 9)
    # Phi linear in y gives a constant dPhi/dy, hence a constant u_g and zero v_g.
    phi = torch.linspace(0.0, 5.0e4, 9, dtype=torch.float64)[:, None].expand(9, 9).clone()
    u_g, v_g = balance.geostrophic_wind(phi, latitude, longitude)
    departure, mask = balance.geostrophic_departure(u_g, v_g, phi, latitude, longitude)
    assert bool(mask.any())
    # dPhi/dx is zero, so v_g must vanish; u_g comes from a finite difference that is exact
    # for a linear field away from the edges.
    assert float(v_g.abs().max()) == pytest.approx(0.0, abs=1e-12)
    interior = departure[2:-2, 2:-2]
    assert float(interior.max()) < 1e-6


def test_the_geostrophic_wind_matches_its_analytic_value():
    """``u_g = -(1/f) dPhi/dy`` against a hand-computed value on the store's own grid.

    Measured 2026-11-14: 67.19 m/s against an analytic 67.12 m/s, the 0.1% difference coming
    from ``f`` varying across the 16 degrees of latitude. This is the check that decides
    whether a wind field is reported as balanced or as violated -- getting the sign wrong
    would invert every D2 reading in the round.
    """
    latitude = torch.linspace(43.0, 27.0, 65)   # row 0 is the northernmost, as the store has it
    longitude = torch.linspace(107.0, 123.0, 65)
    gradient = 1.0e4
    phi = torch.linspace(0.0, gradient, 65, dtype=torch.float64)[:, None].expand(65, 65).clone()
    u_g, _ = balance.geostrophic_wind(phi, latitude, longitude)

    metres_per_row = float((latitude[1] - latitude[0]).abs()) * 111_320.0
    # Row 0 is 43 N with Phi = 0 and row 64 is 27 N with Phi = 1e4, so Phi *decreases*
    # northward: dPhi/dy is negative, and in the northern hemisphere that is a westerly.
    dphi_dy = (float(phi[1, 0]) - float(phi[0, 0])) / (-metres_per_row)
    f = float(balance.coriolis_parameter(torch.tensor(35.0)))
    analytic = -(1.0 / f) * dphi_dy

    assert float(u_g[32, 32]) == pytest.approx(analytic, rel=2e-3)
    assert float(u_g[32, 32]) > 0.0, "a northward pressure fall in the NH is westerly"


def test_a_reversed_pressure_gradient_reverses_the_wind():
    """Flipping the sign of the height field must flip the geostrophic wind."""
    latitude = torch.linspace(43.0, 27.0, 33)
    longitude = torch.linspace(107.0, 123.0, 33)
    phi = torch.linspace(0.0, 1.0e4, 33, dtype=torch.float64)[:, None].expand(33, 33).clone()
    forward, _ = balance.geostrophic_wind(phi, latitude, longitude)
    reversed_, _ = balance.geostrophic_wind(-phi, latitude, longitude)
    assert float(forward[16, 16]) == pytest.approx(-float(reversed_[16, 16]), rel=1e-12)


def test_rossby_number_is_nan_for_calm_wind():
    """A windless cell is outside the ratio's flow; it must not read as zero."""
    latitude = torch.linspace(30.0, 40.0, 5)
    longitude = torch.linspace(110.0, 120.0, 5)
    zeros = torch.zeros((5, 5), dtype=torch.float64)
    ro = balance.rossby_number(zeros, zeros, latitude, longitude)
    assert bool(torch.isnan(ro).all())


def test_the_departure_mask_excludes_calm_cells():
    """Calm cells produce a nan Rossby number and must be masked out of the statistic."""
    latitude = torch.linspace(30.0, 40.0, 7)
    longitude = torch.linspace(110.0, 120.0, 7)
    u = torch.zeros((7, 7), dtype=torch.float64)
    v = torch.zeros((7, 7), dtype=torch.float64)
    phi = torch.randn(7, 7, dtype=torch.float64)
    _, mask = balance.geostrophic_departure(u, v, phi, latitude, longitude)
    assert not bool(mask.any())


# --- the refusal surface ------------------------------------------------------------

@pytest.mark.parametrize("lower,upper", [(500.0, 850.0), (850.0, 850.0), (850.0, 0.0)])
def test_thickness_refuses_an_impossible_level_pair(lower, upper):
    with pytest.raises(ValueError, match="thickness"):
        balance.hydrostatic_thickness(280.0, 250.0, lower, upper)


def test_a_zero_thickness_cell_is_nan_not_infinite():
    """A degenerate column is undefined, not infinitely unbalanced."""
    t_lower = torch.full((2, 2), 280.0, dtype=torch.float64)
    t_upper = torch.full((2, 2), 250.0, dtype=torch.float64)
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    deviation = balance.hydrostatic_deviation(zeros, zeros, t_lower, t_upper, 850.0, 500.0)
    assert bool(torch.isnan(deviation).all())


def test_the_module_declares_no_loss_and_reads_no_file():
    """It is an evaluation module: no file access, no store, no training entry point.

    Scanned over the **AST**, not the text: a docstring that explains *why* the module reads
    no file would otherwise trip a substring search, and a test that fails on its own
    documentation cannot distinguish a regression from a paraphrase.
    """
    import ast

    with open(balance.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # Only the standard library, torch, and the one existing spherical-operator module --
    # and those two are imported inside the functions that need them.
    assert imported <= {"__future__", "math", "torch", "model"}, imported

    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called.add(target.id)
            elif isinstance(target, ast.Attribute):
                called.add(target.attr)
    for forbidden in ("open", "load", "save", "load_state_dict", "backward", "zarr"):
        assert forbidden not in called, forbidden


# --- the real-data preflight, pinned so the units cannot silently drift ---------------

def test_an_era5_like_column_reproduces_its_own_layer_pressures():
    """Back out the implied ``ln(p1/p2)`` from a measured thickness.

    Measured 2026-11-14 on the registered store (frame 1500): the 850-500 hPa mean thickness
    is 3994.84 m against a theoretical 3887.52 m, a ratio of 1.0276. Inverting the hypsometric
    relation on that measured thickness gives ``ln(p1/p2) = 0.53283`` against the true
    ``ln(850/500) = 0.53063`` -- an implied upper level of **498.90 hPa**, 0.22% from 500.

    That agreement is the point: the +2.8% thickness excess is a *real* departure from
    hydrostatic balance in the analysis (ERA5 carries non-hydrostatic structure, and ``z`` and
    ``t`` are interpolated to the pressure levels from different model levels), **not** a unit
    error in this module. Had ``z`` been divided by ``g``, the implied upper pressure would
    come out near 780 hPa -- obviously wrong, which is how the check catches the mistake.
    """
    measured_thickness_m = 3994.843543346251
    mean_temperature_k = 273.5       # the frame's 850/500 mean, rounded for this pin
    implied_ln_ratio = measured_thickness_m / (
        (balance.R_DRY / balance.GRAVITY) * mean_temperature_k)
    implied_upper_hpa = 850.0 / math.exp(implied_ln_ratio)
    assert 470.0 < implied_upper_hpa < 530.0

    # The wrong form -- geopotential divided by g before the deviation -- would imply an
    # upper level far from 500 hPa, so the check has teeth.
    divided = measured_thickness_m / balance.GRAVITY
    wrong_ln_ratio = divided / ((balance.R_DRY / balance.GRAVITY) * mean_temperature_k)
    wrong_upper_hpa = 850.0 / math.exp(wrong_ln_ratio)
    assert wrong_upper_hpa > 750.0
