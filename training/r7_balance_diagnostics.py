"""Physical-balance diagnostics for **evaluating** a forecast, never for training one.

This module answers a question the repo has never asked: given a forecast field, is it
physically credible? It deliberately has no loss, no gradient and no contract dependency --
every function is a pure tensor transform, so the same call can be pointed at a model field,
an analysis field or a hand-built counter-example.

The four balances follow `arXiv:2604.01454` §3.1-3.4 (Pasquini et al., 2026), which is the
only systematic evaluation of physical balance in a regional deep-learning weather model:

* :func:`hydrostatic_deviation` -- their eq. (1)-(3),
  ``dPhi_HB = -R * Tbar / pbar * dp`` and ``delta_HB = (dPhi - dPhi_HB) / dPhi``.
* :func:`rossby_number` / :func:`geostrophic_departure` -- their eq. (4)-(6),
  ``v_g = (1/f) k x grad_p Phi`` restricted to ``Ro < 0.1``.
* :func:`column_divergence_residual` -- their §3.4, the vertical integral of divergence.
* :func:`fine_scale_noise` / :func:`radial_amplitude_spectrum` -- their §4.6, where
  ``std(Phi_f1 - Phi_f13)`` is used as a proxy for fine-scale noise.

**Unit convention, and why it differs from the J1 probe.** The store's ``z*`` channels are
*geopotential* in ``m**2 s**-2``, not height in metres. The deviation in eq. (3) is
dimensionless and needs no ``/g`` because ``R * T`` already carries ``m**2 s**-2``. Dividing
by ``g`` while also leaving ``R`` in the formula is the 224x error recorded in
``scripts/r7_s3_operator_j1_probe.py:70-90``, where a sign slip turned a 36.6 m residual into
8193 m and would have manufactured a direction. :func:`geopotential_to_height` exists only
for reporting in metres; it is never used inside a deviation.

Nothing here reads a file, opens a store or imports a checkpoint.
"""
from __future__ import annotations

import math

import torch

#: Earth radius (m), matching `model/r7_process_tensor_diagnostics.py`.
EARTH_RADIUS_M = 6_371_000.0
#: Earth rotation rate (rad s^-1). IAU mean value; the repo had no Coriolis parameter at all.
OMEGA_EARTH = 7.292_115_0e-5
#: Standard gravity (m s^-2), matching `scripts/r7_s3_operator_j1_probe.py`.
GRAVITY = 9.806_65
#: Gas constant for dry air (J K^-1 kg^-1), matching `scripts/r7_s3_operator_j1_probe.py`.
R_DRY = 287.05
#: ``(R_v / R_d) - 1``. The paper writes ``T_v = T(1 + 0.61 q)``; 0.61 is this rounded.
RV_OVER_RD_MINUS_ONE = 0.608

#: Below this |f| the geostrophic relation is unusable and the caller is refused rather than
#: divided by zero. 1e-5 s^-1 is about 5 degrees from the equator.
MIN_ABS_CORIOLIS = 1.0e-5

#: Default Rossby-number ceiling for the geostrophic test, from the paper's eq. (5) discussion.
DEFAULT_ROSSBY_THRESHOLD = 0.1


def coriolis_parameter(latitude: torch.Tensor) -> torch.Tensor:
    """``f = 2 * Omega * sin(phi)`` for latitude **in degrees**, shape preserved.

    The repo had zero occurrences of a Coriolis parameter in active code before this module,
    so there is no earlier convention to inherit. A grid that reaches the equator is refused:
    ``f -> 0`` makes the geostrophic wind divergent, and silently returning ``inf`` would
    report a perfect violation of a balance that is simply undefined there.
    """
    phi = torch.deg2rad(torch.as_tensor(latitude, dtype=torch.float64))
    f = 2.0 * OMEGA_EARTH * torch.sin(phi)
    if bool((f.abs() < MIN_ABS_CORIOLIS).any()):
        raise ValueError(
            "coriolis_parameter refuses |f| < %.1e s^-1 (within ~5 deg of the equator): "
            "geostrophic balance is undefined there" % MIN_ABS_CORIOLIS)
    return f


def virtual_temperature(temperature: torch.Tensor, humidity: torch.Tensor | None) -> torch.Tensor:
    """``T_v = T (1 + 0.61 q)`` in kelvin.

    ``humidity=None`` means "dry": it returns ``T`` unchanged. That is not a silent fallback --
    the store carries ``q`` at 850 and 500 hPa but **not at 250 hPa**, so a 500-250 thickness
    can only be tested with dry temperature. Callers must say so in their protocol; passing
    ``None`` is how they say it.
    """
    t64 = torch.as_tensor(temperature, dtype=torch.float64)
    if humidity is None:
        return t64.clone()
    return t64 * (1.0 + RV_OVER_RD_MINUS_ONE * torch.as_tensor(humidity, dtype=torch.float64))


def hydrostatic_thickness(theoretical_lower_k, theoretical_upper_k, lower_hpa: float,
                          upper_hpa: float) -> torch.Tensor:
    """``dPhi_HB`` from the paper's eq. (2), in ``m**2 s**-2``.

    ``-R * [(T_i + T_i+1) / 2] / [(p_i + p_i+1) / 2] * (p_i+1 - p_i)``. With
    ``p_i+1 < p_i`` the factor ``(p_i+1 - p_i)`` is negative, so the result is **positive**:
    geopotential increases upward. Getting that sign backwards is the 224x inflation the J1
    probe already recorded.
    """
    if not 0.0 < upper_hpa < lower_hpa:
        raise ValueError("thickness needs 0 < upper < lower pressure")
    mean_temperature = 0.5 * (torch.as_tensor(theoretical_lower_k, dtype=torch.float64)
                             + torch.as_tensor(theoretical_upper_k, dtype=torch.float64))
    mean_pressure = 0.5 * (lower_hpa + upper_hpa)
    return -R_DRY * mean_temperature / mean_pressure * (upper_hpa - lower_hpa)


def hydrostatic_deviation(z_lower, z_upper, t_lower, t_upper, lower_hpa: float,
                          upper_hpa: float, q_lower=None, q_upper=None) -> torch.Tensor:
    """``delta_HB = (dPhi - dPhi_HB) / dPhi``: the paper's eq. (3), dimensionless.

    ``z_*`` are **geopotential** (``m**2 s**-2``), exactly as the store holds them; no ``/g``
    is applied and none may be, see the module docstring. Measured thickness is
    ``z_upper - z_lower`` -- the geopotential *gained* going up.

    A zero denominator is returned as ``nan`` rather than ``inf``: an isopycnal column with
    exactly zero thickness is degenerate, not infinitely unbalanced.
    """
    measured = (torch.as_tensor(z_upper, dtype=torch.float64)
                - torch.as_tensor(z_lower, dtype=torch.float64))
    theoretical = hydrostatic_thickness(
        virtual_temperature(t_lower, q_lower), virtual_temperature(t_upper, q_upper),
        lower_hpa, upper_hpa)
    nonzero = measured != 0.0
    safe = torch.where(nonzero, measured, torch.ones_like(measured))
    deviation = torch.where(nonzero, (measured - theoretical) / safe,
                            torch.full_like(measured, float("nan")))
    return deviation


def geopotential_to_height(geopotential: torch.Tensor) -> torch.Tensor:
    """Reported height in metres. **Never** call this inside a balance deviation."""
    return torch.as_tensor(geopotential, dtype=torch.float64) / GRAVITY


def _horizontal_gradient(field, latitude, longitude):
    """``(d/dx, d/dy)`` eastward and northward, on the sphere.

    Delegates to `model.r7_process_tensor_diagnostics.spherical_scalar_gradient` so there is
    one definition of the spherical metric in the repo, not two. That function uses
    nonuniform second-order differences and rejects near-pole grids.
    """
    from model.r7_process_tensor_diagnostics import spherical_scalar_gradient
    return spherical_scalar_gradient(field, latitude, longitude)


def geostrophic_wind(geopotential, latitude, longitude):
    """``v_g = (1/f) k x grad_p Phi``: the paper's eq. (4), returned as ``(u_g, v_g)``.

    With ``k`` up and ``x`` east, ``k x (A_x, A_y) = (-A_y, A_x)``, so
    ``u_g = -(1/f) dPhi/dy`` and ``v_g = (1/f) dPhi/dx`` -- the sign convention of eq. (4),
    not of a rotated frame.
    """
    phi = torch.as_tensor(geopotential, dtype=torch.float64)
    dx, dy = _horizontal_gradient(phi, latitude, longitude)
    f = coriolis_parameter(latitude).reshape(-1, 1)
    return -(1.0 / f) * dy, (1.0 / f) * dx


def rossby_number(u, v, latitude, longitude):
    """``Ro = ||(v . grad) v|| / (f ||v||)``: the paper's eq. (5), dimensionless.

    Zero wind is returned as ``nan``: it is outside the flow the ratio describes, and forcing
    a 0 would silently place every calm cell inside the ``Ro < 0.1`` mask.
    """
    u64 = torch.as_tensor(u, dtype=torch.float64)
    v64 = torch.as_tensor(v, dtype=torch.float64)
    dudx, dudy = _horizontal_gradient(u64, latitude, longitude)
    dvdx, dvdy = _horizontal_gradient(v64, latitude, longitude)
    advected_u = u64 * dudx + v64 * dudy
    advected_v = u64 * dvdx + v64 * dvdy
    numerator = torch.sqrt(advected_u.square() + advected_v.square())
    speed = torch.sqrt(u64.square() + v64.square())
    f = coriolis_parameter(latitude).reshape(-1, 1)
    denominator = f.abs() * speed
    nonzero = denominator > 0.0
    safe = torch.where(nonzero, denominator, torch.ones_like(denominator))
    return torch.where(nonzero, numerator / safe,
                       torch.full_like(denominator, float("nan")))


def geostrophic_departure(u, v, geopotential, latitude, longitude,
                          threshold: float = DEFAULT_ROSSBY_THRESHOLD):
    """``ws_ag = ||v - v_g||`` in m/s, plus the mask it was evaluated on.

    The mask is returned rather than applied, so the caller can report how many cells the
    statistic actually rests on. A balance measured over 3% of the grid and one measured over
    60% are not the same measurement, and only the caller knows which it needed.
    """
    u64 = torch.as_tensor(u, dtype=torch.float64)
    v64 = torch.as_tensor(v, dtype=torch.float64)
    u_g, v_g = geostrophic_wind(geopotential, latitude, longitude)
    departure = torch.sqrt((u64 - u_g).square() + (v64 - v_g).square())
    ro = rossby_number(u64, v64, latitude, longitude)
    mask = torch.isfinite(ro) & (ro < threshold)
    return departure, mask


def column_divergence_residual(d850, d500, d250):
    """The vertical integral of divergence in ``s**-1``: the paper's §3.4.

    Divergence at the three levels the store carries, summed. This is the same reduction as
    ``scripts/r7_s3_operator_j1_probe.py:93`` -- deliberately, so the two cannot disagree.
    """
    return (torch.as_tensor(d850, dtype=torch.float64)
            + torch.as_tensor(d500, dtype=torch.float64)
            + torch.as_tensor(d250, dtype=torch.float64))


def divergence_from_wind(u, v, latitude, longitude):
    """``grad . v`` on the sphere, delegating to the one existing implementation."""
    from model.r7_process_tensor_diagnostics import spherical_divergence
    return spherical_divergence(u, v, latitude, longitude)


def fine_scale_noise(field, window: int = 13) -> torch.Tensor:
    """``std(field - box_mean(field, window))``: the paper's §4.6 noise proxy.

    A ``13x13`` mean closely approximates an unsmoothed value when the field is smooth, so the
    difference stays narrow; fine-scale noise widens it. The standard deviation of that
    difference is therefore a proxy for how much sub-window structure is present.

    The window is clipped to the field, so a field smaller than ``window`` uses its own size
    rather than raising: the diagnostic still means what it says, over a smaller neighbourhood.
    """
    f64 = torch.as_tensor(field, dtype=torch.float64)
    if f64.dim() < 2:
        raise ValueError("fine_scale_noise needs at least two spatial dimensions")
    height, width = f64.shape[-2], f64.shape[-1]
    kh, kw = min(int(window), height), min(int(window), width)
    if kh < 1 or kw < 1:
        raise ValueError("window must be positive")
    # 'same'-padded box mean, computed as a separable average so the cost is linear in the
    # window rather than quadratic.
    padded = torch.nn.functional.pad(f64.unsqueeze(0).unsqueeze(0),
                                     (kw // 2, kw - 1 - kw // 2, kh // 2, kh - 1 - kh // 2),
                                     mode="replicate")
    rows = torch.nn.functional.avg_pool2d(padded, kernel_size=(1, kw), stride=1)
    smooth = torch.nn.functional.avg_pool2d(rows, kernel_size=(kh, 1), stride=1)
    smooth = smooth[0, 0]
    return (f64 - smooth).std(unbiased=False)


def radial_amplitude_spectrum(field, spacing_km: float):
    """Azimuthally averaged amplitude spectrum as ``(wavelength_km, amplitude)``.

    The paper's §4.6 reports FFT amplitude against wavelength. Amplitude is
    ``sqrt(mean(|F|^2))`` within each radial wavenumber bin, so it is comparable across bins
    holding different numbers of modes. Bin 0 is the domain mean and is returned labelled as
    such by its wavelength being ``inf`` -- a DC term is not a wavelength, and the Stage A
    round already recorded that treating band 0 as variability misreads a 1 K offset as
    spectral content.
    """
    f64 = torch.as_tensor(field, dtype=torch.float64)
    if f64.dim() != 2:
        raise ValueError("radial_amplitude_spectrum takes one [H, W] field")
    if not spacing_km > 0.0:
        raise ValueError("spacing_km must be positive")
    height, width = f64.shape
    centred = f64 - f64.mean()
    spectrum = torch.fft.rfft2(centred)
    ky = torch.fft.fftfreq(height, d=1.0) * height
    kx = torch.fft.rfftfreq(width, d=1.0) * width
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    # Every distinct integer radius is one bin; the map is exact because the frequencies are
    # integer multiples of 1/H and 1/W.
    bins = torch.round(radius).to(torch.int64)
    power = spectrum.abs().square()
    wavelengths, amplitudes = [], []
    for index in range(int(bins.max()) + 1):
        selected = bins == index
        count = int(selected.sum())
        if count == 0:
            continue
        amplitude = float(power[selected].mean().sqrt())
        if index == 0:
            wavelengths.append(float("inf"))
        else:
            wavelengths.append(spacing_km * height / float(index))
        amplitudes.append(amplitude)
    return wavelengths, amplitudes
