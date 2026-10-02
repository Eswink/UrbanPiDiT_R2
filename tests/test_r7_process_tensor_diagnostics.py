"""Analytic spherical torch checks, with numpy only as an offline CPU oracle."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from data.preprocess.process_diagnostics import (
    EARTH_RADIUS_M, REQUIRED_PROCESS_CHANNELS, compute_process_diagnostic_vector,
    horizontal_advection as numpy_advection, spherical_scalar_gradient as numpy_gradient,
)
from training.r7_process_tensor_diagnostics import (
    compute_process_tensor_diagnostics, horizontal_advection, spherical_divergence,
    spherical_scalar_gradient, spherical_vorticity,
)


@pytest.mark.parametrize("descending", [False, True])
def test_nonuniform_scalar_gradient_second_order_boundaries_and_latitude_metric(descending):
    lat = torch.tensor([-20., -7., 9., 24.], dtype=torch.float64)
    lon = torch.tensor([90., 96., 109., 124., 140.], dtype=torch.float64)
    if descending:
        lat, lon = lat.flip(0), lon.flip(0)
    phi, lam = lat.deg2rad()[:, None], lon.deg2rad()[None, :]
    field = EARTH_RADIUS_M * (phi.square() + 2 * lam.square())
    dx, dy = spherical_scalar_gradient(field, lat, lon)
    torch.testing.assert_close(dx, (4 * lam / phi.cos()).expand_as(field), rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(dy, (2 * phi).expand_as(field), rtol=3e-13, atol=3e-13)
    oracle_x, oracle_y = numpy_gradient(field.numpy(), lat.numpy(), lon.numpy())
    torch.testing.assert_close(dx, torch.tensor(oracle_x), rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(dy, torch.tensor(oracle_y), rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("height,width", [(2, 2), (2, 4), (4, 2)])
def test_small_grid_first_order_edges_are_exact_for_linear_fields(height, width):
    lat = torch.linspace(-15., 15., height, dtype=torch.float64)
    lon = torch.linspace(90., 120., width, dtype=torch.float64)
    field = EARTH_RADIUS_M * (lat.deg2rad()[:, None] + 3 * lon.deg2rad()[None, :])
    dx, dy = spherical_scalar_gradient(field, lat, lon)
    torch.testing.assert_close(dx, (3 / lat.deg2rad().cos()[:, None]).expand_as(field), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(dy, torch.ones_like(field), rtol=1e-13, atol=1e-13)


def test_spherical_divergence_vorticity_analytic_and_batched_metrics():
    lat = torch.tensor([[-20., -5., 15.], [30., 40., 60.]], dtype=torch.float64)
    lon = torch.tensor([[80., 94., 118., 135.], [-120., -105., -80., -65.]], dtype=torch.float64)
    phi, lam = lat.deg2rad()[..., None], lon.deg2rad()[:, None, :]
    k = 2.5e-5
    wind = k * EARTH_RADIUS_M * phi.cos() * lam
    zero = torch.zeros_like(wind)
    divergence = spherical_divergence(wind, zero, lat, lon)
    vorticity = spherical_vorticity(zero, wind, lat, lon)
    torch.testing.assert_close(divergence, torch.full_like(wind, k), rtol=2e-13, atol=1e-16)
    torch.testing.assert_close(vorticity, torch.full_like(wind, k), rtol=2e-13, atol=1e-16)


def test_horizontal_advection_sign_and_both_latitude_metrics():
    lat = torch.tensor([-30., -12., 8., 22.], dtype=torch.float64)
    lon = torch.tensor([80., 96., 115., 138.], dtype=torch.float64)
    phi, lam = lat.deg2rad()[:, None], lon.deg2rad()[None, :]
    scalar = EARTH_RADIUS_M * (3 * lam + 2 * phi)
    u, v = torch.full_like(scalar, 4.), torch.full_like(scalar, 5.)
    value = horizontal_advection(scalar, u, v, lat, lon)
    expected = (-12 / phi.cos() - 10).expand_as(scalar)
    torch.testing.assert_close(value, expected, rtol=5e-13, atol=5e-13)
    oracle = numpy_advection(scalar.numpy(), u.numpy(), v.numpy(), lat.numpy(), lon.numpy())
    torch.testing.assert_close(value, torch.tensor(oracle), rtol=5e-13, atol=5e-13)
    assert bool((value < 0).all())


def physical_state(dtype=torch.float64):
    lat = torch.tensor([-20., -4., 12., 25.], dtype=dtype)
    lon = torch.tensor([80., 87., 99., 115., 138.], dtype=dtype)
    yy, xx = torch.meshgrid(torch.arange(4, dtype=dtype), torch.arange(5, dtype=dtype), indexing="ij")
    state = torch.stack((100000. + 25 * xx + yy.square(), 280. + .4 * xx + .1 * yy,
                         .01 + 1e-4 * xx + 2e-5 * yy, 8. + .2 * yy, 2. + .1 * xx,
                         250. + .2 * xx, 20. + .2 * yy, 5. + .1 * xx))
    return state, lat, lon


def test_all_eight_proxies_match_numpy_and_channel_permutation_is_explicit():
    state, lat, lon = physical_state()
    oracle = compute_process_diagnostic_vector(state.numpy(), REQUIRED_PROCESS_CHANNELS, lat.numpy(), lon.numpy())
    value = compute_process_tensor_diagnostics(state, REQUIRED_PROCESS_CHANNELS, lat, lon)
    torch.testing.assert_close(value, torch.tensor(oracle, dtype=torch.float64), rtol=8e-8, atol=1e-12)
    indices = (2, 4, 7, 1, 3, 6, 0, 5)
    names = tuple(REQUIRED_PROCESS_CHANNELS[index] for index in indices)
    permuted = compute_process_tensor_diagnostics(state[list(indices)], names, lat, lon)
    assert torch.equal(value, permuted)
    assert value.shape == (8,) and value.dtype == torch.float64


def test_multiple_batch_and_k_axes_match_independent_coordinates():
    state, lat, lon = physical_state()
    states = torch.stack((state, state * 1.01))[:, None].expand(2, 3, 8, 4, 5).clone()
    latitudes = torch.stack((lat, lat + 20))
    longitudes = torch.stack((lon, lon - 150))
    result = compute_process_tensor_diagnostics(states, REQUIRED_PROCESS_CHANNELS, latitudes, longitudes)
    for b in range(2):
        for k in range(3):
            separate = compute_process_tensor_diagnostics(states[b, k], REQUIRED_PROCESS_CHANNELS,
                                                         latitudes[b], longitudes[b])
            assert torch.equal(result[b, k], separate)
    assert result.shape == (2, 3, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_autocast_disabled_sensitive_math_matches_explicit_fp32(dtype):
    state, lat, lon = physical_state(torch.float32)
    state = state.to(dtype).requires_grad_()
    reference = compute_process_tensor_diagnostics(state.float(), REQUIRED_PROCESS_CHANNELS, lat, lon)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under_amp = compute_process_tensor_diagnostics(state, REQUIRED_PROCESS_CHANNELS, lat, lon)
    assert under_amp.dtype == torch.float32
    assert torch.equal(under_amp, reference)
    under_amp.sum().backward()
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    # This compares the SAME quantized input, not a false claim that BF16
    # atmospheric quantization preserves every physical difference.


def test_zero_wind_constant_fields_have_finite_zero_norm_gradients():
    state = torch.zeros(8, 3, 4, requires_grad=True)
    lat, lon = torch.tensor([-10., 0., 10.]), torch.tensor([80., 90., 100., 110.])
    result = compute_process_tensor_diagnostics(state, REQUIRED_PROCESS_CHANNELS, lat, lon)
    assert torch.equal(result, torch.zeros(8))
    result.sum().backward()
    assert torch.isfinite(state.grad).all()
    assert torch.equal(state.grad[3:5], torch.zeros_like(state.grad[3:5]))
    assert torch.equal(state.grad[6:8], torch.zeros_like(state.grad[6:8]))


def test_large_constant_scalar_offset_cancels_before_fp32_difference():
    lat = torch.tensor([-20., -7., 9., 24.])
    lon = torch.tensor([90., 96., 109., 124., 140.])
    constant = torch.full((4, 5), 100000., requires_grad=True)
    dx, dy = spherical_scalar_gradient(constant, lat, lon)
    assert torch.equal(dx, torch.zeros_like(dx))
    assert torch.equal(dy, torch.zeros_like(dy))
    state = torch.zeros(8, 4, 5)
    state[0] = 100000.
    state[1] = 280.
    state[2] = .01
    state[5] = 250.
    state.requires_grad_()
    result = compute_process_tensor_diagnostics(state, REQUIRED_PROCESS_CHANNELS, lat, lon)
    assert torch.equal(result[:6], torch.zeros(6))
    assert torch.equal(result[7], torch.zeros(()))
    result.sum().backward()
    assert torch.isfinite(state.grad).all()


def test_fp64_tensor_diagnostics_pass_autograd_gradcheck():
    state, lat, lon = physical_state()
    # Keep values moderate for finite-difference resolution of all eight paths.
    state = (state / state.abs().amax((-2, -1), keepdim=True)).requires_grad_()
    assert torch.autograd.gradcheck(lambda value: compute_process_tensor_diagnostics(
        value, REQUIRED_PROCESS_CHANNELS, lat, lon), (state,), atol=2e-7, rtol=2e-4)


@pytest.mark.parametrize("kind", ["nan", "int", "short_axis", "duplicate_lat", "near_pole", "outside_lat",
                                 "wrapped_lon", "duplicate_lon", "batch_axis", "channel_width", "duplicate_names", "missing_channels"])
def test_tensor_diagnostics_reject_invalid_axis_state_and_channel_contracts(kind):
    state, lat, lon = physical_state(torch.float32)
    names = REQUIRED_PROCESS_CHANNELS
    if kind == "nan":
        state[0, 0, 0] = float("nan")
    elif kind == "int":
        state = state.long()
    elif kind == "short_axis":
        lon = lon[:1]
    elif kind == "duplicate_lat":
        lat[1] = lat[0]
    elif kind == "near_pole":
        lat[-1] = 90.
    elif kind == "outside_lat":
        lat[-1] = 91.
    elif kind == "wrapped_lon":
        lon = torch.tensor([350., 355., 0., 5., 10.])
    elif kind == "duplicate_lon":
        lon[-1] = lon[-2]
    elif kind == "batch_axis":
        lat = lat[None, :].expand(2, -1)
    elif kind == "channel_width":
        state = state[:7]
    elif kind == "duplicate_names":
        names = ("mslp",) + REQUIRED_PROCESS_CHANNELS[:-1]
    else:
        names = REQUIRED_PROCESS_CHANNELS[:-1] + ("unknown",)
    with pytest.raises((ValueError, KeyError)):
        compute_process_tensor_diagnostics(state, names, lat, lon)
