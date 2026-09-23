import pytest
import torch

from model.r7_halting import ForecastGainController
from training.r7_halting import next_step_gain_targets, per_sample_latitude_mse


def test_signed_gain_targets_exclude_unobserved_final_transition():
    errors = torch.tensor([[4., 1., 2.], [1., 1., 0.5]], requires_grad=True)
    gains, labels = next_step_gain_targets(errors, threshold=0.5)
    torch.testing.assert_close(gains, torch.tensor([[3., -1.], [0., 0.5]]))
    torch.testing.assert_close(labels, torch.tensor([[1., 0.], [0., 0.]]))
    assert gains.shape == (2, 2)  # E3 has no observed E4; no made-up STOP label.
    assert not gains.requires_grad and not labels.requires_grad


@pytest.mark.parametrize("errors", [torch.zeros(2, 1), torch.tensor([[1., float('nan')]]),
                                    torch.tensor([[1., -1.]]), torch.zeros(0, 2)])
def test_invalid_gain_targets_fail(errors):
    with pytest.raises(ValueError):
        next_step_gain_targets(errors)


def test_latitude_mse_is_per_sample_and_fp32_for_half_inputs():
    prediction = torch.tensor([1., 1000.], dtype=torch.float16).view(2, 1, 1, 1).expand(2, 1, 2, 3)
    actual = per_sample_latitude_mse(prediction, torch.zeros_like(prediction), torch.tensor([0., 60.]))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([1., 1_000_000.]))


def test_latitude_area_weighting_and_batch_coordinates():
    y = torch.tensor([1., 3.]).view(1, 1, 2, 1).expand(2, 2, 2, 3)
    lat = torch.tensor([[0., 60.], [60., 0.]])
    actual = per_sample_latitude_mse(y, torch.zeros_like(y), lat)
    torch.testing.assert_close(actual, torch.tensor([(1. + 9.*0.5)/1.5, (0.5 + 9.)/1.5]))


@pytest.mark.parametrize("latitude", [torch.zeros(3), torch.tensor([0., 91.]),
                                      torch.tensor([float('nan'), 30.]), torch.tensor([90., 90.])])
def test_invalid_latitude_fails(latitude):
    with pytest.raises(ValueError):
        per_sample_latitude_mse(torch.ones(1, 2, 2, 3), torch.zeros(1, 2, 2, 3), latitude)


def test_loss_forbids_accidental_broadcasting():
    with pytest.raises(ValueError, match="equal"):
        per_sample_latitude_mse(torch.ones(2, 1, 2, 3), torch.zeros(1, 1, 2, 3))


def test_controller_features_detach_and_keep_large_half_summaries_finite():
    controller = ForecastGainController(2, 3)
    p = torch.randn(2, 2, requires_grad=True)
    y = torch.full((2, 3, 2, 3), 1000., dtype=torch.float16, requires_grad=True)
    features = controller.features(p, y, y, step=2)
    assert features.dtype == torch.float32 and torch.isfinite(features).all()
    assert not features.requires_grad
    gain, logit = controller(features)
    (gain.square().mean() + logit.square().mean()).backward()
    assert p.grad is None and y.grad is None
    assert controller.net[0].weight.grad is not None
