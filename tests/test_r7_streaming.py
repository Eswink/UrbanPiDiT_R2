from __future__ import annotations

import copy
import gc

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
from training.r7_process_forecast_losses import process_forecast_coreasoning_loss
from training.r7_recursive_losses import deep_supervised_forecast_mse
from training.r7_streaming import backward_streamed_truncated, train_streamed_update
from training.r7_memory import SavedTensorMeter


def model_and_batch(kind="process", batch_size=2, **extra):
    torch.manual_seed(21)
    cfg = dict(in_channels=4, dim=32, depth=1, heads=4, patch_size=2,
               window_size=4, dropout=0.0)
    cfg.update(extra)
    if kind == "process":
        model = ProcessForecastCoReasoner(**cfg, anchored_processes=2, free_processes=2)
    else:
        model = GenericRecursiveWeatherForecaster(**cfg, latent_tokens=4)
    batch = {
        "coarse_history": torch.randn(batch_size, 2, 4, 8, 12),
        "atmos_target": torch.randn(batch_size, 4, 8, 12),
        "process_targets": torch.randn(batch_size, 2),
        "latitude": torch.linspace(45, 30, 8).expand(batch_size, -1).clone(),
        "lead_time_hours": torch.full((batch_size,), 6.0),
    }
    return model.train(), batch


def retained_loss(model, batch, k):
    out = model(batch, reasoning_steps=k, detach_between_steps=True)
    if isinstance(model, ProcessForecastCoReasoner):
        loss = process_forecast_coreasoning_loss(batch, out).total
    else:
        loss = deep_supervised_forecast_mse(
            out.draft_forecasts, batch["atmos_target"], batch["latitude"])
    return out, loss


def assert_gradients_equal(reference, streamed):
    for (name, a), (other, b) in zip(reference.named_parameters(), streamed.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize("kind", ["generic", "process"])
@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_streamed_matches_truncated_reference_all_gradients(kind, k):
    reference, batch = model_and_batch(kind)
    streamed = copy.deepcopy(reference)
    out, loss = retained_loss(reference, batch, k)
    loss.backward()
    log = backward_streamed_truncated(streamed, batch, reasoning_steps=k)
    torch.testing.assert_close(out.forecast, log.final_forecast)
    torch.testing.assert_close(loss.detach(), log.total, rtol=1e-5, atol=1e-6)
    assert log.draft_errors.shape == (k + 1,)
    assert all(not value.requires_grad for value in vars(log).values())
    assert_gradients_equal(reference, streamed)
    assert streamed.backbone.encoder.patch.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("extra", [dict(use_forecast_feedback=False), dict(activation_checkpointing=True)])
def test_streamed_feedback_ablation_and_checkpointing(extra):
    reference, batch = model_and_batch(**extra)
    streamed = copy.deepcopy(reference)
    _, loss = retained_loss(reference, batch, 3)
    loss.backward()
    backward_streamed_truncated(streamed, batch, reasoning_steps=3)
    assert_gradients_equal(reference, streamed)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=1e-3)
        self.updates = 0

    def step(self, *args, **kwargs):
        self.updates += 1
        return super().step(*args, **kwargs)


def test_uneven_microbatches_one_update_and_repeatability():
    full, batch = model_and_batch(batch_size=3)
    micro = copy.deepcopy(full)
    opt_full, opt_micro = CountingSGD(full.parameters()), CountingSGD(micro.parameters())
    groups = [{key: value[sl] for key, value in batch.items()} for sl in (slice(0, 2), slice(2, 3))]
    for _ in range(2):
        train_streamed_update(full, opt_full, [batch], reasoning_steps=3)
        train_streamed_update(micro, opt_micro, groups, reasoning_steps=3)
    assert opt_full.updates == opt_micro.updates == 2
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
        assert b.grad is None


def test_failure_after_partial_accumulation_clears_gradients_without_step():
    model, batch = model_and_batch()
    optimizer = CountingSGD(model.parameters())
    before = copy.deepcopy(model.state_dict())
    bad = dict(batch, atmos_target=torch.full_like(batch["atmos_target"], float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        train_streamed_update(model, optimizer, [batch, bad], reasoning_steps=2)
    assert optimizer.updates == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_optimizer_ownership_guard():
    model, batch = model_and_batch()
    optimizer = CountingSGD(model.correction_head.parameters())
    with pytest.raises(ValueError, match="exactly"):
        train_streamed_update(model, optimizer, [batch])
    assert optimizer.updates == 0


@pytest.mark.parametrize("kind", ["generic", "process"])
def test_bf16_fp32_losses_and_finite_encoder_gradients(kind):
    model, batch = model_and_batch(kind)
    log = backward_streamed_truncated(model, batch, reasoning_steps=3, amp_dtype=torch.bfloat16)
    assert log.total.dtype == torch.float32 and torch.isfinite(log.total)
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
    assert model.backbone.encoder.patch.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("k", [-1, True, 1.5])
def test_bad_depth_rejected(k):
    model, batch = model_and_batch()
    with pytest.raises(ValueError, match="nonnegative integer"):
        backward_streamed_truncated(model, batch, reasoning_steps=k)


def test_fp16_and_wrong_process_dimensions_rejected():
    model, batch = model_and_batch()
    with pytest.raises(ValueError, match="FP16"):
        backward_streamed_truncated(model, batch, amp_dtype=torch.float16)
    with pytest.raises(ValueError, match="exactly match"):
        backward_streamed_truncated(model, dict(batch, process_targets=torch.randn(2, 3)))


def measure_saved_tensors(kind, k, streamed):
    model, batch = model_and_batch(kind)
    model.zero_grad(set_to_none=True)
    with SavedTensorMeter() as meter:
        if streamed:
            log = backward_streamed_truncated(model, batch, reasoning_steps=k)
            del log
        else:
            out, loss = retained_loss(model, batch, k)
            loss.backward()
            del out, loss
    gc.collect()
    assert meter.live_bytes == 0
    return meter.peak_live_bytes


@pytest.mark.parametrize("kind", ["generic", "process"])
def test_live_saved_tensor_peak_does_not_grow_with_k(kind, capsys):
    retained = [measure_saved_tensors(kind, k, False) for k in (1, 2, 4, 8)]
    streamed = [measure_saved_tensors(kind, k, True) for k in (1, 2, 4, 8)]
    with capsys.disabled():
        print(f"\nLogical saved-tensor bytes (not CUDA VRAM), {kind}, K=[1,2,4,8]: "
              f"retained={retained}, streamed={streamed}")
    assert retained[-1] > retained[1] * 2
    assert max(streamed) <= min(streamed) * 1.15
    assert streamed[-1] < retained[-1] * 0.6
