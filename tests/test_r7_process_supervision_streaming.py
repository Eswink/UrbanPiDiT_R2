"""M3 streamed/full loss parity on synthetic, explicitly timed fixtures."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from data.preprocess.r7_process_scale_sidecar import fit_process_scale
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import forecast_inputs
from training.r7_process_forecast_losses import process_forecast_coreasoning_loss
from training.r7_process_supervision import ProcessDiagnosticContext
from training.r7_process_tensor_diagnostics import compute_process_tensor_diagnostics
from training.r7_streaming import backward_streamed_truncated, train_streamed_update

CHANNELS = ("t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
            "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250")
HOUR_NS = 3_600_000_000_000


def fixture():
    torch.manual_seed(52)
    mean = torch.tensor([280., 0., 0., 100000., 100000., 280., .005, 5., 3.,
                         500000., 260., .003, 10., 5., 1000000., 20., 10.])
    std = torch.tensor([5., 3., 3., 500., 1000., 5., .002, 4., 4.,
                       2000., 5., .001, 5., 5., 3000., 5., 5.])
    latitude, longitude = torch.linspace(30., 34., 5), torch.linspace(105., 110., 6)
    fit = torch.randn(12, 17, 5, 6)
    raw = compute_process_tensor_diagnostics(fit * std[:, None, None] + mean[:, None, None],
                                             CHANNELS, latitude, longitude)
    metadata = fit_process_scale(raw.numpy())
    context = ProcessDiagnosticContext(metadata, mean, std, CHANNELS)
    init = torch.full((2,), 100 * HOUR_NS, dtype=torch.int64)
    batch = {"coarse_history": torch.randn(2, 2, 17, 5, 6),
             "atmos_target": torch.randn(2, 17, 5, 6), "latitude": latitude,
             "longitude": longitude, "lead_time_hours": torch.full((2,), 6.),
             "init_time_ns": init, "valid_time_ns": init + 6 * HOUR_NS,
             "target_time_ns": init + 6 * HOUR_NS,
             "history_time_ns": torch.stack((init - 6 * HOUR_NS, init), dim=1),
             "split": ["train", "train"]}
    model = ProcessForecastCoReasoner(in_channels=17, out_channels=17, dim=16, depth=1,
        heads=2, patch_size=2, window_size=2, dropout=0., anchored_processes=8, free_processes=2)
    return model.train(), batch, context


@pytest.mark.parametrize("weights", [dict(input_diagnostic_weight=.1),
    dict(future_diagnostic_weight=.05, draft_diagnostic_weight=.05)])
@pytest.mark.parametrize("steps", [1, 4])
def test_streamed_new_tasks_match_truncated_full_loss_and_all_gradients(weights, steps):
    model, batch, context = fixture()
    streamed = deepcopy(model)
    output = model(forecast_inputs(batch), reasoning_steps=steps, detach_between_steps=True)
    loss = process_forecast_coreasoning_loss(batch, output, process_weight=0.,
        process_supervision_context=context, **weights)
    loss.total.backward()
    log = backward_streamed_truncated(streamed, batch, reasoning_steps=steps, process_weight=0.,
        process_supervision_context=context, **weights)
    torch.testing.assert_close(log.total, loss.total.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(log.final_forecast, output.forecast, rtol=0, atol=0)
    for (name, a), (other, b) in zip(model.named_parameters(), streamed.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=5e-6, msg=name)
    assert streamed.process_readout[1].weight.grad.abs().sum() > 0
    assert streamed.process_queries.grad.abs().sum() > 0
    assert not log.total.requires_grad


def test_streamed_bf16_diagnostics_are_fp32_and_gradients_finite():
    model, batch, context = fixture()
    log = backward_streamed_truncated(model, batch, reasoning_steps=4, process_weight=0.,
        process_supervision_context=context, future_diagnostic_weight=.05,
        draft_diagnostic_weight=.05, amp_dtype=torch.bfloat16)
    assert log.total.dtype == torch.float32
    assert torch.isfinite(log.total)
    assert model.process_readout[1].weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()


def test_bad_future_after_partial_accumulation_never_updates_weights():
    model, batch, context = fixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    before = deepcopy(model.state_dict())
    bad = dict(batch, atmos_target=torch.full_like(batch["atmos_target"], float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        train_streamed_update(model, optimizer, [batch, bad], reasoning_steps=2, process_weight=0.,
            process_supervision_context=context, future_diagnostic_weight=.05, draft_diagnostic_weight=.05)
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_legacy_and_new_tasks_cannot_both_run_streamed():
    model, batch, context = fixture()
    with pytest.raises(ValueError, match="legacy process_weight=0"):
        backward_streamed_truncated(model, batch, process_weight=.1,
            process_supervision_context=context, input_diagnostic_weight=.1)


def test_input_only_task_never_reads_future_diagnostics_or_target_timestamp():
    model, batch, context = fixture()
    prepared = context.fields_from_batch(batch, training=True, need_future=False)
    poison = dict(batch, future_diagnostic_targets=np.full((2, 8), np.nan))
    other = context.fields_from_batch(poison, training=True, need_future=False)
    assert prepared.future_diagnostic_targets is None
    assert other.future_diagnostic_targets is None
    assert torch.equal(prepared.input_diagnostics, other.input_diagnostics)
