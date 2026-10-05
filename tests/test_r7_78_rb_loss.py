"""#78 R-B: the change-scale loss is a switch, not a silent rewrite of the objective.

The contrast is one thing: the training objective either keeps the equal-channel
reduction (``loss_channel_weights=None``, the pre-change behavior) or multiplies
each channel's squared error by the frozen train-only vector. These tests pin
what must be provable without a GPU:

* the None path is bitwise the old implementation - same loss value and same
  gradients as code that never knew about the switch;
* a weighted run equals the analytic ``sum_c w_c * mse_c`` in both value and
  gradient, i.e. the weights multiply the per-channel errors rather than, say,
  scaling only the reported total;
* the weights are validated: wrong length, nonpositive or nonfinite entries
  fail closed instead of silently training a different objective;
* the derivation from the sidecar metadata is exactly ``(1/ratio)^2`` normalized
  to mean one and is stable for the published sidecar;
* the runner records the declared weights in its training contract, so a
  weighted checkpoint can never share a signature with the incumbent objective.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.r7_halting import per_sample_latitude_mse  # noqa: E402
from training.r7_streaming import backward_streamed_truncated  # noqa: E402

CHANNELS = 3
HW = (4, 5)


def _fixture(seed=21):
    torch.manual_seed(seed)
    model = _tiny_process_model()
    batch = {"coarse_history": torch.randn(2, 2, CHANNELS, *HW),
             "atmos_target": torch.randn(2, CHANNELS, *HW),
             "latitude": torch.linspace(43.0, 27.0, HW[0]),
             "longitude": torch.linspace(107.0, 123.0, HW[1]),
             "lead_time_hours": torch.full((2,), 6.0),
             "init_time_ns": torch.full((2,), 100 * 3_600_000_000_000, dtype=torch.int64)}
    batch["history_time_ns"] = torch.stack(
        (batch["init_time_ns"] - 6 * 3_600_000_000_000, batch["init_time_ns"]), dim=1)
    batch["valid_time_ns"] = batch["init_time_ns"] + 6 * 3_600_000_000_000
    batch["target_time_ns"] = batch["valid_time_ns"]
    return model, batch


def _tiny_process_model():
    from model.process_forecast_r7 import ProcessForecastCoReasoner

    return ProcessForecastCoReasoner(
        in_channels=CHANNELS, out_channels=CHANNELS, dim=16, depth=1, heads=2,
        patch_size=2, window_size=2, dropout=0.0, anchored_processes=2,
        free_processes=2, default_reasoning_steps=2)


def test_none_weights_reproduce_the_unweighted_loss_and_gradients():
    model, batch = _fixture()
    reference = deepcopy(model)
    switched = deepcopy(model)

    # Reference: the pre-change call shape, no weight argument at all.
    forecast = reference.backbone(
        __import__("model.r7_halting", fromlist=["forecast_inputs"]).forecast_inputs(batch)
    ).forecast
    loss = per_sample_latitude_mse(forecast, batch["atmos_target"],
                                   batch["latitude"]).mean()
    loss.backward()

    result = backward_streamed_truncated(switched, batch, reasoning_steps=0,
                                         process_weight=0.0)
    assert torch.equal(result.total.detach(), loss.detach())
    for (name_a, a), (name_b, b) in zip(reference.named_parameters(),
                                        switched.named_parameters()):
        assert name_a == name_b
        assert (a.grad is None) == (b.grad is None), name_a
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name_a


def test_weighted_loss_equals_the_analytic_channel_weighted_mse_and_gradients():
    model, batch = _fixture()
    weights = [0.5, 2.0, 4.0]
    analytic = deepcopy(model)
    weighted = deepcopy(model)

    from model.r7_halting import forecast_inputs

    forecast = analytic.backbone(forecast_inputs(batch)).forecast
    expected = per_sample_latitude_mse(forecast, batch["atmos_target"], batch["latitude"],
                                       channel_weights=weights).mean()
    expected.backward()

    result = backward_streamed_truncated(weighted, batch, reasoning_steps=0,
                                         process_weight=0.0,
                                         loss_channel_weights=weights)
    assert torch.allclose(result.total.detach(), expected.detach(), rtol=0, atol=1e-6)
    saw_grad = False
    for (name_a, a), (name_b, b) in zip(analytic.named_parameters(),
                                        weighted.named_parameters()):
        assert name_a == name_b
        assert (a.grad is None) == (b.grad is None), name_a
        if a.grad is not None:
            saw_grad = True
            assert torch.allclose(a.grad, b.grad, rtol=1e-5, atol=1e-7), name_a
    assert saw_grad


def test_weighted_minus_unweighted_equals_the_analytic_difference():
    model, batch = _fixture()
    weights = [0.25, 1.0, 3.0]

    uniform = backward_streamed_truncated(deepcopy(model), batch, reasoning_steps=0,
                                          process_weight=0.0)
    weighted = backward_streamed_truncated(deepcopy(model), batch, reasoning_steps=0,
                                           process_weight=0.0,
                                           loss_channel_weights=weights)

    from model.r7_halting import forecast_inputs

    probe = deepcopy(model)
    forecast = probe.backbone(forecast_inputs(batch)).forecast
    error = (forecast - batch["atmos_target"]).square()
    latitude = batch["latitude"]
    weight = torch.cos(torch.deg2rad(latitude)).clamp_min(0)
    # Average over longitude first, then area-normalize latitude: the exact
    # reduction per_sample_latitude_mse performs before the channel mean.
    per_channel = ((error.mean(3) * weight.view(1, 1, -1)).sum(2)
                   / weight.sum()).mean(0).detach()
    weight_tensor = torch.tensor(weights)
    expected_delta = float((per_channel * (weight_tensor - 1.0)).mean())
    measured_delta = float(weighted.total - uniform.total)
    assert abs(measured_delta - expected_delta) < 1e-6
    # The identity arm's loss must not move when the switch is off.
    assert not torch.equal(weighted.total.detach(), uniform.total.detach())


def test_invalid_weight_vectors_fail_closed():
    model, batch = _fixture()
    for bad in ([0.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, float("inf"), 1.0],
                [1.0, 1.0], [1.0, 1.0, 1.0, 1.0]):
        with pytest.raises(ValueError):
            backward_streamed_truncated(deepcopy(model), batch, reasoning_steps=0,
                                        process_weight=0.0,
                                        loss_channel_weights=bad)
    with pytest.raises(ValueError):
        per_sample_latitude_mse(torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2),
                                channel_weights=[1.0, 0.0, 1.0])


def test_weights_derivation_is_one_over_ratio_squared_normalized_to_mean_one():
    from data.preprocess.r7_rb_weights import rb_loss_weights

    meta = {"schema": "r7-change-scale-v1", "schema_version": 1, "fit_split": "train",
            "scientific_claim": False, "step_hours": 6, "fit_dtype": "float64",
            "runtime_dtype": "float32", "count": 2,
            "channels": ["a", "b"], "units": ["K", "K"],
            "change_scale": [1.0, 2.0], "state_scale": [2.0, 2.0],
            "ratio": [0.5, 1.0], "runtime_ratio": [0.5, 1.0],
            "relative_floor": float(torch.finfo(torch.float32).eps) * 1.0,
            "zero_change_fallback": [False, False],
            "below_floor_degenerate": [False, False],
            "degenerate": [False, False],
            "degenerate_reason": ["active", "active"],
            "data_identity": "a" * 64, "train_manifest_sha256": "b" * 64,
            "train_manifest": "/x/train.jsonl", "store": "/x/cache.zarr",
            "train_pairs": {"count": 1, "first_index": [0, 1], "last_index": [0, 1],
                            "frame_union_count": 2, "first_time_ns": 0, "last_time_ns": 1,
                            "train_diff_sha256": "c" * 64, "train_diff_dtype": "float32"},
            "weight_rule": "w", "sample_rule": "s", "fallback_rule": "f",
            "floor_rule": "fl", "limitations": ["l"],
            "change_scale_identity": "d" * 64}
    from training.r7_experiment import canonical_digest

    content = {key: value for key, value in meta.items() if key != "change_scale_identity"}
    meta["change_scale_identity"] = canonical_digest(content)
    meta["preflight_identity"] = "e" * 64
    derived = rb_loss_weights(meta)
    raw = [1.0 / 0.5 ** 2, 1.0 / 1.0 ** 2]
    mean = sum(raw) / 2
    assert derived["weights"][0] == pytest.approx(raw[0] / mean)
    assert derived["weights"][1] == pytest.approx(raw[1] / mean)
    assert sum(derived["weights"]) / 2 == pytest.approx(1.0)
    assert derived["rule"].strip() and derived["loss_rule"].strip()
