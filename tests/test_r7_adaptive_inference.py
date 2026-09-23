import io

import pytest
import torch

from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import AdaptiveProcessForecaster, ForecastGainController
from training.r7_halting import (calibrate_controller_step, controller_calibration_loss,
                                  next_step_gain_targets, per_sample_latitude_mse)


def make_model(**kwargs):
    torch.manual_seed(73)
    return ProcessForecastCoReasoner(in_channels=4, out_channels=4, dim=32,
                                    depth=1, heads=4, window_size=4, anchored_processes=2,
                                    free_processes=2, default_reasoning_steps=4, **kwargs)


def batch():
    g = torch.Generator().manual_seed(81)
    return {"coarse_history": torch.randn(2, 2, 4, 8, 12, generator=g),
            "atmos_target": torch.randn(2, 4, 8, 12, generator=g),
            "lead_time_hours": torch.tensor([6., 6.]),
            "latitude": torch.linspace(45., 43.25, 8)}


class ScriptedController(ForecastGainController):
    def __init__(self, policy):
        super().__init__(2, 4)
        self.policy, self.calls = policy, []

    def forward(self, features):
        self.calls.append(features.shape[0])
        gain = features.new_ones(features.shape[0])
        if self.policy == "stop":
            gain.fill_(-1.)
        elif self.policy == "mixed" and len(self.calls) == 1:
            gain[0] = -1.
        elif self.policy == "nonfinite":
            gain.fill_(float('nan'))
        return gain, torch.full_like(gain, 20.)


def adapted(policy):
    adapter = AdaptiveProcessForecaster(make_model())
    adapter.controller = ScriptedController(policy)
    return adapter.eval()


@pytest.mark.parametrize("feedback", [True, False])
def test_forced_full_depth_is_existing_fixed_model(feedback):
    model = make_model(use_forecast_feedback=feedback).eval()
    adapter = AdaptiveProcessForecaster(model).eval()
    b = batch()
    with torch.no_grad():
        expected = model(b, reasoning_steps=4)
    result = adapter(b, max_steps=4, force_full_depth=True)
    torch.testing.assert_close(result.forecast, expected.forecast)
    torch.testing.assert_close(result.process_predictions, expected.process_predictions[:, -1])
    assert result.reasoning_steps_per_sample.tolist() == [4, 4]
    assert not result.decision_masks.any()


def test_mixed_stop_executes_only_active_samples_and_preserves_stopped_result():
    adapter, b = adapted("mixed"), batch()
    reason_sizes, correction_sizes = [], []
    rh = adapter.forecaster.reasoning_cell.register_forward_pre_hook(
        lambda module, args: reason_sizes.append(args[0].shape[0]))
    ch = adapter.forecaster.correction_head.register_forward_pre_hook(
        lambda module, args: correction_sizes.append(args[0].shape[0]))
    result = adapter(b, max_steps=4, allow_untrained=True)
    rh.remove(); ch.remove()
    assert reason_sizes == correction_sizes == [2, 1, 1, 1]
    assert adapter.controller.calls == [2, 1, 1]  # no policy query after Kmax
    assert result.reasoning_steps_per_sample.tolist() == [1, 4]
    assert result.active_masks.tolist() == [[True, False, False, False], [True]*4]
    assert result.decision_masks.tolist() == [[True, False, False, False], [True, True, True, False]]
    with torch.no_grad():
        once = adapter.forecaster(b, reasoning_steps=1)
        full = adapter.forecaster(b, reasoning_steps=4)
    torch.testing.assert_close(result.forecast[0], once.forecast[0])
    torch.testing.assert_close(result.forecast[1], full.forecast[1])
    torch.testing.assert_close(result.process_predictions[0], once.process_predictions[0, -1])
    assert not result.forecast.requires_grad


class TargetPoison(dict):
    def __getitem__(self, key):
        if key in {"atmos_target", "process_targets", "atmos_baseline"}:
            raise AssertionError(f"inference touched forbidden field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def test_inference_is_target_free_and_ignores_external_baseline():
    adapter = adapted("continue")
    b = batch()
    clean = {"coarse_history": b["coarse_history"], "lead_time_hours": b["lead_time_hours"]}
    poisoned = TargetPoison({**b, "process_targets": None, "atmos_baseline": None})
    a = adapter(clean, max_steps=3, allow_untrained=True)
    c = adapter(poisoned, max_steps=3, allow_untrained=True)
    torch.testing.assert_close(a.forecast, c.forecast)
    assert torch.equal(a.reasoning_steps_per_sample, c.reasoning_steps_per_sample)


def test_minimum_depth_and_nonfinite_policy_fallback():
    adapter = adapted("stop")
    result = adapter(batch(), max_steps=4, min_steps=2, allow_untrained=True)
    assert result.reasoning_steps_per_sample.tolist() == [2, 2]
    assert not result.decision_masks[:, 0].any()
    invalid = adapted("nonfinite")(batch(), max_steps=3, allow_untrained=True)
    assert invalid.reasoning_steps_per_sample.tolist() == [3, 3]


def test_untrained_and_training_mode_guards():
    adapter = AdaptiveProcessForecaster(make_model())
    with pytest.raises(RuntimeError, match="eval"):
        adapter(batch())
    adapter.eval()
    with pytest.raises(RuntimeError, match="no training updates"):
        adapter(batch())
    assert adapter(batch(), max_steps=1).reasoning_steps_per_sample.tolist() == [1, 1]


@pytest.mark.parametrize("maximum,minimum", [(0, 1), (3, 4), (True, 1), (2.5, 1)])
def test_invalid_depths_fail(maximum, minimum):
    with pytest.raises(ValueError):
        adapted("stop")(batch(), max_steps=maximum, min_steps=minimum, allow_untrained=True)


def test_streamed_calibration_matches_fixed_drafts_and_only_controller_gets_gradients():
    model, b = make_model(), batch()
    adapter = AdaptiveProcessForecaster(model, gain_threshold=0.01)
    model.backbone.eval()  # preserve a deliberately mixed train/eval state
    modes = [module.training for module in model.modules()]
    losses = controller_calibration_loss(adapter, b, max_steps=4)
    assert [module.training for module in model.modules()] == modes
    model.eval()
    with torch.no_grad():
        fixed = model(b, reasoning_steps=4)
        errors = torch.stack([per_sample_latitude_mse(y, b["atmos_target"], b["latitude"])
                              for y in fixed.draft_forecasts[:, 1:].unbind(1)], dim=1)
        expected, labels = next_step_gain_targets(errors, threshold=float(adapter.gain_threshold))
    torch.testing.assert_close(losses.target_gains, expected)
    torch.testing.assert_close(losses.continue_targets, labels)
    assert losses.target_gains.shape == (2, 3)
    losses.total.backward()
    assert all(p.grad is None for p in model.parameters())
    assert adapter.controller.net[-1].weight.grad is not None
    assert adapter.controller.optimizer_updates.item() == 0  # loss alone is not an update


def test_controller_step_and_checkpoint_round_trip():
    adapter, b = AdaptiveProcessForecaster(make_model()), batch()
    original = {name: value.clone() for name, value in adapter.forecaster.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.controller.parameters(), lr=1e-3)
    losses = calibrate_controller_step(adapter, optimizer, b, max_steps=3)
    assert torch.isfinite(losses.total) and not losses.total.requires_grad
    assert adapter.controller.optimizer_updates.item() == 1
    for name, value in adapter.forecaster.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(adapter.state_dict(), stream); stream.seek(0)
    restored = AdaptiveProcessForecaster(make_model(), gain_threshold=10.)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    adapter.eval(); restored.eval()
    torch.testing.assert_close(adapter(b).forecast, restored(b).forecast)
    assert restored.controller.optimizer_updates.item() == 1
    assert restored.gain_threshold.item() == 0.


def test_unsafe_optimizer_is_rejected_before_backbone_update():
    adapter = AdaptiveProcessForecaster(make_model())
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="exactly"):
        calibrate_controller_step(adapter, optimizer, batch(), max_steps=3)
    with pytest.raises(ValueError, match="two"):
        controller_calibration_loss(adapter, batch(), max_steps=1)


def test_cpu_bfloat16_autocast_has_finite_calibration_and_inference():
    adapter = AdaptiveProcessForecaster(make_model())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = controller_calibration_loss(adapter, batch(), max_steps=3)
    assert losses.total.dtype == torch.float32 and torch.isfinite(losses.total)
    losses.total.backward()
    assert all(p.grad is None for p in adapter.forecaster.parameters())
    adapter.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = adapter(batch(), max_steps=3, allow_untrained=True)
    assert torch.isfinite(result.forecast).all()
