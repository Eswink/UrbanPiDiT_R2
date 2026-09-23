from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model.r7_rollout import autoregressive_rollout
from model.weather_forecaster_r7 import NativeAtmosForecaster
from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.r7_halting import AdaptiveProcessForecaster


class IncrementForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []
        self.buffer = None

    def forward(self, batch):
        assert set(batch) == {"coarse_history", "lead_time_hours"}
        self.seen.append({key: value.clone() for key, value in batch.items()})
        nxt = batch["coarse_history"][:, -1] + 1
        if self.buffer is None:
            self.buffer = nxt.clone()
        else:
            self.buffer.copy_(nxt)
        return SimpleNamespace(forecast=self.buffer, reasoning_steps=2)


def test_rollout_is_free_running_snapshots_are_immutable_and_cadence_is_fixed():
    model = IncrementForecast().eval()
    history = torch.zeros(2, 2, 3, 4, 5)
    history[:, -1] = 10
    original = history.clone()
    batch = {"coarse_history": history, "atmos_target": object(),
             "process_targets": object(), "atmos_baseline": object(),
             "lead_time_hours": torch.full((2,), 999.)}
    result = autoregressive_rollout(model, batch)
    assert result.lead_hours == (6, 12, 24, 48, 72)
    assert result.model_calls == 12
    for i, hour in enumerate(result.lead_hours):
        assert torch.all(result.forecasts[:, i] == 10 + hour // 6)
        assert torch.all(result.cumulative_reasoning_steps[:, i] == (hour // 6) * 2)
    for call in model.seen:
        assert torch.all(call["lead_time_hours"] == 6)
    assert torch.all(model.seen[1]["coarse_history"][:, -2] == 10)
    assert torch.all(model.seen[1]["coarse_history"][:, -1] == 11)
    torch.testing.assert_close(history, original, rtol=0, atol=0)
    assert not result.forecasts.requires_grad


@pytest.mark.parametrize("horizons", [(), (12, 6), (6, 6), (7,), (0,), (True,), (6.0,)])
def test_bad_horizons_fail(horizons):
    model = IncrementForecast().eval()
    with pytest.raises(ValueError):
        autoregressive_rollout(model, {"coarse_history": torch.zeros(1, 2, 1, 4, 4)}, lead_hours=horizons)


def test_bad_cadence_training_mode_and_kwargs_fail():
    model = IncrementForecast()
    batch = {"coarse_history": torch.zeros(1, 2, 1, 4, 4)}
    with pytest.raises(RuntimeError, match="eval"):
        autoregressive_rollout(model, batch)
    model.eval()
    with pytest.raises(ValueError, match="cadence"):
        autoregressive_rollout(model, batch, history_interval_hours=12)
    with pytest.raises(ValueError, match="unsupported"):
        autoregressive_rollout(model, batch, inference_kwargs={"atmos_target": torch.zeros(1)})


def test_partial_dynamic_outputs_rejected():
    class Partial(nn.Module):
        def forward(self, batch):
            return SimpleNamespace(forecast=batch["coarse_history"][:, -1, :1])
    with pytest.raises(ValueError, match="every input dynamic channel"):
        autoregressive_rollout(Partial().eval(), {"coarse_history": torch.ones(1, 2, 2, 4, 4)})


@pytest.mark.parametrize("kind", ["native", "generic", "process", "adaptive"])
def test_all_r7_variants_run_to_72h_without_targets(kind):
    torch.manual_seed(22)
    cfg = dict(in_channels=4, dim=32, depth=1, heads=4, window_size=4)
    kwargs = {}
    if kind == "native":
        model = NativeAtmosForecaster(**cfg)
        cost = 0
    elif kind == "generic":
        model = GenericRecursiveWeatherForecaster(**cfg, latent_tokens=4)
        kwargs = {"reasoning_steps": 2}
        cost = 24
    else:
        model = ProcessForecastCoReasoner(**cfg, anchored_processes=2, free_processes=2)
        kwargs = {"reasoning_steps": 2}
        cost = 24
        if kind == "adaptive":
            model = AdaptiveProcessForecaster(model)
            kwargs = {"max_steps": 2, "force_full_depth": True}
    result = autoregressive_rollout(model.eval(),
        {"coarse_history": torch.randn(2, 2, 4, 8, 12)}, inference_kwargs=kwargs)
    assert result.forecasts.shape == (2, 5, 4, 8, 12)
    assert torch.isfinite(result.forecasts).all()
    assert torch.all(result.cumulative_reasoning_steps[:, -1] == cost)
    assert all(p.grad is None for p in model.parameters())
