from __future__ import annotations

import torch

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec


class IncrementRolloutForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="increment_rollout",
        description="synthetic rollout protocol test model",
        uses_static=False,
        trainable=False,
        family="test",
    )

    def __init__(self, *, forecast_protocol: str = "official_rollout", time_step_hours: float = 6.0) -> None:
        super().__init__(
            dynamic_vars=["x"],
            static_vars=[],
            k=3,
            lead_times=[1, 2, 3, 4],
            static_policy="dynamic_only",
            forecast_protocol=forecast_protocol,
            one_step_lead=1,
            time_step_hours=time_step_hours,
        )
        self.seen_last: list[float] = []
        self.seen_hours: list[int | None] = []

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        self.seen_last.append(float(view.last.flatten()[0].detach().cpu()))
        if view.hour_of_day is None:
            self.seen_hours.append(None)
        else:
            self.seen_hours.append(int(view.hour_of_day.flatten()[0].detach().cpu()))
        return view.last + 1.0


def _batch(hour: int = 0) -> dict:
    return {
        "x_ctx": torch.tensor([[[[0.0]], [[10.0]], [[20.0]]]]),
        "lead_times": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "hour_of_day": torch.tensor([hour], dtype=torch.long),
        "norm": {
            "mean": torch.zeros(1, 1, 1),
            "std": torch.ones(1, 1, 1),
        },
    }


def test_dense_rollout_collects_each_step_and_rolls_history() -> None:
    model = IncrementRolloutForecast()
    pred = model(_batch(), lead_times=[1, 2, 3, 4])

    assert tuple(pred.shape) == (1, 4, 1, 1, 1)
    assert pred.flatten().tolist() == [21.0, 22.0, 23.0, 24.0]
    assert model.seen_last == [20.0, 21.0, 22.0, 23.0]
    assert model.seen_hours == [0, 6, 12, 18]


def test_sparse_rollout_keeps_requested_order_after_internal_rollout() -> None:
    model = IncrementRolloutForecast()
    pred = model(_batch(hour=18), lead_times=[2, 4])

    assert tuple(pred.shape) == (1, 2, 1, 1, 1)
    assert pred.flatten().tolist() == [22.0, 24.0]
    assert model.seen_last == [20.0, 21.0, 22.0, 23.0]
    assert model.seen_hours == [18, 0, 6, 12]


def test_direct_mode_keeps_legacy_shared_initial_view() -> None:
    model = IncrementRolloutForecast(forecast_protocol="direct")
    pred = model(_batch(), lead_times=[1, 2, 3, 4])

    assert tuple(pred.shape) == (1, 4, 1, 1, 1)
    assert pred.flatten().tolist() == [21.0, 21.0, 21.0, 21.0]
    assert model.seen_last == [20.0, 20.0, 20.0, 20.0]