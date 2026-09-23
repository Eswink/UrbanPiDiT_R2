import csv
import math

import pytest
import torch
from training.r7_rollout_metrics import RolloutRMSEAccumulator


def test_accumulate_mse_then_sqrt_not_mean_sample_rmse():
    metric = RolloutRMSEAccumulator((6,), ("t2m",))
    pred = torch.tensor([1., 3.]).view(2, 1, 1, 1, 1)
    metric.update(pred[:1], torch.zeros_like(pred[:1]))
    metric.update(pred[1:], torch.zeros_like(pred[1:]))
    assert metric.initializations == 2
    assert float(metric.compute()) == pytest.approx(math.sqrt(5))
    assert float(metric.compute()) != pytest.approx(2.0)


def test_analytic_area_weights_and_physical_normalization():
    metric = RolloutRMSEAccumulator((6,), ("t2m", "u10"), training_std=[2., 3.], units=["K", "m s-1"])
    pred = torch.tensor([1., 3.]).view(1, 1, 1, 2, 1).expand(1, 1, 2, 2, 1)
    metric.update(pred, torch.zeros_like(pred), torch.tensor([0., 60.]))
    expected = torch.tensor([[2 * math.sqrt(11 / 3), 3 * math.sqrt(11 / 3)]], dtype=torch.float64)
    torch.testing.assert_close(metric.compute(), expected)


def test_batch_partition_independence_and_csv(tmp_path):
    torch.manual_seed(22)
    pred = torch.randn(3, 2, 2, 3, 4)
    truth = torch.randn_like(pred)
    lat = torch.tensor([50., 40., 30.])
    whole = RolloutRMSEAccumulator((6, 12), ("t2m", "u10"))
    parts = RolloutRMSEAccumulator((6, 12), ("t2m", "u10"))
    whole.update(pred, truth, lat)
    parts.update(pred[:2], truth[:2], lat)
    parts.update(pred[2:], truth[2:], lat)
    torch.testing.assert_close(parts.compute(), whole.compute())
    path = parts.write_csv(tmp_path / "metrics.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["unit"] == "normalized"
    assert rows[0]["n_initializations"] == "3"
    assert float(rows[0]["rmse"]) == pytest.approx(float(parts.compute()[0, 0]))
    with pytest.raises(FileExistsError):
        parts.write_csv(path)


def test_metrics_reject_nonfinite_and_bad_units():
    metric = RolloutRMSEAccumulator((6,), ("t2m",))
    with pytest.raises(RuntimeError, match="no initialization"):
        metric.compute()
    pred = torch.full((1, 1, 1, 2, 2), float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        metric.update(pred, torch.zeros_like(pred))
    assert metric.initializations == 0
    with pytest.raises(ValueError, match="explicit"):
        RolloutRMSEAccumulator((6,), ("t2m",), training_std=[2])
    with pytest.raises(ValueError, match="positive finite"):
        RolloutRMSEAccumulator((6,), ("t2m",), training_std=[0], units=["K"])
    with pytest.raises(ValueError, match="cannot be claimed"):
        RolloutRMSEAccumulator((6,), ("t2m",), units=["K"])
