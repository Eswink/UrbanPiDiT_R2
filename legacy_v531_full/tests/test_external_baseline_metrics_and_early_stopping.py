from __future__ import annotations

import math

import torch

from baselines.metrics import BatchTensors, compute_batch_metrics, compute_batch_metrics_per_channel
from baselines.train_external_baselines import _early_stopping_config, _is_improvement


def test_external_baseline_metrics_include_bias_and_crps_global_and_per_var():
    pred = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 2.0], [2.0, 2.0]]]],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)
    bt = BatchTensors(
        pred_norm=pred,
        target_norm=target,
        mean=torch.zeros(2, 1, 1),
        std=torch.ones(2, 1, 1),
        clim_denorm=torch.zeros_like(pred),
    )

    global_metrics = compute_batch_metrics(bt)
    per_var = compute_batch_metrics_per_channel(bt)

    assert {"rmse", "mae", "bias", "crps", "acc"}.issubset(global_metrics)
    assert {"rmse_ch", "mae_ch", "bias_ch", "crps_ch", "acc_ch"}.issubset(per_var)
    # Deterministic single-member CRPS is MAE.
    assert torch.allclose(global_metrics["crps"], global_metrics["mae"])
    assert torch.allclose(per_var["crps_ch"], per_var["mae_ch"])
    assert per_var["bias_ch"].shape == (2,)


def test_ensemble_crps_is_supported_and_not_forced_to_mae():
    target = torch.zeros(1, 1, 2, 2)
    pred = torch.ones_like(target)
    ensemble = torch.stack([torch.zeros_like(target), torch.ones_like(target) * 2.0], dim=1)
    bt = BatchTensors(
        pred_norm=pred,
        pred_ensemble_norm=ensemble,
        target_norm=target,
        mean=torch.zeros(1, 1, 1),
        std=torch.ones(1, 1, 1),
    )
    metrics = compute_batch_metrics(bt)
    per_var = compute_batch_metrics_per_channel(bt)
    assert "crps" in metrics and torch.isfinite(metrics["crps"])
    assert "crps_ch" in per_var and per_var["crps_ch"].shape == (1,)
    assert not torch.allclose(metrics["crps"], metrics["mae"])


def test_early_stopping_config_and_improvement_logic():
    cfg = _early_stopping_config(
        {
            "early_stopping": {
                "enabled": True,
                "monitor": "val_mse",
                "mode": "min",
                "patience": 3,
                "min_delta": 0.01,
                "restore_best": True,
            }
        }
    )
    assert cfg["enabled"] is True
    assert cfg["patience"] == 3
    assert _is_improvement(0.98, 1.0, mode="min", min_delta=0.01)
    assert not _is_improvement(0.995, 1.0, mode="min", min_delta=0.01)
    assert _is_improvement(1.02, 1.0, mode="max", min_delta=0.01)
    assert not _is_improvement(float("nan"), 1.0, mode="min", min_delta=0.0)
