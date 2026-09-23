from __future__ import annotations

import torch

from baselines.forecast_models import build_forecast_baseline, available_baselines
from baselines.external_models.urban_loader_adapter import UrbanBatchAdapter


def _dummy_batch(batch_size=2, c=7, k=4, s=5, h=8, w=8, leads=(1, 2, 3, 4)):
    x_ctx = torch.randn(batch_size, c * k + s, h, w)
    static_raw = x_ctx[:, c * k :]
    return {
        "x_ctx": x_ctx,
        "static_raw": static_raw,
        "static_cont": static_raw[:, 1:],
        "static_cat": static_raw[:, :1].round().long().clamp(min=0, max=19),
        "y": torch.randn(batch_size, len(leads), c, h, w),
        "x0": torch.randn(batch_size, c, h, w),
        "lead_times": torch.tensor(leads, dtype=torch.long),
        "hour_of_day": torch.zeros(batch_size, dtype=torch.long),
        "norm": {
            "mean": torch.zeros(c, 1, 1),
            "std": torch.ones(c, 1, 1),
        },
    }


def _build(name: str):
    params = {
        "hidden_channels": 16,
        "include_coords": True,
        "include_hour": True,
        "forecast_protocol": "official_rollout",
        "one_step_lead": 1,
        "time_step_hours": 6.0,
    }
    if name == "fourcastnet":
        params.update({"depth": 1, "modes": 2})
    elif name == "graphcast":
        params.update({"message_passing_steps": 1, "include_time_forcing": True})
    elif name == "gencast":
        params.update({
            "denoising_steps": 1,
            "num_noise_levels": 2,
            "sigma_min": 0.002,
            "sigma_max": 0.1,
            "num_samples": 1,
            "stochastic_eval": False,
        })
    return build_forecast_baseline(
        name,
        dynamic_vars=["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"],
        static_vars=["landcover", "building_surface", "buildings", "building_volume", "population"],
        k=4,
        lead_times=[1, 2, 3, 4],
        static_policy="same_static",
        params=params,
    )


def test_external_baselines_registered():
    names = set(available_baselines())
    assert {"fourcastnet", "graphcast", "gencast"}.issubset(names)


def test_urban_loader_adapter_shapes():
    model = _build("fourcastnet")
    batch = _dummy_batch()
    adapted = UrbanBatchAdapter(model, include_coords=True, include_hour=True)(batch)
    assert adapted.dynamic.shape == (2, 7, 4, 8, 8)
    assert adapted.static.shape == (2, 5, 8, 8)
    assert adapted.features.shape[0] == 2
    assert adapted.target.shape == (2, 4, 7, 8, 8)


@torch.no_grad()
def test_external_baseline_forward_shapes_and_finite():
    batch = _dummy_batch()
    for name in ["fourcastnet", "graphcast", "gencast"]:
        model = _build(name)
        model.eval()
        pred = model(batch, lead_times=[1, 2, 3, 4])
        assert pred.shape == (2, 4, 7, 8, 8), name
        assert torch.isfinite(pred).all(), name


def test_dynamic_only_policy_still_runs():
    model = build_forecast_baseline(
        "graphcast",
        dynamic_vars=["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"],
        static_vars=["landcover", "building_surface", "buildings", "building_volume", "population"],
        k=4,
        lead_times=[1, 2],
        static_policy="dynamic_only",
        params={
            "hidden_channels": 16,
            "message_passing_steps": 1,
            "include_coords": True,
            "include_hour": False,
            "forecast_protocol": "official_rollout",
            "one_step_lead": 1,
        },
    )
    pred = model(_dummy_batch(leads=(1, 2)), lead_times=[1, 2])
    assert pred.shape == (2, 2, 7, 8, 8)


def test_gencast_training_loss_is_finite_and_backpropagates():
    model = _build("gencast")
    batch = _dummy_batch()
    loss = model.training_loss(batch, lead_times=[1])
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@torch.no_grad()
def test_gencast_deterministic_sampling_reproducible_in_eval_mode():
    batch = _dummy_batch()
    model = _build("gencast")
    model.eval()
    first = model(batch, lead_times=[1, 2])
    second = model(batch, lead_times=[1, 2])
    assert first.shape == (2, 2, 7, 8, 8)
    assert torch.allclose(first, second)
