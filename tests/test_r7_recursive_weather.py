import torch
from torch.utils.data import DataLoader
from data import SyntheticAtmosDataset
from model import GenericRecursiveWeatherForecaster
from training.r7_recursive_losses import (
    deep_supervised_forecast_mse,
)


def _batch():
    ds=SyntheticAtmosDataset(
        length=4,hw=(8,12),history_steps=2,channels=4
    )
    return next(iter(DataLoader(ds,batch_size=2)))


def _model(**kwargs):
    cfg=dict(
        in_channels=4,
        history_steps=2,
        out_channels=4,
        dim=32,
        patch_size=2,
        depth=1,
        heads=4,
        window_size=4,
        latent_tokens=4,
        default_reasoning_steps=3,
    )
    cfg.update(kwargs)
    return GenericRecursiveWeatherForecaster(**cfg)


def test_recursive_weather_emits_all_drafts():
    batch=_batch()
    model=_model()
    out=model(batch,reasoning_steps=3)
    assert out.forecast.shape==(2,4,8,12)
    assert out.draft_forecasts.shape==(2,4,4,8,12)
    assert out.reasoning_steps==3
    assert torch.isfinite(out.draft_forecasts).all()


def test_recursive_parameter_count_independent_of_reasoning_depth():
    model=_model()
    count=sum(p.numel() for p in model.parameters())
    model(_batch(),reasoning_steps=1)
    after_one=sum(p.numel() for p in model.parameters())
    model(_batch(),reasoning_steps=4)
    after_four=sum(p.numel() for p in model.parameters())
    assert count==after_one==after_four


def test_detached_recursive_deep_supervision_backward():
    batch=_batch()
    model=_model(detach_between_steps=True)
    out=model(batch,reasoning_steps=3)
    loss=deep_supervised_forecast_mse(
        out.draft_forecasts,
        batch['atmos_target'],
        batch['latitude'],
    )
    loss.backward()
    assert loss.isfinite()
    assert model.cell.self_attn.q.weight.grad is not None
    assert model.correction_head.decode[-1].weight.grad is not None
    assert model.backbone.encoder.patch.weight.grad is not None
