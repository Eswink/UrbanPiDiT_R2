import torch
from torch.utils.data import DataLoader

from data import SyntheticAtmosDataset
from model import (
    GenericRecursiveWeatherForecaster,
    ProcessForecastCoReasoner,
)
from training.r7_process_forecast_losses import (
    process_forecast_coreasoning_loss,
)


def _batch():
    ds=SyntheticAtmosDataset(
        length=4,
        hw=(8,12),
        history_steps=2,
        channels=4,
    )
    return next(iter(DataLoader(ds,batch_size=2)))


def _process_model(**kwargs):
    cfg=dict(
        in_channels=4,
        history_steps=2,
        out_channels=4,
        dim=32,
        patch_size=2,
        depth=1,
        heads=4,
        window_size=4,
        anchored_processes=2,
        free_processes=2,
        default_reasoning_steps=3,
    )
    cfg.update(kwargs)
    return ProcessForecastCoReasoner(**cfg)


def _generic_model(**kwargs):
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


def test_process_forecast_coreasoner_emits_reasoning_trace():
    batch=_batch()
    model=_process_model()
    out=model(batch,reasoning_steps=3)
    assert out.forecast.shape==(2,4,8,12)
    assert out.draft_forecasts.shape==(2,4,4,8,12)
    assert out.process_predictions.shape==(2,3,2)
    assert out.process_state.shape==(2,4,32)
    assert torch.isfinite(out.forecast).all()
    assert torch.isfinite(out.process_predictions).all()


def test_process_model_parameter_budget_matches_generic_recursion():
    process=_process_model()
    generic=_generic_model()
    p_process=sum(p.numel() for p in process.parameters())
    p_generic=sum(p.numel() for p in generic.parameters())

    relative=abs(p_process-p_generic)/p_generic
    assert relative < 0.02


def test_forecast_feedback_path_receives_gradient():
    batch=_batch()
    model=_process_model(use_forecast_feedback=True)
    out=model(batch,reasoning_steps=2)
    loss=process_forecast_coreasoning_loss(
        batch,out,process_weight=0.0
    ).total
    loss.backward()
    grad=model.draft_encoder.patch.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum()>0


def test_process_auxiliary_supervision_reaches_anchored_readout():
    batch=_batch()
    batch=dict(batch)
    batch["process_targets"]=torch.randn(2,2)
    model=_process_model()
    out=model(batch,reasoning_steps=2)
    losses=process_forecast_coreasoning_loss(
        batch,out,process_weight=1.0
    )
    losses.total.backward()
    grad=model.process_readout[-1].weight.grad
    assert losses.process.item()>0
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum()>0


def test_no_feedback_ablation_bypasses_draft_encoder():
    batch=_batch()
    model=_process_model(use_forecast_feedback=False)
    out=model(batch,reasoning_steps=2)
    loss=process_forecast_coreasoning_loss(
        batch,out,process_weight=0.0
    ).total
    loss.backward()
    assert model.draft_encoder.patch.weight.grad is None


def test_process_parameter_count_is_reasoning_depth_invariant():
    model=_process_model()
    count=sum(p.numel() for p in model.parameters())
    model(_batch(),reasoning_steps=1)
    after_one=sum(p.numel() for p in model.parameters())
    model(_batch(),reasoning_steps=4)
    after_four=sum(p.numel() for p in model.parameters())
    assert count==after_one==after_four
