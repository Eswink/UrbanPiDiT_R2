import torch
from torch.utils.data import DataLoader
from data import SyntheticAtmosDataset, validate_forecast_sample
from model import NativeAtmosForecaster
from training.r7_losses import latitude_weighted_mse


def _batch():
    ds=SyntheticAtmosDataset(
        length=4,hw=(9,15),history_steps=2,channels=6
    )
    return next(iter(DataLoader(ds,batch_size=2)))


def _model():
    return NativeAtmosForecaster(
        in_channels=6,
        history_steps=2,
        out_channels=6,
        dim=48,
        patch_size=2,
        depth=2,
        heads=4,
        window_size=4,
    )


def test_r7_forecast_contract_and_shape():
    batch=_batch()
    validate_forecast_sample(batch,batched=True)
    model=_model()
    out=model(batch)
    assert out.forecast.shape==(2,6,9,15)
    assert out.tendency.shape==out.forecast.shape
    assert out.context_tokens.shape[0]==2
    assert torch.isfinite(out.forecast).all()


def test_r7_forecast_backward():
    batch=_batch()
    model=_model()
    out=model(batch)
    loss=latitude_weighted_mse(
        out.forecast,batch['atmos_target'],batch['latitude']
    )
    loss.backward()
    assert loss.isfinite()
    assert model.head.decode[-1].weight.grad is not None
    assert model.encoder.patch.weight.grad is not None
    assert model.encoder.patch.weight.grad.abs().sum()>0


def test_lead_time_changes_context():
    batch=_batch()
    model=_model().eval()
    with torch.no_grad():
        a=model(batch).context_tokens
        changed=dict(batch)
        changed['lead_time_hours']=torch.full_like(
            batch['lead_time_hours'],24.0
        )
        b=model(changed).context_tokens
    assert not torch.allclose(a,b)
