from torch.utils.data import DataLoader

from data import SyntheticAtmosDataset
from training.r7_process_forecast_lit_module import (
    R7ProcessForecastLightningModule,
)


def test_process_forecast_lightning_backward():
    batch=next(iter(DataLoader(
        SyntheticAtmosDataset(
            length=4,
            hw=(8,12),
            history_steps=2,
            channels=4,
        ),
        batch_size=2,
    )))
    model_cfg=dict(
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
        default_reasoning_steps=2,
    )
    lit=R7ProcessForecastLightningModule(
        model_cfg,
        {"lr":1e-3},
        {"process_weight":0.0},
    )
    loss=lit.training_step(batch,0)
    loss.backward()
    assert loss.isfinite()
    assert lit.net.reasoning_cell.self_attn.q.weight.grad is not None
