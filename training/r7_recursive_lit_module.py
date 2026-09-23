from __future__ import annotations
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from model import GenericRecursiveWeatherForecaster
from .r7_losses import latitude_weighted_mse
from .r7_recursive_losses import deep_supervised_forecast_mse


class R7RecursiveLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg:Dict[str,Any],
        optim_cfg:Dict[str,Any],
        loss_cfg:Dict[str,Any]|None=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net=GenericRecursiveWeatherForecaster(**model_cfg)
        self.optim_cfg=dict(optim_cfg)
        self.loss_cfg=dict(loss_cfg or {})

    def _step(self,batch,stage):
        out=self.net(batch)
        deep=deep_supervised_forecast_mse(
            out.draft_forecasts,
            batch['atmos_target'],
            batch.get('latitude'),
            final_weight=float(
                self.loss_cfg.get('final_weight',2.0)
            ),
        )
        final=latitude_weighted_mse(
            out.forecast,
            batch['atmos_target'],
            batch.get('latitude'),
        )
        batch_size=int(batch['atmos_target'].shape[0])
        self.log(
            f'{stage}/loss',deep,
            prog_bar=True,on_epoch=True,on_step=False,batch_size=batch_size,
        )
        self.log(
            f'{stage}/final_mse',final,
            on_epoch=True,on_step=False,batch_size=batch_size,
        )
        return deep

    def training_step(self,batch,batch_idx):
        return self._step(batch,'train')

    def validation_step(self,batch,batch_idx):
        return self._step(batch,'val')

    def test_step(self,batch,batch_idx):
        return self._step(batch,'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.optim_cfg.get('lr',2e-4)),
            weight_decay=float(
                self.optim_cfg.get('weight_decay',1e-4)
            ),
        )
