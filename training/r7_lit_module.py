from __future__ import annotations
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from model import NativeAtmosForecaster
from .r7_losses import latitude_weighted_mse


class R7ForecastLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg:Dict[str,Any],
        optim_cfg:Dict[str,Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net=NativeAtmosForecaster(**model_cfg)
        self.optim_cfg=dict(optim_cfg)

    def _step(self,batch,stage):
        out=self.net(batch)
        loss=latitude_weighted_mse(
            out.forecast,
            batch['atmos_target'],
            batch.get('latitude'),
        )
        batch_size=int(batch['atmos_target'].shape[0])
        self.log(
            f'{stage}/loss',loss,
            prog_bar=True,on_epoch=True,on_step=False,batch_size=batch_size,
        )
        tendency_rms=out.tendency.square().mean().sqrt()
        self.log(
            f'{stage}/tendency_rms',tendency_rms,
            on_epoch=True,on_step=False,batch_size=batch_size,
        )
        return loss

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
            weight_decay=float(self.optim_cfg.get('weight_decay',1e-4)),
        )
