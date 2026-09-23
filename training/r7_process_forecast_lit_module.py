from __future__ import annotations
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from model import ProcessForecastCoReasoner
from .r7_losses import latitude_weighted_mse
from .r7_process_forecast_losses import process_forecast_coreasoning_loss


class R7ProcessForecastLightningModule(pl.LightningModule):
    """Train the R7.3 Process-Forecast Co-Reasoning model."""

    def __init__(
        self,
        model_cfg:Dict[str,Any],
        optim_cfg:Dict[str,Any],
        loss_cfg:Dict[str,Any]|None=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net=ProcessForecastCoReasoner(**model_cfg)
        self.optim_cfg=dict(optim_cfg)
        self.loss_cfg=dict(loss_cfg or {})

    def _step(self,batch,stage):
        out=self.net(batch)
        losses=process_forecast_coreasoning_loss(
            batch,
            out,
            forecast_weight=float(
                self.loss_cfg.get("forecast_weight",1.0)
            ),
            process_weight=float(
                self.loss_cfg.get("process_weight",0.1)
            ),
            final_weight=float(
                self.loss_cfg.get("final_weight",2.0)
            ),
        )
        final=latitude_weighted_mse(
            out.forecast,
            batch["atmos_target"],
            batch.get("latitude"),
        )
        batch_size=int(batch["atmos_target"].shape[0])
        self.log(
            f"{stage}/loss",
            losses.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/forecast_deep",
            losses.forecast,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/final_mse",
            final,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/process",
            losses.process,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return losses.total

    def training_step(self,batch,batch_idx):
        return self._step(batch,"train")

    def validation_step(self,batch,batch_idx):
        return self._step(batch,"val")

    def test_step(self,batch,batch_idx):
        return self._step(batch,"test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.optim_cfg.get("lr",2e-4)),
            weight_decay=float(
                self.optim_cfg.get("weight_decay",1e-4)
            ),
        )
