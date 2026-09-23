from __future__ import annotations
from typing import Dict,Any
import torch
import pytorch_lightning as pl
from model import UrbanPiDiTR2
from .losses import R2Loss

class R2LightningModule(pl.LightningModule):
    def __init__(self,model_cfg:Dict[str,Any],optim_cfg:Dict[str,Any],loss_cfg:Dict[str,Any]|None=None):
        super().__init__(); self.save_hyperparameters(); self.net=UrbanPiDiTR2(**model_cfg); self.optim_cfg=dict(optim_cfg); self.loss_fn=R2Loss(**dict(loss_cfg or {}))
    def _step(self,batch,stage):
        out=self.net(batch,force_zoom=True,adaptive_reasoning=False,hard_route=False); dl=self.net.diffusion_loss(batch,out) if self.net.enable_diffusion else None; losses=self.loss_fn(batch,out,dl)
        batch_size=int(batch['urban_target'].shape[0])
        self.log(f'{stage}/loss',losses.total,prog_bar=True,on_epoch=True,on_step=False,batch_size=batch_size); self.log(f'{stage}/forecast',losses.forecast,on_epoch=True,on_step=False,batch_size=batch_size); self.log(f'{stage}/verifier_loss',losses.verifier,on_epoch=True,on_step=False,batch_size=batch_size); self.log(f'{stage}/verifier_score',out.diagnostics.verifier_score.mean(),on_epoch=True,on_step=False,batch_size=batch_size); self.log(f'{stage}/zoom_prob',out.diagnostics.zoom_probability.mean(),on_epoch=True,on_step=False,batch_size=batch_size)
        return losses.total
    def training_step(self,batch,batch_idx): return self._step(batch,'train')
    def validation_step(self,batch,batch_idx): return self._step(batch,'val')
    def test_step(self,batch,batch_idx): return self._step(batch,'test')
    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=float(self.optim_cfg.get('lr',2e-4)),weight_decay=float(self.optim_cfg.get('weight_decay',1e-4)))
        return opt
