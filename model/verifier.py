from __future__ import annotations
import torch
from torch import nn


class ForecastVerifier(nn.Module):
    """Lightweight verifier with process readout and forecast confidence."""

    def __init__(self,process_dim:int,anchored:int=12,hidden:int=128):
        super().__init__()
        self.anchored=anchored
        self.process_head=nn.Sequential(
            nn.LayerNorm(process_dim),
            nn.Linear(process_dim,1),
        )
        self.conf=nn.Sequential(
            nn.Linear(process_dim+3,hidden),
            nn.GELU(),
            nn.Linear(hidden,1),
        )

    def predict_process(self,process:torch.Tensor)->torch.Tensor:
        return self.process_head(process[:,:self.anchored]).squeeze(-1)

    def forward(self,process,baseline,forecast,residual):
        pp=self.predict_process(process)
        resid_mag=residual.abs().mean(dim=(1,2,3))
        baseline_mag=baseline.abs().mean(dim=(1,2,3))
        delta=(forecast-baseline).abs()
        edge=torch.cat(
            [
                delta[:,:,0,:].flatten(1),
                delta[:,:,-1,:].flatten(1),
                delta[:,:,:,0].flatten(1),
                delta[:,:,:,-1].flatten(1),
            ],
            1,
        )
        boundary_gap=edge.mean(1)
        feat=torch.cat(
            [process.mean(1),resid_mag[:,None],baseline_mag[:,None],boundary_gap[:,None]],
            -1,
        )
        score=torch.sigmoid(self.conf(feat).squeeze(-1))
        return score,pp,{
            'residual_magnitude':resid_mag,
            'cross_scale_delta':boundary_gap,
        }
