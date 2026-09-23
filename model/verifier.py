from __future__ import annotations
import torch
from torch import nn

class ForecastVerifier(nn.Module):
    """轻量 verifier：学习置信度 + 可审计的跨尺度/增量诊断。"""
    def __init__(self,process_dim:int,anchored:int=12,hidden:int=128):
        super().__init__(); self.anchored=anchored
        self.process_head=nn.Sequential(nn.LayerNorm(process_dim),nn.Linear(process_dim,1))
        self.conf=nn.Sequential(nn.Linear(process_dim+3,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,process,baseline,forecast,residual):
        pp=self.process_head(process[:,:self.anchored]).squeeze(-1)
        resid_mag=residual.abs().mean(dim=(1,2,3)); baseline_mag=baseline.abs().mean(dim=(1,2,3))
        delta=(forecast-baseline).abs()
        edge=torch.cat([delta[:,:,0,:].flatten(1),delta[:,:,-1,:].flatten(1),delta[:,:,:,0].flatten(1),delta[:,:,:,-1].flatten(1)],1)
        boundary_gap=edge.mean(1)
        feat=torch.cat([process.mean(1),resid_mag[:,None],baseline_mag[:,None],boundary_gap[:,None]],-1)
        score=torch.sigmoid(self.conf(feat).squeeze(-1))
        return score,pp,{'residual_magnitude':resid_mag,'cross_scale_delta':boundary_gap}
