from __future__ import annotations
import torch
from torch import nn

class ReasoningRouter(nn.Module):
    def __init__(self,process_dim:int,coarse_dim:int,hidden:int=128):
        super().__init__(); self.proc=nn.LayerNorm(process_dim); self.coarse=nn.LayerNorm(coarse_dim)
        self.head=nn.Sequential(nn.Linear(process_dim+coarse_dim,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.roi=nn.Sequential(nn.Linear(coarse_dim,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,process,coarse,coarse_hw):
        pooled=torch.cat([self.proc(process).mean(1),self.coarse(coarse).mean(1)],-1); logit=self.head(pooled).squeeze(-1); prob=torch.sigmoid(logit)
        roi=self.roi(self.coarse(coarse)).squeeze(-1).view(coarse.shape[0],*coarse_hw)
        return logit,prob,roi
