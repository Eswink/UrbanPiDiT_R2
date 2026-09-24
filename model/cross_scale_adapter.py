from __future__ import annotations
import torch
from torch import nn
from .layers.sdpa import CrossBlock

class CrossScaleAdapter(nn.Module):
    def __init__(self,urban_dim:int,coarse_dim:int,process_dim:int,heads:int=8):
        super().__init__()
        self.c=nn.Linear(coarse_dim,urban_dim)
        self.p=nn.Linear(process_dim,urban_dim)
        self.coarse_cross=CrossBlock(urban_dim,heads,2.0,0.0)
        self.process_cross=CrossBlock(urban_dim,heads,2.0,0.0)
    def forward(self,urban,coarse,process):
        urban=self.coarse_cross(urban,self.c(coarse)); urban=self.process_cross(urban,self.p(process)); return urban
