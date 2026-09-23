from __future__ import annotations
import torch
from torch import nn

class UrbanStaticEncoder(nn.Module):
    """廉价形态编码器：保留局地结构，同时产生 patch-level static evidence。"""
    def __init__(self,in_channels:int,dim:int=128,patch_size:int=4):
        super().__init__(); self.in_channels=in_channels; self.patch_size=patch_size
        mid=max(dim//2,32)
        self.net=nn.Sequential(nn.Conv2d(in_channels,mid,3,padding=1),nn.GELU(),nn.Conv2d(mid,dim,kernel_size=patch_size,stride=patch_size),nn.GELU())
    def forward(self,x):
        z=self.net(x); return z.flatten(2).transpose(1,2),(z.shape[-2],z.shape[-1])
