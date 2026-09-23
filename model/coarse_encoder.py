from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers.window_attention import WindowAttentionBlock

class CoarseEncoder(nn.Module):
    def __init__(self,in_channels:int,history_steps:int,dim:int=192,patch_size:int=2,depth:int=4,heads:int=6,window_size:int=8,dropout:float=0.0,use_checkpointing:bool=False):
        super().__init__(); self.in_channels=in_channels; self.history_steps=history_steps; self.dim=dim; self.patch_size=patch_size; self.use_checkpointing=bool(use_checkpointing)
        self.patch=nn.Conv2d(in_channels*history_steps,dim,kernel_size=patch_size,stride=patch_size)
        self.blocks=nn.ModuleList([WindowAttentionBlock(dim,heads,window_size,4.0,dropout,shift=bool(i%2)) for i in range(depth)])
        self.norm=nn.LayerNorm(dim)
    def forward(self,x:torch.Tensor):
        B,T,C,H,W=x.shape
        if T!=self.history_steps or C!=self.in_channels: raise ValueError(f'coarse 输入期望 T,C={self.history_steps},{self.in_channels}，实际 {T},{C}')
        z=self.patch(x.reshape(B,T*C,H,W)); Ht,Wt=z.shape[-2:]; tokens=z.flatten(2).transpose(1,2)
        for b in self.blocks:
            if self.use_checkpointing and self.training and tokens.requires_grad:
                tokens=checkpoint(lambda q, block=b: block(q,(Ht,Wt)), tokens, use_reentrant=False)
            else:
                tokens=b(tokens,(Ht,Wt))
        return self.norm(tokens),(Ht,Wt)
