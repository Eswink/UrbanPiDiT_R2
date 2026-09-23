from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers.window_attention import WindowAttentionBlock
from .layers.patch_grid import pad_patch_grid


class CoarseEncoder(nn.Module):
    def __init__(self,in_channels:int,history_steps:int,dim:int=192,patch_size:int=2,
                 depth:int=4,heads:int=6,window_size:int=8,dropout:float=0.,
                 use_checkpointing:bool=False,periodic_width:bool=False,pad_to_patch:bool=False):
        super().__init__()
        self.in_channels=in_channels
        self.history_steps=history_steps
        self.dim=dim
        self.patch_size=patch_size
        self.use_checkpointing=bool(use_checkpointing)
        self.periodic_width=bool(periodic_width)
        self.pad_to_patch=bool(pad_to_patch)
        self.window_size=window_size
        self.patch=nn.Conv2d(in_channels*history_steps,dim,kernel_size=patch_size,stride=patch_size)
        self.blocks=nn.ModuleList([WindowAttentionBlock(dim,heads,window_size,4.,dropout,
            shift=bool(i%2),periodic_width=self.periodic_width) for i in range(depth)])
        self.norm=nn.LayerNorm(dim)

    def forward(self,x:torch.Tensor):
        B,T,C,H,W=x.shape
        if T!=self.history_steps or C!=self.in_channels:
            raise ValueError(f'coarse 输入期望 T,C={self.history_steps},{self.in_channels}，实际 {T},{C}')
        x=x.reshape(B,T*C,H,W)
        if self.pad_to_patch:
            # A partial periodic window needs a different seam algorithm; reject
            # rather than imply that a masked padding gap is a valid longitude seam.
            if self.periodic_width and (W%(self.patch_size*self.window_size)):
                raise ValueError('periodic R7 width must be divisible by patch_size*window_size')
            x=pad_patch_grid(x,self.patch_size,periodic_width=self.periodic_width)
        z=self.patch(x)
        Ht,Wt=z.shape[-2:]
        tokens=z.flatten(2).transpose(1,2)
        for block in self.blocks:
            if self.use_checkpointing and self.training and tokens.requires_grad:
                tokens=checkpoint(lambda q,b=block:b(q,(Ht,Wt)),tokens,use_reentrant=False)
            else:
                tokens=block(tokens,(Ht,Wt))
        return self.norm(tokens),(Ht,Wt)
