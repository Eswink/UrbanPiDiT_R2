from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers.window_attention import WindowAttentionBlock
from .cross_scale_adapter import CrossScaleAdapter
from .sparse_process_graph import SparseGridProcessGraph

class UrbanExpert(nn.Module):
    def __init__(self,dynamic_channels:int,history_steps:int,static_channels:int,out_channels:int,
        coarse_dim:int,process_dim:int,dim:int=256,patch_size:int=4,depth:int=8,heads:int=8,
        window_size:int=8,dropout:float=0.0,graph_every:int=4,cross_every:int=2,use_checkpointing:bool=False):
        super().__init__()
        self.dynamic_channels=dynamic_channels; self.history_steps=history_steps; self.static_channels=static_channels
        self.out_channels=out_channels; self.patch_size=patch_size
        self.graph_every=max(1,graph_every); self.cross_every=max(1,cross_every)
        self.use_checkpointing=bool(use_checkpointing)
        cin=dynamic_channels*history_steps+static_channels
        self.stem=nn.Sequential(nn.Conv2d(cin,dim//2,3,padding=1),nn.GELU(),nn.Conv2d(dim//2,dim,kernel_size=patch_size,stride=patch_size))
        self.blocks=nn.ModuleList([WindowAttentionBlock(dim,heads,window_size,4.0,dropout,shift=bool(i%2)) for i in range(depth)])
        self.cross=CrossScaleAdapter(dim,coarse_dim,process_dim,heads)
        self.graph=SparseGridProcessGraph(dim)
        self.norm=nn.LayerNorm(dim)
        self.decode=nn.Sequential(nn.ConvTranspose2d(dim,dim//2,kernel_size=patch_size,stride=patch_size),nn.GELU(),nn.Conv2d(dim//2,out_channels,3,padding=1))
        nn.init.zeros_(self.decode[-1].weight); nn.init.zeros_(self.decode[-1].bias)
    def forward(self,history,static,coarse_tokens,process_tokens):
        B,T,C,H,W=history.shape
        if T!=self.history_steps or C!=self.dynamic_channels: raise ValueError('UrbanExpert history T/C 不匹配')
        x=torch.cat([history.reshape(B,T*C,H,W),static],1); z=self.stem(x); ht,wt=z.shape[-2:]; tok=z.flatten(2).transpose(1,2)
        for i,b in enumerate(self.blocks,1):
            if self.use_checkpointing and self.training and tok.requires_grad:
                tok=checkpoint(lambda q, block=b: block(q,(ht,wt)), tok, use_reentrant=False)
            else:
                tok=b(tok,(ht,wt))
            if i%self.cross_every==0:
                if self.use_checkpointing and self.training and tok.requires_grad:
                    tok=checkpoint(lambda q,c,p: self.cross(q,c,p),tok,coarse_tokens,process_tokens,use_reentrant=False)
                else:
                    tok=self.cross(tok,coarse_tokens,process_tokens)
            if i%self.graph_every==0: tok=self.graph(tok,(ht,wt))
        tok=self.norm(tok); fmap=tok.transpose(1,2).reshape(B,-1,ht,wt); residual=self.decode(fmap)
        if residual.shape[-2:]!=(H,W): residual=torch.nn.functional.interpolate(residual,size=(H,W),mode='bilinear',align_corners=False)
        return residual,tok,(ht,wt)
