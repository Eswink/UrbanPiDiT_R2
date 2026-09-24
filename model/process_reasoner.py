from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers.sdpa import SDPAttention, FeedForward, CrossBlock

class SharedProcessReasoningBlock(nn.Module):
    def __init__(self,dim:int=256,heads:int=8,mlp_ratio:float=3.0,dropout:float=0.0):
        super().__init__()
        self.n=nn.LayerNorm(dim)
        self.self_attn=SDPAttention(dim,heads,dropout)
        self.cross=CrossBlock(dim,heads,mlp_ratio,dropout)
        self.ffn=FeedForward(dim,mlp_ratio,dropout)
        self.n2=nn.LayerNorm(dim)
    def forward(self,p,context):
        p=p+self.self_attn(self.n(p)); p=self.cross(p,context); return p+self.ffn(self.n2(p))

class RecurrentProcessReasoner(nn.Module):
    """参数共享的 latent reasoning：reasoning 深度增加不会线性增加参数量。"""
    def __init__(self,dim:int=256,heads:int=8,mlp_ratio:float=3.0,dropout:float=0.0,use_checkpointing:bool=False):
        super().__init__(); self.block=SharedProcessReasoningBlock(dim,heads,mlp_ratio,dropout); self.use_checkpointing=bool(use_checkpointing)
    def forward(self,p,context,steps:int=2):
        states=[]
        for _ in range(int(steps)):
            if self.use_checkpointing and self.training and p.requires_grad:
                p=checkpoint(self.block,p,context,use_reentrant=False)
            else:
                p=self.block(p,context)
            states.append(p)
        return p,states
