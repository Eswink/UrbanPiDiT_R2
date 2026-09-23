from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .sdpa import FeedForward


def _pad_hw(x, window):
    B,H,W,D=x.shape; ph=(window-H%window)%window; pw=(window-W%window)%window
    if ph or pw: x=F.pad(x,(0,0,0,pw,0,ph))
    return x,(H,W)

def _partition(x,w):
    B,H,W,D=x.shape
    return x.view(B,H//w,w,W//w,w,D).permute(0,1,3,2,4,5).reshape(-1,w*w,D)

def _reverse(wins,w,B,H,W,D):
    return wins.view(B,H//w,W//w,w,w,D).permute(0,1,3,2,4,5).reshape(B,H,W,D)

class WindowAttentionBlock(nn.Module):
    """仅在局部窗口构造注意力；复杂度 O(HW*window²)，不构造全图 N² attention。"""
    def __init__(self,dim:int,heads:int=8,window_size:int=8,mlp_ratio:float=4.0,dropout:float=0.0,shift:bool=False):
        super().__init__(); assert dim%heads==0
        self.dim=dim; self.heads=heads; self.ws=window_size; self.shift=shift; self.hd=dim//heads; self.dropout=float(dropout)
        self.n1=nn.LayerNorm(dim); self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
        self.n2=nn.LayerNorm(dim); self.ff=FeedForward(dim,mlp_ratio,dropout)
    def forward(self,tokens:torch.Tensor, hw:tuple[int,int]):
        B,N,D=tokens.shape; H,W=hw
        if N!=H*W: raise ValueError(f'token 数 {N} 与 hw={hw} 不一致')
        x=self.n1(tokens).view(B,H,W,D)
        if self.shift: x=torch.roll(x,shifts=(-self.ws//2,-self.ws//2),dims=(1,2))
        x,(oh,ow)=_pad_hw(x,self.ws); Hp,Wp=x.shape[1:3]; wins=_partition(x,self.ws)
        qkv=self.qkv(wins).view(wins.shape[0],wins.shape[1],3,self.heads,self.hd).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0)
        y=self.proj(y.transpose(1,2).reshape(wins.shape[0],wins.shape[1],D))
        x=_reverse(y,self.ws,B,Hp,Wp,D)[:,:oh,:ow]
        if self.shift: x=torch.roll(x,shifts=(self.ws//2,self.ws//2),dims=(1,2))
        out=tokens+x.reshape(B,N,D)
        return out+self.ff(self.n2(out))
