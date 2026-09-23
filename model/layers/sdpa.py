from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class SDPAttention(nn.Module):
    def __init__(self, dim:int, heads:int=8, dropout:float=0.0, cross:bool=False):
        super().__init__(); assert dim%heads==0
        self.dim=dim; self.heads=heads; self.head_dim=dim//heads; self.dropout=float(dropout); self.cross=cross
        self.q=nn.Linear(dim,dim); self.k=nn.Linear(dim,dim); self.v=nn.Linear(dim,dim); self.out=nn.Linear(dim,dim)
    def forward(self, q_tokens:torch.Tensor, kv_tokens:torch.Tensor|None=None):
        kv=q_tokens if kv_tokens is None else kv_tokens
        B,N,D=q_tokens.shape; M=kv.shape[1]
        q=self.q(q_tokens).view(B,N,self.heads,self.head_dim).transpose(1,2)
        k=self.k(kv).view(B,M,self.heads,self.head_dim).transpose(1,2)
        v=self.v(kv).view(B,M,self.heads,self.head_dim).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1,2).reshape(B,N,D))

class FeedForward(nn.Module):
    def __init__(self,dim:int,ratio:float=4.0,dropout:float=0.0):
        super().__init__(); h=int(dim*ratio)
        self.net=nn.Sequential(nn.Linear(dim,h),nn.GELU(),nn.Dropout(dropout),nn.Linear(h,dim),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class CrossBlock(nn.Module):
    def __init__(self,dim:int,heads:int=8,ratio:float=4.0,dropout:float=0.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.nk=nn.LayerNorm(dim); self.attn=SDPAttention(dim,heads,dropout,True); self.n2=nn.LayerNorm(dim); self.ff=FeedForward(dim,ratio,dropout)
    def forward(self,x,kv):
        x=x+self.attn(self.n1(x),self.nk(kv)); x=x+self.ff(self.n2(x)); return x
