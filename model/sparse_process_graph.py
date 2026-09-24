from __future__ import annotations
from functools import lru_cache
import torch
from torch import nn

@lru_cache(maxsize=64)
def _cpu_grid_edges(h:int,w:int,diagonal:bool=True):
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    if diagonal: dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    src=[]; dst=[]; delta=[]
    for y in range(h):
        for x in range(w):
            i=y*w+x
            for dy,dx in dirs:
                yy,xx=y+dy,x+dx
                if 0<=yy<h and 0<=xx<w:
                    src.append(i); dst.append(yy*w+xx); delta.append((dy,dx))
    return torch.tensor([src,dst],dtype=torch.long), torch.tensor(delta,dtype=torch.float32)

class SparseGridProcessGraph(nn.Module):
    """O(Nk) 局部消息传播，仅构造局部边列表，不生成全连接邻接矩阵。"""
    def __init__(self,dim:int,hidden:int|None=None,diagonal:bool=True):
        super().__init__(); hidden=hidden or dim; self.diagonal=diagonal
        self.msg=nn.Sequential(nn.Linear(dim+2,hidden),nn.GELU(),nn.Linear(hidden,dim))
        self.gate=nn.Sequential(nn.Linear(dim*2+2,hidden),nn.GELU(),nn.Linear(hidden,1),nn.Sigmoid())
        self.norm=nn.LayerNorm(dim)
    def forward(self,tokens:torch.Tensor,hw:tuple[int,int]):
        B,N,D=tokens.shape; h,w=hw
        if N!=h*w: raise ValueError('SparseGridProcessGraph: N 与 hw 不一致')
        edge,delta=_cpu_grid_edges(h,w,self.diagonal); edge=edge.to(tokens.device); delta=delta.to(tokens.device,tokens.dtype)
        s,d=edge[0],edge[1]; xs=tokens[:,s]; xd=tokens[:,d]; ed=delta.unsqueeze(0).expand(B,-1,-1)
        msg=self.msg(torch.cat([xs,ed],-1))*self.gate(torch.cat([xs,xd,ed],-1))
        agg=torch.zeros_like(tokens); agg.index_add_(1,d,msg)
        deg=torch.zeros(N,device=tokens.device,dtype=tokens.dtype); deg.index_add_(0,d,torch.ones_like(d,dtype=tokens.dtype)); agg=agg/deg.clamp_min(1).view(1,N,1)
        return self.norm(tokens+agg)
