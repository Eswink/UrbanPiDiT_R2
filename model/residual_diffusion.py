from __future__ import annotations
import torch
from torch import nn

class ResidualDiffusionRefiner(nn.Module):
    """可选的轻量 residual diffusion scaffold。

    V6 主线默认关闭；只对高不确定性/ZOOM tile 的 residual 进行概率细化。
    """
    def __init__(self,channels:int,hidden:int=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(channels*2+1,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,channels,3,padding=1))
    def _alpha(self,t): return torch.cos(t.clamp(0,1)*torch.pi/2)**2
    def training_loss(self,target_residual,condition):
        B=target_residual.shape[0]; t=torch.rand(B,device=target_residual.device,dtype=target_residual.dtype); a=self._alpha(t).view(B,1,1,1); noise=torch.randn_like(target_residual); noisy=a.sqrt()*target_residual+(1-a).sqrt()*noise
        tt=t.view(B,1,1,1).expand(B,1,*target_residual.shape[-2:]); pred=self.net(torch.cat([noisy,condition,tt],1)); return torch.nn.functional.mse_loss(pred,noise)
    @torch.no_grad()
    def sample(self,condition,steps:int=4):
        x=torch.randn_like(condition); B=condition.shape[0]
        for i in reversed(range(steps)):
            t=torch.full((B,), (i+1)/steps,device=x.device,dtype=x.dtype); tt=t.view(B,1,1,1).expand(B,1,*x.shape[-2:]); eps=self.net(torch.cat([x,condition,tt],1)); x=x-eps/steps
        return x
