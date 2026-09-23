from __future__ import annotations
from typing import Optional
import torch
from torch import nn
from .layers.patch_grid import crop_native_grid


class LeadTimeEmbedding(nn.Module):
    def __init__(self,dim:int,hidden:int|None=None):
        super().__init__()
        hidden=hidden or max(32,dim//2)
        self.net=nn.Sequential(nn.Linear(1,hidden),nn.SiLU(),nn.Linear(hidden,dim))

    def forward(self,lead_time_hours:Optional[torch.Tensor],*,batch:int,device:torch.device,
                dtype:torch.dtype,default_hours:float=6.)->torch.Tensor:
        if lead_time_hours is None:
            hours=torch.full((batch,1),default_hours,device=device,dtype=dtype)
        else:
            hours=torch.as_tensor(lead_time_hours,device=device,dtype=dtype)
            if hours.ndim==0:
                hours=hours.expand(batch).reshape(batch,1)
            else:
                hours=hours.reshape(batch,-1)[:,:1]
        return self.net(hours/24.)


class CoarseForecastHead(nn.Module):
    """Decode full patch cells and crop; never resize a native-grid tendency."""
    def __init__(self,dim:int,out_channels:int,patch_size:int=2,hidden:int|None=None):
        super().__init__()
        hidden=hidden or max(64,dim//2)
        self.out_channels=int(out_channels)
        self.patch_size=int(patch_size)
        self.norm=nn.LayerNorm(dim)
        self.decode=nn.Sequential(nn.ConvTranspose2d(dim,hidden,kernel_size=patch_size,stride=patch_size),
            nn.GELU(),nn.Conv2d(hidden,self.out_channels,3,padding=1))
        nn.init.normal_(self.decode[-1].weight,mean=0.,std=1e-3)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self,tokens:torch.Tensor,token_hw:tuple[int,int],output_hw:tuple[int,int],base_state:torch.Tensor):
        B,N,D=tokens.shape
        ht,wt=token_hw
        if N!=ht*wt:
            raise ValueError('CoarseForecastHead token 数与 token_hw 不一致')
        if base_state.shape!=(B,self.out_channels,*output_hw):
            raise ValueError('base_state must match output batch/channel/native spatial shape')
        z=self.norm(tokens).transpose(1,2).reshape(B,D,ht,wt)
        tendency=crop_native_grid(self.decode(z),output_hw,self.patch_size)
        return base_state+tendency,tendency
