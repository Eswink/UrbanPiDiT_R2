"""Compact regional comparison adaptations; no pretrained/SOTA claim.

Architecture references and deviations are documented in R7_BASELINES.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .coarse_forecast import LeadTimeEmbedding
from .layers.patch_grid import pad_patch_grid,crop_native_grid


@dataclass
class BaselineOutput:
    forecast:torch.Tensor
    tendency:torch.Tensor
    reasoning_steps:int=0


class _Base(nn.Module):
    def __init__(self,in_channels,history_steps,out_channels,dim):
        super().__init__()
        self.in_channels=int(in_channels)
        self.history_steps=int(history_steps)
        self.out_channels=int(in_channels if out_channels is None else out_channels)
        self.dim=int(dim)
        if min(self.in_channels,self.history_steps,self.out_channels,self.dim)<1 or self.out_channels>self.in_channels:
            raise ValueError('invalid channel/history/dimension configuration')
        self.lead=LeadTimeEmbedding(dim)

    def inputs(self,batch):
        x=batch['coarse_history']
        if x.ndim!=5 or x.shape[1:3]!=(self.history_steps,self.in_channels) or min(x.shape)<1:
            raise ValueError('history must match [B,T,C,H,W]')
        lead=self.lead(batch.get('lead_time_hours'),batch=len(x),device=x.device,dtype=x.dtype)
        return x,lead[:,:,None,None]

    def output(self,history,tendency):
        return BaselineOutput(history[:,-1,:self.out_channels]+tendency,tendency)


def _conv_block(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,padding_mode='replicate'),nn.GroupNorm(1,cout),nn.GELU(),
        nn.Conv2d(cout,cout,3,padding=1,padding_mode='replicate'),nn.GroupNorm(1,cout),nn.GELU())


def _small_head(head):
    nn.init.normal_(head.weight,std=1e-3)
    nn.init.zeros_(head.bias)


class UNetForecaster(_Base):
    """Two-level finite-boundary U-Net adaptation for residual weather states."""
    def __init__(self,in_channels,history_steps=2,out_channels=None,dim=64):
        super().__init__(in_channels,history_steps,out_channels,dim)
        self.enc1=_conv_block(in_channels*history_steps,dim)
        self.down1=nn.Conv2d(dim,dim*2,3,stride=2,padding=1)
        self.enc2=_conv_block(dim*2,dim*2)
        self.down2=nn.Conv2d(dim*2,dim*4,3,stride=2,padding=1)
        self.middle=_conv_block(dim*4,dim*4)
        self.up2=nn.ConvTranspose2d(dim*4,dim*2,2,stride=2)
        self.dec2=_conv_block(dim*4,dim*2)
        self.up1=nn.ConvTranspose2d(dim*2,dim,2,stride=2)
        self.dec1=_conv_block(dim*2,dim)
        self.head=nn.Conv2d(dim,self.out_channels,1)
        _small_head(self.head)

    def forward(self,batch):
        x,lead=self.inputs(batch)
        B,T,C,H,W=x.shape
        one=self.enc1(x.reshape(B,T*C,H,W))+lead
        two=self.enc2(self.down1(one))
        mid=self.middle(self.down2(two))
        up=self.up2(mid)[...,:two.shape[-2],:two.shape[-1]]
        up=self.dec2(torch.cat([up,two],1))
        up=self.up1(up)[...,:H,:W]
        return self.output(x,self.head(self.dec1(torch.cat([up,one],1))))


class ConvLSTMForecaster(_Base):
    """Single convolutional recurrent layer, no peepholes or radar-specific decoder."""
    def __init__(self,in_channels,history_steps=2,out_channels=None,dim=64):
        super().__init__(in_channels,history_steps,out_channels,dim)
        self.gates=nn.Conv2d(in_channels+dim,4*dim,3,padding=1,padding_mode='replicate')
        self.head=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,self.out_channels,1))
        _small_head(self.head[-1])

    def forward(self,batch):
        x,lead=self.inputs(batch)
        hidden=x.new_zeros(len(x),self.dim,*x.shape[-2:])
        cell=torch.zeros_like(hidden)
        for frame in x.unbind(1):
            i,f,o,g=self.gates(torch.cat([frame,hidden],1)).chunk(4,1)
            cell=torch.sigmoid(f+1)*cell+torch.sigmoid(i)*torch.tanh(g)
            hidden=torch.sigmoid(o)*torch.tanh(cell)
        return self.output(x,self.head(hidden+lead))


class AFNOMixer(nn.Module):
    """Block-diagonal complex MLP + soft shrinkage, all FFT math in FP32."""
    def __init__(self,dim,blocks=4,shrinkage=.01):
        super().__init__()
        if blocks<1 or dim%blocks or not math.isfinite(shrinkage) or shrinkage<0:
            raise ValueError('invalid block division/shrinkage')
        self.blocks,self.block_dim,self.shrinkage=blocks,dim//blocks,float(shrinkage)
        shape=(2,blocks,self.block_dim,self.block_dim)
        self.w1=nn.Parameter(torch.randn(shape)*.02)
        self.w2=nn.Parameter(torch.randn(shape)*.02)
        self.b1=nn.Parameter(torch.zeros(2,blocks,self.block_dim))
        self.b2=nn.Parameter(torch.zeros(2,blocks,self.block_dim))

    def forward(self,x):
        B,C,H,W=x.shape
        with torch.autocast(x.device.type,enabled=False):
            spectrum=torch.fft.rfft2(x.float(),norm='ortho').permute(0,2,3,1)
            spectrum=spectrum.reshape(B,H,W//2+1,self.blocks,self.block_dim)
            w1=torch.complex(self.w1[0].float(),self.w1[1].float())
            w2=torch.complex(self.w2[0].float(),self.w2[1].float())
            first=torch.einsum('bhwgi,gij->bhwgj',spectrum,w1)
            first=torch.complex(F.relu(first.real+self.b1[0]),F.relu(first.imag+self.b1[1]))
            second=torch.einsum('bhwgi,gij->bhwgj',first,w2)+torch.complex(self.b2[0],self.b2[1])
            pair=F.softshrink(torch.view_as_real(second),lambd=self.shrinkage)
            spectral=torch.view_as_complex(pair.contiguous()).reshape(B,H,W//2+1,C).permute(0,3,1,2)
            result=torch.fft.irfft2(spectral,s=(H,W),norm='ortho')
        return result.to(x.dtype)


class AFNOBlock(nn.Module):
    def __init__(self,dim,blocks,shrinkage):
        super().__init__()
        self.norm1=nn.GroupNorm(1,dim)
        self.mix=AFNOMixer(dim,blocks,shrinkage)
        self.norm2=nn.GroupNorm(1,dim)
        self.mlp=nn.Sequential(nn.Conv2d(dim,dim*2,1),nn.GELU(),nn.Conv2d(dim*2,dim,1))

    def forward(self,x):
        x=x+self.mix(self.norm1(x))
        return x+self.mlp(self.norm2(x))


class AFNOSmallForecaster(_Base):
    """AFNO-inspired patch forecaster, not the FourCastNet architecture/recipe."""
    def __init__(self,in_channels,history_steps=2,out_channels=None,dim=64,patch_size=2,depth=4,blocks=4,shrinkage=.01):
        super().__init__(in_channels,history_steps,out_channels,dim)
        if patch_size<1 or depth<1:
            raise ValueError('positive patch/depth required')
        self.patch_size=patch_size
        self.stem=nn.Conv2d(in_channels*history_steps,dim,patch_size,stride=patch_size)
        self.layers=nn.Sequential(*[AFNOBlock(dim,blocks,shrinkage) for _ in range(depth)])
        self.up=nn.ConvTranspose2d(dim,dim,patch_size,stride=patch_size)
        self.head=nn.Conv2d(dim,self.out_channels,3,padding=1)
        _small_head(self.head)

    def forward(self,batch):
        x,lead=self.inputs(batch)
        B,T,C,H,W=x.shape
        z=self.stem(pad_patch_grid(x.reshape(B,T*C,H,W),self.patch_size))+lead
        y=self.head(self.up(self.layers(z)))
        return self.output(x,crop_native_grid(y,(H,W),self.patch_size))
