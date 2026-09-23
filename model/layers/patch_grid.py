"""Padding/cropping without interpolating the native grid."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def pad_patch_grid(x:torch.Tensor,patch_size:int,*,periodic_width=False):
    if x.ndim!=4 or min(x.shape)<1 or patch_size<1:
        raise ValueError('nonempty [B,C,H,W] and positive patch_size required')
    h,w=x.shape[-2:]
    ph,pw=(-h)%patch_size,(-w)%patch_size
    if pw:
        x=F.pad(x,(0,pw,0,0),mode='circular' if periodic_width else 'replicate')
    if ph:
        x=F.pad(x,(0,0,0,ph),mode='replicate')
    return x


def crop_native_grid(x,output_hw,patch_size):
    h,w=output_hw
    if h<1 or w<1 or x.shape[-2:]!=(((h+patch_size-1)//patch_size)*patch_size,((w+patch_size-1)//patch_size)*patch_size):
        raise ValueError('decoded patch grid does not match requested native-grid geometry')
    return x[...,:h,:w]
