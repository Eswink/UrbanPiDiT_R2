from __future__ import annotations
import torch
from torch import nn
from .layers.sdpa import CrossBlock

DEFAULT_PROCESS_NAMES=(
    'pressure_gradient','advection','moisture_transport','convergence','vorticity','stability',
    'boundary_layer','surface_flux','roughness_drag','heat_storage','evapotranspiration','ventilation'
)

class ProcessStateInitializer(nn.Module):
    def __init__(self,dim:int=256,anchored:int=12,free:int=12,heads:int=8,source_dim:int|None=None):
        super().__init__(); self.anchored=anchored; self.free=free; self.num_tokens=anchored+free; self.dim=dim
        self.queries=nn.Parameter(torch.randn(1,self.num_tokens,dim)*0.02)
        self.src=nn.Identity() if source_dim in (None,dim) else nn.Linear(source_dim,dim)
        self.cross=CrossBlock(dim,heads,2.0,0.0)
        self.process_head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,1))
    def forward(self,context):
        B=context.shape[0]; p=self.queries.expand(B,-1,-1); p=self.cross(p,self.src(context)); preds=self.process_head(p[:,:self.anchored]).squeeze(-1); return p,preds
