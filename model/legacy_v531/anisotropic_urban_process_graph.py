"""扩散时间步自适应的城市过程图。

该模块在现有 wind-aware morphology graph 上添加扩散时间步调制，
让风向各向异性和粗糙度阻塞强度随去噪阶段变化。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn

try:
    from .morphology_graph import WindAwareMorphologyGraph
except ImportError:  # pragma: no cover
    from models.morphology_graph import WindAwareMorphologyGraph


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AnisotropicUrbanProcessGraph(WindAwareMorphologyGraph):
    """带 diffusion timestep 调制的 wind-aware morphology graph。"""

    def __init__(
        self,
        dim: int,
        k: int = 8,
        morphology_sigma: float = 1.0,
        spatial_sigma: float = 1.0,
        wind_temperature: float = 0.25,
        wind_strength: float = 1.0,
        roughness_blocking_strength: float = 0.5,
        time_embed_dim: int = 64,
        time_hidden_dim: int = 32,
        init_gamma: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            k=k,
            morphology_sigma=morphology_sigma,
            spatial_sigma=spatial_sigma,
            wind_temperature=wind_temperature,
            wind_strength=wind_strength,
            roughness_blocking_strength=roughness_blocking_strength,
            **kwargs,
        )
        self.base_wind_strength = float(wind_strength)
        self.base_roughness_blocking_strength = float(roughness_blocking_strength)
        self.time_embed_dim = int(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, int(time_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(time_hidden_dim), 2),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        self.gamma.data.fill_(float(init_gamma))
        self.last_time_diagnostics: Dict[str, torch.Tensor] = {}

    def _time_scales(self, diffusion_t: Optional[torch.Tensor], ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if diffusion_t is None:
            one = ref.new_tensor(1.0)
            return one, one
        emb = _timestep_embedding(diffusion_t.to(device=ref.device, dtype=ref.dtype), self.time_embed_dim)
        raw = self.time_mlp(emb.to(dtype=ref.dtype))
        scales = 1.0 + 0.5 * torch.tanh(raw)
        return scales[:, 0].mean(), scales[:, 1].mean()

    def build_adjacency(
        self,
        static_feat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
        wind_uv: Optional[torch.Tensor] = None,
        *,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        static_hash: Optional[str] = None,
        roughness_proxy: Optional[torch.Tensor] = None,
        diffusion_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ref = self.gamma if static_feat is None else static_feat
        wind_scale, block_scale = self._time_scales(diffusion_t, ref)
        old_wind = self.wind_strength
        old_block = self.roughness_blocking_strength
        self.wind_strength = self.base_wind_strength * float(wind_scale.detach().cpu())
        self.roughness_blocking_strength = self.base_roughness_blocking_strength * float(block_scale.detach().cpu())
        try:
            adj, diagnostics = super().build_adjacency(
                static_feat=static_feat,
                spatial_hw=spatial_hw,
                wind_uv=wind_uv,
                static_cont=static_cont,
                static_cat=static_cat,
                static_hash=static_hash,
                roughness_proxy=roughness_proxy,
            )
        finally:
            self.wind_strength = old_wind
            self.roughness_blocking_strength = old_block
        diagnostics["process_graph/wind_scale_t"] = wind_scale.detach()
        diagnostics["process_graph/block_scale_t"] = block_scale.detach()
        diagnostics["process_graph/graph_entropy"] = diagnostics.get("graph_entropy", adj.new_tensor(0.0))
        self.last_time_diagnostics = diagnostics
        return adj, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
        wind_uv: Optional[torch.Tensor] = None,
        *,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        roughness_proxy: Optional[torch.Tensor] = None,
        diffusion_t: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if spatial_hw is None:
            raise ValueError("forward 需要 spatial_hw")
        adj, diagnostics = self.build_adjacency(
            static_feat=static_feat,
            spatial_hw=spatial_hw,
            wind_uv=wind_uv,
            static_cont=static_cont,
            static_cat=static_cat,
            roughness_proxy=roughness_proxy,
            diffusion_t=diffusion_t,
        )
        if adj.size(0) != x.size(0):
            adj = adj.expand(x.size(0), -1, -1)
        msg = torch.einsum("bnm,bmd->bnd", adj.to(device=x.device, dtype=x.dtype), x)
        delta = self.mlp(self.norm(msg))
        out = x + self.gamma.to(device=x.device, dtype=x.dtype) * delta
        self.last_diagnostics = diagnostics
        if return_diagnostics:
            return out, diagnostics
        return out


__all__ = ["AnisotropicUrbanProcessGraph"]