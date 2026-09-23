r"""
GenCast EDM denoiser。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from baselines.faithful.common import build_graphcast_graph, edm_preconditioning
from baselines.faithful.common.grid_graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from baselines.faithful.common.typed_graph import BipartiteGraphNet
from baselines.faithful.gencast.transformer import MeshTransformer


class FourierFeaturesMLP(nn.Module):
    r"""
    GenCast 风格标量噪声 Fourier 特征编码。
    """

    def __init__(
        self,
        out_dim: int,
        *,
        num_frequencies: int = 16,
        hidden_dim: int | None = None,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        freq_count = max(1, int(num_frequencies))
        hidden = int(hidden_dim or out_dim)
        exponents = torch.arange(freq_count, dtype=torch.float32) / float(freq_count)
        frequencies = torch.exp(-math.log(float(max_period)) * exponents)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(freq_count * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(out_dim)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().flatten()
        frequencies = self.frequencies.to(device=value.device, dtype=value.dtype)
        phase = value[:, None] * frequencies[None, :]
        features = torch.cat([value[:, None], torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.net(features)


def _build_output_mlp(hidden_dim: int, depth: int, dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    for _ in range(max(0, int(depth))):
        layers.extend(
            [
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        )
    if not layers:
        return nn.Identity()
    return nn.Sequential(*layers)


class GenCastDenoiser(nn.Module):
    r"""
    GraphCast 三图骨架 + GenCast mesh Transformer 的 EDM 去噪器。
    """

    def __init__(
        self,
        *,
        cond_channels: int,
        out_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        ffn_hidden: int = 1024,
        dropout: float = 0.0,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        use_graph_attention: bool = True,
        graph_k: int = 8,
        mesh_size: int | None = None,
        mesh_height: int = 4,
        mesh_width: int = 4,
        k_nn: int | None = None,
        radius_query_fraction_edge_length: float = 0.6,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        self.cond_channels = int(cond_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.sigma_data = float(sigma_data)
        self.mesh_size = None if mesh_size is None else int(mesh_size)
        self.mesh_height = int(mesh_height)
        self.mesh_width = int(mesh_width)
        self.k_nn = int(k_nn if k_nn is not None else graph_k)
        self.radius_query_fraction_edge_length = float(radius_query_fraction_edge_length)

        node_dim = self.hidden_channels
        edge_dim = self.hidden_channels
        cond_dim = self.hidden_channels
        gnn_layers = max(1, int(encoder_depth))
        grid_input_dim = self.cond_channels + self.out_channels + NODE_FEATURE_DIM

        self.noise_embedding = FourierFeaturesMLP(cond_dim, num_frequencies=16, hidden_dim=cond_dim)
        self.grid2mesh = BipartiteGraphNet(
            sender_input_dim=grid_input_dim,
            receiver_input_dim=NODE_FEATURE_DIM,
            edge_input_dim=EDGE_FEATURE_DIM,
            node_dim=node_dim,
            edge_dim=edge_dim,
            message_passing_steps=1,
            hidden_layers=gnn_layers,
            dropout=dropout,
            cond_dim=cond_dim,
        )
        self.mesh_transformer = MeshTransformer(
            hidden_dim=node_dim,
            cond_dim=cond_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            use_graph_attention=use_graph_attention,
        )
        self.mesh2grid = BipartiteGraphNet(
            sender_input_dim=node_dim,
            receiver_input_dim=node_dim,
            edge_input_dim=EDGE_FEATURE_DIM,
            node_dim=node_dim,
            edge_dim=edge_dim,
            output_dim=None,
            message_passing_steps=1,
            hidden_layers=max(1, int(decoder_depth)),
            dropout=dropout,
            cond_dim=cond_dim,
            embed_sender=False,
            embed_receiver=False,
        )
        self.output_norm = nn.LayerNorm(node_dim)
        self.output_mlp = _build_output_mlp(node_dim, decoder_depth, dropout)
        self.output_head = nn.Linear(node_dim, self.out_channels)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def _graph(self, height: int, width: int, reference: torch.Tensor):
        return build_graphcast_graph(
            height,
            width,
            mesh_size=self.mesh_size,
            mesh_height=self.mesh_height,
            mesh_width=self.mesh_width,
            k_grid_to_mesh=self.k_nn,
            k_mesh_to_grid=self.k_nn,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _nodes(self, field: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = field.shape
        return field.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)

    def _grid_features(self, noisy_in: torch.Tensor, cond_field: torch.Tensor, graph) -> torch.Tensor:
        bsz = noisy_in.shape[0]
        node_features = graph.grid_node_features.unsqueeze(0).expand(bsz, -1, -1)
        return torch.cat([self._nodes(cond_field), self._nodes(noisy_in), node_features], dim=-1)

    def forward(self, noisy: torch.Tensor, sigma: torch.Tensor, cond_field: torch.Tensor) -> torch.Tensor:
        r"""
        对 noisy target 执行 EDM 预条件去噪。
        """

        if noisy.ndim != 4 or cond_field.ndim != 4:
            raise ValueError("noisy and cond_field must be 4D tensors")
        if noisy.shape[0] != cond_field.shape[0] or noisy.shape[-2:] != cond_field.shape[-2:]:
            raise ValueError("noisy and cond_field must share batch and spatial shape")
        if noisy.shape[1] != self.out_channels:
            raise ValueError(f"Expected noisy channels={self.out_channels}, got {noisy.shape[1]}")
        if cond_field.shape[1] != self.cond_channels:
            raise ValueError(f"Expected cond channels={self.cond_channels}, got {cond_field.shape[1]}")

        bsz, _, height, width = noisy.shape
        coeffs = edm_preconditioning(sigma, noisy, sigma_data=self.sigma_data)
        noisy_in = coeffs.c_in * noisy
        graph = self._graph(height, width, noisy)
        grid_features = self._grid_features(noisy_in, cond_field, graph)
        mesh_features = graph.mesh_node_features.unsqueeze(0).expand(bsz, -1, -1)
        noise_cond = self.noise_embedding(coeffs.c_noise * 4.0)

        grid_latent, mesh_latent, _ = self.grid2mesh(
            grid_features,
            mesh_features,
            graph.grid2mesh_edge_features,
            graph.grid2mesh_senders,
            graph.grid2mesh_receivers,
            graph.mesh_coords.shape[0],
            cond=noise_cond,
        )
        mesh_latent = self.mesh_transformer(
            mesh_latent,
            noise_cond,
            graph.mesh_senders,
            graph.mesh_receivers,
        )
        _, grid_latent, _ = self.mesh2grid(
            mesh_latent,
            grid_latent,
            graph.mesh2grid_edge_features,
            graph.mesh2grid_senders,
            graph.mesh2grid_receivers,
            graph.grid_coords.shape[0],
            cond=noise_cond,
        )
        raw = self.output_head(self.output_mlp(self.output_norm(grid_latent)))
        raw = raw.reshape(bsz, height, width, self.out_channels).permute(0, 3, 1, 2)
        return coeffs.c_skip * noisy + coeffs.c_out * raw


__all__ = ["FourierFeaturesMLP", "GenCastDenoiser"]