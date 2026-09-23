r"""
GenCast 风格 norm-conditioned Transformer。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphMultiheadAttention(nn.Module):
    r"""
    支持图拓扑掩码的多头自注意力。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            int(hidden_dim),
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out


class ConditionalLayerNorm(nn.Module):
    r"""
    由全局条件调制的 LayerNorm。
    """

    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim), elementwise_affine=False)
        self.modulation = nn.Linear(int(cond_dim), int(hidden_dim) * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, N, D), got {tuple(x.shape)}")
        if cond.ndim != 2 or cond.shape[0] != x.shape[0]:
            raise ValueError(f"Expected cond shape ({x.shape[0]}, C), got {tuple(cond.shape)}")
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        y = self.norm(x)
        return y * (1.0 + scale[:, None, :]) + shift[:, None, :]


class NormConditionedTransformerBlock(nn.Module):
    r"""
    GenCast 风格 Transformer 块：条件归一化、图掩码注意力、FFN。
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float = 0.0,
        use_graph_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph_attention = bool(use_graph_attention)
        self.norm1 = ConditionalLayerNorm(hidden_dim, cond_dim)
        self.attn = GraphMultiheadAttention(hidden_dim, num_heads, dropout=float(dropout))
        self.norm2 = ConditionalLayerNorm(hidden_dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(int(hidden_dim), int(ffn_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ffn_hidden), int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = attn_mask if self.use_graph_attention else None
        x = x + self.attn(self.norm1(x, cond), mask)
        return x + self.ffn(self.norm2(x, cond))


class MeshTransformer(nn.Module):
    r"""
    在 mesh 节点上运行的 GenCast 风格稀疏 Transformer。
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float = 0.0,
        use_graph_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph_attention = bool(use_graph_attention)
        self.blocks = nn.ModuleList(
            [
                NormConditionedTransformerBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                    use_graph_attention=use_graph_attention,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def _mask(
        self,
        node_count: int,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_graph_attention:
            return None
        mask = torch.full(
            (int(node_count), int(node_count)),
            float("-inf"),
            device=reference.device,
            dtype=reference.dtype,
        )
        senders = senders.to(device=reference.device, dtype=torch.long)
        receivers = receivers.to(device=reference.device, dtype=torch.long)
        if senders.numel() > 0:
            mask[receivers, senders] = 0.0
        mask.fill_diagonal_(0.0)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._mask(x.shape[1], senders, receivers, x)
        for block in self.blocks:
            x = block(x, cond, attn_mask=mask)
        return x


__all__ = [
    "ConditionalLayerNorm",
    "GraphMultiheadAttention",
    "MeshTransformer",
    "NormConditionedTransformerBlock",
]