"""H-C: one post-backbone context-mixing layer, global or local (``docs/goals/s3-global-mixing.md``).

The registered trunk mixes context tokens only inside ``window_size`` windows
(``model/coarse_encoder.py``), which at the registered ``window_size=4`` and ``patch_size=2``
covers about 2.0 degrees against a 65x65 domain, and there is no N x N attention anywhere in the
backbone. H-C adds exactly one context-mixing layer after the backbone and runs two arms that
differ only in the mixing range: ``global`` attends over every token, ``local`` keeps the trunk's
windowed attention.

Both arms are the same shell -- LayerNorm, ``qkv``, ``proj``, LayerNorm, ``FeedForward(mlp_ratio)``
-- so their parameter counts are identical **by construction**, because the two blocks declare the
same tensors with the same shapes. What cannot be aligned is compute: the global arm's attention
costs about 68x the local arm's at N=1089, d=192, window=4. That difference is declared in the
protocol and reported, never claimed away.

The layer is built last under ``isolated_stream()`` by its caller, so switching it off leaves the
previous implementation bit for bit.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .layers.sdpa import FeedForward
from .layers.window_attention import WindowAttentionBlock

GLOBAL = "global"
LOCAL = "local"
MIXING_MODES = (GLOBAL, LOCAL)


class GlobalAttentionBlock(nn.Module):
    """Full N x N self-attention with ``WindowAttentionBlock``'s parameter layout.

    Same tensors in the same order -- ``n1``, ``qkv``, ``proj``, ``n2``, ``ff`` -- so a global
    block and a windowed block of the same ``dim``/``heads``/``mlp_ratio`` have equal parameter
    counts; only the attention mask and therefore the compute differ.
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = int(dim)
        self.heads = int(heads)
        self.hd = int(dim) // int(heads)
        self.dropout = float(dropout)
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, tokens: torch.Tensor, hw=None) -> torch.Tensor:
        """``[B,N,dim]`` in, ``[B,N,dim]`` out. ``hw`` is accepted and unused for signature parity."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("global attention tokens must be [B,N,dim]")
        batch, length, dim = tokens.shape
        x = self.n1(tokens)
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0)
        mixed = self.proj(attended.transpose(1, 2).reshape(batch, length, dim))
        out = tokens + mixed
        return out + self.ff(self.n2(out))


class ContextMixLayer(nn.Module):
    """The single H-C layer: full attention (``global``) or windowed attention (``local``)."""

    MODES = MIXING_MODES

    def __init__(self, dim: int, heads: int, window_size: int, *, mode: str,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mixing mode must be one of {self.MODES}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.window_size = int(window_size)
        self.mode = mode
        if mode == GLOBAL:
            self.block = GlobalAttentionBlock(dim, heads, mlp_ratio, dropout)
        else:
            self.block = WindowAttentionBlock(dim, heads, window_size, mlp_ratio, dropout, shift=False)

    def forward(self, tokens: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
        """Residual context mixing; the ``[B,N,dim]`` shape is preserved."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("context tokens must be [B,N,dim]")
        return self.block(tokens, token_hw)
