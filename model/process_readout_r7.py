"""Positional process readout (#72 RW-A).

The pre-RW-A bottleneck is one line: the process state is averaged over its
tokens (``process.mean(dim=1)``), projected once, and the same vector is added
to every output position. The solver therefore cannot tell a process token that
belongs to the region it is correcting from one that belongs to the far side of
the domain - every position reads the same pooled summary.

RW-A replaces that read with a per-position one: output position *i* forms its
own query and cross-attends over all M process tokens, so the read costs N x M
and the summary the solver adds is position-dependent. Three boundaries are
deliberate and are what keeps this an *additive* change rather than RW-B:

- no gate: the read is added, never multiplied or masked per position;
- no per-position recurrent state: the process tokens remain the whole state;
- the process recurrence itself is untouched - only how it is read changes, so a
  model with the switch off is bit for bit the previous implementation.

The query is the position's own context token, projected by the attention's own
query weights, plus a fixed (parameter-free) multi-scale sinusoidal encoding of
the token's (row, column) location. A learned query table would have to declare
a maximum token grid - a resolution limit the rest of this model does not have -
and a pooled query would re-introduce the very averaging RW-A removes.

``pooled_readout_query`` (round two) constructs exactly that re-introduced
averaging as a *capacity control*, not as a candidate mechanism: the module, its
parameters and its N x M attention are unchanged, and only the query becomes the
mean over positions (taken before the position encoding is added, so no part of
the encoding survives the average). A run can therefore hold the readout's
parameters and compute fixed and vary whether the read is position-dependent,
which is what separates "the position dependence helps" from "the extra capacity
helps". It is off by default, and off is the pre-change implementation.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .layers.sdpa import SDPAttention


class PositionalProcessReadout(nn.Module):
    """One query per output position reads every process token (N x M)."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0,
                 pooled_readout_query: bool = False):
        super().__init__()
        if type(pooled_readout_query) is not bool:
            raise ValueError("pooled_readout_query must be boolean")
        dim = int(dim)
        if dim % 4:
            raise ValueError("positional readout needs dim divisible by 4 (two axes x sin/cos)")
        self.dim = dim
        self.heads = int(heads)
        self.pooled_readout_query = pooled_readout_query
        self.query_norm = nn.LayerNorm(dim)
        self.process_norm = nn.LayerNorm(dim)
        self.attention = SDPAttention(dim, self.heads, dropout, cross=True)
        # Non-persistent: the basis is a fixed function, not a trained artifact,
        # so it never enters a state_dict or a checkpoint contract.
        self.register_buffer(
            "frequencies", 2.0 ** (-torch.arange(dim // 4, dtype=torch.float32)),
            persistent=False)

    def position_encoding(self, token_hw: tuple[int, int], *,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``[N, dim]`` fixed harmonics of each token's normalized (row, column)."""
        rows, columns = int(token_hw[0]), int(token_hw[1])
        if rows < 1 or columns < 1:
            raise ValueError("token grid must be nonempty")
        row = torch.linspace(0.0, 1.0, rows, device=device, dtype=torch.float32).reshape(-1, 1)
        column = torch.linspace(0.0, 1.0, columns, device=device, dtype=torch.float32).reshape(1, -1)
        frequencies = self.frequencies.to(device).reshape(-1, 1, 1)
        axis_row = torch.cat([torch.sin(math.pi * frequencies * row),
                              torch.cos(math.pi * frequencies * row)], dim=0)
        axis_column = torch.cat([torch.sin(math.pi * frequencies * column),
                                 torch.cos(math.pi * frequencies * column)], dim=0)
        return torch.cat([axis_row.expand(-1, -1, columns),
                          axis_column.expand(-1, rows, -1)],
                         dim=0).reshape(self.dim, rows * columns).transpose(0, 1).to(dtype)

    def forward(self, process: torch.Tensor, context: torch.Tensor,
                token_hw: tuple[int, int]) -> torch.Tensor:
        if process.ndim != 3 or context.ndim != 3:
            raise ValueError("process readout needs [B,M,D] process tokens and [B,N,D] context")
        if process.shape[0] != context.shape[0] or process.shape[2] != self.dim \
                or context.shape[2] != self.dim:
            raise ValueError("process readout batch/feature dimensions do not match")
        positions = int(token_hw[0]) * int(token_hw[1])
        if process.shape[1] < 1 or context.shape[1] != positions:
            raise ValueError(f"context has {context.shape[1]} positions but token_hw "
                             f"{tuple(token_hw)} implies {positions}")
        query = self.query_norm(context)
        if self.pooled_readout_query:
            # Pooled *before* the position encoding is added: averaging the
            # encoded queries would mix the encoding's own mean into the control
            # arm, which is not the same statement as "the read carries no
            # position". Every position then reads the identical query, so the
            # attention returns the identical vector at every position.
            query = query.mean(dim=1, keepdim=True).expand(-1, positions, -1)
        else:
            query = query + self.position_encoding(
                token_hw, device=context.device, dtype=context.dtype)
        return self.attention(query, self.process_norm(process))
