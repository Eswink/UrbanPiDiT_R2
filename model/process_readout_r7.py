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


def require_position_encoding_mode(mode: str, *, positional_process_readout: bool) -> str:
    """Validate the #77 frequency basis and where it may legally be switched on.

    A non-legacy basis only exists inside the positional readout, so turning it
    on with that pathway off would be a silently ignored switch and is refused
    here rather than in every caller's constructor.
    """
    if mode not in PositionalProcessReadout.MODES:
        raise ValueError(f"position_encoding_mode must be one of {PositionalProcessReadout.MODES}")
    if mode != PositionalProcessReadout.LEGACY_FREQUENCIES and not positional_process_readout:
        raise ValueError("a non-legacy position_encoding_mode only exists inside the positional "
                         "process readout; turning it on without positional_process_readout "
                         "would be silently ignored")
    return mode


class PositionalProcessReadout(nn.Module):
    """One query per output position reads every process token (N x M).

    Opt-in draft_query_feedback uses the aligned existing draft encoding inside
    the same query normalization. Off is the original context-only expression;
    neither mode adds a parameter, state key or random draw (ADR 0034).

    ``position_encoding_mode`` (#77) selects the fixed frequency basis:

    - ``'legacy'`` (default) is the pre-change basis ``2**-j`` described in
      ``docs/R7_77_POSITION_ENCODING_BAND.md``: on a 33-point axis only 16 of
      the 192 channels carry any spatial variation at fp32, because the
      geometric decay leaves the phase span far below one radian. It is kept
      bit-for-bit so existing checkpoints keep their exact semantics and a new
      mode is never a silent rewrite of an old digest.
    - ``'nyquist_band'`` spreads the same channel budget geometrically from
      ``band_min_cycles`` to ``band_max_cycles`` cycles per normalized axis,
      with ``band_max_cycles`` at most the axis Nyquist limit ``(N-1)/2`` for
      the largest expected token grid. No new parameter, buffer or random draw
      is added: the basis is still a fixed function of the token coordinates.
    """

    LEGACY_FREQUENCIES = "legacy"
    NYQUIST_BAND_FREQUENCIES = "nyquist_band"
    MODES = (LEGACY_FREQUENCIES, NYQUIST_BAND_FREQUENCIES)
    # Cycles per normalized axis; the top is the Nyquist limit of the largest
    # token grid this model sees (the 33x33 patch grid of a 65x65 field).
    NYQUIST_MIN_CYCLES = 0.5
    NYQUIST_MAX_CYCLES = 16.0

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0,
                 pooled_readout_query: bool = False, draft_query_feedback: bool = False,
                 position_encoding_mode: str = LEGACY_FREQUENCIES):
        super().__init__()
        for value, name in ((pooled_readout_query, 'pooled_readout_query'),
                            (draft_query_feedback, 'draft_query_feedback')):
            if type(value) is not bool:
                raise ValueError(f'{name} must be boolean')
        if position_encoding_mode not in self.MODES:
            raise ValueError(f"position_encoding_mode must be one of {self.MODES}")
        dim = int(dim)
        if dim % 4:
            raise ValueError("positional readout needs dim divisible by 4 (two axes x sin/cos)")
        self.dim = dim
        self.heads = int(heads)
        self.pooled_readout_query = pooled_readout_query
        self.draft_query_feedback = draft_query_feedback
        self.position_encoding_mode = position_encoding_mode
        self.query_norm = nn.LayerNorm(dim)
        self.process_norm = nn.LayerNorm(dim)
        self.attention = SDPAttention(dim, self.heads, dropout, cross=True)
        # Non-persistent: the basis is a fixed function, not a trained artifact,
        # so it never enters a state_dict or a checkpoint contract.
        frequencies = 2.0 ** (-torch.arange(dim // 4, dtype=torch.float32))
        if position_encoding_mode == self.NYQUIST_BAND_FREQUENCIES:
            # Same channel budget, spread geometrically across the usable band
            # instead of decaying to 2**-47. Cycles per normalized axis [0,1].
            # logspace in fp32 can round an endpoint a ulp outside the band, so
            # both ends are pinned exactly rather than merely intended.
            frequencies = torch.logspace(
                math.log10(self.NYQUIST_MIN_CYCLES), math.log10(self.NYQUIST_MAX_CYCLES),
                dim // 4, dtype=torch.float32)
            frequencies[0] = self.NYQUIST_MIN_CYCLES
            frequencies[-1] = self.NYQUIST_MAX_CYCLES
        self.register_buffer("frequencies", frequencies, persistent=False)

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
                token_hw: tuple[int, int], *, draft_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if process.ndim != 3 or context.ndim != 3:
            raise ValueError("process readout needs [B,M,D] process tokens and [B,N,D] context")
        if process.shape[0] != context.shape[0] or process.shape[2] != self.dim \
                or context.shape[2] != self.dim:
            raise ValueError("process readout batch/feature dimensions do not match")
        positions = int(token_hw[0]) * int(token_hw[1])
        if process.shape[1] < 1 or context.shape[1] != positions:
            raise ValueError(f"context has {context.shape[1]} positions but token_hw "
                             f"{tuple(token_hw)} implies {positions}")
        if self.draft_query_feedback:
            if not torch.is_tensor(draft_tokens) or draft_tokens.shape != context.shape:
                raise ValueError('draft_query_feedback requires aligned [B,N,D] draft_tokens')
            if draft_tokens.device != context.device:
                raise ValueError('draft_tokens/context devices differ')
            if draft_tokens.dtype != context.dtype:
                raise ValueError('draft_tokens/context dtypes differ')
            # One normalization of the sum, not separately normalized sources.
            query = self.query_norm(context + draft_tokens)
        else:
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
