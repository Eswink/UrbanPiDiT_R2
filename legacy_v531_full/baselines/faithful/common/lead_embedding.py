r"""
标量 Fourier 嵌入，用于 lead time 与噪声水平条件。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierScalarEmbedding(nn.Module):
    r"""
    将标量条件编码为 MLP 向量。

    Parameters
    ----
    out_dim : int
        输出嵌入维度。
    num_frequencies : int, optional, default=16
        Fourier 频率数量。
    hidden_dim : int, optional, default=None
        MLP 隐层维度。None 时使用 ``out_dim``。
    max_period : float, optional, default=10000.0
        频率尺度。
    """

    def __init__(
        self,
        out_dim: int,
        num_frequencies: int = 16,
        hidden_dim: int | None = None,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        freq_count = max(1, int(num_frequencies))
        hidden = int(hidden_dim or out_dim)
        exponents = torch.arange(freq_count, dtype=torch.float32) / float(freq_count)
        frequencies = torch.exp(-math.log(float(max_period)) * exponents)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(freq_count * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        r"""
        编码标量条件。

        Parameters
        ----
        value : torch.Tensor
            shape :math:`(B,)` 或可展平为 batch 的标量张量。

        Returns
        ----
        torch.Tensor
            shape :math:`(B, D)` 的条件向量。
        """

        value = value.float().flatten()
        phase = value[:, None] * self.frequencies[None, :].to(device=value.device, dtype=value.dtype)
        features = torch.cat([value[:, None], torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.net(features)