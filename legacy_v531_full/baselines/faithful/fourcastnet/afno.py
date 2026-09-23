r"""
FourCastNet AFNO 块。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDiagonalAFNO2D(nn.Module):
    r"""
    真实 AFNO 风格的 block-diagonal 频域 MLP。

    Parameters
    ----
    hidden_channels : int
        token 通道数。
    num_blocks : int, optional, default=2
        通道分组数量。
    hidden_size_factor : int, optional, default=1
        每组频域 MLP 隐层放大倍数。
    sparsity_threshold : float, optional, default=0.01
        softshrink 稀疏阈值。
    hard_thresholding_fraction : float, optional, default=1.0
        保留频域高度方向比例。
    """

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int = 2,
        hidden_size_factor: int = 1,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_channels % num_blocks != 0:
            raise ValueError("hidden_channels must be divisible by num_blocks")
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.block_size = self.hidden_channels // self.num_blocks
        self.hidden_size_factor = max(1, int(hidden_size_factor))
        self.sparsity_threshold = float(sparsity_threshold)
        self.hard_thresholding_fraction = float(hard_thresholding_fraction)
        inner = self.block_size * self.hidden_size_factor
        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(2, self.num_blocks, self.block_size, inner))
        self.b1 = nn.Parameter(scale * torch.randn(2, self.num_blocks, inner))
        self.w2 = nn.Parameter(scale * torch.randn(2, self.num_blocks, inner, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        频域 block-diagonal token 混合。

        Parameters
        ----
        x : torch.Tensor
            输入 token 场，shape :math:`(B, H, W, C)`。

        Returns
        ----
        torch.Tensor
            输出 token 场，shape :math:`(B, H, W, C)`。
        """

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, H, W, C), got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_channels:
            raise ValueError(f"Expected {self.hidden_channels} channels, got {x.shape[-1]}")
        bsz, height, width, channels = x.shape
        dtype = x.dtype
        fft = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        fft = fft.reshape(bsz, height, width // 2 + 1, self.num_blocks, self.block_size)
        keep_modes = int(height * self.hard_thresholding_fraction)
        keep_modes = max(1, min(height, keep_modes))
        out_real = torch.zeros(
            bsz,
            height,
            width // 2 + 1,
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
            device=x.device,
            dtype=torch.float32,
        )
        out_imag = torch.zeros_like(out_real)
        real = fft[:, :keep_modes].real
        imag = fft[:, :keep_modes].imag
        out_real[:, :keep_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", imag, self.w1[1])
            + self.b1[0]
        )
        out_imag[:, :keep_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", real, self.w1[1])
            + self.b1[1]
        )
        real2 = torch.zeros_like(fft.real)
        imag2 = torch.zeros_like(fft.imag)
        real_hidden = out_real[:, :keep_modes]
        imag_hidden = out_imag[:, :keep_modes]
        real2[:, :keep_modes] = (
            torch.einsum("...bi,bio->...bo", real_hidden, self.w2[0])
            - torch.einsum("...bi,bio->...bo", imag_hidden, self.w2[1])
            + self.b2[0]
        )
        imag2[:, :keep_modes] = (
            torch.einsum("...bi,bio->...bo", imag_hidden, self.w2[0])
            + torch.einsum("...bi,bio->...bo", real_hidden, self.w2[1])
            + self.b2[1]
        )
        mixed = torch.stack([real2, imag2], dim=-1)
        mixed = F.softshrink(mixed, lambd=self.sparsity_threshold)
        mixed_complex = torch.view_as_complex(mixed.contiguous()).reshape(bsz, height, width // 2 + 1, channels)
        out = torch.fft.irfft2(mixed_complex, s=(height, width), dim=(1, 2), norm="ortho")
        return out.to(dtype=dtype)


class AFNOBlock(nn.Module):
    r"""
    FourCastNet 风格 AFNO Transformer 块。

    Parameters
    ----
    hidden_channels : int
        token 通道数。
    num_blocks : int, optional, default=2
        AFNO 通道分组数量。
    mlp_ratio : float, optional, default=4.0
        token MLP 隐层比例。
    dropout : float, optional, default=0.0
        dropout 概率。
    sparsity_threshold : float, optional, default=0.01
        AFNO softshrink 阈值。
    """

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        sparsity_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.filter = BlockDiagonalAFNO2D(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
        )
        hidden = max(hidden_channels, int(hidden_channels * float(mlp_ratio)))
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden_channels),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        执行 AFNO 块。

        Parameters
        ----
        x : torch.Tensor
            输入 token 场，shape :math:`(B, H, W, C)`。

        Returns
        ----
        torch.Tensor
            输出 token 场，shape :math:`(B, H, W, C)`。
        """

        x = x + self.filter(self.norm1(x))
        return x + self.mlp(self.norm2(x))