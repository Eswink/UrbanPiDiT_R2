r"""
CorrDiff/SongUNet 风格小型 UNet。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetResBlock(nn.Module):
    r"""
    带条件调制的小型残差块。

    Parameters
    ----
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    cond_channels : int
        条件向量维度。
    dropout : float, optional, default=0.0
        dropout 概率。
    """

    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups_in = max(1, min(8, in_channels))
        groups_out = max(1, min(8, out_channels))
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_channels, out_channels * 2)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r"""
        执行条件残差更新。

        Parameters
        ----
        x : torch.Tensor
            输入特征，shape :math:`(B, C, H, W)`。
        cond : torch.Tensor
            条件向量，shape :math:`(B, D)`。

        Returns
        ----
        torch.Tensor
            更新后的特征。
        """

        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class ConditionalSelfAttention(nn.Module):
    r"""
    带条件调制的空间自注意力块。

    Parameters
    ----
    channels : int
        输入与输出通道数。
    cond_channels : int
        条件向量维度。
    num_heads : int, optional, default=4
        注意力头数量。
    dropout : float, optional, default=0.0
        dropout 概率。
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        heads = max(1, int(num_heads))
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={heads}")
        self.norm = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.cond_proj = nn.Linear(cond_channels, channels * 2)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=float(dropout), batch_first=True)
        self.output = nn.Linear(channels, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r"""
        执行条件自注意力更新。

        Parameters
        ----
        x : torch.Tensor
            输入特征，shape :math:`(B, C, H, W)`。
        cond : torch.Tensor
            条件向量，shape :math:`(B, D)`。

        Returns
        ----
        torch.Tensor
            更新后的特征，shape :math:`(B, C, H, W)`。
        """

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, C, H, W), got {tuple(x.shape)}")
        if cond.ndim != 2 or cond.shape[0] != x.shape[0]:
            raise ValueError(f"Expected cond shape ({x.shape[0]}, D), got {tuple(cond.shape)}")
        bsz, channels, height, width = x.shape
        h = self.norm(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        tokens = h.flatten(2).transpose(1, 2)
        tokens, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        h = self.output(tokens).transpose(1, 2).reshape(bsz, channels, height, width)
        return x + h


class SongUNetSmall(nn.Module):
    r"""
    适配 8x8 城市域的小型 SongUNet。

    Parameters
    ----
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    hidden_channels : int, optional, default=128
        基础隐藏通道数。
    cond_channels : int, optional, default=128
        条件向量维度。
    channel_mult : tuple[int, ...], optional, default=(1, 2)
        UNet 各层通道倍率。
    dropout : float, optional, default=0.0
        dropout 概率。
    use_attention : bool, optional, default=True
        是否在瓶颈层启用空间自注意力。
    num_attention_heads : int, optional, default=4
        注意力头数量。
    attention_resolutions : tuple[int, ...], optional, default=(4,)
        启用注意力的空间分辨率。
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        cond_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2),
        dropout: float = 0.0,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        attention_resolutions: tuple[int, ...] = (4,),
    ) -> None:
        super().__init__()
        mult = tuple(max(1, int(x)) for x in channel_mult)
        if len(mult) < 1:
            raise ValueError("channel_mult must not be empty")
        channels = [int(hidden_channels) * x for x in mult]
        self.attention_resolutions = {int(x) for x in attention_resolutions}
        self.input = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, channel in enumerate(channels):
            in_ch = channels[idx - 1] if idx > 0 else channels[0]
            self.down_blocks.append(UNetResBlock(in_ch, channel, cond_channels, dropout))
            if idx < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1))
        self.mid = UNetResBlock(channels[-1], channels[-1], cond_channels, dropout)
        self.mid_attention = (
            ConditionalSelfAttention(
                channels[-1],
                cond_channels,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            if use_attention
            else None
        )
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx in reversed(range(len(channels))):
            channel = channels[idx]
            if idx < len(channels) - 1:
                self.upsamples.append(nn.ConvTranspose2d(channels[idx + 1], channel, kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(UNetResBlock(channel * 2, channel, cond_channels, dropout))
        self.output_norm = nn.GroupNorm(max(1, min(8, channels[0])), channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r"""
        执行 UNet 前向。

        Parameters
        ----
        x : torch.Tensor
            输入场，shape :math:`(B, C, H, W)`。
        cond : torch.Tensor
            条件向量，shape :math:`(B, D)`。

        Returns
        ----
        torch.Tensor
            输出场，shape :math:`(B, C_{out}, H, W)`。
        """

        h = self.input(x)
        skips = []
        for idx, block in enumerate(self.down_blocks):
            h = block(h, cond)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)
        h = self.mid(h, cond)
        if self.mid_attention is not None and h.shape[-1] in self.attention_resolutions:
            h = self.mid_attention(h, cond)
        upsample_idx = 0
        for idx, block in enumerate(self.up_blocks):
            if idx > 0:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), cond)
        return self.output(F.silu(self.output_norm(h)))