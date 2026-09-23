"""Proxy-conditioned variable channel graph for UrbanPiDiT V5.3.2.

该模块把 morphology-derived process proxies 聚合为样本级上下文，
用于生成变量通道图的动态邻接增量。默认初始化为近似恒等映射，
确保开启模块不会在训练初期破坏主干表示。
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .morpho_process_proxy import PROCESS_PROXY_NAMES
except ImportError:  # pragma: no cover
    from models.morpho_process_proxy import PROCESS_PROXY_NAMES


class ProxyConditionedVariableGraph(nn.Module):
    """代理条件化变量通道图。

    输入为变量通道图像 `[B, C, H, W]`，代理场为 process proxy encoder
    输出的字典。模块先把代理场做 mean/max 聚合，再生成每个样本的变量邻接
    增量，最后通过零初始化门控残差回写到输入。
    """

    def __init__(
        self,
        in_channels: int,
        *,
        proxy_names: Sequence[str] = PROCESS_PROXY_NAMES,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        symmetric: bool = True,
        use_proxy_context: bool = True,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels 必须为正数")
        if int(num_heads) <= 0:
            raise ValueError("num_heads 必须为正数")

        self.c = int(in_channels)
        self.proxy_names = tuple(str(name) for name in proxy_names)
        self.num_heads = int(num_heads)
        self.symmetric = bool(symmetric)
        self.use_proxy_context = bool(use_proxy_context)
        self.context_dim = max(len(self.proxy_names) * 2, 1)

        context_hidden = int(hidden_dim or max(16, self.context_dim * 2, self.c * self.num_heads))
        self.base_logits = nn.Parameter(torch.zeros(self.num_heads, self.c, self.c))
        self.alpha = nn.Parameter(torch.zeros(self.num_heads))
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, context_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(context_hidden, self.num_heads * self.c * self.c),
        )
        nn.init.zeros_(self.context_mlp[-1].weight)
        nn.init.zeros_(self.context_mlp[-1].bias)

        mixer_hidden = max(8, self.c * 4)
        self.norm = nn.LayerNorm(self.c)
        self.mixer = nn.Sequential(
            nn.Linear(self.c, mixer_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mixer_hidden, self.c),
        )
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def _zero_proxy_map(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(ref.size(0), 1, ref.size(2), ref.size(3))

    def _coerce_proxy_map(self, value: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if value is None:
            return self._zero_proxy_map(ref)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return self._zero_proxy_map(ref)

        proxy = value.to(device=ref.device, dtype=ref.dtype)
        if proxy.ndim == 3:
            proxy = proxy.unsqueeze(1)
        elif proxy.ndim != 4:
            return self._zero_proxy_map(ref)

        if proxy.size(0) == 1 and ref.size(0) > 1:
            proxy = proxy.expand(ref.size(0), -1, -1, -1)
        if proxy.size(0) != ref.size(0):
            return self._zero_proxy_map(ref)
        if proxy.size(1) != 1:
            proxy = proxy.mean(dim=1, keepdim=True)
        if proxy.shape[-2:] != ref.shape[-2:]:
            proxy = F.interpolate(proxy, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return torch.nan_to_num(proxy)

    def _aggregate_proxy_fields(
        self,
        proxy_fields: Optional[Mapping[str, torch.Tensor]],
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        if not self.proxy_names:
            return ref.new_zeros(ref.size(0), self.context_dim), 0

        stats = []
        available = 0
        fields = proxy_fields or {}
        for name in self.proxy_names:
            raw = fields.get(name) if isinstance(fields, Mapping) else None
            if isinstance(raw, torch.Tensor) and raw.numel() > 0:
                available += 1
            proxy = self._coerce_proxy_map(raw, ref)
            flat = proxy.flatten(2)
            stats.append(flat.mean(dim=-1))
            stats.append(flat.amax(dim=-1))

        context = torch.cat(stats, dim=1) if stats else ref.new_zeros(ref.size(0), self.context_dim)
        if context.size(1) < self.context_dim:
            pad = ref.new_zeros(context.size(0), self.context_dim - context.size(1))
            context = torch.cat([context, pad], dim=1)
        elif context.size(1) > self.context_dim:
            context = context[:, : self.context_dim]
        return torch.nan_to_num(context), available

    def _dynamic_logits(
        self,
        proxy_context: torch.Tensor,
        batch_size: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_proxy_context:
            return ref.new_zeros(batch_size, self.num_heads, self.c, self.c)
        logits = self.context_mlp(proxy_context)
        return logits.reshape(batch_size, self.num_heads, self.c, self.c)

    def forward(
        self,
        x: torch.Tensor,
        proxy_fields: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x 必须是 [B,C,H,W]，但得到 {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.c:
            raise ValueError(f"输入通道数 C={channels} 与图通道数 {self.c} 不一致")

        proxy_context, available = self._aggregate_proxy_fields(proxy_fields, x)
        delta_logits = self._dynamic_logits(proxy_context, batch, x).to(dtype=x.dtype)
        logits = self.base_logits.to(device=x.device, dtype=x.dtype).unsqueeze(0) + delta_logits
        if self.symmetric:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        adj_prob = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        eye = torch.eye(channels, device=x.device, dtype=x.dtype).view(1, 1, channels, channels)
        alpha = torch.clamp(self.alpha.to(device=x.device, dtype=x.dtype), 0.0, 1.0).view(1, self.num_heads, 1, 1)
        adj = eye + alpha * adj_prob

        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        propagated = torch.einsum("bnc,bhcd->bhnd", tokens, adj).mean(dim=1)
        delta = self.mixer(self.norm(propagated))
        out = tokens + self.gamma.to(device=x.device, dtype=x.dtype) * delta

        self.last_diagnostics = {
            "variable_graph/proxy_fields_available": x.new_tensor(float(available)),
            "variable_graph/proxy_context_norm": proxy_context.detach().norm(dim=1).mean(),
            "variable_graph/delta_logits_abs_mean": delta_logits.detach().abs().mean(),
            "variable_graph/adj_entropy": (-(adj_prob.detach() * adj_prob.detach().clamp_min(1e-8).log()).sum(dim=-1)).mean(),
            "variable_graph/alpha_mean": alpha.detach().mean(),
            "variable_graph/gamma": self.gamma.detach().to(device=x.device, dtype=x.dtype),
            "variable_graph/num_heads": x.new_tensor(float(self.num_heads)),
        }
        return out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


__all__ = ["ProxyConditionedVariableGraph"]