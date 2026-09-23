"""过程代理条件化 AdaLN。

该模块把 morphology-derived process proxy fields 映射为逐 token 的
scale/shift 调制项，用于补充扩散时间步条件。初始 alpha 为 0，
因此不会改变既有模型行为。
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

try:
    from .morpho_process_proxy import PROCESS_PROXY_NAMES
except ImportError:  # pragma: no cover
    from models.morpho_process_proxy import PROCESS_PROXY_NAMES


class ProcessConditionedAdaLN(nn.Module):
    """将 process proxy fields 转换为 AdaLN 的逐 token 调制项。"""

    def __init__(
        self,
        dim: int,
        depth: int,
        proxy_channels: int = len(PROCESS_PROXY_NAMES),
        hidden_dim: Optional[int] = None,
        init_alpha: float = 0.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.proxy_channels = int(proxy_channels)
        self.enabled = bool(enabled)
        hidden = int(hidden_dim or max(self.dim // 2, 32))
        self.proxy_mlp = nn.Sequential(
            nn.Linear(self.proxy_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.dim),
        )
        self.alpha = nn.Parameter(torch.full((self.depth,), float(init_alpha)))
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def _stack_proxy_fields(self, proxy_fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = []
        for name in PROCESS_PROXY_NAMES:
            if name not in proxy_fields:
                raise KeyError(f"proxy_fields 缺少 {name}")
            values.append(proxy_fields[name])
        proxy = torch.cat(values, dim=1)
        if proxy.ndim != 4:
            raise ValueError(f"proxy fields 必须拼成 [B,P,H,W]，但得到 {tuple(proxy.shape)}")
        return proxy.flatten(2).transpose(1, 2)

    def shifts_scales(
        self,
        proxy_fields: Optional[Mapping[str, torch.Tensor]],
        block_idx: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.enabled or proxy_fields is None:
            return None, None
        if block_idx < 0 or block_idx >= self.depth:
            raise IndexError(f"block_idx={block_idx} 超出 depth={self.depth}")
        proxy_tokens = self._stack_proxy_fields(proxy_fields)
        shift, scale = self.proxy_mlp(proxy_tokens).chunk(2, dim=-1)
        alpha = torch.tanh(self.alpha[block_idx]).to(dtype=shift.dtype, device=shift.device)
        self.last_diagnostics = {
            f"process_adaln/alpha_p_{block_idx}": alpha.detach().abs(),
            f"process_adaln/shift_p_norm_{block_idx}": shift.detach().norm(),
            f"process_adaln/scale_p_norm_{block_idx}": scale.detach().norm(),
        }
        return alpha * shift, alpha * scale

    def forward(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        *,
        proxy_fields: Optional[Mapping[str, torch.Tensor]] = None,
        block_idx: int = 0,
    ) -> torch.Tensor:
        proxy_shift, proxy_scale = self.shifts_scales(proxy_fields, block_idx)
        if proxy_shift is None or proxy_scale is None:
            return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x * (1.0 + scale.unsqueeze(1) + proxy_scale) + shift.unsqueeze(1) + proxy_shift


__all__ = ["ProcessConditionedAdaLN"]