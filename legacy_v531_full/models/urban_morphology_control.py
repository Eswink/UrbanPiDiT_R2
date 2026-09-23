"""城市形态控制分支。

该分支把多尺度 process proxy features 转换为逐层 token residuals。
所有输出投影 zero-init，初始时等价于不启用控制分支。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFiLM(nn.Module):
    """用扩散条件向量调制空间特征。"""

    def __init__(self, time_dim: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * channels))

    def forward(self, x: torch.Tensor, t_embed: Optional[torch.Tensor]) -> torch.Tensor:
        if t_embed is None:
            return x
        shift, scale = self.net(t_embed).chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class UrbanMorphologyControlBranch(nn.Module):
    """zero-init 城市形态控制残差分支。"""

    def __init__(
        self,
        feature_channels: Iterable[int],
        dim: int,
        depth: int,
        time_dim: Optional[int] = None,
        enabled: bool = True,
        init_gate: float = 0.0,
    ) -> None:
        super().__init__()
        channels = tuple(int(v) for v in feature_channels)
        if len(channels) < 1:
            raise ValueError("feature_channels 至少需要一个 level")
        self.feature_channels = channels
        self.dim = int(dim)
        self.depth = int(depth)
        self.enabled = bool(enabled)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        time_channels = int(time_dim or dim)

        self.proj_levels = nn.ModuleList([nn.Conv2d(ch, self.dim, kernel_size=1) for ch in channels])
        self.film = TimeFiLM(time_channels, self.dim)
        self.mix = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=self.dim),
            nn.GELU(),
        )
        self.out_heads = nn.ModuleList([nn.Conv2d(self.dim, self.dim, kernel_size=1) for _ in range(self.depth)])
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        self._zero_init_outputs()

    def _zero_init_outputs(self) -> None:
        for head in self.out_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _fuse_levels(self, level_features: List[torch.Tensor]) -> torch.Tensor:
        if not level_features:
            raise ValueError("level_features 不能为空")
        base_hw = level_features[0].shape[-2:]
        fused = None
        for feature, proj in zip(level_features, self.proj_levels):
            item = proj(feature)
            if item.shape[-2:] != base_hw:
                item = F.interpolate(item, size=base_hw, mode="bilinear", align_corners=False)
            fused = item if fused is None else fused + item
        return fused / float(min(len(level_features), len(self.proj_levels)))

    def forward(
        self,
        level_features: Optional[List[torch.Tensor]],
        t_embed: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if not self.enabled or level_features is None:
            return []
        fused = self._fuse_levels(level_features)
        fused = self.mix(self.film(fused, t_embed))
        gate = torch.tanh(self.gate)
        residuals = []
        diagnostics: Dict[str, torch.Tensor] = {"urban_control/gate": gate.detach()}
        for idx, head in enumerate(self.out_heads):
            residual = gate * head(fused).flatten(2).transpose(1, 2)
            residuals.append(residual)
            diagnostics[f"urban_control/residual_norm_{idx}"] = residual.detach().norm()
        self.last_diagnostics = diagnostics
        return residuals


__all__ = ["UrbanMorphologyControlBranch"]