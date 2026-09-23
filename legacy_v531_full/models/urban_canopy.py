"""城市冠层耦合兼容封装。

`UrbanCanopyCoupling` 保留旧配置入口，内部委托 `MicroMetCouplingOperator`。
旧的 `use_urban_canopy_coupling` 路径仍可工作，新实验应优先使用
`ablation.use_micromet_coupling` 与 `micromet_coupling` 配置段。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

try:
    from .micromet_coupling import MicroMetCouplingOperator
except ImportError:  # pragma: no cover
    from models.micromet_coupling import MicroMetCouplingOperator


class UrbanCanopyCoupling(nn.Module):
    """旧城市冠层模块的薄壁兼容封装。"""

    def __init__(
        self,
        in_channels: int,
        static_channels: int = 5,
        dynamic_vars: Optional[Iterable[str]] = None,
        hidden_channels: int = 32,
        dropout: float = 0.0,
        use_wind_drag: bool = True,
        use_thermal: bool = True,
        use_moisture: bool = True,
        use_cross_var: bool = True,
        max_wind_drag: float = 0.20,
        max_branch_scale: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        graph_cfg = kwargs.get("graph_cfg", kwargs.get("morphology_graph_cfg", None))
        self.operator = MicroMetCouplingOperator(
            in_channels=in_channels,
            static_channels=static_channels,
            dynamic_vars=dynamic_vars,
            hidden_channels=hidden_channels,
            dropout=dropout,
            enabled=True,
            mode=str(kwargs.get("mode", "pre")),
            enable_momentum_drag=use_wind_drag,
            enable_thermal_storage=use_thermal,
            enable_moisture_evaporation=use_moisture,
            enable_ventilation_mixing=use_cross_var,
            max_drag=max_wind_drag,
            max_branch_scale=max_branch_scale,
            graph_cfg=graph_cfg,
        )
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None,
        *,
        static_raw: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        hour_of_day: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """应用兼容冠层耦合。

        Args:
            x: 当前动态场 `[B,C,H,W]`。
            static_feat: 旧接口静态字段；会作为 `static_raw` 传给新算子。
            static_raw: 新接口原始静态字段。
            static_cont: 连续静态字段。
            static_cat: 类别静态字段。
            hour_of_day: 可选当前上下文小时，未提供时日周期分支跳过。
            wind_uv: 可选历史风场 `[B,2,H,W]`。
            return_diagnostics: 是否返回诊断信息。

        Returns:
            默认返回耦合后的张量；开启诊断时返回 `(tensor, diagnostics)`。
        """

        raw = static_raw if static_raw is not None else static_feat
        out, diagnostics = self.operator(
            x,
            static_raw=raw,
            static_cont=static_cont,
            static_cat=static_cat,
            hour_of_day=hour_of_day,
            wind_uv=wind_uv,
        )
        self.last_diagnostics = diagnostics
        if return_diagnostics:
            return out, diagnostics
        return out


__all__ = ["UrbanCanopyCoupling"]