r"""
FourCastNet/AFNO 论文忠实复现版本。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec, TensorDict
from baselines.faithful.fourcastnet.afno import AFNOBlock


class FaithfulFourCastNet(ForecastModelBase):
    r"""
    适配 UrbanPiDiT 数据集的 AFNO/FourCastNet 复现基线。

    Parameters
    ----
    dynamic_vars : Sequence[str]
        动态变量名。
    static_vars : Sequence[str], optional
        静态变量名。
    k : int, optional, default=1
        历史帧数量。
    lead_times : Sequence[int], optional
        提前期步数。
    static_policy : str, optional, default="same_static"
        静态信息策略。
    hidden_channels : int, optional, default=128
        token 维度。
    depth : int, optional, default=8
        AFNO 块数量。
    num_blocks : int, optional, default=2
        AFNO block-diagonal 分组数。
    mlp_ratio : float, optional, default=4.0
        token MLP 隐层比例。
    dropout : float, optional, default=0.0
        dropout 概率。
    sparsity_threshold : float, optional, default=0.01
        AFNO softshrink 阈值。
    residual : bool, optional, default=True
        是否预测相对最新场的残差。
    include_coords : bool, optional, default=True
        是否拼接坐标。
    include_hour : bool, optional, default=False
        是否拼接小时条件。
    forecast_protocol : str, optional, default="official_rollout"
        预测协议。
    one_step_lead : int, optional, default=1
        自回归单步长度。
    time_step_hours : float, optional, default=6.0
        单步小时数。
    """

    spec = ForecastModelSpec(
        name="faithful_fourcastnet",
        description="Faithful AFNO/FourCastNet baseline with block-diagonal spectral MLP",
        uses_static=True,
        trainable=True,
        family="faithful_weather_baseline",
    )

    def __init__(
        self,
        *,
        dynamic_vars: Sequence[str],
        static_vars: Sequence[str] | None = None,
        k: int = 1,
        lead_times: Sequence[int] | None = None,
        static_policy: str = "same_static",
        hidden_channels: int = 128,
        depth: int = 8,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        sparsity_threshold: float = 0.01,
        residual: bool = True,
        include_coords: bool = True,
        include_hour: bool = False,
        forecast_protocol: str = "official_rollout",
        one_step_lead: int = 1,
        time_step_hours: float = 6.0,
    ) -> None:
        super().__init__(
            dynamic_vars=dynamic_vars,
            static_vars=static_vars,
            k=k,
            lead_times=lead_times,
            static_policy=static_policy,
            forecast_protocol=forecast_protocol,
            one_step_lead=one_step_lead,
            time_step_hours=time_step_hours,
        )
        self.hidden_channels = int(hidden_channels)
        self.residual = bool(residual)
        self.include_coords = bool(include_coords)
        self.include_hour = bool(include_hour)
        extra = (2 if self.include_coords else 0) + (1 if self.include_hour else 0)
        self.patch_embed = nn.Conv2d(self.feature_channels + extra, self.hidden_channels, kernel_size=1)
        self.position_embedding = nn.Parameter(torch.zeros(1, 8, 8, self.hidden_channels))
        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    hidden_channels=self.hidden_channels,
                    num_blocks=int(num_blocks),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    sparsity_threshold=float(sparsity_threshold),
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_channels)
        self.head = nn.Linear(self.hidden_channels, self.out_channels)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _coords(self, reference: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = reference.shape
        y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(bsz, -1, -1, -1)

    def _hour_map(self, view: ForecastBatchView) -> torch.Tensor:
        bsz, _, height, width = view.last.shape
        if view.hour_of_day is None:
            hour = torch.zeros(bsz, device=view.last.device, dtype=view.last.dtype)
        else:
            hour = view.hour_of_day.to(device=view.last.device, dtype=view.last.dtype)
        return (hour.view(-1, 1, 1, 1) / 23.0).expand(bsz, 1, height, width)

    def _features(self, view: ForecastBatchView) -> torch.Tensor:
        parts = [self.make_features(view)]
        if self.include_coords:
            parts.append(self._coords(view.last))
        if self.include_hour:
            parts.append(self._hour_map(view))
        return torch.cat(parts, dim=1)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        pos = self.position_embedding
        if pos.shape[1] == height and pos.shape[2] == width:
            return pos
        pos_cf = pos.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(pos_cf, size=(height, width), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        r"""
        预测单个提前期。

        Parameters
        ----
        view : ForecastBatchView
            规范化 batch 视图。
        lead_step : torch.Tensor
            提前期步数。rollout 模式下保持接口兼容。

        Returns
        ----
        torch.Tensor
            预测场，shape :math:`(B, C, H, W)`。
        """

        del lead_step
        features = self._features(view)
        tokens = self.patch_embed(features).permute(0, 2, 3, 1)
        tokens = tokens + self._position_embedding(tokens.shape[1], tokens.shape[2])
        for block in self.blocks:
            tokens = block(tokens)
        delta = self.head(self.norm(tokens)).permute(0, 3, 1, 2)
        return view.last + delta if self.residual else delta

    def forward(self, batch: TensorDict, lead_times: Sequence[int] | None = None) -> torch.Tensor:
        return super().forward(batch, lead_times=lead_times)