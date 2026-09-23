r"""
GraphCast 论文结构的 8x8 城市域适配版本。
"""

from __future__ import annotations

from typing import Sequence

import torch

from baselines.faithful.common import FourierScalarEmbedding, build_graphcast_graph
from baselines.faithful.common.grid_graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from baselines.faithful.common.typed_graph import BipartiteGraphNet, MeshGraphNet
from baselines.forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec


class FaithfulGraphCast(ForecastModelBase):
    r"""
    适配 8x8 城市域的 GraphCast 三图消息传递模型。

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
        节点隐藏维度。
    mesh_size : int, optional, default=None
        GraphCast 风格 mesh refinement 级别。None 时使用 ``mesh_height``/``mesh_width``。
    mesh_height : int, optional, default=4
        兼容旧配置的 mesh 高度。
    mesh_width : int, optional, default=4
        兼容旧配置的 mesh 宽度。
    message_passing_steps : int, optional, default=8
        mesh processor 步数。
    k_nn : int, optional, default=4
        radius query 缺失 receiver 时的 fallback kNN 数。
    radius_query_fraction_edge_length : float, optional, default=0.6
        grid2mesh 半径相对 mesh 最长边的比例。
    hidden_layers : int, optional, default=1
        图网络 MLP 隐层数量。
    dropout : float, optional, default=0.0
        dropout 概率。
    use_lead_conditioning : bool, optional, default=True
        是否向 grid node 注入 lead time 条件。
    residual : bool, optional, default=True
        是否预测残差。
    forecast_protocol : str, optional, default="official_rollout"
        预测协议。
    one_step_lead : int, optional, default=1
        自回归单步长度。
    time_step_hours : float, optional, default=6.0
        单步小时数。
    """

    spec = ForecastModelSpec(
        name="faithful_graphcast",
        description="Faithful GraphCast-style grid-mesh-grid message passing baseline",
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
        mesh_size: int | None = None,
        mesh_height: int = 4,
        mesh_width: int = 4,
        message_passing_steps: int = 8,
        k_nn: int = 4,
        radius_query_fraction_edge_length: float = 0.6,
        hidden_layers: int = 1,
        encoder_depth: int | None = None,
        decoder_depth: int | None = None,
        dropout: float = 0.0,
        use_lead_conditioning: bool = True,
        residual: bool = True,
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
        del encoder_depth, decoder_depth
        self.hidden_channels = int(hidden_channels)
        self.mesh_size = None if mesh_size is None else int(mesh_size)
        self.mesh_height = int(mesh_height)
        self.mesh_width = int(mesh_width)
        self.k_nn = int(k_nn)
        self.radius_query_fraction_edge_length = float(radius_query_fraction_edge_length)
        self.residual = bool(residual)
        self.use_lead_conditioning = bool(use_lead_conditioning)
        node_dim = self.hidden_channels
        edge_dim = self.hidden_channels
        grid_input_dim = self.feature_channels + NODE_FEATURE_DIM
        self.lead_embedding = FourierScalarEmbedding(
            node_dim,
            num_frequencies=16,
            hidden_dim=node_dim,
        )
        self.grid2mesh = BipartiteGraphNet(
            sender_input_dim=grid_input_dim,
            receiver_input_dim=NODE_FEATURE_DIM,
            edge_input_dim=EDGE_FEATURE_DIM,
            node_dim=node_dim,
            edge_dim=edge_dim,
            message_passing_steps=1,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        self.processor = MeshGraphNet(
            node_dim=node_dim,
            edge_input_dim=EDGE_FEATURE_DIM,
            edge_dim=edge_dim,
            message_passing_steps=message_passing_steps,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        self.mesh2grid = BipartiteGraphNet(
            sender_input_dim=node_dim,
            receiver_input_dim=node_dim,
            edge_input_dim=EDGE_FEATURE_DIM,
            node_dim=node_dim,
            edge_dim=edge_dim,
            output_dim=self.out_channels,
            message_passing_steps=1,
            hidden_layers=hidden_layers,
            dropout=dropout,
            embed_sender=False,
            embed_receiver=False,
        )

    def _graph(self, height: int, width: int, reference: torch.Tensor):
        return build_graphcast_graph(
            height,
            width,
            mesh_size=self.mesh_size,
            mesh_height=self.mesh_height,
            mesh_width=self.mesh_width,
            k_grid_to_mesh=self.k_nn,
            k_mesh_to_grid=self.k_nn,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _grid_features(self, view: ForecastBatchView, graph) -> torch.Tensor:
        bsz, _, height, width = view.last.shape
        dynamic_static = self.make_features(view).permute(0, 2, 3, 1).reshape(bsz, height * width, -1)
        node_features = graph.grid_node_features.unsqueeze(0).expand(bsz, -1, -1)
        return torch.cat([dynamic_static, node_features], dim=-1)

    def _lead_condition(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        if not self.use_lead_conditioning:
            return view.last.new_zeros((view.last.shape[0], 1, self.hidden_channels))
        lead = lead_step.to(device=view.last.device, dtype=view.last.dtype).flatten()
        lead = lead.expand(view.last.shape[0]) if lead.numel() == 1 else lead
        denom = float(max(self.lead_times) if self.lead_times else max(int(lead.max().item()), 1))
        return self.lead_embedding(lead / max(denom, 1.0)).unsqueeze(1)

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

        bsz, _, height, width = view.last.shape
        graph = self._graph(height, width, view.last)
        grid_features = self._grid_features(view, graph)
        mesh_features = graph.mesh_node_features.unsqueeze(0).expand(bsz, -1, -1)
        grid_latent, mesh_latent, _ = self.grid2mesh(
            grid_features,
            mesh_features,
            graph.grid2mesh_edge_features,
            graph.grid2mesh_senders,
            graph.grid2mesh_receivers,
            graph.mesh_coords.shape[0],
        )
        mesh_latent = mesh_latent + self._lead_condition(view, lead_step)
        mesh_latent, _ = self.processor(
            mesh_latent,
            graph.mesh_edge_features,
            graph.mesh_senders,
            graph.mesh_receivers,
        )
        _, grid_delta, _ = self.mesh2grid(
            mesh_latent,
            grid_latent,
            graph.mesh2grid_edge_features,
            graph.mesh2grid_senders,
            graph.mesh2grid_receivers,
            graph.grid_coords.shape[0],
        )
        delta = grid_delta.reshape(bsz, height, width, self.out_channels).permute(0, 3, 1, 2)
        return view.last + delta if self.residual else delta