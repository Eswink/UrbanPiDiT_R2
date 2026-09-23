r"""
GraphCast/GenCast 风格的局部三图消息传递模块。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .grid_graph import scatter_mean


class ConditionedLayerNorm(nn.Module):
    r"""
    支持全局条件调制的 LayerNorm。

    Parameters
    ----
    hidden_dim : int
        归一化维度。
    cond_dim : int, optional, default=0
        全局条件维度。小于等于 0 时退化为普通 LayerNorm。
    """

    def __init__(self, hidden_dim: int, cond_dim: int = 0) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.norm = nn.LayerNorm(int(hidden_dim), elementwise_affine=self.cond_dim <= 0)
        self.modulation: Optional[nn.Linear]
        if self.cond_dim > 0:
            self.modulation = nn.Linear(self.cond_dim, int(hidden_dim) * 2)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            self.modulation = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        r"""
        执行条件归一化。

        Parameters
        ----
        x : torch.Tensor
            输入张量，shape :math:`(B, N, D)`。
        cond : torch.Tensor, optional
            全局条件，shape :math:`(B, C)`。

        Returns
        ----
        torch.Tensor
            归一化后的张量。
        """

        y = self.norm(x)
        if self.modulation is None:
            return y
        if cond is None:
            raise ValueError("cond must be provided when cond_dim > 0")
        if cond.ndim != 2:
            raise ValueError(f"Expected cond shape (B, C), got {tuple(cond.shape)}")
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        if x.ndim == 2:
            return y.unsqueeze(0) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        if x.ndim == 3 and cond.shape[0] == x.shape[0]:
            return y * (1.0 + scale[:, None, :]) + shift[:, None, :]
        raise ValueError(f"Expected x shape (N, D) or ({cond.shape[0]}, N, D), got {tuple(x.shape)}")


class GraphMLP(nn.Module):
    r"""
    GraphCast 风格 MLP：Linear/SiLU 堆叠后接 LayerNorm。

    Parameters
    ----
    in_dim : int
        输入维度。
    out_dim : int
        输出维度。
    hidden_dim : int
        隐层维度。
    hidden_layers : int, optional, default=1
        隐层数量。
    dropout : float, optional, default=0.0
        dropout 概率。
    cond_dim : int, optional, default=0
        全局条件维度。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = int(in_dim)
        for _ in range(max(1, int(hidden_layers))):
            layers.extend(
                [
                    nn.Linear(current_dim, int(hidden_dim)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.norm = ConditionedLayerNorm(int(out_dim), cond_dim=int(cond_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        r"""
        执行 MLP 映射。

        Parameters
        ----
        x : torch.Tensor
            输入张量。
        cond : torch.Tensor, optional
            全局条件。

        Returns
        ----
        torch.Tensor
            输出张量。
        """

        return self.norm(self.net(x), cond)


class BipartiteInteractionBlock(nn.Module):
    r"""
    二部图的一步边更新与 receiver 节点更新。

    Parameters
    ----
    node_dim : int
        节点隐层维度。
    edge_dim : int
        边隐层维度。
    hidden_dim : int
        MLP 隐层维度。
    hidden_layers : int, optional, default=1
        MLP 隐层数量。
    dropout : float, optional, default=0.0
        dropout 概率。
    cond_dim : int, optional, default=0
        全局条件维度。
    residual : bool, optional, default=True
        是否使用残差更新。
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        cond_dim: int = 0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.edge_mlp = GraphMLP(
            int(node_dim) * 2 + int(edge_dim),
            int(edge_dim),
            int(hidden_dim),
            hidden_layers=int(hidden_layers),
            dropout=float(dropout),
            cond_dim=int(cond_dim),
        )
        self.node_mlp = GraphMLP(
            int(node_dim) + int(edge_dim),
            int(node_dim),
            int(hidden_dim),
            hidden_layers=int(hidden_layers),
            dropout=float(dropout),
            cond_dim=int(cond_dim),
        )

    def forward(
        self,
        sender_nodes: torch.Tensor,
        receiver_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        receiver_count: int,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        执行一次二部图消息传递。

        Parameters
        ----
        sender_nodes : torch.Tensor
            sender 节点，shape :math:`(B, N_s, D)`。
        receiver_nodes : torch.Tensor
            receiver 节点，shape :math:`(B, N_r, D)`。
        edge_features : torch.Tensor
            边特征，shape :math:`(E, D_e)` 或 :math:`(B, E, D_e)`。
        senders : torch.Tensor
            sender 索引。
        receivers : torch.Tensor
            receiver 索引。
        receiver_count : int
            receiver 节点数。
        cond : torch.Tensor, optional
            全局条件。

        Returns
        ----
        tuple[torch.Tensor, torch.Tensor]
            更新后的 receiver 节点与边特征。
        """

        if sender_nodes.ndim != 3 or receiver_nodes.ndim != 3:
            raise ValueError("sender_nodes and receiver_nodes must be 3D")
        bsz = sender_nodes.shape[0]
        if edge_features.ndim == 2:
            edge_features = edge_features.unsqueeze(0).expand(bsz, -1, -1)
        sender_edge = sender_nodes[:, senders.long()]
        receiver_edge = receiver_nodes[:, receivers.long()]
        edge_input = torch.cat([sender_edge, receiver_edge, edge_features], dim=-1)
        edge_update = self.edge_mlp(edge_input, cond)
        edge_out = edge_features + edge_update if self.residual else edge_update
        messages = scatter_mean(edge_out, receivers.long(), receiver_count)
        node_input = torch.cat([receiver_nodes, messages], dim=-1)
        node_update = self.node_mlp(node_input, cond)
        node_out = receiver_nodes + node_update if self.residual else node_update
        return node_out, edge_out


class BipartiteGraphNet(nn.Module):
    r"""
    GraphCast/GenCast 的 grid2mesh 或 mesh2grid GNN。

    Parameters
    ----
    sender_input_dim : int
        sender 输入维度。
    receiver_input_dim : int
        receiver 输入维度。
    edge_input_dim : int
        边输入维度。
    node_dim : int
        节点隐层维度。
    edge_dim : int
        边隐层维度。
    output_dim : int, optional
        receiver 输出维度。None 时输出节点隐层。
    """

    def __init__(
        self,
        *,
        sender_input_dim: int,
        receiver_input_dim: int,
        edge_input_dim: int,
        node_dim: int,
        edge_dim: int,
        output_dim: int | None = None,
        message_passing_steps: int = 1,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        cond_dim: int = 0,
        embed_sender: bool = True,
        embed_receiver: bool = True,
    ) -> None:
        super().__init__()
        self.sender_encoder = (
            GraphMLP(sender_input_dim, node_dim, node_dim, hidden_layers, dropout, cond_dim)
            if embed_sender
            else nn.Identity()
        )
        self.receiver_encoder = (
            GraphMLP(receiver_input_dim, node_dim, node_dim, hidden_layers, dropout, cond_dim)
            if embed_receiver
            else nn.Identity()
        )
        self.edge_encoder = GraphMLP(edge_input_dim, edge_dim, edge_dim, hidden_layers, dropout, cond_dim)
        self.blocks = nn.ModuleList(
            [
                BipartiteInteractionBlock(
                    node_dim,
                    edge_dim,
                    node_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    cond_dim=cond_dim,
                )
                for _ in range(max(1, int(message_passing_steps)))
            ]
        )
        self.output_head: nn.Module
        if output_dim is None:
            self.output_head = nn.Identity()
        else:
            self.output_head = nn.Linear(int(node_dim), int(output_dim))
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        sender_nodes: torch.Tensor,
        receiver_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        receiver_count: int,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        执行二部图 GNN。

        Parameters
        ----
        sender_nodes : torch.Tensor
            sender 节点。
        receiver_nodes : torch.Tensor
            receiver 节点。
        edge_features : torch.Tensor
            原始边特征。
        senders : torch.Tensor
            sender 索引。
        receivers : torch.Tensor
            receiver 索引。
        receiver_count : int
            receiver 节点数量。
        cond : torch.Tensor, optional
            全局条件。

        Returns
        ----
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            sender 隐层、receiver 输出、边隐层。
        """

        sender_hidden = self.sender_encoder(sender_nodes, cond) if isinstance(self.sender_encoder, GraphMLP) else sender_nodes
        receiver_hidden = (
            self.receiver_encoder(receiver_nodes, cond) if isinstance(self.receiver_encoder, GraphMLP) else receiver_nodes
        )
        edge_hidden = self.edge_encoder(edge_features, cond)
        for block in self.blocks:
            receiver_hidden, edge_hidden = block(
                sender_hidden,
                receiver_hidden,
                edge_hidden,
                senders,
                receivers,
                receiver_count,
                cond,
            )
        return sender_hidden, self.output_head(receiver_hidden), edge_hidden


class MeshGraphNet(nn.Module):
    r"""
    GraphCast mesh processor GNN。

    Parameters
    ----
    node_dim : int
        节点隐层维度。
    edge_input_dim : int
        原始边特征维度。
    edge_dim : int
        边隐层维度。
    message_passing_steps : int, optional, default=8
        消息传递步数。
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_input_dim: int,
        edge_dim: int,
        message_passing_steps: int = 8,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.edge_encoder = GraphMLP(edge_input_dim, edge_dim, edge_dim, hidden_layers, dropout, cond_dim)
        self.blocks = nn.ModuleList(
            [
                BipartiteInteractionBlock(
                    node_dim,
                    edge_dim,
                    node_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    cond_dim=cond_dim,
                )
                for _ in range(max(1, int(message_passing_steps)))
            ]
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        执行 mesh graph 消息传递。

        Parameters
        ----
        nodes : torch.Tensor
            mesh 节点，shape :math:`(B, N, D)`。
        edge_features : torch.Tensor
            原始边特征。
        senders : torch.Tensor
            sender 索引。
        receivers : torch.Tensor
            receiver 索引。
        cond : torch.Tensor, optional
            全局条件。

        Returns
        ----
        tuple[torch.Tensor, torch.Tensor]
            更新后的 mesh 节点与边隐层。
        """

        edge_hidden = self.edge_encoder(edge_features, cond)
        for block in self.blocks:
            nodes, edge_hidden = block(nodes, nodes, edge_hidden, senders, receivers, nodes.shape[1], cond)
        return nodes, edge_hidden


__all__ = [
    "BipartiteGraphNet",
    "BipartiteInteractionBlock",
    "ConditionedLayerNorm",
    "GraphMLP",
    "MeshGraphNet",
]