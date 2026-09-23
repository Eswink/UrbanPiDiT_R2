r"""
8x8 城市域的 GraphCast/GenCast 三图结构构建工具。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 5


@dataclass(frozen=True)
class GraphCastGraph:
    r"""
    GraphCast/GenCast 风格三图结构。

    Parameters
    ----
    grid_coords : torch.Tensor
        grid node 坐标，shape :math:`(N_g, 2)`。
    mesh_coords : torch.Tensor
        mesh node 坐标，shape :math:`(N_m, 2)`。
    mesh_faces : torch.Tensor
        triangular mesh 面片，shape :math:`(F, 3)`。
    grid_node_features : torch.Tensor
        grid 结构节点特征，shape :math:`(N_g, D_n)`。
    mesh_node_features : torch.Tensor
        mesh 结构节点特征，shape :math:`(N_m, D_n)`。
    grid2mesh_senders : torch.Tensor
        grid2mesh sender 索引。
    grid2mesh_receivers : torch.Tensor
        grid2mesh receiver 索引。
    grid2mesh_edge_features : torch.Tensor
        grid2mesh 边特征。
    mesh_senders : torch.Tensor
        mesh graph sender 索引。
    mesh_receivers : torch.Tensor
        mesh graph receiver 索引。
    mesh_edge_features : torch.Tensor
        mesh graph 边特征。
    mesh2grid_senders : torch.Tensor
        mesh2grid sender 索引。
    mesh2grid_receivers : torch.Tensor
        mesh2grid receiver 索引。
    mesh2grid_edge_features : torch.Tensor
        mesh2grid 边特征。
    """

    grid_coords: torch.Tensor
    mesh_coords: torch.Tensor
    mesh_faces: torch.Tensor
    grid_node_features: torch.Tensor
    mesh_node_features: torch.Tensor
    grid2mesh_senders: torch.Tensor
    grid2mesh_receivers: torch.Tensor
    grid2mesh_edge_features: torch.Tensor
    mesh_senders: torch.Tensor
    mesh_receivers: torch.Tensor
    mesh_edge_features: torch.Tensor
    mesh2grid_senders: torch.Tensor
    mesh2grid_receivers: torch.Tensor
    mesh2grid_edge_features: torch.Tensor


def grid_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    生成归一化网格坐标。

    Parameters
    ----
    height : int
        网格高度。
    width : int
        网格宽度。
    device : torch.device or str, optional
        输出设备。
    dtype : torch.dtype, optional, default=torch.float32
        输出类型。

    Returns
    ----
    torch.Tensor
        shape :math:`(H W, 2)` 的 ``(lat, lon)`` 局部坐标。
    """

    y = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy, xx], dim=-1).reshape(-1, 2)


def spatial_node_features(coords: torch.Tensor) -> torch.Tensor:
    r"""
    生成 GraphCast 风格节点结构特征。

    Parameters
    ----
    coords : torch.Tensor
        局部经纬度坐标，shape :math:`(N, 2)`。

    Returns
    ----
    torch.Tensor
        节点结构特征，shape :math:`(N, 4)`。
    """

    lat = coords[:, :1]
    lon = coords[:, 1:2]
    return torch.cat([lat, lon, torch.sin(torch.pi * lat), torch.cos(torch.pi * lon)], dim=-1)


def _edge_features(source: torch.Tensor, target: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
    delta = target[receivers] - source[senders]
    dist = delta.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    unit = delta / dist
    return torch.cat([delta, dist, unit], dim=-1)


def knn_edges(
    source: torch.Tensor,
    target: torch.Tensor,
    k: int,
    *,
    exclude_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    基于空间距离构建 kNN 二部边。

    Parameters
    ----
    source : torch.Tensor
        sender 坐标，shape :math:`(N_s, 2)`。
    target : torch.Tensor
        receiver 坐标，shape :math:`(N_r, 2)`。
    k : int
        每个 receiver 连接的 sender 数。
    exclude_self : bool, optional, default=False
        当 source/target 同集合时是否排除自环。

    Returns
    ----
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``senders``、``receivers`` 与边特征。
    """

    distances = torch.cdist(target, source)
    if exclude_self and source.shape[0] == target.shape[0]:
        eye = torch.eye(source.shape[0], device=source.device, dtype=source.dtype)
        distances = distances + eye * 1e6
    k = min(max(1, int(k)), source.shape[0])
    nearest = torch.topk(distances, k=k, largest=False, dim=1).indices
    receivers = torch.arange(target.shape[0], device=target.device).repeat_interleave(k)
    senders = nearest.reshape(-1)
    return senders.long(), receivers.long(), _edge_features(source, target, senders, receivers)


def scatter_mean(values: torch.Tensor, receivers: torch.Tensor, dim_size: int) -> torch.Tensor:
    r"""
    对边消息按 receiver 做 mean 聚合。

    Parameters
    ----
    values : torch.Tensor
        边消息，shape :math:`(B, E, C)`。
    receivers : torch.Tensor
        receiver 索引，shape :math:`(E,)`。
    dim_size : int
        receiver node 数。

    Returns
    ----
    torch.Tensor
        聚合后的 node 消息，shape :math:`(B, N, C)`。
    """

    if values.ndim != 3:
        raise ValueError(f"Expected values shape (B, E, C), got {tuple(values.shape)}")
    bsz, _, channels = values.shape
    out = values.new_zeros((bsz, int(dim_size), channels))
    out.index_add_(1, receivers.long(), values)
    counts = values.new_zeros((int(dim_size), 1))
    counts.index_add_(0, receivers.long(), values.new_ones((receivers.numel(), 1)))
    return out / counts.clamp_min(1.0).view(1, int(dim_size), 1)


def _mesh_resolution(mesh_size: int | None, mesh_height: int | None, mesh_width: int | None) -> tuple[int, int]:
    if mesh_size is not None:
        resolution = max(2, 2 ** max(0, int(mesh_size)) + 1)
        return resolution, resolution
    return max(2, int(mesh_height or 4)), max(2, int(mesh_width or 4))


def _triangular_faces(height: int, width: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    faces: list[list[int]] = []
    for row in range(int(height) - 1):
        for col in range(int(width) - 1):
            tl = row * int(width) + col
            tr = tl + 1
            bl = (row + 1) * int(width) + col
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([br, tr, bl])
    return torch.as_tensor(faces, dtype=torch.long, device=device)


def _mesh_edges_from_faces(faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge_pairs: set[tuple[int, int]] = set()
    for face in faces.detach().cpu().tolist():
        tri_edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        for source, target in tri_edges:
            edge_pairs.add((int(source), int(target)))
            edge_pairs.add((int(target), int(source)))
    ordered = sorted(edge_pairs)
    edge_tensor = torch.as_tensor(ordered, dtype=torch.long, device=faces.device)
    return edge_tensor[:, 0], edge_tensor[:, 1]


def _radius_edges(
    source: torch.Tensor,
    target: torch.Tensor,
    radius: float,
    *,
    fallback_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.cdist(target, source)
    receiver_idx, sender_idx = torch.nonzero(distances <= float(radius), as_tuple=True)
    missing = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    if receiver_idx.numel() > 0:
        missing[receiver_idx.long()] = False
    if missing.any():
        miss = torch.nonzero(missing, as_tuple=False).flatten()
        nearest = torch.topk(distances[miss], k=min(max(1, int(fallback_k)), source.shape[0]), largest=False, dim=1).indices
        receiver_extra = miss.repeat_interleave(nearest.shape[1])
        sender_extra = nearest.reshape(-1)
        receiver_idx = torch.cat([receiver_idx, receiver_extra], dim=0)
        sender_idx = torch.cat([sender_idx, sender_extra], dim=0)
    return sender_idx.long(), receiver_idx.long(), _edge_features(source, target, sender_idx, receiver_idx)


def _mesh2grid_triangle_edges(
    grid: torch.Tensor,
    mesh_height: int,
    mesh_width: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    senders: list[int] = []
    receivers: list[int] = []
    y_step = 2.0 / max(1, int(mesh_height) - 1)
    x_step = 2.0 / max(1, int(mesh_width) - 1)
    for idx, coord in enumerate(grid.detach().cpu().tolist()):
        y_frac = max(0.0, min((float(coord[0]) + 1.0) / y_step, float(mesh_height - 1)))
        x_frac = max(0.0, min((float(coord[1]) + 1.0) / x_step, float(mesh_width - 1)))
        row = min(int(y_frac), int(mesh_height) - 2)
        col = min(int(x_frac), int(mesh_width) - 2)
        local_y = y_frac - row
        local_x = x_frac - col
        tl = row * int(mesh_width) + col
        tr = tl + 1
        bl = (row + 1) * int(mesh_width) + col
        br = bl + 1
        vertices = [tl, bl, tr] if local_y + local_x <= 1.0 else [br, tr, bl]
        senders.extend(vertices)
        receivers.extend([idx, idx, idx])
    return (
        torch.as_tensor(senders, dtype=torch.long, device=device),
        torch.as_tensor(receivers, dtype=torch.long, device=device),
    )


def build_graphcast_graph(
    height: int = 8,
    width: int = 8,
    *,
    mesh_size: int | None = None,
    mesh_height: int | None = 4,
    mesh_width: int | None = 4,
    k_grid_to_mesh: int = 4,
    k_mesh_to_grid: int = 4,
    mesh_k: int | None = None,
    radius_query_fraction_edge_length: float = 0.6,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GraphCastGraph:
    r"""
    构建城市域的 GraphCast/GenCast 三图。

    Parameters
    ----
    height : int, optional, default=8
        grid 高度。
    width : int, optional, default=8
        grid 宽度。
    mesh_size : int, optional
        GraphCast 风格 mesh refinement 级别。本地适配为 ``2 ** mesh_size + 1`` 分辨率。
    mesh_height : int, optional, default=4
        mesh 高度。仅当 ``mesh_size`` 为 None 时使用。
    mesh_width : int, optional, default=4
        mesh 宽度。仅当 ``mesh_size`` 为 None 时使用。
    k_grid_to_mesh : int, optional, default=4
        radius query 缺失 receiver 时的 fallback kNN 数。
    k_mesh_to_grid : int, optional, default=4
        兼容旧配置；mesh2grid 默认使用三角形包含边。
    mesh_k : int, optional, default=None
        兼容旧配置；mesh 内部默认使用三角面片边。
    radius_query_fraction_edge_length : float, optional, default=0.6
        grid2mesh 半径相对 mesh 最长边的比例。
    device : torch.device or str, optional
        输出设备。
    dtype : torch.dtype, optional, default=torch.float32
        输出类型。

    Returns
    ----
    GraphCastGraph
        三图结构。
    """

    del k_mesh_to_grid, mesh_k
    mesh_h, mesh_w = _mesh_resolution(mesh_size, mesh_height, mesh_width)
    grid = grid_coordinates(height, width, device=device, dtype=dtype)
    mesh = grid_coordinates(mesh_h, mesh_w, device=device, dtype=dtype)
    faces = _triangular_faces(mesh_h, mesh_w, device=device)
    mesh_s, mesh_r = _mesh_edges_from_faces(faces)
    mesh_e = _edge_features(mesh, mesh, mesh_s, mesh_r)
    max_edge = mesh_e[:, 2].max().clamp_min(1e-6)
    radius = float(radius_query_fraction_edge_length) * float(max_edge.detach().cpu())
    g2m_s, g2m_r, g2m_e = _radius_edges(grid, mesh, radius, fallback_k=k_grid_to_mesh)
    m2g_s, m2g_r = _mesh2grid_triangle_edges(grid, mesh_h, mesh_w, device=device)
    m2g_e = _edge_features(mesh, grid, m2g_s, m2g_r)
    return GraphCastGraph(
        grid_coords=grid,
        mesh_coords=mesh,
        mesh_faces=faces,
        grid_node_features=spatial_node_features(grid),
        mesh_node_features=spatial_node_features(mesh),
        grid2mesh_senders=g2m_s,
        grid2mesh_receivers=g2m_r,
        grid2mesh_edge_features=g2m_e,
        mesh_senders=mesh_s,
        mesh_receivers=mesh_r,
        mesh_edge_features=mesh_e,
        mesh2grid_senders=m2g_s,
        mesh2grid_receivers=m2g_r,
        mesh2grid_edge_features=m2g_e,
    )


__all__ = [
    "EDGE_FEATURE_DIM",
    "NODE_FEATURE_DIM",
    "GraphCastGraph",
    "build_graphcast_graph",
    "grid_coordinates",
    "knn_edges",
    "scatter_mean",
    "spatial_node_features",
]