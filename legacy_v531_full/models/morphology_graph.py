"""城市形态与风向感知空间图。

该模块把城市网格关系拆成三类可解释来源：
- geographic：网格位置邻近性；
- continuous morphology：连续静态形态字段相似性；
- categorical morphology：类别静态字段一致性。

风向感知图只使用历史风场和静态粗糙度代理，不接触未来真值。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


Diagnostics = Dict[str, torch.Tensor]


def _grid_coordinates(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)


def _flatten_maps(value: Optional[torch.Tensor], *, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if value is None or value.numel() == 0:
        return None
    if value.ndim != 4:
        raise ValueError(f"静态字段必须是 [B,C,H,W]，但得到 {tuple(value.shape)}")
    return value.to(dtype=dtype).flatten(2).transpose(1, 2)


def _coerce_static_maps(
    *,
    static_feat: Optional[torch.Tensor],
    static_cont: Optional[torch.Tensor],
    static_cat: Optional[torch.Tensor],
    categorical_indices: Tuple[int, ...],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if static_feat is None or static_feat.numel() == 0:
        return static_cont, static_cat
    if static_cont is not None or static_cat is not None:
        return static_cont, static_cat
    if not categorical_indices:
        return static_feat, None

    channels = int(static_feat.size(1))
    cat_idx = [idx for idx in categorical_indices if 0 <= int(idx) < channels]
    cont_idx = [idx for idx in range(channels) if idx not in set(cat_idx)]
    cont = static_feat[:, cont_idx] if cont_idx else None
    cat = static_feat[:, cat_idx] if cat_idx else None
    return cont, cat


def _normalize_continuous(flat: torch.Tensor) -> torch.Tensor:
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (flat - mean) / std


def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _apply_knn(distance: torch.Tensor, weights: torch.Tensor, *, k: int, use_self_loop: bool) -> torch.Tensor:
    nodes = int(distance.size(-1))
    if nodes <= 0:
        return weights

    eye = torch.eye(nodes, device=weights.device, dtype=torch.bool).unsqueeze(0)
    rank_distance = distance.masked_fill(eye, float("inf"))
    rank_weights = weights.masked_fill(eye, 0.0)

    if k <= 0 or k >= nodes - 1:
        out = rank_weights.clone()
    else:
        keep = min(nodes - 1, int(k))
        _, indices = torch.topk(rank_distance, k=keep, dim=-1, largest=False)
        out = torch.zeros_like(weights)
        out.scatter_(-1, indices, rank_weights.gather(-1, indices))

    if use_self_loop:
        out = torch.maximum(out, eye.to(dtype=weights.dtype).expand_as(out))
    return _row_normalize(out)


def _edge_mask(adj: torch.Tensor) -> torch.Tensor:
    return adj > 0.0


def _graph_entropy(adj: torch.Tensor) -> torch.Tensor:
    safe = adj.clamp_min(1e-12)
    return -(safe * safe.log()).sum(dim=-1).mean()


def _same_category_ratio(adj: torch.Tensor, cat_flat: Optional[torch.Tensor]) -> torch.Tensor:
    if cat_flat is None or cat_flat.numel() == 0:
        return adj.new_tensor(0.0)
    cat = cat_flat.long()
    same = (cat.unsqueeze(2) == cat.unsqueeze(1)).all(dim=-1)
    mask = _edge_mask(adj)
    denom = mask.float().sum().clamp_min(1.0)
    return (same & mask).float().sum() / denom


def _edge_weight_mean(adj: torch.Tensor) -> torch.Tensor:
    mask = _edge_mask(adj)
    denom = mask.float().sum().clamp_min(1.0)
    return adj.masked_select(mask).sum() / denom


def _edge_weight_std(adj: torch.Tensor) -> torch.Tensor:
    mask = _edge_mask(adj)
    values = adj.masked_select(mask)
    if values.numel() == 0:
        return adj.new_tensor(0.0)
    return values.std(unbiased=False)


def _masked_edge_mean(value: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    mask = _edge_mask(adj)
    denom = mask.to(dtype=value.dtype).sum().clamp_min(1.0)
    return value.to(device=adj.device, dtype=adj.dtype).masked_select(mask).sum() / denom


def _directional_edge_weight_mean(adj: torch.Tensor, alignment: torch.Tensor, *, downwind: bool) -> torch.Tensor:
    direction_mask = alignment > 0.0 if downwind else alignment < 0.0
    mask = _edge_mask(adj) & direction_mask
    denom = mask.float().sum().clamp_min(1.0)
    return adj.masked_select(mask).sum() / denom


def _tensor_hash(parts: Iterable[torch.Tensor], *, prefix: str) -> str:
    h = hashlib.sha1(prefix.encode("utf-8"))
    for tensor in parts:
        t = tensor.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _add_process_graph_aliases(diagnostics: Diagnostics, keys: Iterable[str]) -> None:
    for key in keys:
        if key in diagnostics:
            diagnostics[f"process_graph/{key}"] = diagnostics[key]


PROCESS_GRAPH_RELATION_KEYS = (
    "geographic_relation_mean",
    "morphology_relation_mean",
    "landcover_relation_mean",
    "wind_alignment_mean",
    "roughness_blocking_mean",
    "heat_storage_relation_mean",
    "ventilation_corridor_mean",
    "relation_count",
    "same_landcover_edge_ratio",
    "adjacency_entropy",
)


class HeterogeneousMorphologyGraph(nn.Module):
    """基于地理距离、连续形态相似和类别一致性的空间图。"""

    def __init__(
        self,
        dim: int,
        k: int = 8,
        morphology_sigma: float = 1.0,
        spatial_sigma: float = 1.0,
        dropout: float = 0.0,
        relation_count: int = 3,
        use_self_loop: bool = True,
        alpha_geo: float = 1.0,
        alpha_cont: float = 1.0,
        alpha_cat: float = 1.0,
        graph_temperature: float = 1.0,
        categorical_indices: Optional[Iterable[int]] = None,
        enable_cache: bool = True,
        cache_max_nodes: int = 256,
    ) -> None:
        super().__init__()
        del relation_count
        self.dim = int(dim)
        self.k = int(k)
        self.morphology_sigma = float(max(morphology_sigma, 1e-6))
        self.spatial_sigma = float(max(spatial_sigma, 1e-6))
        self.use_self_loop = bool(use_self_loop)
        self.alpha_geo = float(alpha_geo)
        self.alpha_cont = float(alpha_cont)
        self.alpha_cat = float(alpha_cat)
        self.graph_temperature = float(max(graph_temperature, 1e-6))
        self.categorical_indices = tuple(int(x) for x in (categorical_indices or ()))
        self.enable_cache = bool(enable_cache)
        self.cache_max_nodes = int(cache_max_nodes)
        self._adj_cache: dict[str, tuple[torch.Tensor, Diagnostics]] = {}

        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim * 4, self.dim),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.last_diagnostics: Diagnostics = {}

    def _cache_key(
        self,
        *,
        static_cont: Optional[torch.Tensor],
        static_cat: Optional[torch.Tensor],
        spatial_hw: Tuple[int, int],
        static_hash: Optional[str],
    ) -> Optional[str]:
        if not self.enable_cache:
            return None
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        nodes = height * width
        if nodes > self.cache_max_nodes:
            return None
        if static_hash is not None:
            return f"{static_hash}|{height}x{width}|{self.k}|{self.alpha_geo}|{self.alpha_cont}|{self.alpha_cat}"
        parts = [x for x in (static_cont, static_cat) if x is not None and x.numel() > 0]
        if len(parts) == 0 or parts[0].size(0) != 1:
            return None
        prefix = f"{height}x{width}|{self.k}|{self.alpha_geo}|{self.alpha_cont}|{self.alpha_cat}|{self.spatial_sigma}|{self.morphology_sigma}"
        return _tensor_hash(parts, prefix=prefix)

    def _distance_components(
        self,
        *,
        static_cont: Optional[torch.Tensor],
        static_cat: Optional[torch.Tensor],
        spatial_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        nodes = height * width

        cont_flat = _flatten_maps(static_cont, dtype=dtype)
        cat_flat = _flatten_maps(static_cat, dtype=dtype)
        batch = 1
        if cont_flat is not None:
            batch = int(cont_flat.size(0))
        elif cat_flat is not None:
            batch = int(cat_flat.size(0))

        coords = _grid_coordinates(height, width, device=device, dtype=dtype)
        geo_dist = torch.cdist(coords, coords, p=2.0) / (2.0 * (2.0 ** 0.5))
        geo_dist = geo_dist.unsqueeze(0).expand(batch, -1, -1)

        if cont_flat is None or cont_flat.numel() == 0:
            cont_dist = torch.zeros(batch, nodes, nodes, device=device, dtype=dtype)
        else:
            cont_norm = _normalize_continuous(cont_flat)
            cont_dist = torch.cdist(cont_norm, cont_norm, p=2.0) / max(cont_norm.size(-1), 1) ** 0.5
            cont_dist = cont_dist / self.morphology_sigma

        if cat_flat is None or cat_flat.numel() == 0:
            cat_dist = torch.zeros(batch, nodes, nodes, device=device, dtype=dtype)
            cat_for_diag = None
        else:
            cat_long = cat_flat.long()
            cat_dist = (cat_long.unsqueeze(2) != cat_long.unsqueeze(1)).float().mean(dim=-1).to(dtype=dtype)
            cat_for_diag = cat_long

        geo_dist = geo_dist / self.spatial_sigma
        return geo_dist, cont_dist, cat_dist, cat_for_diag

    def build_adjacency(
        self,
        static_feat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
        wind_uv: Optional[torch.Tensor] = None,
        *,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        static_hash: Optional[str] = None,
    ) -> tuple[torch.Tensor, Diagnostics]:
        """构建邻接矩阵和图诊断。

        Args:
            static_feat: 兼容旧接口的静态字段 `[B,S,H,W]`。
            spatial_hw: 图空间尺寸。
            wind_uv: 兼容旧接口；基础形态图不使用。
            static_cont: 连续静态字段。
            static_cat: 类别静态字段。
            static_hash: 外部提供的缓存键。

        Returns:
            `(A, diagnostics)`，其中 `A` 为 `[B,N,N]` 行归一化邻接。
        """

        del wind_uv
        static_cont, static_cat = _coerce_static_maps(
            static_feat=static_feat,
            static_cont=static_cont,
            static_cat=static_cat,
            categorical_indices=self.categorical_indices,
        )
        if spatial_hw is None:
            source = static_cont if static_cont is not None else static_cat
            if source is None:
                raise ValueError("缺少 spatial_hw，且无法从静态字段推断空间尺寸")
            spatial_hw = (int(source.shape[-2]), int(source.shape[-1]))
        source = static_cont if static_cont is not None else static_cat
        if source is None:
            device = self.gamma.device
            dtype = self.gamma.dtype
        else:
            device = source.device
            dtype = source.dtype if source.is_floating_point() else self.gamma.dtype

        cache_key = self._cache_key(
            static_cont=static_cont,
            static_cat=static_cat,
            spatial_hw=spatial_hw,
            static_hash=static_hash,
        )
        if cache_key is not None and cache_key in self._adj_cache:
            cached_adj, cached_diag = self._adj_cache[cache_key]
            adj = cached_adj.to(device=device, dtype=dtype)
            diagnostics = {key: value.to(device=device, dtype=dtype) for key, value in cached_diag.items()}
            diagnostics["cache_hit"] = adj.new_tensor(1.0)
            return adj, diagnostics

        geo_dist, cont_dist, cat_dist, cat_for_diag = self._distance_components(
            static_cont=static_cont,
            static_cat=static_cat,
            spatial_hw=spatial_hw,
            device=device,
            dtype=dtype,
        )
        total_dist = self.alpha_geo * geo_dist + self.alpha_cont * cont_dist + self.alpha_cat * cat_dist
        weights = torch.exp(-total_dist / self.graph_temperature)
        adj = _apply_knn(total_dist, weights, k=self.k, use_self_loop=self.use_self_loop)

        cont_channels = 0 if static_cont is None else int(static_cont.size(1))
        cat_channels = 0 if static_cat is None else int(static_cat.size(1))
        same_landcover_edge_ratio = _same_category_ratio(adj, cat_for_diag)
        geographic_relation_mean = _masked_edge_mean(torch.exp(-geo_dist / self.graph_temperature), adj)
        morphology_relation_mean = _masked_edge_mean(torch.exp(-cont_dist / self.graph_temperature), adj)
        landcover_relation_mean = _masked_edge_mean((1.0 - cat_dist).clamp(0.0, 1.0), adj)
        adjacency_entropy = _graph_entropy(adj)
        relation_count = 1.0 + float(cont_channels > 0) + float(cat_channels > 0)
        diagnostics: Diagnostics = {
            "mean_degree": _edge_mask(adj).float().sum(dim=-1).mean(),
            "edge_weight_mean": _edge_weight_mean(adj),
            "edge_weight_std": _edge_weight_std(adj),
            "same_category_edge_ratio": same_landcover_edge_ratio,
            "same_landcover_edge_ratio": same_landcover_edge_ratio,
            "geo_distance_mean": _masked_edge_mean(geo_dist, adj),
            "cont_distance_mean": _masked_edge_mean(cont_dist, adj),
            "categorical_mismatch_mean": _masked_edge_mean(cat_dist, adj),
            "graph_entropy": adjacency_entropy,
            "adjacency_entropy": adjacency_entropy,
            "geographic_relation_mean": geographic_relation_mean,
            "morphology_relation_mean": morphology_relation_mean,
            "landcover_relation_mean": landcover_relation_mean,
            "wind_alignment_mean": adj.new_tensor(0.0),
            "heat_storage_relation_mean": morphology_relation_mean,
            "ventilation_corridor_mean": geographic_relation_mean,
            "relation_count": adj.new_tensor(relation_count),
            "continuous_channel_count": adj.new_tensor(float(cont_channels)),
            "categorical_channel_count": adj.new_tensor(float(cat_channels)),
            "static_schema_fallback": adj.new_tensor(float(static_feat is not None and cont_channels + cat_channels > 0)),
            "effective_topk": adj.new_tensor(float(self.k)),
            "cache_hit": adj.new_tensor(0.0),
        }
        _add_process_graph_aliases(diagnostics, PROCESS_GRAPH_RELATION_KEYS)
        if cache_key is not None:
            self._adj_cache[cache_key] = (
                adj.detach().cpu(),
                {key: value.detach().cpu() for key, value in diagnostics.items()},
            )
        return adj, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
        wind_uv: Optional[torch.Tensor] = None,
        *,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Diagnostics]:
        batch, nodes, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"token dim={dim} 与图模块 dim={self.dim} 不一致")
        if spatial_hw is None:
            raise ValueError("forward 需要显式提供 spatial_hw")
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if nodes != height * width:
            raise ValueError(f"token 数 N={nodes} 与 spatial_hw={spatial_hw} 不匹配")

        adj, diagnostics = self.build_adjacency(
            static_feat=static_feat,
            spatial_hw=(height, width),
            wind_uv=wind_uv,
            static_cont=static_cont,
            static_cat=static_cat,
        )
        if adj.size(0) != batch:
            if adj.size(0) == 1:
                adj = adj.expand(batch, -1, -1)
            else:
                raise ValueError(f"adj batch={adj.size(0)} 与 token batch={batch} 不一致")

        msg = torch.einsum("bnm,bmd->bnd", adj.to(dtype=x.dtype, device=x.device), x)
        delta = self.mlp(self.norm(msg))
        out = x + self.gamma.to(dtype=x.dtype, device=x.device) * delta
        self.last_diagnostics = diagnostics
        if return_diagnostics:
            return out, diagnostics
        return out


class WindAwareMorphologyGraph(HeterogeneousMorphologyGraph):
    """在基础形态图上加入历史风向与粗糙度阻塞约束。"""

    def __init__(
        self,
        dim: int,
        k: int = 8,
        morphology_sigma: float = 1.0,
        spatial_sigma: float = 1.0,
        wind_temperature: float = 0.25,
        dropout: float = 0.0,
        use_self_loop: bool = True,
        wind_strength: float = 1.0,
        wind_speed_scale: float = 1.0,
        calm_wind_threshold: float = 1e-4,
        roughness_blocking_strength: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dim=dim,
            k=k,
            morphology_sigma=morphology_sigma,
            spatial_sigma=spatial_sigma,
            dropout=dropout,
            use_self_loop=use_self_loop,
            **kwargs,
        )
        self.wind_temperature = float(max(wind_temperature, 1e-6))
        self.wind_strength = float(max(wind_strength, 0.0))
        self.wind_speed_scale = float(max(wind_speed_scale, 1e-6))
        self.calm_wind_threshold = float(max(calm_wind_threshold, 0.0))
        self.roughness_blocking_strength = float(max(roughness_blocking_strength, 0.0))

    def _coerce_wind(self, wind_uv: torch.Tensor, *, batch: int, spatial_hw: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if wind_uv.ndim != 4 or wind_uv.size(1) != 2:
            raise ValueError(f"wind_uv 必须是 [B,2,H,W]，但得到 {tuple(wind_uv.shape)}")
        wind = wind_uv.to(dtype=dtype)
        if wind.shape[-2:] != (height, width):
            raise ValueError(f"wind_uv 空间尺寸 {tuple(wind.shape[-2:])} 与 {spatial_hw} 不一致")
        if wind.size(0) != batch:
            if wind.size(0) == 1:
                wind = wind.expand(batch, -1, -1, -1)
            else:
                raise ValueError(f"wind_uv batch={wind.size(0)} 与图 batch={batch} 不一致")
        return wind.flatten(2).transpose(1, 2)

    def _coerce_roughness(
        self,
        roughness_proxy: Optional[torch.Tensor],
        static_cont: Optional[torch.Tensor],
        *,
        batch: int,
        spatial_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        if roughness_proxy is None or roughness_proxy.numel() == 0:
            if static_cont is None or static_cont.numel() == 0:
                return torch.zeros(batch, height * width, device=device, dtype=dtype)
            roughness_proxy = static_cont[:, :1]
        if roughness_proxy.ndim == 3:
            roughness_proxy = roughness_proxy.unsqueeze(1)
        if roughness_proxy.ndim != 4:
            raise ValueError(f"roughness_proxy 必须是 [B,1,H,W] 或 [B,H,W]，但得到 {tuple(roughness_proxy.shape)}")
        rough = roughness_proxy.to(device=device, dtype=dtype)
        if rough.shape[-2:] != (height, width):
            raise ValueError(f"roughness_proxy 空间尺寸 {tuple(rough.shape[-2:])} 与 {spatial_hw} 不一致")
        if rough.size(0) != batch:
            if rough.size(0) == 1:
                rough = rough.expand(batch, -1, -1, -1)
            else:
                raise ValueError(f"roughness_proxy batch={rough.size(0)} 与图 batch={batch} 不一致")
        flat = rough[:, :1].flatten(2).squeeze(1)
        lo = flat.amin(dim=1, keepdim=True)
        hi = flat.amax(dim=1, keepdim=True)
        return (flat - lo) / (hi - lo).clamp_min(1e-6)

    def build_wind_adjacency(
        self,
        base_adj: torch.Tensor,
        wind_uv: torch.Tensor,
        *,
        spatial_hw: Tuple[int, int],
        roughness_proxy: Optional[torch.Tensor] = None,
        static_cont: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Diagnostics]:
        batch, nodes, _ = base_adj.shape
        height, width = int(spatial_hw[0]), int(spatial_hw[1])
        wind = self._coerce_wind(wind_uv, batch=batch, spatial_hw=(height, width), dtype=base_adj.dtype)
        speed = wind.norm(dim=-1).clamp_min(0.0)
        mean_speed = speed.mean()
        if float(mean_speed.detach().cpu()) <= self.calm_wind_threshold:
            return base_adj, {
                "wind_anisotropy_mean": base_adj.new_tensor(0.0),
                "anisotropy_strength": base_adj.new_tensor(0.0),
                "upwind_weight_mean": _edge_weight_mean(base_adj),
                "downwind_weight_mean": _edge_weight_mean(base_adj),
                "roughness_blocking_mean": base_adj.new_tensor(1.0),
                "calm_wind": base_adj.new_tensor(1.0),
            }

        coords = _grid_coordinates(height, width, device=base_adj.device, dtype=base_adj.dtype)
        source_to_target = coords.unsqueeze(0) - coords.unsqueeze(1)
        distance = source_to_target.norm(dim=-1).clamp_min(1e-6)
        wind_dir = wind / speed.unsqueeze(-1).clamp_min(1e-6)
        alignment = torch.einsum("nmd,bnd->bnm", source_to_target, wind_dir) / distance.unsqueeze(0)

        anisotropy = torch.tanh(speed / self.wind_speed_scale).unsqueeze(-1)
        directional = torch.exp(self.wind_strength * anisotropy * alignment / self.wind_temperature)

        roughness = self._coerce_roughness(
            roughness_proxy,
            static_cont,
            batch=batch,
            spatial_hw=(height, width),
            device=base_adj.device,
            dtype=base_adj.dtype,
        )
        rough_pair = 0.5 * (roughness.unsqueeze(2) + roughness.unsqueeze(1))
        blocking = torch.exp(-self.roughness_blocking_strength * anisotropy * rough_pair)

        adj = base_adj * directional * blocking
        adj = adj.masked_fill(base_adj <= 0.0, 0.0)
        if self.use_self_loop:
            eye = torch.eye(nodes, device=base_adj.device, dtype=base_adj.dtype).unsqueeze(0)
            adj = torch.maximum(adj, eye.expand_as(adj))
        adj = _row_normalize(adj)
        return adj, {
            "wind_anisotropy_mean": anisotropy.mean(),
            "anisotropy_strength": anisotropy.mean(),
            "upwind_weight_mean": _directional_edge_weight_mean(adj, alignment, downwind=False),
            "downwind_weight_mean": _directional_edge_weight_mean(adj, alignment, downwind=True),
            "roughness_blocking_mean": blocking.masked_select(_edge_mask(base_adj)).mean(),
            "calm_wind": base_adj.new_tensor(0.0),
        }

    def build_adjacency(
        self,
        static_feat: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
        wind_uv: Optional[torch.Tensor] = None,
        *,
        static_cont: Optional[torch.Tensor] = None,
        static_cat: Optional[torch.Tensor] = None,
        static_hash: Optional[str] = None,
        roughness_proxy: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Diagnostics]:
        static_cont, static_cat = _coerce_static_maps(
            static_feat=static_feat,
            static_cont=static_cont,
            static_cat=static_cat,
            categorical_indices=self.categorical_indices,
        )
        base_adj, diagnostics = super().build_adjacency(
            static_feat=None,
            spatial_hw=spatial_hw,
            static_cont=static_cont,
            static_cat=static_cat,
            static_hash=static_hash,
        )
        if wind_uv is None or wind_uv.numel() == 0:
            diagnostics.update(
                {
                    "wind_anisotropy_mean": base_adj.new_tensor(0.0),
                    "anisotropy_strength": base_adj.new_tensor(0.0),
                    "upwind_weight_mean": _edge_weight_mean(base_adj),
                    "downwind_weight_mean": _edge_weight_mean(base_adj),
                    "roughness_blocking_mean": base_adj.new_tensor(1.0),
                    "calm_wind": base_adj.new_tensor(1.0),
                }
            )
            _add_process_graph_aliases(diagnostics, PROCESS_GRAPH_RELATION_KEYS)
            return base_adj, diagnostics
        if spatial_hw is None:
            source = static_cont if static_cont is not None else static_cat
            if source is None:
                raise ValueError("缺少 spatial_hw，且无法从静态字段推断空间尺寸")
            spatial_hw = (int(source.shape[-2]), int(source.shape[-1]))
        wind_adj, wind_diag = self.build_wind_adjacency(
            base_adj,
            wind_uv.to(device=base_adj.device),
            spatial_hw=spatial_hw,
            roughness_proxy=roughness_proxy,
            static_cont=static_cont.to(device=base_adj.device) if static_cont is not None else None,
        )
        diagnostics.update(wind_diag)
        diagnostics["edge_weight_mean"] = _edge_weight_mean(wind_adj)
        diagnostics["edge_weight_std"] = _edge_weight_std(wind_adj)
        diagnostics["graph_entropy"] = _graph_entropy(wind_adj)
        diagnostics["adjacency_entropy"] = diagnostics["graph_entropy"]
        diagnostics["wind_alignment_mean"] = diagnostics.get("downwind_weight_mean", wind_adj.new_tensor(0.0))
        diagnostics["ventilation_corridor_mean"] = diagnostics["wind_alignment_mean"]
        _add_process_graph_aliases(diagnostics, PROCESS_GRAPH_RELATION_KEYS)
        return wind_adj, diagnostics