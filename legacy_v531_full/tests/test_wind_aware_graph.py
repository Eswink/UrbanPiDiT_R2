from __future__ import annotations

import torch

from models.morphology_graph import WindAwareMorphologyGraph


def _line_static() -> torch.Tensor:
    return torch.zeros(1, 1, 1, 5)


def test_wind_anisotropy_favors_downwind_edges() -> None:
    graph = WindAwareMorphologyGraph(
        dim=8,
        k=99,
        use_self_loop=False,
        alpha_geo=1.0,
        alpha_cont=0.0,
        alpha_cat=0.0,
        wind_strength=1.0,
        wind_temperature=0.5,
        roughness_blocking_strength=0.0,
        enable_cache=False,
    )
    base_adj, _ = graph.build_adjacency(static_cont=_line_static(), spatial_hw=(1, 5), wind_uv=None)
    wind_uv = torch.zeros(1, 2, 1, 5)
    wind_uv[:, 0] = 5.0
    wind_adj, diag = graph.build_adjacency(static_cont=_line_static(), spatial_hw=(1, 5), wind_uv=wind_uv)

    assert wind_adj[0, 2, 3] > wind_adj[0, 2, 1]
    assert wind_adj[0, 2, 3] > base_adj[0, 2, 3]
    assert float(diag["wind_anisotropy_mean"]) > 0.0
    assert float(diag["calm_wind"]) == 0.0


def test_calm_wind_degenerates_to_base_graph() -> None:
    graph = WindAwareMorphologyGraph(
        dim=8,
        k=99,
        use_self_loop=True,
        wind_strength=2.0,
        calm_wind_threshold=1e-3,
        enable_cache=False,
    )
    static = torch.randn(1, 1, 2, 2)
    calm = torch.zeros(1, 2, 2, 2)

    base_adj, _ = graph.build_adjacency(static_cont=static, spatial_hw=(2, 2), wind_uv=None)
    wind_adj, diag = graph.build_adjacency(static_cont=static, spatial_hw=(2, 2), wind_uv=calm)

    assert torch.allclose(base_adj, wind_adj, atol=1e-6)
    assert float(diag["calm_wind"]) == 1.0


def test_roughness_blocks_edges_through_rough_cells() -> None:
    graph = WindAwareMorphologyGraph(
        dim=8,
        k=99,
        use_self_loop=False,
        alpha_geo=1.0,
        alpha_cont=0.0,
        alpha_cat=0.0,
        wind_strength=0.0,
        roughness_blocking_strength=3.0,
        enable_cache=False,
    )
    static = _line_static()
    wind_uv = torch.zeros(1, 2, 1, 5)
    wind_uv[:, 0] = 5.0
    rough = torch.tensor([[[[0.0, 0.0, 1.0, 0.0, 0.0]]]])

    no_rough, _ = graph.build_adjacency(
        static_cont=static,
        spatial_hw=(1, 5),
        wind_uv=wind_uv,
        roughness_proxy=torch.zeros_like(rough),
    )
    blocked, diag = graph.build_adjacency(
        static_cont=static,
        spatial_hw=(1, 5),
        wind_uv=wind_uv,
        roughness_proxy=rough,
    )

    assert blocked[0, 1, 2] < no_rough[0, 1, 2]
    assert float(diag["roughness_blocking_mean"]) < 1.0


def test_wind_aware_forward_optional_diagnostics() -> None:
    graph = WindAwareMorphologyGraph(dim=4, k=2, use_self_loop=True, enable_cache=False)
    x = torch.randn(1, 4, 4)
    static = torch.randn(1, 1, 2, 2)
    wind = torch.randn(1, 2, 2, 2)

    out, diag = graph(x, static_cont=static, spatial_hw=(2, 2), wind_uv=wind, return_diagnostics=True)

    assert out.shape == x.shape
    assert "wind_anisotropy_mean" in diag


def test_wind_aware_schema_and_directional_diagnostics_are_reported() -> None:
    graph = WindAwareMorphologyGraph(dim=4, k=2, use_self_loop=True, enable_cache=False)
    static_cont = torch.rand(2, 2, 2, 2)
    static_cat = torch.randint(0, 3, (2, 1, 2, 2))
    wind = torch.zeros(2, 2, 2, 2)
    wind[:, 0] = 2.0

    _, diag = graph.build_adjacency(
        static_cont=static_cont,
        static_cat=static_cat,
        spatial_hw=(2, 2),
        wind_uv=wind,
    )

    assert float(diag["continuous_channel_count"]) == 2.0
    assert float(diag["categorical_channel_count"]) == 1.0
    assert float(diag["static_schema_fallback"]) == 0.0
    for key in (
        "anisotropy_strength",
        "upwind_weight_mean",
        "downwind_weight_mean",
        "roughness_blocking_mean",
        "edge_weight_std",
    ):
        assert key in diag
        assert torch.isfinite(diag[key])