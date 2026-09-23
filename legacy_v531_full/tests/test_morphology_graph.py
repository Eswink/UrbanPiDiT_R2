from __future__ import annotations

import torch

from models.morphology_graph import HeterogeneousMorphologyGraph


def test_same_category_ratio_prefers_category_consistent_edges() -> None:
    graph = HeterogeneousMorphologyGraph(
        dim=8,
        k=1,
        use_self_loop=False,
        alpha_geo=0.0,
        alpha_cont=0.0,
        alpha_cat=5.0,
        enable_cache=False,
    )
    static_cat = torch.tensor([[[[0, 0, 1, 1]]]], dtype=torch.long)

    adj, diag = graph.build_adjacency(static_cat=static_cat, spatial_hw=(1, 4))

    assert adj.shape == (1, 4, 4)
    assert float(diag["same_category_edge_ratio"]) >= 0.75
    assert torch.allclose(adj.sum(dim=-1), torch.ones(1, 4), atol=1e-6)


def test_geo_constraint_keeps_nearest_grid_neighbors() -> None:
    graph = HeterogeneousMorphologyGraph(
        dim=8,
        k=1,
        use_self_loop=False,
        alpha_geo=10.0,
        alpha_cont=0.0,
        alpha_cat=0.0,
        enable_cache=False,
    )
    adj, diag = graph.build_adjacency(static_cont=torch.zeros(1, 1, 1, 4), spatial_hw=(1, 4))
    nbrs = adj[0].argmax(dim=-1).tolist()

    assert nbrs[0] == 1
    assert nbrs[3] == 2
    assert float(diag["mean_degree"]) == 1.0


def test_per_sample_graph_differs_with_static_fields() -> None:
    graph = HeterogeneousMorphologyGraph(
        dim=8,
        k=1,
        use_self_loop=False,
        alpha_geo=0.0,
        alpha_cont=5.0,
        alpha_cat=0.0,
        enable_cache=False,
    )
    sample_a = torch.tensor([[[0.0, 0.1, 5.0, 5.1]]])
    sample_b = torch.tensor([[[0.0, 5.0, 0.1, 5.1]]])
    static_cont = torch.stack([sample_a, sample_b], dim=0)

    adj, _ = graph.build_adjacency(static_cont=static_cont, spatial_hw=(1, 4))

    assert adj.shape == (2, 4, 4)
    assert not torch.allclose(adj[0], adj[1])


def test_sparse_and_dense_shapes_are_valid() -> None:
    static_cont = torch.randn(2, 2, 3, 3)
    sparse = HeterogeneousMorphologyGraph(dim=4, k=2, use_self_loop=True, enable_cache=False)
    dense = HeterogeneousMorphologyGraph(dim=4, k=99, use_self_loop=True, enable_cache=False)

    adj_sparse, diag_sparse = sparse.build_adjacency(static_cont=static_cont, spatial_hw=(3, 3))
    adj_dense, diag_dense = dense.build_adjacency(static_cont=static_cont, spatial_hw=(3, 3))

    assert adj_sparse.shape == (2, 9, 9)
    assert adj_dense.shape == (2, 9, 9)
    assert float(diag_sparse["mean_degree"]) <= float(diag_dense["mean_degree"])
    assert torch.allclose(adj_sparse.sum(dim=-1), torch.ones(2, 9), atol=1e-6)
    assert torch.allclose(adj_dense.sum(dim=-1), torch.ones(2, 9), atol=1e-6)


def test_forward_keeps_legacy_tensor_return_and_optional_diagnostics() -> None:
    graph = HeterogeneousMorphologyGraph(dim=6, k=2, use_self_loop=True, enable_cache=False)
    x = torch.randn(2, 4, 6)
    static_cont = torch.randn(2, 1, 2, 2)

    out = graph(x, static_cont=static_cont, spatial_hw=(2, 2))
    out_diag, diag = graph(x, static_cont=static_cont, spatial_hw=(2, 2), return_diagnostics=True)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out_diag.shape == x.shape
    assert "graph_entropy" in diag


def test_schema_diagnostics_report_continuous_and_categorical_sources() -> None:
    graph = HeterogeneousMorphologyGraph(
        dim=6,
        k=2,
        use_self_loop=True,
        alpha_geo=1.0,
        alpha_cont=1.0,
        alpha_cat=1.0,
        enable_cache=False,
    )
    static_cont = torch.rand(2, 3, 2, 2)
    static_cat = torch.randint(0, 3, (2, 1, 2, 2))

    adj, diag = graph.build_adjacency(static_cont=static_cont, static_cat=static_cat, spatial_hw=(2, 2))

    assert adj.shape == (2, 4, 4)
    assert float(diag["continuous_channel_count"]) == 3.0
    assert float(diag["categorical_channel_count"]) == 1.0
    assert float(diag["static_schema_fallback"]) == 0.0
    for key in (
        "same_category_edge_ratio",
        "geo_distance_mean",
        "cont_distance_mean",
        "categorical_mismatch_mean",
        "edge_weight_mean",
        "edge_weight_std",
        "graph_entropy",
        "effective_topk",
    ):
        assert key in diag
        assert torch.isfinite(diag[key])


def test_legacy_static_feat_fallback_is_explicitly_reported() -> None:
    graph = HeterogeneousMorphologyGraph(
        dim=6,
        k=2,
        use_self_loop=True,
        categorical_indices=(0,),
        enable_cache=False,
    )
    static_feat = torch.cat(
        [
            torch.randint(0, 3, (1, 1, 2, 2)).float(),
            torch.rand(1, 2, 2, 2),
        ],
        dim=1,
    )

    _, diag = graph.build_adjacency(static_feat=static_feat, spatial_hw=(2, 2))

    assert float(diag["continuous_channel_count"]) == 2.0
    assert float(diag["categorical_channel_count"]) == 1.0
    assert float(diag["static_schema_fallback"]) == 1.0