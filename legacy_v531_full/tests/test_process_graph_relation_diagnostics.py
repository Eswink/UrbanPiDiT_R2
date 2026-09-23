from __future__ import annotations

import torch

from models.anisotropic_urban_process_graph import AnisotropicUrbanProcessGraph


RELATION_KEYS = (
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


def test_process_graph_relation_diagnostics_are_finite_scalars() -> None:
    batch, height, width = 2, 4, 4
    graph = AnisotropicUrbanProcessGraph(dim=16, k=2, enable_cache=False)

    _, diagnostics = graph.build_adjacency(
        static_cont=torch.rand(batch, 4, height, width),
        static_cat=torch.randint(0, 4, (batch, 1, height, width)),
        spatial_hw=(height, width),
        wind_uv=torch.randn(batch, 2, height, width),
        roughness_proxy=torch.rand(batch, 1, height, width),
        diffusion_t=torch.tensor([0.1, 0.9]),
    )

    for key in RELATION_KEYS:
        assert key in diagnostics
        assert f"process_graph/{key}" in diagnostics
        value = diagnostics[key]
        assert isinstance(value, torch.Tensor)
        assert value.numel() == 1
        assert torch.isfinite(value).all()