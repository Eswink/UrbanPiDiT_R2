from __future__ import annotations

import torch

from models.anisotropic_urban_process_graph import AnisotropicUrbanProcessGraph


def _inputs(batch: int = 2, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
    static_cat = torch.randint(0, 4, (batch, 1, height, width))
    static_cont = torch.rand(batch, 4, height, width)
    return {
        "x": torch.randn(batch, height * width, 16),
        "static_cont": static_cont,
        "static_cat": static_cat,
        "wind_uv": torch.randn(batch, 2, height, width),
        "roughness_proxy": torch.rand(batch, 1, height, width),
        "spatial_hw": (height, width),
    }


def test_anisotropic_graph_zero_gamma_is_identity() -> None:
    data = _inputs()
    graph = AnisotropicUrbanProcessGraph(dim=16, k=2, enable_cache=False)

    out, diag = graph(**data, diffusion_t=torch.zeros(2), return_diagnostics=True)

    assert out.shape == data["x"].shape
    assert torch.allclose(out, data["x"], atol=1e-6)
    assert "process_graph/wind_scale_t" in diag
    assert "process_graph/block_scale_t" in diag


def test_anisotropic_graph_time_scales_are_initialized_to_one() -> None:
    data = _inputs()
    graph = AnisotropicUrbanProcessGraph(dim=16, k=2, enable_cache=False)

    _, diag = graph(**data, diffusion_t=torch.tensor([0.1, 0.9]), return_diagnostics=True)

    assert torch.allclose(diag["process_graph/wind_scale_t"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(diag["process_graph/block_scale_t"], torch.tensor(1.0), atol=1e-6)


def test_anisotropic_graph_accepts_different_timesteps() -> None:
    data = _inputs()
    graph = AnisotropicUrbanProcessGraph(dim=16, k=2, enable_cache=False)
    with torch.no_grad():
        graph.time_mlp[-1].bias[:] = torch.tensor([0.2, -0.2])

    _, diag = graph(**data, diffusion_t=torch.tensor([0.2, 0.8]), return_diagnostics=True)

    assert torch.isfinite(diag["process_graph/graph_entropy"])
    assert float(diag["process_graph/wind_scale_t"]) != 1.0