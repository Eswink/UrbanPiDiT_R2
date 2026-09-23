from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from baselines.common import load_config

from baselines.faithful.common import build_graphcast_graph, edm_preconditioning, karras_schedule
from baselines.faithful.fourcastnet.afno import BlockDiagonalAFNO2D
from baselines.faithful.gencast.transformer import ConditionalLayerNorm
from baselines.faithful.graphcast.graph_net import InteractionNetwork
from baselines.forecast_models import available_baselines, build_forecast_baseline


DYNAMIC_VARS = ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"]
STATIC_VARS = ["landcover", "building_surface", "buildings", "building_volume", "population"]


def _dummy_batch(batch_size: int = 1, leads: tuple[int, ...] = (1, 2), k: int = 4, h: int = 8, w: int = 8):
    c = len(DYNAMIC_VARS)
    s = len(STATIC_VARS)
    x_ctx = torch.randn(batch_size, c * k + s, h, w)
    static = x_ctx[:, c * k :]
    return {
        "x_ctx": x_ctx,
        "static_raw": static,
        "static_cont": static[:, 1:],
        "static_cat": static[:, :1].round().long().clamp(min=0, max=19),
        "y": torch.randn(batch_size, len(leads), c, h, w),
        "x0": torch.randn(batch_size, c, h, w),
        "lead_times": torch.tensor(leads, dtype=torch.long),
        "hour_of_day": torch.zeros(batch_size, dtype=torch.long),
        "norm": {
            "mean": torch.zeros(c, 1, 1),
            "std": torch.ones(c, 1, 1),
        },
        "clim": torch.zeros(c, h, w),
    }


def _build(name: str, **overrides):
    params = {
        "faithful_fourcastnet": {
            "hidden_channels": 8,
            "depth": 1,
            "num_blocks": 2,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
            "sparsity_threshold": 0.01,
            "include_coords": True,
            "include_hour": True,
            "forecast_protocol": "official_rollout",
        },
        "faithful_graphcast": {
            "hidden_channels": 8,
            "mesh_height": 2,
            "mesh_width": 2,
            "message_passing_steps": 1,
            "k_nn": 2,
            "dropout": 0.0,
            "forecast_protocol": "official_rollout",
        },
        "faithful_gencast": {
            "hidden_channels": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_hidden": 32,
            "sigma_min": 0.002,
            "sigma_max": 0.1,
            "num_noise_levels": 2,
            "num_samples": 1,
            "stochastic_eval": False,
            "forecast_protocol": "direct",
        },
        "faithful_corrdiff": {
            "hidden_channels": 8,
            "cond_channels": 8,
            "channel_mult": (1, 2),
            "sigma_min": 0.002,
            "sigma_max": 0.1,
            "num_noise_levels": 0,
            "num_samples": 1,
            "training_stage": "regression",
            "forecast_protocol": "direct",
        },
    }[name]
    params.update(overrides)
    return build_forecast_baseline(
        name,
        dynamic_vars=DYNAMIC_VARS,
        static_vars=STATIC_VARS,
        k=4,
        lead_times=[1, 2],
        static_policy="same_static",
        params=params,
    )


def test_faithful_registry_entries_available():
    names = set(available_baselines())
    assert {
        "faithful_fourcastnet",
        "faithful_graphcast",
        "faithful_gencast",
        "faithful_corrdiff",
    }.issubset(names)


def test_afno_block_diagonal_frequency_mixing_shape():
    block = BlockDiagonalAFNO2D(hidden_channels=8, num_blocks=2, hidden_size_factor=1, sparsity_threshold=0.01)
    x = torch.randn(2, 8, 8, 8)
    y = block(x)
    assert y.shape == x.shape
    assert block.w1.shape == (2, 2, 4, 4)
    assert block.w2.shape == (2, 2, 4, 4)
    assert torch.isfinite(y).all()


def test_graphcast_graph_structure_matches_8x8_domain():
    graph = build_graphcast_graph(8, 8, mesh_height=4, mesh_width=4, k_grid_to_mesh=4, k_mesh_to_grid=4)
    assert graph.grid_coords.shape == (64, 2)
    assert graph.mesh_coords.shape == (16, 2)
    assert graph.grid2mesh_senders.numel() >= 16
    assert graph.mesh2grid_senders.numel() == 64 * 3
    assert graph.mesh_senders.numel() > 0
    assert graph.grid2mesh_edge_features.shape[-1] == 5


def test_interaction_network_message_passing_backpropagates():
    net = InteractionNetwork(node_dim=4, edge_dim=3, hidden_dim=8)
    sender_nodes = torch.randn(2, 5, 4, requires_grad=True)
    receiver_nodes = torch.randn(2, 6, 4, requires_grad=True)
    edge_features = torch.randn(7, 3)
    senders = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    receivers = torch.tensor([0, 1, 2, 3, 4, 5, 0])
    node_out, edge_out = net(sender_nodes, receiver_nodes, edge_features, senders, receivers, receiver_count=6)
    assert node_out.shape == receiver_nodes.shape
    assert edge_out.shape == (2, 7, 3)
    (node_out.mean() + edge_out.mean()).backward()
    assert sender_nodes.grad is not None
    assert receiver_nodes.grad is not None


def test_norm_conditioning_applies_scale_and_shift():
    layer = ConditionalLayerNorm(hidden_dim=8, cond_dim=4)
    x = torch.randn(2, 5, 8)
    cond = torch.randn(2, 4)
    with torch.no_grad():
        layer.modulation.bias[:8].fill_(0.5)
        layer.modulation.bias[8:].fill_(1.0)
    out = layer(x, cond)
    expected = layer.norm(x) * 1.5 + 1.0
    assert torch.allclose(out, expected, atol=1e-6)


@torch.no_grad()
def test_faithful_models_emit_forecast_contract_shape():
    batch = _dummy_batch()
    for name in ["faithful_fourcastnet", "faithful_graphcast", "faithful_gencast", "faithful_corrdiff"]:
        model = _build(name)
        model.eval()
        pred = model(batch, lead_times=[1, 2])
        assert pred.shape == (1, 2, len(DYNAMIC_VARS), 8, 8), name
        assert torch.isfinite(pred).all(), name


def test_diffusion_training_losses_are_finite_and_trainable():
    batch = _dummy_batch()
    gencast = _build("faithful_gencast")
    gen_loss = gencast.training_loss(batch, lead_times=[1, 2])
    assert torch.isfinite(gen_loss)
    gen_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in gencast.parameters() if p.requires_grad)

    corrdiff = _build(
        "faithful_corrdiff",
        training_stage="diffusion",
        freeze_regression=True,
        num_noise_levels=1,
        sigma_max=0.1,
    )
    assert not any(p.requires_grad for p in corrdiff.regression_unet.parameters())
    corr_loss = corrdiff.training_loss(batch, lead_times=[1, 2])
    assert torch.isfinite(corr_loss)
    corr_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in corrdiff.diffusion_unet.parameters() if p.requires_grad)


def test_edm_schedule_and_preconditioning_are_broadcastable():
    ref = torch.randn(2, 7, 8, 8)
    sigma = torch.tensor([0.01, 0.1])
    coeffs = edm_preconditioning(sigma, ref, sigma_data=0.5)
    schedule = karras_schedule(4, sigma_min=0.002, sigma_max=0.1, device=ref.device, dtype=ref.dtype)
    assert coeffs.c_skip.shape == (2, 1, 1, 1)
    assert coeffs.c_noise.shape == (2,)
    assert schedule.shape == (5,)
    assert torch.all(schedule[:-1] >= schedule[1:])


def test_faithful_training_configs_load_and_target_beijing_data():
    root = Path(__file__).resolve().parents[1]
    expected = "/3240608030/weather-q/data/processed_data/beijing"
    config_names = {
        "fourcastnet_faithful.yaml": ("faithful_fourcastnet", "auto"),
        "graphcast_faithful.yaml": ("faithful_graphcast", "auto"),
        "gencast_faithful.yaml": ("faithful_gencast", "diffusion"),
        "corrdiff_faithful_regression.yaml": ("faithful_corrdiff", "direct"),
        "corrdiff_faithful_diffusion.yaml": ("faithful_corrdiff", "diffusion"),
    }
    for filename, (model_name, train_protocol) in config_names.items():
        cfg = load_config(root / "configs" / "baselines" / filename)
        assert cfg["data_root"] == expected
        assert cfg["dynamic_vars"] == DYNAMIC_VARS
        assert cfg["forecast"]["lead_times"] == [1, 2, 3, 4]
        assert cfg["external_baseline_training"]["train_protocol"] == train_protocol
        assert cfg["baselines"]["models"][0]["name"] == model_name
        assert cfg["baselines"]["models"][0]["family"] == "faithful_weather_baseline"


def test_corrdiff_diffusion_loads_full_training_checkpoint(tmp_path: Path):
    source = _build("faithful_corrdiff", training_stage="regression")
    with torch.no_grad():
        for param in source.regression_unet.parameters():
            param.fill_(0.123)
    checkpoint = tmp_path / "regression_full_payload.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "history": [{"numpy_payload": np.array([1.0], dtype=np.float32)}],
        },
        checkpoint,
    )
    diffusion = _build(
        "faithful_corrdiff",
        training_stage="diffusion",
        regression_checkpoint=str(checkpoint),
        freeze_regression=True,
        num_noise_levels=1,
        sigma_max=0.1,
    )
    first_param = next(diffusion.regression_unet.parameters())
    assert torch.allclose(first_param, torch.full_like(first_param, 0.123))
    assert not any(p.requires_grad for p in diffusion.regression_unet.parameters())