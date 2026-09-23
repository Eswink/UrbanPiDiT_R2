from __future__ import annotations

import torch

from models.urban_pidit import UrbanPiDiT


DYNAMIC_VARS = ("d2m", "sp", "t2m", "tcc", "tp", "u10", "v10")
STATIC_VARS = ("landcover", "building_surface", "buildings", "building_volume", "population")
STATIC_SCHEMA = {
    "categorical": ["landcover"],
    "continuous": ["building_surface", "buildings", "building_volume", "population"],
    "categorical_cardinality": {"landcover": 20},
}


def _kwargs(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "H": 4,
        "W": 4,
        "in_channels": 7,
        "ctx_channels": 33,
        "out_channels": 7,
        "D": 32,
        "depth": 2,
        "heads": 4,
        "mlp_ratio": 2.0,
        "static_channels": 5,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
        "use_lead_time_conditioning": True,
        "use_static_morphology_encoder": True,
        "use_process_proxy_encoder": True,
        "use_process_adaln": True,
        "use_urban_control_branch": True,
        "use_anisotropic_process_graph": True,
        "use_morphology_residual_head": True,
        "dynamic_vars": DYNAMIC_VARS,
        "static_vars": STATIC_VARS,
        "static_schema": STATIC_SCHEMA,
        "morphology_graph_cfg": {"k": 2, "enable_cache": False},
        "morpho_process_cfg": {
            "proxy_encoder": {"hidden_channels": [8, 16, 32], "use_diurnal_modulation": False},
            "process_adaln": {"proxy_proj_dim": 16, "init_alpha": 0.0},
            "anisotropic_process_graph": {"k": 2, "enable_cache": False},
        },
    }
    cfg.update(overrides)
    return cfg


def _batch(batch: int = 2) -> dict[str, torch.Tensor]:
    height = width = 4
    x_t = torch.randn(batch, 7, height, width)
    dyn = torch.randn(batch, 28, height, width)
    static_cat = torch.randint(0, 4, (batch, 1, height, width))
    static_cont = torch.rand(batch, 4, height, width)
    static_raw = torch.cat([static_cat.float(), static_cont], dim=1)
    return {
        "x_t": x_t,
        "x_ctx": torch.cat([dyn, static_raw], dim=1),
        "t": torch.rand(batch),
        "lead_time": torch.full((batch,), 0.25),
        "static_raw": static_raw,
        "static_cont": static_cont,
        "static_cat": static_cat,
        "hour_of_day": torch.tensor([9.0, 21.0])[:batch],
    }


def test_morpho_process_forward_is_shape_and_nan_safe() -> None:
    model = UrbanPiDiT(**_kwargs())
    data = _batch()

    out = model(**data)

    assert out.shape == data["x_t"].shape
    assert torch.isfinite(out).all()
    assert model.process_proxy_encoder is not None
    assert model.process_adaln is not None
    assert model.urban_control_branch is not None
    assert model.anisotropic_process_graph is not None


def test_morpho_process_zero_init_backward_has_gate_gradients() -> None:
    model = UrbanPiDiT(**_kwargs(use_anisotropic_process_graph=False))
    data = _batch()

    loss = model(**data).square().mean()
    loss.backward()

    assert model.process_adaln is not None
    assert model.process_adaln.alpha.grad is not None


def test_degrade_to_v52_disables_new_modules() -> None:
    model = UrbanPiDiT(**_kwargs(degrade_to_v52=True))

    assert model.process_proxy_encoder is None
    assert model.process_adaln is None
    assert model.urban_control_branch is None
    assert model.anisotropic_process_graph is None
    assert model.morphology_residual_head is None