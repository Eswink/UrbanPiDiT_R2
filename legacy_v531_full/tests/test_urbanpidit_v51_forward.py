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


def _base_kwargs(**overrides: object) -> dict[str, object]:
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
        "dynamic_vars": DYNAMIC_VARS,
        "static_vars": STATIC_VARS,
        "static_schema": STATIC_SCHEMA,
        "morphology_graph_cfg": {"k": 2, "enable_cache": False},
        "micromet_coupling_cfg": {"hidden_channels": 8, "graph_cfg": {"k": 2, "enable_cache": False}},
        "urban_canopy_cfg": {"hidden_channels": 8, "graph_cfg": {"k": 2, "enable_cache": False}},
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
    x_ctx = torch.cat([dyn, static_raw], dim=1)
    return {
        "x_t": x_t,
        "x_ctx": x_ctx,
        "t": torch.rand(batch),
        "lead_time": torch.full((batch,), 0.25),
        "static_raw": static_raw,
        "static_cont": static_cont,
        "static_cat": static_cat,
        "hour_of_day": torch.tensor([9.0, 21.0])[:batch],
    }


def _forward(cfg: dict[str, object]) -> UrbanPiDiT:
    model = UrbanPiDiT(**cfg)
    data = _batch()
    out = model(**data)
    assert out.shape == data["x_t"].shape
    assert torch.isfinite(out).all()
    return model


def test_default_forward_without_optional_modules() -> None:
    model = _forward(_base_kwargs())

    assert model.micromet_coupling is None
    assert model.urban_canopy is None


def test_micromet_pre_forward_is_compatible() -> None:
    model = _forward(
        _base_kwargs(
            use_micromet_coupling=True,
            micromet_coupling_cfg={"mode": "pre", "hidden_channels": 8, "graph_cfg": {"k": 2, "enable_cache": False}},
        )
    )

    assert model.micromet_coupling is not None
    assert float(model.last_micromet_diagnostics["micromet_enabled"]) == 1.0


def test_micromet_post_forward_is_compatible() -> None:
    model = _forward(
        _base_kwargs(
            use_micromet_coupling=True,
            micromet_coupling_cfg={"mode": "post", "hidden_channels": 8, "graph_cfg": {"k": 2, "enable_cache": False}},
        )
    )

    assert model.micromet_coupling is not None
    assert float(model.last_micromet_diagnostics["micromet_enabled"]) == 1.0


def test_micromet_interleaved_forward_uses_safe_adapter() -> None:
    model = _forward(
        _base_kwargs(
            use_micromet_coupling=True,
            micromet_coupling_cfg={
                "mode": "interleaved",
                "interleaved_interval": 1,
                "hidden_channels": 8,
                "graph_cfg": {"k": 2, "enable_cache": False},
            },
        )
    )

    assert model.micromet_token_to_state is not None
    assert model.micromet_state_to_token is not None
    assert float(model.last_micromet_diagnostics["micromet_enabled"]) == 1.0
    assert "micromet_block_01_residual_norm" in model.last_micromet_diagnostics
    assert "micromet_block_01_token_residual_norm" in model.last_micromet_diagnostics


def test_legacy_urban_canopy_wrapper_forward_is_compatible() -> None:
    model = _forward(_base_kwargs(use_urban_canopy=True))

    assert model.urban_canopy is not None
    assert float(model.last_urban_canopy_diagnostics["micromet_enabled"]) == 1.0


def test_combined_v51_modules_forward() -> None:
    model = _forward(
        _base_kwargs(
            use_static_morphology_encoder=True,
            use_morphology_graph=True,
            use_wind_aware_graph=True,
            use_dynamic_vg=True,
            use_static_vg=True,
            use_micromet_coupling=True,
            micromet_coupling_cfg={"mode": "post", "hidden_channels": 8, "graph_cfg": {"k": 2, "enable_cache": False}},
        )
    )

    assert model.static_morphology_encoder is not None
    assert model.morphology_graph is not None
    assert model.var_graph_t is not None
    assert model.var_graph_ctx is not None