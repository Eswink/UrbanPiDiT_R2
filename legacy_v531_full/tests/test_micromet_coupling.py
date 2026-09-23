from __future__ import annotations

import inspect

import torch

from models.micromet_coupling import MicroMetCouplingOperator, MomentumDragBranch


DYNAMIC_VARS = ("d2m", "sp", "t2m", "tcc", "tp", "u10", "v10")


def _inputs(batch: int = 2, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
    static_cat = torch.randint(0, 4, (batch, 1, height, width))
    static_cont = torch.rand(batch, 4, height, width)
    static_raw = torch.cat([static_cat.float(), static_cont], dim=1)
    return {
        "x": torch.randn(batch, 7, height, width),
        "static_raw": static_raw,
        "static_cont": static_cont,
        "static_cat": static_cat,
        "wind_uv": torch.randn(batch, 2, height, width),
        "hour_of_day": torch.tensor([9.0, 21.0])[:batch],
    }


def test_zero_initialized_operator_is_shape_stable_identity() -> None:
    data = _inputs()
    op = MicroMetCouplingOperator(
        in_channels=7,
        static_channels=5,
        dynamic_vars=DYNAMIC_VARS,
        hidden_channels=8,
        graph_cfg={"k": 2, "enable_cache": False},
    )

    out, diag = op(
        data["x"],
        static_raw=data["static_raw"],
        static_cont=data["static_cont"],
        static_cat=data["static_cat"],
        hour_of_day=data["hour_of_day"],
        wind_uv=data["wind_uv"],
    )

    assert out.shape == data["x"].shape
    assert torch.allclose(out, data["x"], atol=1e-6)
    assert float(diag["micromet_enabled"]) == 1.0
    assert float(diag["diurnal_available"]) == 1.0
    assert float(diag["gate/momentum_drag"].detach()) == 0.0
    assert float(diag["gate/thermal_storage"].detach()) == 0.0
    assert float(diag["gate/moisture_evaporation"].detach()) == 0.0
    assert float(diag["gate/ventilation_mixing"].detach()) == 0.0
    assert "graph/graph_entropy" in diag


def test_disabled_operator_is_explicit_noop() -> None:
    data = _inputs()
    op = MicroMetCouplingOperator(in_channels=7, static_channels=5, enabled=False)

    out, diag = op(data["x"], static_raw=data["static_raw"])

    assert torch.equal(out, data["x"])
    assert float(diag["micromet_enabled"]) == 0.0
    assert float(diag["diurnal_available"]) == 0.0


def test_hour_none_skips_diurnal_branch_without_shape_change() -> None:
    data = _inputs()
    op = MicroMetCouplingOperator(
        in_channels=7,
        static_channels=5,
        dynamic_vars=DYNAMIC_VARS,
        hidden_channels=8,
        enable_ventilation_mixing=False,
    )

    out_none, diag_none = op(data["x"], static_raw=data["static_raw"], hour_of_day=None)
    out_hour, diag_hour = op(data["x"], static_raw=data["static_raw"], hour_of_day=data["hour_of_day"])

    assert out_none.shape == data["x"].shape
    assert out_hour.shape == data["x"].shape
    assert torch.allclose(out_none, data["x"], atol=1e-6)
    assert torch.allclose(out_hour, data["x"], atol=1e-6)
    assert float(diag_none["diurnal_available"]) == 0.0
    assert float(diag_hour["diurnal_available"]) == 1.0


def test_momentum_drag_residual_increases_with_roughness_proxy() -> None:
    branch = MomentumDragBranch(channels=7, dynamic_vars=DYNAMIC_VARS, max_drag=0.5)
    x = torch.zeros(1, 7, 2, 2)
    x[:, 5] = 2.0
    x[:, 6] = -1.0
    low = torch.zeros(1, 1, 2, 2)
    high = torch.ones(1, 1, 2, 2)

    low_delta = branch.residual(
        x,
        {
            "roughness_proxy": low,
            "drag_proxy": low,
            "ventilation_block_proxy": low,
        },
    )
    high_delta = branch.residual(
        x,
        {
            "roughness_proxy": high,
            "drag_proxy": high,
            "ventilation_block_proxy": high,
        },
    )

    assert high_delta[:, [5, 6]].abs().mean() > low_delta[:, [5, 6]].abs().mean()
    assert torch.all(high_delta[:, 5] < 0.0)
    assert torch.all(high_delta[:, 6] > 0.0)


def test_operator_forward_signature_has_no_future_target_inputs() -> None:
    names = set(inspect.signature(MicroMetCouplingOperator.forward).parameters)

    assert "y" not in names
    assert "x0" not in names
    assert "target" not in names
    assert "future" not in names
    assert {"x", "static_raw", "static_cont", "static_cat", "hour_of_day", "wind_uv"}.issubset(names)


def test_static_schema_inputs_take_priority_over_raw_fallback() -> None:
    data = _inputs()
    op = MicroMetCouplingOperator(
        in_channels=7,
        static_channels=5,
        dynamic_vars=DYNAMIC_VARS,
        hidden_channels=8,
        enable_ventilation_mixing=False,
    )

    _, diag_schema = op(
        data["x"],
        static_raw=torch.zeros_like(data["static_raw"]),
        static_cont=data["static_cont"],
        static_cat=data["static_cat"],
    )
    _, diag_fallback = op(data["x"], static_raw=data["static_raw"])

    assert float(diag_schema["static_schema_fallback"]) == 0.0
    assert float(diag_schema["static_cont_channels"]) == 4.0
    assert float(diag_schema["static_cat_channels"]) == 1.0
    assert float(diag_fallback["static_schema_fallback"]) == 1.0
    assert float(diag_fallback["static_raw_channels"]) == 5.0


def test_proxy_diagnostics_are_named_and_finite() -> None:
    data = _inputs()
    op = MicroMetCouplingOperator(
        in_channels=7,
        static_channels=5,
        dynamic_vars=DYNAMIC_VARS,
        hidden_channels=8,
        graph_cfg={"k": 2, "enable_cache": False},
    )

    _, diag = op(
        data["x"],
        static_cont=data["static_cont"],
        static_cat=data["static_cat"],
        wind_uv=data["wind_uv"],
    )

    for name in (
        "roughness_proxy_mean",
        "drag_proxy_mean",
        "heat_storage_proxy_mean",
        "impervious_proxy_mean",
        "evap_proxy_mean",
        "ventilation_block_proxy_mean",
        "anthropogenic_heat_proxy_mean",
    ):
        assert name in diag
        assert torch.isfinite(diag[name])