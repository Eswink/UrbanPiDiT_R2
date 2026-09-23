from __future__ import annotations

import torch

from models.morpho_process_proxy import MorphologyProcessProxyEncoder, PROCESS_PROXY_NAMES


def _inputs(batch: int = 4, height: int = 8, width: int = 8) -> dict[str, torch.Tensor]:
    static_cat = torch.randint(0, 4, (batch, 1, height, width))
    static_cont = torch.rand(batch, 4, height, width)
    return {
        "ref": torch.zeros(batch, 7, height, width),
        "static_raw": torch.cat([static_cat.float(), static_cont], dim=1),
        "static_cont": static_cont,
        "static_cat": static_cat,
        "hour_of_day": torch.tensor([0.0, 6.0, 12.0, 18.0])[:batch],
    }


def test_proxy_encoder_outputs_expected_shapes() -> None:
    data = _inputs()
    encoder = MorphologyProcessProxyEncoder(static_channels=5, hidden_channels=(8, 16, 32), dropout=0.0)

    out = encoder(**data)

    assert set(out.proxy_fields) == set(PROCESS_PROXY_NAMES)
    for value in out.proxy_fields.values():
        assert value.shape == (4, 1, 8, 8)
        assert torch.isfinite(value).all()
    assert [tuple(x.shape) for x in out.level_features] == [(4, 8, 8, 8), (4, 16, 4, 4), (4, 32, 2, 2)]


def test_proxy_encoder_zero_init_starts_near_half() -> None:
    data = _inputs()
    encoder = MorphologyProcessProxyEncoder(
        static_channels=5,
        hidden_channels=(8, 16, 32),
        use_diurnal_modulation=False,
    )

    out = encoder(**data)

    for value in out.proxy_fields.values():
        assert torch.allclose(value, torch.full_like(value, 0.5), atol=1e-6)


def test_schema_inputs_take_priority_over_raw() -> None:
    data = _inputs()
    encoder = MorphologyProcessProxyEncoder(static_channels=5, hidden_channels=(8, 16, 32))

    out_schema = encoder(
        data["ref"],
        static_raw=torch.zeros_like(data["static_raw"]),
        static_cont=data["static_cont"],
        static_cat=data["static_cat"],
    )
    out_fallback = encoder(data["ref"], static_raw=data["static_raw"])

    assert float(out_schema.diagnostics["proxy/static_schema_fallback"]) == 0.0
    assert float(out_fallback.diagnostics["proxy/static_schema_fallback"]) == 1.0
    assert float(out_schema.diagnostics["proxy/static_cont_channels"]) == 4.0
    assert float(out_schema.diagnostics["proxy/static_cat_channels"]) == 1.0


def test_diurnal_modulation_changes_heat_storage_proxy() -> None:
    data = _inputs(batch=2)
    encoder = MorphologyProcessProxyEncoder(static_channels=5, hidden_channels=(8, 16, 32), proxy_init="zero")

    night = encoder(
        data["ref"],
        static_cont=data["static_cont"],
        static_cat=data["static_cat"],
        hour_of_day=torch.tensor([0.0, 0.0]),
    ).proxy_fields["heat_storage_proxy"]
    noon = encoder(
        data["ref"],
        static_cont=data["static_cont"],
        static_cat=data["static_cat"],
        hour_of_day=torch.tensor([12.0, 12.0]),
    ).proxy_fields["heat_storage_proxy"]

    assert noon.mean() > night.mean()