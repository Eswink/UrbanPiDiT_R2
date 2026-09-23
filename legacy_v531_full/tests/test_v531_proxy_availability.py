from __future__ import annotations

import torch

from models.morpho_process_proxy import MorphologyProcessProxyEncoder


SOURCE_FLAG_KEYS = (
    "proxy/source_has_static_raw",
    "proxy/source_has_static_cont",
    "proxy/source_has_static_cat",
    "proxy/source_has_schema",
    "proxy/source_has_landcover",
    "proxy/source_has_building_surface",
    "proxy/source_has_buildings",
    "proxy/source_has_building_volume",
    "proxy/source_has_population",
    "proxy/source_has_vegetation",
    "proxy/source_has_svf",
    "proxy/estimated_proxy_fields",
)


def test_proxy_availability_flags_are_reported() -> None:
    batch, height, width = 2, 8, 8
    encoder = MorphologyProcessProxyEncoder(
        static_channels=5,
        hidden_channels=(8, 16, 32),
        use_diurnal_modulation=False,
    )

    out = encoder(
        torch.zeros(batch, 7, height, width),
        static_cont=torch.rand(batch, 4, height, width),
        static_cat=torch.randint(0, 4, (batch, 1, height, width)),
    )

    for key in SOURCE_FLAG_KEYS:
        assert key in out.diagnostics
        value = out.diagnostics[key]
        assert isinstance(value, torch.Tensor)
        assert value.numel() == 1
        assert torch.isfinite(value).all()

    assert float(out.diagnostics["proxy/source_has_static_cont"]) == 1.0
    assert float(out.diagnostics["proxy/source_has_static_cat"]) == 1.0
    assert float(out.diagnostics["proxy/source_has_population"]) == 1.0
    assert float(out.diagnostics["proxy/source_has_svf"]) == 0.0