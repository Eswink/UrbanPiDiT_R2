import unittest

import torch

from baselines.forecast_base import ForecastModelBase
from baselines.forecast_models import (
    BASELINE_REGISTRY,
    DEFAULT_BASELINE_NAMES,
    build_baselines_from_config,
    build_forecast_baseline,
)


DYNAMIC_VARS = ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"]
STATIC_VARS = ["landcover", "building_surface", "buildings", "building_volume", "population"]


def _batch(batch_size=2, k=4, h=8, w=8, lead_times=(1, 2, 3, 4), with_static=True):
    c = len(DYNAMIC_VARS)
    s = len(STATIC_VARS) if with_static else 0
    x_ctx = torch.randn(batch_size, c * k + s, h, w)
    out = {
        "x_ctx": x_ctx,
        "lead_times": torch.tensor(list(lead_times), dtype=torch.long),
        "norm": {
            "mean": torch.zeros(c, 1, 1),
            "std": torch.ones(c, 1, 1),
        },
        "clim": torch.randn(c, h, w),
    }
    if with_static:
        out["static_raw"] = x_ctx[:, c * k :].clone()
        out["static_cont"] = x_ctx[:, c * k + 1 :].clone()
        out["static_cat"] = torch.zeros(batch_size, 1, h, w, dtype=torch.long)
    return out


class TestBaselineShapes(unittest.TestCase):
    def test_default_ten_baselines_emit_multi_lead_fields(self):
        batch = _batch()
        for name in DEFAULT_BASELINE_NAMES:
            model = build_forecast_baseline(
                name,
                dynamic_vars=DYNAMIC_VARS,
                static_vars=STATIC_VARS,
                k=4,
                lead_times=[1, 2, 3, 4],
            )
            with self.subTest(name=name):
                pred = model(batch)
                self.assertEqual(tuple(pred.shape), (2, 4, len(DYNAMIC_VARS), 8, 8))
                self.assertTrue(torch.isfinite(pred).all())

    def test_trainable_lightweight_baselines_keep_shape(self):
        batch = _batch(batch_size=1, lead_times=(1, 3))
        for name in ("conv_persistence", "static_conv"):
            model = build_forecast_baseline(
                name,
                dynamic_vars=DYNAMIC_VARS,
                static_vars=STATIC_VARS,
                k=4,
                lead_times=[1, 3],
            )
            with self.subTest(name=name):
                pred = model(batch)
                self.assertEqual(tuple(pred.shape), (1, 2, len(DYNAMIC_VARS), 8, 8))
                self.assertTrue(torch.isfinite(pred).all())

    def test_static_policy_controls_feature_scope(self):
        batch = _batch(batch_size=1)
        dynamic_only = build_forecast_baseline(
            "persistence",
            dynamic_vars=DYNAMIC_VARS,
            static_vars=STATIC_VARS,
            k=4,
            lead_times=[1],
            static_policy="dynamic_only",
        )
        same_static = build_forecast_baseline(
            "static_conv",
            dynamic_vars=DYNAMIC_VARS,
            static_vars=STATIC_VARS,
            k=4,
            lead_times=[1],
            static_policy="same_static",
        )
        self.assertEqual(dynamic_only.prepare_batch(batch).static.shape[1], 0)
        self.assertEqual(same_static.prepare_batch(batch).static.shape[1], len(STATIC_VARS))

    def test_config_builder_creates_requested_models(self):
        cfg = {
            "dynamic_vars": DYNAMIC_VARS,
            "static_vars": STATIC_VARS,
            "k": 4,
            "forecast": {"lead_times": [1, 2]},
            "baselines": {
                "static_policy": "dynamic_only",
                "models": [
                    "persistence",
                    {"name": "static_analog", "alias": "static_analog_same", "static_policy": "same_static"},
                ],
            },
        }
        models = build_baselines_from_config(cfg)
        self.assertEqual(set(models), {"persistence", "static_analog_same"})
        self.assertTrue(all(isinstance(model, ForecastModelBase) for model in models.values()))

    def test_registry_contains_at_least_ten_shape_baselines(self):
        for name in DEFAULT_BASELINE_NAMES:
            self.assertIn(name, BASELINE_REGISTRY)
        self.assertGreaterEqual(len(DEFAULT_BASELINE_NAMES), 10)


if __name__ == "__main__":
    unittest.main()
