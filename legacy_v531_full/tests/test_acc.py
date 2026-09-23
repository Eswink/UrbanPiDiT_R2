import unittest

import torch

from metrics.acc import acc


class TestACC(unittest.TestCase):
    def test_acc_perfect_positive(self):
        B, C, H, W = 2, 1, 5, 6
        lat = torch.linspace(10.0, 50.0, H)
        clim = torch.randn(C, H, W)
        anomaly = torch.randn(B, C, H, W)
        pred = clim.unsqueeze(0) + anomaly
        target = clim.unsqueeze(0) + anomaly
        v = acc(pred, target, clim, lat=lat, center=True, reduction="mean")
        self.assertTrue(torch.isfinite(v))
        self.assertAlmostEqual(float(v), 1.0, places=5)

    def test_acc_perfect_negative(self):
        B, C, H, W = 2, 1, 4, 4
        lat = torch.linspace(-20.0, 20.0, H)
        clim = torch.randn(C, H, W)
        anomaly = torch.randn(B, C, H, W)
        pred = clim.unsqueeze(0) + anomaly
        target = clim.unsqueeze(0) - anomaly
        v = acc(pred, target, clim, lat=lat, center=True, reduction="mean")
        self.assertTrue(torch.isfinite(v))
        self.assertAlmostEqual(float(v), -1.0, places=5)

    def test_acc_zero_variance_returns_nan(self):
        B, C, H, W = 1, 2, 3, 3
        lat = torch.linspace(0.0, 60.0, H)
        clim = torch.zeros(C, H, W)
        pred = torch.ones(B, C, H, W)
        target = torch.ones(B, C, H, W)
        v = acc(pred, target, clim, lat=lat, center=True, reduction="mean")
        self.assertTrue(torch.isnan(v))

    def test_acc_ignores_nan_points(self):
        B, C, H, W = 1, 1, 4, 4
        lat = torch.linspace(0.0, 30.0, H)
        clim = torch.zeros(C, H, W)
        anomaly = torch.randn(B, C, H, W)
        pred = anomaly.clone()
        target = anomaly.clone()
        pred[..., 0, 0] = float("nan")
        target[..., 0, 0] = float("nan")
        v = acc(pred, target, clim, lat=lat, center=True, reduction="mean")
        self.assertTrue(torch.isfinite(v))
        self.assertAlmostEqual(float(v), 1.0, places=5)

    def test_acc_mean_reduces_only_finite_channels(self):
        B, C, H, W = 1, 2, 3, 3
        clim = torch.zeros(C, H, W)
        anomaly = torch.randn(B, 1, H, W)
        pred = torch.cat([anomaly, torch.ones(B, 1, H, W)], dim=1)
        target = torch.cat([anomaly, torch.ones(B, 1, H, W)], dim=1)

        per_channel = acc(pred, target, clim, center=True, reduction="channel_mean")
        self.assertTrue(torch.isfinite(per_channel[0]))
        self.assertTrue(torch.isnan(per_channel[1]))

        v = acc(pred, target, clim, center=True, reduction="mean")
        self.assertTrue(torch.isfinite(v))
        self.assertAlmostEqual(float(v), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

