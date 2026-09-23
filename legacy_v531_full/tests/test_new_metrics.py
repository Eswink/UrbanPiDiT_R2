import unittest

import torch
import torch.nn as nn

from metrics.deterministic import bias, deterministic_metrics, mae, pearson_corr, rmse
from metrics.efficiency import count_parameters, efficiency_summary, model_size_mb, tensor_memory_mb
from metrics.event_metrics import contingency_table, event_scores
from metrics.probabilistic import (
    brier_score,
    ensemble_crps,
    ensemble_mean,
    ensemble_spread,
    probability_of_event,
    spread_skill_ratio,
)
from metrics.significance import paired_bootstrap_delta, paired_permutation_test


class TestNewMetrics(unittest.TestCase):
    def test_deterministic_metrics_known_values(self):
        pred = torch.tensor([[[[1.0, 2.0], [3.0, float("nan")]]]])
        target = torch.tensor([[[[1.0, 1.0], [5.0, 4.0]]]])
        self.assertAlmostEqual(float(mae(pred, target)), 1.0)
        self.assertAlmostEqual(float(rmse(pred, target)), (5.0 / 3.0) ** 0.5, places=6)
        self.assertAlmostEqual(float(bias(pred, target)), -1.0 / 3.0, places=6)
        corr = pearson_corr(pred, target)
        self.assertTrue(torch.isfinite(corr))
        out = deterministic_metrics(pred, target)
        self.assertIn("corr", out)
        self.assertIn("rmse", out)

    def test_probabilistic_metrics_shapes_and_scores(self):
        target = torch.zeros(2, 1, 2, 2)
        ensemble = torch.stack([target - 1.0, target + 1.0], dim=1)
        mean = ensemble_mean(ensemble)
        self.assertTrue(torch.allclose(mean, target))
        self.assertAlmostEqual(float(ensemble_spread(ensemble)), 1.0, places=6)
        self.assertAlmostEqual(float(ensemble_crps(ensemble, target)), 0.5, places=6)
        prob = probability_of_event(ensemble, 0.0, op=">=")
        self.assertTrue(torch.allclose(prob, torch.full_like(target, 0.5)))
        self.assertAlmostEqual(float(brier_score(prob, target.bool())), 0.25, places=6)
        self.assertTrue(torch.isfinite(spread_skill_ratio(ensemble, target)))

    def test_event_metrics_contingency(self):
        pred = torch.tensor([0.1, 0.8, 0.9, 0.2])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        table = contingency_table(pred, target, threshold=0.5)
        self.assertEqual(int(table["hits"]), 1)
        self.assertEqual(int(table["false_alarms"]), 1)
        self.assertEqual(int(table["misses"]), 1)
        self.assertEqual(int(table["correct_negatives"]), 1)
        scores = event_scores(pred, target, threshold=0.5)
        self.assertAlmostEqual(float(scores["csi"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(scores["f1"]), 0.5, places=6)

    def test_significance_helpers_are_deterministic(self):
        a = torch.tensor([2.0, 2.0, 2.0, 2.0])
        b = torch.tensor([1.0, 1.0, 1.0, 1.0])
        boot = paired_bootstrap_delta(a, b, n_boot=64, seed=7)
        perm = paired_permutation_test(a, b, n_permutations=99, seed=7)
        self.assertAlmostEqual(boot["delta_mean"], 1.0, places=6)
        self.assertEqual(boot["ci_low"], 1.0)
        self.assertEqual(boot["ci_high"], 1.0)
        self.assertAlmostEqual(perm["delta_mean"], 1.0, places=6)
        self.assertGreaterEqual(perm["p_value"], 0.0)
        self.assertLessEqual(perm["p_value"], 1.0)

    def test_efficiency_helpers(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        self.assertEqual(count_parameters(model), 26)
        self.assertEqual(count_parameters(model, trainable_only=True), 26)
        self.assertGreater(model_size_mb(model), 0.0)
        x = torch.zeros(2, 3)
        self.assertGreater(tensor_memory_mb(x), 0.0)
        summary = efficiency_summary(model)
        self.assertEqual(summary["params"], 26.0)


if __name__ == "__main__":
    unittest.main()