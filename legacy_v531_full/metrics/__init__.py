"""评估指标子模块。"""

from .acc import acc
from .crps import crps_ensemble
from .deterministic import bias, deterministic_metrics, mae, mse, pearson_corr, rmse
from .efficiency import count_parameters, efficiency_summary, measure_inference_latency_ms, model_size_mb, tensor_memory_mb
from .event_metrics import contingency_table, event_mask, event_scores
from .probabilistic import (
    brier_score,
    ensemble_crps,
    ensemble_mean,
    ensemble_spread,
    probabilistic_metrics,
    probability_of_event,
    spread_skill_ratio,
)
from .significance import paired_bootstrap_delta, paired_permutation_test

__all__ = [
    "rmse",
    "mae",
    "mse",
    "bias",
    "pearson_corr",
    "deterministic_metrics",
    "crps_ensemble",
    "acc",
    "ensemble_mean",
    "ensemble_spread",
    "ensemble_crps",
    "probability_of_event",
    "brier_score",
    "spread_skill_ratio",
    "probabilistic_metrics",
    "event_mask",
    "contingency_table",
    "event_scores",
    "paired_bootstrap_delta",
    "paired_permutation_test",
    "count_parameters",
    "model_size_mb",
    "tensor_memory_mb",
    "measure_inference_latency_ms",
    "efficiency_summary",
]
