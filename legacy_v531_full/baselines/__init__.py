"""Baselines for UrbanPiDiT V5.3.1.

The package keeps the original trainable baselines (linear/tree/MLP),
shape-safe forecast baselines with explicit static-information policies, and
loader-compatible FourCastNet/GraphCast/GenCast adapters for stronger weather
forecasting comparisons.
"""


def run_linear_baseline(*args, **kwargs):
    from .linear import run_linear_baseline as _run

    return _run(*args, **kwargs)


def run_tree_baseline(*args, **kwargs):
    from .tree import run_tree_baseline as _run

    return _run(*args, **kwargs)


def run_mlp_baseline(*args, **kwargs):
    from .mlp import run_mlp_baseline as _run

    return _run(*args, **kwargs)


def train_external_baselines(*args, **kwargs):
    from .train_external_baselines import train_external_baselines as _run

    return _run(*args, **kwargs)


def build_forecast_baseline(*args, **kwargs):
    from .forecast_models import build_forecast_baseline as _build

    return _build(*args, **kwargs)


def build_baselines_from_config(*args, **kwargs):
    from .forecast_models import build_baselines_from_config as _build

    return _build(*args, **kwargs)


def available_baselines(*args, **kwargs):
    from .forecast_models import available_baselines as _available

    return _available(*args, **kwargs)


__all__ = [
    "run_linear_baseline",
    "run_tree_baseline",
    "run_mlp_baseline",
    "train_external_baselines",
    "build_forecast_baseline",
    "build_baselines_from_config",
    "available_baselines",
]
