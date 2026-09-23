from __future__ import annotations

from typing import Dict, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forecast_base import ForecastBatchView, ForecastModelBase, ForecastModelSpec, TensorDict

try:
    from .external_models import UrbanCorrDiffAdapter, UrbanFourCastNetAdapter, UrbanGenCastAdapter, UrbanGraphCastAdapter
except Exception:  # pragma: no cover - adapters are optional during minimal installs
    UrbanCorrDiffAdapter = None  # type: ignore
    UrbanFourCastNetAdapter = None  # type: ignore
    UrbanGenCastAdapter = None  # type: ignore
    UrbanGraphCastAdapter = None  # type: ignore

try:
    from .faithful import FaithfulCorrDiff, FaithfulFourCastNet, FaithfulGenCast, FaithfulGraphCast
except Exception:  # pragma: no cover - faithful baselines are optional during partial imports
    FaithfulCorrDiff = None  # type: ignore
    FaithfulFourCastNet = None  # type: ignore
    FaithfulGenCast = None  # type: ignore
    FaithfulGraphCast = None  # type: ignore


class PersistenceForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="persistence",
        description="copy the latest historical field to every lead",
        uses_static=False,
        trainable=False,
        family="deterministic_naive",
    )

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        return view.last


class ClimatologyForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="climatology",
        description="use train-only climatological mean field when available",
        uses_static=False,
        trainable=False,
        family="deterministic_naive",
    )

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        return self.climatology_norm(view)


class LinearTrendForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="linear_trend",
        description="extrapolate the latest historical tendency by lead steps",
        uses_static=False,
        trainable=False,
        family="deterministic_trend",
    )

    def __init__(self, *args, trend_scale: float = 1.0, max_scale: float = 4.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trend_scale = float(trend_scale)
        self.max_scale = float(max_scale)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(lead_step, device=view.last.device, dtype=view.last.dtype).clamp_min(1.0)
        scale = torch.clamp(scale * self.trend_scale, max=self.max_scale)
        return view.last + scale * (view.last - view.previous)


class MovingAverageForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="moving_average",
        description="average the available historical fields",
        uses_static=False,
        trainable=False,
        family="deterministic_smoother",
    )

    def __init__(self, *args, window: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window = None if window is None else max(1, int(window))

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        values = view.dynamic if self.window is None else view.dynamic[:, :, -self.window :]
        return values.mean(dim=2)


class ExponentialSmoothingForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="exp_smoothing",
        description="exponentially weighted historical average favouring recent fields",
        uses_static=False,
        trainable=False,
        family="deterministic_smoother",
    )

    def __init__(self, *args, alpha: float = 0.65, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        k = view.dynamic.shape[2]
        alpha = min(max(self.alpha, 1e-4), 0.9999)
        powers = torch.arange(k - 1, -1, -1, device=view.last.device, dtype=view.last.dtype)
        weights = (1.0 - alpha) ** powers
        weights = weights / weights.sum().clamp_min(1e-6)
        return (view.dynamic * weights.view(1, 1, k, 1, 1)).sum(dim=2)


class ResidualClimatologyForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="residual_climatology",
        description="blend latest anomaly persistence with train-only climatology",
        uses_static=False,
        trainable=False,
        family="deterministic_climatology",
    )

    def __init__(self, *args, decay: float = 0.75, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decay = float(decay)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        clim = self.climatology_norm(view)
        lead = torch.as_tensor(lead_step, device=view.last.device, dtype=view.last.dtype).clamp_min(1.0)
        weight = torch.clamp(torch.as_tensor(self.decay, device=view.last.device, dtype=view.last.dtype), 0.0, 1.0) ** lead
        return clim + weight * (view.last - clim)


class SpatialMeanForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="spatial_mean",
        description="use the latest domain mean for each dynamic variable",
        uses_static=False,
        trainable=False,
        family="deterministic_spatial",
    )

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        return view.last.mean(dim=(-2, -1), keepdim=True).expand_as(view.last)


class LocalMeanForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="local_mean",
        description="use a local spatial smoother over the latest field",
        uses_static=False,
        trainable=False,
        family="deterministic_spatial",
    )

    def __init__(self, *args, kernel_size: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = max(1, int(kernel_size))

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return view.last
        pad = self.kernel_size // 2
        return F.avg_pool2d(view.last, kernel_size=self.kernel_size, stride=1, padding=pad)


class StaticAnalogForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="static_analog",
        description="modulate latest field with a static morphology contrast map",
        uses_static=True,
        trainable=False,
        family="static_control",
    )

    def __init__(self, *args, strength: float = 0.05, **kwargs) -> None:
        super().__init__(*args, static_policy=kwargs.pop("static_policy", "same_static"), **kwargs)
        self.strength = float(strength)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        if view.static.shape[1] == 0:
            return view.last
        s = view.static.float().mean(dim=1, keepdim=True)
        s = s - s.mean(dim=(-2, -1), keepdim=True)
        scale = s / s.flatten(2).std(dim=-1, keepdim=True).view(s.shape[0], 1, 1, 1).clamp_min(1e-6)
        return view.last + self.strength * scale.expand_as(view.last)


class StaticPersistenceBlendForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="static_persistence_blend",
        description="blend persistence and local smoothing with a static morphology gate",
        uses_static=True,
        trainable=False,
        family="static_control",
    )

    def __init__(self, *args, kernel_size: int = 3, **kwargs) -> None:
        super().__init__(*args, static_policy=kwargs.pop("static_policy", "same_static"), **kwargs)
        self.kernel_size = max(1, int(kernel_size))

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            smooth = view.last
        else:
            smooth = F.avg_pool2d(view.last, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        if view.static.shape[1] == 0:
            return 0.5 * view.last + 0.5 * smooth
        gate = torch.sigmoid(view.static.float().mean(dim=1, keepdim=True))
        return gate * view.last + (1.0 - gate) * smooth


class ConvPersistenceForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="conv_persistence",
        description="depthwise spatial correction initialized as identity persistence",
        uses_static=False,
        trainable=True,
        family="trainable_lightweight",
    )

    def __init__(self, *args, kernel_size: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = max(1, int(kernel_size))
        self.conv = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.out_channels,
            bias=True,
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        return view.last + self.conv(view.last)


class StaticConvForecast(ForecastModelBase):
    spec = ForecastModelSpec(
        name="static_conv",
        description="single convolution over dynamic history plus controlled static channels",
        uses_static=True,
        trainable=True,
        family="trainable_lightweight",
    )

    def __init__(self, *args, hidden_channels: int = 32, **kwargs) -> None:
        super().__init__(*args, static_policy=kwargs.pop("static_policy", "same_static"), **kwargs)
        hidden_channels = max(1, int(hidden_channels))
        self.net = nn.Sequential(
            nn.Conv2d(self.feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forecast_lead(self, view: ForecastBatchView, lead_step: torch.Tensor) -> torch.Tensor:
        return view.last + self.net(self.make_features(view))


BASELINE_REGISTRY: Dict[str, Type[ForecastModelBase]] = {
    PersistenceForecast.spec.name: PersistenceForecast,
    ClimatologyForecast.spec.name: ClimatologyForecast,
    LinearTrendForecast.spec.name: LinearTrendForecast,
    MovingAverageForecast.spec.name: MovingAverageForecast,
    ExponentialSmoothingForecast.spec.name: ExponentialSmoothingForecast,
    ResidualClimatologyForecast.spec.name: ResidualClimatologyForecast,
    SpatialMeanForecast.spec.name: SpatialMeanForecast,
    LocalMeanForecast.spec.name: LocalMeanForecast,
    StaticAnalogForecast.spec.name: StaticAnalogForecast,
    StaticPersistenceBlendForecast.spec.name: StaticPersistenceBlendForecast,
    ConvPersistenceForecast.spec.name: ConvPersistenceForecast,
    StaticConvForecast.spec.name: StaticConvForecast,
}

if UrbanFourCastNetAdapter is not None:
    BASELINE_REGISTRY[UrbanFourCastNetAdapter.spec.name] = UrbanFourCastNetAdapter
if UrbanGraphCastAdapter is not None:
    BASELINE_REGISTRY[UrbanGraphCastAdapter.spec.name] = UrbanGraphCastAdapter
if UrbanGenCastAdapter is not None:
    BASELINE_REGISTRY[UrbanGenCastAdapter.spec.name] = UrbanGenCastAdapter
if UrbanCorrDiffAdapter is not None:
    BASELINE_REGISTRY[UrbanCorrDiffAdapter.spec.name] = UrbanCorrDiffAdapter
if FaithfulFourCastNet is not None:
    BASELINE_REGISTRY[FaithfulFourCastNet.spec.name] = FaithfulFourCastNet
if FaithfulGraphCast is not None:
    BASELINE_REGISTRY[FaithfulGraphCast.spec.name] = FaithfulGraphCast
if FaithfulGenCast is not None:
    BASELINE_REGISTRY[FaithfulGenCast.spec.name] = FaithfulGenCast
if FaithfulCorrDiff is not None:
    BASELINE_REGISTRY[FaithfulCorrDiff.spec.name] = FaithfulCorrDiff


DEFAULT_BASELINE_NAMES = [
    "persistence",
    "climatology",
    "linear_trend",
    "moving_average",
    "exp_smoothing",
    "residual_climatology",
    "spatial_mean",
    "local_mean",
    "static_analog",
    "static_persistence_blend",
]


def available_baselines() -> Dict[str, ForecastModelSpec]:
    return {name: cls.spec for name, cls in BASELINE_REGISTRY.items()}


def build_forecast_baseline(
    name: str,
    *,
    dynamic_vars: Sequence[str],
    static_vars: Sequence[str] | None = None,
    k: int = 1,
    lead_times: Sequence[int] | None = None,
    static_policy: str | None = None,
    params: Dict | None = None,
) -> ForecastModelBase:
    key = str(name).strip().lower()
    if key not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline={name!r}; available={sorted(BASELINE_REGISTRY)}")
    cls = BASELINE_REGISTRY[key]
    kwargs = dict(params or {})
    if static_policy is not None:
        kwargs["static_policy"] = static_policy
    return cls(
        dynamic_vars=dynamic_vars,
        static_vars=static_vars,
        k=k,
        lead_times=lead_times,
        **kwargs,
    )


def build_baselines_from_config(cfg: Dict) -> Dict[str, ForecastModelBase]:
    data = dict(cfg.get("data", {}) or {})
    dyn = list(data.get("dynamic_vars", cfg.get("dynamic_vars", [])) or [])
    stat = list(data.get("static_vars", cfg.get("static_vars", [])) or [])
    k = int(data.get("k", cfg.get("k", 1)))
    forecast = dict(cfg.get("forecast", {}) or {})
    lead_times = forecast.get("lead_times", cfg.get("lead_times", None))
    baseline_cfg = dict(cfg.get("baselines", {}) or {})
    static_policy = str(baseline_cfg.get("static_policy", "dynamic_only"))
    items = baseline_cfg.get("models", None)
    if items is None:
        items = [{"name": name} for name in DEFAULT_BASELINE_NAMES]

    out: Dict[str, ForecastModelBase] = {}
    for item in items:
        if isinstance(item, str):
            name = item
            params = {}
            policy = static_policy
            alias = name
        else:
            name = str(item["name"])
            params = dict(item.get("params", {}) or {})
            policy = str(item.get("static_policy", static_policy))
            alias = str(item.get("alias", name))
            for alias_key in ("prediction_protocol", "protocol", "rollout_mode"):
                if alias_key in params:
                    params.setdefault("forecast_protocol", params.pop(alias_key))
            protocol = item.get("forecast_protocol", item.get("prediction_protocol", item.get("protocol", forecast.get("protocol", None))))
            if protocol is not None:
                params.setdefault("forecast_protocol", protocol)
            if "one_step_lead" in item:
                params.setdefault("one_step_lead", int(item["one_step_lead"]))
            if "time_step_hours" in forecast:
                params.setdefault("time_step_hours", float(forecast["time_step_hours"]))
        out[alias] = build_forecast_baseline(
            name,
            dynamic_vars=dyn,
            static_vars=stat,
            k=k,
            lead_times=lead_times,
            static_policy=policy,
            params=params,
        )
    return out


__all__ = [
    "BASELINE_REGISTRY",
    "DEFAULT_BASELINE_NAMES",
    "PersistenceForecast",
    "ClimatologyForecast",
    "LinearTrendForecast",
    "MovingAverageForecast",
    "ExponentialSmoothingForecast",
    "ResidualClimatologyForecast",
    "SpatialMeanForecast",
    "LocalMeanForecast",
    "StaticAnalogForecast",
    "StaticPersistenceBlendForecast",
    "ConvPersistenceForecast",
    "StaticConvForecast",
    "UrbanGenCastAdapter",
    "UrbanGraphCastAdapter",
    "UrbanFourCastNetAdapter",
    "UrbanCorrDiffAdapter",
    "FaithfulFourCastNet",
    "FaithfulGraphCast",
    "FaithfulGenCast",
    "FaithfulCorrDiff",
    "available_baselines",
    "build_forecast_baseline",
    "build_baselines_from_config",
]