"""统一配置构造工具。

该模块把 YAML 配置转换为训练、评估、推理共用的结构化字典，避免
`train.py`、`evaluate.py`、`benchmark_inference.py` 各自维护一套不一致逻辑。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


def deep_update(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """递归合并字典，返回新对象。"""

    out = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _as_list(value: Optional[Iterable[Any]]) -> list:
    return list(value or [])


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)


def _data_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("data", {}) or {})


def _get_data_value(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    data = _data_section(cfg)
    if key in data:
        return data[key]
    return cfg.get(key, default)


def _get_static_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_schema = dict(_data_section(cfg).get("static_schema", {}) or {})
    root_schema = dict(cfg.get("static_schema", {}) or {})
    return deep_update(data_schema, root_schema)


def _get_ablation_bool(ablation: Dict[str, Any], keys: Iterable[str], default: bool = False) -> bool:
    for key in keys:
        if key in ablation:
            return _as_bool(ablation.get(key), default)
    return bool(default)


def build_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dyn = _as_list(_get_data_value(cfg, "dynamic_vars"))
    stat = _as_list(_get_data_value(cfg, "static_vars"))
    include_static = _as_bool(_get_data_value(cfg, "include_static", True), True)
    k = int(_get_data_value(cfg, "k", 1))

    model = dict(cfg.get("model", {}) or {})
    ablation = dict(cfg.get("ablation", {}) or {})
    urban_graph = dict(cfg.get("urban_graph", {}) or {})
    morphology_graph = dict(cfg.get("morphology_graph", {}) or {})
    micromet_coupling = dict(cfg.get("micromet_coupling", {}) or {})
    morpho_process = dict(cfg.get("morpho_process", {}) or {})
    urban_canopy = dict(cfg.get("urban_canopy", {}) or {})
    static_schema = _get_static_schema(cfg)

    drop_path_rate = float(model.get("drop_path_rate", 0.0))
    if not _as_bool(ablation.get("use_drop_path", True), True):
        drop_path_rate = 0.0

    return {
        "H": int(_get_data_value(cfg, "H", cfg.get("H", 8))),
        "W": int(_get_data_value(cfg, "W", cfg.get("W", 8))),
        "D": int(model.get("D", 512)),
        "depth": int(model.get("depth", 12)),
        "heads": int(model.get("heads", 8)),
        "mlp_ratio": float(model.get("mlp_ratio", 4.0)),
        "drop_path_rate": drop_path_rate,
        "dropout": float(model.get("dropout", 0.0)),
        "in_channels": len(dyn),
        "ctx_channels": len(dyn) * k + (len(stat) if include_static else 0),
        "static_channels": len(stat) if include_static else 0,
        "out_channels": len(dyn),
        "dynamic_vars": dyn,
        "static_vars": stat,
        "static_schema": static_schema,
        "use_cross_attention": _as_bool(ablation.get("use_cross_attention", True), True),
        "use_position_encoding": _as_bool(ablation.get("use_position_encoding", True), True),
        "use_timestep_conditioning": _as_bool(ablation.get("use_timestep_conditioning", True), True),
        "use_lead_time_conditioning": _as_bool(ablation.get("use_lead_time_conditioning", False), False),
        "use_hybrid_attention": _get_ablation_bool(ablation, ["use_hybrid_attention"], False),
        "use_multi_scale": _get_ablation_bool(ablation, ["use_multi_scale", "use_multiscale_fusion"], False),
        "use_variable_graph": _get_ablation_bool(ablation, ["use_variable_graph", "use_variable_channel_attention"], False),
        "use_proxy_conditioned_vg": _get_ablation_bool(ablation, ["use_proxy_conditioned_vg", "use_proxy_conditioned_variable_graph"], False),
        "vg_num_heads": int(ablation.get("vg_num_heads", model.get("vg_num_heads", 4))),
        "use_static_morphology_encoder": _get_ablation_bool(ablation, ["use_static_morphology_encoder"], False),
        "use_dynamic_vg": _get_ablation_bool(
            ablation,
            ["use_dynamic_vg"],
            _get_ablation_bool(ablation, ["use_variable_graph", "use_variable_channel_attention"], False),
        ),
        "use_static_vg": _get_ablation_bool(
            ablation,
            ["use_static_vg"],
            _get_ablation_bool(ablation, ["use_variable_graph", "use_variable_channel_attention"], False),
        ),
        "use_urban_graph": _get_ablation_bool(ablation, ["use_urban_graph", "use_urban_spatial_graph"], False),
        "use_legacy_urban_graph": _get_ablation_bool(
            ablation,
            ["use_legacy_urban_graph"],
            _get_ablation_bool(ablation, ["use_urban_graph", "use_urban_spatial_graph"], False),
        ),
        "use_morphology_graph": _as_bool(ablation.get("use_morphology_graph", False), False),
        "use_wind_aware_graph": _as_bool(ablation.get("use_wind_aware_graph", False), False),
        "use_urban_canopy_coupling": _get_ablation_bool(
            ablation,
            ["use_urban_canopy_coupling", "use_urban_canopy"],
            False,
        ),
        "use_urban_canopy": _get_ablation_bool(
            ablation,
            ["use_urban_canopy"],
            _get_ablation_bool(ablation, ["use_urban_canopy_coupling"], False),
        ),
        "use_micromet_coupling": _as_bool(ablation.get("use_micromet_coupling", False), False),
        "use_process_proxy_encoder": _as_bool(ablation.get("use_process_proxy_encoder", False), False),
        "use_process_adaln": _as_bool(ablation.get("use_process_adaln", False), False),
        "use_urban_control_branch": _as_bool(ablation.get("use_urban_control_branch", False), False),
        "use_anisotropic_process_graph": _as_bool(ablation.get("use_anisotropic_process_graph", False), False),
        "use_micromet_token_branches": _as_bool(ablation.get("use_micromet_token_branches", False), False),
        "use_morphology_residual_head": _as_bool(ablation.get("use_morphology_residual_head", False), False),
        "degrade_to_v52": _as_bool(ablation.get("degrade_to_v52", False), False),
        "morpho_process_cfg": morpho_process,
        "canopy_coupling_mode": str(ablation.get("canopy_coupling_mode", "pre_transformer")),
        "urban_graph_k": int(urban_graph.get("k", morphology_graph.get("k", 8))),
        "urban_graph_sigma": float(urban_graph.get("sigma", morphology_graph.get("sigma", 1.0))),
        "morphology_graph_cfg": morphology_graph,
        "micromet_coupling_cfg": micromet_coupling,
        "urban_canopy_cfg": urban_canopy,
    }


def build_optim_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train = dict(cfg.get("train", {}) or {})
    ablation = dict(cfg.get("ablation", {}) or {})
    losses = dict(cfg.get("losses", {}) or {})
    recon = dict(losses.get("reconstruction", {}) or {})

    loss_weights = recon.get("weights", train.get("loss_weights", None))
    out = {
        "lr": float(train.get("lr", 5e-4)),
        "weight_decay": float(train.get("weight_decay", 1e-4)),
        "max_epochs": int(train.get("max_epochs", 200)),
        "loss_weights": loss_weights,
        "use_channel_weights": _as_bool(recon.get("use_channel_weights", ablation.get("use_channel_weights", True)), True),
        "use_adaptive_weights": _as_bool(ablation.get("use_adaptive_weights", False), False),
    }
    if out["loss_weights"] is None:
        out.pop("loss_weights")
    return out


def build_diffusion_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("diffusion", {}) or {})


def build_physics_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ablation = dict(cfg.get("ablation", {}) or {})
    legacy = dict(cfg.get("physics", {}) or {})
    losses = dict(cfg.get("losses", {}) or {})

    feasibility = dict(losses.get("feasibility", {}) or {})
    structure = dict(losses.get("structure", {}) or {})
    process = dict(losses.get("process", {}) or {})
    process_consistency = dict(losses.get("process_consistency", {}) or {})
    if process_consistency:
        process = deep_update(process, process_consistency)

    out = dict(legacy)
    out.update(
        {
            "use_physics_loss": _as_bool(feasibility.get("enabled", ablation.get("use_physics_loss", True)), True),
            "use_feasibility_loss": _as_bool(feasibility.get("enabled", ablation.get("use_physics_loss", True)), True),
            "use_fft_loss": _as_bool(structure.get("fft_enabled", ablation.get("use_fft_loss", True)), True),
            "use_gradient_loss": _as_bool(structure.get("grad_enabled", ablation.get("use_gradient_loss", True)), True),
            "use_structure_loss": _as_bool(structure.get("enabled", True), True),
            "use_process_consistency": _as_bool(
                ablation.get("use_process_consistency_loss", process.get("enabled", False)),
                False,
            ),
            "use_hard_physics": _as_bool(ablation.get("use_hard_physics", False), False),
            "use_adversarial": _as_bool(ablation.get("use_adversarial", False), False),
            "use_distillation": _as_bool(ablation.get("use_distillation", False), False),
            "lambda": float(feasibility.get("lambda", legacy.get("lambda", 0.1))),
            "lambda_fft": float(structure.get("lambda_fft", legacy.get("lambda_fft", 0.0))),
            "lambda_grad": float(structure.get("lambda_grad", legacy.get("lambda_grad", 0.0))),
            "lambda_process_rh": float(process.get("lambda_rh", legacy.get("lambda_process_rh", 0.0))),
            "lambda_process_drag": float(process.get("lambda_drag", legacy.get("lambda_process_drag", 0.0))),
            "lambda_process_diurnal": float(process.get("lambda_diurnal", legacy.get("lambda_process_diurnal", 0.0))),
            "lambda_process_proxy": float(process.get("lambda", legacy.get("lambda_process_proxy", 0.0))),
            "lambda_proxy_roughness_wind": float(
                process.get("lambda_roughness_wind", legacy.get("lambda_proxy_roughness_wind", 0.0))
            ),
            "lambda_proxy_diurnal_building": float(
                process.get("lambda_diurnal_building", legacy.get("lambda_proxy_diurnal_building", 0.0))
            ),
            "lambda_proxy_impervious_d2m": float(
                process.get("lambda_impervious_d2m", legacy.get("lambda_proxy_impervious_d2m", 0.0))
            ),
            "lambda_proxy_wind_tcc": float(process.get("lambda_wind_tcc", legacy.get("lambda_proxy_wind_tcc", 0.0))),
        }
    )
    return out


def build_metrics_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg.get("metrics", {}) or {})
    out["var_names"] = _as_list(_get_data_value(cfg, "dynamic_vars"))
    return out


def build_forecast_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("forecast", {}) or {})


def build_inference_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("inference", {}) or {})


def build_litmodule_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_cfg": build_model_cfg(cfg),
        "optim_cfg": build_optim_cfg(cfg),
        "diffusion_cfg": build_diffusion_cfg(cfg),
        "physics_cfg": build_physics_cfg(cfg),
        "metrics_cfg": build_metrics_cfg(cfg),
        "inference_cfg": build_inference_cfg(cfg),
        "forecast_cfg": build_forecast_cfg(cfg),
    }


def build_datamodule_kwargs(cfg: Dict[str, Any], *, batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> Dict[str, Any]:
    forecast = build_forecast_cfg(cfg)
    train = dict(cfg.get("train", {}) or {})
    return {
        "data_root": _get_data_value(cfg, "data_root"),
        "k": int(_get_data_value(cfg, "k")),
        "delta_t": int(_get_data_value(cfg, "delta_t")),
        "lead_times": forecast.get("lead_times", None),
        "batch_size": int(batch_size if batch_size is not None else train.get("batch_size", 1)),
        "num_workers": int(num_workers if num_workers is not None else train.get("num_workers", 8)),
        "dynamic_vars": _as_list(_get_data_value(cfg, "dynamic_vars")),
        "static_vars": _as_list(_get_data_value(cfg, "static_vars")),
        "include_static": _as_bool(_get_data_value(cfg, "include_static", True), True),
        "broadcast_static": _as_bool(_get_data_value(cfg, "broadcast_static", False), False),
        "static_perturb": _get_data_value(cfg, "static_perturb", None),
        "dynamic_perturb": _get_data_value(cfg, "dynamic_perturb", None),
        "static_schema": _get_static_schema(cfg),
        "expose_hour": _as_bool(_get_data_value(cfg, "expose_hour", False), False),
    }
