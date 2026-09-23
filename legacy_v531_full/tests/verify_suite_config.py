r"""UrbanPiDiT V5.3.1 suite 配置完整性验证脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_registry import build_model_from_config
from utils.config_builder import build_datamodule_kwargs, build_model_cfg, build_physics_cfg, deep_update

EXPECTED_EXPERIMENT_COUNT = 29

ABLATION_KEYS = (
    "use_drop_path",
    "use_cross_attention",
    "use_position_encoding",
    "use_timestep_conditioning",
    "use_lead_time_conditioning",
    "use_static_morphology_encoder",
    "use_morphology_graph",
    "use_wind_aware_graph",
    "use_micromet_coupling",
    "use_process_proxy_encoder",
    "use_process_adaln",
    "use_urban_control_branch",
    "use_anisotropic_process_graph",
    "use_micromet_token_branches",
    "use_morphology_residual_head",
    "use_process_consistency_loss",
    "degrade_to_v52",
    "use_urban_canopy_coupling",
    "use_urban_canopy",
    "use_urban_graph",
    "use_hybrid_attention",
    "use_multi_scale",
    "use_variable_graph",
    "use_dynamic_vg",
    "use_static_vg",
    "use_legacy_urban_graph",
)

FULL_FLAGS: dict[str, bool] = {
    "use_drop_path": True,
    "use_cross_attention": True,
    "use_position_encoding": True,
    "use_timestep_conditioning": True,
    "use_lead_time_conditioning": True,
    "use_static_morphology_encoder": True,
    "use_morphology_graph": True,
    "use_wind_aware_graph": True,
    "use_micromet_coupling": True,
    "use_process_proxy_encoder": True,
    "use_process_adaln": True,
    "use_urban_control_branch": True,
    "use_anisotropic_process_graph": True,
    "use_micromet_token_branches": True,
    "use_morphology_residual_head": True,
    "use_process_consistency_loss": True,
    "degrade_to_v52": False,
    "use_urban_canopy_coupling": False,
    "use_urban_canopy": False,
    "use_urban_graph": False,
    "use_hybrid_attention": False,
    "use_multi_scale": False,
    "use_variable_graph": False,
    "use_dynamic_vg": False,
    "use_static_vg": False,
    "use_legacy_urban_graph": False,
}

MODULE_FLAGS = (
    "use_static_morphology_encoder",
    "use_morphology_graph",
    "use_wind_aware_graph",
    "use_micromet_coupling",
    "use_process_proxy_encoder",
    "use_process_adaln",
    "use_urban_control_branch",
    "use_anisotropic_process_graph",
    "use_micromet_token_branches",
    "use_morphology_residual_head",
    "use_process_consistency_loss",
)

CUMULATIVE_FLAGS: dict[str, dict[str, bool]] = {
    "cumulative_v4_base": dict.fromkeys(MODULE_FLAGS, False),
    "cumulative_v5_static_enc": {**dict.fromkeys(MODULE_FLAGS, False), "use_static_morphology_encoder": True},
    "cumulative_v5_morph_graph": {
        **dict.fromkeys(MODULE_FLAGS, False),
        "use_static_morphology_encoder": True,
        "use_morphology_graph": True,
    },
    "cumulative_v5_wind_graph": {
        **dict.fromkeys(MODULE_FLAGS, False),
        "use_static_morphology_encoder": True,
        "use_morphology_graph": True,
        "use_wind_aware_graph": True,
    },
    "cumulative_v51_micromet": {
        **dict.fromkeys(MODULE_FLAGS, False),
        "use_static_morphology_encoder": True,
        "use_morphology_graph": True,
        "use_wind_aware_graph": True,
        "use_micromet_coupling": True,
    },
    "cumulative_v52_process_loss": {
        **dict.fromkeys(MODULE_FLAGS, False),
        "use_static_morphology_encoder": True,
        "use_morphology_graph": True,
        "use_wind_aware_graph": True,
        "use_micromet_coupling": True,
        "use_process_consistency_loss": True,
    },
    "cumulative_v53_proxy_adaln": {
        **dict.fromkeys(MODULE_FLAGS, False),
        "use_static_morphology_encoder": True,
        "use_morphology_graph": True,
        "use_wind_aware_graph": True,
        "use_micromet_coupling": True,
        "use_process_proxy_encoder": True,
        "use_process_adaln": True,
        "use_urban_control_branch": True,
        "use_process_consistency_loss": True,
    },
    "cumulative_v531_full": {key: FULL_FLAGS[key] for key in MODULE_FLAGS},
}

CORE_OVERRIDES: dict[str, dict[str, bool]] = {
    "ablation_no_proxy_encoder": {
        "use_process_proxy_encoder": False,
        "use_process_adaln": False,
        "use_urban_control_branch": False,
    },
    "ablation_proxy_encoder_alone": {
        "use_process_adaln": False,
        "use_urban_control_branch": False,
        "use_anisotropic_process_graph": False,
        "use_micromet_token_branches": False,
        "use_morphology_residual_head": False,
        "use_process_consistency_loss": False,
    },
    "ablation_no_process_adaln": {"use_process_adaln": False},
    "ablation_no_urban_control": {"use_urban_control_branch": False},
    "ablation_no_aniso_graph": {"use_anisotropic_process_graph": False},
    "ablation_no_graph_all": {
        "use_morphology_graph": False,
        "use_wind_aware_graph": False,
        "use_anisotropic_process_graph": False,
    },
    "ablation_no_micromet_tokens": {"use_micromet_token_branches": False},
    "ablation_no_residual_head": {"use_morphology_residual_head": False},
    "ablation_no_process_consistency": {"use_process_consistency_loss": False},
}

STATIC_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "static_same": {"include_static": True, "static_mode": "none", "dynamic_mode": "none"},
    "static_dynamic_only": {
        "include_static": False,
        "static_mode": "none",
        "dynamic_mode": "none",
        "static_vars_len": 0,
    },
    "static_zero": {"include_static": True, "static_mode": "fill_zero", "dynamic_mode": "none"},
    "static_shuffle": {"include_static": True, "static_mode": "shuffle_spatial", "dynamic_mode": "none"},
}

COUNTERFACTUAL_MODES: dict[str, dict[str, str]] = {
    "counterfactual_zero_static": {"static_mode": "fill_zero", "dynamic_mode": "none"},
    "counterfactual_no_building_vol": {"static_mode": "remove_building_volume", "dynamic_mode": "none"},
    "counterfactual_no_population": {"static_mode": "remove_population", "dynamic_mode": "none"},
    "counterfactual_rotate_wind": {"static_mode": "none", "dynamic_mode": "rotate_wind_90"},
    "counterfactual_permute_lc": {"static_mode": "permute_landcover", "dynamic_mode": "none"},
}

LOSS_DISABLED = {"feasibility": False, "structure": False, "process": False}
LOSS_ENABLED = {"feasibility": True, "structure": True, "process": True}
FORWARD_ALIASES = (
    "cumulative_v4_base",
    "cumulative_v531_full",
    "ablation_no_proxy_encoder",
    "ablation_no_process_adaln",
    "ablation_no_urban_control",
    "static_dynamic_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify UrbanPiDiT V5.3.1 suite configuration")
    parser.add_argument("--suite", default="configs/urbanpidit_v531_suite.yaml", help="suite YAML 路径")
    parser.add_argument("--report", default="tests/verification_report.json", help="JSON 报告输出路径")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return dict(data or {})


def resolve_path(raw_path: str, suite_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    root = suite_path.parent.parent if suite_path.parent.name == "configs" else suite_path.parent
    for candidate in (Path.cwd() / path, suite_path.parent / path, root / path):
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def resolve_suite(suite_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    suite_cfg = read_yaml(suite_path)
    base_path = resolve_path(str(suite_cfg["base"]), suite_path)
    base_cfg = deep_update(read_yaml(base_path), suite_cfg.get("base_overrides", {}))
    base_cfg = deep_update(base_cfg, {"train": {"two_stage": suite_cfg.get("two_stage", {})}})
    resolved = {}
    for experiment in suite_cfg.get("experiments", []):
        alias = str(experiment["alias"])
        resolved[alias] = deep_update(base_cfg, experiment.get("overrides", {}))
    return suite_cfg, resolved


def expected_flags(alias: str) -> dict[str, bool]:
    flags = dict(FULL_FLAGS)
    if alias in CUMULATIVE_FLAGS:
        flags.update(CUMULATIVE_FLAGS[alias])
    if alias in CORE_OVERRIDES:
        flags.update(CORE_OVERRIDES[alias])
    if alias == "static_dynamic_only":
        flags.update({key: False for key in MODULE_FLAGS})
    if alias == "ablation_no_process_consistency":
        flags["use_process_consistency_loss"] = False
    return flags


def expected_losses(alias: str) -> dict[str, bool]:
    if alias in {
        "cumulative_v4_base",
        "cumulative_v5_static_enc",
        "cumulative_v5_morph_graph",
        "cumulative_v5_wind_graph",
        "cumulative_v51_micromet",
    }:
        return dict(LOSS_DISABLED)
    if alias in {"ablation_no_process_consistency", "ablation_proxy_encoder_alone", "static_dynamic_only"}:
        return {"feasibility": True, "structure": True, "process": False}
    if alias == "ablation_no_feasibility_only":
        return {"feasibility": False, "structure": True, "process": True}
    if alias == "ablation_no_structure_only":
        return {"feasibility": True, "structure": False, "process": True}
    if alias == "ablation_no_process_proxy_only":
        return {"feasibility": True, "structure": True, "process": False}
    return dict(LOSS_ENABLED)


def expected_policy(alias: str) -> dict[str, Any]:
    policy: dict[str, Any] = {"include_static": True, "static_mode": "none", "dynamic_mode": "none"}
    if alias in STATIC_EXPECTATIONS:
        policy.update(STATIC_EXPECTATIONS[alias])
    if alias in COUNTERFACTUAL_MODES:
        policy.update(COUNTERFACTUAL_MODES[alias])
    return policy


def record(checks: list[dict[str, Any]], layer: str, alias: str, key: str, expected: Any, actual: Any) -> None:
    checks.append(
        {
            "layer": layer,
            "alias": alias,
            "key": key,
            "expected": expected,
            "actual": actual,
            "ok": expected == actual,
        }
    )


def mode_of(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    return str(value.get("mode", "none"))


def ablation_value(cfg: Mapping[str, Any], key: str) -> bool:
    ablation = cfg.get("ablation", {})
    if not isinstance(ablation, Mapping):
        return False
    return bool(ablation.get(key, False))


def verify_suite_structure(checks: list[dict[str, Any]], suite_cfg: Mapping[str, Any]) -> None:
    experiments = list(suite_cfg.get("experiments", []))
    aliases = [str(item.get("alias", "")) for item in experiments]
    record(checks, "layer0_suite", "__suite__", "experiment_count", EXPECTED_EXPERIMENT_COUNT, len(aliases))
    record(checks, "layer0_suite", "__suite__", "aliases_unique", True, len(set(aliases)) == len(aliases))
    record(checks, "layer0_suite", "__suite__", "two_stage.enabled", True, bool(suite_cfg.get("two_stage", {}).get("enabled")))


def verify_yaml_layer(checks: list[dict[str, Any]], resolved: Mapping[str, dict[str, Any]]) -> None:
    for alias, cfg in resolved.items():
        flags = expected_flags(alias)
        losses = expected_losses(alias)
        policy = expected_policy(alias)
        for key in ABLATION_KEYS:
            record(checks, "layer1_yaml", alias, f"ablation.{key}", flags[key], ablation_value(cfg, key))
        for key, expected in losses.items():
            record(checks, "layer1_yaml", alias, f"losses.{key}.enabled", expected, bool(cfg["losses"][key]["enabled"]))
        record(checks, "layer1_yaml", alias, "include_static", policy["include_static"], bool(cfg.get("include_static", True)))
        record(checks, "layer1_yaml", alias, "static_perturb.mode", policy["static_mode"], mode_of(cfg.get("static_perturb")))
        record(checks, "layer1_yaml", alias, "dynamic_perturb.mode", policy["dynamic_mode"], mode_of(cfg.get("dynamic_perturb")))
        record(checks, "layer1_yaml", alias, "train.two_stage.enabled", True, bool(cfg.get("train", {}).get("two_stage", {}).get("enabled")))


def verify_builder_layer(checks: list[dict[str, Any]], resolved: Mapping[str, dict[str, Any]]) -> None:
    for alias, cfg in resolved.items():
        model_cfg = build_model_cfg(cfg)
        physics_cfg = build_physics_cfg(cfg)
        data_kwargs = build_datamodule_kwargs(cfg)
        flags = expected_flags(alias)
        losses = expected_losses(alias)
        policy = expected_policy(alias)
        for key in ABLATION_KEYS:
            if key in model_cfg:
                record(checks, "layer2_builder", alias, f"model_cfg.{key}", flags[key], bool(model_cfg[key]))
        record(checks, "layer2_builder", alias, "physics.use_feasibility_loss", losses["feasibility"], bool(physics_cfg["use_feasibility_loss"]))
        record(checks, "layer2_builder", alias, "physics.use_structure_loss", losses["structure"], bool(physics_cfg["use_structure_loss"]))
        record(checks, "layer2_builder", alias, "physics.use_process_consistency", flags["use_process_consistency_loss"], bool(physics_cfg["use_process_consistency"]))
        record(checks, "layer2_builder", alias, "data.include_static", policy["include_static"], bool(data_kwargs["include_static"]))
        record(checks, "layer2_builder", alias, "data.static_perturb.mode", policy["static_mode"], mode_of(data_kwargs["static_perturb"]))
        record(checks, "layer2_builder", alias, "data.dynamic_perturb.mode", policy["dynamic_mode"], mode_of(data_kwargs["dynamic_perturb"]))
        if "static_vars_len" in policy:
            record(checks, "layer2_builder", alias, "model_cfg.static_channels", 0, int(model_cfg["static_channels"]))
            record(checks, "layer2_builder", alias, "model_cfg.ctx_channels", 28, int(model_cfg["ctx_channels"]))


def compact_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return deep_update(
        dict(deepcopy(cfg)),
        {
            "H": 4,
            "W": 4,
            "model": {"D": 32, "depth": 3, "heads": 4, "mlp_ratio": 2.0, "drop_path_rate": 0.0, "dropout": 0.0},
            "morphology_graph": {"k": 2, "enable_cache": False},
            "micromet_coupling": {"hidden_channels": 8, "graph_cfg": {"k": 2, "enable_cache": False}},
            "morpho_process": {
                "proxy_encoder": {"hidden_channels": [8, 16, 32], "use_diurnal_modulation": False},
                "process_adaln": {"proxy_proj_dim": 16, "init_alpha": 0.0},
                "anisotropic_process_graph": {"k": 2, "enable_cache": False},
            },
        },
    )


def build_compact_model(cfg: Mapping[str, Any]) -> Any:
    model_cfg = build_model_cfg(compact_cfg(cfg))
    return build_model_from_config(model_cfg)


def module_presence(model: Any) -> dict[str, bool]:
    return {
        "static_morphology_encoder": model.static_morphology_encoder is not None,
        "morphology_graph": model.morphology_graph is not None,
        "micromet_coupling": model.micromet_coupling is not None,
        "process_proxy_encoder": model.process_proxy_encoder is not None,
        "process_adaln": model.process_adaln is not None,
        "urban_control_branch": model.urban_control_branch is not None,
        "anisotropic_process_graph": model.anisotropic_process_graph is not None,
        "morphology_residual_head": model.morphology_residual_head is not None,
    }


def verify_model_layer(checks: list[dict[str, Any]], resolved: Mapping[str, dict[str, Any]]) -> None:
    for alias, cfg in resolved.items():
        model = build_compact_model(cfg)
        flags = expected_flags(alias)
        for key in ABLATION_KEYS:
            if hasattr(model, key):
                record(checks, "layer3_model", alias, key, flags[key], bool(getattr(model, key)))
        modules = module_presence(model)
        record(checks, "layer3_model", alias, "module.static_morphology_encoder", flags["use_static_morphology_encoder"], modules["static_morphology_encoder"])
        record(checks, "layer3_model", alias, "module.morphology_graph", flags["use_morphology_graph"] or flags["use_wind_aware_graph"], modules["morphology_graph"])
        record(checks, "layer3_model", alias, "module.micromet_coupling", flags["use_micromet_coupling"], modules["micromet_coupling"])
        record(checks, "layer3_model", alias, "module.process_proxy_encoder", flags["use_process_proxy_encoder"], modules["process_proxy_encoder"])
        record(checks, "layer3_model", alias, "module.process_adaln", flags["use_process_adaln"], modules["process_adaln"])
        record(checks, "layer3_model", alias, "module.urban_control_branch", flags["use_urban_control_branch"], modules["urban_control_branch"])
        record(checks, "layer3_model", alias, "module.anisotropic_process_graph", flags["use_anisotropic_process_graph"], modules["anisotropic_process_graph"])
        record(checks, "layer3_model", alias, "module.morphology_residual_head", flags["use_morphology_residual_head"], modules["morphology_residual_head"])


def synthetic_batch(model: Any) -> dict[str, torch.Tensor | None]:
    batch, height, width = 2, model.base_H, model.base_W
    x_t = torch.randn(batch, model.in_channels, height, width)
    dyn = torch.randn(batch, model.in_channels * model.real_k, height, width)
    static_raw = None
    static_cont = None
    static_cat = None
    if model.real_static > 0:
        static_cat = torch.randint(0, 4, (batch, 1, height, width))
        cont_channels = max(model.real_static - 1, 0)
        static_cont = torch.rand(batch, cont_channels, height, width)
        static_raw = torch.cat([static_cat.float(), static_cont], dim=1)
    x_ctx = dyn if static_raw is None else torch.cat([dyn, static_raw], dim=1)
    return {
        "x_t": x_t,
        "x_ctx": x_ctx,
        "t": torch.rand(batch),
        "lead_time": torch.full((batch,), 0.25),
        "static_raw": static_raw,
        "static_cont": static_cont,
        "static_cat": static_cat,
        "hour_of_day": torch.tensor([9.0, 21.0]),
    }


def track_forward(module: Any, calls: dict[str, int], key: str) -> None:
    original = module.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls[key] += 1
        return original(*args, **kwargs)

    module.forward = wrapped


def instrument_model(model: Any) -> dict[str, int]:
    calls = {"proxy_context": 0, "proxy_returned": 0, "adaln": 0, "control": 0, "graph": 0, "residual": 0}
    original_proxy = model._build_process_proxy_context

    def wrapped_proxy(*args: Any, **kwargs: Any) -> Any:
        calls["proxy_context"] += 1
        out = original_proxy(*args, **kwargs)
        calls["proxy_returned"] += int(out[0] is not None)
        return out

    model._build_process_proxy_context = wrapped_proxy
    if model.process_adaln is not None:
        track_forward(model.process_adaln, calls, "adaln")
    if model.urban_control_branch is not None:
        track_forward(model.urban_control_branch, calls, "control")
    if model.anisotropic_process_graph is not None:
        track_forward(model.anisotropic_process_graph, calls, "graph")
    if model.morphology_residual_head is not None:
        track_forward(model.morphology_residual_head, calls, "residual")
    return calls


def verify_forward_layer(checks: list[dict[str, Any]], resolved: Mapping[str, dict[str, Any]]) -> None:
    for alias in FORWARD_ALIASES:
        if alias not in resolved:
            continue
        model = build_compact_model(resolved[alias])
        model.eval()
        calls = instrument_model(model)
        with torch.no_grad():
            out = model(**synthetic_batch(model))
        flags = expected_flags(alias)
        record(checks, "layer4_forward", alias, "output_finite", True, bool(torch.isfinite(out).all().item()))
        record(checks, "layer4_forward", alias, "proxy_context.called", True, calls["proxy_context"] > 0)
        record(checks, "layer4_forward", alias, "proxy_context.returned", flags["use_process_proxy_encoder"], calls["proxy_returned"] > 0)
        record(checks, "layer4_forward", alias, "process_adaln.called", flags["use_process_adaln"], calls["adaln"] > 0)
        record(checks, "layer4_forward", alias, "urban_control.called", flags["use_urban_control_branch"], calls["control"] > 0)
        residual_should_run = flags["use_morphology_residual_head"] and flags["use_process_proxy_encoder"]
        record(checks, "layer4_forward", alias, "residual_head.called", residual_should_run, calls["residual"] > 0)


def build_report(suite_path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [item for item in checks if not item["ok"]]
    by_layer: dict[str, dict[str, int]] = {}
    for item in checks:
        layer = item["layer"]
        if layer not in by_layer:
            by_layer[layer] = {"passed": 0, "failed": 0}
        by_layer[layer]["passed" if item["ok"] else "failed"] += 1
    return {"suite": str(suite_path), "total_checks": len(checks), "failure_count": len(failures), "by_layer": by_layer, "failures": failures, "checks": checks}


def main() -> int:
    args = parse_args()
    suite_path = resolve_path(args.suite, PROJECT_ROOT / "configs" / "urbanpidit_v531_suite.yaml")
    suite_cfg, resolved = resolve_suite(suite_path)
    checks: list[dict[str, Any]] = []
    verify_suite_structure(checks, suite_cfg)
    verify_yaml_layer(checks, resolved)
    verify_builder_layer(checks, resolved)
    verify_model_layer(checks, resolved)
    verify_forward_layer(checks, resolved)
    report = build_report(suite_path, checks)
    report_path = resolve_path(args.report, suite_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"suite={suite_path}")
    print(f"checks={report['total_checks']} failures={report['failure_count']} report={report_path}")
    if report["failure_count"]:
        for item in report["failures"][:20]:
            print(f"FAIL {item['layer']} {item['alias']} {item['key']}: expected={item['expected']} actual={item['actual']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())