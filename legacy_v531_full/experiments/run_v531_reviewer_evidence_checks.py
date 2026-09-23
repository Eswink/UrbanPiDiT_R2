#!/usr/bin/env python3
"""UrbanPiDiT V5.3.1 reviewer-evidence harness.

该脚本执行最小可复现证据链：加载 YAML 配置、构建模型、创建无目标 dummy batch、
运行 forward(return_diagnostics=True)，并输出 JSON-safe reviewer diagnostics。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_registry import build_model_v531_from_config, dropped_model_kwargs, filter_model_kwargs
from urbanpidit_version import CLAIM_BOUNDARY, DIAGNOSTIC_PREFIXES, RELEASE_NAME, VERSION, VERSION_NAME
from utils.config_builder import build_model_cfg


DEFAULT_REQUIRED_PREFIXES = (
    "version/",
    "model/",
    "input/",
    "output/",
    "proxy/",
    "process_adaln/",
    "urban_control/",
    "process_graph/",
    "micromet/",
    "residual_prediction/",
    "leakage/",
)


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML mapping: {config_path}")
    return data


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu")
        if tensor.numel() == 1:
            return tensor.item()
        tensor_f = tensor.float()
        return {
            "shape": list(tensor.shape),
            "mean": tensor_f.mean().item(),
            "std": tensor_f.std(unbiased=False).item(),
            "norm": tensor_f.norm().item(),
        }
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cfg_list(cfg: Mapping[str, Any], key: str) -> list[Any]:
    value = cfg.get(key, [])
    return list(value or [])


def _static_schema(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("static_schema", {}) or {})


def make_dummy_batch(cfg: Mapping[str, Any], model_cfg: Mapping[str, Any], *, batch_size: int = 2) -> Dict[str, torch.Tensor]:
    torch.manual_seed(int(cfg.get("train", {}).get("seed", 42)))
    dynamic_vars = _cfg_list(cfg, "dynamic_vars")
    static_vars = _cfg_list(cfg, "static_vars")
    schema = _static_schema(cfg)
    categorical = [str(v) for v in schema.get("categorical", []) if str(v) in {str(s) for s in static_vars}]
    continuous = [str(v) for v in schema.get("continuous", []) if str(v) in {str(s) for s in static_vars}]
    if not categorical and "landcover" in {str(s) for s in static_vars}:
        categorical = ["landcover"]
    if not continuous:
        continuous = [str(v) for v in static_vars if str(v) not in set(categorical)]

    batch = int(batch_size)
    channels = int(model_cfg["in_channels"])
    ctx_channels = int(model_cfg["ctx_channels"])
    height = int(model_cfg["H"])
    width = int(model_cfg["W"])
    static_channels = int(model_cfg.get("static_channels", 0))
    dynamic_ctx_channels = max(ctx_channels - static_channels, channels)

    x_t = torch.randn(batch, channels, height, width)
    x_ctx_dynamic = torch.randn(batch, dynamic_ctx_channels, height, width)

    cat_cardinality = dict(schema.get("categorical_cardinality", {}) or {})
    if categorical:
        cat_parts = []
        for name in categorical:
            card = max(int(cat_cardinality.get(name, 20)), 1)
            cat_parts.append(torch.randint(0, card, (batch, 1, height, width), dtype=torch.long))
        static_cat = torch.cat(cat_parts, dim=1)
    else:
        static_cat = torch.zeros(batch, 0, height, width, dtype=torch.long)

    if continuous:
        static_cont = torch.randn(batch, len(continuous), height, width)
    else:
        static_cont = torch.zeros(batch, 0, height, width)

    static_parts = []
    if static_cat.numel() > 0:
        static_parts.append(static_cat.float())
    if static_cont.numel() > 0:
        static_parts.append(static_cont)
    if static_parts:
        static_raw = torch.cat(static_parts, dim=1)
    else:
        static_raw = torch.zeros(batch, static_channels, height, width)
    if static_raw.size(1) < static_channels:
        pad = torch.zeros(batch, static_channels - static_raw.size(1), height, width)
        static_raw = torch.cat([static_raw, pad], dim=1)
    elif static_raw.size(1) > static_channels:
        static_raw = static_raw[:, :static_channels]

    if static_channels > 0:
        x_ctx = torch.cat([x_ctx_dynamic[:, : ctx_channels - static_channels], static_raw], dim=1)
    else:
        x_ctx = x_ctx_dynamic[:, :ctx_channels]

    return {
        "x_t": x_t,
        "x_ctx": x_ctx,
        "t": torch.linspace(0.05, 0.95, steps=batch),
        "lead_time": torch.linspace(0.25, 1.0, steps=batch),
        "static_raw": static_raw,
        "static_cont": static_cont,
        "static_cat": static_cat,
        "hour_of_day": torch.linspace(9.0, 15.0, steps=batch),
    }


def parameter_report(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gates: Dict[str, Any] = {}
    for name, param in model.named_parameters():
        lname = name.lower()
        if "gate" in lname or "alpha" in lname or "gamma" in lname:
            gates[name] = to_jsonable(param)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "gate_parameter_count": len(gates),
        "gate_parameters": gates,
    }


def prefix_report(diagnostics: Mapping[str, Any], required_prefixes: tuple[str, ...]) -> Dict[str, Any]:
    keys = sorted(str(k) for k in diagnostics)
    observed = sorted({key.split("/", 1)[0] + "/" for key in keys if "/" in key})
    return {
        "observed_prefixes": observed,
        "required_prefixes": list(required_prefixes),
        "missing_required_prefixes": [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in keys)],
        "diagnostic_key_count": len(keys),
        "diagnostic_keys": keys,
    }


def build_report(
    cfg: Mapping[str, Any],
    *,
    config_path: str | Path,
    batch_size: int = 2,
    required_prefixes: tuple[str, ...] = DEFAULT_REQUIRED_PREFIXES,
) -> Dict[str, Any]:
    model_cfg = build_model_cfg(dict(cfg))
    model = build_model_v531_from_config(model_cfg)
    model.eval()

    batch = make_dummy_batch(cfg, model_cfg, batch_size=batch_size)
    with torch.no_grad():
        prediction, diagnostics = model(
            batch["x_t"],
            batch["x_ctx"],
            batch["t"],
            lead_time=batch["lead_time"],
            static_raw=batch["static_raw"],
            static_cont=batch["static_cont"],
            static_cat=batch["static_cat"],
            hour_of_day=batch["hour_of_day"],
            return_diagnostics=True,
        )

    diagnostics = to_jsonable(diagnostics)
    prefixes = prefix_report(diagnostics, required_prefixes)
    return {
        "release_name": RELEASE_NAME,
        "version": VERSION,
        "version_name": VERSION_NAME,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "trained_checkpoint": False,
        "model_cfg": to_jsonable(filter_model_kwargs(model_cfg)),
        "dropped_model_cfg_keys": sorted(dropped_model_kwargs(model_cfg).keys()),
        "enabled_modules": {
            "process_proxy_encoder": bool(model_cfg.get("use_process_proxy_encoder", False)),
            "process_adaln": bool(model_cfg.get("use_process_adaln", False)),
            "urban_control_branch": bool(model_cfg.get("use_urban_control_branch", False)),
            "anisotropic_process_graph": bool(model_cfg.get("use_anisotropic_process_graph", False)),
            "micromet_coupling": bool(model_cfg.get("use_micromet_coupling", False)),
            "micromet_token_branches": bool(model_cfg.get("use_micromet_token_branches", False)),
            "morphology_residual_head": bool(model_cfg.get("use_morphology_residual_head", False)),
        },
        "parameter_report": parameter_report(model),
        "batch_report": {
            "input_keys": sorted(batch.keys()),
            "target_tensor_present": False,
            "x_t_shape": list(batch["x_t"].shape),
            "x_ctx_shape": list(batch["x_ctx"].shape),
            "static_raw_shape": list(batch["static_raw"].shape),
            "static_cont_shape": list(batch["static_cont"].shape),
            "static_cat_shape": list(batch["static_cat"].shape),
        },
        "forward_report": {
            "prediction_shape": list(prediction.shape),
            "prediction_mean": prediction.detach().float().mean().cpu().item(),
            "prediction_std": prediction.detach().float().std(unbiased=False).cpu().item(),
            "prediction_finite": bool(torch.isfinite(prediction).all().cpu().item()),
        },
        "diagnostic_prefix_contract": prefixes,
        "diagnostics": diagnostics,
        "diagnostic_prefixes_declared": list(DIAGNOSTIC_PREFIXES),
    }


def write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run UrbanPiDiT V5.3.1 reviewer-evidence checks.")
    parser.add_argument("--config", type=str, default="configs/urbanpidit_v531_reviewer_evidence.yaml")
    parser.add_argument("--output", type=str, default="outputs/v531_reviewer_evidence.json")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    report = build_report(cfg, config_path=args.config, batch_size=args.batch_size)
    write_report(report, args.output)
    print(f"Wrote reviewer evidence JSON: {Path(args.output).expanduser().resolve()}")
    print(f"Diagnostic keys: {report['diagnostic_prefix_contract']['diagnostic_key_count']}")
    missing = report["diagnostic_prefix_contract"]["missing_required_prefixes"]
    if missing:
        print(f"Missing required prefixes: {', '.join(missing)}")
    return report


if __name__ == "__main__":
    main()