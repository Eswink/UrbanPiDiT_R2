from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from .common import load_yaml, save_json
except ImportError:  # pragma: no cover - 支持 python experiments/*.py 直接运行
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.common import load_yaml, save_json

from utils.config_builder import build_model_cfg

try:
    from models.morphology_graph import HeterogeneousMorphologyGraph, WindAwareMorphologyGraph
except ImportError:  # pragma: no cover
    from ..models.morphology_graph import HeterogeneousMorphologyGraph, WindAwareMorphologyGraph


def _static_tensors(cfg: Dict[str, Any], *, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height = int(cfg.get("H", dict(cfg.get("data", {}) or {}).get("H", 8)))
    width = int(cfg.get("W", dict(cfg.get("data", {}) or {}).get("W", 8)))
    static_vars = list(cfg.get("static_vars", dict(cfg.get("data", {}) or {}).get("static_vars", [])) or [])
    static_channels = max(len(static_vars), 1)
    cat_channels = 1 if static_channels > 0 else 0
    cont_channels = max(static_channels - cat_channels, 1)
    static_cat = torch.randint(0, 4, (batch, cat_channels, height, width))
    static_cont = torch.rand(batch, cont_channels, height, width)
    static_raw = torch.cat([static_cat.float(), static_cont], dim=1)
    return static_raw, static_cont, static_cat


def build_graph_diagnostics(config: str, *, dry_run: bool = True) -> Dict[str, Any]:
    cfg = load_yaml(config)
    model_cfg = build_model_cfg(cfg)
    graph_cfg = dict(model_cfg.get("morphology_graph_cfg", {}) or {})
    height = int(model_cfg.get("H", 8))
    width = int(model_cfg.get("W", 8))
    dim = int(model_cfg.get("D", 64))
    nodes = height * width
    batch = 2
    tokens = torch.randn(batch, nodes, dim)
    _, static_cont, static_cat = _static_tensors(cfg, batch=batch)
    wind_uv = torch.zeros(batch, 2, height, width)
    wind_uv[:, 0] = 2.0

    base_graph = HeterogeneousMorphologyGraph(
        dim=dim,
        k=int(graph_cfg.get("k", 4)),
        morphology_sigma=float(graph_cfg.get("morphology_sigma", graph_cfg.get("sigma", 1.0))),
        spatial_sigma=float(graph_cfg.get("spatial_sigma", graph_cfg.get("sigma", 1.0))),
        dropout=0.0,
        use_self_loop=True,
        relation_count=int(graph_cfg.get("relation_count", 2)),
        enable_cache=bool(graph_cfg.get("enable_cache", False)),
    )
    _ = base_graph(tokens, static_cont=static_cont, static_cat=static_cat, spatial_hw=(height, width))
    base_diag = {key: float(value.detach().cpu()) for key, value in base_graph.last_diagnostics.items()}

    wind_graph = WindAwareMorphologyGraph(
        dim=dim,
        k=int(graph_cfg.get("k", 4)),
        morphology_sigma=float(graph_cfg.get("morphology_sigma", graph_cfg.get("sigma", 1.0))),
        spatial_sigma=float(graph_cfg.get("spatial_sigma", graph_cfg.get("sigma", 1.0))),
        dropout=0.0,
        use_self_loop=True,
        wind_temperature=float(graph_cfg.get("wind_temperature", 0.25)),
        wind_strength=float(graph_cfg.get("wind_strength", 1.0)),
        roughness_blocking_strength=float(graph_cfg.get("roughness_blocking_strength", 0.5)),
        enable_cache=bool(graph_cfg.get("enable_cache", False)),
    )
    _ = wind_graph(
        tokens,
        static_cont=static_cont,
        static_cat=static_cat,
        spatial_hw=(height, width),
        wind_uv=wind_uv,
    )
    wind_diag = {key: float(value.detach().cpu()) for key, value in wind_graph.last_diagnostics.items()}

    return {
        "protocol": "v52_graph_diagnostics" if "v52" in Path(config).name else "v51_graph_diagnostics",
        "config": config,
        "dry_run": bool(dry_run),
        "schema_faithful_fields": [
            "continuous_channel_count",
            "categorical_channel_count",
            "same_category_edge_ratio",
            "geo_distance_mean",
            "cont_distance_mean",
            "categorical_mismatch_mean",
            "edge_weight_std",
            "anisotropy_strength",
            "upwind_weight_mean",
            "downwind_weight_mean",
            "roughness_blocking_mean",
        ],
        "spatial_hw": [height, width],
        "node_count": nodes,
        "base_graph": base_diag,
        "wind_aware_graph": wind_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight V5.1 graph diagnostics")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="results/graph_diagnostics")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="keep output as manifest-style diagnostics only")
    args = parser.parse_args()

    diagnostics = build_graph_diagnostics(args.config, dry_run=True)
    out_path = Path(args.out_dir) / "graph_diagnostics.json"
    save_json(diagnostics, out_path)
    print(diagnostics)


if __name__ == "__main__":
    main()