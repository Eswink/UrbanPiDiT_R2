from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from .common import load_yaml
except ImportError:  # pragma: no cover - 支持 python experiments/*.py 直接运行
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.common import load_yaml

from models.model_registry import build_model_from_config
from utils.config_builder import build_model_cfg


def make_smoke_model_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """压缩模型尺寸，只保留配置开关链路用于 CPU smoke。"""

    cfg = deepcopy(model_cfg)
    cfg.update({"H": 4, "W": 4, "D": 32, "depth": 2, "heads": 4, "mlp_ratio": 2.0, "dropout": 0.0, "drop_path_rate": 0.0})
    graph_cfg = dict(cfg.get("morphology_graph_cfg", {}) or {})
    graph_cfg.update({"k": min(int(graph_cfg.get("k", 2)), 2), "enable_cache": False})
    cfg["morphology_graph_cfg"] = graph_cfg
    micromet_cfg = dict(cfg.get("micromet_coupling_cfg", {}) or {})
    micromet_graph_cfg = dict(micromet_cfg.get("graph_cfg", {}) or {})
    micromet_graph_cfg.update({"k": min(int(micromet_graph_cfg.get("k", 2)), 2), "enable_cache": False})
    if str(micromet_cfg.get("mode", "")).lower() in {"interleaved", "inside", "mid"}:
        micromet_cfg["interleaved_interval"] = 1
    micromet_cfg.update({"hidden_channels": min(int(micromet_cfg.get("hidden_channels", 8)), 8), "graph_cfg": micromet_graph_cfg})
    cfg["micromet_coupling_cfg"] = micromet_cfg
    canopy_cfg = dict(cfg.get("urban_canopy_cfg", {}) or {})
    canopy_cfg.update({"hidden_channels": min(int(canopy_cfg.get("hidden_channels", 8)), 8)})
    cfg["urban_canopy_cfg"] = canopy_cfg
    return cfg


def build_dummy_batch(model_cfg: Dict[str, Any], *, batch: int = 2) -> Dict[str, torch.Tensor]:
    height = int(model_cfg.get("H", 8))
    width = int(model_cfg.get("W", 8))
    in_channels = int(model_cfg.get("in_channels", 7))
    ctx_channels = int(model_cfg.get("ctx_channels", in_channels * 4))
    static_channels = int(model_cfg.get("static_channels", 0))
    dynamic_ctx_channels = max(ctx_channels - static_channels, in_channels)

    x_t = torch.randn(batch, in_channels, height, width)
    dyn_ctx = torch.randn(batch, dynamic_ctx_channels, height, width)
    data: Dict[str, torch.Tensor] = {
        "x_t": x_t,
        "x_ctx": dyn_ctx,
        "t": torch.rand(batch),
    }
    if bool(model_cfg.get("use_lead_time_conditioning", False)):
        data["lead_time"] = torch.full((batch,), 0.25)
    else:
        data["lead_time"] = None  # type: ignore[assignment]

    if static_channels > 0:
        static_cat = torch.randint(0, 4, (batch, 1, height, width))
        cont_channels = max(static_channels - 1, 1)
        static_cont = torch.rand(batch, cont_channels, height, width)
        static_raw = torch.cat([static_cat.float(), static_cont], dim=1)[:, :static_channels]
        if static_raw.size(1) < static_channels:
            pad = torch.zeros(batch, static_channels - static_raw.size(1), height, width)
            static_raw = torch.cat([static_raw, pad], dim=1)
        data["x_ctx"] = torch.cat([dyn_ctx, static_raw], dim=1)
        data["static_raw"] = static_raw
        data["static_cont"] = static_cont
        data["static_cat"] = static_cat
        data["hour_of_day"] = torch.full((batch,), 12.0)
    return data


def run_smoke(config: str) -> Dict[str, Any]:
    cfg = load_yaml(config)
    model_cfg = make_smoke_model_cfg(build_model_cfg(cfg))
    model = build_model_from_config(model_cfg)
    model.eval()
    data = build_dummy_batch(model_cfg)
    with torch.no_grad():
        out = model(**data)

    diagnostics = {
        "micromet": sorted(str(key) for key in getattr(model, "last_micromet_diagnostics", {}).keys()),
        "urban_canopy": sorted(str(key) for key in getattr(model, "last_urban_canopy_diagnostics", {}).keys()),
    }
    return {
        "protocol": "v51_smoke_forward",
        "config": config,
        "output_shape": list(out.shape),
        "finite": bool(torch.isfinite(out).all().item()),
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V5.1 lightweight dummy forward smoke test")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    result = run_smoke(args.config)
    print(result)
    if not result["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()