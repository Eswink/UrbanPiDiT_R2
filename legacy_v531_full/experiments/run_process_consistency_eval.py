from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from .common import load_yaml, save_json
except ImportError:  # pragma: no cover - 支持 python experiments/*.py 直接运行
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.common import load_yaml, save_json

from losses import FeasibilityProxyLoss, PhysicalConsistencyConfig, ProcessConsistencyLoss, StructureProxyLoss
from utils.config_builder import build_physics_cfg


def _sample_fields(batch: int = 2, height: int = 8, width: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.zeros(batch, 7, height, width)
    pred[:, 0] = 12.0
    pred[:, 1] = 100000.0
    pred[:, 2] = 20.0
    pred[:, 3] = 0.5
    pred[:, 4] = 1.0
    pred[:, 5] = 2.0
    pred[:, 6] = 1.0
    target = pred.clone()
    static_cont = torch.linspace(0.0, 1.0, steps=height * width).view(1, 1, height, width).repeat(batch, 1, 1, 1)
    return pred, target, static_cont


def build_process_consistency_eval(config: str, *, dry_run: bool = True) -> Dict[str, Any]:
    cfg = load_yaml(config)
    height = int(cfg.get("H", dict(cfg.get("data", {}) or {}).get("H", 8)))
    width = int(cfg.get("W", dict(cfg.get("data", {}) or {}).get("W", 8)))
    physics_cfg = build_physics_cfg(cfg)
    physics_cfg["use_process_consistency"] = True
    losses_cfg = dict(cfg.get("losses", {}) or {})
    proc = dict(losses_cfg.get("process", {}) or {})
    config_obj = PhysicalConsistencyConfig.from_mapping(physics_cfg)
    pred, target, static_cont = _sample_fields(height=height, width=width)
    latest_state = pred.clone()
    pred[:, 5, :, width // 2 :] = pred[:, 5, :, width // 2 :] + 1.5
    pred[:, 2, height // 2 :, :] = pred[:, 2, height // 2 :, :] + 2.0

    feasibility = FeasibilityProxyLoss(config_obj)(pred)
    process = ProcessConsistencyLoss(config_obj)(
        pred,
        latest_state=latest_state,
        static_cont=static_cont,
        hour_of_day=torch.tensor([12.0, 22.0])[: pred.size(0)],
    )
    structure = {
        "wind_div": StructureProxyLoss.wind_divergence(pred),
    }

    violation_rates = {
        "tp_negative_rate": float((pred[:, 4] < 0.0).float().mean()),
        "tcc_out_of_range_rate": float(((pred[:, 3] < 0.0) | (pred[:, 3] > 1.0)).float().mean()),
        "dew_above_temp_rate": float((pred[:, 0] > pred[:, 2]).float().mean()),
        "sp_negative_rate": float((pred[:, 1] < 0.0).float().mean()),
    }

    return {
        "protocol": "v52_process_consistency_eval" if "v52" in Path(config).name else "v51_process_consistency_eval",
        "config": config,
        "dry_run": bool(dry_run),
        "evaluation_protocol": {
            "uses_latest_state_from_context": True,
            "uses_static_schema": True,
            "uses_target_as_error_reference_only": True,
            "reports_drag_residual_gap": True,
            "reports_daynight_diurnal_gap": True,
        },
        "enabled": {
            "process_consistency": bool(config_obj.use_process_consistency),
            "lambda_rh": float(config_obj.lambda_process_rh),
            "lambda_drag": float(config_obj.lambda_process_drag),
            "lambda_diurnal": float(config_obj.lambda_process_diurnal),
        },
        "feasibility": {key: float(value.detach().cpu()) for key, value in feasibility.items()},
        "process": {key: float(value.detach().cpu()) for key, value in process.items()},
        "structure": {key: float(value.detach().cpu()) for key, value in structure.items()},
        "violation_rates": violation_rates,
        "process_config": proc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V5.1 process-consistency proxy diagnostics")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="results/process_consistency")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="write proxy diagnostics only; no checkpoint evaluation")
    args = parser.parse_args()

    result = build_process_consistency_eval(args.config, dry_run=True)
    out_path = Path(args.out_dir) / "process_consistency_eval.json"
    save_json(result, out_path)
    print(result)


if __name__ == "__main__":
    main()