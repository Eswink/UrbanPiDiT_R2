from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping

try:
    from .common import ExperimentCommand, deep_update, load_yaml, run_commands, save_json, write_manifest, write_yaml
except ImportError:  # pragma: no cover - 支持 python experiments/*.py 直接运行
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.common import ExperimentCommand, deep_update, load_yaml, run_commands, save_json, write_manifest, write_yaml


MECHANISM_OVERRIDES: Dict[str, Dict] = {
    "base_dit_cross_attention": {
        "ablation": {
            "use_static_morphology_encoder": False,
            "use_morphology_graph": False,
            "use_wind_aware_graph": False,
            "use_dynamic_vg": False,
            "use_static_vg": False,
            "use_urban_canopy_coupling": False,
            "use_urban_canopy": False,
            "use_micromet_coupling": False,
        },
        "losses": {"process": {"enabled": False, "lambda_rh": 0.0, "lambda_drag": 0.0, "lambda_diurnal": 0.0}},
    },
    "plus_hetero_graph": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": False,
            "use_micromet_coupling": False,
        },
        "losses": {"process": {"enabled": False, "lambda_rh": 0.0, "lambda_drag": 0.0, "lambda_diurnal": 0.0}},
    },
    "plus_wind_aware_graph": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_micromet_coupling": False,
        },
        "losses": {"process": {"enabled": False, "lambda_rh": 0.0, "lambda_drag": 0.0, "lambda_diurnal": 0.0}},
    },
    "plus_interleaved_micromet": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "interleaved", "interleaved_interval": 3},
        "losses": {"process": {"enabled": False, "lambda_rh": 0.0, "lambda_drag": 0.0, "lambda_diurnal": 0.0}},
    },
    "plus_process_rh": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "interleaved", "interleaved_interval": 3},
        "losses": {"process": {"enabled": True, "lambda_rh": 0.02, "lambda_drag": 0.0, "lambda_diurnal": 0.0}},
    },
    "plus_process_drag": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "interleaved", "interleaved_interval": 3},
        "losses": {"process": {"enabled": True, "lambda_rh": 0.02, "lambda_drag": 0.01, "lambda_diurnal": 0.0}},
    },
    "plus_process_diurnal": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "interleaved", "interleaved_interval": 3},
        "losses": {"process": {"enabled": True, "lambda_rh": 0.02, "lambda_drag": 0.01, "lambda_diurnal": 0.01}},
    },
    "full_v52_micromet_refine": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_dynamic_vg": True,
            "use_static_vg": True,
            "use_urban_canopy_coupling": False,
            "use_urban_canopy": False,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "interleaved", "interleaved_interval": 3},
        "losses": {"process": {"enabled": True, "lambda_rh": 0.02, "lambda_drag": 0.01, "lambda_diurnal": 0.01}},
    },
    "v4_base": {
        "ablation": {
            "use_static_morphology_encoder": False,
            "use_morphology_graph": False,
            "use_wind_aware_graph": False,
            "use_dynamic_vg": False,
            "use_static_vg": False,
            "use_urban_canopy_coupling": False,
            "use_urban_canopy": False,
            "use_micromet_coupling": False,
        }
    },
    "static_encoder": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": False,
            "use_wind_aware_graph": False,
            "use_micromet_coupling": False,
        }
    },
    "morphology_graph": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": False,
            "use_micromet_coupling": False,
        }
    },
    "wind_aware_graph": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_micromet_coupling": False,
        }
    },
    "micromet_only": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": False,
            "use_wind_aware_graph": False,
            "use_urban_canopy_coupling": False,
            "use_urban_canopy": False,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "pre"},
    },
    "full_v51": {
        "ablation": {
            "use_static_morphology_encoder": True,
            "use_morphology_graph": True,
            "use_wind_aware_graph": True,
            "use_dynamic_vg": True,
            "use_static_vg": True,
            "use_urban_canopy_coupling": False,
            "use_urban_canopy": False,
            "use_micromet_coupling": True,
        },
        "data": {"expose_hour": True},
        "micromet_coupling": {"enabled": True, "mode": "pre"},
    },
}


def build_mechanism_ablation_configs(
    base_config: str,
    out_dir: str,
    *,
    names: List[str] | None = None,
) -> Dict[str, str]:
    cfg = load_yaml(base_config)
    experiment = dict(cfg.get("experiment", {}) or {})
    selected = names or list(experiment.get("variants", []) or MECHANISM_OVERRIDES.keys())
    protocol = str(experiment.get("protocol", "v51_mechanism_ablation"))
    version = "v52" if "v52" in protocol or "v52" in Path(base_config).name else "v51"
    out: Dict[str, str] = {}
    for name in selected:
        if name not in MECHANISM_OVERRIDES:
            raise KeyError(f"unknown mechanism ablation={name}")
        variant = deep_update(cfg, MECHANISM_OVERRIDES[name])
        variant.setdefault("logging", {})["run_name"] = f"urbanpidit_{version}_mechanism_{name}"
        path = Path(out_dir) / "configs" / f"{name}.yaml"
        write_yaml(variant, path)
        out[name] = str(path)
    return out


def build_mechanism_ablation_commands(configs: Mapping[str, str]) -> List[ExperimentCommand]:
    commands: List[ExperimentCommand] = []
    v52_names = {
        "base_dit_cross_attention",
        "plus_hetero_graph",
        "plus_wind_aware_graph",
        "plus_interleaved_micromet",
        "plus_process_rh",
        "plus_process_drag",
        "plus_process_diurnal",
        "full_v52_micromet_refine",
    }
    for name, cfg_path in configs.items():
        version = "V5.2" if name in v52_names or "v52" in cfg_path else "V5.1"
        commands.append(
            ExperimentCommand(
                name=f"train_{name}",
                command=["python", "train.py", "--config", cfg_path],
                description=f"train {version} mechanism ablation variant {name}",
            )
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5.1/V5.2 mechanism ablation manifest")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/experiments/mechanism_ablation")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="write manifest/configs only; this is the default behavior")
    parser.add_argument("--execute", action="store_true", help="execute generated training commands")
    args = parser.parse_args()

    names = [x.strip() for x in args.only.split(",") if x.strip()] or None
    configs = build_mechanism_ablation_configs(args.config, args.out_dir, names=names)
    commands = build_mechanism_ablation_commands(configs)
    protocol = str(dict(load_yaml(args.config).get("experiment", {}) or {}).get("protocol", "v51_mechanism_ablation"))
    manifest = write_manifest(
        commands,
        Path(args.out_dir) / "manifest.json",
        extra={
            "protocol": protocol,
            "config": args.config,
            "configs": configs,
            "dry_run": not bool(args.execute),
            "v52_refine_axes": [
                "schema_faithful_graph",
                "wind_aware_graph",
                "interleaved_micromet",
                "rh_process_proxy",
                "residual_drag_proxy",
                "daynight_diurnal_proxy",
            ],
        },
    )
    if args.execute:
        manifest["runs"] = run_commands(commands, execute=True)
        save_json(manifest, Path(args.out_dir) / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()