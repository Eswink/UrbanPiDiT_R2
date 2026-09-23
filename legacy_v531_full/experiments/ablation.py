from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping

from .common import ExperimentCommand, deep_update, load_yaml, run_commands, save_json, write_manifest, write_yaml


ABLATION_OVERRIDES: Dict[str, Dict] = {
    "base": {"ablation": {"use_static_morphology_encoder": False, "use_morphology_graph": False, "use_wind_aware_graph": False, "use_dynamic_vg": False, "use_static_vg": False, "use_urban_canopy_coupling": False}},
    "static_encoder_only": {"ablation": {"use_static_morphology_encoder": True, "use_morphology_graph": False, "use_wind_aware_graph": False, "use_dynamic_vg": False, "use_static_vg": False, "use_urban_canopy_coupling": False}},
    "morphology_graph_only": {"ablation": {"use_static_morphology_encoder": False, "use_morphology_graph": True, "use_wind_aware_graph": False, "use_dynamic_vg": False, "use_static_vg": False, "use_urban_canopy_coupling": False}},
    "wind_aware_graph_only": {"ablation": {"use_static_morphology_encoder": False, "use_morphology_graph": True, "use_wind_aware_graph": True, "use_dynamic_vg": False, "use_static_vg": False, "use_urban_canopy_coupling": False}},
    "dynamic_vg_only": {"ablation": {"use_variable_graph": False, "use_dynamic_vg": True, "use_static_vg": False, "use_urban_canopy_coupling": False}},
    "static_vg_only": {"ablation": {"use_variable_graph": False, "use_dynamic_vg": False, "use_static_vg": True, "use_urban_canopy_coupling": False}},
    "urban_canopy_only": {"ablation": {"use_static_morphology_encoder": False, "use_morphology_graph": False, "use_wind_aware_graph": False, "use_dynamic_vg": False, "use_static_vg": False, "use_urban_canopy_coupling": True}},
    "combined": {"ablation": {"use_static_morphology_encoder": True, "use_morphology_graph": True, "use_wind_aware_graph": True, "use_dynamic_vg": True, "use_static_vg": True, "use_urban_canopy_coupling": True}},
}


def build_ablation_configs(base_config: str, out_dir: str, *, names: List[str] | None = None) -> Dict[str, str]:
    cfg = load_yaml(base_config)
    selected = names or list(ABLATION_OVERRIDES.keys())
    out: Dict[str, str] = {}
    for name in selected:
        if name not in ABLATION_OVERRIDES:
            raise KeyError(f"unknown ablation={name}")
        ab_cfg = deep_update(cfg, ABLATION_OVERRIDES[name])
        ab_cfg.setdefault("logging", {})["run_name"] = f"urbanpidit_ablation_{name}"
        path = Path(out_dir) / "configs" / f"{name}.yaml"
        write_yaml(ab_cfg, path)
        out[name] = str(path)
    return out


def build_ablation_commands(configs: Mapping[str, str], out_dir: str) -> List[ExperimentCommand]:
    commands = []
    for name, cfg_path in configs.items():
        commands.append(
            ExperimentCommand(
                name=f"train_{name}",
                command=["python", "train.py", "--config", cfg_path],
                description=f"train ablation variant {name}",
            )
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UrbanPiDiT module ablation configs and commands")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/experiments/ablation")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    names = [x.strip() for x in args.only.split(",") if x.strip()] or None
    configs = build_ablation_configs(args.config, args.out_dir, names=names)
    commands = build_ablation_commands(configs, args.out_dir)
    manifest = write_manifest(commands, Path(args.out_dir) / "manifest.json", extra={"configs": configs})
    if args.execute:
        manifest["runs"] = run_commands(commands, execute=True)
        save_json(manifest, Path(args.out_dir) / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()