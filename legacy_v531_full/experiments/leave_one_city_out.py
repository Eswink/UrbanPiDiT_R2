from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from data.splits import leave_one_region_splits, normalize_region_specs

from .common import ExperimentCommand, load_yaml, run_commands, save_json, write_manifest, write_yaml


def build_leave_one_city_out_configs(config: str, out_dir: str) -> Dict[str, str]:
    cfg = load_yaml(config)
    regions = normalize_region_specs(cfg)
    splits = leave_one_region_splits(regions)
    out: Dict[str, str] = {}
    for name, split in splits.items():
        run_cfg = dict(cfg)
        run_cfg.setdefault("multi_region", {})["split"] = split
        run_cfg.setdefault("logging", {})["run_name"] = f"urbanpidit_{name}"
        path = Path(out_dir) / "configs" / f"{name}.yaml"
        write_yaml(run_cfg, path)
        out[name] = str(path)
    return out


def build_leave_one_city_out_commands(configs: Dict[str, str]) -> List[ExperimentCommand]:
    return [
        ExperimentCommand(
            name=f"manifest_{name}",
            command=["python", "run_leave_one_city_out.py", "--config", cfg_path, "--out", f"outputs/multi_region/{name}.json"],
            description=f"build and validate leave-one-city-out manifest for {name}",
        )
        for name, cfg_path in configs.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leave-one-city-out experiment configs")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/experiments/leave_one_city_out")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    configs = build_leave_one_city_out_configs(args.config, args.out_dir)
    commands = build_leave_one_city_out_commands(configs)
    manifest = write_manifest(commands, Path(args.out_dir) / "manifest.json", extra={"configs": configs})
    if args.execute:
        manifest["runs"] = run_commands(commands, execute=True)
        save_json(manifest, Path(args.out_dir) / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()