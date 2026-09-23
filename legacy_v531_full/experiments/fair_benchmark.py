from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .common import ExperimentCommand, load_yaml, run_commands, save_json, write_manifest


def build_fair_benchmark_commands(config: str, out_dir: str) -> List[ExperimentCommand]:
    return [
        ExperimentCommand(
            name="fair_forecast_baselines",
            command=["python", "-m", "baselines.forecast_runner", "--config", config, "--out_dir", str(Path(out_dir) / "forecast")],
            description="shape-safe baselines under controlled static-information policies",
        ),
        ExperimentCommand(
            name="legacy_linear_ridge",
            command=["python", "-m", "baselines.linear", "--config", config, "--kind", "ridge", "--out_dir", str(Path(out_dir) / "linear")],
            description="legacy ridge baseline with the same data loader and static schema",
        ),
        ExperimentCommand(
            name="legacy_tree_rf",
            command=["python", "-m", "baselines.tree", "--config", config, "--model", "rf", "--out_dir", str(Path(out_dir) / "tree_rf")],
            description="legacy random forest baseline with the same data loader and static schema",
        ),
    ]


def build_fair_benchmark_manifest(config: str, out_dir: str) -> Dict:
    cfg = load_yaml(config)
    commands = build_fair_benchmark_commands(config, out_dir)
    return {
        "protocol": "fair_static_information_benchmark",
        "config": config,
        "out_dir": out_dir,
        "static_schema": cfg.get("static_schema", dict(cfg.get("data", {}) or {}).get("static_schema", {})),
        "commands": [
            {"name": c.name, "description": c.description, "command": c.command, "shell": c.shell()} for c in commands
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair static-information benchmark orchestrator")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/experiments/fair_benchmark")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    commands = build_fair_benchmark_commands(args.config, args.out_dir)
    manifest = write_manifest(
        commands,
        Path(args.out_dir) / "manifest.json",
        extra={"protocol": "fair_static_information_benchmark", "config": args.config},
    )
    if args.execute:
        manifest["runs"] = run_commands(commands, execute=True)
        save_json(manifest, Path(args.out_dir) / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()