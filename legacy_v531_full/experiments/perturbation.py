from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .common import ExperimentCommand, deep_update, load_yaml, run_commands, save_json, write_manifest, write_yaml


PERTURB_MODES = ["none", "fill_mean", "fill_zero", "shuffle_hw", "rot90", "flip_ud", "flip_lr"]


def build_perturbation_configs(base_config: str, out_dir: str, *, modes: List[str] | None = None) -> Dict[str, str]:
    cfg = load_yaml(base_config)
    selected = modes or PERTURB_MODES
    out: Dict[str, str] = {}
    for mode in selected:
        p_cfg = deep_update(cfg, {"static_perturb": {"mode": mode, "seed": 42, "deterministic": True}})
        p_cfg.setdefault("logging", {})["run_name"] = f"urbanpidit_static_perturb_{mode}"
        path = Path(out_dir) / "configs" / f"static_{mode}.yaml"
        write_yaml(p_cfg, path)
        out[mode] = str(path)
    return out


def build_perturbation_commands(configs: Dict[str, str], *, ckpt: str | None = None) -> List[ExperimentCommand]:
    commands = []
    for mode, cfg_path in configs.items():
        if ckpt:
            cmd = ["python", "evaluate.py", "--config", cfg_path, "--ckpt", ckpt]
            desc = f"evaluate checkpoint with static perturbation {mode}"
        else:
            cmd = ["python", "train.py", "--config", cfg_path]
            desc = f"train/evaluate perturbation config {mode}"
        commands.append(ExperimentCommand(name=f"static_{mode}", command=cmd, description=desc))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static perturbation experiment configs")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/experiments/perturbation")
    parser.add_argument("--modes", type=str, default="")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()] or None
    configs = build_perturbation_configs(args.config, args.out_dir, modes=modes)
    commands = build_perturbation_commands(configs, ckpt=args.ckpt)
    manifest = write_manifest(commands, Path(args.out_dir) / "manifest.json", extra={"configs": configs})
    if args.execute:
        manifest["runs"] = run_commands(commands, execute=True)
        save_json(manifest, Path(args.out_dir) / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()