from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from .common import ExperimentCommand, deep_update, load_yaml, run_commands, save_json, write_manifest, write_yaml
except ImportError:  # pragma: no cover - 支持 python experiments/*.py 直接运行
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.common import ExperimentCommand, deep_update, load_yaml, run_commands, save_json, write_manifest, write_yaml


STATIC_PERTURBATION_MODES: tuple[str, ...] = (
    "none",
    "fill_mean",
    "fill_zero",
    "shuffle_spatial",
    "shuffle_batch",
    "landcover_only",
    "continuous_only",
    "wrong_city_static",
)


def default_static_perturbation_modes(static_vars: Sequence[str]) -> List[str]:
    modes = list(STATIC_PERTURBATION_MODES)
    for name in static_vars:
        if str(name) not in {"landcover"}:
            modes.append(f"remove_{name}")
    return modes


def build_static_perturbation_spec(mode: str, cfg: Mapping[str, Any], *, seed: int = 42) -> Dict[str, Any]:
    static_schema = dict(cfg.get("static_schema", dict(cfg.get("data", {}) or {}).get("static_schema", {})) or {})
    cardinality = dict(static_schema.get("categorical_cardinality", {}) or {})
    spec: Dict[str, Any] = {"mode": mode, "seed": int(seed), "deterministic": True}
    if cardinality:
        spec["categorical_modes"] = {name: 0 for name in cardinality}
    if mode == "wrong_city_static":
        wrong_city = dict(cfg.get("wrong_city_static", {}) or {})
        if wrong_city.get("source_path"):
            spec["source_path"] = str(wrong_city["source_path"])
        if wrong_city.get("source_key"):
            spec["source_key"] = str(wrong_city["source_key"])
        spec["skip_if_missing"] = True
    return spec


def build_static_perturbation_configs(
    base_config: str,
    out_dir: str,
    *,
    modes: List[str] | None = None,
) -> Dict[str, str]:
    cfg = load_yaml(base_config)
    experiment = dict(cfg.get("experiment", {}) or {})
    protocol = str(experiment.get("protocol", "v51_static_perturbation"))
    version = "v52" if "v52" in protocol or "v52" in Path(base_config).name else "v51"
    static_vars = list(cfg.get("static_vars", dict(cfg.get("data", {}) or {}).get("static_vars", [])) or [])
    seed = int(dict(cfg.get("train", {}) or {}).get("seed", 42))
    selected = modes or list(experiment.get("modes", []) or default_static_perturbation_modes(static_vars))
    out: Dict[str, str] = {}
    for mode in selected:
        spec = build_static_perturbation_spec(mode, cfg, seed=seed)
        variant = deep_update(cfg, {"static_perturb": spec})
        variant.setdefault("logging", {})["run_name"] = f"urbanpidit_{version}_static_perturb_{mode}"
        path = Path(out_dir) / "configs" / f"static_{mode}.yaml"
        write_yaml(variant, path)
        out[mode] = str(path)
    return out


def build_static_perturbation_commands(configs: Mapping[str, str], *, ckpt: str | None = None) -> List[ExperimentCommand]:
    commands: List[ExperimentCommand] = []
    for mode, cfg_path in configs.items():
        if ckpt:
            cmd = ["python", "evaluate.py", "--config", cfg_path, "--ckpt", ckpt]
            desc = f"evaluate static perturbation mode {mode}"
        else:
            cmd = ["python", "evaluate.py", "--config", cfg_path, "--ckpt", "${CKPT_PATH}"]
            desc = f"manifest placeholder for static perturbation mode {mode}"
        commands.append(ExperimentCommand(name=f"static_{mode}", command=cmd, description=desc))
    return commands


def compute_degradation_rows(results: Mapping[str, Mapping[str, float]], *, baseline: str = "none") -> List[Dict[str, Any]]:
    base = dict(results.get(baseline, {}) or {})
    rows: List[Dict[str, Any]] = []
    metric_names = sorted({key for item in results.values() for key in item.keys()})
    for mode, metrics in results.items():
        for metric in metric_names:
            value = metrics.get(metric)
            base_value = base.get(metric)
            delta = None
            ratio = None
            if value is not None and base_value is not None:
                delta = float(value) - float(base_value)
                denom = abs(float(base_value))
                ratio = delta / denom if denom > 1e-12 else None
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "value": value,
                    "baseline": base_value,
                    "delta": delta,
                    "degradation_ratio": ratio,
                }
            )
    return rows


def write_degradation_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "metric", "value", "baseline", "delta", "degradation_ratio"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_results(path: str | Path | None) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): dict(v or {}) for k, v in dict(data or {}).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5.1 static perturbation manifest")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/experiments/static_perturbation")
    parser.add_argument("--modes", type=str, default="")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--results", type=str, default=None, help="optional evaluated metrics json for degradation CSV")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="write manifest/configs only; this is the default behavior")
    parser.add_argument("--execute", action="store_true", help="execute generated evaluation commands")
    args = parser.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()] or None
    configs = build_static_perturbation_configs(args.config, args.out_dir, modes=modes)
    commands = build_static_perturbation_commands(configs, ckpt=args.ckpt)
    result_rows = compute_degradation_rows(load_results(args.results)) if args.results else []
    if result_rows:
        write_degradation_csv(result_rows, Path(args.out_dir) / "degradation_ratio.csv")
    cfg = load_yaml(args.config)
    protocol = str(dict(cfg.get("experiment", {}) or {}).get("protocol", "v51_static_perturbation"))
    manifest = write_manifest(
        commands,
        Path(args.out_dir) / "manifest.json",
        extra={
            "protocol": protocol,
            "config": args.config,
            "configs": configs,
            "dry_run": not bool(args.execute),
            "degradation_csv": str(Path(args.out_dir) / "degradation_ratio.csv") if result_rows else None,
        },
    )
    if args.execute:
        manifest["runs"] = run_commands(commands, execute=True)
        save_json(manifest, Path(args.out_dir) / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()