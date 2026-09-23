from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

from data.multi_region_loader import build_leave_one_region_datamodules
from data.splits import leave_one_region_splits, normalize_region_specs
from data.static_preprocess import describe_static_schema, merge_static_schemas


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_leave_one_city_out_manifest(config_path: str | Path, *, validate_data: bool = False) -> Dict:
    cfg = load_yaml(config_path)
    regions = normalize_region_specs(cfg)
    splits = leave_one_region_splits(regions)
    static_vars = list(dict(cfg.get("data", {}) or {}).get("static_vars", cfg.get("static_vars", [])) or [])
    base_schema = dict(dict(cfg.get("data", {}) or {}).get("static_schema", cfg.get("static_schema", {})) or {})
    merged_schema = merge_static_schemas(static_vars, [base_schema, *[r.static_schema for r in regions]])
    manifest = {
        "config": str(config_path),
        "num_regions": len(regions),
        "regions": [
            {"name": r.name, "data_root": str(r.data_root), "weight": r.weight, "static_schema": r.static_schema}
            for r in regions
        ],
        "static_schema": describe_static_schema(static_vars, merged_schema),
        "splits": splits,
    }

    if validate_data:
        modules = build_leave_one_region_datamodules(cfg)
        validation = {}
        for name, dm in modules.items():
            dm.prepare_data()
            dm.setup()
            validation[name] = {
                "train_samples": len(dm.train_ds),
                "val_samples": len(dm.val_ds),
                "test_samples": len(dm.test_ds),
                "train_regions": dm.split.get("train", []),
                "test_regions": dm.split.get("test", []),
            }
        manifest["validation"] = validation
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leave-one-city-out split manifest")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out", type=str, default="outputs/multi_region/leave_one_city_out_manifest.json")
    parser.add_argument("--validate_data", action="store_true")
    args = parser.parse_args()

    manifest = build_leave_one_city_out_manifest(args.config, validate_data=bool(args.validate_data))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()